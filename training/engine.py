"""Shared optimization, loss, metric, and prediction utilities."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.geometry_adaptive_normst import build_visible_native_neighbor_graph


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _reshape_gene_weight(gene_weight, prediction):
    if gene_weight is None:
        return None
    weight = torch.as_tensor(
        gene_weight, device=prediction.device, dtype=prediction.dtype
    )
    genes = prediction.shape[-1] if prediction.ndim == 3 else prediction.shape[1]
    if weight.ndim != 1 or weight.shape[0] != genes:
        raise ValueError("gene_weight must have shape [G]")
    if not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("gene_weight must be finite and positive")
    if prediction.ndim == 3:
        return weight.reshape(1, 1, genes)
    if prediction.ndim == 4:
        return weight.reshape(1, genes, 1, 1)
    raise ValueError("gene weighting expects a 3D or 4D prediction")


def masked_smooth_l1(prediction, target, mask, gene_weight=None):
    expanded = mask.to(prediction.dtype).expand_as(prediction)
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    weight = _reshape_gene_weight(gene_weight, prediction)
    if weight is not None:
        elementwise = elementwise * weight
    return (elementwise * expanded).sum() / expanded.sum().clamp_min(1.0)


def structure_aware_visium_loss(
    prediction,
    target,
    mask,
    gene_correlation_weight=0.1,
    variance_weight=0.01,
    negative_weight=0.1,
    min_target_variance=1e-6,
    epsilon=1e-8,
    gene_weight=None,
):
    """SmoothL1 plus spatial gene-structure and non-negativity penalties.

    Correlation and variance are computed per gene across target spots. Genes
    whose target map is effectively flat are excluded from both structural
    terms. A constant prediction is retained with correlation zero, rather
    than being dropped as an invalid correlation, so collapse is penalized.
    """
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError(
            "structure-aware loss expects matching [batch, spots, genes] tensors"
        )
    if mask.shape != (*prediction.shape[:2], 1):
        raise ValueError("structure-aware loss expects mask shape [batch, spots, 1]")

    base = masked_smooth_l1(
        prediction, target, mask, gene_weight=gene_weight
    )
    prediction_float = prediction.float().transpose(1, 2)
    target_float = target.float().transpose(1, 2)
    weight = mask.to(prediction_float.dtype).transpose(1, 2).expand_as(
        prediction_float
    )
    count = weight.sum(dim=-1).clamp_min(1.0)
    prediction_mean = (prediction_float * weight).sum(dim=-1) / count
    target_mean = (target_float * weight).sum(dim=-1) / count
    prediction_centered = (
        prediction_float - prediction_mean[..., None]
    ) * weight
    target_centered = (target_float - target_mean[..., None]) * weight
    prediction_energy = prediction_centered.square().sum(dim=-1)
    target_energy = target_centered.square().sum(dim=-1)
    target_variance = target_energy / count
    valid = (count >= 2) & (target_variance > min_target_variance)

    zero = prediction_float.sum() * 0.0
    if valid.any():
        correlation = (
            (prediction_centered * target_centered).sum(dim=-1)
            / torch.sqrt(
                (prediction_energy * target_energy).clamp_min(epsilon)
            )
        ).clamp(-1.0, 1.0)
        gene_correlation_loss = 1.0 - correlation[valid].mean()
        prediction_std = torch.sqrt(
            (prediction_energy[valid] / count[valid]).clamp_min(epsilon)
        )
        target_std = torch.sqrt(target_variance[valid])
        log_std_error = torch.log(prediction_std + epsilon) - torch.log(
            target_std + epsilon
        )
        variance_loss = F.smooth_l1_loss(
            log_std_error, torch.zeros_like(log_std_error)
        )
    else:
        gene_correlation_loss = zero
        variance_loss = zero

    expanded = mask.to(prediction_float.dtype).expand_as(prediction)
    negative_loss = (
        F.relu(-prediction.float()).square() * expanded
    ).sum() / expanded.sum().clamp_min(1.0)
    total = (
        base
        + gene_correlation_weight * gene_correlation_loss
        + variance_weight * variance_loss
        + negative_weight * negative_loss
    )
    return total, {
        "smooth_l1": base,
        "gene_correlation": gene_correlation_loss,
        "variance": variance_loss,
        "negative": negative_loss,
        "valid_genes": valid.sum(),
    }


def _masked_correlations(prediction, target, mask, epsilon=1e-8):
    """Vectorized Pearson for rows with a shared last-axis definition."""
    prediction = prediction.float()
    target = target.float()
    weight = mask.to(prediction.dtype)
    count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    prediction_centered = prediction - (
        prediction * weight
    ).sum(dim=-1, keepdim=True) / count
    target_centered = target - (
        target * weight
    ).sum(dim=-1, keepdim=True) / count
    prediction_centered = prediction_centered * weight
    target_centered = target_centered * weight
    numerator = (prediction_centered * target_centered).sum(dim=-1)
    prediction_energy = prediction_centered.square().sum(dim=-1)
    target_energy = target_centered.square().sum(dim=-1)
    valid = (
        (count.squeeze(-1) >= 2)
        & (prediction_energy > epsilon)
        & (target_energy > epsilon)
    )
    denominator = torch.sqrt(
        prediction_energy.clamp_min(epsilon)
        * target_energy.clamp_min(epsilon)
    )
    return (numerator / denominator)[valid]


def _gene_variance_mean(prediction, mask, task):
    if task == "visium":
        values = prediction.transpose(1, 2)
        weight = mask.transpose(1, 2).expand(-1, prediction.shape[2], -1)
    else:
        values = prediction.flatten(2)
        weight = mask.flatten(2).expand(-1, prediction.shape[1], -1)
    weight = weight.to(values.dtype)
    count = weight.sum(dim=-1).clamp_min(1.0)
    mean = (values.float() * weight).sum(dim=-1) / count
    variance = (
        (values.float() - mean[..., None]).square() * weight
    ).sum(dim=-1) / count
    valid = count >= 2
    return variance[valid].mean() if valid.any() else values.sum() * 0.0


def _gene_value_moments(prediction, mask, task):
    """Return pooled per-gene sum, square sum, and valid count."""
    values = prediction.detach().float()
    if task == "visium":
        weight = mask.expand_as(values).to(values.dtype)
        dimensions = (0, 1)
    else:
        weight = mask.expand_as(values).to(values.dtype)
        dimensions = (0, 2, 3)
    weighted = values * weight
    return (
        weighted.sum(dim=dimensions, dtype=torch.float64).cpu(),
        (values.square() * weight).sum(
            dim=dimensions, dtype=torch.float64
        ).cpu(),
        weight.sum(dim=dimensions, dtype=torch.float64).cpu(),
    )


def correlation_values(prediction, target, mask, task):
    """Return valid gene-wise and spot-wise Pearson values."""
    if task == "visium":
        batch, points, genes = prediction.shape
        gene_prediction = prediction.transpose(1, 2).reshape(batch * genes, points)
        gene_target = target.transpose(1, 2).reshape(batch * genes, points)
        gene_mask = mask.transpose(1, 2).expand(-1, genes, -1).reshape(
            batch * genes, points
        )
        spot_prediction = prediction.reshape(batch * points, genes)
        spot_target = target.reshape(batch * points, genes)
        spot_mask = mask.expand(-1, -1, genes).reshape(batch * points, genes)
    else:
        batch, genes = prediction.shape[:2]
        points = prediction.shape[2] * prediction.shape[3]
        gene_prediction = prediction.flatten(2).reshape(batch * genes, points)
        gene_target = target.flatten(2).reshape(batch * genes, points)
        flat_mask = mask.flatten(2)
        gene_mask = flat_mask.expand(-1, genes, -1).reshape(
            batch * genes, points
        )
        spot_prediction = prediction.flatten(2).transpose(1, 2).reshape(
            batch * points, genes
        )
        spot_target = target.flatten(2).transpose(1, 2).reshape(
            batch * points, genes
        )
        spot_mask = flat_mask.transpose(1, 2).expand(-1, -1, genes).reshape(
            batch * points, genes
        )
    return (
        _masked_correlations(gene_prediction, gene_target, gene_mask),
        _masked_correlations(spot_prediction, spot_target, spot_mask),
    )


def visium_prediction(
    model, batch, full_neighbor, full_xy, return_latents=False
):
    if batch["visible_spots"].shape[0] != 1:
        raise ValueError("compact Visium graph construction requires batch size one")
    if isinstance(full_neighbor, (list, tuple)):
        if "slice_index" not in batch:
            raise ValueError("multi-slice batches require slice_index")
        slice_index = int(batch["slice_index"][0].item())
        full_neighbor = full_neighbor[slice_index]
        full_xy = full_xy[slice_index]
    visible_spots = batch["visible_spots"][0]
    geometry = build_visible_native_neighbor_graph(
        full_neighbor, full_xy, visible_spots
    )
    prediction, auxiliary = model(
        batch["visible_expression"],
        batch["visible_coord"],
        batch["query_coord"],
        geometry,
        return_auxiliary=True,
    )
    target = batch["target_values"]
    mask = torch.ones(
        (*target.shape[:2], 1), dtype=torch.bool, device=target.device
    )
    result = (prediction, target, mask, auxiliary["baseline"])
    if return_latents:
        return (*result, auxiliary)
    return result


def hd_prediction(model, batch):
    prediction = model(
        batch["inp"],
        coarse_valid_mask=batch["input_mask"],
        fine_valid_mask=batch["target_mask"],
        baseline_scale=batch["baseline_scale"],
    )
    baseline = model._interpolate(
        batch["inp"], model.scale, model.baseline_mode
    ) * batch["baseline_scale"][:, :, None, None]
    baseline = baseline * batch["target_mask"].to(baseline.dtype)
    return prediction, batch["gt"], batch["target_mask"], baseline


def run_epoch(
    task,
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    use_amp=True,
    max_grad_norm=0.0,
    report_baseline=False,
    description="train",
    full_neighbor=None,
    full_xy=None,
    detailed_metrics=True,
    loss_config=None,
):
    if report_baseline and not detailed_metrics:
        raise ValueError("baseline reporting requires detailed metrics")
    training = optimizer is not None
    loss_config = {} if loss_config is None else dict(loss_config)
    loss_mode = loss_config.pop("mode", "smooth_l1")
    gene_weight = loss_config.pop("gene_weight", None)
    if loss_mode not in {"smooth_l1", "structure_aware"}:
        raise ValueError(f"unsupported loss mode: {loss_mode}")
    if loss_config and loss_mode == "smooth_l1":
        raise ValueError("loss weights require loss mode 'structure_aware'")
    if loss_mode == "structure_aware" and task != "visium":
        raise ValueError("structure-aware loss is currently defined for Visium only")
    model.train(training)
    loss_only_sum = torch.zeros((), device=device, dtype=torch.float32)
    loss_only_count = torch.zeros((), device=device, dtype=torch.long)
    totals = {
        "smooth_l1_sum": 0.0, "squared_error_sum": 0.0,
        "absolute_error_sum": 0.0, "element_count": 0,
        "positive_squared_error_sum": 0.0,
        "positive_absolute_error_sum": 0.0, "positive_count": 0,
        "negative_prediction_count": 0, "near_zero_prediction_count": 0,
        "gene_value_sum": None, "gene_value_square_sum": None,
        "gene_value_count": None,
        "gene_pearson_sum": 0.0, "gene_pearson_count": 0,
        "spot_pearson_sum": 0.0, "spot_pearson_count": 0,
        "baseline_smooth_l1_sum": 0.0, "baseline_element_count": 0,
        "baseline_squared_error_sum": 0.0,
        "baseline_absolute_error_sum": 0.0,
        "baseline_gene_pearson_sum": 0.0,
        "baseline_gene_pearson_count": 0,
        "baseline_spot_pearson_sum": 0.0,
        "baseline_spot_pearson_count": 0,
        "batches": 0,
        "objective_sum": 0.0,
        "gene_correlation_loss_sum": 0.0,
        "variance_loss_sum": 0.0,
        "negative_loss_sum": 0.0,
        "structure_valid_gene_count": 0,
    }
    gradient_context = torch.enable_grad if training else torch.no_grad
    with gradient_context():
        progress = tqdm(loader, desc=description, leave=False)
        for cpu_batch in progress:
            batch = move_batch(cpu_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if task == "visium":
                    prediction, target, mask, baseline = visium_prediction(
                        model, batch, full_neighbor, full_xy
                    )
                else:
                    prediction, target, mask, baseline = hd_prediction(model, batch)
                if loss_mode == "structure_aware":
                    loss, loss_terms = structure_aware_visium_loss(
                        prediction,
                        target,
                        mask,
                        gene_weight=gene_weight,
                        **loss_config,
                    )
                else:
                    loss = masked_smooth_l1(
                        prediction, target, mask, gene_weight=gene_weight
                    )
                    loss_terms = None
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"{description}: non-finite loss; prediction_finite="
                    f"{torch.isfinite(prediction).all().item()}"
                )
            if training:
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if not detailed_metrics:
                if loss_mode == "structure_aware":
                    loss_only_sum += loss.detach().float()
                    loss_only_count += 1
                else:
                    expansion = prediction.numel() // mask.numel()
                    element_count = mask.count_nonzero() * expansion
                    loss_only_sum += loss.detach().float() * element_count
                    loss_only_count += element_count
                continue

            expanded_mask = mask.bool().expand_as(prediction)
            error = (prediction.detach().float() - target.float())[expanded_mask]
            selected_prediction = prediction.detach().float()[expanded_mask]
            selected_target = target.float()[expanded_mask]
            positive = selected_target > 0
            genes, spots = correlation_values(
                prediction.detach(), target, mask, task
            )
            element_count = error.numel()
            totals["smooth_l1_sum"] += float(F.smooth_l1_loss(
                selected_prediction, selected_target, reduction="sum"
            ))
            totals["squared_error_sum"] += float(error.square().sum())
            totals["absolute_error_sum"] += float(error.abs().sum())
            totals["element_count"] += element_count
            if positive.any():
                positive_error = error[positive]
                totals["positive_squared_error_sum"] += float(
                    positive_error.square().sum()
                )
                totals["positive_absolute_error_sum"] += float(
                    positive_error.abs().sum()
                )
                totals["positive_count"] += positive_error.numel()
            totals["negative_prediction_count"] += int(
                (selected_prediction < 0).sum()
            )
            totals["near_zero_prediction_count"] += int(
                (selected_prediction.abs() < 1e-8).sum()
            )
            gene_sum, gene_square_sum, gene_count = _gene_value_moments(
                prediction, mask, task
            )
            if totals["gene_value_sum"] is None:
                totals["gene_value_sum"] = gene_sum
                totals["gene_value_square_sum"] = gene_square_sum
                totals["gene_value_count"] = gene_count
            else:
                totals["gene_value_sum"] += gene_sum
                totals["gene_value_square_sum"] += gene_square_sum
                totals["gene_value_count"] += gene_count
            totals["gene_pearson_sum"] += float(genes.sum())
            totals["gene_pearson_count"] += genes.numel()
            totals["spot_pearson_sum"] += float(spots.sum())
            totals["spot_pearson_count"] += spots.numel()

            if report_baseline:
                selected_baseline = baseline.float()[expanded_mask]
                baseline_genes, baseline_spots = correlation_values(
                    baseline, target, mask, task
                )
                totals["baseline_smooth_l1_sum"] += float(F.smooth_l1_loss(
                    selected_baseline, selected_target, reduction="sum"
                ))
                baseline_error = selected_baseline - selected_target
                totals["baseline_squared_error_sum"] += float(
                    baseline_error.square().sum()
                )
                totals["baseline_absolute_error_sum"] += float(
                    baseline_error.abs().sum()
                )
                totals["baseline_element_count"] += element_count
                totals["baseline_gene_pearson_sum"] += float(
                    baseline_genes.sum()
                )
                totals["baseline_gene_pearson_count"] += baseline_genes.numel()
                totals["baseline_spot_pearson_sum"] += float(
                    baseline_spots.sum()
                )
                totals["baseline_spot_pearson_count"] += baseline_spots.numel()
            totals["batches"] += 1
            totals["objective_sum"] += float(loss.detach())
            if loss_terms is not None:
                totals["gene_correlation_loss_sum"] += float(
                    loss_terms["gene_correlation"].detach()
                )
                totals["variance_loss_sum"] += float(
                    loss_terms["variance"].detach()
                )
                totals["negative_loss_sum"] += float(
                    loss_terms["negative"].detach()
                )
                totals["structure_valid_gene_count"] += int(
                    loss_terms["valid_genes"].detach()
                )
            progress.set_postfix(
                loss=(
                    f"{totals['smooth_l1_sum'] / max(totals['element_count'], 1):.4f}"
                )
            )

    if not detailed_metrics:
        pooled_loss = loss_only_sum / loss_only_count.clamp_min(1)
        return {"loss": float(pooled_loss)}

    element_count = max(totals["element_count"], 1)
    positive_count = max(totals["positive_count"], 1)
    pooled_loss = totals["smooth_l1_sum"] / element_count
    gene_count = totals["gene_value_count"]
    if gene_count is None:
        prediction_variance = 0.0
    else:
        valid_gene = gene_count >= 2
        gene_mean = totals["gene_value_sum"] / gene_count.clamp_min(1.0)
        gene_variance = (
            totals["gene_value_square_sum"] / gene_count.clamp_min(1.0)
            - gene_mean.square()
        ).clamp_min(0.0)
        prediction_variance = float(
            gene_variance[valid_gene].mean() if valid_gene.any() else 0.0
        )
    objective_loss = (
        totals["objective_sum"] / max(totals["batches"], 1)
        if loss_mode == "structure_aware" or gene_weight is not None
        else pooled_loss
    )
    metrics = {
        "loss": objective_loss,
        "reconstruction": pooled_loss,
        "rmse": (totals["squared_error_sum"] / element_count) ** 0.5,
        "mae": totals["absolute_error_sum"] / element_count,
        "positive_rmse": (
            totals["positive_squared_error_sum"] / positive_count
        ) ** 0.5,
        "positive_mae": (
            totals["positive_absolute_error_sum"] / positive_count
        ),
        "positive_count": totals["positive_count"],
        "negative_fraction": (
            totals["negative_prediction_count"] / element_count
        ),
        "near_zero_fraction": (
            totals["near_zero_prediction_count"] / element_count
        ),
        "prediction_variance": prediction_variance,
        "element_count": totals["element_count"],
    }
    metrics.update({
        "gene_pearson": totals["gene_pearson_sum"]
        / max(totals["gene_pearson_count"], 1),
        "gene_pearson_valid": totals["gene_pearson_count"],
        "spot_pearson": totals["spot_pearson_sum"]
        / max(totals["spot_pearson_count"], 1),
        "spot_pearson_valid": totals["spot_pearson_count"],
    })
    if loss_mode == "structure_aware":
        batches = max(totals["batches"], 1)
        metrics.update({
            "gene_correlation_loss": (
                totals["gene_correlation_loss_sum"] / batches
            ),
            "variance_loss": totals["variance_loss_sum"] / batches,
            "negative_loss": totals["negative_loss_sum"] / batches,
            "structure_valid_gene_count": totals[
                "structure_valid_gene_count"
            ],
        })
    if report_baseline:
        baseline_reconstruction = totals["baseline_smooth_l1_sum"] / max(
            totals["baseline_element_count"], 1
        )
        baseline_gene = totals["baseline_gene_pearson_sum"] / max(
            totals["baseline_gene_pearson_count"], 1
        )
        baseline_spot = totals["baseline_spot_pearson_sum"] / max(
            totals["baseline_spot_pearson_count"], 1
        )
        baseline_element_count = max(totals["baseline_element_count"], 1)
        baseline_rmse = (
            totals["baseline_squared_error_sum"] / baseline_element_count
        ) ** 0.5
        baseline_mae = (
            totals["baseline_absolute_error_sum"] / baseline_element_count
        )
        metrics.update({
            "baseline_reconstruction": baseline_reconstruction,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
            "baseline_gene_pearson": baseline_gene,
            "baseline_gene_pearson_valid": totals[
                "baseline_gene_pearson_count"
            ],
            "baseline_spot_pearson": baseline_spot,
            "baseline_spot_pearson_valid": totals[
                "baseline_spot_pearson_count"
            ],
            "reconstruction_gain": baseline_reconstruction
            - metrics["reconstruction"],
            "gene_pearson_gain": metrics["gene_pearson"] - baseline_gene,
            "spot_pearson_gain": metrics["spot_pearson"] - baseline_spot,
            "idw_reconstruction": baseline_reconstruction,
            "idw_rmse": baseline_rmse,
            "idw_mae": baseline_mae,
            "idw_gene_pearson": baseline_gene,
            "idw_spot_pearson": baseline_spot,
        })
    return metrics


@torch.no_grad()
def collect_predictions(task, model, loader, device, use_amp, full_neighbor, full_xy):
    model.eval()
    collected = {"prediction": [], "truth": [], "baseline": [], "mask": []}
    target_spots = []
    origins = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if task == "visium":
                prediction, target, mask, baseline = visium_prediction(
                    model, batch, full_neighbor, full_xy
                )
            else:
                prediction, target, mask, baseline = hd_prediction(model, batch)
        for key, value in (
            ("prediction", prediction), ("truth", target),
            ("baseline", baseline), ("mask", mask),
        ):
            collected[key].append(value.float().cpu().numpy())
        if "origin" in batch:
            origins.append(batch["origin"].cpu().numpy())
        if "target_spots" in batch:
            target_spots.append(batch["target_spots"].cpu().numpy())
    result = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    if origins:
        result["origins"] = np.concatenate(origins, axis=0)
    if target_spots:
        result["target_spots"] = np.concatenate(target_spots, axis=0)
    return result
