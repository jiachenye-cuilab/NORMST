"""Optimization and evaluation for composition-latent AE-NORMST."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.geometry_adaptive_normst import build_visible_native_neighbor_graph
from training.engine import masked_smooth_l1, move_batch


def _select_slice(values, slice_index: int, label: str):
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be a per-slice list")
    try:
        return values[slice_index]
    except IndexError as error:
        raise IndexError(f"invalid slice_index={slice_index} for {label}") from error


def ae_visium_prediction(
    model,
    batch,
    full_neighbors,
    full_xy,
    full_composition,
    full_library_context,
    full_counts,
    library_context_mode: str,
):
    """Build one masked-spot forward pass without exposing hidden library size."""
    if library_context_mode not in {"zero", "visible"}:
        raise ValueError("library_context_mode must be zero or visible")
    if batch["visible_spots"].shape[0] != 1:
        raise ValueError("AE-NORMST compact graphs require batch size one")
    if "slice_index" not in batch:
        raise ValueError("AE-NORMST batches require slice_index")
    slice_index = int(batch["slice_index"][0].item())
    neighbor = _select_slice(full_neighbors, slice_index, "full_neighbors")
    xy = _select_slice(full_xy, slice_index, "full_xy")
    composition = _select_slice(
        full_composition, slice_index, "full_composition"
    )
    library_context = _select_slice(
        full_library_context, slice_index, "full_library_context"
    )
    counts = _select_slice(full_counts, slice_index, "full_counts")

    visible_spots = batch["visible_spots"][0].to(
        device=neighbor.device, dtype=torch.long, non_blocking=True
    )
    target_spots = batch["target_spots"][0].to(
        device=neighbor.device, dtype=torch.long, non_blocking=True
    )
    if bool(torch.isin(visible_spots, target_spots).any()):
        raise RuntimeError("visible and target spots overlap")

    visible_composition = composition.index_select(
        0, visible_spots
    ).unsqueeze(0)
    target = composition.index_select(0, target_spots).unsqueeze(0)
    visible_xy = xy.index_select(0, visible_spots).unsqueeze(0)
    query_xy = xy.index_select(0, target_spots).unsqueeze(0)
    if library_context_mode == "visible":
        visible_context = library_context.index_select(0, visible_spots)
    else:
        visible_context = torch.zeros(
            (visible_spots.numel(), 1),
            device=composition.device,
            dtype=composition.dtype,
        )
    if visible_context.ndim == 1:
        visible_context = visible_context[:, None]
    visible_context = visible_context.unsqueeze(0)
    target_counts = counts.index_select(0, target_spots).unsqueeze(0)

    geometry = build_visible_native_neighbor_graph(
        neighbor, xy, visible_spots, validate_indices=False
    )
    prediction, auxiliary = model(
        visible_composition,
        visible_context,
        visible_xy,
        query_xy,
        geometry,
        return_auxiliary=True,
    )
    if auxiliary.get("hidden_library_used") is not False:
        raise RuntimeError("model did not preserve the hidden-library contract")
    mask = torch.ones(
        (*target.shape[:2], 1), dtype=torch.bool, device=target.device
    )
    return prediction, target, mask, auxiliary["baseline"], target_counts


def _masked_correlations(prediction, target, mask, epsilon=1e-8):
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
    return numerator / denominator, valid


def _correlation_sums(prediction, target, mask):
    batch, points, dimensions = prediction.shape
    gene_values, gene_valid = _masked_correlations(
        prediction.transpose(1, 2).reshape(batch * dimensions, points),
        target.transpose(1, 2).reshape(batch * dimensions, points),
        mask.transpose(1, 2).expand(-1, dimensions, -1).reshape(
            batch * dimensions, points
        ),
    )
    spot_values, spot_valid = _masked_correlations(
        prediction.reshape(batch * points, dimensions),
        target.reshape(batch * points, dimensions),
        mask.expand(-1, -1, dimensions).reshape(batch * points, dimensions),
    )
    return {
        "gene_sum": float(gene_values[gene_valid].sum()),
        "gene_count": int(gene_valid.sum()),
        "spot_sum": float(spot_values[spot_valid].sum()),
        "spot_count": int(spot_valid.sum()),
        "gene_values": gene_values,
        "gene_valid": gene_valid,
        "spot_values": spot_values,
        "spot_valid": spot_valid,
    }


def _decoded_statistics(frozen_ae, latent, target_counts, mask):
    probability = frozen_ae.decode_standardized(latent)["probability"].float()
    target = frozen_ae.target_composition(target_counts).float()
    valid_spot = target_counts.sum(dim=-1) > 0
    if not bool(valid_spot.any()):
        raise ValueError("decoded composition metrics require non-empty target spots")
    log_probability = probability.clamp_min(1e-8).log()
    log_target = target.clamp_min(1e-8).log()
    ce = -(target * log_probability).sum(dim=-1)
    kl = (target * (log_target - log_probability)).sum(dim=-1)
    l1 = (probability - target).abs().sum(dim=-1)
    decoded_mask = mask & valid_spot[..., None]
    correlations = _correlation_sums(probability, target, decoded_mask)
    return {
        "ce_sum": float(ce[valid_spot].sum()),
        "kl_sum": float(kl[valid_spot].sum()),
        "l1_sum": float(l1[valid_spot].sum()),
        "spot_count": int(valid_spot.sum()),
        "gene_pearson_sum": correlations["gene_sum"],
        "gene_pearson_count": correlations["gene_count"],
        "spot_pearson_sum": correlations["spot_sum"],
        "spot_pearson_count": correlations["spot_count"],
    }


def _empty_totals():
    prefixes = ("", "baseline_")
    totals = {
        "objective_sum": 0.0,
        "objective_count": 0,
        "objective_latent_sum": 0.0,
        "objective_composition_ce_sum": 0.0,
        "latent_smooth_l1_sum": 0.0,
        "latent_squared_error_sum": 0.0,
        "latent_absolute_error_sum": 0.0,
        "latent_element_count": 0,
        "prediction_sum": None,
        "prediction_square_sum": None,
        "prediction_count": 0,
    }
    for prefix in prefixes:
        totals.update({
            f"{prefix}gene_pearson_sum": 0.0,
            f"{prefix}gene_pearson_count": 0,
            f"{prefix}spot_pearson_sum": 0.0,
            f"{prefix}spot_pearson_count": 0,
            f"{prefix}decoded_ce_sum": 0.0,
            f"{prefix}decoded_kl_sum": 0.0,
            f"{prefix}decoded_l1_sum": 0.0,
            f"{prefix}decoded_spot_count": 0,
            f"{prefix}decoded_gene_pearson_sum": 0.0,
            f"{prefix}decoded_gene_pearson_count": 0,
            f"{prefix}decoded_spot_pearson_sum": 0.0,
            f"{prefix}decoded_spot_pearson_count": 0,
        })
    totals.update({
        "baseline_latent_smooth_l1_sum": 0.0,
        "baseline_latent_squared_error_sum": 0.0,
        "baseline_latent_absolute_error_sum": 0.0,
        "baseline_latent_element_count": 0,
        "paired_gene_prediction_sum": 0.0,
        "paired_gene_baseline_sum": 0.0,
        "paired_gene_count": 0,
        "paired_spot_prediction_sum": 0.0,
        "paired_spot_baseline_sum": 0.0,
        "paired_spot_count": 0,
    })
    return totals


def _add_decoded(totals, prefix, values):
    for name in (
        "ce_sum", "kl_sum", "l1_sum", "spot_count",
        "gene_pearson_sum", "gene_pearson_count",
        "spot_pearson_sum", "spot_pearson_count",
    ):
        totals[f"{prefix}decoded_{name}"] += values[name]


def _mean(total, count):
    return float(total) / max(int(count), 1)


def run_ae_epoch(
    model,
    frozen_ae,
    loader,
    device,
    full_neighbors,
    full_xy,
    full_composition,
    full_library_context,
    full_counts,
    library_context_mode,
    composition_loss_weight=0.1,
    optimizer=None,
    scaler=None,
    use_amp=True,
    max_grad_norm=0.0,
    report_baseline=False,
    detailed_metrics=True,
    description="train",
):
    """Run an epoch with latent SmoothL1 and optional decoded composition CE."""
    if composition_loss_weight < 0 or not math.isfinite(composition_loss_weight):
        raise ValueError("composition_loss_weight must be finite and non-negative")
    if report_baseline and not detailed_metrics:
        raise ValueError("baseline reporting requires detailed metrics")
    training = optimizer is not None
    if training and scaler is None:
        raise ValueError("training requires a GradScaler")
    model.train(training)
    frozen_ae.eval()
    totals = _empty_totals()
    gradient_context = torch.enable_grad if training else torch.no_grad
    with gradient_context():
        progress = tqdm(loader, desc=description, leave=False)
        for cpu_batch in progress:
            batch = move_batch(cpu_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction, target, mask, baseline, target_counts = (
                    ae_visium_prediction(
                        model, batch, full_neighbors, full_xy,
                        full_composition, full_library_context, full_counts,
                        library_context_mode,
                    )
                )
                latent_loss = masked_smooth_l1(prediction, target, mask)
                composition_ce, _ = frozen_ae.composition_cross_entropy(
                    prediction, target_counts
                )
                loss = latent_loss + composition_loss_weight * composition_ce
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{description}: non-finite AE-NORMST loss")
            if training:
                scaler.scale(loss).backward()
                if any(parameter.grad is not None for parameter in frozen_ae.parameters()):
                    raise RuntimeError("frozen AE accumulated gradients")
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            totals["objective_sum"] += float(loss.detach())
            totals["objective_latent_sum"] += float(latent_loss.detach())
            totals["objective_composition_ce_sum"] += float(
                composition_ce.detach()
            )
            totals["objective_count"] += 1
            progress.set_postfix(loss=f"{_mean(totals['objective_sum'], totals['objective_count']):.4f}")
            if not detailed_metrics:
                continue

            expanded = mask.expand_as(prediction)
            selected_prediction = prediction.detach().float()[expanded]
            selected_target = target.float()[expanded]
            error = selected_prediction - selected_target
            elements = error.numel()
            totals["latent_smooth_l1_sum"] += float(F.smooth_l1_loss(
                selected_prediction, selected_target, reduction="sum"
            ))
            totals["latent_squared_error_sum"] += float(error.square().sum())
            totals["latent_absolute_error_sum"] += float(error.abs().sum())
            totals["latent_element_count"] += elements
            dimension_sum = prediction.detach().float().sum(dim=(0, 1)).cpu()
            dimension_square_sum = prediction.detach().float().square().sum(
                dim=(0, 1)
            ).cpu()
            if totals["prediction_sum"] is None:
                totals["prediction_sum"] = dimension_sum
                totals["prediction_square_sum"] = dimension_square_sum
            else:
                totals["prediction_sum"] += dimension_sum
                totals["prediction_square_sum"] += dimension_square_sum
            totals["prediction_count"] += prediction.shape[0] * prediction.shape[1]

            latent_correlations = _correlation_sums(
                prediction.detach(), target, mask
            )
            totals["gene_pearson_sum"] += latent_correlations["gene_sum"]
            totals["gene_pearson_count"] += latent_correlations["gene_count"]
            totals["spot_pearson_sum"] += latent_correlations["spot_sum"]
            totals["spot_pearson_count"] += latent_correlations["spot_count"]
            decoded = _decoded_statistics(
                frozen_ae, prediction.detach(), target_counts, mask
            )
            _add_decoded(totals, "", decoded)

            if report_baseline:
                selected_baseline = baseline.detach().float()[expanded]
                baseline_error = selected_baseline - selected_target
                totals["baseline_latent_smooth_l1_sum"] += float(
                    F.smooth_l1_loss(
                        selected_baseline, selected_target, reduction="sum"
                    )
                )
                totals["baseline_latent_squared_error_sum"] += float(
                    baseline_error.square().sum()
                )
                totals["baseline_latent_absolute_error_sum"] += float(
                    baseline_error.abs().sum()
                )
                totals["baseline_latent_element_count"] += elements
                baseline_correlations = _correlation_sums(baseline, target, mask)
                totals["baseline_gene_pearson_sum"] += baseline_correlations["gene_sum"]
                totals["baseline_gene_pearson_count"] += baseline_correlations["gene_count"]
                totals["baseline_spot_pearson_sum"] += baseline_correlations["spot_sum"]
                totals["baseline_spot_pearson_count"] += baseline_correlations["spot_count"]
                gene_valid = (
                    latent_correlations["gene_valid"]
                    & baseline_correlations["gene_valid"]
                )
                spot_valid = (
                    latent_correlations["spot_valid"]
                    & baseline_correlations["spot_valid"]
                )
                totals["paired_gene_prediction_sum"] += float(
                    latent_correlations["gene_values"][gene_valid].sum()
                )
                totals["paired_gene_baseline_sum"] += float(
                    baseline_correlations["gene_values"][gene_valid].sum()
                )
                totals["paired_gene_count"] += int(gene_valid.sum())
                totals["paired_spot_prediction_sum"] += float(
                    latent_correlations["spot_values"][spot_valid].sum()
                )
                totals["paired_spot_baseline_sum"] += float(
                    baseline_correlations["spot_values"][spot_valid].sum()
                )
                totals["paired_spot_count"] += int(spot_valid.sum())
                baseline_decoded = _decoded_statistics(
                    frozen_ae, baseline, target_counts, mask
                )
                _add_decoded(totals, "baseline_", baseline_decoded)

    if not totals["objective_count"]:
        raise ValueError(f"{description}: empty loader")
    if not detailed_metrics:
        return {
            "loss": _mean(totals["objective_sum"], totals["objective_count"]),
            "reconstruction": _mean(
                totals["objective_latent_sum"], totals["objective_count"]
            ),
            "decoded_composition_ce": _mean(
                totals["objective_composition_ce_sum"],
                totals["objective_count"],
            ),
        }

    element_count = totals["latent_element_count"]
    spot_count = totals["decoded_spot_count"]
    prediction_count = max(totals["prediction_count"], 1)
    mean = totals["prediction_sum"] / prediction_count
    variance = (
        totals["prediction_square_sum"] / prediction_count - mean.square()
    ).clamp_min(0.0)
    metrics = {
        "loss": _mean(totals["objective_sum"], totals["objective_count"]),
        "reconstruction": _mean(totals["latent_smooth_l1_sum"], element_count),
        "rmse": math.sqrt(_mean(totals["latent_squared_error_sum"], element_count)),
        "mae": _mean(totals["latent_absolute_error_sum"], element_count),
        "prediction_variance": float(variance.mean()),
        "element_count": element_count,
        "gene_pearson": _mean(totals["gene_pearson_sum"], totals["gene_pearson_count"]),
        "gene_pearson_valid": totals["gene_pearson_count"],
        "spot_pearson": _mean(totals["spot_pearson_sum"], totals["spot_pearson_count"]),
        "spot_pearson_valid": totals["spot_pearson_count"],
        "decoded_composition_ce": _mean(totals["decoded_ce_sum"], spot_count),
        "decoded_composition_kl": _mean(totals["decoded_kl_sum"], spot_count),
        "decoded_composition_l1": _mean(totals["decoded_l1_sum"], spot_count),
        "decoded_spot_count": spot_count,
        "decoded_gene_pearson": _mean(
            totals["decoded_gene_pearson_sum"], totals["decoded_gene_pearson_count"]
        ),
        "decoded_gene_pearson_valid": totals["decoded_gene_pearson_count"],
        "decoded_spot_pearson": _mean(
            totals["decoded_spot_pearson_sum"], totals["decoded_spot_pearson_count"]
        ),
        "decoded_spot_pearson_valid": totals["decoded_spot_pearson_count"],
    }
    if report_baseline:
        baseline_elements = totals["baseline_latent_element_count"]
        baseline_spots = totals["baseline_decoded_spot_count"]
        paired_gene = _mean(totals["paired_gene_prediction_sum"], totals["paired_gene_count"])
        paired_baseline_gene = _mean(totals["paired_gene_baseline_sum"], totals["paired_gene_count"])
        paired_spot = _mean(totals["paired_spot_prediction_sum"], totals["paired_spot_count"])
        paired_baseline_spot = _mean(totals["paired_spot_baseline_sum"], totals["paired_spot_count"])
        metrics.update({
            "baseline_reconstruction": _mean(
                totals["baseline_latent_smooth_l1_sum"], baseline_elements
            ),
            "baseline_rmse": math.sqrt(_mean(
                totals["baseline_latent_squared_error_sum"], baseline_elements
            )),
            "baseline_mae": _mean(
                totals["baseline_latent_absolute_error_sum"], baseline_elements
            ),
            "baseline_gene_pearson": _mean(
                totals["baseline_gene_pearson_sum"], totals["baseline_gene_pearson_count"]
            ),
            "baseline_gene_pearson_valid": totals["baseline_gene_pearson_count"],
            "baseline_spot_pearson": _mean(
                totals["baseline_spot_pearson_sum"], totals["baseline_spot_pearson_count"]
            ),
            "baseline_spot_pearson_valid": totals["baseline_spot_pearson_count"],
            "gene_pearson_paired": paired_gene,
            "baseline_gene_pearson_paired": paired_baseline_gene,
            "gene_pearson_paired_valid": totals["paired_gene_count"],
            "spot_pearson_paired": paired_spot,
            "baseline_spot_pearson_paired": paired_baseline_spot,
            "spot_pearson_paired_valid": totals["paired_spot_count"],
            "reconstruction_gain": _mean(
                totals["baseline_latent_smooth_l1_sum"], baseline_elements
            ) - metrics["reconstruction"],
            "gene_pearson_gain": paired_gene - paired_baseline_gene,
            "spot_pearson_gain": paired_spot - paired_baseline_spot,
            "baseline_decoded_composition_ce": _mean(
                totals["baseline_decoded_ce_sum"], baseline_spots
            ),
            "baseline_decoded_composition_kl": _mean(
                totals["baseline_decoded_kl_sum"], baseline_spots
            ),
            "baseline_decoded_composition_l1": _mean(
                totals["baseline_decoded_l1_sum"], baseline_spots
            ),
            "baseline_decoded_gene_pearson": _mean(
                totals["baseline_decoded_gene_pearson_sum"],
                totals["baseline_decoded_gene_pearson_count"],
            ),
            "baseline_decoded_spot_pearson": _mean(
                totals["baseline_decoded_spot_pearson_sum"],
                totals["baseline_decoded_spot_pearson_count"],
            ),
            "decoded_composition_ce_gain": _mean(
                totals["baseline_decoded_ce_sum"], baseline_spots
            ) - metrics["decoded_composition_ce"],
            "decoded_composition_kl_gain": _mean(
                totals["baseline_decoded_kl_sum"], baseline_spots
            ) - metrics["decoded_composition_kl"],
        })
    return metrics


@torch.no_grad()
def collect_ae_predictions(
    model,
    loader,
    device,
    use_amp,
    full_neighbors,
    full_xy,
    full_composition,
    full_library_context,
    full_counts,
    library_context_mode,
):
    """Collect compact latent predictions; decoded 1000-D arrays are omitted."""
    model.eval()
    result = {"prediction": [], "truth": [], "baseline": [], "target_spots": []}
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction, target, _, baseline, _ = ae_visium_prediction(
                model, batch, full_neighbors, full_xy, full_composition,
                full_library_context, full_counts, library_context_mode,
            )
        result["prediction"].append(prediction.float().cpu().numpy())
        result["truth"].append(target.float().cpu().numpy())
        result["baseline"].append(baseline.float().cpu().numpy())
        result["target_spots"].append(
            batch["target_spots"].cpu().numpy()
        )
    return {
        key: np.concatenate(values, axis=0) for key, values in result.items()
    }


__all__ = ["ae_visium_prediction", "collect_ae_predictions", "run_ae_epoch"]
