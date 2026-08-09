"""Train the unified geometry-adaptive NORMST models.

Standard Visium example::

    python train_geometry_adaptive_normst.py --task visium \
      --data-dir /data/151673 --output-dir save/151673/unified_seed2027 \
      --seed 2027

Paired Visium HD example::

    python train_geometry_adaptive_normst.py --task visium_hd \
      --lr-dir /data/square_016um --hr-dir /data/square_008um \
      --output-dir save/HBCHD/unified_16_to_8

The standard route removes target spots before model input and restricts the
native six-neighbour graph to the remaining visible spots.  The HD route uses
all selected genes jointly and keeps LR validity separate from the HR target
validity mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.joint_masked_visium import PointJointMaskedVisiumDataset
from datasets.masked_visium import prepare_masked_visium
from datasets.paired_visium_hd import (
    JointPairedVisiumHDDataset,
    prepare_visium_hd_pair,
)
from models.geometry_adaptive_normst import (
    VisiumHDNORMST,
    VisiumNORMST,
    build_native_hex_neighbors,
    build_visible_native_neighbor_graph,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("visium", "visium_hd"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--target-sum", type=float, default=1e4)

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--operator-layers", type=int, default=4)
    parser.add_argument(
        "--operator-mode",
        choices=(
            "local_only", "galerkin_only", "parallel",
            "local_then_global", "global_then_local",
        ),
        default="parallel",
    )
    parser.add_argument("--fusion", choices=("add", "concat"), default="add")
    parser.add_argument("--learnable-alpha", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save all HD test patches; standard Visium test predictions are always saved",
    )

    # Standard Visium options.
    parser.add_argument("--data-dir")
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--observed-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--train-target-fraction", type=float, default=0.25)
    parser.add_argument("--masks-per-epoch", type=int, default=64)
    parser.add_argument("--query-neighbors", type=int, default=6)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--query-chunk-size", type=int, default=1024)

    # Paired Visium HD options.
    parser.add_argument("--lr-dir")
    parser.add_argument("--hr-dir")
    parser.add_argument("--h5-name", default="raw_feature_bc_matrix.h5")
    parser.add_argument(
        "--positions-name", default="spatial/tissue_positions.parquet"
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch-size-lr", type=int, default=16)
    parser.add_argument("--patches-per-epoch", type=int, default=64)
    parser.add_argument("--eval-origin-stride", type=int, default=0)
    parser.add_argument("--eval-max-origins", type=int, default=0)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.1)
    parser.add_argument("--split-axis", choices=("row", "col"), default="col")
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15)
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("nearest", "bilinear", "bicubic"),
        default="bilinear",
    )
    return parser.parse_args(argv)


def validate_args(args):
    positive = (
        args.n_genes, args.width, args.num_heads, args.operator_layers,
        args.epochs, args.batch_size, args.query_neighbors,
        args.query_chunk_size, args.scale, args.patch_size_lr,
        args.patches_per_epoch, args.masks_per_epoch,
    )
    if min(positive) < 1:
        raise ValueError("model, data, and training sizes must be positive")
    if args.width % args.num_heads:
        raise ValueError("width must be divisible by num_heads")
    if args.workers < 0 or args.patience < 0 or args.eval_max_origins < 0:
        raise ValueError("workers, patience, and eval_max_origins must be non-negative")
    if args.min_delta < 0 or args.max_grad_norm < 0:
        raise ValueError("min_delta and max_grad_norm must be non-negative")
    if args.idw_power <= 0:
        raise ValueError("idw_power must be positive")
    if args.task == "visium":
        if not args.data_dir:
            raise ValueError("--data-dir is required for --task visium")
        if args.batch_size != 1:
            raise ValueError(
                "standard Visium currently requires --batch-size 1 because "
                "each random mask has a different compact native graph"
            )
        if not 0.0 < args.train_target_fraction < 1.0:
            raise ValueError("train_target_fraction must be between zero and one")
    elif not args.lr_dir or not args.hr_dir:
        raise ValueError("--lr-dir and --hr-dir are required for --task visium_hd")


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


def masked_smooth_l1(prediction, target, mask):
    expanded = mask.to(prediction.dtype).expand_as(prediction)
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    return (elementwise * expanded).sum() / expanded.sum().clamp_min(1.0)


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


def visium_prediction(model, batch, full_neighbor, full_xy):
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
    prediction = model(
        batch["visible_expression"],
        batch["visible_coord"],
        batch["query_coord"],
        geometry,
    )
    target = batch["target_values"]
    mask = torch.ones(
        (*target.shape[:2], 1), dtype=torch.bool, device=target.device
    )
    return prediction, target, mask, batch["baseline"]


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
):
    training = optimizer is not None
    model.train(training)
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
        "baseline_gene_pearson_sum": 0.0,
        "baseline_gene_pearson_count": 0,
        "baseline_spot_pearson_sum": 0.0,
        "baseline_spot_pearson_count": 0,
        "batches": 0,
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
                loss = masked_smooth_l1(prediction, target, mask)
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
            progress.set_postfix(
                loss=(
                    f"{totals['smooth_l1_sum'] / max(totals['element_count'], 1):.4f}"
                )
            )

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
    metrics = {
        "loss": pooled_loss,
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
        metrics.update({
            "baseline_reconstruction": baseline_reconstruction,
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
        })
    return metrics


def prepare_visium(args, device):
    data = prepare_masked_visium(
        data_dir=args.data_dir,
        count_file=args.count_file,
        n_genes=args.n_genes,
        target_sum=args.target_sum,
        context_dim=0,
        observed_fraction=args.observed_fraction,
        validation_fraction=args.validation_fraction,
        build_physical_query_graph=False,
        seed=args.seed,
    )
    datasets = {
        split: PointJointMaskedVisiumDataset(
            data,
            split,
            masks_per_epoch=args.masks_per_epoch,
            train_target_fraction=args.train_target_fraction,
            idw_neighbors=args.query_neighbors,
            seed=args.seed,
        )
        for split in ("train", "val", "test")
    }
    array_row = torch.from_numpy(data.spot_rows.astype(np.int64))
    row_parity = data.row_parity[data.spot_rows].astype(np.int64)
    array_col_np = data.spot_cols.astype(np.int64) * 2 + row_parity
    array_col = torch.from_numpy(array_col_np)
    full_neighbor_cpu = build_native_hex_neighbors(array_row, array_col)
    full_xy_cpu = torch.from_numpy(data.physical_xy.astype(np.float32))
    model = VisiumNORMST(
        n_genes=len(data.genes),
        width=args.width,
        num_heads=args.num_heads,
        num_layers=args.operator_layers,
        operator_mode=args.operator_mode,
        fusion=args.fusion,
        learnable_alpha=args.learnable_alpha,
        query_neighbors=args.query_neighbors,
        idw_power=args.idw_power,
        query_chunk_size=args.query_chunk_size,
    )
    metadata = {
        "genes": data.genes,
        "gene_scale": data.gene_scale,
        "physical_xy": data.physical_xy,
        "array_row": data.spot_rows,
        "array_col": array_col_np,
        "native_neighbor_index": full_neighbor_cpu.numpy(),
        "observed_spots": data.observed_spots,
        "validation_spots": data.validation_spots,
        "test_spots": data.test_spots,
    }
    config = {
        "n_observed_spots": len(data.observed_spots),
        "n_validation_spots": len(data.validation_spots),
        "n_test_spots": len(data.test_spots),
        "spatial_representation": "compact_native_hex_points",
    }
    return (
        model,
        datasets,
        metadata,
        config,
        full_neighbor_cpu.to(device),
        full_xy_cpu.to(device),
    )


def prepare_hd(args, _device):
    pair = prepare_visium_hd_pair(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        n_genes=args.n_genes,
        scale=args.scale,
        target_sum=args.target_sum,
        split_axis=args.split_axis,
        split_ratios=args.split_ratios,
        h5_name=args.h5_name,
        positions_name=args.positions_name,
        context_dim=0,
    )
    patch = (args.patch_size_lr, args.patch_size_lr)
    eval_stride = args.eval_origin_stride or args.patch_size_lr
    datasets = {
        "train": JointPairedVisiumHDDataset(
            pair, "train", patch, repeat=args.patches_per_epoch,
            min_tissue_fraction=args.min_tissue_fraction,
            deterministic=False, seed=args.seed,
        ),
        "val": JointPairedVisiumHDDataset(
            pair, "val", patch, repeat=1,
            min_tissue_fraction=args.min_tissue_fraction,
            deterministic=True, origin_stride=eval_stride,
            max_origins=args.eval_max_origins, seed=args.seed,
        ),
        "test": JointPairedVisiumHDDataset(
            pair, "test", patch, repeat=1,
            min_tissue_fraction=args.min_tissue_fraction,
            deterministic=True, origin_stride=eval_stride,
            max_origins=args.eval_max_origins, seed=args.seed,
        ),
    }
    model = VisiumHDNORMST(
        n_genes=len(pair.genes),
        width=args.width,
        num_heads=args.num_heads,
        num_layers=args.operator_layers,
        operator_mode=args.operator_mode,
        fusion=args.fusion,
        learnable_alpha=args.learnable_alpha,
        scale=args.scale,
        baseline_mode=args.baseline_mode,
    )
    metadata = {
        "genes": pair.genes,
        "lr_gene_scale": pair.lr_gene_scale,
        "hr_gene_scale": pair.hr_gene_scale,
        "lr_row_map": pair.lr_row_map,
        "hr_row_map": pair.hr_row_map,
        "split_names": np.asarray(list(pair.split_ranges)),
        "split_bounds": np.asarray(list(pair.split_ranges.values())),
    }
    config = {
        "split_ranges": pair.split_ranges,
        "split_axis": pair.split_axis,
        "n_validation_origins": len(datasets["val"].origins),
        "n_test_origins": len(datasets["test"].origins),
        "spatial_representation": "paired_cartesian_grids",
    }
    return model, datasets, metadata, config, None, None


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


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    seed_everything(args.seed)
    requested_device = torch.device(args.device)
    device = (
        torch.device("cpu")
        if requested_device.type == "cuda" and not torch.cuda.is_available()
        else requested_device
    )
    use_amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {args.task} data ...")
    prepared = prepare_visium(args, device) if args.task == "visium" else prepare_hd(args, device)
    model, datasets, preprocessing, task_config, full_neighbor, full_xy = prepared
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else 1,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        for split, dataset in datasets.items()
    }
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config.update(task_config)
    config.update({
        "model": "VisiumNORMST" if args.task == "visium" else "VisiumHDNORMST",
        "n_selected_genes": len(preprocessing["genes"]),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "device_resolved": str(device),
        "amp_enabled": use_amp,
    })
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    np.savetxt(args.output_dir / "genes.txt", preprocessing["genes"], fmt="%s")
    np.savez(args.output_dir / "preprocessing.npz", **preprocessing)
    print(f"Trainable parameters: {config['trainable_parameters']:,}")

    history = []
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        if hasattr(datasets["train"], "set_epoch"):
            datasets["train"].set_epoch(epoch)
        learning_rate_used = optimizer.param_groups[0]["lr"]
        train_started = perf_counter()
        train_metrics = run_epoch(
            args.task, model, loaders["train"], device,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
            description=f"train {epoch + 1}/{args.epochs}",
            full_neighbor=full_neighbor, full_xy=full_xy,
        )
        train_seconds = perf_counter() - train_started
        val_started = perf_counter()
        val_metrics = run_epoch(
            args.task, model, loaders["val"], device,
            use_amp=use_amp, report_baseline=True,
            description=f"val {epoch + 1}/{args.epochs}",
            full_neighbor=full_neighbor, full_xy=full_xy,
        )
        val_seconds = perf_counter() - val_started
        record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate_used,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(record))
        scheduler.step()
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "config": config,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["loss"] < best_val - args.min_delta:
            best_val = val_metrics["loss"]
            stale_epochs = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best validation loss={best_val:.6f}"
            )
            break

    best = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(
        args.task, model, loaders["test"], device,
        use_amp=use_amp, report_baseline=True, description="test",
        full_neighbor=full_neighbor, full_xy=full_xy,
    )
    test_metrics["best_epoch"] = best["epoch"]
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    if args.task == "visium" or args.save_predictions:
        predictions = collect_predictions(
            args.task, model, loaders["test"], device, use_amp,
            full_neighbor, full_xy,
        )
        predictions["genes"] = preprocessing["genes"]
        if args.task == "visium":
            predictions["target_spots"] = preprocessing["test_spots"]
        np.savez(args.output_dir / "test_predictions.npz", **predictions)
    print("Test:", json.dumps(test_metrics))


if __name__ == "__main__":
    main()
