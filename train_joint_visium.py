"""Train joint multi-gene NORMST on masked spots from one Visium slice.

One sample is one random whole-spot mask. Every gene at a target spot is
hidden together, and the model predicts all selected genes in one forward
pass. This is a within-slide feasibility path, not yet multi-slice training.
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

from datasets.joint_masked_visium import JointMaskedVisiumDataset
from datasets.masked_visium import prepare_masked_visium
from models.joint_sronet_st import JointSTSRNO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--observed-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--train-target-fraction", type=float, default=0.25)
    parser.add_argument("--masks-per-epoch", type=int, default=64)
    parser.add_argument("--idw-neighbors", type=int, default=6)
    parser.add_argument("--physical-query-neighbors", type=int, default=6)
    parser.add_argument(
        "--physical-query-candidate-multiplier", type=int, default=8
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience in validation epochs; zero disables it",
    )
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--operator-layers", type=int, default=2)
    parser.add_argument("--encoder-blocks", type=int, default=16)
    parser.add_argument("--encoder-channels", type=int, default=64)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument("--hex-residual-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


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


def gather_targets(prediction, target_indices):
    if prediction.ndim != 4 or target_indices.ndim != 2:
        raise ValueError("prediction and target indices have invalid shapes")
    index = target_indices[:, None, :].expand(
        -1, prediction.shape[1], -1
    )
    return torch.gather(prediction.flatten(2), 2, index)


def correlation_values(prediction, target, axis=-1, epsilon=1e-8):
    prediction = prediction.float()
    target = target.float()
    prediction = prediction - prediction.mean(dim=axis, keepdim=True)
    target = target - target.mean(dim=axis, keepdim=True)
    numerator = (prediction * target).sum(dim=axis)
    prediction_energy = prediction.square().sum(dim=axis)
    target_energy = target.square().sum(dim=axis)
    valid = (prediction_energy > epsilon) & (target_energy > epsilon)
    denominator = torch.sqrt(
        prediction_energy.clamp_min(epsilon)
        * target_energy.clamp_min(epsilon)
    )
    return (numerator / denominator)[valid]


def reconstruction_metrics(prediction, target):
    prediction = prediction.float()
    target = target.float()
    error = prediction - target
    positive = target > 0
    metrics = {
        "reconstruction": F.smooth_l1_loss(prediction, target),
        "rmse": error.square().mean().sqrt(),
        "mae": error.abs().mean(),
        "negative_fraction": (prediction < 0).float().mean(),
        "near_zero_fraction": (prediction.abs() < 1e-8).float().mean(),
        "prediction_variance": prediction.var(
            dim=-1, unbiased=False
        ).mean(),
    }
    if positive.any():
        metrics["positive_rmse"] = error[positive].square().mean().sqrt()
        metrics["positive_mae"] = error[positive].abs().mean()
    else:
        zero = prediction.sum() * 0.0
        metrics["positive_rmse"] = zero
        metrics["positive_mae"] = zero
    return metrics


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    use_amp=True,
    report_baseline=False,
    description="train",
):
    training = optimizer is not None
    model.train(training)
    scalar_names = (
        "loss", "reconstruction", "rmse", "mae", "positive_rmse",
        "positive_mae", "negative_fraction", "near_zero_fraction",
        "prediction_variance",
    )
    totals = {name: 0.0 for name in scalar_names}
    gene_pearson_sum = 0.0
    gene_pearson_count = 0
    spot_pearson_sum = 0.0
    spot_pearson_count = 0
    baseline_totals = {
        "baseline_reconstruction": 0.0,
        "baseline_gene_pearson_sum": 0.0,
        "baseline_gene_pearson_count": 0,
        "baseline_spot_pearson_sum": 0.0,
        "baseline_spot_pearson_count": 0,
    }
    samples = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, desc=description, leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                full_prediction = model(
                    batch["inp"], batch["input_mask"]
                )
                prediction = gather_targets(
                    full_prediction, batch["target_indices"]
                )
                target = batch["target_values"]
                values = reconstruction_metrics(prediction, target)
                loss = values["reconstruction"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"{description}: non-finite loss; "
                    f"prediction_finite={torch.isfinite(prediction).all().item()}"
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            detached = prediction.detach()
            gene_pearsons = correlation_values(detached, target, axis=-1)
            spot_pearsons = correlation_values(
                detached.transpose(1, 2), target.transpose(1, 2), axis=-1
            )
            batch_size = prediction.shape[0]
            values["loss"] = loss
            for name in scalar_names:
                totals[name] += float(values[name].item()) * batch_size
            gene_pearson_sum += float(gene_pearsons.sum().item())
            gene_pearson_count += gene_pearsons.numel()
            spot_pearson_sum += float(spot_pearsons.sum().item())
            spot_pearson_count += spot_pearsons.numel()

            if report_baseline:
                baseline = batch["baseline"]
                baseline_values = reconstruction_metrics(baseline, target)
                baseline_gene = correlation_values(baseline, target, axis=-1)
                baseline_spot = correlation_values(
                    baseline.transpose(1, 2),
                    target.transpose(1, 2),
                    axis=-1,
                )
                baseline_totals["baseline_reconstruction"] += (
                    float(baseline_values["reconstruction"].item())
                    * batch_size
                )
                baseline_totals["baseline_gene_pearson_sum"] += float(
                    baseline_gene.sum().item()
                )
                baseline_totals["baseline_gene_pearson_count"] += (
                    baseline_gene.numel()
                )
                baseline_totals["baseline_spot_pearson_sum"] += float(
                    baseline_spot.sum().item()
                )
                baseline_totals["baseline_spot_pearson_count"] += (
                    baseline_spot.numel()
                )

            samples += batch_size
            progress.set_postfix(
                loss=f"{totals['loss'] / max(samples, 1):.4f}"
            )

    metrics = {
        name: value / max(samples, 1) for name, value in totals.items()
    }
    metrics["gene_pearson"] = gene_pearson_sum / max(gene_pearson_count, 1)
    metrics["gene_pearson_valid"] = gene_pearson_count
    metrics["spot_pearson"] = spot_pearson_sum / max(spot_pearson_count, 1)
    metrics["spot_pearson_valid"] = spot_pearson_count
    if report_baseline:
        baseline_reconstruction = (
            baseline_totals["baseline_reconstruction"] / max(samples, 1)
        )
        baseline_gene = (
            baseline_totals["baseline_gene_pearson_sum"]
            / max(baseline_totals["baseline_gene_pearson_count"], 1)
        )
        baseline_spot = (
            baseline_totals["baseline_spot_pearson_sum"]
            / max(baseline_totals["baseline_spot_pearson_count"], 1)
        )
        metrics.update({
            "baseline_reconstruction": baseline_reconstruction,
            "baseline_gene_pearson": baseline_gene,
            "baseline_gene_pearson_valid": baseline_totals[
                "baseline_gene_pearson_count"
            ],
            "baseline_spot_pearson": baseline_spot,
            "baseline_spot_pearson_valid": baseline_totals[
                "baseline_spot_pearson_count"
            ],
            "reconstruction_gain": (
                baseline_reconstruction - metrics["reconstruction"]
            ),
            "gene_pearson_gain": metrics["gene_pearson"] - baseline_gene,
            "spot_pearson_gain": metrics["spot_pearson"] - baseline_spot,
        })
    return metrics


@torch.no_grad()
def collect_test_predictions(model, loader, device, use_amp):
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        full_prediction = model(batch["inp"], batch["input_mask"])
        prediction = gather_targets(
            full_prediction, batch["target_indices"]
        )
    return {
        "prediction": prediction[0].float().T.cpu().numpy(),
        "truth": batch["target_values"][0].float().T.cpu().numpy(),
        "baseline": batch["baseline"][0].float().T.cpu().numpy(),
        "target_indices": batch["target_indices"][0].cpu().numpy(),
    }


def main():
    args = parse_args()
    if min(
        args.n_genes,
        args.masks_per_epoch,
        args.idw_neighbors,
        args.physical_query_neighbors,
        args.physical_query_candidate_multiplier,
        args.batch_size,
        args.epochs,
        args.width,
        args.num_heads,
        args.encoder_blocks,
        args.encoder_channels,
        args.decoder_hidden,
    ) < 1:
        raise ValueError("model, data, and training sizes must be positive")
    if args.workers < 0 or args.patience < 0:
        raise ValueError("workers and patience must be non-negative")
    if args.min_delta < 0 or args.hex_residual_scale < 0:
        raise ValueError("min_delta and residual scale must be non-negative")
    if not 0.0 < args.train_target_fraction < 1.0:
        raise ValueError("train_target_fraction must be between 0 and 1")

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing joint masked standard Visium data (no histology) ...")
    data = prepare_masked_visium(
        data_dir=args.data_dir,
        count_file=args.count_file,
        n_genes=args.n_genes,
        target_sum=args.target_sum,
        context_dim=0,
        observed_fraction=args.observed_fraction,
        validation_fraction=args.validation_fraction,
        build_physical_query_graph=True,
        physical_query_neighbors=args.physical_query_neighbors,
        physical_query_candidate_multiplier=(
            args.physical_query_candidate_multiplier
        ),
        seed=args.seed,
    )
    print(
        "Spots:",
        f"observed={len(data.observed_spots)},",
        f"val={len(data.validation_spots)}, test={len(data.test_spots)}",
    )

    datasets = {
        split: JointMaskedVisiumDataset(
            data,
            split,
            masks_per_epoch=args.masks_per_epoch,
            train_target_fraction=args.train_target_fraction,
            idw_neighbors=args.idw_neighbors,
            seed=args.seed,
        )
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else 1,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            # Recreate workers after set_epoch so deterministic epoch-specific
            # masks are visible to worker processes.
            persistent_workers=False,
        )
        for split, dataset in datasets.items()
    }

    model = JointSTSRNO(
        n_genes=len(data.genes),
        width=args.width,
        num_heads=args.num_heads,
        num_operator_layers=args.operator_layers,
        encoder_blocks=args.encoder_blocks,
        encoder_channels=args.encoder_channels,
        hex_residual_scale=args.hex_residual_scale,
        physical_query_neighbors=args.physical_query_neighbors,
        decoder_hidden=args.decoder_hidden,
    )
    tissue_mask = (data.row_map >= 0).astype(np.float32)[None]
    model.set_spatial_context(
        torch.from_numpy(tissue_mask),
        torch.from_numpy(data.physical_coord_grid),
        torch.from_numpy(data.row_parity),
        torch.from_numpy(data.physical_query_indices),
        torch.from_numpy(data.physical_query_relative),
        torch.from_numpy(data.physical_query_mask),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config.update({
        "model_type": "joint_multigene",
        "grid_shape": data.shape,
        "n_observed_spots": len(data.observed_spots),
        "n_validation_spots": len(data.validation_spots),
        "n_test_spots": len(data.test_spots),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "optimizer_steps_per_epoch": (
            len(loaders["train"])
        ),
    })
    print(f"Trainable parameters: {config['trainable_parameters']:,}")
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    np.savetxt(args.output_dir / "genes.txt", data.genes, fmt="%s")
    np.savez(
        args.output_dir / "preprocessing.npz",
        genes=data.genes,
        gene_scale=data.gene_scale,
        row_map=data.row_map,
        physical_coord_grid=data.physical_coord_grid,
        row_parity=data.row_parity,
        physical_query_indices=data.physical_query_indices,
        physical_query_relative=data.physical_query_relative,
        physical_query_mask=data.physical_query_mask,
        observed_spots=data.observed_spots,
        validation_spots=data.validation_spots,
        test_spots=data.test_spots,
    )

    history = []
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        datasets["train"].set_epoch(epoch)
        train_started = perf_counter()
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            report_baseline=False,
            description=f"train {epoch + 1}/{args.epochs}",
        )
        train_seconds = perf_counter() - train_started
        val_started = perf_counter()
        val_metrics = run_epoch(
            model,
            loaders["val"],
            device,
            use_amp=use_amp,
            report_baseline=True,
            description=f"val {epoch + 1}/{args.epochs}",
        )
        val_seconds = perf_counter() - val_started
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
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

    best = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(
        model,
        loaders["test"],
        device,
        use_amp=use_amp,
        report_baseline=True,
        description="test",
    )
    test_metrics["best_epoch"] = best["epoch"]
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    predictions = collect_test_predictions(
        model, loaders["test"], device, use_amp
    )
    np.savez(
        args.output_dir / "test_predictions.npz",
        prediction=predictions["prediction"],
        truth=predictions["truth"],
        baseline=predictions["baseline"],
        target_spots=data.test_spots,
        target_flat_indices=predictions["target_indices"],
        genes=data.genes,
    )
    print("Test:", json.dumps(test_metrics))


if __name__ == "__main__":
    main()
