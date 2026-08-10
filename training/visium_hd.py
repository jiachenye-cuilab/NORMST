"""Train VisiumHDNORMST on paired 16-to-8 micrometre Visium HD grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.paired_visium_hd import (
    JointPairedVisiumHDDataset,
    prepare_visium_hd_pair,
)
from models.geometry_adaptive_normst import VisiumHDNORMST
from training.engine import collect_predictions, run_epoch, seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="save every evaluated Visium HD test patch",
    )

    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
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
        args.n_genes, args.target_sum, args.width, args.num_heads,
        args.operator_layers, args.epochs, args.batch_size, args.scale,
        args.patch_size_lr, args.patches_per_epoch,
    )
    if min(positive) <= 0:
        raise ValueError("model, data, and training sizes must be positive")
    if args.width % args.num_heads:
        raise ValueError("width must be divisible by num_heads")
    if args.workers < 0 or args.patience < 0 or args.eval_max_origins < 0:
        raise ValueError(
            "workers, patience, and eval_max_origins must be non-negative"
        )
    if args.eval_origin_stride < 0:
        raise ValueError("eval_origin_stride must be non-negative")
    if args.min_delta < 0 or args.max_grad_norm < 0:
        raise ValueError("min_delta and max_grad_norm must be non-negative")
    if not 0.0 <= args.min_tissue_fraction <= 1.0:
        raise ValueError("min_tissue_fraction must be between zero and one")


def prepare_hd(args):
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
    return model, datasets, metadata, config


def _loader(dataset, args, device, split, generator):
    return DataLoader(
        dataset,
        batch_size=args.batch_size if split == "train" else 1,
        shuffle=split == "train",
        generator=generator if split == "train" else None,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


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

    print("Preparing visium_hd data ...")
    model, datasets, preprocessing, task_config = prepare_hd(args)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: _loader(dataset, args, device, split, generator)
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
        "task": "visium_hd",
        "model": "VisiumHDNORMST",
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
        learning_rate_used = optimizer.param_groups[0]["lr"]
        train_started = perf_counter()
        train_metrics = run_epoch(
            "visium_hd", model, loaders["train"], device,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
            description=f"train {epoch + 1}/{args.epochs}",
        )
        train_seconds = perf_counter() - train_started
        val_started = perf_counter()
        val_metrics = run_epoch(
            "visium_hd", model, loaders["val"], device,
            use_amp=use_amp, report_baseline=True,
            description=f"val {epoch + 1}/{args.epochs}",
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
        "visium_hd", model, loaders["test"], device,
        use_amp=use_amp, report_baseline=True, description="test",
    )
    test_metrics["best_epoch"] = best["epoch"]
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    if args.save_predictions:
        predictions = collect_predictions(
            "visium_hd", model, loaders["test"], device, use_amp, None, None
        )
        predictions["genes"] = preprocessing["genes"]
        np.savez(args.output_dir / "test_predictions.npz", **predictions)
    print("Test:", json.dumps(test_metrics))
