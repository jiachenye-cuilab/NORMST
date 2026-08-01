"""Train ST-SRNO on an official paired Visium HD resolution hierarchy.

Example:
python train_visium_hd_paired.py \
  --lr-dir /home2/yejiachen/ST/HBCHD/binned_outputs/square_016um \
  --hr-dir /home2/yejiachen/ST/HBCHD/binned_outputs/square_008um \
  --output-dir save/HBCHD/paired_16_to_8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.paired_visium_hd import (
    PairedVisiumHDDataset,
    prepare_visium_hd_pair,
)
from models.sronet_st import STSRNO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--patch-size-lr", type=int, default=32)
    parser.add_argument("--train-repeat", type=int, default=4)
    parser.add_argument("--eval-repeat", type=int, default=1,
                        help="Deprecated compatibility option; deterministic evaluation covers origins")
    parser.add_argument("--eval-origin-stride", type=int, default=0,
                        help="LR stride for deterministic evaluation; 0 uses patch size")
    parser.add_argument("--eval-max-origins", type=int, default=0,
                        help="Evenly subsample evaluation origins; 0 evaluates all")
    parser.add_argument("--context-dim", type=int, default=16,
                        help="PCA gene-context channels; 0 reproduces the single-gene model")
    parser.add_argument("--context-mode", choices=("pca", "shuffled"), default="pca",
                        help="Use real context or a spatially shuffled negative control")
    parser.add_argument("--min-tissue-fraction", type=float, default=0.1)
    parser.add_argument("--split-axis", choices=("row", "col"), default="col")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--operator-layers", type=int, default=2)
    parser.add_argument("--encoder-blocks", type=int, default=16)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--lambda-negative", type=float, default=0.01)
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


def masked_smooth_l1(prediction, target, mask):
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def masked_pearson_values(prediction, target, mask):
    batch = prediction.shape[0]
    values = []
    for index in range(batch):
        valid = mask[index].bool()
        pred = prediction[index][valid]
        truth = target[index][valid]
        if pred.numel() < 2:
            continue
        pred = pred - pred.mean()
        truth = truth - truth.mean()
        denominator = torch.sqrt((pred.square().sum()) * (truth.square().sum()))
        if denominator > 0:
            values.append((pred * truth).sum() / denominator)
    if not values:
        return prediction.new_empty(0)
    return torch.stack(values)


def aggregation_consistency_loss(
    prediction_scaled,
    input_scaled,
    input_mask,
    target_mask,
    hr_library,
    lr_gene_scale,
    hr_gene_scale,
    scale,
    target_sum,
):
    """Reaggregate predicted HR expression and compare with observed LR.

    Gene expression is converted from log-CP10K to an estimated raw-count
    contribution using the observed HR library sizes, summed over scale x
    scale bins, and normalized again at the coarse resolution.
    """
    pred_log = prediction_scaled * hr_gene_scale[:, None, None, None]
    input_log = input_scaled * lr_gene_scale[:, None, None, None]
    pred_cp = torch.expm1(pred_log.clamp(min=0, max=20))
    estimated_counts = pred_cp / target_sum * hr_library * target_mask
    area = float(scale * scale)
    coarse_counts = F.avg_pool2d(
        estimated_counts, kernel_size=scale, stride=scale
    ) * area
    coarse_library = F.avg_pool2d(
        hr_library * target_mask, kernel_size=scale, stride=scale
    ) * area
    reconstructed_log = torch.log1p(
        coarse_counts * target_sum / coarse_library.clamp_min(1.0)
    )
    loss = F.smooth_l1_loss(reconstructed_log, input_log, reduction="none")
    return (loss * input_mask).sum() / input_mask.sum().clamp_min(1.0)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(
    model,
    loader,
    device,
    target_sum,
    lambda_consistency,
    lambda_negative,
    optimizer=None,
    scaler=None,
    use_amp=True,
    description="train",
):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "consistency": 0.0,
              "negative": 0.0, "pearson_sum": 0.0, "pearson_count": 0,
              "baseline_reconstruction": 0.0, "baseline_pearson_sum": 0.0,
              "baseline_pearson_count": 0, "batches": 0}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, desc=description, leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction = model(
                    batch["inp"], batch["input_mask"], batch["coord"],
                    batch["cell"], batch["scale"], batch["target_mask"],
                    batch["gene_context"],
                    batch["lr_gene_scale"] / batch["hr_gene_scale"],
                )
                reconstruction = masked_smooth_l1(
                    prediction, batch["gt"], batch["target_mask"]
                )
                consistency = aggregation_consistency_loss(
                    prediction, batch["inp"], batch["input_mask"],
                    batch["target_mask"], batch["hr_library"],
                    batch["lr_gene_scale"], batch["hr_gene_scale"],
                    scale=int(batch["scale"][0].item()), target_sum=target_sum,
                )
                negative = (
                    F.relu(-prediction) * batch["target_mask"]
                ).sum() / batch["target_mask"].sum().clamp_min(1.0)
                loss = (
                    reconstruction
                    + lambda_consistency * consistency
                    + lambda_negative * negative
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            baseline = F.grid_sample(
                batch["inp"], batch["coord"].flip(-1), mode="bilinear",
                padding_mode="border", align_corners=False,
            )
            # Input and target use separately fitted gene-wise RMS scales.
            baseline = baseline * (
                batch["lr_gene_scale"] / batch["hr_gene_scale"]
            )[:, None, None, None]
            pearsons = masked_pearson_values(
                prediction.detach(), batch["gt"], batch["target_mask"]
            )
            baseline_pearsons = masked_pearson_values(
                baseline, batch["gt"], batch["target_mask"]
            )
            baseline_reconstruction = masked_smooth_l1(
                baseline, batch["gt"], batch["target_mask"]
            )
            totals["loss"] += loss.item()
            totals["reconstruction"] += reconstruction.item()
            totals["consistency"] += consistency.item()
            totals["negative"] += negative.item()
            totals["pearson_sum"] += pearsons.sum().item()
            totals["pearson_count"] += pearsons.numel()
            totals["baseline_reconstruction"] += baseline_reconstruction.item()
            totals["baseline_pearson_sum"] += baseline_pearsons.sum().item()
            totals["baseline_pearson_count"] += baseline_pearsons.numel()
            totals["batches"] += 1
            progress.set_postfix(loss=f"{totals['loss']/totals['batches']:.4f}")
    batches = max(totals.pop("batches"), 1)
    pearson = totals.pop("pearson_sum") / max(totals.pop("pearson_count"), 1)
    baseline_pearson = (
        totals.pop("baseline_pearson_sum")
        / max(totals.pop("baseline_pearson_count"), 1)
    )
    metrics = {key: value / batches for key, value in totals.items()}
    metrics["pearson"] = pearson
    metrics["baseline_pearson"] = baseline_pearson
    metrics["pearson_gain"] = pearson - baseline_pearson
    metrics["reconstruction_gain"] = (
        metrics["baseline_reconstruction"] - metrics["reconstruction"]
    )
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing paired official Visium HD data ...")
    pair = prepare_visium_hd_pair(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        n_genes=args.n_genes,
        scale=args.scale,
        target_sum=args.target_sum,
        split_axis=args.split_axis,
        split_ratios=args.split_ratios,
        context_dim=args.context_dim,
    )
    datasets = {
        "train": PairedVisiumHDDataset(
            pair, "train", (args.patch_size_lr, args.patch_size_lr),
            args.train_repeat, args.min_tissue_fraction, deterministic=False,
            context_mode=args.context_mode, seed=args.seed,
        ),
        "val": PairedVisiumHDDataset(
            pair, "val", (args.patch_size_lr, args.patch_size_lr),
            args.eval_repeat, args.min_tissue_fraction, deterministic=True,
            origin_stride=args.eval_origin_stride or args.patch_size_lr,
            max_origins=args.eval_max_origins,
            context_mode=args.context_mode, seed=args.seed,
        ),
        "test": PairedVisiumHDDataset(
            pair, "test", (args.patch_size_lr, args.patch_size_lr),
            args.eval_repeat, args.min_tissue_fraction, deterministic=True,
            origin_stride=args.eval_origin_stride or args.patch_size_lr,
            max_origins=args.eval_max_origins,
            context_mode=args.context_mode, seed=args.seed,
        ),
    }
    print(
        "Evaluation origins:",
        f"val={len(datasets['val'].origins)}, test={len(datasets['test'].origins)}",
    )
    if args.context_dim:
        explained = float(pair.context_explained_variance_ratio.sum())
        print(f"PCA context: {args.context_dim} dimensions, explained variance={explained:.4f}")
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for name, dataset in datasets.items()
    }
    model = STSRNO(
        width=args.width,
        num_heads=args.num_heads,
        num_operator_layers=args.operator_layers,
        encoder_blocks=args.encoder_blocks,
        context_dim=args.context_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    np.savetxt(args.output_dir / "genes.txt", pair.genes, fmt="%s")
    np.savez(
        args.output_dir / "gene_scales.npz",
        genes=pair.genes,
        lr_scale=pair.lr_gene_scale,
        hr_scale=pair.hr_gene_scale,
        context_mean=pair.context_mean,
        context_components=pair.context_components,
        context_scale=pair.context_scale,
        context_explained_variance_ratio=pair.context_explained_variance_ratio,
    )
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["split_ranges"] = pair.split_ranges
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    history, best_val = [], float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, loaders["train"], device, args.target_sum,
            args.lambda_consistency, args.lambda_negative,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            description=f"train {epoch + 1}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model, loaders["val"], device, args.target_sum,
            args.lambda_consistency, args.lambda_negative,
            use_amp=use_amp, description=f"val {epoch + 1}/{args.epochs}",
        )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
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
            "epoch": epoch + 1,
            "config": config,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")

    best = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(
        model, loaders["test"], device, args.target_sum,
        args.lambda_consistency, args.lambda_negative,
        use_amp=use_amp, description="test",
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print("Test:", json.dumps(test_metrics))


if __name__ == "__main__":
    main()
