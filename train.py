"""Train expression-only SRNO by recovering held-out standard Visium spots.

The target spots are removed before the model sees target-gene expression or
multi-gene PCA context. No histology image is loaded.
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

from datasets.masked_visium import MaskedVisiumDataset, prepare_masked_visium
from models.sronet_st import STSRNO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--count-file", default="raw_feature_bc_matrix.h5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--observed-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--train-target-fraction", type=float, default=0.25)
    parser.add_argument("--train-repeat", type=int, default=8)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--context-mode", choices=("pca", "shuffled"), default="pca")
    parser.add_argument("--gene-embedding-dim", type=int, default=16)
    parser.add_argument(
        "--spatial-encoder",
        choices=("rectangular", "rectangular_coord", "hex_coord"),
        default="rectangular",
        help=(
            "rectangular keeps the original EDSR; rectangular_coord adds true "
            "physical coordinate channels; hex_coord also replaces EDSR with "
            "six-neighbor residual message passing"
        ),
    )
    parser.add_argument(
        "--hex-residual-scale",
        type=float,
        default=0.1,
        help="Residual-branch scale used only by the hexagonal encoder",
    )
    parser.add_argument("--idw-neighbors", type=int, default=8)
    parser.add_argument("--reconstruction-loss", choices=("standard", "balanced"),
                        default="standard")
    parser.add_argument("--positive-loss-weight", type=float, default=0.2)
    parser.add_argument("--lambda-pearson", type=float, default=0.05)
    parser.add_argument("--lambda-negative", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--operator-layers", type=int, default=2)
    parser.add_argument("--encoder-blocks", type=int, default=16)
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


def balanced_masked_smooth_l1(prediction, target, mask, positive_weight):
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    valid = mask.bool()
    positive = valid & (target > 0)
    zero = valid & ~positive
    terms, weights = [], []
    if positive.any():
        terms.append(elementwise[positive].mean())
        weights.append(positive_weight)
    if zero.any():
        terms.append(elementwise[zero].mean())
        weights.append(1.0 - positive_weight)
    if not terms:
        return prediction.sum() * 0.0
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return sum(terms) / len(terms)
    return sum(term * weight for term, weight in zip(terms, weights)) / weight_sum


def masked_pearson_values(prediction, target, mask, epsilon=1e-8):
    values = []
    for index in range(prediction.shape[0]):
        valid = mask[index].bool()
        pred = prediction[index][valid]
        truth = target[index][valid]
        if pred.numel() < 2:
            continue
        pred = pred - pred.mean()
        truth = truth - truth.mean()
        pred_energy = pred.square().sum()
        truth_energy = truth.square().sum()
        if pred_energy > epsilon and truth_energy > epsilon:
            values.append(
                (pred * truth).sum() / torch.sqrt(pred_energy * truth_energy)
            )
    if not values:
        return prediction.new_empty(0)
    return torch.stack(values)


def masked_pearson_loss(prediction, target, mask, epsilon=1e-8):
    losses = []
    for index in range(prediction.shape[0]):
        valid = mask[index].bool()
        pred = prediction[index][valid]
        truth = target[index][valid]
        if pred.numel() < 2:
            continue
        pred = pred - pred.mean()
        truth = truth - truth.mean()
        truth_energy = truth.square().sum()
        if truth_energy <= epsilon:
            continue
        denominator = torch.sqrt(
            pred.square().sum().clamp_min(epsilon) * truth_energy
        )
        losses.append(1.0 - (pred * truth).sum() / denominator)
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(
    model,
    loader,
    device,
    reconstruction_loss,
    positive_loss_weight,
    lambda_pearson,
    lambda_negative,
    optimizer=None,
    scaler=None,
    use_amp=True,
    description="train",
):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "balanced_reconstruction": 0.0,
        "pearson_loss": 0.0,
        "negative": 0.0,
        "baseline_reconstruction": 0.0,
        "pearson_sum": 0.0,
        "pearson_count": 0,
        "baseline_pearson_sum": 0.0,
        "baseline_pearson_count": 0,
        "samples": 0,
    }
    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, desc=description, leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction = model(
                    batch["inp"],
                    batch["input_mask"],
                    batch["coord"],
                    batch["cell"],
                    batch["scale"],
                    target_mask=batch["tissue_mask"],
                    gene_context=batch["gene_context"],
                    gene_index=batch["gene_index"],
                    tissue_mask=batch["tissue_mask"],
                    physical_coord=batch["physical_coord"],
                    row_parity=batch["row_parity"],
                )
                reconstruction = masked_smooth_l1(
                    prediction, batch["gt"], batch["target_mask"]
                )
                balanced = balanced_masked_smooth_l1(
                    prediction, batch["gt"], batch["target_mask"],
                    positive_loss_weight,
                )
                pearson_loss = masked_pearson_loss(
                    prediction, batch["gt"], batch["target_mask"]
                )
                negative = (
                    F.relu(-prediction) * batch["target_mask"]
                ).sum() / batch["target_mask"].sum().clamp_min(1.0)
                reconstruction_objective = (
                    reconstruction if reconstruction_loss == "standard" else balanced
                )
                loss = (
                    reconstruction_objective
                    + lambda_pearson * pearson_loss
                    + lambda_negative * negative
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"{description}: non-finite loss; "
                    f"prediction_finite={torch.isfinite(prediction).all().item()}, "
                    f"reconstruction={reconstruction.item()}, "
                    f"pearson_loss={pearson_loss.item()}, negative={negative.item()}"
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            pearsons = masked_pearson_values(
                prediction.detach(), batch["gt"], batch["target_mask"]
            )
            baseline_pearsons = masked_pearson_values(
                batch["baseline"], batch["gt"], batch["target_mask"]
            )
            baseline_reconstruction = masked_smooth_l1(
                batch["baseline"], batch["gt"], batch["target_mask"]
            )
            batch_size = batch["inp"].shape[0]
            totals["loss"] += loss.item() * batch_size
            totals["reconstruction"] += reconstruction.item() * batch_size
            totals["balanced_reconstruction"] += balanced.item() * batch_size
            totals["pearson_loss"] += pearson_loss.item() * batch_size
            totals["negative"] += negative.item() * batch_size
            totals["baseline_reconstruction"] += baseline_reconstruction.item() * batch_size
            totals["pearson_sum"] += pearsons.sum().item()
            totals["pearson_count"] += pearsons.numel()
            totals["baseline_pearson_sum"] += baseline_pearsons.sum().item()
            totals["baseline_pearson_count"] += baseline_pearsons.numel()
            totals["samples"] += batch_size
            progress.set_postfix(loss=f"{totals['loss']/totals['samples']:.4f}")

    samples = max(totals.pop("samples"), 1)
    pearson = totals.pop("pearson_sum") / max(totals.pop("pearson_count"), 1)
    baseline_pearson = (
        totals.pop("baseline_pearson_sum")
        / max(totals.pop("baseline_pearson_count"), 1)
    )
    metrics = {key: value / samples for key, value in totals.items()}
    metrics["pearson"] = pearson
    metrics["baseline_pearson"] = baseline_pearson
    metrics["pearson_gain"] = pearson - baseline_pearson
    metrics["reconstruction_gain"] = (
        metrics["baseline_reconstruction"] - metrics["reconstruction"]
    )
    return metrics


def main():
    args = parse_args()
    if args.context_dim < 0 or args.gene_embedding_dim < 0:
        raise ValueError("context and embedding dimensions must be non-negative")
    if args.hex_residual_scale < 0:
        raise ValueError("hex residual scale must be non-negative")
    if not 0.0 <= args.positive_loss_weight <= 1.0:
        raise ValueError("positive_loss_weight must be between 0 and 1")
    if min(args.lambda_pearson, args.lambda_negative) < 0:
        raise ValueError("loss weights must be non-negative")
    if min(args.train_repeat, args.batch_size, args.workers + 1, args.epochs) < 1:
        raise ValueError("repeat, batch size and epochs must be positive; workers non-negative")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing masked standard Visium data (no histology) ...")
    data = prepare_masked_visium(
        data_dir=args.data_dir,
        count_file=args.count_file,
        n_genes=args.n_genes,
        target_sum=args.target_sum,
        context_dim=args.context_dim,
        observed_fraction=args.observed_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(
        "Spots:",
        f"observed={len(data.observed_spots)},",
        f"val={len(data.validation_spots)}, test={len(data.test_spots)}",
    )
    if args.context_dim:
        explained = data.context_explained_variance_ratio.sum()
        print(f"PCA context explained variance={explained:.4f}")

    datasets = {
        split: MaskedVisiumDataset(
            data,
            split,
            repeat=args.train_repeat,
            train_target_fraction=args.train_target_fraction,
            context_mode=args.context_mode,
            idw_neighbors=args.idw_neighbors,
            seed=args.seed,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items()
    }
    model = STSRNO(
        width=args.width,
        num_heads=args.num_heads,
        num_operator_layers=args.operator_layers,
        encoder_blocks=args.encoder_blocks,
        context_dim=args.context_dim,
        n_genes=len(data.genes),
        gene_embedding_dim=args.gene_embedding_dim,
        include_tissue_mask=True,
        spatial_encoder=args.spatial_encoder,
        hex_residual_scale=args.hex_residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["grid_shape"] = data.shape
    config["n_observed_spots"] = len(data.observed_spots)
    config["n_validation_spots"] = len(data.validation_spots)
    config["n_test_spots"] = len(data.test_spots)
    config["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {config['trainable_parameters']:,}")
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    np.savetxt(args.output_dir / "genes.txt", data.genes, fmt="%s")
    np.savez(
        args.output_dir / "preprocessing.npz",
        genes=data.genes,
        gene_scale=data.gene_scale,
        context_mean=data.context_mean,
        context_components=data.context_components,
        context_scale=data.context_scale,
        context_explained_variance_ratio=data.context_explained_variance_ratio,
        row_map=data.row_map,
        physical_coord_grid=data.physical_coord_grid,
        row_parity=data.row_parity,
        observed_spots=data.observed_spots,
        validation_spots=data.validation_spots,
        test_spots=data.test_spots,
    )

    history, best_val = [], float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, loaders["train"], device,
            args.reconstruction_loss, args.positive_loss_weight,
            args.lambda_pearson, args.lambda_negative,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            description=f"train {epoch + 1}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model, loaders["val"], device,
            args.reconstruction_loss, args.positive_loss_weight,
            args.lambda_pearson, args.lambda_negative,
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
        model, loaders["test"], device,
        args.reconstruction_loss, args.positive_loss_weight,
        args.lambda_pearson, args.lambda_negative,
        use_amp=use_amp, description="test",
    )
    test_metrics["best_epoch"] = best["epoch"]
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print("Test:", json.dumps(test_metrics))


if __name__ == "__main__":
    main()
