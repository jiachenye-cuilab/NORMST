"""Train the standalone count-aware expression encoder."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePath
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from checkpoint_io import (
    load_checkpoint,
    save_checkpoint as save_portable_checkpoint,
)

from data import (
    PretrainData,
    SpotCountDataset,
    assert_output_outside_sources,
    load_pretrain_data,
    parse_manifest,
    verify_source_contract,
)
from losses import LossConfig, denoising_objective
from metrics import ReconstructionMetrics
from model import CountAwareAutoencoder, ModelConfig, parse_hidden_dims


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pretrain a count-aware composition encoder"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--count-file", default="filtered_feature_bc_matrix.h5"
    )
    parser.add_argument(
        "--fixed-genes",
        type=Path,
        default=None,
        help="optional genes.txt whose exact order is reused for a matched comparison",
    )
    parser.add_argument("--min-train-gene-count", type=int, default=1)
    parser.add_argument("--composition-dim", type=int, default=255)
    parser.add_argument("--hidden-dims", default="512,512")
    parser.add_argument("--composition-scale", type=float, default=1e4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dispersion-init", type=float, default=10.0)
    parser.add_argument(
        "--decoder-type",
        choices=("nonlinear", "linear"),
        default="nonlinear",
        help="nonlinear MLP decoder or GLM-PCA-style linear logit decoder",
    )
    parser.add_argument("--thinning-probability", type=float, default=0.5)
    parser.add_argument("--nb-weight", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--latent-variance-weight", type=float, default=0.0)
    parser.add_argument("--latent-covariance-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("batch size, epochs and patience must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.gradient_clip < 0:
        raise ValueError("invalid optimizer hyperparameters")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    parse_hidden_dims(args.hidden_dims)
    LossConfig(
        thinning_probability=args.thinning_probability,
        nb_weight=args.nb_weight,
        consistency_weight=args.consistency_weight,
        latent_variance_weight=args.latent_variance_weight,
        latent_covariance_weight=args.latent_covariance_weight,
    )


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_value(value):
    if isinstance(value, PurePath):
        return str(value.resolve())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: dict) -> None:
    save_portable_checkpoint(path, payload)


def prepare_output_dir(args, entries) -> Path:
    output = assert_output_outside_sources(args.output_dir, entries)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def make_loader(
    dataset: SpotCountDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
        drop_last=False,
    )


def thinning_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def run_loss_epoch(
    model: CountAwareAutoencoder,
    loader: DataLoader,
    loss_config: LossConfig,
    device: torch.device,
    use_amp: bool,
    thinning_seed: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    generator = thinning_generator(device, thinning_seed)
    sums = {
        "total_loss": 0.0,
        "composition_cross_entropy": 0.0,
        "negative_binomial_nll": 0.0,
        "latent_consistency": 0.0,
        "latent_variance": 0.0,
        "latent_covariance": 0.0,
    }
    observations = 0
    valid_directions = 0
    valid_pairs = 0

    context = nullcontext() if training else torch.no_grad()
    with context:
        for batch in loader:
            counts = batch["counts"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                values = denoising_objective(
                    model,
                    counts,
                    loss_config,
                    generator=generator,
                )
            loss = values["total_loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite pretraining loss")
            if training:
                if scaler is None:
                    loss.backward()
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    if gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()

            batch_size = int(counts.shape[0])
            observations += batch_size
            for key in sums:
                sums[key] += float(values[key].detach()) * batch_size
            valid_directions += int(values["valid_directions"].detach())
            valid_pairs += int(values["valid_pairs"].detach())

    if observations < 1:
        raise ValueError("empty data loader")
    result = {key: value / observations for key, value in sums.items()}
    result.update(
        {
            "spots": observations,
            "valid_directions": valid_directions,
            "valid_pairs": valid_pairs,
        }
    )
    return result


@torch.no_grad()
def reconstruction_metrics(
    model: CountAwareAutoencoder,
    loader: DataLoader,
    train_log_mean: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()
    accumulator = ReconstructionMetrics(train_log_mean)
    for batch in loader:
        counts = batch["counts"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(counts)["mean"]
        accumulator.update(prediction, counts)
    return accumulator.compute()


@torch.no_grad()
def latent_statistics(
    model: CountAwareAutoencoder,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, np.ndarray | int]:
    """Fit downstream target standardization on training spots only."""
    model.eval()
    latent_sum = torch.zeros(model.config.composition_dim, dtype=torch.float64)
    latent_square_sum = torch.zeros_like(latent_sum)
    library_sum = 0.0
    library_square_sum = 0.0
    count = 0
    for batch in loader:
        counts = batch["counts"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            latent, log_library = model.encode(counts)
        latent = latent.float().cpu().double()
        log_library = log_library.float().cpu().double().ravel()
        latent_sum += latent.sum(dim=0)
        latent_square_sum += latent.square().sum(dim=0)
        library_sum += float(log_library.sum())
        library_square_sum += float(log_library.square().sum())
        count += int(latent.shape[0])
    if count < 2:
        raise ValueError("at least two training spots are required")
    latent_mean = latent_sum / count
    latent_variance = (
        latent_square_sum / count - latent_mean.square()
    ).clamp_min(0.0)
    library_mean = library_sum / count
    library_variance = max(library_square_sum / count - library_mean**2, 0.0)
    return {
        "spots": count,
        "composition_mean": latent_mean.float().numpy(),
        "composition_scale": latent_variance.sqrt().clamp_min(1e-6).float().numpy(),
        "log_library_mean": np.asarray([library_mean], dtype=np.float32),
        "log_library_scale": np.asarray(
            [max(library_variance**0.5, 1e-6)], dtype=np.float32
        ),
    }


def checkpoint_payload(
    model: CountAwareAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    run_config: dict,
    data: PretrainData,
    history: list[dict],
) -> dict:
    return {
        "format_version": 1,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "run_config": run_config,
        "manifest_sha256": data.manifest_sha256,
        "genes": data.genes.tolist(),
        "train_log_mean": data.train_log_mean,
        "source_contract": data.source_contract,
        "history": history,
    }


def final_role_metrics(
    model: CountAwareAutoencoder,
    data: PretrainData,
    role: str,
    args,
    device: torch.device,
    use_amp: bool,
    loss_config: LossConfig,
) -> dict:
    combined = data.dataset(role)
    combined_loader = make_loader(
        combined,
        args.batch_size,
        args.workers,
        device,
        shuffle=False,
        seed=args.seed + 600,
    )
    denoising = run_loss_epoch(
        model,
        combined_loader,
        loss_config,
        device,
        use_amp,
        thinning_seed=args.seed + (700 if role == "val" else 800),
    )
    overall = reconstruction_metrics(
        model, combined_loader, data.train_log_mean, device, use_amp
    )
    by_slice = {}
    for index, item in enumerate(data.for_role(role)):
        dataset = SpotCountDataset([item])
        loader = make_loader(
            dataset,
            args.batch_size,
            args.workers,
            device,
            shuffle=False,
            seed=args.seed + 900 + index,
        )
        by_slice[item.name] = reconstruction_metrics(
            model, loader, data.train_log_mean, device, use_amp
        )
    return {
        "role": role,
        "reconstruction_library_mode": "oracle true library from encoder input",
        "denoising": denoising,
        "reconstruction_overall": overall,
        "reconstruction_by_slice": by_slice,
    }


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    seed_everything(args.seed)
    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    manifest = args.manifest.resolve()
    entries = parse_manifest(manifest, args.count_file)
    fixed_genes = None
    fixed_genes_path = None
    if args.fixed_genes is not None:
        fixed_genes_path = args.fixed_genes.resolve()
        if not fixed_genes_path.is_file():
            raise FileNotFoundError(fixed_genes_path)
        fixed_genes = [
            line
            for line in fixed_genes_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not fixed_genes or len(fixed_genes) != len(set(fixed_genes)):
            raise ValueError("fixed genes must be non-empty and unique")

    output = prepare_output_dir(args, entries)
    shutil.copyfile(manifest, output / "manifest.json")

    data = load_pretrain_data(
        manifest,
        count_file=args.count_file,
        min_train_gene_count=args.min_train_gene_count,
        fixed_genes=fixed_genes,
    )
    genes_path = output / "genes.txt"
    genes_path.write_text(
        "\n".join(data.genes.tolist()) + "\n", encoding="utf-8"
    )

    model_config = ModelConfig(
        n_genes=len(data.genes),
        composition_dim=args.composition_dim,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        composition_scale=args.composition_scale,
        dropout=args.dropout,
        dispersion_init=args.dispersion_init,
        decoder_type=args.decoder_type,
    )
    loss_config = LossConfig(
        thinning_probability=args.thinning_probability,
        nb_weight=args.nb_weight,
        consistency_weight=args.consistency_weight,
        latent_variance_weight=args.latent_variance_weight,
        latent_covariance_weight=args.latent_covariance_weight,
    )
    model = CountAwareAutoencoder(model_config).to(device)
    model.initialize_decoder_bias(
        torch.from_numpy(data.train_gene_probability).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3),
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_config = _json_value(
        {
            **vars(args),
            "manifest": str(manifest),
            "output_dir": str(output),
            "device": str(device),
            "amp_effective": use_amp,
            "model": model_config.to_dict(),
            "loss": asdict(loss_config),
            "gene_selection": (
                {
                    "method": "exact external genes.txt order",
                    "source": str(fixed_genes_path),
                    "source_sha256": _sha256(fixed_genes_path),
                    "selected_genes": len(data.genes),
                    "validation_or_test_expression_used_by_ae": False,
                }
                if fixed_genes_path is not None
                else {
                    "method": (
                        "all genes shared by manifest slices with training total count >= "
                        f"{args.min_train_gene_count}"
                    ),
                    "validation_or_test_expression_used": False,
                }
            ),
            "genes_sha256": _sha256(genes_path),
            "checkpoint_selection": "minimum fixed-thinning validation total_loss",
            "feature_layout": "composition latent first, log1p(total UMI) last",
            "data_writes": "output_dir only; source contract is read-only",
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "manifest_sha256": data.manifest_sha256,
            "manifest_meta": json.loads(
                manifest.read_text(encoding="utf-8")
            ).get("_meta"),
            "source_contract": data.source_contract,
        }
    )
    write_json(output / "config.json", run_config)

    train_loader = make_loader(
        data.dataset("train"),
        args.batch_size,
        args.workers,
        device,
        shuffle=True,
        seed=args.seed,
    )
    train_stats_loader = make_loader(
        data.dataset("train"),
        args.batch_size,
        args.workers,
        device,
        shuffle=False,
        seed=args.seed + 1,
    )
    val_loader = make_loader(
        data.dataset("val"),
        args.batch_size,
        args.workers,
        device,
        shuffle=False,
        seed=args.seed + 2,
    )

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_metrics = run_loss_epoch(
            model,
            train_loader,
            loss_config,
            device,
            use_amp,
            thinning_seed=args.seed + 10_000 + epoch,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=args.gradient_clip,
        )
        val_metrics = run_loss_epoch(
            model,
            val_loader,
            loss_config,
            device,
            use_amp,
            thinning_seed=args.seed + 20_000,
        )
        scheduler.step(val_metrics["total_loss"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        write_json(output / "history.json", history)

        improved = val_metrics["total_loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_epoch,
            best_val_loss,
            run_config,
            data,
            history,
        )
        save_checkpoint(output / "last.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)
        print(json.dumps(_json_value(row), ensure_ascii=False), flush=True)
        verify_source_contract(data.source_contract)

        if epochs_without_improvement >= args.patience:
            break

    best = load_checkpoint(output / "best.pt", map_location=device)
    model.load_state_dict(best["model_state"], strict=True)
    statistics = latent_statistics(
        model, train_stats_loader, device=device, use_amp=use_amp
    )
    np.savez_compressed(
        output / "latent_statistics.npz",
        **statistics,
    )
    best["latent_statistics"] = statistics
    best["representation_frozen_for_downstream"] = True
    save_checkpoint(output / "best.pt", best)

    val_metrics = final_role_metrics(
        model, data, "val", args, device, use_amp, loss_config
    )
    test_metrics = final_role_metrics(
        model, data, "test", args, device, use_amp, loss_config
    )
    write_json(output / "val_metrics.json", val_metrics)
    write_json(output / "test_metrics.json", test_metrics)
    verify_source_contract(data.source_contract)
    print(
        json.dumps(
            {
                "status": "complete",
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "feature_dim": model.feature_dim,
                "genes": len(data.genes),
                "output_dir": str(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
