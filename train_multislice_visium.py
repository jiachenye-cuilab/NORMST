"""Train VisiumNORMST across slice-level train/val/test partitions.

Pass ``--visium-root`` to discover its direct child slice directories and
randomly split them 4:1:1. The resolved split is always saved as
``output_dir/manifest.json``. A prebuilt manifest can alternatively be passed
with ``--manifest``; it must contain non-empty ``train``, ``val`` and ``test``
groups, for example::

    {
      "train": {"151507": "/data/151507", "151508": "/data/151508"},
      "val": {"151509": "/data/151509"},
      "test": {"151673": "/data/151673"}
    }

HVGs are fitted on all spots from training slices only. Gene-wise RMS scales
are also fitted there by default and can be disabled with ``--no-rms-scale``.
Every assigned slice contributes all of its generated masks to its own role
and retains its own physical coordinates and native hex graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.multislice_masked_visium import (
    MultiSlicePointDataset,
    prepare_multislice_visium,
)
from models.geometry_adaptive_normst import (
    VisiumNORMST,
    build_native_hex_neighbors,
)
from train_geometry_adaptive_normst import (
    collect_predictions,
    run_epoch,
    seed_everything,
)


POSITION_FILES = (
    "tissue_positions.csv",
    "tissue_positions_list.csv",
    "tissue_positions_list.txt",
    "tissue_positions.parquet",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--visium-root", type=Path,
        help="directory whose direct children are standard Visium slices",
    )
    source.add_argument(
        "--manifest", type=Path,
        help="optional prebuilt slice-level split manifest",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument(
        "--no-rms-scale", action="store_true",
        help="keep expression in log1p(CP10K) space without gene-wise RMS scaling",
    )
    parser.add_argument(
        "--mask-target-fraction", "--train-target-fraction",
        dest="mask_target_fraction", type=float, default=0.25,
        help=(
            "fraction of spots hidden inside every train/val/test mask; "
            "--train-target-fraction remains as a compatibility alias"
        ),
    )
    parser.add_argument("--masks-per-slice", type=int, default=64)
    parser.add_argument("--query-neighbors", type=int, default=6)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--query-chunk-size", type=int, default=1024)

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
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def discover_visium_slices(data_root: Path, count_file: str):
    root = data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Visium data root does not exist: {root}")
    valid = []
    skipped = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        if not candidate.is_dir():
            continue
        count_path = candidate / count_file
        spatial_dir = candidate / "spatial"
        position_path = next(
            (spatial_dir / name for name in POSITION_FILES
             if (spatial_dir / name).is_file()),
            None,
        )
        if count_path.is_file() and position_path is not None:
            valid.append(candidate.resolve())
        else:
            missing = []
            if not count_path.is_file():
                missing.append(count_file)
            if position_path is None:
                missing.append("spatial/tissue_positions*")
            skipped.append({"directory": str(candidate.resolve()), "missing": missing})
    return valid, skipped


def ratio_4_1_1_counts(n_slices: int):
    if n_slices < 6:
        raise ValueError(
            "a non-empty 4:1:1 train/val/test split requires at least 6 slices; "
            f"found {n_slices}"
        )
    train_count = round(n_slices * 4 / 6)
    val_count = round(n_slices / 6)
    test_count = n_slices - train_count - val_count
    if min(train_count, val_count, test_count) < 1:
        raise ValueError("4:1:1 rounding produced an empty split")
    return train_count, val_count, test_count


def build_random_manifest(slice_paths, seed, data_root, count_file):
    paths = list(slice_paths)
    train_count, val_count, test_count = ratio_4_1_1_counts(len(paths))
    random.Random(seed).shuffle(paths)
    groups = {
        "train": paths[:train_count],
        "val": paths[train_count:train_count + val_count],
        "test": paths[train_count + val_count:],
    }
    manifest = {
        "_meta": {
            "source": "auto_discovered_visium_root",
            "seed": seed,
            "ratio": [4, 1, 1],
            "counts": {
                "train": train_count, "val": val_count, "test": test_count,
            },
            "data_root": str(data_root.resolve()),
            "count_file": count_file,
        }
    }
    for role, role_paths in groups.items():
        manifest[role] = {path.name: str(path) for path in role_paths}
    return manifest


def resolve_run_manifest(args):
    output_manifest = args.output_dir / "manifest.json"
    if args.manifest is not None:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        slices, skipped = discover_visium_slices(
            args.visium_root, args.count_file
        )
        payload = build_random_manifest(
            slices, args.seed, args.visium_root, args.count_file
        )
        counts = payload["_meta"]["counts"]
        print(
            f"Discovered {len(slices)} slices: "
            f"train={counts['train']}, val={counts['val']}, "
            f"test={counts['test']}"
        )
        if skipped:
            print(f"Skipped {len(skipped)} incomplete directories:")
            for item in skipped:
                print(
                    f"  {item['directory']}: missing "
                    f"{', '.join(item['missing'])}"
                )
    output_manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def validate_args(args):
    positive = (
        args.n_genes, args.target_sum, args.masks_per_slice,
        args.query_neighbors, args.idw_power, args.query_chunk_size,
        args.width, args.num_heads, args.operator_layers, args.epochs,
    )
    if min(positive) <= 0:
        raise ValueError("model, preprocessing, and training sizes must be positive")
    if args.width % args.num_heads:
        raise ValueError("width must be divisible by num_heads")
    if not 0.0 < args.mask_target_fraction < 1.0:
        raise ValueError("mask_target_fraction must be between zero and one")
    if args.workers < 0 or args.patience < 0:
        raise ValueError("workers and patience must be non-negative")
    if args.min_delta < 0 or args.max_grad_norm < 0:
        raise ValueError("min_delta and max_grad_norm must be non-negative")


def _loader(dataset, device, workers, shuffle=False, generator=None):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def _macro_average(per_slice):
    if not per_slice:
        raise ValueError("cannot aggregate an empty slice metric mapping")
    names = list(next(iter(per_slice.values())))
    macro = {}
    for name in names:
        values = [metrics[name] for metrics in per_slice.values()]
        if name.endswith("_valid") or name.endswith("_count"):
            macro[name] = int(sum(values))
        else:
            macro[name] = float(np.mean(values))
    return macro


def evaluate_by_slice(
    model,
    dataset,
    device,
    use_amp,
    full_neighbors,
    full_xy,
    workers,
    description,
):
    per_slice = {}
    for local_index, name in enumerate(dataset.slice_names):
        loader = _loader(
            dataset.tagged_slice_dataset(local_index), device, workers
        )
        per_slice[name] = run_epoch(
            "visium",
            model,
            loader,
            device,
            use_amp=use_amp,
            report_baseline=True,
            description=f"{description}:{name}",
            full_neighbor=full_neighbors,
            full_xy=full_xy,
        )
    return _macro_average(per_slice), per_slice


def _safe_name(name: str):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "slice"


def save_preprocessing(output_dir, prepared, full_neighbors):
    np.savez(
        output_dir / "preprocessing.npz",
        genes=prepared.genes,
        gene_scale=prepared.gene_scale,
        slice_names=np.asarray([item.name for item in prepared.slices]),
        slice_roles=np.asarray([item.role for item in prepared.slices]),
        slice_paths=np.asarray([item.path for item in prepared.slices]),
        target_sum=np.asarray(prepared.target_sum, dtype=np.float32),
    )
    slice_dir = output_dir / "preprocessing_slices"
    slice_dir.mkdir(exist_ok=True)
    for index, item in enumerate(prepared.slices):
        np.savez(
            slice_dir / f"{index:03d}_{_safe_name(item.name)}.npz",
            name=np.asarray(item.name),
            role=np.asarray(item.role),
            physical_xy=item.data.physical_xy,
            array_row=item.array_row,
            array_col=item.array_col,
            native_neighbor_index=full_neighbors[index].cpu().numpy(),
            observed_spots=item.data.observed_spots,
            validation_spots=item.data.validation_spots,
            test_spots=item.data.test_spots,
        )


@torch.no_grad()
def save_test_predictions(
    output_dir,
    model,
    dataset,
    prepared,
    device,
    use_amp,
    full_neighbors,
    full_xy,
    workers,
):
    prediction_dir = output_dir / "test_predictions"
    prediction_dir.mkdir(exist_ok=True)
    index_payload = []
    for local_index, name in enumerate(dataset.slice_names):
        tagged = dataset.tagged_slice_dataset(local_index)
        loader = _loader(tagged, device, workers)
        values = collect_predictions(
            "visium", model, loader, device, use_amp,
            full_neighbors, full_xy,
        )
        values["genes"] = prepared.genes
        filename = f"{local_index:03d}_{_safe_name(name)}.npz"
        np.savez(prediction_dir / filename, **values)
        index_payload.append({"slice": name, "file": filename})
    (output_dir / "test_predictions_index.json").write_text(
        json.dumps(index_payload, indent=2), encoding="utf-8"
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
    manifest_path = resolve_run_manifest(args)

    print("Preparing leakage-safe multi-slice Visium data ...")
    prepared = prepare_multislice_visium(
        manifest_path=str(manifest_path),
        count_file=args.count_file,
        n_genes=args.n_genes,
        target_sum=args.target_sum,
        seed=args.seed,
        apply_rms_scale=not args.no_rms_scale,
    )
    datasets = {
        role: MultiSlicePointDataset(
            prepared,
            role,
            masks_per_slice=args.masks_per_slice,
            mask_target_fraction=args.mask_target_fraction,
            idw_neighbors=args.query_neighbors,
            seed=args.seed,
        )
        for role in ("train", "val", "test")
    }
    full_neighbors = [
        build_native_hex_neighbors(
            torch.from_numpy(item.array_row),
            torch.from_numpy(item.array_col),
        ).to(device)
        for item in prepared.slices
    ]
    full_xy = [
        torch.from_numpy(item.data.physical_xy).to(device, dtype=torch.float32)
        for item in prepared.slices
    ]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = _loader(
        datasets["train"], device, args.workers,
        shuffle=True, generator=generator,
    )

    model = VisiumNORMST(
        n_genes=len(prepared.genes),
        width=args.width,
        num_heads=args.num_heads,
        num_layers=args.operator_layers,
        operator_mode=args.operator_mode,
        fusion=args.fusion,
        learnable_alpha=args.learnable_alpha,
        query_neighbors=args.query_neighbors,
        idw_power=args.idw_power,
        query_chunk_size=args.query_chunk_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = vars(args).copy()
    for key in ("output_dir", "visium_root", "manifest"):
        if config[key] is not None:
            config[key] = str(config[key])
    config.update({
        "model": "VisiumNORMST",
        "protocol": "slice_level_train_val_test",
        "resolved_manifest": str(manifest_path.resolve()),
        "batch_size": 1,
        "n_selected_genes": len(prepared.genes),
        "train_slices": datasets["train"].slice_names,
        "val_slices": datasets["val"].slice_names,
        "test_slices": datasets["test"].slice_names,
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
    np.savetxt(args.output_dir / "genes.txt", prepared.genes, fmt="%s")
    save_preprocessing(args.output_dir, prepared, full_neighbors)
    print(
        "Slices:",
        f"train={len(datasets['train'].slice_names)},",
        f"val={len(datasets['val'].slice_names)},",
        f"test={len(datasets['test'].slice_names)}",
    )
    print(f"Trainable parameters: {config['trainable_parameters']:,}")

    history = []
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        datasets["train"].set_epoch(epoch)
        learning_rate_used = optimizer.param_groups[0]["lr"]
        train_started = perf_counter()
        train_metrics = run_epoch(
            "visium",
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
            description=f"train {epoch + 1}/{args.epochs}",
            full_neighbor=full_neighbors,
            full_xy=full_xy,
        )
        train_seconds = perf_counter() - train_started
        val_started = perf_counter()
        val_macro, val_per_slice = evaluate_by_slice(
            model, datasets["val"], device, use_amp,
            full_neighbors, full_xy, args.workers,
            description=f"val {epoch + 1}/{args.epochs}",
        )
        val_seconds = perf_counter() - val_started
        record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate_used,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_macro_{key}": value for key, value in val_macro.items()},
            "val_per_slice": val_per_slice,
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
        if val_macro["loss"] < best_val - args.min_delta:
            best_val = val_macro["loss"]
            stale_epochs = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best validation macro loss={best_val:.6f}"
            )
            break

    best = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best["model"])
    test_macro, test_per_slice = evaluate_by_slice(
        model, datasets["test"], device, use_amp,
        full_neighbors, full_xy, args.workers, description="test",
    )
    test_metrics = {
        "best_epoch": best["epoch"],
        "macro": test_macro,
        "per_slice": test_per_slice,
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    save_test_predictions(
        args.output_dir, model, datasets["test"], prepared,
        device, use_amp, full_neighbors, full_xy, args.workers,
    )
    print("Test:", json.dumps(test_metrics))


if __name__ == "__main__":
    main()
