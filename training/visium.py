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

HVGs are fitted on raw counts from all spots in training slices only. A
positive ``--target-sum`` applies library-size normalization before ``log1p``;
zero or a negative value skips it. Gene-wise RMS scales are fitted on training
slices by default and can be disabled with ``--no-rms-scale``.
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
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.multislice_masked_visium import (
    MultiSlicePointDataset,
    prepare_multislice_visium,
)
from models.geometry_adaptive_normst import (
    VisiumNORMST,
    build_native_hex_neighbors,
)
from training.engine import (
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
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--visium-root", type=Path,
        help="directory whose direct children are standard Visium slices",
    )
    source.add_argument(
        "--manifest", type=Path,
        help="optional prebuilt slice-level split manifest",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="checkpoint for --predict-only; defaults to output_dir/best.pt",
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help=(
            "load a completed run and export test predictions without training; "
            "model and preprocessing settings come from output_dir"
        ),
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="export test_predictions after training (disabled by default)",
    )
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument(
        "--target-sum", type=float, default=1e4,
        help=(
            "positive value enables library-size normalization before log1p; "
            "use 0 or -1 to apply log1p directly to raw counts"
        ),
    )
    parser.add_argument(
        "--no-rms-scale", action="store_true",
        help="skip gene-wise RMS scaling after the configured log1p transform",
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

    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--num-layers", "--operator-layers",
        dest="operator_layers", type=int, default=4,
        help=(
            "number of local-global operator blocks; --operator-layers is a "
            "compatibility alias"
        ),
    )
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
    parser.add_argument(
        "--alpha-global", type=float, default=1.0,
        help=(
            "initial global residual scale; fixed unless --learnable-alpha is set"
        ),
    )
    parser.add_argument(
        "--baseline-calibration", action="store_true",
        help="enable identity-initialized learned gene-wise IDW scale and bias",
    )
    parser.add_argument(
        "--residual-head-width-multiplier",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "residual MLP hidden width relative to model width; 1 reproduces "
            "the legacy-width head while 2 preserves the current default"
        ),
    )
    parser.add_argument(
        "--calibration-only", action="store_true",
        help=(
            "train and evaluate only GeneAffine(IDW); requires "
            "--baseline-calibration and fixes all other parameters"
        ),
    )
    parser.add_argument(
        "--input-coordinate-lifting", action="store_true",
        help=(
            "inject normalized within-slice coordinates into initial tokens; "
            "disabled by default and intended for the phase-two ablation"
        ),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--loss-mode", choices=("smooth_l1", "structure_aware"),
        default="smooth_l1",
        help=(
            "training objective; checkpoint selection remains validation "
            "SmoothL1 reconstruction, and the default preserves the old baseline"
        ),
    )
    parser.add_argument("--gene-correlation-loss-weight", type=float, default=0.1)
    parser.add_argument("--variance-loss-weight", type=float, default=0.01)
    parser.add_argument("--negative-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--min-structure-target-variance", type=float, default=1e-6,
        help="exclude effectively flat target gene maps from structural losses",
    )
    parser.add_argument(
        "--loss-gene-weighting",
        choices=("none", "inv_sqrt_std", "inv_std"),
        default="none",
        help=(
            "optional training-only gene weighting fitted from training slices; "
            "input normalization is unchanged"
        ),
    )
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
    if not args.predict_only and args.visium_root is None and args.manifest is None:
        raise ValueError("training requires --visium-root or --manifest")
    if args.predict_only and args.visium_root is not None:
        raise ValueError(
            "--predict-only uses the saved split; pass a local --manifest instead "
            "of --visium-root"
        )
    if args.predict_only and args.save_predictions:
        raise ValueError("--predict-only already exports predictions")
    if not args.predict_only and args.checkpoint is not None:
        raise ValueError("--checkpoint is only valid with --predict-only")
    positive = (
        args.n_genes, args.masks_per_slice,
        args.query_neighbors, args.idw_power, args.query_chunk_size,
        args.width, args.num_heads, args.operator_layers, args.epochs,
    )
    if min(positive) <= 0:
        raise ValueError("model, preprocessing, and training sizes must be positive")
    if not np.isfinite(args.target_sum):
        raise ValueError("target_sum must be finite")
    if not np.isfinite(args.alpha_global) or args.alpha_global < 0:
        raise ValueError("alpha_global must be finite and non-negative")
    if args.calibration_only and not args.baseline_calibration:
        raise ValueError(
            "--calibration-only requires --baseline-calibration"
        )
    if args.calibration_only and args.input_coordinate_lifting:
        raise ValueError(
            "--input-coordinate-lifting has no effect with --calibration-only"
        )
    if args.width % args.num_heads:
        raise ValueError("width must be divisible by num_heads")
    if not 0.0 < args.mask_target_fraction < 1.0:
        raise ValueError("mask_target_fraction must be between zero and one")
    if args.workers < 0 or args.patience < 0:
        raise ValueError("workers and patience must be non-negative")
    if args.min_delta < 0 or args.max_grad_norm < 0:
        raise ValueError("min_delta and max_grad_norm must be non-negative")
    loss_values = (
        args.gene_correlation_loss_weight,
        args.variance_loss_weight,
        args.negative_loss_weight,
        args.min_structure_target_variance,
    )
    if min(loss_values) < 0:
        raise ValueError("loss weights and variance threshold must be non-negative")
    if args.loss_mode == "structure_aware" and not any(loss_values[:3]):
        raise ValueError("structure_aware loss requires at least one positive weight")


def build_loss_config(args):
    if args.loss_mode == "smooth_l1":
        return {"mode": "smooth_l1"}
    return {
        "mode": "structure_aware",
        "gene_correlation_weight": args.gene_correlation_loss_weight,
        "variance_weight": args.variance_loss_weight,
        "negative_weight": args.negative_loss_weight,
        "min_target_variance": args.min_structure_target_variance,
    }


def fit_training_gene_weights(prepared, mode: str, epsilon: float = 1e-8):
    """Fit gene standard deviations and optional weights on train slices only."""
    training = [
        item.data.expression.astype(np.float64, copy=False)
        for item in prepared.slices
        if item.role == "train"
    ]
    if not training:
        raise ValueError("gene loss weighting requires at least one training slice")
    count = sum(len(values) for values in training)
    value_sum = sum(
        (values.sum(axis=0) for values in training),
        start=np.zeros(len(prepared.genes), dtype=np.float64),
    )
    square_sum = sum(
        (np.square(values).sum(axis=0) for values in training),
        start=np.zeros(len(prepared.genes), dtype=np.float64),
    )
    mean = value_sum / max(count, 1)
    variance = np.maximum(square_sum / max(count, 1) - np.square(mean), 0.0)
    standard_deviation = np.sqrt(variance)
    if mode == "none":
        weight = np.ones_like(standard_deviation)
    elif mode == "inv_sqrt_std":
        weight = 1.0 / np.sqrt(standard_deviation + epsilon)
    elif mode == "inv_std":
        weight = 1.0 / (standard_deviation + epsilon)
    else:
        raise ValueError(f"unsupported gene loss weighting: {mode}")
    weight /= np.mean(weight)
    return standard_deviation.astype(np.float32), weight.astype(np.float32)


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
    full_expression,
    workers,
    description,
    loss_config=None,
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
            full_expression=full_expression,
            loss_config=loss_config,
        )
    return _macro_average(per_slice), per_slice


def _safe_name(name: str):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "slice"


def save_preprocessing(
    output_dir,
    prepared,
    full_neighbors,
    training_gene_std,
    loss_gene_weight,
):
    np.savez(
        output_dir / "preprocessing.npz",
        genes=prepared.genes,
        gene_scale=prepared.gene_scale,
        slice_names=np.asarray([item.name for item in prepared.slices]),
        slice_roles=np.asarray([item.role for item in prepared.slices]),
        slice_paths=np.asarray([item.path for item in prepared.slices]),
        target_sum=np.asarray(prepared.target_sum, dtype=np.float32),
        expression_transform=np.asarray(prepared.expression_transform),
        training_gene_std=training_gene_std,
        loss_gene_weight=loss_gene_weight,
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
    full_expression,
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
            full_neighbors, full_xy, full_expression,
        )
        values["genes"] = prepared.genes
        filename = f"{local_index:03d}_{_safe_name(name)}.npz"
        np.savez(prediction_dir / filename, **values)
        index_payload.append({"slice": name, "file": filename})
    (output_dir / "test_predictions_index.json").write_text(
        json.dumps(index_payload, indent=2), encoding="utf-8"
    )


def trainable_parameter_breakdown(model):
    groups = {
        "expression_encoder": model.expression_encoder,
        "coordinate_lifting": getattr(model, "coordinate_lifting", nn.Identity()),
        "operator_blocks": model.blocks,
        "physical_query_decoder": model.query_decoder,
        "residual_decoder": model.residual_projection,
        "baseline_calibration": model.baseline_calibration,
    }
    result = {
        name: sum(
            parameter.numel() for parameter in module.parameters()
            if parameter.requires_grad
        )
        for name, module in groups.items()
    }
    result["total"] = sum(result.values())
    return result


def configure_trainable_parameters(model, calibration_only):
    if not calibration_only:
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.baseline_calibration.parameters():
        parameter.requires_grad_(True)


def _existing_file(label, candidates):
    checked = []
    for value in candidates:
        if value is None:
            continue
        path = Path(value)
        checked.append(str(path))
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"{label} was not found; checked: {', '.join(checked)}")


def _model_from_config(config, n_genes, device):
    return VisiumNORMST(
        n_genes=n_genes,
        width=int(config.get("width", 256)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get(
            "operator_layers", config.get("num_layers", 4)
        )),
        operator_mode=config.get("operator_mode", "parallel"),
        fusion=config.get("fusion", "add"),
        learnable_alpha=bool(config.get("learnable_alpha", False)),
        alpha_global=float(config.get("alpha_global", 1.0)),
        query_neighbors=int(config.get("query_neighbors", 6)),
        idw_power=float(config.get("idw_power", 2.0)),
        query_chunk_size=int(config.get("query_chunk_size", 1024)),
        baseline_calibration=bool(config.get("baseline_calibration", False)),
        residual_head_width_multiplier=int(config.get(
            "residual_head_width_multiplier", 2
        )),
        calibration_only=bool(config.get("calibration_only", False)),
        input_coordinate_lifting=bool(config.get(
            "input_coordinate_lifting", False
        )),
    ).to(device)


def run_prediction_only(args, device, use_amp):
    run_dir = args.output_dir.resolve()
    config_path = _existing_file("training config", [run_dir / "config.json"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_path = _existing_file(
        "checkpoint", [args.checkpoint, run_dir / "best.pt"]
    )
    manifest_path = _existing_file(
        "manifest",
        [
            args.manifest,
            run_dir / "manifest.json",
            config.get("resolved_manifest"),
        ],
    )
    preprocessing_path = _existing_file(
        "preprocessing", [run_dir / "preprocessing.npz"]
    )
    genes_path = _existing_file("genes", [run_dir / "genes.txt"])
    genes = np.atleast_1d(np.loadtxt(genes_path, dtype=str))
    with np.load(preprocessing_path, allow_pickle=False) as preprocessing:
        preprocessing_genes = preprocessing["genes"].astype(str)
        gene_scale = preprocessing["gene_scale"].astype(np.float32)
    if not np.array_equal(genes, preprocessing_genes):
        raise ValueError(
            "genes.txt and preprocessing.npz contain different gene orders"
        )

    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    prepared = prepare_multislice_visium(
        manifest_path=str(manifest_path),
        count_file=config.get("count_file", "filtered_feature_bc_matrix.h5"),
        n_genes=len(genes),
        target_sum=float(config.get("target_sum", 1e4)),
        seed=seed,
        apply_rms_scale=not bool(config.get("no_rms_scale", False)),
        fixed_genes=genes,
        fixed_gene_scale=gene_scale,
    )
    query_neighbors = int(config.get(
        "query_neighbors", config.get("idw_neighbors", 6)
    ))
    dataset = MultiSlicePointDataset(
        prepared,
        "test",
        masks_per_slice=int(config.get("masks_per_slice", 64)),
        mask_target_fraction=float(config.get("mask_target_fraction", 0.25)),
        idw_neighbors=query_neighbors,
        seed=seed,
        materialize_values=False,
    )
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
    full_expression = [
        torch.from_numpy(item.data.expression).to(device, dtype=torch.float32)
        for item in prepared.slices
    ]
    model = _model_from_config(config, len(prepared.genes), device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    save_test_predictions(
        run_dir,
        model,
        dataset,
        prepared,
        device,
        use_amp,
        full_neighbors,
        full_xy,
        full_expression,
        args.workers,
    )
    print(f"Saved test predictions to {run_dir / 'test_predictions'}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    requested_device = torch.device(args.device)
    device = (
        torch.device("cpu")
        if requested_device.type == "cuda" and not torch.cuda.is_available()
        else requested_device
    )
    use_amp = device.type == "cuda" and not args.no_amp
    if args.predict_only:
        return run_prediction_only(args, device, use_amp)

    seed_everything(args.seed)
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
    training_gene_std, loss_gene_weight = fit_training_gene_weights(
        prepared, args.loss_gene_weighting
    )
    loss_config = build_loss_config(args)
    if args.loss_gene_weighting != "none":
        loss_config["gene_weight"] = torch.from_numpy(
            loss_gene_weight
        ).to(device)
    datasets = {
        role: MultiSlicePointDataset(
            prepared,
            role,
            masks_per_slice=args.masks_per_slice,
            mask_target_fraction=args.mask_target_fraction,
            idw_neighbors=args.query_neighbors,
            seed=args.seed,
            materialize_values=False,
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
    full_expression = [
        torch.from_numpy(item.data.expression).to(
            device, dtype=torch.float32
        )
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
        alpha_global=args.alpha_global,
        query_neighbors=args.query_neighbors,
        idw_power=args.idw_power,
        query_chunk_size=args.query_chunk_size,
        baseline_calibration=args.baseline_calibration,
        residual_head_width_multiplier=args.residual_head_width_multiplier,
        calibration_only=args.calibration_only,
        input_coordinate_lifting=args.input_coordinate_lifting,
    ).to(device)
    configure_trainable_parameters(model, args.calibration_only)
    optimized_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not optimized_parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = vars(args).copy()
    for key in ("output_dir", "visium_root", "manifest", "checkpoint"):
        if config[key] is not None:
            config[key] = str(config[key])
    config.update({
        "task": "visium",
        "model": "VisiumNORMST",
        "protocol": "slice_level_train_val_test",
        "expression_transform": prepared.expression_transform,
        "checkpoint_metric": "val_macro_reconstruction",
        "resolved_manifest": str(manifest_path.resolve()),
        "batch_size": 1,
        "n_selected_genes": len(prepared.genes),
        "num_layers": args.operator_layers,
        "train_slices": datasets["train"].slice_names,
        "val_slices": datasets["val"].slice_names,
        "test_slices": datasets["test"].slice_names,
        "parameter_breakdown": trainable_parameter_breakdown(model),
        "device_resolved": str(device),
        "amp_enabled": use_amp,
        "slice_expression_resident_on_device": True,
        "slice_expression_bytes": sum(
            value.numel() * value.element_size()
            for value in full_expression
        ),
    })
    config["trainable_parameters"] = config["parameter_breakdown"]["total"]
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    np.savetxt(args.output_dir / "genes.txt", prepared.genes, fmt="%s")
    save_preprocessing(
        args.output_dir,
        prepared,
        full_neighbors,
        training_gene_std,
        loss_gene_weight,
    )
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
            full_expression=full_expression,
            detailed_metrics=False,
            loss_config=loss_config,
        )
        train_seconds = perf_counter() - train_started
        val_started = perf_counter()
        val_macro, val_per_slice = evaluate_by_slice(
            model, datasets["val"], device, use_amp,
            full_neighbors, full_xy, full_expression, args.workers,
            description=f"val {epoch + 1}/{args.epochs}",
            loss_config=loss_config,
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
        if val_macro["reconstruction"] < best_val - args.min_delta:
            best_val = val_macro["reconstruction"]
            stale_epochs = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best validation macro reconstruction={best_val:.6f}"
            )
            break

    best = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best["model"])
    test_macro, test_per_slice = evaluate_by_slice(
        model, datasets["test"], device, use_amp,
        full_neighbors, full_xy, full_expression, args.workers,
        description="test",
        loss_config=loss_config,
    )
    test_metrics = {
        "best_epoch": best["epoch"],
        "macro": test_macro,
        "per_slice": test_per_slice,
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    if args.save_predictions:
        save_test_predictions(
            args.output_dir, model, datasets["test"], prepared,
            device, use_amp, full_neighbors, full_xy, full_expression,
            args.workers,
        )
    print("Test:", json.dumps(test_metrics))
    return 0
