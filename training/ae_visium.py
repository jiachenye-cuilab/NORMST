"""Train AE-NORMST on frozen 32-D composition coordinates.

The frozen AE defines the selected genes and composition coordinates. NORMST
predicts only those coordinates. Full-gene library size may be supplied for
visible spots as context, but is never a hidden-spot target or model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.ae_masked_visium import (
    MultiSlicePointDataset,
    prepare_ae_multislice_visium,
    verify_source_contract,
)
from models.ae_normst import AENORMST
from models.frozen_composition_ae import FrozenCompositionAE, sha256_file
from models.geometry_adaptive_normst import build_native_hex_neighbors
from training.ae_engine import collect_ae_predictions, run_ae_epoch
from training.engine import seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        help="fixed train/val/test manifest used to fit the frozen AE",
    )
    parser.add_argument(
        "--ae-checkpoint", type=Path,
        help="completed frozen composition AE best.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="AE-NORMST checkpoint for --predict-only; default: output_dir/best.pt",
    )
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="save compact 32-D test predictions after training",
    )
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--ae-encode-batch-size", type=int, default=256)
    parser.add_argument(
        "--library-context", choices=("zero", "visible"), default="zero",
        help=(
            "zero is the composition-only control; visible injects train-fitted "
            "standardized full-gene log-library for visible spots only"
        ),
    )
    parser.add_argument("--composition-loss-weight", type=float, default=0.1)
    parser.add_argument("--mask-target-fraction", type=float, default=0.25)
    parser.add_argument("--masks-per-slice", type=int, default=64)
    parser.add_argument("--query-neighbors", type=int, default=6)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--num-layers", "--operator-layers", dest="operator_layers",
        type=int, default=4,
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
    parser.add_argument("--alpha-global", type=float, default=1.0)
    parser.add_argument(
        "--residual-head-width-multiplier", type=int, choices=(1, 2), default=2
    )
    parser.add_argument("--input-coordinate-lifting", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    if not args.predict_only and (args.manifest is None or args.ae_checkpoint is None):
        raise ValueError("training requires --manifest and --ae-checkpoint")
    if args.predict_only and args.save_predictions:
        raise ValueError("--predict-only already saves test predictions")
    if not args.predict_only and args.checkpoint is not None:
        raise ValueError("--checkpoint is only valid with --predict-only")
    positive = (
        args.ae_encode_batch_size, args.masks_per_slice, args.query_neighbors,
        args.idw_power, args.query_chunk_size, args.width, args.num_heads,
        args.operator_layers, args.epochs,
    )
    if min(positive) <= 0:
        raise ValueError("model, masking, encoding, and training sizes must be positive")
    if args.width % args.num_heads:
        raise ValueError("width must be divisible by num_heads")
    if not 0 < args.mask_target_fraction < 1:
        raise ValueError("mask_target_fraction must be between zero and one")
    finite_nonnegative = (
        args.alpha_global, args.composition_loss_weight, args.min_delta,
        args.weight_decay, args.max_grad_norm,
    )
    if not all(np.isfinite(value) and value >= 0 for value in finite_nonnegative):
        raise ValueError("loss weights and non-negative model settings must be finite")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if args.workers < 0 or args.patience < 0:
        raise ValueError("workers and patience must be non-negative")


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
        raise ValueError("cannot aggregate empty slice metrics")
    macro = {}
    for name in next(iter(per_slice.values())):
        values = [metrics[name] for metrics in per_slice.values()]
        if name.endswith("_valid") or name.endswith("_count"):
            macro[name] = int(sum(values))
        else:
            macro[name] = float(np.mean(values))
    return macro


def evaluate_by_slice(
    model, frozen_ae, dataset, device, use_amp, tensors, workers,
    library_context_mode, composition_loss_weight, description,
):
    per_slice = {}
    for local_index, name in enumerate(dataset.slice_names):
        loader = _loader(dataset.tagged_slice_dataset(local_index), device, workers)
        per_slice[name] = run_ae_epoch(
            model, frozen_ae, loader, device,
            full_neighbors=tensors["neighbors"],
            full_xy=tensors["xy"],
            full_composition=tensors["composition"],
            full_library_context=tensors["library_context"],
            full_counts=tensors["counts"],
            library_context_mode=library_context_mode,
            composition_loss_weight=composition_loss_weight,
            use_amp=use_amp,
            report_baseline=True,
            detailed_metrics=True,
            description=f"{description}:{name}",
        )
    return _macro_average(per_slice), per_slice


def _safe_name(name):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "slice"


def _genes_sha256(genes):
    return hashlib.sha256(
        ("\n".join(np.asarray(genes).astype(str)) + "\n").encode("utf-8")
    ).hexdigest()


def parameter_breakdown(model):
    groups = {
        "composition_encoder": model.expression_encoder,
        "visible_library_context": model.library_context_lifting,
        "coordinate_lifting": getattr(model, "coordinate_lifting", torch.nn.Identity()),
        "operator_blocks": model.blocks,
        "physical_query_decoder": model.query_decoder,
        "latent_residual_decoder": model.residual_projection,
    }
    result = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in groups.items()
    }
    result["total"] = sum(result.values())
    return result


def _full_tensors(prepared, device):
    return {
        "neighbors": [
            build_native_hex_neighbors(
                torch.from_numpy(item.array_row), torch.from_numpy(item.array_col)
            ).to(device)
            for item in prepared.slices
        ],
        "xy": [
            torch.from_numpy(item.data.physical_xy).to(device, dtype=torch.float32)
            for item in prepared.slices
        ],
        "composition": [
            torch.from_numpy(item.data.expression).to(device, dtype=torch.float32)
            for item in prepared.slices
        ],
        "library_context": [
            torch.from_numpy(item.standardized_library_context[:, None]).to(
                device, dtype=torch.float32
            )
            for item in prepared.slices
        ],
        "counts": [
            torch.from_numpy(item.selected_counts).to(device, dtype=torch.float32)
            for item in prepared.slices
        ],
    }


def _model_from_config(config, composition_dim, device):
    return AENORMST(
        composition_dim=composition_dim,
        width=int(config.get("width", 256)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get("operator_layers", config.get("num_layers", 4))),
        operator_mode=config.get("operator_mode", "parallel"),
        fusion=config.get("fusion", "add"),
        learnable_alpha=bool(config.get("learnable_alpha", False)),
        alpha_global=float(config.get("alpha_global", 1.0)),
        query_neighbors=int(config.get("query_neighbors", 6)),
        idw_power=float(config.get("idw_power", 2.0)),
        query_chunk_size=int(config.get("query_chunk_size", 1024)),
        residual_head_width_multiplier=int(
            config.get("residual_head_width_multiplier", 2)
        ),
        input_coordinate_lifting=bool(config.get("input_coordinate_lifting", False)),
    ).to(device)


def save_preprocessing(output_dir, prepared, frozen_ae, tensors):
    np.savez(
        output_dir / "preprocessing.npz",
        genes=prepared.genes,
        composition_mean=frozen_ae.composition_mean.cpu().numpy(),
        composition_scale=frozen_ae.composition_scale.cpu().numpy(),
        library_context_mean=np.asarray(prepared.library_context_mean, dtype=np.float32),
        library_context_scale=np.asarray(prepared.library_context_scale, dtype=np.float32),
        library_context_definition=np.asarray(
            "log1p full-gene total UMI; mean/std fitted on train spots only"
        ),
        slice_names=np.asarray([item.name for item in prepared.slices]),
        slice_roles=np.asarray([item.role for item in prepared.slices]),
        slice_paths=np.asarray([item.path for item in prepared.slices]),
    )
    slice_dir = output_dir / "preprocessing_slices"
    slice_dir.mkdir(exist_ok=False)
    for index, item in enumerate(prepared.slices):
        np.savez(
            slice_dir / f"{index:03d}_{_safe_name(item.name)}.npz",
            name=np.asarray(item.name),
            role=np.asarray(item.role),
            barcodes=item.barcodes,
            physical_xy=item.data.physical_xy,
            array_row=item.array_row,
            array_col=item.array_col,
            native_neighbor_index=tensors["neighbors"][index].cpu().numpy(),
        )


@torch.no_grad()
def save_test_predictions(
    output_dir, model, dataset, prepared, device, use_amp, tensors,
    workers, library_context_mode,
):
    prediction_dir = output_dir / "test_predictions"
    prediction_dir.mkdir(exist_ok=True)
    index_payload = []
    for local_index, name in enumerate(dataset.slice_names):
        loader = _loader(dataset.tagged_slice_dataset(local_index), device, workers)
        values = collect_ae_predictions(
            model, loader, device, use_amp,
            tensors["neighbors"], tensors["xy"], tensors["composition"],
            tensors["library_context"], tensors["counts"], library_context_mode,
        )
        values["latent_names"] = np.asarray([
            f"ae_composition_{index:02d}" for index in range(prepared.composition_dim)
        ])
        filename = f"{local_index:03d}_{_safe_name(name)}.npz"
        np.savez(prediction_dir / filename, **values)
        index_payload.append({"slice": name, "file": filename})
    (output_dir / "test_predictions_index.json").write_text(
        json.dumps(index_payload, indent=2), encoding="utf-8"
    )


def _prepare(args, manifest_path, ae_checkpoint, device, use_amp, config=None):
    frozen_ae = FrozenCompositionAE.from_checkpoint(ae_checkpoint, map_location=device)
    prepared = prepare_ae_multislice_visium(
        manifest_path=manifest_path,
        frozen_ae=frozen_ae,
        device=device,
        count_file=(config or {}).get("count_file", args.count_file),
        encode_batch_size=int((config or {}).get(
            "ae_encode_batch_size", args.ae_encode_batch_size
        )),
        use_amp=use_amp,
    )
    seed = int((config or {}).get("seed", args.seed))
    datasets = {
        role: MultiSlicePointDataset(
            prepared,
            role,
            masks_per_slice=int((config or {}).get(
                "masks_per_slice", args.masks_per_slice
            )),
            mask_target_fraction=float((config or {}).get(
                "mask_target_fraction", args.mask_target_fraction
            )),
            idw_neighbors=int((config or {}).get(
                "query_neighbors", args.query_neighbors
            )),
            seed=seed,
            materialize_values=False,
        )
        for role in ("train", "val", "test")
    }
    return frozen_ae, prepared, datasets, _full_tensors(prepared, device)


def run_prediction_only(args, device, use_amp):
    run_dir = args.output_dir.resolve()
    config_path = _existing_file("training config", [run_dir / "config.json"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_path = _existing_file(
        "AE-NORMST checkpoint", [args.checkpoint, run_dir / "best.pt"]
    )
    manifest_path = _existing_file(
        "manifest", [args.manifest, config.get("resolved_manifest"), run_dir / "manifest.json"]
    )
    ae_checkpoint = _existing_file(
        "frozen AE checkpoint", [args.ae_checkpoint, config.get("ae_checkpoint")]
    )
    if sha256_file(checkpoint_path) != config.get("best_checkpoint_sha256", sha256_file(checkpoint_path)):
        raise ValueError("AE-NORMST best checkpoint hash differs from config")
    seed_everything(int(config.get("seed", 2027)))
    frozen_ae, prepared, datasets, tensors = _prepare(
        args, manifest_path, ae_checkpoint, device, use_amp, config=config
    )
    if frozen_ae.checkpoint_sha256 != config.get("ae_checkpoint_sha256"):
        raise ValueError("frozen AE checkpoint hash differs from training config")
    if _genes_sha256(prepared.genes) != config.get("genes_sha256"):
        raise ValueError("selected gene contract differs from training config")
    model = _model_from_config(config, prepared.composition_dim, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    save_test_predictions(
        run_dir, model, datasets["test"], prepared, device, use_amp, tensors,
        args.workers, config.get("library_context", "zero"),
    )
    print(f"Saved latent test predictions to {run_dir / 'test_predictions'}")
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

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"training output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()
    ae_checkpoint = args.ae_checkpoint.resolve()
    seed_everything(args.seed)
    print("Preparing frozen AE composition data ...")
    frozen_ae, prepared, datasets, tensors = _prepare(
        args, manifest_path, ae_checkpoint, device, use_amp
    )
    shutil.copyfile(manifest_path, output_dir / "manifest.json")
    if sha256_file(output_dir / "manifest.json") != frozen_ae.manifest_sha256:
        raise RuntimeError("copied manifest no longer matches the frozen AE")

    model = _model_from_config(vars(args), prepared.composition_dim, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = _loader(
        datasets["train"], device, args.workers, shuffle=True, generator=generator
    )

    config = vars(args).copy()
    for key in ("manifest", "ae_checkpoint", "output_dir", "checkpoint"):
        if config[key] is not None:
            config[key] = str(Path(config[key]).resolve())
    breakdown = parameter_breakdown(model)
    config.update({
        "task": "ae_visium",
        "model": "AENORMST",
        "protocol": "random_pair_grouped_8_2_2_from_frozen_AE_manifest",
        "prediction_target": "standardized frozen-AE composition latent only",
        "composition_dim": prepared.composition_dim,
        "n_selected_genes": len(prepared.genes),
        "resolved_manifest": str(manifest_path),
        "manifest_sha256": prepared.manifest_sha256,
        "ae_checkpoint": str(ae_checkpoint),
        "ae_checkpoint_sha256": frozen_ae.checkpoint_sha256,
        "genes_sha256": _genes_sha256(prepared.genes),
        "checkpoint_weights_frozen": True,
        "hidden_library_used_by_model": False,
        "library_is_prediction_target": False,
        "library_context_scope": "visible spots only",
        "library_context_definition": (
            "log1p full-gene total UMI; standardized with train-spot mean/std"
        ),
        "checkpoint_metric": "val_macro_reconstruction_latent_smooth_l1",
        "batch_size": 1,
        "num_layers": args.operator_layers,
        "train_slices": datasets["train"].slice_names,
        "val_slices": datasets["val"].slice_names,
        "test_slices": datasets["test"].slice_names,
        "parameter_breakdown": breakdown,
        "trainable_parameters": breakdown["total"],
        "frozen_ae_parameters": sum(
            parameter.numel() for parameter in frozen_ae.parameters()
        ),
        "device_resolved": str(device),
        "amp_enabled": use_amp,
        "source_contract": prepared.source_contract,
    })
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    np.savetxt(output_dir / "genes.txt", prepared.genes, fmt="%s")
    save_preprocessing(output_dir, prepared, frozen_ae, tensors)
    print(
        f"Slices: train={len(datasets['train'].slice_names)}, "
        f"val={len(datasets['val'].slice_names)}, "
        f"test={len(datasets['test'].slice_names)}"
    )
    print(
        f"Composition={prepared.composition_dim}, genes={len(prepared.genes)}, "
        f"trainable parameters={breakdown['total']:,}"
    )

    history = []
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        verify_source_contract(prepared.source_contract)
        datasets["train"].set_epoch(epoch)
        learning_rate_used = optimizer.param_groups[0]["lr"]
        train_started = perf_counter()
        train_metrics = run_ae_epoch(
            model, frozen_ae, train_loader, device,
            full_neighbors=tensors["neighbors"], full_xy=tensors["xy"],
            full_composition=tensors["composition"],
            full_library_context=tensors["library_context"],
            full_counts=tensors["counts"],
            library_context_mode=args.library_context,
            composition_loss_weight=args.composition_loss_weight,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            max_grad_norm=args.max_grad_norm, detailed_metrics=False,
            description=f"train {epoch + 1}/{args.epochs}",
        )
        train_seconds = perf_counter() - train_started
        val_started = perf_counter()
        val_macro, val_per_slice = evaluate_by_slice(
            model, frozen_ae, datasets["val"], device, use_amp, tensors,
            args.workers, args.library_context, args.composition_loss_weight,
            f"val {epoch + 1}/{args.epochs}",
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
        (output_dir / "history.json").write_text(
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
            "ae_checkpoint_sha256": frozen_ae.checkpoint_sha256,
            "manifest_sha256": prepared.manifest_sha256,
            "genes_sha256": config["genes_sha256"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_macro["reconstruction"] < best_val - args.min_delta:
            best_val = val_macro["reconstruction"]
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best validation latent SmoothL1={best_val:.6f}"
            )
            break

    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    val_macro, val_per_slice = evaluate_by_slice(
        model, frozen_ae, datasets["val"], device, use_amp, tensors,
        args.workers, args.library_context, args.composition_loss_weight,
        "best-val",
    )
    val_metrics = {
        "best_epoch": best["epoch"],
        "macro": val_macro,
        "per_slice": val_per_slice,
    }
    (output_dir / "val_metrics.json").write_text(
        json.dumps(val_metrics, indent=2), encoding="utf-8"
    )
    test_macro, test_per_slice = evaluate_by_slice(
        model, frozen_ae, datasets["test"], device, use_amp, tensors,
        args.workers, args.library_context, args.composition_loss_weight, "test",
    )
    test_metrics = {
        "best_epoch": best["epoch"],
        "macro": test_macro,
        "per_slice": test_per_slice,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    config["best_checkpoint_sha256"] = sha256_file(output_dir / "best.pt")
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    if args.save_predictions:
        save_test_predictions(
            output_dir, model, datasets["test"], prepared, device, use_amp,
            tensors, args.workers, args.library_context,
        )
    print("Test:", json.dumps(test_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
