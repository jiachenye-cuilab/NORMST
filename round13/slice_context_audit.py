"""Validation-only Round13 audit of visible-slice context corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from datasets.pro_normst import ProNORMSTData, ProNORMSTSlice, prepare_pro_normst_data
from models.pro_normst import ProNORMST
from training.pro_normst_engine import (
    build_padded_model_batch,
    scientific_metrics,
    weighted_gene_smooth_l1_per_item,
)
from training.pro_normst_masks import ProMask, build_mask_geometry, fixed_mask_bank


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROUND9_ROOT = SOURCE_ROOT / "save" / "pro_normst" / "pro-v2-round-009"
DEFAULT_OUTPUT = (
    SOURCE_ROOT
    / "save"
    / "pro_normst"
    / "pro-v2-round-013-audit"
    / "slice_context_audit.json"
)
FOLDS = ("lodo_d1", "lodo_d2", "lodo_d3")
FAMILIES = ("ordinary", "gap")
METHODS = ("baseline", "train_bias", "context_affine")
TRAIN_AUDIT_ROLE = "context-audit-train"
VALIDATION_ROLE = "val"
MASKS_PER_SLICE_FAMILY = 16
EVALUATION_BATCH_SIZE = 4
SEED = 2027
VARIANT = "full"
PROTOCOL = "pair_grouped_lodo"
REPLAY_TOLERANCE = 1e-5
FAMILY_NONWORSE_TOLERANCE = 1e-5
PEARSON_DROP_TOLERANCE = 0.001
RAW_METRIC_KEYS = (
    "smooth_l1",
    "mae",
    "rmse",
    "gene_pearson",
    "spot_pearson",
    "variance_ratio_median",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round9-root", type=Path, default=ROUND9_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("cannot average empty or non-finite values")
    return float(array.mean())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace audit artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fit_context_affine(
    contexts: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit train-only per-gene residual bias and centered context slope."""
    context = np.asarray(contexts, dtype=np.float64)
    residual = np.asarray(residuals, dtype=np.float64)
    if (
        context.shape != residual.shape
        or context.ndim != 2
        or context.shape[0] < 2
        or not np.isfinite(context).all()
        or not np.isfinite(residual).all()
    ):
        raise ValueError("contexts and residuals must be finite matching [M,G] arrays")
    context_mean = context.mean(axis=0)
    residual_mean = residual.mean(axis=0)
    centered_context = context - context_mean[None, :]
    centered_residual = residual - residual_mean[None, :]
    denominator = np.sum(centered_context * centered_context, axis=0)
    numerator = np.sum(centered_context * centered_residual, axis=0)
    slope = np.zeros_like(residual_mean)
    nonconstant = denominator > 0
    slope[nonconstant] = numerator[nonconstant] / denominator[nonconstant]
    return {
        "context_mean": context_mean.astype(np.float32),
        "residual_mean": residual_mean.astype(np.float32),
        "slope": slope.astype(np.float32),
        "context_nonconstant_genes": np.asarray(int(nonconstant.sum())),
    }


def context_correction(
    context: np.ndarray,
    fit: dict[str, np.ndarray],
) -> np.ndarray:
    value = np.asarray(context, dtype=np.float32)
    return (
        fit["residual_mean"]
        + fit["slope"] * (value - fit["context_mean"])
    ).astype(np.float32)


def aggregate_validation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-weight masks within slice, then slices within each family."""
    if not records:
        raise ValueError("validation records must not be empty")
    output: dict[str, Any] = {}
    metric_keys = ("weighted_z_smooth_l1", *RAW_METRIC_KEYS)
    for method in METHODS:
        families: dict[str, Any] = {}
        for family in FAMILIES:
            family_records = [record for record in records if record["family"] == family]
            slice_ids = sorted({record["slice_id"] for record in family_records})
            if not slice_ids:
                raise ValueError(f"missing validation records for family {family}")
            slice_metrics: dict[str, dict[str, float]] = {}
            for slice_id in slice_ids:
                selected = [
                    record["methods"][method]
                    for record in family_records
                    if record["slice_id"] == slice_id
                ]
                slice_metrics[slice_id] = {
                    key: _mean(record[key] for record in selected) for key in metric_keys
                }
            summary = {
                key: _mean(item[key] for item in slice_metrics.values())
                for key in metric_keys
            }
            families[family] = {
                "n_slices": len(slice_ids),
                "n_masks": len(family_records),
                "summary": summary,
                "slices": slice_metrics,
            }
        criterion = _mean(
            families[family]["summary"]["weighted_z_smooth_l1"]
            for family in FAMILIES
        )
        output[method] = {
            "criterion_weighted_z_smooth_l1": criterion,
            "families": families,
        }
    return output


def paired_gains(methods: dict[str, Any]) -> dict[str, Any]:
    baseline = methods["baseline"]
    gains: dict[str, Any] = {}
    for method in METHODS[1:]:
        candidate = methods[method]
        family_gains: dict[str, Any] = {}
        for family in FAMILIES:
            left = baseline["families"][family]["summary"]
            right = candidate["families"][family]["summary"]
            family_gains[family] = {
                "weighted_z_smooth_l1_gain": float(
                    left["weighted_z_smooth_l1"] - right["weighted_z_smooth_l1"]
                ),
                "smooth_l1_gain": float(left["smooth_l1"] - right["smooth_l1"]),
                "mae_gain": float(left["mae"] - right["mae"]),
                "rmse_gain": float(left["rmse"] - right["rmse"]),
                "gene_pearson_delta": float(
                    right["gene_pearson"] - left["gene_pearson"]
                ),
                "spot_pearson_delta": float(
                    right["spot_pearson"] - left["spot_pearson"]
                ),
            }
        gains[method] = {
            "criterion_gain": float(
                baseline["criterion_weighted_z_smooth_l1"]
                - candidate["criterion_weighted_z_smooth_l1"]
            ),
            "families": family_gains,
        }
    return gains


def audit_decision(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared validation-only FiLM feasibility gates."""
    if len(folds) != len(FOLDS):
        raise ValueError("the decision requires exactly three LODO folds")
    overall_methods: dict[str, Any] = {}
    for method in METHODS:
        family_summaries: dict[str, dict[str, float]] = {}
        for family in FAMILIES:
            family_summaries[family] = {
                key: _mean(
                    fold["methods"][method]["families"][family]["summary"][key]
                    for fold in folds
                )
                for key in ("weighted_z_smooth_l1", *RAW_METRIC_KEYS)
            }
        overall_methods[method] = {
            "criterion_weighted_z_smooth_l1": _mean(
                fold["methods"][method]["criterion_weighted_z_smooth_l1"]
                for fold in folds
            ),
            "families": family_summaries,
        }

    baseline = overall_methods["baseline"]
    bias = overall_methods["train_bias"]
    context = overall_methods["context_affine"]
    context_gain = (
        baseline["criterion_weighted_z_smooth_l1"]
        - context["criterion_weighted_z_smooth_l1"]
    )
    context_over_bias_gain = (
        bias["criterion_weighted_z_smooth_l1"]
        - context["criterion_weighted_z_smooth_l1"]
    )
    positive_folds = sum(
        fold["gains"]["context_affine"]["criterion_gain"] > 0 for fold in folds
    )
    family_regressions = {
        family: float(
            context["families"][family]["weighted_z_smooth_l1"]
            - baseline["families"][family]["weighted_z_smooth_l1"]
        )
        for family in FAMILIES
    }
    pearson_deltas = {
        f"{family}_{metric}": float(
            context["families"][family][metric]
            - baseline["families"][family][metric]
        )
        for family in FAMILIES
        for metric in ("gene_pearson", "spot_pearson")
    }
    gates = {
        "baseline_replay_all_folds": all(fold["replay"]["passed"] for fold in folds),
        "context_gain_positive": context_gain > 0,
        "context_beats_train_bias": context_over_bias_gain > 0,
        "positive_in_at_least_two_folds": positive_folds >= 2,
        "families_not_worse_beyond_tolerance": all(
            value <= FAMILY_NONWORSE_TOLERANCE
            for value in family_regressions.values()
        ),
        "pearson_not_worse_beyond_tolerance": all(
            value >= -PEARSON_DROP_TOLERANCE for value in pearson_deltas.values()
        ),
    }
    return {
        "supports_round13_film": all(gates.values()),
        "gates": gates,
        "overall": overall_methods,
        "context_criterion_gain": float(context_gain),
        "context_over_train_bias_criterion_gain": float(context_over_bias_gain),
        "context_positive_fold_count": int(positive_folds),
        "family_context_regression": family_regressions,
        "context_pearson_deltas": pearson_deltas,
    }


def _visible_context(item: ProNORMSTSlice, mask: ProMask) -> np.ndarray:
    return np.mean(
        item.expression_z[mask.visible_index].astype(np.float64), axis=0
    ).astype(np.float32)


@torch.no_grad()
def _predict_batch(
    model: ProNORMST,
    item: ProNORMSTSlice,
    masks: Sequence[ProMask],
    device: torch.device,
) -> tuple[torch.Tensor, Any]:
    packed = build_padded_model_batch([item] * len(masks), masks, device)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        output = model(
            packed.visible_z,
            packed.visible_index,
            packed.query_index,
            packed.geometry,
        )
    prediction = output[0] if isinstance(output, tuple) else output
    return prediction.detach().float(), packed


def _build_banks(
    data: ProNORMSTData,
    fold: str,
) -> tuple[dict[str, dict[str, tuple[ProMask, ...]]], dict[str, dict[str, tuple[ProMask, ...]]]]:
    train_banks: dict[str, dict[str, tuple[ProMask, ...]]] = {}
    validation_banks: dict[str, dict[str, tuple[ProMask, ...]]] = {}
    for item in data.roles["train"]:
        geometry = build_mask_geometry(item.neighbor_index)
        train_banks[item.slice_id] = {
            family: fixed_mask_bank(
                geometry,
                protocol=PROTOCOL,
                fold=fold,
                role=TRAIN_AUDIT_ROLE,
                slice_id=item.slice_id,
                family=family,
                size=MASKS_PER_SLICE_FAMILY,
            )
            for family in FAMILIES
        }
    for item in data.roles[VALIDATION_ROLE]:
        geometry = build_mask_geometry(item.neighbor_index)
        validation_banks[item.slice_id] = {
            family: fixed_mask_bank(
                geometry,
                protocol=PROTOCOL,
                fold=fold,
                role=VALIDATION_ROLE,
                slice_id=item.slice_id,
                family=family,
                size=MASKS_PER_SLICE_FAMILY,
            )
            for family in FAMILIES
        }
    return train_banks, validation_banks


def _fit_from_train_masks(
    model: ProNORMST,
    data: ProNORMSTData,
    banks: dict[str, dict[str, tuple[ProMask, ...]]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    contexts: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    counts = {family: 0 for family in FAMILIES}
    for item in data.roles["train"]:
        for family in FAMILIES:
            masks = banks[item.slice_id][family]
            for start in range(0, len(masks), EVALUATION_BATCH_SIZE):
                batch = masks[start : start + EVALUATION_BATCH_SIZE]
                prediction, packed = _predict_batch(model, item, batch, device)
                prediction_np = prediction.cpu().numpy()
                target_np = packed.target_z.detach().float().cpu().numpy()
                for offset, mask in enumerate(batch):
                    length = packed.query_lengths[offset]
                    contexts.append(_visible_context(item, mask))
                    residuals.append(
                        np.mean(
                            target_np[offset, :length] - prediction_np[offset, :length],
                            axis=0,
                            dtype=np.float64,
                        ).astype(np.float32)
                    )
                    counts[family] += 1
    fit = fit_context_affine(np.stack(contexts), np.stack(residuals))
    diagnostics = {
        "n_observations": len(contexts),
        "n_masks_by_family": counts,
        "context_nonconstant_genes": int(fit["context_nonconstant_genes"]),
        "residual_bias_rms": float(np.sqrt(np.mean(fit["residual_mean"] ** 2))),
        "context_slope_rms": float(np.sqrt(np.mean(fit["slope"] ** 2))),
    }
    return fit, diagnostics


def _validation_records(
    model: ProNORMST,
    data: ProNORMSTData,
    banks: dict[str, dict[str, tuple[ProMask, ...]]],
    fit: dict[str, np.ndarray],
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    weights = torch.as_tensor(
        data.preprocessing.positive_weight, dtype=torch.float32, device=device
    )
    gene_scale = data.preprocessing.gene_scale
    for item in data.roles[VALIDATION_ROLE]:
        for family in FAMILIES:
            masks = banks[item.slice_id][family]
            for start in range(0, len(masks), EVALUATION_BATCH_SIZE):
                batch = masks[start : start + EVALUATION_BATCH_SIZE]
                prediction, packed = _predict_batch(model, item, batch, device)
                contexts = np.stack([_visible_context(item, mask) for mask in batch])
                correction = {
                    "baseline": np.zeros_like(contexts, dtype=np.float32),
                    "train_bias": np.broadcast_to(
                        fit["residual_mean"], contexts.shape
                    ).copy(),
                    "context_affine": np.stack(
                        [context_correction(value, fit) for value in contexts]
                    ),
                }
                method_predictions: dict[str, np.ndarray] = {}
                method_losses: dict[str, list[float]] = {}
                for method in METHODS:
                    delta = torch.as_tensor(
                        correction[method], dtype=torch.float32, device=device
                    )[:, None, :]
                    corrected = prediction + delta
                    method_losses[method] = (
                        weighted_gene_smooth_l1_per_item(
                            corrected,
                            packed.target_z,
                            weights,
                            packed.query_valid,
                        )
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    method_predictions[method] = corrected.detach().cpu().numpy()
                for offset, mask in enumerate(batch):
                    length = packed.query_lengths[offset]
                    target_x = item.expression_x[mask.query_index]
                    methods: dict[str, Any] = {}
                    for method in METHODS:
                        prediction_x = (
                            method_predictions[method][offset, :length]
                            * gene_scale[None, :]
                        )
                        raw_metrics = scientific_metrics(prediction_x, target_x)
                        methods[method] = {
                            "weighted_z_smooth_l1": float(
                                method_losses[method][offset]
                            ),
                            **{key: float(raw_metrics[key]) for key in RAW_METRIC_KEYS},
                        }
                    records.append(
                        {
                            "slice_id": item.slice_id,
                            "family": family,
                            "mask_index": start + offset,
                            "methods": methods,
                        }
                    )
    return records


def _load_locked_fold(
    round9_root: Path,
    fold: str,
    device: torch.device,
) -> tuple[ProNORMSTData, ProNORMST, dict[str, Any]]:
    run_dir = round9_root / fold / f"seed{SEED}" / VARIANT
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best.pt"
    status_path = run_dir / "run_status.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected_config = {
        "fold": fold,
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "initialization_seed": SEED,
        "evidence_tier": "formal-lodo",
        "round_identity": "pro-v2-round-009",
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Round9 run config mismatch for {fold}: {mismatches}")
    if status.get("status") != "complete":
        raise ValueError(f"Round9 run is not complete for {fold}")
    actual_checkpoint_hash = _file_sha256(checkpoint_path)
    if actual_checkpoint_hash != status.get("checkpoint_sha256"):
        raise ValueError(f"Round9 best checkpoint hash mismatch for {fold}")

    data = prepare_pro_normst_data(
        config["manifest"], config["panel"], count_file=config["count_file"]
    )
    model = ProNORMST(
        torch.as_tensor(data.preprocessing.gene_mean_z, dtype=torch.float32),
        variant=VARIANT,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "pro-normst-checkpoint-v1":
        raise ValueError(f"invalid Round9 checkpoint schema for {fold}")
    if checkpoint.get("contract_hash") != config["contract_hash"]:
        raise ValueError(f"Round9 checkpoint contract mismatch for {fold}")
    model.load_state_dict(checkpoint["model"], strict=True)
    locked_best = float(status["best_validation"])
    if abs(float(checkpoint["best_value"]) - locked_best) > 1e-12:
        raise ValueError(f"Round9 checkpoint/status best value mismatch for {fold}")
    model.eval()
    provenance = {
        "run_dir": str(run_dir.resolve()),
        "config_sha256": _file_sha256(config_path),
        "checkpoint_sha256": actual_checkpoint_hash,
        "contract_hash": config["contract_hash"],
        "locked_best_validation": locked_best,
        "best_epoch": int(status["best_epoch"]),
    }
    return data, model, provenance


def run_fold(round9_root: Path, fold: str, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    data, model, provenance = _load_locked_fold(round9_root, fold, device)
    load_seconds = time.perf_counter() - started
    for role in ("train", VALIDATION_ROLE):
        for item in data.roles[role]:
            item.preload(device)
    banks_started = time.perf_counter()
    train_banks, validation_banks = _build_banks(data, fold)
    bank_seconds = time.perf_counter() - banks_started
    fit_started = time.perf_counter()
    fit, fit_diagnostics = _fit_from_train_masks(
        model, data, train_banks, device
    )
    fit_seconds = time.perf_counter() - fit_started
    validation_started = time.perf_counter()
    records = _validation_records(
        model, data, validation_banks, fit, device
    )
    validation_seconds = time.perf_counter() - validation_started
    methods = aggregate_validation_records(records)
    replay_value = methods["baseline"]["criterion_weighted_z_smooth_l1"]
    replay_error = abs(replay_value - provenance["locked_best_validation"])
    result = {
        "fold": fold,
        "provenance": provenance,
        "fit": fit_diagnostics,
        "validation": {
            "role": VALIDATION_ROLE,
            "n_masks": len(records),
            "n_slices": len(data.roles[VALIDATION_ROLE]),
        },
        "methods": methods,
        "gains": paired_gains(methods),
        "replay": {
            "locked": provenance["locked_best_validation"],
            "observed": replay_value,
            "absolute_error": replay_error,
            "tolerance": REPLAY_TOLERANCE,
            "passed": replay_error <= REPLAY_TOLERANCE,
        },
        "runtime_seconds": {
            "load": load_seconds,
            "build_masks": bank_seconds,
            "fit_train_masks": fit_seconds,
            "evaluate_validation": validation_seconds,
            "total": time.perf_counter() - started,
        },
    }
    del model, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.perf_counter()
    folds: list[dict[str, Any]] = []
    for fold in FOLDS:
        result = run_fold(args.round9_root.resolve(), fold, device)
        folds.append(result)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "replay": result["replay"],
                    "context_gain": result["gains"]["context_affine"][
                        "criterion_gain"
                    ],
                    "seconds": result["runtime_seconds"]["total"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    decision = audit_decision(folds)
    payload = {
        "schema": "pro-normst-slice-context-audit-v1",
        "evidence_tier": "validation-only-diagnostic",
        "round_identity": "pro-v2-round-013-audit",
        "source_round": "pro-v2-round-009",
        "test_metrics_used": False,
        "model_structure_changed": False,
        "training_performed": False,
        "device": str(device),
        "protocol": {
            "folds": list(FOLDS),
            "variant": VARIANT,
            "seed": SEED,
            "train_mask_role": TRAIN_AUDIT_ROLE,
            "validation_mask_role": VALIDATION_ROLE,
            "masks_per_slice_family": MASKS_PER_SLICE_FAMILY,
            "families": list(FAMILIES),
            "replay_tolerance": REPLAY_TOLERANCE,
            "family_nonworse_tolerance": FAMILY_NONWORSE_TOLERANCE,
            "pearson_drop_tolerance": PEARSON_DROP_TOLERANCE,
        },
        "folds": folds,
        "decision": decision,
        "runtime_seconds": {"total": time.perf_counter() - started},
    }
    _write_json_once(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), **decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
