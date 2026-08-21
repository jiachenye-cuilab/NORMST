#!/usr/bin/env python
"""Validation-only attribution audit for ProNORMST variance compression.

The audit has two parts.  First, it reads the validation summaries saved at the
locked best epoch for all Round9 formal runs.  Second, it replays only the
validation masks of representative full checkpoints and decomposes prediction
amplitude, calibration slope, and the effect of oracle variance restoration.
No test metric or test prediction participates in the output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

import numpy as np
import torch

from datasets.pro_normst import prepare_pro_normst_data
from models.pro_normst import ProNORMST
from training.pro_normst import _build_fixed_banks, _build_geometries, _split_identity
from training.pro_normst_engine import build_padded_model_batch, scientific_metrics


SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = SOURCE_ROOT / "save/pro_normst/pro-v2-round-009/formal_matrix.json"
DEFAULT_PILOT = SOURCE_ROOT / "save/pro_normst/pro-v2-round-009/pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replay-seed", type=int, default=2027)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_mean(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return fmean(finite) if finite else None


def _finite_pstdev(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return pstdev(finite) if len(finite) > 1 else (0.0 if finite else None)


def _best_validation_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in matrix["runs"]:
        run_dir = Path(run["run_dir"])
        history = _read_json(run_dir / "history.json")
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        best_epoch = int(checkpoint["best_epoch"])
        matches = [item for item in history if int(item["epoch"]) == best_epoch]
        if len(matches) != 1:
            raise ValueError(f"locked best epoch is absent or duplicated: {run_dir}")
        best = matches[0]
        saved_criterion = float(checkpoint["best_value"])
        replayed_criterion = float(best["validation"]["criterion_weighted_z_smooth_l1"])
        if abs(saved_criterion - replayed_criterion) > 1e-12:
            raise ValueError(f"best criterion mismatch: {run_dir}")
        for family in ("ordinary", "gap"):
            summary = best["validation"]["families"][family]["summary"]["model"]
            rows.append(
                {
                    "variant": run["variant"],
                    "fold": run["fold"],
                    "seed": int(run["seed"]),
                    "family": family,
                    "best_epoch": int(best["epoch"]),
                    "criterion": replayed_criterion,
                    **{
                        key: summary.get(key)
                        for key in (
                            "smooth_l1",
                            "rmse",
                            "mae",
                            "gene_pearson",
                            "spot_pearson",
                            "variance_ratio_median",
                            "variance_ratio_q25",
                            "variance_ratio_q75",
                        )
                    },
                }
            )
    return rows


def _aggregate_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "criterion",
        "best_epoch",
        "smooth_l1",
        "rmse",
        "mae",
        "gene_pearson",
        "spot_pearson",
        "variance_ratio_median",
        "variance_ratio_q25",
        "variance_ratio_q75",
    )
    output: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in rows}):
        output[variant] = {}
        for family in ("ordinary", "gap"):
            selected = [
                row for row in rows if row["variant"] == variant and row["family"] == family
            ]
            by_fold = {}
            for fold in sorted({row["fold"] for row in selected}):
                fold_rows = [row for row in selected if row["fold"] == fold]
                by_fold[fold] = {
                    key: _finite_mean(row[key] for row in fold_rows) for key in metrics
                }
            output[variant][family] = {
                "n_runs": len(selected),
                "mean": {key: _finite_mean(row[key] for row in selected) for key in metrics},
                "seed_and_fold_pstdev": {
                    key: _finite_pstdev(row[key] for row in selected) for key in metrics
                },
                "range": {
                    key: [
                        min(float(row[key]) for row in selected if row[key] is not None),
                        max(float(row[key]) for row in selected if row[key] is not None),
                    ]
                    for key in metrics
                },
                "fold_first_mean": {
                    key: _finite_mean(value[key] for value in by_fold.values()) for key in metrics
                },
                "by_fold": by_fold,
            }
    return output


def _pearson_per_gene(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    left = prediction.astype(np.float64) - prediction.mean(axis=0, keepdims=True)
    right = target.astype(np.float64) - target.mean(axis=0, keepdims=True)
    numerator = (left * right).sum(axis=0)
    denominator = np.sqrt((left * left).sum(axis=0) * (right * right).sum(axis=0))
    result = np.full(prediction.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _mask_calibration(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction64 = prediction.astype(np.float64)
    target64 = target.astype(np.float64)
    prediction_centered = prediction64 - prediction64.mean(axis=0, keepdims=True)
    target_centered = target64 - target64.mean(axis=0, keepdims=True)
    prediction_variance = np.mean(np.square(prediction_centered), axis=0)
    target_variance = np.mean(np.square(target_centered), axis=0)
    covariance = np.mean(prediction_centered * target_centered, axis=0)
    scale_floor = max(float(np.median(target_variance)) * 1e-8, 1e-12)
    valid_target = target_variance > scale_floor
    valid_prediction = prediction_variance > scale_floor
    valid_both = valid_target & valid_prediction
    variance_ratio = prediction_variance[valid_target] / target_variance[valid_target]
    std_ratio = np.sqrt(variance_ratio)
    prediction_on_truth_slope = covariance[valid_target] / target_variance[valid_target]
    gene_pearson = _pearson_per_gene(prediction64, target64)

    variance_restored = prediction64.copy()
    if valid_both.any():
        restored_scale = np.sqrt(target_variance[valid_both] / prediction_variance[valid_both])
        variance_restored[:, valid_both] = (
            target64[:, valid_both].mean(axis=0, keepdims=True)
            + prediction_centered[:, valid_both] * restored_scale[None, :]
        )
    oracle_affine = prediction64.copy()
    if valid_both.any():
        truth_on_prediction_slope = covariance[valid_both] / prediction_variance[valid_both]
        oracle_affine[:, valid_both] = (
            target64[:, valid_both].mean(axis=0, keepdims=True)
            + prediction_centered[:, valid_both] * truth_on_prediction_slope[None, :]
        )

    raw_metrics = scientific_metrics(prediction64, target64)
    variance_restored_metrics = scientific_metrics(variance_restored, target64)
    oracle_affine_metrics = scientific_metrics(oracle_affine, target64)
    return {
        "n_queries": int(target.shape[0]),
        "n_genes": int(target.shape[1]),
        "valid_target_variance_genes": int(valid_target.sum()),
        "valid_prediction_variance_genes": int(valid_both.sum()),
        "variance_ratio_median": float(np.median(variance_ratio)),
        "variance_ratio_q25": float(np.quantile(variance_ratio, 0.25)),
        "variance_ratio_q75": float(np.quantile(variance_ratio, 0.75)),
        "std_ratio_median": float(np.median(std_ratio)),
        "prediction_on_truth_slope_median": float(np.median(prediction_on_truth_slope)),
        "prediction_on_truth_slope_q25": float(np.quantile(prediction_on_truth_slope, 0.25)),
        "prediction_on_truth_slope_q75": float(np.quantile(prediction_on_truth_slope, 0.75)),
        "gene_pearson_median": float(np.nanmedian(gene_pearson)),
        "global_centered_energy_ratio": float(
            np.square(prediction_centered).sum() / np.square(target_centered).sum()
        ),
        "mean_bias_rmse": float(
            np.sqrt(np.mean((prediction64.mean(axis=0) - target64.mean(axis=0)) ** 2))
        ),
        "raw": raw_metrics,
        "oracle_variance_restored": variance_restored_metrics,
        "oracle_per_gene_affine": oracle_affine_metrics,
    }


def _mean_nested_numeric(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for record in records for key in record})
    output: dict[str, Any] = {}
    for key in keys:
        values = [record[key] for record in records if key in record]
        if values and all(isinstance(value, dict) for value in values):
            output[key] = _mean_nested_numeric(values)
        elif values and all(value is None or isinstance(value, (int, float)) for value in values):
            output[key] = _finite_mean(values)
    return output


@torch.no_grad()
def _replay_run(run_dir: Path, device: torch.device) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    if config["variant"] != "full":
        raise ValueError(f"raw replay is restricted to full checkpoints: {run_dir}")
    data = prepare_pro_normst_data(
        Path(config["manifest"]),
        Path(config["panel"]),
        count_file=config["count_file"],
    )
    protocol, fold = _split_identity(data)
    geometries = _build_geometries(data)
    banks = _build_fixed_banks(data, geometries, protocol, fold)
    for item in data.roles["val"]:
        item.preload(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = ProNORMST(
        torch.as_tensor(data.preprocessing.gene_mean_z), variant="full"
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    use_amp = device.type == "cuda"
    depths: dict[str, Any] = {}
    for round_limit in (1, 2, 4):
        family_slices: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for family in ("ordinary", "gap"):
            for item in data.roles["val"]:
                mask_records: list[dict[str, Any]] = []
                masks = banks["val"][item.slice_id][family]
                for offset in range(0, len(masks), 4):
                    mask_batch = masks[offset : offset + 4]
                    packed = build_padded_model_batch(
                        [item] * len(mask_batch), mask_batch, device
                    )
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        prediction_z = model(
                            packed.visible_z,
                            packed.visible_index,
                            packed.query_index,
                            packed.geometry,
                            round_limit=round_limit,
                        )
                    prediction = prediction_z.detach().float().cpu().numpy()
                    for batch_index, mask in enumerate(mask_batch):
                        length = packed.query_lengths[batch_index]
                        prediction_x = (
                            prediction[batch_index, :length]
                            * data.preprocessing.gene_scale[None, :]
                        )
                        target_x = item.expression_x[mask.query_index]
                        mask_records.append(_mask_calibration(prediction_x, target_x))
                family_slices[family].append(
                    {
                        "slice_id": item.slice_id,
                        "summary": _mean_nested_numeric(mask_records),
                    }
                )
        depths[f"round{round_limit}"] = {
            family: {
                "summary": _mean_nested_numeric(
                    [record["summary"] for record in family_slices[family]]
                ),
                "slices": family_slices[family],
            }
            for family in ("ordinary", "gap")
        }
    return {
        "run_dir": str(run_dir.resolve()),
        "protocol": protocol,
        "fold": fold,
        "seed": int(config["initialization_seed"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_criterion": float(checkpoint["best_value"]),
        "depths": depths,
    }


def main() -> int:
    args = parse_args()
    matrix = _read_json(args.matrix.resolve())
    if matrix.get("schema") != "pro-normst-formal-matrix-v1":
        raise ValueError("formal matrix schema is incompatible")
    rows = _best_validation_rows(matrix)
    selected = [
        Path(run["run_dir"])
        for run in matrix["runs"]
        if run["variant"] == "full" and int(run["seed"]) == args.replay_seed
    ]
    selected.append(args.pilot_dir.resolve())
    device = torch.device(args.device)
    replay = [_replay_run(run_dir, device) for run_dir in selected]
    payload = {
        "schema": "pro-normst-round10-variance-audit-v1",
        "scope": "validation_only",
        "test_metrics_used": False,
        "selection_role": "next_round_hypothesis_only",
        "formal_history": {
            "n_rows": len(rows),
            "rows": rows,
            "aggregate": _aggregate_history(rows),
        },
        "raw_validation_replay": replay,
        "oracle_diagnostics": {
            "selection_allowed": False,
            "meaning": (
                "Variance-restored and per-gene affine metrics use the same validation "
                "targets and are descriptive upper bounds, never deployable transforms."
            ),
        },
        "limitations": [
            "Raw replay uses seed2027 because formal diagnostics show initialization variation is much smaller than held-out-donor variation.",
            "The shared dataset constructor materializes every role, but this audit only indexes and forwards validation slices.",
            "Oracle calibration uses validation truth and may not be used for checkpoint selection, hyperparameter fitting, or test transformation.",
        ],
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
