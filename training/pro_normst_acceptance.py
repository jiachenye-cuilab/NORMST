"""Fail-closed aggregation for the frozen ProNORMST formal matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from training.pro_normst_engine import canonical_json, file_sha256


FOLDS = ("lodo_d1", "lodo_d2", "lodo_d3")
SEEDS = (2027, 2028, 2029)
VARIANTS = ("full", "one-shot", "local-only", "global-only")
DEPTH_ACCEPTANCE_FAMILY = "gap"
ROUND_INVARIANCE_RTOL = 2e-3
ROUND_INVARIANCE_ATOL = 2e-4
FORMAL_ARTIFACTS_DIRECTORY = "formal_artifacts"
SCIENTIFIC_METRICS = (
    "smooth_l1",
    "mae",
    "rmse",
    "gene_pearson",
    "spot_pearson",
    "variance_ratio_median",
    "variance_ratio_q25",
    "variance_ratio_q75",
    "variance_ratio_defined",
    "negative_fraction",
    "positive_mae",
    "positive_rmse",
    "zero_mae",
    "zero_rmse",
)
PAIRED_ERROR_METRICS = (
    "smooth_l1",
    "mae",
    "rmse",
    "positive_mae",
    "positive_rmse",
    "zero_mae",
    "zero_rmse",
)
PAIRED_CORRELATION_METRICS = ("gene_pearson", "spot_pearson")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _prediction_tree_sha256(root: Path) -> str:
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise ValueError(f"test prediction tree is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _load_run(run_dir: Path, identity: tuple[str, int, str]) -> dict[str, Any]:
    fold, seed, variant = identity
    test_artifacts = run_dir / "test_artifacts"
    config = _read_json(run_dir / "config.json")
    contract = _read_json(run_dir / "contract_manifest.json")
    metrics = _read_json(test_artifacts / "test_metrics.json")
    complete = _read_json(test_artifacts / "test_complete.json")
    checkpoint_lock_path = run_dir / "run_checkpoint_lock.json"
    checkpoint_lock = _read_json(checkpoint_lock_path)
    status = _read_json(run_dir / "run_status.json")
    gradient = _read_json(run_dir / "gradient_gate.json")
    bptt = _read_json(run_dir / "final_loss_bptt_gate.json")
    expected = {
        "protocol": "pair_grouped_lodo",
        "fold": fold,
        "initialization_seed": seed,
        "variant": variant,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{run_dir}: config {key}={config.get(key)!r}, expected {value!r}")
    if config.get("model") != "pro-normst" or config.get("evidence_tier") != "formal-lodo":
        raise ValueError(f"{run_dir}: run is not a formal ProNORMST run")
    if not config.get("candidate_lock"):
        raise ValueError(f"{run_dir}: formal run has no candidate lock")
    if status.get("status") != "complete" or status.get("test_run") is not True:
        raise ValueError(f"{run_dir}: formal test is incomplete")
    if complete.get("status") != "complete":
        raise ValueError(f"{run_dir}: test completion marker is invalid")
    if (
        checkpoint_lock.get("schema") != "pro-normst-run-checkpoint-lock-v1"
        or checkpoint_lock.get("status") != "locked"
        or checkpoint_lock.get("contract_hash") != config.get("contract_hash")
        or checkpoint_lock.get("checkpoint_sha256") != complete.get("checkpoint_sha256")
        or complete.get("run_checkpoint_lock_sha256")
        != file_sha256(checkpoint_lock_path)
    ):
        raise ValueError(f"{run_dir}: formal run checkpoint lock is invalid")
    lock_identity = {
        "human_contract_version": config.get("human_contract_version"),
        "round_identity": config.get("round_identity"),
        "fold": fold,
        "initialization_seed": seed,
        "variant": variant,
    }
    if any(checkpoint_lock.get(key) != value for key, value in lock_identity.items()):
        raise ValueError(f"{run_dir}: formal run checkpoint lock identity drifted")
    if gradient.get("passed") is not True or bptt.get("passed") is not True:
        raise ValueError(f"{run_dir}: gradient/BPTT gate failed")
    contract_hash = config.get("contract_hash")
    if metrics.get("_meta", {}).get("contract_hash") != contract_hash:
        raise ValueError(f"{run_dir}: test metrics contract hash mismatch")
    if complete.get("contract_hash") != contract_hash:
        raise ValueError(f"{run_dir}: test marker contract hash mismatch")
    if complete.get("test_metrics_sha256") != file_sha256(test_artifacts / "test_metrics.json"):
        raise ValueError(f"{run_dir}: test metrics changed after completion")
    if complete.get("prediction_tree_sha256") != _prediction_tree_sha256(
        test_artifacts / "predictions"
    ):
        raise ValueError(f"{run_dir}: test predictions changed after completion")
    expected_rounds = {"round1", "round2", "round4"} if variant == "full" else {
        "round1" if variant == "one-shot" else "round4"
    }
    actual_rounds = {key for key in metrics if key.startswith("round")}
    if actual_rounds != expected_rounds:
        raise ValueError(
            f"{run_dir}: round artifacts {sorted(actual_rounds)} != {sorted(expected_rounds)}"
        )
    return {
        "run_dir": run_dir,
        "test_artifacts": test_artifacts,
        "config": config,
        "contract": contract,
        "metrics": metrics,
        "complete": complete,
        "checkpoint_lock": checkpoint_lock,
        "gradient": gradient,
        "bptt": bptt,
    }


def _idw_prediction_tree_hash(run: dict[str, Any]) -> str:
    round_number = 1 if run["config"]["variant"] == "one-shot" else 4
    root = run["test_artifacts"] / "predictions" / f"test_round{round_number}"
    files = sorted(root.glob("*/*/mask_*.npz"))
    if not files:
        raise ValueError(f"{run['run_dir']}: missing raw IDW predictions")
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        with np.load(path, allow_pickle=False) as arrays:
            for key in ("query_index", "visible_index", "idw_x"):
                value = np.ascontiguousarray(arrays[key])
                digest.update(key.encode("ascii"))
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
                digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _matched_fold_contract_value(run: dict[str, Any], field: str) -> str:
    contract = run.get("contract")
    if not isinstance(contract, dict) or field not in contract:
        raise ValueError(f"{run['run_dir']}: contract is missing {field}")
    value = contract[field]
    if not isinstance(value, dict):
        raise ValueError(f"{run['run_dir']}: contract field {field} must be an object")
    return canonical_json(value)


def _matched_train_val_slice_contract(run: dict[str, Any]) -> str:
    contract = run.get("contract")
    slices = contract.get("slice_data_and_geometry") if isinstance(contract, dict) else None
    if not isinstance(slices, dict):
        raise ValueError(
            f"{run['run_dir']}: contract is missing slice_data_and_geometry"
        )
    selected: dict[str, Any] = {}
    roles: set[str] = set()
    for slice_id, item in slices.items():
        if not isinstance(item, dict):
            raise ValueError(
                f"{run['run_dir']}: slice_data_and_geometry entry is invalid: {slice_id}"
            )
        role = item.get("role")
        if role in {"train", "val"}:
            selected[str(slice_id)] = item
            roles.add(str(role))
    if roles != {"train", "val"}:
        raise ValueError(
            f"{run['run_dir']}: contract must contain train and val slice data"
        )
    return canonical_json(selected)


def load_formal_matrix(path: str | Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    matrix_path = Path(path).resolve()
    payload = _read_json(matrix_path)
    if payload.get("schema") != "pro-normst-formal-matrix-v1":
        raise ValueError("formal matrix schema is incompatible")
    round_identity = payload.get("round_identity")
    if not isinstance(round_identity, str) or not round_identity:
        raise ValueError("formal matrix round identity is missing")
    entries = payload.get("runs")
    if not isinstance(entries, list):
        raise ValueError("formal matrix runs must be a list")
    expected = {(fold, seed, variant) for fold in FOLDS for seed in SEEDS for variant in VARIANTS}
    runs: dict[tuple[str, int, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("formal matrix run entries must be objects")
        identity = (str(entry.get("fold")), int(entry.get("seed")), str(entry.get("variant")))
        if identity not in expected:
            raise ValueError(f"unexpected formal run identity: {identity}")
        if identity in runs:
            raise ValueError(f"duplicate formal run identity: {identity}")
        raw_dir = Path(str(entry.get("run_dir", "")))
        run_dir = raw_dir if raw_dir.is_absolute() else matrix_path.parent / raw_dir
        runs[identity] = _load_run(run_dir.resolve(), identity)
    missing = sorted(expected.difference(runs))
    if missing:
        raise ValueError(f"formal matrix is incomplete; missing {missing}")

    candidate_lock_hashes = {
        run["contract"].get("run", {}).get("candidate_lock_sha256")
        for run in runs.values()
    }
    if None in candidate_lock_hashes or len(candidate_lock_hashes) != 1:
        raise ValueError("formal runs do not share one frozen candidate lock")
    run_rounds = {run["config"].get("round_identity") for run in runs.values()}
    human_versions = {
        run["config"].get("human_contract_version") for run in runs.values()
    }
    if run_rounds != {round_identity}:
        raise ValueError("formal runs do not share the matrix round identity")
    if human_versions != {"pro-normst-human-v9"}:
        raise ValueError("formal runs do not share the frozen human contract version")

    for fold in FOLDS:
        fold_runs = [run for identity, run in runs.items() if identity[0] == fold]
        preprocessing_contracts = {
            _matched_fold_contract_value(run, "preprocessing") for run in fold_runs
        }
        split_contracts = {
            _matched_fold_contract_value(run, "split") for run in fold_runs
        }
        train_val_slice_contracts = {
            _matched_train_val_slice_contract(run) for run in fold_runs
        }
        mask_contracts = {
            canonical_json(run["contract"]["fixed_mask_banks"]) for run in fold_runs
        }
        test_identities = {
            canonical_json(run["metrics"]["_meta"]["test_expression_x_sha256"])
            for run in fold_runs
        }
        idw_summaries = {
            canonical_json(
                {
                    family: run["metrics"][
                        "round1" if run["config"]["variant"] == "one-shot" else "round4"
                    ]["families"][family]["summary"]["idw"]
                    for family in ("ordinary", "gap")
                }
            )
            for run in fold_runs
        }
        idw_prediction_hashes = {_idw_prediction_tree_hash(run) for run in fold_runs}
        if len(preprocessing_contracts) != 1:
            raise ValueError(f"{fold}: matched preprocessing contract drifted")
        if len(split_contracts) != 1:
            raise ValueError(f"{fold}: matched split contract drifted")
        if len(train_val_slice_contracts) != 1:
            raise ValueError(
                f"{fold}: matched train/val slice data or geometry drifted"
            )
        if len(mask_contracts) != 1:
            raise ValueError(f"{fold}: matched fixed mask banks drifted")
        if (
            len(test_identities) != 1
            or len(idw_summaries) != 1
            or len(idw_prediction_hashes) != 1
        ):
            raise ValueError(f"{fold}: matched test identity or IDW baseline drifted")
    return runs


def _smooth_l1(
    run: dict[str, Any],
    round_number: int,
    family: str,
    section: str = "model",
    depth: str | None = None,
) -> float:
    summary = run["metrics"][f"round{round_number}"]["families"][family]["summary"]
    if depth is None:
        value = summary[section].get("smooth_l1")
    else:
        stratum = summary.get("strata", {}).get(f"depth:{depth}", {})
        value = stratum.get(section, {}).get("smooth_l1")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(
            f"{run['run_dir']}: undefined smooth_l1 round={round_number} "
            f"family={family} section={section} depth={depth}"
        )
    return float(value)


def _effect_summary(values: dict[tuple[str, int], float]) -> dict[str, Any]:
    expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if set(values) != expected:
        raise ValueError("paired effect does not contain all fold x seed identities")
    fold_effect = {
        fold: float(np.mean([values[(fold, seed)] for seed in SEEDS]))
        for fold in FOLDS
    }
    effects = np.asarray([fold_effect[fold] for fold in FOLDS], dtype=np.float64)
    return {
        "fold_x_init": [
            {"fold": fold, "seed": seed, "effect": values[(fold, seed)]}
            for fold in FOLDS
            for seed in SEEDS
        ],
        "fold_effect": fold_effect,
        "overall_mean": float(effects.mean()),
        "fold_sample_sd": float(effects.std(ddof=1)),
        "fold_range": [float(effects.min()), float(effects.max())],
        "positive_folds": int(np.count_nonzero(effects > 0)),
    }


def _descriptive_summary(
    values: dict[tuple[str, int], float | int | None],
) -> dict[str, Any]:
    """Summarize possibly undefined metrics without treating runs as donors."""

    expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if set(values) != expected:
        raise ValueError("descriptive metric does not contain all fold x seed identities")
    rows = []
    fold_effect: dict[str, float | None] = {}
    for fold in FOLDS:
        fold_values = []
        for seed in SEEDS:
            raw = values[(fold, seed)]
            value = (
                float(raw)
                if isinstance(raw, (int, float))
                and not isinstance(raw, bool)
                and math.isfinite(float(raw))
                else None
            )
            rows.append({"fold": fold, "seed": seed, "value": value})
            if value is not None:
                fold_values.append(value)
        fold_effect[fold] = float(np.mean(fold_values)) if fold_values else None
    defined_folds = [
        float(fold_effect[fold])
        for fold in FOLDS
        if fold_effect[fold] is not None
    ]
    return {
        "fold_x_init": rows,
        "fold_effect": fold_effect,
        "overall_mean": float(np.mean(defined_folds)) if defined_folds else None,
        "fold_sample_sd": (
            float(np.std(defined_folds, ddof=1)) if len(defined_folds) >= 2 else None
        ),
        "fold_range": (
            [float(np.min(defined_folds)), float(np.max(defined_folds))]
            if defined_folds
            else None
        ),
        "defined_runs": sum(row["value"] is not None for row in rows),
        "defined_folds": len(defined_folds),
    }


def _primary_round(variant: str) -> int:
    return 1 if variant == "one-shot" else 4


def _summary_section(
    run: dict[str, Any],
    variant: str,
    family: str,
    section: str,
) -> dict[str, Any]:
    summary = run["metrics"][f"round{_primary_round(variant)}"]["families"][family][
        "summary"
    ]
    value = summary.get(section, {})
    return value if isinstance(value, dict) else {}


def _scientific_report(
    runs: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for variant in VARIANTS:
        methods[variant] = {}
        for family in ("ordinary", "gap"):
            methods[variant][family] = {
                metric: _descriptive_summary(
                    {
                        (fold, seed): _summary_section(
                            runs[(fold, seed, variant)], variant, family, "model"
                        ).get(metric)
                        for fold in FOLDS
                        for seed in SEEDS
                    }
                )
                for metric in SCIENTIFIC_METRICS
            }

    strict_idw: dict[str, Any] = {}
    clipped_secondary: dict[str, Any] = {}
    for family in ("ordinary", "gap"):
        strict_idw[family] = {
            metric: _descriptive_summary(
                {
                    (fold, seed): _summary_section(
                        runs[(fold, seed, "full")], "full", family, "idw"
                    ).get(metric)
                    for fold in FOLDS
                    for seed in SEEDS
                }
            )
            for metric in SCIENTIFIC_METRICS
        }
        clipped_secondary[family] = {
            metric: _descriptive_summary(
                {
                    (fold, seed): _summary_section(
                        runs[(fold, seed, "full")],
                        "full",
                        family,
                        "model_clipped_zero",
                    ).get(metric)
                    for fold in FOLDS
                    for seed in SEEDS
                }
            )
            for metric in SCIENTIFIC_METRICS
        }

    controls = {
        "full_vs_idw": "idw",
        "full_vs_one_shot": "one-shot",
        "full_vs_local_only": "local-only",
        "full_vs_global_only": "global-only",
    }
    paired: dict[str, Any] = {}
    for comparison, control in controls.items():
        paired[comparison] = {}
        for family in ("ordinary", "gap"):
            family_metrics: dict[str, Any] = {}
            for metric in PAIRED_ERROR_METRICS + PAIRED_CORRELATION_METRICS:
                values: dict[tuple[str, int], float | None] = {}
                for fold in FOLDS:
                    for seed in SEEDS:
                        full_value = _summary_section(
                            runs[(fold, seed, "full")], "full", family, "model"
                        ).get(metric)
                        if control == "idw":
                            control_value = _summary_section(
                                runs[(fold, seed, "full")], "full", family, "idw"
                            ).get(metric)
                        else:
                            control_value = _summary_section(
                                runs[(fold, seed, control)], control, family, "model"
                            ).get(metric)
                        if all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                            for value in (full_value, control_value)
                        ):
                            if metric in PAIRED_CORRELATION_METRICS:
                                values[(fold, seed)] = float(full_value) - float(control_value)
                            else:
                                values[(fold, seed)] = float(control_value) - float(full_value)
                        else:
                            values[(fold, seed)] = None
                family_metrics[metric] = _descriptive_summary(values)
            paired[comparison][family] = family_metrics

    full_strata: dict[str, Any] = {}
    full_supported: dict[str, Any] = {}
    for family in ("ordinary", "gap"):
        summaries = {
            (fold, seed): runs[(fold, seed, "full")]["metrics"]["round4"][
                "families"
            ][family]["summary"]
            for fold in FOLDS
            for seed in SEEDS
        }
        stratum_names = sorted(
            {
                name
                for summary in summaries.values()
                for name in summary.get("strata", {})
            }
        )
        full_strata[family] = {}
        for name in stratum_names:
            full_strata[family][name] = {
                metric: _descriptive_summary(
                    {
                        identity: summary.get("strata", {})
                        .get(name, {})
                        .get("model", {})
                        .get(metric)
                        for identity, summary in summaries.items()
                    }
                )
                for metric in SCIENTIFIC_METRICS
            }
            full_strata[family][name]["coverage"] = _descriptive_summary(
                {
                    identity: summary.get("strata", {}).get(name, {}).get("coverage")
                    for identity, summary in summaries.items()
                }
            )
        supported_names = sorted(
            {
                name
                for summary in summaries.values()
                for name in summary.get("supported_genes", {})
            }
        )
        full_supported[family] = {
            name: {
                metric: _descriptive_summary(
                    {
                        identity: summary.get("supported_genes", {})
                        .get(name, {})
                        .get(metric)
                        for identity, summary in summaries.items()
                    }
                )
                for metric in SCIENTIFIC_METRICS
            }
            for name in supported_names
        }
    return {
        "methods_raw_x": methods,
        "strict_idw_raw_x": strict_idw,
        "full_clipped_zero_secondary": clipped_secondary,
        "paired_control_gains": paired,
        "full_stratified_raw_x": full_strata,
        "full_supported_genes_raw_x": full_supported,
    }
def _round_invariance(runs: dict[tuple[str, int, str], dict[str, Any]]) -> dict[str, Any]:
    comparisons = 0
    mismatches: list[dict[str, Any]] = []
    max_abs_error = 0.0
    max_relative_error = 0.0
    for fold in FOLDS:
        for seed in SEEDS:
            run = runs[(fold, seed, "full")]
            base = run["test_artifacts"] / "predictions" / "test_round4"
            files = sorted(base.glob("*/*/mask_*.npz"))
            if not files:
                raise ValueError(f"{run['run_dir']}: missing round4 raw predictions")
            for round4_path in files:
                relative = round4_path.relative_to(base)
                paths = {
                    number: run["test_artifacts"] / "predictions" / f"test_round{number}" / relative
                    for number in (1, 2, 4)
                }
                if not all(path.is_file() for path in paths.values()):
                    raise ValueError(f"{run['run_dir']}: incomplete round prediction set {relative}")
                arrays = {number: np.load(path, allow_pickle=False) for number, path in paths.items()}
                try:
                    query = arrays[4]["query_index"]
                    depth = arrays[4]["depth"]
                    if not all(
                        np.array_equal(arrays[number]["query_index"], query)
                        and np.array_equal(arrays[number]["depth"], depth)
                        for number in (1, 2)
                    ):
                        raise ValueError(f"{run['run_dir']}: round query/depth identity drifted")
                    prediction = {number: arrays[number]["prediction_x"] for number in (1, 2, 4)}
                    checks = {
                        "depth1_round1_round2": (depth == 1, 1, 2),
                        "depth1_round1_round4": (depth == 1, 1, 4),
                        "depth2_round2_round4": (depth == 2, 2, 4),
                    }
                    for label, (selector, left, right) in checks.items():
                        if not bool(selector.any()):
                            continue
                        comparisons += 1
                        left_values = prediction[left][selector]
                        right_values = prediction[right][selector]
                        absolute_error = np.abs(
                            np.asarray(left_values, dtype=np.float64)
                            - np.asarray(right_values, dtype=np.float64)
                        )
                        denominator = np.maximum(
                            np.abs(np.asarray(right_values, dtype=np.float64)),
                            ROUND_INVARIANCE_ATOL,
                        )
                        relative_error = absolute_error / denominator
                        comparison_max_abs_error = float(np.max(absolute_error))
                        comparison_max_relative_error = float(np.max(relative_error))
                        max_abs_error = max(max_abs_error, comparison_max_abs_error)
                        max_relative_error = max(max_relative_error, comparison_max_relative_error)
                        close = np.isclose(
                            left_values,
                            right_values,
                            rtol=ROUND_INVARIANCE_RTOL,
                            atol=ROUND_INVARIANCE_ATOL,
                            equal_nan=False,
                        )
                        if not bool(np.all(close)):
                            mismatches.append(
                                {
                                    "fold": fold,
                                    "seed": seed,
                                    "file": str(relative).replace("\\", "/"),
                                    "check": label,
                                    "mismatched_values": int(np.count_nonzero(~close)),
                                    "max_abs_error": comparison_max_abs_error,
                                    "max_relative_error": comparison_max_relative_error,
                                }
                            )
                finally:
                    for array in arrays.values():
                        array.close()
    return {
        "passed": not mismatches,
        "comparisons": comparisons,
        "tolerance": {
            "rtol": ROUND_INVARIANCE_RTOL,
            "atol": ROUND_INVARIANCE_ATOL,
        },
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def summarize_formal_matrix(
    runs: dict[tuple[str, int, str], dict[str, Any]]
) -> dict[str, Any]:
    gap_idw: dict[tuple[str, int], float] = {}
    gap_one: dict[tuple[str, int], float] = {}
    ordinary_idw: dict[tuple[str, int], float] = {}
    ordinary_one: dict[tuple[str, int], float] = {}
    ordinary_local: dict[tuple[str, int], float] = {}
    ordinary_global: dict[tuple[str, int], float] = {}
    gap_local: dict[tuple[str, int], float] = {}
    gap_global: dict[tuple[str, int], float] = {}
    depth2: dict[tuple[str, int], float] = {}
    depth34: dict[tuple[str, int], float] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            full = runs[(fold, seed, "full")]
            one = runs[(fold, seed, "one-shot")]
            local = runs[(fold, seed, "local-only")]
            global_only = runs[(fold, seed, "global-only")]
            identity = (fold, seed)
            full_gap = _smooth_l1(full, 4, "gap")
            gap_idw[identity] = _smooth_l1(full, 4, "gap", section="idw") - full_gap
            gap_one[identity] = _smooth_l1(one, 1, "gap") - full_gap
            gap_local[identity] = _smooth_l1(local, 4, "gap") - full_gap
            gap_global[identity] = _smooth_l1(global_only, 4, "gap") - full_gap
            full_ordinary = _smooth_l1(full, 4, "ordinary")
            idw_ordinary = _smooth_l1(full, 4, "ordinary", section="idw")
            one_ordinary = _smooth_l1(one, 1, "ordinary")
            ordinary_local[identity] = (
                _smooth_l1(local, 4, "ordinary") - full_ordinary
            )
            ordinary_global[identity] = (
                _smooth_l1(global_only, 4, "ordinary") - full_ordinary
            )
            if idw_ordinary <= 0 or one_ordinary <= 0:
                raise ValueError("ordinary relative deterioration denominator must be positive")
            ordinary_idw[identity] = (full_ordinary - idw_ordinary) / idw_ordinary
            ordinary_one[identity] = (full_ordinary - one_ordinary) / one_ordinary
            depth2[identity] = _smooth_l1(
                full, 1, DEPTH_ACCEPTANCE_FAMILY, depth="2"
            ) - _smooth_l1(full, 2, DEPTH_ACCEPTANCE_FAMILY, depth="2")
            depth34[identity] = _smooth_l1(
                full, 2, DEPTH_ACCEPTANCE_FAMILY, depth="3-4"
            ) - _smooth_l1(full, 4, DEPTH_ACCEPTANCE_FAMILY, depth="3-4")

    effects = {
        "gap_full_vs_idw_gain": _effect_summary(gap_idw),
        "gap_full_vs_one_shot_gain": _effect_summary(gap_one),
        "ordinary_full_vs_idw_relative_deterioration": _effect_summary(ordinary_idw),
        "ordinary_full_vs_one_shot_relative_deterioration": _effect_summary(ordinary_one),
        "ordinary_full_vs_local_only_gain": _effect_summary(ordinary_local),
        "ordinary_full_vs_global_only_gain": _effect_summary(ordinary_global),
        "gap_full_vs_local_only_gain": _effect_summary(gap_local),
        "gap_full_vs_global_only_gain": _effect_summary(gap_global),
        "gap_depth2_round2_vs_round1_gain": _effect_summary(depth2),
        "gap_depth3_4_round4_vs_round2_gain": _effect_summary(depth34),
    }
    invariance = _round_invariance(runs)
    scientific_report = _scientific_report(runs)
    checks = {
        "gap_vs_idw": (
            effects["gap_full_vs_idw_gain"]["overall_mean"] > 0
            and effects["gap_full_vs_idw_gain"]["positive_folds"] >= 2
        ),
        "gap_vs_one_shot": (
            effects["gap_full_vs_one_shot_gain"]["overall_mean"] > 0
            and effects["gap_full_vs_one_shot_gain"]["positive_folds"] >= 2
        ),
        "ordinary_vs_idw_within_one_percent": (
            effects["ordinary_full_vs_idw_relative_deterioration"]["overall_mean"] <= 0.01
        ),
        "ordinary_vs_one_shot_within_one_percent": (
            effects["ordinary_full_vs_one_shot_relative_deterioration"]["overall_mean"] <= 0.01
        ),
        "round_invariance": invariance["passed"],
        "depth2_round_gain": (
            effects["gap_depth2_round2_vs_round1_gain"]["overall_mean"] > 0
            and effects["gap_depth2_round2_vs_round1_gain"]["positive_folds"] >= 2
        ),
        "depth3_4_round_gain": (
            effects["gap_depth3_4_round4_vs_round2_gain"]["overall_mean"] > 0
            and effects["gap_depth3_4_round4_vs_round2_gain"]["positive_folds"] >= 2
        ),
        "final_loss_full_bptt": all(
            runs[(fold, seed, "full")]["bptt"].get("passed") is True
            for fold in FOLDS
            for seed in SEEDS
        ),
    }
    return {
        "schema": "pro-normst-formal-acceptance-v2",
        "depth_acceptance_family": DEPTH_ACCEPTANCE_FAMILY,
        "accepted": all(checks.values()),
        "checks": checks,
        "effects": effects,
        "scientific_report": scientific_report,
        "round_invariance": invariance,
    }


def _formal_acceptance_artifacts(result: dict[str, Any]) -> dict[str, bytes]:
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=("comparison", "fold", "seed", "effect"))
    writer.writeheader()
    for comparison, summary in result["effects"].items():
        for row in summary["fold_x_init"]:
            writer.writerow({"comparison": comparison, **row})
    scientific_buffer = io.StringIO(newline="")
    scientific_writer = csv.DictWriter(
        scientific_buffer,
        fieldnames=("method", "family", "metric", "fold", "seed", "value"),
    )
    scientific_writer.writeheader()
    for method, families in result["scientific_report"]["methods_raw_x"].items():
        for family, metrics in families.items():
            for metric, summary in metrics.items():
                for row in summary["fold_x_init"]:
                    scientific_writer.writerow(
                        {"method": method, "family": family, "metric": metric, **row}
                    )
    for family, metrics in result["scientific_report"]["strict_idw_raw_x"].items():
        for metric, summary in metrics.items():
            for row in summary["fold_x_init"]:
                scientific_writer.writerow(
                    {"method": "strict-idw", "family": family, "metric": metric, **row}
                )
    paired_buffer = io.StringIO(newline="")
    paired_writer = csv.DictWriter(
        paired_buffer,
        fieldnames=("comparison", "family", "metric", "fold", "seed", "value"),
    )
    paired_writer.writeheader()
    for comparison, families in result["scientific_report"][
        "paired_control_gains"
    ].items():
        for family, metrics in families.items():
            for metric, summary in metrics.items():
                for row in summary["fold_x_init"]:
                    paired_writer.writerow(
                        {
                            "comparison": comparison,
                            "family": family,
                            "metric": metric,
                            **row,
                        }
                    )
    return {
        "formal_acceptance.json": (
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "formal_run_effects.csv": csv_buffer.getvalue().encode("utf-8"),
        "formal_scientific_metrics.csv": scientific_buffer.getvalue().encode("utf-8"),
        "formal_paired_metrics.csv": paired_buffer.getvalue().encode("utf-8"),
    }


def _existing_acceptance_is_identical(artifact_dir: Path, artifacts: dict[str, bytes]) -> bool:
    if not artifact_dir.is_dir():
        return False
    return all(
        (artifact_dir / name).is_file() and (artifact_dir / name).read_bytes() == content
        for name, content in artifacts.items()
    )


def write_formal_acceptance(result: dict[str, Any], output_dir: str | Path) -> None:
    """Atomically publish the two managed artifacts without touching output-dir extras."""
    destination = Path(output_dir).resolve()
    artifacts = _formal_acceptance_artifacts(result)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"formal acceptance output is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    artifact_dir = destination / FORMAL_ARTIFACTS_DIRECTORY
    if artifact_dir.exists():
        if _existing_acceptance_is_identical(artifact_dir, artifacts):
            return
        raise ValueError(
            f"formal acceptance artifacts are incomplete or differ from this result: {artifact_dir}"
        )

    staging = destination / f".{FORMAL_ARTIFACTS_DIRECTORY}.staging-{uuid4().hex}"
    try:
        staging.mkdir()
        for name, content in artifacts.items():
            path = staging / name
            with path.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, artifact_dir)
    except Exception:
        if staging.exists():
            for path in staging.iterdir():
                if path.is_file():
                    path.unlink()
            staging.rmdir()
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = summarize_formal_matrix(load_formal_matrix(args.matrix))
    write_formal_acceptance(result, args.output_dir)
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEPTH_ACCEPTANCE_FAMILY",
    "FOLDS",
    "FORMAL_ARTIFACTS_DIRECTORY",
    "SEEDS",
    "VARIANTS",
    "load_formal_matrix",
    "summarize_formal_matrix",
    "write_formal_acceptance",
]
