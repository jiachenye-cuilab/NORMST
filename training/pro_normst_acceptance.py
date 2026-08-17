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


def load_formal_matrix(path: str | Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    matrix_path = Path(path).resolve()
    payload = _read_json(matrix_path)
    if payload.get("schema") != "pro-normst-formal-matrix-v1":
        raise ValueError("formal matrix schema is incompatible")
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

    for fold in FOLDS:
        fold_runs = [run for identity, run in runs.items() if identity[0] == fold]
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
        if (
            len(mask_contracts) != 1
            or len(test_identities) != 1
            or len(idw_summaries) != 1
            or len(idw_prediction_hashes) != 1
        ):
            raise ValueError(f"{fold}: matched masks, test identity, or IDW baseline drifted")
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
    depth2: dict[tuple[str, int], float] = {}
    depth34: dict[tuple[str, int], float] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            full = runs[(fold, seed, "full")]
            one = runs[(fold, seed, "one-shot")]
            identity = (fold, seed)
            full_gap = _smooth_l1(full, 4, "gap")
            gap_idw[identity] = _smooth_l1(full, 4, "gap", section="idw") - full_gap
            gap_one[identity] = _smooth_l1(one, 1, "gap") - full_gap
            full_ordinary = _smooth_l1(full, 4, "ordinary")
            idw_ordinary = _smooth_l1(full, 4, "ordinary", section="idw")
            one_ordinary = _smooth_l1(one, 1, "ordinary")
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
        "gap_depth2_round2_vs_round1_gain": _effect_summary(depth2),
        "gap_depth3_4_round4_vs_round2_gain": _effect_summary(depth34),
    }
    invariance = _round_invariance(runs)
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
        "schema": "pro-normst-formal-acceptance-v1",
        "depth_acceptance_family": DEPTH_ACCEPTANCE_FAMILY,
        "accepted": all(checks.values()),
        "checks": checks,
        "effects": effects,
        "round_invariance": invariance,
    }


def _formal_acceptance_artifacts(result: dict[str, Any]) -> dict[str, bytes]:
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=("comparison", "fold", "seed", "effect"))
    writer.writeheader()
    for comparison, summary in result["effects"].items():
        for row in summary["fold_x_init"]:
            writer.writerow({"comparison": comparison, **row})
    return {
        "formal_acceptance.json": (
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "formal_run_effects.csv": csv_buffer.getvalue().encode("utf-8"),
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
