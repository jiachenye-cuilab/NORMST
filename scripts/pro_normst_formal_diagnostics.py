"""Aggregate convergence, runtime, health, and mechanism diagnostics for a formal matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SUMMARY_METRICS = (
    "smooth_l1",
    "mae",
    "rmse",
    "gene_pearson",
    "spot_pearson",
    "variance_ratio_median",
)

MECHANISM_METRICS = (
    "gate_mean",
    "coverage_mean",
    "confidence_mean",
    "activation_round_mean",
    "gated_local_global_norm_ratio",
    "global_input_residual_ratio",
    "global_local_normalized_linear_cka",
    "local_state_enhancer_residual_ratio",
    "local_state_effective_rank",
    "local_state_enhanced_effective_rank",
    "routing_entropy",
    "routing_channel_max_probability_mean",
    "local_direction_entropy",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "sample_sd": None, "median": None, "min": None, "max": None}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "sample_sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _launcher_and_runtime(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    launchers = []
    for path in Path(f"{run_dir}.control").glob("train.*/launcher.json"):
        value = _read_json(path)
        if (
            value.get("status") == "complete"
            and value.get("exit_code") == 0
            and Path(str(value.get("output_dir", ""))).resolve() == run_dir
        ):
            launchers.append((path, value))
    if len(launchers) != 1:
        raise ValueError(f"expected one complete launcher for {run_dir}, found {len(launchers)}")
    launcher_path, launcher = launchers[0]
    log_path = Path(str(launcher["log"]))
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    events = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and str(value.get("event", "")).startswith("runtime_"):
            events.append(value)
    setup = next((value for value in events if value.get("event") == "runtime_setup"), None)
    test = next((value for value in events if value.get("event") == "runtime_test"), None)
    epochs = [value for value in events if value.get("event") == "runtime_epoch"]
    if setup is None or test is None or not epochs:
        raise ValueError(f"runtime events are incomplete for {run_dir}")
    peak_allocated = max(
        int(value.get("cuda_peak_memory", {}).get("allocated_bytes", 0))
        for value in epochs
    )
    peak_reserved = max(
        int(value.get("cuda_peak_memory", {}).get("reserved_bytes", 0))
        for value in epochs
    )
    runtime = {
        "setup_seconds": float(setup["seconds"]),
        "epoch_total_seconds": sum(float(value["total_seconds"]) for value in epochs),
        "train_total_seconds": sum(float(value["train_seconds"]) for value in epochs),
        "validation_total_seconds": sum(float(value["validation_seconds"]) for value in epochs),
        "test_seconds": float(test["seconds"]),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }
    runtime["recorded_total_seconds"] = (
        runtime["setup_seconds"] + runtime["epoch_total_seconds"] + runtime["test_seconds"]
    )
    return {
        "path": str(launcher_path.resolve()),
        "physical_gpu": int(launcher["physical_gpu"]),
        "preflight_exit_code": int(launcher["preflight_exit_code"]),
        "exit_code": int(launcher["exit_code"]),
        "log": str(log_path.resolve()),
    }, runtime


def _test_family(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        source: {
            metric: value.get(metric)
            for metric in SUMMARY_METRICS
        }
        for source, value in (
            ("model", summary["model"]),
            ("idw", summary["idw"]),
        )
    }


def _load_run(entry: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(entry["run_dir"]).resolve()
    config = _read_json(run_dir / "config.json")
    status = _read_json(run_dir / "run_status.json")
    history = _read_json(run_dir / "history.json")
    gradient = _read_json(run_dir / "gradient_gate.json")
    bptt = _read_json(run_dir / "final_loss_bptt_gate.json")
    checkpoint_lock = _read_json(run_dir / "run_checkpoint_lock.json")
    test_complete = _read_json(run_dir / "test_artifacts" / "test_complete.json")
    test_metrics = _read_json(run_dir / "test_artifacts" / "test_metrics.json")
    if status.get("status") != "complete" or not status.get("test_run"):
        raise ValueError(f"run is incomplete: {run_dir}")
    if not gradient.get("passed") or not bptt.get("passed"):
        raise ValueError(f"gradient/BPTT gate failed: {run_dir}")
    if checkpoint_lock.get("status") != "locked" or test_complete.get("status") != "complete":
        raise ValueError(f"checkpoint/test commit is incomplete: {run_dir}")
    variant = str(entry["variant"])
    primary_round = "round1" if variant == "one-shot" else "round4"
    primary = test_metrics[primary_round]
    epochs = len(history)
    best_epoch = int(status["best_epoch"])
    best_record = history[best_epoch - 1]
    launcher, runtime = _launcher_and_runtime(run_dir)
    diagnostics_path = run_dir / "best_diagnostics.json"
    diagnostics = _read_json(diagnostics_path) if diagnostics_path.is_file() else {}
    return {
        "fold": str(entry["fold"]),
        "seed": int(entry["seed"]),
        "variant": variant,
        "run_dir": str(run_dir),
        "contract_hash": config["contract_hash"],
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "epochs_after_best": epochs - best_epoch,
        "early_stopped": epochs < 50,
        "best_validation": float(status["best_validation"]),
        "final_validation": float(history[-1]["validation"]["criterion_weighted_z_smooth_l1"]),
        "final_minus_best_validation": float(
            history[-1]["validation"]["criterion_weighted_z_smooth_l1"]
            - status["best_validation"]
        ),
        "best_train_combined_loss": float(best_record["train_loss"]["combined"]),
        "best_validation_minus_train": float(
            status["best_validation"] - best_record["train_loss"]["combined"]
        ),
        "gradient_norm_preclip_mean_at_best": float(best_record["gradient_norm_preclip_mean"]),
        "gradient_norm_preclip_max_at_best": float(best_record["gradient_norm_preclip_max"]),
        "test_primary_round": primary_round,
        "test_criterion_weighted_z_smooth_l1": float(primary["criterion_weighted_z_smooth_l1"]),
        "test": {
            family: _test_family(primary["families"][family]["summary"])
            for family in ("ordinary", "gap")
        },
        "health": {
            "gradient_gate": bool(gradient["passed"]),
            "bptt_gate": bool(bptt["passed"]),
            "checkpoint_lock": checkpoint_lock["status"],
            "test_commit": test_complete["status"],
        },
        "launcher": launcher,
        "runtime": runtime,
        "best_diagnostics": {
            key: diagnostics.get(key)
            for key in MECHANISM_METRICS
            if isinstance(diagnostics.get(key), (int, float))
        },
    }


def _variant_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = [record["runtime"] for record in records]
    result = {
        "runs": len(records),
        "epochs_completed": _numeric_summary(record["epochs_completed"] for record in records),
        "best_epoch": _numeric_summary(record["best_epoch"] for record in records),
        "best_validation": _numeric_summary(record["best_validation"] for record in records),
        "test_criterion_weighted_z_smooth_l1": _numeric_summary(
            record["test_criterion_weighted_z_smooth_l1"] for record in records
        ),
        "best_validation_minus_train": _numeric_summary(
            record["best_validation_minus_train"] for record in records
        ),
        "early_stopped_runs": sum(bool(record["early_stopped"]) for record in records),
        "runtime": {
            "total_gpu_hours": sum(value["recorded_total_seconds"] for value in runtime) / 3600.0,
            "mean_run_hours": statistics.fmean(value["recorded_total_seconds"] for value in runtime) / 3600.0,
            "mean_train_seconds_per_epoch": statistics.fmean(
                value["train_total_seconds"] / record["epochs_completed"]
                for value, record in zip(runtime, records, strict=True)
            ),
            "mean_validation_seconds_per_epoch": statistics.fmean(
                value["validation_total_seconds"] / record["epochs_completed"]
                for value, record in zip(runtime, records, strict=True)
            ),
            "mean_test_seconds": statistics.fmean(value["test_seconds"] for value in runtime),
            "max_peak_allocated_gib": max(value["peak_allocated_bytes"] for value in runtime) / 2**30,
            "max_peak_reserved_gib": max(value["peak_reserved_bytes"] for value in runtime) / 2**30,
        },
        "test": {},
    }
    for family in ("ordinary", "gap"):
        result["test"][family] = {
            metric: _numeric_summary(
                record["test"][family]["model"][metric]
                for record in records
                if isinstance(record["test"][family]["model"].get(metric), (int, float))
            )
            for metric in SUMMARY_METRICS
        }
    return result


def _full_fold_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for fold in ("lodo_d1", "lodo_d2", "lodo_d3"):
        selected = [record for record in records if record["fold"] == fold and record["variant"] == "full"]
        result[fold] = {
            family: {
                metric: statistics.fmean(
                    record["test"][family]["model"][metric]
                    for record in selected
                )
                for metric in SUMMARY_METRICS
            }
            for family in ("ordinary", "gap")
        }
    return result


def _mechanism_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [record for record in records if record["variant"] == "full"]
    return {
        metric: _numeric_summary(
            record["best_diagnostics"][metric]
            for record in selected
            if metric in record["best_diagnostics"]
        )
        for metric in MECHANISM_METRICS
    }


def _compact_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": bool(payload["accepted"]),
        "checks": payload["checks"],
        "round_invariance": {
            key: payload["round_invariance"].get(key)
            for key in ("passed", "mismatch_count", "max_abs_error", "max_relative_error")
        },
        "effects": {
            name: {
                key: value.get(key)
                for key in ("overall_mean", "fold_sample_sd", "fold_range", "positive_folds", "fold_effect")
            }
            for name, value in payload["effects"].items()
        },
        "method_metrics": {
            method: {
                family: {
                    metric: values[metric].get("overall_mean")
                    for metric in SUMMARY_METRICS
                }
                for family, values in families.items()
            }
            for method, families in payload["scientific_report"]["methods_raw_x"].items()
        },
        "strict_idw_metrics": {
            family: {
                metric: values[metric].get("overall_mean")
                for metric in SUMMARY_METRICS
            }
            for family, values in payload["scientific_report"]["strict_idw_raw_x"].items()
        },
    }


def _conditioning_summary(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current_model = current["rounds"]["round8"]
    previous_model = previous["rounds"]["round8"]
    fields = (
        "weighted_z_smooth_l1",
        "raw_x_smooth_l1",
        "global_local_normalized_linear_cka",
        "local_conditional_error_gain",
        "global_conditional_error_gain",
    )
    execution_delta = {
        family: {
            key: current_model["families"][family][key] - previous_model["families"][family][key]
            for key in fields
        }
        for family in ("ordinary", "gap")
    }
    return {
        "current_replay": current_model,
        "current_minus_round7": current["round8_minus_round7"],
        "current_minus_previous_round8_execution": execution_delta,
        "limitations": current["limitations"],
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "fold", "seed", "variant", "epochs_completed", "best_epoch", "early_stopped",
        "best_validation", "final_validation", "best_train_combined_loss",
        "best_validation_minus_train", "test_primary_round", "test_criterion_weighted_z_smooth_l1",
        "ordinary_smooth_l1", "ordinary_gene_pearson", "ordinary_spot_pearson",
        "gap_smooth_l1", "gap_gene_pearson", "gap_spot_pearson",
        "runtime_hours", "train_seconds_per_epoch", "validation_seconds_per_epoch",
        "test_seconds", "peak_allocated_gib", "physical_gpu", "run_dir",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            runtime = record["runtime"]
            writer.writerow({
                "fold": record["fold"],
                "seed": record["seed"],
                "variant": record["variant"],
                "epochs_completed": record["epochs_completed"],
                "best_epoch": record["best_epoch"],
                "early_stopped": record["early_stopped"],
                "best_validation": record["best_validation"],
                "final_validation": record["final_validation"],
                "best_train_combined_loss": record["best_train_combined_loss"],
                "best_validation_minus_train": record["best_validation_minus_train"],
                "test_primary_round": record["test_primary_round"],
                "test_criterion_weighted_z_smooth_l1": record["test_criterion_weighted_z_smooth_l1"],
                "ordinary_smooth_l1": record["test"]["ordinary"]["model"]["smooth_l1"],
                "ordinary_gene_pearson": record["test"]["ordinary"]["model"]["gene_pearson"],
                "ordinary_spot_pearson": record["test"]["ordinary"]["model"]["spot_pearson"],
                "gap_smooth_l1": record["test"]["gap"]["model"]["smooth_l1"],
                "gap_gene_pearson": record["test"]["gap"]["model"]["gene_pearson"],
                "gap_spot_pearson": record["test"]["gap"]["model"]["spot_pearson"],
                "runtime_hours": runtime["recorded_total_seconds"] / 3600.0,
                "train_seconds_per_epoch": runtime["train_total_seconds"] / record["epochs_completed"],
                "validation_seconds_per_epoch": runtime["validation_total_seconds"] / record["epochs_completed"],
                "test_seconds": runtime["test_seconds"],
                "peak_allocated_gib": runtime["peak_allocated_bytes"] / 2**30,
                "physical_gpu": record["launcher"]["physical_gpu"],
                "run_dir": record["run_dir"],
            })


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--round1-comparison", type=Path, required=True)
    parser.add_argument("--queue-result", type=Path, required=True)
    parser.add_argument("--conditioning-audit", type=Path, required=True)
    parser.add_argument("--previous-conditioning-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        matrix = _read_json(args.matrix.resolve())
        records = [_load_run(entry) for entry in matrix["runs"]]
        if len(records) != 36:
            raise ValueError(f"expected 36 formal runs, found {len(records)}")
        acceptance = _read_json(args.acceptance.resolve())
        comparison = _read_json(args.round1_comparison.resolve())
        queue_result = _read_json(args.queue_result.resolve())
        conditioning = _read_json(args.conditioning_audit.resolve())
        previous_conditioning = _read_json(args.previous_conditioning_audit.resolve())
        by_variant = {
            variant: _variant_summary([record for record in records if record["variant"] == variant])
            for variant in ("full", "one-shot", "local-only", "global-only")
        }
        queued_at = _parse_datetime(queue_result["queued_at"])
        finished_at = _parse_datetime(queue_result["finished_at"])
        formal_gpu_seconds = sum(record["runtime"]["recorded_total_seconds"] for record in records)
        report = {
            "schema": "pro-normst-formal-diagnostics-v1",
            "selection_use": False,
            "round_identity": matrix["round_identity"],
            "scope": "post-hoc analysis of the locked 36-run formal matrix",
            "health": {
                "runs": len(records),
                "complete_runs": sum(record["health"]["test_commit"] == "complete" for record in records),
                "gradient_gates_passed": sum(record["health"]["gradient_gate"] for record in records),
                "bptt_gates_passed": sum(record["health"]["bptt_gate"] for record in records),
                "checkpoint_locks": sum(record["health"]["checkpoint_lock"] == "locked" for record in records),
                "formal_accepted": bool(acceptance["accepted"]),
            },
            "convergence_and_runtime": {
                "by_variant": by_variant,
                "formal_total_gpu_hours": formal_gpu_seconds / 3600.0,
                "detached_pipeline_wall_hours_including_pilot": (finished_at - queued_at).total_seconds() / 3600.0,
                "four_gpu_parallel_utilization_proxy": formal_gpu_seconds / (
                    4.0 * (finished_at - queued_at).total_seconds()
                ),
            },
            "full_test_metrics_by_fold": _full_fold_summary(records),
            "full_best_checkpoint_mechanism": _mechanism_summary(records),
            "formal_acceptance": _compact_acceptance(acceptance),
            "round1_comparison": comparison,
            "conditioning_audit": _conditioning_summary(conditioning, previous_conditioning),
            "runs": records,
            "limitations": [
                "Test artifacts are analyzed only after every validation-selected checkpoint was locked; no result is used for model selection.",
                "The nine fold-by-seed runs are not nine independent donors; held-out donor folds are the biological generalization units.",
                "Best-diagnostics use one fixed validation gap mask per run and are mechanistic summaries, not population estimates.",
                "Branch-removal conditioning diagnostics reuse the full decoder and are descriptive; independently trained variants provide the formal causal controls.",
            ],
        }
        (staging / "formal_diagnostics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(staging / "formal_run_summary.csv", records)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output_dir": str(output_dir), "runs": 36, "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
