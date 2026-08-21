#!/usr/bin/env python
"""Apply the predeclared Round10 validation-only promotion gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


FORMAL_CRITERION_TARGET = 0.22244
MEAN_GENE_PEARSON_GAIN_TARGET = 0.001
MAX_PEARSON_DROP = 0.001
MAX_ERROR_RELATIVE_DETERIORATION = 0.0025
METRICS = ("rmse", "mae", "gene_pearson", "spot_pearson", "variance_ratio_median")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {path}")
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


def _locked_validation(run_dir: Path) -> dict[str, Any]:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    best_epoch = int(checkpoint["best_epoch"])
    matches = [
        record
        for record in _read_json(run_dir / "history.json")
        if int(record["epoch"]) == best_epoch
    ]
    if len(matches) != 1:
        raise ValueError(f"locked epoch absent or duplicated: {run_dir}")
    record = matches[0]
    criterion = float(record["validation"]["criterion_weighted_z_smooth_l1"])
    if abs(criterion - float(checkpoint["best_value"])) > 1e-12:
        raise ValueError(f"locked criterion mismatch: {run_dir}")
    families = {
        family: {
            key: record["validation"]["families"][family]["summary"]["model"].get(key)
            for key in METRICS
        }
        for family in ("ordinary", "gap")
    }
    return {"best_epoch": best_epoch, "criterion": criterion, "families": families}


def main() -> int:
    args = parse_args()
    baseline = _locked_validation(args.baseline_dir.resolve())
    candidate = _locked_validation(args.candidate_dir.resolve())
    deltas = {
        family: {
            key: float(candidate["families"][family][key])
            - float(baseline["families"][family][key])
            for key in METRICS
        }
        for family in ("ordinary", "gap")
    }
    error_relative_deterioration = {
        family: {
            key: deltas[family][key] / float(baseline["families"][family][key])
            for key in ("rmse", "mae")
        }
        for family in ("ordinary", "gap")
    }
    health = {
        "pilot": bool(_read_json(args.candidate_dir / "pilot_gate.json")["passed"]),
        "gradient": bool(_read_json(args.candidate_dir / "gradient_gate.json")["passed"]),
        "final_loss_bptt": bool(
            _read_json(args.candidate_dir / "final_loss_bptt_gate.json")["passed"]
        ),
    }
    gene_gains = [deltas[family]["gene_pearson"] for family in ("ordinary", "gap")]
    gates = {
        "health": all(health.values()),
        "formal_criterion": candidate["criterion"] <= FORMAL_CRITERION_TARGET,
        "gene_pearson_each_positive": all(value > 0 for value in gene_gains),
        "gene_pearson_mean_gain": sum(gene_gains) / len(gene_gains)
        >= MEAN_GENE_PEARSON_GAIN_TARGET,
        "pearson_guardrail": all(
            deltas[family][key] >= -MAX_PEARSON_DROP
            for family in ("ordinary", "gap")
            for key in ("gene_pearson", "spot_pearson")
        ),
        "raw_error_guardrail": all(
            value <= MAX_ERROR_RELATIVE_DETERIORATION
            for family in error_relative_deterioration.values()
            for value in family.values()
        ),
        "variance_health": all(
            1e-3 <= float(candidate["families"][family]["variance_ratio_median"]) <= 10
            for family in ("ordinary", "gap")
        ),
    }
    if not all(math.isfinite(float(value)) for family in deltas.values() for value in family.values()):
        raise FloatingPointError("pilot comparison contains non-finite metrics")
    payload = {
        "schema": "pro-normst-round10-selection-v1",
        "round_identity": "pro-v2-round-010",
        "selection_basis": "validation_only",
        "test_metrics_used": False,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": deltas,
        "error_relative_deterioration": error_relative_deterioration,
        "health": health,
        "thresholds": {
            "formal_criterion_target": FORMAL_CRITERION_TARGET,
            "mean_gene_pearson_gain_target": MEAN_GENE_PEARSON_GAIN_TARGET,
            "maximum_pearson_drop": MAX_PEARSON_DROP,
            "maximum_error_relative_deterioration": MAX_ERROR_RELATIVE_DETERIORATION,
        },
        "gates": gates,
        "selected_for_formal_matrix": all(gates.values()),
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "selected": all(gates.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
