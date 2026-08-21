#!/usr/bin/env python
"""Apply the frozen Round11 validation-only promotion gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from round10.evaluate_pilot import (
    FORMAL_CRITERION_TARGET,
    MAX_ERROR_RELATIVE_DETERIORATION,
    MAX_PEARSON_DROP,
    MEAN_GENE_PEARSON_GAIN_TARGET,
    METRICS,
    _atomic_json,
    _locked_validation,
    _read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round9-dir", type=Path, required=True)
    parser.add_argument("--round10-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _deltas(candidate: dict, baseline: dict) -> dict:
    return {
        family: {
            key: float(candidate["families"][family][key])
            - float(baseline["families"][family][key])
            for key in METRICS
        }
        for family in ("ordinary", "gap")
    }


def main() -> int:
    args = parse_args()
    round9 = _locked_validation(args.round9_dir.resolve())
    round10 = _locked_validation(args.round10_dir.resolve())
    candidate = _locked_validation(args.candidate_dir.resolve())
    versus_round9 = _deltas(candidate, round9)
    versus_round10 = _deltas(candidate, round10)
    error_relative_deterioration = {
        family: {
            key: versus_round9[family][key] / float(round9["families"][family][key])
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
    gene_gains = [versus_round9[family]["gene_pearson"] for family in ("ordinary", "gap")]
    gates = {
        "health": all(health.values()),
        "formal_criterion": candidate["criterion"] <= FORMAL_CRITERION_TARGET,
        "gene_pearson_each_positive": all(value > 0 for value in gene_gains),
        "gene_pearson_mean_gain": sum(gene_gains) / len(gene_gains)
        >= MEAN_GENE_PEARSON_GAIN_TARGET,
        "pearson_guardrail": all(
            versus_round9[family][key] >= -MAX_PEARSON_DROP
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
    if not all(
        math.isfinite(float(value))
        for comparison in (versus_round9, versus_round10)
        for family in comparison.values()
        for value in family.values()
    ):
        raise FloatingPointError("pilot comparison contains non-finite metrics")
    payload = {
        "schema": "pro-normst-round11-selection-v1",
        "round_identity": "pro-v2-round-011",
        "selection_basis": "validation_only",
        "test_metrics_used": False,
        "round9_baseline": round9,
        "round10_fixed_weight": round10,
        "candidate": candidate,
        "candidate_minus_round9": versus_round9,
        "candidate_minus_round10": versus_round10,
        "error_relative_deterioration_vs_round9": error_relative_deterioration,
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
