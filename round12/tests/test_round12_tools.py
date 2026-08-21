"""Tests for Round12 promotion and runtime reporting tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from round12.compare_runtime import _comparison, _summary
from round12.evaluate_pilot import main as evaluate_main


def _validation_record(
    criterion: float,
    *,
    gene_pearson: float,
    spot_pearson: float,
    rmse: float,
    mae: float,
) -> dict:
    family = {
        "summary": {
            "model": {
                "gene_pearson": gene_pearson,
                "spot_pearson": spot_pearson,
                "rmse": rmse,
                "mae": mae,
                "variance_ratio_median": 0.07,
            }
        }
    }
    return {
        "epoch": 1,
        "validation": {
            "criterion_weighted_z_smooth_l1": criterion,
            "families": {"ordinary": family, "gap": family},
        },
    }


class Round12ToolTest(unittest.TestCase):
    def test_runtime_comparison_reports_speedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_log = root / "baseline.log"
            candidate_log = root / "candidate.log"
            baseline_log.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "runtime_setup", "seconds": 10.0}),
                        json.dumps(
                            {
                                "event": "runtime_epoch",
                                "epoch": 1,
                                "train_seconds": 40.0,
                                "validation_seconds": 40.0,
                                "total_seconds": 81.0,
                            }
                        ),
                        json.dumps(
                            {
                                "event": "runtime_epoch",
                                "epoch": 2,
                                "train_seconds": 38.0,
                                "validation_seconds": 38.0,
                                "total_seconds": 77.0,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            candidate_log.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "runtime_setup", "seconds": 9.0}),
                        json.dumps(
                            {
                                "event": "runtime_epoch",
                                "epoch": 1,
                                "train_seconds": 41.0,
                                "validation_seconds": 20.0,
                                "total_seconds": 62.0,
                            }
                        ),
                        json.dumps(
                            {
                                "event": "runtime_epoch",
                                "epoch": 2,
                                "train_seconds": 39.0,
                                "validation_seconds": 19.0,
                                "total_seconds": 59.0,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            baseline = _summary(baseline_log)
            candidate = _summary(candidate_log)
            comparison = _comparison(baseline, candidate)
        self.assertGreater(
            comparison["warm_epoch_mean"]["total_seconds"]["speedup"],
            1.3,
        )

    def test_validation_only_evaluator_requires_residual_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            baseline_record = _validation_record(
                0.2232179318089038,
                gene_pearson=0.25,
                spot_pearson=0.45,
                rmse=1.58,
                mae=1.38,
            )
            candidate_record = _validation_record(
                0.2224,
                gene_pearson=0.251,
                spot_pearson=0.451,
                rmse=1.579,
                mae=1.379,
            )
            for run_dir, record in (
                (baseline, baseline_record),
                (candidate, candidate_record),
            ):
                torch.save(
                    {
                        "best_epoch": 1,
                        "best_value": record["validation"][
                            "criterion_weighted_z_smooth_l1"
                        ],
                    },
                    run_dir / "best.pt",
                )
                (run_dir / "history.json").write_text(
                    json.dumps([record]),
                    encoding="utf-8",
                )
            for name in ("pilot_gate.json", "gradient_gate.json"):
                (candidate / name).write_text(
                    json.dumps({"passed": True}),
                    encoding="utf-8",
                )
            (candidate / "final_loss_bptt_gate.json").write_text(
                json.dumps({"passed": True}),
                encoding="utf-8",
            )
            (candidate / "best_diagnostics.json").write_text(
                json.dumps(
                    {
                        "local_gene_residual_rms": 0.01,
                        "local_gene_residual_finite": True,
                        "local_gene_residual_shared_prediction_rms_ratio": 0.02,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "evaluation.json"
            with mock.patch(
                "sys.argv",
                [
                    "evaluate_pilot",
                    "--baseline-dir",
                    str(baseline),
                    "--candidate-dir",
                    str(candidate),
                    "--output",
                    str(output),
                ],
            ):
                self.assertEqual(evaluate_main(), 0)
            evaluation = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(evaluation["selected_for_formal_matrix"])
        self.assertEqual(evaluation["selection_basis"], "validation_only")
        self.assertFalse(evaluation["test_metrics_used"])


if __name__ == "__main__":
    unittest.main()
