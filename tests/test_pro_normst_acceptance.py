"""Synthetic aggregation tests for the formal ProNORMST acceptance gate."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from training.pro_normst_acceptance import (
    FOLDS,
    FORMAL_ARTIFACTS_DIRECTORY,
    ROUND_INVARIANCE_ATOL,
    SEEDS,
    VARIANTS,
    _prediction_tree_sha256,
    _round_invariance,
    load_formal_matrix,
    summarize_formal_matrix,
    write_formal_acceptance,
)


def _family_summary(model: float, idw: float, depth2: float, depth34: float):
    return {
        "model": {"smooth_l1": model},
        "idw": {"smooth_l1": idw},
        "strata": {
            "depth:2": {"model": {"smooth_l1": depth2}},
            "depth:3-4": {"model": {"smooth_l1": depth34}},
        },
    }


def _metrics(variant: str):
    if variant == "full":
        return {
            "round1": {
                "families": {
                    "ordinary": {"summary": _family_summary(1.005, 1.0, 1.2, 1.3)},
                    "gap": {"summary": _family_summary(1.1, 1.1, 1.2, 1.3)},
                }
            },
            "round2": {
                "families": {
                    "ordinary": {"summary": _family_summary(1.005, 1.0, 1.0, 1.2)},
                    "gap": {"summary": _family_summary(0.9, 1.1, 1.0, 1.2)},
                }
            },
            "round4": {
                "families": {
                    "ordinary": {"summary": _family_summary(1.005, 1.0, 1.0, 1.0)},
                    "gap": {"summary": _family_summary(0.8, 1.1, 1.0, 1.0)},
                }
            },
        }
    primary_round = "round1" if variant == "one-shot" else "round4"
    branch_model = {
        "one-shot": (1.0, 1.0),
        "local-only": (1.2, 1.1),
        "global-only": (1.3, 1.2),
    }
    ordinary_model, gap_model = branch_model[variant]
    return {
        primary_round: {
            "families": {
                "ordinary": {
                    "summary": _family_summary(ordinary_model, 1.0, 1.0, 1.0)
                },
                "gap": {"summary": _family_summary(gap_model, 1.1, 1.0, 1.0)},
            }
        }
    }


def _runs():
    return {
        (fold, seed, variant): {
            "run_dir": Path(f"/{fold}/{seed}/{variant}"),
            "metrics": _metrics(variant),
            "bptt": {"passed": True},
        }
        for fold in FOLDS
        for seed in SEEDS
        for variant in VARIANTS
    }


def _formal_contract(fold: str):
    return {
        "run": {"candidate_lock_sha256": "candidate-lock"},
        "preprocessing": {
            "schema": "pro-normst-preprocessing-v2",
            "fold": fold,
            "gene_scale_sha256": f"scale-{fold}",
            "detection_rate_sha256": f"detection-{fold}",
            "positive_weight_sha256": f"weight-{fold}",
        },
        "split": {
            "protocol": "pair_grouped_lodo",
            "fold": fold,
            "role_sizes": {"train": 4, "val": 4, "test": 4},
        },
        "slice_data_and_geometry": {
            f"{fold}-train": {
                "role": "train",
                "expression_x_sha256": f"train-expression-{fold}",
                "neighbor_index_sha256": f"train-geometry-{fold}",
            },
            f"{fold}-val": {
                "role": "val",
                "expression_x_sha256": f"val-expression-{fold}",
                "neighbor_index_sha256": f"val-geometry-{fold}",
            },
            f"{fold}-test": {
                "role": "test",
                "expression_x_sha256": f"test-expression-{fold}",
                "neighbor_index_sha256": f"test-geometry-{fold}",
            },
        },
        "fixed_mask_banks": {"fold": fold, "bank": "matched"},
    }


def _loaded_runs():
    runs = _runs()
    for (fold, _seed, variant), run in runs.items():
        run["config"] = {"variant": variant}
        run["contract"] = copy.deepcopy(_formal_contract(fold))
        run["metrics"]["_meta"] = {
            "test_expression_x_sha256": {f"{fold}-test": f"test-expression-{fold}"}
        }
    return runs


def _formal_matrix_payload():
    return {
        "schema": "pro-normst-formal-matrix-v1",
        "runs": [
            {
                "fold": fold,
                "seed": seed,
                "variant": variant,
                "run_dir": f"runs/{fold}/{seed}/{variant}",
            }
            for fold in FOLDS
            for seed in SEEDS
            for variant in VARIANTS
        ],
    }


def _runs_with_round_predictions(root: Path) -> dict[tuple[str, int, str], dict[str, object]]:
    runs = _runs()
    for fold in FOLDS:
        for seed in SEEDS:
            run_dir = root / fold / str(seed) / "full"
            runs[(fold, seed, "full")]["run_dir"] = run_dir
            runs[(fold, seed, "full")]["test_artifacts"] = run_dir / "test_artifacts"
            for round_number in (1, 2, 4):
                path = (
                    runs[(fold, seed, "full")]["test_artifacts"]
                    / "predictions"
                    / f"test_round{round_number}"
                    / "gap"
                    / "slice"
                    / "mask_0.npz"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                prediction = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
                if round_number in (2, 4):
                    prediction[0, 0] += ROUND_INVARIANCE_ATOL / 2
                np.savez(
                    path,
                    query_index=np.asarray([10, 20, 30], dtype=np.int64),
                    depth=np.asarray([1, 2, 3], dtype=np.int64),
                    prediction_x=prediction,
                )
    return runs


class FormalAcceptanceTest(unittest.TestCase):
    @mock.patch(
        "training.pro_normst_acceptance._round_invariance",
        return_value={"passed": True, "comparisons": 1, "mismatches": []},
    )
    def test_complete_positive_matrix_is_accepted(self, _invariance):
        result = summarize_formal_matrix(_runs())
        self.assertTrue(result["accepted"])
        self.assertEqual(result["schema"], "pro-normst-formal-acceptance-v2")
        self.assertEqual(result["depth_acceptance_family"], "gap")
        self.assertEqual(
            result["effects"]["gap_full_vs_idw_gain"]["positive_folds"], 3
        )
        self.assertAlmostEqual(
            result["effects"][
                "ordinary_full_vs_idw_relative_deterioration"
            ]["overall_mean"],
            0.005,
        )

    @mock.patch(
        "training.pro_normst_acceptance._round_invariance",
        return_value={"passed": True, "comparisons": 1, "mismatches": []},
    )
    def test_branch_ablation_effects_are_reported_without_new_acceptance_gates(
        self, _invariance
    ):
        result = summarize_formal_matrix(_runs())
        self.assertAlmostEqual(
            result["effects"]["ordinary_full_vs_local_only_gain"]["overall_mean"],
            0.195,
        )
        self.assertAlmostEqual(
            result["effects"]["ordinary_full_vs_global_only_gain"]["overall_mean"],
            0.295,
        )
        self.assertAlmostEqual(
            result["effects"]["gap_full_vs_local_only_gain"]["overall_mean"],
            0.3,
        )
        self.assertAlmostEqual(
            result["effects"]["gap_full_vs_global_only_gain"]["overall_mean"],
            0.4,
        )
        self.assertFalse(any("local" in key or "global" in key for key in result["checks"]))

    def test_formal_matrix_rejects_matched_training_contract_drift(self):
        mutations = {
            "preprocessing": lambda run: run["contract"]["preprocessing"].__setitem__(
                "gene_scale_sha256", "drifted"
            ),
            "split": lambda run: run["contract"]["split"]["role_sizes"].__setitem__(
                "train", 3
            ),
            "train/val slice": lambda run: run["contract"][
                "slice_data_and_geometry"
            ]["lodo_d1-val"].__setitem__("neighbor_index_sha256", "drifted"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            matrix_path = Path(temporary) / "formal_matrix.json"
            matrix_path.write_text(json.dumps(_formal_matrix_payload()), encoding="utf-8")
            for expected_error, mutate in mutations.items():
                with self.subTest(expected_error=expected_error):
                    runs = _loaded_runs()
                    mutate(runs[("lodo_d1", 2028, "local-only")])
                    with mock.patch(
                        "training.pro_normst_acceptance._load_run",
                        side_effect=lambda _path, identity: runs[identity],
                    ), mock.patch(
                        "training.pro_normst_acceptance._idw_prediction_tree_hash",
                        return_value="matched-idw",
                    ):
                        with self.assertRaisesRegex(ValueError, expected_error):
                            load_formal_matrix(matrix_path)

    @mock.patch(
        "training.pro_normst_acceptance._round_invariance",
        return_value={"passed": True, "comparisons": 1, "mismatches": []},
    )
    def test_gap_direction_and_one_percent_gates_fail_closed(self, _invariance):
        runs = _runs()
        for seed in SEEDS:
            runs[("lodo_d1", seed, "full")]["metrics"]["round4"]["families"][
                "gap"
            ]["summary"]["model"]["smooth_l1"] = 1.2
            runs[("lodo_d2", seed, "full")]["metrics"]["round4"]["families"][
                "gap"
            ]["summary"]["model"]["smooth_l1"] = 1.2
            runs[("lodo_d1", seed, "full")]["metrics"]["round4"]["families"][
                "ordinary"
            ]["summary"]["model"]["smooth_l1"] = 1.04
        result = summarize_formal_matrix(runs)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["gap_vs_idw"])
        self.assertFalse(result["checks"]["gap_vs_one_shot"])
        self.assertFalse(result["checks"]["ordinary_vs_idw_within_one_percent"])

    @mock.patch(
        "training.pro_normst_acceptance._round_invariance",
        return_value={"passed": False, "comparisons": 1, "mismatches": [{}]},
    )
    def test_round_invariance_is_mandatory(self, _invariance):
        result = summarize_formal_matrix(copy.deepcopy(_runs()))
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["round_invariance"])

    def test_round_invariance_uses_tolerance_and_records_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            invariance = _round_invariance(_runs_with_round_predictions(Path(temporary)))
        self.assertTrue(invariance["passed"])
        self.assertEqual(invariance["mismatch_count"], 0)
        self.assertGreater(invariance["max_abs_error"], 0.0)
        self.assertEqual(invariance["tolerance"]["atol"], ROUND_INVARIANCE_ATOL)

    def test_round_invariance_rejects_above_tolerance_and_reports_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = _runs_with_round_predictions(Path(temporary))
            path = (
                runs[("lodo_d1", 2027, "full")]["run_dir"]
                / "test_artifacts"
                / "predictions"
                / "test_round4"
                / "gap"
                / "slice"
                / "mask_0.npz"
            )
            with np.load(path, allow_pickle=False) as arrays:
                np.savez(
                    path,
                    query_index=arrays["query_index"],
                    depth=arrays["depth"],
                    prediction_x=np.asarray([[1.1], [2.0], [3.0]], dtype=np.float32),
                )
            invariance = _round_invariance(runs)
        self.assertFalse(invariance["passed"])
        self.assertEqual(invariance["mismatch_count"], 1)
        mismatch = invariance["mismatches"][0]
        self.assertEqual(mismatch["check"], "depth1_round1_round4")
        self.assertGreater(mismatch["max_abs_error"], 0.01)

    def test_round_invariance_keeps_query_and_depth_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = _runs_with_round_predictions(Path(temporary))
            path = (
                runs[("lodo_d1", 2027, "full")]["run_dir"]
                / "test_artifacts"
                / "predictions"
                / "test_round2"
                / "gap"
                / "slice"
                / "mask_0.npz"
            )
            with np.load(path, allow_pickle=False) as arrays:
                np.savez(
                    path,
                    query_index=np.asarray([11, 20, 30], dtype=np.int64),
                    depth=arrays["depth"],
                    prediction_x=arrays["prediction_x"],
                )
            with self.assertRaisesRegex(ValueError, "query/depth identity drifted"):
                _round_invariance(runs)

    def test_prediction_tree_hash_detects_raw_prediction_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "predictions"
            path = root / "test_round4" / "slice" / "gap" / "mask_00.npz"
            path.parent.mkdir(parents=True)
            np.savez(path, prediction_x=np.asarray([[1.0]], dtype=np.float32))
            original = _prediction_tree_sha256(root)
            np.savez(path, prediction_x=np.asarray([[2.0]], dtype=np.float32))
            self.assertNotEqual(_prediction_tree_sha256(root), original)

    def test_write_formal_acceptance_is_atomic_and_idempotent(self):
        with mock.patch(
            "training.pro_normst_acceptance._round_invariance",
            return_value={"passed": True, "comparisons": 1, "mismatches": []},
        ):
            result = summarize_formal_matrix(_runs())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "acceptance"
            destination.mkdir()
            readme = destination / "README.md"
            readme.write_text("do not touch\n", encoding="utf-8")
            logs = destination / "logs"
            logs.mkdir()
            (logs / "run.log").write_text("preserve\n", encoding="utf-8")
            write_formal_acceptance(result, destination)
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                {"README.md", "logs", FORMAL_ARTIFACTS_DIRECTORY},
            )
            artifact_dir = destination / FORMAL_ARTIFACTS_DIRECTORY
            self.assertEqual(
                {path.name for path in artifact_dir.iterdir()},
                {"formal_acceptance.json", "formal_run_effects.csv"},
            )
            write_formal_acceptance(result, destination)
            self.assertEqual(readme.read_text(encoding="utf-8"), "do not touch\n")
            self.assertEqual((logs / "run.log").read_text(encoding="utf-8"), "preserve\n")
            changed = copy.deepcopy(result)
            changed["accepted"] = not result["accepted"]
            with self.assertRaisesRegex(ValueError, "differ"):
                write_formal_acceptance(changed, destination)
            self.assertEqual(readme.read_text(encoding="utf-8"), "do not touch\n")

    def test_write_formal_acceptance_rejects_partial_managed_artifacts(self):
        with mock.patch(
            "training.pro_normst_acceptance._round_invariance",
            return_value={"passed": True, "comparisons": 1, "mismatches": []},
        ):
            result = summarize_formal_matrix(_runs())
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "acceptance"
            artifact_dir = destination / FORMAL_ARTIFACTS_DIRECTORY
            artifact_dir.mkdir(parents=True)
            (destination / "README.md").write_text("preserve\n", encoding="utf-8")
            (artifact_dir / "formal_acceptance.json").write_text("partial\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete or differ"):
                write_formal_acceptance(result, destination)
            self.assertEqual((destination / "README.md").read_text(encoding="utf-8"), "preserve\n")

    def test_write_formal_acceptance_cleans_staging_on_interruption(self):
        with mock.patch(
            "training.pro_normst_acceptance._round_invariance",
            return_value={"passed": True, "comparisons": 1, "mismatches": []},
        ):
            result = summarize_formal_matrix(_runs())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "acceptance"
            destination.mkdir()
            (destination / "README.md").write_text("preserve\n", encoding="utf-8")
            with mock.patch(
                "training.pro_normst_acceptance.os.replace",
                side_effect=OSError("interrupted"),
            ):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    write_formal_acceptance(result, destination)
            self.assertEqual(
                (destination / "README.md").read_text(encoding="utf-8"), "preserve\n"
            )
            self.assertFalse((destination / FORMAL_ARTIFACTS_DIRECTORY).exists())
            self.assertFalse(list(destination.glob(".formal_artifacts.staging-*")))


if __name__ == "__main__":
    unittest.main()
