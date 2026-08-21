from __future__ import annotations

import copy
import unittest

import numpy as np

from round13.slice_context_audit import (
    FAMILIES,
    METHODS,
    aggregate_validation_records,
    audit_decision,
    context_correction,
    fit_context_affine,
)


def _metrics(value: float) -> dict[str, float]:
    return {
        "weighted_z_smooth_l1": value,
        "smooth_l1": value,
        "mae": value,
        "rmse": value,
        "gene_pearson": 1.0 - value,
        "spot_pearson": 1.0 - value,
        "variance_ratio_median": value,
    }


def _fold(context_gain: float, bias_gain: float = 0.005) -> dict:
    baseline = 0.2
    methods = {}
    for method, criterion in (
        ("baseline", baseline),
        ("train_bias", baseline - bias_gain),
        ("context_affine", baseline - context_gain),
    ):
        methods[method] = {
            "criterion_weighted_z_smooth_l1": criterion,
            "families": {
                family: {
                    "summary": {
                        **_metrics(criterion),
                        "gene_pearson": 0.5,
                        "spot_pearson": 0.6,
                    }
                }
                for family in FAMILIES
            },
        }
    return {
        "methods": methods,
        "gains": {
            "context_affine": {"criterion_gain": context_gain},
        },
        "replay": {"passed": True},
    }


class SliceContextAuditTest(unittest.TestCase):
    def test_fit_context_affine_recovers_bias_and_slope(self) -> None:
        context = np.asarray(
            [[-2.0, 7.0], [-1.0, 7.0], [1.0, 7.0], [2.0, 7.0]],
            dtype=np.float32,
        )
        residual = np.column_stack(
            (1.5 + 0.25 * context[:, 0], np.full(4, -0.75))
        )
        fit = fit_context_affine(context, residual)

        np.testing.assert_allclose(fit["residual_mean"], [1.5, -0.75])
        np.testing.assert_allclose(fit["slope"], [0.25, 0.0])
        np.testing.assert_allclose(
            context_correction(context[0], fit), [1.0, -0.75]
        )
        self.assertEqual(int(fit["context_nonconstant_genes"]), 1)

    def test_aggregation_equal_weights_masks_then_slices(self) -> None:
        records = []
        for family in FAMILIES:
            for slice_id, values in (("a", [1.0, 3.0]), ("b", [9.0])):
                for index, value in enumerate(values):
                    records.append(
                        {
                            "slice_id": slice_id,
                            "family": family,
                            "mask_index": index,
                            "methods": {
                                method: _metrics(value) for method in METHODS
                            },
                        }
                    )

        result = aggregate_validation_records(records)

        # Slice a mean is 2, slice b mean is 9, and slices are equal-weighted.
        self.assertEqual(
            result["baseline"]["families"]["ordinary"]["summary"][
                "weighted_z_smooth_l1"
            ],
            5.5,
        )
        self.assertEqual(
            result["baseline"]["criterion_weighted_z_smooth_l1"], 5.5
        )

    def test_decision_requires_context_to_beat_bias_and_two_folds(self) -> None:
        passing = [_fold(0.01), _fold(0.02), _fold(-0.001)]
        decision = audit_decision(passing)
        self.assertTrue(decision["supports_round13_film"])

        loses_to_bias = copy.deepcopy(passing)
        for fold in loses_to_bias:
            fold["methods"]["train_bias"][
                "criterion_weighted_z_smooth_l1"
            ] = 0.18
        decision = audit_decision(loses_to_bias)
        self.assertFalse(decision["gates"]["context_beats_train_bias"])
        self.assertFalse(decision["supports_round13_film"])

        only_one_positive = [_fold(0.01), _fold(-0.001), _fold(-0.002)]
        decision = audit_decision(only_one_positive)
        self.assertFalse(decision["gates"]["positive_in_at_least_two_folds"])
        self.assertFalse(decision["supports_round13_film"])

    def test_decision_fails_closed_on_replay_or_pearson_drop(self) -> None:
        folds = [_fold(0.01), _fold(0.02), _fold(-0.001)]
        folds[0]["replay"]["passed"] = False
        decision = audit_decision(folds)
        self.assertFalse(decision["gates"]["baseline_replay_all_folds"])

        folds = [_fold(0.01), _fold(0.02), _fold(-0.001)]
        for fold in folds:
            fold["methods"]["context_affine"]["families"]["gap"]["summary"][
                "gene_pearson"
            ] = 0.498
        decision = audit_decision(folds)
        self.assertFalse(
            decision["gates"]["pearson_not_worse_beyond_tolerance"]
        )


if __name__ == "__main__":
    unittest.main()
