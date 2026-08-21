from __future__ import annotations

import unittest

import numpy as np
import torch

from round13.loss_alignment_audit import (
    aggregate_gene_records,
    concentration_summary,
    detection_rate_strata,
    per_gene_components,
)
from round13.slice_context_audit import FAMILIES, METHODS


class LossAlignmentAuditTest(unittest.TestCase):
    def test_components_are_additive_and_respect_positive_weight(self) -> None:
        prediction = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        target = torch.tensor([[[1.0, -1.0], [0.0, 1.0]]])
        valid = torch.tensor([[True, True]])
        result = per_gene_components(
            prediction,
            target,
            torch.tensor([3.0, 2.0]),
            torch.ones(2),
            valid,
        )

        torch.testing.assert_close(
            result["weighted_z"],
            result["positive_target_contribution"]
            + result["nonpositive_target_contribution"],
        )
        # Gene 0: positive loss has weight 3 and zero-target loss is 0.
        self.assertAlmostEqual(float(result["weighted_z"][0, 0]), 0.375)
        # Gene 1: equal 0.5 losses with weights 1 and 2 => 0.5.
        self.assertAlmostEqual(float(result["weighted_z"][0, 1]), 0.5)

    def test_detection_strata_cover_every_boundary_once(self) -> None:
        rate = np.asarray([0.0, 0.01, 0.1, 0.2, 0.5, 0.9, 1.0])
        strata = detection_rate_strata(rate)
        coverage = np.stack(tuple(strata.values())).sum(axis=0)
        np.testing.assert_array_equal(coverage, np.ones(rate.size))
        self.assertTrue(strata["very_sparse_weight3"][2])
        self.assertTrue(strata["common_weight1"][4])

    def test_gene_aggregation_equal_weights_slices(self) -> None:
        records = []
        for family in FAMILIES:
            for slice_id, values in (("a", [1.0, 3.0]), ("b", [9.0])):
                for index, value in enumerate(values):
                    vector = np.asarray([value, value])
                    records.append(
                        {
                            "slice_id": slice_id,
                            "family": family,
                            "mask_index": index,
                            "methods": {
                                method: {
                                    key: vector.copy()
                                    for key in (
                                        "weighted_z",
                                        "positive_target_contribution",
                                        "nonpositive_target_contribution",
                                        "raw_x_smooth_l1",
                                        "raw_x_mae",
                                    )
                                }
                                for method in METHODS
                            },
                            "target": {
                                "positive_weight_mass_fraction": vector.copy(),
                                "positive_element_fraction": vector.copy(),
                            },
                        }
                    )
        aggregate = aggregate_gene_records(records)
        np.testing.assert_allclose(
            aggregate["methods"]["baseline"]["overall"]["weighted_z"],
            [5.5, 5.5],
        )

    def test_concentration_reports_top_mass(self) -> None:
        summary = concentration_summary(np.asarray([4.0, 1.0, -3.0, 0.0]))
        self.assertEqual(summary["n_worse"], 2)
        self.assertEqual(summary["n_better"], 1)
        self.assertAlmostEqual(
            summary["top_positive_regression_mass_share"]["top_1"], 0.8
        )
        self.assertEqual(summary["improvement_mass"], 3.0)


if __name__ == "__main__":
    unittest.main()
