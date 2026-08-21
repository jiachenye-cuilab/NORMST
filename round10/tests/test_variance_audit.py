from __future__ import annotations

import unittest

import numpy as np

from round10.variance_audit import _mask_calibration


class VarianceAuditTest(unittest.TestCase):
    def test_known_shrinkage_is_recovered(self) -> None:
        target = np.asarray(
            [
                [0.0, 2.0, 1.0],
                [1.0, 0.0, 2.0],
                [2.0, 1.0, 0.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        prediction = target.mean(axis=0, keepdims=True) + 0.25 * (
            target - target.mean(axis=0, keepdims=True)
        )
        result = _mask_calibration(prediction, target)
        self.assertAlmostEqual(result["variance_ratio_median"], 0.0625, places=7)
        self.assertAlmostEqual(result["std_ratio_median"], 0.25, places=7)
        self.assertAlmostEqual(
            result["prediction_on_truth_slope_median"], 0.25, places=7
        )
        self.assertAlmostEqual(
            result["oracle_variance_restored"]["rmse"], 0.0, places=7
        )
        self.assertAlmostEqual(
            result["oracle_per_gene_affine"]["rmse"], 0.0, places=7
        )


if __name__ == "__main__":
    unittest.main()
