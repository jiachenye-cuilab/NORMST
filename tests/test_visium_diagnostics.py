"""Numerical tests for optional Visium diagnostic analyses."""

from __future__ import annotations

import unittest

import numpy as np

from training.diagnostics import (
    analyze_gene_loss_contribution,
    analyze_latent_effective_rank,
    analyze_pca_reconstruction,
    analyze_residual_pca,
)


class VisiumDiagnosticTest(unittest.TestCase):
    def test_truth_fitted_pca_compares_idw_and_normst_on_one_basis(self):
        rng = np.random.default_rng(12)
        truth = rng.normal(size=(80, 60))
        centered = truth - truth.mean(axis=0, keepdims=True)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        first_score = centered @ components[0]
        idw = truth.mean(axis=0, keepdims=True) + np.outer(
            first_score, components[0]
        )
        prediction = truth + rng.normal(scale=1e-3, size=truth.shape)
        rows = analyze_pca_reconstruction(
            truth, idw, prediction, max_components=50
        )
        self.assertEqual(len(rows), 50)
        self.assertGreater(rows[0]["idw_corr"], 0.999)
        self.assertTrue(np.isnan(rows[1]["idw_corr"]))
        self.assertGreater(rows[1]["normst_corr"], 0.999)

    def test_residual_pca_truncates_thresholds_to_available_dimensions(self):
        rng = np.random.default_rng(13)
        truth = rng.normal(size=(20, 12))
        idw = truth * 0.25
        rows, summary = analyze_residual_pca(truth, idw)
        self.assertEqual(len(rows), 12)
        self.assertEqual(summary["pc256_actual_component"], 12)
        self.assertAlmostEqual(
            summary["pc256_cumulative_explained_variance"], 1.0
        )

    def test_latent_effective_rank_uses_centered_singular_values(self):
        rng = np.random.default_rng(14)
        basis = rng.normal(size=(100, 5))
        h0 = basis @ rng.normal(size=(5, 16))
        hl = basis[:, :2] @ rng.normal(size=(2, 16))
        rows, summary = analyze_latent_effective_rank(h0, hl)
        self.assertTrue(rows)
        self.assertIn("singular_value/sum", summary["definition"])
        self.assertGreater(summary["h0"]["effective_rank"], 3.0)
        self.assertLess(summary["hl"]["effective_rank"], 2.1)

    def test_gene_loss_contribution_is_sorted_and_reports_top_shares(self):
        truth = np.zeros((10, 4), dtype=np.float64)
        prediction = np.zeros_like(truth)
        prediction[:, 0] = 3.0
        prediction[:, 1] = 1.0
        rows, summary = analyze_gene_loss_contribution(
            truth, prediction, ["g0", "g1", "g2", "g3"]
        )
        self.assertEqual(rows[0]["gene"], "g0")
        self.assertGreater(rows[0]["loss_fraction"], rows[1]["loss_fraction"])
        self.assertGreater(summary["top_20_percent_loss_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
