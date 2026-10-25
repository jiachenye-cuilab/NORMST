"""Discrete capture boundaries, output coverage, and publication after failure."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from normst.conditioning import condition_with_prior
from normst.data import load_settings, read_json
from normst.geometry import capture_query_quadrature, circle_quadrature, geometry
from normst.model import NORMST
from normst.predict import load_model, predict


def legacy_checkpoint(folder, *, radii=(8., 16.), coverage=False, constant=False):
    centers = np.array([[25., 17.], [25., 17. + sum(radii)]])
    data = dict(bounds=np.array([0., 0., 64., 64.]), centers=centers, radii=np.array(radii),
        counts=np.array([[10., 20.], [30., 40.]], dtype=np.float32), support_bins=None,
        quadrature=dict(kind="native_bin_centers", bin_um=2.))
    geometry(data)
    if constant:
        data["counts"] = (data["areas"][:, None] * [1., 2.]).astype(np.float32)
    context = condition_with_prior(data["counts"], centers, data["areas"], np.ones(2, bool),
                                   data["bounds"], load_settings()["conditioning"])
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    if coverage:
        valid[:] = False
        valid[:, :, ::2, ::2] = True
    torch.manual_seed(2027)
    field = NORMST(context, data["bounds"], torch.zeros(1, 7, 8, 8), valid)
    path = folder / "legacy.pt"
    # The original format has no completion or provenance fields.
    torch.save(dict(format_version=1, state_dict=field.state_dict(),
        observations={name: torch.from_numpy(data[name]) for name in ("counts", "centers", "radii")},
        gene_ids=["a", "b"], barcodes=["left", "right"], config=load_settings(),
        epochs=0, optimizer_updates=0, feature_tile_offset_yx=[0, 0]), path)
    return path, data


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory(prefix="normst-readouts-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_standard_visium_quadrature_is_unchanged(self):
        centers = np.array([[60., 60.], [60., 160.], [146.60254037844385, 110.]])
        data = dict(centers=centers, radii=np.full(3, 27.5), counts=np.ones((3, 2)), support_bins=None,
                    quadrature=dict(kind="native_bin_centers", bin_um=2.))
        expected = circle_quadrature(centers, data["radii"], data["quadrature"])
        actual = geometry(data)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a.numpy(), b)

    def test_equal_and_unequal_tangent_circles_have_unique_consistent_nodes(self):
        for radii in ((8., 8.), (8., 16.)):
            with self.subTest(radii=radii):
                checkpoint, data = legacy_checkpoint(self.folder, radii=radii)
                points, weights, owners = geometry(data)
                self.assertEqual(len(np.unique(points.numpy(), axis=0)), len(points))
                shared = (points.numpy() == [25., 25.]).all(1)
                self.assertEqual(int(shared.sum()), 1)
                self.assertEqual(int(owners[shared][0]), 0)
                query_points, query_weights, query_rows = capture_query_quadrature(data["centers"], data["radii"], data)
                for a, b in zip((points, weights, owners), (query_points, query_weights, query_rows)):
                    np.testing.assert_array_equal(a.numpy(), b)
                query = self.folder / "queries.npz"
                np.savez(query, centers_yx_um=data["centers"], radius_um=data["radii"])
                for batch in (1, 2):
                    result = predict(checkpoint, self.folder / f"r{radii[1]}-batch{batch}", device="cpu", queries=query, batch=batch)
                    np.testing.assert_allclose(np.load(result), data["counts"], rtol=2e-6, atol=1e-7)

    def test_partial_coverage_integrates_whole_selected_bins(self):
        checkpoint, _ = legacy_checkpoint(self.folder, coverage=True, constant=True)
        fine = predict(checkpoint, self.folder / "fine", device="cpu", bin_um=8)
        coarse = predict(checkpoint, self.folder / "coarse", device="cpu", bin_um=16)
        # Each 16 um square contains one covered and three uncovered 8 um cells.
        # Coverage selects the square; all of its 256 um² contributes counts.
        np.testing.assert_allclose(np.load(fine), np.tile([64., 128.], (16, 1)), rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(np.load(coarse), np.tile([256., 512.], (16, 1)), rtol=2e-6, atol=1e-6)
        np.testing.assert_array_equal(np.load(coarse.parent / "coverage_fraction.npy"), np.full(16, .25))
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        np.save(self.folder / "mask.npy", mask)
        selected = predict(checkpoint, self.folder / "selected", device="cpu", bin_um=16, mask=self.folder / "mask.npy")
        np.testing.assert_array_equal(np.load(selected), np.load(coarse)[:1])
        # Circular queries use their physical support even outside image coverage.
        np.savez(self.folder / "gap.npz", centers_yx_um=np.array([[11., 11.]]), radius_um=1.)
        gap = predict(checkpoint, self.folder / "gap", device="cpu", queries=self.folder / "gap.npz")
        np.testing.assert_allclose(np.load(gap), [[4., 8.]], rtol=2e-6, atol=1e-7)

    def test_legacy_checkpoint_load_and_prediction_failure_recovery(self):
        checkpoint, _ = legacy_checkpoint(self.folder)
        field, _ = load_model(checkpoint)
        field.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"], strict=True)
        output = self.folder / "prediction"
        with patch("normst.predict.projected_integrals", side_effect=RuntimeError("simulated prediction interruption")):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                predict(checkpoint, output, device="cpu")
        self.assertFalse(output.exists())
        failed = list(self.folder.glob(".prediction.partial-*"))
        self.assertEqual(len(failed), 1)
        self.assertEqual(read_json(failed[0] / "status.json")["status"], "failed")
        result = predict(checkpoint, output, device="cpu")
        self.assertTrue(result.exists())
        self.assertEqual(read_json(output / "status.json")["status"], "complete")
        self.assertEqual(read_json(output / "prediction.json")["status"], "complete")
        with self.assertRaises(FileExistsError):
            predict(checkpoint, output, device="cpu")


if __name__ == "__main__":
    unittest.main()
