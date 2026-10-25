"""Small CPU checks of scientific behavior, independent of real data or weights."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial import cKDTree
import torch

from examples.synthetic import make_inputs
from normst.conditioning import condition_with_prior, group_batch
from normst.data import load_observations, load_settings, manifest, write_json
from normst.geometry import BasisIntegrals, circle_quadrature, knot_points
from normst.losses import allocation_deviance, coarse_deviance, count_deviance
from normst.model import NORMST
from normst.predict import load_model, multiply, predict, projected_integrals
from normst.train import backward_views, fit, make_contexts


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="normst-test-")
        cls.folder = Path(cls.temporary.name)
        cls.dataset = make_inputs(cls.folder / "data")
        write_json(cls.folder / "config.json", {"epochs": 4})
        cls.checkpoint = fit(cls.dataset, cls.folder / "model", device="cpu", config=cls.folder / "config.json")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_default_parameter_count_and_initial_prior(self):
        settings = load_settings()["conditioning"]
        bounds = np.array([0., 0., 64., 64.])
        counts = np.full((2, 2000), 3.)
        context = condition_with_prior(counts, np.array([[16., 16.], [48., 48.]]),
                                       np.full(2, 100.), np.ones(2, bool), bounds, settings)
        torch.manual_seed(2027)
        field = NORMST(context, bounds, torch.zeros(1, 2563, 8, 8), torch.ones(1, 1, 8, 8, dtype=torch.bool))
        self.assertEqual(sum(p.numel() for p in field.parameters()), 504048)
        points = knot_points(torch.arange(64), field.bounds, 8., field.basis_shape)
        with torch.no_grad():
            density = field.coefficients(field.encode(), points)
        torch.testing.assert_close(density, torch.full_like(density, .03), rtol=1e-6, atol=1e-8)

    def test_shared_image_gradients_match_two_complete_forwards(self):
        settings = load_settings()
        data = load_observations(manifest(self.dataset))
        full, contexts, owners = make_contexts(data, settings, "cpu")
        field, _ = load_model(self.checkpoint)
        field.requires_grad_(True)
        reference = copy.deepcopy(field)
        rows, parents = group_batch(owners[0], np.unique(owners[0]))
        observed = torch.as_tensor(data["counts"][rows])
        quadrature = BasisIntegrals(data, "cpu", 8.)
        backward_views(field, full, contexts[0], quadrature, rows, observed, parents, 64)
        for coarse, context in ((True, contexts[0]), (False, full)):
            reference.set_context(context)
            prediction = reference.regional_counts(reference.encode(), *quadrature.batch(rows), len(rows), 64)
            loss = (allocation_deviance(prediction, observed, parents) + coarse_deviance(prediction, observed, parents)
                    if coarse else count_deviance(prediction.double(), observed.double()))
            loss.backward()
        actual = dict(field.named_parameters())
        for name, parameter in reference.named_parameters():
            self.assertIsNotNone(actual[name].grad, name)
            self.assertTrue(torch.isfinite(actual[name].grad).all(), name)
            torch.testing.assert_close(actual[name].grad, parameter.grad, rtol=3e-5, atol=1e-7)
        for name in ("encoder.0.weight", "he_encoder.0.weight", "he_projection.weight",
                     "image_bypass.0.weight", "decoder.2.weight", "adapter.2.weight"):
            self.assertGreater(float(actual[name].grad.abs().sum()), 0., name)

    def test_native_region_conservation_and_chunk_invariance(self):
        field, data = load_model(self.checkpoint)
        with torch.no_grad():
            latent = field.encode()
            points = knot_points(torch.arange(np.prod(field.basis_shape)), field.bounds, 8., field.basis_shape)
            coefficients = field.coefficients(latent, points, 71).numpy()
            np.testing.assert_allclose(coefficients, field.coefficients(latent, points, 512).numpy(), rtol=1e-6, atol=1e-8)
        quadrature = BasisIntegrals(data, "cpu", 8.)
        raw = multiply(quadrature.matrix.astype(np.float64), coefficients, "cpu")
        factors = data["counts"] / raw
        nodes, weights, owners = circle_quadrature(data["centers"], data["radii"], data["quadrature"])
        values = projected_integrals(nodes, weights, owners, len(data["counts"]), field,
            coefficients, data, factors, cKDTree(data["centers"]))
        np.testing.assert_allclose(values, data["counts"], rtol=1e-10, atol=1e-8)

    def test_checkpoint_reload_and_additive_grid_readouts(self):
        fine = predict(self.checkpoint, self.folder / "grid8", device="cpu", bin_um=8)
        coarse = predict(self.checkpoint, self.folder / "grid16", device="cpu", bin_um=16)
        fine_values, coarse_values = np.load(fine), np.load(coarse)
        aggregated = fine_values.reshape(20, 2, 20, 2, 12).sum(axis=(1, 3)).reshape(400, 12)
        np.testing.assert_allclose(coarse_values, aggregated, rtol=2e-6, atol=1e-7)
        self.assertTrue(np.isfinite(fine_values).all() and (fine_values >= 0).all())
        query = predict(self.checkpoint, self.folder / "query", device="cpu",
                        queries=self.folder / "data/observations.npz")
        with np.load(self.folder / "data/observations.npz") as data:
            np.testing.assert_allclose(np.load(query), data["counts"], rtol=2e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
