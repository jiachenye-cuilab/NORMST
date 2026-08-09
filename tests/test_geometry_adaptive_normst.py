"""Synthetic sanity tests for the geometry-adaptive local-global framework."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from models.geometry_adaptive_normst import (
    VisiumHDNORMST,
    VisiumNORMST,
    build_native_hex_neighbors,
    build_visible_native_neighbor_graph,
)
from models.local_global_operator import (
    GalerkinOperator,
    GridGeometry,
    GridLocalOperator,
    HexGeometry,
    HexNativeLocalOperator,
    NeuralOperatorBlock,
    grid_to_tokens,
    tokens_to_grid,
)


def artificial_hex() -> tuple[torch.Tensor, torch.Tensor]:
    root3 = 3.0 ** 0.5
    xy = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, root3 / 2.0],
        [-0.5, root3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -root3 / 2.0],
        [0.5, -root3 / 2.0],
    ])
    neighbor = torch.full((7, 6), -1, dtype=torch.long)
    neighbor[0] = torch.arange(1, 7)
    neighbor[1:, 0] = 0
    return xy, neighbor


def manual_idw(
    expression: torch.Tensor,
    visible_xy: torch.Tensor,
    query_xy: torch.Tensor,
    neighbors: int,
) -> torch.Tensor:
    distance = torch.cdist(query_xy.float(), visible_xy.float())
    distance, index = torch.topk(
        distance, k=min(neighbors, visible_xy.shape[0]), largest=False
    )
    gathered = expression[index]
    coincident = distance <= 1e-8
    has_coincident = coincident.any(dim=-1, keepdim=True)
    coincident_weight = coincident.float()
    coincident_weight = coincident_weight / coincident_weight.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    inverse = distance.clamp_min(1e-8).pow(-2.0)
    inverse = inverse / inverse.sum(dim=-1, keepdim=True)
    weight = torch.where(has_coincident, coincident_weight, inverse)
    return (gathered * weight[..., None]).sum(dim=1)


class GalerkinSanityTest(unittest.TestCase):
    def test_shape_variable_n_mask_and_padding_invariance(self):
        torch.manual_seed(1)
        operator = GalerkinOperator(width=8, num_heads=2)
        for points in (5, 11):
            tokens = torch.randn(2, points, 8)
            mask = torch.ones(2, points, dtype=torch.bool)
            mask[1, -2:] = False
            output = operator(tokens, mask)
            self.assertEqual(output.shape, tokens.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.equal(output[1, -2:], torch.zeros_like(output[1, -2:])))

        core = torch.randn(1, 5, 8)
        padded = torch.cat([core, torch.randn(1, 3, 8)], dim=1)
        core_output = operator(core, torch.ones(1, 5, dtype=torch.bool))
        padded_output = operator(
            padded,
            torch.tensor([[True, True, True, True, True, False, False, False]]),
        )
        torch.testing.assert_close(core_output, padded_output[:, :5], atol=1e-6, rtol=1e-6)

    def test_optional_quadrature_is_finite(self):
        operator = GalerkinOperator(width=8, num_heads=2)
        tokens = torch.randn(1, 6, 8)
        weights = torch.tensor([[1.0, 0.5, 2.0, 1.0, 0.25, 1.0]])
        output = operator(tokens, quadrature_weight=weights)
        self.assertTrue(torch.isfinite(output).all())


class HexLocalSanityTest(unittest.TestCase):
    def test_raw_array_coordinates_produce_six_native_neighbors(self):
        rows = torch.tensor([1, 1, 1, 0, 0, 2, 2])
        cols = torch.tensor([2, 0, 4, 1, 3, 1, 3])
        neighbor = build_native_hex_neighbors(rows, cols)
        self.assertEqual(int((neighbor[0] >= 0).sum()), 6)
        self.assertLessEqual(int((neighbor >= 0).sum(dim=1).max()), 6)

    def test_native_topology_survives_random_visibility(self):
        xy, full_neighbor = artificial_hex()
        full = build_visible_native_neighbor_graph(
            full_neighbor, xy, torch.arange(7)
        )
        self.assertEqual(int(full.neighbor_mask[0].sum()), 6)
        self.assertLessEqual(int(full.neighbor_mask.sum(dim=1).max()), 6)

        # Hide one first-order neighbor.  The center retains five neighbors;
        # no more distant point is inserted to restore a count of six.
        visible = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        restricted = build_visible_native_neighbor_graph(
            full_neighbor, xy, visible
        )
        self.assertEqual(int(restricted.neighbor_mask[0].sum()), 5)
        self.assertTrue((restricted.neighbor_index[0] < len(visible)).all())

    def test_order_invariance_relative_xy_and_edges(self):
        torch.manual_seed(2)
        xy, full_neighbor = artificial_hex()
        geometry = build_visible_native_neighbor_graph(
            full_neighbor, xy, torch.arange(7)
        )
        operator = HexNativeLocalOperator(width=4)
        tokens = torch.randn(1, 7, 4)
        output = operator(tokens, geometry)
        self.assertEqual(output.shape, tokens.shape)
        self.assertTrue(torch.isfinite(output).all())

        permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        permuted = HexGeometry(
            geometry.neighbor_index[:, permutation],
            geometry.relative_xy[:, permutation],
            geometry.neighbor_mask[:, permutation],
        )
        permuted_output = operator(tokens, permuted)
        torch.testing.assert_close(output, permuted_output, atol=1e-6, rtol=1e-6)

        changed_relative = geometry.relative_xy.clone()
        changed_relative[0, 0] += torch.tensor([0.7, -0.3])
        changed = HexGeometry(
            geometry.neighbor_index, changed_relative, geometry.neighbor_mask
        )
        changed_output = operator(tokens, changed)
        self.assertGreater(
            float((output[0, 0] - changed_output[0, 0]).abs().max().detach()), 1e-7
        )


class GridAndBlockSanityTest(unittest.TestCase):
    def test_grid_roundtrip_local_shape_and_alignment(self):
        torch.manual_seed(3)
        grid = torch.randn(2, 4, 3, 5)
        geometry = GridGeometry(3, 5)
        tokens = grid_to_tokens(grid)
        torch.testing.assert_close(tokens_to_grid(tokens, geometry), grid)
        local = GridLocalOperator(width=4)
        self.assertEqual(local(tokens, geometry).shape, tokens.shape)

        model = VisiumHDNORMST(
            n_genes=2, width=8, num_heads=2, num_layers=1, scale=2
        )
        coarse = torch.randn(1, 2, 2, 3)
        baseline_scale = torch.tensor([0.5, 2.0])
        prediction = model(coarse, baseline_scale=baseline_scale)
        expected = F.interpolate(
            coarse, scale_factor=2, mode="bilinear", align_corners=False
        ) * baseline_scale.reshape(1, 2, 1, 1)
        self.assertEqual(prediction.shape, (1, 2, 4, 6))
        torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-6)

    def test_block_ablations_gradients_and_learnable_alpha(self):
        torch.manual_seed(4)
        xy, full_neighbor = artificial_hex()
        geometry = build_visible_native_neighbor_graph(
            full_neighbor, xy, torch.arange(7)
        )
        tokens = torch.randn(1, 7, 8)
        for mode in (
            "local_only",
            "galerkin_only",
            "parallel",
            "local_then_global",
            "global_then_local",
        ):
            block = NeuralOperatorBlock(
                width=8,
                num_heads=2,
                local_operator=HexNativeLocalOperator(8),
                mode=mode,
            )
            output = block(tokens, geometry)
            self.assertEqual(output.shape, tokens.shape)
            self.assertTrue(torch.isfinite(output).all())

        concat = NeuralOperatorBlock(
            width=8,
            num_heads=2,
            local_operator=HexNativeLocalOperator(8),
            mode="parallel",
            fusion="concat",
            learnable_alpha=True,
        )
        train_tokens = tokens.clone().requires_grad_(True)
        loss = concat(train_tokens, geometry).square().mean()
        loss.backward()
        self.assertIsNotNone(concat.alpha_local.grad)
        self.assertIsNotNone(concat.alpha_global.grad)
        self.assertTrue(torch.isfinite(concat.alpha_local.grad))
        self.assertTrue(torch.isfinite(concat.alpha_global.grad))
        finite_gradients = [
            torch.isfinite(parameter.grad).all()
            for parameter in concat.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(finite_gradients and all(finite_gradients))


class EndToEndSanityTest(unittest.TestCase):
    def test_joint_1000_gene_shapes_and_independent_parameters(self):
        torch.manual_seed(7)
        xy, full_neighbor = artificial_hex()
        geometry = build_visible_native_neighbor_graph(
            full_neighbor, xy, torch.arange(7)
        )
        visium = VisiumNORMST(
            n_genes=1000, width=8, num_heads=2, num_layers=1
        )
        hd = VisiumHDNORMST(
            n_genes=1000, width=8, num_heads=2, num_layers=1
        )
        point_output = visium(
            torch.randn(1, 7, 1000),
            xy[None],
            torch.tensor([[[0.1, 0.1], [-0.2, 0.3]]]),
            geometry,
        )
        grid_output = hd(torch.randn(1, 1000, 2, 2))
        self.assertEqual(point_output.shape, (1, 2, 1000))
        self.assertEqual(grid_output.shape, (1, 1000, 4, 4))
        self.assertIsInstance(visium.blocks[0], NeuralOperatorBlock)
        self.assertIsInstance(hd.blocks[0], NeuralOperatorBlock)
        visium_storage = {parameter.data_ptr() for parameter in visium.parameters()}
        hd_storage = {parameter.data_ptr() for parameter in hd.parameters()}
        self.assertTrue(visium_storage.isdisjoint(hd_storage))

    def test_visium_initial_prediction_is_exact_idw(self):
        torch.manual_seed(5)
        xy, full_neighbor = artificial_hex()
        expression = torch.randn(7, 3)
        query = torch.tensor([[0.0, 0.0], [0.2, 0.15]])
        geometry = build_visible_native_neighbor_graph(
            full_neighbor, xy, torch.arange(7)
        )
        model = VisiumNORMST(
            n_genes=3,
            width=8,
            num_heads=2,
            num_layers=1,
            query_neighbors=3,
        )
        prediction = model(
            expression[None], xy[None], query[None], geometry
        )[0]
        expected = manual_idw(expression, xy, query, neighbors=3)
        torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-6)

    def test_tiny_visium_and_hd_overfit(self):
        torch.manual_seed(6)
        xy, full_neighbor = artificial_hex()
        visible_index = torch.arange(1, 7)
        visible_xy = xy[visible_index]
        visible_expression = torch.randn(1, 6, 2)
        query_xy = torch.tensor([[[0.0, 0.0], [0.25, -0.1]]])
        geometry = build_visible_native_neighbor_graph(
            full_neighbor, xy, visible_index
        )
        visium = VisiumNORMST(
            n_genes=2,
            width=8,
            num_heads=2,
            num_layers=1,
            query_neighbors=3,
        )
        with torch.no_grad():
            baseline = visium(
                visible_expression, visible_xy[None], query_xy, geometry
            )
        target = baseline + torch.tensor([[[0.4, -0.2], [-0.3, 0.5]]])
        optimizer = torch.optim.Adam(visium.parameters(), lr=3e-2)
        initial = None
        for _ in range(60):
            optimizer.zero_grad()
            prediction = visium(
                visible_expression, visible_xy[None], query_xy, geometry
            )
            loss = F.mse_loss(prediction, target)
            if initial is None:
                initial = float(loss.detach())
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), initial * 0.1)

        hd = VisiumHDNORMST(
            n_genes=1,
            width=8,
            num_heads=2,
            num_layers=1,
            scale=2,
        )
        coarse = torch.randn(1, 1, 2, 2)
        with torch.no_grad():
            hd_baseline = hd(coarse)
        hd_target = hd_baseline + torch.linspace(
            -0.3, 0.3, 16
        ).reshape(1, 1, 4, 4)
        optimizer = torch.optim.Adam(hd.parameters(), lr=3e-2)
        hd_initial = None
        for _ in range(60):
            optimizer.zero_grad()
            prediction = hd(coarse)
            hd_loss = F.mse_loss(prediction, hd_target)
            if hd_initial is None:
                hd_initial = float(hd_loss.detach())
            hd_loss.backward()
            optimizer.step()
        self.assertLess(float(hd_loss.detach()), hd_initial * 0.1)


if __name__ == "__main__":
    unittest.main()
