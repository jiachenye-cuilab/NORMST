"""Contract tests for the isolated Round12 local gene residual head."""

from __future__ import annotations

import importlib
import unittest

import torch

from models.pro_normst import FullHexGeometry, ProNORMST
from round12.model import LocalGeneResidualHead, Round12ProNORMST
from training.pro_normst_engine import optimizer_for_model


def line_geometry(nodes: int = 6) -> FullHexGeometry:
    xy = torch.stack(
        [torch.arange(nodes, dtype=torch.float32), torch.zeros(nodes)],
        dim=-1,
    )
    neighbor = torch.full((nodes, 6), -1, dtype=torch.long)
    neighbor[1:, 0] = torch.arange(nodes - 1)
    neighbor[:-1, 1] = torch.arange(1, nodes)
    return FullHexGeometry(
        xy=xy,
        neighbor_index=neighbor,
        indices_validated=True,
    )


class Round12IsolationTest(unittest.TestCase):
    def test_import_does_not_modify_v9(self):
        import training.pro_normst as base

        original_model = base.ProNORMST
        original_human = base.HUMAN_CONTRACT_VERSION
        original_numerical = base.NUMERICAL_IMPLEMENTATION_SCHEMA
        importlib.import_module("round12.train")
        self.assertIs(base.ProNORMST, original_model)
        self.assertEqual(base.HUMAN_CONTRACT_VERSION, original_human)
        self.assertEqual(base.NUMERICAL_IMPLEMENTATION_SCHEMA, original_numerical)

    def test_activation_is_explicit_and_process_local(self):
        import training.pro_normst as base
        import round12.train as round12_train

        original = (
            base.ProNORMST,
            base.HUMAN_CONTRACT_VERSION,
            base.NUMERICAL_IMPLEMENTATION_SCHEMA,
            base._contract_manifest,
        )
        try:
            round12_train.activate_round12()
            self.assertIs(base.ProNORMST, Round12ProNORMST)
            self.assertEqual(base.HUMAN_CONTRACT_VERSION, "pro-normst-human-v12")
            self.assertEqual(
                base.NUMERICAL_IMPLEMENTATION_SCHEMA,
                "pro-normst-numerical-v12",
            )
        finally:
            (
                base.ProNORMST,
                base.HUMAN_CONTRACT_VERSION,
                base.NUMERICAL_IMPLEMENTATION_SCHEMA,
                base._contract_manifest,
            ) = original


class Round12ModelTest(unittest.TestCase):
    def setUp(self):
        self.geometry = line_geometry()
        self.visible = torch.linspace(0.0, 1.0, 512).reshape(1, 1, 512)
        self.visible_index = torch.tensor([0])
        self.query_index = torch.tensor([1, 2, 3, 4, 5])

    def _predictions(
        self,
        variant: str,
    ) -> tuple[torch.Tensor, torch.Tensor, ProNORMST, Round12ProNORMST]:
        torch.manual_seed(2027)
        baseline = ProNORMST(torch.zeros(512), variant=variant).eval()
        torch.manual_seed(2027)
        candidate = Round12ProNORMST(torch.zeros(512), variant=variant).eval()
        baseline_prediction = baseline(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
        )
        candidate_prediction = candidate(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
        )
        return baseline_prediction, candidate_prediction, baseline, candidate

    def test_inherited_initialization_and_step_zero_prediction_are_exact(self):
        for variant in ProNORMST.VALID_VARIANTS:
            baseline_prediction, candidate_prediction, baseline, candidate = (
                self._predictions(variant)
            )
            for name, value in baseline.state_dict().items():
                self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
            self.assertTrue(torch.equal(candidate_prediction, baseline_prediction))

    def test_head_output_projection_is_zero_initialized(self):
        model = Round12ProNORMST(torch.zeros(512))
        head = model.local_gene_residual_head
        torch.testing.assert_close(
            head.output_projection.weight,
            torch.zeros_like(head.output_projection.weight),
        )
        torch.testing.assert_close(
            head.output_projection.bias,
            torch.zeros_like(head.output_projection.bias),
        )

    def test_variant_scope_freezes_and_skips_global_only_head(self):
        global_only = Round12ProNORMST(
            torch.zeros(512),
            variant="global-only",
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in global_only.local_gene_residual_head.parameters()
            )
        )
        calls = []
        handle = global_only.local_gene_residual_head.register_forward_pre_hook(
            lambda _module, _inputs: calls.append(1)
        )
        try:
            _, auxiliary = global_only(
                self.visible,
                self.visible_index,
                self.query_index,
                self.geometry,
                return_auxiliary=True,
            )
        finally:
            handle.remove()
        self.assertEqual(calls, [])
        torch.testing.assert_close(
            auxiliary["local_gene_residual"],
            torch.zeros_like(auxiliary["local_gene_residual"]),
        )
        for variant in ("full", "one-shot", "local-only"):
            model = Round12ProNORMST(torch.zeros(512), variant=variant)
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for parameter in model.local_gene_residual_head.parameters()
                )
            )

    def test_inactive_queries_receive_zero_residual(self):
        model = Round12ProNORMST(torch.zeros(512)).eval()
        with torch.no_grad():
            model.local_gene_residual_head.output_projection.weight.fill_(0.01)
        _, auxiliary = model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            return_auxiliary=True,
        )
        inactive = ~auxiliary["active_query"]
        self.assertTrue(bool(inactive.any()))
        torch.testing.assert_close(
            auxiliary["local_gene_residual"][inactive],
            torch.zeros_like(auxiliary["local_gene_residual"][inactive]),
        )

    def test_metadata_is_detached_and_local_input_receives_gradient(self):
        head = LocalGeneResidualHead()
        with torch.no_grad():
            head.output_projection.weight.fill_(0.01)
        local = torch.randn(5, 256, requires_grad=True)
        rounds = torch.tensor([1, 1, 2, 3, 4])
        coverage = torch.rand(5, 1, requires_grad=True)
        confidence = torch.rand(5, 1, requires_grad=True)
        head(local, rounds, coverage, confidence).square().mean().backward()
        self.assertIsNotNone(local.grad)
        self.assertGreater(float(local.grad.abs().sum()), 0.0)
        self.assertIsNone(coverage.grad)
        self.assertIsNone(confidence.grad)

    def test_all_head_parameters_receive_gradient_after_zero_start_update(self):
        model = Round12ProNORMST(torch.zeros(512))
        optimizer, _ = optimizer_for_model(model)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                self.visible,
                self.visible_index,
                self.query_index,
                self.geometry,
            )
            prediction.square().mean().backward()
            optimizer.step()
        for name, parameter in model.local_gene_residual_head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_nonzero_head_preserves_early_exit_invariance(self):
        model = Round12ProNORMST(torch.zeros(512)).eval()
        with torch.no_grad():
            model.local_gene_residual_head.output_projection.weight.fill_(0.01)
        prediction1, auxiliary1 = model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            round_limit=1,
            return_auxiliary=True,
        )
        prediction2, auxiliary2 = model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            round_limit=2,
            return_auxiliary=True,
        )
        prediction4, auxiliary4 = model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            round_limit=4,
            return_auxiliary=True,
        )
        depth1 = auxiliary4["activation_round"] == 1
        depth2 = auxiliary4["activation_round"] == 2
        self.assertTrue(torch.equal(prediction1[depth1], prediction2[depth1]))
        self.assertTrue(torch.equal(prediction1[depth1], prediction4[depth1]))
        self.assertTrue(torch.equal(prediction2[depth2], prediction4[depth2]))
        self.assertTrue(
            torch.equal(
                auxiliary1["local_gene_residual"][depth1],
                auxiliary4["local_gene_residual"][depth1],
            )
        )
        self.assertTrue(
            torch.equal(
                auxiliary2["local_gene_residual"][depth2],
                auxiliary4["local_gene_residual"][depth2],
            )
        )

    def test_manifest_freezes_round12_head(self):
        manifest = Round12ProNORMST(torch.zeros(512)).contract_manifest()
        self.assertEqual(manifest["schema"], "pro-normst-direct-512-v7")
        self.assertEqual(
            manifest["local_gene_residual_head"],
            "gated-local-round-reliability-mlp-v1",
        )
        self.assertEqual(manifest["local_gene_residual_round_embedding_dim"], 8)
        self.assertEqual(manifest["local_gene_residual_hidden_dim"], 256)
        self.assertEqual(manifest["local_gene_residual_output_dim"], 512)
        self.assertEqual(manifest["local_gene_residual_output_init"], "zeros")
        self.assertTrue(manifest["local_gene_residual_activation_grouped"])
        self.assertTrue(manifest["local_gene_residual_metadata_detached"])
        self.assertFalse(manifest["local_gene_residual_global_input"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_amp_step_zero_and_round_invariance(self):
        device = torch.device("cuda")
        torch.manual_seed(2027)
        baseline = ProNORMST(torch.zeros(512)).to(device).eval()
        torch.manual_seed(2027)
        candidate = Round12ProNORMST(torch.zeros(512)).to(device).eval()
        visible = self.visible.to(device)
        visible_index = self.visible_index.to(device)
        query_index = self.query_index.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            baseline_prediction = baseline(
                visible,
                visible_index,
                query_index,
                self.geometry,
            )
            candidate_prediction = candidate(
                visible,
                visible_index,
                query_index,
                self.geometry,
            )
        self.assertTrue(torch.equal(candidate_prediction, baseline_prediction))

        with torch.no_grad():
            candidate.local_gene_residual_head.output_projection.weight.fill_(0.01)
        outputs = {}
        auxiliaries = {}
        for round_limit in (1, 2, 4):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs[round_limit], auxiliaries[round_limit] = candidate(
                    visible,
                    visible_index,
                    query_index,
                    self.geometry,
                    round_limit=round_limit,
                    return_auxiliary=True,
                )
        depth1 = auxiliaries[4]["activation_round"] == 1
        depth2 = auxiliaries[4]["activation_round"] == 2
        torch.testing.assert_close(
            outputs[1][depth1],
            outputs[4][depth1],
            rtol=2e-3,
            atol=2e-4,
        )
        torch.testing.assert_close(
            outputs[2][depth2],
            outputs[4][depth2],
            rtol=2e-3,
            atol=2e-4,
        )


if __name__ == "__main__":
    unittest.main()
