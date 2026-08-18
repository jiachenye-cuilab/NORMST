"""Contract tests for the independent direct-512 ProNORMST path."""

from __future__ import annotations

import copy
import inspect
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from datasets.pro_normst import (
    ProNORMSTSlice,
    SliceSpec,
    _read_slice,
    load_panel,
    load_split_manifest,
    prepare_pro_normst_data,
)
from models.pro_normst import FullHexGeometry, ProNORMST
from training.pro_normst import (
    _checkpoint_payload,
    _evaluate_role,
    _final_loss_bptt_gate,
    _load_checkpoint,
    _numerical_contract_hash,
    _pilot_gate,
    _run_test_once,
    _validate_args,
    _validate_candidate_lock,
    _validate_existing_run,
    _write_candidate_lock,
)
from training.pro_normst_engine import (
    learning_rate_for_step,
    evaluate_mask,
    optimizer_for_model,
    strict_visible_idw,
    weighted_gene_smooth_l1,
)
from training.pro_normst_masks import (
    build_mask_geometry,
    generate_gap_mask,
    generate_ordinary_mask,
    make_mask_identity,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def line_geometry(nodes: int = 6) -> FullHexGeometry:
    xy = torch.stack(
        [torch.arange(nodes, dtype=torch.float32), torch.zeros(nodes)], dim=-1
    )
    neighbor = torch.full((nodes, 6), -1, dtype=torch.long)
    neighbor[1:, 0] = torch.arange(nodes - 1)
    neighbor[:-1, 1] = torch.arange(1, nodes)
    return FullHexGeometry(xy=xy, neighbor_index=neighbor, indices_validated=True)


def rectangular_hex_graph(rows: int = 15, columns: int = 15) -> np.ndarray:
    coordinates = []
    for row in range(rows):
        parity = row % 2
        coordinates.extend((row, 2 * column + parity) for column in range(columns))
    lookup = {coordinate: index for index, coordinate in enumerate(coordinates)}
    deltas = ((0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1))
    neighbor = np.full((len(coordinates), 6), -1, dtype=np.int64)
    for index, (row, column) in enumerate(coordinates):
        for direction, (delta_row, delta_column) in enumerate(deltas):
            neighbor[index, direction] = lookup.get(
                (row + delta_row, column + delta_column), -1
            )
    return neighbor


class DirectModelContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        self.model = ProNORMST(torch.zeros(512), variant="full")
        self.visible = torch.linspace(0.0, 1.0, 512).reshape(1, 1, 512)
        self.visible_index = torch.tensor([0])
        self.query_index = torch.tensor([1, 2, 3, 4, 5])
        self.geometry = line_geometry()

    def test_forward_has_no_query_expression_or_autoencoder(self):
        signature = inspect.signature(self.model.forward)
        self.assertNotIn("query_expression", signature.parameters)
        self.assertNotIn("target", signature.parameters)
        self.assertFalse(hasattr(self.model, "latent_encoder"))
        self.assertFalse(any("encoder" in name for name, _ in self.model.named_parameters()))
        self.assertEqual(self.model.state_dim, 512)
        self.assertFalse(self.model.contract_manifest()["query_truth_in_forward"])

    def test_visible_expression_is_the_exact_initial_state(self):
        _, auxiliary = self.model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            return_auxiliary=True,
        )
        torch.testing.assert_close(auxiliary["visible_state"], self.visible)
        torch.testing.assert_close(auxiliary["full_state"][0, 0], self.visible[0, 0])

    def test_global_memory_contains_original_visible_nodes_only(self):
        _, auxiliary = self.model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            return_auxiliary=True,
            return_diagnostics=True,
        )
        attention = auxiliary["global_diagnostics"]["attention"]
        self.assertEqual(attention.shape, (1, 8, 5, 1))
        torch.testing.assert_close(attention, torch.ones_like(attention))

    def test_same_checkpoint_early_exit_invariance(self):
        prediction1, auxiliary1 = self.model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            round_limit=1,
            return_auxiliary=True,
        )
        prediction2, auxiliary2 = self.model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            round_limit=2,
            return_auxiliary=True,
        )
        prediction4, auxiliary4 = self.model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            round_limit=4,
            return_auxiliary=True,
        )
        torch.testing.assert_close(prediction1[:, 0], prediction2[:, 0])
        torch.testing.assert_close(prediction1[:, 0], prediction4[:, 0])
        torch.testing.assert_close(prediction2[:, 1], prediction4[:, 1])
        self.assertEqual(auxiliary1["activation_round"][0, 0].item(), 1)
        self.assertEqual(auxiliary2["activation_round"][0, 1].item(), 2)
        self.assertEqual(auxiliary4["activation_round"][0, 3].item(), 4)
        self.assertEqual(auxiliary4["activation_round"][0, 4].item(), -1)

    def test_full_bptt_reaches_visible_state_and_path_scorer(self):
        visible = self.visible.clone().requires_grad_(True)
        _, auxiliary = self.model(
            visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            return_auxiliary=True,
        )
        loss = auxiliary["local_state"][0, 3].square().mean()
        loss.backward()
        self.assertIsNotNone(visible.grad)
        self.assertGreater(float(visible.grad.abs().sum()), 0.0)
        gradient = self.model.local_operator.lambda_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_final_gene_loss_backpropagates_through_shared_local_rounds(self):
        prediction, auxiliary = self.model(
            self.visible,
            self.visible_index,
            self.query_index,
            self.geometry,
            return_auxiliary=True,
            return_diagnostics=True,
        )
        round_states = auxiliary["local_diagnostics"]["round_states"]
        for state in round_states:
            state.retain_grad()
        target = torch.zeros_like(prediction)
        loss = weighted_gene_smooth_l1(prediction, target, torch.ones(512))
        loss.backward()
        for parameter in (
            self.model.local_operator.path_trunk[0].weight,
            self.model.local_operator.lambda_head.weight,
            self.model.local_operator.routing_logits,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        for state in round_states:
            self.assertIsNotNone(state.grad)
            self.assertTrue(torch.isfinite(state.grad).all())
            self.assertGreater(float(state.grad.abs().sum()), 0.0)

    def test_local_projection_receives_only_activated_queries(self):
        projected_rows = []

        def record_rows(_module, inputs):
            projected_rows.append(int(inputs[0].shape[0]))

        handle = self.model.local_projection.register_forward_pre_hook(record_rows)
        try:
            _, auxiliary = self.model(
                self.visible,
                self.visible_index,
                self.query_index,
                self.geometry,
                return_auxiliary=True,
            )
        finally:
            handle.remove()
        active = auxiliary["active_query"]
        self.assertEqual(projected_rows, [int(active.sum().item())])
        self.assertFalse(active[0, -1])
        torch.testing.assert_close(
            auxiliary["local_projected"][~active],
            torch.zeros_like(auxiliary["local_projected"][~active]),
        )
        torch.testing.assert_close(
            auxiliary["gated_local"][~active],
            torch.zeros_like(auxiliary["gated_local"][~active]),
        )

    def test_ablation_inactive_branch_is_frozen_and_skipped(self):
        local = ProNORMST(torch.zeros(512), variant="local-only")
        global_only = ProNORMST(torch.zeros(512), variant="global-only")
        self.assertTrue(all(not value.requires_grad for value in local.global_branch.parameters()))
        self.assertTrue(
            all(not value.requires_grad for value in global_only.local_operator.parameters())
        )
        self.assertFalse(any("gate" in name for name, _ in self.model.named_parameters()))

    def test_indices_must_partition_the_complete_graph(self):
        with self.assertRaisesRegex(ValueError, "partition"):
            self.model(
                self.visible,
                self.visible_index,
                torch.tensor([1, 2, 3, 4]),
                self.geometry,
            )

    def test_changing_query_truth_cannot_change_forward_prediction(self):
        neighbor = self.geometry.neighbor_index.numpy()
        mask_geometry = build_mask_geometry(neighbor)
        mask = generate_ordinary_mask(
            mask_geometry,
            make_mask_identity(
                protocol="synthetic",
                fold="fold0",
                role="val",
                slice_id="slice0",
                family="ordinary",
                mask_index=0,
            ),
        )
        expression = np.linspace(0.0, 2.0, 6 * 512, dtype=np.float32).reshape(6, 512)
        item = ProNORMSTSlice(
            slice_id="slice0",
            role="val",
            donor="donor0",
            pair="pair0",
            barcodes=tuple(f"b{index}" for index in range(6)),
            gene_ids=tuple(f"g{index}" for index in range(512)),
            expression_x=expression.copy(),
            expression_z=expression.copy(),
            array_row=np.arange(6, dtype=np.int64),
            array_col=np.arange(6, dtype=np.int64),
            full_xy=np.column_stack((np.arange(6), np.zeros(6))).astype(np.float32),
            neighbor_index=neighbor,
            native_scale=1.0,
            component_id=np.zeros(6, dtype=np.int32),
        )
        first = evaluate_mask(
            self.model.eval(),
            item,
            mask,
            np.ones(512, dtype=np.float32),
            np.ones(512, dtype=np.float32),
            np.full(512, 0.5, dtype=np.float32),
            torch.device("cpu"),
            use_amp=False,
            return_prediction=True,
        )
        item.expression_x[mask.query_index] += 100.0
        item.expression_z[mask.query_index] += 100.0
        second = evaluate_mask(
            self.model.eval(),
            item,
            mask,
            np.ones(512, dtype=np.float32),
            np.ones(512, dtype=np.float32),
            np.full(512, 0.5, dtype=np.float32),
            torch.device("cpu"),
            use_amp=False,
            return_prediction=True,
        )
        np.testing.assert_array_equal(first["prediction_z"], second["prediction_z"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_cuda_and_amp_forward_tolerance(self):
        cpu_model = self.model.eval()
        cuda_model = ProNORMST(torch.zeros(512), variant="full").cuda().eval()
        cuda_model.load_state_dict(cpu_model.state_dict())
        with torch.no_grad():
            cpu_prediction = cpu_model(
                self.visible,
                self.visible_index,
                self.query_index,
                self.geometry,
            )
            cuda_geometry = FullHexGeometry(
                xy=self.geometry.xy.cuda(),
                neighbor_index=self.geometry.neighbor_index.cuda(),
                indices_validated=True,
            )
            cuda_prediction = cuda_model(
                self.visible.cuda(),
                self.visible_index.cuda(),
                self.query_index.cuda(),
                cuda_geometry,
            ).cpu()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                amp_prediction = cuda_model(
                    self.visible.cuda(),
                    self.visible_index.cuda(),
                    self.query_index.cuda(),
                    cuda_geometry,
                ).float().cpu()
        torch.testing.assert_close(cpu_prediction, cuda_prediction, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(cuda_prediction, amp_prediction, rtol=2e-2, atol=2e-2)


class MaskContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = build_mask_geometry(rectangular_hex_graph())

    def identity(self, family: str, index: int = 0):
        return make_mask_identity(
            protocol="synthetic",
            fold="fold0",
            role="val",
            slice_id="slice0",
            family=family,
            mask_index=index,
        )

    def test_ordinary_is_exact_half_deterministic_and_a_partition(self):
        first = generate_ordinary_mask(self.geometry, self.identity("ordinary"))
        second = generate_ordinary_mask(self.geometry, self.identity("ordinary"))
        self.assertEqual(first.query_index.size, self.geometry.n_nodes // 2)
        np.testing.assert_array_equal(first.query_index, second.query_index)
        self.assertEqual(
            set(first.query_index).union(first.visible_index),
            set(range(self.geometry.n_nodes)),
        )
        self.assertFalse(set(first.query_index).intersection(first.visible_index))

    def test_gap_is_one_attempt_nonovershooting_and_preserves_standard_rings(self):
        mask = generate_gap_mask(self.geometry, self.identity("gap"))
        repeated = generate_gap_mask(self.geometry, self.identity("gap"))
        np.testing.assert_array_equal(mask.query_index, repeated.query_index)
        self.assertLessEqual(mask.query_index.size, self.geometry.n_nodes // 2)
        self.assertGreater(mask.query_index.size, 0)
        query = set(mask.query_index.tolist())
        for hole in mask.holes:
            self.assertTrue(set(hole.nodes).issubset(query))
            if hole.radius in (3, 4):
                self.assertFalse(query.intersection(hole.protected_ring))
            if hole.radius == 2:
                self.assertGreaterEqual(len(hole.nodes), 15)
        self.assertEqual(mask.identity["attempt_index"], 0)
        self.assertTrue(set(mask.provenance).issubset({"standard-r3", "standard-r4", "r2", "random"}))


class FrozenSplitAndDeferredTestDataTest(unittest.TestCase):
    def _mutated_manifest(self, name: str, mutate) -> Path:
        source = SOURCE_ROOT / "pre-train" / "manifests" / name
        payload = json.loads(source.read_text(encoding="utf-8"))
        mutate(payload)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_pilot_seed_and_exact_pair_assignment_are_frozen(self):
        wrong_seed = self._mutated_manifest(
            "random_pair_8_2_2_seed2027.json",
            lambda payload: payload["_meta"].__setitem__("split_seed", 2028),
        )
        with self.assertRaisesRegex(ValueError, "split_seed"):
            load_split_manifest(wrong_seed)

        def swap_complete_pairs(payload):
            train_pair = {key: payload["train"].pop(key) for key in ("151509", "151510")}
            val_pair = {key: payload["val"].pop(key) for key in ("151507", "151508")}
            payload["train"].update(val_pair)
            payload["val"].update(train_pair)

        wrong_roles = self._mutated_manifest(
            "random_pair_8_2_2_seed2027.json", swap_complete_pairs
        )
        with self.assertRaisesRegex(ValueError, "frozen split"):
            load_split_manifest(wrong_roles)

    def test_lodo_fold_and_train_validation_mapping_are_frozen(self):
        wrong_fold = self._mutated_manifest(
            "lodo_d1.json", lambda payload: payload["_meta"].__setitem__("fold", "custom")
        )
        with self.assertRaisesRegex(ValueError, "LODO fold"):
            load_split_manifest(wrong_fold)

        def swap_complete_pairs(payload):
            train_pair = {key: payload["train"].pop(key) for key in ("151671", "151672")}
            val_pair = {key: payload["val"].pop(key) for key in ("151669", "151670")}
            payload["train"].update(val_pair)
            payload["val"].update(train_pair)

        wrong_roles = self._mutated_manifest("lodo_d1.json", swap_complete_pairs)
        with self.assertRaisesRegex(ValueError, "frozen split"):
            load_split_manifest(wrong_roles)

    def test_formal_arguments_require_a_candidate_lock(self):
        common = dict(
            resume=False,
            predict_only=False,
            smoke=False,
            epochs=50,
            seed=2027,
            variant="full",
            candidate_lock=None,
        )
        with self.assertRaisesRegex(ValueError, "candidate-lock"):
            _validate_args(Namespace(**common), "pair_grouped_lodo")
        common["candidate_lock"] = Path("candidate_lock.json")
        _validate_args(Namespace(**common), "pair_grouped_lodo")
        with self.assertRaisesRegex(ValueError, "must not consume"):
            _validate_args(Namespace(**common), "pair_grouped_random_split")

    def test_scanpy_loader_materializes_standard_visium_expression(self):
        panel = tuple(f"ENSG{index:09d}" for index in range(512))
        barcodes = tuple(f"barcode-{index}" for index in range(4))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spatial = root / "spatial"
            spatial.mkdir()
            position_lines = [
                "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres"
            ]
            coordinates = ((0, 0), (0, 2), (1, 1), (1, 3))
            for index, (row, column) in enumerate(coordinates):
                position_lines.append(
                    f"{barcodes[index]},1,{row},{column},{10 * row},{10 * column}"
                )
            (spatial / "tissue_positions.csv").write_text(
                "\n".join(position_lines) + "\n", encoding="utf-8"
            )
            count_path = root / "filtered_feature_bc_matrix.h5"
            count_path.touch()
            adata = mock.Mock()
            adata.obs_names = pd.Index(barcodes)
            adata.var = pd.DataFrame({"gene_ids": panel})
            adata.X = np.ones((4, 512), dtype=np.float32)
            spec = SliceSpec(
                slice_id="synthetic",
                role="test",
                path=root,
                count_file=count_path.name,
                donor="d",
                pair="p",
                position="x",
                serial="a",
            )
            with mock.patch(
                "datasets.pro_normst.sc.read_10x_h5", return_value=adata
            ) as reader:
                item = _read_slice(spec, panel)
            reader.assert_called_once_with(count_path, gex_only=True)
            self.assertEqual(item.expression_x.shape, (4, 512))
            expected = np.log1p(10000.0 / 512.0)
            np.testing.assert_allclose(item.expression_x, expected, rtol=1e-6)

    def test_test_expression_cannot_change_train_fitted_preprocessing(self):
        panel = tuple(f"ENSG{index:09d}" for index in range(512))
        specs = {
            role: [
                SliceSpec(
                    slice_id=role,
                    role=role,
                    path=Path(role),
                    count_file="counts.h5",
                    donor=f"donor-{role}",
                    pair=f"pair-{role}",
                    position="x",
                    serial="a",
                )
            ]
            for role in ("train", "val", "test")
        }

        def prepared(test_value):
            def fake_read(spec, _panel):
                value = {"train": 2.0, "val": 20.0, "test": test_value}[spec.role]
                expression = np.full((2, 512), value, dtype=np.float32)
                neighbor = np.full((2, 6), -1, dtype=np.int64)
                neighbor[0, 1] = 1
                neighbor[1, 0] = 0
                return ProNORMSTSlice(
                    slice_id=spec.slice_id,
                    role=spec.role,
                    donor=spec.donor,
                    pair=spec.pair,
                    barcodes=(f"{spec.role}-0", f"{spec.role}-1"),
                    gene_ids=panel,
                    expression_x=expression,
                    expression_z=np.empty_like(expression),
                    array_row=np.array([0, 0], dtype=np.int64),
                    array_col=np.array([0, 2], dtype=np.int64),
                    full_xy=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
                    neighbor_index=neighbor,
                    native_scale=1.0,
                    component_id=np.zeros(2, dtype=np.int32),
                )

            with mock.patch("datasets.pro_normst.load_panel", return_value=panel), mock.patch(
                "datasets.pro_normst.load_split_manifest",
                return_value=(specs, {"protocol": "synthetic"}),
            ), mock.patch("datasets.pro_normst._read_slice", side_effect=fake_read):
                return prepare_pro_normst_data("split.json", "panel.txt")

        first = prepared(200.0)
        second = prepared(2000.0)
        preprocessing_manifest = first.preprocessing.manifest()
        self.assertEqual(
            preprocessing_manifest["schema"], "pro-normst-preprocessing-v2"
        )
        self.assertNotIn("manifest_sha256", preprocessing_manifest)
        self.assertIn("panel_ordered_sha256", preprocessing_manifest)
        self.assertIn("gene_scale_sha256", preprocessing_manifest)
        for name in ("gene_scale", "detection_rate", "positive_weight", "gene_mean_z"):
            np.testing.assert_array_equal(
                getattr(first.preprocessing, name), getattr(second.preprocessing, name)
            )
        np.testing.assert_array_equal(
            first.roles["train"][0].expression_z,
            second.roles["train"][0].expression_z,
        )
        self.assertFalse(
            np.array_equal(
                first.roles["test"][0].expression_z,
                second.roles["test"][0].expression_z,
            )
        )


class DataAndObjectiveContractTest(unittest.TestCase):
    def test_unified_entrypoint_dispatches_to_direct_training(self):
        import train

        with mock.patch("training.pro_normst.main", return_value=0) as task_main:
            result = train.main(
                [
                    "--task",
                    "visium",
                    "--model",
                    "pro-normst",
                    "--manifest",
                    "split.json",
                    "--output-dir",
                    "run",
                ]
            )
        self.assertEqual(result, 0)
        task_main.assert_called_once_with(
            ["--manifest", "split.json", "--output-dir", "run"]
        )

    @staticmethod
    def _contract_fixture():
        return {
            "schema": "pro-normst-training-contract-v3",
            "numerical_implementation_schema": "pro-normst-numerical-v1",
            "model": {
                "schema": "model",
                "variant": "full",
                "max_rounds": 4,
                "n_genes": 512,
            },
            "preprocessing": {
                "panel_size": 512,
                "panel_ordered_sha256": "panel",
                "transform": "log1p(panel_only_cp10k)",
                "scale": "train_slice_balanced_uncentered_rms",
            },
            "mask_schema": "mask",
            "fixed_mask_banks": {"val": {}, "test": {}},
            "slice_data_and_geometry": {"slice": {"role": "train"}},
            "loss_schema": "loss",
            "metric_schema": "metric",
            "optimization": {"max_steps": 3200},
            "split": {
                "protocol": "pair_grouped_random_split",
                "fold": "pilot_seed2027",
            },
            "run": {
                "variant": "full",
                "initialization_seed": 2027,
                "smoke": False,
                "epochs": 50,
                "precision": "cuda-fp16-amp",
                "candidate_lock_sha256": None,
            },
            "runtime": {"torch": "test", "numpy": "test"},
        }

    def test_numerical_hash_ignores_non_numerical_provenance(self):
        contract = self._contract_fixture()
        baseline = _numerical_contract_hash(contract)
        audit_drift = copy.deepcopy(contract)
        audit_drift["runtime"] = {"torch": "new", "numpy": "new"}
        audit_drift["run"]["candidate_lock_sha256"] = "portable-lock"
        self.assertEqual(_numerical_contract_hash(audit_drift), baseline)

        numerical_drift = copy.deepcopy(contract)
        numerical_drift["preprocessing"]["panel_ordered_sha256"] = "different-panel"
        self.assertNotEqual(_numerical_contract_hash(numerical_drift), baseline)

    def test_resume_allows_runtime_provenance_drift_with_warning(self):
        saved_contract = self._contract_fixture()
        current_contract = copy.deepcopy(saved_contract)
        current_contract["runtime"] = {"torch": "new", "numpy": "new"}
        contract_hash = _numerical_contract_hash(current_contract)
        preprocessing = mock.Mock(
            gene_ids=("ENSG000000001",),
            gene_scale=np.asarray([1.0], dtype=np.float32),
            detection_rate=np.asarray([0.5], dtype=np.float32),
            positive_weight=np.asarray([1.0], dtype=np.float32),
            gene_mean_z=np.asarray([0.0], dtype=np.float32),
        )
        data = mock.Mock(preprocessing=preprocessing)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "split.json"
            manifest.write_text(json.dumps({"split": "fixed"}), encoding="utf-8")
            (root / "config.json").write_text(
                json.dumps({"contract_hash": contract_hash}), encoding="utf-8"
            )
            (root / "contract_manifest.json").write_text(
                json.dumps(saved_contract), encoding="utf-8"
            )
            (root / "genes.txt").write_text("ENSG000000001\n", encoding="utf-8")
            np.savez(
                root / "preprocessing.npz",
                gene_scale=preprocessing.gene_scale,
                detection_rate=preprocessing.detection_rate,
                positive_weight=preprocessing.positive_weight,
                gene_mean_z=preprocessing.gene_mean_z,
            )
            (root / "split_manifest.snapshot.json").write_text(
                manifest.read_text(encoding="utf-8"), encoding="utf-8"
            )
            with self.assertWarnsRegex(RuntimeWarning, "audit provenance changed"):
                _validate_existing_run(
                    root,
                    Namespace(manifest=manifest),
                    data,
                    current_contract,
                    contract_hash,
                )

    def test_candidate_lock_is_portable_and_rejects_candidate_mutation(self):
        contract = self._contract_fixture()
        artifacts = {
            "config.json": {
                "protocol": "pair_grouped_random_split",
                "fold": "pilot_seed2027",
                "variant": "full",
                "initialization_seed": 2027,
                "smoke": False,
            },
            "gradient_gate.json": {
                "passed": True,
                "missing_nonzero_gradient": [],
            },
            "final_loss_bptt_gate.json": {
                "passed": True,
                "applicable": True,
                "required_rounds": 4,
                "records": [
                    {
                        "epoch": 1,
                        "global_step": 64,
                        "round_gradient_norm": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
            },
            "pilot_gate.json": {
                "passed": True,
                "families": {
                    family: {
                        "passed": True,
                        "model_smooth_l1": 1.0,
                        "idw_smooth_l1": 1.0,
                        "variance_ratio_median": 1.0,
                    }
                    for family in ("ordinary", "gap")
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in artifacts.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            (root / "best.pt").write_bytes(b"checkpoint")
            candidate_lock = _write_candidate_lock(root, contract)
            payload = _validate_candidate_lock(candidate_lock, contract)
            self.assertEqual(payload["schema"], "pro-normst-candidate-lock-v3")
            self.assertEqual(
                set(payload),
                {
                    "schema",
                    "status",
                    "candidate_signature",
                    "candidate_signature_sha256",
                    "pilot_identity",
                    "gates",
                    "checkpoint_sha256",
                },
            )
            self.assertFalse((root / "test_artifacts").exists())
            self.assertFalse((root / "run_status.json").exists())
            portable_dir = root / "portable"
            portable_dir.mkdir()
            portable_lock = portable_dir / "candidate_lock.json"
            portable_lock.write_bytes(candidate_lock.read_bytes())

            # Validation is self-contained: the original pilot artifacts are audit
            # provenance, not a permanent path dependency.
            (root / "gradient_gate.json").write_text(
                json.dumps({"passed": False}), encoding="utf-8"
            )
            audit_drift = copy.deepcopy(contract)
            audit_drift["runtime"] = {"torch": "updated", "numpy": "updated"}
            _validate_candidate_lock(portable_lock, audit_drift)

            malformed_checkpoint = copy.deepcopy(payload)
            malformed_checkpoint["checkpoint_sha256"] = "not-a-sha256"
            malformed_checkpoint_path = root / "malformed_checkpoint_lock.json"
            malformed_checkpoint_path.write_text(
                json.dumps(malformed_checkpoint), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checkpoint hash"):
                _validate_candidate_lock(malformed_checkpoint_path, contract)

            numerical_drift = copy.deepcopy(contract)
            numerical_drift["model"]["n_genes"] = 256
            with self.assertRaisesRegex(ValueError, "does not match"):
                _validate_candidate_lock(portable_lock, numerical_drift)

            tampered = copy.deepcopy(payload)
            tampered["gates"]["final_loss_bptt"]["witness"][
                "round_gradient_norm"
            ][0] = 0.0
            tampered_path = root / "tampered_candidate_lock.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "BPTT gate"):
                _validate_candidate_lock(tampered_path, contract)

    def test_candidate_lock_rejects_invalid_pretest_gate(self):
        contract = self._contract_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                "config.json": {
                    "protocol": "pair_grouped_random_split",
                    "fold": "pilot_seed2027",
                    "variant": "full",
                    "initialization_seed": 2027,
                    "smoke": False,
                },
                "gradient_gate.json": {
                    "passed": True,
                    "missing_nonzero_gradient": [],
                },
                "final_loss_bptt_gate.json": {
                    "passed": True,
                    "applicable": True,
                    "required_rounds": 4,
                    "records": [
                        {
                            "epoch": 1,
                            "global_step": 64,
                            "round_gradient_norm": [0.0, 0.2, 0.3, 0.4],
                        }
                    ],
                },
                "pilot_gate.json": {
                    "passed": True,
                    "families": {
                        family: {
                            "passed": True,
                            "model_smooth_l1": 1.0,
                            "idw_smooth_l1": 1.0,
                            "variance_ratio_median": 1.0,
                        }
                        for family in ("ordinary", "gap")
                    },
                },
            }
            for name, payload in artifacts.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            (root / "best.pt").write_bytes(b"checkpoint")
            with self.assertRaisesRegex(RuntimeError, "candidate-lock contract"):
                _write_candidate_lock(root, contract)

    def test_validation_summary_omits_per_mask_history_without_changing_gate(self):
        item = mock.Mock(slice_id="slice")
        data = mock.Mock()
        data.roles = {"val": [item]}
        data.preprocessing = mock.Mock(
            gene_scale=np.ones(1, dtype=np.float32),
            positive_weight=np.ones(1, dtype=np.float32),
            detection_rate=np.ones(1, dtype=np.float32),
        )
        mask = mock.Mock()
        banks = {
            "val": {
                "slice": {
                    "ordinary": (mask,),
                    "gap": (mask,),
                }
            }
        }
        metric_record = {
            "mask": {
                "identity": "mask",
                "n_target": 1,
                "n_query": 1,
                "realized_fraction": 0.5,
                "provenance_counts": {"ordinary": 1},
            }
        }
        aggregate = {
            "weighted_z_smooth_l1": 1.0,
            "model": {"smooth_l1": 1.0, "variance_ratio_median": 1.0},
            "idw": {"smooth_l1": 1.0},
        }

        def evaluate(*_args, **_kwargs):
            return copy.deepcopy(metric_record)

        with mock.patch(
            "training.pro_normst.evaluate_mask", side_effect=evaluate
        ), mock.patch(
            "training.pro_normst.aggregate_slice_mask_records",
            return_value=aggregate,
        ), mock.patch(
            "training.pro_normst.mean_slice_summaries", return_value=aggregate
        ):
            full = _evaluate_role(
                mock.Mock(),
                data,
                "val",
                banks,
                torch.device("cpu"),
                use_amp=False,
                detail="full",
            )
            summary = _evaluate_role(
                mock.Mock(),
                data,
                "val",
                banks,
                torch.device("cpu"),
                use_amp=False,
                detail="summary",
            )

        self.assertEqual(
            full["criterion_weighted_z_smooth_l1"],
            summary["criterion_weighted_z_smooth_l1"],
        )
        self.assertEqual(
            full["families"]["ordinary"]["summary"],
            summary["families"]["ordinary"]["summary"],
        )
        self.assertIn("slices", full["families"]["ordinary"])
        self.assertNotIn("slices", summary["families"]["ordinary"])
        self.assertEqual(_pilot_gate(full), _pilot_gate(summary))

    def test_test_artifact_commit_is_atomic_idempotent_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            checkpoint = output_dir / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            staging = output_dir / ".test_artifacts.tmp"
            staging.mkdir()
            (staging / "partial.npz").write_bytes(b"partial")
            model = mock.Mock(variant="one-shot")

            def fake_evaluate(*_args, **kwargs):
                prediction = (
                    kwargs["output_dir"]
                    / "predictions"
                    / kwargs["prediction_label"]
                    / "slice"
                    / "ordinary"
                    / "mask_00.npz"
                )
                prediction.parent.mkdir(parents=True, exist_ok=True)
                np.savez(prediction, prediction_x=np.ones((1, 1), dtype=np.float32))
                return {"role": "test", "round_limit": kwargs["round_limit"]}

            identity = {"slice": "expression-hash"}
            with mock.patch(
                "training.pro_normst.test_expression_identity", return_value=identity
            ), mock.patch(
                "training.pro_normst._evaluate_role", side_effect=fake_evaluate
            ) as evaluator:
                first = _run_test_once(
                    model=model,
                    data=mock.Mock(),
                    banks={},
                    output_dir=output_dir,
                    device=torch.device("cpu"),
                    use_amp=False,
                    checkpoint_path=checkpoint,
                    protocol="pair_grouped_random_split",
                    contract_hash="contract",
                )
                second = _run_test_once(
                    model=model,
                    data=mock.Mock(),
                    banks={},
                    output_dir=output_dir,
                    device=torch.device("cpu"),
                    use_amp=False,
                    checkpoint_path=checkpoint,
                    protocol="pair_grouped_random_split",
                    contract_hash="contract",
                )
                self.assertEqual(evaluator.call_count, 1)
                self.assertEqual(first, second)
                self.assertFalse(staging.exists())

                metrics = output_dir / "test_artifacts" / "test_metrics.json"
                metrics.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "do not match"):
                    _run_test_once(
                        model=model,
                        data=mock.Mock(),
                        banks={},
                        output_dir=output_dir,
                        device=torch.device("cpu"),
                        use_amp=False,
                        checkpoint_path=checkpoint,
                        protocol="pair_grouped_random_split",
                        contract_hash="contract",
                    )
                self.assertEqual(evaluator.call_count, 1)

    def test_complete_test_staging_is_committed_without_reevaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            checkpoint = output_dir / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            staging = output_dir / ".test_artifacts.tmp"
            artifact_dir = output_dir / "test_artifacts"
            model = mock.Mock(variant="one-shot")

            def fake_evaluate(*_args, **kwargs):
                prediction = (
                    kwargs["output_dir"]
                    / "predictions"
                    / kwargs["prediction_label"]
                    / "slice"
                    / "ordinary"
                    / "mask_00.npz"
                )
                prediction.parent.mkdir(parents=True, exist_ok=True)
                np.savez(prediction, prediction_x=np.ones((1, 1), dtype=np.float32))
                return {"role": "test", "round_limit": kwargs["round_limit"]}

            real_replace = os.replace

            def interrupt_final_commit(source, destination):
                if Path(source) == staging and Path(destination) == artifact_dir:
                    raise OSError("interrupted before directory commit")
                return real_replace(source, destination)

            call = dict(
                model=model,
                data=mock.Mock(),
                banks={},
                output_dir=output_dir,
                device=torch.device("cpu"),
                use_amp=False,
                checkpoint_path=checkpoint,
                protocol="pair_grouped_random_split",
                contract_hash="contract",
            )
            with mock.patch(
                "training.pro_normst.test_expression_identity",
                return_value={"slice": "expression-hash"},
            ), mock.patch(
                "training.pro_normst._evaluate_role", side_effect=fake_evaluate
            ) as evaluator:
                with mock.patch(
                    "training.pro_normst.os.replace",
                    side_effect=interrupt_final_commit,
                ):
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        _run_test_once(**call)
                self.assertTrue(staging.is_dir())
                result = _run_test_once(**call)
                self.assertEqual(evaluator.call_count, 1)
                self.assertEqual(result["round1"]["round_limit"], 1)
                self.assertTrue(artifact_dir.is_dir())
                self.assertFalse(staging.exists())

    def test_frozen_panel_hash_and_lodo_manifest(self):
        panel = load_panel(
            SOURCE_ROOT
            / "diagnostics"
            / "dlpfc_151676_shared_panel_20260817"
            / "shared_panel_512_ensembl.txt"
        )
        self.assertEqual(len(panel), 512)
        roles, metadata = load_split_manifest(
            SOURCE_ROOT / "pre-train" / "manifests" / "lodo_d1.json"
        )
        self.assertEqual({key: len(value) for key, value in roles.items()}, {"train": 4, "val": 4, "test": 4})
        self.assertEqual(metadata["held_out_donor"], "Br5292")
        self.assertEqual({item.donor for item in roles["test"]}, {"Br5292"})

    def test_weighted_loss_is_gene_equal_not_query_pooled(self):
        prediction = torch.tensor([[[2.0, 2.0], [0.0, 4.0]]])
        target = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        positive_weight = torch.tensor([3.0, 2.0])
        actual = weighted_gene_smooth_l1(prediction, target, positive_weight)
        element = torch.nn.functional.smooth_l1_loss(
            prediction, target, reduction="none", beta=1.0
        )
        weights = torch.tensor([[[3.0, 1.0], [1.0, 2.0]]])
        expected = ((element * weights).sum(dim=(0, 1)) / weights.sum(dim=(0, 1))).mean()
        torch.testing.assert_close(actual, expected)

    def test_strict_idw_uses_canonical_tie_break_for_six_neighbors(self):
        visible_index = np.arange(7, dtype=np.int64)
        visible_xy = np.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]],
            dtype=np.float32,
        )
        values = visible_index[:, None].astype(np.float32)
        prediction = strict_visible_idw(
            values, visible_xy, visible_index, np.array([[0.0, 0.0]], dtype=np.float32)
        )
        self.assertLess(float(prediction[0, 0]), 3.5)

    def test_contracted_learning_rate_endpoints(self):
        self.assertAlmostEqual(learning_rate_for_step(128), 2e-5)
        self.assertAlmostEqual(learning_rate_for_step(3200), 2e-6)
        self.assertLess(learning_rate_for_step(1), learning_rate_for_step(2))

    def test_checkpoint_round_trip_and_contract_mismatch(self):
        model = ProNORMST(torch.zeros(512), variant="full")
        optimizer, _ = optimizer_for_model(model)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        generator = torch.Generator(device="cpu").manual_seed(2027)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            data_order_generator=generator,
            contract_hash="contract",
            epoch=1,
            global_step=1,
            best_value=1.0,
            best_epoch=1,
            bad_epochs=0,
            history=[],
            gradient_cache={"1": 1.0},
            gradient_seen={name: True for name in model.trainable_parameter_names()},
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save(payload, checkpoint)
            restored = ProNORMST(torch.ones(512), variant="full")
            _load_checkpoint(
                checkpoint, restored, "contract", torch.device("cpu")
            )
            for left, right in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right)
            with self.assertRaisesRegex(ValueError, "contract hash"):
                _load_checkpoint(
                    checkpoint, restored, "different", torch.device("cpu")
                )


if __name__ == "__main__":
    unittest.main()
