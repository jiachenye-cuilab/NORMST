"""Synthetic smoke tests for the unified training entry."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from datasets.paired_visium_hd import (
    JointPairedVisiumHDDataset,
    VisiumHDPair,
)
from models.geometry_adaptive_normst import (
    VisiumHDNORMST,
    VisiumNORMST,
    build_native_hex_neighbors,
)
from train_geometry_adaptive_normst import main, run_epoch


def artificial_hex():
    root3 = 3.0 ** 0.5
    xy = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [0.5, root3 / 2.0],
        [-0.5, root3 / 2.0], [-1.0, 0.0],
        [-0.5, -root3 / 2.0], [0.5, -root3 / 2.0],
    ])
    rows = torch.tensor([1, 1, 1, 0, 0, 2, 2])
    cols = torch.tensor([2, 0, 4, 1, 3, 1, 3])
    return xy, build_native_hex_neighbors(rows, cols)


class JointHDAdapterTest(unittest.TestCase):
    def test_all_gene_patch_shapes_and_scales(self):
        genes = np.asarray(["g0", "g1", "g2"])
        lr_counts = sp.csc_matrix(np.arange(12, dtype=np.float32).reshape(4, 3))
        hr_counts = sp.csc_matrix(np.arange(48, dtype=np.float32).reshape(16, 3))
        pair = VisiumHDPair(
            lr_matrix=lr_counts,
            hr_matrix=hr_counts,
            lr_library=np.full(4, 100.0, dtype=np.float32),
            hr_library=np.full(16, 100.0, dtype=np.float32),
            lr_row_map=np.arange(4, dtype=np.int32).reshape(2, 2),
            hr_row_map=np.arange(16, dtype=np.int32).reshape(4, 4),
            genes=genes,
            lr_gene_scale=np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
            hr_gene_scale=np.asarray([2.0, 4.0, 8.0], dtype=np.float32),
            split_ranges={"train": (0, 2), "val": (0, 2), "test": (0, 2)},
            split_axis="col",
            scale=2,
            target_sum=1e4,
            lr_context=np.empty((4, 0), dtype=np.float32),
            context_mean=np.empty((0,), dtype=np.float32),
            context_components=np.empty((0, 3), dtype=np.float32),
            context_scale=np.empty((0,), dtype=np.float32),
            context_explained_variance_ratio=np.empty((0,), dtype=np.float32),
        )
        dataset = JointPairedVisiumHDDataset(
            pair,
            "train",
            patch_size_lr=(2, 2),
            repeat=1,
            min_tissue_fraction=0.0,
            deterministic=True,
            origin_stride=2,
        )
        item = dataset[0]
        self.assertEqual(item["inp"].shape, (3, 2, 2))
        self.assertEqual(item["gt"].shape, (3, 4, 4))
        self.assertEqual(item["input_mask"].shape, (1, 2, 2))
        self.assertEqual(item["target_mask"].shape, (1, 4, 4))
        torch.testing.assert_close(
            item["baseline_scale"], torch.full((3,), 0.5)
        )


class TrainingStepTest(unittest.TestCase):
    def test_loss_only_metrics_preserve_parameter_update(self):
        batches = [{
            "features": torch.tensor([[1.0], [2.0], [3.0]]),
            "target": torch.tensor([[0.5], [1.0], [1.5]]),
            "mask": torch.ones(3, 1, dtype=torch.bool),
        }]

        def fake_prediction(model, batch, _neighbor, _xy):
            prediction = model(batch["features"])
            return prediction, batch["target"], batch["mask"], torch.zeros_like(
                prediction
            )

        full_model = torch.nn.Linear(1, 1)
        loss_only_model = torch.nn.Linear(1, 1)
        loss_only_model.load_state_dict(full_model.state_dict())
        full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
        loss_only_optimizer = torch.optim.SGD(
            loss_only_model.parameters(), lr=0.1,
        )
        full_scaler = torch.amp.GradScaler("cpu", enabled=False)
        loss_only_scaler = torch.amp.GradScaler("cpu", enabled=False)
        with patch(
            "train_geometry_adaptive_normst.visium_prediction",
            side_effect=fake_prediction,
        ):
            full_metrics = run_epoch(
                "visium", full_model, DataLoader(batches, batch_size=1),
                torch.device("cpu"), optimizer=full_optimizer,
                scaler=full_scaler, use_amp=False,
            )
            loss_only_metrics = run_epoch(
                "visium", loss_only_model, DataLoader(batches, batch_size=1),
                torch.device("cpu"), optimizer=loss_only_optimizer,
                scaler=loss_only_scaler, use_amp=False,
                detailed_metrics=False,
            )
        self.assertAlmostEqual(
            full_metrics["loss"], loss_only_metrics["loss"], places=6,
        )
        for full, loss_only in zip(
            full_model.parameters(), loss_only_model.parameters(),
        ):
            torch.testing.assert_close(full, loss_only, rtol=0.0, atol=0.0)

    def test_epoch_metrics_pool_elements_instead_of_averaging_batches(self):
        batches = [
            {
                "prediction": torch.tensor([[0.0]]),
                "target": torch.tensor([[1.0]]),
                "mask": torch.ones(1, 1, dtype=torch.bool),
            },
            {
                "prediction": torch.tensor([[3.0], [3.0], [3.0]]),
                "target": torch.tensor([[0.0], [0.0], [0.0]]),
                "mask": torch.ones(3, 1, dtype=torch.bool),
            },
        ]

        def fake_prediction(_model, batch, _neighbor, _xy):
            prediction = batch["prediction"]
            return prediction, batch["target"], batch["mask"], torch.zeros_like(
                prediction
            )

        with patch(
            "train_geometry_adaptive_normst.visium_prediction",
            side_effect=fake_prediction,
        ):
            metrics = run_epoch(
                "visium", torch.nn.Identity(),
                DataLoader(batches, batch_size=1), torch.device("cpu"),
                use_amp=False,
            )
        self.assertAlmostEqual(metrics["loss"], 2.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(7.0))
        self.assertAlmostEqual(metrics["mae"], 2.5)
        self.assertAlmostEqual(metrics["positive_rmse"], 1.0)
        self.assertAlmostEqual(metrics["positive_mae"], 1.0)
        self.assertEqual(metrics["positive_count"], 1)
        self.assertEqual(metrics["element_count"], 4)

        with patch(
            "train_geometry_adaptive_normst.visium_prediction",
            side_effect=fake_prediction,
        ):
            loss_only = run_epoch(
                "visium", torch.nn.Identity(),
                DataLoader(batches, batch_size=1), torch.device("cpu"),
                use_amp=False, detailed_metrics=False,
            )
        self.assertEqual(set(loss_only), {"loss"})
        self.assertAlmostEqual(loss_only["loss"], metrics["loss"])

    def test_visium_and_hd_one_optimizer_step(self):
        torch.manual_seed(11)
        device = torch.device("cpu")
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        xy, full_neighbor = artificial_hex()
        visible_spots = torch.arange(1, 7)
        visium_batch = {
            "visible_expression": torch.randn(6, 3),
            "visible_coord": xy[visible_spots],
            "query_coord": xy[:1],
            "target_values": torch.randn(1, 3),
            "target_spots": torch.tensor([0]),
            "visible_spots": visible_spots,
            "baseline": torch.zeros(1, 3),
        }
        visium = VisiumNORMST(
            n_genes=3, width=8, num_heads=2, num_layers=1
        )
        visium_optimizer = torch.optim.Adam(visium.parameters(), lr=1e-3)
        visium_metrics = run_epoch(
            "visium",
            visium,
            DataLoader([visium_batch], batch_size=1),
            device,
            optimizer=visium_optimizer,
            scaler=scaler,
            use_amp=False,
            full_neighbor=full_neighbor,
            full_xy=xy,
        )
        self.assertTrue(np.isfinite(visium_metrics["loss"]))

        hd_batch = {
            "inp": torch.randn(2, 2, 2),
            "input_mask": torch.ones(1, 2, 2),
            "gt": torch.randn(2, 4, 4),
            "target_mask": torch.ones(1, 4, 4),
            "baseline_scale": torch.ones(2),
            "origin": torch.zeros(2, dtype=torch.long),
        }
        hd = VisiumHDNORMST(
            n_genes=2, width=8, num_heads=2, num_layers=1
        )
        hd_optimizer = torch.optim.Adam(hd.parameters(), lr=1e-3)
        hd_metrics = run_epoch(
            "visium_hd",
            hd,
            DataLoader([hd_batch], batch_size=1),
            device,
            optimizer=hd_optimizer,
            scaler=scaler,
            use_amp=False,
        )
        self.assertTrue(np.isfinite(hd_metrics["loss"]))

    def test_visium_main_writes_complete_training_artifacts(self):
        xy, full_neighbor = artificial_hex()
        visible_spots = torch.arange(1, 7)
        batch = {
            "visible_expression": torch.randn(6, 3),
            "visible_coord": xy[visible_spots],
            "query_coord": xy[:1],
            "target_values": torch.randn(1, 3),
            "target_spots": torch.tensor([0]),
            "visible_spots": visible_spots,
            "baseline": torch.zeros(1, 3),
        }
        prepared = (
            VisiumNORMST(n_genes=3, width=8, num_heads=2, num_layers=1),
            {"train": [batch], "val": [batch], "test": [batch]},
            {
                "genes": np.asarray(["g0", "g1", "g2"]),
                "test_spots": np.asarray([0]),
            },
            {"spatial_representation": "synthetic_test"},
            full_neighbor,
            xy,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with patch(
                "train_geometry_adaptive_normst.prepare_visium",
                return_value=prepared,
            ):
                main([
                    "--task", "visium",
                    "--data-dir", "unused",
                    "--output-dir", str(output),
                    "--n-genes", "3",
                    "--width", "8",
                    "--num-heads", "2",
                    "--operator-layers", "1",
                    "--epochs", "1",
                    "--masks-per-epoch", "1",
                    "--device", "cpu",
                    "--no-amp",
                ])
            expected = {
                "config.json", "history.json", "best.pt", "last.pt",
                "test_metrics.json", "test_predictions.npz",
                "preprocessing.npz", "genes.txt",
            }
            self.assertEqual(
                expected, {path.name for path in output.iterdir()}
            )
            history = json.loads(
                (output / "history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(history[0]["learning_rate"], 2e-5)


if __name__ == "__main__":
    unittest.main()
