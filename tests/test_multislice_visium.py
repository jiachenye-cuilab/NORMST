"""Synthetic tests for leakage-safe standard Visium multi-slice training."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import scipy.sparse as sp

from datasets.multislice_masked_visium import (
    MultiSlicePointDataset,
    prepare_multislice_visium,
)
from train_multislice_visium import main


ROWS = np.asarray([1, 1, 1, 0, 0, 2, 2], dtype=np.int64)
COLS = np.asarray([2, 0, 4, 1, 3, 1, 3], dtype=np.int64)


def make_adata(offset: float):
    counts = np.arange(1, 22, dtype=np.float32).reshape(7, 3) + offset
    result = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs={
            "array_row": ROWS,
            "array_col": COLS,
            "pxl_row_in_fullres": ROWS.astype(np.float32),
            "pxl_col_in_fullres": COLS.astype(np.float32),
        },
    )
    result.obs_names = [f"spot_{index}" for index in range(7)]
    result.var_names = ["g0", "g1", "g2"]
    return result


class MultiSlicePreparationTest(unittest.TestCase):
    def _prepare(self, directory: Path):
        manifest = {
            "train": {"train_a": "train_a", "train_b": "train_b"},
            "val": {"val_a": "val_a"},
            "test": {"test_a": "test_a"},
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        offsets = {
            "train_a": 0.0,
            "train_b": 3.0,
            # Large held-out-slice offsets make leakage into RMS easy to detect.
            "val_a": 1000.0,
            "test_a": 2000.0,
        }

        def fake_reader(data_dir, _count_file):
            return make_adata(offsets[Path(data_dir).name])

        with patch(
            "datasets.multislice_masked_visium._read_standard_visium",
            side_effect=fake_reader,
        ):
            prepared = prepare_multislice_visium(
                str(manifest_path), n_genes=3, target_sum=1.0,
                observed_fraction=0.5, seed=7,
            )
        return manifest_path, prepared

    def test_shared_scale_uses_only_observed_training_spots(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, prepared = self._prepare(Path(temporary))
        expected_rows = []
        observed = np.asarray([0, 3, 5])
        for offset in (0.0, 3.0):
            counts = np.arange(1, 22, dtype=np.float32).reshape(7, 3) + offset
            library = counts.sum(axis=1)
            normalized = np.log1p(counts / library[:, None])
            expected_rows.append(normalized[observed])
        expected_scale = np.sqrt(np.mean(np.vstack(expected_rows) ** 2, axis=0))
        np.testing.assert_allclose(
            prepared.gene_scale, expected_scale, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(prepared.genes, ["g0", "g1", "g2"])

    def test_balanced_dataset_and_complete_entry_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, prepared = self._prepare(root)
            train = MultiSlicePointDataset(
                prepared, "train", masks_per_slice=2,
                train_target_fraction=0.25, idw_neighbors=2, seed=9,
            )
            self.assertEqual(len(train), 4)
            self.assertEqual(int(train[0]["slice_index"]), 0)
            self.assertEqual(int(train[2]["slice_index"]), 1)
            for index in (0, 2):
                item = train[index]
                self.assertTrue(set(item["visible_spots"].tolist()).isdisjoint(
                    item["target_spots"].tolist()
                ))

            output = root / "output"
            with patch(
                "train_multislice_visium.prepare_multislice_visium",
                return_value=prepared,
            ):
                main([
                    "--manifest", str(manifest_path),
                    "--output-dir", str(output),
                    "--n-genes", "3",
                    "--masks-per-slice", "1",
                    "--query-neighbors", "2",
                    "--width", "8",
                    "--num-heads", "2",
                    "--operator-layers", "1",
                    "--epochs", "1",
                    "--device", "cpu",
                    "--no-amp",
                ])
            expected_files = {
                "config.json", "manifest.json", "genes.txt",
                "preprocessing.npz", "history.json", "best.pt", "last.pt",
                "test_metrics.json", "test_predictions_index.json",
                "preprocessing_slices", "test_predictions",
            }
            self.assertEqual(expected_files, {item.name for item in output.iterdir()})
            metrics = json.loads(
                (output / "test_metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("macro", metrics)
            self.assertEqual(set(metrics["per_slice"]), {"test_a"})
            prediction_files = list((output / "test_predictions").glob("*.npz"))
            self.assertEqual(len(prediction_files), 1)


if __name__ == "__main__":
    unittest.main()
