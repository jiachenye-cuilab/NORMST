"""Synthetic tests for leakage-safe standard Visium multi-slice training."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import scipy.sparse as sp

from datasets.multislice_masked_visium import (
    MultiSlicePointDataset,
    prepare_multislice_visium,
)
from train_multislice_visium import (
    build_random_manifest,
    discover_visium_slices,
    main,
    ratio_4_1_1_counts,
    resolve_run_manifest,
)


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


def make_slice_directory(root: Path, name: str, complete=True):
    directory = root / name
    spatial = directory / "spatial"
    spatial.mkdir(parents=True)
    if complete:
        (directory / "filtered_feature_bc_matrix.h5").touch()
        (spatial / "tissue_positions.csv").touch()
    return directory


class MultiSlicePreparationTest(unittest.TestCase):
    def test_training_entry_discovers_and_saves_reproducible_4_1_1_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "visium"
            root.mkdir()
            for index in range(12):
                make_slice_directory(root, f"slice_{index:02d}")
            make_slice_directory(root, "incomplete", complete=False)

            slices, skipped = discover_visium_slices(
                root, "filtered_feature_bc_matrix.h5"
            )
            self.assertEqual(len(slices), 12)
            self.assertEqual(len(skipped), 1)
            self.assertEqual(ratio_4_1_1_counts(12), (8, 2, 2))
            first = build_random_manifest(
                slices, 17, root, "filtered_feature_bc_matrix.h5"
            )
            second = build_random_manifest(
                slices, 17, root, "filtered_feature_bc_matrix.h5"
            )
            self.assertEqual(first, second)

            output_dir = Path(temporary) / "output"
            output_dir.mkdir()
            resolved = resolve_run_manifest(SimpleNamespace(
                output_dir=output_dir,
                manifest=None,
                visium_root=root,
                count_file="filtered_feature_bc_matrix.h5",
                seed=17,
            ))
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            self.assertEqual(payload["_meta"]["counts"], {
                "train": 8, "val": 2, "test": 2,
            })
            assigned = [
                path for role in ("train", "val", "test")
                for path in payload[role].values()
            ]
            self.assertEqual(len(assigned), len(set(assigned)))

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
                seed=7,
            )
        return manifest_path, prepared

    def test_shared_scale_uses_all_spots_from_training_slices_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, prepared = self._prepare(Path(temporary))
        expected_rows = []
        for offset in (0.0, 3.0):
            counts = np.arange(1, 22, dtype=np.float32).reshape(7, 3) + offset
            library = counts.sum(axis=1)
            normalized = np.log1p(counts / library[:, None])
            expected_rows.append(normalized)
        expected_scale = np.sqrt(np.mean(np.vstack(expected_rows) ** 2, axis=0))
        np.testing.assert_allclose(
            prepared.gene_scale, expected_scale, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(prepared.genes, ["g0", "g1", "g2"])

    def test_hvg_selection_is_batch_aware_and_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "train": {"train_a": "train_a", "train_b": "train_b"},
                "val": {"val_a": "val_a"},
                "test": {"test_a": "test_a"},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            def fake_reader(data_dir, _count_file):
                return make_adata({
                    "train_a": 0.0, "train_b": 3.0,
                    "val_a": 1000.0, "test_a": 2000.0,
                }[Path(data_dir).name])

            def fake_hvg(adata, **_kwargs):
                adata.var["highly_variable"] = [True, True, False]

            with patch(
                "datasets.multislice_masked_visium._read_standard_visium",
                side_effect=fake_reader,
            ), patch(
                "datasets.multislice_masked_visium.sc.pp.highly_variable_genes",
                side_effect=fake_hvg,
            ) as hvg:
                prepared = prepare_multislice_visium(
                    str(manifest_path), n_genes=2, target_sum=1.0, seed=7,
                )
            self.assertEqual(hvg.call_args.kwargs["flavor"], "seurat_v3_paper")
            self.assertEqual(hvg.call_args.kwargs["batch_key"], "slice_id")
            self.assertEqual(
                set(hvg.call_args.args[0].obs["slice_id"]),
                {"train_a", "train_b"},
            )
            np.testing.assert_array_equal(prepared.genes, ["g0", "g1"])

    def test_balanced_dataset_and_complete_entry_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, prepared = self._prepare(root)
            train = MultiSlicePointDataset(
                prepared, "train", masks_per_slice=2,
                mask_target_fraction=0.25, idw_neighbors=2, seed=9,
            )
            self.assertEqual(len(train), 4)
            self.assertEqual(int(train[0]["slice_index"]), 0)
            self.assertEqual(int(train[2]["slice_index"]), 1)
            for index in (0, 2):
                item = train[index]
                self.assertTrue(set(item["visible_spots"].tolist()).isdisjoint(
                    item["target_spots"].tolist()
                ))
                self.assertEqual(
                    set(item["visible_spots"].tolist())
                    | set(item["target_spots"].tolist()),
                    set(range(7)),
                )

            validation = MultiSlicePointDataset(
                prepared, "val", masks_per_slice=2,
                mask_target_fraction=0.25, idw_neighbors=2, seed=9,
            )
            self.assertEqual(len(validation), 2)
            self.assertEqual(int(validation[0]["slice_index"]), 2)
            first_validation_targets = validation[0]["target_spots"].clone()
            np.testing.assert_array_equal(
                first_validation_targets, validation[0]["target_spots"]
            )

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
