"""Check registration, gene alignment, token coordinates, and image storage."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import torch

from normst.data import load_image_features, load_observations, manifest
from normst.features import extract, token_maps
from normst.memory import compact_features
from normst.prepare import prepare_visium


class InputTests(unittest.TestCase):
    def test_token_cell_centers_and_register_exclusion(self):
        axis = (torch.arange(16) + .5) * 7
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        tokens = torch.zeros(1, 261, 1280)
        tokens[:, 0] = 3
        tokens[:, 1:5] = 1e6
        tokens[:, 5:, :2] = torch.stack((y, x), -1).reshape(256, 2)
        maps = token_maps(tokens)
        expected = (torch.arange(14) + .5) * 8
        yy, xx = torch.meshgrid(expected, expected, indexing="ij")
        torch.testing.assert_close(maps[0, 1280:1282], torch.stack((yy, xx)), rtol=2e-7, atol=1e-5)
        self.assertTrue(torch.equal(maps[0, :1280], torch.full((1280, 14, 14), 3.)))

    def test_spaceranger_to_normalized_features(self):
        try:
            import h5py
            import tifffile
        except ImportError:
            self.skipTest("install the histology extra to check Space Ranger inputs")
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="normst-input-") as temporary:
            folder = Path(temporary)
            arrays = np.array([[row, 2 * col + row % 2] for row in range(3) for col in range(3)])
            centers = 112 + np.column_stack((arrays[:, 0] * 50 * np.sqrt(3), arrays[:, 1] * 50))
            image_points = centers * 2
            counts = np.arange(1, 73).reshape(9, 8).astype(np.int32)
            matrix = sparse.csc_matrix(counts.T)
            with h5py.File(folder / "matrix.h5", "w") as handle:
                group = handle.create_group("matrix")
                for key in ("data", "indices", "indptr"):
                    group[key] = getattr(matrix, key)
                group["shape"] = matrix.shape
                group["barcodes"] = np.array([f"spot{i}".encode() for i in range(9)])
                features = group.create_group("features")
                features["id"] = np.array([f"g{i}".encode() for i in range(8)])
                features["name"] = np.array([f"G{i}".encode() for i in range(8)])
                features["feature_type"] = np.array([b"Gene Expression"] * 7 + [b"Antibody Capture"])
            lines = ["barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres"]
            for i, (array, pixel) in enumerate(zip(arrays, image_points)):
                lines.append(f"spot{i},1,{array[0]},{array[1]},{pixel[0]},{pixel[1]}")
            (folder / "positions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
            y, x = np.indices((1024, 1024))
            rgb = np.stack((y // 4, x // 4, (y + x) // 8), -1).astype(np.uint8)
            tifffile.imwrite(folder / "image.tiff", rgb, photometric="rgb")
            (folder / "genes.txt").write_text("g5\ng1\ng3\n", encoding="utf-8")
            dataset = prepare_visium(folder / "matrix.h5", folder / "positions.csv", folder / "image.tiff",
                                      folder / "prepared", gene_list=folder / "genes.txt")
            spec = manifest(dataset)
            data = load_observations(spec)
            self.assertEqual(data["gene_ids"], ["g5", "g1", "g3"])
            np.testing.assert_array_equal(data["counts"], counts[:, [5, 1, 3]])
            np.testing.assert_allclose(data["centers"], centers, rtol=0, atol=1e-10)
            image = tifffile.memmap(spec["image"], mode="r")
            pixel = np.rint((centers[4] - data["bounds"][:2]) * 2 - .5).astype(int)
            expected = rgb[tuple(np.rint(image_points[4]).astype(int))]
            np.testing.assert_allclose(image[tuple(pixel)], expected, rtol=0, atol=1)

            class ToyBackbone:
                def forward_features(self, batch):
                    # Known finite token values isolate layout/normalization from downloaded weights.
                    base = torch.arange(261, dtype=batch.dtype, device=batch.device)[None, :, None]
                    scale = torch.arange(1280, dtype=batch.dtype, device=batch.device)[None, None, :] / 1280
                    return (base + scale + batch.mean((1, 2, 3))[:, None, None]).expand(len(batch), -1, -1)

            with patch("normst.features.load_backbone", return_value=ToyBackbone()):
                extract(dataset, folder / "unused.safetensors", device="cpu", batch_tiles=4)
            features, valid = load_image_features(spec)
            self.assertEqual(features.shape[1], 2563)
            selected = features[0, :, valid[0, 0]].numpy()
            np.testing.assert_allclose(selected.mean(1), 0, rtol=0, atol=2e-4)
            np.testing.assert_allclose(selected.std(1), 1, rtol=0, atol=2e-4)
            cls, local, indices = compact_features(features, valid, dict(dimension=1280, tile_cells=14))
            restored = torch.cat((cls[:, indices] * valid[0, 0].flatten(), local[0].flatten(1)), 0)
            torch.testing.assert_close(restored, features[0].flatten(1), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
