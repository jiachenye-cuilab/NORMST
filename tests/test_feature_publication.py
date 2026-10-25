"""Feature extraction must sync writable files before publishing completion."""

import errno
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from normst.data import load_image_features, manifest, read_json, write_json
from normst.features import extract


class FeaturePublicationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory(prefix="normst-feature-publication-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def _inputs(self, name):
        import tifffile

        folder = self.folder / name
        folder.mkdir()
        np.save(folder / "feature_valid.npy", np.ones((8, 8), dtype=bool))
        tifffile.imwrite(folder / "he.tiff", np.zeros((128, 128, 3), dtype=np.uint8), photometric="rgb")
        weights = folder / "toy-weights.bin"
        weights.write_bytes(b"synthetic test double")
        dataset = folder / "dataset.json"
        write_json(dataset, dict(features="features.npy", image="he.tiff", feature_valid="feature_valid.npy",
                                 bounds_yxyx_um=[0, 0, 64, 64]))
        return dataset, weights

    def _extract_checked(self, dataset, weights, events, fail_on=None):
        spec = manifest(dataset)
        destination = spec["features"]
        normalization = destination.with_suffix(".normalization.npz")
        marker = destination.with_suffix(".json")
        real_open, real_sync, real_replace = Path.open, os.fsync, os.replace
        opened = []

        class Backbone:
            def forward_features(self, batch):
                return torch.zeros(len(batch), 261, 1280, device=batch.device)

        def opened_path(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            opened.append((Path(path), stream))
            return stream

        def sync(descriptor):
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                # Enforce Windows' writable-handle requirement on any host.
                # The zero-byte write leaves file contents and offsets unchanged.
                os.write(descriptor, b"")
                for path, stream in opened:
                    if not stream.closed and stream.fileno() == descriptor:
                        if path.name in ("features.npy", "normalization.npz"):
                            self.assertTrue(stream.writable())
                            events.append(("sync", path.name))
                            if fail_on and path.name == fail_on[0]:
                                raise fail_on[1]
                        break
            real_sync(descriptor)

        def replace(source, target):
            source, target = Path(source), Path(target)
            if target in (destination, normalization):
                self.assertEqual(events[:2], [("sync", "features.npy"), ("sync", "normalization.npz")])
                self.assertEqual(source.parent.parent, target.parent)
                self.assertTrue(all(stream.closed for path, stream in opened if path == source))
                events.append(("replace", target.name))
            elif target == marker and json.loads(source.read_text(encoding="utf-8"))["status"] == "complete":
                self.assertTrue(destination.is_file() and normalization.is_file())
                events.append(("complete", target.name))
            real_replace(source, target)

        with patch("normst.features.load_backbone", return_value=Backbone()), \
                patch.object(Path, "open", opened_path), \
                patch("normst.features.os.fsync", sync), \
                patch("normst.features.os.replace", replace):
            return extract(dataset, weights, device="cpu")

    def test_extract_syncs_both_writable_files_before_replacement_and_completion(self):
        dataset, weights = self._inputs("complete")
        events = []
        destination = self._extract_checked(dataset, weights, events)
        self.assertEqual(events, [("sync", "features.npy"), ("sync", "normalization.npz"),
                                 ("replace", "features.normalization.npz"), ("replace", "features.npy"),
                                 ("complete", "features.json")])
        self.assertEqual(read_json(destination.with_suffix(".json"))["status"], "complete")
        features, _ = load_image_features(manifest(dataset))
        self.assertEqual(tuple(features.shape), (1, 2563, 8, 8))
        self.assertTrue(torch.isfinite(features).all())

    def test_either_feature_file_sync_failure_prevents_publication_and_can_retry(self):
        for name in ("features.npy", "normalization.npz"):
            with self.subTest(file=name):
                dataset, weights = self._inputs(name)
                spec = manifest(dataset)
                destination = spec["features"]
                events = []
                error = OSError(errno.EIO, "injected feature file sync failure")
                with self.assertRaises(OSError) as caught:
                    self._extract_checked(dataset, weights, events, fail_on=(name, error))
                self.assertIs(caught.exception, error)
                self.assertFalse(destination.exists())
                self.assertFalse(destination.with_suffix(".normalization.npz").exists())
                self.assertEqual(read_json(destination.with_suffix(".json"))["status"], "failed")
                self.assertTrue(all(event[0] == "sync" for event in events))
                self.assertEqual(list(destination.parent.glob(".features.npy.partial-*")), [])
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    load_image_features(spec)
                self._extract_checked(dataset, weights, [])
                self.assertEqual(read_json(destination.with_suffix(".json"))["status"], "complete")
                load_image_features(spec)


if __name__ == "__main__":
    unittest.main()
