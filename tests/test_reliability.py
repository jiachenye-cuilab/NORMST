"""Failure recovery, input contracts, and seeded compatibility on small CPU inputs."""

import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import torch

from examples.synthetic import make_inputs
from normst.artifacts import save_checkpoint
from normst.conditioning import condition_with_prior
from normst.data import load_image_features, load_settings, manifest, read_json, write_json
from normst.features import extract, validate_external_features
from normst.model import NORMST
from normst.predict import predict
from normst.prepare import read_spaceranger
from normst.train import fit


class InputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="normst-safety-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_nested_configuration_merge_and_validation(self):
        path = self.folder / "settings.json"
        write_json(path, {"conditioning": {"neighbors": 3}})
        settings = load_settings(path)
        expected = load_settings()
        expected["conditioning"]["neighbors"] = 3
        self.assertEqual(settings, expected)
        invalid = ({"conditioning": None}, {"conditioning": {"neighbors": 0}},
            {"conditioning": {"neighbors": 1.5}}, {"conditioning": {"unknown": 1}},
            {"conditioning": {"kernel_um": "100"}}, {"epochs": True}, {"epochs": 0},
            {"minimum_learning_rate": 1}, {"weight_decay": -1}, {"offsets_um": []},
            {"offsets_um": [[0, 200]]}, {"schedule_horizon_epochs": 5},
            {"gradient_clip": "5"}, {"seed": 2**32}, {"query_chunk": False})
        for override in invalid:
            with self.subTest(override=override):
                write_json(path, override)
                with self.assertRaises(ValueError):
                    load_settings(path)
        path.write_text('{"learning_rate": NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_settings(path)

    def test_incomplete_features_and_changed_content_are_rejected(self):
        dataset = make_inputs(self.folder / "data")
        spec = manifest(dataset)
        marker = spec["features"].with_suffix(".json")
        marker.unlink()
        # A fully allocated, finite zero-filled file can still be unfinished.
        np.save(spec["features"], np.zeros((7, 40, 40), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "completion"):
            load_image_features(spec)
        with self.assertRaisesRegex(ValueError, "confirm_complete"):
            validate_external_features(dataset)
        validate_external_features(dataset, confirm_complete=True)
        load_image_features(spec)
        array = np.load(spec["features"], mmap_mode="r+")
        array[0, 0, 0] = 1
        array.flush()
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_image_features(spec)
        with self.assertRaises(ValueError):
            fit(dataset, self.folder / "model", device="cpu")
        self.assertFalse((self.folder / "model").exists())
        validate_external_features(dataset, confirm_complete=True)
        array[0, 0, 0] = np.nan
        array.flush()
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_external_features(dataset, confirm_complete=True)
        del array
        with spec["features"].open("r+b") as stream:
            stream.truncate(spec["features"].stat().st_size - 4)
        with self.assertRaisesRegex(ValueError, "truncated"):
            load_image_features(spec)

    def test_feature_mask_and_geometry_are_part_of_completion(self):
        dataset = make_inputs(self.folder / "data")
        spec = manifest(dataset)
        array = np.load(spec["features"], mmap_mode="r+")
        array[:, 0, 0] = 0
        array.flush()
        validate_external_features(dataset, confirm_complete=True)
        valid = np.load(spec["feature_valid"])
        valid[0, 0] = False
        np.save(spec["feature_valid"], valid)
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_image_features(spec)
        validate_external_features(dataset, confirm_complete=True)
        load_image_features(spec)
        spec["bounds_yxyx_um"] = [2, 2, 322, 322]
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_image_features(spec)

    def test_external_arrays_can_keep_metadata_in_a_separate_writable_directory(self):
        arrays = self.folder / "arrays"
        arrays.mkdir()
        np.save(arrays / "precomputed.npy", np.zeros((7, 8, 8), dtype=np.float32))
        np.save(arrays / "mask.npy", np.ones((8, 8), dtype=bool))
        task = self.folder / "task"
        task.mkdir()
        dataset = task / "dataset.json"
        write_json(dataset, dict(features="../arrays/precomputed.npy", feature_valid="../arrays/mask.npy",
            feature_metadata="completion.json", bounds_yxyx_um=[0, 0, 64, 64]))
        before = sorted(path.name for path in arrays.iterdir())
        validate_external_features(dataset, confirm_complete=True)
        load_image_features(manifest(dataset))
        self.assertEqual(sorted(path.name for path in arrays.iterdir()), before)
        self.assertTrue((task / "completion.json").exists())

    def _spaceranger(self):
        import h5py

        matrix = sparse.csc_matrix(np.arange(1, 13).reshape(4, 3).T)
        path = self.folder / "matrix.h5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("matrix")
            for key in ("data", "indices", "indptr"):
                group[key] = getattr(matrix, key)
            group["shape"] = matrix.shape
            group["barcodes"] = np.array([b"a", b"b", b"c", b"d"])
            features = group.create_group("features")
            features["id"] = np.array([b"g0", b"g1", b"g2"])
            features["name"] = np.array([b"G0", b"G1", b"G2"])
        positions = self.folder / "positions.csv"
        # Deliberately shuffled; output must follow the expression matrix.
        positions.write_text("barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
            "d,1,2,2,40,41\nb,1,0,2,20,21\na,1,0,0,10,11\nc,1,2,0,30,31\n", encoding="utf-8")
        return path, positions

    def test_barcode_order_and_duplicate_positions(self):
        matrix, positions = self._spaceranger()
        counts, _, _, barcodes, _, pixels = read_spaceranger(matrix, positions)
        np.testing.assert_array_equal(barcodes, ["a", "b", "c", "d"])
        np.testing.assert_array_equal(counts.toarray(), np.arange(1, 13).reshape(4, 3))
        np.testing.assert_array_equal(pixels[:, 0], [10, 20, 30, 40])
        with positions.open("a", encoding="utf-8") as stream:
            stream.write("a,1,0,0,10,11\n")
        with self.assertRaisesRegex(ValueError, "position table barcodes: 1 duplicate IDs"):
            read_spaceranger(matrix, positions)

    def test_duplicate_matrix_barcodes(self):
        import h5py

        matrix, positions = self._spaceranger()
        with h5py.File(matrix, "r+") as handle:
            handle["matrix/barcodes"][1] = b"a"
        with self.assertRaisesRegex(ValueError, "expression matrix barcodes: 1 duplicate IDs"):
            read_spaceranger(matrix, positions)

    def test_missing_positions_require_explicit_exclusion(self):
        matrix, positions = self._spaceranger()
        positions.write_text(positions.read_text(encoding="utf-8").replace("d,1,2,2,40,41\n", ""), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "1 of 4 expression barcodes lack positions"):
            read_spaceranger(matrix, positions)
        with self.assertWarnsRegex(UserWarning, "explicitly excluded"):
            counts, _, _, barcodes, _, _ = read_spaceranger(matrix, positions, skip_missing_positions=True)
        np.testing.assert_array_equal(barcodes, ["a", "b", "c"])
        np.testing.assert_array_equal(counts.toarray(), np.arange(1, 10).reshape(3, 3))

    def test_extraction_and_normalization_failures_can_restart(self):
        import tifffile

        torch.set_num_threads(2)
        np.save(self.folder / "feature_valid.npy", np.ones((8, 8), dtype=bool))
        tifffile.imwrite(self.folder / "he.tiff", np.zeros((128, 128, 3), dtype=np.uint8), photometric="rgb")
        weights = self.folder / "toy-weights.bin"
        weights.write_bytes(b"synthetic test double")
        dataset = self.folder / "dataset.json"
        write_json(dataset, dict(features="features.npy", image="he.tiff", feature_valid="feature_valid.npy",
                                 bounds_yxyx_um=[0, 0, 64, 64]))
        spec = manifest(dataset)

        class Backbone:
            def forward_features(self, batch):
                return torch.zeros(len(batch), 261, 1280, device=batch.device)

        for target in ("normst.features.token_maps", "normst.features.normalize_features"):
            with self.subTest(target=target), patch("normst.features.load_backbone", return_value=Backbone()):
                with patch(target, side_effect=RuntimeError("simulated interruption")):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        extract(dataset, weights, device="cpu")
                self.assertFalse(spec["features"].exists())
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    load_image_features(spec)
                with self.assertRaisesRegex(ValueError, "extract-managed"):
                    validate_external_features(dataset, confirm_complete=True)
        with patch("normst.features.load_backbone", return_value=Backbone()):
            # Interrupt after the array is renamed but before its commit record.
            original = write_json
            def fail_commit(path, value):
                if value.get("status") == "complete":
                    raise RuntimeError("simulated commit interruption")
                original(path, value)
            with patch("normst.features.write_json", side_effect=fail_commit):
                with self.assertRaisesRegex(RuntimeError, "commit interruption"):
                    extract(dataset, weights, device="cpu")
            self.assertTrue(spec["features"].exists())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_image_features(spec)
            extract(dataset, weights, device="cpu")
        features, _ = load_image_features(spec)
        self.assertEqual(tuple(features.shape), (1, 2563, 8, 8))
        self.assertEqual(read_json(self.folder / "features.json")["status"], "complete")


class CompatibilityTests(unittest.TestCase):
    def test_seeded_parameters_and_rng_match_reference(self):
        # Parameter and post-construction RNG fingerprints from the 1.0.0
        # initializer, evaluated on CPU with PyTorch 2.5.1 / Python 3.12.
        references = {
            (7, 17): ("a37a6519dbc3d503ab09342a7ef0da0ba9be5d8ffb91cf76dc9b1c4532095ff4", "36d583e156ee36befefbe13d9ab07b5ae753fb1bc9c7a921f5fccb5e9e85818b"),
            (7, 2027): ("028b4182c1e70c00478ffb5fbf57c8f7c228b0f1163426e5743a07fac1e6d412", "a13987d2fb681c9ddae4c2239b08cc970b889b7c2998594b837dde31eadf1c25"),
            (2563, 17): ("d09a04e1f2ed7969a3404c450fbd924650b18949d2e433387381e33ea58c32aa", "15837f598ffcfd0ead67f923e53ae073ef021dee21de7b7600c686e219edcff9"),
            (2563, 2027): ("bf724394668c7798016357e6ada6a89e58ae2fe050c2fc1f493534afe6088586", "6c67b7559681fb4dfbf8c41dcc7ba247b3506ffccb31fe8a65e97045eb706651"),
        }
        bounds = np.array([0., 0., 64., 64.])
        context = condition_with_prior(np.arange(1, 25).reshape(2, 12), np.array([[16., 16.], [48., 48.]]),
            np.full(2, 100.), np.ones(2, bool), bounds, load_settings()["conditioning"])
        for (channels, seed), (parameters, rng) in references.items():
            with self.subTest(channels=channels, seed=seed):
                torch.manual_seed(seed)
                field = NORMST(context, bounds, torch.zeros(1, channels, 8, 8), torch.ones(1, 1, 8, 8, dtype=torch.bool))
                digest = hashlib.sha256()
                for name, value in field.named_parameters():
                    digest.update(name.encode())
                    digest.update(value.detach().numpy().tobytes())
                self.assertEqual(digest.hexdigest(), parameters)
                self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(), rng)
                clone = NORMST(context, bounds, field.histology_features, field.feature_valid)
                clone.load_state_dict(field.state_dict(), strict=True)
                with torch.no_grad():
                    query = torch.tensor([[20., 20.], [44., 36.]])
                    torch.testing.assert_close(field.coefficients(field.encode(), query),
                                               clone.coefficients(clone.encode(), query), rtol=0, atol=0)

    def test_atomic_checkpoint_keeps_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory(prefix="normst-atomic-") as temporary:
            path = Path(temporary) / "resume.pt"
            save_checkpoint(path, {"step": 1})
            def incomplete(value, output):
                Path(output).write_bytes(b"partial checkpoint")
                raise RuntimeError("simulated disk error")
            with patch("normst.artifacts.torch.save", side_effect=incomplete):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(path, {"step": 2})
            self.assertEqual(torch.load(path, weights_only=True), {"step": 1})


class ResumeTests(unittest.TestCase):
    def test_compact_2563_channel_model_resumes_exactly(self):
        with tempfile.TemporaryDirectory(prefix="normst-compact-resume-") as temporary:
            folder = Path(temporary)
            rng = np.random.default_rng(2027)
            np.savez(folder / "observations.npz", counts=rng.poisson(10, (2, 2000)).astype(np.float32),
                centers_yx_um=np.array([[16., 16.], [48., 48.]]), radius_um=6.,
                gene_ids=np.asarray([f"gene_{i}" for i in range(2000)]))
            features = rng.standard_normal((2563, 8, 8)).astype(np.float32)
            features[:1280] = 0  # Single normalized CLS tile.
            np.save(folder / "features.npy", features)
            np.save(folder / "valid.npy", np.ones((8, 8), dtype=bool))
            dataset = folder / "dataset.json"
            write_json(dataset, dict(observations="observations.npz", features="features.npy", feature_valid="valid.npy",
                                     bounds_yxyx_um=[0, 0, 64, 64]))
            validate_external_features(dataset, confirm_complete=True)
            config = folder / "config.json"
            write_json(config, dict(epochs=2, batch_groups=1, block_um=32,
                offsets_um=[[0, 0], [16, 0], [0, 16], [16, 16]]))
            reference = fit(dataset, folder / "reference", device="cpu", config=config)
            def interrupt(path, payload):
                save_checkpoint(path, payload)
                if payload["optimizer_updates"] == 1:
                    raise RuntimeError("simulated compact-model interruption")
            with patch("normst.train.save_checkpoint", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "compact-model"):
                    fit(dataset, folder / "resumed", device="cpu", config=config, checkpoint_every=1)
            actual = fit(dataset, folder / "resumed", device="cpu", resume=folder / "resumed/resume.pt")
            a, b = torch.load(reference, weights_only=True), torch.load(actual, weights_only=True)
            for name in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], rtol=0, atol=0)

    def test_mid_epoch_resume_is_identical_to_continuous_training(self):
        with tempfile.TemporaryDirectory(prefix="normst-resume-") as temporary:
            folder = Path(temporary)
            dataset = make_inputs(folder / "data")
            config = folder / "config.json"
            write_json(config, dict(epochs=7, batch_groups=1))
            random.seed(101)
            np.random.seed(101)
            continuous = fit(dataset, folder / "continuous", device="cpu", config=config, checkpoint_every=2)
            random.seed(101)
            np.random.seed(101)
            def interrupt(path, payload):
                save_checkpoint(path, payload)
                if Path(path).name == "resume.pt" and payload["optimizer_updates"] == 2:
                    self.assertGreater(len(payload["cursor"]["order"]), payload["cursor"]["next_group"])
                    raise RuntimeError("simulated mid-epoch interruption")
            with patch("normst.train.save_checkpoint", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "mid-epoch"):
                    fit(dataset, folder / "resumed", device="cpu", config=config, checkpoint_every=2)
            recovery = folder / "resumed/resume.pt"
            self.assertFalse((folder / "resumed/model.pt").exists())
            self.assertEqual(read_json(folder / "resumed/status.json")["status"], "failed")
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            changed = folder / "changed.json"
            write_json(changed, dict(epochs=4, batch_groups=1))
            with self.assertRaisesRegex(ValueError, "configuration"):
                fit(dataset, folder / "resumed", device="cpu", config=changed, resume=recovery)
            alternative = make_inputs(folder / "alternative")
            spec = manifest(alternative)
            values = np.load(spec["features"], mmap_mode="r+")
            values[0, 0, 0] += 1
            values.flush()
            validate_external_features(alternative, confirm_complete=True)
            with self.assertRaisesRegex(ValueError, "inputs differ"):
                fit(alternative, folder / "resumed", device="cpu", resume=recovery)
            resumed = fit(dataset, folder / "resumed", device="cpu", resume=recovery, checkpoint_every=2)
            a = torch.load(continuous, weights_only=True)
            b = torch.load(resumed, weights_only=True)
            self.assertEqual(a.keys(), b.keys())
            for name in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], rtol=0, atol=0)
            self.assertEqual(a["optimizer_updates"], b["optimizer_updates"])
            self.assertEqual(read_json(folder / "continuous/training.json")["history"],
                             read_json(folder / "resumed/training.json")["history"])
            ca = torch.load(folder / "continuous/resume.pt", weights_only=True)
            cb = torch.load(recovery, weights_only=True)
            self.assertEqual(ca["optimizer"]["param_groups"], cb["optimizer"]["param_groups"])
            for parameter, state in ca["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, cb["optimizer"]["state"][parameter][key], rtol=0, atol=0)
            for name in ("python", "numpy_global", "grouping"):
                self.assertEqual(ca["random_state"][name], cb["random_state"][name])
            torch.testing.assert_close(ca["random_state"]["torch_cpu"], cb["random_state"]["torch_cpu"], rtol=0, atol=0)
            self.assertEqual(read_json(folder / "resumed/status.json")["status"], "complete")
            pa = predict(continuous, folder / "prediction-a", device="cpu")
            pb = predict(resumed, folder / "prediction-b", device="cpu")
            np.testing.assert_array_equal(np.load(pa), np.load(pb))

    def test_resume_at_last_update_and_epoch_boundary(self):
        with tempfile.TemporaryDirectory(prefix="normst-boundary-") as temporary:
            folder = Path(temporary)
            dataset = make_inputs(folder / "data")
            config = folder / "config.json"
            write_json(config, dict(epochs=2, batch_groups=1))
            reference = fit(dataset, folder / "reference", device="cpu", config=config)
            for boundary in (False, True):
                with self.subTest(epoch_boundary=boundary):
                    output = folder / str(boundary)
                    def interrupt(path, payload):
                        save_checkpoint(path, payload)
                        cursor = payload["cursor"]
                        end = cursor["epoch"] == 2 if boundary else cursor["order"] is not None and cursor["next_group"] == len(cursor["order"])
                        if end:
                            raise RuntimeError("simulated epoch-end interruption")
                    with patch("normst.train.save_checkpoint", side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError, "epoch-end"):
                            fit(dataset, output, device="cpu", config=config, checkpoint_every=1)
                    actual = fit(dataset, output, device="cpu", resume=output / "resume.pt")
                    expected = torch.load(reference, weights_only=True)["state_dict"]
                    restored = torch.load(actual, weights_only=True)["state_dict"]
                    for name in expected:
                        torch.testing.assert_close(expected[name], restored[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
