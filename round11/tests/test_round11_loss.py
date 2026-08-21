from __future__ import annotations

import unittest
from unittest import mock

import torch

import training.pro_normst as base
import round11.train as round11_train
from round10.losses import gene_pearson_penalty_per_item
from round11.losses import (
    correlation_weight_for_epoch,
    loss_contract,
    round11_training_loss_per_item,
    training_epoch,
)
from training.pro_normst_engine import weighted_gene_smooth_l1_per_item


class Round11ScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = torch.tensor(
            [
                [
                    [0.0, 2.0, 1.0],
                    [1.0, 0.0, 2.0],
                    [2.0, 1.0, 0.0],
                    [3.0, 3.0, 3.0],
                ]
            ]
        )
        self.prediction = (0.2 - 0.4 * self.target).requires_grad_()
        self.valid = torch.ones(1, 4, dtype=torch.bool)
        self.positive_weight = torch.ones(3)

    def test_frozen_epoch_schedule(self) -> None:
        expected = [0.01, 0.01, 0.01, 0.01, 0.01, 0.008, 0.006, 0.004, 0.002, 0.0, 0.0]
        actual = [correlation_weight_for_epoch(epoch) for epoch in range(len(expected))]
        self.assertEqual(actual, expected)
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                correlation_weight_for_epoch(invalid)  # type: ignore[arg-type]

    def test_loss_requires_explicit_epoch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "explicit epoch"):
            round11_training_loss_per_item(
                self.prediction,
                self.target,
                self.positive_weight,
                self.valid,
            )

    def test_early_loss_and_late_base_loss_are_exact(self) -> None:
        base_loss = weighted_gene_smooth_l1_per_item(
            self.prediction,
            self.target,
            self.positive_weight,
            self.valid,
        )
        pearson = gene_pearson_penalty_per_item(
            self.prediction,
            self.target,
            self.valid,
        )
        with training_epoch(0):
            early = round11_training_loss_per_item(
                self.prediction,
                self.target,
                self.positive_weight,
                self.valid,
            )
        with training_epoch(9):
            late = round11_training_loss_per_item(
                self.prediction,
                self.target,
                self.positive_weight,
                self.valid,
            )
        torch.testing.assert_close(early, base_loss + 0.01 * pearson)
        torch.testing.assert_close(late, base_loss)

    def test_epoch_context_is_reset(self) -> None:
        with training_epoch(5):
            value = round11_training_loss_per_item(
                self.prediction,
                self.target,
                self.positive_weight,
                self.valid,
            )
            self.assertTrue(torch.isfinite(value).all())
        with self.assertRaises(RuntimeError):
            round11_training_loss_per_item(
                self.prediction,
                self.target,
                self.positive_weight,
                self.valid,
            )

    def test_contract_records_schedule(self) -> None:
        schedule = loss_contract()["schedule"]
        self.assertEqual(schedule["epochs_1_to_5"], 0.01)
        self.assertEqual(schedule["epoch_6"], 0.008)
        self.assertEqual(schedule["epoch_9"], 0.002)
        self.assertEqual(schedule["epoch_10_and_later"], 0.0)

    def test_epoch_runner_reconstructs_resume_weight_from_epoch(self) -> None:
        observed: list[torch.Tensor] = []

        def fake_epoch_runner(*args, **kwargs):
            del args, kwargs
            observed.append(
                round11_training_loss_per_item(
                    self.prediction,
                    self.target,
                    self.positive_weight,
                    self.valid,
                )
            )
            return "complete"

        with mock.patch.object(
            round11_train, "_V9_TRAIN_ONE_EPOCH", fake_epoch_runner
        ):
            self.assertEqual(round11_train._round11_train_one_epoch(epoch=5), "complete")
            self.assertEqual(round11_train._round11_train_one_epoch(epoch=5), "complete")
            self.assertEqual(round11_train._round11_train_one_epoch(epoch=9), "complete")
        torch.testing.assert_close(observed[0], observed[1])
        base_loss = weighted_gene_smooth_l1_per_item(
            self.prediction,
            self.target,
            self.positive_weight,
            self.valid,
        )
        torch.testing.assert_close(observed[2], base_loss)


class Round11IsolationTest(unittest.TestCase):
    def test_import_does_not_modify_v9(self) -> None:
        self.assertEqual(base.HUMAN_CONTRACT_VERSION, "pro-normst-human-v9")
        self.assertEqual(base.NUMERICAL_IMPLEMENTATION_SCHEMA, "pro-normst-numerical-v9")
        self.assertIs(base._train_one_epoch, round11_train._V9_TRAIN_ONE_EPOCH)

    def test_activation_is_explicit_and_process_local(self) -> None:
        with (
            mock.patch.object(base, "HUMAN_CONTRACT_VERSION", "pro-normst-human-v9"),
            mock.patch.object(base, "NUMERICAL_IMPLEMENTATION_SCHEMA", "pro-normst-numerical-v9"),
            mock.patch.object(base, "weighted_gene_smooth_l1_per_item", round11_train._V9_TRAINING_LOSS),
            mock.patch.object(base, "_contract_manifest", round11_train._V9_CONTRACT_MANIFEST),
            mock.patch.object(base, "_train_one_epoch", round11_train._V9_TRAIN_ONE_EPOCH),
        ):
            round11_train.activate_round11()
            self.assertEqual(base.HUMAN_CONTRACT_VERSION, "pro-normst-human-v11")
            self.assertEqual(base.NUMERICAL_IMPLEMENTATION_SCHEMA, "pro-normst-numerical-v11")
            self.assertIs(base._train_one_epoch, round11_train._round11_train_one_epoch)


if __name__ == "__main__":
    unittest.main()
