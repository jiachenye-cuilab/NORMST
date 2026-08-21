from __future__ import annotations

import unittest
from unittest import mock

import torch

import training.pro_normst as base
import round10.train as round10_train
from round10.losses import (
    CORRELATION_WEIGHT,
    gene_pearson_penalty_per_item,
    loss_contract,
    round10_training_loss_per_item,
)
from training.pro_normst_engine import weighted_gene_smooth_l1_per_item


class Round10LossTest(unittest.TestCase):
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
        self.valid = torch.ones(1, 4, dtype=torch.bool)

    def test_perfect_and_inverse_correlation(self) -> None:
        perfect = gene_pearson_penalty_per_item(self.target, self.target, self.valid)
        inverse = gene_pearson_penalty_per_item(-self.target, self.target, self.valid)
        torch.testing.assert_close(perfect, torch.zeros_like(perfect), atol=1e-6, rtol=0)
        torch.testing.assert_close(inverse, torch.full_like(inverse, 2.0), atol=1e-6, rtol=0)

    def test_positive_affine_and_padding_invariance(self) -> None:
        scale = torch.tensor([2.0, 0.5, 3.0])
        offset = torch.tensor([4.0, -2.0, 7.0])
        transformed = self.target * scale + offset
        baseline = gene_pearson_penalty_per_item(transformed, self.target, self.valid)
        padded_prediction = torch.cat([transformed, torch.randn(1, 3, 3)], dim=1)
        padded_target = torch.cat([self.target, torch.randn(1, 3, 3)], dim=1)
        padded_valid = torch.tensor([[True, True, True, True, False, False, False]])
        padded = gene_pearson_penalty_per_item(
            padded_prediction, padded_target, padded_valid
        )
        torch.testing.assert_close(baseline, padded, atol=1e-6, rtol=0)

    def test_batch_items_are_independent(self) -> None:
        prediction = torch.cat([self.target, -self.target], dim=0)
        target = self.target.expand(2, -1, -1).clone()
        valid = self.valid.expand(2, -1).clone()
        batched = gene_pearson_penalty_per_item(prediction, target, valid)
        separate = torch.cat(
            [
                gene_pearson_penalty_per_item(
                    prediction[index : index + 1],
                    target[index : index + 1],
                    valid[index : index + 1],
                )
                for index in range(2)
            ]
        )
        torch.testing.assert_close(batched, separate)

    def test_constant_prediction_has_finite_nonzero_gradient(self) -> None:
        prediction = torch.zeros_like(self.target, requires_grad=True)
        penalty = gene_pearson_penalty_per_item(prediction, self.target, self.valid)
        penalty.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(bool((prediction.grad != 0).any()))

    def test_total_loss_is_frozen_weighted_sum(self) -> None:
        prediction = (0.3 * self.target + 0.2).requires_grad_()
        positive_weight = torch.ones(self.target.shape[-1])
        base_loss = weighted_gene_smooth_l1_per_item(
            prediction, self.target, positive_weight, self.valid
        )
        pearson = gene_pearson_penalty_per_item(prediction, self.target, self.valid)
        actual = round10_training_loss_per_item(
            prediction, self.target, positive_weight, self.valid
        )
        torch.testing.assert_close(actual, base_loss + CORRELATION_WEIGHT * pearson)
        self.assertEqual(loss_contract()["pearson"]["weight"], 0.01)


class Round10IsolationTest(unittest.TestCase):
    def test_import_does_not_modify_v9(self) -> None:
        self.assertEqual(base.HUMAN_CONTRACT_VERSION, "pro-normst-human-v9")
        self.assertEqual(
            base.NUMERICAL_IMPLEMENTATION_SCHEMA, "pro-normst-numerical-v9"
        )
        self.assertIs(
            base.weighted_gene_smooth_l1_per_item,
            round10_train._V9_TRAINING_LOSS,
        )

    def test_activation_is_explicit_and_process_local(self) -> None:
        with (
            mock.patch.object(base, "HUMAN_CONTRACT_VERSION", "pro-normst-human-v9"),
            mock.patch.object(
                base,
                "NUMERICAL_IMPLEMENTATION_SCHEMA",
                "pro-normst-numerical-v9",
            ),
            mock.patch.object(
                base,
                "weighted_gene_smooth_l1_per_item",
                round10_train._V9_TRAINING_LOSS,
            ),
            mock.patch.object(
                base, "_contract_manifest", round10_train._V9_CONTRACT_MANIFEST
            ),
        ):
            round10_train.activate_round10()
            self.assertEqual(
                base.HUMAN_CONTRACT_VERSION, round10_train.HUMAN_CONTRACT_VERSION
            )
            self.assertEqual(
                base.NUMERICAL_IMPLEMENTATION_SCHEMA,
                round10_train.NUMERICAL_IMPLEMENTATION_SCHEMA,
            )
            self.assertIs(
                base.weighted_gene_smooth_l1_per_item,
                round10_training_loss_per_item,
            )


if __name__ == "__main__":
    unittest.main()
