"""Streaming count-space metrics for full-count autoencoder evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ReconstructionMetrics:
    """Accumulate Oracle reconstruction metrics without storing a dense role."""

    def __init__(self, train_log_mean: np.ndarray):
        baseline = np.asarray(train_log_mean, dtype=np.float64)
        if baseline.ndim != 1 or baseline.size < 2:
            raise ValueError("train_log_mean must be a non-empty gene vector")
        self.baseline = torch.from_numpy(baseline)
        self.n_genes = int(baseline.size)
        self.n_spots = 0
        self.sum_prediction = torch.zeros(self.n_genes, dtype=torch.float64)
        self.sum_target = torch.zeros(self.n_genes, dtype=torch.float64)
        self.sum_prediction2 = torch.zeros(self.n_genes, dtype=torch.float64)
        self.sum_target2 = torch.zeros(self.n_genes, dtype=torch.float64)
        self.sum_cross = torch.zeros(self.n_genes, dtype=torch.float64)
        self.sum_prediction_library = torch.zeros(self.n_genes, dtype=torch.float64)
        self.sum_target_library = torch.zeros(self.n_genes, dtype=torch.float64)
        self.sum_library = 0.0
        self.sum_library2 = 0.0
        self.sse = 0.0
        self.baseline_sse = 0.0
        self.smooth_sum = 0.0
        self.smooth_count = 0
        self.positive_smooth_sum = 0.0
        self.positive_count = 0
        self.zero_smooth_sum = 0.0
        self.zero_count = 0
        self.spot_pearson_sum = 0.0
        self.spot_pearson_count = 0
        self.composition_l1_sum = 0.0
        self.library_relative_error_sum = 0.0

    def update(
        self,
        prediction_counts: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> None:
        prediction = prediction_counts.detach().float().cpu().clamp_min(0.0)
        target = target_counts.detach().float().cpu()
        if prediction.shape != target.shape or prediction.shape[1] != self.n_genes:
            raise ValueError("metric tensors do not align with train_log_mean")
        batch = prediction.shape[0]
        if batch < 1:
            return
        prediction_log = torch.log1p(prediction).double()
        target_log = torch.log1p(target).double()
        log_library = torch.log1p(target.sum(dim=1).double())

        self.n_spots += batch
        self.sum_prediction += prediction_log.sum(dim=0)
        self.sum_target += target_log.sum(dim=0)
        self.sum_prediction2 += prediction_log.square().sum(dim=0)
        self.sum_target2 += target_log.square().sum(dim=0)
        self.sum_cross += (prediction_log * target_log).sum(dim=0)
        self.sum_prediction_library += (prediction_log * log_library[:, None]).sum(dim=0)
        self.sum_target_library += (target_log * log_library[:, None]).sum(dim=0)
        self.sum_library += float(log_library.sum())
        self.sum_library2 += float(log_library.square().sum())

        residual = prediction_log - target_log
        self.sse += float(residual.square().sum())
        self.baseline_sse += float(
            (target_log - self.baseline[None, :]).square().sum()
        )

        element = F.smooth_l1_loss(
            prediction_log.float(), target_log.float(), reduction="none"
        ).double()
        positive = target > 0
        zero = ~positive
        self.smooth_sum += float(element.sum())
        self.smooth_count += int(element.numel())
        self.positive_smooth_sum += float(element[positive].sum())
        self.positive_count += int(positive.sum())
        self.zero_smooth_sum += float(element[zero].sum())
        self.zero_count += int(zero.sum())

        centered_prediction = prediction_log - prediction_log.mean(dim=1, keepdim=True)
        centered_target = target_log - target_log.mean(dim=1, keepdim=True)
        denominator = torch.sqrt(
            centered_prediction.square().sum(dim=1)
            * centered_target.square().sum(dim=1)
        )
        valid = denominator > 1e-12
        if bool(valid.any()):
            values = (centered_prediction * centered_target).sum(dim=1)[valid]
            values = values / denominator[valid]
            self.spot_pearson_sum += float(values.sum())
            self.spot_pearson_count += int(valid.sum())

        true_library = target.sum(dim=1, keepdim=True).clamp_min(1.0)
        true_composition = target / true_library
        predicted_composition = prediction / prediction.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.composition_l1_sum += float(
            (predicted_composition - true_composition).abs().sum(dim=1).sum()
        )
        self.library_relative_error_sum += float(
            (
                (prediction.sum(dim=1) - target.sum(dim=1)).abs()
                / target.sum(dim=1).clamp_min(1.0)
            ).sum()
        )

    @staticmethod
    def _mean_valid(values: torch.Tensor, valid: torch.Tensor) -> tuple[float, int]:
        count = int(valid.sum())
        return (
            float(values[valid].mean()) if count else float("nan"),
            count,
        )

    def compute(self) -> dict[str, float | int]:
        if self.n_spots < 1:
            raise ValueError("no observations were accumulated")
        n = float(self.n_spots)
        var_prediction = self.sum_prediction2 - self.sum_prediction.square() / n
        var_target = self.sum_target2 - self.sum_target.square() / n
        covariance = self.sum_cross - self.sum_prediction * self.sum_target / n
        gene_denominator = torch.sqrt(var_prediction.clamp_min(0) * var_target.clamp_min(0))
        gene_valid = gene_denominator > 1e-12
        gene_values = covariance / gene_denominator.clamp_min(1e-12)
        gene_pearson, gene_count = self._mean_valid(gene_values, gene_valid)

        library_variance = self.sum_library2 - self.sum_library**2 / n
        cov_prediction_library = (
            self.sum_prediction_library
            - self.sum_prediction * self.sum_library / n
        )
        cov_target_library = (
            self.sum_target_library - self.sum_target * self.sum_library / n
        )
        if library_variance > 1e-12:
            residual_covariance = (
                covariance
                - cov_prediction_library * cov_target_library / library_variance
            )
            residual_prediction_var = (
                var_prediction - cov_prediction_library.square() / library_variance
            )
            residual_target_var = (
                var_target - cov_target_library.square() / library_variance
            )
            partial_denominator = torch.sqrt(
                residual_prediction_var.clamp_min(0)
                * residual_target_var.clamp_min(0)
            )
            partial_valid = partial_denominator > 1e-12
            partial_values = residual_covariance / partial_denominator.clamp_min(1e-12)
            partial_pearson, partial_count = self._mean_valid(
                partial_values, partial_valid
            )
        else:
            partial_pearson, partial_count = float("nan"), 0

        def safe_ratio(numerator: float, denominator: int) -> float:
            return numerator / denominator if denominator else float("nan")

        return {
            "spots": self.n_spots,
            "genes": self.n_genes,
            "lograw_r2_vs_train_mean": 1.0
            - self.sse / max(self.baseline_sse, 1e-12),
            "gene_pearson": gene_pearson,
            "gene_pearson_valid": gene_count,
            "partial_gene_pearson_controlling_log_library": partial_pearson,
            "partial_gene_pearson_valid": partial_count,
            "spot_pearson": safe_ratio(
                self.spot_pearson_sum, self.spot_pearson_count
            ),
            "spot_pearson_valid": self.spot_pearson_count,
            "smooth_l1": safe_ratio(self.smooth_sum, self.smooth_count),
            "positive_smooth_l1": safe_ratio(
                self.positive_smooth_sum, self.positive_count
            ),
            "zero_smooth_l1": safe_ratio(self.zero_smooth_sum, self.zero_count),
            "positive_elements": self.positive_count,
            "zero_elements": self.zero_count,
            "composition_l1": self.composition_l1_sum / self.n_spots,
            "reconstructed_library_relative_error_with_true_library": (
                self.library_relative_error_sum / self.n_spots
            ),
        }
