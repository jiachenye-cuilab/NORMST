"""Round10-only training loss; importing this module does not patch v9."""

from __future__ import annotations

from typing import Any

import torch

from training.pro_normst_engine import weighted_gene_smooth_l1_per_item


CORRELATION_WEIGHT = 0.01
CORRELATION_ENERGY_EPS = 1e-6


def gene_pearson_penalty_per_item(
    prediction_z: torch.Tensor,
    target_z: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    energy_eps: float = CORRELATION_ENERGY_EPS,
) -> torch.Tensor:
    """Return gene-equal ``1-rho`` for each padded slice independently."""
    prediction = prediction_z.float()
    target = target_z.detach().float()
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction_z and target_z must be matching [B,Nq,G] tensors")
    if query_valid.shape != prediction.shape[:2]:
        raise ValueError("query_valid must align with [B,Nq]")
    if not isinstance(energy_eps, (int, float)) or not 0 < float(energy_eps):
        raise ValueError("energy_eps must be finite and positive")

    valid = query_valid[..., None].to(prediction.dtype)
    counts = valid.sum(dim=1).clamp_min(1.0)
    prediction_mean = (prediction * valid).sum(dim=1) / counts
    target_mean = (target * valid).sum(dim=1) / counts
    prediction_centered = (prediction - prediction_mean[:, None, :]) * valid
    target_centered = (target - target_mean[:, None, :]) * valid
    prediction_energy = prediction_centered.square().sum(dim=1)
    target_energy = target_centered.square().sum(dim=1)
    cross_energy = (prediction_centered * target_centered).sum(dim=1)
    defined = target_energy > float(energy_eps)
    denominator = (
        prediction_energy.clamp_min(float(energy_eps))
        * target_energy.clamp_min(float(energy_eps))
    ).sqrt()
    correlation = (cross_energy / denominator).clamp(min=-1.0, max=1.0)
    defined_float = defined.to(correlation.dtype)
    defined_count = defined_float.sum(dim=1).clamp_min(1.0)
    penalty = ((1.0 - correlation) * defined_float).sum(dim=1) / defined_count
    return penalty


def round10_training_loss_per_item(
    prediction_z: torch.Tensor,
    target_z: torch.Tensor,
    positive_weight: torch.Tensor,
    query_valid: torch.Tensor,
) -> torch.Tensor:
    """Add the frozen Round10 Pearson term to the unchanged v9 base loss."""
    base = weighted_gene_smooth_l1_per_item(
        prediction_z,
        target_z,
        positive_weight,
        query_valid,
    )
    correlation = gene_pearson_penalty_per_item(
        prediction_z,
        target_z,
        query_valid,
    )
    return base + CORRELATION_WEIGHT * correlation


def loss_contract() -> dict[str, Any]:
    return {
        "schema": "gene-equal-target-weighted-smooth-l1-plus-gene-pearson-v1",
        "base": "gene-equal-target-weighted-smooth-l1-beta1-v1",
        "pearson": {
            "axis": "query-within-slice-per-gene",
            "penalty": "mean-defined-genes(1-clamp(rho,-1,1))",
            "weight": CORRELATION_WEIGHT,
            "target_energy_defined_threshold": CORRELATION_ENERGY_EPS,
            "prediction_energy_floor": CORRELATION_ENERGY_EPS,
            "padding": "excluded",
            "target": "detached",
            "compute_dtype": "float32",
        },
        "batch_reduction": "per-slice-independent-then-slice-equal",
        "checkpoint_selection": "unchanged-weighted-z-smooth-l1-only",
    }


__all__ = [
    "CORRELATION_ENERGY_EPS",
    "CORRELATION_WEIGHT",
    "gene_pearson_penalty_per_item",
    "loss_contract",
    "round10_training_loss_per_item",
]
