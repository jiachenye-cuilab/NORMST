"""Round11 Pearson warm-start schedule layered over the unchanged v9 loss."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch

from round10.losses import gene_pearson_penalty_per_item
from training.pro_normst_engine import weighted_gene_smooth_l1_per_item


PEAK_CORRELATION_WEIGHT = 0.01
FULL_WEIGHT_EPOCHS = 5
ZERO_WEIGHT_EPOCH = 10
_TRAINING_EPOCH: ContextVar[int | None] = ContextVar(
    "round11_training_epoch", default=None
)


def correlation_weight_for_epoch(epoch: int) -> float:
    """Return the frozen weight for a zero-based training epoch."""
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    epoch_number = epoch + 1
    if epoch_number <= FULL_WEIGHT_EPOCHS:
        return PEAK_CORRELATION_WEIGHT
    if epoch_number >= ZERO_WEIGHT_EPOCH:
        return 0.0
    remaining = ZERO_WEIGHT_EPOCH - epoch_number
    decay_span = ZERO_WEIGHT_EPOCH - FULL_WEIGHT_EPOCHS
    return PEAK_CORRELATION_WEIGHT * remaining / decay_span


@contextmanager
def training_epoch(epoch: int) -> Iterator[None]:
    """Set the deterministic epoch context for one base training epoch."""
    correlation_weight_for_epoch(epoch)
    token = _TRAINING_EPOCH.set(epoch)
    try:
        yield
    finally:
        _TRAINING_EPOCH.reset(token)


def round11_training_loss_per_item(
    prediction_z: torch.Tensor,
    target_z: torch.Tensor,
    positive_weight: torch.Tensor,
    query_valid: torch.Tensor,
) -> torch.Tensor:
    """Use Pearson only during the frozen warm-start/decay epochs."""
    epoch = _TRAINING_EPOCH.get()
    if epoch is None:
        raise RuntimeError("Round11 training loss requires an explicit epoch context")
    base = weighted_gene_smooth_l1_per_item(
        prediction_z,
        target_z,
        positive_weight,
        query_valid,
    )
    weight = correlation_weight_for_epoch(epoch)
    if weight == 0.0:
        return base
    correlation = gene_pearson_penalty_per_item(
        prediction_z,
        target_z,
        query_valid,
    )
    return base + weight * correlation


def loss_contract() -> dict[str, Any]:
    return {
        "schema": "gene-equal-weighted-smooth-l1-plus-scheduled-gene-pearson-v1",
        "base": "gene-equal-target-weighted-smooth-l1-beta1-v1",
        "pearson_dependency": "round10-gene-wise-pearson-v1",
        "schedule": {
            "epoch_indexing": "one-based-human-zero-based-runtime",
            "peak_weight": PEAK_CORRELATION_WEIGHT,
            "epochs_1_to_5": PEAK_CORRELATION_WEIGHT,
            "epoch_6": 0.008,
            "epoch_7": 0.006,
            "epoch_8": 0.004,
            "epoch_9": 0.002,
            "epoch_10_and_later": 0.0,
            "resume_semantics": "weight-is-pure-function-of-completed-epoch-index",
        },
        "batch_reduction": "per-slice-independent-then-slice-equal",
        "checkpoint_selection": "unchanged-weighted-z-smooth-l1-only",
    }


__all__ = [
    "FULL_WEIGHT_EPOCHS",
    "PEAK_CORRELATION_WEIGHT",
    "ZERO_WEIGHT_EPOCH",
    "correlation_weight_for_epoch",
    "loss_contract",
    "round11_training_loss_per_item",
    "training_epoch",
]
