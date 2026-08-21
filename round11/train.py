#!/usr/bin/env python
"""Opt-in Round11 trainer with an epoch-derived Pearson loss schedule."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import training.pro_normst as base
from round11.losses import (
    loss_contract,
    round11_training_loss_per_item,
    training_epoch,
)


HUMAN_CONTRACT_VERSION = "pro-normst-human-v11"
NUMERICAL_IMPLEMENTATION_SCHEMA = "pro-normst-numerical-v11"
_EXPECTED_BASE_HUMAN = "pro-normst-human-v9"
_EXPECTED_BASE_NUMERICAL = "pro-normst-numerical-v9"
_V9_CONTRACT_MANIFEST = base._contract_manifest
_V9_TRAINING_LOSS = base.weighted_gene_smooth_l1_per_item
_V9_TRAIN_ONE_EPOCH = base._train_one_epoch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round11_contract_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    manifest = _V9_CONTRACT_MANIFEST(*args, **kwargs)
    manifest["schema"] = "pro-normst-training-contract-v6"
    root = Path(__file__).resolve().parents[1]
    manifest["loss_schema"] = {
        **loss_contract(),
        "implementation_sha256": {
            "round10/losses.py": _sha256(root / "round10/losses.py"),
            "round11/losses.py": _sha256(Path(__file__).with_name("losses.py")),
            "round11/train.py": _sha256(Path(__file__)),
        },
    }
    return manifest


def _round11_train_one_epoch(*args: Any, **kwargs: Any) -> Any:
    epoch = kwargs.get("epoch")
    if not isinstance(epoch, int):
        raise RuntimeError("Round11 requires _train_one_epoch(epoch=<int>)")
    with training_epoch(epoch):
        return _V9_TRAIN_ONE_EPOCH(*args, **kwargs)


def activate_round11() -> None:
    """Patch only the current process and fail closed on any non-v9 base."""
    if (
        base.HUMAN_CONTRACT_VERSION == HUMAN_CONTRACT_VERSION
        and base.NUMERICAL_IMPLEMENTATION_SCHEMA == NUMERICAL_IMPLEMENTATION_SCHEMA
        and base.weighted_gene_smooth_l1_per_item is round11_training_loss_per_item
        and base._contract_manifest is _round11_contract_manifest
        and base._train_one_epoch is _round11_train_one_epoch
    ):
        return
    if base.HUMAN_CONTRACT_VERSION != _EXPECTED_BASE_HUMAN:
        raise RuntimeError("Round11 requires the frozen v9 human contract")
    if base.NUMERICAL_IMPLEMENTATION_SCHEMA != _EXPECTED_BASE_NUMERICAL:
        raise RuntimeError("Round11 requires the frozen v9 numerical implementation")
    if base.weighted_gene_smooth_l1_per_item is not _V9_TRAINING_LOSS:
        raise RuntimeError("Round11 refuses an already modified v9 training loss")
    if base._contract_manifest is not _V9_CONTRACT_MANIFEST:
        raise RuntimeError("Round11 refuses an already modified v9 contract builder")
    if base._train_one_epoch is not _V9_TRAIN_ONE_EPOCH:
        raise RuntimeError("Round11 refuses an already modified v9 epoch runner")
    base.HUMAN_CONTRACT_VERSION = HUMAN_CONTRACT_VERSION
    base.NUMERICAL_IMPLEMENTATION_SCHEMA = NUMERICAL_IMPLEMENTATION_SCHEMA
    base.weighted_gene_smooth_l1_per_item = round11_training_loss_per_item
    base._contract_manifest = _round11_contract_manifest
    base._train_one_epoch = _round11_train_one_epoch


def main(argv: list[str] | None = None) -> int:
    activate_round11()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
