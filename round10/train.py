#!/usr/bin/env python
"""Opt-in Round10 trainer layered over the frozen v9 implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import training.pro_normst as base
from round10.losses import loss_contract, round10_training_loss_per_item


HUMAN_CONTRACT_VERSION = "pro-normst-human-v10"
NUMERICAL_IMPLEMENTATION_SCHEMA = "pro-normst-numerical-v10"
_EXPECTED_BASE_HUMAN = "pro-normst-human-v9"
_EXPECTED_BASE_NUMERICAL = "pro-normst-numerical-v9"
_V9_CONTRACT_MANIFEST = base._contract_manifest
_V9_TRAINING_LOSS = base.weighted_gene_smooth_l1_per_item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round10_contract_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    manifest = _V9_CONTRACT_MANIFEST(*args, **kwargs)
    manifest["schema"] = "pro-normst-training-contract-v5"
    manifest["loss_schema"] = {
        **loss_contract(),
        "implementation_sha256": {
            "round10/losses.py": _sha256(Path(__file__).with_name("losses.py")),
            "round10/train.py": _sha256(Path(__file__)),
        },
    }
    return manifest


def activate_round10() -> None:
    """Patch only this Python process; the repository's default v9 path is untouched."""
    if (
        base.HUMAN_CONTRACT_VERSION == HUMAN_CONTRACT_VERSION
        and base.NUMERICAL_IMPLEMENTATION_SCHEMA == NUMERICAL_IMPLEMENTATION_SCHEMA
        and base.weighted_gene_smooth_l1_per_item is round10_training_loss_per_item
        and base._contract_manifest is _round10_contract_manifest
    ):
        return
    if base.HUMAN_CONTRACT_VERSION != _EXPECTED_BASE_HUMAN:
        raise RuntimeError("Round10 requires the frozen v9 human contract")
    if base.NUMERICAL_IMPLEMENTATION_SCHEMA != _EXPECTED_BASE_NUMERICAL:
        raise RuntimeError("Round10 requires the frozen v9 numerical implementation")
    if base.weighted_gene_smooth_l1_per_item is not _V9_TRAINING_LOSS:
        raise RuntimeError("Round10 refuses an already modified v9 training loss")
    if base._contract_manifest is not _V9_CONTRACT_MANIFEST:
        raise RuntimeError("Round10 refuses an already modified v9 contract builder")
    base.HUMAN_CONTRACT_VERSION = HUMAN_CONTRACT_VERSION
    base.NUMERICAL_IMPLEMENTATION_SCHEMA = NUMERICAL_IMPLEMENTATION_SCHEMA
    base.weighted_gene_smooth_l1_per_item = round10_training_loss_per_item
    base._contract_manifest = _round10_contract_manifest


def main(argv: list[str] | None = None) -> int:
    activate_round10()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
