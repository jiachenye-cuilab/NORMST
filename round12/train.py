#!/usr/bin/env python
"""Opt-in Round12 trainer layered over the frozen v9 training semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import training.pro_normst as base
from round12.model import Round12ProNORMST


HUMAN_CONTRACT_VERSION = "pro-normst-human-v12"
NUMERICAL_IMPLEMENTATION_SCHEMA = "pro-normst-numerical-v12"
_EXPECTED_BASE_HUMAN = "pro-normst-human-v9"
_EXPECTED_BASE_NUMERICAL = "pro-normst-numerical-v9"
_V9_PRO_NORMST = base.ProNORMST
_V9_CONTRACT_MANIFEST = base._contract_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round12_contract_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    manifest = _V9_CONTRACT_MANIFEST(*args, **kwargs)
    manifest["schema"] = "pro-normst-training-contract-v7"
    manifest["model"]["round12_implementation_sha256"] = {
        "round12/model.py": _sha256(Path(__file__).with_name("model.py")),
        "round12/train.py": _sha256(Path(__file__)),
    }
    return manifest


def activate_round12() -> None:
    """Patch only this process and fail closed on any non-v9 base."""
    if (
        base.HUMAN_CONTRACT_VERSION == HUMAN_CONTRACT_VERSION
        and base.NUMERICAL_IMPLEMENTATION_SCHEMA
        == NUMERICAL_IMPLEMENTATION_SCHEMA
        and base.ProNORMST is Round12ProNORMST
        and base._contract_manifest is _round12_contract_manifest
    ):
        return
    if base.HUMAN_CONTRACT_VERSION != _EXPECTED_BASE_HUMAN:
        raise RuntimeError("Round12 requires the frozen v9 human contract")
    if base.NUMERICAL_IMPLEMENTATION_SCHEMA != _EXPECTED_BASE_NUMERICAL:
        raise RuntimeError("Round12 requires the frozen v9 numerical implementation")
    if base.ProNORMST is not _V9_PRO_NORMST:
        raise RuntimeError("Round12 refuses an already modified v9 model class")
    if base._contract_manifest is not _V9_CONTRACT_MANIFEST:
        raise RuntimeError("Round12 refuses an already modified v9 contract builder")
    base.HUMAN_CONTRACT_VERSION = HUMAN_CONTRACT_VERSION
    base.NUMERICAL_IMPLEMENTATION_SCHEMA = NUMERICAL_IMPLEMENTATION_SCHEMA
    base.ProNORMST = Round12ProNORMST
    base._contract_manifest = _round12_contract_manifest


def main(argv: list[str] | None = None) -> int:
    activate_round12()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HUMAN_CONTRACT_VERSION",
    "NUMERICAL_IMPLEMENTATION_SCHEMA",
    "activate_round12",
    "main",
]
