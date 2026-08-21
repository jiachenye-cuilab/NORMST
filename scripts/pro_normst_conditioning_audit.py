#!/usr/bin/env python
"""Validation-only audit of Round7 versus Round8 global conditioning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from datasets.pro_normst import prepare_pro_normst_data
from models.pro_normst import ProNORMST, ResidualLocalStateEnhancer
from scripts.pro_normst_representation_audit import (
    _atomic_json,
    _evaluate_model,
    file_sha256,
)
from training.pro_normst import (
    DEFAULT_PANEL,
    _build_fixed_banks,
    _build_geometries,
    _split_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--round7-dir", type=Path, required=True)
    parser.add_argument("--round8-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _checkpoint(run_dir: Path, device: torch.device) -> tuple[Path, dict[str, Any]]:
    path = run_dir / "best.pt"
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get("schema") != "pro-normst-checkpoint-v1":
        raise ValueError(f"checkpoint schema is incompatible: {path}")
    return path, value


def _load_round7(
    run_dir: Path,
    gene_mean: Any,
    device: torch.device,
) -> tuple[ProNORMST, dict[str, Any]]:
    path, checkpoint = _checkpoint(run_dir, device)
    model = ProNORMST(torch.as_tensor(gene_mean), variant="full").to(device)
    model.local_state_enhancer = ResidualLocalStateEnhancer(
        state_dim=model.state_dim,
        hidden_dim=model.LOCAL_STATE_HIDDEN_DIM,
        norm_eps=model.local_norm.eps,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "saved_best_epoch": int(checkpoint["best_epoch"]),
        "saved_best_validation": float(checkpoint["best_value"]),
        "historical_unconditioned_enhancer_reconstructed": True,
    }


def _load_round8(
    run_dir: Path,
    gene_mean: Any,
    device: torch.device,
) -> tuple[ProNORMST, dict[str, Any]]:
    path, checkpoint = _checkpoint(run_dir, device)
    model = ProNORMST(torch.as_tensor(gene_mean), variant="full").to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "saved_best_epoch": int(checkpoint["best_epoch"]),
        "saved_best_validation": float(checkpoint["best_value"]),
        "historical_unconditioned_enhancer_reconstructed": False,
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    data = prepare_pro_normst_data(
        args.manifest,
        args.panel,
        count_file=args.count_file,
    )
    protocol, fold = _split_identity(data)
    if protocol != "pair_grouped_random_split" or fold != "pilot_seed2027":
        raise ValueError("conditioning audit requires the frozen pilot split")
    geometries = _build_geometries(data)
    banks = _build_fixed_banks(data, geometries, protocol, fold)
    round7, round7_identity = _load_round7(
        args.round7_dir,
        data.preprocessing.gene_mean_z,
        device,
    )
    round8, round8_identity = _load_round8(
        args.round8_dir,
        data.preprocessing.gene_mean_z,
        device,
    )
    rounds = {
        "round7": {
            **round7_identity,
            **_evaluate_model(round7, data, banks, device),
        },
        "round8": {
            **round8_identity,
            **_evaluate_model(round8, data, banks, device),
        },
    }
    for value in rounds.values():
        value["replay_minus_saved_criterion"] = (
            value["criterion_weighted_z_smooth_l1"]
            - value["saved_best_validation"]
        )
        if abs(value["replay_minus_saved_criterion"]) > 5e-5:
            raise ValueError("validation replay does not reproduce saved criterion")
    comparisons = {}
    for family in ("ordinary", "gap"):
        baseline = rounds["round7"]["families"][family]
        candidate = rounds["round8"]["families"][family]
        comparisons[family] = {
            key: candidate[key] - baseline[key]
            for key in (
                "raw_x_smooth_l1",
                "global_local_normalized_linear_cka",
                "local_conditional_error_gain",
                "global_conditional_error_gain",
            )
        }
    payload = {
        "schema": "pro-normst-conditioned-innovation-audit-v1",
        "selection_use": False,
        "data_scope": "validation_only_same_2_slices_x_2_families_x_16_masks",
        "aggregation": "masks_equal_within_slice_then_slices_equal_within_family",
        "rounds": rounds,
        "round8_minus_round7": comparisons,
        "limitations": [
            "Branch removal uses the same full-model decoder and is descriptive, not a separately trained matched branch control.",
            "Linear CKA measures shared representation structure but does not prove causal independence.",
            "No test expression or test metric is used in this audit.",
        ],
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
