#!/usr/bin/env python
"""Post-hoc validation-only representation audit for Round5 versus Round7.

This diagnostic never participates in checkpoint selection.  It reconstructs
the Round5 model under the current implementation by leaving the Round7 local
enhancer at its exact zero-start identity, then replays both locked best
checkpoints on the same fixed validation mask bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from datasets.pro_normst import prepare_pro_normst_data
from models.pro_normst import ProNORMST, ResidualLocalStateEnhancer
from training.pro_normst import (
    DEFAULT_PANEL,
    _build_fixed_banks,
    _build_geometries,
    _split_identity,
)
from training.pro_normst_engine import scientific_metrics, weighted_gene_smooth_l1


ROUND5_MISSING_STATE = {
    "local_state_enhancer.input_projection.weight",
    "local_state_enhancer.input_projection.bias",
    "local_state_enhancer.output_projection.weight",
    "local_state_enhancer.output_projection.bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--round5-dir", type=Path, required=True)
    parser.add_argument("--round7-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(
    run_dir: Path,
    gene_mean: np.ndarray,
    device: torch.device,
    *,
    round5_identity_reconstruction: bool,
) -> tuple[ProNORMST, dict[str, Any]]:
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "pro-normst-checkpoint-v1":
        raise ValueError(f"checkpoint schema is incompatible: {checkpoint_path}")
    model = ProNORMST(torch.as_tensor(gene_mean), variant="full").to(device)
    # The repository may have advanced beyond v7.  Reinstall the exact
    # unconditioned enhancer used by Round7 before loading historical states;
    # current forward accepts it through the legacy optional condition input.
    model.local_state_enhancer = ResidualLocalStateEnhancer(
        state_dim=model.state_dim,
        hidden_dim=model.LOCAL_STATE_HIDDEN_DIM,
        norm_eps=model.local_norm.eps,
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    expected_missing = ROUND5_MISSING_STATE if round5_identity_reconstruction else set()
    if missing != expected_missing or unexpected:
        raise ValueError(
            f"checkpoint state mismatch for {run_dir}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if round5_identity_reconstruction:
        enhancer = model.local_state_enhancer
        if not bool((enhancer.output_projection.weight == 0).all()) or not bool(
            (enhancer.output_projection.bias == 0).all()
        ):
            raise ValueError("Round5 identity enhancer is not exactly zero-start")
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "saved_best_epoch": int(checkpoint["best_epoch"]),
        "saved_best_validation": float(checkpoint["best_value"]),
        "round5_identity_reconstruction": round5_identity_reconstruction,
        "missing_state_filled_by_identity_enhancer": sorted(missing),
    }


def _linear_cka(global_feature: torch.Tensor, local_feature: torch.Tensor) -> float | None:
    if global_feature.shape != local_feature.shape or global_feature.shape[0] < 2:
        return None
    left = global_feature.float() - global_feature.float().mean(dim=0, keepdim=True)
    right = local_feature.float() - local_feature.float().mean(dim=0, keepdim=True)
    cross_energy = (left.transpose(0, 1) @ right).square().sum()
    left_energy = (left.transpose(0, 1) @ left).square().sum()
    right_energy = (right.transpose(0, 1) @ right).square().sum()
    denominator = (left_energy * right_energy).sqrt()
    if float(denominator.item()) <= 0.0:
        return None
    return float((cross_energy / denominator).item())


@torch.no_grad()
def _evaluate_model(
    model: ProNORMST,
    data: Any,
    banks: Any,
    device: torch.device,
) -> dict[str, Any]:
    use_amp = device.type == "cuda"
    positive_weight = torch.as_tensor(
        data.preprocessing.positive_weight, dtype=torch.float32, device=device
    )
    gene_scale = data.preprocessing.gene_scale
    family_slices: dict[str, list[dict[str, float]]] = {
        "ordinary": [],
        "gap": [],
    }
    for family in ("ordinary", "gap"):
        for item in data.roles["val"]:
            records: list[dict[str, float]] = []
            for mask in banks["val"][item.slice_id][family]:
                visible_z = torch.as_tensor(
                    item.expression_z[mask.visible_index],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                target_z = torch.as_tensor(
                    item.expression_z[mask.query_index],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    prediction_z, auxiliary = model(
                        visible_z,
                        torch.as_tensor(mask.visible_index, device=device),
                        torch.as_tensor(mask.query_index, device=device),
                        item.geometry(device),
                        return_auxiliary=True,
                    )
                    global_feature = auxiliary["global_normalized"]
                    local_feature = auxiliary["gated_local"]
                    no_local_z = model.gene_decoder(
                        torch.cat([global_feature, torch.zeros_like(local_feature)], dim=-1)
                    )
                    no_global_z = model.gene_decoder(
                        torch.cat([torch.zeros_like(global_feature), local_feature], dim=-1)
                    )
                active = auxiliary["active_query"] & auxiliary["query_valid"]
                cka = _linear_cka(
                    auxiliary["global_normalized"][active],
                    auxiliary["local_normalized"][active],
                )
                target_x = item.expression_x[mask.query_index]

                def raw_x(value: torch.Tensor) -> np.ndarray:
                    return (
                        value.float().squeeze(0).cpu().numpy() * gene_scale[None, :]
                    )

                full_smooth = scientific_metrics(
                    raw_x(prediction_z), target_x
                )["smooth_l1"]
                no_local_smooth = scientific_metrics(
                    raw_x(no_local_z), target_x
                )["smooth_l1"]
                no_global_smooth = scientific_metrics(
                    raw_x(no_global_z), target_x
                )["smooth_l1"]
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in (full_smooth, no_local_smooth, no_global_smooth)
                ):
                    raise FloatingPointError("non-finite branch audit metric")
                record = {
                    "weighted_z_smooth_l1": float(
                        weighted_gene_smooth_l1(
                            prediction_z, target_z, positive_weight
                        ).item()
                    ),
                    "raw_x_smooth_l1": float(full_smooth),
                    "raw_x_smooth_l1_without_local": float(no_local_smooth),
                    "raw_x_smooth_l1_without_global": float(no_global_smooth),
                    "local_conditional_error_gain": float(no_local_smooth - full_smooth),
                    "global_conditional_error_gain": float(no_global_smooth - full_smooth),
                }
                if cka is not None:
                    record["global_local_normalized_linear_cka"] = cka
                records.append(record)
            keys = sorted({key for record in records for key in record})
            family_slices[family].append(
                {
                    key: fmean(float(record[key]) for record in records if key in record)
                    for key in keys
                }
            )
    families = {
        family: {
            key: fmean(float(record[key]) for record in records if key in record)
            for key in sorted({key for record in records for key in record})
        }
        for family, records in family_slices.items()
    }
    return {
        "families": families,
        "criterion_weighted_z_smooth_l1": 0.5
        * (
            families["ordinary"]["weighted_z_smooth_l1"]
            + families["gap"]["weighted_z_smooth_l1"]
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    data = prepare_pro_normst_data(
        args.manifest, args.panel, count_file=args.count_file
    )
    protocol, fold = _split_identity(data)
    if protocol != "pair_grouped_random_split" or fold != "pilot_seed2027":
        raise ValueError("representation audit requires the frozen pilot split")
    geometries = _build_geometries(data)
    banks = _build_fixed_banks(data, geometries, protocol, fold)
    round5, round5_identity = _load_model(
        args.round5_dir,
        data.preprocessing.gene_mean_z,
        device,
        round5_identity_reconstruction=True,
    )
    round7, round7_identity = _load_model(
        args.round7_dir,
        data.preprocessing.gene_mean_z,
        device,
        round5_identity_reconstruction=False,
    )
    rounds = {
        "round5": {
            **round5_identity,
            **_evaluate_model(round5, data, banks, device),
        },
        "round7": {
            **round7_identity,
            **_evaluate_model(round7, data, banks, device),
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
        left = rounds["round5"]["families"][family]
        right = rounds["round7"]["families"][family]
        comparisons[family] = {
            key: right[key] - left[key]
            for key in (
                "global_local_normalized_linear_cka",
                "local_conditional_error_gain",
                "global_conditional_error_gain",
            )
        }
    payload = {
        "schema": "pro-normst-representation-audit-v1",
        "selection_use": False,
        "data_scope": "validation_only_same_2_slices_x_2_families_x_16_masks",
        "aggregation": "masks_equal_within_slice_then_slices_equal_within_family",
        "rounds": rounds,
        "round7_minus_round5": comparisons,
        "limitations": [
            "Branch removal uses the same full-model decoder and is descriptive, not a separately trained matched branch control.",
            "Linear CKA measures shared representation structure but does not prove causal redundancy.",
            "No test expression or test metric is used in this audit.",
        ],
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
