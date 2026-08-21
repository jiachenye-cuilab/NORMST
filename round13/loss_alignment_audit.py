"""Validation-only decomposition of Round13 correction/loss misalignment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from round13.slice_context_audit import (
    EVALUATION_BATCH_SIZE,
    FAMILIES,
    FOLDS,
    METHODS,
    ROUND9_ROOT,
    SOURCE_ROOT,
    VALIDATION_ROLE,
    _build_banks,
    _file_sha256,
    _fit_from_train_masks,
    _load_locked_fold,
    _predict_batch,
    _visible_context,
    _write_json_once,
    context_correction,
)


DEFAULT_SOURCE_AUDIT = (
    SOURCE_ROOT
    / "save"
    / "pro_normst"
    / "pro-v2-round-013-audit"
    / "slice_context_audit.json"
)
DEFAULT_OUTPUT = (
    SOURCE_ROOT
    / "save"
    / "pro_normst"
    / "pro-v2-round-013-loss-alignment"
    / "loss_alignment_audit.json"
)
COMPONENTS = (
    "weighted_z",
    "positive_target_contribution",
    "nonpositive_target_contribution",
    "raw_x_smooth_l1",
    "raw_x_mae",
)
RECONSTRUCTION_TOLERANCE = 1e-6
LOCKED_REPLAY_TOLERANCE = 1e-5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round9-root", type=Path, default=ROUND9_ROOT)
    parser.add_argument("--source-audit", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def _mean_arrays(values: Iterable[np.ndarray]) -> np.ndarray:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if not arrays:
        raise ValueError("cannot aggregate an empty array collection")
    stacked = np.stack(arrays)
    if not np.isfinite(stacked).all():
        raise ValueError("cannot aggregate non-finite arrays")
    return stacked.mean(axis=0)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def detection_rate_strata(detection_rate: np.ndarray) -> dict[str, np.ndarray]:
    rate = np.asarray(detection_rate, dtype=np.float64)
    if rate.ndim != 1 or not np.isfinite(rate).all() or (rate < 0).any() or (rate > 1).any():
        raise ValueError("detection_rate must be a finite [G] vector in [0,1]")
    return {
        "undetected_train": rate == 0,
        "very_sparse_weight3": (rate > 0) & (rate <= 0.1),
        "sparse_weight1to3": (rate > 0.1) & (rate < 0.5),
        "common_weight1": (rate >= 0.5) & (rate < 1),
        "always_detected_train": rate == 1,
    }


def positive_weight_groups(positive_weight: np.ndarray) -> dict[str, np.ndarray]:
    weight = np.asarray(positive_weight, dtype=np.float64)
    if weight.ndim != 1 or not np.isfinite(weight).all():
        raise ValueError("positive_weight must be a finite [G] vector")
    return {
        "weight_eq_3": np.isclose(weight, 3.0, rtol=0.0, atol=1e-7),
        "weight_between_1_and_3": (weight > 1.0 + 1e-7) & (weight < 3.0 - 1e-7),
        "weight_eq_1": np.isclose(weight, 1.0, rtol=0.0, atol=1e-7),
    }


def per_gene_components(
    prediction_z: torch.Tensor,
    target_z: torch.Tensor,
    positive_weight: torch.Tensor,
    gene_scale: torch.Tensor,
    query_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return additive contracted loss and raw-x errors for each item/gene."""
    prediction = prediction_z.float()
    target = target_z.detach().float()
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must be matching [B,N,G] tensors")
    if query_valid.shape != prediction.shape[:2]:
        raise ValueError("query_valid must match [B,N]")
    weight = positive_weight.float().reshape(1, 1, -1)
    scale = gene_scale.float().reshape(1, 1, -1)
    if weight.shape[-1] != prediction.shape[-1] or scale.shape[-1] != prediction.shape[-1]:
        raise ValueError("positive_weight/gene_scale do not match the gene dimension")
    valid = query_valid[..., None].to(prediction.dtype)
    positive = (target > 0).to(prediction.dtype)
    element_weight = torch.where(target > 0, weight, torch.ones_like(target))
    weighted_denominator = (element_weight * valid).sum(dim=1)
    if bool((weighted_denominator <= 0).any()):
        raise ValueError("every item/gene needs a positive weighted denominator")
    element_z_loss = F.smooth_l1_loss(
        prediction, target, reduction="none", beta=1.0
    )
    weighted_numerator = element_z_loss * element_weight * valid
    positive_contribution = (weighted_numerator * positive).sum(dim=1) / weighted_denominator
    nonpositive_contribution = (
        weighted_numerator * (1.0 - positive)
    ).sum(dim=1) / weighted_denominator

    prediction_x = prediction * scale
    target_x = target * scale
    raw_delta = prediction_x - target_x
    raw_absolute = torch.abs(raw_delta)
    raw_smooth = torch.where(
        raw_absolute < 1.0,
        0.5 * raw_delta * raw_delta,
        raw_absolute - 0.5,
    )
    query_count = valid.sum(dim=1)
    return {
        "weighted_z": positive_contribution + nonpositive_contribution,
        "positive_target_contribution": positive_contribution,
        "nonpositive_target_contribution": nonpositive_contribution,
        "raw_x_smooth_l1": (raw_smooth * valid).sum(dim=1) / query_count,
        "raw_x_mae": (raw_absolute * valid).sum(dim=1) / query_count,
        "positive_weight_mass_fraction": (
            element_weight * positive * valid
        ).sum(dim=1)
        / weighted_denominator,
        "positive_element_fraction": (positive * valid).sum(dim=1) / query_count,
    }


def aggregate_gene_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-weight masks inside slice, slices inside family, then families."""
    if not records:
        raise ValueError("gene records must not be empty")
    methods: dict[str, Any] = {method: {} for method in METHODS}
    target: dict[str, Any] = {}
    for family in FAMILIES:
        selected_family = [record for record in records if record["family"] == family]
        slice_ids = sorted({record["slice_id"] for record in selected_family})
        if not slice_ids:
            raise ValueError(f"missing records for family {family}")
        for method in METHODS:
            slice_values = []
            for slice_id in slice_ids:
                masks = [
                    record["methods"][method]
                    for record in selected_family
                    if record["slice_id"] == slice_id
                ]
                slice_values.append(
                    {key: _mean_arrays(mask[key] for mask in masks) for key in COMPONENTS}
                )
            methods[method][family] = {
                key: _mean_arrays(value[key] for value in slice_values)
                for key in COMPONENTS
            }
        target_slices = []
        for slice_id in slice_ids:
            masks = [
                record["target"]
                for record in selected_family
                if record["slice_id"] == slice_id
            ]
            target_slices.append(
                {
                    key: _mean_arrays(mask[key] for mask in masks)
                    for key in (
                        "positive_weight_mass_fraction",
                        "positive_element_fraction",
                    )
                }
            )
        target[family] = {
            key: _mean_arrays(value[key] for value in target_slices)
            for key in (
                "positive_weight_mass_fraction",
                "positive_element_fraction",
            )
        }
    for method in METHODS:
        methods[method]["overall"] = {
            key: _mean_arrays(methods[method][family][key] for family in FAMILIES)
            for key in COMPONENTS
        }
    target["overall"] = {
        key: _mean_arrays(target[family][key] for family in FAMILIES)
        for key in ("positive_weight_mass_fraction", "positive_element_fraction")
    }
    return {"methods": methods, "target": target}


def _method_scalar_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in METHODS:
        output[method] = {
            "overall": {
                key: float(np.mean(aggregate["methods"][method]["overall"][key]))
                for key in COMPONENTS
            },
            "families": {
                family: {
                    key: float(np.mean(aggregate["methods"][method][family][key]))
                    for key in COMPONENTS
                }
                for family in FAMILIES
            },
        }
    return output


def _method_delta_arrays(
    aggregate: dict[str, Any], method: str, level: str = "overall"
) -> dict[str, np.ndarray]:
    return {
        key: (
            aggregate["methods"][method][level][key]
            - aggregate["methods"]["baseline"][level][key]
        )
        for key in COMPONENTS
    }


def _delta_scalar_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in METHODS[1:]:
        output[method] = {
            "overall": {
                key: float(np.mean(value))
                for key, value in _method_delta_arrays(aggregate, method).items()
            },
            "families": {
                family: {
                    key: float(np.mean(value))
                    for key, value in _method_delta_arrays(
                        aggregate, method, family
                    ).items()
                }
                for family in FAMILIES
            },
        }
    return output


def _selector_summary(
    aggregate: dict[str, Any], selectors: dict[str, np.ndarray]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, selector_value in selectors.items():
        selector = np.asarray(selector_value, dtype=bool)
        count = int(selector.sum())
        item: dict[str, Any] = {"n_genes": count}
        if count:
            item["positive_weight_mass_fraction"] = float(
                np.mean(
                    aggregate["target"]["overall"][
                        "positive_weight_mass_fraction"
                    ][selector]
                )
            )
            item["positive_element_fraction"] = float(
                np.mean(
                    aggregate["target"]["overall"]["positive_element_fraction"][
                        selector
                    ]
                )
            )
            item["deltas"] = {
                method: {
                    key: float(np.mean(value[selector]))
                    for key, value in _method_delta_arrays(
                        aggregate, method
                    ).items()
                }
                for method in METHODS[1:]
            }
            positive_regression = np.maximum(
                _method_delta_arrays(aggregate, "context_affine")["weighted_z"],
                0.0,
            )
            total_positive = float(positive_regression.sum())
            item["context_positive_regression_mass_share"] = (
                float(positive_regression[selector].sum() / total_positive)
                if total_positive > 0
                else None
            )
        output[name] = item
    return output


def concentration_summary(delta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("concentration delta must be a finite vector")
    positive = np.maximum(values, 0.0)
    negative = np.minimum(values, 0.0)
    positive_mass = float(positive.sum())
    order = np.argsort(-positive, kind="stable")
    shares = {}
    for count in (1, 5, 10, 25, 50):
        take = min(count, values.size)
        shares[f"top_{count}"] = (
            float(positive[order[:take]].sum() / positive_mass)
            if positive_mass > 0
            else None
        )
    return {
        "n_genes": int(values.size),
        "n_worse": int((values > 0).sum()),
        "n_better": int((values < 0).sum()),
        "net_mean_delta": float(values.mean()),
        "positive_regression_mass": positive_mass,
        "improvement_mass": float(-negative.sum()),
        "top_positive_regression_mass_share": shares,
    }


def stability_summary(fold_deltas: np.ndarray) -> dict[str, Any]:
    values = np.asarray(fold_deltas, dtype=np.float64)
    if values.shape[0] != len(FOLDS) or values.ndim != 2:
        raise ValueError("stability requires [3,G] fold deltas")
    worse_count = (values > 0).sum(axis=0)
    better_count = (values < 0).sum(axis=0)
    correlations: dict[str, float | None] = {}
    for left in range(len(FOLDS)):
        for right in range(left + 1, len(FOLDS)):
            label = f"{FOLDS[left]}__{FOLDS[right]}"
            if np.std(values[left]) == 0 or np.std(values[right]) == 0:
                correlations[label] = None
            else:
                correlations[label] = float(
                    np.corrcoef(values[left], values[right])[0, 1]
                )
    return {
        "worse_in_3_of_3": int((worse_count == 3).sum()),
        "worse_in_at_least_2_of_3": int((worse_count >= 2).sum()),
        "better_in_3_of_3": int((better_count == 3).sum()),
        "better_in_at_least_2_of_3": int((better_count >= 2).sum()),
        "fold_pair_pearson": correlations,
    }


def _validation_gene_records(
    model: Any,
    data: Any,
    banks: dict[str, Any],
    fit: dict[str, np.ndarray],
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    positive_weight = torch.as_tensor(
        data.preprocessing.positive_weight, dtype=torch.float32, device=device
    )
    gene_scale = torch.as_tensor(
        data.preprocessing.gene_scale, dtype=torch.float32, device=device
    )
    for item in data.roles[VALIDATION_ROLE]:
        for family in FAMILIES:
            masks = banks[item.slice_id][family]
            for start in range(0, len(masks), EVALUATION_BATCH_SIZE):
                batch = masks[start : start + EVALUATION_BATCH_SIZE]
                prediction, packed = _predict_batch(model, item, batch, device)
                contexts = np.stack([_visible_context(item, mask) for mask in batch])
                corrections = {
                    "baseline": np.zeros_like(contexts, dtype=np.float32),
                    "train_bias": np.broadcast_to(
                        fit["residual_mean"], contexts.shape
                    ).copy(),
                    "context_affine": np.stack(
                        [context_correction(value, fit) for value in contexts]
                    ),
                }
                method_components: dict[str, dict[str, np.ndarray]] = {}
                target_components: dict[str, np.ndarray] | None = None
                for method in METHODS:
                    delta = torch.as_tensor(
                        corrections[method], dtype=torch.float32, device=device
                    )[:, None, :]
                    components = per_gene_components(
                        prediction + delta,
                        packed.target_z,
                        positive_weight,
                        gene_scale,
                        packed.query_valid,
                    )
                    method_components[method] = {
                        key: components[key].detach().cpu().numpy()
                        for key in COMPONENTS
                    }
                    if target_components is None:
                        target_components = {
                            key: components[key].detach().cpu().numpy()
                            for key in (
                                "positive_weight_mass_fraction",
                                "positive_element_fraction",
                            )
                        }
                if target_components is None:
                    raise RuntimeError("target components were not computed")
                for offset, _mask in enumerate(batch):
                    records.append(
                        {
                            "slice_id": item.slice_id,
                            "family": family,
                            "mask_index": start + offset,
                            "methods": {
                                method: {
                                    key: method_components[method][key][offset]
                                    for key in COMPONENTS
                                }
                                for method in METHODS
                            },
                            "target": {
                                key: target_components[key][offset]
                                for key in target_components
                            },
                        }
                    )
    return records


def run_fold(
    round9_root: Path,
    fold: str,
    source_fold: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    data, model, provenance = _load_locked_fold(round9_root, fold, device)
    for role in ("train", VALIDATION_ROLE):
        for item in data.roles[role]:
            item.preload(device)
    train_banks, validation_banks = _build_banks(data, fold)
    fit, fit_diagnostics = _fit_from_train_masks(model, data, train_banks, device)
    records = _validation_gene_records(model, data, validation_banks, fit, device)
    aggregate = aggregate_gene_records(records)
    methods = _method_scalar_summary(aggregate)
    deltas = _delta_scalar_summary(aggregate)
    reconstruction: dict[str, Any] = {}
    for method in METHODS:
        observed = methods[method]["overall"]["weighted_z"]
        frozen = float(
            source_fold["methods"][method]["criterion_weighted_z_smooth_l1"]
        )
        error = abs(observed - frozen)
        reconstruction[method] = {
            "observed": observed,
            "frozen_slice_context_audit": frozen,
            "absolute_error": error,
            "tolerance": RECONSTRUCTION_TOLERANCE,
            "passed": error <= RECONSTRUCTION_TOLERANCE,
        }
    locked_error = abs(
        methods["baseline"]["overall"]["weighted_z"]
        - provenance["locked_best_validation"]
    )
    integrity = {
        "locked_round9_replay": {
            "absolute_error": locked_error,
            "tolerance": LOCKED_REPLAY_TOLERANCE,
            "passed": locked_error <= LOCKED_REPLAY_TOLERANCE,
        },
        "slice_context_audit_reconstruction": reconstruction,
    }
    integrity["passed"] = bool(
        integrity["locked_round9_replay"]["passed"]
        and all(item["passed"] for item in reconstruction.values())
    )
    if not integrity["passed"]:
        raise RuntimeError(f"loss-alignment integrity failed for {fold}: {integrity}")
    detection = data.preprocessing.detection_rate.copy()
    weight = data.preprocessing.positive_weight.copy()
    result = {
        "fold": fold,
        "provenance": provenance,
        "n_validation_masks": len(records),
        "fit": fit_diagnostics,
        "integrity": integrity,
        "methods": methods,
        "deltas_method_minus_baseline": deltas,
        "detection_rate_strata": _selector_summary(
            aggregate, detection_rate_strata(detection)
        ),
        "positive_weight_groups": _selector_summary(
            aggregate, positive_weight_groups(weight)
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    gene_ids = data.preprocessing.gene_ids
    del model, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, aggregate, gene_ids, detection, weight


def _aggregate_fold_selector_sections(
    folds: list[dict[str, Any]], section: str
) -> dict[str, Any]:
    names = tuple(folds[0][section])
    output: dict[str, Any] = {}
    for name in names:
        items = [fold[section][name] for fold in folds]
        result: dict[str, Any] = {
            "mean_n_genes": float(np.mean([item["n_genes"] for item in items])),
            "fold_n_genes": {
                fold["fold"]: fold[section][name]["n_genes"] for fold in folds
            },
        }
        nonempty = [item for item in items if item["n_genes"] > 0]
        if nonempty:
            result["positive_weight_mass_fraction"] = _mean_optional(
                item.get("positive_weight_mass_fraction") for item in nonempty
            )
            result["positive_element_fraction"] = _mean_optional(
                item.get("positive_element_fraction") for item in nonempty
            )
            result["context_positive_regression_mass_share"] = _mean_optional(
                item.get("context_positive_regression_mass_share")
                for item in nonempty
            )
            result["deltas"] = {
                method: {
                    key: _mean_optional(
                        item["deltas"][method][key] for item in nonempty
                    )
                    for key in COMPONENTS
                }
                for method in METHODS[1:]
            }
        output[name] = result
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_path = args.source_audit.resolve()
    source_audit = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source_audit.get("schema") != "pro-normst-slice-context-audit-v1"
        or source_audit.get("test_metrics_used") is not False
        or source_audit.get("training_performed") is not False
    ):
        raise ValueError("source slice-context audit is incompatible")
    source_folds = {item["fold"]: item for item in source_audit["folds"]}
    started = time.perf_counter()
    fold_results: list[dict[str, Any]] = []
    fold_aggregates: list[dict[str, Any]] = []
    fold_detection: list[np.ndarray] = []
    fold_weights: list[np.ndarray] = []
    canonical_gene_ids: tuple[str, ...] | None = None
    for fold in FOLDS:
        result, aggregate, gene_ids, detection, weight = run_fold(
            args.round9_root.resolve(), fold, source_folds[fold], device
        )
        if canonical_gene_ids is None:
            canonical_gene_ids = gene_ids
        elif gene_ids != canonical_gene_ids:
            raise ValueError("fold gene order mismatch")
        fold_results.append(result)
        fold_aggregates.append(aggregate)
        fold_detection.append(detection)
        fold_weights.append(weight)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "integrity": result["integrity"]["passed"],
                    "context_weighted_z_delta": result[
                        "deltas_method_minus_baseline"
                    ]["context_affine"]["overall"]["weighted_z"],
                    "seconds": result["runtime_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if canonical_gene_ids is None:
        raise RuntimeError("no folds were evaluated")

    overall_method_arrays = {
        method: {
            key: _mean_arrays(
                aggregate["methods"][method]["overall"][key]
                for aggregate in fold_aggregates
            )
            for key in COMPONENTS
        }
        for method in METHODS
    }
    overall_deltas = {
        method: {
            key: overall_method_arrays[method][key]
            - overall_method_arrays["baseline"][key]
            for key in COMPONENTS
        }
        for method in METHODS[1:]
    }
    context_fold_deltas = np.stack(
        [
            _method_delta_arrays(aggregate, "context_affine")["weighted_z"]
            for aggregate in fold_aggregates
        ]
    )
    bias_fold_deltas = np.stack(
        [
            _method_delta_arrays(aggregate, "train_bias")["weighted_z"]
            for aggregate in fold_aggregates
        ]
    )
    context_delta = overall_deltas["context_affine"]["weighted_z"]
    order = np.argsort(-context_delta, kind="stable")
    mean_detection = np.stack(fold_detection).mean(axis=0)
    mean_weight = np.stack(fold_weights).mean(axis=0)
    per_gene = []
    for index, gene_id in enumerate(canonical_gene_ids):
        per_gene.append(
            {
                "gene_id": gene_id,
                "context_weighted_z_delta": float(context_delta[index]),
                "train_bias_weighted_z_delta": float(
                    overall_deltas["train_bias"]["weighted_z"][index]
                ),
                "context_positive_target_delta": float(
                    overall_deltas["context_affine"][
                        "positive_target_contribution"
                    ][index]
                ),
                "context_nonpositive_target_delta": float(
                    overall_deltas["context_affine"][
                        "nonpositive_target_contribution"
                    ][index]
                ),
                "context_raw_x_smooth_l1_delta": float(
                    overall_deltas["context_affine"]["raw_x_smooth_l1"][index]
                ),
                "context_raw_x_mae_delta": float(
                    overall_deltas["context_affine"]["raw_x_mae"][index]
                ),
                "mean_train_detection_rate": float(mean_detection[index]),
                "mean_positive_weight": float(mean_weight[index]),
                "fold_context_weighted_z_delta": {
                    fold: float(context_fold_deltas[offset, index])
                    for offset, fold in enumerate(FOLDS)
                },
            }
        )
    payload = {
        "schema": "pro-normst-loss-alignment-audit-v1",
        "evidence_tier": "validation-only-diagnostic",
        "source_round": "pro-v2-round-009",
        "source_slice_context_audit": {
            "path": str(source_path),
            "sha256": _file_sha256(source_path),
        },
        "test_metrics_used": False,
        "model_structure_changed": False,
        "training_performed": False,
        "device": str(device),
        "folds": fold_results,
        "integrity_passed": all(fold["integrity"]["passed"] for fold in fold_results),
        "overall": {
            "methods": {
                method: {
                    key: float(np.mean(value))
                    for key, value in overall_method_arrays[method].items()
                }
                for method in METHODS
            },
            "deltas_method_minus_baseline": {
                method: {
                    key: float(np.mean(value)) for key, value in values.items()
                }
                for method, values in overall_deltas.items()
            },
            "detection_rate_strata": _aggregate_fold_selector_sections(
                fold_results, "detection_rate_strata"
            ),
            "positive_weight_groups": _aggregate_fold_selector_sections(
                fold_results, "positive_weight_groups"
            ),
            "context_concentration": concentration_summary(context_delta),
            "train_bias_concentration": concentration_summary(
                overall_deltas["train_bias"]["weighted_z"]
            ),
            "context_stability": stability_summary(context_fold_deltas),
            "train_bias_stability": stability_summary(bias_fold_deltas),
            "top_25_context_regression_genes": [per_gene[index] for index in order[:25]],
        },
        "per_gene": per_gene,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json_once(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "integrity_passed": payload["integrity_passed"],
                "overall_deltas": payload["overall"][
                    "deltas_method_minus_baseline"
                ],
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
