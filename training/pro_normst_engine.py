"""Loss, baseline, metrics, and contract utilities for ProNORMST training."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from datasets.pro_normst import ProNORMSTSlice
from models.pro_normst import ProNORMST
from training.pro_normst_masks import ProMask


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_initialization(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        torch.use_deterministic_algorithms(False)


def capture_rng_state(data_order_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_order": data_order_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], data_order_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    data_order_generator.set_state(state["data_order"])


def weighted_gene_smooth_l1(
    prediction_z: torch.Tensor,
    target_z: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    """Gene-equal SmoothL1 with target-dependent positive element weights."""
    prediction = prediction_z.float()
    target = target_z.detach().float()
    weight = positive_weight.float().reshape(1, 1, -1)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction_z and target_z must be matching [B,Nq,G] tensors")
    if weight.shape[-1] != prediction.shape[-1]:
        raise ValueError("positive_weight does not match the gene dimension")
    element_weight = torch.where(target > 0, weight, torch.ones_like(target))
    element_loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)
    numerator = (element_loss * element_weight).sum(dim=(0, 1))
    denominator = element_weight.sum(dim=(0, 1))
    if bool((denominator <= 0).any()):
        raise RuntimeError("gene loss denominator must be positive")
    loss = (numerator / denominator).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("weighted SmoothL1 is non-finite")
    return loss


def strict_visible_idw(
    visible_expression_x: np.ndarray,
    visible_xy: np.ndarray,
    visible_index: np.ndarray,
    query_xy: np.ndarray,
    *,
    neighbors: int = 6,
    power: float = 2.0,
) -> np.ndarray:
    """Six-nearest original-visible IDW with canonical-index tie breaking."""
    values = np.asarray(visible_expression_x, dtype=np.float64)
    source_xy = np.asarray(visible_xy, dtype=np.float64)
    target_xy = np.asarray(query_xy, dtype=np.float64)
    canonical = np.asarray(visible_index, dtype=np.int64)
    if values.ndim != 2 or source_xy.shape != (values.shape[0], 2):
        raise ValueError("visible IDW inputs are misaligned")
    if canonical.shape != (values.shape[0],) or target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError("IDW index/target geometry is invalid")
    if values.shape[0] < 1 or neighbors != 6 or power != 2.0:
        raise ValueError("strict IDW requires original-visible values, k=6, power=2")
    output = np.empty((target_xy.shape[0], values.shape[1]), dtype=np.float64)
    take = min(neighbors, values.shape[0])
    for query_offset, coordinate in enumerate(target_xy):
        distance = np.linalg.norm(source_xy - coordinate[None, :], axis=1)
        order = np.lexsort((canonical, distance))[:take]
        selected_distance = distance[order]
        if (selected_distance <= 0).any():
            zero = order[selected_distance <= 0]
            output[query_offset] = values[zero].mean(axis=0)
        else:
            weight = selected_distance ** (-power)
            weight /= weight.sum()
            output[query_offset] = weight @ values[order]
    return output.astype(np.float32)


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(a_centered * a_centered, axis=1)
        * np.sum(b_centered * b_centered, axis=1)
    )
    result = np.full(a.shape[0], np.nan, dtype=np.float32)
    valid = denominator > 0
    result[valid] = np.sum(a_centered[valid] * b_centered[valid], axis=1) / denominator[valid]
    return result


def scientific_metrics(
    prediction_x: np.ndarray,
    target_x: np.ndarray,
    *,
    min_queries: int = 1,
) -> dict[str, float | int | None]:
    prediction = np.asarray(prediction_x, dtype=np.float32)
    target = np.asarray(target_x, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("metric arrays must have matching [Nq,G] shape")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("metric arrays contain NaN/Inf")
    result: dict[str, float | int | None] = {
        "n_queries": int(prediction.shape[0]),
        "n_genes": int(prediction.shape[1]),
    }
    if prediction.shape[0] < min_queries:
        result["coverage"] = 0.0
        return result
    delta = prediction - target
    absolute = np.abs(delta)
    smooth = np.where(absolute < 1.0, 0.5 * delta * delta, absolute - 0.5)
    result.update(
        smooth_l1=float(smooth.mean()),
        mae=float(absolute.mean()),
        rmse=float(np.sqrt(np.mean(delta * delta))),
        negative_fraction=float(np.mean(prediction < 0)),
        coverage=1.0,
    )
    gene_pearson = _pearson_rows(prediction.T, target.T)
    spot_pearson = _pearson_rows(prediction, target)
    result["gene_pearson"] = (
        float(np.nanmean(gene_pearson)) if np.isfinite(gene_pearson).any() else None
    )
    result["gene_pearson_defined"] = int(np.isfinite(gene_pearson).sum())
    result["spot_pearson"] = (
        float(np.nanmean(spot_pearson)) if np.isfinite(spot_pearson).any() else None
    )
    result["spot_pearson_defined"] = int(np.isfinite(spot_pearson).sum())

    prediction_variance = np.var(prediction, axis=0)
    target_variance = np.var(target, axis=0)
    valid_variance = target_variance > 0
    variance_ratio = prediction_variance[valid_variance] / target_variance[valid_variance]
    if variance_ratio.size:
        result["variance_ratio_median"] = float(np.median(variance_ratio))
        result["variance_ratio_q25"] = float(np.quantile(variance_ratio, 0.25))
        result["variance_ratio_q75"] = float(np.quantile(variance_ratio, 0.75))
    else:
        result["variance_ratio_median"] = None
        result["variance_ratio_q25"] = None
        result["variance_ratio_q75"] = None
    result["variance_ratio_defined"] = int(variance_ratio.size)

    for label, selector in (("positive", target > 0), ("zero", target == 0)):
        count = int(selector.sum())
        result[f"{label}_elements"] = count
        result[f"{label}_mae"] = float(absolute[selector].mean()) if count else None
        result[f"{label}_rmse"] = (
            float(np.sqrt(np.mean(delta[selector] ** 2))) if count else None
        )
    return result


def paired_metric_gain(
    model_metrics: dict[str, Any],
    idw_metrics: dict[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("smooth_l1", "mae", "rmse", "positive_mae", "positive_rmse", "zero_mae", "zero_rmse"):
        left, right = model_metrics.get(key), idw_metrics.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[f"{key}_gain"] = float(right - left)
    for key in ("gene_pearson", "spot_pearson"):
        left, right = model_metrics.get(key), idw_metrics.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[f"{key}_gain"] = float(left - right)
    return result


def metric_strata(mask: ProMask) -> dict[str, np.ndarray]:
    strata: dict[str, np.ndarray] = {"all": np.ones(mask.query_index.size, dtype=bool)}
    for value in np.unique(mask.provenance):
        strata[f"provenance:{value}"] = mask.provenance == value
    for value in np.unique(mask.depth):
        label = "disconnected" if value < 0 else ("5+" if value >= 5 else str(int(value)))
        selector = mask.depth < 0 if value < 0 else (mask.depth >= 5 if value >= 5 else mask.depth == value)
        strata[f"depth:{label}"] = selector
    depth_three_four = (mask.depth == 3) | (mask.depth == 4)
    if depth_three_four.any():
        strata["depth:3-4"] = depth_three_four
    for value in np.unique(mask.full_degree):
        strata[f"degree:{int(value)}"] = mask.full_degree == value
    component_bins = {
        "1": mask.query_component_size == 1,
        "2-4": (mask.query_component_size >= 2) & (mask.query_component_size <= 4),
        "5-14": (mask.query_component_size >= 5) & (mask.query_component_size <= 14),
        "15+": mask.query_component_size >= 15,
    }
    for label, selector in component_bins.items():
        if selector.any():
            strata[f"component_size:{label}"] = selector
    return strata


@torch.no_grad()
def evaluate_mask(
    model: ProNORMST,
    slice_data: ProNORMSTSlice,
    mask: ProMask,
    gene_scale: np.ndarray,
    positive_weight: np.ndarray,
    detection_rate: np.ndarray,
    device: torch.device,
    *,
    use_amp: bool,
    round_limit: int | None = None,
    return_prediction: bool = False,
    return_auxiliary: bool = False,
) -> dict[str, Any]:
    visible = mask.visible_index
    query = mask.query_index
    visible_z = torch.as_tensor(
        slice_data.expression_z[visible], dtype=torch.float32, device=device
    ).unsqueeze(0)
    target_z = torch.as_tensor(
        slice_data.expression_z[query], dtype=torch.float32, device=device
    ).unsqueeze(0)
    weights = torch.as_tensor(positive_weight, dtype=torch.float32, device=device)
    amp_context = torch.amp.autocast(device_type=device.type, enabled=use_amp)
    with amp_context:
        output = model(
            visible_z,
            torch.as_tensor(visible, dtype=torch.long, device=device),
            torch.as_tensor(query, dtype=torch.long, device=device),
            slice_data.geometry(device),
            round_limit=round_limit,
            return_auxiliary=return_auxiliary,
            return_diagnostics=return_auxiliary,
        )
    prediction_z, auxiliary = output if isinstance(output, tuple) else (output, None)
    loss = weighted_gene_smooth_l1(prediction_z, target_z, weights)
    prediction_x = prediction_z.float().squeeze(0).cpu().numpy() * gene_scale[None, :]
    target_x = slice_data.expression_x[query]
    idw_x = strict_visible_idw(
        slice_data.expression_x[visible],
        slice_data.full_xy[visible],
        visible,
        slice_data.full_xy[query],
    )
    model_metrics = scientific_metrics(prediction_x, target_x)
    idw_metrics = scientific_metrics(idw_x, target_x)
    strata: dict[str, Any] = {}
    for name, selector in metric_strata(mask).items():
        strata[name] = {
            "model": scientific_metrics(prediction_x[selector], target_x[selector], min_queries=10),
            "idw": scientific_metrics(idw_x[selector], target_x[selector], min_queries=10),
            "n_queries": int(selector.sum()),
        }
        strata[name]["gain"] = paired_metric_gain(
            strata[name]["model"], strata[name]["idw"]
        )

    supported: dict[str, Any] = {}
    interior = (detection_rate > 0) & (detection_rate < 1)
    if interior.any():
        supported["both_positive_and_zero"] = scientific_metrics(
            prediction_x[:, interior], target_x[:, interior]
        )
    positive_supported = detection_rate > 0
    zero_supported = detection_rate < 1
    if positive_supported.any():
        supported["positive"] = scientific_metrics(
            prediction_x[:, positive_supported], target_x[:, positive_supported]
        )
    if zero_supported.any():
        supported["zero"] = scientific_metrics(
            prediction_x[:, zero_supported], target_x[:, zero_supported]
        )
    result: dict[str, Any] = {
        "weighted_z_smooth_l1": float(loss.item()),
        "model": model_metrics,
        "model_clipped_zero": scientific_metrics(
            np.maximum(prediction_x, 0.0), target_x
        ),
        "idw": idw_metrics,
        "gain": paired_metric_gain(model_metrics, idw_metrics),
        "strata": strata,
        "supported_genes": supported,
        "mask": mask.manifest(),
    }
    if return_prediction:
        result["prediction_z"] = prediction_z.float().squeeze(0).cpu().numpy()
        result["prediction_x"] = prediction_x.astype(np.float32)
        result["target_x"] = target_x.astype(np.float32)
        result["idw_x"] = idw_x.astype(np.float32)
    if return_auxiliary:
        result["auxiliary"] = auxiliary
    return result


def _mean_defined(records: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [record.get(key) for record in records]
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return float(np.mean(numeric)) if numeric else None


def aggregate_slice_mask_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-weight masks within one slice; require 8/16 for strata."""
    if not records:
        raise ValueError("cannot aggregate an empty mask record list")
    summary: dict[str, Any] = {
        "n_masks": len(records),
        "weighted_z_smooth_l1": _mean_defined(records, "weighted_z_smooth_l1"),
    }
    for section in ("model", "model_clipped_zero", "idw", "gain"):
        keys = sorted({key for record in records for key in record[section]})
        summary[section] = {
            key: _mean_defined([record[section] for record in records], key) for key in keys
        }
    stratum_names = sorted({name for record in records for name in record["strata"]})
    summary["strata"] = {}
    for name in stratum_names:
        valid = [
            record["strata"][name]
            for record in records
            if name in record["strata"]
            and record["strata"][name]["model"].get("coverage") == 1.0
        ]
        item: dict[str, Any] = {
            "valid_masks": len(valid),
            "coverage": len(valid) / len(records),
        }
        if len(valid) >= 8:
            for section in ("model", "idw", "gain"):
                keys = sorted({key for record in valid for key in record[section]})
                item[section] = {
                    key: _mean_defined([record[section] for record in valid], key)
                    for key in keys
                }
        summary["strata"][name] = item
    supported_names = sorted(
        {name for record in records for name in record.get("supported_genes", {})}
    )
    summary["supported_genes"] = {}
    for name in supported_names:
        values = [
            record["supported_genes"][name]
            for record in records
            if name in record.get("supported_genes", {})
        ]
        keys = sorted({key for value in values for key in value})
        summary["supported_genes"][name] = {
            key: _mean_defined(values, key) for key in keys
        }
    return summary


def mean_slice_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-weight the slices available in one role/family."""
    if not summaries:
        raise ValueError("cannot aggregate empty slice summaries")
    output: dict[str, Any] = {
        "n_slices": len(summaries),
        "weighted_z_smooth_l1": _mean_defined(summaries, "weighted_z_smooth_l1"),
    }
    for section in ("model", "model_clipped_zero", "idw", "gain"):
        keys = sorted({key for summary in summaries for key in summary[section]})
        output[section] = {
            key: _mean_defined([summary[section] for summary in summaries], key)
            for key in keys
        }
    stratum_names = sorted(
        {name for summary in summaries for name in summary.get("strata", {})}
    )
    output["strata"] = {}
    for name in stratum_names:
        eligible = [
            summary["strata"][name]
            for summary in summaries
            if name in summary.get("strata", {})
            and "model" in summary["strata"][name]
        ]
        item: dict[str, Any] = {
            "valid_slices": len(eligible),
            "coverage": len(eligible) / len(summaries),
        }
        if eligible:
            for section in ("model", "idw", "gain"):
                keys = sorted({key for value in eligible for key in value[section]})
                item[section] = {
                    key: _mean_defined([value[section] for value in eligible], key)
                    for key in keys
                }
        output["strata"][name] = item
    supported_names = sorted(
        {name for summary in summaries for name in summary.get("supported_genes", {})}
    )
    output["supported_genes"] = {}
    for name in supported_names:
        values = [
            summary["supported_genes"][name]
            for summary in summaries
            if name in summary.get("supported_genes", {})
        ]
        keys = sorted({key for value in values for key in value})
        output["supported_genes"][name] = {
            key: _mean_defined(values, key) for key in keys
        }
    return output


def learning_rate_for_step(step: int, *, max_steps: int = 3200) -> float:
    if step < 1 or step > max_steps:
        raise ValueError("optimizer step is outside the contracted budget")
    peak, minimum, warmup = 2e-5, 2e-6, 128
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def optimizer_for_model(model: ProNORMST) -> tuple[torch.optim.AdamW, dict[str, list[str]]]:
    decay, no_decay = [], []
    names = {"decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_embedding = "embedding" in name
        is_routing = name.endswith("routing_logits")
        use_decay = parameter.ndim >= 2 and name.endswith("weight") and not is_embedding and not is_routing
        group = "decay" if use_decay else "no_decay"
        (decay if use_decay else no_decay).append(parameter)
        names[group].append(name)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer, names


def diagnostic_summary(auxiliary: dict[str, Any], gradient_norm: float | None) -> dict[str, Any]:
    def tensor_mean(name: str) -> float | None:
        value = auxiliary.get(name)
        return float(value.float().mean().item()) if isinstance(value, torch.Tensor) else None

    global_value = auxiliary.get("global_normalized")
    local_value = auxiliary.get("gated_local")
    routing = auxiliary.get("routing_probability")
    summary: dict[str, Any] = {
        "gradient_norm_preclip": gradient_norm,
        "gate_mean": tensor_mean("gate"),
        "coverage_mean": tensor_mean("coverage"),
        "confidence_mean": tensor_mean("confidence"),
        "activation_round_mean": tensor_mean("activation_round"),
    }
    for label, key in (
        ("global_pre_norm_rms", "global_raw"),
        ("global_post_norm_rms", "global_normalized"),
        ("local_pre_norm_rms", "local_projected"),
        ("local_post_norm_rms", "local_normalized"),
        ("gated_local_rms", "gated_local"),
    ):
        value = auxiliary.get(key)
        if isinstance(value, torch.Tensor):
            summary[label] = float(value.detach().float().square().mean().sqrt().item())
    if isinstance(global_value, torch.Tensor):
        summary["global_rms"] = float(global_value.float().square().mean().sqrt().item())
    if isinstance(local_value, torch.Tensor):
        summary["gated_local_rms"] = float(local_value.float().square().mean().sqrt().item())
    if isinstance(global_value, torch.Tensor) and isinstance(local_value, torch.Tensor):
        denominator = global_value.float().norm().clamp_min(1e-12)
        summary["gated_local_global_norm_ratio"] = float(
            (local_value.float().norm() / denominator).item()
        )
    if isinstance(routing, torch.Tensor):
        probability = routing.float().clamp_min(1e-12)
        summary["routing_entropy"] = float(
            (-(probability * probability.log()).sum(dim=-1).mean()).item()
        )
        summary["routing_head_utilization"] = probability.mean(dim=0).detach().cpu().tolist()
        summary["routing_channel_max_probability_mean"] = float(
            probability.max(dim=-1).values.mean().item()
        )
    local_state = auxiliary.get("local_state")
    if isinstance(local_state, torch.Tensor):
        values = local_state.detach().float().reshape(-1, local_state.shape[-1])
        active = auxiliary.get("active_query")
        if isinstance(active, torch.Tensor):
            selector = active.detach().reshape(-1).to(torch.bool)
            values = values[selector]
        if values.numel():
            summary["local_state_norm_mean"] = float(values.norm(dim=-1).mean().item())
            summary["local_state_variance_mean"] = float(values.var(dim=0, unbiased=False).mean().item())
            centered = values[:512] - values[:512].mean(dim=0, keepdim=True)
            singular = torch.linalg.svdvals(centered)
            energy = singular.square()
            probability = energy / energy.sum().clamp_min(1e-12)
            summary["local_state_effective_rank"] = float(
                torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()).item()
            )
    global_diagnostics = auxiliary.get("global_diagnostics", {})
    if isinstance(global_diagnostics, dict) and "attention" in global_diagnostics:
        attention = global_diagnostics["attention"].float()
        distance = global_diagnostics["normalized_distance"].float()
        summary["attention_read_distance"] = float(
            (attention * distance[:, None]).sum(dim=-1).mean().item()
        )
        attention_probability = attention.clamp_min(1e-12)
        summary["attention_entropy"] = float(
            (-(attention_probability * attention_probability.log()).sum(dim=-1).mean()).item()
        )
    local_diagnostics = auxiliary.get("local_diagnostics", {})
    if isinstance(local_diagnostics, dict) and "round_states" in local_diagnostics:
        round_gradient = []
        for state in local_diagnostics["round_states"]:
            gradient = state.grad if isinstance(state, torch.Tensor) else None
            round_gradient.append(
                float(gradient.detach().float().norm().item())
                if isinstance(gradient, torch.Tensor)
                else None
            )
        summary["final_loss_round_state_gradient_norm"] = round_gradient
    if isinstance(local_diagnostics, dict) and "direction_attention" in local_diagnostics:
        direction = local_diagnostics["direction_attention"].detach().float()
        if direction.numel():
            mass = direction.sum(dim=(0, 1, 2))
            denominator = mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            summary["local_head_direction_utilization"] = (
                mass / denominator
            ).cpu().tolist()
            probability = direction.clamp_min(1e-12)
            entropy = -(probability * probability.log()).sum(dim=-1)
            valid = direction.sum(dim=-1) > 0
            summary["local_direction_entropy"] = (
                float(entropy[valid].mean().item()) if bool(valid.any()) else None
            )
    return summary


__all__ = [
    "aggregate_slice_mask_records",
    "canonical_json",
    "canonical_sha256",
    "capture_rng_state",
    "diagnostic_summary",
    "evaluate_mask",
    "file_sha256",
    "learning_rate_for_step",
    "mean_slice_summaries",
    "metric_strata",
    "optimizer_for_model",
    "paired_metric_gain",
    "restore_rng_state",
    "scientific_metrics",
    "seed_initialization",
    "strict_visible_idw",
    "weighted_gene_smooth_l1",
]
