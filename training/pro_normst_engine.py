"""Loss, baseline, metrics, and contract utilities for ProNORMST training."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from datasets.pro_normst import ProNORMSTSlice
from models.pro_normst import FullHexGeometry, ProNORMST
from training.pro_normst_masks import ProMask


EvaluationControlCache = dict[tuple[int, str], dict[str, Any]]
IDW_CACHE_SCHEMA = "pro-normst-strict-idw-cache-v1"

# Round-level optimization constants.  Keep the schedule and optimizer
# construction on the same source of truth so the recorded contract cannot
# drift from the executable values.
PEAK_LEARNING_RATE = 5e-5
MINIMUM_LEARNING_RATE = 5e-6
WARMUP_STEPS = 128
MATRIX_WEIGHT_DECAY = 1e-4
OTHER_WEIGHT_DECAY = 0.0


@dataclass(frozen=True)
class PaddedModelBatch:
    """Variable-length complete-slice inputs packed without changing indices."""

    visible_z: torch.Tensor
    visible_index: torch.Tensor
    query_index: torch.Tensor
    target_z: torch.Tensor
    query_valid: torch.Tensor
    geometry: FullHexGeometry
    query_lengths: tuple[int, ...]


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


def _numpy_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class PersistentIDWCache:
    """Content-addressed strict-IDW predictions reusable across matched runs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._slice_identities: dict[int, tuple[ProNORMSTSlice, dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _slice_identity(self, slice_data: ProNORMSTSlice) -> dict[str, Any]:
        object_id = id(slice_data)
        cached = self._slice_identities.get(object_id)
        if cached is not None and cached[0] is slice_data:
            return cached[1]
        identity = {
            "slice_id": slice_data.slice_id,
            "expression_x_sha256": _numpy_array_sha256(slice_data.expression_x),
            "full_xy_sha256": _numpy_array_sha256(slice_data.full_xy),
        }
        self._slice_identities[object_id] = (slice_data, identity)
        return identity

    def _key_payload(
        self,
        slice_data: ProNORMSTSlice,
        mask: ProMask,
    ) -> dict[str, Any]:
        return {
            "schema": IDW_CACHE_SCHEMA,
            "baseline": "strict-original-visible-idw",
            "neighbors": 6,
            "power": 2.0,
            "slice": self._slice_identity(slice_data),
            "visible_index_sha256": _numpy_array_sha256(mask.visible_index),
            "query_index_sha256": _numpy_array_sha256(mask.query_index),
        }

    @staticmethod
    def _load(
        path: Path,
        key: str,
        key_payload: dict[str, Any],
    ) -> np.ndarray:
        try:
            with np.load(path, allow_pickle=False) as archive:
                manifest = json.loads(str(archive["manifest_json"].item()))
                prediction = np.asarray(archive["prediction_x"], dtype=np.float32)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"strict IDW cache artifact is invalid: {path}") from error
        valid = (
            manifest.get("schema") == IDW_CACHE_SCHEMA
            and manifest.get("key") == key
            and manifest.get("key_payload") == key_payload
            and manifest.get("prediction_shape") == list(prediction.shape)
            and manifest.get("prediction_dtype") == prediction.dtype.str
            and manifest.get("prediction_sha256")
            == _numpy_array_sha256(prediction)
        )
        if not valid:
            raise ValueError(f"strict IDW cache artifact failed validation: {path}")
        prediction.setflags(write=False)
        return prediction

    @staticmethod
    def _write(
        path: Path,
        key: str,
        key_payload: dict[str, Any],
        prediction: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        value = np.ascontiguousarray(prediction, dtype=np.float32)
        manifest = {
            "schema": IDW_CACHE_SCHEMA,
            "key": key,
            "key_payload": key_payload,
            "prediction_shape": list(value.shape),
            "prediction_dtype": value.dtype.str,
            "prediction_sha256": _numpy_array_sha256(value),
        }
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                np.savez(
                    handle,
                    manifest_json=np.asarray(canonical_json(manifest)),
                    prediction_x=value,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def get_or_compute(
        self,
        slice_data: ProNORMSTSlice,
        mask: ProMask,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        key_payload = self._key_payload(slice_data, mask)
        key = canonical_sha256(key_payload)
        parent = self.root / key[:2]
        path = parent / f"{key}.npz"
        lock_path = parent / f"{key}.lock"
        parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if path.exists():
                    value = self._load(path, key, key_payload)
                    self.hits += 1
                    return value
                self.misses += 1
                value = np.asarray(compute(), dtype=np.float32)
                expected_shape = (mask.query_index.size, slice_data.expression_x.shape[1])
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise ValueError("strict IDW cache producer returned an invalid prediction")
                self._write(path, key, key_payload, value)
                self.writes += 1
                return self._load(path, key, key_payload)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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
    # Checkpoints are loaded with ``map_location=device`` so CUDA resume can
    # move every tensor, including CPU-generator states, onto the GPU.  PyTorch
    # requires these opaque ByteTensor states on CPU when restoring the CPU and
    # data-order generators.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    data_order_generator.set_state(state["data_order"].cpu())


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


def weighted_gene_smooth_l1_per_item(
    prediction_z: torch.Tensor,
    target_z: torch.Tensor,
    positive_weight: torch.Tensor,
    query_valid: torch.Tensor,
) -> torch.Tensor:
    """Return gene-equal losses per padded item without query pooling."""
    prediction = prediction_z.float()
    target = target_z.detach().float()
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction_z and target_z must be matching [B,Nq,G] tensors")
    if query_valid.shape != prediction.shape[:2]:
        raise ValueError("query_valid must align with [B,Nq]")
    weight = positive_weight.float().reshape(1, 1, -1)
    if weight.shape[-1] != prediction.shape[-1]:
        raise ValueError("positive_weight does not match the gene dimension")
    valid = query_valid[..., None].to(prediction.dtype)
    element_weight = torch.where(target > 0, weight, torch.ones_like(target))
    element_loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=1.0,
    )
    numerator = (element_loss * element_weight * valid).sum(dim=1)
    denominator = (element_weight * valid).sum(dim=1)
    if bool((denominator <= 0).any()):
        raise RuntimeError("every batch item must have a positive gene loss denominator")
    losses = (numerator / denominator).mean(dim=-1)
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("batched weighted SmoothL1 is non-finite")
    return losses


def build_padded_model_batch(
    slices: Sequence[ProNORMSTSlice],
    masks: Sequence[ProMask],
    device: torch.device,
) -> PaddedModelBatch:
    """Pack complete slice-mask pairs with ``-1`` compact-index padding."""
    if not slices or len(slices) != len(masks):
        raise ValueError("slices and masks must be matching non-empty sequences")
    batch = len(slices)
    visible_lengths = tuple(int(mask.visible_index.size) for mask in masks)
    query_lengths = tuple(int(mask.query_index.size) for mask in masks)
    if min(*visible_lengths, *query_lengths) < 1:
        raise ValueError("every packed mask must contain visible and query nodes")
    max_visible = max(visible_lengths)
    max_query = max(query_lengths)
    max_nodes = max(item.n_nodes for item in slices)

    first_expression = slices[0].expression_z_tensor(device)
    visible_z = first_expression.new_zeros(batch, max_visible, first_expression.shape[1])
    target_z = first_expression.new_zeros(batch, max_query, first_expression.shape[1])
    visible_index = torch.full(
        (batch, max_visible),
        -1,
        dtype=torch.long,
        device=device,
    )
    query_index = torch.full(
        (batch, max_query),
        -1,
        dtype=torch.long,
        device=device,
    )
    query_valid = torch.zeros(batch, max_query, dtype=torch.bool, device=device)

    same_slice = all(item is slices[0] for item in slices)
    if same_slice:
        geometry = slices[0].geometry(device)
    else:
        xy = first_expression.new_zeros(batch, max_nodes, 2)
        neighbor_index = torch.full(
            (batch, max_nodes, 6),
            -1,
            dtype=torch.long,
            device=device,
        )
        node_mask = torch.zeros(batch, max_nodes, dtype=torch.bool, device=device)
        native_scale = torch.empty(batch, dtype=torch.float32, device=device)

    for offset, (item, mask) in enumerate(zip(slices, masks, strict=True)):
        expression = item.expression_z_tensor(device)
        if expression.shape != (item.n_nodes, first_expression.shape[1]):
            raise ValueError("packed slices must share the contracted gene dimension")
        visible = torch.as_tensor(mask.visible_index, dtype=torch.long, device=device)
        query = torch.as_tensor(mask.query_index, dtype=torch.long, device=device)
        n_visible = visible_lengths[offset]
        n_query = query_lengths[offset]
        visible_index[offset, :n_visible] = visible
        query_index[offset, :n_query] = query
        query_valid[offset, :n_query] = True
        visible_z[offset, :n_visible] = expression.index_select(0, visible)
        target_z[offset, :n_query] = expression.index_select(0, query)

        if not same_slice:
            source_geometry = item.geometry(device)
            if source_geometry.xy.ndim != 2 or source_geometry.neighbor_index.ndim != 2:
                raise ValueError("resident slice geometry must be unbatched")
            nodes = item.n_nodes
            xy[offset, :nodes] = source_geometry.xy
            neighbor_index[offset, :nodes] = source_geometry.neighbor_index
            if source_geometry.node_mask is None:
                node_mask[offset, :nodes] = True
            else:
                node_mask[offset, :nodes] = source_geometry.node_mask
            if source_geometry.native_scale is None:
                raise ValueError("resident slice geometry must define native_scale")
            native_scale[offset] = source_geometry.native_scale

    if not same_slice:
        geometry = FullHexGeometry(
            xy=xy,
            neighbor_index=neighbor_index,
            node_mask=node_mask,
            native_scale=native_scale,
            indices_validated=True,
        )
    return PaddedModelBatch(
        visible_z=visible_z,
        visible_index=visible_index,
        query_index=query_index,
        target_z=target_z,
        query_valid=query_valid,
        geometry=geometry,
        query_lengths=query_lengths,
    )


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
        if take == values.shape[0]:
            candidate = np.arange(values.shape[0], dtype=np.int64)
        else:
            partition = np.argpartition(distance, take - 1)
            boundary = distance[partition[take - 1]]
            candidate = np.flatnonzero(distance <= boundary)
        candidate_order = np.lexsort(
            (canonical[candidate], distance[candidate])
        )
        order = candidate[candidate_order[:take]]
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


def _scientific_metrics_for_selectors(
    prediction_x: np.ndarray,
    target_x: np.ndarray,
    selectors: dict[str, np.ndarray],
    *,
    axis: int,
    min_queries: int,
    full_metrics: dict[str, float | int | None],
) -> dict[str, dict[str, float | int | None]]:
    """Evaluate unique row/gene selectors once while preserving exact values."""
    prediction = np.asarray(prediction_x, dtype=np.float32)
    target = np.asarray(target_x, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("metric arrays must have matching [Nq,G] shape")
    if axis not in {0, 1}:
        raise ValueError("metric selector axis must be 0 or 1")
    selector_size = prediction.shape[axis]
    all_selector = np.ones(selector_size, dtype=bool)
    cache: dict[bytes, dict[str, float | int | None]] = {}
    # Row boolean indexing preserves the reduction layout used by the full
    # metric call.  Column boolean indexing does not: even an all-True gene
    # selector can change NumPy's reduction order and last-bit results.  Seed
    # only the row cache from ``full_metrics``; duplicate gene selectors still
    # reuse the first explicitly selected computation.
    if axis == 0 and prediction.shape[0] >= min_queries:
        cache[all_selector.tobytes()] = full_metrics
    result: dict[str, dict[str, float | int | None]] = {}
    for name, raw_selector in selectors.items():
        selector = np.asarray(raw_selector, dtype=bool)
        if selector.shape != (selector_size,):
            raise ValueError(f"metric selector {name!r} has an invalid shape")
        key = np.ascontiguousarray(selector).tobytes()
        metrics = cache.get(key)
        if metrics is None:
            selected_prediction = (
                prediction[selector] if axis == 0 else prediction[:, selector]
            )
            selected_target = target[selector] if axis == 0 else target[:, selector]
            metrics = scientific_metrics(
                selected_prediction,
                selected_target,
                min_queries=min_queries,
            )
            cache[key] = metrics
        result[name] = dict(metrics)
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


def _fixed_mask_control(
    slice_data: ProNORMSTSlice,
    mask: ProMask,
    target_x: np.ndarray,
    selectors: dict[str, np.ndarray],
    control_cache: EvaluationControlCache | None,
    *,
    retain_idw: bool,
    persistent_idw_cache: PersistentIDWCache | None,
) -> dict[str, Any]:
    """Return immutable IDW controls, reusing exact fixed-mask results."""

    def compute_idw() -> np.ndarray:
        value = strict_visible_idw(
            slice_data.expression_x[mask.visible_index],
            slice_data.full_xy[mask.visible_index],
            mask.visible_index,
            slice_data.full_xy[mask.query_index],
        )
        value.setflags(write=False)
        return value

    def load_idw() -> np.ndarray:
        return (
            compute_idw()
            if persistent_idw_cache is None
            else persistent_idw_cache.get_or_compute(slice_data, mask, compute_idw)
        )

    cache_key = (id(slice_data.expression_x), canonical_json(mask.identity))
    if control_cache is not None and cache_key in control_cache:
        cached = control_cache[cache_key]
        if not np.array_equal(
            cached["visible_index"], mask.visible_index
        ) or not np.array_equal(
            cached["query_index"], mask.query_index
        ):
            raise ValueError("fixed-mask control cache identity collision")
        if retain_idw and cached["idw_x"] is None:
            cached["idw_x"] = load_idw()
        return cached

    idw_x = load_idw()
    idw_metrics = scientific_metrics(idw_x, target_x)
    control = {
        "visible_index": mask.visible_index.copy(),
        "query_index": mask.query_index.copy(),
        "idw_x": idw_x if retain_idw else None,
        "metrics": idw_metrics,
        "strata": _scientific_metrics_for_selectors(
            idw_x,
            target_x,
            selectors,
            axis=0,
            min_queries=10,
            full_metrics=idw_metrics,
        ),
    }
    if control_cache is not None:
        control_cache[cache_key] = control
    return control


@torch.no_grad()
def evaluate_masks(
    model: ProNORMST,
    slice_data: ProNORMSTSlice,
    masks: Sequence[ProMask],
    gene_scale: np.ndarray,
    positive_weight: np.ndarray | torch.Tensor,
    detection_rate: np.ndarray,
    device: torch.device,
    *,
    use_amp: bool,
    round_limit: int | None = None,
    return_prediction: bool = False,
    return_auxiliary: bool = False,
    control_cache: EvaluationControlCache | None = None,
    persistent_idw_cache: PersistentIDWCache | None = None,
) -> list[dict[str, Any]]:
    mask_batch = tuple(masks)
    if not mask_batch:
        raise ValueError("evaluation mask batch must not be empty")
    if return_auxiliary and len(mask_batch) != 1:
        raise ValueError("auxiliary evaluation is restricted to one mask")
    packed = build_padded_model_batch(
        [slice_data] * len(mask_batch),
        mask_batch,
        device,
    )
    weights = torch.as_tensor(positive_weight, dtype=torch.float32, device=device)
    amp_context = torch.amp.autocast(device_type=device.type, enabled=use_amp)
    with amp_context:
        output = model(
            packed.visible_z,
            packed.visible_index,
            packed.query_index,
            packed.geometry,
            round_limit=round_limit,
            return_auxiliary=return_auxiliary,
            return_diagnostics=return_auxiliary,
        )
    prediction_z, auxiliary = output if isinstance(output, tuple) else (output, None)
    losses = weighted_gene_smooth_l1_per_item(
        prediction_z,
        packed.target_z,
        weights,
        packed.query_valid,
    ).detach().cpu().tolist()
    predictions = prediction_z.detach().float().cpu().numpy()
    results: list[dict[str, Any]] = []
    for offset, mask in enumerate(mask_batch):
        query = mask.query_index
        query_length = packed.query_lengths[offset]
        prediction_z_item = predictions[offset, :query_length]
        prediction_x = prediction_z_item * gene_scale[None, :]
        target_x = slice_data.expression_x[query]
        selectors = metric_strata(mask)
        control = _fixed_mask_control(
            slice_data,
            mask,
            target_x,
            selectors,
            control_cache,
            retain_idw=return_prediction,
            persistent_idw_cache=persistent_idw_cache,
        )
        model_metrics = scientific_metrics(prediction_x, target_x)
        idw_metrics = dict(control["metrics"])
        model_strata = _scientific_metrics_for_selectors(
            prediction_x,
            target_x,
            selectors,
            axis=0,
            min_queries=10,
            full_metrics=model_metrics,
        )
        strata: dict[str, Any] = {}
        for name, selector in selectors.items():
            strata[name] = {
                "model": model_strata[name],
                "idw": dict(control["strata"][name]),
                "n_queries": int(selector.sum()),
            }
            strata[name]["gain"] = paired_metric_gain(
                strata[name]["model"], strata[name]["idw"]
            )

        supported_selectors: dict[str, np.ndarray] = {}
        interior = (detection_rate > 0) & (detection_rate < 1)
        if interior.any():
            supported_selectors["both_positive_and_zero"] = interior
        positive_supported = detection_rate > 0
        zero_supported = detection_rate < 1
        if positive_supported.any():
            supported_selectors["positive"] = positive_supported
        if zero_supported.any():
            supported_selectors["zero"] = zero_supported
        supported = _scientific_metrics_for_selectors(
            prediction_x,
            target_x,
            supported_selectors,
            axis=1,
            min_queries=1,
            full_metrics=model_metrics,
        )
        clipped_metrics = (
            dict(model_metrics)
            if model_metrics.get("negative_fraction") == 0.0
            else scientific_metrics(np.maximum(prediction_x, 0.0), target_x)
        )
        result: dict[str, Any] = {
            "weighted_z_smooth_l1": float(losses[offset]),
            "model": model_metrics,
            "model_clipped_zero": clipped_metrics,
            "idw": idw_metrics,
            "gain": paired_metric_gain(model_metrics, idw_metrics),
            "strata": strata,
            "supported_genes": supported,
            "mask": mask.manifest(),
        }
        if return_prediction:
            idw_x = control["idw_x"]
            if not isinstance(idw_x, np.ndarray):
                raise RuntimeError(
                    "fixed-mask control cache did not retain IDW prediction"
                )
            result["prediction_z"] = prediction_z_item
            result["prediction_x"] = prediction_x.astype(np.float32)
            result["target_x"] = target_x.astype(np.float32)
            result["idw_x"] = idw_x.astype(np.float32)
        if return_auxiliary:
            result["auxiliary"] = auxiliary
        results.append(result)
    return results


def evaluate_mask(
    model: ProNORMST,
    slice_data: ProNORMSTSlice,
    mask: ProMask,
    gene_scale: np.ndarray,
    positive_weight: np.ndarray | torch.Tensor,
    detection_rate: np.ndarray,
    device: torch.device,
    *,
    use_amp: bool,
    round_limit: int | None = None,
    return_prediction: bool = False,
    return_auxiliary: bool = False,
    control_cache: EvaluationControlCache | None = None,
    persistent_idw_cache: PersistentIDWCache | None = None,
) -> dict[str, Any]:
    return evaluate_masks(
        model,
        slice_data,
        (mask,),
        gene_scale,
        positive_weight,
        detection_rate,
        device,
        use_amp=use_amp,
        round_limit=round_limit,
        return_prediction=return_prediction,
        return_auxiliary=return_auxiliary,
        control_cache=control_cache,
        persistent_idw_cache=persistent_idw_cache,
    )[0]


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
    peak = PEAK_LEARNING_RATE
    minimum = MINIMUM_LEARNING_RATE
    warmup = WARMUP_STEPS
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
            {"params": decay, "weight_decay": MATRIX_WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": OTHER_WEIGHT_DECAY},
        ],
        lr=PEAK_LEARNING_RATE,
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
    global_input = auxiliary.get("global_input_base")
    global_encoded_input = auxiliary.get("global_input_state")
    global_input_residual = auxiliary.get("global_input_residual")
    if isinstance(global_input, torch.Tensor):
        input_rms = global_input.detach().float().square().mean().sqrt()
        summary["global_input_rms"] = float(input_rms.item())
    else:
        input_rms = None
    if isinstance(global_input_residual, torch.Tensor):
        residual_rms = global_input_residual.detach().float().square().mean().sqrt()
        summary["global_input_residual_rms"] = float(residual_rms.item())
        if isinstance(input_rms, torch.Tensor):
            summary["global_input_residual_ratio"] = float(
                (residual_rms / input_rms.clamp_min(1e-12)).item()
            )
    if isinstance(global_encoded_input, torch.Tensor):
        summary["global_encoded_input_rms"] = float(
            global_encoded_input.detach().float().square().mean().sqrt().item()
        )
    local_gene_residual = auxiliary.get("local_gene_residual")
    shared_decoder_prediction = auxiliary.get("shared_decoder_prediction")
    if isinstance(local_gene_residual, torch.Tensor):
        residual_value = local_gene_residual.detach().float()
        residual_rms = residual_value.square().mean().sqrt()
        summary["local_gene_residual_rms"] = float(residual_rms.item())
        summary["local_gene_residual_finite"] = bool(
            torch.isfinite(residual_value).all()
        )
        if isinstance(shared_decoder_prediction, torch.Tensor):
            base_rms = (
                shared_decoder_prediction.detach().float().square().mean().sqrt()
            )
            summary["local_gene_residual_shared_prediction_rms_ratio"] = float(
                (residual_rms / base_rms.clamp_min(1e-12)).item()
            )
    active_query = auxiliary.get("active_query")
    query_valid = auxiliary.get("query_valid")
    active_selector = None
    if isinstance(active_query, torch.Tensor) and isinstance(query_valid, torch.Tensor):
        active_selector = active_query.detach().to(torch.bool) & query_valid.detach().to(
            torch.bool
        )
    if isinstance(global_value, torch.Tensor):
        summary["global_rms"] = float(global_value.float().square().mean().sqrt().item())
    if isinstance(local_value, torch.Tensor):
        summary["gated_local_rms"] = float(local_value.float().square().mean().sqrt().item())
    if isinstance(global_value, torch.Tensor) and isinstance(local_value, torch.Tensor):
        denominator = global_value.float().norm().clamp_min(1e-12)
        summary["gated_local_global_norm_ratio"] = float(
            (local_value.float().norm() / denominator).item()
        )
    local_normalized = auxiliary.get("local_normalized")
    if (
        isinstance(global_value, torch.Tensor)
        and isinstance(local_normalized, torch.Tensor)
        and isinstance(active_selector, torch.Tensor)
        and bool(active_selector.any())
    ):
        global_rows = global_value.detach().float()[active_selector]
        local_rows = local_normalized.detach().float()[active_selector]
        if global_rows.shape[0] >= 2:
            global_rows = global_rows - global_rows.mean(dim=0, keepdim=True)
            local_rows = local_rows - local_rows.mean(dim=0, keepdim=True)
            cross_energy = (global_rows.transpose(0, 1) @ local_rows).square().sum()
            global_energy = (
                global_rows.transpose(0, 1) @ global_rows
            ).square().sum()
            local_energy = (
                local_rows.transpose(0, 1) @ local_rows
            ).square().sum()
            denominator = (global_energy * local_energy).sqrt()
            summary["global_local_normalized_linear_cka"] = (
                float((cross_energy / denominator).item())
                if float(denominator.item()) > 0.0
                else None
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
    def selected_local_rows(name: str) -> torch.Tensor | None:
        value = auxiliary.get(name)
        if not isinstance(value, torch.Tensor):
            return None
        rows = value.detach().float().reshape(-1, value.shape[-1])
        if isinstance(active_selector, torch.Tensor):
            rows = rows[active_selector.reshape(-1)]
        return rows if rows.numel() else None

    def add_state_statistics(prefix: str, values: torch.Tensor | None) -> None:
        if values is None:
            return
        summary[f"{prefix}_norm_mean"] = float(values.norm(dim=-1).mean().item())
        summary[f"{prefix}_variance_mean"] = float(
            values.var(dim=0, unbiased=False).mean().item()
        )
        centered = values[:512] - values[:512].mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        energy = singular.square()
        if float(energy.sum().item()) > 0.0:
            probability = energy / energy.sum()
            summary[f"{prefix}_effective_rank"] = float(
                torch.exp(
                    -(probability * probability.clamp_min(1e-12).log()).sum()
                ).item()
            )
        else:
            summary[f"{prefix}_effective_rank"] = None

    local_state = selected_local_rows("local_state")
    local_state_enhanced = selected_local_rows("local_state_enhanced")
    local_state_residual = selected_local_rows("local_state_residual")
    add_state_statistics("local_state", local_state)
    add_state_statistics("local_state_enhanced", local_state_enhanced)
    if local_state_residual is not None:
        residual_rms = local_state_residual.square().mean().sqrt()
        summary["local_state_enhancer_residual_rms"] = float(residual_rms.item())
        if local_state is not None:
            input_rms = local_state.square().mean().sqrt().clamp_min(1e-12)
            summary["local_state_enhancer_residual_ratio"] = float(
                (residual_rms / input_rms).item()
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
    "EvaluationControlCache",
    "IDW_CACHE_SCHEMA",
    "MATRIX_WEIGHT_DECAY",
    "MINIMUM_LEARNING_RATE",
    "OTHER_WEIGHT_DECAY",
    "PEAK_LEARNING_RATE",
    "PaddedModelBatch",
    "PersistentIDWCache",
    "WARMUP_STEPS",
    "aggregate_slice_mask_records",
    "build_padded_model_batch",
    "canonical_json",
    "canonical_sha256",
    "capture_rng_state",
    "diagnostic_summary",
    "evaluate_mask",
    "evaluate_masks",
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
    "weighted_gene_smooth_l1_per_item",
]
