"""Leakage-safe multi-slice preparation for standard 10x Visium."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from datasets.visium_common import (
    VisiumSliceData,
    as_csr,
    read_standard_visium,
    spot_geometry,
)


@dataclass
class PreparedVisiumSlice:
    name: str
    role: str
    path: str
    data: VisiumSliceData
    array_row: np.ndarray
    array_col: np.ndarray


@dataclass
class MultiSliceVisiumData:
    slices: list[PreparedVisiumSlice]
    genes: np.ndarray
    gene_scale: np.ndarray
    target_sum: float
    expression_transform: str
    manifest_path: str

    def for_role(self, role: str) -> list[PreparedVisiumSlice]:
        return [item for item in self.slices if item.role == role]


@dataclass
class _RawSlice:
    name: str
    role: str
    path: str
    adata: Any
    raw_matrix: Any
    library_size: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    row_map: np.ndarray
    physical_xy: np.ndarray
    array_row: np.ndarray
    array_col: np.ndarray
    observed_spots: np.ndarray


def _manifest_entries(manifest_path: str, default_count_file: str):
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("multi-slice manifest must be a JSON object")
    entries = []
    for role in ("train", "val", "test"):
        group = payload.get(role)
        if not group:
            raise ValueError(f"manifest requires at least one {role} slice")
        normalized = []
        if isinstance(group, dict):
            for name, value in group.items():
                if isinstance(value, str):
                    normalized.append({"name": name, "path": value})
                elif isinstance(value, dict):
                    normalized.append({"name": name, **value})
                else:
                    raise ValueError(f"invalid manifest entry for {role}/{name}")
        elif isinstance(group, list):
            normalized = group
        else:
            raise ValueError(f"manifest {role} must be an object or list")

        for item in normalized:
            if not isinstance(item, dict):
                raise ValueError(f"manifest {role} list entries must be objects")
            name = item.get("name", item.get("id"))
            data_path = item.get("path")
            if not name or not data_path:
                raise ValueError(f"manifest {role} entries require name and path")
            resolved = Path(data_path)
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            entries.append({
                "name": str(name),
                "role": role,
                "path": str(resolved),
                "count_file": item.get("count_file", default_count_file),
            })
    names = [entry["name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError(
            "slice names must be unique across train/val/test; use distinct "
            "names for a slice-level generalization protocol"
        )
    paths = [str(Path(entry["path"]).resolve()) for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError(
            "the same slice path cannot appear in multiple manifest roles; "
            "this would leak a training slice into slice-level evaluation"
        )
    return entries


def _load_raw_slices(entries):
    raw_slices = []
    for entry in entries:
        adata = read_standard_visium(entry["path"], entry["count_file"])
        keep = ~adata.var_names.str.startswith("DEPRECATED_")
        adata = adata[:, keep].copy()
        rows, cols, row_map, physical_xy = spot_geometry(adata)
        raw_matrix = as_csr(adata.X)
        library_size = np.asarray(raw_matrix.sum(axis=1)).ravel().astype(np.float32)
        all_spots = np.arange(len(rows), dtype=np.int64)
        array_row = adata.obs["array_row"].to_numpy(np.int64).copy()
        array_col = adata.obs["array_col"].to_numpy(np.int64).copy()
        raw_slices.append(_RawSlice(
            name=entry["name"],
            role=entry["role"],
            path=entry["path"],
            adata=adata,
            raw_matrix=raw_matrix,
            library_size=library_size,
            rows=rows,
            cols=cols,
            row_map=row_map,
            physical_xy=physical_xy,
            array_row=array_row,
            array_col=array_col,
            observed_spots=all_spots,
        ))
    return raw_slices


def _common_genes(raw_slices: list[_RawSlice]):
    availability = [set(item.adata.var_names) for item in raw_slices]
    first_train = next(item for item in raw_slices if item.role == "train")
    common = np.asarray([
        gene for gene in first_train.adata.var_names
        if all(gene in available for available in availability)
    ])
    if not len(common):
        raise ValueError("the manifest slices have no common genes")
    return common


def _select_training_hvgs(
    raw_slices: list[_RawSlice],
    common_genes: np.ndarray,
    n_genes: int,
):
    if n_genes < 1:
        raise ValueError("n_genes must be positive")
    if n_genes > len(common_genes):
        raise ValueError(
            f"requested {n_genes} genes but only {len(common_genes)} are "
            "shared by every manifest slice"
        )
    if n_genes == len(common_genes):
        return common_genes.copy()
    training = [item for item in raw_slices if item.role == "train"]
    sources = [item.adata[:, common_genes].copy() for item in training]
    pooled = sc.concat(
        sources,
        join="inner",
        label="slice_id",
        keys=[item.name for item in training],
        index_unique="-",
    )
    sc.pp.highly_variable_genes(
        pooled,
        flavor="seurat_v3_paper",
        n_top_genes=n_genes,
        batch_key="slice_id",
        inplace=True,
    )
    genes = pooled.var_names[pooled.var["highly_variable"]].to_numpy()
    if len(genes) != min(n_genes, len(common_genes)):
        raise ValueError("HVG selection returned an unexpected gene count")
    return genes


def prepare_multislice_visium(
    manifest_path: str,
    count_file: str = "filtered_feature_bc_matrix.h5",
    n_genes: int = 1000,
    target_sum: float = 1e4,
    seed: int = 2026,
    apply_rms_scale: bool = True,
    fixed_genes: np.ndarray | None = None,
    fixed_gene_scale: np.ndarray | None = None,
) -> MultiSliceVisiumData:
    """Prepare slice-isolated data without fitting on val/test expression.

    A positive ``target_sum`` applies library-size normalization before
    ``log1p``; zero or a negative value skips it and applies ``log1p`` directly
    to selected raw counts. When ``apply_rms_scale`` is false, ``gene_scale``
    is stored as ones for artifact compatibility.
    """
    if fixed_gene_scale is not None and fixed_genes is None:
        raise ValueError("fixed_gene_scale requires fixed_genes")
    entries = _manifest_entries(manifest_path, count_file)
    raw_slices = _load_raw_slices(entries)
    common = _common_genes(raw_slices)
    if fixed_genes is None:
        genes = _select_training_hvgs(raw_slices, common, n_genes)
    else:
        genes = np.asarray(fixed_genes, dtype=str)
        if genes.ndim != 1 or len(genes) < 1:
            raise ValueError("fixed_genes must be a non-empty one-dimensional array")
        if len(np.unique(genes)) != len(genes):
            raise ValueError("fixed_genes must not contain duplicates")
        missing = np.setdiff1d(genes, common)
        if len(missing):
            raise ValueError(
                f"fixed genes are missing from one or more slices: {missing[:5].tolist()}"
            )

    unscaled = []
    squared_sum = np.zeros(len(genes), dtype=np.float64)
    training_count = 0
    for item in raw_slices:
        gene_indices = item.adata.var_names.get_indexer(genes)
        if np.any(gene_indices < 0):
            raise ValueError(f"selected genes are missing from slice {item.name}")
        counts = item.raw_matrix[:, gene_indices].toarray().astype(np.float32)
        if target_sum > 0:
            expression = np.log1p(
                counts * (
                    target_sum / np.maximum(item.library_size, 1.0)
                )[:, None]
            ).astype(np.float32, copy=False)
        else:
            expression = np.log1p(counts).astype(np.float32, copy=False)
        unscaled.append(expression)
        if apply_rms_scale and item.role == "train":
            selected = expression.astype(np.float64)
            squared_sum += np.square(selected).sum(axis=0)
            training_count += len(selected)
    if fixed_gene_scale is not None:
        gene_scale = np.asarray(fixed_gene_scale, dtype=np.float32)
        if gene_scale.shape != (len(genes),):
            raise ValueError("fixed_gene_scale must align with fixed_genes")
        if not np.isfinite(gene_scale).all() or np.any(gene_scale <= 0):
            raise ValueError("fixed_gene_scale must be finite and positive")
    elif apply_rms_scale:
        if training_count < 1:
            raise ValueError("no training spots are available for RMS fitting")
        gene_scale = np.sqrt(squared_sum / training_count).astype(np.float32)
        gene_scale = np.maximum(gene_scale, 1e-6)
    else:
        gene_scale = np.ones(len(genes), dtype=np.float32)

    prepared = []
    for item, expression in zip(raw_slices, unscaled):
        if apply_rms_scale:
            expression = expression / gene_scale[None, :]
        empty_index = np.empty((0,), dtype=np.int64)
        data = VisiumSliceData(
            expression=expression,
            genes=genes,
            gene_scale=gene_scale,
            row_map=item.row_map,
            spot_rows=item.rows,
            spot_cols=item.cols,
            physical_xy=item.physical_xy,
            observed_spots=item.observed_spots,
            validation_spots=empty_index,
            test_spots=empty_index,
        )
        prepared.append(PreparedVisiumSlice(
            name=item.name,
            role=item.role,
            path=item.path,
            data=data,
            array_row=item.array_row,
            array_col=item.array_col,
        ))
    return MultiSliceVisiumData(
        slices=prepared,
        genes=genes,
        gene_scale=gene_scale,
        target_sum=target_sum,
        expression_transform=(
            "log1p_library_size_normalized"
            if target_sum > 0 else "log1p_raw_counts"
        ),
        manifest_path=str(Path(manifest_path).resolve()),
    )


class _TaggedSliceDataset(Dataset):
    def __init__(self, dataset: Dataset, slice_index: int):
        self.dataset = dataset
        self.slice_index = slice_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["slice_index"] = torch.tensor(self.slice_index, dtype=torch.long)
        return item


class _WholeSliceMaskDataset(Dataset):
    """Generate deterministic masks from every spot in one assigned slice.

    The slice role is fixed by the manifest. Training masks change by epoch;
    validation and test masks stay fixed. No spot from another role is used.
    """

    _ROLE_SEED = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        data: VisiumSliceData,
        role: str,
        masks_per_slice: int,
        target_fraction: float,
        idw_neighbors: int,
        seed: int,
        materialize_values: bool = True,
    ):
        if role not in self._ROLE_SEED:
            raise ValueError("role must be train, val, or test")
        if masks_per_slice < 1:
            raise ValueError("masks_per_slice must be positive")
        if not 0.0 < target_fraction < 1.0:
            raise ValueError("target_fraction must be between zero and one")
        if idw_neighbors < 1:
            raise ValueError("idw_neighbors must be positive")
        if len(data.expression) < 2:
            raise ValueError("a slice needs at least two tissue spots")
        self.materialize_values = materialize_values
        self.data = data if materialize_values else None
        self.num_spots = len(data.expression)
        self.role = role
        self.masks_per_slice = masks_per_slice
        self.target_fraction = target_fraction
        self.idw_neighbors = idw_neighbors
        self.seed = seed
        self.epoch = 0
        self.all_spots = np.arange(self.num_spots, dtype=np.int64)

    def __len__(self):
        return self.masks_per_slice

    def set_epoch(self, epoch: int):
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.role == "train":
            self.epoch = epoch

    def _partition(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        target_count = min(
            len(self.all_spots) - 1,
            max(1, round(len(self.all_spots) * self.target_fraction)),
        )
        mask_epoch = self.epoch if self.role == "train" else 0
        rng = np.random.default_rng(np.random.SeedSequence([
            self.seed, self._ROLE_SEED[self.role], mask_epoch, index,
        ]))
        target_spots = np.sort(
            rng.choice(self.all_spots, size=target_count, replace=False)
            .astype(np.int64, copy=False)
        )
        hidden = np.zeros(len(self.all_spots), dtype=bool)
        hidden[target_spots] = True
        return self.all_spots[~hidden], target_spots

    def _baseline(self, visible_spots, target_spots):
        if self.data is None:
            raise RuntimeError("baseline materialization is disabled")
        if self.role == "train":
            return np.zeros(
                (len(target_spots), len(self.data.genes)), dtype=np.float32
            )
        neighbors = min(self.idw_neighbors, len(visible_spots))
        distances, indices = cKDTree(
            self.data.physical_xy[visible_spots]
        ).query(self.data.physical_xy[target_spots], k=neighbors)
        if neighbors == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = 1.0 / np.maximum(distances, 1e-6) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        neighbor_expression = self.data.expression[visible_spots[indices]]
        return np.einsum(
            "qk,qkg->qg", weights, neighbor_expression, optimize=True
        ).astype(np.float32)

    def __getitem__(self, index):
        visible_spots, target_spots = self._partition(index)
        item = {
            "target_spots": torch.from_numpy(target_spots),
            "visible_spots": torch.from_numpy(visible_spots),
        }
        if not self.materialize_values:
            return item
        assert self.data is not None
        item.update({
            "visible_expression": torch.from_numpy(
                self.data.expression[visible_spots].astype(np.float32, copy=False)
            ),
            "visible_coord": torch.from_numpy(
                self.data.physical_xy[visible_spots].astype(np.float32, copy=False)
            ),
            "query_coord": torch.from_numpy(
                self.data.physical_xy[target_spots].astype(np.float32, copy=False)
            ),
            "target_values": torch.from_numpy(
                self.data.expression[target_spots].astype(np.float32, copy=False)
            ),
            "baseline": torch.from_numpy(
                self._baseline(visible_spots, target_spots)
            ),
        })
        return item


class MultiSlicePointDataset(Dataset):
    """Balanced concatenation of per-slice compact point datasets."""

    def __init__(
        self,
        prepared: MultiSliceVisiumData,
        role: str,
        masks_per_slice: int = 64,
        mask_target_fraction: float = 0.25,
        idw_neighbors: int = 6,
        seed: int = 2026,
        materialize_values: bool = True,
    ):
        if role not in {"train", "val", "test"}:
            raise ValueError("role must be train, val, or test")
        selected = [
            (index, item) for index, item in enumerate(prepared.slices)
            if item.role == role
        ]
        if not selected:
            raise ValueError(f"no {role} slices are available")
        self.role = role
        self.slice_names = [item.name for _, item in selected]
        self.children = []
        self.global_slice_indices = []
        for global_index, item in selected:
            child = _WholeSliceMaskDataset(
                item.data,
                role=role,
                masks_per_slice=masks_per_slice,
                target_fraction=mask_target_fraction,
                idw_neighbors=idw_neighbors,
                seed=seed + global_index * 100003,
                materialize_values=materialize_values,
            )
            self.children.append(_TaggedSliceDataset(child, global_index))
            self.global_slice_indices.append(global_index)
        lengths = np.asarray([len(child) for child in self.children], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)]).tolist()

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        child_index = bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[child_index]
        return self.children[child_index][local_index]

    def set_epoch(self, epoch: int):
        for child in self.children:
            child.dataset.set_epoch(epoch)

    def tagged_slice_dataset(self, local_slice_index: int):
        return self.children[local_slice_index]
