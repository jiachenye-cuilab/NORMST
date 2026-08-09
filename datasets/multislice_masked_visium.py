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
from torch.utils.data import Dataset

from datasets.joint_masked_visium import PointJointMaskedVisiumDataset
from datasets.masked_visium import (
    MaskedVisiumData,
    _as_csr,
    _read_standard_visium,
    _spot_geometry,
)


@dataclass
class PreparedVisiumSlice:
    name: str
    role: str
    path: str
    data: MaskedVisiumData
    array_row: np.ndarray
    array_col: np.ndarray


@dataclass
class MultiSliceVisiumData:
    slices: list[PreparedVisiumSlice]
    genes: np.ndarray
    gene_scale: np.ndarray
    target_sum: float
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
    physical_coord_grid: np.ndarray
    row_parity: np.ndarray
    array_row: np.ndarray
    array_col: np.ndarray
    observed_spots: np.ndarray
    held_out_spots: np.ndarray


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


def _visibility_split(
    rows: np.ndarray,
    cols: np.ndarray,
    observed_fraction: float,
    seed: int,
):
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("observed_fraction must be between zero and one")
    if np.isclose(observed_fraction, 0.5):
        observed_mask = (rows + cols) % 2 == 0
    else:
        observed_mask = np.random.default_rng(seed).random(len(rows)) < observed_fraction
    observed = np.flatnonzero(observed_mask)
    held_out = np.flatnonzero(~observed_mask)
    if min(len(observed), len(held_out)) == 0:
        raise ValueError("a multi-slice visibility split produced an empty partition")
    return np.sort(observed), np.sort(held_out)


def _load_raw_slices(entries, observed_fraction: float, seed: int):
    raw_slices = []
    for index, entry in enumerate(entries):
        adata = _read_standard_visium(entry["path"], entry["count_file"])
        keep = ~adata.var_names.str.startswith("DEPRECATED_")
        adata = adata[:, keep].copy()
        rows, cols, row_map, physical_xy, coord_grid, row_parity = _spot_geometry(
            adata
        )
        raw_matrix = _as_csr(adata.X)
        library_size = np.asarray(raw_matrix.sum(axis=1)).ravel().astype(np.float32)
        observed, held_out = _visibility_split(
            rows, cols, observed_fraction, seed + index * 100003
        )
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
            physical_coord_grid=coord_grid,
            row_parity=row_parity,
            array_row=array_row,
            array_col=array_col,
            observed_spots=observed,
            held_out_spots=held_out,
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
    sources = [
        item.adata[item.observed_spots, common_genes].copy()
        for item in raw_slices if item.role == "train"
    ]
    pooled = sc.concat(sources, join="inner", index_unique="-")
    sc.pp.highly_variable_genes(
        pooled, flavor="seurat_v3", n_top_genes=n_genes, inplace=True
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
    observed_fraction: float = 0.5,
    seed: int = 2026,
) -> MultiSliceVisiumData:
    """Prepare shared genes/scales without fitting on val or test expression."""
    entries = _manifest_entries(manifest_path, count_file)
    raw_slices = _load_raw_slices(entries, observed_fraction, seed)
    common = _common_genes(raw_slices)
    genes = _select_training_hvgs(raw_slices, common, n_genes)

    unscaled = []
    squared_sum = np.zeros(len(genes), dtype=np.float64)
    training_count = 0
    for item in raw_slices:
        gene_indices = item.adata.var_names.get_indexer(genes)
        if np.any(gene_indices < 0):
            raise ValueError(f"selected genes are missing from slice {item.name}")
        counts = item.raw_matrix[:, gene_indices].toarray().astype(np.float32)
        expression = np.log1p(
            counts * (
                target_sum / np.maximum(item.library_size, 1.0)
            )[:, None]
        ).astype(np.float32, copy=False)
        unscaled.append(expression)
        if item.role == "train":
            selected = expression[item.observed_spots].astype(np.float64)
            squared_sum += np.square(selected).sum(axis=0)
            training_count += len(selected)
    if training_count < 1:
        raise ValueError("no observed training spots are available for RMS fitting")
    gene_scale = np.sqrt(squared_sum / training_count).astype(np.float32)
    gene_scale = np.maximum(gene_scale, 1e-6)

    prepared = []
    for item, expression in zip(raw_slices, unscaled):
        expression = expression / gene_scale[None, :]
        empty_context = np.empty((len(expression), 0), dtype=np.float32)
        empty = np.empty((0,), dtype=np.float32)
        empty_index = np.empty((0,), dtype=np.int64)
        data = MaskedVisiumData(
            expression=expression,
            genes=genes,
            gene_scale=gene_scale,
            context=empty_context,
            context_mean=empty,
            context_components=np.empty((0, len(genes)), dtype=np.float32),
            context_scale=empty,
            context_explained_variance_ratio=empty,
            row_map=item.row_map,
            spot_rows=item.rows,
            spot_cols=item.cols,
            physical_xy=item.physical_xy,
            physical_coord_grid=item.physical_coord_grid,
            row_parity=item.row_parity,
            physical_query_indices=empty_index,
            physical_query_relative=empty,
            physical_query_mask=empty,
            observed_spots=item.observed_spots,
            validation_spots=(
                item.held_out_spots if item.role == "val" else empty_index
            ),
            test_spots=(
                item.held_out_spots if item.role == "test" else empty_index
            ),
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


class MultiSlicePointDataset(Dataset):
    """Balanced concatenation of per-slice compact point datasets."""

    def __init__(
        self,
        prepared: MultiSliceVisiumData,
        role: str,
        masks_per_slice: int = 64,
        train_target_fraction: float = 0.25,
        idw_neighbors: int = 6,
        seed: int = 2026,
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
            child = PointJointMaskedVisiumDataset(
                item.data,
                split=role,
                masks_per_epoch=masks_per_slice,
                train_target_fraction=train_target_fraction,
                idw_neighbors=idw_neighbors,
                seed=seed + global_index * 100003,
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
