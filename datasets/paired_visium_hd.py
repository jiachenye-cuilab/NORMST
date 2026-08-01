"""Paired official Visium HD datasets for real-resolution SR training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import IncrementalPCA

from utils import make_coord


def _read_positions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "barcode" in frame.columns:
        frame = frame.set_index("barcode")
    elif "array_row" not in frame.columns:
        frame = pd.read_csv(
            path,
            header=None,
            names=(
                "barcode", "in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres",
            ),
            index_col="barcode",
        )
    frame.index = frame.index.astype(str)
    return frame


def _read_level(
    directory: Path,
    h5_name: str,
    positions_name: str,
):
    adata = sc.read_10x_h5(directory / h5_name, gex_only=True)
    adata.var_names_make_unique()
    keep_genes = ~adata.var_names.str.startswith("DEPRECATED_")
    adata = adata[:, keep_genes].copy()
    positions = _read_positions(directory / positions_name)
    common = positions.index.intersection(adata.obs_names, sort=False)
    if len(common) == 0:
        raise ValueError(f"No shared barcodes in {directory}")
    positions = positions.loc[common].copy()
    adata = adata[common, :].copy()
    valid = (
        positions["in_tissue"].to_numpy().astype(bool)
        if "in_tissue" in positions else np.ones(len(positions), dtype=bool)
    )
    return adata, positions, valid


def _as_csr(matrix) -> sp.csr_matrix:
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)


def _library_size(adata) -> np.ndarray:
    return np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float32)


def _row_map(positions: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
    height = int(positions["array_row"].max()) + 1
    width = int(positions["array_col"].max()) + 1
    result = np.full((height, width), -1, dtype=np.int32)
    indices = np.flatnonzero(valid)
    rows = positions.iloc[indices]["array_row"].to_numpy(np.int64)
    cols = positions.iloc[indices]["array_col"].to_numpy(np.int64)
    result[rows, cols] = indices
    return result


def _split_ranges(
    row_map_lr: np.ndarray,
    axis: str,
    ratios: Sequence[float],
) -> Dict[str, Tuple[int, int]]:
    occupied = np.argwhere(row_map_lr >= 0)
    dimension = occupied[:, 1] if axis == "col" else occupied[:, 0]
    start, stop = int(dimension.min()), int(dimension.max()) + 1
    length = stop - start
    train_stop = start + round(length * ratios[0])
    val_stop = train_stop + round(length * ratios[1])
    return {
        "train": (start, train_stop),
        "val": (train_stop, val_stop),
        "test": (val_stop, stop),
    }


def _observation_indices_for_split(
    positions: pd.DataFrame,
    valid: np.ndarray,
    split_range: Tuple[int, int],
    axis: str,
    coordinate_scale: int = 1,
) -> np.ndarray:
    column = "array_col" if axis == "col" else "array_row"
    coordinate = positions[column].to_numpy(np.int64) // coordinate_scale
    return np.flatnonzero(
        valid & (coordinate >= split_range[0]) & (coordinate < split_range[1])
    )


def _gene_rms(
    matrix: sp.csr_matrix,
    observation_indices: np.ndarray,
    gene_indices: np.ndarray,
    library_size: np.ndarray,
    target_sum: float,
    batch_genes: int = 64,
) -> np.ndarray:
    scales = []
    denominator = np.maximum(library_size[observation_indices], 1.0)
    for start in range(0, len(gene_indices), batch_genes):
        batch = gene_indices[start:start + batch_genes]
        counts = matrix[observation_indices, :][:, batch].toarray().T.astype(np.float32)
        values = np.log1p(counts * (target_sum / denominator)[None, :])
        scales.append(np.sqrt(np.mean(values ** 2, axis=1)))
    result = np.concatenate(scales).astype(np.float32)
    return np.maximum(result, 1e-6)


def _normalized_expression_batch(
    matrix: sp.csr_matrix,
    observations: np.ndarray,
    gene_indices: np.ndarray,
    library_size: np.ndarray,
    target_sum: float,
    gene_scale: np.ndarray,
) -> np.ndarray:
    """Return gene-wise scaled log-CP10K without materializing the full matrix."""
    counts = matrix[observations, :][:, gene_indices].toarray().astype(np.float32)
    values = np.log1p(
        counts * (target_sum / np.maximum(library_size[observations], 1.0))[:, None]
    )
    return values / gene_scale[None, :]


def _fit_lr_pca_context(
    matrix: sp.csr_matrix,
    train_observations: np.ndarray,
    gene_indices: np.ndarray,
    library_size: np.ndarray,
    target_sum: float,
    gene_scale: np.ndarray,
    n_components: int,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on LR training observations and transform every LR observation."""
    if n_components <= 0:
        return (
            np.empty((matrix.shape[0], 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, len(gene_indices)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    if len(train_observations) < n_components:
        raise ValueError("PCA context dimension exceeds the number of training bins")

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    # IncrementalPCA requires every partial_fit batch to contain at least K rows.
    fit_batch = max(batch_size, n_components)
    stop = len(train_observations)
    start = 0
    while start < stop:
        end = min(start + fit_batch, stop)
        # Fold a too-small final remainder into the current batch so every
        # observation is fitted exactly once.
        if 0 < stop - end < n_components:
            end = stop
        observations = train_observations[start:end]
        values = _normalized_expression_batch(
            matrix, observations, gene_indices, library_size,
            target_sum, gene_scale,
        )
        pca.partial_fit(values)
        start = end

    scores = np.empty((matrix.shape[0], n_components), dtype=np.float32)
    all_observations = np.arange(matrix.shape[0], dtype=np.int64)
    for start in range(0, len(all_observations), batch_size):
        observations = all_observations[start:start + batch_size]
        values = _normalized_expression_batch(
            matrix, observations, gene_indices, library_size,
            target_sum, gene_scale,
        )
        scores[observations] = pca.transform(values).astype(np.float32)
    context_scale = np.sqrt(np.mean(scores[train_observations] ** 2, axis=0))
    context_scale = np.maximum(context_scale.astype(np.float32), 1e-6)
    scores /= context_scale[None, :]
    return (
        scores,
        pca.mean_.astype(np.float32),
        pca.components_.astype(np.float32),
        context_scale,
        pca.explained_variance_ratio_.astype(np.float32),
    )


@dataclass
class VisiumHDPair:
    lr_matrix: sp.csc_matrix
    hr_matrix: sp.csc_matrix
    lr_library: np.ndarray
    hr_library: np.ndarray
    lr_row_map: np.ndarray
    hr_row_map: np.ndarray
    genes: np.ndarray
    lr_gene_scale: np.ndarray
    hr_gene_scale: np.ndarray
    split_ranges: Dict[str, Tuple[int, int]]
    split_axis: str
    scale: int
    target_sum: float
    lr_context: np.ndarray
    context_mean: np.ndarray
    context_components: np.ndarray
    context_scale: np.ndarray
    context_explained_variance_ratio: np.ndarray


def prepare_visium_hd_pair(
    lr_dir: str,
    hr_dir: str,
    n_genes: int = 1000,
    scale: int = 2,
    target_sum: float = 1e4,
    split_axis: str = "col",
    split_ratios: Sequence[float] = (0.7, 0.15, 0.15),
    h5_name: str = "raw_feature_bc_matrix.h5",
    positions_name: str = "spatial/tissue_positions.parquet",
    context_dim: int = 0,
) -> VisiumHDPair:
    if split_axis not in {"row", "col"}:
        raise ValueError("split_axis must be 'row' or 'col'")
    if len(split_ratios) != 3 or not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("split_ratios must contain train/val/test values summing to one")

    lr, lr_positions, lr_valid = _read_level(Path(lr_dir), h5_name, positions_name)
    hr, hr_positions, hr_valid = _read_level(Path(hr_dir), h5_name, positions_name)
    # Compute sequencing-depth factors before selecting common genes/HVGs.
    # This preserves the full-library denominator at each resolution.
    lr_library = _library_size(lr)
    hr_library = _library_size(hr)
    common_genes = lr.var_names.intersection(hr.var_names, sort=False)
    lr = lr[:, common_genes].copy()
    hr = hr[:, common_genes].copy()
    lr_map = _row_map(lr_positions, lr_valid)
    hr_map = _row_map(hr_positions, hr_valid)
    ranges = _split_ranges(lr_map, split_axis, split_ratios)

    hr_train_obs = _observation_indices_for_split(
        hr_positions, hr_valid, ranges["train"], split_axis, coordinate_scale=scale
    )
    hvg_source = hr[hr_train_obs, :].copy()
    sc.pp.highly_variable_genes(
        hvg_source, flavor="seurat_v3", n_top_genes=n_genes, inplace=True
    )
    genes = hvg_source.var_names[hvg_source.var["highly_variable"]].to_numpy()
    lr_gene_indices = lr.var_names.get_indexer(genes)
    hr_gene_indices = hr.var_names.get_indexer(genes)
    lr_matrix_all = _as_csr(lr.X)
    hr_matrix_all = _as_csr(hr.X)
    lr_train_obs = _observation_indices_for_split(
        lr_positions, lr_valid, ranges["train"], split_axis
    )

    lr_scales = _gene_rms(
        lr_matrix_all, lr_train_obs, lr_gene_indices, lr_library, target_sum
    )
    hr_scales = _gene_rms(
        hr_matrix_all, hr_train_obs, hr_gene_indices, hr_library, target_sum
    )
    (
        lr_context, context_mean, context_components, context_scale,
        context_explained_variance_ratio,
    ) = _fit_lr_pca_context(
        lr_matrix_all, lr_train_obs, lr_gene_indices, lr_library,
        target_sum, lr_scales, context_dim,
    )
    return VisiumHDPair(
        lr_matrix=lr_matrix_all[:, lr_gene_indices].tocsc(),
        hr_matrix=hr_matrix_all[:, hr_gene_indices].tocsc(),
        lr_library=lr_library,
        hr_library=hr_library,
        lr_row_map=lr_map,
        hr_row_map=hr_map,
        genes=genes,
        lr_gene_scale=lr_scales,
        hr_gene_scale=hr_scales,
        split_ranges=ranges,
        split_axis=split_axis,
        scale=scale,
        target_sum=target_sum,
        lr_context=lr_context,
        context_mean=context_mean,
        context_components=context_components,
        context_scale=context_scale,
        context_explained_variance_ratio=context_explained_variance_ratio,
    )


class PairedVisiumHDDataset(Dataset):
    def __init__(
        self,
        pair: VisiumHDPair,
        split: str,
        patch_size_lr: Tuple[int, int] = (64, 64),
        repeat: int = 20,
        min_tissue_fraction: float = 0.1,
        deterministic: bool = False,
        origin_stride: Optional[int] = None,
        max_origins: int = 0,
        context_mode: str = "pca",
        seed: int = 2026,
    ):
        self.pair = pair
        self.split = split
        self.patch_h, self.patch_w = patch_size_lr
        self.repeat = repeat
        self.deterministic = deterministic
        if context_mode not in {"pca", "shuffled"}:
            raise ValueError("context_mode must be 'pca' or 'shuffled'")
        self.context_mode = context_mode
        self.seed = seed
        self.origin_stride = origin_stride
        self.origins = self._candidate_origins(min_tissue_fraction)
        if max_origins > 0 and len(self.origins) > max_origins:
            chosen = np.linspace(0, len(self.origins) - 1, max_origins).round().astype(int)
            self.origins = [self.origins[index] for index in chosen]
        self.context_permutation = np.arange(len(pair.lr_library), dtype=np.int64)
        if context_mode == "shuffled" and pair.lr_context.shape[1]:
            np.random.default_rng(seed).shuffle(self.context_permutation)
        if not self.origins:
            raise ValueError(f"No usable {split} patches; reduce patch size or tissue threshold")

    def _candidate_origins(self, minimum: float):
        row_map = self.pair.lr_row_map
        low, high = self.pair.split_ranges[self.split]
        if self.pair.split_axis == "col":
            row_limits = (0, row_map.shape[0])
            col_limits = (low, high)
        else:
            row_limits = (low, high)
            col_limits = (0, row_map.shape[1])
        stride = self.origin_stride or max(self.patch_h // 4, 1)
        step_h, step_w = stride, stride
        origins = []
        for row in range(row_limits[0], row_limits[1] - self.patch_h + 1, step_h):
            for col in range(col_limits[0], col_limits[1] - self.patch_w + 1, step_w):
                patch = row_map[row:row + self.patch_h, col:col + self.patch_w]
                if np.mean(patch >= 0) >= minimum:
                    origins.append((row, col))
        return origins

    def __len__(self):
        if self.deterministic:
            return len(self.pair.genes) * len(self.origins)
        return len(self.pair.genes) * self.repeat

    def _extract_context(self, row: int, col: int) -> np.ndarray:
        channels = self.pair.lr_context.shape[1]
        result = np.zeros((channels, self.patch_h, self.patch_w), dtype=np.float32)
        if channels == 0:
            return result
        indices = self.pair.lr_row_map[
            row:row + self.patch_h, col:col + self.patch_w
        ]
        valid = indices >= 0
        observations = indices[valid]
        if self.context_mode == "shuffled":
            observations = self.context_permutation[observations]
        result[:, valid] = self.pair.lr_context[observations].T
        return result

    @staticmethod
    def _extract(
        matrix: sp.csc_matrix,
        library: np.ndarray,
        row_map: np.ndarray,
        gene: int,
        row: int,
        col: int,
        height: int,
        width: int,
        target_sum: float,
    ):
        indices = row_map[row:row + height, col:col + width]
        valid = indices >= 0
        values = np.zeros((height, width), dtype=np.float32)
        lib_grid = np.zeros((height, width), dtype=np.float32)
        obs = indices[valid]
        if len(obs):
            counts = matrix[obs, gene].toarray().ravel().astype(np.float32)
            values[valid] = np.log1p(
                counts * (target_sum / np.maximum(library[obs], 1.0))
            )
            lib_grid[valid] = library[obs]
        return values, valid.astype(np.float32), lib_grid

    def __getitem__(self, index):
        gene = index % len(self.pair.genes)
        if self.deterministic:
            origin = self.origins[(index // len(self.pair.genes)) % len(self.origins)]
        else:
            origin = random.choice(self.origins)
        row_lr, col_lr = origin
        scale = self.pair.scale
        inp, input_mask, _ = self._extract(
            self.pair.lr_matrix, self.pair.lr_library, self.pair.lr_row_map,
            gene, row_lr, col_lr, self.patch_h, self.patch_w, self.pair.target_sum,
        )
        gt, target_mask, hr_library = self._extract(
            self.pair.hr_matrix, self.pair.hr_library, self.pair.hr_row_map,
            gene, row_lr * scale, col_lr * scale,
            self.patch_h * scale, self.patch_w * scale, self.pair.target_sum,
        )
        inp = inp / self.pair.lr_gene_scale[gene]
        gt = gt / self.pair.hr_gene_scale[gene]
        context = self._extract_context(row_lr, col_lr)
        target_shape = (self.patch_h * scale, self.patch_w * scale)
        return {
            "inp": torch.from_numpy(inp).unsqueeze(0),
            "input_mask": torch.from_numpy(input_mask).unsqueeze(0),
            "gene_context": torch.from_numpy(context),
            "gt": torch.from_numpy(gt).unsqueeze(0),
            "target_mask": torch.from_numpy(target_mask).unsqueeze(0),
            "hr_library": torch.from_numpy(hr_library).unsqueeze(0),
            "coord": make_coord(target_shape),
            "cell": torch.tensor([2 / target_shape[0], 2 / target_shape[1]], dtype=torch.float32),
            "scale": torch.tensor(float(scale), dtype=torch.float32),
            "lr_gene_scale": torch.tensor(self.pair.lr_gene_scale[gene]),
            "hr_gene_scale": torch.tensor(self.pair.hr_gene_scale[gene]),
            "gene_index": torch.tensor(gene, dtype=torch.long),
        }
