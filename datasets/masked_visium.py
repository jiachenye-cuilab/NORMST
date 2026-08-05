"""Expression-only masked-spot recovery for standard 10x Visium."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from utils import make_coord


@dataclass
class MaskedVisiumData:
    expression: np.ndarray
    genes: np.ndarray
    gene_scale: np.ndarray
    context: np.ndarray
    context_mean: np.ndarray
    context_components: np.ndarray
    context_scale: np.ndarray
    context_explained_variance_ratio: np.ndarray
    row_map: np.ndarray
    spot_rows: np.ndarray
    spot_cols: np.ndarray
    physical_xy: np.ndarray
    physical_coord_grid: np.ndarray
    row_parity: np.ndarray
    observed_spots: np.ndarray
    validation_spots: np.ndarray
    test_spots: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.row_map.shape


def _as_csr(matrix) -> sp.csr_matrix:
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)


def _read_standard_visium(data_dir: str, count_file: str):
    """Read expression and positions without requiring histology image files."""
    directory = Path(data_dir)
    adata = sc.read_10x_h5(directory / count_file, gex_only=True)
    adata.var_names_make_unique()

    spatial = directory / "spatial"
    candidates = (
        spatial / "tissue_positions.csv",
        spatial / "tissue_positions_list.csv",
        spatial / "tissue_positions_list.txt",
    )
    positions_path = next((path for path in candidates if path.exists()), None)
    if positions_path is None:
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(f"Expected one of {names} under {spatial}")

    positions = pd.read_csv(positions_path)
    if "barcode" not in positions.columns:
        positions = pd.read_csv(
            positions_path,
            header=None,
            names=(
                "barcode", "in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres",
            ),
        )
    positions["barcode"] = positions["barcode"].astype(str)
    positions = positions.set_index("barcode")

    common = adata.obs_names[adata.obs_names.isin(positions.index)]
    if len(common) == 0:
        raise ValueError("No shared barcodes between the count matrix and positions")
    positions = positions.loc[common]
    if "in_tissue" in positions:
        in_tissue = pd.to_numeric(
            positions["in_tissue"], errors="raise"
        ).astype(bool)
        common = positions.index[in_tissue]
        positions = positions.loc[common]
    if len(common) == 0:
        raise ValueError("No in-tissue spots remain after position filtering")
    adata = adata[common, :].copy()
    for column in (
        "in_tissue", "array_row", "array_col",
        "pxl_row_in_fullres", "pxl_col_in_fullres",
    ):
        if column in positions:
            adata.obs[column] = positions.loc[adata.obs_names, column].to_numpy()
    return adata


def _spot_geometry(adata):
    rows_raw = adata.obs["array_row"].to_numpy(np.int64)
    cols_raw = adata.obs["array_col"].to_numpy(np.int64)
    # Visium array_col advances by two within a row. Integer division converts
    # the staggered hexagonal layout to an offset grid without reordering spots.
    rows = rows_raw - rows_raw.min()
    cols = cols_raw // 2
    cols = cols - cols.min()
    row_map = np.full((rows.max() + 1, cols.max() + 1), -1, dtype=np.int32)
    for spot, (row, col) in enumerate(zip(rows, cols)):
        if row_map[row, col] >= 0:
            raise ValueError("Visium coordinates collide in the offset-grid mapping")
        row_map[row, col] = spot

    pixel_columns = ("pxl_col_in_fullres", "pxl_row_in_fullres")
    if all(column in adata.obs for column in pixel_columns):
        physical_xy = adata.obs[list(pixel_columns)].to_numpy(np.float32).copy()
    else:
        physical_xy = np.column_stack([cols_raw, rows_raw]).astype(np.float32)
    physical_xy -= physical_xy.mean(axis=0, keepdims=True)
    # Use one isotropic scale so IDW distances retain the physical aspect ratio.
    physical_xy /= max(float(physical_xy.std()), 1e-6)

    # Physical-coordinate channels are isotropically normalized so the real
    # aspect ratio and the half-column shift between alternating rows remain.
    coord_values = physical_xy / max(float(np.abs(physical_xy).max()), 1e-6)
    physical_coord_grid = np.zeros((2, *row_map.shape), dtype=np.float32)
    physical_coord_grid[:, rows, cols] = coord_values.T

    # All spots in one Visium array row share the same raw array_col parity.
    # It determines which two offset-grid columns are the true neighbors in an
    # adjacent row.
    row_parity = np.zeros(row_map.shape[0], dtype=np.float32)
    for row in np.unique(rows):
        parities = np.unique(cols_raw[rows == row] % 2)
        if len(parities) != 1:
            raise ValueError("A Visium array row contains mixed column parities")
        row_parity[row] = parities[0]
    return (
        rows, cols, row_map, physical_xy, physical_coord_grid, row_parity
    )


def prepare_masked_visium(
    data_dir: str,
    count_file: str = "raw_feature_bc_matrix.h5",
    n_genes: int = 1000,
    target_sum: float = 1e4,
    context_dim: int = 16,
    observed_fraction: float = 0.5,
    validation_fraction: float = 0.5,
    seed: int = 2026,
) -> MaskedVisiumData:
    """Load one slice and reserve spots before fitting any expression statistic."""
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("observed_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    adata = _read_standard_visium(data_dir, count_file)
    keep = ~adata.var_names.str.startswith("DEPRECATED_")
    adata = adata[:, keep].copy()
    (
        rows, cols, row_map, physical_xy, physical_coord_grid, row_parity
    ) = _spot_geometry(adata)

    rng = np.random.default_rng(seed)
    if np.isclose(observed_fraction, 0.5):
        observed_mask = (rows + cols) % 2 == 0
    else:
        observed_mask = rng.random(adata.n_obs) < observed_fraction
    observed_spots = np.flatnonzero(observed_mask)
    held_out = np.flatnonzero(~observed_mask)
    rng.shuffle(held_out)
    validation_count = round(len(held_out) * validation_fraction)
    validation_spots = np.sort(held_out[:validation_count])
    test_spots = np.sort(held_out[validation_count:])
    if min(len(observed_spots), len(validation_spots), len(test_spots)) == 0:
        raise ValueError("Spot split produced an empty partition")

    raw = _as_csr(adata.X)
    library_size = np.asarray(raw.sum(axis=1)).ravel().astype(np.float32)
    hvg_source = adata[observed_spots, :].copy()
    sc.pp.highly_variable_genes(
        hvg_source, flavor="seurat_v3", n_top_genes=n_genes, inplace=True
    )
    genes = hvg_source.var_names[hvg_source.var["highly_variable"]].to_numpy()
    gene_indices = adata.var_names.get_indexer(genes)
    counts = raw[:, gene_indices].toarray().astype(np.float32)
    expression = np.log1p(
        counts * (target_sum / np.maximum(library_size, 1.0))[:, None]
    )
    gene_scale = np.sqrt(np.mean(expression[observed_spots] ** 2, axis=0))
    gene_scale = np.maximum(gene_scale.astype(np.float32), 1e-6)
    expression /= gene_scale[None, :]

    if context_dim > 0:
        if context_dim > min(len(observed_spots), len(genes)):
            raise ValueError("context_dim exceeds available training data")
        pca = PCA(n_components=context_dim, svd_solver="randomized", random_state=seed)
        pca.fit(expression[observed_spots])
        context = pca.transform(expression).astype(np.float32)
        context_scale = np.sqrt(np.mean(context[observed_spots] ** 2, axis=0))
        context_scale = np.maximum(context_scale.astype(np.float32), 1e-6)
        context /= context_scale[None, :]
        context_mean = pca.mean_.astype(np.float32)
        context_components = pca.components_.astype(np.float32)
        explained = pca.explained_variance_ratio_.astype(np.float32)
    else:
        context = np.empty((adata.n_obs, 0), dtype=np.float32)
        context_mean = np.empty((0,), dtype=np.float32)
        context_components = np.empty((0, len(genes)), dtype=np.float32)
        context_scale = np.empty((0,), dtype=np.float32)
        explained = np.empty((0,), dtype=np.float32)

    return MaskedVisiumData(
        expression=expression,
        genes=genes,
        gene_scale=gene_scale,
        context=context,
        context_mean=context_mean,
        context_components=context_components,
        context_scale=context_scale,
        context_explained_variance_ratio=explained,
        row_map=row_map,
        spot_rows=rows,
        spot_cols=cols,
        physical_xy=physical_xy,
        physical_coord_grid=physical_coord_grid,
        row_parity=row_parity,
        observed_spots=np.sort(observed_spots),
        validation_spots=validation_spots,
        test_spots=test_spots,
    )


class MaskedVisiumDataset(Dataset):
    def __init__(
        self,
        data: MaskedVisiumData,
        split: str,
        repeat: int = 8,
        train_target_fraction: float = 0.25,
        context_mode: str = "pca",
        idw_neighbors: int = 8,
        seed: int = 2026,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not 0.0 < train_target_fraction < 1.0:
            raise ValueError("train_target_fraction must be between 0 and 1")
        if context_mode not in {"pca", "shuffled"}:
            raise ValueError("context_mode must be pca or shuffled")
        if idw_neighbors < 1:
            raise ValueError("idw_neighbors must be at least 1")
        self.data = data
        self.split = split
        self.repeat = repeat
        self.train_target_fraction = train_target_fraction
        self.context_mode = context_mode
        self.seed = seed
        self.height, self.width = data.shape
        self.tissue_mask = (data.row_map >= 0).astype(np.float32)
        self.coord = make_coord((self.height, self.width))
        self.cell = torch.tensor(
            [2 / self.height, 2 / self.width], dtype=torch.float32
        )

        self.context_source = np.arange(len(data.expression), dtype=np.int64)
        if context_mode == "shuffled" and data.context.shape[1]:
            shuffled = data.observed_spots.copy()
            np.random.default_rng(seed).shuffle(shuffled)
            self.context_source[data.observed_spots] = shuffled

        self.eval_targets = (
            data.validation_spots if split == "val" else data.test_spots
        ) if split != "train" else np.empty((0,), dtype=np.int64)
        self.neighbor_spots = None
        self.neighbor_weights = None
        if split != "train":
            neighbors = min(idw_neighbors, len(data.observed_spots))
            tree = cKDTree(data.physical_xy[data.observed_spots])
            distances, indices = tree.query(
                data.physical_xy[self.eval_targets], k=neighbors
            )
            if neighbors == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            weights = 1.0 / np.maximum(distances, 1e-6) ** 2
            weights /= weights.sum(axis=1, keepdims=True)
            self.neighbor_spots = data.observed_spots[indices]
            self.neighbor_weights = weights.astype(np.float32)

    def __len__(self):
        multiplier = self.repeat if self.split == "train" else 1
        return len(self.data.genes) * multiplier

    def _put(self, values: np.ndarray, spots: np.ndarray, channels: int = 1):
        grid = np.zeros((channels, self.height, self.width), dtype=np.float32)
        if len(spots):
            rows = self.data.spot_rows[spots]
            cols = self.data.spot_cols[spots]
            if channels == 1:
                grid[0, rows, cols] = values
            else:
                grid[:, rows, cols] = values.T
        return grid

    def __getitem__(self, index):
        gene = index % len(self.data.genes)
        if self.split == "train":
            count = max(
                1, round(len(self.data.observed_spots) * self.train_target_fraction)
            )
            target_spots = np.asarray(
                random.sample(self.data.observed_spots.tolist(), count),
                dtype=np.int64,
            )
            target_lookup = set(target_spots.tolist())
            observed_spots = np.asarray(
                [spot for spot in self.data.observed_spots if spot not in target_lookup],
                dtype=np.int64,
            )
        else:
            target_spots = self.eval_targets
            observed_spots = self.data.observed_spots

        inp = self._put(self.data.expression[observed_spots, gene], observed_spots)
        input_mask = self._put(np.ones(len(observed_spots), np.float32), observed_spots)
        context_values = self.data.context[
            self.context_source[observed_spots]
        ]
        gene_context = self._put(
            context_values, observed_spots, channels=self.data.context.shape[1]
        )
        gt = self._put(self.data.expression[target_spots, gene], target_spots)
        target_mask = self._put(np.ones(len(target_spots), np.float32), target_spots)

        baseline = np.zeros((1, self.height, self.width), dtype=np.float32)
        if self.split != "train":
            neighbor_values = self.data.expression[self.neighbor_spots, gene]
            predictions = (neighbor_values * self.neighbor_weights).sum(axis=1)
            baseline = self._put(predictions, target_spots)

        return {
            "inp": torch.from_numpy(inp),
            "input_mask": torch.from_numpy(input_mask),
            "tissue_mask": torch.from_numpy(self.tissue_mask).unsqueeze(0),
            "physical_coord": torch.from_numpy(self.data.physical_coord_grid),
            "row_parity": torch.from_numpy(self.data.row_parity),
            "gene_context": torch.from_numpy(gene_context),
            "gt": torch.from_numpy(gt),
            "target_mask": torch.from_numpy(target_mask),
            "baseline": torch.from_numpy(baseline),
            "coord": self.coord,
            "cell": self.cell,
            "scale": torch.tensor(1.0, dtype=torch.float32),
            "gene_index": torch.tensor(gene, dtype=torch.long),
        }
