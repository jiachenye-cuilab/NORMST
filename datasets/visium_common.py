"""Shared standard-Visium I/O and geometry used by multi-slice training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


@dataclass
class VisiumSliceData:
    """Prepared expression and geometry for one assigned Visium slice."""

    expression: np.ndarray
    genes: np.ndarray
    gene_scale: np.ndarray
    row_map: np.ndarray
    spot_rows: np.ndarray
    spot_cols: np.ndarray
    physical_xy: np.ndarray
    observed_spots: np.ndarray
    validation_spots: np.ndarray
    test_spots: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.row_map.shape


def as_csr(matrix) -> sp.csr_matrix:
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)


def read_standard_visium(data_dir: str, count_file: str):
    """Read expression and positions without requiring histology images."""
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


def spot_geometry(adata):
    """Build the offset-grid index and isotropic physical coordinates."""
    rows_raw = adata.obs["array_row"].to_numpy(np.int64)
    cols_raw = adata.obs["array_col"].to_numpy(np.int64)
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
    physical_xy /= max(float(physical_xy.std()), 1e-6)
    return rows, cols, row_map, physical_xy
