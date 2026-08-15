"""Train-only gene detection filtering and donor-aware HVG selection."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data import CountSlice, PretrainData


def training_gene_statistics(data: PretrainData) -> dict[str, np.ndarray | int]:
    """Return raw-count summaries fitted only on manifest training spots."""
    genes = len(data.genes)
    gene_sum = np.zeros(genes, dtype=np.float64)
    gene_square_sum = np.zeros(genes, dtype=np.float64)
    detected_spots = np.zeros(genes, dtype=np.int64)
    train_spots = 0
    for item in data.for_role("train"):
        counts = item.counts.astype(np.float64, copy=False)
        gene_sum += np.asarray(counts.sum(axis=0)).ravel()
        gene_square_sum += np.asarray(counts.power(2).sum(axis=0)).ravel()
        detected_spots += np.asarray(counts.getnnz(axis=0)).ravel()
        train_spots += item.n_spots
    if train_spots < 1:
        raise ValueError("gene selection requires non-empty training spots")
    mean = gene_sum / train_spots
    variance = np.maximum(gene_square_sum / train_spots - np.square(mean), 0.0)
    return {
        "train_spots": train_spots,
        "gene_sum": gene_sum,
        "mean": mean,
        "variance": variance,
        "detection_fraction": detected_spots / train_spots,
    }


def _subset_pretrain_data(
    data: PretrainData,
    indices: np.ndarray,
) -> PretrainData:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("gene indices must be a non-empty unique vector")
    slices = [
        CountSlice(
            name=item.name,
            role=item.role,
            path=item.path,
            counts=item.counts[:, indices].tocsr(),
            barcodes=item.barcodes,
        )
        for item in data.slices
    ]
    train_total = np.zeros(len(indices), dtype=np.float64)
    train_log_sum = np.zeros(len(indices), dtype=np.float64)
    train_spots = 0
    for item in slices:
        if item.role != "train":
            continue
        train_total += np.asarray(item.counts.sum(axis=0)).ravel()
        log_counts = item.counts.astype(np.float64, copy=True)
        log_counts.data = np.log1p(log_counts.data)
        train_log_sum += np.asarray(log_counts.sum(axis=0)).ravel()
        train_spots += item.n_spots
    if train_spots < 1 or np.any(train_total <= 0):
        raise ValueError("selected genes require positive training counts")
    return replace(
        data,
        slices=slices,
        genes=data.genes[indices],
        train_gene_probability=(train_total / train_total.sum()).astype(np.float32),
        train_log_mean=(train_log_sum / train_spots).astype(np.float32),
    )


def _training_anndata(
    slices: Sequence[CountSlice],
    genes: np.ndarray,
    slice_metadata: dict,
) -> ad.AnnData:
    matrices = []
    donor = []
    slice_name = []
    obs_names = []
    for item in slices:
        metadata = slice_metadata.get(item.name)
        if metadata is None or "donor" not in metadata:
            raise ValueError(f"manifest metadata missing donor for {item.name}")
        matrices.append(item.counts)
        donor.extend([str(metadata["donor"])] * item.n_spots)
        slice_name.extend([item.name] * item.n_spots)
        obs_names.extend(f"{item.name}:{barcode}" for barcode in item.barcodes)
    matrix = sp.vstack(matrices, format="csr", dtype=np.float32)
    obs = pd.DataFrame(
        {
            "donor": pd.Categorical(donor),
            "slice": pd.Categorical(slice_name),
        },
        index=pd.Index(obs_names, name="spot"),
    )
    var = pd.DataFrame(index=pd.Index(genes.astype(str), name="gene"))
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    if not result.obs_names.is_unique or not result.var_names.is_unique:
        raise ValueError("HVG AnnData identifiers must be unique")
    return result


def select_donor_aware_hvgs(
    data: PretrainData,
    n_top_genes: int,
    min_train_detection_fraction: float,
    slice_metadata: dict,
) -> tuple[PretrainData, dict, pd.DataFrame]:
    """Apply a train-only detection floor, then donor-aware Seurat-v3 HVGs."""
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be positive")
    if not 0.0 <= min_train_detection_fraction < 1.0:
        raise ValueError("min_train_detection_fraction must be in [0, 1)")
    statistics = training_gene_statistics(data)
    detection = np.asarray(statistics["detection_fraction"])
    eligible = np.flatnonzero(detection >= min_train_detection_fraction)
    if len(eligible) < n_top_genes:
        raise ValueError(
            f"detection floor {min_train_detection_fraction} leaves {len(eligible)} "
            f"genes, fewer than requested {n_top_genes}"
        )

    import scanpy as sc

    eligible_slices = [
        CountSlice(
            name=item.name,
            role=item.role,
            path=item.path,
            counts=item.counts[:, eligible].tocsr(),
            barcodes=item.barcodes,
        )
        for item in data.for_role("train")
    ]
    train = _training_anndata(
        eligible_slices,
        data.genes[eligible],
        slice_metadata,
    )
    sc.pp.highly_variable_genes(
        train,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        batch_key="donor",
        subset=False,
        check_values=True,
    )
    local = np.flatnonzero(train.var["highly_variable"].to_numpy(dtype=bool))
    if len(local) != n_top_genes:
        raise RuntimeError(f"Scanpy selected {len(local)} genes, expected {n_top_genes}")
    rank = train.var["highly_variable_rank"].to_numpy(dtype=np.float64)
    order = np.lexsort((data.genes[eligible[local]].astype(str), rank[local]))
    selected_indices = eligible[local[order]]
    selected = _subset_pretrain_data(data, selected_indices)

    mean = np.asarray(statistics["mean"])
    variance = np.asarray(statistics["variance"])
    columns = {
        "rank": np.arange(1, n_top_genes + 1, dtype=np.int64),
        "gene": selected.genes,
        "train_raw_mean": mean[selected_indices],
        "train_raw_variance": variance[selected_indices],
        "train_detection_fraction": detection[selected_indices],
        "train_zero_fraction": 1.0 - detection[selected_indices],
    }
    for source, target in (
        ("variances_norm", "train_normalized_variance"),
        ("highly_variable_nbatches", "hvg_donor_batches"),
    ):
        if source in train.var:
            columns[target] = train.var[source].to_numpy()[local[order]]
    table = pd.DataFrame(columns)
    audit = {
        "method": "Scanpy Seurat-v3 HVG after train-only detection filtering",
        "flavor": "seurat_v3",
        "input": "integer raw counts",
        "batch_key": "donor",
        "fit_roles": ["train"],
        "fit_slices": [item.name for item in data.for_role("train")],
        "fit_donors": sorted(train.obs["donor"].astype(str).unique().tolist()),
        "fit_spots": int(statistics["train_spots"]),
        "candidate_genes_before_detection_floor": len(data.genes),
        "min_train_detection_fraction": min_train_detection_fraction,
        "max_train_zero_fraction": 1.0 - min_train_detection_fraction,
        "eligible_genes_after_detection_floor": len(eligible),
        "selected_genes": n_top_genes,
        "tie_breaker": "ascending gene name for equal Seurat-v3 rank",
        "validation_or_test_expression_used": False,
    }
    return selected, audit, table
