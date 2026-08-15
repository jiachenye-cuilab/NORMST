"""Read-only Visium preparation in a frozen AE composition coordinate system."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from datasets.multislice_masked_visium import (
    MultiSlicePointDataset,
    _common_genes,
    _load_raw_slices,
    _manifest_entries,
)
from datasets.visium_common import VisiumSliceData
from models.frozen_composition_ae import FrozenCompositionAE


POSITION_FILES = (
    "tissue_positions.csv",
    "tissue_positions_list.csv",
    "tissue_positions_list.txt",
)


@dataclass
class PreparedAEVisiumSlice:
    name: str
    role: str
    path: str
    data: VisiumSliceData
    array_row: np.ndarray
    array_col: np.ndarray
    selected_counts: np.ndarray
    full_gene_log_library: np.ndarray
    standardized_library_context: np.ndarray
    barcodes: np.ndarray


@dataclass
class AEMultiSliceVisiumData:
    slices: list[PreparedAEVisiumSlice]
    genes: np.ndarray
    composition_dim: int
    library_context_mean: float
    library_context_scale: float
    manifest_path: str
    manifest_sha256: str
    ae_checkpoint_sha256: str
    source_contract: list[dict]

    def for_role(self, role: str) -> list[PreparedAEVisiumSlice]:
        return [item for item in self.slices if item.role == role]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_source_contract(entries: list[dict]) -> list[dict]:
    result = []
    for entry in entries:
        root = Path(entry["path"]).resolve()
        count_path = root / entry["count_file"]
        position_path = next(
            (root / "spatial" / name for name in POSITION_FILES
             if (root / "spatial" / name).is_file()),
            None,
        )
        if not count_path.is_file() or position_path is None:
            raise FileNotFoundError(f"incomplete Visium source: {root}")
        for kind, path in (("raw_counts", count_path), ("positions", position_path)):
            value = _snapshot(path)
            value.update({
                "slice": entry["name"], "role": entry["role"], "kind": kind
            })
            result.append(value)
    return result


def verify_source_contract(contract: list[dict]) -> None:
    for expected in contract:
        current = _snapshot(Path(expected["path"]))
        if any(current[key] != expected[key] for key in ("size", "mtime_ns")):
            raise RuntimeError(f"source data changed: {expected['path']}")


def _encode_counts(
    frozen_ae: FrozenCompositionAE,
    counts: np.ndarray,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    values = []
    for start in range(0, len(counts), batch_size):
        batch = torch.from_numpy(counts[start:start + batch_size]).to(
            device, non_blocking=True
        )
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            encoded = frozen_ae.encode_standardized(batch)
        values.append(encoded.float().cpu().numpy())
    return np.concatenate(values).astype(np.float32, copy=False)


def prepare_ae_multislice_visium(
    manifest_path: str | Path,
    frozen_ae: FrozenCompositionAE,
    device: torch.device,
    count_file: str = "filtered_feature_bc_matrix.h5",
    encode_batch_size: int = 256,
    use_amp: bool = True,
) -> AEMultiSliceVisiumData:
    """Encode every spot while fitting only full-library context on train spots."""
    if encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive")
    manifest = Path(manifest_path).resolve()
    manifest_sha256 = sha256_file(manifest)
    if not frozen_ae.manifest_sha256:
        raise ValueError("AE checkpoint does not record its training manifest hash")
    if manifest_sha256 != frozen_ae.manifest_sha256:
        raise ValueError(
            "AE checkpoint and AE-NORMST manifest differ; refit the AE for this split"
        )

    entries = _manifest_entries(str(manifest), count_file)
    contract = build_source_contract(entries)
    raw_slices = _load_raw_slices(entries)
    common = set(_common_genes(raw_slices).tolist())
    genes = np.asarray(frozen_ae.genes, dtype=str)
    missing = [gene for gene in genes if gene not in common]
    if missing:
        raise ValueError(
            f"AE genes are missing from one or more slices: {missing[:5]}"
        )

    train_log_library = np.concatenate([
        np.log1p(item.library_size.astype(np.float64))
        for item in raw_slices if item.role == "train"
    ])
    if train_log_library.size < 2:
        raise ValueError("at least two training spots are required")
    library_mean = float(train_log_library.mean())
    library_scale = max(float(train_log_library.std()), 1e-6)

    frozen_ae.to(device).eval()
    latent_names = np.asarray([
        f"ae_composition_{index:02d}" for index in range(frozen_ae.composition_dim)
    ])
    prepared = []
    for item in raw_slices:
        gene_indices = item.adata.var_names.get_indexer(genes)
        if np.any(gene_indices < 0):
            raise ValueError(f"AE genes are missing from slice {item.name}")
        counts = item.raw_matrix[:, gene_indices].toarray().astype(np.float32)
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError(f"slice {item.name} contains invalid raw counts")
        if counts.size and not np.allclose(counts, np.rint(counts), atol=1e-5):
            raise ValueError(f"slice {item.name} counts are not integer-valued")
        composition = _encode_counts(
            frozen_ae,
            counts,
            device,
            encode_batch_size,
            bool(use_amp and device.type == "cuda"),
        )
        full_log_library = np.log1p(item.library_size).astype(np.float32)
        library_context = (
            (full_log_library.astype(np.float64) - library_mean) / library_scale
        ).astype(np.float32)
        empty = np.empty((0,), dtype=np.int64)
        data = VisiumSliceData(
            expression=composition,
            genes=latent_names,
            gene_scale=np.ones(frozen_ae.composition_dim, dtype=np.float32),
            row_map=item.row_map,
            spot_rows=item.rows,
            spot_cols=item.cols,
            physical_xy=item.physical_xy,
            observed_spots=item.observed_spots,
            validation_spots=empty,
            test_spots=empty,
        )
        prepared.append(PreparedAEVisiumSlice(
            name=item.name,
            role=item.role,
            path=item.path,
            data=data,
            array_row=item.array_row,
            array_col=item.array_col,
            selected_counts=counts,
            full_gene_log_library=full_log_library,
            standardized_library_context=library_context,
            barcodes=item.adata.obs_names.to_numpy(dtype=str).copy(),
        ))

    verify_source_contract(contract)
    return AEMultiSliceVisiumData(
        slices=prepared,
        genes=genes,
        composition_dim=frozen_ae.composition_dim,
        library_context_mean=library_mean,
        library_context_scale=library_scale,
        manifest_path=str(manifest),
        manifest_sha256=manifest_sha256,
        ae_checkpoint_sha256=frozen_ae.checkpoint_sha256,
        source_contract=contract,
    )


__all__ = [
    "AEMultiSliceVisiumData",
    "MultiSlicePointDataset",
    "PreparedAEVisiumSlice",
    "prepare_ae_multislice_visium",
    "verify_source_contract",
]
