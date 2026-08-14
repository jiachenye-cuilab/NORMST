"""Read-only all-gene Visium loading for count-aware pretraining."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.visium_common import as_csr, read_standard_visium  # noqa: E402


ROLES = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    role: str
    path: Path
    count_file: str

    @property
    def count_path(self) -> Path:
        return self.path / self.count_file

    @property
    def position_path(self) -> Path:
        spatial = self.path / "spatial"
        candidates = (
            spatial / "tissue_positions.csv",
            spatial / "tissue_positions_list.csv",
            spatial / "tissue_positions_list.txt",
        )
        match = next((path for path in candidates if path.is_file()), None)
        if match is None:
            names = ", ".join(path.name for path in candidates)
            raise FileNotFoundError(f"expected one of {names} under {spatial}")
        return match


@dataclass
class CountSlice:
    name: str
    role: str
    path: Path
    counts: sp.csr_matrix
    barcodes: np.ndarray

    @property
    def n_spots(self) -> int:
        return int(self.counts.shape[0])


@dataclass
class PretrainData:
    manifest_path: Path
    manifest_sha256: str
    entries: list[ManifestEntry]
    slices: list[CountSlice]
    genes: np.ndarray
    train_gene_probability: np.ndarray
    train_log_mean: np.ndarray
    source_contract: list[dict]

    def for_role(self, role: str) -> list[CountSlice]:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        return [item for item in self.slices if item.role == role]

    def dataset(self, role: str) -> "SpotCountDataset":
        return SpotCountDataset(self.for_role(role))


def _normalize_manifest_group(group, role: str):
    if not group:
        raise ValueError(f"manifest requires a non-empty {role} group")
    if isinstance(group, dict):
        normalized = []
        for name, value in group.items():
            if isinstance(value, str):
                normalized.append({"name": name, "path": value})
            elif isinstance(value, dict):
                normalized.append({"name": name, **value})
            else:
                raise ValueError(f"invalid manifest entry for {role}/{name}")
        return normalized
    if isinstance(group, list):
        return group
    raise ValueError(f"manifest {role} must be an object or list")


def parse_manifest(
    manifest_path: str | Path,
    default_count_file: str = "filtered_feature_bc_matrix.h5",
) -> list[ManifestEntry]:
    """Parse the existing slice-level contract without writing beside it."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    if payload.get("_meta", {}).get("protocol") == "pair_grouped_lodo":
        from lodo import validate_lodo_payload

        validate_lodo_payload(payload)

    entries: list[ManifestEntry] = []
    for role in ROLES:
        for item in _normalize_manifest_group(payload.get(role), role):
            if not isinstance(item, dict):
                raise ValueError(f"manifest {role} entries must be objects")
            name = item.get("name", item.get("id"))
            raw_path = item.get("path")
            if not name or not raw_path:
                raise ValueError(f"manifest {role} entries require name and path")
            data_path = Path(raw_path)
            if not data_path.is_absolute():
                data_path = (path.parent / data_path).resolve()
            entries.append(
                ManifestEntry(
                    name=str(name),
                    role=role,
                    path=data_path,
                    count_file=str(item.get("count_file", default_count_file)),
                )
            )

    names = [item.name for item in entries]
    paths = [str(item.path).casefold() for item in entries]
    if len(names) != len(set(names)):
        raise ValueError("slice names must be unique across manifest roles")
    if len(paths) != len(set(paths)):
        raise ValueError("the same slice path cannot occur in multiple roles")
    return entries


def _file_snapshot(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def source_contract(entries: Sequence[ManifestEntry]) -> list[dict]:
    """Record cheap source metadata so a run can verify that inputs stayed fixed."""
    result = []
    for item in entries:
        if not item.count_path.is_file():
            raise FileNotFoundError(item.count_path)
        for kind, path in (
            ("raw_counts", item.count_path),
            ("tissue_positions", item.position_path),
        ):
            snapshot = _file_snapshot(path)
            snapshot.update({"slice": item.name, "role": item.role, "kind": kind})
            result.append(snapshot)
    return result


def verify_source_contract(contract: Sequence[dict]) -> None:
    for expected in contract:
        current = _file_snapshot(Path(expected["path"]))
        for key in ("size", "mtime_ns"):
            if current[key] != expected[key]:
                raise RuntimeError(
                    f"source count file changed during the run: {expected['path']}"
                )


def assert_output_outside_sources(
    output_dir: str | Path,
    entries: Sequence[ManifestEntry],
) -> Path:
    output = Path(output_dir).resolve()
    protected_roots = {item.path.resolve() for item in entries}
    protected_roots.update(item.path.resolve().parent for item in entries)
    for item in entries:
        protected_roots.update(
            parent
            for parent in item.path.resolve().parents
            if parent.name.casefold() == "data"
        )
    for protected in protected_roots:
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        else:
            raise ValueError(
                f"output directory must not be inside a source data root: {protected}"
            )
    return output


def _validate_counts(matrix: sp.csr_matrix, slice_name: str) -> sp.csr_matrix:
    values = matrix.data
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{slice_name} contains invalid raw counts")
    if values.size and not np.allclose(values, np.rint(values), atol=1e-5):
        raise ValueError(
            f"{slice_name} does not look like an integer raw-count matrix"
        )
    matrix = matrix.astype(np.float32, copy=False)
    matrix.eliminate_zeros()
    return matrix


def _training_summaries(
    slices: Sequence[CountSlice],
    n_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    gene_total = np.zeros(n_genes, dtype=np.float64)
    log_sum = np.zeros(n_genes, dtype=np.float64)
    spots = 0
    for item in slices:
        if item.role != "train":
            continue
        gene_total += np.asarray(item.counts.sum(axis=0)).ravel()
        log_counts = item.counts.copy().astype(np.float64)
        log_counts.data = np.log1p(log_counts.data)
        log_sum += np.asarray(log_counts.sum(axis=0)).ravel()
        spots += item.n_spots
    if spots < 1 or gene_total.sum() <= 0:
        raise ValueError("training slices contain no positive counts")
    probability = gene_total / gene_total.sum()
    return probability.astype(np.float32), (log_sum / spots).astype(np.float32)


def load_pretrain_data(
    manifest_path: str | Path,
    count_file: str = "filtered_feature_bc_matrix.h5",
    min_train_gene_count: int = 1,
    fixed_genes: Iterable[str] | None = None,
) -> PretrainData:
    """Load aligned raw counts; expression-based fitting uses train only."""
    if min_train_gene_count < 1:
        raise ValueError("min_train_gene_count must be positive")
    manifest = Path(manifest_path).resolve()
    entries = parse_manifest(manifest, count_file)
    contract = source_contract(entries)

    loaded = []
    availability = []
    for entry in entries:
        adata = read_standard_visium(str(entry.path), entry.count_file)
        keep = ~adata.var_names.str.startswith("DEPRECATED_")
        adata = adata[:, keep].copy()
        loaded.append((entry, adata))
        availability.append(set(map(str, adata.var_names)))

    if fixed_genes is None:
        first_train = next(adata for entry, adata in loaded if entry.role == "train")
        common = np.asarray(
            [
                str(gene)
                for gene in first_train.var_names
                if all(str(gene) in present for present in availability)
            ],
            dtype=str,
        )
    else:
        common = np.asarray(list(map(str, fixed_genes)), dtype=str)
        missing = {
            entry.name: [gene for gene in common if gene not in present]
            for (entry, _), present in zip(loaded, availability)
        }
        missing = {name: genes for name, genes in missing.items() if genes}
        if missing:
            summary = ", ".join(f"{name}:{len(genes)}" for name, genes in missing.items())
            raise ValueError(f"fixed genes are absent from slices: {summary}")
    if common.size < 1:
        raise ValueError("manifest slices have no common genes")

    preliminary: list[CountSlice] = []
    for entry, adata in loaded:
        aligned = _validate_counts(as_csr(adata[:, common].X), entry.name)
        library = np.asarray(aligned.sum(axis=1)).ravel()
        if np.any(library <= 0):
            raise ValueError(f"{entry.name} contains zero-library in-tissue spots")
        preliminary.append(
            CountSlice(
                name=entry.name,
                role=entry.role,
                path=entry.path,
                counts=aligned,
                barcodes=np.asarray(adata.obs_names, dtype=str),
            )
        )

    if fixed_genes is None:
        train_total = np.zeros(len(common), dtype=np.float64)
        for item in preliminary:
            if item.role == "train":
                train_total += np.asarray(item.counts.sum(axis=0)).ravel()
        keep = train_total >= min_train_gene_count
        if not np.any(keep):
            raise ValueError("no genes satisfy the training-only count filter")
        genes = common[keep]
        slices = [
            CountSlice(
                name=item.name,
                role=item.role,
                path=item.path,
                counts=item.counts[:, keep].tocsr(),
                barcodes=item.barcodes,
            )
            for item in preliminary
        ]
    else:
        genes = common
        slices = preliminary

    probability, train_log_mean = _training_summaries(slices, len(genes))
    if np.any(probability <= 0):
        raise ValueError("fixed gene set contains genes with zero training counts")
    verify_source_contract(contract)
    return PretrainData(
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        entries=entries,
        slices=slices,
        genes=genes,
        train_gene_probability=probability,
        train_log_mean=train_log_mean,
        source_contract=contract,
    )


class SpotCountDataset(Dataset):
    """Map global spot indices to rows of immutable CSR count matrices."""

    def __init__(self, slices: Sequence[CountSlice]):
        if not slices:
            raise ValueError("SpotCountDataset requires at least one slice")
        gene_counts = {item.counts.shape[1] for item in slices}
        if len(gene_counts) != 1:
            raise ValueError("all slices must use the same gene order")
        self.slices = list(slices)
        self.ends = np.cumsum([item.n_spots for item in self.slices]).tolist()
        self.n_genes = next(iter(gene_counts))

    def __len__(self) -> int:
        return int(self.ends[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slice_index = bisect_right(self.ends, index)
        start = 0 if slice_index == 0 else self.ends[slice_index - 1]
        return slice_index, index - start

    def __getitem__(self, index: int):
        slice_index, spot_index = self._locate(index)
        row = self.slices[slice_index].counts.getrow(spot_index).toarray().ravel()
        return {
            "counts": torch.from_numpy(row.astype(np.float32, copy=False)),
            "slice_index": slice_index,
            "spot_index": spot_index,
        }

    def __getitems__(self, indices):
        """Densify one CSR block per slice instead of one row per Python call."""
        located = [self._locate(int(index)) for index in indices]
        groups: dict[int, list[tuple[int, int]]] = {}
        for output_index, (slice_index, spot_index) in enumerate(located):
            groups.setdefault(slice_index, []).append((output_index, spot_index))

        result = [None] * len(located)
        for slice_index, members in groups.items():
            spot_indices = [spot_index for _, spot_index in members]
            dense = self.slices[slice_index].counts[spot_indices].toarray()
            dense = dense.astype(np.float32, copy=False)
            for row_index, (output_index, spot_index) in enumerate(members):
                result[output_index] = {
                    "counts": torch.from_numpy(dense[row_index]),
                    "slice_index": slice_index,
                    "spot_index": spot_index,
                }
        return result
