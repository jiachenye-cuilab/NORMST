"""Strict real-data adapter for the direct-512 ProNORMST contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from models.pro_normst import FullHexGeometry


PANEL_SIZE = 512
PANEL_ORDERED_SHA256 = "72562d01005a5078a0d95b38a050824299fa906f4c9888ff989c2aba9a73a7ce"
NEIGHBOR_DELTAS = ((0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1))
OPPOSITE_DIRECTIONS = (1, 0, 5, 4, 3, 2)

FROZEN_SPLITS: dict[str, dict[str, tuple[str, ...]]] = {
    "pilot_seed2027": {
        "train": ("151509", "151510", "151669", "151670", "151671", "151672", "151675", "151676"),
        "val": ("151507", "151508"),
        "test": ("151673", "151674"),
    },
    "lodo_d1": {
        "train": ("151671", "151672", "151673", "151674"),
        "val": ("151669", "151670", "151675", "151676"),
        "test": ("151507", "151508", "151509", "151510"),
    },
    "lodo_d2": {
        "train": ("151509", "151510", "151673", "151674"),
        "val": ("151507", "151508", "151675", "151676"),
        "test": ("151669", "151670", "151671", "151672"),
    },
    "lodo_d3": {
        "train": ("151507", "151508", "151671", "151672"),
        "val": ("151509", "151510", "151669", "151670"),
        "test": ("151673", "151674", "151675", "151676"),
    },
}
FROZEN_LODO_DONORS = {
    "lodo_d1": "Br5292",
    "lodo_d2": "Br5595",
    "lodo_d3": "Br8100",
}
FROZEN_SLICE_METADATA = {
    "151507": ("Br5292", "Br5292_anterior", "anterior", "a"),
    "151508": ("Br5292", "Br5292_anterior", "anterior", "b"),
    "151509": ("Br5292", "Br5292_posterior", "posterior", "a"),
    "151510": ("Br5292", "Br5292_posterior", "posterior", "b"),
    "151669": ("Br5595", "Br5595_anterior", "anterior", "a"),
    "151670": ("Br5595", "Br5595_anterior", "anterior", "b"),
    "151671": ("Br5595", "Br5595_posterior", "posterior", "a"),
    "151672": ("Br5595", "Br5595_posterior", "posterior", "b"),
    "151673": ("Br8100", "Br8100_anterior", "anterior", "a"),
    "151674": ("Br8100", "Br8100_anterior", "anterior", "b"),
    "151675": ("Br8100", "Br8100_posterior", "posterior", "a"),
    "151676": ("Br8100", "Br8100_posterior", "posterior", "b"),
}


def ordered_text_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class SliceSpec:
    slice_id: str
    role: str
    path: Path
    count_file: str
    donor: str
    pair: str
    position: str
    serial: str


@dataclass
class ProNORMSTSlice:
    slice_id: str
    role: str
    donor: str
    pair: str
    barcodes: tuple[str, ...]
    gene_ids: tuple[str, ...]
    expression_x: np.ndarray
    expression_z: np.ndarray
    array_row: np.ndarray
    array_col: np.ndarray
    full_xy: np.ndarray
    neighbor_index: np.ndarray
    native_scale: float
    component_id: np.ndarray
    source_barcodes_sha256: str = ""
    source_gex_gene_ids_sha256: str = ""
    source_panel_indices_sha256: str = ""

    @property
    def n_nodes(self) -> int:
        return len(self.barcodes)

    def geometry(self, device: torch.device | str = "cpu") -> FullHexGeometry:
        target = torch.device(device)
        return FullHexGeometry(
            xy=torch.as_tensor(self.full_xy, dtype=torch.float32, device=target),
            neighbor_index=torch.as_tensor(
                self.neighbor_index, dtype=torch.long, device=target
            ),
            native_scale=torch.tensor(self.native_scale, dtype=torch.float32, device=target),
            indices_validated=True,
        )


@dataclass(frozen=True)
class ProNORMSTPreprocessing:
    gene_ids: tuple[str, ...]
    gene_scale: np.ndarray
    detection_rate: np.ndarray
    positive_weight: np.ndarray
    gene_mean_z: np.ndarray
    panel_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "pro-normst-preprocessing-v2",
            "panel_size": len(self.gene_ids),
            "panel_ordered_sha256": self.panel_sha256,
            "gene_scale_sha256": _array_sha256(self.gene_scale.astype("<f4")),
            "detection_rate_sha256": _array_sha256(
                self.detection_rate.astype("<f4")
            ),
            "positive_weight_sha256": _array_sha256(
                self.positive_weight.astype("<f4")
            ),
            "gene_mean_z_sha256": _array_sha256(self.gene_mean_z.astype("<f4")),
            "transform": "log1p(panel_only_cp10k)",
            "scale": "train_slice_balanced_uncentered_rms",
        }


@dataclass
class ProNORMSTData:
    roles: dict[str, list[ProNORMSTSlice]]
    preprocessing: ProNORMSTPreprocessing
    split_metadata: dict[str, Any]


def load_panel(path: str | Path) -> tuple[str, ...]:
    panel_path = Path(path).resolve()
    values = tuple(
        line.strip()
        for line in panel_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(values) != PANEL_SIZE or len(set(values)) != PANEL_SIZE:
        raise ValueError("Shared panel must contain exactly 512 unique Ensembl IDs")
    actual_hash = ordered_text_sha256(values)
    if actual_hash != PANEL_ORDERED_SHA256:
        raise ValueError(
            f"Shared panel ordered hash mismatch: {actual_hash} != {PANEL_ORDERED_SHA256}"
        )
    return values


def _validate_frozen_split(
    roles: dict[str, list[SliceSpec]], metadata: dict[str, Any]
) -> None:
    protocol = str(metadata.get("protocol", ""))
    if protocol == "pair_grouped_random_split":
        try:
            split_seed = int(metadata.get("split_seed"))
        except (TypeError, ValueError) as error:
            raise ValueError("pilot split_seed must be the frozen value 2027") from error
        if split_seed != 2027:
            raise ValueError("pilot split_seed must be the frozen value 2027")
        identity = "pilot_seed2027"
    elif protocol == "pair_grouped_lodo":
        identity = str(metadata.get("fold", ""))
        expected_donor = FROZEN_LODO_DONORS.get(identity)
        if expected_donor is None:
            raise ValueError(f"LODO fold must be one of {sorted(FROZEN_LODO_DONORS)}")
        if str(metadata.get("held_out_donor", "")) != expected_donor:
            raise ValueError(
                f"{identity} must hold out donor {expected_donor}"
            )
    else:
        return

    expected = FROZEN_SPLITS[identity]
    actual = {
        role: tuple(sorted(spec.slice_id for spec in role_specs))
        for role, role_specs in roles.items()
    }
    if actual != expected:
        raise ValueError(
            f"{identity} slice roles do not match the frozen split: "
            f"actual={actual}, expected={expected}"
        )
    for role_specs in roles.values():
        for spec in role_specs:
            actual_metadata = (spec.donor, spec.pair, spec.position, spec.serial)
            expected_metadata = FROZEN_SLICE_METADATA.get(spec.slice_id)
            if actual_metadata != expected_metadata:
                raise ValueError(
                    f"{spec.slice_id} donor/pair metadata does not match the frozen DLPFC identity"
                )


def load_split_manifest(
    path: str | Path,
    default_count_file: str = "filtered_feature_bc_matrix.h5",
) -> tuple[dict[str, list[SliceSpec]], dict[str, Any]]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = payload.get("_meta", {})
    role_aliases = (("train", "train"), ("val", "val"), ("validation", "val"), ("test", "test"))
    roles: dict[str, list[SliceSpec]] = {"train": [], "val": [], "test": []}
    seen_slices: dict[str, str] = {}
    for source_role, canonical_role in role_aliases:
        if source_role not in payload:
            continue
        entries = payload[source_role]
        if not isinstance(entries, dict):
            raise ValueError(f"manifest role {source_role} must be an object")
        for slice_id, raw in entries.items():
            if slice_id in seen_slices:
                raise ValueError(
                    f"slice {slice_id} appears in both {seen_slices[slice_id]} and {canonical_role}"
                )
            seen_slices[slice_id] = canonical_role
            item = raw if isinstance(raw, dict) else {"path": raw}
            slice_meta = metadata.get("slice_metadata", {}).get(slice_id, {})
            field = lambda name: str(item.get(name, slice_meta.get(name, "")))
            required = {name: field(name) for name in ("donor", "pair", "position", "serial")}
            if not all(required.values()):
                raise ValueError(f"slice {slice_id} is missing donor/pair metadata")
            raw_path_value = item.get("path")
            if raw_path_value is None or not str(raw_path_value).strip():
                raise ValueError(f"slice {slice_id} is missing path")
            raw_path = Path(str(raw_path_value))
            resolved = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
            roles[canonical_role].append(
                SliceSpec(
                    slice_id=str(slice_id),
                    role=canonical_role,
                    path=resolved.resolve(),
                    count_file=str(item.get("count_file", default_count_file)),
                    donor=required["donor"],
                    pair=required["pair"],
                    position=required["position"],
                    serial=required["serial"],
                )
            )
    if not all(roles.values()):
        raise ValueError("manifest must contain non-empty train, val, and test roles")
    for role in roles:
        roles[role].sort(key=lambda item: item.slice_id)

    pair_roles: dict[str, set[str]] = {}
    donor_roles: dict[str, set[str]] = {}
    for role, specs in roles.items():
        for spec in specs:
            pair_roles.setdefault(spec.pair, set()).add(role)
            donor_roles.setdefault(spec.donor, set()).add(role)
    split_pairs = {pair: sorted(value) for pair, value in pair_roles.items() if len(value) > 1}
    if split_pairs:
        raise ValueError(f"serial pairs cross manifest roles: {split_pairs}")
    if metadata.get("protocol") == "pair_grouped_lodo":
        held_out = str(metadata.get("held_out_donor", ""))
        if not held_out or donor_roles.get(held_out) != {"test"}:
            raise ValueError("LODO held-out donor must be test-only")
        if any(spec.donor != held_out for spec in roles["test"]):
            raise ValueError("LODO test role contains a non-held-out donor")
    _validate_frozen_split(roles, metadata)
    return roles, metadata


def _read_positions(directory: Path) -> pd.DataFrame:
    spatial = directory / "spatial"
    candidates = (
        spatial / "tissue_positions.csv",
        spatial / "tissue_positions_list.csv",
        spatial / "tissue_positions_list.txt",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"missing tissue positions under {spatial}")
    positions = pd.read_csv(path)
    if "barcode" not in positions.columns:
        positions = pd.read_csv(
            path,
            header=None,
            names=(
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ),
        )
    required = {
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    }
    missing = sorted(required.difference(positions.columns))
    if missing:
        raise ValueError(f"positions missing columns: {missing}")
    positions["barcode"] = positions["barcode"].astype(str)
    if positions["barcode"].duplicated().any():
        raise ValueError("positions contain duplicate barcodes")
    positions = positions.set_index("barcode")
    in_tissue = pd.to_numeric(positions["in_tissue"], errors="raise").astype(np.int64)
    if not bool(in_tissue.isin((0, 1)).all()):
        raise ValueError("in_tissue must contain only 0/1")
    return positions.loc[in_tissue == 1].copy()


def _build_graph(
    rows: np.ndarray,
    cols: np.ndarray,
    xy: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    nodes = len(rows)
    coordinates = [(int(row), int(col)) for row, col in zip(rows, cols)]
    if len(set(coordinates)) != nodes:
        raise ValueError("array_row/array_col coordinates are not unique")
    lookup = {coordinate: index for index, coordinate in enumerate(coordinates)}
    neighbor = np.full((nodes, 6), -1, dtype=np.int64)
    for node, (row, col) in enumerate(coordinates):
        for direction, (delta_row, delta_col) in enumerate(NEIGHBOR_DELTAS):
            neighbor[node, direction] = lookup.get((row + delta_row, col + delta_col), -1)
    for node in range(nodes):
        for direction, other in enumerate(neighbor[node]):
            if other >= 0 and neighbor[other, OPPOSITE_DIRECTIONS[direction]] != node:
                raise ValueError("native graph has an inconsistent opposite edge")

    lengths = []
    for node in range(nodes):
        for other in neighbor[node]:
            if other > node:
                lengths.append(float(np.linalg.norm(xy[node].astype(np.float64) - xy[other])))
    if not lengths or not np.isfinite(lengths).all() or min(lengths) <= 0:
        raise ValueError("slice has no valid positive-length native edge")
    native_scale = float(np.median(np.asarray(lengths, dtype=np.float64)))

    component = np.full(nodes, -1, dtype=np.int32)
    next_component = 0
    for start in range(nodes):
        if component[start] >= 0:
            continue
        stack = [start]
        component[start] = next_component
        while stack:
            current = stack.pop()
            for other in neighbor[current]:
                if other >= 0 and component[other] < 0:
                    component[other] = next_component
                    stack.append(int(other))
        next_component += 1
    return neighbor, native_scale, component


def _read_slice(spec: SliceSpec, panel: tuple[str, ...]) -> ProNORMSTSlice:
    """Load one standard 10x Visium slice through Scanpy."""

    count_path = spec.path / spec.count_file
    if not count_path.exists():
        raise FileNotFoundError(count_path)
    adata = sc.read_10x_h5(count_path, gex_only=True)
    count_barcodes = pd.Index(adata.obs_names.astype(str))
    if count_barcodes.has_duplicates:
        raise ValueError(f"{count_path}: count matrix has duplicate barcodes")
    if "gene_ids" not in adata.var:
        raise ValueError(f"{count_path}: Scanpy result is missing canonical gene_ids")
    gex_gene_ids = tuple(adata.var["gene_ids"].astype(str))
    gene_ids = pd.Index(gex_gene_ids)
    if gene_ids.has_duplicates:
        raise ValueError(f"{count_path}: Gene Expression Ensembl IDs are not unique")
    panel_indices = gene_ids.get_indexer(panel)
    if bool((panel_indices < 0).any()):
        missing = [panel[index] for index in np.flatnonzero(panel_indices < 0)]
        raise ValueError(f"{count_path}: missing panel genes: {missing[:5]}")
    selected = adata.X[:, panel_indices]
    counts = selected.toarray() if sp.issparse(selected) else np.asarray(selected)
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (len(count_barcodes), len(panel)):
        raise ValueError(f"{count_path}: Scanpy expression dimensions are inconsistent")
    if not np.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError(f"{count_path}: counts must be finite and non-negative")

    positions = _read_positions(spec.path)
    position_barcodes = pd.Index(positions.index.astype(str))
    if set(count_barcodes) != set(position_barcodes):
        missing_count = sorted(set(position_barcodes).difference(count_barcodes))[:5]
        missing_position = sorted(set(count_barcodes).difference(position_barcodes))[:5]
        raise ValueError(
            f"{spec.slice_id}: counts/in-tissue positions are not one-to-one; "
            f"missing_count={missing_count}, missing_position={missing_position}"
        )

    positions = positions.loc[count_barcodes]
    rows = pd.to_numeric(positions["array_row"], errors="raise").to_numpy(np.int64)
    cols = pd.to_numeric(positions["array_col"], errors="raise").to_numpy(np.int64)
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    pixel_row = pd.to_numeric(
        positions["pxl_row_in_fullres"], errors="raise"
    ).to_numpy(np.float64)[order]
    pixel_col = pd.to_numeric(
        positions["pxl_col_in_fullres"], errors="raise"
    ).to_numpy(np.float64)[order]
    full_xy = np.column_stack((pixel_col, pixel_row))
    if not np.isfinite(full_xy).all():
        raise ValueError(f"{spec.slice_id}: full-resolution pixel coordinates are non-finite")
    neighbor, native_scale, component = _build_graph(rows, cols, full_xy)
    counts = counts[order]
    panel_library = counts.sum(axis=1)
    if bool((panel_library <= 0).any()):
        raise ValueError(f"{spec.slice_id}: panel-only library must be positive at every spot")
    expression_x = np.log1p(
        10000.0 * counts / panel_library[:, None]
    ).astype(np.float32)
    return ProNORMSTSlice(
        slice_id=spec.slice_id,
        role=spec.role,
        donor=spec.donor,
        pair=spec.pair,
        barcodes=tuple(count_barcodes[order].astype(str)),
        gene_ids=panel,
        expression_x=expression_x,
        expression_z=np.empty_like(expression_x),
        array_row=rows,
        array_col=cols,
        full_xy=full_xy.astype(np.float32),
        neighbor_index=neighbor,
        native_scale=native_scale,
        component_id=component,
        source_barcodes_sha256=ordered_text_sha256(count_barcodes),
        source_gex_gene_ids_sha256=ordered_text_sha256(gex_gene_ids),
        source_panel_indices_sha256=_array_sha256(
            panel_indices.astype("<i8")
        ),
    )


def prepare_pro_normst_data(
    manifest_path: str | Path,
    panel_path: str | Path,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
) -> ProNORMSTData:
    """Load all roles, fitting every learned statistic on train slices only."""
    panel = load_panel(panel_path)
    specs, metadata = load_split_manifest(manifest_path, count_file)
    roles = {
        role: [_read_slice(spec, panel) for spec in specs[role]]
        for role in ("train", "val", "test")
    }
    train_slices = roles["train"]
    mean_squares = np.stack(
        [
            np.mean(item.expression_x.astype(np.float64) ** 2, axis=0)
            for item in train_slices
        ]
    )
    rms64 = np.sqrt(mean_squares.mean(axis=0))
    gene_scale = np.maximum(rms64.astype(np.float32), np.float32(1e-6))
    detection_rate = np.stack(
        [
            np.mean(item.expression_x > 0, axis=0, dtype=np.float64)
            for item in train_slices
        ]
    ).mean(axis=0)
    positive_weight = np.ones(PANEL_SIZE, dtype=np.float64)
    interior = (detection_rate > 0) & (detection_rate < 1)
    positive_weight[interior] = np.clip(
        np.sqrt((1.0 - detection_rate[interior]) / detection_rate[interior]), 1.0, 3.0
    )

    for role in ("train", "val", "test"):
        for item in roles[role]:
            item.expression_z = (item.expression_x / gene_scale[None, :]).astype(np.float32)
    gene_mean_z = np.stack(
        [
            np.mean(item.expression_z, axis=0, dtype=np.float64)
            for item in train_slices
        ]
    ).mean(axis=0).astype(np.float32)
    preprocessing = ProNORMSTPreprocessing(
        gene_ids=panel,
        gene_scale=gene_scale,
        detection_rate=detection_rate.astype(np.float32),
        positive_weight=positive_weight.astype(np.float32),
        gene_mean_z=gene_mean_z,
        panel_sha256=PANEL_ORDERED_SHA256,
    )
    return ProNORMSTData(roles=roles, preprocessing=preprocessing, split_metadata=metadata)


def test_expression_identity(data: ProNORMSTData) -> dict[str, str]:
    """Hash Scanpy-loaded test expression for the locked evaluation audit."""

    identity: dict[str, str] = {}
    for item in data.roles["test"]:
        identity[item.slice_id] = _array_sha256(item.expression_x.astype("<f4"))
    return identity


__all__ = [
    "NEIGHBOR_DELTAS",
    "OPPOSITE_DIRECTIONS",
    "FROZEN_LODO_DONORS",
    "FROZEN_SLICE_METADATA",
    "FROZEN_SPLITS",
    "PANEL_ORDERED_SHA256",
    "PANEL_SIZE",
    "ProNORMSTData",
    "ProNORMSTPreprocessing",
    "ProNORMSTSlice",
    "SliceSpec",
    "load_panel",
    "load_split_manifest",
    "ordered_text_sha256",
    "prepare_pro_normst_data",
    "test_expression_identity",
]
