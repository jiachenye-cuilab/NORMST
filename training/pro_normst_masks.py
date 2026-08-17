"""Deterministic ordinary and spatial-gap masks for ProNORMST."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch


MASK_SCHEMA = "pro-normst-mask-v1"
BASE_MASK_SEED = 2027
EXPECTED_HEX_BALL_SIZE = {2: 19, 3: 37, 4: 61, 5: 91}


MaskFamily = Literal["ordinary", "gap"]


@dataclass(frozen=True)
class GapHole:
    center: int
    radius: int
    nodes: tuple[int, ...]
    protected_ring: tuple[int, ...] = ()


@dataclass(frozen=True)
class ProMask:
    family: MaskFamily
    query_index: np.ndarray
    visible_index: np.ndarray
    n_target: int
    realized_fraction: float
    provenance: np.ndarray
    depth: np.ndarray
    query_component: np.ndarray
    query_component_size: np.ndarray
    full_degree: np.ndarray
    holes: tuple[GapHole, ...]
    identity: dict[str, Any]
    substream_seeds: dict[str, int]

    def manifest(self) -> dict[str, Any]:
        counts = {
            value: int(np.count_nonzero(self.provenance == value))
            for value in np.unique(self.provenance)
        }
        return {
            "schema": MASK_SCHEMA,
            "identity": self.identity,
            "family": self.family,
            "n_target": self.n_target,
            "n_query": int(self.query_index.size),
            "realized_fraction": self.realized_fraction,
            "query_index": self.query_index.tolist(),
            "provenance_counts": counts,
            "hole_sizes": [len(hole.nodes) for hole in self.holes],
            "holes": [
                {
                    "center": hole.center,
                    "radius": hole.radius,
                    "nodes": list(hole.nodes),
                    "protected_ring": list(hole.protected_ring),
                }
                for hole in self.holes
            ],
            "depth": self.depth.tolist(),
            "query_component": self.query_component.tolist(),
            "query_component_size": self.query_component_size.tolist(),
            "full_degree": self.full_degree.tolist(),
            "substream_seeds": self.substream_seeds,
            "torch_version": str(torch.__version__),
        }


@dataclass(frozen=True)
class MaskGeometry:
    neighbor_index: np.ndarray
    degree: np.ndarray
    standard_candidates: dict[int, tuple[GapHole, ...]]
    r2_candidates: tuple[GapHole, ...]

    @property
    def n_nodes(self) -> int:
        return int(self.neighbor_index.shape[0])


def canonical_seed(identity: dict[str, Any], domain: str) -> int:
    payload = {
        "base_mask_seed": BASE_MASK_SEED,
        "domain": domain,
        **identity,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def make_mask_identity(
    *,
    protocol: str,
    fold: str,
    role: str,
    slice_id: str,
    family: MaskFamily,
    mask_index: int,
    attempt_index: int = 0,
) -> dict[str, Any]:
    if family not in ("ordinary", "gap"):
        raise ValueError(f"unknown mask family: {family}")
    if mask_index < 0 or attempt_index != 0:
        raise ValueError("mask_index must be non-negative and attempt_index must be 0")
    return {
        "schema": MASK_SCHEMA,
        "protocol": str(protocol),
        "fold": str(fold),
        "role": str(role),
        "slice": str(slice_id),
        "family": family,
        "mask_index": int(mask_index),
        "attempt_index": int(attempt_index),
    }


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _distances_within(
    neighbor_index: np.ndarray,
    center: int,
    max_radius: int,
) -> dict[int, int]:
    distance = {int(center): 0}
    queue = [int(center)]
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        current_distance = distance[node]
        if current_distance >= max_radius:
            continue
        for other in neighbor_index[node]:
            other = int(other)
            if other >= 0 and other not in distance:
                distance[other] = current_distance + 1
                queue.append(other)
    return distance


def build_mask_geometry(neighbor_index: np.ndarray) -> MaskGeometry:
    neighbor = np.asarray(neighbor_index, dtype=np.int64)
    if neighbor.ndim != 2 or neighbor.shape[1] != 6:
        raise ValueError("neighbor_index must have shape [N,6]")
    nodes = neighbor.shape[0]
    if nodes < 2 or (neighbor < -1).any() or (neighbor >= nodes).any():
        raise ValueError("neighbor_index contains invalid node indices")
    degree = np.count_nonzero(neighbor >= 0, axis=1).astype(np.int16)
    standard: dict[int, list[GapHole]] = {3: [], 4: []}
    r2: list[GapHole] = []
    for center in range(nodes):
        distances = _distances_within(neighbor, center, 5)
        ball2 = tuple(sorted(node for node, value in distances.items() if value <= 2))
        eccentricity2 = max((distances[node] for node in ball2), default=0)
        if eccentricity2 == 2 and len(ball2) >= 15:
            r2.append(GapHole(center=center, radius=2, nodes=ball2))
        for radius in (3, 4):
            outer_radius = radius + 1
            outer = tuple(
                sorted(node for node, value in distances.items() if value <= outer_radius)
            )
            if len(outer) != EXPECTED_HEX_BALL_SIZE[outer_radius]:
                continue
            core = tuple(sorted(node for node, value in distances.items() if value <= radius))
            if len(core) != EXPECTED_HEX_BALL_SIZE[radius]:
                continue
            core_set = set(core)
            ring = tuple(node for node in outer if node not in core_set)
            standard[radius].append(
                GapHole(center=center, radius=radius, nodes=core, protected_ring=ring)
            )
    return MaskGeometry(
        neighbor_index=neighbor,
        degree=degree,
        standard_candidates={radius: tuple(value) for radius, value in standard.items()},
        r2_candidates=tuple(r2),
    )


def _query_metadata(
    geometry: MaskGeometry,
    query_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = geometry.n_nodes
    query_mask = np.zeros(nodes, dtype=bool)
    query_mask[query_index] = True
    visible = ~query_mask

    distance = np.full(nodes, -1, dtype=np.int32)
    queue = np.flatnonzero(visible).astype(np.int64).tolist()
    distance[visible] = 0
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for other in geometry.neighbor_index[node]:
            other = int(other)
            if other >= 0 and distance[other] < 0:
                distance[other] = distance[node] + 1
                queue.append(other)

    component = np.full(nodes, -1, dtype=np.int32)
    component_sizes: list[int] = []
    next_component = 0
    for start in query_index:
        start = int(start)
        if component[start] >= 0:
            continue
        members = [start]
        component[start] = next_component
        member_cursor = 0
        while member_cursor < len(members):
            node = members[member_cursor]
            member_cursor += 1
            for other in geometry.neighbor_index[node]:
                other = int(other)
                if other >= 0 and query_mask[other] and component[other] < 0:
                    component[other] = next_component
                    members.append(other)
        component_sizes.append(len(members))
        next_component += 1
    size_by_node = np.zeros(nodes, dtype=np.int32)
    for component_id, size in enumerate(component_sizes):
        size_by_node[component == component_id] = size
    return (
        distance[query_index],
        component[query_index],
        size_by_node[query_index],
        geometry.degree[query_index].astype(np.int16),
    )


def _finish_mask(
    geometry: MaskGeometry,
    family: MaskFamily,
    query_index: np.ndarray,
    provenance_by_node: np.ndarray,
    holes: list[GapHole],
    identity: dict[str, Any],
    seeds: dict[str, int],
) -> ProMask:
    query = np.asarray(np.sort(query_index), dtype=np.int64)
    if query.size < 1 or np.unique(query).size != query.size:
        raise ValueError("query set must be non-empty and unique")
    visible_mask = np.ones(geometry.n_nodes, dtype=bool)
    visible_mask[query] = False
    visible = np.flatnonzero(visible_mask).astype(np.int64)
    depth, component, component_size, degree = _query_metadata(geometry, query)
    target = geometry.n_nodes // 2
    return ProMask(
        family=family,
        query_index=query,
        visible_index=visible,
        n_target=target,
        realized_fraction=float(query.size / geometry.n_nodes),
        provenance=provenance_by_node[query].astype("U16"),
        depth=depth,
        query_component=component,
        query_component_size=component_size,
        full_degree=degree,
        holes=tuple(holes),
        identity=dict(identity),
        substream_seeds=dict(seeds),
    )


def generate_ordinary_mask(
    geometry: MaskGeometry,
    identity: dict[str, Any],
) -> ProMask:
    if identity.get("family") != "ordinary":
        raise ValueError("ordinary generator requires an ordinary identity")
    seed = canonical_seed(identity, "ordinary-query")
    order = torch.randperm(geometry.n_nodes, generator=_generator(seed)).numpy()
    query = order[: geometry.n_nodes // 2].astype(np.int64)
    provenance = np.full(geometry.n_nodes, "", dtype="U16")
    provenance[query] = "ordinary"
    return _finish_mask(
        geometry, "ordinary", query, provenance, [], identity, {"ordinary-query": seed}
    )


def generate_gap_mask(
    geometry: MaskGeometry,
    identity: dict[str, Any],
) -> ProMask:
    if identity.get("family") != "gap":
        raise ValueError("gap generator requires a gap identity")
    target = geometry.n_nodes // 2
    domains = ("standard-r3", "standard-r4", "radius-start", "r2", "random-fill")
    seeds = {domain: canonical_seed(identity, domain) for domain in domains}
    permutations = {
        radius: torch.randperm(
            len(geometry.standard_candidates[radius]),
            generator=_generator(seeds[f"standard-r{radius}"]),
        ).tolist()
        for radius in (3, 4)
    }
    start_radius = 3 + int(
        torch.randint(0, 2, (1,), generator=_generator(seeds["radius-start"])).item()
    )
    query = np.zeros(geometry.n_nodes, dtype=bool)
    protected = np.zeros(geometry.n_nodes, dtype=bool)
    provenance = np.full(geometry.n_nodes, "", dtype="U16")
    holes: list[GapHole] = []
    accepted_mass = {3: 0, 4: 0}
    cursor = {3: 0, 4: 0}
    query_count = 0

    while cursor[3] < len(permutations[3]) or cursor[4] < len(permutations[4]):
        available = [radius for radius in (3, 4) if cursor[radius] < len(permutations[radius])]
        if len(available) == 1:
            radius = available[0]
        elif accepted_mass[3] == accepted_mass[4]:
            radius = start_radius
        else:
            radius = 3 if accepted_mass[3] < accepted_mass[4] else 4
        candidate_index = permutations[radius][cursor[radius]]
        cursor[radius] += 1
        candidate = geometry.standard_candidates[radius][candidate_index]
        core = np.asarray(candidate.nodes, dtype=np.int64)
        ring = np.asarray(candidate.protected_ring, dtype=np.int64)
        if query_count + core.size > target:
            continue
        # Protected rings may overlap each other: they are shared visible
        # buffer nodes.  A core may not enter any ring, and a ring may not
        # cover an already accepted core.
        if query[core].any() or protected[core].any() or query[ring].any():
            continue
        query[core] = True
        protected[ring] = True
        provenance[core] = f"standard-r{radius}"
        accepted_mass[radius] += int(core.size)
        query_count += int(core.size)
        holes.append(candidate)

    r2_order = torch.randperm(
        len(geometry.r2_candidates), generator=_generator(seeds["r2"])
    ).tolist()
    for candidate_index in r2_order:
        candidate = geometry.r2_candidates[candidate_index]
        core = np.asarray(candidate.nodes, dtype=np.int64)
        if query_count + core.size > target:
            continue
        if query[core].any() or protected[core].any():
            continue
        query[core] = True
        provenance[core] = "r2"
        query_count += int(core.size)
        holes.append(candidate)

    remaining = target - query_count
    if remaining > 0:
        eligible = np.flatnonzero(~query & ~protected).astype(np.int64)
        order = torch.randperm(
            eligible.size, generator=_generator(seeds["random-fill"])
        ).numpy()
        selected = eligible[order[: min(remaining, eligible.size)]]
        query[selected] = True
        provenance[selected] = "random"
    return _finish_mask(
        geometry,
        "gap",
        np.flatnonzero(query).astype(np.int64),
        provenance,
        holes,
        identity,
        seeds,
    )


def generate_mask(geometry: MaskGeometry, identity: dict[str, Any]) -> ProMask:
    family = identity.get("family")
    if family == "ordinary":
        return generate_ordinary_mask(geometry, identity)
    if family == "gap":
        return generate_gap_mask(geometry, identity)
    raise ValueError(f"unknown mask family: {family}")


def fixed_mask_bank(
    geometry: MaskGeometry,
    *,
    protocol: str,
    fold: str,
    role: str,
    slice_id: str,
    family: MaskFamily,
    size: int = 16,
) -> tuple[ProMask, ...]:
    if size != 16:
        raise ValueError("contracted validation/test bank size is 16")
    return tuple(
        generate_mask(
            geometry,
            make_mask_identity(
                protocol=protocol,
                fold=fold,
                role=role,
                slice_id=slice_id,
                family=family,
                mask_index=index,
            ),
        )
        for index in range(size)
    )


__all__ = [
    "BASE_MASK_SEED",
    "GapHole",
    "MASK_SCHEMA",
    "MaskGeometry",
    "ProMask",
    "build_mask_geometry",
    "canonical_seed",
    "fixed_mask_bank",
    "generate_gap_mask",
    "generate_mask",
    "generate_ordinary_mask",
    "make_mask_identity",
]
