"""Minimal geometry-adaptive local-global NORMST models.

The two task models instantiate the same :class:`NeuralOperatorBlock` design
but own independent parameters and use different local discretizations:

* standard Visium: compact visible point tokens plus native hex neighbors;
* Visium HD: coarse Cartesian grid tokens plus 3x3 convolutions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.local_global_operator import (
    GridGeometry,
    GridLocalOperator,
    HexGeometry,
    HexNativeLocalOperator,
    NeuralOperatorBlock,
    grid_to_tokens,
    tokens_to_grid,
)


def build_native_hex_neighbors(
    array_row: torch.Tensor,
    array_col: torch.Tensor,
) -> torch.Tensor:
    """Build the complete six-neighbor graph from raw 10x array coordinates.

    In native Visium coordinates, within-row neighbors differ by two columns
    and the four diagonal neighbors differ by one row and one column.  Missing
    tissue/boundary locations remain ``-1``.
    """

    if array_row.ndim != 1 or array_col.shape != array_row.shape:
        raise ValueError("array_row and array_col must have shape [N]")
    if array_row.numel() < 1:
        raise ValueError("at least one Visium spot is required")
    rows = array_row.detach().cpu().tolist()
    cols = array_col.detach().cpu().tolist()
    if any(int(value) != value for value in rows + cols):
        raise ValueError("Visium array coordinates must be integer-valued")
    coordinate_to_index = {}
    for index, (row, col) in enumerate(zip(rows, cols)):
        key = (int(row), int(col))
        if key in coordinate_to_index:
            raise ValueError("duplicate Visium array coordinate")
        coordinate_to_index[key] = index
    offsets = ((0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1))
    neighbor = torch.full(
        (len(rows), 6),
        -1,
        dtype=torch.long,
        device=array_row.device,
    )
    for index, (row, col) in enumerate(zip(rows, cols)):
        for direction, (delta_row, delta_col) in enumerate(offsets):
            neighbor[index, direction] = coordinate_to_index.get(
                (int(row) + delta_row, int(col) + delta_col), -1
            )
    return neighbor


def build_visible_native_neighbor_graph(
    full_neighbor_index: torch.Tensor,
    full_xy: torch.Tensor,
    visible_index: torch.Tensor,
) -> HexGeometry:
    """Restrict a complete native hex graph to one visible subset.

    Hidden native neighbors become ``-1``.  No distant point is searched for or
    substituted, which is the essential distinction between the local kernel
    and the later physical query interpolation.
    """

    if full_neighbor_index.ndim != 2 or full_neighbor_index.dtype != torch.long:
        raise ValueError("full_neighbor_index must have shape [N,K]")
    points, _ = full_neighbor_index.shape
    if full_xy.shape != (points, 2):
        raise ValueError("full_xy must have shape [N,2]")
    if visible_index.ndim != 1 or visible_index.dtype != torch.long:
        raise ValueError("visible_index must be a one-dimensional long tensor")
    if visible_index.numel() < 1:
        raise ValueError("at least one visible point is required")
    if visible_index.min().item() < 0 or visible_index.max().item() >= points:
        raise ValueError("visible_index is out of bounds")
    if visible_index.unique().numel() != visible_index.numel():
        raise ValueError("visible_index must not contain duplicates")

    device = full_neighbor_index.device
    visible_index = visible_index.to(device)
    lookup = torch.full((points,), -1, dtype=torch.long, device=device)
    lookup[visible_index] = torch.arange(visible_index.numel(), device=device)
    native = full_neighbor_index[visible_index]
    safe_native = native.clamp(min=0)
    if safe_native.numel() and safe_native.max().item() >= points:
        raise ValueError("full native neighbor index is out of bounds")
    present = native >= 0
    compact = lookup[safe_native]
    present = present & (compact >= 0)
    compact = torch.where(present, compact, torch.full_like(compact, -1))

    source_xy = full_xy.to(device)[safe_native]
    target_xy = full_xy.to(device)[visible_index, None, :]
    relative = source_xy - target_xy
    relative = torch.where(present[..., None], relative, torch.zeros_like(relative))
    return HexGeometry(compact, relative, present)


def _normalize_point_coordinates(
    visible_xy: torch.Tensor,
    query_xy: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = valid_mask[..., None].to(torch.float32)
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    visible_float = visible_xy.float()
    center = (visible_float * weight).sum(dim=1, keepdim=True) / count
    squared = ((visible_float - center).square() * weight).sum(
        dim=(1, 2), keepdim=True
    )
    scale = (squared / (count * 2.0)).sqrt().clamp_min(1e-6)
    normalized_visible = (visible_float - center) / scale
    normalized_query = (query_xy.float() - center) / scale
    return normalized_visible, normalized_query


class GeneAffine(nn.Module):
    """Identity-initialized, gene-wise baseline calibration."""

    def __init__(self, n_genes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_genes))
        self.bias = nn.Parameter(torch.zeros(n_genes))

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return expression * self.weight + self.bias


class ExpressionEncoder(nn.Module):
    """Residual gene-expression lifting without coordinate injection."""

    def __init__(self, n_genes: int, width: int):
        super().__init__()
        if min(n_genes, width) < 1:
            raise ValueError("expression encoder dimensions must be positive")
        self.n_genes = n_genes
        self.width = width
        self.skip = nn.Linear(n_genes, width)
        self.main = nn.Sequential(
            nn.Linear(n_genes, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 3 or expression.shape[-1] != self.n_genes:
            raise ValueError(
                f"expression must have shape [B,N,{self.n_genes}]"
            )
        return self.skip(expression) + self.main(expression)


class PhysicalQueryDecoder(nn.Module):
    """Query visible latent tokens and compute an expression IDW baseline."""

    def __init__(
        self,
        width: int,
        n_genes: int,
        neighbors: int = 6,
        idw_power: float = 2.0,
        query_chunk_size: int = 1024,
    ):
        super().__init__()
        if min(width, n_genes, neighbors, query_chunk_size) < 1:
            raise ValueError("query dimensions must be positive")
        if idw_power <= 0:
            raise ValueError("idw_power must be positive")
        self.width = width
        self.n_genes = n_genes
        self.neighbors = neighbors
        self.idw_power = float(idw_power)
        self.query_chunk_size = query_chunk_size
        self.edge_encoder = nn.Sequential(
            nn.Linear(width + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.edge_score[-1].weight)
        nn.init.zeros_(self.edge_score[-1].bias)

    @staticmethod
    def _gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        batch_index = torch.arange(batch, device=values.device)[:, None, None]
        return values[batch_index, index]

    def _idw(self, distance: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        coincident = (distance <= 1e-8) & available
        has_coincident = coincident.any(dim=-1, keepdim=True)
        coincident_weight = coincident.to(distance.dtype)
        coincident_weight = coincident_weight / coincident_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        inverse = distance.clamp_min(1e-8).pow(-self.idw_power)
        inverse = inverse * available.to(inverse.dtype)
        inverse = inverse / inverse.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.where(has_coincident, coincident_weight, inverse)

    def _chunk(
        self,
        visible_features: torch.Tensor,
        visible_expression: torch.Tensor,
        visible_xy: torch.Tensor,
        query_xy: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative_all = query_xy[:, :, None, :] - visible_xy[:, None, :, :]
        distance_all = relative_all.square().sum(dim=-1).sqrt()
        distance_all = distance_all.masked_fill(
            ~visible_mask[:, None, :], float("inf")
        )
        neighbors = min(self.neighbors, visible_xy.shape[1])
        distance, index = torch.topk(
            distance_all, k=neighbors, dim=-1, largest=False, sorted=True
        )
        available = torch.gather(
            visible_mask[:, None, :].expand(-1, query_xy.shape[1], -1),
            2,
            index,
        )
        relative = torch.gather(
            relative_all,
            2,
            index[..., None].expand(-1, -1, -1, 2),
        )
        geometry = torch.cat([relative, distance[..., None]], dim=-1)
        geometry = torch.where(
            available[..., None], geometry, torch.zeros_like(geometry)
        )
        neighbor_features = self._gather(visible_features, index)
        neighbor_expression = self._gather(visible_expression, index)

        message = self.edge_encoder(
            torch.cat([neighbor_features, geometry.to(neighbor_features.dtype)], dim=-1)
        )
        logits = self.edge_score(geometry.to(message.dtype)).squeeze(-1)
        logits = logits.masked_fill(~available, -1e4)
        learned_weight = torch.softmax(logits, dim=-1)
        learned_weight = learned_weight * available.to(learned_weight.dtype)
        learned_weight = learned_weight / learned_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        queried = (message * learned_weight[..., None]).sum(dim=2)

        idw_weight = self._idw(distance, available)
        baseline = (
            neighbor_expression.float() * idw_weight[..., None]
        ).sum(dim=2).to(visible_expression.dtype)
        return queried, baseline

    def forward(
        self,
        visible_features: torch.Tensor,
        visible_expression: torch.Tensor,
        visible_xy: torch.Tensor,
        query_xy: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visible_features.ndim != 3 or visible_features.shape[-1] != self.width:
            raise ValueError("visible_features must have shape [B,Nv,C]")
        if visible_expression.shape[:2] != visible_features.shape[:2]:
            raise ValueError("visible expression and features are misaligned")
        if visible_expression.shape[-1] != self.n_genes:
            raise ValueError("visible expression has an unexpected gene count")
        if visible_xy.shape != (*visible_features.shape[:2], 2):
            raise ValueError("visible_xy must have shape [B,Nv,2]")
        if query_xy.ndim != 3 or query_xy.shape[0] != visible_features.shape[0]:
            raise ValueError("query_xy must have shape [B,Nq,2]")
        if query_xy.shape[-1] != 2:
            raise ValueError("query_xy must have shape [B,Nq,2]")
        if visible_mask.shape != visible_features.shape[:2]:
            raise ValueError("visible_mask must have shape [B,Nv]")
        if not visible_mask.any(dim=1).all():
            raise ValueError("every query domain needs a visible source point")

        feature_chunks = []
        baseline_chunks = []
        for start in range(0, query_xy.shape[1], self.query_chunk_size):
            queried, baseline = self._chunk(
                visible_features,
                visible_expression,
                visible_xy,
                query_xy[:, start:start + self.query_chunk_size],
                visible_mask,
            )
            feature_chunks.append(queried)
            baseline_chunks.append(baseline)
        if not feature_chunks:
            batch = visible_features.shape[0]
            return (
                visible_features.new_empty(batch, 0, self.width),
                visible_expression.new_empty(batch, 0, self.n_genes),
            )
        return torch.cat(feature_chunks, dim=1), torch.cat(
            baseline_chunks, dim=1
        )


class VisiumNORMST(nn.Module):
    """Masked-location reconstruction on compact visible Visium point tokens.

    Initial token content comes only from expression. Coordinates remain
    separate and are used by native relative-edge geometry and physical query.
    """

    def __init__(
        self,
        n_genes: int,
        width: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        operator_mode: str = "parallel",
        fusion: str = "add",
        learnable_alpha: bool = False,
        alpha_global: float = 1.0,
        query_neighbors: int = 6,
        idw_power: float = 2.0,
        query_chunk_size: int = 1024,
        baseline_calibration: bool = False,
    ):
        super().__init__()
        if min(n_genes, width, num_heads, num_layers) < 1:
            raise ValueError("model dimensions must be positive")
        self.n_genes = n_genes
        self.width = width
        self.baseline_calibration_enabled = bool(baseline_calibration)
        self.expression_encoder = ExpressionEncoder(n_genes, width)
        self.blocks = nn.ModuleList([
            NeuralOperatorBlock(
                width=width,
                num_heads=num_heads,
                local_operator=HexNativeLocalOperator(width),
                mode=operator_mode,
                fusion=fusion,
                learnable_alpha=learnable_alpha,
                alpha_global=alpha_global,
            )
            for _ in range(num_layers)
        ])
        self.query_decoder = PhysicalQueryDecoder(
            width,
            n_genes,
            neighbors=query_neighbors,
            idw_power=idw_power,
            query_chunk_size=query_chunk_size,
        )
        self.residual_projection = nn.Sequential(
            nn.Linear(width + 2, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, n_genes),
        )
        self.baseline_calibration = (
            GeneAffine(n_genes) if baseline_calibration else nn.Identity()
        )
        nn.init.normal_(
            self.residual_projection[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.residual_projection[-1].bias)

    def forward(
        self,
        visible_expression: torch.Tensor,
        visible_xy: torch.Tensor,
        query_xy: torch.Tensor,
        native_geometry: HexGeometry,
        visible_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        quadrature_weight: Optional[torch.Tensor] = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if visible_expression.ndim != 3:
            raise ValueError("visible_expression must have shape [B,Nv,G]")
        batch, visible_points, genes = visible_expression.shape
        if genes != self.n_genes or visible_points < 1:
            raise ValueError("visible expression has an invalid shape")
        if visible_xy.shape != (batch, visible_points, 2):
            raise ValueError("visible_xy must have shape [B,Nv,2]")
        if query_xy.ndim != 3 or query_xy.shape[0] != batch or query_xy.shape[-1] != 2:
            raise ValueError("query_xy must have shape [B,Nq,2]")
        if visible_mask is None:
            visible_mask = torch.ones(
                batch, visible_points, dtype=torch.bool, device=visible_expression.device
            )
        else:
            if visible_mask.shape != (batch, visible_points):
                raise ValueError("visible_mask must have shape [B,Nv]")
            visible_mask = visible_mask.to(
                device=visible_expression.device, dtype=torch.bool
            )
        if query_mask is None:
            query_mask = torch.ones(
                query_xy.shape[:2], dtype=torch.bool, device=query_xy.device
            )
        elif query_mask.shape != query_xy.shape[:2]:
            raise ValueError("query_mask must have shape [B,Nq]")
        else:
            query_mask = query_mask.to(device=query_xy.device, dtype=torch.bool)

        normalized_visible, normalized_query = _normalize_point_coordinates(
            visible_xy, query_xy, visible_mask
        )
        tokens = self.expression_encoder(visible_expression)
        tokens = tokens * visible_mask[..., None].to(tokens.dtype)
        initial_tokens = tokens
        for block in self.blocks:
            tokens = block(
                tokens,
                native_geometry,
                visible_mask,
                quadrature_weight,
            )
        query_feature, baseline = self.query_decoder(
            tokens,
            visible_expression,
            normalized_visible,
            normalized_query,
            visible_mask,
        )
        residual = self.residual_projection(
            torch.cat([
                query_feature,
                normalized_query.to(query_feature.dtype),
            ], dim=-1)
        )
        query_weight = query_mask[..., None].to(residual.dtype)
        baseline = baseline * query_weight
        prediction = (
            residual + self.baseline_calibration(baseline)
        ) * query_weight
        if not return_auxiliary:
            return prediction
        return prediction, {
            "baseline": baseline,
            "h0": initial_tokens,
            "hl": tokens,
            "visible_mask": visible_mask,
        }


class VisiumHDNORMST(nn.Module):
    """Joint-gene 16-to-8 micrometre grid super-resolution model."""

    def __init__(
        self,
        n_genes: int,
        width: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        operator_mode: str = "parallel",
        fusion: str = "add",
        learnable_alpha: bool = False,
        scale: int = 2,
        baseline_mode: str = "bilinear",
    ):
        super().__init__()
        if min(n_genes, width, num_heads, num_layers, scale) < 1:
            raise ValueError("model dimensions must be positive")
        if baseline_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("unsupported baseline interpolation mode")
        self.n_genes = n_genes
        self.width = width
        self.scale = scale
        self.baseline_mode = baseline_mode
        self.grid_lifting = nn.Conv2d(n_genes, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            NeuralOperatorBlock(
                width=width,
                num_heads=num_heads,
                local_operator=GridLocalOperator(width),
                mode=operator_mode,
                fusion=fusion,
                learnable_alpha=learnable_alpha,
            )
            for _ in range(num_layers)
        ])
        self.fine_decoder = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, n_genes, kernel_size=1),
        )
        nn.init.zeros_(self.fine_decoder[-1].weight)
        nn.init.zeros_(self.fine_decoder[-1].bias)

    @staticmethod
    def _interpolate(grid: torch.Tensor, scale: int, mode: str) -> torch.Tensor:
        options = {"scale_factor": scale, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            options["align_corners"] = False
        return F.interpolate(grid, **options)

    @staticmethod
    def _reshape_baseline_scale(
        scale: torch.Tensor,
        batch: int,
        genes: int,
    ) -> torch.Tensor:
        if scale.ndim == 1 and scale.shape[0] == genes:
            return scale.reshape(1, genes, 1, 1)
        if scale.ndim == 2 and scale.shape == (batch, genes):
            return scale.reshape(batch, genes, 1, 1)
        raise ValueError("baseline_scale must have shape [G] or [B,G]")

    def forward(
        self,
        coarse_expression: torch.Tensor,
        coarse_valid_mask: Optional[torch.Tensor] = None,
        fine_valid_mask: Optional[torch.Tensor] = None,
        baseline_scale: Optional[torch.Tensor] = None,
        quadrature_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if coarse_expression.ndim != 4 or coarse_expression.shape[1] != self.n_genes:
            raise ValueError("coarse_expression must have shape [B,G,H,W]")
        batch, _, height, width = coarse_expression.shape
        if coarse_valid_mask is None:
            coarse_valid_mask = torch.ones(
                batch, 1, height, width,
                dtype=torch.bool,
                device=coarse_expression.device,
            )
        elif coarse_valid_mask.shape != (batch, 1, height, width):
            raise ValueError("coarse_valid_mask must have shape [B,1,H,W]")
        else:
            coarse_valid_mask = coarse_valid_mask.to(
                device=coarse_expression.device, dtype=torch.bool
            )
        valid_tokens = coarse_valid_mask.flatten(2).squeeze(1)
        geometry = GridGeometry(height, width)
        coarse_features = self.grid_lifting(coarse_expression)
        coarse_features = coarse_features * coarse_valid_mask.to(
            coarse_features.dtype
        )
        tokens = grid_to_tokens(coarse_features)
        for block in self.blocks:
            tokens = block(
                tokens,
                geometry,
                valid_tokens,
                quadrature_weight,
            )
        latent_grid = tokens_to_grid(tokens, geometry)
        fine_latent = self._interpolate(
            latent_grid, self.scale, self.baseline_mode
        )
        residual = self.fine_decoder(fine_latent)
        baseline = self._interpolate(
            coarse_expression, self.scale, self.baseline_mode
        )
        if baseline_scale is not None:
            baseline = baseline * self._reshape_baseline_scale(
                baseline_scale.to(device=baseline.device, dtype=baseline.dtype),
                batch,
                self.n_genes,
            )

        target_shape = (height * self.scale, width * self.scale)
        if fine_valid_mask is None:
            fine_valid_mask = F.interpolate(
                coarse_valid_mask.to(coarse_expression.dtype),
                size=target_shape,
                mode="nearest",
            ) > 0.5
        elif fine_valid_mask.shape != (batch, 1, *target_shape):
            raise ValueError("fine_valid_mask is not aligned to the fine grid")
        else:
            fine_valid_mask = fine_valid_mask.to(
                device=coarse_expression.device, dtype=torch.bool
            )
        return (baseline + residual) * fine_valid_mask.to(residual.dtype)
