"""AE-latent NORMST without changes to the existing Visium model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models.geometry_adaptive_normst import (
    ExpressionEncoder,
    PhysicalQueryDecoder,
    _normalize_point_coordinates,
)
from models.local_global_operator import (
    HexGeometry,
    HexNativeLocalOperator,
    NeuralOperatorBlock,
)


class AENORMST(nn.Module):
    """Predict frozen AE composition coordinates at held-out Visium spots."""

    def __init__(
        self,
        composition_dim: int,
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
        residual_head_width_multiplier: int = 2,
        input_coordinate_lifting: bool = False,
    ):
        super().__init__()
        if min(
            composition_dim,
            width,
            num_heads,
            num_layers,
            residual_head_width_multiplier,
        ) < 1:
            raise ValueError("model dimensions must be positive")
        self.composition_dim = int(composition_dim)
        self.width = int(width)
        self.expression_encoder = ExpressionEncoder(composition_dim, width)
        # Always instantiated. Passing zeros creates the matched no-library
        # control with exactly the same shape and parameter count.
        self.library_context_lifting = nn.Linear(1, width, bias=False)
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
            composition_dim,
            neighbors=query_neighbors,
            idw_power=idw_power,
            query_chunk_size=query_chunk_size,
        )
        self.residual_projection = nn.Sequential(
            nn.Linear(width + 2, residual_head_width_multiplier * width),
            nn.GELU(),
            nn.Linear(residual_head_width_multiplier * width, composition_dim),
        )
        nn.init.normal_(self.residual_projection[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_projection[-1].bias)
        if input_coordinate_lifting:
            with torch.random.fork_rng(devices=[]):
                self.coordinate_lifting = nn.Linear(2, width, bias=False)

    def forward(
        self,
        visible_composition: torch.Tensor,
        visible_library_context: torch.Tensor,
        visible_xy: torch.Tensor,
        query_xy: torch.Tensor,
        native_geometry: HexGeometry,
        visible_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        quadrature_weight: Optional[torch.Tensor] = None,
        return_auxiliary: bool = False,
        return_block_diagnostics: bool = False,
    ):
        if visible_composition.ndim != 3:
            raise ValueError("visible_composition must have shape [B,Nv,D]")
        batch, visible_points, dimensions = visible_composition.shape
        if dimensions != self.composition_dim or visible_points < 1:
            raise ValueError("visible composition has an invalid shape")
        if visible_library_context.shape != (batch, visible_points, 1):
            raise ValueError("visible_library_context must have shape [B,Nv,1]")
        if visible_xy.shape != (batch, visible_points, 2):
            raise ValueError("visible_xy must have shape [B,Nv,2]")
        if query_xy.ndim != 3 or query_xy.shape[0] != batch or query_xy.shape[-1] != 2:
            raise ValueError("query_xy must have shape [B,Nq,2]")
        if visible_mask is None:
            visible_mask = torch.ones(
                batch,
                visible_points,
                dtype=torch.bool,
                device=visible_composition.device,
            )
        elif visible_mask.shape != (batch, visible_points):
            raise ValueError("visible_mask must have shape [B,Nv]")
        else:
            visible_mask = visible_mask.to(
                device=visible_composition.device, dtype=torch.bool
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
        expression_tokens = self.expression_encoder(visible_composition)
        library_tokens = self.library_context_lifting(
            visible_library_context.to(expression_tokens.dtype)
        )
        tokens = expression_tokens + library_tokens
        coordinate_tokens = None
        if hasattr(self, "coordinate_lifting"):
            coordinate_tokens = self.coordinate_lifting(
                normalized_visible.to(tokens.dtype)
            )
            tokens = tokens + coordinate_tokens
        tokens = tokens * visible_mask[..., None].to(tokens.dtype)
        initial_tokens = tokens

        block_diagnostics = []
        for block in self.blocks:
            if return_block_diagnostics:
                tokens, block_values = block(
                    tokens,
                    native_geometry,
                    visible_mask,
                    quadrature_weight,
                    return_diagnostics=True,
                )
                block_diagnostics.append(block_values)
            else:
                tokens = block(
                    tokens, native_geometry, visible_mask, quadrature_weight
                )

        query_feature, baseline = self.query_decoder(
            tokens,
            visible_composition,
            normalized_visible,
            normalized_query,
            visible_mask,
        )
        residual = self.residual_projection(
            torch.cat(
                [query_feature, normalized_query.to(query_feature.dtype)], dim=-1
            )
        )
        query_weight = query_mask[..., None].to(residual.dtype)
        prediction = (residual + baseline) * query_weight
        if not return_auxiliary:
            return prediction
        auxiliary = {
            "baseline": baseline * query_weight,
            "h0": initial_tokens,
            "hl": tokens,
            "expression_lifting": expression_tokens,
            "library_context_lifting": library_tokens,
            "visible_mask": visible_mask,
            "hidden_library_used": False,
        }
        if coordinate_tokens is not None:
            auxiliary["coordinate_lifting"] = coordinate_tokens
        if return_block_diagnostics:
            auxiliary["block_diagnostics"] = block_diagnostics
        return prediction, auxiliary
