"""Geometry-independent global operator and geometry-specific local kernels.

All operator blocks consume the same token layout ``[B, N, C]``.  Geometry is
carried separately, so standard Visium can remain a point set while Visium HD
can retain its Cartesian grid for the local branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HexGeometry:
    """Native first-order Visium topology for the currently visible tokens.

    ``neighbor_index`` contains compact token indices and uses ``-1`` for a
    missing or currently hidden native neighbor.  It must be precomputed from
    the complete Visium lattice; it is deliberately not a KNN graph.
    """

    neighbor_index: torch.Tensor
    relative_xy: torch.Tensor
    neighbor_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class GridGeometry:
    """Cartesian layout associated with a flattened token sequence."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("grid dimensions must be positive")


def grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,C,H,W]`` to ``[B,H*W,C]`` without changing order."""

    if grid.ndim != 4:
        raise ValueError("grid must have shape [B,C,H,W]")
    return grid.flatten(2).transpose(1, 2).contiguous()


def tokens_to_grid(tokens: torch.Tensor, geometry: GridGeometry) -> torch.Tensor:
    """Restore row-major ``[B,H*W,C]`` tokens to ``[B,C,H,W]``."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B,N,C]")
    if tokens.shape[1] != geometry.height * geometry.width:
        raise ValueError("token count does not match grid geometry")
    return tokens.transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], geometry.height, geometry.width
    ).contiguous()


def _expand_batch(tensor: torch.Tensor, batch: int, ndim: int, name: str) -> torch.Tensor:
    if tensor.ndim == ndim - 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != ndim or tensor.shape[0] not in {1, batch}:
        raise ValueError(f"{name} has an invalid batch shape")
    return tensor.expand(batch, *tensor.shape[1:])


class GalerkinOperator(nn.Module):
    """Token-form Galerkin kernel integral with masked empirical quadrature.

    This preserves the SRNO core ``Q (K^T V / measure)`` while removing every
    dependence on grid height/width and replacing 1x1 convolutions with
    token-wise linear maps.  ``quadrature_weight`` is optional; uniform weights
    over valid source tokens are the default.
    """

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        if min(width, num_heads) < 1 or width % num_heads:
            raise ValueError("width must be positive and divisible by num_heads")
        self.width = width
        self.num_heads = num_heads
        self.head_width = width // num_heads
        self.qkv = nn.Linear(width, 3 * width)
        self.key_norm = nn.LayerNorm(self.head_width)
        self.value_norm = nn.LayerNorm(self.head_width)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        quadrature_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(f"tokens must have shape [B,N,{self.width}]")
        batch, points, _ = tokens.shape
        if points < 1:
            raise ValueError("the operator domain must contain at least one token")

        if valid_mask is None:
            valid = torch.ones(
                batch, points, dtype=torch.bool, device=tokens.device
            )
        else:
            if valid_mask.shape != (batch, points):
                raise ValueError("valid_mask must have shape [B,N]")
            valid = valid_mask.to(device=tokens.device, dtype=torch.bool)
        if not valid.any(dim=1).all():
            raise ValueError("every batch item must contain a valid source token")

        if quadrature_weight is None:
            source_weight = valid.to(dtype=tokens.dtype)
        else:
            if quadrature_weight.shape != (batch, points):
                raise ValueError("quadrature_weight must have shape [B,N]")
            if not torch.isfinite(quadrature_weight).all():
                raise ValueError("quadrature_weight must be finite")
            if (quadrature_weight < 0).any():
                raise ValueError("quadrature_weight must be non-negative")
            source_weight = quadrature_weight.to(
                device=tokens.device, dtype=tokens.dtype
            ) * valid.to(dtype=tokens.dtype)
        measure = source_weight.sum(dim=1, keepdim=True).clamp_min(1e-12)

        qkv = self.qkv(tokens).reshape(
            batch, points, self.num_heads, 3 * self.head_width
        )
        query, key, value = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)
        key = self.key_norm(key)
        value = self.value_norm(value)
        # The empirical integral can sum thousands of source tokens.  Accumulate
        # it in float32 even under AMP/FP16, then cast the branch result back.
        kernel = torch.einsum(
            "bhnd,bhne,bn->bhde",
            key.float(),
            value.float(),
            source_weight.float(),
        ) / measure.float()[:, None, :, None]
        integral = torch.einsum("bhnd,bhde->bhne", query.float(), kernel)
        integral = integral.permute(0, 2, 1, 3).reshape(
            batch, points, self.width
        ).to(tokens.dtype)
        return integral * valid[..., None].to(integral.dtype)


class HexNativeLocalOperator(nn.Module):
    """Lightweight kernel over native first-order hexagonal neighbors only."""

    def __init__(self, width: int):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        self.width = width
        self.feature_projection = nn.Linear(width, width)
        self.edge_kernel = nn.Sequential(
            nn.Linear(2, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.Sigmoid(),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        geometry: HexGeometry,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(f"tokens must have shape [B,N,{self.width}]")
        batch, points, _ = tokens.shape
        index = _expand_batch(
            geometry.neighbor_index.to(tokens.device), batch, 3, "neighbor_index"
        )
        relative = _expand_batch(
            geometry.relative_xy.to(device=tokens.device, dtype=tokens.dtype),
            batch,
            4,
            "relative_xy",
        )
        if index.shape[1] != points or relative.shape != (*index.shape, 2):
            raise ValueError("hex geometry does not match token shape")
        if geometry.neighbor_mask is None:
            available = index >= 0
        else:
            available = _expand_batch(
                geometry.neighbor_mask.to(tokens.device),
                batch,
                3,
                "neighbor_mask",
            ).to(torch.bool) & (index >= 0)

        safe_index = index.clamp(min=0)
        if safe_index.numel() and safe_index.max().item() >= points:
            raise ValueError("native neighbor index is out of bounds")
        if valid_mask is None:
            valid = torch.ones(
                batch, points, dtype=torch.bool, device=tokens.device
            )
        else:
            if valid_mask.shape != (batch, points):
                raise ValueError("valid_mask must have shape [B,N]")
            valid = valid_mask.to(device=tokens.device, dtype=torch.bool)
        source_valid = torch.gather(
            valid[:, None, :].expand(-1, points, -1), 2, safe_index
        )
        available = available & source_valid & valid[..., None]

        batch_index = torch.arange(batch, device=tokens.device)[:, None, None]
        neighbor = tokens[batch_index, safe_index]
        message = self.feature_projection(neighbor) * self.edge_kernel(relative)
        weight = available[..., None].to(message.dtype)
        output = (message * weight).sum(dim=2)
        count = weight.sum(dim=2).clamp_min(1.0)
        output = output / count
        return output * valid[..., None].to(output.dtype)


class GridLocalOperator(nn.Module):
    """Standard 3x3 convolutional local operator for a Cartesian grid."""

    def __init__(self, width: int):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        self.width = width
        self.convolution = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(
        self,
        tokens: torch.Tensor,
        geometry: GridGeometry,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        grid = tokens_to_grid(tokens, geometry)
        if valid_mask is None:
            valid = torch.ones(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
        else:
            if valid_mask.shape != tokens.shape[:2]:
                raise ValueError("valid_mask must have shape [B,N]")
            valid = valid_mask.to(device=tokens.device, dtype=torch.bool)
        mask_grid = valid.reshape(
            tokens.shape[0], 1, geometry.height, geometry.width
        ).to(tokens.dtype)
        output = self.convolution(grid * mask_grid) * mask_grid
        return grid_to_tokens(output)


class NeuralOperatorBlock(nn.Module):
    """Unified local-global block with explicit ablation controls."""

    _MODES = {
        "local_only",
        "galerkin_only",
        "parallel",
        "local_then_global",
        "global_then_local",
    }

    def __init__(
        self,
        width: int,
        num_heads: int,
        local_operator: Optional[nn.Module],
        mode: str = "parallel",
        fusion: str = "add",
        learnable_alpha: bool = False,
        project_branches: bool = True,
        ffn_expansion: int = 2,
    ):
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"unsupported operator mode: {mode}")
        if fusion not in {"add", "concat"}:
            raise ValueError("fusion must be add or concat")
        if width < 1 or ffn_expansion < 1:
            raise ValueError("width and ffn_expansion must be positive")
        uses_local = mode != "galerkin_only"
        uses_global = mode != "local_only"
        if uses_local and local_operator is None:
            raise ValueError("the selected mode requires a local operator")

        self.width = width
        self.mode = mode
        self.fusion = fusion
        self.local_operator = local_operator
        self.global_operator = (
            GalerkinOperator(width, num_heads) if uses_global else None
        )
        projection = nn.Linear if project_branches else lambda _a, _b: nn.Identity()
        self.local_projection = projection(width, width) if uses_local else None
        self.global_projection = projection(width, width) if uses_global else None
        if learnable_alpha:
            self.alpha_local = nn.Parameter(torch.ones(()))
            self.alpha_global = nn.Parameter(torch.ones(()))
        else:
            self.register_buffer("alpha_local", torch.ones(()), persistent=True)
            self.register_buffer("alpha_global", torch.ones(()), persistent=True)
        self.concat_projection = (
            nn.Sequential(
                nn.Linear(3 * width, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            if fusion == "concat"
            else None
        )
        hidden = width * ffn_expansion
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )

    def _local(
        self,
        tokens: torch.Tensor,
        geometry: HexGeometry | GridGeometry,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert self.local_operator is not None
        assert self.local_projection is not None
        return self.local_projection(
            self.local_operator(tokens, geometry, valid_mask)
        )

    def _global(
        self,
        tokens: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        quadrature_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert self.global_operator is not None
        assert self.global_projection is not None
        return self.global_projection(
            self.global_operator(tokens, valid_mask, quadrature_weight)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        geometry: HexGeometry | GridGeometry,
        valid_mask: Optional[torch.Tensor] = None,
        quadrature_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(f"tokens must have shape [B,N,{self.width}]")
        zeros = torch.zeros_like(tokens)
        local_delta = zeros
        global_delta = zeros

        if self.mode == "local_only":
            local_delta = self._local(tokens, geometry, valid_mask)
        elif self.mode == "galerkin_only":
            global_delta = self._global(
                tokens, valid_mask, quadrature_weight
            )
        elif self.mode == "parallel":
            local_delta = self._local(tokens, geometry, valid_mask)
            global_delta = self._global(
                tokens, valid_mask, quadrature_weight
            )
        elif self.mode == "local_then_global":
            local_delta = self._local(tokens, geometry, valid_mask)
            global_input = tokens + self.alpha_local * local_delta
            global_delta = self._global(
                global_input, valid_mask, quadrature_weight
            )
        else:
            global_delta = self._global(
                tokens, valid_mask, quadrature_weight
            )
            local_input = tokens + self.alpha_global * global_delta
            local_delta = self._local(local_input, geometry, valid_mask)

        scaled_local = self.alpha_local * local_delta
        scaled_global = self.alpha_global * global_delta
        if self.fusion == "add":
            fused = tokens + scaled_local + scaled_global
        else:
            assert self.concat_projection is not None
            fused = self.concat_projection(
                torch.cat([tokens, scaled_local, scaled_global], dim=-1)
            )
        output = fused + self.ffn(self.norm(fused))
        if valid_mask is not None:
            output = output * valid_mask[..., None].to(output.dtype)
        return output
