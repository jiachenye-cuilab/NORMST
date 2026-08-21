"""Visible-anchored progressive query propagation for masked Visium spots.

This module is intentionally independent from the existing one-shot
``VisiumNORMST`` and ``AENORMST`` implementations.  It contains only the
proposed model topology and geometry contract; training/data integration is a
separate concern.

The forward interface receives expression for original-visible spots only.
Hidden/query expression is deliberately absent, so it cannot leak into global
memory or recurrent local propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class NonAffineRMSNorm(nn.Module):
    """RMS normalization without trainable scale or offset."""

    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.width = int(width)
        self.eps = float(eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.width:
            raise ValueError(f"last dimension must be {self.width}")
        inverse_rms = values.float().square().mean(
            dim=-1, keepdim=True
        ).add(self.eps).rsqrt()
        return values * inverse_rms.to(values.dtype)


class DeterministicExpressionAutoencoder(nn.Module):
    """The frozen-representation AE topology selected for phase one.

    This class defines only the model.  Fold construction, reconstruction loss,
    model selection, and checkpoint serialization remain training contracts.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        if min(n_genes, latent_dim, hidden_dim) < 1:
            raise ValueError("autoencoder dimensions must be positive")
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_genes, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.n_genes),
        )

    def encode(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.shape[-1] != self.n_genes:
            raise ValueError(f"expression last dimension must be {self.n_genes}")
        return self.encoder(expression)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent last dimension must be {self.latent_dim}")
        return self.decoder(latent)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(expression))


class FrozenLatentEncoder(nn.Module):
    """Freeze a pointwise AE encoder and apply fixed channel standardization."""

    def __init__(
        self,
        encoder: nn.Module,
        n_genes: int,
        latent_mean: torch.Tensor,
        latent_scale: torch.Tensor,
    ):
        super().__init__()
        if n_genes < 1:
            raise ValueError("n_genes must be positive")
        mean = torch.as_tensor(latent_mean, dtype=torch.float32)
        scale = torch.as_tensor(latent_scale, dtype=torch.float32)
        if mean.ndim != 1 or scale.shape != mean.shape or mean.numel() < 1:
            raise ValueError("latent statistics must be matching non-empty vectors")
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("latent mean must be finite")
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("latent scale must be finite and positive")

        self.encoder = encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.n_genes = int(n_genes)
        self.latent_dim = int(mean.numel())
        self.register_buffer("latent_mean", mean)
        self.register_buffer("latent_scale", scale)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 3 or expression.shape[-1] != self.n_genes:
            raise ValueError(
                f"expression must have shape [B,N,{self.n_genes}]"
            )
        shape = expression.shape
        latent = self.encoder(expression.reshape(-1, self.n_genes))
        if not isinstance(latent, torch.Tensor):
            raise TypeError("frozen encoder must return a tensor")
        if latent.shape != (shape[0] * shape[1], self.latent_dim):
            raise ValueError(
                "frozen encoder output does not match the configured latent dimension"
            )
        latent = latent.reshape(shape[0], shape[1], self.latent_dim).float()
        return (latent - self.latent_mean) / self.latent_scale

    encode_standardized = forward


@dataclass(frozen=True)
class FullHexGeometry:
    """Complete native Visium topology, including visible and query nodes.

    ``neighbor_index`` has six globally consistent direction slots and uses
    ``-1`` only for a missing tissue neighbor.  Hidden nodes must never be
    removed from this table.  Tensors may be unbatched (``[N,...]``) or batched
    (``[B,N,...]``); a leading singleton batch is broadcast when needed.
    """

    xy: torch.Tensor
    neighbor_index: torch.Tensor
    node_mask: Optional[torch.Tensor] = None
    native_scale: Optional[torch.Tensor] = None
    indices_validated: bool = False


def _expand_batch(
    tensor: torch.Tensor,
    batch: int,
    unbatched_ndim: int,
    name: str,
) -> torch.Tensor:
    if tensor.ndim == unbatched_ndim:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != unbatched_ndim + 1 or tensor.shape[0] not in {1, batch}:
        raise ValueError(f"{name} has an invalid batch shape")
    return tensor.expand(batch, *tensor.shape[1:])


def _batch_index(batch: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch, device=device)[:, None, None]


def _gather_nodes(
    values: torch.Tensor,
    index: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    safe = index.clamp(min=0)
    batch = torch.arange(values.shape[0], device=values.device)[:, None]
    gathered = values[batch, safe]
    trailing = (1,) * (gathered.ndim - valid.ndim)
    return gathered * valid.reshape(*valid.shape, *trailing).to(gathered.dtype)


def _expand_compact_index(
    index: torch.Tensor,
    batch: int,
    points: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(index, device=device, dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] not in {1, batch} or value.shape[1] != points:
        raise ValueError(f"{name} must have shape [B,{points}] or [{points}]")
    return value.expand(batch, points)


def _native_edge_scale(
    xy: torch.Tensor,
    neighbor_index: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Median valid native-edge length for each complete tissue graph."""

    batch, nodes, directions = neighbor_index.shape
    safe = neighbor_index.clamp(min=0)
    batch_id = _batch_index(batch, neighbor_index.device)
    neighbor_xy = xy[batch_id, safe]
    length = (neighbor_xy.float() - xy.float()[:, :, None, :]).square().sum(
        dim=-1
    ).sqrt()
    scales = []
    for batch_id_value in range(batch):
        valid_length = length[batch_id_value][
            edge_mask[batch_id_value]
            & (length[batch_id_value] > eps)
        ]
        if valid_length.numel() == 0:
            raise ValueError(
                "each full tissue graph must contain a positive-length native edge"
            )
        scales.append(valid_length.median())
    return torch.stack(scales).to(device=xy.device, dtype=torch.float32)


def materialize_full_hex_geometry(
    geometry: FullHexGeometry,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Validate and batch a complete geometry without consulting visibility."""

    xy = _expand_batch(
        torch.as_tensor(geometry.xy, device=device, dtype=dtype),
        batch,
        2,
        "geometry.xy",
    )
    if xy.shape[-1] != 2 or not bool(torch.isfinite(xy).all()):
        raise ValueError("geometry.xy must be finite with shape [B,N,2]")
    nodes = xy.shape[1]
    neighbor = _expand_batch(
        torch.as_tensor(geometry.neighbor_index, device=device, dtype=torch.long),
        batch,
        2,
        "geometry.neighbor_index",
    )
    if neighbor.shape != (batch, nodes, 6):
        raise ValueError("geometry.neighbor_index must have shape [B,N,6]")
    if bool((neighbor < -1).any()):
        raise ValueError("native neighbor indices must be -1 or non-negative")
    if not geometry.indices_validated and bool((neighbor >= nodes).any()):
        raise ValueError("native neighbor index is out of bounds")

    if geometry.node_mask is None:
        node_mask = torch.ones(batch, nodes, device=device, dtype=torch.bool)
    else:
        node_mask = _expand_batch(
            torch.as_tensor(geometry.node_mask, device=device, dtype=torch.bool),
            batch,
            1,
            "geometry.node_mask",
        )
        if node_mask.shape != (batch, nodes):
            raise ValueError("geometry.node_mask must have shape [B,N]")
    if not bool(node_mask.any(dim=1).all()):
        raise ValueError("each batch item must contain a tissue node")

    safe = neighbor.clamp(min=0)
    source_node_valid = node_mask[_batch_index(batch, device), safe]
    edge_mask = (
        (neighbor >= 0)
        & node_mask[:, :, None]
        & source_node_valid
    )

    if geometry.native_scale is None:
        native_scale = _native_edge_scale(
            xy, neighbor, node_mask, edge_mask, eps
        )
    else:
        native_scale = torch.as_tensor(
            geometry.native_scale, device=device, dtype=torch.float32
        )
        if native_scale.ndim == 0:
            native_scale = native_scale[None]
        if native_scale.ndim != 1 or native_scale.shape[0] not in {1, batch}:
            raise ValueError("geometry.native_scale must be scalar or have shape [B]")
        native_scale = native_scale.expand(batch)
        if not bool(torch.isfinite(native_scale).all()) or bool(
            (native_scale <= eps).any()
        ):
            raise ValueError("geometry.native_scale must be finite and positive")

    return {
        "xy": xy,
        "neighbor_index": neighbor,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "degree": edge_mask.sum(dim=-1),
        "native_scale": native_scale,
    }


class VisibleOnlyRadialCrossAttention(nn.Module):
    """Bare multi-head softmax readout from original-visible latent memory."""

    def __init__(
        self,
        latent_dim: int,
        width: int = 256,
        num_heads: int = 8,
        radial_hidden_dim: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if min(latent_dim, width, num_heads, radial_hidden_dim) < 1:
            raise ValueError("attention dimensions must be positive")
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.latent_dim = int(latent_dim)
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.head_dim = self.width // self.num_heads

        self.mask_token = nn.Parameter(torch.empty(self.width))
        self.query_projection = nn.Linear(self.width, self.width)
        self.key_projection = nn.Linear(self.latent_dim, self.width)
        self.value_projection = nn.Linear(self.latent_dim, self.width)
        self.output_projection = nn.Linear(self.width, self.width)
        self.radial_bias = nn.Sequential(
            nn.Linear(2, radial_hidden_dim),
            nn.GELU(),
            nn.Linear(radial_hidden_dim, self.num_heads),
        )
        self.output_norm = NonAffineRMSNorm(self.width, eps=norm_eps)

        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        nn.init.zeros_(self.radial_bias[-1].weight)
        nn.init.zeros_(self.radial_bias[-1].bias)

    def forward(
        self,
        visible_latent: torch.Tensor,
        visible_xy: torch.Tensor,
        query_xy: torch.Tensor,
        native_scale: torch.Tensor,
        visible_mask: torch.Tensor,
        query_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if visible_latent.ndim != 3 or visible_latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"visible_latent must have shape [B,Nv,{self.latent_dim}]"
            )
        batch, visible_points, _ = visible_latent.shape
        query_points = query_xy.shape[1]
        if visible_xy.shape != (batch, visible_points, 2):
            raise ValueError("visible_xy does not align with visible_latent")
        if query_xy.shape != (batch, query_points, 2):
            raise ValueError("query_xy must have shape [B,Nq,2]")
        if visible_mask.shape != (batch, visible_points):
            raise ValueError("visible_mask does not align with visible_latent")
        if query_mask.shape != (batch, query_points):
            raise ValueError("query_mask does not align with query_xy")
        if not bool(visible_mask.any(dim=1).all()):
            raise ValueError("every batch item must contain an original-visible spot")
        if native_scale.shape != (batch,):
            raise ValueError("native_scale must have shape [B]")

        query_token = self.mask_token.view(1, 1, self.width).expand(
            batch, query_points, self.width
        )
        query = self.query_projection(query_token).reshape(
            batch, query_points, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = self.key_projection(visible_latent).reshape(
            batch, visible_points, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value = self.value_projection(visible_latent).reshape(
            batch, visible_points, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        displacement = (
            query_xy.float()[:, :, None, :]
            - visible_xy.float()[:, None, :, :]
        )
        distance = displacement.square().sum(dim=-1).sqrt()
        distance = distance / native_scale[:, None, None].clamp_min(1e-8)
        radial_feature = torch.stack([distance, distance.square()], dim=-1)
        radial_bias = self.radial_bias(radial_feature).permute(0, 3, 1, 2)

        logits = torch.matmul(query.float(), key.float().transpose(-1, -2))
        logits = logits / (self.head_dim ** 0.5)
        logits = logits + radial_bias.float()
        logits = logits.masked_fill(
            ~visible_mask[:, None, None, :], -torch.inf
        )
        attention = torch.softmax(logits, dim=-1)
        context = torch.matmul(attention.to(value.dtype), value)
        context = context.permute(0, 2, 1, 3).reshape(
            batch, query_points, self.width
        )
        raw = self.output_projection(context)
        raw = raw * query_mask[..., None].to(raw.dtype)
        normalized = self.output_norm(raw)
        normalized = normalized * query_mask[..., None].to(normalized.dtype)

        diagnostics: dict[str, torch.Tensor] = {}
        if return_diagnostics:
            diagnostics = {
                "attention": attention * query_mask[:, None, :, None].to(
                    attention.dtype
                ),
                "radial_bias": radial_bias,
                "normalized_distance": distance,
            }
        return raw, normalized, diagnostics


class SharedAlignedLocalOperator(nn.Module):
    """Synchronous, direction-equivariant frontier propagation in AE space."""

    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        source_type_dim: int = 8,
        gamma: float = 0.95,
        reliability_eps: float = 1e-8,
        direction_init_std: float = 1e-3,
        routing_init_std: float = 1e-3,
    ):
        super().__init__()
        if min(latent_dim, num_heads, hidden_dim, source_type_dim) < 1:
            raise ValueError("local operator dimensions must be positive")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0,1]")
        if reliability_eps <= 0:
            raise ValueError("reliability_eps must be positive")
        if direction_init_std <= 0 or routing_init_std <= 0:
            raise ValueError("local routing initialization scales must be positive")
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.source_type_dim = int(source_type_dim)
        self.gamma = float(gamma)
        self.reliability_eps = float(reliability_eps)
        self.direction_init_std = float(direction_init_std)
        self.routing_init_std = float(routing_init_std)

        # Type 0 is original-visible; type 1 is an earlier-round query.
        self.source_type_embedding = nn.Embedding(2, self.source_type_dim)
        descriptor_dim = 2 * self.latent_dim + 3 + self.source_type_dim
        self.path_trunk = nn.Sequential(
            nn.Linear(descriptor_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.lambda_head = nn.Linear(self.hidden_dim, 1)
        self.direction_head = nn.Linear(self.hidden_dim, self.num_heads)
        self.routing_logits = nn.Parameter(
            torch.empty(self.latent_dim, self.num_heads)
        )

        nn.init.normal_(self.source_type_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lambda_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.lambda_head.bias)
        nn.init.normal_(
            self.direction_head.weight, mean=0.0, std=self.direction_init_std
        )
        nn.init.zeros_(self.direction_head.bias)
        nn.init.normal_(self.routing_logits, mean=0.0, std=self.routing_init_std)

    def forward(
        self,
        initial_state: torch.Tensor,
        visible_nodes: torch.Tensor,
        query_nodes: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_mask: torch.Tensor,
        degree: torch.Tensor,
        max_rounds: int,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if initial_state.ndim != 3 or initial_state.shape[-1] != self.latent_dim:
            raise ValueError(
                f"initial_state must have shape [B,N,{self.latent_dim}]"
            )
        batch, nodes, _ = initial_state.shape
        expected_graph = (batch, nodes, 6)
        if neighbor_index.shape != expected_graph or edge_mask.shape != expected_graph:
            raise ValueError("native graph does not align with initial_state")
        if visible_nodes.shape != (batch, nodes) or query_nodes.shape != (batch, nodes):
            raise ValueError("visible/query node masks do not align with state")
        if degree.shape != (batch, nodes):
            raise ValueError("degree does not align with state")
        if bool((visible_nodes & query_nodes).any()):
            raise ValueError("visible and query node masks must not overlap")

        state = initial_state
        active = visible_nodes.clone()
        confidence = visible_nodes.to(initial_state.dtype)
        coverage = torch.zeros_like(confidence)
        activation_round = torch.full(
            (batch, nodes), -1, device=initial_state.device, dtype=torch.long
        )
        activation_round = torch.where(
            visible_nodes, torch.zeros_like(activation_round), activation_round
        )
        source_type = query_nodes.to(torch.long)
        routing = torch.softmax(self.routing_logits, dim=-1)

        safe_neighbor = neighbor_index.clamp(min=0)
        batch_id = _batch_index(batch, initial_state.device)
        direction_id = torch.arange(
            6, device=initial_state.device
        )[None, None, :]

        frontier_masks: list[torch.Tensor] = []
        round_states: list[torch.Tensor] = []
        lambda_rounds: list[torch.Tensor] = []
        reliability_rounds: list[torch.Tensor] = []
        attention_rounds: list[torch.Tensor] = []
        source_rounds: list[torch.Tensor] = []

        for round_index in range(1, max_rounds + 1):
            # Every tensor below reads the previous-round snapshot.  State and
            # active are committed only after all frontier candidates exist.
            direct_source_active = (
                edge_mask
                & active[batch_id, safe_neighbor]
            )
            frontier = (
                query_nodes
                & ~active
                & direct_source_active.any(dim=-1)
            )

            direct_state = state[batch_id, safe_neighbor]
            direct_confidence = confidence[batch_id, safe_neighbor]
            direct_type = source_type[batch_id, safe_neighbor]

            predecessor_index = neighbor_index[
                batch_id, safe_neighbor, direction_id
            ]
            safe_predecessor = predecessor_index.clamp(min=0)
            predecessor_active = (
                direct_source_active
                & (predecessor_index >= 0)
                & active[batch_id, safe_predecessor]
            )
            predecessor_state = state[batch_id, safe_predecessor]
            predecessor_confidence = confidence[batch_id, safe_predecessor]

            residual = torch.where(
                predecessor_active[..., None],
                direct_state - predecessor_state,
                torch.zeros_like(direct_state),
            )
            c_j = torch.where(
                direct_source_active,
                direct_confidence,
                torch.zeros_like(direct_confidence),
            )
            c_p = torch.where(
                predecessor_active,
                predecessor_confidence,
                torch.zeros_like(predecessor_confidence),
            )
            has_predecessor = predecessor_active.to(initial_state.dtype)
            type_embedding = self.source_type_embedding(direct_type)
            descriptor = torch.cat(
                [
                    direct_state,
                    residual,
                    c_j[..., None],
                    c_p[..., None],
                    has_predecessor[..., None],
                    type_embedding,
                ],
                dim=-1,
            )
            path_hidden = self.path_trunk(descriptor)
            path_lambda = torch.sigmoid(self.lambda_head(path_hidden).squeeze(-1))
            path_lambda = torch.where(
                predecessor_active,
                path_lambda,
                torch.zeros_like(path_lambda),
            )
            direction_score = self.direction_head(path_hidden)

            candidate = direct_state + path_lambda[..., None] * residual
            candidate = candidate * direct_source_active[..., None].to(
                candidate.dtype
            )
            detached_lambda = path_lambda.detach()
            path_reliability = c_j * (
                (1.0 - detached_lambda) + detached_lambda * c_p
            )
            path_reliability = path_reliability * direct_source_active.to(
                path_reliability.dtype
            )

            # [B,N,K,H] -> [B,N,H,K], then normalize over directions K.
            attention_logit = direction_score.permute(0, 1, 3, 2)
            attention_logit = attention_logit + torch.log(
                path_reliability[:, :, None, :] + self.reliability_eps
            )
            valid_direction = direct_source_active[:, :, None, :]
            attention_logit = attention_logit.masked_fill(
                ~valid_direction, -torch.inf
            )
            has_source = direct_source_active.any(dim=-1)
            attention_logit = torch.where(
                has_source[:, :, None, None],
                attention_logit,
                torch.zeros_like(attention_logit),
            )
            direction_attention = torch.softmax(attention_logit, dim=-1)
            direction_attention = direction_attention * valid_direction.to(
                direction_attention.dtype
            )
            direction_attention = direction_attention / direction_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(self.reliability_eps)

            # routing[D,H] x alpha[B,N,H,K] -> weights[B,N,D,K]
            channel_direction = torch.einsum(
                "dh,bnhk->bndk", routing, direction_attention
            )
            local_candidate = (
                channel_direction * candidate.permute(0, 1, 3, 2)
            ).sum(dim=-1)

            mean_direction = channel_direction.mean(dim=2)
            inherited_confidence = self.gamma * (
                mean_direction * path_reliability
            ).sum(dim=-1)
            inherited_confidence = inherited_confidence.detach()
            source_coverage = direct_source_active.sum(dim=-1).to(
                initial_state.dtype
            ) / degree.clamp_min(1).to(initial_state.dtype)

            state = torch.where(
                frontier[..., None], local_candidate, state
            )
            confidence = torch.where(
                frontier, inherited_confidence, confidence
            )
            coverage = torch.where(frontier, source_coverage, coverage)
            active = active | frontier
            activation_round = torch.where(
                frontier,
                torch.full_like(activation_round, round_index),
                activation_round,
            )
            frontier_masks.append(frontier)

            if return_diagnostics:
                round_states.append(state)
                source_rounds.append(direct_source_active)
                lambda_rounds.append(
                    path_lambda * frontier[..., None].to(path_lambda.dtype)
                )
                reliability_rounds.append(
                    path_reliability
                    * frontier[..., None].to(path_reliability.dtype)
                )
                attention_rounds.append(
                    direction_attention
                    * frontier[:, :, None, None].to(direction_attention.dtype)
                )

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "state": state,
            "active": active,
            "confidence": confidence,
            "coverage": coverage,
            "gate": (
                query_nodes.to(initial_state.dtype)
                * active.to(initial_state.dtype)
                * coverage
                * confidence
            ),
            "activation_round": activation_round,
            "frontier_masks": torch.stack(frontier_masks, dim=1),
            "routing_probability": routing,
        }
        if return_diagnostics:
            result.update({
                "round_states": round_states,
                "source_masks": torch.stack(source_rounds, dim=1),
                "path_lambda": torch.stack(lambda_rounds, dim=1),
                "path_reliability": torch.stack(reliability_rounds, dim=1),
                "direction_attention": torch.stack(attention_rounds, dim=1),
            })
        return result


class ProgressiveNORMST(nn.Module):
    """Fixed global readout plus confidence-gated progressive local frontier."""

    def __init__(
        self,
        latent_encoder: FrozenLatentEncoder,
        gene_mean: torch.Tensor,
        width: int = 256,
        global_heads: int = 8,
        local_heads: int = 8,
        max_rounds: int = 4,
        global_radial_hidden_dim: int = 32,
        local_hidden_dim: int = 256,
        source_type_dim: int = 8,
        decoder_hidden_dim: int = 512,
        gamma: float = 0.95,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if min(
            width,
            global_heads,
            local_heads,
            max_rounds,
            global_radial_hidden_dim,
            local_hidden_dim,
            source_type_dim,
            decoder_hidden_dim,
        ) < 1:
            raise ValueError("model dimensions must be positive")
        mean = torch.as_tensor(gene_mean, dtype=torch.float32)
        if mean.shape != (latent_encoder.n_genes,):
            raise ValueError(
                f"gene_mean must have shape [{latent_encoder.n_genes}]"
            )
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("gene_mean must be finite")

        self.latent_encoder = latent_encoder
        self.n_genes = int(latent_encoder.n_genes)
        self.latent_dim = int(latent_encoder.latent_dim)
        self.width = int(width)
        self.max_rounds = int(max_rounds)

        self.global_branch = VisibleOnlyRadialCrossAttention(
            latent_dim=self.latent_dim,
            width=self.width,
            num_heads=global_heads,
            radial_hidden_dim=global_radial_hidden_dim,
            norm_eps=norm_eps,
        )
        self.local_operator = SharedAlignedLocalOperator(
            latent_dim=self.latent_dim,
            num_heads=local_heads,
            hidden_dim=local_hidden_dim,
            source_type_dim=source_type_dim,
            gamma=gamma,
        )
        self.local_projection = nn.Linear(
            self.latent_dim, self.width, bias=False
        )
        self.local_norm = NonAffineRMSNorm(self.width, eps=norm_eps)
        self.gene_decoder = nn.Sequential(
            nn.Linear(2 * self.width, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, self.n_genes),
        )
        nn.init.normal_(self.gene_decoder[-1].weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            self.gene_decoder[-1].bias.copy_(mean)

    def _validate_and_scatter_nodes(
        self,
        visible_latent: torch.Tensor,
        visible_index: torch.Tensor,
        query_index: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, visible_points, _ = visible_latent.shape
        nodes = node_mask.shape[1]
        visible_valid = visible_index >= 0
        query_valid = query_index >= 0
        if bool((visible_index >= nodes).any()) or bool((query_index >= nodes).any()):
            raise ValueError("visible/query node index is out of bounds")

        full_state = visible_latent.new_zeros(batch, nodes, self.latent_dim)
        visible_nodes = torch.zeros(
            batch, nodes, device=visible_latent.device, dtype=torch.bool
        )
        query_nodes = torch.zeros_like(visible_nodes)
        for batch_index in range(batch):
            visible_values = visible_index[batch_index, visible_valid[batch_index]]
            query_values = query_index[batch_index, query_valid[batch_index]]
            if visible_values.numel() < 1:
                raise ValueError("every batch item must contain a visible node")
            if visible_values.unique().numel() != visible_values.numel():
                raise ValueError("visible_node_index contains duplicates")
            if query_values.unique().numel() != query_values.numel():
                raise ValueError("query_node_index contains duplicates")
            if not bool(node_mask[batch_index, visible_values].all()) or not bool(
                node_mask[batch_index, query_values].all()
            ):
                raise ValueError("visible/query index points outside the tissue mask")
            if query_values.numel() and bool(
                torch.isin(visible_values, query_values).any()
            ):
                raise ValueError("visible and query node indices must not overlap")
            full_state[batch_index].index_copy_(
                0,
                visible_values,
                visible_latent[batch_index, visible_valid[batch_index]],
            )
            visible_nodes[batch_index, visible_values] = True
            query_nodes[batch_index, query_values] = True

        return (
            full_state,
            visible_nodes,
            query_nodes,
            visible_valid,
            query_valid,
        )

    def forward(
        self,
        visible_expression: torch.Tensor,
        visible_node_index: torch.Tensor,
        query_node_index: torch.Tensor,
        geometry: FullHexGeometry,
        return_auxiliary: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if visible_expression.ndim != 3 or visible_expression.shape[-1] != self.n_genes:
            raise ValueError(
                f"visible_expression must have shape [B,Nv,{self.n_genes}]"
            )
        if not bool(torch.isfinite(visible_expression).all()):
            raise ValueError("visible_expression must be finite")
        batch, visible_points, _ = visible_expression.shape
        visible_index = _expand_compact_index(
            visible_node_index,
            batch,
            visible_points,
            "visible_node_index",
            visible_expression.device,
        )
        query_value = torch.as_tensor(
            query_node_index, device=visible_expression.device, dtype=torch.long
        )
        if query_value.ndim == 1:
            query_value = query_value.unsqueeze(0)
        if query_value.ndim != 2 or query_value.shape[0] not in {1, batch}:
            raise ValueError("query_node_index must have shape [B,Nq] or [Nq]")
        query_index = query_value.expand(batch, query_value.shape[1])
        query_points = query_index.shape[1]

        materialized = materialize_full_hex_geometry(
            geometry,
            batch=batch,
            device=visible_expression.device,
            dtype=visible_expression.dtype,
        )
        visible_latent = self.latent_encoder(visible_expression)
        (
            initial_state,
            visible_nodes,
            query_nodes,
            visible_valid,
            query_valid,
        ) = self._validate_and_scatter_nodes(
            visible_latent,
            visible_index,
            query_index,
            materialized["node_mask"],
        )
        visible_latent = visible_latent * visible_valid[..., None].to(
            visible_latent.dtype
        )

        visible_xy = _gather_nodes(
            materialized["xy"], visible_index, visible_valid
        )
        query_xy = _gather_nodes(
            materialized["xy"], query_index, query_valid
        )
        global_raw, global_normalized, global_diagnostics = self.global_branch(
            visible_latent,
            visible_xy,
            query_xy,
            materialized["native_scale"],
            visible_valid,
            query_valid,
            return_diagnostics=return_diagnostics,
        )

        local_result = self.local_operator(
            initial_state=initial_state,
            visible_nodes=visible_nodes,
            query_nodes=query_nodes,
            neighbor_index=materialized["neighbor_index"],
            edge_mask=materialized["edge_mask"],
            degree=materialized["degree"],
            max_rounds=self.max_rounds,
            return_diagnostics=return_diagnostics,
        )
        local_state = _gather_nodes(
            local_result["state"], query_index, query_valid
        )
        local_projected = self.local_projection(local_state)
        local_normalized = self.local_norm(local_projected)
        query_gate = _gather_nodes(
            local_result["gate"][..., None], query_index, query_valid
        )
        gated_local = query_gate * local_normalized
        fused = torch.cat([global_normalized, gated_local], dim=-1)
        prediction = self.gene_decoder(fused)
        prediction = prediction * query_valid[..., None].to(prediction.dtype)

        if not return_auxiliary and not return_diagnostics:
            return prediction

        query_activation_round = _gather_nodes(
            local_result["activation_round"][..., None],
            query_index,
            query_valid,
        ).squeeze(-1).to(torch.long)
        query_activation_round = torch.where(
            query_valid,
            query_activation_round,
            torch.full_like(query_activation_round, -1),
        )
        auxiliary: dict[str, object] = {
            "visible_latent": visible_latent,
            "global_raw": global_raw,
            "global_normalized": global_normalized,
            "local_state": local_state,
            "local_projected": local_projected,
            "local_normalized": local_normalized,
            "gate": query_gate,
            "gated_local": gated_local,
            "fused_feature": fused,
            "active_query": _gather_nodes(
                local_result["active"][..., None], query_index, query_valid
            ).squeeze(-1).to(torch.bool),
            "activation_round": query_activation_round,
            "confidence": _gather_nodes(
                local_result["confidence"][..., None], query_index, query_valid
            ),
            "coverage": _gather_nodes(
                local_result["coverage"][..., None], query_index, query_valid
            ),
            "frontier_masks": local_result["frontier_masks"],
            "routing_probability": local_result["routing_probability"],
            "native_scale": materialized["native_scale"],
            "query_valid": query_valid,
            "visible_nodes": visible_nodes,
            "query_nodes": query_nodes,
            "full_state": local_result["state"],
            "checkpoint_contract": {
                "visible_memory_frozen": True,
                "global_key_value": "original_visible_only",
                "query_truth_in_forward": False,
                "propagated_state_detached_between_rounds": False,
                "confidence_metadata_detached": True,
                "ae_decoder_in_forward": False,
                "idw_in_forward": False,
            },
        }
        if return_diagnostics:
            auxiliary["global_diagnostics"] = global_diagnostics
            auxiliary["local_diagnostics"] = {
                key: local_result[key]
                for key in (
                    "round_states",
                    "source_masks",
                    "path_lambda",
                    "path_reliability",
                    "direction_attention",
                )
            }
        return prediction, auxiliary


__all__ = [
    "DeterministicExpressionAutoencoder",
    "FrozenLatentEncoder",
    "FullHexGeometry",
    "NonAffineRMSNorm",
    "ProgressiveNORMST",
    "SharedAlignedLocalOperator",
    "VisibleOnlyRadialCrossAttention",
    "materialize_full_hex_geometry",
]
