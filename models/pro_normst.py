"""Direct-expression Progressive NORMST.

This module is the real-data model defined by ``Pro_contract.md``.  It keeps
the historical frozen-AE prototype untouched and reuses only its validated,
AE-agnostic geometry and operator primitives.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .progressive_normst import (
    FullHexGeometry,
    NonAffineRMSNorm,
    SharedAlignedLocalOperator,
    VisibleOnlyRadialCrossAttention,
    materialize_full_hex_geometry,
)


ProNORMSTVariant = Literal["full", "one-shot", "local-only", "global-only"]


class ResidualGeneProgramEncoder(nn.Module):
    """Zero-start residual MLP used only by the global visible memory."""

    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 256,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(state_dim, hidden_dim) < 1:
            raise ValueError("gene-program encoder dimensions must be positive")
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = NonAffineRMSNorm(self.state_dim, eps=norm_eps)
        self.input_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(self.hidden_dim, self.state_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, visible_state: torch.Tensor) -> torch.Tensor:
        residual = self.output_projection(
            self.activation(self.input_projection(self.input_norm(visible_state)))
        )
        return visible_state + residual


class ResidualLocalStateEnhancer(nn.Module):
    """Zero-start residual MLP applied after local aggregation."""

    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 256,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(state_dim, hidden_dim) < 1:
            raise ValueError("local-state enhancer dimensions must be positive")
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = NonAffineRMSNorm(self.state_dim, eps=norm_eps)
        self.input_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(self.hidden_dim, self.state_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        local_state: torch.Tensor,
        global_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del global_context
        residual = self.output_projection(
            self.activation(self.input_projection(self.input_norm(local_state)))
        )
        return local_state + residual


class GlobalConditionedResidualLocalStateEnhancer(nn.Module):
    """Zero-start local innovation MLP conditioned on detached global context."""

    def __init__(
        self,
        state_dim: int = 512,
        global_dim: int = 256,
        hidden_dim: int = 256,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(state_dim, global_dim, hidden_dim) < 1:
            raise ValueError("conditioned local enhancer dimensions must be positive")
        self.state_dim = int(state_dim)
        self.global_dim = int(global_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = NonAffineRMSNorm(self.state_dim, eps=norm_eps)
        self.input_projection = nn.Linear(
            self.state_dim + self.global_dim,
            self.hidden_dim,
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(self.hidden_dim, self.state_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        local_state: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        if local_state.shape[:-1] != global_context.shape[:-1]:
            raise ValueError("local state and global context must align")
        if local_state.shape[-1] != self.state_dim:
            raise ValueError("local state has an incompatible channel dimension")
        if global_context.shape[-1] != self.global_dim:
            raise ValueError("global context has an incompatible channel dimension")
        conditioned_input = torch.cat(
            [self.input_norm(local_state), global_context.detach()],
            dim=-1,
        )
        residual = self.output_projection(
            self.activation(self.input_projection(conditioned_input))
        )
        return local_state + residual


def _expand_index(
    index: torch.Tensor,
    batch: int,
    expected_points: int | None,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(index, device=device, dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] not in {1, batch}:
        raise ValueError(f"{name} must have shape [B,N] or [N]")
    if expected_points is not None and value.shape[1] != expected_points:
        raise ValueError(f"{name} does not align with its compact values")
    return value.expand(batch, value.shape[1])


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


class ProNORMST(nn.Module):
    """Shared-512 input/output model with no AE or query-expression input."""

    N_GENES = 512
    GLOBAL_INPUT_HIDDEN_DIM = 256
    LOCAL_STATE_HIDDEN_DIM = 256
    LOCAL_DIRECTION_INIT_STD = 1e-3
    LOCAL_ROUTING_INIT_STD = 1e-3
    VALID_VARIANTS = ("full", "one-shot", "local-only", "global-only")

    def __init__(
        self,
        gene_mean: torch.Tensor,
        *,
        variant: ProNORMSTVariant = "full",
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
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unsupported ProNORMST variant: {variant}")
        if max_rounds != 4 and variant != "one-shot":
            raise ValueError("contracted progressive variants require max_rounds=4")
        if variant == "one-shot":
            max_rounds = 1
        if width != 256 or decoder_hidden_dim != 512:
            raise ValueError("contracted widths are width=256 and decoder_hidden_dim=512")

        mean = torch.as_tensor(gene_mean, dtype=torch.float32)
        if mean.shape != (self.N_GENES,) or not bool(torch.isfinite(mean).all()):
            raise ValueError("gene_mean must be a finite vector with shape [512]")

        self.n_genes = self.N_GENES
        self.state_dim = self.N_GENES
        self.width = int(width)
        self.max_rounds = int(max_rounds)
        self.variant: ProNORMSTVariant = variant

        self.global_branch = VisibleOnlyRadialCrossAttention(
            latent_dim=self.state_dim,
            width=self.width,
            num_heads=global_heads,
            radial_hidden_dim=global_radial_hidden_dim,
            norm_eps=norm_eps,
        )
        self.local_operator = SharedAlignedLocalOperator(
            latent_dim=self.state_dim,
            num_heads=local_heads,
            hidden_dim=local_hidden_dim,
            source_type_dim=source_type_dim,
            gamma=gamma,
            direction_init_std=self.LOCAL_DIRECTION_INIT_STD,
            routing_init_std=self.LOCAL_ROUTING_INIT_STD,
        )
        self.local_projection = nn.Linear(self.state_dim, self.width, bias=False)
        self.local_norm = NonAffineRMSNorm(self.width, eps=norm_eps)
        self.gene_decoder = nn.Sequential(
            nn.Linear(2 * self.width, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, self.N_GENES),
        )
        nn.init.normal_(self.gene_decoder[-1].weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            self.gene_decoder[-1].bias.copy_(mean)
        # Register new modules after all Round3 modules so the original seeded
        # initialization remains unchanged.  Both residual output projections
        # start at zero, making the initial Round8 forward exactly match the
        # zero-start Round7/Round5 forward while adding only new conditioning.
        self.global_input_encoder = ResidualGeneProgramEncoder(
            state_dim=self.state_dim,
            hidden_dim=self.GLOBAL_INPUT_HIDDEN_DIM,
            norm_eps=norm_eps,
        )
        self.local_state_enhancer = GlobalConditionedResidualLocalStateEnhancer(
            state_dim=self.state_dim,
            global_dim=self.width,
            hidden_dim=self.LOCAL_STATE_HIDDEN_DIM,
            norm_eps=norm_eps,
        )
        if self.variant == "local-only":
            self.global_branch.requires_grad_(False)
            self.global_input_encoder.requires_grad_(False)
        elif self.variant == "global-only":
            self.local_operator.requires_grad_(False)
            self.local_state_enhancer.requires_grad_(False)
            self.local_projection.requires_grad_(False)

    def contract_manifest(self) -> dict[str, object]:
        """Return the immutable model-side contract for checkpoints."""
        conditioned_enhancer = isinstance(
            self.local_state_enhancer,
            GlobalConditionedResidualLocalStateEnhancer,
        )
        manifest: dict[str, object] = {
            "schema": (
                "pro-normst-direct-512-v6"
                if conditioned_enhancer
                else "pro-normst-direct-512-v5"
            ),
            "variant": self.variant,
            "n_genes": self.N_GENES,
            "state_dim": self.state_dim,
            "width": self.width,
            "max_rounds": self.max_rounds,
            "global_heads": self.global_branch.num_heads,
            "global_radial_hidden_dim": self.global_branch.radial_bias[0].out_features,
            "local_heads": self.local_operator.num_heads,
            "local_hidden_dim": self.local_operator.hidden_dim,
            "source_type_dim": self.local_operator.source_type_dim,
            "decoder_hidden_dim": self.gene_decoder[0].out_features,
            "gamma": self.local_operator.gamma,
            "norm_eps": self.global_branch.output_norm.eps,
            "path_reliability_eps": self.local_operator.reliability_eps,
            "local_direction_init_std": self.local_operator.direction_init_std,
            "local_routing_init_std": self.local_operator.routing_init_std,
            "direct_expression_adapter": (
                "global_conditioned_residual_mlp_local_residual_mlp_global"
                if conditioned_enhancer
                else "residual_mlp_local_residual_mlp_global"
            ),
            "global_input_encoder": "residual-prenorm-gene-mlp-v1",
            "global_input_encoder_hidden_dim": self.global_input_encoder.hidden_dim,
            "global_input_encoder_output_init": "zeros",
            "local_state_enhancer": (
                "global-conditioned-residual-prenorm-local-mlp-v1"
                if conditioned_enhancer
                else "residual-prenorm-local-mlp-v1"
            ),
            "local_state_enhancer_hidden_dim": self.local_state_enhancer.hidden_dim,
            "local_state_enhancer_input_dim": (
                self.local_state_enhancer.state_dim
                + (
                    self.local_state_enhancer.global_dim
                    if conditioned_enhancer
                    else 0
                )
            ),
            "local_state_enhancer_output_init": "zeros",
            "local_state_enhancer_position": (
                "after_local_aggregation_before_local_projection"
            ),
            "local_state_enhancer_activation_grouped": True,
            "ae_encoder": False,
            "ae_decoder": False,
            "query_truth_in_forward": False,
            "global_key_value": "original_visible_only",
            "synchronous_frontier": True,
            "confidence_metadata_detached": True,
            "gate_parameters": False,
            "local_fusion_gate": "active_x_coverage_x_confidence_fixed",
        }
        if conditioned_enhancer:
            manifest.update(
                {
                    "local_state_enhancer_global_condition_dim": (
                        self.local_state_enhancer.global_dim
                    ),
                    "local_state_enhancer_global_condition_source": (
                        "fixed_original_visible_global_context"
                    ),
                    "local_state_enhancer_global_condition_stop_gradient": True,
                }
            )
        return manifest

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def _scatter_visible(
        self,
        visible_state: torch.Tensor,
        visible_index: torch.Tensor,
        query_index: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, _ = visible_state.shape
        nodes = node_mask.shape[1]
        visible_valid = visible_index >= 0
        query_valid = query_index >= 0
        if bool((visible_index >= nodes).any()) or bool((query_index >= nodes).any()):
            raise ValueError("visible/query node index is out of bounds")

        full_state = visible_state.new_zeros(batch, nodes, self.state_dim)
        visible_nodes = torch.zeros(batch, nodes, device=visible_state.device, dtype=torch.bool)
        query_nodes = torch.zeros_like(visible_nodes)
        for batch_index in range(batch):
            visible_values = visible_index[batch_index, visible_valid[batch_index]]
            query_values = query_index[batch_index, query_valid[batch_index]]
            if visible_values.numel() < 1 or query_values.numel() < 1:
                raise ValueError("every batch item must contain visible and query nodes")
            if visible_values.unique().numel() != visible_values.numel():
                raise ValueError("visible_node_index contains duplicates")
            if query_values.unique().numel() != query_values.numel():
                raise ValueError("query_node_index contains duplicates")
            if not bool(node_mask[batch_index, visible_values].all()) or not bool(
                node_mask[batch_index, query_values].all()
            ):
                raise ValueError("visible/query index points outside the tissue graph")
            if bool(torch.isin(visible_values, query_values).any()):
                raise ValueError("visible and query node indices must not overlap")
            full_state[batch_index].index_copy_(
                0,
                visible_values,
                visible_state[batch_index, visible_valid[batch_index]],
            )
            visible_nodes[batch_index, visible_values] = True
            query_nodes[batch_index, query_values] = True
            if not bool(
                torch.equal(
                    visible_nodes[batch_index] | query_nodes[batch_index],
                    node_mask[batch_index],
                )
            ):
                raise ValueError(
                    "visible and query node indices must partition the full tissue graph"
                )
        return full_state, visible_nodes, query_nodes, visible_valid, query_valid

    def forward(
        self,
        visible_expression_z: torch.Tensor,
        visible_node_index: torch.Tensor,
        query_node_index: torch.Tensor,
        geometry: FullHexGeometry,
        *,
        round_limit: int | None = None,
        return_auxiliary: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if visible_expression_z.ndim != 3 or visible_expression_z.shape[-1] != self.N_GENES:
            raise ValueError("visible_expression_z must have shape [B,N_visible,512]")
        if not bool(torch.isfinite(visible_expression_z).all()):
            raise ValueError("visible_expression_z must be finite")
        batch, visible_points, _ = visible_expression_z.shape
        visible_index = _expand_index(
            visible_node_index,
            batch,
            visible_points,
            "visible_node_index",
            visible_expression_z.device,
        )
        query_index = _expand_index(
            query_node_index,
            batch,
            None,
            "query_node_index",
            visible_expression_z.device,
        )
        query_points = query_index.shape[1]
        if query_points < 1:
            raise ValueError("query_node_index must not be empty")

        rounds = self.max_rounds if round_limit is None else int(round_limit)
        allowed_rounds = {1} if self.variant == "one-shot" else {1, 2, 4}
        if rounds not in allowed_rounds or rounds > self.max_rounds:
            raise ValueError(f"round_limit must be one of {sorted(allowed_rounds)}")

        materialized = materialize_full_hex_geometry(
            geometry,
            batch=batch,
            device=visible_expression_z.device,
            dtype=visible_expression_z.dtype,
        )
        (
            initial_state,
            visible_nodes,
            query_nodes,
            visible_valid,
            query_valid,
        ) = self._scatter_visible(
            visible_expression_z,
            visible_index,
            query_index,
            materialized["node_mask"],
        )
        visible_state = visible_expression_z * visible_valid[..., None].to(
            visible_expression_z.dtype
        )
        visible_xy = _gather_nodes(materialized["xy"], visible_index, visible_valid)
        query_xy = _gather_nodes(materialized["xy"], query_index, query_valid)

        need_global = self.variant != "local-only"
        if need_global:
            global_input_state = self.global_input_encoder(visible_state)
            global_input_residual = global_input_state - visible_state
            global_raw, global_normalized, global_diagnostics = self.global_branch(
                global_input_state,
                visible_xy,
                query_xy,
                materialized["native_scale"],
                visible_valid,
                query_valid,
                return_diagnostics=return_diagnostics,
            )
        else:
            global_input_state = visible_state
            global_input_residual = torch.zeros_like(visible_state)
            global_raw = visible_state.new_zeros(batch, query_points, self.width)
            global_normalized = torch.zeros_like(global_raw)
            global_diagnostics = {}

        need_local = self.variant != "global-only"
        local_result: dict[str, object]
        if need_local:
            local_result = self.local_operator(
                initial_state=initial_state,
                visible_nodes=visible_nodes,
                query_nodes=query_nodes,
                neighbor_index=materialized["neighbor_index"],
                edge_mask=materialized["edge_mask"],
                degree=materialized["degree"],
                max_rounds=rounds,
                return_diagnostics=return_diagnostics,
            )
            local_state = _gather_nodes(local_result["state"], query_index, query_valid)
            active_query = _gather_nodes(
                local_result["active"][..., None], query_index, query_valid
            ).squeeze(-1).to(torch.bool)
            active_query = active_query & query_valid
            activation_round = _gather_nodes(
                local_result["activation_round"][..., None], query_index, query_valid
            ).squeeze(-1).to(torch.long)
            activation_round = torch.where(
                query_valid, activation_round, torch.full_like(activation_round, -1)
            )
            flat_active = active_query.reshape(-1)
            flat_activation_round = activation_round.reshape(-1)
            flat_state = local_state.reshape(-1, self.state_dim)
            flat_global_context = global_normalized.reshape(-1, self.width)
            flat_enhanced = None
            flat_enhancer_residual = None
            flat_projected = None
            flat_normalized = None
            # Keep each activation depth's GEMM row shape independent of the
            # requested early-exit limit.  Under FP16, projecting all active
            # rows at once lets cuBLAS choose shape-dependent rounding, which
            # can change depth-1/2 predictions between round limits.
            for activation in range(1, rounds + 1):
                round_offset = torch.nonzero(
                    flat_active & (flat_activation_round == activation),
                    as_tuple=False,
                ).squeeze(-1)
                if round_offset.numel() < 1:
                    continue
                state_round = flat_state.index_select(0, round_offset)
                global_context_round = flat_global_context.index_select(
                    0, round_offset
                )
                enhanced_round = self.local_state_enhancer(
                    state_round,
                    global_context_round,
                )
                enhancer_residual_round = enhanced_round - state_round
                projected_round = self.local_projection(
                    enhanced_round
                )
                normalized_round = self.local_norm(projected_round)
                if flat_projected is None:
                    flat_enhanced = enhanced_round.new_zeros(
                        flat_state.shape[0], self.state_dim
                    )
                    flat_enhancer_residual = torch.zeros_like(flat_enhanced)
                    flat_projected = projected_round.new_zeros(
                        flat_state.shape[0], self.width
                    )
                    flat_normalized = torch.zeros_like(flat_projected)
                flat_enhanced = flat_enhanced.index_copy(
                    0, round_offset, enhanced_round
                )
                flat_enhancer_residual = flat_enhancer_residual.index_copy(
                    0, round_offset, enhancer_residual_round
                )
                flat_projected = flat_projected.index_copy(
                    0, round_offset, projected_round
                )
                flat_normalized = flat_normalized.index_copy(
                    0, round_offset, normalized_round
                )
            if (
                flat_enhanced is None
                or flat_enhancer_residual is None
                or flat_projected is None
                or flat_normalized is None
            ):
                flat_enhanced = local_state.new_zeros(
                    flat_state.shape[0], self.state_dim
                )
                flat_enhancer_residual = torch.zeros_like(flat_enhanced)
                flat_projected = local_state.new_zeros(
                    flat_state.shape[0], self.width
                )
                flat_normalized = torch.zeros_like(flat_projected)
            local_state_enhanced = flat_enhanced.reshape(
                batch, query_points, self.state_dim
            )
            local_state_residual = flat_enhancer_residual.reshape(
                batch, query_points, self.state_dim
            )
            local_projected = flat_projected.reshape(
                batch, query_points, self.width
            )
            local_normalized = flat_normalized.reshape(
                batch, query_points, self.width
            )
            query_gate = _gather_nodes(
                local_result["gate"][..., None], query_index, query_valid
            ).detach()
            query_confidence = _gather_nodes(
                local_result["confidence"][..., None], query_index, query_valid
            ).detach()
            query_coverage = _gather_nodes(
                local_result["coverage"][..., None], query_index, query_valid
            ).detach()
            gated_local = query_gate * local_normalized
        else:
            local_state = visible_state.new_zeros(batch, query_points, self.state_dim)
            local_state_enhanced = torch.zeros_like(local_state)
            local_state_residual = torch.zeros_like(local_state)
            local_projected = visible_state.new_zeros(batch, query_points, self.width)
            local_normalized = torch.zeros_like(local_projected)
            query_gate = visible_state.new_zeros(batch, query_points, 1)
            query_confidence = visible_state.new_zeros(batch, query_points, 1)
            query_coverage = torch.zeros_like(query_confidence)
            gated_local = torch.zeros_like(local_projected)
            local_result = {
                "active": query_nodes.new_zeros(query_nodes.shape),
                "activation_round": torch.full_like(query_nodes, -1, dtype=torch.long),
                "confidence": initial_state.new_zeros(query_nodes.shape),
                "coverage": initial_state.new_zeros(query_nodes.shape),
                "frontier_masks": query_nodes.new_zeros(
                    (batch, rounds, query_nodes.shape[1]), dtype=torch.bool
                ),
                "routing_probability": initial_state.new_zeros((self.state_dim, 8)),
                "state": initial_state,
            }
            active_query = query_valid.new_zeros(query_valid.shape)
            activation_round = torch.full_like(query_valid, -1, dtype=torch.long)

        fused = torch.cat([global_normalized, gated_local], dim=-1)
        prediction = self.gene_decoder(fused)
        prediction = prediction * query_valid[..., None].to(prediction.dtype)
        if not return_auxiliary and not return_diagnostics:
            return prediction

        auxiliary: dict[str, object] = {
            "visible_state": visible_state,
            "global_raw": global_raw,
            "global_normalized": global_normalized,
            "local_state": local_state,
            "local_state_enhanced": local_state_enhanced,
            "local_state_residual": local_state_residual,
            "local_projected": local_projected,
            "local_normalized": local_normalized,
            "gate": query_gate,
            "gated_local": gated_local,
            "fused_feature": fused,
            "active_query": active_query,
            "activation_round": activation_round,
            "confidence": query_confidence,
            "coverage": query_coverage,
            "global_input_base": visible_state,
            "global_input_state": global_input_state,
            "global_input_residual": global_input_residual,
            "frontier_masks": local_result["frontier_masks"],
            "routing_probability": local_result["routing_probability"],
            "native_scale": materialized["native_scale"],
            "query_valid": query_valid,
            "visible_nodes": visible_nodes,
            "query_nodes": query_nodes,
            "full_state": local_result["state"],
            "checkpoint_contract": self.contract_manifest(),
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
                if key in local_result
            }
        return prediction, auxiliary


__all__ = [
    "FullHexGeometry",
    "GlobalConditionedResidualLocalStateEnhancer",
    "ProNORMST",
    "ProNORMSTVariant",
    "ResidualLocalStateEnhancer",
    "ResidualGeneProgramEncoder",
]
