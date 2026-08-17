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

        if self.variant == "local-only":
            self.global_branch.requires_grad_(False)
        elif self.variant == "global-only":
            self.local_operator.requires_grad_(False)
            self.local_projection.requires_grad_(False)

    def contract_manifest(self) -> dict[str, object]:
        """Return the immutable model-side contract for checkpoints."""
        return {
            "schema": "pro-normst-direct-512-v1",
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
            "direct_expression_adapter": "identity_no_parameters",
            "ae_encoder": False,
            "ae_decoder": False,
            "query_truth_in_forward": False,
            "global_key_value": "original_visible_only",
            "synchronous_frontier": True,
            "confidence_metadata_detached": True,
            "gate_parameters": False,
        }

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
            global_raw, global_normalized, global_diagnostics = self.global_branch(
                visible_state,
                visible_xy,
                query_xy,
                materialized["native_scale"],
                visible_valid,
                query_valid,
                return_diagnostics=return_diagnostics,
            )
        else:
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
            flat_active = active_query.reshape(-1)
            active_offset = torch.nonzero(flat_active, as_tuple=False).squeeze(-1)
            flat_state = local_state.reshape(-1, self.state_dim)
            if active_offset.numel() > 0:
                projected_active = self.local_projection(
                    flat_state.index_select(0, active_offset)
                )
                normalized_active = self.local_norm(projected_active)
                flat_projected = projected_active.new_zeros(
                    flat_state.shape[0], self.width
                )
                flat_normalized = torch.zeros_like(flat_projected)
                flat_projected = flat_projected.index_copy(
                    0, active_offset, projected_active
                )
                flat_normalized = flat_normalized.index_copy(
                    0, active_offset, normalized_active
                )
            else:
                flat_projected = local_state.new_zeros(
                    flat_state.shape[0], self.width
                )
                flat_normalized = torch.zeros_like(flat_projected)
            local_projected = flat_projected.reshape(
                batch, query_points, self.width
            )
            local_normalized = flat_normalized.reshape(
                batch, query_points, self.width
            )
            query_gate = _gather_nodes(
                local_result["gate"][..., None], query_index, query_valid
            )
            gated_local = query_gate * local_normalized
        else:
            local_state = visible_state.new_zeros(batch, query_points, self.state_dim)
            local_projected = visible_state.new_zeros(batch, query_points, self.width)
            local_normalized = torch.zeros_like(local_projected)
            query_gate = visible_state.new_zeros(batch, query_points, 1)
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

        fused = torch.cat([global_normalized, gated_local], dim=-1)
        prediction = self.gene_decoder(fused)
        prediction = prediction * query_valid[..., None].to(prediction.dtype)
        if not return_auxiliary and not return_diagnostics:
            return prediction

        activation_round = _gather_nodes(
            local_result["activation_round"][..., None], query_index, query_valid
        ).squeeze(-1).to(torch.long)
        activation_round = torch.where(
            query_valid, activation_round, torch.full_like(activation_round, -1)
        )
        auxiliary: dict[str, object] = {
            "visible_state": visible_state,
            "global_raw": global_raw,
            "global_normalized": global_normalized,
            "local_state": local_state,
            "local_projected": local_projected,
            "local_normalized": local_normalized,
            "gate": query_gate,
            "gated_local": gated_local,
            "fused_feature": fused,
            "active_query": active_query,
            "activation_round": activation_round,
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


__all__ = ["FullHexGeometry", "ProNORMST", "ProNORMSTVariant"]
