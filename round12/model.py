"""Opt-in Round12 model with a dedicated local-to-gene residual path."""

from __future__ import annotations

import torch
from torch import nn

from models.pro_normst import FullHexGeometry, ProNORMST as FrozenV9ProNORMST


class LocalGeneResidualHead(nn.Module):
    """Depth/reliability-aware local gene residual with a zero-start output."""

    def __init__(
        self,
        *,
        local_dim: int = 256,
        round_embedding_dim: int = 8,
        hidden_dim: int = 256,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        if min(local_dim, round_embedding_dim, hidden_dim, output_dim) < 1:
            raise ValueError("local gene residual dimensions must be positive")
        self.local_dim = int(local_dim)
        self.round_embedding_dim = int(round_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.round_embedding = nn.Embedding(4, self.round_embedding_dim)
        self.input_projection = nn.Linear(
            self.local_dim + self.round_embedding_dim + 2,
            self.hidden_dim,
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.normal_(self.round_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        gated_local: torch.Tensor,
        activation_round: torch.Tensor,
        coverage: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if gated_local.ndim != 2 or gated_local.shape[-1] != self.local_dim:
            raise ValueError("gated_local must have shape [N,local_dim]")
        rows = gated_local.shape[0]
        if activation_round.shape != (rows,):
            raise ValueError("activation_round must align with local rows")
        if coverage.shape != (rows, 1) or confidence.shape != (rows, 1):
            raise ValueError("coverage/confidence must align with local rows")
        if activation_round.device.type == "cpu" and bool(
            ((activation_round < 1) | (activation_round > 4)).any()
        ):
            raise ValueError("active local rows require activation_round in [1,4]")
        round_feature = self.round_embedding(activation_round - 1)
        features = torch.cat(
            [
                gated_local,
                round_feature,
                coverage.detach(),
                confidence.detach(),
            ],
            dim=-1,
        )
        return self.output_projection(
            self.activation(self.input_projection(features))
        )


class Round12ProNORMST(FrozenV9ProNORMST):
    """Frozen v9 model plus one opt-in local gene residual head."""

    LOCAL_GENE_ROUND_EMBEDDING_DIM = 8
    LOCAL_GENE_HIDDEN_DIM = 256

    def __init__(
        self,
        gene_mean: torch.Tensor,
        *,
        variant: str = "full",
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
        super().__init__(
            gene_mean,
            variant=variant,
            width=width,
            global_heads=global_heads,
            local_heads=local_heads,
            max_rounds=max_rounds,
            global_radial_hidden_dim=global_radial_hidden_dim,
            local_hidden_dim=local_hidden_dim,
            source_type_dim=source_type_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            gamma=gamma,
            norm_eps=norm_eps,
        )
        # Construct only after every v9 module so all inherited seeded
        # parameters remain exactly unchanged.
        self.local_gene_residual_head = LocalGeneResidualHead(
            local_dim=self.width,
            round_embedding_dim=self.LOCAL_GENE_ROUND_EMBEDDING_DIM,
            hidden_dim=self.LOCAL_GENE_HIDDEN_DIM,
            output_dim=self.N_GENES,
        )
        if self.variant == "global-only":
            self.local_gene_residual_head.requires_grad_(False)

    def contract_manifest(self) -> dict[str, object]:
        manifest = super().contract_manifest()
        manifest.update(
            {
                "schema": "pro-normst-direct-512-v7",
                "local_gene_residual_head": (
                    "gated-local-round-reliability-mlp-v1"
                ),
                "local_gene_residual_input": (
                    "gated_local+activation_round_embedding+detached_coverage+"
                    "detached_confidence"
                ),
                "local_gene_residual_round_embedding_dim": (
                    self.local_gene_residual_head.round_embedding_dim
                ),
                "local_gene_residual_hidden_dim": (
                    self.local_gene_residual_head.hidden_dim
                ),
                "local_gene_residual_output_dim": (
                    self.local_gene_residual_head.output_dim
                ),
                "local_gene_residual_output_init": "zeros",
                "local_gene_residual_position": "after_shared_gene_decoder",
                "local_gene_residual_active_rows_only": True,
                "local_gene_residual_activation_grouped": True,
                "local_gene_residual_global_input": False,
                "local_gene_residual_metadata_detached": True,
                "local_gene_residual_global_only": "frozen-and-skipped",
            }
        )
        return manifest

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
        base_prediction, auxiliary = super().forward(
            visible_expression_z,
            visible_node_index,
            query_node_index,
            geometry,
            round_limit=round_limit,
            return_auxiliary=True,
            return_diagnostics=return_diagnostics,
        )
        query_valid = auxiliary["query_valid"]
        if not isinstance(query_valid, torch.Tensor):
            raise RuntimeError("v9 auxiliary output omitted query_valid")
        residual = base_prediction.new_zeros(base_prediction.shape)
        if self.variant != "global-only":
            gated_local = auxiliary["gated_local"]
            activation_round = auxiliary["activation_round"]
            coverage = auxiliary["coverage"]
            confidence = auxiliary["confidence"]
            active_query = auxiliary["active_query"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    gated_local,
                    activation_round,
                    coverage,
                    confidence,
                    active_query,
                )
            ):
                raise RuntimeError("v9 auxiliary output is incomplete")
            flat_active = active_query.reshape(-1).to(torch.bool)
            flat_round = activation_round.reshape(-1).to(torch.long)
            flat_local = gated_local.reshape(-1, self.width)
            flat_coverage = coverage.reshape(-1, 1)
            flat_confidence = confidence.reshape(-1, 1)
            flat_residual = residual.reshape(-1, self.N_GENES)
            rounds = self.max_rounds if round_limit is None else int(round_limit)
            for activation in range(1, rounds + 1):
                offsets = torch.nonzero(
                    flat_active & (flat_round == activation),
                    as_tuple=False,
                ).squeeze(-1)
                if offsets.numel() < 1:
                    continue
                residual_round = self.local_gene_residual_head(
                    flat_local.index_select(0, offsets),
                    flat_round.index_select(0, offsets),
                    flat_coverage.index_select(0, offsets),
                    flat_confidence.index_select(0, offsets),
                )
                flat_residual = flat_residual.index_copy(
                    0,
                    offsets,
                    residual_round,
                )
            residual = flat_residual.reshape_as(base_prediction)
        prediction = base_prediction + residual
        prediction = prediction * query_valid[..., None].to(prediction.dtype)
        auxiliary.update(
            {
                "shared_decoder_prediction": base_prediction,
                "local_gene_residual": residual,
            }
        )
        if not return_auxiliary and not return_diagnostics:
            return prediction
        return prediction, auxiliary


__all__ = ["LocalGeneResidualHead", "Round12ProNORMST"]
