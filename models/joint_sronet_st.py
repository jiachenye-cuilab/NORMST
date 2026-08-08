"""Joint multi-gene NORMST for masked standard Visium spot recovery."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.hex_encoder import HexSpatialEncoder
from models.sronet_st import MaskedGalerkinAttention


class JointSTSRNO(nn.Module):
    """Run the spatial backbone once and predict all genes together.

    The data-specific tissue geometry is attached once as non-persistent model
    buffers. Checkpoints therefore remain model-only and preprocessing stores
    the slice-specific graph separately.
    """

    def __init__(
        self,
        n_genes: int,
        width: int = 256,
        num_heads: int = 16,
        num_operator_layers: int = 2,
        encoder_blocks: int = 16,
        encoder_channels: int = 64,
        hex_residual_scale: float = 0.1,
        physical_query_neighbors: int = 6,
        decoder_hidden: int = 256,
    ):
        super().__init__()
        if n_genes < 1:
            raise ValueError("n_genes must be positive")
        if physical_query_neighbors < 1:
            raise ValueError("physical_query_neighbors must be positive")
        if encoder_channels < 1 or decoder_hidden < 1:
            raise ValueError("encoder and decoder dimensions must be positive")
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")

        self.n_genes = n_genes
        self.encoder_channels = encoder_channels
        self.physical_query_neighbors = physical_query_neighbors
        # All genes are compressed jointly at each spot. The four additional
        # channels are input mask, tissue mask, physical x, and physical y.
        self.encoder = HexSpatialEncoder(
            in_channels=n_genes + 4,
            channels=encoder_channels,
            blocks=encoder_blocks,
            residual_scale=hex_residual_scale,
        )
        self.physical_edge_encoder = nn.Sequential(
            nn.Linear(encoder_channels + 3, encoder_channels),
            nn.GELU(),
            nn.Linear(encoder_channels, encoder_channels),
        )
        self.physical_edge_score = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.physical_edge_score[-1].weight)
        nn.init.zeros_(self.physical_edge_score[-1].bias)

        self.lifting = nn.Conv2d(encoder_channels + 2, width, 1)
        self.scale_conditioner = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 2 * width)
        )
        nn.init.zeros_(self.scale_conditioner[-1].weight)
        nn.init.zeros_(self.scale_conditioner[-1].bias)
        self.operators = nn.ModuleList([
            MaskedGalerkinAttention(width, num_heads)
            for _ in range(num_operator_layers)
        ])
        self.projection = nn.Sequential(
            nn.Conv2d(width, decoder_hidden, 1),
            nn.GELU(),
            nn.Conv2d(decoder_hidden, n_genes, 1),
        )
        # A grouped 1x1 layer is a separate calibrated skip for every gene.
        self.skip_calibration = nn.Conv2d(
            n_genes, n_genes, 1, groups=n_genes
        )
        nn.init.ones_(self.skip_calibration.weight)
        nn.init.zeros_(self.skip_calibration.bias)

        self.register_buffer("_tissue_mask", torch.empty(0), persistent=False)
        self.register_buffer("_physical_coord", torch.empty(0), persistent=False)
        self.register_buffer("_row_parity", torch.empty(0), persistent=False)
        self.register_buffer(
            "_query_positions", torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_neighbor_indices", torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("_neighbor_relative", torch.empty(0), persistent=False)
        self.register_buffer(
            "_neighbor_mask", torch.empty(0, dtype=torch.bool),
            persistent=False,
        )

    def set_spatial_context(
        self,
        tissue_mask: torch.Tensor,
        physical_coord: torch.Tensor,
        row_parity: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_relative: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ):
        """Attach one slice's grid and flattened physical-query graph."""
        if tissue_mask.ndim != 3 or tissue_mask.shape[0] != 1:
            raise ValueError("tissue_mask must have shape [1,H,W]")
        _, height, width = tissue_mask.shape
        if physical_coord.shape != (2, height, width):
            raise ValueError("physical_coord must have shape [2,H,W]")
        if row_parity.shape != (height,):
            raise ValueError("row_parity must have shape [H]")
        if neighbor_indices.ndim != 3:
            raise ValueError("neighbor_indices must have shape [H,W,K]")
        if neighbor_indices.shape[:2] != (height, width):
            raise ValueError("physical query grid does not match tissue grid")
        candidates = neighbor_indices.shape[-1]
        if candidates < self.physical_query_neighbors:
            raise ValueError("physical query candidate count is too small")
        if neighbor_relative.shape != (3, height, width, candidates):
            raise ValueError("neighbor_relative must have shape [3,H,W,K]")
        if neighbor_mask.shape != (1, height, width, candidates):
            raise ValueError("neighbor_mask must have shape [1,H,W,K]")

        points = height * width
        if neighbor_indices.numel() and (
            neighbor_indices.min().item() < 0
            or neighbor_indices.max().item() >= points
        ):
            raise ValueError("physical query neighbor index is out of bounds")
        tissue_flat = tissue_mask.reshape(points) > 0.5
        graph_flat = neighbor_mask[0].reshape(points, candidates) > 0.5
        graph_queries = graph_flat.any(dim=1)
        if not torch.equal(graph_queries, tissue_flat):
            raise ValueError(
                "physical query graph must cover exactly the tissue spots"
            )
        query_positions = torch.nonzero(
            tissue_flat, as_tuple=False
        ).flatten()
        relative_flat = neighbor_relative.permute(1, 2, 3, 0).reshape(
            points, candidates, 3
        )

        device = self.skip_calibration.weight.device
        self._tissue_mask = tissue_mask.to(device=device, dtype=torch.float32)
        self._physical_coord = physical_coord.to(
            device=device, dtype=torch.float32
        )
        self._row_parity = row_parity.to(device=device, dtype=torch.float32)
        self._query_positions = query_positions.to(
            device=device, dtype=torch.long
        )
        self._neighbor_indices = neighbor_indices.reshape(
            points, candidates
        )[query_positions].to(device=device, dtype=torch.long)
        self._neighbor_relative = relative_flat[query_positions].to(
            device=device, dtype=torch.float32
        ).contiguous()
        self._neighbor_mask = graph_flat[query_positions].to(
            device=device, dtype=torch.bool
        )

    def _physical_query_features(
        self,
        features: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = features.shape
        points = height * width
        queries = len(self._query_positions)

        flat_features = features.flatten(2).transpose(1, 2)
        flat_observed = input_mask.flatten(2).squeeze(1)
        indices = self._neighbor_indices.unsqueeze(0).expand(batch, -1, -1)
        relative = self._neighbor_relative.to(features.dtype).unsqueeze(0).expand(
            batch, -1, -1, -1
        )
        static_mask = self._neighbor_mask.unsqueeze(0).expand(batch, -1, -1)
        candidate_observed = flat_observed[:, self._neighbor_indices] > 0.5
        available = static_mask & candidate_observed

        masked_distance = relative[..., 2].masked_fill(
            ~available, float("inf")
        )
        selected = torch.topk(
            masked_distance,
            k=self.physical_query_neighbors,
            dim=2,
            largest=False,
        ).indices
        selected_indices = torch.gather(indices, 2, selected)
        batch_indices = torch.arange(
            batch, device=features.device
        )[:, None, None]
        selected_features = flat_features[batch_indices, selected_indices]
        selected_relative = torch.gather(
            relative,
            2,
            selected[..., None].expand(-1, -1, -1, 3),
        )
        selected_mask = torch.gather(available, 2, selected)

        edge_input = torch.cat(
            [selected_features, selected_relative], dim=-1
        )
        messages = self.physical_edge_encoder(edge_input)
        logits = self.physical_edge_score(selected_relative).squeeze(-1)
        logits = logits.masked_fill(~selected_mask, -1e4)
        weights = torch.softmax(logits, dim=2) * selected_mask.to(logits.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        aggregated = (messages * weights[..., None]).sum(dim=2)

        # Match the native-grid cell feature used by the existing physical
        # query: [2/H,2/W] scaled by [H,W] equals [2,2].
        cell = features.new_full((batch, queries, 2), 2.0)
        valid_query = torch.cat([aggregated, cell], dim=-1)
        flat_query = features.new_zeros(
            batch, points, channels + 2
        )
        scatter_index = self._query_positions.view(1, queries, 1).expand(
            batch, queries, channels + 2
        )
        flat_query = flat_query.scatter(1, scatter_index, valid_query)
        return flat_query.transpose(1, 2).reshape(
            batch, channels + 2, height, width
        )

    def forward(
        self,
        expression: torch.Tensor,
        input_mask: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._tissue_mask.numel() == 0:
            raise ValueError("spatial context has not been attached")
        if expression.ndim != 4 or expression.shape[1] != self.n_genes:
            raise ValueError("expression must have shape [B,G,H,W]")
        batch, _, height, width = expression.shape
        if input_mask.shape != (batch, 1, height, width):
            raise ValueError("input_mask must have shape [B,1,H,W]")
        if self._tissue_mask.shape[-2:] != (height, width):
            raise ValueError("input grid does not match attached spatial context")

        tissue_mask = self._tissue_mask.unsqueeze(0).expand(batch, -1, -1, -1)
        physical_coord = self._physical_coord.unsqueeze(0).expand(
            batch, -1, -1, -1
        )
        row_parity = self._row_parity.unsqueeze(0).expand(batch, -1)
        encoder_input = torch.cat([
            expression,
            input_mask.to(expression.dtype),
            tissue_mask.to(expression.dtype),
            physical_coord.to(expression.dtype),
        ], dim=1)
        features = self.encoder(encoder_input, tissue_mask, row_parity)
        query_features = self._physical_query_features(features, input_mask)
        latent = self.lifting(query_features) * tissue_mask

        if scale is None:
            scale = expression.new_ones(batch)
        scale_feature = torch.log2(scale.clamp_min(1.0)).reshape(-1, 1)
        gamma, beta = self.scale_conditioner(scale_feature).chunk(2, dim=1)
        latent = (
            latent * (1 + gamma[:, :, None, None])
            + beta[:, :, None, None]
        ) * tissue_mask
        for operator in self.operators:
            latent = operator(latent, tissue_mask) * tissue_mask

        residual = self.projection(latent)
        skip = self.skip_calibration(expression)
        return (residual + skip) * tissue_mask
