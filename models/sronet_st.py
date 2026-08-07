"""ST-specific SRNO with masks, scale conditioning, and calibrated skip."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.edsr import make_edsr_baseline
from models.hex_encoder import HexSpatialEncoder
from utils import make_coord


class MaskedGalerkinAttention(nn.Module):
    def __init__(self, width: int, num_heads: int):
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_width = width // num_heads
        self.width = width
        self.qkv = nn.Conv2d(width, 3 * width, 1)
        self.key_norm = nn.LayerNorm(self.head_width)
        self.value_norm = nn.LayerNorm(self.head_width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch, channels, height, width = x.shape
        qkv = self.qkv(x).permute(0, 2, 3, 1).reshape(
            batch, height * width, self.num_heads, 3 * self.head_width
        )
        q, key, value = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)
        key = self.key_norm(key)
        value = self.value_norm(value)
        if mask is None:
            denominator = float(height * width)
        else:
            flat_mask = mask.reshape(batch, 1, height * width, 1).to(x.dtype)
            key = key * flat_mask
            value = value * flat_mask
            denominator = flat_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        kernel = torch.matmul(key.transpose(-2, -1), value) / denominator
        integral = torch.matmul(q, kernel)
        integral = integral.permute(0, 2, 1, 3).reshape(
            batch, height, width, channels
        ).permute(0, 3, 1, 2)
        attended = x + integral
        return x + self.ffn(attended)


class STSRNO(nn.Module):
    def __init__(
        self,
        width: int = 256,
        num_heads: int = 16,
        num_operator_layers: int = 2,
        encoder_blocks: int = 16,
        context_dim: int = 0,
        n_genes: int = 0,
        gene_embedding_dim: int = 0,
        include_tissue_mask: bool = False,
        spatial_encoder: str = "rectangular",
        hex_residual_scale: float = 0.1,
        query_mode: str = "rectangular",
        physical_query_neighbors: int = 6,
    ):
        super().__init__()
        self.width = width
        # Channel 0 is expression; channel 1 explicitly distinguishes valid
        # tissue observations from biological zeros and padding.
        self.context_dim = context_dim
        self.gene_embedding_dim = gene_embedding_dim
        self.include_tissue_mask = include_tissue_mask
        if spatial_encoder not in {
            "rectangular", "rectangular_coord", "hex_coord"
        }:
            raise ValueError(
                "spatial_encoder must be rectangular, rectangular_coord, or hex_coord"
            )
        self.spatial_encoder = spatial_encoder
        self.use_physical_coordinates = spatial_encoder != "rectangular"
        if query_mode not in {"rectangular", "physical"}:
            raise ValueError("query_mode must be rectangular or physical")
        if physical_query_neighbors < 1:
            raise ValueError("physical_query_neighbors must be positive")
        self.query_mode = query_mode
        self.physical_query_neighbors = physical_query_neighbors
        # The query graph is shared by every gene sample from one tissue
        # slice. Keep one device copy instead of duplicating it in each batch.
        # It is saved separately with preprocessing rather than in checkpoints.
        self.register_buffer(
            "_physical_query_indices", torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_physical_query_relative", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_physical_query_mask", torch.empty(0), persistent=False
        )
        if gene_embedding_dim < 0:
            raise ValueError("gene_embedding_dim must be non-negative")
        if gene_embedding_dim > 0:
            if n_genes <= 0:
                raise ValueError("n_genes must be positive when using gene embeddings")
            self.gene_embedding = nn.Embedding(n_genes, gene_embedding_dim)
        else:
            self.gene_embedding = None
        encoder_channels = (
            2 + context_dim + gene_embedding_dim
            + int(include_tissue_mask)
            + 2 * int(self.use_physical_coordinates)
        )
        if spatial_encoder == "hex_coord":
            self.encoder = HexSpatialEncoder(
                in_channels=encoder_channels,
                channels=64,
                blocks=encoder_blocks,
                residual_scale=hex_residual_scale,
            )
        else:
            self.encoder = make_edsr_baseline(
                n_resblocks=encoder_blocks,
                n_feats=64,
                n_colors=encoder_channels,
            )
        if query_mode == "physical":
            # A shared edge encoder makes physical-neighbor aggregation
            # permutation invariant. The three geometric inputs are dx, dy,
            # and Euclidean distance in units of one median spot spacing.
            self.physical_edge_encoder = nn.Sequential(
                nn.Linear(64 + 3, 64),
                nn.GELU(),
                nn.Linear(64, 64),
            )
            self.physical_edge_score = nn.Sequential(
                nn.Linear(3, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            # Start from uniform averaging over available physical neighbors.
            nn.init.zeros_(self.physical_edge_score[-1].weight)
            nn.init.zeros_(self.physical_edge_score[-1].bias)
            self.lifting = nn.Conv2d(64 + 2, width, 1)
        else:
            self.physical_edge_encoder = None
            self.physical_edge_score = None
            self.lifting = nn.Conv2d((64 + 2) * 4 + 2, width, 1)
        self.scale_conditioner = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 2 * width)
        )
        # Start as identity FiLM so scale conditioning is introduced smoothly.
        nn.init.zeros_(self.scale_conditioner[-1].weight)
        nn.init.zeros_(self.scale_conditioner[-1].bias)
        self.operators = nn.ModuleList([
            MaskedGalerkinAttention(width, num_heads)
            for _ in range(num_operator_layers)
        ])
        self.projection = nn.Sequential(
            nn.Conv2d(width, 256, 1), nn.GELU(), nn.Conv2d(256, 1, 1)
        )
        # Learn whether/how the interpolated LR expression should enter the
        # HR prediction. Identity initialization recovers the original SRNO.
        self.skip_calibration = nn.Conv2d(1, 1, 1)
        nn.init.ones_(self.skip_calibration.weight)
        nn.init.zeros_(self.skip_calibration.bias)

    def set_physical_query_graph(
        self,
        neighbor_indices: torch.Tensor,
        neighbor_relative: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ):
        """Attach one unbatched, data-specific physical query graph."""
        if neighbor_indices.ndim != 3:
            raise ValueError("neighbor_indices must have shape [H,W,K]")
        height, width, candidates = neighbor_indices.shape
        if candidates < self.physical_query_neighbors:
            raise ValueError("physical query candidate count is too small")
        points = height * width
        if neighbor_indices.numel() and (
            neighbor_indices.min().item() < 0
            or neighbor_indices.max().item() >= points
        ):
            raise ValueError("physical query neighbor index is out of bounds")
        if neighbor_relative.shape != (3, height, width, candidates):
            raise ValueError("neighbor_relative must have shape [3,H,W,K]")
        if neighbor_mask.shape != (1, height, width, candidates):
            raise ValueError("neighbor_mask must have shape [1,H,W,K]")
        device = self.skip_calibration.weight.device
        self._physical_query_indices = neighbor_indices.to(
            device=device, dtype=torch.long
        )
        self._physical_query_relative = neighbor_relative.to(
            device=device, dtype=torch.float32
        )
        self._physical_query_mask = neighbor_mask.to(
            device=device, dtype=torch.float32
        )

    def _query_features(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        pos_lr = make_coord(feat.shape[-2:], flatten=False).to(
            device=feat.device, dtype=feat.dtype
        ).permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:]
        )
        radius_row = 1 / feat.shape[-2]
        radius_col = 1 / feat.shape[-1]
        relative_coordinates, sampled_features, areas = [], [], []
        for direction_row in (-1, 1):
            for direction_col in (-1, 1):
                shifted = coord.clone()
                shifted[..., 0] += direction_row * radius_row + 1e-6
                shifted[..., 1] += direction_col * radius_col + 1e-6
                shifted.clamp_(-1 + 1e-6, 1 - 1e-6)
                sampled = F.grid_sample(
                    feat, shifted.flip(-1), mode="nearest", align_corners=False
                )
                old_coord = F.grid_sample(
                    pos_lr, shifted.flip(-1), mode="nearest", align_corners=False
                )
                relative = coord.permute(0, 3, 1, 2) - old_coord
                relative[:, 0] *= feat.shape[-2]
                relative[:, 1] *= feat.shape[-1]
                relative_coordinates.append(relative)
                sampled_features.append(sampled)
                areas.append(torch.abs(relative[:, 0] * relative[:, 1]) + 1e-9)
        total_area = torch.stack(areas).sum(dim=0)
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]
        sampled_features = [
            value * (area / total_area).unsqueeze(1)
            for value, area in zip(sampled_features, areas)
        ]
        relative_cell = cell.clone()
        relative_cell[:, 0] *= feat.shape[-2]
        relative_cell[:, 1] *= feat.shape[-1]
        relative_cell = relative_cell[:, :, None, None].expand(
            -1, -1, coord.shape[1], coord.shape[2]
        )
        return torch.cat(
            [*relative_coordinates, *sampled_features, relative_cell], dim=1
        )

    def _physical_query_features(
        self,
        feat: torch.Tensor,
        input_mask: torch.Tensor,
        cell: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_relative: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate the nearest currently observed spots in physical space."""
        batch, channels, height, width = feat.shape
        if input_mask.shape != (batch, 1, height, width):
            raise ValueError("input_mask must have shape [B,1,H,W]")
        if neighbor_indices.ndim != 4:
            raise ValueError("physical_query_indices must have shape [B,H,W,K]")
        if neighbor_indices.shape[1:3] != (height, width):
            raise ValueError("physical query grid does not match encoder features")
        candidates = neighbor_indices.shape[-1]
        if candidates < self.physical_query_neighbors:
            raise ValueError("physical query candidate count is too small")
        if neighbor_relative.shape != (
            batch, 3, height, width, candidates
        ):
            raise ValueError(
                "physical_query_relative must have shape [B,3,H,W,K]"
            )
        if neighbor_mask.shape != (
            batch, 1, height, width, candidates
        ):
            raise ValueError(
                "physical_query_mask must have shape [B,1,H,W,K]"
            )

        points = height * width
        indices = neighbor_indices.reshape(batch, points, candidates).long()
        relative = neighbor_relative.permute(0, 2, 3, 4, 1).reshape(
            batch, points, candidates, 3
        ).to(feat.dtype)
        static_mask = neighbor_mask[:, 0].reshape(
            batch, points, candidates
        ).bool()

        flat_feat = feat.flatten(2).transpose(1, 2)
        flat_observed = input_mask.flatten(2).squeeze(1)
        batch_indices = torch.arange(
            batch, device=feat.device
        )[:, None, None]
        candidate_observed = flat_observed[batch_indices, indices] > 0.5
        available = static_mask & candidate_observed

        # Candidate lists are longer than K because training masks change for
        # every sample. Select the nearest K that remain observed at runtime.
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
        selected_feat = flat_feat[batch_indices, selected_indices]
        selected_relative = torch.gather(
            relative,
            2,
            selected[..., None].expand(-1, -1, -1, 3),
        )
        selected_mask = torch.gather(available, 2, selected)

        edge_input = torch.cat([selected_feat, selected_relative], dim=-1)
        messages = self.physical_edge_encoder(edge_input)
        logits = self.physical_edge_score(selected_relative).squeeze(-1)
        logits = logits.masked_fill(~selected_mask, -1e4)
        weights = torch.softmax(logits, dim=2) * selected_mask.to(logits.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        aggregated = (messages * weights[..., None]).sum(dim=2)
        aggregated = aggregated.transpose(1, 2).reshape(
            batch, channels, height, width
        )

        relative_cell = cell.clone()
        relative_cell[:, 0] *= height
        relative_cell[:, 1] *= width
        relative_cell = relative_cell[:, :, None, None].expand(
            -1, -1, height, width
        )
        return torch.cat([aggregated, relative_cell], dim=1)

    def forward(
        self,
        expression: torch.Tensor,
        input_mask: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor,
        scale: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        gene_context: Optional[torch.Tensor] = None,
        baseline_scale: Optional[torch.Tensor] = None,
        gene_index: Optional[torch.Tensor] = None,
        tissue_mask: Optional[torch.Tensor] = None,
        physical_coord: Optional[torch.Tensor] = None,
        row_parity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gene_context is None:
            gene_context = expression.new_zeros(
                expression.shape[0], self.context_dim, *expression.shape[-2:]
            )
        if gene_context.shape[1] != self.context_dim:
            raise ValueError(
                f"Expected {self.context_dim} context channels, got {gene_context.shape[1]}"
            )
        encoder_inputs = [expression, input_mask, gene_context]
        if self.include_tissue_mask:
            if tissue_mask is None:
                raise ValueError("tissue_mask is required by this model")
            encoder_inputs.append(tissue_mask)
        if self.gene_embedding is not None:
            if gene_index is None:
                raise ValueError("gene_index is required when using gene embeddings")
            embedding = self.gene_embedding(gene_index).to(expression.dtype)
            embedding = embedding[:, :, None, None].expand(
                -1, -1, *expression.shape[-2:]
            )
            encoder_inputs.append(embedding)
        if self.use_physical_coordinates:
            if physical_coord is None:
                raise ValueError("physical_coord is required by this spatial encoder")
            if physical_coord.shape[1] != 2:
                raise ValueError("physical_coord must contain x and y channels")
            encoder_inputs.append(physical_coord.to(expression.dtype))
        encoder_input = torch.cat(encoder_inputs, dim=1)
        if self.spatial_encoder == "hex_coord":
            if row_parity is None:
                raise ValueError("row_parity is required by the hex encoder")
            encoder_mask = tissue_mask if tissue_mask is not None else input_mask
            feat = self.encoder(encoder_input, encoder_mask, row_parity)
        else:
            feat = self.encoder(encoder_input)
        if self.query_mode == "physical":
            if self._physical_query_indices.numel() == 0:
                raise ValueError("physical query graph has not been attached")
            batch = feat.shape[0]
            query_features = self._physical_query_features(
                feat,
                input_mask,
                cell,
                self._physical_query_indices.unsqueeze(0).expand(
                    batch, -1, -1, -1
                ),
                self._physical_query_relative.unsqueeze(0).expand(
                    batch, -1, -1, -1, -1
                ),
                self._physical_query_mask.unsqueeze(0).expand(
                    batch, -1, -1, -1, -1
                ),
            )
        else:
            query_features = self._query_features(feat, coord, cell)
        latent = self.lifting(query_features)
        scale_feature = torch.log2(scale.clamp_min(1.0)).reshape(-1, 1)
        gamma, beta = self.scale_conditioner(scale_feature).chunk(2, dim=1)
        latent = latent * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for operator in self.operators:
            latent = operator(latent, target_mask)
        residual = self.projection(latent)
        baseline = F.grid_sample(
            expression,
            coord.flip(-1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        if baseline_scale is not None:
            baseline = baseline * baseline_scale[:, None, None, None]
        return residual + self.skip_calibration(baseline)
