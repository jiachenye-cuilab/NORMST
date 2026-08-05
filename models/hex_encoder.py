"""Hexagonal residual encoder for standard 10x Visium offset grids."""

from __future__ import annotations

import torch
import torch.nn as nn


def _shift_to_neighbor(x: torch.Tensor, delta_row: int, delta_col: int):
    """Return x at (row + delta_row, col + delta_col), zero outside bounds."""
    height, width = x.shape[-2:]
    output = torch.zeros_like(x)

    if delta_row >= 0:
        target_rows = slice(0, height - delta_row)
        source_rows = slice(delta_row, height)
    else:
        target_rows = slice(-delta_row, height)
        source_rows = slice(0, height + delta_row)

    if delta_col >= 0:
        target_cols = slice(0, width - delta_col)
        source_cols = slice(delta_col, width)
    else:
        target_cols = slice(-delta_col, width)
        source_cols = slice(0, width + delta_col)

    output[:, :, target_rows, target_cols] = x[
        :, :, source_rows, source_cols
    ]
    return output


class HexConv2d(nn.Module):
    """Direction-aware convolution over a center and its six hex neighbors."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.center = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        # Direction order: left, right, upper-left, upper-right,
        # lower-left, lower-right. Neighbor projections have no bias so a
        # missing neighbor contributes exactly zero.
        self.neighbors = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(6)
        ])

    @staticmethod
    def _parity_mask(row_parity: torch.Tensor, x: torch.Tensor):
        if row_parity.ndim == 2:
            row_parity = row_parity[:, None, :, None]
        elif row_parity.ndim == 3:
            row_parity = row_parity[:, :, :, None]
        if row_parity.ndim != 4 or row_parity.shape[-2] != x.shape[-2]:
            raise ValueError("row_parity must have shape [B,H] or [B,1,H,1]")
        return row_parity.to(device=x.device, dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        tissue_mask: torch.Tensor,
        row_parity: torch.Tensor,
    ) -> torch.Tensor:
        mask = tissue_mask.to(device=x.device, dtype=x.dtype)
        masked = x * mask
        odd = self._parity_mask(row_parity, x)
        even = 1.0 - odd

        left = _shift_to_neighbor(masked, 0, -1)
        right = _shift_to_neighbor(masked, 0, 1)
        left_mask = _shift_to_neighbor(mask, 0, -1)
        right_mask = _shift_to_neighbor(mask, 0, 1)
        upper_left = (
            even * _shift_to_neighbor(masked, -1, -1)
            + odd * _shift_to_neighbor(masked, -1, 0)
        )
        upper_left_mask = (
            even * _shift_to_neighbor(mask, -1, -1)
            + odd * _shift_to_neighbor(mask, -1, 0)
        )
        upper_right = (
            even * _shift_to_neighbor(masked, -1, 0)
            + odd * _shift_to_neighbor(masked, -1, 1)
        )
        upper_right_mask = (
            even * _shift_to_neighbor(mask, -1, 0)
            + odd * _shift_to_neighbor(mask, -1, 1)
        )
        lower_left = (
            even * _shift_to_neighbor(masked, 1, -1)
            + odd * _shift_to_neighbor(masked, 1, 0)
        )
        lower_left_mask = (
            even * _shift_to_neighbor(mask, 1, -1)
            + odd * _shift_to_neighbor(mask, 1, 0)
        )
        lower_right = (
            even * _shift_to_neighbor(masked, 1, 0)
            + odd * _shift_to_neighbor(masked, 1, 1)
        )
        lower_right_mask = (
            even * _shift_to_neighbor(mask, 1, 0)
            + odd * _shift_to_neighbor(mask, 1, 1)
        )

        output = self.center(masked)
        neighbor_values = (
            left, right, upper_left, upper_right, lower_left, lower_right
        )
        neighbor_masks = (
            left_mask, right_mask, upper_left_mask, upper_right_mask,
            lower_left_mask, lower_right_mask,
        )
        for projection, neighbor in zip(
            self.neighbors,
            neighbor_values,
        ):
            output = output + projection(neighbor)
        # The seven projections are initialized independently. Normalize by
        # the square root of the actual contributor count to preserve feature
        # variance at both interior spots and irregular tissue boundaries.
        contributors = mask
        for neighbor_mask in neighbor_masks:
            contributors = contributors + neighbor_mask
        output = output / contributors.clamp_min(1.0).sqrt()
        return output * mask


class HexResidualBlock(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.1):
        super().__init__()
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        self.residual_scale = residual_scale
        self.conv1 = HexConv2d(channels, channels)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = HexConv2d(channels, channels)

    def forward(self, x, tissue_mask, row_parity):
        residual = self.conv1(x, tissue_mask, row_parity)
        residual = self.activation(residual)
        residual = self.conv2(residual, tissue_mask, row_parity)
        return (x + self.residual_scale * residual) * tissue_mask


class HexSpatialEncoder(nn.Module):
    """EDSR-like residual encoder using the true six-neighbor topology."""

    def __init__(
        self,
        in_channels: int,
        channels: int = 64,
        blocks: int = 16,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.head = nn.Conv2d(in_channels, channels, 1)
        self.blocks = nn.ModuleList([
            HexResidualBlock(channels, residual_scale) for _ in range(blocks)
        ])
        self.body = HexConv2d(channels, channels)

    def forward(self, x, tissue_mask, row_parity):
        mask = tissue_mask.to(device=x.device, dtype=x.dtype)
        head = self.head(x) * mask
        residual = head
        for block in self.blocks:
            residual = block(residual, mask, row_parity)
        residual = self.body(residual, mask, row_parity)
        return (head + residual) * mask
