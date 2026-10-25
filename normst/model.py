"""The final CNN expression field with a 32-channel image bypass."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import basis_shape, knot_points, stencil
from .memory import CompactHistologyMixin


class ResidualEncoderBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value):
        return value + .1 * self.layers(value)


class PointwiseImageProjection(nn.Conv2d):
    """A 1x1 image projection with bounded convolution workspace."""
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1)

    def forward(self, features):
        compact = getattr(self, "compact_inputs", None)
        if compact is not None:
            cls, local, indices, valid = compact
            dimension = cls.shape[0]
            weight = self.weight[:, :, 0, 0]
            projected = torch.addmm(self.bias[:, None], weight[:, dimension:], local[0].flatten(1))
            global_projected = torch.mm(weight[:, :dimension], cls)
            projected = projected + global_projected[:, indices] * valid.flatten()
            return projected.reshape(1, self.out_channels, *local.shape[-2:])
        projected = torch.addmm(self.bias[:, None], self.weight[:, :, 0, 0], features[0].flatten(1))
        return projected.reshape(1, self.out_channels, *features.shape[-2:])


def _seeded_layers(genes, channels):
    """Fixed initialization layout, including reserved random-number slots.

    Keeping this draw order makes seeded weights reproducible. Temporary layers
    below reserve draws only; they are not parameters of the returned network.
    """
    encoder = nn.Sequential(nn.Conv2d(genes + 3, 64, 1), nn.GELU(), ResidualEncoderBlock(64))
    for _ in range(2):
        nn.Linear(64, 192)
        for _ in range(3):
            nn.Linear(64, 64)
    nn.Linear(66, 64)
    nn.Linear(64, genes)
    nn.Conv2d(3, 8, 5, padding=2)
    nn.Conv2d(8, 16, 3, padding=1)
    ResidualEncoderBlock(16)
    projection = nn.Conv2d(16, 64, 1, bias=False)
    decoder = nn.Sequential(nn.Linear(82, 64), nn.GELU(), nn.Linear(64, genes))
    nn.Linear(64, 2 * genes)
    image_encoder = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.GELU(), ResidualEncoderBlock(16))
    decoder[2] = nn.Linear(64, genes)
    nn.init.zeros_(decoder[2].weight)
    nn.init.zeros_(decoder[2].bias)
    with torch.random.fork_rng(devices=[]):
        bypass = nn.Sequential(nn.Linear(channels, 16), nn.GELU())
        expanded = nn.Linear(98, 64)
    with torch.no_grad():
        expanded.weight[:, :82].copy_(decoder[0].weight)
        expanded.weight[:, 82:].zero_()
        expanded.bias.copy_(decoder[0].bias)
    decoder[0] = expanded
    adapter = nn.Sequential(nn.Linear(98, 16), nn.GELU(), nn.Linear(16, genes))
    nn.init.zeros_(adapter[2].weight)
    nn.init.zeros_(adapter[2].bias)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        expanded = nn.Linear(channels, 32)
        expanded.weight[:16].copy_(bypass[0].weight)
        expanded.bias[:16].copy_(bypass[0].bias)
        bypass[0] = expanded
        for head in (decoder, adapter):
            expanded = nn.Linear(114, head[0].out_features)
            expanded.weight.zero_()
            expanded.weight[:, :98].copy_(head[0].weight)
            expanded.bias.copy_(head[0].bias)
            head[0] = expanded
        projected = PointwiseImageProjection(channels, 16)
        projected.weight, projected.bias = image_encoder[0].weight, image_encoder[0].bias
        image_encoder[0] = projected
    return encoder, image_encoder, projection, decoder, bypass, adapter


class _Field(nn.Module):
    def __init__(self, context, bounds, features, feature_valid):
        super().__init__()
        self.basis_pitch_um = 8.
        self.basis_shape = basis_shape(bounds, self.basis_pitch_um)
        self.bypass_mode = "local8"
        if (features.ndim != 4 or features.shape[0] != 1 or
                tuple(features.shape[-2:]) != self.basis_shape or
                feature_valid.shape != features[:, :1].shape or features.dtype != torch.float32):
            raise ValueError("features must be float32 [1,C,H,W] on the 8 um field grid")
        if feature_valid.dtype != torch.bool or not feature_valid.any():
            raise ValueError("feature_valid must be a nonempty boolean [1,1,H,W] mask")
        if (torch.as_tensor(bounds)[2:] - torch.as_tensor(bounds)[:2]).remainder(32).any():
            raise ValueError("field extent must be a multiple of 32 um")
        for name in ("conditioning", "reference_rate", "prior_density"):
            self.register_buffer(name, context[name])
        self.register_buffer("bounds", torch.as_tensor(bounds, dtype=torch.float32))
        self.register_buffer("histology_features", features)
        self.register_buffer("feature_valid", feature_valid)
        encoder, image_encoder, projection, decoder, bypass, adapter = _seeded_layers(
            len(self.reference_rate), features.shape[1])
        self.encoder = encoder
        self.decoder = decoder
        self.he_encoder = image_encoder
        self.he_projection = projection
        self.image_bypass = bypass
        self.adapter = adapter
        self._he_features = None

    def set_context(self, context):
        for name in ("conditioning", "reference_rate", "prior_density"):
            setattr(self, name, context[name])
        self._he_features = None

    def encode(self, image_features=None):
        latent = self.encoder(self.conditioning)
        image = self.he_encoder(self.histology_features) if image_features is None else image_features
        self._he_features = image * self.feature_valid
        context = F.avg_pool2d(self._he_features, 4)
        if context.shape[-2:] != latent.shape[-2:]:
            raise ValueError("8 um image and 32 um expression grids must align")
        return latent + self.he_projection(context)

    def bypass_features(self, points):
        indices, weights = stencil(points, self.bounds, 8., self.basis_shape)
        source_valid = self.feature_valid[0, 0].flatten()
        valid = (weights * source_valid[indices]).sum(1)
        raw = self.histology_features[0].flatten(1)
        values = raw.new_zeros(len(points), raw.shape[0])
        for corner in range(4):
            rows = indices[:, corner]
            values = values + raw[:, rows].T * (weights[:, corner] * source_valid[rows])[:, None]
        return self.image_bypass(values) * valid[:, None]

    def coefficients(self, latent, points, chunk=1024):
        result = []
        for begin in range(0, len(points), chunk):
            current = points[begin:begin + chunk]
            relative = 2 * (current - self.bounds[:2]) / (self.bounds[2:] - self.bounds[:2]) - 1
            grid = relative.flip(-1)[None, None]

            def sample(array):
                return F.grid_sample(array, grid, mode="bilinear", padding_mode="border",
                                     align_corners=False)[0, :, 0].T

            features = torch.cat((sample(latent), relative, sample(self._he_features),
                                  self.bypass_features(current)), 1)
            prior = sample(self.prior_density).clamp_min(1e-8)
            base = prior * F.softplus(self.decoder(features) + math.log(math.expm1(1.)))
            result.append(base * torch.exp(.1 * torch.tanh(self.adapter(features))))
        return torch.cat(result)

    def density(self, latent, points):
        indices, weights = stencil(points, self.bounds, self.basis_pitch_um, self.basis_shape)
        unique, inverse = torch.unique(indices.reshape(-1), sorted=True, return_inverse=True)
        values = self.coefficients(latent, knot_points(unique, self.bounds, 8., self.basis_shape))
        inverse = inverse.reshape(-1, 4)
        return sum(values[inverse[:, corner]] * weights[:, corner, None] for corner in range(4))

    def regional_counts(self, latent, points, weights, owners, count, chunk=1024):
        if weights.layout == torch.sparse_coo:
            if weights.shape != (count, len(points)) or owners is not None:
                raise ValueError("invalid region integral matrix")
            return torch.sparse.mm(weights, self.coefficients(latent, points, chunk))
        total = latent.new_zeros(count, len(self.reference_rate))
        for begin in range(0, len(points), chunk):
            values = self.density(latent, points[begin:begin + chunk]) * weights[begin:begin + chunk, None]
            total = total.index_add(0, owners[begin:begin + chunk], values)
        return total


class NORMST(CompactHistologyMixin, _Field):
    """Trainable expression network; frozen image features are supplied separately.

    Contexts use raw observed counts. All physical coordinates are (y, x) in um.
    The final architecture has 504,048 parameters for 2,000 genes and 2,563 image
    channels. Compact storage is lossless and does not change the architecture.
    """
    def __init__(self, context, bounds, features, feature_valid, *, compact=True, tile_offset_yx=(0, 0)):
        super().__init__(context, bounds, features, feature_valid)
        self.tile_offset_yx = tuple(tile_offset_yx)
        options = (dict(dimension=1280, tile_cells=14, crop_offset_yx=list(tile_offset_yx))
                   if compact and features.shape[1] == 2563 else None)
        self._prepare_compact_histology(options)
