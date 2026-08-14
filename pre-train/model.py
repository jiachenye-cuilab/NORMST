"""All-gene count-aware denoising autoencoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_genes: int
    composition_dim: int = 255
    hidden_dims: tuple[int, ...] = (512, 512)
    composition_scale: float = 1e4
    dropout: float = 0.1
    dispersion_init: float = 10.0

    @property
    def feature_dim(self) -> int:
        return self.composition_dim + 1

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        payload["feature_dim"] = self.feature_dim
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelConfig":
        accepted = {
            "n_genes",
            "composition_dim",
            "hidden_dims",
            "composition_scale",
            "dropout",
            "dispersion_init",
        }
        values = {key: value for key, value in payload.items() if key in accepted}
        values["hidden_dims"] = tuple(values["hidden_dims"])
        return cls(**values)


def _validate_config(config: ModelConfig) -> None:
    if config.n_genes < 2:
        raise ValueError("n_genes must be at least two")
    if config.composition_dim < 2:
        raise ValueError("composition_dim must be at least two")
    if not config.hidden_dims or any(width < 2 for width in config.hidden_dims):
        raise ValueError("hidden_dims must contain positive widths >= 2")
    if config.composition_scale <= 0:
        raise ValueError("composition_scale must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if config.dispersion_init <= 0:
        raise ValueError("dispersion_init must be positive")


def _hidden_block(input_dim: int, output_dim: int, dropout: float):
    return [
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
    ]


class CountAwareAutoencoder(nn.Module):
    """Encode gene composition separately from the observed library size.

    The composition branch receives all genes after row-wise factorization.
    The final feature is ``[z_composition, log1p(total_UMI)]``.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        _validate_config(config)
        self.config = config

        encoder_layers: list[nn.Module] = []
        previous = config.n_genes
        for width in config.hidden_dims:
            encoder_layers.extend(_hidden_block(previous, width, config.dropout))
            previous = width
        encoder_layers.append(nn.Linear(previous, config.composition_dim))
        encoder_layers.append(nn.LayerNorm(config.composition_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous = config.composition_dim
        for width in reversed(config.hidden_dims):
            decoder_layers.extend(_hidden_block(previous, width, config.dropout))
            previous = width
        decoder_layers.append(nn.Linear(previous, config.n_genes))
        self.decoder = nn.Sequential(*decoder_layers)

        raw_init = math.log(math.expm1(config.dispersion_init))
        self.raw_dispersion = nn.Parameter(torch.full((config.n_genes,), raw_init))
        self.reset_parameters()

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    @property
    def dispersion(self) -> torch.Tensor:
        return F.softplus(self.raw_dispersion) + 1e-4

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initialize_decoder_bias(self, gene_probability: torch.Tensor) -> None:
        """Initialize only from training gene totals."""
        probability = torch.as_tensor(
            gene_probability,
            dtype=self.raw_dispersion.dtype,
            device=self.raw_dispersion.device,
        )
        if probability.shape != (self.config.n_genes,):
            raise ValueError("gene_probability has the wrong shape")
        if bool((probability <= 0).any()) or not bool(torch.isfinite(probability).all()):
            raise ValueError("gene_probability must be finite and strictly positive")
        probability = probability / probability.sum()
        final = self.decoder[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("decoder final layer is not linear")
        with torch.no_grad():
            final.bias.copy_(probability.log())

    def composition_input(self, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if counts.ndim != 2 or counts.shape[1] != self.config.n_genes:
            raise ValueError(
                f"counts must have shape [batch, {self.config.n_genes}]"
            )
        if bool((counts < 0).any()):
            raise ValueError("counts must be non-negative")
        library = counts.sum(dim=1, keepdim=True)
        composition = counts / library.clamp_min(1.0)
        transformed = torch.log1p(composition * self.config.composition_scale)
        return transformed, library

    def encode_composition(self, counts: torch.Tensor) -> torch.Tensor:
        transformed, _ = self.composition_input(counts)
        return self.encoder(transformed)

    def encode(self, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed, library = self.composition_input(counts)
        latent = self.encoder(transformed)
        log_library = torch.log1p(library)
        return latent, log_library

    def feature_vector(self, counts: torch.Tensor) -> torch.Tensor:
        latent, log_library = self.encode(counts)
        return torch.cat([latent, log_library], dim=1)

    def decode_logits(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.config.composition_dim:
            raise ValueError(
                "latent must have shape "
                f"[batch, {self.config.composition_dim}]"
            )
        return self.decoder(latent)

    def decode_mean(
        self,
        latent: torch.Tensor,
        log_library: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.decode_logits(latent)
        probability = torch.softmax(logits, dim=1)
        if log_library.ndim == 1:
            log_library = log_library[:, None]
        if log_library.shape != (latent.shape[0], 1):
            raise ValueError("log_library must have shape [batch, 1]")
        library = torch.expm1(log_library).clamp_min(0.0)
        return library * probability, probability

    def forward(self, counts: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, log_library = self.encode(counts)
        logits = self.decode_logits(latent)
        probability = torch.softmax(logits, dim=1)
        mean = torch.expm1(log_library).clamp_min(0.0) * probability
        return {
            "composition_latent": latent,
            "log_library": log_library,
            "feature": torch.cat([latent, log_library], dim=1),
            "logits": logits,
            "probability": probability,
            "mean": mean,
            "dispersion": self.dispersion,
        }

    def freeze_representation(self) -> "CountAwareAutoencoder":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


def parse_hidden_dims(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("hidden dims cannot be empty")
        return tuple(int(part) for part in parts)
    return tuple(int(part) for part in value)
