"""Frozen representation interface for a future spatial downstream model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from checkpoint_io import load_checkpoint
from losses import composition_cross_entropy, negative_binomial_nll
from model import CountAwareAutoencoder, ModelConfig


class FrozenCountRepresentation(nn.Module):
    """Standardize targets and decode downstream 256-D predictions.

    Model parameters remain frozen, but gradients can flow from count-space
    losses through the decoder into a downstream model's predicted features.
    """

    def __init__(self, model: CountAwareAutoencoder, statistics: dict):
        super().__init__()
        self.model = model.freeze_representation()
        composition_dim = model.config.composition_dim
        mean = torch.as_tensor(
            np.asarray(statistics["composition_mean"], dtype=np.float32)
        )
        scale = torch.as_tensor(
            np.asarray(statistics["composition_scale"], dtype=np.float32)
        )
        library_mean = torch.as_tensor(
            np.asarray(statistics["log_library_mean"], dtype=np.float32)
        )
        library_scale = torch.as_tensor(
            np.asarray(statistics["log_library_scale"], dtype=np.float32)
        )
        if mean.shape != (composition_dim,) or scale.shape != (composition_dim,):
            raise ValueError("composition statistics have the wrong shape")
        if library_mean.shape != (1,) or library_scale.shape != (1,):
            raise ValueError("library statistics have the wrong shape")
        if bool((scale <= 0).any()) or bool((library_scale <= 0).any()):
            raise ValueError("standardization scales must be positive")
        self.register_buffer("composition_mean", mean)
        self.register_buffer("composition_scale", scale)
        self.register_buffer("log_library_mean", library_mean)
        self.register_buffer("log_library_scale", library_scale)

    @property
    def feature_dim(self) -> int:
        return self.model.feature_dim

    def train(self, mode: bool = True):
        """Allow wrapper use in a train graph while keeping dropout frozen."""
        super().train(mode)
        self.model.eval()
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "FrozenCountRepresentation":
        checkpoint = load_checkpoint(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        if "latent_statistics" not in checkpoint:
            raise ValueError("completed best.pt with latent_statistics is required")
        model = CountAwareAutoencoder(
            ModelConfig.from_dict(checkpoint["model_config"])
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        return cls(model, checkpoint["latent_statistics"])

    def standardize(
        self,
        latent: torch.Tensor,
        log_library: torch.Tensor,
    ) -> torch.Tensor:
        if log_library.ndim == 1:
            log_library = log_library[:, None]
        composition = (
            latent - self.composition_mean[None, :]
        ) / self.composition_scale[None, :]
        library = (
            log_library - self.log_library_mean[None, :]
        ) / self.log_library_scale[None, :]
        return torch.cat([composition, library], dim=1)

    def unstandardize(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 2 or feature.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature must have shape [batch, {self.feature_dim}]"
            )
        latent = (
            feature[:, :-1] * self.composition_scale[None, :]
            + self.composition_mean[None, :]
        )
        log_library = (
            feature[:, -1:] * self.log_library_scale[None, :]
            + self.log_library_mean[None, :]
        )
        return latent, log_library

    @torch.no_grad()
    def encode_target(self, counts: torch.Tensor) -> torch.Tensor:
        latent, log_library = self.model.encode(counts)
        return self.standardize(latent, log_library)

    def decode_feature(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, log_library = self.unstandardize(feature)
        logits = self.model.decode_logits(latent)
        probability = torch.softmax(logits.float(), dim=1)
        library = torch.expm1(log_library.float()).clamp_min(0.0)
        return {
            "composition_latent": latent,
            "log_library": log_library,
            "logits": logits,
            "probability": probability,
            "mean": library * probability,
            "dispersion": self.model.dispersion,
        }

    def latent_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        composition_weight: float = 1.0,
        library_weight: float = 1.0,
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target features must align")
        if prediction.ndim != 2 or prediction.shape[1] != self.feature_dim:
            raise ValueError("feature tensors have the wrong shape")
        if composition_weight < 0 or library_weight < 0 or beta <= 0:
            raise ValueError("invalid latent loss weights")
        composition = F.smooth_l1_loss(
            prediction[:, :-1], target[:, :-1], reduction="mean", beta=beta
        )
        library = F.smooth_l1_loss(
            prediction[:, -1:], target[:, -1:], reduction="mean", beta=beta
        )
        total = composition_weight * composition + library_weight * library
        return {
            "latent_loss": total,
            "composition_latent_loss": composition,
            "library_latent_loss": library,
        }

    def count_auxiliary_loss(
        self,
        predicted_feature: torch.Tensor,
        target_counts: torch.Tensor,
        nb_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        if nb_weight < 0:
            raise ValueError("nb_weight must be non-negative")
        decoded = self.decode_feature(predicted_feature)
        cross_entropy, valid = composition_cross_entropy(
            decoded["logits"], target_counts
        )
        nb_element = negative_binomial_nll(
            target_counts,
            decoded["mean"],
            decoded["dispersion"],
        )
        nb_per_spot = nb_element.mean(dim=1)
        nb = (
            nb_per_spot[valid].mean()
            if bool(valid.any())
            else predicted_feature.sum() * 0.0
        )
        return {
            "count_auxiliary_loss": cross_entropy + nb_weight * nb,
            "composition_cross_entropy": cross_entropy,
            "negative_binomial_nll": nb,
        }
