"""Composition-only interface to a completed frozen pretraining checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


_PRETRAIN_MODEL_PATH = Path(__file__).resolve().parents[1] / "pre-train" / "model.py"
_PRETRAIN_MODULE_NAME = "_normst_frozen_pretrain_model"
_PRETRAIN_CHECKPOINT_IO_PATH = (
    Path(__file__).resolve().parents[1] / "pre-train" / "checkpoint_io.py"
)
_PRETRAIN_CHECKPOINT_IO_MODULE_NAME = "_normst_frozen_pretrain_checkpoint_io"


def _pretrain_classes():
    module = sys.modules.get(_PRETRAIN_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            _PRETRAIN_MODULE_NAME, _PRETRAIN_MODEL_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pretraining model from {_PRETRAIN_MODEL_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PRETRAIN_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module.CountAwareAutoencoder, module.ModelConfig


def _load_pretrain_checkpoint(path: Path, map_location):
    module = sys.modules.get(_PRETRAIN_CHECKPOINT_IO_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            _PRETRAIN_CHECKPOINT_IO_MODULE_NAME, _PRETRAIN_CHECKPOINT_IO_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"cannot load checkpoint I/O from {_PRETRAIN_CHECKPOINT_IO_PATH}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PRETRAIN_CHECKPOINT_IO_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module.load_checkpoint(
        path,
        map_location=map_location,
        weights_only=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenCompositionAE(nn.Module):
    """Freeze the AE and expose only its standardized composition coordinates.

    Encoding is used during data preparation under ``no_grad``. Decoding is
    deliberately differentiable with respect to a downstream predicted latent,
    while every checkpoint parameter remains frozen.
    """

    def __init__(self, model: nn.Module, checkpoint: dict, checkpoint_path: Path):
        super().__init__()
        if "latent_statistics" not in checkpoint:
            raise ValueError("completed best.pt with latent_statistics is required")
        if not checkpoint.get("representation_frozen_for_downstream", False):
            raise ValueError("checkpoint is not marked frozen for downstream use")
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        statistics = checkpoint["latent_statistics"]
        mean = torch.as_tensor(
            np.asarray(statistics["composition_mean"], dtype=np.float32)
        )
        scale = torch.as_tensor(
            np.asarray(statistics["composition_scale"], dtype=np.float32)
        )
        composition_dim = int(model.config.composition_dim)
        if mean.shape != (composition_dim,) or scale.shape != (composition_dim,):
            raise ValueError("checkpoint composition statistics have the wrong shape")
        if bool((scale <= 0).any()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("checkpoint composition scale is invalid")
        self.register_buffer("composition_mean", mean)
        self.register_buffer("composition_scale", scale)

        genes = np.asarray(checkpoint["genes"], dtype=str)
        if genes.shape != (model.config.n_genes,) or len(np.unique(genes)) != len(genes):
            raise ValueError("checkpoint genes are invalid")
        self.genes = tuple(genes.tolist())
        self.checkpoint_path = str(checkpoint_path.resolve())
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.manifest_sha256 = str(checkpoint.get("manifest_sha256", ""))
        self.genes_sha256 = hashlib.sha256(
            ("\n".join(self.genes) + "\n").encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "FrozenCompositionAE":
        path = Path(checkpoint_path).resolve()
        checkpoint = _load_pretrain_checkpoint(path, map_location)
        CountAwareAutoencoder, ModelConfig = _pretrain_classes()
        model = CountAwareAutoencoder(
            ModelConfig.from_dict(checkpoint["model_config"])
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        return cls(model, checkpoint, path)

    @property
    def composition_dim(self) -> int:
        return int(self.model.config.composition_dim)

    @property
    def n_genes(self) -> int:
        return int(self.model.config.n_genes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode_standardized(self, counts: torch.Tensor) -> torch.Tensor:
        latent = self.model.encode_composition(counts)
        return (
            latent.float() - self.composition_mean[None, :]
        ) / self.composition_scale[None, :]

    def unstandardize(self, standardized_latent: torch.Tensor) -> torch.Tensor:
        if standardized_latent.shape[-1] != self.composition_dim:
            raise ValueError(
                f"latent last dimension must be {self.composition_dim}"
            )
        return (
            standardized_latent * self.composition_scale
            + self.composition_mean
        )

    def decode_standardized(self, standardized_latent: torch.Tensor) -> dict:
        shape = standardized_latent.shape
        if len(shape) not in (2, 3) or shape[-1] != self.composition_dim:
            raise ValueError("latent must have shape [N,D] or [B,N,D]")
        flat = standardized_latent.reshape(-1, self.composition_dim)
        latent = self.unstandardize(flat)
        logits = self.model.decode_logits(latent)
        probability = torch.softmax(logits.float(), dim=-1)
        return {
            "latent": latent.reshape(*shape[:-1], self.composition_dim),
            "logits": logits.reshape(*shape[:-1], self.n_genes),
            "probability": probability.reshape(*shape[:-1], self.n_genes),
        }

    def composition_cross_entropy(
        self,
        standardized_latent: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = self.decode_standardized(standardized_latent)
        if decoded["logits"].shape != target_counts.shape:
            raise ValueError("decoded logits and target counts do not align")
        target = target_counts.float()
        library = target.sum(dim=-1)
        valid = library > 0
        if not bool(valid.any()):
            return decoded["logits"].sum() * 0.0, valid
        per_spot = -(
            target * F.log_softmax(decoded["logits"].float(), dim=-1)
        ).sum(dim=-1) / library.clamp_min(1.0)
        return per_spot[valid].mean(), valid

    def target_composition(self, target_counts: torch.Tensor) -> torch.Tensor:
        target = target_counts.float()
        return target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def audit(self) -> dict:
        return {
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_sha256": self.manifest_sha256,
            "genes_sha256": self.genes_sha256,
            "genes": self.n_genes,
            "composition_dim": self.composition_dim,
            "checkpoint_weights_frozen": all(
                not parameter.requires_grad for parameter in self.model.parameters()
            ),
            "predicted_feature": "composition latent only; library is excluded",
        }
