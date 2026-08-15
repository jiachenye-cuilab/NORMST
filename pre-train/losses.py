"""UMI thinning and count-aware denoising objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import CountAwareAutoencoder


@dataclass(frozen=True)
class LossConfig:
    thinning_probability: float = 0.5
    nb_weight: float = 0.1
    consistency_weight: float = 0.05
    latent_variance_weight: float = 0.0
    latent_covariance_weight: float = 0.0

    def __post_init__(self):
        if not 0 < self.thinning_probability < 1:
            raise ValueError("thinning_probability must be in (0, 1)")
        if (
            self.nb_weight < 0
            or self.consistency_weight < 0
            or self.latent_variance_weight < 0
            or self.latent_covariance_weight < 0
        ):
            raise ValueError("loss weights must be non-negative")


def latent_variance_covariance(
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style scale floor and correlation penalty across spots."""
    if latent.ndim != 2 or latent.shape[1] < 2:
        raise ValueError("latent must be a [spots, dimensions] matrix")
    values = latent.float()
    centered = values - values.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    standard_deviation = torch.sqrt(variance + 1e-4)
    variance_penalty = F.relu(1.0 - standard_deviation).mean()
    normalized = centered / standard_deviation.clamp_min(1e-4)
    correlation = normalized.T @ normalized / max(values.shape[0], 1)
    off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
    covariance_penalty = off_diagonal.square().sum() / values.shape[1]
    return variance_penalty, covariance_penalty


def binomial_thinning(
    counts: torch.Tensor,
    probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complementary integer-valued views whose sum is exact."""
    if not 0 < probability < 1:
        raise ValueError("probability must be in (0, 1)")
    if counts.ndim != 2 or bool((counts < 0).any()):
        raise ValueError("counts must be a non-negative matrix")
    if not bool(torch.isfinite(counts).all()):
        raise ValueError("counts must be finite")
    rounded = counts.round()
    if not bool(torch.allclose(counts, rounded, atol=1e-5, rtol=0.0)):
        raise ValueError("binomial thinning requires integer-valued counts")
    probabilities = torch.full_like(counts, probability)
    first = torch.binomial(rounded, probabilities, generator=generator)
    second = rounded - first
    return first, second


def composition_cross_entropy(
    logits: torch.Tensor,
    target_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-UMI cross entropy, averaged only over positive-library targets."""
    if logits.shape != target_counts.shape:
        raise ValueError("logits and target_counts must align")
    target = target_counts.float()
    library = target.sum(dim=1)
    valid = library > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0, valid
    per_spot = -(target * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
    per_spot = per_spot / library.clamp_min(1.0)
    return per_spot[valid].mean(), valid


def negative_binomial_nll(
    target_counts: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
) -> torch.Tensor:
    """Elementwise NB2 negative log likelihood with Var=x+x^2/theta."""
    target = target_counts.float()
    mu = mean.float().clamp_min(1e-8)
    theta = dispersion.float().clamp_min(1e-4)
    if theta.ndim == 1:
        theta = theta[None, :]
    if target.shape != mu.shape or theta.shape[-1] != target.shape[-1]:
        raise ValueError("NB target, mean and dispersion must align")
    log_theta_mu = torch.log(theta + mu)
    log_prob = (
        torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + target * (torch.log(mu) - log_theta_mu)
    )
    return -log_prob


def _direction_loss(
    logits: torch.Tensor,
    target_counts: torch.Tensor,
    dispersion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cross_entropy, valid = composition_cross_entropy(logits, target_counts)
    target_library = target_counts.float().sum(dim=1, keepdim=True)
    mean = target_library * torch.softmax(logits.float(), dim=1)
    nb_element = negative_binomial_nll(target_counts, mean, dispersion)
    nb_per_spot = nb_element.mean(dim=1)
    nb = nb_per_spot[valid].mean() if bool(valid.any()) else logits.sum() * 0.0
    return cross_entropy, nb, valid


def denoising_objective(
    model: CountAwareAutoencoder,
    counts: torch.Tensor,
    config: LossConfig,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Symmetric A->B/B->A thinning objective plus latent consistency."""
    first, second = binomial_thinning(
        counts,
        probability=config.thinning_probability,
        generator=generator,
    )
    latent_first = model.encode_composition(first)
    latent_second = model.encode_composition(second)
    logits_first = model.decode_logits(latent_first)
    logits_second = model.decode_logits(latent_second)

    ce_ab, nb_ab, valid_ab = _direction_loss(
        logits_first, second, model.dispersion
    )
    ce_ba, nb_ba, valid_ba = _direction_loss(
        logits_second, first, model.dispersion
    )
    cross_entropy = 0.5 * (ce_ab + ce_ba)
    nb = 0.5 * (nb_ab + nb_ba)

    valid_pair = valid_ab & valid_ba
    if bool(valid_pair.any()):
        first_normalized = F.normalize(latent_first[valid_pair].float(), dim=1)
        second_normalized = F.normalize(latent_second[valid_pair].float(), dim=1)
        consistency = (1.0 - (first_normalized * second_normalized).sum(dim=1)).mean()
    else:
        consistency = latent_first.sum() * 0.0

    variance, covariance = latent_variance_covariance(
        torch.cat([latent_first, latent_second], dim=0)
    )

    total = (
        cross_entropy
        + config.nb_weight * nb
        + config.consistency_weight * consistency
        + config.latent_variance_weight * variance
        + config.latent_covariance_weight * covariance
    )
    return {
        "total_loss": total,
        "composition_cross_entropy": cross_entropy,
        "negative_binomial_nll": nb,
        "latent_consistency": consistency,
        "latent_variance": variance,
        "latent_covariance": covariance,
        "valid_directions": valid_ab.sum() + valid_ba.sum(),
        "valid_pairs": valid_pair.sum(),
    }
