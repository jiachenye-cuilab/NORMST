"""Count and conditional allocation objectives on observed capture regions."""

import torch


def count_deviance(prediction, observed):
    mean = prediction.clamp_min(1e-12)
    target = observed.clamp_min(1e-12)
    return 2 * (mean - observed + observed * (target.log() - mean.log())).mean()


def allocation_deviance(prediction, observed, owners):
    """Conditional part of Poisson deviance, averaged per child spot and gene.

    Predicted totals are normalized only inside the objective. The density
    field itself is not projected onto observed totals.
    """
    owners = torch.as_tensor(owners, device=prediction.device, dtype=torch.long)
    predicted, target = prediction.double().clamp_min(1e-12), observed.double()
    groups = int(owners.max()) + 1
    mean = predicted.new_zeros(groups, predicted.shape[1]).index_add(0, owners, predicted)
    total = target.new_zeros(groups, target.shape[1]).index_add(0, owners, target)
    log_p = predicted.log() - mean[owners].log()
    log_y = target.clamp_min(1e-12).log() - total[owners].clamp_min(1e-12).log()
    return 2 * (target * (log_y - log_p)).mean()


def coarse_deviance(prediction, observed, owners):
    """Poisson deviance of group totals, divided by child spots and genes.

    Restriction sums the original 55um capture supports; it assigns no area or
    invented target counts to gaps. Conditional + this term equals direct
    Poisson deviance on child counts, but now under the coarsened input view.
    """
    owners = torch.as_tensor(owners, device=prediction.device, dtype=torch.long)
    groups = int(owners.max()) + 1
    predicted, target = prediction.double().clamp_min(1e-12), observed.double()
    total_p = predicted.new_zeros(groups, predicted.shape[1]).index_add(0, owners, predicted)
    total_y = target.new_zeros(groups, target.shape[1]).index_add(0, owners, target)
    return count_deviance(total_p, total_y) * groups / len(predicted)
