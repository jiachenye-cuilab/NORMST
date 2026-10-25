"""Encode observed counts and spatially coarsened input views."""

import numpy as np
import torch


def condition_with_prior(counts, centers, areas, visible, bounds, settings):
    """Build log-count features and a raw-density prior from visible spots only."""
    counts = np.asarray(counts, dtype=np.float64)
    available = np.flatnonzero(visible)
    if not len(available) or not np.isfinite(counts[available]).all() or (counts[available] < 0).any():
        raise ValueError("valid visible counts are required")
    available = available[np.lexsort((centers[available, 1], centers[available, 0]))]
    lo, hi = np.asarray(bounds[:2]), np.asarray(bounds[2:])
    shape = np.ceil((hi - lo) / settings["latent_pitch_um"]).astype(int)
    axis = [lo[d] + (np.arange(shape[d]) + .5) * (hi[d] - lo[d]) / shape[d] for d in range(2)]
    points = np.stack(np.meshgrid(*axis, indexing="ij"), -1).reshape(-1, 2)
    rates = counts[available] / areas[available, None]
    reference = np.maximum(counts[available].sum(0) / areas[available].sum(), 1e-8)
    encoded = np.log1p(rates / reference[None])
    features = np.empty((len(points), counts.shape[1] + 3), dtype=np.float32)
    prior = np.empty((len(points), counts.shape[1]), dtype=np.float32)
    k, width = min(settings["neighbors"], len(available)), settings["kernel_um"]
    chunk = settings["conditioning_chunk_points"]
    for begin in range(0, len(points), chunk):
        query = points[begin:begin + chunk]
        distance = np.square(query[:, None] - centers[available][None]).sum(2)
        nearest = np.argsort(distance, axis=1, kind="stable")[:, :k]
        d2 = np.take_along_axis(distance, nearest, axis=1)
        weight = 1 / (1 + d2 / width ** 2)
        weight /= weight.sum(1, keepdims=True)
        lifted = (encoded[nearest] * weight[..., None]).sum(1)
        relative = 2 * (query - lo) / (hi - lo) - 1
        features[begin:begin + len(query)] = np.concatenate((lifted, np.sqrt(d2[:, :1]) / width, relative), 1)
        prior[begin:begin + len(query)] = (rates[nearest] * weight[..., None]).sum(1)
    return dict(conditioning=torch.from_numpy(features.T.reshape(1, counts.shape[1] + 3, *shape)),
                reference_rate=torch.tensor(reference, dtype=torch.float32),
                prior_density=torch.from_numpy(prior.T.reshape(1, counts.shape[1], *shape)))


def spatial_groups(centers, bounds, block_um, offset_um):
    if block_um <= 0 or np.shape(offset_um) != (2,):
        raise ValueError("positive grouping width and two-dimensional offset required")
    cells = np.floor((np.asarray(centers) - np.asarray(bounds[:2]) + offset_um) / block_um).astype(np.int64)
    return np.unique(cells, axis=0, return_inverse=True)[1]


def grouped_sum(values, owners):
    result = np.zeros((int(owners.max()) + 1, *values.shape[1:]), dtype=np.float64)
    np.add.at(result, owners, values)
    return result


def coarsened_counts(counts, areas, owners):
    """Spread each observed group total in proportion to original capture area.

    Geometry is unchanged. These are input features, never invented targets;
    no expression or area is assigned to gaps between the original supports.
    """
    totals, area = grouped_sum(counts, owners), grouped_sum(areas, owners)
    if (area <= 0).any() or not np.isfinite(totals).all() or (totals < 0).any():
        raise ValueError("finite nonnegative counts and positive capture area required")
    result = totals[owners] * (areas / area[owners])[:, None]
    np.testing.assert_allclose(grouped_sum(result, owners), totals, rtol=2e-14, atol=1e-10)
    return result


def group_batch(owners, selected):
    rows = np.flatnonzero(np.isin(owners, selected))
    _, local = np.unique(owners[rows], return_inverse=True)
    if not len(rows) or len(np.unique(local)) != len(selected):
        raise ValueError("every selected group must be present exactly once")
    return rows, local
