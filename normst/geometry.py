"""Physical capture geometry and reusable bilinear basis integrals."""

import time
import numpy as np
from scipy import sparse
import torch


def circle_quadrature(centers, radii, specification):
    """Return physical (y,x) nodes, area weights and their observed spot owners.

    The native-bin rule matches a simulated assay's capture footprint, without
    accessing native-bin expression. Polar quadrature models physical disks.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(centers),))
    if centers.shape != (len(radii), 2) or not np.isfinite(centers).all() or not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("finite physical spot centers and positive radii are required")
    points, weights, owners = [], [], []
    kind = specification["kind"]
    if kind == "polar":
        nr, na = specification["radial_nodes"], specification["angular_nodes"]
        if nr < 2 or na < 8:
            raise ValueError("disk quadrature requires at least 2 radial and 8 angular nodes")
        nodes, radial_weights = np.polynomial.legendre.leggauss(nr)
        radius = np.sqrt((nodes + 1) / 2)
        angles = (np.arange(na) + .5) * (2 * np.pi / na)
        offsets = (radius[:, None, None] * np.stack((np.sin(angles), np.cos(angles)), -1)[None]).reshape(-1, 2)
    for index, (center, r) in enumerate(zip(centers, radii)):
        if kind == "native_bin_centers":
            pitch = float(specification["bin_um"])
            if not np.isfinite(pitch) or pitch <= 0:
                raise ValueError("invalid integration bin pitch")
            lower, upper = np.floor((center - r) / pitch).astype(int), np.ceil((center + r) / pitch).astype(int)
            yy, xx = np.meshgrid(np.arange(lower[0], upper[0]), np.arange(lower[1], upper[1]), indexing="ij")
            p = (np.stack((yy.ravel(), xx.ravel()), 1) + .5) * pitch
            p = p[np.square(p - center).sum(1) <= r * r]
            w = np.full(len(p), pitch * pitch)
        elif kind == "polar":
            p = center + r * offsets
            w = np.repeat(radial_weights / 2, na) * (np.pi * r * r / na)
        else:
            raise ValueError(f"unknown observation geometry: {kind}")
        if not len(p):
            raise ValueError("a spot has no integration support")
        points.append(p)
        weights.append(w)
        owners.append(np.full(len(p), index, dtype=np.int64))
    return np.concatenate(points), np.concatenate(weights), np.concatenate(owners)


def geometry(data):
    points, weights, owners = circle_quadrature(data["centers"], data["radii"], data["quadrature"])
    areas = np.bincount(owners, weights=weights, minlength=len(data["centers"]))
    if data["support_bins"] is not None:
        np.testing.assert_array_equal(areas, data["support_bins"] * data["quadrature"]["bin_um"] ** 2)
    data["areas"] = areas
    return (torch.tensor(points, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32),
            torch.tensor(owners, dtype=torch.int64))


def basis_shape(bounds, pitch):
    bounds = np.asarray(bounds, dtype=np.float64)
    shape = (bounds[2:] - bounds[:2]) / pitch
    if (bounds.shape != (4,) or not np.isfinite(bounds).all() or not np.isfinite(pitch)
            or pitch <= 0 or (shape < 2).any() or not np.allclose(shape, np.rint(shape), rtol=0, atol=1e-8)):
        raise ValueError("basis spacing must divide finite field bounds, with at least two knots per axis")
    return tuple(int(value) for value in np.rint(shape))


def stencil(points, bounds, pitch, shape):
    """Four cell-centred hat functions, with constant extension to domain edges."""
    maximum = points.new_tensor(shape) - 1
    relative = torch.minimum(((points - bounds[:2]) / pitch - .5).clamp_min(0), maximum)
    lower = relative.floor().long()
    upper = torch.minimum(lower + 1, maximum.long())
    fraction = relative - lower
    y, x = fraction.unbind(1)
    indices = torch.stack((lower[:, 0] * shape[1] + lower[:, 1],
                           lower[:, 0] * shape[1] + upper[:, 1],
                           upper[:, 0] * shape[1] + lower[:, 1],
                           upper[:, 0] * shape[1] + upper[:, 1]), 1)
    weights = torch.stack(((1 - y) * (1 - x), (1 - y) * x, y * (1 - x), y * x), 1)
    return indices, weights


def knot_points(indices, bounds, pitch, shape):
    coordinates = torch.stack((indices // shape[1], indices % shape[1]), 1)
    return bounds[:2] + (coordinates.to(bounds.dtype) + .5) * pitch


class BasisIntegrals:
    """Precompute geometry on CPU; move only a spot batch's sparse weights to GPU."""

    def __init__(self, data, device, pitch):
        started = time.monotonic()
        points, weights, owners = geometry(data)
        self.device, self.pitch = device, float(pitch)
        self.shape = basis_shape(data["bounds"], pitch)
        bounds = torch.as_tensor(data["bounds"], dtype=points.dtype)
        self.bounds = bounds.to(device)
        self.sizes = np.bincount(owners.numpy(), minlength=len(data["counts"]))
        offsets = np.r_[0, self.sizes.cumsum()]
        parts = []
        for first in range(0, len(self.sizes), 128):
            last = min(first + 128, len(self.sizes))
            sl = slice(offsets[first], offsets[last])
            indices, fractions = stencil(points[sl], bounds, pitch, self.shape)
            values = (fractions * weights[sl, None]).numpy().reshape(-1)
            rows = np.repeat(owners[sl].numpy() - first, 4)
            part = sparse.coo_matrix((values, (rows, indices.numpy().reshape(-1))),
                                     shape=(last - first, int(np.prod(self.shape)))).tocsr()
            part.eliminate_zeros()
            parts.append(part)
        self.matrix = sparse.vstack(parts, format="csr")
        np.testing.assert_allclose(np.asarray(self.matrix.sum(1)).ravel(), data["areas"], rtol=2e-6, atol=1e-4)
        self.statistics = dict(basis_pitch_um=self.pitch, basis_shape=list(self.shape),
            original_quadrature_points=len(points), nonzero_region_weights=self.matrix.nnz,
            geometry_seconds=time.monotonic() - started,
            sparse_geometry_MiB=sum(a.nbytes for a in (self.matrix.data, self.matrix.indices, self.matrix.indptr)) / 2**20)

    def batch(self, rows):
        sub = self.matrix[np.asarray(rows, dtype=np.int64)]
        unique, columns = np.unique(sub.indices, return_inverse=True)
        owners = np.repeat(np.arange(len(rows)), np.diff(sub.indptr))
        indices = torch.as_tensor(np.stack((owners, columns)), dtype=torch.int64, device=self.device)
        weights = torch.sparse_coo_tensor(indices, torch.as_tensor(sub.data, device=self.device),
            (len(rows), len(unique)), device=self.device, is_coalesced=True)
        points = knot_points(torch.as_tensor(unique, dtype=torch.int64, device=self.device),
                             self.bounds, self.pitch, self.shape)
        return points, weights, None
