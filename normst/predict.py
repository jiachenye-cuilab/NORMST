"""Regional count readouts with gene-wise conservation of observed spot totals."""

from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch

from .artifacts import environment, file_identity, staged_directory
from .data import runtime, write_json
from .geometry import BasisIntegrals, capture_query_quadrature, geometry, knot_points, observed_owners, stencil
from .model import NORMST


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported NORMST checkpoint format")
    state = payload["state_dict"]
    context = {name: state[name] for name in ("conditioning", "reference_rate", "prior_density")}
    field = NORMST(context, state["bounds"].numpy(), state["histology_features"], state["feature_valid"],
                   tile_offset_yx=payload["feature_tile_offset_yx"])
    field.load_state_dict(state, strict=True)
    field = field.to(device).eval().requires_grad_(False)
    data = {name: value.numpy() for name, value in payload["observations"].items()}
    data.update(bounds=state["bounds"].numpy().astype(np.float64), gene_ids=payload["gene_ids"],
                support_bins=None, quadrature=dict(kind="native_bin_centers", bin_um=2.))
    geometry(data)
    return field, data


def point_matrix(points, weights, rows, count, field):
    points = torch.as_tensor(points, dtype=torch.float64)
    bounds = field.bounds.detach().cpu().double()
    indices, fractions = stencil(points, bounds, 8., field.basis_shape)
    matrix = sparse.coo_matrix(((fractions.numpy() * np.asarray(weights)[:, None]).reshape(-1),
        (np.repeat(rows, 4), indices.numpy().reshape(-1))), shape=(count, int(np.prod(field.basis_shape)))).tocsr()
    matrix.eliminate_zeros()
    return matrix


def multiply(matrix, coefficients, device):
    """Promote only the selected FP32 knots to FP64 integration arithmetic."""
    columns, local = np.unique(matrix.indices, return_inverse=True)
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    indices = torch.as_tensor(np.stack((rows, local)), dtype=torch.int64, device=device)
    values = torch.as_tensor(matrix.data, dtype=torch.float64, device=device)
    weights = torch.sparse_coo_tensor(indices, values, (matrix.shape[0], len(columns)),
                                      device=device, is_coalesced=True)
    selected = torch.as_tensor(np.array(coefficients[columns], copy=True), device=device).double()
    return torch.sparse.mm(weights, selected).cpu().numpy()


def projected_integrals(points, weights, rows, count, field, coefficients, data, factors, tree):
    device = field.bounds.device
    raw = multiply(point_matrix(points, weights, rows, count, field), coefficients, device)
    owners = observed_owners(points, data, tree)
    selected = owners >= 0
    if selected.any():
        pairs, inverse = np.unique(np.column_stack((rows[selected], owners[selected])), axis=0, return_inverse=True)
        matrix = point_matrix(points[selected], weights[selected], inverse, len(pairs), field)
        captured = multiply(matrix, coefficients, device)
        # Correct only each bin's captured fragment. Gap contributions stay raw.
        np.add.at(raw, pairs[:, 0], captured * (factors[pairs[:, 1]] - 1))
    if not np.isfinite(raw).all() or (raw < -1e-8).any():
        raise ValueError("invalid projected counts")
    return raw.clip(0)


@torch.no_grad()
def predict(checkpoint, output, *, device="cuda:0", bin_um=8, mask=None, queries=None, batch=128):
    with staged_directory(output) as staging:
        _predict(checkpoint, staging, device=device, bin_um=bin_um, mask=mask, queries=queries, batch=batch)
    return Path(output) / "counts.npy"


def _predict(checkpoint, output, *, device, bin_um, mask, queries, batch):
    device = runtime(device)
    if type(batch) is not int or batch < 1:
        raise ValueError("batch must be positive")
    field, data = load_model(checkpoint, device)
    bounds = data["bounds"]
    if (bounds % 2).any():
        raise ValueError("prediction requires field bounds on the native 2 um integration lattice")
    if queries is None:
        if type(bin_um) is not int or bin_um < 8 or bin_um % 8 or np.any((bounds[2:] - bounds[:2]) % bin_um):
            raise ValueError("bin_um must be a multiple of 8 um and tile the field bounds")
        shape = np.rint((bounds[2:] - bounds[:2]) / bin_um).astype(int)
        valid = field.feature_valid[0, 0].cpu().numpy()
        factor = bin_um // 8
        coverage = valid.reshape(shape[0], factor, shape[1], factor).mean(axis=(1, 3))
        valid = coverage > 0
        if mask is not None:
            selected = np.load(mask, allow_pickle=False)
            if selected.shape != tuple(shape) or selected.dtype != np.bool_:
                raise ValueError("output mask must be boolean and match the requested output grid")
            valid &= selected
        cells = np.argwhere(valid)
        np.save(output / "coverage_fraction.npy", coverage[valid].astype(np.float32), allow_pickle=False)
        centers = bounds[:2] + (cells + .5) * bin_um
        offsets = (np.indices((bin_um // 2, bin_um // 2)).reshape(2, -1).T + .5) * 2 - bin_um / 2
        radii = None
    else:
        if mask is not None:
            raise ValueError("use either region queries or a grid mask")
        with np.load(queries, allow_pickle=False) as archive:
            centers = np.asarray(archive["centers_yx_um"], dtype=np.float64)
            radii = np.broadcast_to(np.asarray(archive["radius_um"], dtype=np.float64), (len(centers),)).copy()
        if (centers.shape != (len(radii), 2) or not np.isfinite(centers).all() or not np.isfinite(radii).all()
                or (radii <= 0).any() or (centers - radii[:, None] < bounds[:2]).any()
                or (centers + radii[:, None] > bounds[2:]).any()):
            raise ValueError("query circles must be finite and lie entirely inside the field")
    if not len(centers):
        raise ValueError("no query regions selected")
    destination = output / "counts.npy"
    result = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                     shape=(len(centers), len(data["gene_ids"])))
    # Temporary disk cache bounds GPU memory and is removed after inference.
    with tempfile.TemporaryDirectory(prefix=".normst-knots-", dir=output) as temporary:
        coefficients = np.lib.format.open_memmap(Path(temporary) / "knots.npy", mode="w+", dtype=np.float32,
            shape=(int(np.prod(field.basis_shape)), len(data["gene_ids"])))
        latent = field.encode()
        for first in range(0, len(coefficients), 1024):
            indices = torch.arange(first, min(first + 1024, len(coefficients)), device=device)
            points = knot_points(indices, field.bounds, 8., field.basis_shape)
            coefficients[first:first + len(indices)] = field.coefficients(latent, points).cpu().numpy()
        coefficients.flush()
        del latent
        captured = BasisIntegrals(data, device, 8.).matrix.astype(np.float64)
        raw_spots = np.vstack([multiply(captured[first:first + 16], coefficients, device)
                               for first in range(0, len(data["counts"]), 16)])
        if not np.isfinite(raw_spots).all() or ((raw_spots <= 0) & (data["counts"] > 0)).any():
            raise ValueError("positive observed counts have zero or invalid predicted integral")
        factors = np.divide(data["counts"], raw_spots, out=np.zeros_like(raw_spots), where=raw_spots > 0)
        np.save(output / "raw_spot_counts.npy", raw_spots, allow_pickle=False)
        tree = cKDTree(data["centers"])
        for first in range(0, len(centers), batch):
            current = centers[first:first + batch]
            if radii is None:
                points = (current[:, None] + offsets[None]).reshape(-1, 2)
                weights = np.full(len(points), 4.)
                owners = np.repeat(np.arange(len(current)), len(offsets))
            else:
                points, weights, owners = capture_query_quadrature(current, radii[first:first + batch], data)
            result[first:first + len(current)] = projected_integrals(
                points, weights, owners, len(current), field, coefficients, data, factors, tree).astype(np.float32)
            if first == 0 or first + batch >= len(centers) or first // batch % 32 == 0:
                print(f"predicted regions {first + len(current)}/{len(centers)}", flush=True)
        result.flush()
        del coefficients
    for first in range(0, len(result), batch):
        if not np.isfinite(result[first:first + batch]).all() or (result[first:first + batch] < 0).any():
            raise ValueError("prediction output contains invalid values")
    np.save(output / "centers_yx_um.npy", centers, allow_pickle=False)
    if radii is not None:
        np.save(output / "radius_um.npy", radii, allow_pickle=False)
    write_json(output / "genes.json", data["gene_ids"])
    write_json(output / "prediction.json", dict(status="complete", shape=list(result.shape), units="expected captured UMI counts",
        region="square" if radii is None else "circle", bin_um=bin_um if radii is None else None,
        integration_pitch_um=2, observed_total_projection=True, bounds_yxyx_um=bounds.tolist(),
        coverage_rule="any valid 8 um cell; integrate the full square" if radii is None else "query geometry independent of image coverage",
        environment=environment(device), checkpoint=file_identity(checkpoint),
        mask=file_identity(mask) if mask is not None else None,
        queries=file_identity(queries) if queries is not None else None, batch=batch,
        outputs={path.name: file_identity(path) for path in sorted(output.iterdir()) if path.suffix in (".npy", ".json")
                 and path.name not in ("status.json", "prediction.json")}))
    del result
    return destination
