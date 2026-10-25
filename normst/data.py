"""Portable, explicit input arrays; no dataset-specific paths or downloads."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from .artifacts import file_identity, write_json
from .geometry import basis_shape, geometry


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_settings(path=None):
    settings = read_json(Path(__file__).with_name("defaults.json"))
    if path is not None:
        def merge(base, overrides, prefix=""):
            if not isinstance(overrides, dict):
                raise ValueError(f"{prefix or 'settings'} must be a JSON object")
            for key, value in overrides.items():
                name = prefix + key
                if key not in base:
                    raise ValueError(f"unknown training setting: {name}")
                if isinstance(base[key], dict):
                    merge(base[key], value, name + ".")
                else:
                    base[key] = value
        merge(settings, read_json(path))
    validate_settings(settings)
    return settings


def validate_settings(settings):
    def number(value, name, *, integer=False, minimum=0, strict=True):
        if (type(value) not in ((int,) if integer else (int, float)) or not np.isfinite(value)
                or (value <= minimum if strict else value < minimum)):
            relation = ">" if strict else ">="
            raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'} {relation} {minimum}")

    number(settings["seed"], "seed", integer=True, strict=False)
    if settings["seed"] > 2**32 - 1:
        raise ValueError("seed must be at most 2**32 - 1")
    for name in ("epochs", "batch_groups", "query_chunk", "cpu_threads"):
        number(settings[name], name, integer=True)
    for name in ("learning_rate", "gradient_clip", "block_um", "warmup_epochs", "schedule_horizon_epochs"):
        number(settings[name], name)
    for name in ("minimum_learning_rate", "weight_decay"):
        number(settings[name], name, strict=False)
    if settings["minimum_learning_rate"] > settings["learning_rate"]:
        raise ValueError("minimum_learning_rate must not exceed learning_rate")
    if settings["warmup_epochs"] >= settings["schedule_horizon_epochs"]:
        raise ValueError("warmup_epochs must be smaller than schedule_horizon_epochs")
    offsets = settings["offsets_um"]
    if not isinstance(offsets, list) or not offsets:
        raise ValueError("offsets_um must be a nonempty list of [y, x] offsets")
    for offset in offsets:
        if not isinstance(offset, list) or len(offset) != 2:
            raise ValueError("each offsets_um entry must contain two numbers")
        for value in offset:
            number(value, "offsets_um", strict=False)
            if value >= settings["block_um"]:
                raise ValueError("offsets_um entries must be smaller than block_um")
    conditioning = settings["conditioning"]
    for name in ("neighbors", "conditioning_chunk_points"):
        number(conditioning[name], "conditioning." + name, integer=True)
    for name in ("latent_pitch_um", "kernel_um"):
        number(conditioning[name], "conditioning." + name)
    if conditioning["latent_pitch_um"] != 32:
        raise ValueError("conditioning.latent_pitch_um must be 32")


def manifest(path):
    path = Path(path).resolve()
    spec = read_json(path)
    for name in ("observations", "features", "feature_valid", "feature_metadata", "tissue_mask", "image"):
        if name in spec:
            spec[name] = path.parent / spec[name]
    if "features" in spec:
        destination = spec["features"].resolve()
        reserved = {path, *(spec[name].resolve() for name in ("observations", "feature_valid", "image", "tissue_mask") if name in spec)}
        products = {destination, feature_metadata_path(spec).resolve(), destination.with_suffix(".normalization.npz"), destination.with_suffix(".lock")}
        if products & reserved:
            raise ValueError("feature artifacts must use paths distinct from other dataset inputs")
        if feature_metadata_path(spec).resolve() in {destination, destination.with_suffix(".normalization.npz"), destination.with_suffix(".lock")}:
            raise ValueError("feature metadata must use a path distinct from the array, normalization and lock")
    return spec


def load_observations(spec):
    with np.load(spec["observations"], allow_pickle=False) as archive:
        data = dict(counts=np.asarray(archive["counts"], dtype=np.float32),
                    centers=np.asarray(archive["centers_yx_um"], dtype=np.float64),
                    radii=np.asarray(archive["radius_um"], dtype=np.float64),
                    gene_ids=archive["gene_ids"].astype(str).tolist(),
                    support_bins=archive["support_bins"] if "support_bins" in archive else None)
        if data["counts"].ndim != 2 or not all(data["counts"].shape):
            raise ValueError("counts must be a nonempty two-dimensional spot-by-gene array")
        n = len(data["counts"])
        data["barcodes"] = archive["barcodes"].astype(str).tolist() if "barcodes" in archive else [str(i) for i in range(n)]
    counts = data["counts"]
    try:
        data["radii"] = np.broadcast_to(data["radii"], (n,)).copy()
    except ValueError as error:
        raise ValueError(f"radius_um must be scalar or have {n} values aligned to count rows") from error
    for name, length in (("gene_ids", counts.shape[1]), ("barcodes", n)):
        ids = data[name]
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids) or len(ids) != length:
            raise ValueError(f"{name} must contain {length} nonempty strings in count-array order")
        duplicates = len(ids) - len(set(ids))
        if duplicates:
            raise ValueError(f"{name} contain {duplicates} duplicate entries")
    data["bounds"] = np.asarray(spec["bounds_yxyx_um"], dtype=np.float64)
    basis_shape(data["bounds"], 32.)
    if (data["bounds"] % 2).any():
        raise ValueError("bounds must align with the fixed 2 um integration lattice")
    if (not np.isfinite(counts).all()
            or (counts < 0).any() or (counts != np.floor(counts)).any()
            or data["centers"].shape != (n, 2) or not np.isfinite(data["centers"]).all()
            or not np.isfinite(data["radii"]).all() or (data["radii"] <= 0).any()):
        raise ValueError("require aligned raw integer counts and finite capture geometry")
    if ((data["centers"] - data["radii"][:, None] < data["bounds"][:2]).any()
            or (data["centers"] + data["radii"][:, None] > data["bounds"][2:]).any()):
        raise ValueError("field bounds must contain every complete capture circle")
    # Total projection has a unique meaning only for disjoint observed supports.
    pairs = cKDTree(data["centers"]).query_pairs(2 * data["radii"].max(), output_type="ndarray")
    if len(pairs):
        separation = np.linalg.norm(data["centers"][pairs[:, 0]] - data["centers"][pairs[:, 1]], axis=1)
        if (separation < data["radii"][pairs].sum(1)).any():
            raise ValueError("overlapping capture circles are not supported")
        data["_touching_pairs"] = pairs[separation == data["radii"][pairs].sum(1)]
    else:
        data["_touching_pairs"] = pairs
    data["quadrature"] = dict(kind="native_bin_centers", bin_um=2.)
    geometry(data)
    return data


def validate_feature_arrays(spec, path=None):
    path = Path(spec["features"] if path is None else path)
    try:
        features = np.load(path, mmap_mode="c", allow_pickle=False)
    except (ValueError, OSError, EOFError) as error:
        raise ValueError("features file is truncated or is not a valid NPY array") from error
    if not isinstance(features, np.memmap):
        raise ValueError("features must be a single NPY array")
    try:
        valid = np.load(spec["feature_valid"], mmap_mode="c", allow_pickle=False)
    except (ValueError, OSError, EOFError) as error:
        raise ValueError("feature_valid file is truncated or is not a valid NPY array") from error
    if not isinstance(valid, np.memmap):
        raise ValueError("feature_valid must be a single NPY array")
    shape = basis_shape(spec["bounds_yxyx_um"], 8.)
    if (features.ndim != 3 or features.dtype != np.float32 or features.shape[0] < 1
            or features.shape[1:] != shape):
        raise ValueError("features must be float32 [C,H,W] on the 8 um grid")
    if path.stat().st_size != features.offset + features.nbytes:
        raise ValueError("features NPY byte length does not match its declared array")
    if valid.shape != shape or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("feature_valid must be a nonempty boolean [H,W] array")
    if Path(spec["feature_valid"]).stat().st_size != valid.offset + valid.nbytes:
        raise ValueError("feature_valid NPY byte length does not match its declared array")
    for first in range(0, len(features), 32):
        block = features[first:first + 32]
        if not np.isfinite(block).all() or np.any(block[:, ~valid] != 0):
            raise ValueError("features must be finite; invalid image cells must contain zeros")
    offset = spec.get("feature_tile_offset_yx", [0, 0])
    if (not isinstance(offset, (list, tuple)) or len(offset) != 2
            or any(type(v) is not int or not 0 <= v < 14 for v in offset)):
        raise ValueError("feature_tile_offset_yx must contain two integers in [0, 14)")
    return features, valid


def feature_record(spec, features, *, path=None, source="external", **metadata):
    return dict(format_version=1, status="complete", source=source, shape=list(features.shape),
        dtype="float32", bounds_yxyx_um=spec["bounds_yxyx_um"],
        feature_tile_offset_yx=list(spec.get("feature_tile_offset_yx", [0, 0])),
        files=dict(features=file_identity(spec["features"] if path is None else path),
                   feature_valid=file_identity(spec["feature_valid"])), **metadata)


def feature_metadata_path(spec):
    return Path(spec.get("feature_metadata", Path(spec["features"]).with_suffix(".json")))


def load_image_features(spec, *, return_record=False):
    marker = feature_metadata_path(spec)
    if not marker.exists():
        raise ValueError("features lack completion metadata; finish extract or explicitly validate completed external features")
    record = read_json(marker)
    if record.get("status") != "complete" or record.get("format_version") != 1:
        raise ValueError("features are incomplete or unregistered; rerun extraction, or validate known-complete external features")
    features, valid = validate_feature_arrays(spec)
    expected = feature_record(spec, features, source=record.get("source"))
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("feature completion metadata does not match the arrays or geometry; revalidate the completed inputs")
    if record.get("source") == "normst":
        normalization = Path(spec["features"]).with_suffix(".normalization.npz")
        if not normalization.exists() or file_identity(normalization) != record.get("normalization"):
            raise ValueError("completed features have missing or mismatched normalization statistics")
    result = torch.from_numpy(features)[None], torch.from_numpy(valid)[None, None]
    return (*result, record) if return_record else result


def runtime(device, threads=2):
    device = torch.device(device)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for the small example")
    return device


def context_to(context, device):
    return {name: value.to(device) for name, value in context.items()}
