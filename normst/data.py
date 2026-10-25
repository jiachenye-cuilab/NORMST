"""Portable, explicit input arrays; no dataset-specific paths or downloads."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from .geometry import basis_shape, geometry


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_settings(path=None):
    settings = read_json(Path(__file__).with_name("defaults.json"))
    if path is not None:
        overrides = read_json(path)
        unknown = overrides.keys() - settings.keys()
        if unknown:
            raise ValueError(f"unknown training settings: {sorted(unknown)}")
        settings.update(overrides)
    if (settings["epochs"] < 1 or settings["batch_groups"] < 1 or settings["query_chunk"] < 1
            or settings["cpu_threads"] < 1 or not 0 < settings["warmup_epochs"] < settings["schedule_horizon_epochs"]
            or settings["conditioning"]["latent_pitch_um"] != 32):
        raise ValueError("invalid training settings")
    return settings


def manifest(path):
    path = Path(path).resolve()
    spec = read_json(path)
    for name in ("observations", "features", "feature_valid", "tissue_mask", "image"):
        if name in spec:
            spec[name] = path.parent / spec[name]
    return spec


def load_observations(spec):
    with np.load(spec["observations"], allow_pickle=False) as archive:
        data = dict(counts=np.asarray(archive["counts"], dtype=np.float32),
                    centers=np.asarray(archive["centers_yx_um"], dtype=np.float64),
                    radii=np.asarray(archive["radius_um"], dtype=np.float64),
                    gene_ids=archive["gene_ids"].astype(str).tolist(),
                    support_bins=archive["support_bins"] if "support_bins" in archive else None)
        n = len(data["counts"])
        data["barcodes"] = archive["barcodes"].astype(str).tolist() if "barcodes" in archive else [str(i) for i in range(n)]
    counts = data["counts"]
    data["radii"] = np.broadcast_to(data["radii"], (n,)).copy()
    data["bounds"] = np.asarray(spec["bounds_yxyx_um"], dtype=np.float64)
    basis_shape(data["bounds"], 32.)
    if (data["bounds"] % 2).any():
        raise ValueError("bounds must align with the fixed 2 um integration lattice")
    if (counts.ndim != 2 or n < 1 or counts.shape[1] < 1 or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts != np.floor(counts)).any()
            or len(data["gene_ids"]) != counts.shape[1] or len(set(data["gene_ids"])) != counts.shape[1]
            or len(data["barcodes"]) != n or len(set(data["barcodes"])) != n
            or data["centers"].shape != (n, 2) or not np.isfinite(data["centers"]).all()
            or not np.isfinite(data["radii"]).all() or (data["radii"] <= 0).any()):
        raise ValueError("require aligned raw integer counts, unique genes/spots, and finite geometry")
    if ((data["centers"] - data["radii"][:, None] < data["bounds"][:2]).any()
            or (data["centers"] + data["radii"][:, None] > data["bounds"][2:]).any()):
        raise ValueError("field bounds must contain every complete capture circle")
    # Total projection has a unique meaning only for disjoint observed supports.
    pairs = cKDTree(data["centers"]).query_pairs(2 * data["radii"].max(), output_type="ndarray")
    if len(pairs):
        separation = np.linalg.norm(data["centers"][pairs[:, 0]] - data["centers"][pairs[:, 1]], axis=1)
        if (separation < data["radii"][pairs].sum(1)).any():
            raise ValueError("overlapping capture circles are not supported")
    data["quadrature"] = dict(kind="native_bin_centers", bin_um=2.)
    geometry(data)
    return data


def load_image_features(spec):
    features = np.load(spec["features"], mmap_mode="c", allow_pickle=False)
    valid = np.load(spec["feature_valid"], allow_pickle=False)
    shape = basis_shape(spec["bounds_yxyx_um"], 8.)
    if features.ndim != 3 or features.dtype != np.float32 or features.shape[1:] != shape:
        raise ValueError("features must be float32 [C,H,W] on the 8 um grid")
    if valid.shape != shape or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("feature_valid must be a nonempty boolean [H,W] array")
    for first in range(0, len(features), 32):
        block = features[first:first + 32]
        if not np.isfinite(block).all() or np.any(block[:, ~valid] != 0):
            raise ValueError("features must be finite; invalid image cells must contain zeros")
    return torch.from_numpy(features)[None], torch.from_numpy(valid)[None, None]


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
