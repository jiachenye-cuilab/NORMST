"""Frozen Virchow2 CLS/patch tokens and RGB on an aligned 8 um grid."""

import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .artifacts import _sync_file, environment, file_identity, writer_lock
from .data import feature_metadata_path, feature_record, manifest, read_json, runtime, validate_feature_arrays, write_json
from .geometry import basis_shape


def load_backbone(weights, device):
    import timm
    from timm.layers import SwiGLUPacked
    from safetensors.torch import load_file

    model = timm.create_model("vit_huge_patch14_224", pretrained=False, img_size=224,
        init_values=1e-5, num_classes=0, reg_tokens=4, mlp_ratio=5.3375,
        global_pool="", dynamic_img_size=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    weights = Path(weights)
    state = (load_file(str(weights), device="cpu") if weights.suffix == ".safetensors"
             else torch.load(weights, map_location="cpu", weights_only=True))
    model.load_state_dict(state, strict=True)
    if model.num_features != 1280 or model.num_prefix_tokens != 5:
        raise ValueError("weights must match the Virchow2 token layout")
    return model.eval().requires_grad_(False).to(device)


def token_maps(tokens):
    patches = tokens[:, 5:]  # Exclude the CLS token and four register tokens.
    side = math.isqrt(patches.shape[1])
    if side * side != patches.shape[1] or tokens.shape[-1] != 1280:
        raise ValueError("unexpected Virchow2 spatial tokens")
    local = patches.transpose(1, 2).reshape(len(tokens), 1280, side, side)
    local = F.interpolate(local, size=(14, 14), mode="bilinear", align_corners=False)
    cls = tokens[:, 0, :, None, None].expand(-1, -1, 14, 14)
    return torch.cat((cls, local), 1)


def normalize_features(features, valid):
    means, scales = [], []
    for begin in range(0, len(features), 32):
        values = np.array(features[begin:begin + 32], copy=True)
        visible = values[:, valid]
        mean = visible.mean(axis=1, dtype=np.float64)
        scale = np.maximum(visible.std(axis=1, dtype=np.float64), 1e-6)
        values -= mean.astype(np.float32)[:, None, None]
        values /= scale.astype(np.float32)[:, None, None]
        values *= valid
        features[begin:begin + len(values)] = values
        means.extend(mean.tolist())
        scales.extend(scale.tolist())
    return np.asarray(means), np.asarray(scales)


def extract(data_path, weights, *, device="cuda:0", batch_tiles=8):
    spec = manifest(data_path)
    destination = Path(spec["features"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = feature_metadata_path(spec)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(destination.with_suffix(".lock")):
        previous = read_json(marker) if marker.exists() else {}
        if previous.get("status") == "complete" or (destination.exists() and previous.get("source") != "normst"):
            raise FileExistsError(f"{destination}: use a new feature path to preserve existing inputs")
        write_json(marker, dict(format_version=1, status="running", source="normst"))
        try:
            with tempfile.TemporaryDirectory(prefix=f".{destination.name}.partial-", dir=destination.parent) as temporary:
                staged = Path(temporary) / destination.name
                _extract(spec, weights, staged, device=device, batch_tiles=batch_tiles)
                features, _ = validate_feature_arrays(spec, staged)
                normalization = staged.with_name("normalization.npz")
                with np.load(normalization, allow_pickle=False) as values:
                    if (values["mean"].shape != (len(features),) or values["scale"].shape != (len(features),)
                            or not np.isfinite(values["mean"]).all() or not np.isfinite(values["scale"]).all()
                            or (values["scale"] <= 0).any()):
                        raise ValueError("invalid feature normalization statistics")
                record = feature_record(spec, features, path=staged, source="normst", encoder="Virchow2",
                    pitch_um=8, tile_um=112, image_pixel_um=.5, frozen=True,
                    normalization=file_identity(normalization), normalization_file=destination.with_suffix(".normalization.npz").name,
                    weights=file_identity(weights),
                    image=file_identity(spec["image"]), environment=environment(device), batch_tiles=batch_tiles)
                del features
                # Completion metadata is the commit record; readers reject earlier states.
                for path in (staged, normalization):
                    _sync_file(path)
                os.replace(normalization, destination.with_suffix(".normalization.npz"))
                os.replace(staged, destination)
                write_json(marker, record)
        except BaseException:
            write_json(marker, dict(format_version=1, status="failed", source="normst"))
            raise
    return destination


def validate_external_features(data_path, *, confirm_complete=False):
    """Register an external producer's finished arrays without modifying their values."""
    if not confirm_complete:
        raise ValueError("confirm_complete is required: array checks cannot establish that an external producer finished")
    spec = manifest(data_path)
    destination = Path(spec["features"])
    marker = feature_metadata_path(spec)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(marker.with_suffix(".lock")):
        previous = read_json(marker) if marker.exists() else {}
        adjacent = destination.with_suffix(".json")
        managed = read_json(adjacent) if adjacent != marker and adjacent.exists() else {}
        if previous.get("source") == "normst" or managed.get("source") == "normst":
            raise ValueError("extract-managed features must be completed by extract, not registered as external")
        features, _ = validate_feature_arrays(spec)
        write_json(marker, feature_record(spec, features, producer_completion_confirmed=True))
    return marker


def _extract(spec, weights, destination, *, device, batch_tiles):
    import tifffile

    device = runtime(device)
    height, width = basis_shape(spec["bounds_yxyx_um"], 8.)
    if type(batch_tiles) is not int or batch_tiles < 1:
        raise ValueError("batch_tiles must be positive")
    if tuple(spec.get("feature_tile_offset_yx", (0, 0))) != (0, 0):
        raise ValueError("extract expects tiles anchored to the supplied image origin")
    valid = np.load(spec["feature_valid"], allow_pickle=False)
    if valid.dtype != np.bool_ or valid.shape != (height, width) or not valid.any():
        raise ValueError("a nonempty boolean image-valid mask on the 8 um grid is required")
    # The supplied lossless RGB image is registered to the exact field bounds.
    try:
        pixels = tifffile.memmap(spec["image"], mode="r")
    except ValueError:
        pixels = tifffile.imread(spec["image"], maxworkers=2)
    if pixels.shape != (height * 16, width * 16, 3) or pixels.dtype != np.uint8:
        raise ValueError("registered RGB TIFF must use 0.5 um pixels and cover the field bounds")
    model = load_backbone(weights, device)
    features = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=(2563, height, width))
    features[:] = 0
    jobs = [(top, left) for top in range(0, height, 14) for left in range(0, width, 14)
            if valid[top:top + 14, left:left + 14].any()]
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    with torch.inference_mode():
        for begin in range(0, len(jobs), batch_tiles):
            batch = jobs[begin:begin + batch_tiles]
            tiles = []
            for top, left in batch:
                tile = pixels[top * 16:top * 16 + 224, left * 16:left * 16 + 224]
                tiles.append(np.pad(tile, ((0, 224 - len(tile)), (0, 224 - tile.shape[1]), (0, 0)), mode="edge"))
            rgb = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).to(device, torch.float32) / 255
            maps = torch.cat((token_maps(model.forward_features((rgb - mean) / std)), F.avg_pool2d(rgb, 16)), 1)
            if not torch.isfinite(maps).all():
                raise ValueError("nonfinite image features")
            for values, (top, left) in zip(maps.cpu().numpy(), batch):
                hh, ww = min(14, height - top), min(14, width - left)
                features[:, top:top + hh, left:left + ww] = values[:, :hh, :ww]
            print(f"image tiles {begin + len(batch)}/{len(jobs)}", flush=True)
    means, scales = normalize_features(features, valid)
    features.flush()
    np.savez(destination.with_name("normalization.npz"), mean=means, scale=scales)
