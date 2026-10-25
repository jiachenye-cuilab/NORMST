"""Frozen Virchow2 CLS/patch tokens and RGB on an aligned 8 um grid."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import manifest, runtime, write_json
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
    import tifffile

    device = runtime(device)
    spec = manifest(data_path)
    height, width = basis_shape(spec["bounds_yxyx_um"], 8.)
    if batch_tiles < 1:
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
    destination = Path(spec["features"])
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    write_json(destination.with_suffix(".json"), dict(encoder="Virchow2", channels=2563,
        pitch_um=8, tile_um=112, image_pixel_um=.5, shape=list(features.shape), frozen=True))
    return destination
