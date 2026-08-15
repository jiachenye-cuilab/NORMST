"""Export frozen per-spot features without changing the source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from checkpoint_io import load_checkpoint
from data import (
    SpotCountDataset,
    assert_output_outside_sources,
    load_pretrain_data,
    parse_manifest,
    verify_source_contract,
)
from model import CountAwareAutoencoder, ModelConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export frozen count encoder features")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--count-file", default="filtered_feature_bc_matrix.h5"
    )
    parser.add_argument("--roles", default="train,val,test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--allow-different-manifest",
        action="store_true",
        help="allow encoding a new split contract with the checkpoint gene order",
    )
    return parser.parse_args(argv)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics(checkpoint: dict, composition_dim: int):
    if "latent_statistics" not in checkpoint:
        raise ValueError(
            "checkpoint has no train-fitted latent statistics; use completed best.pt"
        )
    payload = checkpoint["latent_statistics"]
    composition_mean = np.asarray(payload["composition_mean"], dtype=np.float32)
    composition_scale = np.asarray(payload["composition_scale"], dtype=np.float32)
    library_mean = np.asarray(payload["log_library_mean"], dtype=np.float32)
    library_scale = np.asarray(payload["log_library_scale"], dtype=np.float32)
    if composition_mean.shape != (composition_dim,):
        raise ValueError("checkpoint composition_mean has the wrong shape")
    if composition_scale.shape != (composition_dim,) or np.any(composition_scale <= 0):
        raise ValueError("checkpoint composition_scale is invalid")
    if library_mean.shape != (1,) or library_scale.shape != (1,):
        raise ValueError("checkpoint library statistics are invalid")
    return composition_mean, composition_scale, library_mean, library_scale


@torch.no_grad()
def encode_slice(model, item, batch_size, workers, device, use_amp):
    dataset = SpotCountDataset([item])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    latents = []
    libraries = []
    for batch in loader:
        counts = batch["counts"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            latent, log_library = model.encode(counts)
        latents.append(latent.float().cpu().numpy())
        libraries.append(log_library.float().cpu().numpy())
    return np.concatenate(latents), np.concatenate(libraries)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid loader settings")
    roles = tuple(part.strip() for part in args.roles.split(",") if part.strip())
    if not roles or any(role not in ("train", "val", "test") for role in roles):
        raise ValueError("roles must be selected from train,val,test")

    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    genes = np.asarray(checkpoint["genes"], dtype=str)
    if genes.shape != (model_config.n_genes,):
        raise ValueError("checkpoint genes do not match model_config")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (
        not args.allow_different_manifest
        and manifest_sha != checkpoint.get("manifest_sha256")
    ):
        raise ValueError(
            "manifest differs from pretraining; pass --allow-different-manifest "
            "only for an intentional external encoding"
        )

    entries = parse_manifest(manifest_path, args.count_file)
    output = assert_output_outside_sources(args.output_dir, entries)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = load_pretrain_data(
        manifest_path,
        count_file=args.count_file,
        fixed_genes=genes,
    )
    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    model = CountAwareAutoencoder(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.freeze_representation()
    mean, scale, library_mean, library_scale = _statistics(
        checkpoint, model_config.composition_dim
    )

    files = []
    for role in roles:
        for item in data.for_role(role):
            latent, log_library = encode_slice(
                model,
                item,
                args.batch_size,
                args.workers,
                device,
                use_amp,
            )
            raw_feature = np.concatenate([latent, log_library], axis=1)
            standardized = np.concatenate(
                [
                    (latent - mean[None, :]) / scale[None, :],
                    (log_library - library_mean[None, :])
                    / library_scale[None, :],
                ],
                axis=1,
            ).astype(np.float32)
            destination = output / f"{item.name}.npz"
            np.savez_compressed(
                destination,
                composition_latent=latent.astype(np.float32),
                log_library=log_library.astype(np.float32),
                feature_raw=raw_feature.astype(np.float32),
                feature_standardized=standardized,
                barcodes=item.barcodes,
                slice_name=np.asarray(item.name),
                role=np.asarray(role),
            )
            files.append(
                {
                    "slice": item.name,
                    "role": role,
                    "path": str(destination),
                    "spots": item.n_spots,
                    "feature_dim": int(standardized.shape[1]),
                }
            )

    (output / "genes.txt").write_text(
        "\n".join(genes.tolist()) + "\n", encoding="utf-8"
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "same_manifest_as_pretraining": manifest_sha
        == checkpoint.get("manifest_sha256"),
        "roles": list(roles),
        "genes": len(genes),
        "composition_dim": model_config.composition_dim,
        "feature_dim": model_config.feature_dim,
        "feature_layout": "standardized composition latent, standardized log library",
        "standardization_fit": "pretraining train spots only",
        "model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "files": files,
        "source_contract": data.source_contract,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    verify_source_contract(data.source_contract)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
