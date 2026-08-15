"""Portable checkpoint I/O for Windows/Linux pretraining workflows."""

from __future__ import annotations

import os
from pathlib import Path, PurePath
import pathlib
import threading

import torch


_PATHLIB_UNPICKLE_LOCK = threading.RLock()


def portable_checkpoint_value(value):
    """Recursively replace OS-specific pathlib objects with plain strings."""
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {
            portable_checkpoint_value(key): portable_checkpoint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portable_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable_checkpoint_value(item) for item in value)
    if isinstance(value, set):
        return {portable_checkpoint_value(item) for item in value}
    return value


def save_checkpoint(path: str | Path, payload: dict) -> None:
    """Atomically save a checkpoint containing no concrete OS path classes."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(portable_checkpoint_value(payload), temporary)
    temporary.replace(target)


def _torch_load(path: Path, map_location, weights_only: bool):
    return torch.load(
        path,
        map_location=map_location,
        weights_only=weights_only,
    )


def load_checkpoint(
    path: str | Path,
    map_location="cpu",
    weights_only: bool = False,
):
    """Load new portable files and legacy checkpoints from the other OS.

    Historical checkpoints could pickle ``WindowsPath`` or ``PosixPath`` in
    metadata. A normal load is attempted first. Only the specific pathlib
    cross-platform failure activates the process-global alias, protected by a
    lock and restored immediately after loading.
    """
    source = Path(path)
    try:
        return _torch_load(source, map_location, weights_only)
    except NotImplementedError as error:
        if "cannot instantiate" not in str(error):
            raise

    with _PATHLIB_UNPICKLE_LOCK:
        if os.name == "nt":
            incompatible_name = "PosixPath"
            replacement = pathlib.WindowsPath
        else:
            incompatible_name = "WindowsPath"
            replacement = pathlib.PosixPath
        original = getattr(pathlib, incompatible_name)
        setattr(pathlib, incompatible_name, replacement)
        try:
            checkpoint = _torch_load(source, map_location, weights_only)
        finally:
            setattr(pathlib, incompatible_name, original)
    return portable_checkpoint_value(checkpoint)


__all__ = ["load_checkpoint", "portable_checkpoint_value", "save_checkpoint"]
