"""Atomic local artifacts and content identifiers for reproducible runs."""

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

from filelock import FileLock, Timeout
import torch


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def sync_directory(path):
    # Windows does not support opening directories for fsync through os.
    # File contents are still synced before publication on every platform.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_file(path):
    # Windows FlushFileBuffers requires a writable handle. Do not truncate.
    with Path(path).open("r+b") as stream:
        os.fsync(stream.fileno())


@contextmanager
def atomic_file(path):
    """Replace one file only after its writer closes and syncs successfully."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    try:
        yield Path(temporary)
        _sync_file(temporary)
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path, value):
    with atomic_file(path) as temporary:
        torch.save(value, temporary)


@contextmanager
def writer_lock(path):
    """Acquire an exclusive native file lock without waiting for another writer."""
    lock = FileLock(path)
    try:
        lock.acquire(blocking=False)
    except Timeout as error:
        raise RuntimeError(f"another writer holds {Path(path).name}") from error
    try:
        yield
    finally:
        lock.release()


@contextmanager
def staged_directory(destination):
    """Publish a complete output directory; retain failed staging for inspection."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(destination.parent / f".{destination.name}.lock"):
        if destination.exists():
            raise FileExistsError(destination)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
        write_json(staging / "status.json", dict(status="running"))
        try:
            yield staging
            write_json(staging / "status.json", dict(status="complete"))
            for path in staging.iterdir():
                if path.is_file():
                    _sync_file(path)
            os.rename(staging, destination)
            sync_directory(destination.parent)
        except BaseException:
            if staging.exists():
                write_json(staging / "status.json", dict(status="failed"))
            raise


def environment(device):
    packages = {}
    for name in ("numpy", "scipy", "torch", "filelock", "timm", "safetensors", "Pillow", "tifffile", "h5py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.suffix in (".py", ".json"):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    device = torch.device(device)
    return dict(python=platform.python_version(), system=platform.system(), machine=platform.machine(),
        packages=packages, code_sha256=digest.hexdigest(), device=str(device),
        torch_build=str(torch.__version__), torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        accelerator=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        cpu_threads=torch.get_num_threads(), tf32=False, cudnn_deterministic=True)
