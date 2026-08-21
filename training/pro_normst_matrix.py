"""Create the immutable 36-run ProNORMST formal-matrix manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from training.pro_normst_acceptance import FOLDS, SEEDS, VARIANTS


def build_formal_matrix(round_identity: str, runs_root: str | Path) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", round_identity):
        raise ValueError("round identity is invalid")
    root = Path(runs_root).resolve()
    return {
        "schema": "pro-normst-formal-matrix-v1",
        "round_identity": round_identity,
        "runs": [
            {
                "fold": fold,
                "seed": seed,
                "variant": variant,
                "run_dir": str(root / round_identity / fold / f"seed{seed}" / variant),
            }
            for fold in FOLDS
            for seed in SEEDS
            for variant in VARIANTS
        ],
    }


def write_formal_matrix(
    payload: dict[str, Any],
    output: str | Path,
    *,
    require_complete: bool = False,
) -> None:
    path = Path(output).resolve()
    if require_complete:
        missing = [
            entry["run_dir"]
            for entry in payload["runs"]
            if not (Path(entry["run_dir"]) / "run_status.json").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"formal matrix has incomplete run directories: {missing}")
    content = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"refusing to replace formal matrix: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require every matrix run to contain run_status.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_formal_matrix(args.round_id, args.runs_root)
    write_formal_matrix(payload, args.output, require_complete=args.require_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_formal_matrix", "write_formal_matrix"]
