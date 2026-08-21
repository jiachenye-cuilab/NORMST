"""Compare two complete ProNORMST formal-acceptance summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _resolve_acceptance(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    return resolved / "formal_artifacts" / "formal_acceptance.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    if value.get("schema") != "pro-normst-formal-acceptance-v2":
        raise ValueError(f"unexpected formal acceptance schema: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact(path: Path) -> dict[str, Any]:
    artifact = _resolve_acceptance(path)
    payload = _read_json(artifact)
    effects = {}
    for name, value in payload.get("effects", {}).items():
        if not isinstance(value, dict):
            continue
        effects[name] = {
            "overall_mean": value.get("overall_mean"),
            "positive_folds": value.get("positive_folds"),
            "fold_effect": value.get("fold_effect"),
        }
    invariance = payload.get("round_invariance", {})
    return {
        "artifact": str(artifact),
        "sha256": _sha256(artifact),
        "accepted": payload.get("accepted"),
        "checks": payload.get("checks"),
        "effects": effects,
        "round_invariance": {
            "passed": invariance.get("passed"),
            "mismatch_count": invariance.get("mismatch_count"),
            "max_abs_error": invariance.get("max_abs_error"),
            "max_relative_error": invariance.get("max_relative_error"),
        },
    }


def _deltas(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    effect_delta = {}
    common = sorted(set(baseline["effects"]) & set(current["effects"]))
    for name in common:
        left = baseline["effects"][name].get("overall_mean")
        right = current["effects"][name].get("overall_mean")
        effect_delta[name] = (
            float(right) - float(left)
            if isinstance(left, (int, float)) and isinstance(right, (int, float))
            else None
        )
    return {
        "check_changes": {
            key: {
                "baseline": baseline.get("checks", {}).get(key),
                "current": current.get("checks", {}).get(key),
            }
            for key in sorted(
                set(baseline.get("checks", {})) | set(current.get("checks", {}))
            )
        },
        "effect_overall_mean_current_minus_baseline": effect_delta,
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = _compact(args.baseline)
    current = _compact(args.current)
    _write_exclusive(
        args.output,
        {
            "schema": "pro-normst-formal-acceptance-comparison-v1",
            "baseline": baseline,
            "current": current,
            "delta": _deltas(baseline, current),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
