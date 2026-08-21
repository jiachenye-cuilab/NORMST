#!/usr/bin/env python
"""Compare Round12 runtime events with the frozen Round9 pilot log."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _events(path: Path) -> list[dict[str, Any]]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and str(value.get("event", "")).startswith(
            "runtime_"
        ):
            values.append(value)
    if not values:
        raise ValueError(f"runtime log contains no events: {path}")
    return values


def _summary(path: Path) -> dict[str, Any]:
    events = _events(path)
    epochs = [value for value in events if value.get("event") == "runtime_epoch"]
    if not epochs:
        raise ValueError(f"runtime log contains no epochs: {path}")
    warm = epochs[1:] if len(epochs) > 1 else epochs

    def mean(records: list[dict[str, Any]], key: str) -> float:
        return statistics.fmean(float(value[key]) for value in records)

    setup = next(
        (value for value in events if value.get("event") == "runtime_setup"),
        None,
    )
    test = next(
        (value for value in reversed(events) if value.get("event") == "runtime_test"),
        None,
    )
    return {
        "log": str(path.resolve()),
        "epochs": len(epochs),
        "setup_seconds": float(setup["seconds"]) if setup is not None else None,
        "first_epoch": {
            key: float(epochs[0][key])
            for key in ("train_seconds", "validation_seconds", "total_seconds")
        },
        "warm_epoch_mean": {
            key: mean(warm, key)
            for key in ("train_seconds", "validation_seconds", "total_seconds")
        },
        "all_epoch_mean": {
            key: mean(epochs, key)
            for key in ("train_seconds", "validation_seconds", "total_seconds")
        },
        "test_seconds": float(test["seconds"]) if test is not None else None,
        "final_idw_cache": (
            test.get("persistent_idw_cache")
            if test is not None
            else epochs[-1].get("persistent_idw_cache")
        ),
    }


def _comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for section in ("first_epoch", "warm_epoch_mean", "all_epoch_mean"):
        output[section] = {}
        for key in ("train_seconds", "validation_seconds", "total_seconds"):
            left = float(baseline[section][key])
            right = float(candidate[section][key])
            output[section][key] = {
                "baseline": left,
                "candidate": right,
                "candidate_minus_baseline": right - left,
                "relative_change": (right - left) / left,
                "speedup": left / right,
            }
    return output


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    baseline = _summary(args.baseline_log.resolve())
    candidate = _summary(args.candidate_log.resolve())
    payload = {
        "schema": "pro-normst-round12-runtime-comparison-v1",
        "baseline": baseline,
        "candidate": candidate,
        "comparison": _comparison(baseline, candidate),
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
