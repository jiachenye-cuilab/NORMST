"""Build and audit pair-aware leave-one-donor-out DLPFC manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SLICE_METADATA = {
    "151507": {"donor": "Br5292", "pair": "Br5292_anterior", "position": "anterior", "serial": "a"},
    "151508": {"donor": "Br5292", "pair": "Br5292_anterior", "position": "anterior", "serial": "b"},
    "151509": {"donor": "Br5292", "pair": "Br5292_posterior", "position": "posterior", "serial": "a"},
    "151510": {"donor": "Br5292", "pair": "Br5292_posterior", "position": "posterior", "serial": "b"},
    "151669": {"donor": "Br5595", "pair": "Br5595_anterior", "position": "anterior", "serial": "a"},
    "151670": {"donor": "Br5595", "pair": "Br5595_anterior", "position": "anterior", "serial": "b"},
    "151671": {"donor": "Br5595", "pair": "Br5595_posterior", "position": "posterior", "serial": "a"},
    "151672": {"donor": "Br5595", "pair": "Br5595_posterior", "position": "posterior", "serial": "b"},
    "151673": {"donor": "Br8100", "pair": "Br8100_anterior", "position": "anterior", "serial": "a"},
    "151674": {"donor": "Br8100", "pair": "Br8100_anterior", "position": "anterior", "serial": "b"},
    "151675": {"donor": "Br8100", "pair": "Br8100_posterior", "position": "posterior", "serial": "a"},
    "151676": {"donor": "Br8100", "pair": "Br8100_posterior", "position": "posterior", "serial": "b"},
}


LODO_SPLITS = {
    "lodo_d1": {
        "held_out_donor": "Br5292",
        "train": ["151671", "151672", "151673", "151674"],
        "val": ["151669", "151670", "151675", "151676"],
        "test": ["151507", "151508", "151509", "151510"],
    },
    "lodo_d2": {
        "held_out_donor": "Br5595",
        "train": ["151509", "151510", "151673", "151674"],
        "val": ["151507", "151508", "151675", "151676"],
        "test": ["151669", "151670", "151671", "151672"],
    },
    "lodo_d3": {
        "held_out_donor": "Br8100",
        "train": ["151507", "151508", "151671", "151672"],
        "val": ["151509", "151510", "151669", "151670"],
        "test": ["151673", "151674", "151675", "151676"],
    },
}


def _role_names(group) -> list[str]:
    if not isinstance(group, dict):
        raise ValueError("LODO manifest role groups must be objects")
    return [str(name) for name in group]


def validate_lodo_payload(payload: dict) -> dict:
    """Fail closed on donor leakage, split pairs, or fold drift."""
    if not isinstance(payload, dict) or not isinstance(payload.get("_meta"), dict):
        raise ValueError("LODO manifest requires a top-level _meta object")
    meta = payload["_meta"]
    fold = str(meta.get("fold", "")).casefold()
    if fold not in LODO_SPLITS:
        raise ValueError(f"unknown LODO fold: {fold}")
    expected = LODO_SPLITS[fold]
    if meta.get("protocol") != "pair_grouped_lodo":
        raise ValueError("LODO protocol must be pair_grouped_lodo")
    if meta.get("held_out_donor") != expected["held_out_donor"]:
        raise ValueError("held_out_donor does not match the frozen fold")

    roles = {role: _role_names(payload.get(role)) for role in ("train", "val", "test")}
    flattened = [name for names in roles.values() for name in names]
    if len(flattened) != len(set(flattened)):
        raise ValueError("a slice occurs in more than one manifest role")
    if set(flattened) != set(SLICE_METADATA):
        missing = sorted(set(SLICE_METADATA) - set(flattened))
        extra = sorted(set(flattened) - set(SLICE_METADATA))
        raise ValueError(f"LODO slice coverage mismatch; missing={missing}, extra={extra}")
    for role, expected_names in expected.items():
        if role == "held_out_donor":
            continue
        if roles[role] != expected_names:
            raise ValueError(f"{fold}/{role} differs from the frozen split")

    role_by_slice = {
        name: role for role, names in roles.items() for name in names
    }
    pair_roles: dict[str, set[str]] = {}
    for name, metadata in SLICE_METADATA.items():
        pair_roles.setdefault(metadata["pair"], set()).add(role_by_slice[name])
    split_pairs = {pair: values for pair, values in pair_roles.items() if len(values) != 1}
    if split_pairs:
        raise ValueError(f"serial pairs cross manifest roles: {split_pairs}")

    held_out = expected["held_out_donor"]
    test_donors = {SLICE_METADATA[name]["donor"] for name in roles["test"]}
    if test_donors != {held_out}:
        raise ValueError("test must contain exactly one complete held-out donor")
    held_out_slices = {
        name for name, values in SLICE_METADATA.items() if values["donor"] == held_out
    }
    if set(roles["test"]) != held_out_slices:
        raise ValueError("all four held-out donor slices must be test-only")

    remaining = {"Br5292", "Br5595", "Br8100"} - {held_out}
    for role in ("train", "val"):
        donor_counts = {
            donor: sum(
                SLICE_METADATA[name]["donor"] == donor for name in roles[role]
            )
            for donor in remaining
        }
        if set(donor_counts.values()) != {2}:
            raise ValueError(
                f"{role} must contain one complete pair from each remaining donor"
            )
    return {
        "fold": fold,
        "held_out_donor": held_out,
        "role_slice_counts": {role: len(names) for role, names in roles.items()},
        "pair_roles": {pair: next(iter(values)) for pair, values in pair_roles.items()},
        "valid": True,
    }


def build_lodo_payload(
    fold: str,
    visium_root: str | Path,
    output_directory: str | Path,
    relative_paths: bool = True,
) -> dict:
    fold = fold.casefold()
    if fold not in LODO_SPLITS:
        raise ValueError(f"unknown LODO fold: {fold}")
    root = Path(visium_root).resolve()
    output = Path(output_directory).resolve()
    split = LODO_SPLITS[fold]

    def entry(name: str) -> dict:
        path = root / name
        rendered = os.path.relpath(path, output) if relative_paths else str(path)
        return {
            "path": rendered.replace("\\", "/"),
            **SLICE_METADATA[name],
        }

    payload = {
        "_meta": {
            "protocol": "pair_grouped_lodo",
            "fold": fold,
            "held_out_donor": split["held_out_donor"],
            "test_split_unit": "donor",
            "train_validation_split_unit": "serial_pair",
            "description": (
                "One donor is test-only. Each remaining donor contributes one "
                "complete serial pair to train and its other pair to validation."
            ),
            "slice_metadata": SLICE_METADATA,
        },
    }
    for role in ("train", "val", "test"):
        payload[role] = {name: entry(name) for name in split[role]}
    validate_lodo_payload(payload)
    return payload


def audit_manifest(path: str | Path, check_paths: bool = False) -> dict:
    manifest = Path(path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    result = validate_lodo_payload(payload)
    if check_paths:
        missing = []
        for role in ("train", "val", "test"):
            for name, item in payload[role].items():
                raw_path = Path(item["path"] if isinstance(item, dict) else item)
                resolved = raw_path if raw_path.is_absolute() else manifest.parent / raw_path
                if not resolved.resolve().is_dir():
                    missing.append(name)
        if missing:
            raise FileNotFoundError(f"slice directories are missing: {missing}")
        result["data_paths_exist"] = True
    result["manifest"] = str(manifest)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit existing LODO manifests")
    audit.add_argument("manifests", type=Path, nargs="+")
    audit.add_argument("--check-paths", action="store_true")

    generate = subparsers.add_parser(
        "generate", help="generate portable manifests for a local data root"
    )
    generate.add_argument("--visium-root", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--absolute-paths", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "audit":
        results = [audit_manifest(path, args.check_paths) for path in args.manifests]
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for fold in LODO_SPLITS:
        destination = output / f"{fold}.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing manifest: {destination}")
        payload = build_lodo_payload(
            fold,
            args.visium_root,
            output,
            relative_paths=not args.absolute_paths,
        )
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(audit_manifest(destination, check_paths=True))
    print(json.dumps(written, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
