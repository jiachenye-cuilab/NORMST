"""Select a frozen train-only gene set for matched pretraining runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data import (
    assert_output_outside_sources,
    load_pretrain_data,
    parse_manifest,
    verify_source_contract,
)
from gene_selection import select_donor_aware_hvgs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--n-top-genes", type=int, default=1000)
    parser.add_argument("--min-train-detection-fraction", type=float, default=0.1)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    manifest = args.manifest.resolve()
    entries = parse_manifest(manifest, args.count_file)
    output = assert_output_outside_sources(args.output_dir, entries)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    slice_metadata = payload.get("_meta", {}).get("slice_metadata", {})
    data = load_pretrain_data(manifest, count_file=args.count_file)
    selected, audit, table = select_donor_aware_hvgs(
        data,
        args.n_top_genes,
        args.min_train_detection_fraction,
        slice_metadata,
    )
    output.mkdir(parents=True, exist_ok=True)
    genes = output / "genes.txt"
    genes.write_text("\n".join(selected.genes.tolist()) + "\n", encoding="utf-8")
    table.to_csv(output / "gene_selection.csv", index=False)
    config = {
        **audit,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "genes": str(genes),
        "genes_sha256": _sha256(genes),
        "data_writes": "output_dir only; source count matrices remain read-only",
        "source_contract": data.source_contract,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    verify_source_contract(data.source_contract)
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
