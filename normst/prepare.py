"""Prepare classic Visium Space Ranger counts and a registered H&E image."""

import csv
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import affine_transform

from .data import write_json


def read_spaceranger(matrix_path, positions_path):
    import h5py

    with h5py.File(matrix_path, "r") as handle:
        group = handle["matrix"]
        matrix = sparse.csc_matrix((group["data"][:], group["indices"][:], group["indptr"][:]),
                                   shape=tuple(group["shape"][:])).T.tocsr()
        genes = group["features/id"][:].astype(str)
        symbols = group["features/name"][:].astype(str)
        barcodes = group["barcodes"][:].astype(str)
        types = group["features/feature_type"][:].astype(str) if "feature_type" in group["features"] else None
    with Path(positions_path).open(newline="", encoding="utf-8") as stream:
        first = next(csv.reader(stream))
        stream.seek(0)
        names = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        rows = list(csv.DictReader(stream, fieldnames=None if first[0] == "barcode" else names))
    lookup = {row["barcode"]: row for row in rows}
    keep = np.array([i for i, b in enumerate(barcodes) if b in lookup and int(lookup[b]["in_tissue"]) == 1])
    if len(keep) < 3:
        raise ValueError("at least three in-tissue spots are required")
    selected = [lookup[b] for b in barcodes[keep]]
    array = np.array([[int(row[k]) for k in ("array_row", "array_col")] for row in selected])
    pixels = np.array([[float(row[k]) for k in ("pxl_row_in_fullres", "pxl_col_in_fullres")] for row in selected])
    columns = np.flatnonzero(types == "Gene Expression") if types is not None else np.arange(len(genes))
    return matrix[keep][:, columns].astype(np.float64), genes[columns], symbols[columns], barcodes[keep], array, pixels


def select_genes(counts, number):
    library = np.asarray(counts.sum(1)).ravel()
    normalized = sparse.diags(10000 / np.maximum(library, 1)) @ counts
    normalized.data = np.log1p(normalized.data)
    variance = np.asarray(normalized.power(2).mean(0)).ravel() - np.square(np.asarray(normalized.mean(0)).ravel())
    eligible = np.flatnonzero(np.asarray((counts > 0).sum(0)).ravel() >= 3)
    if number < 1 or len(eligible) < number:
        raise ValueError(f"requested {number} genes but only {len(eligible)} have counts in at least three spots")
    return np.sort(eligible[np.argsort(-variance[eligible], kind="stable")[:number]])


def prepare_visium(matrix_path, positions_path, image_path, output, *, n_genes=2000, gene_list=None):
    import tifffile
    from PIL import Image

    counts, genes, symbols, barcodes, array, pixels = read_spaceranger(matrix_path, positions_path)
    if gene_list is None:
        columns = select_genes(counts, n_genes)
    else:
        requested = [line.strip() for line in Path(gene_list).read_text(encoding="utf-8").splitlines() if line.strip()]
        lookup = {gene: i for i, gene in enumerate(genes)}
        if not requested or len(set(requested)) != len(requested) or any(gene not in lookup for gene in requested):
            raise ValueError("gene list must contain unique gene IDs present in the input matrix")
        columns = np.asarray([lookup[gene] for gene in requested])
    # Classic Visium uses a 100 um hexagonal lattice and 55 um capture diameter.
    centers = 112 + np.column_stack((array[:, 0] * 50 * np.sqrt(3), array[:, 1] * 50))
    design = np.c_[centers, np.ones(len(centers))]
    affine, _, rank, _ = np.linalg.lstsq(design, pixels, rcond=None)
    residual = np.linalg.norm(design @ affine - pixels, axis=1)
    if rank != 3 or residual.max() >= 1:
        raise ValueError("array-to-image calibration must have full rank and residual below one full-resolution pixel")
    bounds = np.r_[np.floor((centers - 27.5).min(0) / 32) * 32,
                   np.ceil((centers + 27.5).max(0) / 32) * 32]
    height, width = (int(value) for value in np.rint((bounds[2:] - bounds[:2]) * 2))
    try:
        original = tifffile.memmap(image_path, mode="r")
    except (ValueError, tifffile.TiffFileError):
        if str(image_path).lower().endswith((".tif", ".tiff", ".btf")):
            original = tifffile.imread(image_path, maxworkers=2)
        else:
            with Image.open(image_path) as image:
                original = np.asarray(image.convert("RGB"))
    if original.ndim != 3 or original.shape[2] != 3 or original.dtype != np.uint8:
        raise ValueError("provide an 8-bit RGB image in the Space Ranger full-resolution coordinate frame")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    registered = tifffile.memmap(output / "he.tiff", shape=(height, width, 3), dtype=np.uint8, photometric="rgb")
    matrix, offset = .5 * affine[:2].T, np.r_[bounds[:2] + .25, 1] @ affine
    for channel in range(3):
        affine_transform(original[:, :, channel], matrix, offset, output_shape=(height, width),
            output=registered[:, :, channel], order=1, mode="nearest", prefilter=False)
    registered.flush()
    # Match image coverage at 2 um centers, then require every subcell of an 8 um cell.
    y = bounds[0] + (np.arange(height // 4) + .5) * 2
    x = bounds[1] + (np.arange(width // 4) + .5) * 2
    sy = y[:, None] * affine[0, 0] + x[None] * affine[1, 0] + affine[2, 0]
    sx = y[:, None] * affine[0, 1] + x[None] * affine[1, 1] + affine[2, 1]
    valid = (sy >= 0) & (sy <= original.shape[0] - 1) & (sx >= 0) & (sx <= original.shape[1] - 1)
    valid = valid.reshape(height // 16, 4, width // 16, 4).all(axis=(1, 3))
    if not valid.any():
        raise ValueError("image has no valid coverage of the physical field")
    np.save(output / "feature_valid.npy", valid, allow_pickle=False)
    np.savez_compressed(output / "observations.npz", counts=counts[:, columns].toarray().astype(np.float32),
        centers_yx_um=centers, radius_um=np.full(len(centers), 27.5), gene_ids=genes[columns], barcodes=barcodes)
    write_json(output / "genes.json", dict(gene_ids=genes[columns].tolist(), gene_symbols=symbols[columns].tolist()))
    write_json(output / "registration.json", dict(image_from_um_yx=affine.tolist(),
        max_residual_pixels=float(residual.max()), pixel_um=.5))
    write_json(output / "dataset.json", dict(observations="observations.npz", image="he.tiff",
        features="features.npy", feature_valid="feature_valid.npy", bounds_yxyx_um=bounds.tolist()))
    return output / "dataset.json"
