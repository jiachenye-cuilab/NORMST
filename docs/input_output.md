# Input and output formats

## Prepared inputs

Use `normst prepare-visium` for a classic Space Ranger section, or provide a `dataset.json` with paths relative to that file:

```json
{
  "observations": "observations.npz",
  "image": "he.tiff",
  "features": "features.npy",
  "feature_valid": "feature_valid.npy",
  "bounds_yxyx_um": [0, 0, 6720, 6720]
}
```

| Array | Shape | Meaning |
| --- | --- | --- |
| `counts` in `observations.npz` | spots × genes | Raw nonnegative integer counts; do not log-normalize |
| `centers_yx_um` in `observations.npz` | spots × 2 | Capture centers in physical `(y, x)` micrometers |
| `radius_um` in `observations.npz` | spots, or scalar | Capture radii; 27.5 µm for classic Visium |
| `gene_ids` in `observations.npz` | genes | Unique string IDs in count-column order |
| `barcodes` in `observations.npz` | spots | Optional unique string IDs |
| `features.npy` | 2563 × H × W | Float32 normalized Virchow2 CLS, patch, and RGB features |
| `feature_valid.npy` | H × W | Boolean image-coverage mask; invalid features must be zero |

All input spots are used for training. For held-out prediction, exclude those spots from the input NPZ entirely. Gene selection and image registration must also respect the intended evaluation design.

Bounds are `[y_min, x_min, y_max, x_max]` in micrometers. They must contain every complete, nonoverlapping capture circle, start on the 2 µm integration lattice, and have extents divisible by 32 µm, with at least two 32 µm cells per axis. The image-feature grid has cell centers at the lower bound plus `(index + 0.5) × 8 µm`. Expression contexts use a 32 µm grid.

## Image features

`normst extract` reads an 8-bit RGB TIFF registered to the exact field bounds at 0.5 µm per pixel. Its shape is `(H × 16, W × 16, 3)`. No stain normalization or image augmentation is applied. `prepare-visium` performs affine registration from Space Ranger array/image coordinates and rejects calibration errors of one full-resolution pixel or more.

The frozen Virchow2 encoder processes 224 × 224 pixel tiles (112 µm). Tile CLS tokens and spatial patch tokens are concatenated; register tokens are excluded. Spatial tokens are bilinearly sampled onto a 14 × 14 grid, followed by three pooled RGB channels. Each of the 2,563 channels is standardized over valid image cells only. Invalid cells are set to zero.

Tiles are anchored to the supplied image origin. For an externally prepared crop of an existing feature map, `feature_tile_offset_yx` can specify its offset, in 8 µm cells modulo 14, for lossless CLS storage. The supplied CLS channels must reconstruct exactly within their tiles.

The 7-channel generated features in the CPU example are only a compact software test. Use the full 2,563-channel Virchow2 representation for analysis.

## Training settings

`normst train --config settings.json` overrides named entries in [defaults.json](../normst/defaults.json). The defaults are the final training recipe: 180 epochs, peak learning rate 0.0006, five warmup epochs, cosine decay to 0.00001, AdamW weight decay 0.0001, gradient clipping at 5, and batches of 32 spatial groups. Four 200 µm groupings with half-block offsets cycle across epochs. All three loss terms have weight 1.

The final epoch is saved; there is no best-epoch selection. Changing the epoch count changes the training budget. The cosine horizon is a separate setting. Full-slide feature extraction and fitting are intended for CUDA; memory and temporary disk use scale with field area and gene count.

## Query grids and regions

`normst predict --bin-um 16` changes the output bin size without refitting. The bin size must be a multiple of 8 µm and divide the field extents. This is a readout of the same fitted field, not a change in its learned resolution.

By default, grid output includes bins with valid image coverage; this is **not a tissue segmentation**. To restrict output to tissue, use `--mask tissue_mask.npy`, a boolean mask on the requested output grid. The mask only selects output rows and does not truncate the capture supports used for total correction.

For circular query regions, supply `--queries queries.npz` containing `centers_yx_um` and `radius_um`. All query circles must lie inside the field bounds. Output rows follow query-file order; grid rows follow increasing `(y, x)` order. `counts.npy`, `centers_yx_um.npy`, and `genes.json` share row/column ordering. These are expected counts per region, not log-normalized expression or density.

Counts are computed with the fixed 2 µm midpoint integration rule. Within each observed capture region, the field is rescaled gene by gene to reproduce the measured total; between observed regions, the raw field is retained. A bin crossing a capture boundary combines corrected capture and uncorrected gap contributions. `raw_spot_counts.npy` contains the model's uncorrected integrals over the supplied input spots.

The checkpoint is section-specific and includes its observations and image features. Fit a new checkpoint for a new section. No pretrained expression checkpoint or real dataset is bundled.
