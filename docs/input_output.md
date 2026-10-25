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

Space Ranger matrix and position-table barcodes must each be unique. Missing positions for any expression barcode cause an error with counts and examples. `--skip-missing-positions` explicitly excludes these spots and emits a warning; positions marked `in_tissue=0` are excluded as usual. Extra position rows are allowed. Prepared rows follow expression-matrix barcode order, independent of position-table order. Gene columns follow the selected panel, including the order supplied with `--gene-list`.

Bounds are `[y_min, x_min, y_max, x_max]` in micrometers. They must contain every complete, nonoverlapping capture circle, start on the 2 µm integration lattice, and have extents divisible by 32 µm, with at least two 32 µm cells per axis. The image-feature grid has cell centers at the lower bound plus `(index + 0.5) × 8 µm`. Expression contexts use a 32 µm grid.

For custom tangential capture circles, a shared 2 µm integration node belongs to the lowest observation-row index. Training, correction factors and queries exactly matching an observed circle use this same discrete support. Standard, separated Visium capture circles are unaffected. Other circular queries use their own geometric footprint and may overlap one another.

## Image features

`normst extract` reads an 8-bit RGB TIFF registered to the exact field bounds at 0.5 µm per pixel. Its shape is `(H × 16, W × 16, 3)`. No stain normalization or image augmentation is applied. `prepare-visium` performs affine registration from Space Ranger array/image coordinates and rejects calibration errors of one full-resolution pixel or more.

The frozen Virchow2 encoder processes 224 × 224 pixel tiles (112 µm). Tile CLS tokens and spatial patch tokens are concatenated; register tokens are excluded. Spatial tokens are bilinearly sampled onto a 14 × 14 grid, followed by three pooled RGB channels. Each of the 2,563 channels is standardized over valid image cells only. Invalid cells are set to zero.

Tiles are anchored to the supplied image origin. For an externally prepared crop of an existing feature map, `feature_tile_offset_yx` can specify its offset, in 8 µm cells modulo 14, for lossless CLS storage. The supplied CLS channels must reconstruct exactly within their tiles.

### External precomputed features

Use the 2,563-channel CLS/patch/RGB representation for the documented analysis workflow. The loader also accepts other positive channel counts for custom inputs. NORMST validates a float32 `C × H × W` NPY array, exact file byte length, every value's finiteness, bounds/grid alignment, a nonempty boolean coverage mask, and zero values outside that mask. The 2,563-channel model additionally verifies lossless tile-CLS reconstruction when constructed. These checks establish array integrity, not whether an external producer finished its computation or used the intended encoder and normalization.

After the external producer has finished and closed its output files, explicitly confirm completion:

```bash
normst validate-features --data data/section/dataset.json --confirm-complete
```

This leaves array values unchanged and writes a companion `features.json` with shape, bounds, tile offset and SHA-256 identifiers for both arrays. Training requires the completion record and repeats the full array and identifier checks. A file with the expected name or size alone is insufficient. Keep registered inputs unchanged while a run is using them. Existing completed arrays without this record can be registered in the same way after their provenance and completion have been established. Failed `extract` outputs must be completed by rerunning `extract`.

For external arrays in a read-only directory, add `"feature_metadata": "features.json"` to `dataset.json` to keep the completion record and validation lock beside the manifest. This optional path is resolved relative to the manifest; by default the record is beside the feature array. Registration needs read access to the arrays and write access to the metadata directory.

## Training settings

`normst train --config settings.json` recursively merges named entries into [defaults.json](../normst/defaults.json). For example, `{"conditioning": {"neighbors": 4}}` changes that value while retaining the other conditioning settings. Unknown keys, incorrect types, nonfinite values and invalid ranges are rejected before fitting. The defaults are 180 epochs, peak learning rate 0.0006, five warmup epochs, cosine decay to 0.00001, AdamW weight decay 0.0001, gradient clipping at 5, and batches of 32 spatial groups. Four 200 µm groupings with half-block offsets cycle across epochs. All three loss terms have weight 1.

The final epoch is saved; there is no best-epoch selection. Changing the epoch count changes the training budget. The cosine horizon is a separate setting. Full-slide feature extraction and fitting are intended for CUDA; memory and temporary disk use scale with field area and gene count.

## Completion and recovery

**Features.** Extraction writes a temporary array, finishes every tile and channel normalization, and validates the full result and normalization statistics. It atomically replaces the data files and publishes `features.json` with `status: complete` last. A partial or mismatched set is rejected by training. On failure the marker remains incomplete; rerun the same extraction command to restart from the image. Completed features are preserved; use a new feature path to recompute them. An existing unregistered feature file is also preserved for explicit external validation.

**Training.** `resume.pt` is atomically saved before training, every 100 optimizer updates, and at each epoch boundary. Set `--checkpoint-every N` to change the update interval without changing the training schedule. Recovery includes model parameters and context buffers, AdamW state, parameter-group order, epoch and next group, the current permutation, accumulated losses, and Python/NumPy/PyTorch random states. Immutable image arrays are reloaded from the original inputs and checked against their saved content identifiers.

```bash
normst train --data data/section/dataset.json --output outputs/section/model \
  --resume outputs/section/model/resume.pt --device cuda:0
```

Resume uses the saved configuration when `--config` is omitted; if supplied, it must resolve to the same settings, including the epoch endpoint. Inputs, source code, software versions, device and thread settings must match the saved run. Updates after the last saved checkpoint are recomputed. The fixed endpoint still produces `model.pt`; intermediate recovery states are not selected as inference models. Original version-1 inference checkpoints remain readable but lack the optimizer/random state required for training recovery.

`run.json` records the resolved configuration, environment, content identifiers and feature provenance, including the encoder-weight identifier when features were extracted by NORMST. `training.json` records the completed loss history; `status.json` distinguishes running, failed and complete training runs.

**Predictions.** Prediction uses a sibling `.partial-*` directory. It validates all counts and writes row/column metadata and content identifiers before atomically renaming the complete directory to the requested output. Use the final directory with `status.json` and `prediction.json` both marked complete. If interrupted, the final path remains absent; rerun the same command to recompute. Completed directories are preserved. Failed staging directories can be inspected and removed once their process has stopped. An abrupt process kill may leave a `running` marker in staging; this does not indicate completion.

Local writers use non-blocking `filelock` locks with native Windows and POSIX backends, released automatically on process exit. The small lock files may remain in place. Atomic publication requires staging and destination to be on the same filesystem; staging is created beside its destination for this reason. File contents are synced through writable handles, which are closed before replacement. POSIX also syncs directory metadata; Windows skips the unsupported directory sync. File-sync and publication errors propagate to the caller. Sufficient space is needed for staging, the last recovery checkpoint and temporary integration arrays.

## Query grids and regions

`normst predict --bin-um 16` changes the output bin size without refitting. The bin size must be a multiple of 8 µm and divide the field extents. This reads out the same fitted field. Grid spacing specifies the sampling of model estimates; it does not establish a validated biological resolution or constitute direct single-cell measurement.

By default, grid output includes bins containing **any** valid 8 µm image cell; this is **not a tissue segmentation**. Every selected square is integrated over its full area, including uncovered subcells. `coverage_fraction.npy` reports the covered fraction in output-row order. Thus a partially covered 16 µm bin need not equal the sum of the emitted, coverage-filtered 8 µm rows. Additivity holds when the compared outputs cover the same physical support. To restrict output to tissue, use `--mask tissue_mask.npy`, a boolean mask on the requested output grid. The mask selects rows without truncating capture supports used for total correction.

For circular query regions, supply `--queries queries.npz` containing `centers_yx_um` and `radius_um`. All query circles must lie inside the field bounds and include at least one 2 µm integration node. Circular queries are evaluated over their supplied support regardless of image coverage; assess their image support separately. Output rows follow query-file order; grid rows follow increasing `(y, x)` order. `counts.npy`, `centers_yx_um.npy`, and `genes.json` share row/column ordering. These are expected counts per region, not log-normalized expression or density.

Counts are computed with the fixed 2 µm midpoint integration rule. Within each observed capture region, the field is rescaled gene by gene to reproduce the measured total; between observed regions, the raw field is retained. A bin crossing a capture boundary combines corrected capture and uncorrected gap contributions. `raw_spot_counts.npy` contains the model's uncorrected integrals over the supplied input spots.

Observed-total correction enforces conservation by construction. This checks readout consistency; fine-scale accuracy requires independent evidence.

The checkpoint is section-specific and includes its observations and image features. Fit a new checkpoint for a new section. No pretrained expression checkpoint or real dataset is bundled.
