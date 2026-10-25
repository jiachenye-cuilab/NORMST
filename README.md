# NORMST

**Histology-guided super-resolution of spatial transcriptomics.**

NORMST combines raw Visium spot counts with a registered H&E image to learn a continuous spatial expression field. It produces gene-expression estimates on an **8 µm grid**, larger grids, or circular query regions, while preserving measured gene-wise totals within observed capture regions.

## Installation

Tested with Python 3.12 and PyTorch 2.5.1. Use a PyTorch build compatible with your CUDA installation for full-slide analysis.

```bash
git clone https://github.com/jiachenye-cuilab/NORMST.git
cd NORMST
pip install -e ".[histology]"
```

For a pinned environment, use the [reproducibility instructions](docs/reproducibility.md) and [version lock](requirements-lock.txt). If image features are already prepared, the core package can be installed with `pip install -e .`.

## Quick example

This synthetic CPU example uses generated counts and seven-channel features to check loading, four training epochs, checkpoint saving, and prediction. It requires no dataset or pretrained weights and serves as a software demonstration rather than biological validation.

```bash
python examples/synthetic.py --output outputs/demo
```

## Analyze a Visium section

Prepare the Space Ranger expression matrix, tissue positions, and the **full-resolution RGB H&E image used by Space Ranger**. Obtain the [Virchow2 weights](https://huggingface.co/paige-ai/Virchow2) separately under the provider's access and license terms.

```bash
# 1. Prepare counts, coordinates, and the registered image.
normst prepare-visium \
  --matrix data/filtered_feature_bc_matrix.h5 \
  --positions data/spatial/tissue_positions.csv \
  --image data/he_fullres.tiff \
  --output data/section

# 2. Extract frozen image features.
normst extract --data data/section/dataset.json \
  --weights weights/virchow2/model.safetensors --device cuda:0

# 3. Fit the expression model to this section.
normst train --data data/section/dataset.json \
  --output outputs/section/model --device cuda:0

# 4. Export 8 µm expression estimates.
normst predict --checkpoint outputs/section/model/model.pt \
  --output outputs/section/prediction --device cuda:0
```

`prepare-visium` supports classic Visium arrays with 55 µm spots and 100 µm spacing. It uses all supplied in-tissue spots and selects 2,000 genes by variance after log normalization. Use `--gene-list` to supply a fixed gene panel. For prepared inputs, output masks, or query regions, see [Input and output formats](docs/input_output.md).

## Outputs and reproducibility

| File | Contents |
| --- | --- |
| `model/model.pt` | Fitted model, image features, and observed counts needed for inference |
| `model/resume.pt` | Periodic training state for recovery after interruption |
| `prediction/counts.npy` | Nonnegative expected counts, with regions as rows and genes as columns |
| `prediction/centers_yx_um.npy` | Output region centers in physical `(y, x)` coordinates |
| `prediction/genes.json` | Gene IDs in column order |
| `prediction/coverage_fraction.npy` | Fraction of each grid bin covered by valid image features |

Outputs are model estimates of captured counts per region, not direct single-cell measurements. Grid spacing describes the readout, rather than a validated biological resolution. Image coverage selects output bins; a separate mask can restrict them to tissue. Each checkpoint is section-specific. All supplied spots participate in fitting, so held-out spots must be excluded from the input; see the [input and evaluation requirements](docs/input_output.md).

The default training configuration uses seed **2027**, **180 fixed epochs**, a 32-channel image bypass, and frozen Virchow2 features. The expression network has **504,048 trainable parameters** for 2,000 genes. Training combines conditional allocation, coarse-group total, and observed-spot count losses using observed counts as supervision. Settings are in [defaults.json](normst/defaults.json).

Feature extraction and prediction publish completion records after validation. Training writes configuration, software versions and input identifiers, and supports `--resume`. See [artifact completion and recovery](docs/input_output.md#completion-and-recovery).

```bash
python -m unittest discover -s tests
```

Tests cover input integrity, interrupted training, initialization compatibility, gradients, conservation and grid additivity. Conservation is enforced by the readout and does not establish super-resolution accuracy. Use `normst --help` for available commands.

## License and support

Code is released under the [MIT License](LICENSE). Third-party notices are in [NOTICE](NOTICE). Virchow2 weights are not distributed with this repository and have separate terms.

For questions or bug reports, please open a [GitHub issue](https://github.com/jiachenye-cuilab/NORMST/issues).
