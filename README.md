# NORMST

**Histology-guided super-resolution of spatial transcriptomics.**

NORMST combines measured Visium spot counts with a registered H&E image to estimate spatial gene expression on a finer grid. It learns from the measured capture regions and preserves their gene-wise totals in the final readout. The default output spacing is **8 µm**; these are model estimates, not direct single-cell measurements.

## Installation

Tested with Python 3.12 and PyTorch 2.5.1. Use a PyTorch build compatible with your CUDA installation for full-slide analysis.

```bash
git clone https://github.com/jiachenye-cuilab/NORMST.git
cd NORMST
pip install -e ".[histology]"
```

The tested dependency versions are listed in [requirements.txt](requirements.txt). If image features are already prepared, the core package can be installed with `pip install -e .`.

## Quick example

Run a small CPU example with generated counts and image features—no data or pretrained weights are required:

```bash
python examples/synthetic.py --output outputs/demo
```

This exercises input loading, four training epochs, checkpoint saving, and prediction. It is a workflow check, not a biological example.

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
| `prediction/counts.npy` | Nonnegative expected counts, with regions as rows and genes as columns |
| `prediction/centers_yx_um.npy` | Output region centers in physical `(y, x)` coordinates |
| `prediction/genes.json` | Gene IDs in column order |

The default recipe uses seed **2027**, **180 fixed epochs**, a 32-channel image bypass, and frozen Virchow2 features. The expression network has **504,048 trainable parameters** for 2,000 genes. Training combines conditional allocation, coarse-group total, and observed-spot count losses; it requires no fine-resolution expression labels. Settings are in [defaults.json](normst/defaults.json).

```bash
python -m unittest discover -s tests
```

These checks cover training gradients, checkpoint reloads, and observed-count conservation. Use `normst --help` or a subcommand's `--help` for available options.

## License and support

Code is released under the [MIT License](LICENSE). Third-party notices are in [NOTICE](NOTICE). Virchow2 weights are not distributed with this repository and have separate terms.

For questions or bug reports, please open a [GitHub issue](https://github.com/jiachenye-cuilab/NORMST/issues).
