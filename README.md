# NORMST
NORMST: a neural operator for resolution magnification in spatial transcriptomics

#### Overview

![Overview](Overview.png)

## Geometry-adaptive local-global models

The unified implementation supports standard Visium masked-spot recovery and
official paired Visium HD 16-to-8 micrometre prediction. Standard Visium uses
compact visible point tokens and the native six-neighbour topology; Visium HD
uses the original Cartesian bin grid. Both routes combine a local operator,
Galerkin global operator and interpolation residual baseline.

### Standard Visium

Place each standard Visium slice in a direct child directory such as
`/data/yejiachen/Workdir/Data/visium/151673`. The training entry discovers
valid slices, randomly splits them 4:1:1 at slice level using `--seed`, and
saves the exact split to `output_dir/manifest.json`. A prebuilt split can be
supplied with `--manifest` instead of `--visium-root`.

```bash
python /data/yejiachen/Workdir/NORMST/train.py \
  --task visium \
  --visium-root /data/yejiachen/Workdir/Data/visium \
  --output-dir /data/yejiachen/Workdir/NORMST/save/multislice/seed2027 \
  --seed 2027
```

Every mask from a training slice is used for training; validation and test
slices never contribute masks to training. HVGs and default RMS scales are
fitted only on training slices. Positive `--target-sum` values apply
library-size normalization before `log1p`; use `--target-sum -1` or `0` for
`log1p(raw counts) + RMS`. Add `--no-rms-scale` only to skip RMS. Training
records pooled loss only; validation and test retain complete per-slice and
equal-slice macro metrics.

Test prediction arrays are not saved after training by default. Add
`--save-predictions` to export them immediately. A downloaded run can recreate
the same test predictions later by loading its saved model and preprocessing
contract; the supplied manifest must point to locally accessible slice paths:

```bash
python train.py --task visium \
  --output-dir save/multislice/seed2027 \
  --manifest /local/path/manifest.json \
  --predict-only
```

`--predict-only` loads `best.pt` by default; use `--checkpoint` to select a
different checkpoint.

### Paired Visium HD

```bash
python /data/yejiachen/Workdir/NORMST/train.py \
  --task visium_hd \
  --lr-dir /home2/yejiachen/ST/HBCHD/binned_outputs/square_016um \
  --hr-dir /home2/yejiachen/ST/HBCHD/binned_outputs/square_008um \
  --output-dir /data/yejiachen/Workdir/NORMST/save/HBCHD/geometry_adaptive_seed2027 \
  --seed 2027
```

### Synthetic validation

```bash
python -m unittest discover -s tests
```
