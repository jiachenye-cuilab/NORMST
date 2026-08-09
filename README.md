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

```bash
python /data/yejiachen/Workdir/NORMST/train_geometry_adaptive_normst.py \
  --task visium \
  --data-dir /data/yejiachen/Workdir/Data/151673 \
  --output-dir /data/yejiachen/Workdir/NORMST/save/151673/geometry_adaptive_seed2027 \
  --n-genes 1000 \
  --seed 2027 \
  --device cuda
```

### Paired Visium HD

```bash
python /data/yejiachen/Workdir/NORMST/train_geometry_adaptive_normst.py \
  --task visium_hd \
  --lr-dir /home2/yejiachen/ST/HBCHD/binned_outputs/square_016um \
  --hr-dir /home2/yejiachen/ST/HBCHD/binned_outputs/square_008um \
  --output-dir /data/yejiachen/Workdir/NORMST/save/HBCHD/geometry_adaptive_seed2027 \
  --n-genes 1000 \
  --scale 2 \
  --patch-size-lr 16 \
  --patches-per-epoch 64 \
  --width 128 \
  --num-heads 8 \
  --operator-layers 4 \
  --batch-size 1 \
  --epochs 150 \
  --seed 2027 \
  --device cuda
```

### Multi-slice standard Visium

Place each standard Visium slice in a direct child directory such as
`/data/yejiachen/Workdir/Data/visium/151673`. The training entry discovers
valid slices, randomly splits them 4:1:1 at slice level using `--seed`, and
saves the exact split to `output_dir/manifest.json`. A prebuilt split can still
be supplied with `--manifest` instead of `--visium-root`.

```bash
python /data/yejiachen/Workdir/NORMST/train_multislice_visium.py \
  --visium-root /data/yejiachen/Workdir/Data/visium \
  --output-dir /data/yejiachen/Workdir/NORMST/save/multislice/seed2027 \
  --n-genes 1000 \
  --masks-per-slice 64 \
  --mask-target-fraction 0.25 \
  --seed 2027 \
  --device cuda
```

Every mask from a training slice is used for training; validation and test
slices never contribute masks to training. HVGs and RMS scales are fitted only
on training slices. The route uses batch size one because each random mask has
a different compact native graph, and reports pooled element-level metrics,
per-slice metrics, and an equal-slice macro average.

### Synthetic validation

```bash
python -m unittest \
  tests/test_geometry_adaptive_normst.py \
  tests/test_geometry_adaptive_training.py \
  tests/test_multislice_visium.py
```
