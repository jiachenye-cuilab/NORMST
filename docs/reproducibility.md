# Reproducible installation and checks

The regression suite was run on Linux x86_64, Python **3.12.9**, NumPy **2.3.5**, SciPy **1.18.0**, and PyTorch **2.5.1**, using **CPU with two threads**. The installed PyTorch build is `py3.12_cuda12.1_cudnn9.1.0_0`; torchvision is `0.20.1 / py312_cu121`. GPU training and Virchow2 inference with real weights are separate validation steps.

## Pinned installation

From the repository root, create a separate environment and install the specified binary builds before the pinned Python packages:

```bash
conda create -n normst-repro -c pytorch -c nvidia -c conda-forge \
  python=3.12.9=h9e4cc4f_1_cpython \
  pytorch=2.5.1=py3.12_cuda12.1_cudnn9.1.0_0 \
  torchvision=0.20.1=py312_cu121 pytorch-cuda=12.1=ha16c6d3_6 pip
conda activate normst-repro
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
```

[requirements-lock.txt](../requirements-lock.txt) pins the installed Python dependencies and their active Python dependency closure. [requirements.txt](../requirements.txt) lists the direct runtime packages. The lock does not describe the host driver, operating system, or every native Conda library; preserve the binary-build choices above and retain each run's environment record. A fresh online installation from these instructions has not been tested as part of the local regression suite. Other PyTorch builds can be used, but numerical equivalence across platforms and library builds must be checked separately.

## Run the checks

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s tests -v
```

The tests cover complete feature publication, corruption and interruption rejection, barcode uniqueness/alignment, missing-position opt-in, nested configuration overrides, training recovery, initialization and inference-checkpoint compatibility, tangential capture supports, partial image coverage, gradients, conservation and grid additivity. Histology input tests use local test doubles and generated images; they do not fetch pretrained weights.

For the recovery test, a seven-epoch CPU run with one spatial group per batch is compared against a run interrupted after update 2 within the first epoch. This exercises both warmup and cosine learning rates. Final model tensors, AdamW state, training history, predictions and saved random states must match exactly (`rtol=0`, `atol=0`). Additional checks interrupt at an epoch's last update and after its completion. The original scientific tests retain their tolerances: shared-image gradients `rtol=3e-5, atol=1e-7`; floating-point region readouts and grid additivity up to `rtol=2e-6, atol=1e-7`. These checks establish software consistency for the tested environment, not bitwise GPU reproducibility.

Seeded initialization deliberately retains reserved random draws and the staged layer expansions in `_seeded_layers`. Changing them can change the initialized parameters even if the final shapes stay the same. Tests check parameter and RNG fingerprints for seeds 17 and 2027 with 7 and 2,563 image channels, strict state loading and output equality. The version-1 inference checkpoint schema remains supported.

A separate two-epoch recovery check exercises the complete 2,000-gene, 2,563-channel expression network with compact CLS storage and requires exact equality of all saved model tensors.

## Run records and remaining validation

Training stores resolved settings, software and source identifiers, input hashes, feature provenance and the loss history alongside its checkpoints. Extraction additionally records image and encoder-weight hashes. Prediction records checkpoint, query/mask and output identifiers. Supply the same immutable inputs and software when [resuming training](input_output.md#completion-and-recovery).

Real-data benchmark reproduction, model ablations, biological validation and full-slide CUDA recovery/inference still require the corresponding data, licensed Virchow2 weights, evaluation splits and reference results. They are outside the synthetic regression suite. In particular, the observed-total correction enforces conservation by construction; passing that check is not evidence of fine-scale prediction accuracy.
