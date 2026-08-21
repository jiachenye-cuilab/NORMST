# NORMST code architecture

This document describes the supported runtime paths and the status of retained
compatibility and experiment code. It intentionally contains no mutable
training-round status.

## Public entry points

`train.py` is the unified dispatcher:

| Arguments | Dataset adapter | Model | Training workflow |
|---|---|---|---|
| `--task visium --model legacy` | `datasets/multislice_masked_visium.py` | `models/geometry_adaptive_normst.py` | `training/visium.py` |
| `--task visium --model pro-normst` | `datasets/pro_normst.py` | `models/pro_normst.py` | `training/pro_normst.py` |
| `--task ae_visium` | `datasets/ae_masked_visium.py` | `models/ae_normst.py` | `training/ae_visium.py` |
| `--task visium_hd` | `datasets/paired_visium_hd.py` | `models/geometry_adaptive_normst.py` | `training/visium_hd.py` |

The route named `legacy` is still a documented and executable standard-Visium
workflow. It is not dead code and must not be confused with the incompatible
historical ProNORMST prototype.

## ProNORMST core

- `datasets/pro_normst.py` owns the frozen split validation, Shared-512 data
  loading, train-only preprocessing, complete hex graph, and resident tensor
  cache.
- `models/pro_normst.py` is the canonical direct-expression model. It imports
  only AE-independent geometry, radial attention, local propagation, and RMS
  normalization primitives from `models/progressive_normst.py`.
- `training/pro_normst_masks.py` defines deterministic ordinary/gap masks and
  their identities.
- `training/pro_normst_engine.py` contains batching, contracted loss, strict
  visible-only IDW, scientific metrics, aggregation, diagnostics, optimizer,
  and the content-addressed IDW cache.
- `training/pro_normst.py` implements the controlled smoke/train/resume/predict
  lifecycle, immutable locks, checkpoints, and one-time test publication.
- `training/pro_normst_matrix.py` and
  `training/pro_normst_acceptance.py` build and validate the formal LODO matrix.
- `scripts/pro_normst_run.sh` is the generic GPU-aware launcher; the remaining
  `scripts/pro_normst_*.py` files are reproducible formal/post-hoc audits.

The old `DeterministicExpressionAutoencoder`, `FrozenLatentEncoder`, and
`ProgressiveNORMST` classes in `models/progressive_normst.py` are not part of
the canonical direct-512 path. They remain only as the frozen historical
prototype/compatibility surface required by the ProNORMST contract; the shared
primitives in the same module are active production dependencies.

## Versioned experiment modules

`round10/`, `round11/`, and `round12/` are isolated, frozen experiment layers.
They explicitly patch or subclass the v9 implementation without changing the
default ProNORMST entry. Their code and tests are retained to reproduce failed
promotion experiments and must not be folded silently into the canonical
model.

`round13/` contains validation-only feasibility and loss-alignment audits. It
does not define a trainable model or a production training route.

## AE pretraining

`pre-train/pre-train.py` is the sole count-aware AE training entry. Supporting
data, model, loss, metric, checkpoint, export, gene-selection, and LODO tools
remain under `pre-train/`. The obsolete duplicate `pre-train/train.py` was
removed; it lacked the fixed-gene, portable-checkpoint, decoder, and latent
regularization behavior of the canonical entry.

## Local-only files

Agent instructions, session handoff/objective documents, the local technical
contract, server-absolute manifests, checkpoints, logs, and one-off launchers
are intentionally excluded from Git. They may describe the local execution
environment but are not portable runtime source.

## Tests

- `tests/`: canonical ProNORMST model/data/mask/lifecycle and formal acceptance.
- `round10/tests/`, `round11/tests/`, `round12/tests/`: frozen experiment
  isolation and numerical behavior.
- `round13/tests/`: validation-audit fitting, aggregation, and decomposition.

Run the repository suite with the required environment:

```bash
conda run -n NORMST python -m unittest discover -s tests -v
conda run -n NORMST python -m unittest discover -s round10/tests -v
conda run -n NORMST python -m unittest discover -s round11/tests -v
conda run -n NORMST python -m unittest discover -s round12/tests -v
conda run -n NORMST python -m unittest discover -s round13/tests -v
```
