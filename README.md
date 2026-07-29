# Beyond L2

Code accompanying **"Beyond $L_2$: Generalizing Abductive Latent Explanations to Diverse
Prototype-Based Architectures"** (ECML PKDD 2026).

Abductive Latent Explanations (ALE) compute formal, subset-minimal guarantees on the
predictions of prototype-based networks by bounding similarity/activation values in the
latent space. The original ALE framework only handled Euclidean ($L_2$) similarity. This
repository extends it to five paradigms, each targeting a different family of prototype
architectures:

| Paradigm(s) | Targets | Idea |
|---|---|---|
| `top_k` | any architecture | Greedily reports the $k$ most-activated prototypes needed to guarantee the prediction. |
| `triangle`, `hypersphere` | ProtoPNet, ProtoPool (Euclidean) | Original ALE: Euclidean Triangle Inequality / Hypersphere Intersection Approximation (HIA). |
| `cosine`, `cosine_hypersphere` | TesNet (cosine similarity) | Angular Triangle Inequality and Spherical Cap Intersection Approximation. |
| `simplex`, `pip_sparse` | PIP-Net | Bounds derived from the softmax spatial-simplex constraint and PIP-Net's sparse non-negative decision head. |
| `isotropic_gaussian[_hypersphere]`, `isotropic_log[_hypersphere]` | Isotropic Gaussian ProtoPNet | Maps any monotonic similarity function to a universal Euclidean space, reuses the base Euclidean HIA/TI, maps bounds back. |

Focal Similarity (ProtoPool's max-minus-average pooling) is handled inside the base
`triangle`/`hypersphere` paradigms rather than as a separate one -- see
`explain/subset_minimal_axp.py`.

This code is built on top of [CaBRNet](https://github.com/aiser-team/cabrnet), an
open-source library for prototype-based classifiers. TesNet and the Isotropic Gaussian
similarity layer are not (yet) part of the public CaBRNet package, so this repository
ships them as a small plugin package, `beyond_l2_ext`, loaded by CaBRNet's own
configuration-driven plugin mechanism -- no fork of CaBRNet is required.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This pulls in `cabrnet` (PyPI) plus `torch`/`torchvision`/`pandas`/`scikit-learn`/etc.,
and installs `beyond_l2_ext` in editable mode so CaBRNet's config loader can import it by
module path (`beyond_l2_ext.similarities`, `beyond_l2_ext.tesnet.model`, ...).

## Repository layout

```text
beyond_l2_ext/          Plugin package: Isotropic Gaussian similarity layers, TesNet
                        architecture, and a ProtoPNetClassifier subclass exposing
                        covariances -- everything CaBRNet's public package is missing.
explain/                The ALE explanation engine (flat scripts, run from this directory
                        or with it on PYTHONPATH):
  subset_minimal_axp.py   Core engine: base Euclidean Triangle Inequality / HIA, Top-k.
  extended_explainers.py  Cosine, Isotropic Gaussian, and PIP-Net-specific explainers.
  run_formal_exp.py       CLI driver: trains nothing, loads a checkpoint and runs a
                          chosen paradigm over a set of test-set indices.
configs/                Per-architecture, per-dataset {model_arch,dataset,training}.yml,
                        taken from the checkpoints actually used to produce the paper's
                        results (protopnet, tesnet, pipnet, protopool, protopnet_gaussian_iso).
scripts/summarize_results.py   Aggregates `{paradigm}_explanations.csv` files into
                        per-file mean/std tables (Exp Size, Runtime, accuracy).
tests/                  Unit tests for the explainer classes (paradigm validation,
                        explanation-dict shape, ProtoPool weight compression).
```

## Datasets

Oxford Flowers 102 and Oxford-IIIT Pet are fetched automatically by torchvision
(`download: true` in the vendored `dataset.yml` files). CUB-200-2011 must be downloaded
manually (see [the CUB-200-2011 homepage](https://www.vision.caltech.edu/datasets/cub_200_2011/))
and extracted so that `dataset.yml`'s `root` points at it.

## Train an architecture

```bash
cabrnet train --device cuda:0 --seed 42 \
  --config-dir configs/tesnet/flowers102 \
  --output-dir runs/tesnet_flowers102
```

Repeat for `configs/{protopnet,pipnet,protopool,protopnet_gaussian_iso}/<dataset>`. Note:

- **ProtoPool** is only provided for CUB-200: the paper reports that it could not be
  trained successfully on Flowers/Pets with this codebase.
- The vendored `model_arch.yml` files reference the exact backbone/similarity/classifier
  configuration used for the paper's checkpoints. A few reference a local pretrained
  backbone file (`examples/pretrained_conv_extractors/resnet50_inat.pth` from the
  CaBRNet repository) that isn't bundled here; CaBRNet tolerates its absence when you are
  about to load a full checkpoint on top (see "Explain" below) and will simply skip that
  initial weight-loading step, but a *fresh* training run needs either that file or an
  edit to `weights:` (e.g. `IMAGENET1K_V1`).
- If you retrain `protopnet_gaussian_iso`, note the covariance-regularization hook that
  exists in some CaBRNet forks (`similarity_layer.regularization_loss()`) was not active
  for the checkpoint these results were derived from (`loss_coefficients` has no
  `volume` term) -- the vendored `training.yml` reflects that.

Trained checkpoints are not included in this repository (see `checkpoints/README.md`).

## Run an ALE paradigm

`explain/run_formal_exp.py` loads `<config>/final/{model_state.pth,model_arch.yml,...}`
(i.e. `--config` is a training `--output-dir`) and explains a batch of test-set indices
(by default, a cached `random_indices.txt`, or all of them with `--all_indices`).

```bash
# Top-k (any architecture)
python explain/run_formal_exp.py --paradigm top_k --data flowers102 --arch protopnet \
    --config runs/protopnet_flowers102

# Original Euclidean ALE (ProtoPNet baseline, and ProtoPool on CUB-200)
python explain/run_formal_exp.py --paradigm triangle     --data cub200 --arch protopool --config runs/protopool_cub200
python explain/run_formal_exp.py --paradigm hypersphere  --data cub200 --arch protopool --config runs/protopool_cub200

# Cosine / Spherical Cap (TesNet)
python explain/run_formal_exp.py --paradigm cosine            --data flowers102 --arch tesnet --config runs/tesnet_flowers102
python explain/run_formal_exp.py --paradigm cosine_hypersphere --data flowers102 --arch tesnet --config runs/tesnet_flowers102

# Dimensional Projection (PIP-Net)
python explain/run_formal_exp.py --paradigm simplex    --data flowers102 --arch pipnet --config runs/pipnet_flowers102
python explain/run_formal_exp.py --paradigm pip_sparse --data flowers102 --arch pipnet --config runs/pipnet_flowers102

# Isotropic Gaussian ProtoPNet
python explain/run_formal_exp.py --paradigm isotropic_log             --data flowers102 --arch protopnet_gaussian_iso --config runs/protopnet_gaussian_iso_flowers102
python explain/run_formal_exp.py --paradigm isotropic_log_hypersphere --data flowers102 --arch protopnet_gaussian_iso --config runs/protopnet_gaussian_iso_flowers102
```

Add `--all_indices` to explain the whole test set, `--overwrite` to regenerate an
existing CSV, and `--time_bench` to print aggregate timing. `--arch` only affects the
default checkpoint path used when `--config` is omitted -- it does not change how the
model is loaded (that's entirely driven by `<config>/final/model_arch.yml`).

Each run appends to `<config>/final/explanations/<paradigm>_explanations.csv`, with one
row per sample: `Idx, True Label, Pred Label, Correct, Exp Size, ..., Runtime (s)`.
`Exp Size` and `Runtime (s)` are the two quantities reported in the paper's Tables 2-4
(absolute size; relative size is `Exp Size` normalized by $|P|$ for Top-k/PIP-Sparse, or
by $|P|\times|L|$ for the spatial paradigms -- see the paper, Section 5).

```bash
python scripts/summarize_results.py runs/tesnet_flowers102/final/explanations/cosine_explanations.csv
```

## Tests

```bash
pip install -e ".[dev]" pytest  # or just: pip install pytest
PYTHONPATH=. pytest tests/
```

These are unit tests for the explainer classes' construction and control flow (paradigm
validation, explanation-dict shapes, ProtoPool weight compression) -- they don't replace
the end-to-end check of running a paradigm against a real checkpoint above, since the
geometric correctness of the bounds depends on real prototype/feature geometry.

## Citation

```bibtex
@inproceedings{soria2026beyond,
  title     = {Beyond L2: Generalizing Abductive Latent Explanations to Diverse Prototype-Based Architectures},
  author    = {Soria, Jules and Grastien, Alban and Xu-Darme, Romain and Girard-Satabin, Julien and Chihani, Zakaria and Cancila, Daniela},
  booktitle = {ECML PKDD},
  year      = {2026}
}
```

## License

LGPL-2.1-or-later (see `LICENSE`), matching the license of the CaBRNet library this code
depends on.
