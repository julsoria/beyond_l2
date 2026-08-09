# Beyond L2

Code for **"Beyond $L_2$: Generalizing Abductive Latent Explanations to Diverse
Prototype-Based Architectures"** (ECML PKDD 2026).

Abductive Latent Explanations (ALE) compute formal, subset-minimal guarantees on the
predictions of prototype-based networks by bounding similarity/activation values in the
latent space. The original ALE framework only handled Euclidean ($L_2$) similarity; this
repository extends it to five paradigms, each targeting a different family of prototype
architectures:

| Paradigm(s) (`--paradigm`) | `--arch` | Idea |
|---|---|---|
| `top_k` | any | Greedily reports the $k$ most-activated prototypes needed to guarantee the prediction. |
| `triangle`, `hypersphere` | `protopnet`, `protopool` | Original ALE: Euclidean Triangle Inequality / Hypersphere Intersection Approximation (HIA). |
| `cosine`, `cosine_hypersphere` | `tesnet` | Angular Triangle Inequality and Spherical Cap Intersection Approximation. |
| `simplex`, `pip_sparse` | `pipnet` | Bounds from the softmax spatial-simplex constraint and PIP-Net's sparse non-negative decision head. |
| `isotropic_gaussian[_hypersphere]`, `isotropic_log[_hypersphere]` | `protopnet_gaussian_iso` | Maps any monotonic similarity to a universal Euclidean space, reuses the base HIA/TI, maps bounds back. |

Focal Similarity (ProtoPool's max-minus-average pooling) is handled inside the base
`triangle`/`hypersphere` paradigms, not as a separate one -- see `explain/subset_minimal_axp.py`.

This code is built on top of [CaBRNet](https://github.com/aiser-team/cabrnet), an
open-source library for prototype-based classifiers. TesNet and the Isotropic Gaussian
similarity layer are not (yet) part of the public CaBRNet package, so this repository
ships them as a small plugin package, `beyond_l2_ext`, loaded through CaBRNet's own
configuration-driven `module`/`name` mechanism -- no fork of CaBRNet is required.

## Install

```bash
git clone https://github.com/julsoria/beyond_l2
cd beyond_l2
python3 -m venv .venv && source .venv/bin/activate
./scripts/setup_cabrnet.sh
```

`setup_cabrnet.sh` clones CaBRNet straight from its public GitHub repository, pinned to
the exact commit behind the published `1.2` release, and installs it plus `beyond_l2_ext`
in editable mode. It requires Python 3.10-3.12 (CaBRNet 1.2's own constraint) and refuses
to run outside that range rather than silently resolving an older CaBRNet version.

## Examples: paper figures

`examples/figures/` reproduces the paper's conceptual diagrams (Hypersphere Intersection
Approximation in Gaussian/cosine space, the PIP-Net simplex bound) as plain headless
scripts -- no CaBRNet, checkpoint, or dataset needed, just `pip install -e ".[figures]"`:

```bash
python examples/figures/gaussian_hia.py --out hia_gaussian.svg
```

See `examples/figures/README.md` for the full list and what each one draws.

## Repository layout

```text
beyond_l2/
├── beyond_l2_ext/                  CaBRNet plugin package
│   ├── similarities.py               Isotropic Gaussian similarity layers
│   ├── protopnet_gaussian.py          ProtoPNetClassifier subclass exposing covariances
│   └── tesnet/                        TesNet architecture (model.py, decision.py)
├── explain/                        The ALE explanation engine (flat scripts)
│   ├── subset_minimal_axp.py         Core engine: Euclidean Triangle Inequality / HIA, Top-k
│   ├── extended_explainers.py        Cosine, Isotropic Gaussian, PIP-Net explainers
│   ├── extra_utils.py                Small shared helpers
│   └── run_formal_exp.py             CLI driver: loads a checkpoint, runs one paradigm
├── configs/                        {dataset,model_arch,training}.yml per checkpoint used
│   ├── protopnet/{cub200,flowers102,oxford_iiit_pet}/
│   ├── tesnet/{cub200,flowers102,oxford_iiit_pet}/
│   ├── pipnet/{cub200,flowers102,oxford_iiit_pet}/
│   ├── protopool/cub200/             only dataset ProtoPool trained on (see Datasets)
│   └── protopnet_gaussian_iso/{cub200,flowers102,oxford_iiit_pet}/
├── examples/figures/                Headless scripts reproducing the paper's diagrams
├── scripts/
│   ├── setup_cabrnet.sh              Clones + pins CaBRNet from GitHub, installs everything
│   └── summarize_results.py          Aggregates *_explanations.csv into mean/std tables
├── checkpoints/README.md           Checkpoints aren't committed here (see Datasets)
└── pyproject.toml, Makefile, environment.yml, LICENSE, CITATION.cff
```

## Datasets

Oxford Flowers-102 and Oxford-IIIT Pet are fetched automatically by torchvision
(`download: true` in the vendored `dataset.yml` files). CUB-200-2011 must be downloaded
manually (see [the CUB-200-2011 homepage](https://www.vision.caltech.edu/datasets/cub_200_2011/))
and extracted so that `dataset.yml`'s `root` points at it.

**ProtoPool** is only provided for CUB-200: the paper reports that it could not be
trained successfully on Flowers/Pets with this codebase.

Trained checkpoints (~100-300MB each) are not committed to this repository -- see
`checkpoints/README.md`.

## Train an architecture

```bash
cabrnet train --device cuda:0 --seed 42 \
  --config-dir configs/tesnet/flowers102 \
  --output-dir runs/tesnet_flowers102
```

Repeat for any `configs/<arch>/<dataset>` combination (see the layout above). Two notes:

- A few `model_arch.yml` files reference a local pretrained backbone file
  (`examples/pretrained_conv_extractors/resnet50_inat.pth` from the CaBRNet repo) that
  isn't bundled here. CaBRNet tolerates its absence when a full checkpoint is loaded on
  top (see "Run an ALE paradigm" below), but a *fresh* training run needs either that file
  or an edited `weights:` value (e.g. `IMAGENET1K_V1`).
- `protopnet_gaussian_iso`'s covariance-regularization hook
  (`similarity_layer.regularization_loss()`) exists but was inactive for the checkpoints
  behind the paper's results -- the vendored `training.yml` has no `volume` loss term,
  reflecting that.

## Run an ALE paradigm

`run_formal_exp.py` loads `<config>/final/{model_state.pth,model_arch.yml,...}` (i.e.
`--config` is a training `--output-dir`) and explains a batch of test-set indices (by
default a cached `random_indices.txt`, or all of them with `--all_indices`):

```bash
python explain/run_formal_exp.py --paradigm cosine --data flowers102 --arch tesnet \
    --config runs/tesnet_flowers102
```

Swap `--paradigm`/`--arch`/`--config` per the table at the top of this README to run any
other combination. Add `--overwrite` to regenerate an existing CSV, `--time_bench` to
print aggregate timing.

Each run appends to `<config>/final/explanations/<paradigm>_explanations.csv`, one row
per sample (`Idx, True Label, Pred Label, Correct, Exp Size, ..., Runtime (s)`) --
`Exp Size` and `Runtime (s)` are what the paper's Tables 2-4 report. Summarize a CSV with:

```bash
python scripts/summarize_results.py runs/tesnet_flowers102/final/explanations/cosine_explanations.csv
```

## Citation

```bibtex
@inproceedings{soria2026beyond,
  title={Beyond L2: Generalizing Abductive Latent Explanations to Diverse Prototype-Based Architectures},
  author={Soria, Jules and Grastien, Alban and Xu-Darme, Romain and Girard-Satabin, Julien and Chihani, Zakaria and Cancila, Daniela},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year={2026},
  organization={Springer}
}
```

## License

LGPL-2.1-or-later (see `LICENSE`), matching the license of the CaBRNet library this code
depends on.
