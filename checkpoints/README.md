# Trained checkpoints

Trained model checkpoints (13 combinations: `{protopnet,tesnet,pipnet,protopnet_gaussian_iso}`
x `{cub200,flowers102,oxford_iiit_pet}`, plus `protopool` x `cub200`) are hosted on Zenodo
rather than committed to this git repository (~1.2GB total as a single archive).

## Download

```bash
python scripts/download_checkpoints.py
```

This fetches the archive from Zenodo, verifies its checksum, and extracts each checkpoint
into `runs/<arch>_<dataset>/final/` -- the same layout `cabrnet train` produces, so
`run_formal_exp.py --config runs/<arch>_<dataset>` works identically whether you trained a
checkpoint yourself or downloaded it. Use `--only tesnet_flowers102 protopnet_cub200 ...`
to extract just specific combos (the archive itself is downloaded in full either way).

Alternatively, train each architecture yourself following the "Train an architecture"
section of the top-level `README.md` -- every config used to produce the paper's results
is included under `configs/`.

## Publishing (maintainer notes)

`scripts/package_checkpoints.py` builds the archive from local training runs (see that
script's docstring for exactly what it includes and what it cleans up). After building it:

1. Go to [zenodo.org/uploads/new](https://zenodo.org/uploads/new).
2. Upload the generated `beyond_l2_checkpoints.zip`.
3. Fill in the metadata:
   - **Title:** Beyond L2: Trained Model Checkpoints
   - **Upload type:** Dataset
   - **Description:** Trained model checkpoints accompanying the code at
     [github.com/julsoria/beyond_l2](https://github.com/julsoria/beyond_l2) and the paper
     "Beyond L2: Generalizing Abductive Latent Explanations to Diverse Prototype-Based
     Architectures" (ECML PKDD 2026, [arXiv:2608.16773](https://arxiv.org/abs/2608.16773)).
     Contains 13 architecture/dataset combinations (ProtoPNet, TesNet, PIP-Net, ProtoPool,
     and an Isotropic Gaussian ProtoPNet variant, over Flowers-102, Oxford-IIIT Pet, and
     CUB-200-2011). See the repository's README for how to load and use these checkpoints.
   - **License:** CC-BY-4.0 (or your preference -- distinct from the code's LGPL-2.1)
   - **Keywords:** prototype-based networks, explainable AI, abductive reasoning, ProtoPNet
   - **Related works:** link the GitHub repo and the arXiv paper as "is supplement to"
4. Publish, then take the resulting record ID (the number in the record's URL, e.g.
   `zenodo.org/records/1234567` -> `1234567`) and set it as `ZENODO_RECORD_ID` at the top of
   `scripts/download_checkpoints.py`.
