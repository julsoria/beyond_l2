#!/usr/bin/env python3
"""Maintainer tool: packages trained checkpoints into a single archive for
redistribution (e.g. as a Zenodo upload). Run this after training all
architecture/dataset combinations (see the "Train an architecture" section of
the README), then upload the resulting archive to Zenodo by hand and wire the
resulting record ID into scripts/download_checkpoints.py.

Each packaged `final/` directory contains just what run_formal_exp.py needs
to load the checkpoint and run explanations: model weights, configs, and
cached test indices -- prior explanation CSVs, training logs, and optimizer
state are left out.

`model_state.pth` is cleaned before being added to the archive:
 - keys left over from an abandoned, unpublished sparsity experiment
   (classifier.threshold_ema, classifier.cumsum_matrix -- confirmed dead in
   every released checkpoint: threshold_ema is 0.0 everywhere and
   cumsum_matrix only ever fed a logging dict, never the prediction) are
   dropped, since the public CaBRNet ProtoPNetClassifier doesn't define them
   and strict state_dict loading would otherwise crash.
 - an `_orig_mod.` prefix (left behind by saving a torch.compile()-wrapped
   model directly) is stripped if present.

`model_arch.yml`/`dataset.yml`/`training.yml` are taken from this repo's own
configs/<arch>/<dataset>/, not the checkpoint's raw saved copies -- those
already have the correct beyond_l2_ext module paths (the checkpoint's own
saved copies still reference the private cabrnet.archs.tesnet /
cabrnet.core.utils.similarities.MahalanobisLogDistance paths they were
trained against, which don't exist in the public CaBRNet release). If the
checkpoint's actual `classifier.prototypes` count differs from the config's
declared `num_prototypes` (ProtoPool prunes prototypes after training, and
the saved config isn't always updated to match), the bundled model_arch.yml
is patched to the real, loadable count.
"""
import argparse
import hashlib
import io
import re
import zipfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# Dead keys from an abandoned sparsity experiment on a private branch -- never
# part of this paper, provably inert (see module docstring above).
DEAD_KEYS = {"classifier.threshold_ema", "classifier.cumsum_matrix"}

# (arch, dataset) combinations shipped in configs/. Matches the layout
# `make train-<arch>-<dataset>` produces: <source-root>/<arch>_<dataset>/final/.
COMBOS = [
    ("protopnet", "cub200"), ("protopnet", "flowers102"), ("protopnet", "oxford_iiit_pet"),
    ("tesnet", "cub200"), ("tesnet", "flowers102"), ("tesnet", "oxford_iiit_pet"),
    ("pipnet", "cub200"), ("pipnet", "flowers102"), ("pipnet", "oxford_iiit_pet"),
    ("protopool", "cub200"),
    ("protopnet_gaussian_iso", "cub200"), ("protopnet_gaussian_iso", "flowers102"),
    ("protopnet_gaussian_iso", "oxford_iiit_pet"),
]

NUM_PROTOTYPES_RE = re.compile(r"^(\s*num_prototypes:\s*)(\d+)(\s*)$", re.MULTILINE)


def hash_file(path, algo="sha256", chunk_size=1024 * 1024):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clean_state_dict(path):
    """Loads a model_state.pth and strips dead/compile-artifact keys.
    Returns (state_dict, list of dropped key names)."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    dropped = [k for k in DEAD_KEYS if k in state_dict]
    for k in dropped:
        del state_dict[k]

    return state_dict, dropped


def serialize_state_dict(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def patch_num_prototypes(yaml_text, actual_count):
    """Rewrites a `num_prototypes: N` line to match the checkpoint's actual
    prototype count, if it differs. Returns (patched_text, declared_count_or_None)."""
    match = NUM_PROTOTYPES_RE.search(yaml_text)
    if not match or int(match.group(2)) == actual_count:
        return yaml_text, None
    declared = int(match.group(2))
    patched = NUM_PROTOTYPES_RE.sub(rf"\g<1>{actual_count}\g<3>", yaml_text, count=1)
    return patched, declared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("runs"),
                         help="Directory containing <arch>_<dataset>/final/ subfolders (default: runs/)")
    parser.add_argument("--configs-root", type=Path, default=REPO_ROOT / "configs",
                         help="Directory containing <arch>/<dataset>/*.yml (default: this repo's configs/)")
    parser.add_argument("--out", type=Path, default=Path("beyond_l2_checkpoints.zip"))
    args = parser.parse_args()

    missing = []
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for arch, dataset in COMBOS:
            final_dir = args.source_root / f"{arch}_{dataset}" / "final"
            config_dir = args.configs_root / arch / dataset
            if not final_dir.is_dir():
                missing.append(final_dir)
                continue
            prefix = f"{arch}/{dataset}/final"

            state_dict, dropped = load_clean_state_dict(final_dir / "model_state.pth")
            zf.writestr(f"{prefix}/model_state.pth", serialize_state_dict(state_dict))
            print(f"{prefix}/model_state.pth" + (f" (stripped dead keys: {dropped})" if dropped else ""))

            actual_num_prototypes = None
            if "classifier.prototypes" in state_dict:
                actual_num_prototypes = state_dict["classifier.prototypes"].shape[0]

            for name in ("model_arch.yml", "dataset.yml", "training.yml"):
                text = (config_dir / name).read_text()
                if name == "model_arch.yml" and actual_num_prototypes is not None:
                    text, declared = patch_num_prototypes(text, actual_num_prototypes)
                    if declared is not None:
                        print(f"  patched num_prototypes: {declared} -> {actual_num_prototypes} "
                              f"(checkpoint was pruned after this config was saved)")
                zf.writestr(f"{prefix}/{name}", text)
                print(f"{prefix}/{name}")

            idx_src = final_dir / "random_indices.txt"
            if idx_src.is_file():
                zf.write(idx_src, f"{prefix}/random_indices.txt")
                print(f"{prefix}/random_indices.txt")
            else:
                print(f"warning: {idx_src} not found, skipping")

    if missing:
        print("\nMissing checkpoint directories (not included in the archive):")
        for m in missing:
            print(f"  {m}")

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {args.out} ({size_mb:.1f} MB)")
    print(f"SHA256: {hash_file(args.out)}")


if __name__ == "__main__":
    main()
