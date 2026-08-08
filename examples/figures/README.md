# Paper figures

Standalone scripts that reproduce the conceptual diagrams from the paper. Each
one only needs `numpy` + `matplotlib` -- no CaBRNet, no trained checkpoint, no
dataset. Run any of them directly:

```bash
pip install matplotlib numpy   # or: pip install -e ".[figures]"
python examples/figures/gaussian_hia.py --out hia_gaussian.svg
```

| Script | Figure |
|---|---|
| `gaussian_hia.py` | Hypersphere Intersection Approximation in isotropic Gaussian ($L_2$) space. |
| `cosine_hia.py` | Spherical Cap Intersection Approximation (cosine similarity, TesNet). |
| `cosine_hia_euclidean.py` | The cosine case mapped to Euclidean space and projected back onto the sphere. |
| `pip_simplex.py` | Probability-simplex slicing that bounds a target prototype in PIP-Net. |
| `pip_sparse_evidence.py` | Toy walkthrough of PIP-Net's sparse-evidence verification loop. |

Every script exposes its inputs (prototype/query positions, the observed
activation mass, etc.) as `--flags` with defaults matching the paper's
figures -- run `--help` on any of them to see what can be varied. The
geometry-based ones (`gaussian_hia.py`, `cosine_hia.py`, `pip_simplex.py`)
also accept `--tikz FILE.tex` to export the exact same figure as a standalone
TikZ file, compilable with pdflatex/lualatex.

These were originally interactive matplotlib scripts (drag sliders, then press
a key to save); they've been converted to plain, headless, single-shot scripts
here so they run the same way in a terminal, a notebook, or CI.
