#!/usr/bin/env python3
"""Static generator for the Isotropic Gaussian Hypersphere Intersection
Approximation (HIA) figure. Headless: renders the same geometry used in the
paper and saves it directly, no GUI/slider interaction required.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def compute_geometry(mu1, sig1, mu2, sig2, muk, sigk, z):
    mu1, mu2, muk, z = (np.asarray(v, dtype=float) for v in (mu1, mu2, muk, z))

    R_1 = np.linalg.norm(z - mu1)
    R_2 = np.linalg.norm(z - mu2)

    d_12 = np.linalg.norm(mu1 - mu2)
    if d_12 > 1e-5:
        h = (R_1**2 - R_2**2 + d_12**2) / (2 * d_12)
        R_int = np.sqrt(max(0.0, R_1**2 - h**2))
        C_int = mu1 + h * (mu2 - mu1) / d_12
    else:
        C_int = mu1
        R_int = R_1

    d_k_int = np.linalg.norm(muk - C_int)
    v_k_int = (muk - C_int) / d_k_int if d_k_int > 1e-5 else np.array([0.0, 1.0])

    p_min = C_int + v_k_int * R_int
    p_max = C_int - v_k_int * R_int

    return {
        "mu1": mu1, "sig1": sig1, "R_1": R_1,
        "mu2": mu2, "sig2": sig2, "R_2": R_2,
        "muk": muk, "sigk": sigk,
        "z": z,
        "C_int": C_int, "R_int": R_int,
        "p_min": p_min, "p_max": p_max,
    }


def render(ax, d):
    def gaussian_contours(center, sigma, color, levels=3):
        for i in range(1, levels + 1):
            alpha = max(0.05, 0.4 - (i * 0.08))
            ax.add_patch(patches.Circle(center, i * sigma, color=color, alpha=alpha, zorder=1))

    gaussian_contours(d["mu1"], d["sig1"], "blue")
    gaussian_contours(d["mu2"], d["sig2"], "orange")
    gaussian_contours(d["muk"], d["sigk"], "purple")

    ax.add_patch(patches.Circle(d["mu1"], d["R_1"], fill=False, color="blue", linewidth=2.5, zorder=3))
    ax.add_patch(patches.Circle(d["mu2"], d["R_2"], fill=False, color="orange", linewidth=2.5, zorder=3))
    ax.add_patch(patches.Circle(d["C_int"], d["R_int"], fill=False, color="red", linestyle="--", linewidth=2.5, zorder=4))

    mu1, mu2, muk, z, C_int = d["mu1"], d["mu2"], d["muk"], d["z"], d["C_int"]
    R_1, R_2 = d["R_1"], d["R_2"]
    p_min, p_max = d["p_min"], d["p_max"]

    ax.plot(*mu1, marker="o", color="blue", markersize=8, zorder=5)
    ax.plot(*mu2, marker="o", color="orange", markersize=8, zorder=5)
    ax.plot(*muk, marker="o", color="purple", markersize=8, zorder=5)
    ax.plot(*z, marker="o", color="black", markersize=8, zorder=5)
    ax.plot(*C_int, marker="X", color="red", markersize=8, zorder=5)

    ax.plot([mu1[0], mu1[0] - R_1 / np.sqrt(2)], [mu1[1], mu1[1] + R_1 / np.sqrt(2)], "b--", lw=1)
    ax.plot([mu2[0], mu2[0] + R_2 / np.sqrt(2)], [mu2[1], mu2[1] + R_2 / np.sqrt(2)], "g--", lw=1)
    ax.plot([muk[0], p_min[0]], [muk[1], p_min[1]], color="red", lw=2.5, zorder=2)
    ax.plot([muk[0], p_max[0]], [muk[1], p_max[1]], color="darkred", lw=1.5, linestyle=":", zorder=2)

    offset = 0.4
    ax.text(mu1[0], mu1[1] - offset * 2.5, r"$\mu_1$", color="blue", fontsize=14, ha="center")
    ax.text(mu2[0], mu2[1] - offset * 2.5, r"$\mu_2$", color="orange", fontsize=14, ha="center")
    ax.text(muk[0], muk[1] - offset * 2.5, r"$\mu_k$", color="purple", fontsize=14, ha="center")
    ax.text(z[0], z[1] - offset * 2, r"$z$", color="black", fontsize=14, fontweight="bold", ha="center")
    ax.text(C_int[0], C_int[1] - offset * 2.5, r"$C_{int}$", color="red", fontsize=14, ha="center")
    ax.text(mu1[0] - R_1 / 2 - 0.2, mu1[1] + R_1 / 2 + 0.2, r"$R_1 = \sigma_1 C_1$",
            color="blue", fontsize=12, fontweight="bold")
    ax.text((mu2[0] + R_2 / 2) - 1.0, (mu2[1] + R_2 / 2) - 1.5, r"$R_2 = \sigma_2 C_2$",
            color="orange", fontsize=12, fontweight="bold")


def export_to_tikz(d, filename):
    """Dumps the 2D geometry into a standalone TikZ file (compile with pdflatex/lualatex)."""
    with open(filename, "w") as f:
        f.write("% Compile with pdflatex or lualatex\n")
        f.write("\\begin{tikzpicture}[scale=0.35, line join=round, line cap=round]\n\n")

        f.write("  % Gaussian Density Contours\n")
        for i in range(1, 4):
            alpha = max(0.05, 0.4 - (i * 0.08))
            f.write(f"  \\fill[blue, opacity={alpha:.2f}] ({d['mu1'][0]:.3f},{d['mu1'][1]:.3f}) circle ({i * d['sig1']:.3f});\n")
            f.write(f"  \\fill[orange, opacity={alpha:.2f}] ({d['mu2'][0]:.3f},{d['mu2'][1]:.3f}) circle ({i * d['sig2']:.3f});\n")
            f.write(f"  \\fill[purple, opacity={alpha:.2f}] ({d['muk'][0]:.3f},{d['muk'][1]:.3f}) circle ({i * d['sigk']:.3f});\n")

        f.write("\n  % Bounding Caps & Intersection\n")
        f.write(f"  \\draw[blue, very thick] ({d['mu1'][0]:.3f},{d['mu1'][1]:.3f}) circle ({d['R_1']:.3f});\n")
        f.write(f"  \\draw[orange, very thick] ({d['mu2'][0]:.3f},{d['mu2'][1]:.3f}) circle ({d['R_2']:.3f});\n")
        f.write(f"  \\draw[red, very thick, dashed] ({d['C_int'][0]:.3f},{d['C_int'][1]:.3f}) circle ({d['R_int']:.3f});\n")

        f.write("\n  % Lines\n")
        f.write(f"  \\draw[blue, dashed] ({d['mu1'][0]:.3f},{d['mu1'][1]:.3f}) -- ({d['mu1'][0] - d['R_1']/np.sqrt(2):.3f},{d['mu1'][1] + d['R_1']/np.sqrt(2):.3f});\n")
        f.write(f"  \\draw[orange, dashed] ({d['mu2'][0]:.3f},{d['mu2'][1]:.3f}) -- ({d['mu2'][0] + d['R_2']/np.sqrt(2):.3f},{d['mu2'][1] + d['R_2']/np.sqrt(2):.3f});\n")
        f.write(f"  \\draw[red, ultra thick] ({d['muk'][0]:.3f},{d['muk'][1]:.3f}) -- ({d['p_min'][0]:.3f},{d['p_min'][1]:.3f});\n")
        f.write(f"  \\draw[darkgray, thick, dotted] ({d['muk'][0]:.3f},{d['muk'][1]:.3f}) -- ({d['p_max'][0]:.3f},{d['p_max'][1]:.3f});\n")

        f.write("\n  % Points\n")
        f.write(f"  \\node[circle, fill=blue, inner sep=2pt] at ({d['mu1'][0]:.3f},{d['mu1'][1]:.3f}) {{}};\n")
        f.write(f"  \\node[circle, fill=orange, inner sep=2pt] at ({d['mu2'][0]:.3f},{d['mu2'][1]:.3f}) {{}};\n")
        f.write(f"  \\node[circle, fill=purple, inner sep=2pt] at ({d['muk'][0]:.3f},{d['muk'][1]:.3f}) {{}};\n")
        f.write(f"  \\node[circle, fill=black, inner sep=2pt] at ({d['z'][0]:.3f},{d['z'][1]:.3f}) {{}};\n")
        cx, cy = d["C_int"][0], d["C_int"][1]
        s = 0.3
        f.write(f"  \\draw[red, thick] ({cx-s:.3f},{cy-s:.3f}) -- ({cx+s:.3f},{cy+s:.3f});\n")
        f.write(f"  \\draw[red, thick] ({cx-s:.3f},{cy+s:.3f}) -- ({cx+s:.3f},{cy-s:.3f});\n")

        f.write("\n  % Labels\n")
        offset = 0.4 * 1.5
        f.write(f"  \\node[blue, below] at ({d['mu1'][0]:.3f},{d['mu1'][1] - offset:.3f}) {{$\\mu_1$}};\n")
        f.write(f"  \\node[orange, below] at ({d['mu2'][0]:.3f},{d['mu2'][1] - offset:.3f}) {{$\\mu_2$}};\n")
        f.write(f"  \\node[purple, below] at ({d['muk'][0]:.3f},{d['muk'][1] - offset:.3f}) {{$\\mu_k$}};\n")
        f.write(f"  \\node[black, above] at ({d['z'][0]:.3f},{d['z'][1] + 0.4:.3f}) {{$\\mathbf{{z}}$}};\n")
        f.write(f"  \\node[red, below] at ({cx:.3f},{cy - offset:.3f}) {{$C_{{int}}$}};\n")

        f.write("\\end{tikzpicture}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu1", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
    parser.add_argument("--sig1", type=float, default=1.5)
    parser.add_argument("--mu2", type=float, nargs=2, default=[4.5, 1.5], metavar=("X", "Y"))
    parser.add_argument("--sig2", type=float, default=0.8)
    parser.add_argument("--muk", type=float, nargs=2, default=[1.5, -3.5], metavar=("X", "Y"))
    parser.add_argument("--sigk", type=float, default=1.2)
    parser.add_argument("--z", type=float, nargs=2, default=[2.0, 2.5], metavar=("X", "Y"))
    parser.add_argument("--out", default="hia_gaussian_l2.svg", help="Output image path (svg/png/pdf)")
    parser.add_argument("--tikz", default=None, help="Optional path to also export a standalone TikZ file")
    args = parser.parse_args()

    d = compute_geometry(args.mu1, args.sig1, args.mu2, args.sig2, args.muk, args.sigk, args.z)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_title("HIA in Isotropic Gaussian ($L_2$) Space", fontsize=16, pad=15)
    render(ax, d)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")

    if args.tikz:
        export_to_tikz(d, args.tikz)
        print(f"Saved {args.tikz}")


if __name__ == "__main__":
    main()
