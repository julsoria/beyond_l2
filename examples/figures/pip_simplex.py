#!/usr/bin/env python3
"""Static generator for the "Probability Simplex Slicing in PIP-Net" figure --
how observing one prototype's activation mass bounds a target prototype's
possible range. Headless: no GUI/slider interaction required.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_plane_verts(s, size=1.2):
    """Vertices of the cutting hyperplane Y = s."""
    return [[(size, s, -0.1), (-0.1, s, -0.1), (-0.1, s, size), (size, s, size)]]


def render(ax, s):
    rem = 1.0 - s

    v_simplex = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ax.add_collection3d(Poly3DCollection([v_simplex], facecolors="gray", edgecolors="k",
                                          alpha=0.15, linewidths=1, linestyles="--"))
    ax.add_collection3d(Poly3DCollection(get_plane_verts(s), facecolors="blue",
                                          edgecolors="blue", alpha=0.15, linestyles="--"))

    ax.plot([rem, 0], [s, s], [0, rem], color="blue", linewidth=4, zorder=5)
    ax.plot([rem], [s], [0], marker="o", color="purple", markersize=8, linestyle="none", zorder=6)
    ax.plot([0], [s], [rem], marker="o", color="purple", markersize=8, linestyle="none", zorder=6)

    ax.plot([0, 0], [s, 0], [rem, rem], color="red", linestyle="--", linewidth=1.5)
    ax.plot([0, 0], [0, 0], [0, rem], color="red", linewidth=3, zorder=4)

    ax.text(rem - 0.05, s, 0, "$z_{worst}$", color="purple", fontsize=12, fontweight="bold")
    ax.text(0 - 0.05, s, rem + 0.05, "$z_{best}$", color="purple", fontsize=12, fontweight="bold")
    ax.text(0, -0.1, rem / 2, f"$S_{{tgt}}^{{max}} = {rem:.2f}$", color="red", fontsize=12, fontweight="bold")
    ax.text(0.0, 0.0, -0.1, "$S_{tgt}^{min} = 0.0$", color="red", fontsize=12, fontweight="bold")

    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    ax.set_zlim([0, 1.1])
    ax.set_box_aspect([1, 1, 1])
    ax.plot([0, 1.2], [0, 0], [0, 0], color="gray", linewidth=1.5)
    ax.plot([0, 0], [0, 1.2], [0, 0], color="gray", linewidth=1.5)
    ax.plot([0, 0], [0, 0], [0, 1.2], color="gray", linewidth=1.5)
    ax.set_xlabel("\nOther Prototypes ($p_{other}$)", fontsize=11)
    ax.set_ylabel("\nObserved Prototype ($p_j$)", fontsize=11)
    ax.set_zlabel("\nTarget Prototype ($p_k$)", fontsize=11)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_zticks([0, 0.5, 1.0])
    ax.set_title("Probability Simplex Slicing in PIP-Net\nResidual Mass bounds the Target Prototype",
                 fontsize=15, pad=20)


def export_to_tikz(s, filename, elev, azim):
    """Dumps the geometry into a standalone TikZ file (compile with pdflatex/lualatex)."""
    rem = 1.0 - s
    theta_x = elev + 30
    phi_z = azim + 90

    with open(filename, "w") as f:
        f.write("% Compile with pdflatex or lualatex\n")
        f.write(f"\\tdplotsetmaincoords{{{theta_x:.2f}}}{{{phi_z:.2f}}}\n")
        f.write("\\begin{tikzpicture}[tdplot_main_coords, scale=5.0, line join=round, line cap=round]\n\n")

        f.write("  % Custom Axes\n")
        f.write("  \\draw[gray, thick, ->] (0,0,0) -- (1.2,0,0) node[anchor=west]{Other ($\\mathbf{p}_{other}$)};\n")
        f.write("  \\draw[gray, thick, ->] (0,0,0) -- (0,1.2,0) node[anchor=north west]{Observed ($\\mathbf{p}_{j}$)};\n")
        f.write("  \\draw[gray, thick, ->] (0,0,0) -- (0,0,1.2) node[anchor=south]{Target ($\\mathbf{p}_{k}$)};\n")

        f.write("\n  % Simplex Surface\n")
        f.write("  \\filldraw[fill=gray!20, draw=black, dashed, opacity=0.5] (1,0,0) -- (0,1,0) -- (0,0,1) -- cycle;\n")

        f.write("\n  % Constraint Plane\n")
        f.write(f"  \\filldraw[fill=blue!10, draw=blue, dashed, opacity=0.6] (1.2,{s},-0.1) -- (-0.1,{s},-0.1) -- (-0.1,{s},1.2) -- (1.2,{s},1.2) -- cycle;\n")

        f.write("\n  % Lines\n")
        f.write(f"  \\draw[blue, line width=2pt] ({rem},{s},0) -- (0,{s},{rem});\n")
        f.write(f"  \\draw[red, dashed, thick] (0,{s},{rem}) -- (0,0,{rem});\n")
        f.write(f"  \\draw[red, line width=1.5pt] (0,0,0) -- (0,0,{rem});\n")

        f.write("\n  % Points\n")
        f.write(f"  \\node[circle, fill=purple, inner sep=1.5pt] at ({rem},{s},0) {{}};\n")
        f.write(f"  \\node[circle, fill=purple, inner sep=1.5pt] at (0,{s},{rem}) {{}};\n")

        f.write("\n  % Labels\n")
        f.write(f"  \\node[purple, anchor=east] at ({rem},{s},0) {{$\\mathbf{{z}}_{{worst}}$}};\n")
        f.write(f"  \\node[purple, anchor=west] at (0,{s},{rem}) {{$\\mathbf{{z}}_{{best}}$}};\n")
        f.write(f"  \\node[red, anchor=east] at (0,-0.1,{rem}/2) {{$S_{{tgt}}^{{max}} = {rem:.2f}$}};\n")
        f.write("  \\node[red, anchor=north] at (0,0,-0.1) {\\textbf{$S_{tgt}^{min} = 0.0$}};\n")

        f.write("\\end{tikzpicture}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s", type=float, default=0.6, help="Observed activation mass on the known prototype (0-1)")
    parser.add_argument("--elev", type=float, default=20)
    parser.add_argument("--azim", type=float, default=45)
    parser.add_argument("--out", default="pipnet_simplex.svg", help="Output image path (svg/png/pdf)")
    parser.add_argument("--tikz", default=None, help="Optional path to also export a standalone TikZ file")
    args = parser.parse_args()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)
    render(ax, args.s)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")

    if args.tikz:
        export_to_tikz(args.s, args.tikz, args.elev, args.azim)
        print(f"Saved {args.tikz}")


if __name__ == "__main__":
    main()
