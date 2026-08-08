#!/usr/bin/env python3
"""Static generator for the Spherical Cap Intersection Approximation figure
(cosine-similarity HIA on the unit sphere). Headless: renders the same
geometry used in the paper and saves it directly, no GUI/slider interaction
required.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3D projection)


def spherical_to_cartesian(theta, phi):
    return np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def generate_3d_circle(center, angular_radius, num_points=100):
    """3D points for a circle on the surface of the unit sphere."""
    v1 = np.cross(center, np.array([0, 0, 1]))
    if np.linalg.norm(v1) < 1e-5:
        v1 = np.cross(center, np.array([0, 1, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(center, v1)

    t = np.linspace(0, 2 * np.pi, num_points)
    return (
        np.cos(angular_radius) * center[:, np.newaxis]
        + np.sin(angular_radius) * np.cos(t) * v1[:, np.newaxis]
        + np.sin(angular_radius) * np.sin(t) * v2[:, np.newaxis]
    )


def compute_geometry(p1_sph, p2_sph, z_sph, pk_sph):
    p1 = spherical_to_cartesian(*p1_sph)
    p2 = spherical_to_cartesian(*p2_sph)
    z = spherical_to_cartesian(*z_sph)
    pk = spherical_to_cartesian(*pk_sph)

    C1 = np.arccos(np.clip(np.dot(p1, z), -1.0, 1.0))
    C2 = np.arccos(np.clip(np.dot(p2, z), -1.0, 1.0))
    d_angle = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))

    if d_angle < 1e-5:
        p_int = p1
    else:
        num = np.cos(C2) - np.cos(C1) * np.cos(d_angle)
        den = np.cos(C1) * np.sin(d_angle)
        alpha = np.arctan2(num, den)
        p_int = (np.sin(d_angle - alpha) * p1 + np.sin(alpha) * p2) / np.sin(d_angle)
        p_int = p_int / np.linalg.norm(p_int)

    r_int = np.arccos(np.clip(np.dot(p_int, z), -1.0, 1.0))

    t_geo = np.linspace(0, 1, 50)
    geodesic = np.array([pk * (1 - t) + p_int * t for t in t_geo])
    geodesic = geodesic / np.linalg.norm(geodesic, axis=1)[:, np.newaxis]

    return {
        "p1": p1, "p2": p2, "z": z, "pk": pk,
        "C1": C1, "C2": C2, "p_int": p_int, "r_int": r_int,
        "cap1": generate_3d_circle(p1, C1),
        "cap2": generate_3d_circle(p2, C2),
        "cap_int": generate_3d_circle(p_int, r_int),
        "geodesic": geodesic, "geo_mid": geodesic[25],
    }


def render(ax, d):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.1, linewidth=0.5)

    ax.plot(*d["cap1"], color="blue", linewidth=2, label=r"$d_\angle(z, p_1) \leq C_1$")
    ax.plot(*d["cap2"], color="orange", linewidth=2, label=r"$d_\angle(z, p_2) \leq C_2$")
    ax.plot(*d["cap_int"], color="red", linestyle="--", linewidth=2, label=r"Approximated Cap ($r_{int}$)")
    ax.plot(*d["geodesic"].T, color="purple", linestyle=":", linewidth=1.5)

    for name, color in (("p1", "blue"), ("p2", "orange"), ("z", "black"), ("p_int", "red"), ("pk", "purple")):
        p = d[name]
        ax.plot([p[0]], [p[1]], [p[2]], marker="o", color=color, linestyle="none", markersize=6)
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=color, linestyle="--", alpha=0.5)

    labels = {
        "p1": ("$p_1$", "blue"), "p2": ("$p_2$", "orange"), "z": ("$z$", "black"),
        "p_int": ("$p_{int}$", "red"), "pk": ("$p_k$", "purple"),
    }
    for key, (text, color) in labels.items():
        p = d[key]
        ax.text(p[0], p[1], p[2] * 1.05, text, color=color, fontsize=12, fontweight="bold")
    gm = d["geo_mid"]
    ax.text(gm[0], gm[1], gm[2] * 1.05, r"$d_\angle(p_k, p_{int})$", color="purple", fontsize=11)


def export_to_tikz(d, filename, elev, azim):
    theta_x = elev + 45
    phi_z = azim + 90

    with open(filename, "w") as f:
        f.write("% Compile with pdflatex or lualatex\n")
        f.write(f"\\tdplotsetmaincoords{{{theta_x:.2f}}}{{{phi_z:.2f}}}\n")
        f.write("\\begin{tikzpicture}[tdplot_main_coords, scale=4, line join=round, line cap=round]\n\n")

        f.write("  % Sphere Wireframe\n")
        f.write("  \\tikzset{wireframe/.style={gray, very thin, opacity=0.15}}\n")
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        for lat in v:
            xs, ys, zs = np.cos(u) * np.sin(lat), np.sin(u) * np.sin(lat), np.ones_like(u) * np.cos(lat)
            coords = " -- ".join(f"({x:.3f},{y:.3f},{z:.3f})" for x, y, z in zip(xs, ys, zs))
            f.write(f"  \\draw[wireframe] {coords};\n")
        for lon in u:
            xs, ys, zs = np.cos(lon) * np.sin(v), np.sin(lon) * np.sin(v), np.ones_like(v) * np.cos(v)
            coords = " -- ".join(f"({x:.3f},{y:.3f},{z:.3f})" for x, y, z in zip(xs, ys, zs))
            f.write(f"  \\draw[wireframe] {coords};\n")

        f.write("\n  % Caps, Geodesics, and Radii\n")
        lines = [
            (d["cap1"], "blue", "thick"), (d["cap2"], "orange", "thick"),
            (d["cap_int"], "red", "thick, dashed"), (d["geodesic"].T, "purple", "thick, dotted"),
        ]
        for name, color in (("p1", "blue"), ("p2", "orange"), ("z", "black"), ("p_int", "red"), ("pk", "purple")):
            p = d[name]
            lines.append((np.array([[0, p[0]], [0, p[1]], [0, p[2]]]), color, "dashed, opacity=0.5"))
        for coords_3d, color, style in lines:
            xs, ys, zs = coords_3d
            if len(xs) > 0:
                coords = " -- ".join(f"({x:.3f},{y:.3f},{z:.3f})" for x, y, z in zip(xs, ys, zs))
                f.write(f"  \\draw[{color}, {style}] {coords};\n")

        f.write("\n  % Points\n")
        points = [("blue", d["p1"]), ("orange", d["p2"]), ("black", d["z"]), ("red", d["p_int"]), ("purple", d["pk"])]
        for color, pt in points:
            f.write(f"  \\node[circle, fill={color}, inner sep=1.2pt] at ({pt[0]:.3f},{pt[1]:.3f},{pt[2]:.3f}) {{}};\n")

        f.write("\n  % Labels\n")
        labels = [
            ("blue", d["p1"], "$\\mathbf{p}_1$"), ("orange", d["p2"], "$\\mathbf{p}_2$"),
            ("black", d["z"], "$\\mathbf{z}$"), ("red", d["p_int"], "$\\mathbf{p}_{int}$"),
            ("purple", d["pk"], "$\\mathbf{p}_k$"),
            ("purple", d["geo_mid"], "$d_\\angle(\\mathbf{p}_k, \\mathbf{p}_{int})$"),
        ]
        for color, pt, text in labels:
            f.write(f"  \\node[color={color}, anchor=south west] at ({pt[0]:.3f},{pt[1]:.3f},{pt[2]*1.05:.3f}) {{{text}}};\n")

        f.write("\\end{tikzpicture}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", type=float, nargs=2, default=[np.pi / 4, 3 * np.pi / 8], metavar=("THETA", "PHI"))
    parser.add_argument("--p2", type=float, nargs=2, default=[3 * np.pi / 5, np.pi / 4], metavar=("THETA", "PHI"))
    parser.add_argument("--z", type=float, nargs=2, default=[np.pi / 3, 3 * np.pi / 8], metavar=("THETA", "PHI"))
    parser.add_argument("--pk", type=float, nargs=2, default=[2 * np.pi / 3, 3 * np.pi / 5], metavar=("THETA", "PHI"))
    parser.add_argument("--elev", type=float, default=20)
    parser.add_argument("--azim", type=float, default=45)
    parser.add_argument("--out", default="hia_cosine_manifold.svg", help="Output image path (svg/png/pdf)")
    parser.add_argument("--tikz", default=None, help="Optional path to also export a standalone TikZ file")
    args = parser.parse_args()

    d = compute_geometry(args.p1, args.p2, args.z, args.pk)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title("Hypersphere Intersection Approximation", fontsize=14, pad=20)

    render(ax, d)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05), fontsize=10)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")

    if args.tikz:
        export_to_tikz(d, args.tikz, args.elev, args.azim)
        print(f"Saved {args.tikz}")


if __name__ == "__main__":
    main()
