#!/usr/bin/env python3
"""Static generator for the "HIA via Euclidean Mapping & Manifold Projection"
figure -- the supplementary diagram showing the cosine/angular case mapped to
Euclidean space and projected back onto the sphere. Headless: no GUI/slider
interaction required.
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


def compute_geometry(p1_sph, p2_sph, z_sph):
    p1 = spherical_to_cartesian(*p1_sph)
    p2 = spherical_to_cartesian(*p2_sph)
    z = spherical_to_cartesian(*z_sph)

    S1 = np.clip(np.dot(p1, z), -1.0, 1.0)
    S2 = np.clip(np.dot(p2, z), -1.0, 1.0)

    R1 = np.sqrt(max(0.0, 2.0 - 2.0 * S1))
    R2 = np.sqrt(max(0.0, 2.0 - 2.0 * S2))

    d12 = np.linalg.norm(p1 - p2)
    if d12 > 1e-5:
        h = (R1**2 - R2**2 + d12**2) / (2 * d12)
        C_int_euc = p1 + h * (p2 - p1) / d12
        p_int = C_int_euc / np.linalg.norm(C_int_euc)
    else:
        C_int_euc = p1
        p_int = p1

    C1_angle = np.arccos(S1)
    C2_angle = np.arccos(S2)
    r_int_angle = np.arccos(np.clip(np.dot(p_int, z), -1.0, 1.0))

    return {
        "p1": p1, "p2": p2, "z": z, "p_int": p_int, "C_int_euc": C_int_euc,
        "cap1": generate_3d_circle(p1, C1_angle),
        "cap2": generate_3d_circle(p2, C2_angle),
        "cap_int": generate_3d_circle(p_int, r_int_angle),
    }


def render(ax, d):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.1, linewidth=0.5)

    ax.plot(*d["cap1"], color="blue", linewidth=2)
    ax.plot(*d["cap2"], color="green", linewidth=2)
    ax.plot(*d["cap_int"], color="red", linestyle="--", linewidth=2)

    p1, p2, z, p_int, c_euc = d["p1"], d["p2"], d["z"], d["p_int"], d["C_int_euc"]
    ax.plot([p1[0]], [p1[1]], [p1[2]], marker="o", color="blue", linestyle="none", markersize=6)
    ax.plot([p2[0]], [p2[1]], [p2[2]], marker="o", color="green", linestyle="none", markersize=6)
    ax.plot([z[0]], [z[1]], [z[2]], marker="o", color="black", linestyle="none", markersize=6)
    ax.plot([p_int[0]], [p_int[1]], [p_int[2]], marker="o", color="red", linestyle="none", markersize=6)
    ax.plot([c_euc[0]], [c_euc[1]], [c_euc[2]], marker="x", color="orange", linestyle="none", markersize=8)

    ax.plot([c_euc[0], p_int[0]], [c_euc[1], p_int[1]], [c_euc[2], p_int[2]], color="orange", linewidth=2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", linestyle=":", linewidth=1.5)

    ax.text(p1[0], p1[1], p1[2] * 1.05, "$p_1$", color="blue", fontsize=12, fontweight="bold")
    ax.text(p2[0], p2[1], p2[2] * 1.05, "$p_2$", color="green", fontsize=12, fontweight="bold")
    ax.text(z[0], z[1], z[2] * 1.05, "$z$", color="black", fontsize=12, fontweight="bold")
    ax.text(p_int[0], p_int[1], p_int[2] * 1.05, "$p_{int}$", color="red", fontsize=12, fontweight="bold")
    ax.text(c_euc[0], c_euc[1], c_euc[2] * 1.05, "$C_{euc}$", color="orange", fontsize=11, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", type=float, nargs=2, default=[np.pi / 4, 3 * np.pi / 8], metavar=("THETA", "PHI"))
    parser.add_argument("--p2", type=float, nargs=2, default=[3 * np.pi / 5, np.pi / 4], metavar=("THETA", "PHI"))
    parser.add_argument("--z", type=float, nargs=2, default=[np.pi / 3, 3 * np.pi / 8], metavar=("THETA", "PHI"))
    parser.add_argument("--elev", type=float, default=20)
    parser.add_argument("--azim", type=float, default=45)
    parser.add_argument("--out", default="hia_euclidean_mapping.svg", help="Output image path (svg/png/pdf)")
    args = parser.parse_args()

    d = compute_geometry(args.p1, args.p2, args.z)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title("HIA via Euclidean Mapping & Manifold Projection", fontsize=14, pad=20)

    render(ax, d)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
