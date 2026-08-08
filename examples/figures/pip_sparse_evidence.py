#!/usr/bin/env python3
"""Static generator for the "PIP-Net Sparse Evidence Algorithm" toy figure --
a 3-step walkthrough of how competitor prototypes are progressively revealed
until the predicted class's lead is either verified or refuted.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="pipnet_sparse_evidence.png", help="Output image path (svg/png/pdf)")
    args = parser.parse_args()

    classes = ["Bird (Predicted)", "Plane (Competitor)"]
    prototypes = ["P0 (Feathers)", "P1 (Wings)", "P2 (Engine)"]

    # Weights (K x P)
    W = np.array([
        [2.5, 0.0, 0.0],  # Bird
        [0.0, 2.0, 3.0],  # Plane
    ])

    # Observed global max activations (A)
    A = np.array([0.8, 0.6, 0.1])

    steps = [[0], [0, 1], [0, 1, 2]]
    step_titles = [
        "Step 0: Init\n$E = \\{P_0\\}$",
        "Step 1: Reveal Wings\n$E = \\{P_0, P_1\\}$",
        "Step 2: Reveal Engine\n$E = \\{P_0, P_1, P_2\\}$",
    ]

    bird_score = W[0, 0] * A[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    fig.suptitle("PIP-Net Sparse Evidence Algorithm: Bounding the Worst-Case", fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes):
        E = steps[i]
        unobserved = [j for j in range(3) if j not in E]

        plane_observed = sum(W[1, j] * A[j] for j in E)
        plane_unobserved_max = sum(W[1, j] * 1.0 for j in unobserved)

        ax.bar(0, bird_score, color="#1f77b4", edgecolor="black", width=0.6,
               label="Predicted Class Score" if i == 0 else "")
        ax.text(0, bird_score / 2, f"Exact Score:\n{bird_score:.2f}", ha="center", va="center",
                color="white", fontweight="bold")

        ax.bar(1, plane_observed, color="#d62728", edgecolor="black", width=0.6,
               label="Competitor Observed" if i == 0 else "")
        if plane_observed > 0:
            ax.text(1, plane_observed / 2, f"Observed:\n{plane_observed:.2f}", ha="center", va="center",
                    color="white", fontweight="bold")

        if plane_unobserved_max > 0:
            ax.bar(1, plane_unobserved_max, bottom=plane_observed, color="#ff9896", edgecolor="red",
                   hatch="//", width=0.6,
                   label="Unobserved Worst-Case\n(Max Activation = 1.0)" if i == 0 else "")
            ax.text(1, plane_observed + plane_unobserved_max / 2, f"Unobserved Max:\n+{plane_unobserved_max:.2f}",
                    ha="center", va="center", color="black", fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Bird\n(Predicted)", "Plane\n(Competitor)"], fontsize=12)
        ax.set_title(step_titles[i], fontsize=14, pad=10)
        ax.axhline(bird_score, color="green", linestyle="--", linewidth=2,
                   label="Threshold to Beat" if i == 0 else "")

        total_competitor_bound = plane_observed + plane_unobserved_max
        verified = total_competitor_bound < bird_score
        ax.text(0.5, 5.2, "VERIFIED" if verified else "UNVERIFIED",
                ha="center", color="green" if verified else "red", fontsize=14, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="green" if verified else "red", pad=5))

        ax.set_ylim(0, 5.8)
        if i == 0:
            ax.set_ylabel("Logit Score", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
