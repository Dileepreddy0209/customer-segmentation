"""
Compare segmentations produced by different algorithms on the same customers.

Reads the rfm_labels.csv written by segmentation_pipeline.py and produces:
  * a side-by-side scatter (log Recency vs log Monetary) per algorithm
  * a pairwise Adjusted Rand Index (ARI) table showing how much the algorithms agree

Usage: python plot_segments.py --labels results_k4/rfm_labels.csv --out results_k4
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

MODELS = ["K-Means", "Agglomerative (Ward)", "GMM"]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # colour-blind-checked order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results_k4/rfm_labels.csv")
    ap.add_argument("--out", default="results_k4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.labels, index_col=0)

    # pairwise agreement between all algorithms (DBSCAN noise kept as its own label)
    algos = [c for c in df.columns if c not in ("Recency", "Frequency", "Monetary")]
    ari = pd.DataFrame(index=algos, columns=algos, dtype=float)
    for i in algos:
        for j in algos:
            ari.loc[i, j] = adjusted_rand_score(df[i], df[j])
    ari = ari.round(3)
    print("Pairwise Adjusted Rand Index (1 = identical, 0 = chance):")
    print(ari.to_string())
    ari.to_csv(os.path.join(a.out, "pairwise_ari.csv"))

    # scatter: segments ordered by median spend so colours mean the same thing in every panel
    k = df[MODELS[0]].nunique()
    fig, axs = plt.subplots(1, len(MODELS), figsize=(12, 4), sharex=True, sharey=True)
    for ax, m in zip(axs, MODELS):
        order = df.groupby(m)["Monetary"].median().sort_values().index
        rank = df[m].map({s: r for r, s in enumerate(order)})
        for r in range(k):
            sel = rank == r
            ax.scatter(df.Recency[sel] + 1, df.Monetary[sel].clip(lower=1), s=5, alpha=.55,
                       lw=0, color=COLOURS[r % len(COLOURS)])
        ax.set(xscale="log", yscale="log", title=f"{m} (k = {k})", xlabel="Recency (days, log scale)")
        ax.grid(alpha=.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("Monetary (£, log scale)")
    names = [f"Segment {chr(65 + r)}" for r in range(k)]
    names[0] += " (lowest value)"; names[-1] += " (highest value)"
    fig.legend([plt.Line2D([], [], marker="o", ls="", color=COLOURS[r % len(COLOURS)], markersize=7) for r in range(k)],
               names, loc="lower center", ncol=k, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = os.path.join(a.out, "segments_comparison.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
