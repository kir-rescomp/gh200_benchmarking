#!/usr/bin/env python
"""
Generate the two-panel poster figure from benchmark outputs:
    Panel A  -- biology:  UMAP coloured by Leiden cluster
    Panel B  -- scaling:  per-stage wall-clock time + peak HBM annotation

Run after gh200_rsc_benchmark.py has produced its outputs.
"""

import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ── load results ─────────────────────────────────────────────────────────────
adata    = sc.read_h5ad("gh200_results.h5ad")
timings  = pd.read_csv("gh200_timings.csv")
summary  = json.load(open("gh200_summary.json"))

# ── figure scaffold ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7), dpi=200)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)

# ── Panel A: biology — UMAP coloured by Leiden cluster ───────────────────────
axA = fig.add_subplot(gs[0])
umap = adata.obsm["X_umap"]
clusters = adata.obs["leiden"].astype("category")
codes = clusters.cat.codes.to_numpy()
n_clusters = len(clusters.cat.categories)

cmap = plt.cm.get_cmap("tab20", n_clusters)
axA.scatter(umap[:, 0], umap[:, 1], c=codes, cmap=cmap,
            s=0.3, alpha=0.6, linewidths=0, rasterized=True)

# label each cluster at its median position
for i, cat in enumerate(clusters.cat.categories):
    mask = codes == i
    cx, cy = np.median(umap[mask, 0]), np.median(umap[mask, 1])
    axA.text(cx, cy, str(cat), fontsize=8, fontweight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.7))

axA.set_title(f"A.  {summary['n_cells']:,} mouse brain cells — "
              f"{n_clusters} Leiden clusters",
              fontsize=13, fontweight="bold", loc="left")
axA.set_xlabel("UMAP 1"); axA.set_ylabel("UMAP 2")
axA.set_xticks([]); axA.set_yticks([])
for s in axA.spines.values():
    s.set_visible(False)

# ── Panel B: scaling — per-stage timing ──────────────────────────────────────
axB = fig.add_subplot(gs[1])
t = timings.sort_values("seconds", ascending=True)
bars = axB.barh(t["stage"], t["seconds"], color="#76b900")  # NVIDIA green

for bar, (_, row) in zip(bars, t.iterrows()):
    axB.text(bar.get_width() + max(t["seconds"]) * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f"{row['seconds']:.1f}s", va="center", fontsize=8)

axB.set_title("B.  Per-stage wall-clock (single GH200)",
              fontsize=13, fontweight="bold", loc="left")
axB.set_xlabel("seconds")
axB.grid(axis="x", alpha=0.25)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)

# headline annotation box
txt = (f"Total pipeline: {summary['total_pipeline_seconds']}s\n"
       f"Peak HBM3e: {summary['peak_hbm_gb']} / 96 GB\n"
       f"Entire dataset resident on-GPU")
axB.text(0.98, 0.04, txt, transform=axB.transAxes, ha="right", va="bottom",
         fontsize=9, bbox=dict(boxstyle="round,pad=0.5",
                               fc="#f0f7e6", ec="#76b900"))

fig.suptitle("End-to-end single-cell clustering on NVIDIA GH200",
             fontsize=15, fontweight="bold", x=0.07, ha="left")

fig.savefig("gh200_poster_figure.png", bbox_inches="tight", dpi=200)
fig.savefig("gh200_poster_figure.pdf", bbox_inches="tight")  # vector for print
print("Saved gh200_poster_figure.png / .pdf")
