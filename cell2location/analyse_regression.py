#!/usr/bin/env python
"""
analyse_regression.py
─────────────────────
Visualise and summarise cell2location RegressionModel outputs.

Inputs (all under OUTDIR):
  - inf_aver.tsv                 cell-type × gene inferred average expression
  - rdata_with_posterior.h5ad    reference AnnData with exported posterior
  - regression_model/model.pt    PyTorch weights (inspected for architecture)

Outputs (all under OUTDIR/regression_analysis/):
  - inf_aver_heatmap.png         top marker genes × cell types
  - top_markers.tsv              top N marker genes per cell type
  - model_architecture.txt       state-dict key/shape summary of model.pt
  - umap_celltypes.png           UMAP coloured by Lineage_LoRes
  - umap_samples.png             UMAP coloured by sample_name
  - qc_total_counts.png          violin: total counts per cell type
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch

# ── paths ──────────────────────────────────────────────────────────────────────

OUTDIR   = "/gpfs3/well/kir-scratch/sansom/mat611/GH200_benchmarking/haoliang_cell2location/from_drokhlyansky2020"
SAVEDIR  = os.path.join(OUTDIR, "regression_analysis")
os.makedirs(SAVEDIR, exist_ok=True)

INF_AVER_PATH  = os.path.join(OUTDIR, "inf_aver.tsv")
RDATA_PATH     = os.path.join(OUTDIR, "rdata_with_posterior.h5ad")
MODEL_PT_PATH  = os.path.join(OUTDIR, "regression_model", "model.pt")

TOP_N_MARKERS  = 10   # top marker genes to show per cell type in heatmap
TOP_N_TABLE    = 20   # top markers to write to TSV

print(f"Saving all outputs to: {SAVEDIR}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Inspect model.pt  — architecture summary
# ══════════════════════════════════════════════════════════════════════════════

print("─── 1. model.pt architecture ───")
state_dict = torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=False)

arch_path = os.path.join(SAVEDIR, "model_architecture.txt")
with open(arch_path, "w") as fh:
    fh.write(f"{'Layer':<60}  {'Shape':<30}  {'Parameters':>12}\n")
    fh.write("─" * 108 + "\n")
    total_params = 0
    for key, tensor in state_dict.items():
        n = tensor.numel()
        total_params += n
        fh.write(f"{key:<60}  {str(tuple(tensor.shape)):<30}  {n:>12,}\n")
    fh.write("─" * 108 + "\n")
    fh.write(f"{'TOTAL PARAMETERS':<60}  {'':30}  {total_params:>12,}\n")

print(f"  Layers     : {len(state_dict)}")
print(f"  Parameters : {total_params:,}")
print(f"  Saved      → {arch_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. inf_aver — inferred average expression per cell type
# ══════════════════════════════════════════════════════════════════════════════

print("─── 2. inf_aver analysis ───")
inf_aver = pd.read_csv(INF_AVER_PATH, sep="\t", index_col=0)
cell_types = inf_aver.columns.tolist()
print(f"  Cell types : {len(cell_types)}")
print(f"  Genes      : {len(inf_aver)}\n")

# ── 2a. top marker gene per cell type (highest specificity = highest ratio
#        of that cell type's expression to the mean of all others)
scores = inf_aver.copy()
for ct in cell_types:
    others_mean = inf_aver.drop(columns=ct).mean(axis=1)
    scores[ct]  = inf_aver[ct] / (others_mean + 1e-6)

# Build marker table
rows = []
for ct in cell_types:
    top_genes = scores[ct].sort_values(ascending=False).head(TOP_N_TABLE)
    for rank, (gene, spec) in enumerate(top_genes.items(), 1):
        rows.append({
            "cell_type"      : ct,
            "rank"           : rank,
            "gene"           : gene,
            "specificity"    : spec,
            "mean_expression": inf_aver.loc[gene, ct],
        })

markers_df = pd.DataFrame(rows)
markers_path = os.path.join(SAVEDIR, "top_markers.tsv")
markers_df.to_csv(markers_path, sep="\t", index=False)
print(f"  Saved top markers → {markers_path}")

# ── 2b. heatmap: top-N markers × cell types
top_genes_per_ct = (
    markers_df[markers_df["rank"] <= TOP_N_MARKERS]
    .groupby("cell_type")["gene"]
    .apply(list)
    .to_dict()
)

heatmap_genes = []
heatmap_labels = {}          # gene → cell type label for row annotation
for ct, genes in top_genes_per_ct.items():
    for g in genes:
        if g not in heatmap_genes:
            heatmap_genes.append(g)
            heatmap_labels[g] = ct

plot_df = inf_aver.loc[heatmap_genes, :].copy()
# log1p-scale for display
plot_df = np.log1p(plot_df)

n_genes = len(heatmap_genes)
n_cts   = len(cell_types)

fig, ax = plt.subplots(figsize=(max(12, n_cts * 0.8), max(10, n_genes * 0.25)))
sns.heatmap(
    plot_df,
    ax=ax,
    cmap="viridis",
    xticklabels=True,
    yticklabels=True,
    linewidths=0,
    cbar_kws={"label": "log1p(inf_aver)"},
)
ax.set_title(f"Top {TOP_N_MARKERS} marker genes per cell type\n(RegressionModel inferred average expression)", fontsize=13)
ax.set_xlabel("Cell type")
ax.set_ylabel("Gene")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=7)
plt.tight_layout()

heatmap_path = os.path.join(SAVEDIR, "inf_aver_heatmap.png")
fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  Saved heatmap      → {heatmap_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Reference AnnData — UMAP and QC plots
# ══════════════════════════════════════════════════════════════════════════════

print("─── 3. Reference AnnData plots ───")
rdata = sc.read_h5ad(RDATA_PATH)
print(f"  {rdata}\n")

sc.settings.verbosity = 1

# Compute UMAP if not already present
if "X_umap" not in rdata.obsm:
    print("  Computing PCA + neighbours + UMAP …")
    sc.pp.highly_variable_genes(rdata, n_top_genes=3000, flavor="seurat", layer="counts")
    sc.pp.pca(rdata, use_highly_variable=True)
    sc.pp.neighbors(rdata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(rdata)
else:
    print("  Using existing UMAP embedding.")

# ── 3a. UMAP coloured by cell type
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    rdata,
    color="Lineage_LoRes",
    ax=ax,
    show=False,
    frameon=False,
    title="Reference cells — Lineage_LoRes",
    legend_loc="right margin",
    legend_fontsize=7,
)
umap_ct_path = os.path.join(SAVEDIR, "umap_celltypes.png")
fig.savefig(umap_ct_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved UMAP (cell types) → {umap_ct_path}")

# ── 3b. UMAP coloured by sample
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    rdata,
    color="sample_name",
    ax=ax,
    show=False,
    frameon=False,
    title="Reference cells — sample_name",
    legend_loc="right margin",
    legend_fontsize=6,
)
umap_sample_path = os.path.join(SAVEDIR, "umap_samples.png")
fig.savefig(umap_sample_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved UMAP (samples)    → {umap_sample_path}")

# ── 3c. Total counts violin per cell type
fig, ax = plt.subplots(figsize=(max(12, len(cell_types)), 5))
sc.pl.violin(
    rdata,
    keys="n_counts" if "n_counts" in rdata.obs else "total_counts"
          if "total_counts" in rdata.obs else None,
    groupby="Lineage_LoRes",
    ax=ax,
    show=False,
    rotation=45,
)
if "n_counts" in rdata.obs or "total_counts" in rdata.obs:
    ax.set_title("Total counts per cell type (reference)")
    qc_path = os.path.join(SAVEDIR, "qc_total_counts.png")
    fig.savefig(qc_path, dpi=150, bbox_inches="tight")
    print(f"  Saved QC violin         → {qc_path}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n─── Summary of outputs ───")
for fname in sorted(os.listdir(SAVEDIR)):
    fpath = os.path.join(SAVEDIR, fname)
    size  = os.path.getsize(fpath)
    print(f"  {fname:<40}  {size/1024:>8.1f} KB")

print("\nAll done.")
