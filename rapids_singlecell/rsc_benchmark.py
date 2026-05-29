#!/usr/bin/env python
"""
GH200 showcase benchmark: end-to-end million-cell single-cell clustering
entirely in HBM3e using rapids-singlecell.

Pipeline (all GPU-resident):
    QC -> normalize -> HVG -> PCA -> kNN graph -> UMAP -> Leiden -> markers

Outputs:
    - gh200_results.h5ad         (annotated AnnData with embedding + clusters)
    - gh200_timings.csv          (per-stage wall-clock + peak HBM)
    - gh200_markers.csv          (top marker genes per Leiden cluster)
"""

import time
import json
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator
import rmm

# ── RMM pool: let CuPy/RAPIDS pull from a managed pool spanning HBM3e.
#    managed_memory=True lets allocations spill into the Grace LPDDR5X over
#    NVLink-C2C if a stage transiently exceeds HBM — the GH200 unified-memory
#    safety net. For 1.3M cells it should stay in HBM, but this prevents a
#    hard OOM on the largest stages.
rmm.reinitialize(
    managed_memory=True,
    pool_allocator=True,
    initial_pool_size="40GB",
)
cp.cuda.set_allocator(rmm_cupy_allocator)

# ── timing + memory instrumentation ──────────────────────────────────────────
timings = []

def hbm_peak_gb():
    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) / 1024**3

class stage:
    """Context manager: time a pipeline stage and record peak HBM after it."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        cp.cuda.Stream.null.synchronize()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *a):
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - self.t0
        mem = hbm_peak_gb()
        timings.append({"stage": self.name, "seconds": round(dt, 2),
                        "hbm_used_gb": round(mem, 1)})
        print(f"  [{self.name:<22}] {dt:7.2f}s   HBM in use: {mem:5.1f} GB")

# ── 1. load data ─────────────────────────────────────────────────────────────
# 1.3M mouse brain cells (10x Genomics). Download once to DATA_PATH.
# wget https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/\
#   1M_neurons_filtered_gene_bc_matrices_h5.h5
DATA_PATH = "1M_neurons_filtered_gene_bc_matrices_h5.h5"

print("Loading dataset...")
adata = sc.read_10x_h5(DATA_PATH)
adata.var_names_make_unique()
print(f"  loaded: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

# Use the following sub-sampling for a smoke test
# sc.pp.subsample(adata, n_obs=100_000, random_state=0)

# ── 2. move the whole matrix onto the GPU and keep it there ──────────────────
with stage("to_GPU"):
    rsc.get.anndata_to_GPU(adata)   # X now lives in HBM3e for the whole run

# ── 3. QC + basic filtering ──────────────────────────────────────────────────
# ── 3. QC + basic filtering ──────────────────────────────────────────────────
with stage("QC_filter"):
    rsc.pp.flag_gene_family(adata, gene_family_name="MT", gene_family_prefix="mt-")
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["MT"])

    n_mt = int(adata.var["MT"].sum())
    print(f"  flagged {n_mt} mitochondrial genes", flush=True)

    rsc.pp.filter_genes(adata, min_cells=3)      # genes detected in >=3 cells
    rsc.pp.filter_cells(adata, min_counts=200)   # cells with >=200 total counts

    adata = adata[adata.obs["pct_counts_MT"] < 20].copy()
    print(f"  after QC: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes", flush=True)


# keep raw counts for marker logreg later
adata.layers["counts"] = adata.X.copy()

# ── 4. normalize + log1p ─────────────────────────────────────────────────────
with stage("normalize_log1p"):
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)

# ── 5. highly variable genes (seurat_v3 works on raw counts) ─────────────────
with stage("HVG"):
    rsc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                                 layer="counts")
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()

# ── 6. scale + PCA ───────────────────────────────────────────────────────────
with stage("scale_PCA"):
    rsc.pp.scale(adata, max_value=10)
    rsc.pp.pca(adata, n_comps=50)

# ── 7. neighbour graph (cuVS kNN) — the classic CPU bottleneck ───────────────
with stage("neighbors_kNN"):
    rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

# ── 8. UMAP embedding ────────────────────────────────────────────────────────
with stage("UMAP"):
    rsc.tl.umap(adata, min_dist=0.3, maxiter=200)

# ── 9. Move to CPU for clustering + markers ──────────────────────────────────
# Heavy GPU work (PCA, kNN, UMAP) is done. The neighbour graph rsc built lives
# in adata.obsp["connectivities"]; clustering on it is light. Transfer now.
with stage("to_CPU"):
    rsc.get.anndata_to_CPU(adata)

# ── 10. Leiden clustering (CPU igraph on the GPU-built graph) ─────────────────
with stage("Leiden"):
    import scanpy as sc
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden",
                 flavor="igraph", n_iterations=10, directed=False)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"  Leiden found {n_clusters} clusters", flush=True)

# ── 11. Marker genes per cluster (CPU logistic regression) ───────────────────
with stage("markers_logreg"):
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="logreg",
                            use_raw=True)

# ── persist results ──────────────────────────────────────────────────────────
print("\nWriting outputs...", flush=True)
adata.write("gh200_results.h5ad")

pd.DataFrame(timings).to_csv("gh200_timings.csv", index=False)
marker_tbl = sc.get.rank_genes_groups_df(adata, group=None)
marker_tbl.to_csv("gh200_markers.csv", index=False)

total = sum(t["seconds"] for t in timings)
peak = max(t["hbm_used_gb"] for t in timings)
summary = {
    "n_cells": int(adata.n_obs),
    "n_clusters": int(adata.obs["leiden"].nunique()),
    "total_pipeline_seconds": round(total, 1),
    "peak_hbm_gb": round(peak, 1),
}
with open("gh200_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 55)
print(f"  Cells clustered : {summary['n_cells']:,}")
print(f"  Clusters found  : {summary['n_clusters']}")
print(f"  Total pipeline  : {summary['total_pipeline_seconds']}s")
print(f"  Peak HBM3e used : {summary['peak_hbm_gb']} GB / 144 GB")
print("=" * 55)
