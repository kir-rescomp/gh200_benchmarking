#!/usr/bin/env python
# coding: utf-8
"""
Cell2location benchmark script for GH200 nodes.
Converted from Jupyter notebook — all outputs written to files.
"""

import gc
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

import scanpy as sc
import numpy as np
import cell2location as c2l
import seaborn as sns


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

H5AD_REF = "/gpfs3/well/kir-scratch/sansom/mat611/GH200_benchmarking/haoliang_cell2location/Drokhlyansky2020.h5ad"
H5AD_QRY = "/gpfs3/well/kir-scratch/sansom/mat611/GH200_benchmarking/haoliang_cell2location/merged.square_008um.h5ad"
OUTDIR   = "/gpfs3/well/kir-scratch/sansom/mat611/GH200_benchmarking/haoliang_cell2location/from_drokhlyansky2020"

os.makedirs(OUTDIR, exist_ok=True)

# Redirect stdout/stderr to a log file alongside the SLURM .out file
log_path = os.path.join(OUTDIR, "run.log")
log_fh   = open(log_path, "w", buffering=1)   # line-buffered

class Tee:
    """Write to both the original stream and a log file."""
    def __init__(self, stream, fh):
        self.stream = stream
        self.fh     = fh
    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
    def flush(self):
        self.stream.flush()
        self.fh.flush()

sys.stdout = Tee(sys.stdout, log_fh)
sys.stderr = Tee(sys.stderr, log_fh)

print(f"Output directory : {OUTDIR}")
print(f"Log file         : {log_path}")


def save_plot(path, *plot_call_args, plot_fn, dpi=150, **plot_call_kwargs):
    """
    Call plot_fn(*args, **kwargs), save whatever ends up in the current figure,
    then close. Works whether plot_fn returns a Figure or None.
    """
    result = plot_fn(*plot_call_args, **plot_call_kwargs)
    fig = result if isinstance(result, plt.Figure) else plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {path}")


# ---------------------------------------------------------------------------
# Diagnostic: list output directory
# ---------------------------------------------------------------------------

print("\n--- OUTDIR listing ---")
subprocess.run(["ls", "-lh", OUTDIR])
print("----------------------\n")


# ---------------------------------------------------------------------------
# Reference data
# Drokhlyansky et al. Cell (2020) https://doi.org/10.1016/j.cell.2020.08.003
# ---------------------------------------------------------------------------

print("Loading reference data...")
rdata = sc.read_h5ad(H5AD_REF)
print(rdata)

# Save obs metadata summary to TSV
obs_summary = (
    rdata.obs[["sample_name", "condition", "inflamed", "tissue",
               "mouse", "exp_batch", "tech", "chemistry", "dataset", "tech_chem"]]
    .value_counts()
    .sort_index()
    .reset_index()
)
obs_summary_path = os.path.join(OUTDIR, "reference_obs_summary.tsv")
obs_summary.to_csv(obs_summary_path, sep="\t", index=False)
print(f"Saved reference obs summary → {obs_summary_path}")


# ---------------------------------------------------------------------------
# Regression model — reference signature
# ---------------------------------------------------------------------------

c2l.models.RegressionModel.setup_anndata(
    adata=rdata,
    batch_key="sample_name",
    labels_key="Lineage_LoRes",
    categorical_covariate_keys=None,
)
mod = c2l.models.RegressionModel(rdata)
try:
    mod.view_anndata_setup()
except AttributeError as e:
    print(f"view_anndata_setup() skipped (rich version bug): {e}")

print("\nTraining RegressionModel (max_epochs=100)...")
mod.train(max_epochs=100)

save_plot(
    os.path.join(OUTDIR, "regression_training_history.png"),
    plot_fn=mod.plot_history,
)

print(rdata)

# Export posterior
print("Exporting posterior (num_samples=1000, batch_size=2500)...")
rdata = mod.export_posterior(
    rdata,
    sample_kwargs={"num_samples": 1000, "batch_size": 2500},
)
print(rdata)

save_plot(
    os.path.join(OUTDIR, "regression_QC.png"),
    plot_fn=mod.plot_QC,
)


# ---------------------------------------------------------------------------
# Extract inferred average expression per cell type
# ---------------------------------------------------------------------------

if "means_per_cluster_mu_fg" in rdata.varm.keys():
    inf_aver = rdata.varm["means_per_cluster_mu_fg"][
        [f"means_per_cluster_mu_fg_{i}" for i in rdata.uns["mod"]["factor_names"]]
    ].copy()
else:
    inf_aver = rdata.var[
        [f"means_per_cluster_mu_fg_{i}" for i in rdata.uns["mod"]["factor_names"]]
    ].copy()

inf_aver.columns = rdata.uns["mod"]["factor_names"]

print("\ninf_aver head (5×5):")
print(inf_aver.iloc[0:5, 0:5])

print("\nTop 20 genes for 'Stromal cells':")
print(inf_aver.sort_values("Stromal cells", ascending=False).head(20))

# Save full inf_aver to file
inf_aver_path = os.path.join(OUTDIR, "inf_aver.tsv")
inf_aver.to_csv(inf_aver_path, sep="\t")
print(f"Saved inf_aver → {inf_aver_path}")


# ---------------------------------------------------------------------------
# Free reference data before loading spatial query
# ---------------------------------------------------------------------------

del rdata, mod
gc.collect()
print("Reference data freed from memory.")


# ---------------------------------------------------------------------------
# Query (spatial) data
# ---------------------------------------------------------------------------

# The query h5ad was written with a newer anndata that serialises log1p(base=None)
# as encoding_type='null', which this anndata version cannot deserialise.
# Workaround: strip uns/log1p/base from a scratch copy, read that, then delete it.
import h5py, shutil

tmp_qry = H5AD_QRY + ".readfix.h5ad"
print(f"\nCopying query h5ad to temp path for null-encoding fix: {tmp_qry}")
shutil.copy2(H5AD_QRY, tmp_qry)
try:
    with h5py.File(tmp_qry, "r+") as f:
        if "uns/log1p" in f and "base" in f["uns/log1p"]:
            del f["uns/log1p/base"]
            print("Patched uns/log1p/base (null encoding) in temp copy.")
    print("Loading spatial query data...")
    qdata = sc.read_h5ad(tmp_qry)
finally:
    os.remove(tmp_qry)
    print("Temp copy removed.")

print(qdata)

# Restrict to shared genes
intersect = np.intersect1d(qdata.var_names, inf_aver.index)
print(f"Shared genes: {len(intersect)}")
qdata    = qdata[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# Use raw counts layer
qdata.X = qdata.layers["counts"].copy()

# Summarise high-count entries (was a bare expression in the notebook)
high_counts = qdata.X[qdata.X > 50]
print(f"Entries with count > 50: {high_counts.shape[0]}  "
      f"(max={high_counts.max():.0f})")


# ---------------------------------------------------------------------------
# Cell2location model
# ---------------------------------------------------------------------------

c2l.models.Cell2location.setup_anndata(adata=qdata, batch_key="sample")

mod = c2l.models.Cell2location(
    qdata,
    cell_state_df=inf_aver,
    N_cells_per_location=30,
    detection_alpha=20,
)
try:
    mod.view_anndata_setup()
except AttributeError as e:
    print(f"view_anndata_setup() skipped (rich version bug): {e}")

print("\nTraining Cell2location model (max_epochs=30000)...")
mod.train(max_epochs=30000, batch_size=None, train_size=1)

save_plot(
    os.path.join(OUTDIR, "c2l_training_history.png"),
    1000,                   # skip first 1000 epochs for clarity
    plot_fn=mod.plot_history,
)

print("\nAll done.")
log_fh.close()
