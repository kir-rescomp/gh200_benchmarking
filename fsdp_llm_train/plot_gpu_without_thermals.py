#!/usr/bin/env python3

# If you are running this as a Python script and not a notebook,
# kaleido==0.2.1 is required to suppress number of warning related to Chrome back-end
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fn = "gpu-stats-19699360.csv"
title = "GH200 FSDP LLM TRAIN - 19694471"
output_file = fn.rsplit(".", 1)[0]

# prepare dataframe
df = pd.read_csv(fn)
df.rename(columns=lambda c: c.strip(), inplace=True)
df.dropna(inplace=True)
df["time"] = pd.to_datetime(df["timestamp"])

# one trace per GPU
gpus = df["uuid"].unique()

# Only plot columns that are actually present in the CSV
candidate_cols = [
    "utilization.gpu [%]",
    "utilization.memory [%]",
    "memory.used [MiB]",
    "temperature.gpu",
    "clocks.current.sm [MHz]",
    "power.draw [W]",
    "sw_thermal_slowdown",
]

# Handle sw_thermal_slowdown if the raw column exists
if "clocks_event_reasons.sw_thermal_slowdown" in df.columns:
    df["sw_thermal_slowdown"] = df["clocks_event_reasons.sw_thermal_slowdown"].apply(
        lambda x: 0 if "Not" in str(x) else 1
    )

cols_to_plot = [c for c in candidate_cols if c in df.columns]

fig = make_subplots(rows=len(cols_to_plot), cols=1, subplot_titles=cols_to_plot, shared_xaxes=True)

for row0, measure in enumerate(cols_to_plot):
    for gpu in gpus:
        gdf = df[df["uuid"] == gpu]
        fig.append_trace(go.Scatter(
            x=gdf["time"],
            y=gdf[measure],
            name=gpu[-8:],
            legendgroup=gpu,
            showlegend=(row0 == 0),
        ), row=row0+1, col=1)

fig.update_layout(
    height=220 * len(cols_to_plot),
    width=1100,
    title_text=title,
    template="plotly_dark",
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
)
fig.update_traces(line=dict(width=1))

for a in fig.layout.annotations:
    if a["text"] == "utilization.gpu [%]":
        a["text"] = "utilization.gpu [%%] (mean = %.1f %%)" % (df["utilization.gpu [%]"].mean(),)

fig.write_html(output_file + ".html")
fig.write_image(output_file + ".png")
print(f"Saved {output_file}.html and {output_file}.png")
