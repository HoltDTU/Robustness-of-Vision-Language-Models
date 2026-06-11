#!/usr/bin/env python3
"""
Generate a Mean Absolute Error (MAE) vs. complexity plot for the VLM
graph-reading pilot study.

X-axis: complexity (number of nodes in the graph, = node level)
Y-axis: mean absolute error of the model's count, averaged over the
        10 graphs at each node level.

Two panels: node-count MAE (left) and edge-count MAE (right),
one line per model.

Inputs (same folder):
    predictions.csv     image_id, model, pred_nodes, pred_edges, pred_directed
    pilot_manifest.csv  image_id, node_level, n_nodes_true, n_edges_true, ...

Output:
    mae_vs_complexity.pdf   (vector, for LaTeX)
    mae_vs_complexity.png   (raster preview)
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Load and merge
# ---------------------------------------------------------------------
pred = pd.read_csv("predictions.csv")
man = pd.read_csv("pilot_manifest.csv")
df = pred.merge(man, on="image_id")

# Short, readable model names
short = {
    "google/gemma-4-26b-a4b": "Gemma-4",
    "alibaba/qwen-3.6-35b-a3b": "Qwen-3.6",
}
df["model_short"] = df["model"].map(short)

# Absolute errors
df["nodes_abserr"] = (df["pred_nodes"] - df["n_nodes_true"]).abs()
df["edges_abserr"] = (df["pred_edges"] - df["n_edges_true"]).abs()

# Per node level (= complexity) MAE, one column per model
order = ["Gemma-4", "Qwen-3.6"]
node_mae = df.pivot_table(index="node_level", columns="model_short",
                          values="nodes_abserr", aggfunc="mean").reindex(columns=order)
edge_mae = df.pivot_table(index="node_level", columns="model_short",
                          values="edges_abserr", aggfunc="mean").reindex(columns=order)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
style = {
    "Qwen-3.6": dict(color="tab:blue", marker="o", linestyle="-"),
    "Gemma-4": dict(color="tab:orange", marker="s", linestyle="--"),
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

for ax, data, title in [
    (axes[0], node_mae, "Node-count MAE"),
    (axes[1], edge_mae, "Edge-count MAE"),
]:
    for m in order:
        ax.plot(data.index, data[m], label=m, markersize=4, linewidth=1.6, **style[m])
    ax.set_title(title)
    ax.set_xlabel("Number of nodes in the graph")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(1, 24)
    ax.set_ylim(bottom=0)
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.legend()

fig.tight_layout()
fig.savefig("mae_vs_complexity.pdf", bbox_inches="tight")
fig.savefig("mae_vs_complexity.png", dpi=200, bbox_inches="tight")
print("Saved mae_vs_complexity.pdf and mae_vs_complexity.png")