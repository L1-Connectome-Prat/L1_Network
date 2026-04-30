"""
Per-compartment distribution of layer membership and signal-flow scores.

For every L1 Group we have:
    - Compartment   : from Group_Hybrid-Placement.parquet
    - Layer         : from the k = 10 PPR layer assignment
    - Integration_Drive, Final_Score : from the Signal-Integration table

"""
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
from prats_helpers import place_panel_label

from _compartment_palette import COMP_HEX, COMP_RGB, COMP_ORDER

#------------------------------------------------------------------------------
# Config
folder = "Analysis_Outputs/Group_Layer-Network"
k      = 10

# Figure params
linewidth      = 0.75
scatter_size   = 2
scatter_alpha  = 0.75
title_fontsize = 7.5
label_fontsize = 7
tick_fontsize  = 6

label_kwargs = dict(fontsize = 12, va = "center", ha = "left")

#------------------------------------------------------------------------------
# Load data
layer_df = pd.read_parquet(f"{folder}/Signal-Integration_Group_Layers-Assignment_k-{k}.parquet",
                           engine = "pyarrow")
pos_df   = pd.read_parquet(f"{folder}/Group_Hybrid-Placement.parquet",
                           engine = "pyarrow")

# Compartment is constant across hemispheres → one mapping per Group.
comp_map = (pos_df[["Group", "Compartment"]]
            .drop_duplicates()
            .set_index("Group")["Compartment"])

#------------------------------------------------------------------------------
# Merge
group_df = layer_df.copy()
group_df["Compartment"] = group_df["Group"].map(comp_map)
group_df = group_df.dropna(subset = ["Compartment"]).reset_index(drop = True)

# Restrict to compartments that are actually populated in the data.
present       = [c for c in COMP_ORDER if c in set(group_df["Compartment"])]
group_df      = group_df[group_df["Compartment"].isin(present)].copy()
group_df["Compartment"] = pd.Categorical(group_df["Compartment"],
                                          categories = present,
                                          ordered    = True)

print(f"{len(group_df)} groups across {len(present)} compartments")

# Grouping
counts = (group_df.groupby(["Compartment", "Layer"], observed = True)
                  .size()
                  .unstack(fill_value = 0)
                  .reindex(index   = present,
                           columns = np.arange(0, k + 1),
                           fill_value = 0))

#------------------------------------------------------------------------------
# Figure
fig, ax = plt.subplots(figsize = (3.5, 8),
                       nrows   = 4,
                       ncols   = 1,
                       height_ratios = [2, 1, 1, 1])

# Major tick params
for a in ax.flatten():
    # Despine
    sns.despine(ax = a)
    # Params
    a.tick_params(axis = "both",
                  which = "major",
                  labelsize = tick_fontsize,
                  length    = 1.5,
                  width     = 0.5,
                  pad       = 1.5)

# Group Count Heatmap
# Create annotations
annot = counts.T.astype(int).astype(str)
annot[annot == "0"] = ""
sns.heatmap(counts.T,
            annot      = annot,
            fmt        = "s",
            cmap       = "magma",
            linewidths = 0.4,
            square     = True,
            linecolor  = "white",
            # cbar_kws   = {"label"  : "Number of groups",
            #               "shrink" : 0.25},
            cbar       = False,
            annot_kws  = {"fontsize" : tick_fontsize - 1},
            ax         = ax[0])

ax[0].set_xlabel("",)
ax[0].set_ylabel("Layer", fontsize = label_fontsize, labelpad = 5)
ax[0].set_title("")

# Tint the y-axis labels with each compartment's color.
for tick, c in zip(ax[0].get_xticklabels(), counts.index):
    tick.set_color(tuple(COMP_RGB[c]))

# Rotate
for tick in ax[0].get_yticklabels():
    tick.set_rotation(0)

# Metrics
metrics = ["Layer", "Integration_Drive", "Final_Score"]
for a, metric in zip(ax[1:], metrics):
    # Scatterplot
    sns.stripplot(data      = group_df,
                  x         = "Compartment",
                  y         = metric,
                  order     = present,
                  hue       = "Compartment",
                  hue_order = present,
                  palette   = COMP_RGB,
                  size      = scatter_size,
                  alpha     = scatter_alpha,
                  jitter    = 0.28,
                  edgecolor = "black",
                  linewidth = 0.15,
                  legend    = False,
                  ax        = a)

    # Median marker per compartment
    medians = group_df.groupby("Compartment", observed = True)[metric].median()
    for xpos, c in enumerate(present):
        m = medians.get(c, np.nan)
        if np.isnan(m):
            continue
        a.hlines(m, xpos - 0.32, xpos + 0.32,
                 colors    = "black",
                 linewidth = linewidth,
                 zorder    = 100)

    # Labels
    a.set_ylabel(metric.replace("_", " "), fontsize = label_fontsize, labelpad = 5)
    a.set_xlabel("")
    a.set_title("", fontsize = title_fontsize, pad = 3)
    # Labels only on the last time
    if a != ax[-1]:
        a.set_xticklabels([])
    else:
        # Place and Rotate
        a.set_xticklabels(present, rotation = 90)
        # Color
        for t, c in zip(a.get_xticklabels(), present):
            t.set_color(tuple(COMP_RGB[c]))

    # Limits
    a.set_xlim(-0.5, len(present) - 0.5)

# Label
ax[-1].set_xlabel("Compartment", fontsize = label_fontsize, labelpad = 5)

# Custom Limits
ax[1].set_ylim(-0.5, 11)
ax[2].set_ylim(-0.25, 2)
ax[3].set_ylim(-0.1, 0.8)

# Lines
ax[1].set_yticks([0, 2, 4, 6, 8, 10])

# Remove spine from the heatmap
sns.despine(ax = ax[0], left = True, bottom = True)

# Panel labels
shx, shy = -0.145, 0
place_panel_label(fig, ax[0], "A", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1], "B", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[2], "C", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[3], "D", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)

# Save the Figure
plt.savefig(f"{folder}/Figure_05_Supplementary_Compartment-Layer-Heatmap.pdf",
            dpi         = 1200,
            bbox_inches = "tight")
plt.close()

#------------------------------------------------------------------------------
print("हो गया दोस्तों!")
