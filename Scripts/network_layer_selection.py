"""
Organizing the Groups into signal flow layers.

"""
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import jenkspy
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.colors import PowerNorm
import seaborn as sns
sns.set_style("ticks")
from prats_helpers import place_panel_label

#------------------------------------------------------------------------------
folder = "Analysis_Outputs/Group_Layer-Network"

#------------------------------------------------------------------------------
sig_int = pd.read_parquet("Analysis_Outputs/Group_Signal-Flow/Signal-Integration_Group.parquet",
                           engine = "pyarrow")

# Get all the sources
sources = np.load("Analysis_Outputs/Group_Layer-Network/Source_Groups.npy")
# sources = []
# Removing the sources 
sig_int_no_sources = sig_int.loc[~sig_int["Group"].isin(sources)].reset_index(drop = True)

#------------------------------------------------------------------------------
# Params
linewidth      = 0.75
scatter_size   = 6
scatter_alpha  = 0.75
linewidth_hist = 0.5
alpha_hist     = 0.35
title_fontsize = 8.5
label_fontsize = 7
tick_fontsize  = 6
panel_fontsize = 16

# Panel Labels
label_kwargs = dict(fontsize = 16, va = "top", ha = "right")

#------------------------------------------------------------------------------
# We already have selected k = 11 layers based on this analysis
# Adding that to the plot   
final_k = 10
start   = 5
stop    = 21

#------------------------------------------------------------------------------
# Cutting into k layers
sil_scores = []
max_sizes  = []
mean_cvs   = []

# Get the values
scores = sig_int_no_sources["Final_Score"].values.reshape(-1, 1)
scores_flat = sig_int_no_sources["Final_Score"].values

# Loop
for k in range(start, stop):
    # --- Quantile-based cutting (original) ---
    # labels = pd.qcut(sig_int_no_sources["Final_Score"],
    #                  k,
    #                  labels = False).values

    # --- Jenks natural breaks ---
    breaks = jenkspy.jenks_breaks(scores_flat, n_classes = k)
    labels = pd.cut(sig_int_no_sources["Final_Score"],
                    bins = breaks,
                    labels = False,
                    include_lowest = True).values

    # Silhouette Score
    sil_scores.append(silhouette_score(scores, labels))

    # Max Counts
    _, counts = np.unique(labels, return_counts = True)
    max_sizes.append(counts.max())

    # Mean CV
    cvs = []
    for l in range(k):
        vals = sig_int_no_sources["Final_Score"].values[labels == l]
        if len(vals) > 1 and vals.mean() > 0:
            cvs.append(vals.std() / vals.mean())
    # Mean
    mean_cvs.append(np.mean(cvs))

#------------------------------------------------------------------------------
sel_df = pd.DataFrame(zip(range(start, stop),
                          sil_scores,
                          max_sizes,
                          mean_cvs),
                      columns = ["k",
                                 "Silhouette_Score",
                                 "Max_Size",
                                 "Mean_CV"])

# Save it
sel_df.to_parquet(f"{folder}/Layer-Selection_Metrics.parquet",
                  index = False,
                  engine = "pyarrow")

#------------------------------------------------------------------------------
# Assigning Layers
# --- Quantile-based cutting (original) ---
# sig_int_no_sources["Layer"] = pd.qcut(sig_int_no_sources["Final_Score"],
#                                        final_k,
#                                        labels = False).astype(int) + 1

# --- Jenks natural breaks ---
final_breaks = jenkspy.jenks_breaks(sig_int_no_sources["Final_Score"].values,
                                    n_classes = final_k)
sig_int_no_sources["Layer"] = pd.cut(sig_int_no_sources["Final_Score"],
                                      bins = final_breaks,
                                      labels = False,
                                      include_lowest = True).astype(int) + 1

# Renumbering ONLY if needed - starting at 1 since the sources are zero!
# sig_int_no_sources["Layer"] = final_k - sig_int_no_sources["Layer"]

# Creating a mapping
layer_map = dict(zip(sig_int_no_sources["Group"],
                     sig_int_no_sources["Layer"]))

# Map
sig_int_layers = sig_int.copy()
sig_int_layers["Layer"] = sig_int_layers["Group"].map(layer_map)
# The sources dont get a value, as intended. Set them to zero
sig_int_layers["Layer"] = sig_int_layers["Layer"].fillna(0).astype(int)

# Save it
sig_int_layers.to_parquet(f"{folder}/Signal-Integration_Group_Layers-Assignment_k-{final_k}.parquet",
                          index = False,
                          engine = "pyarrow")

#------------------------------------------------------------------------------
# Plot
fig, ax = plt.subplots(figsize = (8, 2.5), ncols = 3)
plt.subplots_adjust(wspace = 0.4)

# Silhouette Score
ax[0].plot(range(start, stop), sil_scores,
           linewidth = linewidth,
           marker = "o",
           alpha = scatter_alpha,
           markersize = scatter_size / 2,
           markerfacecolor = "red",
           markeredgecolor = "blue",
           markeredgewidth = linewidth_hist)

# Circle
ax[0].scatter(final_k, sil_scores[final_k - start],
           s = scatter_size * 5,
           facecolors = "none",
           edgecolors = "green",
           linewidth = linewidth_hist)

# Set the labels
ax[0].set_xlabel("Number of Layers (k)", fontsize = label_fontsize, labelpad = 5)
ax[0].set_ylabel("Silhouette Score", fontsize = label_fontsize, labelpad = 5)

# Layer Crowding
ax[1].plot(range(start, stop), max_sizes,
           linewidth = linewidth,
           marker = "o",
           alpha = scatter_alpha,
           markersize = scatter_size / 2,
           markerfacecolor = "red",
           markeredgecolor = "blue",
           markeredgewidth = linewidth_hist)

# Circle
ax[1].scatter(final_k, max_sizes[final_k - start],
           s = scatter_size * 5,
           facecolors = "none",
           edgecolors = "green",
           linewidth = linewidth_hist)

# Set the labels
ax[1].set_xlabel("Number of Layers (k)", fontsize = label_fontsize, labelpad = 5)
ax[1].set_ylabel("Max Layer Size", fontsize = label_fontsize, labelpad = 5)

# Within Layer Homogeniety
ax[2].plot(range(start, stop), mean_cvs,
           linewidth = linewidth,
           marker = "o",
           alpha = scatter_alpha,
           markersize = scatter_size / 2,
           markerfacecolor = "red",
           markeredgecolor = "blue",
           markeredgewidth = linewidth_hist)

# Circle
ax[2].scatter(final_k, mean_cvs[final_k - start],
           s = scatter_size * 5,
           facecolors = "none",
           edgecolors = "green",
           linewidth = linewidth_hist)

# Set the labels
ax[2].set_xlabel("Number of Layers (k)", fontsize = label_fontsize, labelpad = 5)
ax[2].set_ylabel("Mean Within-Layer CV", fontsize = label_fontsize, labelpad = 5)

# Ticks
ticks = np.linspace(start, stop, stop - start + 1).astype(int)
labels = [str(x) if x % 2 == 0 else "" for x in ticks]

# Despine
for i, a in enumerate(ax):
    # Despine
    sns.despine(ax = a)
    # Params
    a.tick_params(axis = "both",
                  which = "major",
                  labelsize = tick_fontsize,
                  length    = 1.5,
                  width     = 0.5,
                  pad       = 1.5)
    # X-Range
    a.set_xlim(start - 1, stop)
    a.set_xticks(ticks)
    a.set_xticklabels(labels)

    # Vertical line
    a.axvline(x = final_k,
             color = "grey",
             linewidth = linewidth,
             linestyle = "--",
             alpha = 0.5)
    # To align to the top
    trans = blended_transform_factory(a.transData, a.transAxes)
    a.text(x = final_k + 0.5,
           y = 0.95,
           s = f"k = {final_k}",
           fontsize = tick_fontsize,
           color = "grey",
           ha = "left",
           va = "center",
           transform = trans)

# Add Panel Labels
shx = -0.04
shy = 0.065
place_panel_label(fig, ax[0], "A", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1], "B", shx = -0.03, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[2], "C", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)

# Save the figure
plt.savefig(f"{folder}/Figure_05_Supplementary_Layers-Selection.pdf",
            dpi = 1200,
            bbox_inches = "tight")

#------------------------------------------------------------------------------
# Layer vs Integration Index to see flattening out in the later layers
fig, ax = plt.subplots(figsize = (7, 2.5), nrows = 1, ncols = 2)
plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

# Scatter for Final Score vs Integration Index
sns.regplot(data = sig_int_layers,
            x = "Final_Score",
            y = "Integration_Index",
            lowess = True,
            scatter_kws = {"s"     : scatter_size,
                           "color" : "purple",
                           "alpha" : scatter_alpha / 2},
            line_kws = {"color"     : "purple",
                        "linewidth" : linewidth},
            ax = ax[0])

# Labels
ax[0].set_xlabel("Topological Depth (Final Score)", fontsize = label_fontsize, labelpad = 5)
ax[0].set_ylabel("Integration Index", fontsize = label_fontsize, labelpad = 5)
ax[0].set_xlim(-0.01, 0.7)

# Scatter for Final Score vs Integration Index
sns.stripplot(data = sig_int_layers,
            x = "Layer",
            y = "Integration_Index",
            s = scatter_size / 2.5,
            color  = "purple",
            jitter = True,
            alpha  = scatter_alpha / 2,
            ax     = ax[1])

# Place the mean + CI
sns.pointplot(data = sig_int_layers,
            x = "Layer",
            y = "Integration_Index",
            color    = "black",
            markersize = scatter_size / 1.5,
            linewidth  = linewidth,
            errorbar = "ci",
            capsize  = 0.1,
            ax       = ax[1])

# Labels
ax[1].set_xlabel("Layer", fontsize = label_fontsize, labelpad = 5)
ax[1].set_ylabel("Integration Index", fontsize = label_fontsize, labelpad = 5)

# Flatten and Format
for a in ax:
    # Ticks
    a.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = tick_fontsize - 1,
        length    = 1.5,
        width     = 0.5,
        pad       = 1.5)
    # Despine
    sns.despine(ax = a)
    # Limits
    a.set_ylim(-0.05, 1)

# Labels
place_panel_label(fig, ax[0], "A", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1], "B", shx = -0.03, shy = shy,
                  label_kwargs = label_kwargs)

# Save the figure
plt.savefig(f"{folder}/Figure_05_Supplementary_Layers-vs-Integration.pdf",
            dpi = 1200,
            bbox_inches = "tight")
plt.close()

#------------------------------------------------------------------------------
# FEEDFORWARD / FEEDBACK / LATERAL DECOMPOSITION
conn_df = pd.read_csv("Data_Connectivity/Connectivity_Group.csv")

# Map each group to its layer
conn_df["Layer_pre"]  = conn_df["ConnectivityGroup_pre"].map(layer_map).fillna(0).astype(int)
conn_df["Layer_post"] = conn_df["ConnectivityGroup_post"].map(layer_map).fillna(0).astype(int)

# Direction: positive = feedforward, negative = feedback, zero = lateral
conn_df["Direction"] = conn_df["Layer_post"] - conn_df["Layer_pre"]
conn_df["Edge_Type"] = np.where(conn_df["Direction"] > 0, "Feedforward",
                       np.where(conn_df["Direction"] < 0, "Feedback",
                                "Lateral"))

# Exclude self-connections
conn_no_self = conn_df.loc[conn_df["Pre"] != conn_df["Post"]]

# Per-layer FF/FB/Lateral weight (outgoing from each layer)
ff_fb = conn_no_self.groupby(["Layer_pre", "Hemisphere_pre", "Edge_Type"])["weight"].sum().reset_index()
ff_fb.columns = ["Layer", "Hemisphere", "Edge_Type", "Weight"]

# FB/FF ratio per layer per hemisphere
ff_fb_pivot = ff_fb.pivot_table(index   = ["Layer", "Hemisphere"],
                                columns = "Edge_Type",
                                values  = "Weight",
                                fill_value = 0).reset_index()
ff_fb_pivot.columns.name = None  # Clear MultiIndex column name

# Ensure all edge type columns exist
for et in ["Feedforward", "Feedback", "Lateral"]:
    if et not in ff_fb_pivot.columns:
        ff_fb_pivot[et] = 0

ff_fb_pivot["FB_FF_Ratio"] = ff_fb_pivot["Feedback"] / (ff_fb_pivot["Feedforward"] + 1e-12)

# Proportions
ff_fb_pivot["Total"] = ff_fb_pivot["Feedforward"] + ff_fb_pivot["Feedback"] + ff_fb_pivot["Lateral"]
for et in ["Feedforward", "Feedback", "Lateral"]:
    ff_fb_pivot[f"{et}_Frac"] = ff_fb_pivot[et] / (ff_fb_pivot["Total"] + 1e-12)

# Save
ff_fb_pivot.to_parquet(f"{folder}/Layer_FF-FB-Lateral.parquet",
                       index = False,
                       engine = "pyarrow")

#--------------------------------------
# LAYER-TO-LAYER CONNECTIVITY MATRIX (hemisphere separated)
conn_df["LayerHemi_pre"]  = conn_df["Hemisphere_pre"] + "_" + conn_df["Layer_pre"].astype(str).str.zfill(2)
conn_df["LayerHemi_post"] = conn_df["Hemisphere_post"] + "_" + conn_df["Layer_post"].astype(str).str.zfill(2)

# Matrix
layer_matrix = conn_df.pivot_table(index   = "LayerHemi_pre",
                                   columns = "LayerHemi_post",
                                   values  = "weight",
                                   aggfunc = "sum",
                                   fill_value = 0)

# Save
layer_matrix.to_parquet(f"{folder}/Layer_Connectivity-Matrix.parquet",
                        engine = "pyarrow")

#--------------------------------------
# LAYER-TO-LAYER OUTGOING FRACTION: RELATIVE HEMISPHERE
layer_conn_df = conn_df.groupby(["Layer_pre", "Hemisphere_pre", "Layer_post", "Hemisphere_post"])["weight"].sum().reset_index()
# Whether or not both hemispheres are the same
layer_conn_df["Direction"] = layer_conn_df["Hemisphere_pre"] == layer_conn_df["Hemisphere_post"]
# Replace True with Ipsi and False with Contra
layer_conn_df["Direction"] = layer_conn_df["Direction"].replace({True: "Ipsi", False: "Contra"})
# Outgoing fraction: normalize by each source layer-hemisphere's total output
layer_conn_df["norm_weight"] = layer_conn_df["weight"] / (layer_conn_df.groupby(["Layer_pre", "Hemisphere_pre"])["weight"].transform("sum") + 1e-12)

# Average over hemispheres
layer_conn_df = layer_conn_df.groupby(["Layer_pre", "Layer_post", "Direction"])["norm_weight"].mean().reset_index()

# Correcting the Names
layer_conn_df["Layer_pre"] = layer_conn_df["Layer_pre"].astype(str).str.zfill(2)
layer_conn_df["Layer_post"] = layer_conn_df["Direction"] + "_" + layer_conn_df["Layer_post"].astype(str).str.zfill(2)

# Matrix
dir_matrix = layer_conn_df.pivot_table(index   = "Layer_pre",
                                       columns = "Layer_post",
                                       values  = "norm_weight",
                                       aggfunc = "sum",
                                       fill_value = 0)

# Ipsilateral ordered first in columns
dir_order = [f"{d}_{l:02d}" for d in ["Ipsi", "Contra"] for l in range(11)]
dir_matrix = dir_matrix[dir_order]

# Just making it ipsilateral vs contralateral
ic_matrix = layer_conn_df.pivot_table(index   = "Layer_pre",
                                      columns = "Direction",
                                      values  = "norm_weight",
                                      aggfunc = "sum",
                                      fill_value = 0)
# Reorder
ic_matrix = ic_matrix[["Ipsi", "Contra"]]

#--------------------------------------
# Plot: Panel A = Stacked Bar, Panel B = Hemisphere Heatmap
fig, ax = plt.subplots(figsize = (8.5, 9),
                       ncols = 2,
                       nrows = 3,
                       gridspec_kw = {"width_ratios" : [1, 1.4], "height_ratios" : [1, 0.5, 0.5]})

plt.subplots_adjust(wspace = 0.75)

edge_cmap = {"Feedforward" : "steelblue",
             "Feedback"    : "indianred",
             "Lateral"     : "grey"}

# Panel A: Stacked bar - Proportions per layer (Right hemisphere)
right_fb = ff_fb_pivot.loc[ff_fb_pivot["Hemisphere"] == "Right"].sort_values("Layer")
bottom_ff = np.zeros(len(right_fb))

for et in ["Feedforward", "Lateral", "Feedback"]:
    ax[0, 0].bar(right_fb["Layer"], right_fb[f"{et}_Frac"],
              bottom = bottom_ff,
              color  = edge_cmap[et],
              label  = et,
              edgecolor = "k",
              linewidth = 0.3,
              width  = 0.8)
    bottom_ff += right_fb[f"{et}_Frac"].values

ax[0, 0].set_xlabel("Layer", fontsize = label_fontsize, labelpad = 5)
ax[0, 0].set_ylabel("Fraction of Output", fontsize = label_fontsize, labelpad = 5)
ax[0, 0].legend(fontsize = tick_fontsize, loc = "upper left",
                bbox_to_anchor = (1.01, 1.01))
ax[0, 0].set_ylim(0, 1)
# Labels
ax[0, 0].set_xticks(np.arange(0, 11), np.arange(0, 11))

# Panel B: Hemisphere-separated heatmap
sns.heatmap(layer_matrix,
            cmap = "magma",
            norm = PowerNorm(gamma = 0.3, vmin = 0,
                             vmax = layer_matrix.values.max()),
            square = True,
            linewidths = 0.3,
            ax = ax[0, 1],
            cbar_kws = {"label"  : "Connection Weight",
                        "shrink" : 0.6,
                        "aspect" : 15},
            xticklabels = True,
            yticklabels = True)
ax[0, 1].set_xlabel("PostSynaptic Layer", fontsize = label_fontsize, labelpad = 5)
ax[0, 1].set_ylabel("PreSynaptic Layer", fontsize = label_fontsize, labelpad = 5)
ax[0, 1].tick_params(labelsize = tick_fontsize - 1)
plt.setp(ax[0, 1].get_xticklabels(), rotation = 90)
plt.setp(ax[0, 1].get_yticklabels(), rotation = 0)

# Colorbar font sizes
cbar = ax[0, 1].collections[0].colorbar
cbar.ax.tick_params(labelsize = tick_fontsize - 1)
cbar.set_label("Connection Weight", fontsize = tick_fontsize - 1)

# Add a slightly thicked dividing line across hemipheres
ax[0, 1].axhline(y = 11, color = "white", linewidth = 1.5)
ax[0, 1].axvline(x = 11, color = "white", linewidth = 1.5)

# Panel Bii: Relative Hemisphere  
sns.heatmap(dir_matrix,
            cmap = "magma",
            vmin = 0,
            vmax = 0.3,
            square = True,
            linewidths = 0.3,
            ax = ax[1, 1],
            cbar_kws = {"label"  : "Output Fraction",
                        "shrink" : 0.25,
                        "aspect" : 7,
                        "extend" : "max"},
            xticklabels = True,
            yticklabels = True)

# Colorbar font sizes
cbar = ax[1, 1].collections[0].colorbar
cbar.ax.tick_params(labelsize = tick_fontsize - 1)
cbar.set_label("Output Fraction", fontsize = tick_fontsize - 1)

# Vertical Line
ax[1, 1].axvline(x = 11, color = "white", linewidth = 1.5)

# Labels
ax[1, 1].set_xlabel("PostSynaptic Layer", fontsize = label_fontsize, labelpad = 5)
ax[1, 1].set_ylabel("PreSynaptic Layer", fontsize = label_fontsize, labelpad = 5)

# Panel Biii: Relative Hemisphere  
sns.heatmap(ic_matrix.T,
            cmap = "magma",
            vmin = 0,
            vmax = 1,
            square = True,
            linewidths = 0.3,
            annot = True,
            fmt   = ".2f",
            annot_kws = {"size" :  4},
            ax = ax[2, 1],
            cbar_kws = {"label"  : "Output Fraction",
                        "shrink" : 0.2,
                        "aspect" : 7},
            xticklabels = True,
            yticklabels = True)

# Colorbar font sizes
cbar = ax[2, 1].collections[0].colorbar
cbar.ax.tick_params(labelsize = tick_fontsize - 1)
cbar.set_label("Output Fraction", fontsize = tick_fontsize - 1)

# Labels
ax[2, 1].set_ylabel("", fontsize = label_fontsize, labelpad = 5)
ax[2, 1].set_xlabel("PreSynaptic Layer", fontsize = label_fontsize, labelpad = 5)

#--------------------------------------
# Format
for a in ax.flatten():
    a.tick_params(axis = "both",
                  which = "major",
                  labelsize = tick_fontsize,
                  length = 1.5,
                  width = 0.5,
                  pad = 1.5)
    sns.despine(ax = a, left = False, bottom = False)

# Turn Off
ax[1, 0].axis("off")
ax[2, 0].axis("off")

# Aspect
ax[0, 0].set_aspect(13.5, adjustable = "box")

# Args
label_kwargs = dict(fontsize = 16, va = "bottom", ha = "right")

# Panel Labels
place_panel_label(fig, ax[0, 0], "A", shx = -0.04, shy = 0,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[0, 1], "Bi", shx = -0.07, shy = 0,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 1], "ii", shx = -0.07, shy = 0,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[2, 1], "iii", shx = -0.07, shy = 0,
                  label_kwargs = label_kwargs)

# Save
plt.savefig(f"{folder}/Figure_05_Supplementary_Layer-Connectivity.pdf",
            dpi = 1200,
            bbox_inches = "tight")
plt.close()

#------------------------------------------------------------------------------
# Status
print("हो गया दोस्तों!")