"""
Signal flow through the Connectivity Groups.

"""
import pandas as pd
import numpy as np
from glob import glob
import os
import networkx as nx

# Connectome
import navis
import pymaid
import connectome_analysis
import flybrains # Always required to run the xform and mirroring

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style("ticks")

# Fits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from prats_helpers import place_panel_label

#------------------------------------------------------------------------------
# Folders
folder = "Analysis_Outputs/Group_Signal-Flow"
home   = os.environ["PROJECTS_HOME"]

# Fragment to plot
fragment = "Neuron"
k_selected = 7

# Params
linewidth      = 0.75
scatter_size   = 6
scatter_alpha  = 0.75
linewidth_hist = 0.5
alpha_hist     = 0.35
title_fontsize = 8
label_fontsize = 7
tick_fontsize  = 6

# Panel Labels
label_kwargs = dict(fontsize = 16, va = "top", ha = "right")

#------------------------------------------------------------------------------
neuropil = navis.Volume(flybrains.PK_L1CBNeuropilsym.mesh)
mbr = connectome_analysis.stl_to_navis(f"{home}/L1_Compartment-Atlas/Central-Brain_Compartments/MB/MB_Right.stl")
mbl = connectome_analysis.stl_to_navis(f"{home}/L1_Compartment-Atlas/Central-Brain_Compartments/MB/MB_Left.stl")

# Symmetrize 
mbr = navis.xform_brain(mbr, source = "PK_L1CNS", target = "PK_L1CNSsym")
mbl = navis.xform_brain(mbl, source = "PK_L1CNS", target = "PK_L1CNSsym")

# Colors
neuropil.color = (0, 0, 0, 0.35)
mbr.color = (0, 0, 0, 0.25)
mbl.color = (0, 0, 0, 0.25)

#------------------------------------------------------------------------------
# Connectivity
conn_df = pd.read_csv("Data_Connectivity/Connectivity_Group.csv")

# Convert to NetworkX
G = nx.DiGraph()
G.add_weighted_edges_from(conn_df[["Pre", "Post", "weight"]].itertuples(index = False, name = None))

def flux_balance_group(G, g):
    """
    Return f_within , f_in , f_out  for GROUP node g in weighted DiGraph G.
    """
    # Self Loops
    w_within = G[g][g]["weight"] if G.has_edge(g, g) else 0.0

    # Outflow
    out_weights = {v: d["weight"] for _, v, d in G.out_edges(g, data = True) if v != g}
    w_out       = sum(out_weights.values())
    out_probs   = [w / w_out for w in out_weights.values()] if w_out > 0 else [0]
    out_entropy = -sum(p * np.log(p) for p in out_probs if p > 0) if out_probs else 0

    # Inflow
    in_weights = {u: d["weight"] for u, _, d in G.in_edges(g, data = True) if u != g}
    w_in       = sum(in_weights.values())
    in_probs   = [w / w_in for w in in_weights.values()] if w_in > 0 else [0]
    in_entropy = -sum(p * np.log(p) for p in in_probs if p > 0) if in_probs else 0

    # Total
    tot = w_within + w_in + w_out
    f_in     = w_in / (tot + 1e-12)
    f_out    = w_out / (tot + 1e-12)
    f_within = w_within / (tot + 1e-12)

    # Return Raw and Fractions
    info = (w_in, w_out, w_within, f_in, f_out, f_within, tot, in_entropy, out_entropy) if tot else 1e-12

    return info

# Measure metrics
records = []
for g in G.nodes():
    try:
        # Get the metrics
        w_in, w_out, w_within, f_in, f_out, f_within, tot, in_entropy, out_entropy = flux_balance_group(G, g)
    except Exception as e:
        print(f"Error in group {g} : {e}")
        continue


    # Append
    records.append(dict(Group    = g,
                        w_in     = w_in,
                        w_out    = w_out,
                        w_within = w_within,
                        f_in     = f_in,
                        f_out    = f_out,
                        f_within = f_within,
                        tot      = tot,
                        in_entropy  = in_entropy,
                        out_entropy = out_entropy))

# Convert to DataFrame
metrics = pd.DataFrame(records).sort_values("Group")

# Calculate in vs out degree centrality
metrics["Flow_Ratio"] = (metrics["w_out"] - metrics["w_in"]) / (metrics["w_out"] + metrics["w_in"])
# Effective input partners and output partners
metrics["Neff_inputs"]  = np.exp(metrics["in_entropy"])
metrics["Neff_outputs"] = np.exp(metrics["out_entropy"])

# Effective Breadth Ratio
metrics["Neff_Total"] = metrics["Neff_inputs"] + metrics["Neff_outputs"]
metrics["Neff_flow_direction"] = (metrics["Neff_outputs"] - metrics["Neff_inputs"]) / (metrics["Neff_Total"] + 1e-12)

#------------------------------------------------------------------------------
# SCORE DISTRIBUTIONS
fig, ax = plt.subplots(figsize = (8.5, 3), ncols = 2, sharex = False)
plt.subplots_adjust(wspace = 0.5)

# Plotting
sns.ecdfplot(data = metrics,
             x = "f_within",
             ax = ax[0],
             linewidth = linewidth,
             label = r"f$_{within}$")
sns.ecdfplot(data = metrics,
             x = "f_out",
             ax = ax[0],
             linewidth = linewidth,
             label = r"f$_{output}$")
sns.ecdfplot(data = metrics,
             x = "f_in",
             ax = ax[0],
             linewidth = linewidth,
             label = r"f$_{input}$")

# Plotting
sns.ecdfplot(data = metrics,
             x = "in_entropy",
             ax = ax[1],
             linewidth = linewidth,
             label = "Input Entropy")
sns.ecdfplot(data = metrics,
             x = "out_entropy",
             ax = ax[1],
             linewidth = linewidth,
             label = "Output Entropy")

# Labels
ax[0].set_xlabel("Fraction of Connectivity", fontsize = label_fontsize, labelpad = 5)
ax[0].set_ylabel("Cumulative Proportion", fontsize = label_fontsize, labelpad = 5)
ax[1].set_xlabel("Entropy", fontsize = label_fontsize, labelpad = 5)
ax[1].set_ylabel("Cumulative Proportion", fontsize = label_fontsize, labelpad = 5)


# Ticks
for a in ax.flatten():
    a.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = tick_fontsize,
        length    = 1.5,
        width     = 0.5,
        pad       = 1.5)

    # Despine
    sns.despine(ax = a)

# Limits
ax[0].set_xlim(0, 1)
ax[1].set_xlim(0, 5)

# Legend
ax[0].legend(fontsize = label_fontsize, loc = "lower right", bbox_to_anchor = (1, 0))
ax[1].legend(fontsize = label_fontsize, loc = "lower right", bbox_to_anchor = (1, 0))

# Main Labels
shx = -0.03
shy = 0.07
place_panel_label(fig, ax[0], "A", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1], "B", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)

# Save
plt.savefig(f"{folder}/Figure_05_Supplementary_Signal-Metrics-Distribution.pdf",
            dpi = 1200, 
            bbox_inches = "tight")
plt.close()

#------------------------------------------------------------------------------
# ROLES
# Clustering into Roles
cols = metrics.columns
cols = [c for c in cols if c != "Group"]
metrics[["Group", "Hemisphere"]] = metrics["Group"].str.split("_", expand = True)
group_df = metrics.groupby("Group")[cols].mean().reset_index()

# Select the features to use for clustering
features = ["f_in",
            "f_out",
            "f_within",
            "in_entropy",
            "out_entropy"]

# Scaling Features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(group_df[features])

# Running KMeans Clustering
k_vals = range(2, 21)
inertias   = []
sil_scores = []

# Loop
for k in k_vals:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    labels  = kmeans.fit_predict(features_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(features_scaled, labels))

# Save the results
k_results = pd.DataFrame({"k"                : list(k_vals),
                          "inertia"          : inertias,
                          "silhouette_score" : sil_scores})

# After examining the plots, we settle on 6 clusters
kmeans  = KMeans(n_clusters = k_selected, random_state = 42)
labels  = kmeans.fit_predict(features_scaled)
group_df["Cluster_Label"] = labels
sample_sils = silhouette_samples(features_scaled, labels)
avg_sil     = silhouette_score(features_scaled, labels)
# Mean for plotting
cluster_means = pd.Series(sample_sils).groupby(labels).mean()

# Mapping to Centroids
mapping = dict(zip(group_df["Group"], group_df["Cluster_Label"]))
# Centroids of the clusters
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns = features)

# Show it
print()
print(centroids)
# Assigning roles
cluster_names = {
    0 : "Broad_Relay",
    1 : "Focused_Convergence",
    2 : "Sensory_Broadcast",
    3 : "Focused_Relay",
    4 : "Broad_Convergence",
    5 : "Recurrent_Processor",
    6 : "Signal_Sink"
}
print(cluster_names)
print()

# Assign
group_df["Role"] = group_df["Cluster_Label"].map(cluster_names)

# Save this
group_df.to_parquet(f"{folder}/Group_Functional-Roles.parquet",
                    engine = "pyarrow",
                    index = False)
# Since we use the same seed every time, we should get the same roles, 
# but just in case, we'll reload the final fixed and tested version
group_df = pd.read_parquet(f"{folder}/Group_Functional-Roles_2026-04-03.parquet",
                           engine = "pyarrow")

#------------------------------------------------------------------------------
# Plotting the performance
fig, ax = plt.subplots(figsize = (7.5, 6), ncols = 2, nrows = 2,
                       sharex = False)
plt.subplots_adjust(wspace = 0.25, hspace = 0.25)

# Plot
ax[0, 0].plot(k_vals, sil_scores,
              linewidth = 1,
              markersize = 3,
              marker = "o",
              mfc = "red",
              mec = "red")
ax[0, 1].plot(k_vals, inertias,
              linewidth = 1,
              markersize = 3,
              marker = "o",
              mfc = "red",
              mec = "red")

# Limits and labels
for a in ax.flatten():
    # Despine
    sns.despine(ax = a)
    a.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = label_fontsize,
        length    = 1.5,
        width     = 0.5,
        pad       = 1.5)

for a in ax[0]:
    a.set_xlim(0, 22)
    a.set_xticks(ticks = np.linspace(0, 22, 12), labels = np.linspace(0, 22, 12).astype(int), fontsize = tick_fontsize)
    a.set_xlabel("# Clusters", labelpad = 5, fontsize = label_fontsize)

# Labels
ax[0, 0].set_ylabel("Silhouette Score", labelpad = 5, fontsize = label_fontsize)
ax[0, 1].set_ylabel("Inertia", labelpad = 5, fontsize = label_fontsize)

# Select
ax[0, 0].scatter(k_selected, sil_scores[k_selected - 2], s = 50, c = "none", edgecolors = "green", linewidth = linewidth)
ax[0, 1].scatter(k_selected, inertias[k_selected - 2], s = 50, c = "none", edgecolors = "green", linewidth = linewidth)

# Limit
ax[0, 0].set_ylim(0.1, 0.6)
ax[0, 1].set_ylim(0, 800)

# Silhouette Means
cluster_means.plot.bar(ax = ax[1, 0],
                       color = sns.color_palette("tab10", k_selected),
                       alpha = alpha_hist,
                       edgecolor = "k",
                       linewidth = linewidth_hist)
ax[1, 0].set_xlabel("Cluster", labelpad = 5, fontsize = label_fontsize)
ax[1, 0].set_ylabel("Mean Silhouette Score", labelpad = 5, fontsize = label_fontsize)
ax[1, 0].axhline(avg_sil,
                 color = "grey",
                 linestyle = "--",
                 linewidth = linewidth_hist)
ax[1, 0].text(x = -0.25,
              y = avg_sil + 0.025,
              s = f"Mean : {avg_sil:.2f}",
              fontsize = tick_fontsize,
              color = "grey",
              ha = "left",
              va = "bottom")
# Limits
ax[1, 0].set_ylim(0, 0.6)
ax[1, 0].set_xticklabels(ax[1, 0].get_xticklabels(), rotation = 0)
# Turn Off Final Axis
ax[1, 1].axis("off")

# Main Labels
shx = -0.05
shy = 0.03
place_panel_label(fig, ax[0, 0], "A", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[0, 1], "B", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 0], "C", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)

# Save the figure
plt.savefig(f"{folder}/Figure_05_Supplementary_KMeans-Performance.pdf",
            dpi = 1200, 
            bbox_inches = "tight")
plt.close()

#------------------------------------------------------------------------------
# Anatomical Signal Flow Merge
# Read in the positions
pos_df = pd.read_parquet("Analysis_Outputs/Group_Anatomy/Group_Positions.parquet",
                         engine = "pyarrow")

# Merge
group_df = group_df.merge(pos_df, on = "Group", how = "left")
group_df.to_parquet(f"{folder}/Group_Signal-Flow-Merged.parquet",
                    engine = "pyarrow",
                    index = False)

# Read in the pre-worked option
group_df = pd.read_parquet(f"{folder}/Group_Signal-Flow-Merged_2026-04-03.parquet",
                           engine = "pyarrow")
# Name
group_df["Group"] = group_df["Group"] + "_" + group_df["Hemisphere"]
# Which fragment we want to plot
fragment_df = group_df.loc[group_df["Fragment"] == fragment].reset_index(drop = True)

#------------------------------------------------------------------------------
# Figure
fig, ax = plt.subplots(figsize = (15, 18), ncols = 3, nrows = 4,
                gridspec_kw  ={'height_ratios' : [1, 1, 1, 0.75]})
plt.subplots_adjust(hspace = 0.05, wspace = 0.05)
sns.set_style("ticks")

# Group Role cmaps
role_order = ["Sensory_Broadcast",
              "Focused_Relay",
              "Broad_Relay",
              "Focused_Convergence",
              "Broad_Convergence",
              "Recurrent_Processor",
              "Signal_Sink"
              ]
role_cmap = dict(zip(role_order, sns.color_palette("tab10", k_selected)))

# Sizes
sizes = (5, 150)
alpha = 0.5

# Hemisphere
sns.scatterplot(fragment_df,
                x = "Pos_x",
                y = "Pos_y",
                hue = "Hemisphere",
                palette = {"Right" : "red", "Left" : "blue"},
                ax     = ax[0, 0],
                legend = False,
                size   = "Size",
                sizes  = sizes,
                alpha  = alpha)

# Flow Ratio
norm = mpl.colors.Normalize(vmin = -1, vmax = 1)
cmap = mpl.colormaps['coolwarm'] 
# Scatter
sns.scatterplot(
                data  = fragment_df,
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "Flow_Ratio",
                palette = cmap,
                hue_norm = norm,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = 0.2,
                edgecolor = "k",
                ax      = ax[0, 1],
                legend  = False)
# Tiny Cbar
cax1 = ax[0, 1].inset_axes([0.8, 0.07, 0.15, 0.03])
cb1 = plt.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap),
                   cax = cax1, 
                   orientation = 'horizontal')
cb1.set_label("Flow Index", fontsize = 8)
cb1.ax.tick_params(labelsize = tick_fontsize)

# Local Connectivity
within_cmap = mpl.colormaps['Greys']
within_norm = mpl.colors.Normalize(vmin = 0.0, vmax = 0.25)
# Scatter
sns.scatterplot(
                data  = fragment_df,
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "f_within",
                palette  = within_cmap,
                hue_norm = within_norm,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = 0.2,
                edgecolor = "k",
                ax      = ax[0, 2],
                legend  = False)
# Tiny Cbar
cax2 = ax[0, 2].inset_axes([0.8, 0.07, 0.15, 0.03])
cb2 = plt.colorbar(mpl.cm.ScalarMappable(norm = within_norm, cmap = within_cmap),
                   cax = cax2, 
                   orientation = 'horizontal')
cb2.set_label("Recurrent Fraction", fontsize = 8)
cb2.ax.tick_params(labelsize = tick_fontsize)

# Input Entropy based Neff
cmap_inentropy = mpl.colormaps['Blues']
norm_inentropy = mpl.colors.Normalize(vmin = 0.0, vmax = 80)
# Scatter
sns.scatterplot(data  = fragment_df,
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "Neff_inputs",
                palette  = cmap_inentropy,
                hue_norm = norm_inentropy,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = 0.2,
                edgecolor = "k",
                ax      = ax[1, 0],
                legend  = False)
# Tiny Cbar
cax3 = ax[1, 0].inset_axes([0.8, 0.07, 0.15, 0.03])
cb3 = plt.colorbar(mpl.cm.ScalarMappable(norm = norm_inentropy, cmap = cmap_inentropy),
                   cax = cax3, 
                   orientation = 'horizontal')
cb3.set_label(r"$N_{eff}$ Inputs", fontsize = 8)
cb3.ax.tick_params(labelsize = tick_fontsize)

# Output Entropy based Neff
cmap_outentropy = mpl.colormaps['Reds']
norm_outentropy = mpl.colors.Normalize(vmin = 0.0, vmax = 80)
# Scatter
sns.scatterplot(data  = fragment_df,
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "Neff_outputs",
                palette  = cmap_outentropy,
                hue_norm = norm_outentropy,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = 0.2,
                edgecolor = "k",
                ax      = ax[1, 1],
                legend  = False)
# Tiny Cbar
cax4 = ax[1, 1].inset_axes([0.8, 0.07, 0.15, 0.03])
cb4 = plt.colorbar(mpl.cm.ScalarMappable(norm = norm_outentropy, cmap = cmap_outentropy),
                   cax = cax4, 
                   orientation = 'horizontal')
cb4.set_label(r"$N_{eff}$ Outputs", fontsize = 8)
cb4.ax.tick_params(labelsize = tick_fontsize)

# Flow Direction 
cmap_pflow = mpl.colormaps['coolwarm']
norm_pflow = mpl.colors.Normalize(vmin = -1, vmax = 1)
# Scatter
sns.scatterplot(data  = fragment_df,
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "Neff_flow_direction",
                palette  = cmap_pflow,
                hue_norm = norm_pflow,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = 0.2,
                edgecolor = "k",
                ax      = ax[1, 2],
                legend  = False)
# Tiny Cbar
cax5 = ax[1, 2].inset_axes([0.8, 0.07, 0.15, 0.03])
cb5 = plt.colorbar(mpl.cm.ScalarMappable(norm = norm_pflow, cmap = cmap_pflow),
                   cax = cax5, 
                   orientation = 'horizontal')
cb5.set_label("Partner Breadth", fontsize = 8)
cb5.ax.tick_params(labelsize = tick_fontsize)

# Group Roles
sns.scatterplot(data  = fragment_df,
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "Role",
                hue_order = role_order,
                palette  = role_cmap,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha / 1.25,
                linewidth = 0.2,
                edgecolor = "k",
                legend  = True,
                ax = ax[2, 0])

# Legend
ax[2, 0].legend(title = "",
            bbox_to_anchor = (1.02, 0.95),
            loc = "upper left",
            fontsize = label_fontsize,
            title_fontsize = label_fontsize)

# Add the compartments and neuropil
for a in ax.flatten()[:7]:
    # Plot compartments
    navis.plot2d([mbr, mbl, neuropil],
                 ax = a,
                 view = ("x", "y"),
                 method = "2d", 
                 volume_outlines = True,
                 linewidth = 0.75,
                 rasterize = True)

    # Text
    for i, row in fragment_df.iterrows():
        a.text(x = row["Pos_x"],
               y = row["Pos_y"],
               s = int(row["Group"].split("_")[0][1:]),
               ha = "center",
               va = "center",
               fontsize = 2)

    # Invert axis
    a.invert_yaxis()
    # Remove all ticks
    a.axis("off")

# Turn Off Some Axes
ax[2, 1].axis("off")

#-----------------------------------------------------------------------
# Replace the default 2D axis with a 3D one (must happen after subplots creation)
ax[2, 2].remove()
ax3d = fig.add_subplot(4, 3, 12, projection = '3d')

# Replicate the exact size mapping that sns.scatterplot uses everywhere else
# This guarantees identical point sizes across the whole figure
size_col = fragment_df["Size"]
size_min = size_col.min()
size_max = size_col.max()
size_range = size_max - size_min if size_max > size_min else 1.0

# ---------------------------------------------------------------------
# Plot grouped by role → much more readable + easy legend control
for role in role_order:
    subset = fragment_df[fragment_df["Role"] == role]
    if len(subset) == 0:
        continue

    # Map "Size" column exactly the same way sns does
    mapped_sizes = 5 + (subset["Size"] - size_min) / size_range * 145.0

    ax3d.scatter(
        xs = subset["Neff_inputs"],
        ys = subset["Neff_outputs"],
        zs = subset["f_within"],
        s = mapped_sizes,
        c = [role_cmap[role]],
        alpha = alpha / 1.25,
        edgecolor = "k",
        linewidth = 0.2,
        label = role
    )

# Styling – consistent with the rest of your figure
ax3d.set_xlabel(r"$N_{eff}$ Inputs",   fontsize = label_fontsize, labelpad = 12)
ax3d.set_ylabel(r"$N_{eff}$ Outputs",  fontsize = label_fontsize, labelpad = 12)
ax3d.set_zlabel(r"$f_{within}$",           fontsize = label_fontsize, labelpad = 12)

ax3d.set_xlim(-2, 100)
ax3d.set_ylim(-2, 100)
ax3d.set_zlim(0, 0.4)

# Recommended starting camera angle (shows f_in separation nicely)
ax3d.view_init(elev = 19, azim = 12)

# Subtle grid for depth perception in 3D
ax3d.grid(True, alpha = 0.15)
ax3d.tick_params(labelsize = tick_fontsize)

#--------------------------------------
# 2D Scatter plots for better view
right_df = fragment_df.loc[fragment_df["Hemisphere"] == "Right"]
sns.scatterplot(right_df,
                x = "Neff_inputs",
                y = "Neff_outputs",
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = 0.2,
                edgecolor = "k",
                hue   = "f_within",
                palette  = within_cmap,
                hue_norm = within_norm,
                ax      = ax[3, 0],
                legend  = False)

sns.scatterplot(right_df,
                x = "Neff_inputs",
                y = "Neff_outputs",
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = 0.2,
                edgecolor = "k",
                hue   = "Role",
                palette  = role_cmap,
                hue_order = role_order,
                ax      = ax[3, 1],
                legend  = False)

# Turn Off final axis
ax[3, 2].axis("off")

# Limits and labels
for a in [ax[3, 0], ax[3, 1]]:
    a.set_xlim(-2, 100)
    a.set_ylim(-2, 100)
    a.set_xlabel(r"$N_{eff}$ Inputs", fontsize = label_fontsize, labelpad = 5)
    a.set_ylabel(r"$N_{eff}$ Outputs", fontsize = label_fontsize, labelpad = 5)
    # Place Text
    for i, row in right_df.iterrows():
        a.text(x = row["Neff_inputs"],
               y = row["Neff_outputs"],
               s = int(row["Group"].split("_")[0][1:]),
               ha = "center",
               va = "center",
               fontsize = 2)
    # Line
    a.plot(np.linspace(-2, 100, 100), np.linspace(-2, 100, 100), linestyle = "--", color = "k", linewidth = 0.5, zorder = 0)
    # Aspect
    a.set_aspect("equal")

    # Text
    a.text(5, 90, s = "x = y", ha = "left", va = "top", fontsize = tick_fontsize)
    
    # Tick
    a.tick_params(labelsize = tick_fontsize)

    # Despine
    sns.despine(ax = a)


# Main Labels
shx = 0.01
shy = 0
place_panel_label(fig, ax[0, 0], "A", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[0, 1], "B", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[0, 2], "C", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 0], "D", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 1], "E", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 2], "F", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[2, 0], "G", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)

# Scatters' Labels
place_panel_label(fig, ax[3, 0], "H", shx = -0.03, shy = 0.01,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[3, 1], "I", shx = -0.03, shy = 0.01,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[3, 2], "J", shx = 0, shy = 0.01,
                  label_kwargs = label_kwargs)

# Save the figure
plt.savefig(f"{folder}/Figure_05_Supplementary_Flow-Map.pdf",
            dpi = 1200, 
            bbox_inches = "tight")
plt.close()

#------------------------------------------------------------------------------
# Status
print("हो गया दोस्तों!")