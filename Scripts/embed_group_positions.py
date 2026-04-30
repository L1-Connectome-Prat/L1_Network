"""
Determining overall, dendritic, and axonal synaptic 'position' of connectivity groups

"""
import numpy as np
import pandas as pd
import os
from glob import glob

import navis
import flybrains
import connectome_analysis

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import seaborn as sns
sns.set_style("ticks")
from prats_helpers import place_panel_label

#------------------------------------------------------------------------------
home = os.environ["PROJECTS_HOME"]
output_folder = "Analysis_Outputs/Group_Anatomy"

#------------------------------------------------------------------------------
# All differentiated skeletons
neurons = glob(f"{home}/L1_Skeletons/Skeletons/Diff_Skeletons/*/*/*.swc")
neurons = navis.read_swc(neurons,
                         fmt = "{id}.swc",
                         connector_labels = {0 : 7, 1 : 8})

# Add size information for later
cnt = pd.DataFrame(zip(neurons.connectivitygroup, neurons.hemisphere)).value_counts().reset_index()
cnt.sort_values(by = [0, 1], inplace = True)
cnt.columns = ["Group", "Hemisphere", "Size"]

#------------------------------------------------------------------------------
# Get all the groups
groups = np.unique(neurons.connectivitygroup)
groups = groups[groups != "NA"]

# Keep only the neurons with groups 
neurons = neurons[neurons.connectivitygroup != "NA"]
# There are some incomplete strange neurons
neurons = neurons[pd.Series(neurons.n_connectors).fillna(0) > 0]

#------------------------------------------------------------------------------
# Split Axon and Dendrites
pieces    = navis.split_axon_dendrite(neurons, metric = "synapse_flow_centrality")
dendrites = pieces[pieces.compartment == "dendrite"]
axons     = pieces[pieces.compartment == "axon"]

# We only want to measure the centroids for the T-Bars in the Axons and the PSDs
# in the dendrites
# For easy loop later, we just discard these features here
# 0 : 7 --> 7 is presynapses i.e. Tbars
# 1 : 8 --> 8 is postsynapses i.e. PSDs
# Even though this method does not prop to all neurons, it is ok at the level we
# are looking at
# PSDs
conns = dendrites.connectors
conns = conns.loc[conns["type"] == 1]
dendrites.connectors = conns
# TBars
conns = axons.connectors
conns = conns.loc[conns["type"] == 0]
axons.connectors = conns

# Set case to use later
neurons.case   = "Neuron"
dendrites.case = "Dendrite"
axons.case     = "Axon"

#------------------------------------------------------------------------------
# Loop over each group and find the centroids for each condition
pos = []
for group in groups:
    for hemisphere in ["Right", "Left"]:
        # Loop through each case
        for case in [neurons, dendrites, axons]:
            # Get the group
            grp = case[(case.connectivitygroup == group) & 
                       (case.hemisphere == hemisphere)]
            
            # Centroid
            try:
                centroid = np.mean(grp.connectors[["x", "y", "z"]], axis = 0).values
            except:
                centroid = [0, 0, 0]

            # Append
            pos.append({"Group" : group,
                        "Hemisphere" : hemisphere,
                        "Fragment"   : case.case,
                        "Pos_x" : centroid[0],
                        "Pos_y" : centroid[1],
                        "Pos_z" : centroid[2]})

# Convert to DataFrame
pos = pd.DataFrame(pos)

#------------------------------------------------------------------------------
# Symmetrize
pos[["Pos_x", "Pos_y", "Pos_z"]] = navis.xform_brain(pos[["Pos_x", "Pos_y", "Pos_z"]].values,
                                                        source = "PK_L1CNS",
                                                        target = "PK_L1CNSsym")

# Merge
pos = pos.merge(cnt, on = ["Group", "Hemisphere"], how = "left")

# NOTE : Manually fixing g011 Right Dendrites
p  = pos.loc[(pos["Group"] == "g011") & (pos["Hemisphere"] == "Left") & (pos["Fragment"] == "Dendrite"), ["Pos_x", "Pos_y", "Pos_z"]]
p2 = navis.mirror_brain(p.values, template = "PK_L1CNS", warp = True)
pos.loc[(pos["Group"] == "g011") & (pos["Hemisphere"] == "Right") & (pos["Fragment"] == "Dendrite"), ["Pos_x", "Pos_y", "Pos_z"]] = p2

#------------------------------------------------------------------------------
# Save
# This is run from the main folder - so be careful with the paths
pos.to_csv(f"{output_folder}/Group_Positions.csv",
           index = False)
pos.to_parquet(f"{output_folder}/Group_Positions.parquet",
               engine = "pyarrow",
               index = False)

#------------------------------------------------------------------------------
# Axon Dendrite Distance
dend = pos.loc[pos["Fragment"] == "Dendrite"].set_index(["Group", "Hemisphere"])
axon = pos.loc[pos["Fragment"] == "Axon"].set_index(["Group", "Hemisphere"])

# Scale it to um
dend[["Pos_x", "Pos_y", "Pos_z"]] /= 1000
axon[["Pos_x", "Pos_y", "Pos_z"]] /= 1000

# Join the two
flow = dend[["Pos_x", "Pos_y", "Pos_z"]].join(
       axon[["Pos_x", "Pos_y", "Pos_z"]],
       lsuffix = "_dend",
       rsuffix = "_axon")

# Euclidean Distance 
flow["distance"] = np.sqrt(
    (flow.Pos_x_dend - flow.Pos_x_axon)**2 +
    (flow.Pos_y_dend - flow.Pos_y_axon)**2 +
    (flow.Pos_z_dend - flow.Pos_z_axon)**2)

flow = flow.reset_index().merge(
    pos.loc[pos["Fragment"] == "Neuron", ["Group", "Hemisphere", "Size"]],
    on = ["Group", "Hemisphere"])

# Use the Right Hemisphere to Mark the Quartiles
# This is what we will be plotting
fr = flow.loc[flow["Hemisphere"] == "Right"].reset_index(drop = True)
cuts = [fr["distance"].quantile(q) for q in [0.25, 0.5, 0.75]]

# Calculating the Delta Z to show the depth of the projection
# Axon - Dendrite (posterior projecting = +ve and Anterior Projecting = -ve)
fr["dz"] = fr["Pos_z_axon"] - fr["Pos_z_dend"]
fr["Q"]  = pd.qcut(fr["distance"], q = 4, labels = [1, 2, 3, 4])

# Merge the Q into the Flow
flow = flow.merge(fr[["Group", "dz", "Q"]],
                  on = "Group",
                  how = "left")

# Save the Flow
flow.to_csv(f"{output_folder}/Group_Dendrite-to-Axon_Distance.csv",
           index = False)
flow.to_parquet(f"{output_folder}/Group_Dendrite-to-Axon_Distance.parquet",
               engine = "pyarrow",
               index = False)

#------------------------------------------------------------------------------
# Making a plot of the flow
neuropil = navis.Volume(flybrains.PK_L1CBNeuropilsym.mesh)
neuropil.color = (0, 0, 0, 0.25)

# Load the MBs
mbr = connectome_analysis.stl_to_navis(f"{home}/L1_Compartment-Atlas/Central-Brain_Compartments/MB/MB_Right.stl")
mbl = connectome_analysis.stl_to_navis(f"{home}/L1_Compartment-Atlas/Central-Brain_Compartments/MB/MB_Left.stl")

# Symmetrize 
mbr = navis.xform_brain(mbr, source = "PK_L1CNS", target = "PK_L1CNSsym")
mbl = navis.xform_brain(mbl, source = "PK_L1CNS", target = "PK_L1CNSsym")

# Set color
mbr.color = (0, 0, 0, 0.15)
mbl.color = (0, 0, 0, 0.15)

# Params
sizes = (5, 75)
cmap  = {"Right" : "red", "Left" : "blue"}
fmap  = {"Dendrite" : "green", "Axon" : "magenta"}
alpha = 0.25
linewidth = 0.3

# Normalization for Depth
vabs    = np.percentile(np.abs(fr["dz"]), 95)
dz_norm = TwoSlopeNorm(vmin = -vabs, vcenter = 0, vmax = vabs)
dz_cmap = cm.ScalarMappable(norm = dz_norm, cmap = "vlag")

# Arrow thickness: group size (sqrt-scaled)
smin, smax = fr["Size"].min(), fr["Size"].max()
def size2lw(s):
    f = (np.sqrt(s) - np.sqrt(smin)) / (np.sqrt(smax) - np.sqrt(smin))
    return 0.3 + f * 2.0

#------------------------------------------------------------------------------
# Define VNC groups
vnc = connectome_analysis.stl_to_navis(f"{home}/L1_Compartment-Atlas/Ventral-Nerve-Cord/VNC.stl")
in_vnc = navis.in_volume(neurons, mode = "IN", volume = vnc)

# Threshold to 30 connectors
sel = in_vnc[in_vnc.n_connectors > 30]
vnc_groups = np.unique(sel.connectivitygroup)

# Save it
np.save(f"{output_folder}/Groups_w-VNC-innervation.npy", vnc_groups)

# Merge
fr["VNC"] = fr["Group"].isin(vnc_groups)
fr["VNC"] = fr["VNC"].replace({True : "VNC", False : "Non-VNC"})
vnc_cmap  = {"VNC" : "orange", "Non-VNC" : "teal"}

# Panel Labels
label_kwargs = dict(fontsize = 16, va = "top", ha = "left")

#------------------------------------------------------------------------------
# Three Panels
fig = plt.figure(figsize = (12, 12))
# Create Grid
gs = gridspec.GridSpec(12, 12)

# Top Row
ax_a = fig.add_subplot(gs[0:4, 0:4])
ax_b = fig.add_subplot(gs[0:4, 4:8])
ax_c = fig.add_subplot(gs[0:4, 8:12])

# Middle Row
ax_d = fig.add_subplot(gs[5:8, 0:3])
ax_e = fig.add_subplot(gs[5:8, 4:7])

# Bottom Row
ax_h = fig.add_subplot(gs[9:12, 0:3])
ax_i = fig.add_subplot(gs[9:12, 3:6])
ax_j = fig.add_subplot(gs[9:12, 6:9])
ax_k = fig.add_subplot(gs[9:12, 9:12])

#--------------------------------------
# Dendritic Positions
sns.scatterplot(data  = pos.loc[pos["Fragment"] == "Dendrite"],
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "Hemisphere",
                palette  = cmap,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = linewidth,
                edgecolor = "k",
                legend  = True,
                ax = ax_a)

# Names
for i, row in pos.loc[pos["Fragment"] == "Dendrite"].iterrows():
    ax_a.text(row["Pos_x"], row["Pos_y"],
               int(row["Group"][1:]), fontsize = 2,
               ha = "center", va = "center")
    

# Axonal Positions
sns.scatterplot(data  = pos.loc[pos["Fragment"] == "Axon"],
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "Hemisphere",
                palette  = cmap,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = linewidth,
                edgecolor = "k",
                legend  = True,
                ax = ax_b)

# Names
for i, row in pos.loc[pos["Fragment"] == "Axon"].iterrows():
    ax_b.text(row["Pos_x"], row["Pos_y"],
               int(row["Group"][1:]), fontsize = 2,
               ha = "center", va = "center")

# Neuronal Positions
sns.scatterplot(data  = pos.loc[pos["Fragment"] == "Neuron"],
                x     = "Pos_x",
                y     = "Pos_y",
                hue   = "Hemisphere",
                palette  = cmap,
                size     = "Size",
                sizes    = sizes,
                alpha    = alpha,
                linewidth = linewidth,
                edgecolor = "k",
                legend  = True,
                ax = ax_c)

# Names
for i, row in pos.loc[pos["Fragment"] == "Neuron"].iterrows():
    ax_c.text(row["Pos_x"], row["Pos_y"],
               int(row["Group"][1:]), fontsize = 2,
               ha = "center", va = "center")

#--------------------------------------
# Loop
for a in [ax_a, ax_b, ax_c, ax_h, ax_i, ax_j, ax_k]:
    # Plot the neuropil
    navis.plot2d([mbr, mbl, neuropil],
              ax     = a,
              view   = ("x", "y"),
              method = "2d",
              volume_outlines = True,
              linewidth       = 0.05,
              rasterize       = True)

    # Axis is off
    a.axis("off")
    a.legend().set_visible(False)
    # Aspect
    a.set_aspect("equal")
    # Invert
    a.invert_yaxis()

# Titles
ax_a.set_title("Dendritic Position", fontsize = 10, y = 0.98)
ax_b.set_title("Axonal Position", fontsize = 10, y = 0.98)
ax_c.set_title("Neuron Position", fontsize = 10, y = 0.98)

#--------------------------------------
# Axon to Dendrite Distances
sns.ecdfplot(data = flow,
             x    = "distance",
             hue  = "Hemisphere",
             palette = cmap,
             ax    = ax_d, 
             alpha = 0.86,
             linewidth = 0.5,
             legend    = True)
# Despine
sns.despine(ax = ax_d)
ax_d.set_xlabel(r"Dendrite-to-Axon Distance ($\mu$m)", fontsize = 8)
ax_d.set_ylabel("Cumulative Proportion", fontsize = 8)
ax_d.set_xlim(0, 100)
ax_d.set_ylim(0, 1)
ax_d.tick_params(labelsize = 6)
sns.move_legend(ax_d, loc = "upper right", bbox_to_anchor = (0.95, 0.95),
                fontsize = 6, title = "")

# Quartile Lines
for cut in cuts:
    ax_d.axvline(x = cut, color = "grey", linestyle = "--", linewidth = linewidth)
    ax_d.text(95, 0.15 - (cuts.index(cut) * 0.05),
              f"Q{cuts.index(cut) + 1}: {cut:.1f} $\mu$m",
              fontsize = 6,
              color = "grey",
              ha = "right", va = "center")

#--------------------------------------
# Anatomy Panels for the Flow Directions
# Appending the Max so that it is easy to make the splits
ax_list = [ax_h, ax_i, ax_j, ax_k]
for q in range(1, 5):
    # Subset
    sub1 = fr.loc[fr["Q"] == q]
    grps = np.unique(sub1["Group"])
    # Posn
    sub2 = pos.loc[(pos["Fragment"] != "Neuron") &
                  (pos["Hemisphere"] == "Right") &
                  (pos["Group"]).isin(grps)]

    # Plot
    sns.scatterplot(data = sub2,
                    x = "Pos_x",
                    y = "Pos_y",
                    hue     = "Fragment",
                    palette = fmap,
                    s       = 10,
                    alpha   = alpha * 2.5,
                    linewidth = linewidth,
                    edgecolor = "k",
                    legend = False,
                    ax     = ax_list[q - 1])

    # Looping over each dendrite-axon
    for _, row in sub1.iterrows():
        ax_list[q - 1].annotate(
            "",
            xytext = (row["Pos_x_dend"] * 1000, row["Pos_y_dend"] * 1000),
            xy     = (row["Pos_x_axon"] * 1000, row["Pos_y_axon"] * 1000),
            arrowprops = dict(
                arrowstyle = "-|>",
                color      = dz_cmap.to_rgba(row["dz"]),
                lw         = size2lw(row["Size"]),
                alpha      = 1,
                shrinkA    = 0,
                shrinkB    = 0,
                mutation_scale = 6),
            zorder = 0
        )

        # Add the dendrite group info
        ax_list[q - 1].text(row["Pos_x_dend"] * 1000,
                           row["Pos_y_dend"] * 1000,
                           int(row["Group"][1:]),
                           fontsize = 2,
                           ha = "center",
                           va = "center")

#--------------------------------------
# Colorbar for dz
cax = fig.add_axes([0.42, 0.07, 0.10, 0.006])
cbar = plt.colorbar(dz_cmap, cax = cax, orientation = "horizontal")
cbar.set_label(r"Projection Direction $\Delta$Z ($\mu$m)" +
               "\n" +
               r"$\leftarrow$ Anterior | Posterior $\rightarrow$",
               fontsize = 7)
cbar.ax.tick_params(labelsize = 5)


#--------------------------------------
# Size + color legend next to colorbar
circle_sizes = [2, 10, 30]
leg1 = fig.legend(
    handles = [Line2D([0],[0], color = "gray", lw = size2lw(s), label = f"n = {s}") for s in circle_sizes],
    loc = "lower center",
    bbox_to_anchor = (0.6, 0.065),
    ncol = 3,
    fontsize = 6,
    frameon = False,
    handlelength  = 2,
    handletextpad = 0.4,
    columnspacing = 1.0)

# Row 2: dendrite/axon markers
fig.legend(
    handles = [
        Line2D([0],[0], marker = "o", color = "w", markerfacecolor = "green",   markersize = 3, label = "Dendritic Posn."),
        Line2D([0],[0], marker = "o", color = "w", markerfacecolor = "magenta", markersize = 3, label = "Axonal Posn."),
    ],
    loc = "lower center",
    bbox_to_anchor = (0.6, 0.045),
    ncol     = 2,
    fontsize = 6,
    frameon  = False,
    handletextpad = 0.4,
    columnspacing = 1.0)

#--------------------------------------
# Scatter for distance and dz
sns.scatterplot(fr,
                x = "distance",
                y = "dz",
                hue = "VNC",
                palette = vnc_cmap,
                s = 5,
                ax = ax_e,
                legend = True)

# Line
ax_e.axhline(y = 0, color = "grey", linestyle = "--", linewidth = linewidth)

# Despine
sns.despine(ax = ax_e)
ax_e.set_xlabel(r"Dendrite-to-Axon Distance ($\mu$m)", fontsize = 8)
ax_e.set_ylabel("Projection Direction ($\mu$m)", fontsize = 8)
ax_e.set_xlim(0, 100)
ax_e.set_ylim(-40, 80)
ax_e.tick_params(labelsize = 6)
sns.move_legend(ax_e, loc = "upper left", bbox_to_anchor = (0.95, 0.95),
                fontsize = 6, title = "")

# Direction Indicators
ax_e.text(90, -5,
          "$\\leftarrow$ Anterior", fontsize = 6,
          ha = "center",
          va = "top",
          rotation = 90)

ax_e.text(90, 5,
          "Posterior $\\rightarrow$", fontsize = 6,
          ha = "center",
          va = "bottom",
          rotation = 90)

# Main Labels
shx = -0.005
shy = 0.01
place_panel_label(fig, ax_a, "Ai", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax_b, "ii", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax_c, "iii", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax_d, "B", shx = -0.02, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax_e, "C", shx = -0.02, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax_h, "Di", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax_i, "ii", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax_j, "iii", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax_k, "iv", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)

# Save
plt.savefig(f"{output_folder}/Figure_05_Supplementary_Group-Positions.pdf",
            dpi = 450,
            bbox_inches = "tight")

print("हो गया दोस्तों!")