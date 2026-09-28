"""
Compiled figure generatrion for neuron Groups.
Connectivity-based neuron classes. The name is helpful to keep all the grouping
files together.

"""
# General
import numpy as np
import pandas as pd
import os
from glob import glob
import pickle as pkl
import random

# Connectome
import navis
import flybrains

# Plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly
import connectome_analysis
import seaborn as sns
sns.set_style("white")

#------------------------------------------------------------------------------
home   = os.environ["PROJECTS_HOME"]
folder = "Analysis_Outputs/Group_Anatomy"
plot_3d  = False

#------------------------------------------------------------------------------
# Load in all the neurons
neuron_df = pd.read_csv(f"{home}/L1_Skeletons/Data_Summary/L1_NeuronInformation_PK.csv")
# Isolating only the neurons needed
neuron_df = neuron_df.loc[(neuron_df["Category"] == "Diff") &
                          (neuron_df["Lineage"]  != "S-PaN")]
# Fill
neuron_df.fillna("NA", inplace = True)
group_dict = dict(zip(neuron_df.bodyId, neuron_df.ConnectivityGroup))

#------------------------------------------------------------------------------
# Read in the lineage colors
with open(f"{home}/L1_Lineages/Data_Helpers/Lineage_Colors.pkl", "rb") as f:
    lineage_colors = pkl.load(f)
lineage_colors["Unbound"] = (0.5, 0.5, 0.5)
# Map the colors
neuron_df["Color"] = neuron_df["Lineage"].map(lineage_colors)
color_dict = dict(zip(neuron_df["bodyId"].astype(str), neuron_df["Color"]))

#------------------------------------------------------------------------------
# Load in all the Right neurons
neuron_paths = glob(f"{home}/L1_Skeletons/Skeletons/Diff_Skeletons/*/Right/*.swc")
# Shuffle for layering that is easier on the eyes
random.shuffle(neuron_paths)
neurons = navis.read_swc(neuron_paths, fmt = "{id}.swc", connector_labels = {0: 7, 1: 8})
neurons = navis.xform_brain(neurons, source = "PK_L1CNS", target = "PK_L1CNSsym")

# Group Dictionary
groups = np.unique(neurons.connectivitygroup)
groups = groups[groups != "NA"]

#------------------------------------------------------------------------------
# Volumes
brain = navis.Volume(flybrains.PK_L1CNSsym.mesh, name = "CNS")
mbr = connectome_analysis.stl_to_navis(f"{home}/L1_Compartment-Atlas/Central-Brain_Compartments/MB/MB_Right.stl")
mbl = connectome_analysis.stl_to_navis(f"{home}/L1_Compartment-Atlas/Central-Brain_Compartments/MB/MB_Left.stl")

# Symemtrize the Left Side and change colors
mbl = navis.xform_brain(mbl, source = "PK_L1CNS", target = "PK_L1CNSsym")
mbl.color = (0.85, 0.85, 0.85, 0.45)
mbr.color = (0.85, 0.85, 0.85, 0.45)

#------------------------------------------------------------------------------
# Counts
# Number of Neurons per group per hemisphere
neuron_counts = neuron_df.groupby(["ConnectivityGroup", "Hemisphere"]).size().reset_index()
neuron_counts.columns = ["ConnectivityGroup", "Hemisphere", "NumNeurons"]

# Number of acronyms per group per hemisphere
acronym_counts = neuron_df.groupby(["ConnectivityGroup", "Hemisphere"]).agg({"VolkerClass": "nunique"}).reset_index()
acronym_counts.columns = ["ConnectivityGroup", "Hemisphere", "NumAcronyms"]

# Merge the counts
merged_counts = pd.merge(neuron_counts, acronym_counts, on = ["ConnectivityGroup", "Hemisphere"])
merged_counts = merged_counts.loc[merged_counts.ConnectivityGroup != "NA"].reset_index(drop = True)

# Easy lookup to plot
right_merged = merged_counts.loc[merged_counts.Hemisphere == "Right"]
left_merged  = merged_counts.loc[merged_counts.Hemisphere == "Left"]
neu_lookup  = dict(zip(right_merged.ConnectivityGroup, right_merged.NumNeurons))
acro_lookup = dict(zip(right_merged.ConnectivityGroup, right_merged.NumAcronyms))

# Mean calculation
mean_df = pd.merge(right_merged, left_merged, on = "ConnectivityGroup",
                   suffixes = ("_Right", "_Left"))
mean_df["MeanNeurons"]  = (mean_df.NumNeurons_Right + mean_df.NumNeurons_Left) / 2
mean_df["MeanAcronyms"] = (mean_df.NumAcronyms_Right + mean_df.NumAcronyms_Left) / 2
mean_neurons  = np.mean(mean_df.MeanNeurons)
mean_acronyms = np.mean(mean_df.MeanAcronyms)

#------------------------------------------------------------------------------
# Plotting two figures
# The first one is with the first 96 groups
# The second one is with the remaining groups and the counts

#------------------------------------------------------------------------------
# FIGURE 01
fig, ax = plt.subplots(figsize = (10 * 4, 8 * 4),
                       nrows = 8, ncols = 10)
plt.subplots_adjust(wspace = 0.15, hspace = 0.2)
ax = ax.flatten()

# Go over every group and plot
for i, group in enumerate(groups[:80]):
    neus = neurons[neurons.connectivitygroup == group]
    navis.plot2d([brain, mbr, mbl, neus],
                 method    = "2d",
                 view      = ("x", "-y"),
                 color     = color_dict,
                 radius    = False,
                 linewidth = 0.5,
                 ax        = ax[i],
                 alpha     = 0.5,
                 rasterize = True)
    # Place the Label
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])
    ax[i].set_frame_on(False)

    # Turn OFF grid
    ax[i].grid(visible = False)

    # Limits
    ax[i].set_xlim(-2000, 107000)
    ax[i].set_ylim(125000, -2500)
    ax[i].set_aspect("equal")
    ax[i].set_title(group[1:], fontsize = 13) # Remove the 'G'

    # Add Text
    ax[i].text(0.05, 0.05,
               f"#Neu: {neu_lookup[group]}\n#Acr: {acro_lookup[group]}",
               transform = ax[i].transAxes,
               ha = "left", va = "bottom", fontsize = 10)

# Saving the figure
plt.savefig(f"{folder}/Figure_05_Supplementary_GroupAnatomy_01.pdf",
            bbox_inches = "tight",
            backend = "pdf", 
            transparent = False,
            dpi = 275)
plt.close()

#------------------------------------------------------------------------------
# FIGURE 02
fig, ax = plt.subplots(figsize = (10 * 4, 8 * 4),
                       nrows = 8, ncols = 10)
plt.subplots_adjust(wspace = 0.15, hspace = 0.2)
ax = ax.flatten()

# Go over every group and plot
for i, group in enumerate(groups[80:160]):
    neus = neurons[neurons.connectivitygroup == group]
    navis.plot2d([brain, mbr, mbl, neus],
                 method    = "2d",
                 view      = ("x", "-y"),
                 color     = color_dict,
                 radius    = False,
                 linewidth = 0.5,
                 ax        = ax[i],
                 alpha     = 0.5,
                 rasterize = True)
    # Place the Label
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])
    ax[i].set_frame_on(False)

    # Turn OFF grid
    ax[i].grid(visible = False)

    # Limits
    ax[i].set_xlim(-2000, 107000)
    ax[i].set_ylim(125000, -2500)
    ax[i].set_aspect("equal")
    ax[i].set_title(group[1:], fontsize = 13) # Remove the 'G'

    # Add Text
    ax[i].text(0.05, 0.05,
               f"#Neu: {neu_lookup[group]}\n#Acr: {acro_lookup[group]}",
               transform = ax[i].transAxes,
               ha = "left", va = "bottom", fontsize = 10)

# Saving the figure
plt.savefig(f"{folder}/Figure_05_Supplementary_GroupAnatomy_02.pdf",
            bbox_inches = "tight",
            backend = "pdf", 
            transparent = False,
            dpi = 275)
plt.close()

#------------------------------------------------------------------------------
# FIGURE 03
fig, ax = plt.subplots(figsize = (10 * 4, 8 * 4),
                       nrows = 8, ncols = 10)
plt.subplots_adjust(wspace = 0.15, hspace = 0.2)
ax = ax.flatten()

# Go over every group and plot
for i, group in enumerate(groups[160:]):
    neus = neurons[neurons.connectivitygroup == group]
    navis.plot2d([brain, mbr, mbl, neus],
                 method    = "2d",
                 view      = ("x", "-y"),
                 color     = color_dict,
                 radius    = False,
                 linewidth = 0.5,
                 ax        = ax[i],
                 alpha     = 0.5,
                 rasterize = True)
    # Place the Label
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])
    ax[i].set_frame_on(False)

    # Turn OFF grid
    ax[i].grid(visible = False)

    # Limits
    ax[i].set_xlim(-2000, 107000)
    ax[i].set_ylim(125000, -2500)
    ax[i].set_aspect("equal")
    ax[i].set_title(group[1:], fontsize = 13) # Remove the 'G'

    # Add Text
    ax[i].text(0.05, 0.05,
               f"#Neu: {neu_lookup[group]}\n#Acr: {acro_lookup[group]}",
               transform = ax[i].transAxes,
               ha = "left", va = "bottom", fontsize = 10)

#--------------------------------------
# Histograms with counts
sns.histplot(right_merged,
             x = "NumNeurons",
             color = "red",
             alpha = 0.35,
             binwidth = 5,
             binrange = (0, 60),
             edgecolor = "black",
             linewidth = 0.45,
             ax = ax[-2]
             )

sns.histplot(left_merged,
             x = "NumNeurons",
             color = "blue",
             alpha = 0.35,
             binwidth = 5,
             binrange = (0, 60),
             edgecolor = "black",
             linewidth = 0.45,
             ax = ax[-2]
             )

ax[-2].set_ylabel("# Groups")
ax[-2].set_xlabel("Number of Neurons")
ax[-2].axvline(x = mean_neurons, color = "purple", linestyle = "--",
               linewidth = 0.75, alpha = 1)
ax[-2].text(12, 50,
            f"Mean: {mean_neurons:.2f}",
            color = "purple",
            ha = "left", va = "center", fontsize = 10)

sns.histplot(right_merged,
             x = "NumAcronyms",
             color = "red",
             alpha = 0.35,
             binwidth = 1,
             binrange = (0, 8),
             edgecolor = "black",
             linewidth = 0.45,
             ax = ax[-1]
             )

sns.histplot(left_merged,
             x = "NumAcronyms",
             color = "blue",
             alpha = 0.35,
             binwidth = 1,
             binrange = (0, 8),
             edgecolor = "black",
             linewidth = 0.45,
             ax = ax[-1]
             )
             
ax[-1].set_ylabel("# Groups")
ax[-1].set_xlabel("Number of Acronyms")
ax[-1].axvline(x = mean_acronyms, color = "purple", linestyle = "--",
               linewidth = 0.75, alpha = 1)
ax[-1].text(4, 54,
            f"Mean: {mean_acronyms:.2f}",
            color = "purple",
            ha = "left", va = "center", fontsize = 10)

# Limits
ax[-2].set_xlim(0, 65)
ax[-1].set_xlim(0, 10)

# Turn OFF grid
ax[-2].grid(visible = False)
ax[-1].grid(visible = False)

# Despine
sns.despine(ax = ax[-2])
sns.despine(ax = ax[-1])

# Ticks
for a in [ax[-1], ax[-2]]:
    a.tick_params(axis  = 'both',
                  which = 'major',
                  length = 4,
                  width  = 1,
                  direction = "out",
                  bottom = True,
                  top    = False,
                  left   = True,
                  right  = False,
                  pad = 1.6,
                  labelsize = 8)

# Turn Off Remaining Items
for a in ax[-8:-2]:
    a.axis("off")

# Saving the figure
plt.savefig(f"{folder}/Figure_05_Supplementary_GroupAnatomy_03.pdf",
            bbox_inches = "tight",
            backend = "pdf", 
            transparent = False,
            dpi = 275)
plt.close()

#------------------------------------------------------------------------------
# 3D Figure
if plot_3d:
    figure_3d = navis.plot3d(
			[neurons, flybrains.PK_L1CNS],
			color        = color_dict,
			legend_group = group_dict,
			width   = 1600,
			height  = 1600,
			inline  = False,
			backend = "plotly")

    # Save the plot
    plotly.offline.plot(figure_3d,
               filename  = f"{folder}/Compiled_Groups_3D.html",
               auto_open = False)

#--------------------------------------------------------------------------
# Status
print("हो गया दोस्तों!")
