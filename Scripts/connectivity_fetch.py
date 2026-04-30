#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:02:21 2025

@author: @KandimallaPrat

Complete connectivity and organization of the L1 Neurons by Acronym.

"""
# General
import pandas as pd
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, leaves_list

# Connectome interface
import pymaid
# Analysis Package
import connectome_analysis

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns
sns.set_style("ticks")

#------------------------------------------------------------------------------
def merge_level(df, level = "VolkerClass"):
    """
    Helper to merge the connectivity at different levels.

    """
    # Grouped
    grouped = df.groupby([f"{level}_pre", f"{level}_post",
                          "Hemisphere_pre", "Hemisphere_post"
                         ])["weight"].sum().reset_index()

    # Merging the name just for easy future use
    grouped["Pre"]  = grouped[f"{level}_pre"].astype(str) + "_" + grouped["Hemisphere_pre"]
    grouped["Post"] = grouped[f"{level}_post"].astype(str) + "_" + grouped["Hemisphere_post"]

    # Normalize the Output Direction
    grouped["total_out"]   = grouped.groupby(["Pre"])["weight"].transform("sum")
    grouped["norm_weight"] = grouped["weight"] / grouped["total_out"]

    return grouped

#------------------------------------------------------------------------------
def plot_connectivity_heatmap(df,
                              norm,
                              ax,
                              label  = True,
                              method = "ward", 
                              metric = "euclidean"):
    """
    Helper to plot the connectivity heatmaps and required levels

    """
    # Get all the items
    items = set(df["Pre"]) | set(df["Post"])
    items = list(items)

    # Convert to matrix
    matrix = df.pivot(index = "Pre", columns = "Post", values = "weight").fillna(0)
    # Reindex to ensure that all items are present and ordered the same
    matrix = matrix.reindex(columns = items, index = items).fillna(0)
    # Fuse
    fused = pd.concat((matrix, matrix.T), axis = 1)

    # Linkage
    fused_linkage = linkage(fused, method = method, metric = metric)
    order         = leaves_list(fused_linkage)

    # Reorder
    matrix = matrix.iloc[order, order]

    #----------------------------------
    # Plotting
    sns.heatmap(
        matrix.fillna(0),
        cmap = "magma",
        norm = norm,
        ax   = ax,
        square = True,
        cbar_kws = {"label" : "Connection Weight",
                    "shrink" : 0.2,
                    "aspect" : 20},
        xticklabels = label,
        yticklabels = label)

    # Tick Params
    if label:
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 4)
        ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 4)


#------------------------------------------------------------------------------
# Folders
home   = os.environ["PROJECTS_HOME"]
folder = "Data_Connectivity"

#------------------------------------------------------------------------------
# Read in the L1 CATMAID Information
http_user     = os.environ["L1_USERNAME"]
http_password = os.environ["L1_PASSWORD"]
api_token     = os.environ["L1_TOKEN"]
server        = os.environ["L1_SERVER"]

#------------------------------------------------------------------------------
# Read in the lineage acronyms
class_df = pd.read_csv(f"{home}/L1_Skeletons/Data_Summary/L1_NeuronInformation_PK.csv")
class_df = class_df.loc[class_df["Category"] == "Diff"]

# Drop the Large Unneeded Identifiers and Monikers
# Drop the Category since connections are by definition only between differentiated
# neurons
class_df.drop(columns = ["Identifier", "Moniker", "Category"], inplace = True)

# LAZY way to do this 
# Creating a pre and post df
pre_df   = class_df.rename(columns = {name : f"{name}_pre" for name in class_df.columns})
post_df  = class_df.rename(columns = {name : f"{name}_post" for name in class_df.columns})

# Integers
pre_df["bodyId_pre"]   = pre_df["bodyId_pre"].astype(int)
post_df["bodyId_post"] = post_df["bodyId_post"].astype(int)

#------------------------------------------------------------------------------
# Connect to L1 CATMAID
larval_analysis = connectome_analysis.CATMAIDConnect(
                            http_user,
                            http_password,
                            api_token,
                            server)

#------------------------------------------------------------------------------
# Neuron IDs
neurons = pymaid.find_neurons(annotations = ["Kandimalla-etal.", "Diff"],
                              intersect = True)
print(f"\nWorking on {len(neurons)} neurons.\n")
neu_ids = list(neurons.id)

# Fetch Connectivity
conn = larval_analysis.fetch_connectivity(neu_ids, neu_ids, min_weight = 0)

# Reorg
conn_df = conn[1]
conn_df["PreSynaptic_ID"]  = conn_df["PreSynaptic_ID"].astype(int)
conn_df["PostSynaptic_ID"] = conn_df["PostSynaptic_ID"].astype(int)

# Rename
conn_df.rename(columns = {"PreSynaptic_ID" : "bodyId_pre",
                          "PostSynaptic_ID" : "bodyId_post"},
               inplace = True)

#------------------------------------------------------------------------------
# Merge
conn_df = pd.merge(conn_df, pre_df, 
                   left_on  = "bodyId_pre",
                   right_on = "bodyId_pre",
                   how      = "left")
conn_df = pd.merge(conn_df, post_df, 
                   left_on  = "bodyId_post",
                   right_on = "bodyId_post",
                   how      = "left")

# Integers
conn_df["bodyId_pre"]  = conn_df["bodyId_pre"].astype(int)
conn_df["bodyId_post"] = conn_df["bodyId_post"].astype(int)

# Reset
conn_df.reset_index(drop = True, inplace = True)

# Save the Matrix
conn_df.to_csv("Data_Connectivity/Connectivity.csv", index = False)
conn_df.to_parquet("Data_Connectivity/Connectivity.parquet", index = False,
                   engine = "pyarrow")

#------------------------------------------------------------------------------
# Merging at Different Levels
# Lineage
lineage_df = merge_level(conn_df, level = "Lineage")
# Acronym
acronym_df = merge_level(conn_df, level = "VolkerClass")
# Group
group_df   = merge_level(conn_df, level = "ConnectivityGroup")

# Save it
lineage_df.to_csv("Data_Connectivity/Connectivity_Lineage.csv", index = False)
lineage_df.to_parquet("Data_Connectivity/Connectivity_Lineage.parquet", index = False,
                   engine = "pyarrow")

acronym_df.to_csv("Data_Connectivity/Connectivity_Acronym.csv", index = False)
acronym_df.to_parquet("Data_Connectivity/Connectivity_Acronym.parquet", index = False,
                   engine = "pyarrow")

group_df.to_csv("Data_Connectivity/Connectivity_Group.csv", index = False)
group_df.to_parquet("Data_Connectivity/Connectivity_Group.parquet", index = False,
                   engine = "pyarrow")

#--------------------------------------
# Just adding a Cluster case as well
conn_df.fillna("NA", inplace = True)
conn_df["Cluster_pre"]  = conn_df["Lineage_pre"] + "_" + conn_df["Cluster_pre"].astype(str) + "_" + conn_df["Instance_pre"]
conn_df["Cluster_post"] = conn_df["Lineage_post"] + "_" + conn_df["Cluster_post"].astype(str) + "_" + conn_df["Instance_post"]

# Merging
cluster_df = merge_level(conn_df, level = "Cluster")

# Save it
cluster_df.to_csv("Data_Connectivity/Connectivity_Cluster.csv", index = False)
cluster_df.to_parquet("Data_Connectivity/Connectivity_Cluster.parquet", index = False,
                   engine = "pyarrow")

#--------------------------------------
# Just for safety, the bodyId
conn_merged = merge_level(conn_df, level = "bodyId")
# Save it
conn_merged.to_csv("Data_Connectivity/Connectivity_bodyId.csv", index = False)
conn_merged.to_parquet("Data_Connectivity/Connectivity_bodyId.parquet", index = False,
                       engine = "pyarrow")

#------------------------------------------------------------------------------
# Plotting the Distributions
# ZIP
pairs = [ 
    (lineage_df,  "Lineage"),
    (group_df,    "Group"),
    (acronym_df,  "Acronym"),
    (cluster_df,  "Cluster"),
    (conn_merged, "Neuron")
]

# Palette
palette = sns.color_palette("tab10", n_colors = len(pairs))
palette = dict(zip(["Lineage", "Group", "Acronym", "Cluster", "Neuron"],
                    palette))

# Create Figure
fig, ax = plt.subplots(figsize = (8.5, 3.25), ncols = 2)

# Plotting the ECDFs for Complete Weight
for sub_df, label in pairs:
    sns.ecdfplot(data = sub_df,
                 x    = "weight",
                 ax   = ax[0],
                 linewidth = 1,
                 label = label,
                 alpha = 0.75,
                 color = palette[label])

# Plotting the ECDFs for Normalized Weight
for sub_df, label in pairs:
    sns.ecdfplot(data = sub_df,
                 x    = "norm_weight",
                 ax   = ax[1],
                 linewidth = 1,
                 label = label,
                 alpha = 0.75,
                 color = palette[label])

# Params
for a in ax:
    # Despine
    sns.despine(ax = a)
    # Params
    a.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = 6,
        length    = 1.5,
        width     = 0.5,
        pad       = 1.5)
    # Labels
    a.set_ylabel("Cumulative Proportion", fontsize = 8)
    # Scale
    a.set_xscale("log")

# Labels
ax[0].set_xlabel("Weight", fontsize = 8)
ax[1].set_xlabel("Normalized Output Weight", fontsize = 8)

# Limits
ax[0].set_xlim(-50, 3200)
ax[1].set_xlim(1e-4, 1)

# Legend
ax[1].legend(title = "Connectivity Level",
            bbox_to_anchor = (1.05, 1),
            loc = "upper left",
            fontsize = 8,
            title_fontsize = 8)

# Save it
plt.savefig("Data_Connectivity/Figure_05_Supplementary_Conn-Weight-Distribution.pdf",
            dpi = 600,
            bbox_inches = "tight")
plt.close()

#------------------------------------------------------------------------------
# Plot Heatmaps

#--------------------------------------
# Lineage
fig, ax = plt.subplots(figsize = (30, 30))
plot_connectivity_heatmap(lineage_df,
                          norm = PowerNorm(gamma = 0.15, vmin = 0, vmax = 3200),
                          ax = ax)
# Labels
ax.set_xlabel("PostSynaptic Lineage", fontsize = 18)
ax.set_ylabel("PreSynaptic Lineage", fontsize = 18)

# Save it
plt.savefig("Data_Connectivity/Figure_05_Supplementary_Connectivity-Lineage.pdf",
            dpi = 600,
            bbox_inches = "tight")
plt.close()

#--------------------------------------
# Acronym
fig, ax = plt.subplots(figsize = (30, 30))
plot_connectivity_heatmap(acronym_df,
                          norm = PowerNorm(gamma = 0.15, vmin = 0, vmax = 650),
                          ax = ax,
                          label = False)
# Labels
ax.set_xlabel("PostSynaptic Class", fontsize = 18)
ax.set_ylabel("PreSynaptic Class", fontsize = 18)

# Save it
plt.savefig("Data_Connectivity/Figure_05_Supplementary_Connectivity-Acronym.pdf",
            dpi = 600,
            bbox_inches = "tight")

plt.close()

#--------------------------------------
# Group
fig, ax = plt.subplots(figsize = (35, 35))
plot_connectivity_heatmap(group_df,
                          norm = PowerNorm(gamma = 0.15, vmin = 0, vmax = 2500),
                          ax = ax)
# Labels
ax.set_xlabel("PostSynaptic Group", fontsize = 18)
ax.set_ylabel("PreSynaptic Group", fontsize = 18)

# Save it
plt.savefig("Data_Connectivity/Figure_05_Supplementary_Connectivity-Group.pdf",
            dpi = 600,
            bbox_inches = "tight")

plt.close()

#--------------------------------------
# Status
print("हो गया दोस्तों!")