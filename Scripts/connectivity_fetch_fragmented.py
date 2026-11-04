#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: @KandimallaPrat

Connectivity of the L1 Neurons broken down by fragment (axon / dendrite) and
by neuropil compartment (level1 and level2).

Each synaptic link (one presynaptic node -> one postsynaptic node through a
connector) is labelled with:

    Fragment_pre  - fragment of the presynaptic neuron holding the pre node
    Fragment_post - fragment of the postsynaptic neuron holding the post node
    Compartment   - compartment containing the connector position

Splits prefer Winding et al. (2023) tags (``mw axon split``) where present
and fall back to navis.split_axon_dendrite via synapse flow centrality, same
as L1_Neuron-Typing/Scripts/identifiers_split.py.

Fragment labels:
    axon / dendrite            - from either split
    linker / cellbodyfiber     - extra pieces from the navis fallback
    unsplit                    - both split strategies failed

Compartment labels:
    <name>                     - connector inside exactly one compartment
    <name1>|<name2>            - connector inside overlapping compartments
    None                       - connector outside all compartments

Every link gets exactly one row, so summing the weights by
(bodyId_pre, bodyId_post) recovers the neuron x neuron connectivity.

"""
# General
import pandas as pd
import numpy as np
import os

# Connectome interface
import pymaid
import navis
# Analysis Package
import connectome_analysis

#------------------------------------------------------------------------------
# Folders
home   = os.environ["PROJECTS_HOME"]
folder = "Data_Connectivity"
atlas  = f"{home}/L1_Compartment-Atlas"
levels = ["level1", "level2"]

#------------------------------------------------------------------------------
def load_compartments(level):
    """
    Load compartments listed in the atlas level file.

    """
    compartment_names = np.load(f"{atlas}/{level.capitalize()}.npy",
                                allow_pickle = True)

    compartments = []
    for comp in compartment_names:
        stl = connectome_analysis.stl_to_navis(f"{atlas}/{comp}")
        stl.fix_normals()
        stl.name = comp.split("/")[-1].replace(".stl", "")
        compartments.append(stl)

    return compartments

#------------------------------------------------------------------------------
def node_fragments(skel, metric = "synapse_flow_centrality"):
    """
    Label every node of a skeleton by fragment. Prefer Winding cuts where
    tagged; fall back to navis.split_axon_dendrite.

    Returns (Series node_id -> fragment, method used)

    """
    node_ids = skel.nodes.node_id.values

    try:
        cut_posn = skel.tags["mw axon split"][0]
        # navis.cut_skeleton returns (distal, proximal) — axon is distal
        axon, dendrite = navis.cut_skeleton(skel, cut_posn)

        labels = pd.Series("unsplit", index = node_ids)
        labels[labels.index.isin(dendrite.nodes.node_id)] = "dendrite"
        labels[labels.index.isin(axon.nodes.node_id)]     = "axon"
        return labels, "winding"

    except (KeyError, IndexError, TypeError):
        pass

    try:
        labelled = navis.split_axon_dendrite(skel.copy(),
                                             metric     = metric,
                                             label_only = True)
        # compartment comes back categorical; plain strings so fillna works
        labels = pd.Series(labelled.nodes.compartment.astype(str).values,
                           index = labelled.nodes.node_id.values)
        return labels.reindex(node_ids).fillna("unsplit"), "navis"

    except Exception:
        return pd.Series("unsplit", index = node_ids), "failed"

#------------------------------------------------------------------------------
def label_compartments(cn_df, compartments):
    """
    Assign each connector position to the compartment(s) containing it.

    """
    points = cn_df[["x", "y", "z"]].values
    hits   = np.zeros((len(points), len(compartments)), dtype = bool)

    for i, vol in enumerate(compartments):
        hits[:, i] = navis.in_volume(points, vol)

    names  = np.array([vol.name for vol in compartments])
    labels = ["|".join(names[row]) if row.any() else "None" for row in hits]

    return pd.Series(labels, index = cn_df["connector_id"].values), hits.sum(axis = 1)

#------------------------------------------------------------------------------
# Read in the L1 CATMAID Information
http_user     = os.environ["L1_USERNAME"]
http_password = os.environ["L1_PASSWORD"]
api_token     = os.environ["L1_TOKEN"]
server        = os.environ["L1_SERVER"]

#------------------------------------------------------------------------------
# Read in the lineage acronyms (same as connectivity_fetch.py)
class_df = pd.read_csv(f"{home}/L1_Skeletons/Data_Summary/L1_NeuronInformation_PK.csv")
class_df = class_df.loc[class_df["Category"] == "Diff"]
class_df.drop(columns = ["Identifier", "Moniker", "Category"], inplace = True)

pre_df   = class_df.rename(columns = {name : f"{name}_pre" for name in class_df.columns})
post_df  = class_df.rename(columns = {name : f"{name}_post" for name in class_df.columns})

pre_df["bodyId_pre"]   = pre_df["bodyId_pre"].astype(int)
post_df["bodyId_post"] = post_df["bodyId_post"].astype(int)

#------------------------------------------------------------------------------
# Connect to L1 CATMAID
rm = pymaid.CatmaidInstance(server        = server,
                            api_token     = api_token,
                            http_user     = http_user,
                            http_password = http_password)

#------------------------------------------------------------------------------
# Neurons, pulled fresh with all connectors
neurons = pymaid.find_neurons(annotations = ["Kandimalla-etal.", "Diff"],
                              intersect = True)
print(f"\nWorking on {len(neurons)} neurons.\n")
skeletons = pymaid.get_neurons(list(neurons.id))

#------------------------------------------------------------------------------
# Fragment label for every connector link of every neuron
link_dfs = []
methods  = {}
for skel in navis.config.tqdm(skeletons, desc = "Splitting"):
    labels, method = node_fragments(skel)
    methods[int(skel.id)] = method

    cn = skel.connectors[["node_id", "connector_id", "type", "x", "y", "z"]].copy()
    cn["bodyId"]   = int(skel.id)
    cn["Fragment"] = labels.reindex(cn["node_id"].values).fillna("unsplit").values
    link_dfs.append(cn)

links = pd.concat(link_dfs, ignore_index = True)
links["type"] = links["type"].astype(int)

methods = pd.Series(methods)
print(f"\nSplit methods:\n{methods.value_counts()}\n")

#------------------------------------------------------------------------------
# Pair presynaptic (type 0) with postsynaptic (type 1) links through the connector
pre_links  = links.loc[links["type"] == 0].rename(columns = {"node_id"  : "node_pre",
                                                             "bodyId"   : "bodyId_pre",
                                                             "Fragment" : "Fragment_pre"})
post_links = links.loc[links["type"] == 1].rename(columns = {"node_id"  : "node_post",
                                                             "bodyId"   : "bodyId_post",
                                                             "Fragment" : "Fragment_post"})

conn_links = pd.merge(pre_links[["connector_id", "x", "y", "z",
                                 "bodyId_pre", "node_pre", "Fragment_pre"]],
                      post_links[["connector_id", "bodyId_post",
                                  "node_post", "Fragment_post"]],
                      on  = "connector_id",
                      how = "inner")

print(f"{len(conn_links)} synaptic links between {len(skeletons)} neurons.\n")

#------------------------------------------------------------------------------
# Compartment labels on unique connector positions
connectors = conn_links.drop_duplicates("connector_id")[["connector_id", "x", "y", "z"]]

for level in levels:
    compartments = load_compartments(level)
    comp_map, n_hits = label_compartments(connectors, compartments)
    conn_links[f"Compartment_{level}"] = conn_links["connector_id"].map(comp_map)

    print(f"{level}: {len(connectors)} connectors | "
          f"outside all = {(n_hits == 0).sum()} | "
          f"overlapping = {(n_hits > 1).sum()}")

# Save the raw links so relabelling never needs a refetch
conn_links.to_parquet(f"{folder}/Connectivity_Fragmented_Links.parquet",
                      index = False, engine = "pyarrow")

#------------------------------------------------------------------------------
# Collapse to fragment x compartment connectivity per level
for level in levels:
    conn_df = (conn_links.groupby(["bodyId_pre", "Fragment_pre",
                                   "bodyId_post", "Fragment_post",
                                   f"Compartment_{level}"])
                         .size()
                         .reset_index(name = "weight")
                         .rename(columns = {f"Compartment_{level}" : "Compartment"}))

    # Merge
    conn_df = pd.merge(conn_df, pre_df,  on = "bodyId_pre",  how = "left")
    conn_df = pd.merge(conn_df, post_df, on = "bodyId_post", how = "left")

    # Save
    conn_df.to_csv(f"{folder}/Connectivity_Fragmented_{level}.csv", index = False)
    conn_df.to_parquet(f"{folder}/Connectivity_Fragmented_{level}.parquet",
                       index = False, engine = "pyarrow")

#--------------------------------------
# Status
print("हो गया दोस्तों!")
