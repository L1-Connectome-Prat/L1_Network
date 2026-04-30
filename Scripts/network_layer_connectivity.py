"""
Anatomical connectivity diagram among L1 Groups -- compartment-coloured variant.

Positions    : Pos_x / Pos_y from Group_Hybrid-Placement (one row per Group x Hemi)
Node fill    : Compartment (Mixed = grey; left hemisphere greyed out)
Node outline : Fragment   (Dendrite = solid black halo,
                           Axon     = dashed black outline,
                           Neuron   = none)
Node size    : number of neurons (Size column)
Node alpha   : self-connection norm_weight (stronger recurrence = darker)
Node label   : group id (gXXX) printed in the centre of each circle
Edge color   : source node's compartment color
Edge width   : log-scaled true weight

"""
import flybrains
import pandas as pd
import numpy as np
import networkx as nx
import navis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _compartment_palette import COMP_HEX, COMP_RGB, COMP_ORDER

#------------------------------------------------------------------------------
# Config
folder    = "Analysis_Outputs/Group_Layer-Network"
k         = 10
EDGE_FRAC = 0.075   # keep top fraction of edges by norm_weight

label_fontsize = 7
tick_fontsize  = 6

#------------------------------------------------------------------------------
# Brain
neuropil       = navis.Volume(flybrains.PK_L1CBNeuropilsym.mesh)
neuropil.color = (0, 0, 0, 0.35)

#------------------------------------------------------------------------------
# Load data
layer_df = pd.read_parquet(f"{folder}/Signal-Integration_Group_Layers-Assignment_k-{k}.parquet",
                           engine = "pyarrow")
conn_df  = pd.read_csv("Data_Connectivity/Connectivity_Group.csv")
pos_df   = pd.read_parquet(f"{folder}/Group_Hybrid-Placement.parquet",
                           engine = "pyarrow")

#------------------------------------------------------------------------------
# Anatomical positions: hybrid placement (one row per Group x Hemisphere)
frag = pos_df[["Group", "Hemisphere", "Compartment", "Fragment",
               "Pos_x", "Pos_y", "Size"]].drop_duplicates()

#------------------------------------------------------------------------------
# Build node table: one row per group x hemisphere
nodes = []
for _, row in frag.iterrows():
    grp  = row["Group"]
    info = layer_df.loc[layer_df["Group"] == grp]
    if info.empty:
        continue
    info = info.iloc[0]

    nodes.append({"Node"        : f"{grp}_{row['Hemisphere']}",
                  "Group"       : grp,
                  "Hemisphere"  : row["Hemisphere"],
                  "x"           : row["Pos_x"],
                  "y"           : row["Pos_y"],
                  "Size"        : row["Size"],
                  "Compartment" : row["Compartment"],
                  "Fragment"    : row["Fragment"],
                  "Layer"       : info["Layer"]})

nodes = pd.DataFrame(nodes)

# Self-connection strength → node alpha
self_conn = conn_df.loc[conn_df["Pre"] == conn_df["Post"],
                        ["Pre", "norm_weight"]]
self_map  = dict(zip(self_conn["Pre"], self_conn["norm_weight"]))
self_max  = self_conn["norm_weight"].max() if len(self_conn) > 0 else 1.0

nodes["Self_W"] = nodes["Node"].map(self_map).fillna(0)
nodes["Alpha"]  = 0.5 + 0.5 * (nodes["Self_W"] / (self_max + 1e-12))

#------------------------------------------------------------------------------
# Subset edges: right-hemisphere origins, no self, top fraction by norm_weight
edges = conn_df.loc[(conn_df["Hemisphere_pre"] == "Right") &
                    (conn_df["Pre"] != conn_df["Post"])].copy()

n_top = int(len(edges) * EDGE_FRAC)
edges = edges.nlargest(n_top, "norm_weight")

# Keep only edges between nodes we have positions for
valid_nodes = set(nodes["Node"])
edges       = edges.loc[edges["Pre"].isin(valid_nodes) &
                        edges["Post"].isin(valid_nodes)].reset_index(drop = True)

#------------------------------------------------------------------------------
# Build NetworkX graph
G        = nx.DiGraph()
node_pos = {}
for _, r in nodes.iterrows():
    node_pos[r["Node"]] = (r["x"], r["y"])
    G.add_node(r["Node"],
               layer    = r["Layer"],
               size     = r["Size"],
               comp     = r["Compartment"],
               fragment = r["Fragment"],
               alpha    = r["Alpha"],
               hemi     = r["Hemisphere"])

G.add_edges_from([(r["Pre"], r["Post"],
                   {"weight" : r["weight"], "norm_weight" : r["norm_weight"]})
                  for _, r in edges.iterrows()])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

#------------------------------------------------------------------------------
# Shared geometry
node_sizes   = {n : 8 + G.nodes[n]["size"] * (140 - 8) / 75 for n in G.nodes()}
global_sizes = [node_sizes[n] for n in G.nodes()]

all_x, all_y = zip(*node_pos.values())
midline_x    = (min(all_x) + max(all_x)) / 2
pad_x        = (max(all_x) - min(all_x)) * 0.05
pad_y        = (max(all_y) - min(all_y)) * 0.05
xlims        = (min(all_x) - pad_x, max(all_x) + pad_x)
ylims        = (min(all_y) - pad_y, max(all_y) + pad_y)

#------------------------------------------------------------------------------
# Helpers
def _draw_edge(ax, u, v, d, rgb, arrowsize = 5):
    """One curved, arrow-headed edge. Arc direction flips across the midline."""
    lw    = 0.05 + np.log1p(d["weight"]) * 0.12
    a     = min(0.55 + d["norm_weight"] * 0.45, 0.95)
    rad   = -0.22 if node_pos[u][0] > midline_x else 0.22

    nx.draw_networkx_edges(G, pos = node_pos,
                           edgelist        = [(u, v)],
                           ax              = ax,
                           edge_color      = [(*rgb, a)],
                           width           = lw,
                           node_size       = global_sizes,
                           arrows          = True,
                           arrowsize       = arrowsize,
                           arrowstyle      = "-|>",
                           connectionstyle = f"arc3,rad={rad}")


def _draw_fragment_outline(ax, xy, sz, fragment, lw = 0.35):
    """
    Fragment outline: Dendrite = solid black halo ring,
                      Axon     = dashed black halo ring,
                      Neuron   = nothing.
    Ring sits just outside the marker (radius ~7% larger).
    """
    if fragment == "Neuron":
        return
    sc = ax.scatter(*xy,
                    s          = sz * 1.15,
                    facecolors = "none",
                    edgecolors = "black",
                    linewidths = lw,
                    zorder     = 4)
    if fragment == "Axon":
        sc.set_linestyle("--")


def _label_node(ax, n, xy, fontsize = 1.5):
    """Centred group label (e.g. 'g042')."""
    ax.annotate(n.rsplit("_", 1)[0],
                xy       = xy,
                fontsize = fontsize,
                color    = "black",
                alpha    = 0.85,
                ha       = "center",
                va       = "center",
                zorder   = 6)

#==============================================================================
# FIGURE 1 — whole-network overview
#==============================================================================
fig, ax = plt.subplots(figsize = (6.5, 6.5))

# Neuropil outline
navis.plot2d(neuropil,
             ax              = ax,
             view            = ("x", "y"),
             method          = "2d",
             volume_outlines = True,
             linewidth       = 0.4)

# Midline
y_min, y_max = min(all_y), max(all_y)
y_span       = y_max - y_min
ax.plot([midline_x, midline_x],
        [y_min - 0.1 * y_span, y_max + 0.1 * y_span],
        color     = "k",
        linestyle = (0, (4, 4)),
        linewidth = 0.35,
        alpha     = 0.35,
        zorder    = 0)

# Edges coloured by source compartment
for u, v, d in G.edges(data = True):
    _draw_edge(ax, u, v, d,
               rgb = COMP_RGB[G.nodes[u]["comp"]])

# Nodes: compartment fill + fragment outline (grey on left)
for n, attr in G.nodes(data = True):
    sz      = node_sizes[n]
    n_alpha = attr["alpha"]
    is_left = attr["hemi"] == "Left"

    # Fragment outline ring (skipped for Neuron)
    _draw_fragment_outline(ax, node_pos[n], sz, attr["fragment"], lw = 0.35)

    fill_c = (0.65, 0.65, 0.65) if is_left else tuple(COMP_RGB[attr["comp"]])
    ax.scatter(*node_pos[n],
               s          = sz,
               c          = [fill_c],
               alpha      = 0.5 if is_left else n_alpha,
               edgecolors = "none",
               zorder     = 5)

    _label_node(ax, n, node_pos[n], fontsize = 1.8)

#-- Compartment legend (horizontal, below the panel) ---------------------------
comp_handles = [Line2D([0], [0],
                       marker          = "o",
                       color           = "none",
                       markerfacecolor = tuple(COMP_RGB[c]),
                       markeredgecolor = "none",
                       markersize      = 5,
                       label           = c,
                       linestyle       = "None")
                for c in COMP_ORDER]

ax.legend(handles        = comp_handles,
          loc            = "upper center",
          bbox_to_anchor = (0.5, -0.01),
          ncol           = len(COMP_ORDER),
          fontsize       = tick_fontsize - 1,
          frameon        = False,
          handletextpad  = 0.3,
          columnspacing  = 0.8,
          borderpad      = 0.0)

ax.axis("off")
ax.invert_yaxis()

plt.savefig(f"{folder}/Figure_05_Compartment-Symmetric-Network.pdf",
            bbox_inches = "tight",
            dpi         = 1200)
plt.close()

#==============================================================================
# FIGURE 2 — per-layer decomposition (4 x 3 grid)
#==============================================================================
fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize = (16, 16))
plt.subplots_adjust(wspace = 0.02, hspace = 0.02)
axes = axes.flatten()

# Precompute which nodes belong to each layer
layer_nodes = {}
for n, attr in G.nodes(data = True):
    layer_nodes.setdefault(attr["layer"], []).append(n)

for idx, a in enumerate(axes):
    a.axis("off")
    a.set_xlim(xlims)
    a.set_ylim(ylims)
    a.invert_yaxis()

    # Last panel (past L10) is the compartment legend.
    if idx > k:
        comp_handles = [Line2D([0], [0],
                               marker          = "o",
                               color           = "none",
                               markerfacecolor = tuple(COMP_RGB[c]),
                               markersize      = 8,
                               label           = c,
                               linestyle       = "None")
                        for c in COMP_ORDER]
        a.legend(handles        = comp_handles,
                 title          = "Compartment",
                 title_fontsize = tick_fontsize + 1,
                 fontsize       = tick_fontsize,
                 loc            = "center",
                 framealpha     = 0.7,
                 edgecolor      = "grey",
                 ncol           = 2)
        continue

    lay    = idx
    active = set(layer_nodes.get(lay, []))

    # Neuropil outline
    navis.plot2d(neuropil,
                 ax              = a,
                 view            = ("x", "y"),
                 method          = "2d",
                 volume_outlines = True,
                 linewidth       = 0.5)

    # Background: all nodes in very pale grey for context
    for n in G.nodes():
        a.scatter(*node_pos[n],
                  s          = node_sizes[n] * 0.6,
                  c          = [(0.8, 0.8, 0.8)],
                  alpha      = 0.25,
                  edgecolors = "none",
                  zorder     = 2)

    # Edges from this layer coloured by source node's compartment
    for u, v, d in G.edges(data = True):
        if u not in active:
            continue
        _draw_edge(a, u, v, d,
                   rgb       = COMP_RGB[G.nodes[u]["comp"]],
                   arrowsize = 4)

    # Target nodes outside this layer: slightly darker grey
    targets = {v for u, v, d in G.edges(data = True)
               if u in active and v not in active}
    for n in targets:
        a.scatter(*node_pos[n],
                  s          = node_sizes[n] * 0.8,
                  c          = [(0.55, 0.55, 0.55)],
                  alpha      = 0.6,
                  edgecolors = "none",
                  zorder     = 3)

    # Highlighted nodes for this layer
    for n in layer_nodes.get(lay, []):
        attr    = G.nodes[n]
        sz      = node_sizes[n]
        n_alpha = attr["alpha"]
        is_left = attr["hemi"] == "Left"

        if is_left:
            fill_c, al = (0.55, 0.55, 0.55), 0.6
        else:
            fill_c, al = tuple(COMP_RGB[attr["comp"]]), n_alpha
            _draw_fragment_outline(a, node_pos[n], sz,
                                   attr["fragment"], lw = 0.25)

        a.scatter(*node_pos[n],
                  s          = sz,
                  c          = [fill_c],
                  alpha      = al,
                  edgecolors = "none",
                  zorder     = 5)

        _label_node(a, n, node_pos[n], fontsize = 1.5)

    # Panel title: number of right-hemi groups in this layer
    n_groups = sum(1 for n in layer_nodes.get(lay, [])
                   if G.nodes[n]["hemi"] == "Right")
    a.set_title(f"L{lay}  ({n_groups} groups)",
                fontsize = tick_fontsize + 1, pad = 2)

plt.savefig(f"{folder}/Figure_05_Supplementary_Layering-Compartment.pdf",
            bbox_inches = "tight",
            dpi         = 1200)
plt.close()

#------------------------------------------------------------------------------
print("हो गया दोस्तों!")
