"""
Bundles connectome analysis outputs (layer assignments, signal-flow metrics,
group positions, raw connectivity) into a compact `docs/viewer_data.js`
that `docs/index.html` loads via a <script> tag. Coordinate values stay in brain
space (no projection) so they align with the neuropil outline.

Run from repo root:
    python Scripts/viz/export_viewer_data.py

GitHub Pages serves /docs, so once this is committed and pushed, the
interactive viewer is live at:
    https://<user>.github.io/L1_Network/

"""
import pandas as pd
import json

#------------------------------------------------------------------------------
folder = "Analysis_Outputs/Group_Layer-Network"
k      = 10

layer_path   = f"{folder}/Signal-Integration_Group_Layers-Assignment_k-{k}.parquet"
conn_path    = "Data_Connectivity/Connectivity_Group.csv"
pos_path     = f"{folder}/Group_Hybrid-Placement.parquet"
role_path    = "Analysis_Outputs/Group_Signal-Flow/Group_Functional-Roles.parquet"
outline_path = "Scripts/viz/neuropil_outline.json"

#------------------------------------------------------------------------------
layer_df = pd.read_parquet(layer_path, engine = "pyarrow")
conn_df  = pd.read_csv(conn_path)
pos_df   = pd.read_parquet(pos_path,  engine = "pyarrow")
role_df  = pd.read_parquet(role_path, engine = "pyarrow")

# Merge functional roles
role_cols = ["Group", "Role", "Flow_Ratio", "Neff_inputs", "Neff_outputs"]
layer_df  = layer_df.merge(role_df[role_cols], on = "Group", how = "left")

#------------------------------------------------------------------------------
# Hybrid placement: one row per Group x Hemisphere with Compartment + Fragment.
frag = pos_df[["Group", "Hemisphere", "Compartment", "Fragment",
               "Pos_x", "Pos_y", "Size"]].drop_duplicates()

#------------------------------------------------------------------------------
# Build node list
nodes = []
for _, row in frag.iterrows():
    grp  = row["Group"]
    info = layer_df.loc[layer_df["Group"] == grp]
    if info.empty:
        continue
    info = info.iloc[0]

    def _safe_float(col, default, decimals):
        val = info.get(col)
        try:
            return round(float(val), decimals)
        except (TypeError, ValueError):
            return default

    nodes.append({
        "id"       : f"{grp}_{row['Hemisphere']}",
        "group"    : str(grp),
        "hemi"     : row["Hemisphere"],
        "x"        : float(row["Pos_x"]),
        "y"        : float(row["Pos_y"]),
        "size"     : int(row["Size"]),
        "comp"     : str(row["Compartment"]),
        "fragment" : str(row["Fragment"]),
        "layer"    : int(info["Layer"]),
        "mod"      : str(info["Primary_Modality"]),
        # Signal integration
        "integration" : round(float(info["Integration_Index"]), 4),
        "final_score" : round(float(info["Final_Score"]),       4),
        # PPR drive per modality
        "olf_drive"    : round(float(info["Olf_Scaled_Drive"]),    4),
        "gust_drive"   : round(float(info["Gust_Scaled_Drive"]),   4),
        "vis_drive"    : round(float(info["Vis_Scaled_Drive"]),    4),
        "thermo_drive" : round(float(info["Thermo_Scaled_Drive"]), 4),
        # Modality enrichment
        "olf_enr"    : round(float(info["Olf_Enrichment"]),    4),
        "gust_enr"   : round(float(info["Gust_Enrichment"]),   4),
        "vis_enr"    : round(float(info["Vis_Enrichment"]),    4),
        "thermo_enr" : round(float(info["Thermo_Enrichment"]), 4),
        # Functional role
        "role"       : str(info.get("Role") or ""),
        "flow_ratio" : _safe_float("Flow_Ratio",   0.0, 4),
        "neff_in"    : _safe_float("Neff_inputs",  0.0, 2),
        "neff_out"   : _safe_float("Neff_outputs", 0.0, 2),
    })

node_ids = {n["id"] for n in nodes}

#------------------------------------------------------------------------------
# Self-connection strength → node alpha
self_conn = conn_df.loc[conn_df["Pre"] == conn_df["Post"],
                        ["Pre", "norm_weight"]]
self_map  = dict(zip(self_conn["Pre"], self_conn["norm_weight"]))
self_max  = float(self_conn["norm_weight"].max()) if len(self_conn) > 0 else 1.0

for n in nodes:
    sw         = self_map.get(n["id"], 0.0)
    n["self_w"] = round(float(sw), 5)
    n["alpha"]  = round(0.5 + 0.5 * (sw / (self_max + 1e-12)), 3)

#------------------------------------------------------------------------------
# Edges (unfiltered — threshold is applied interactively in the viewer)
edges    = conn_df.loc[conn_df["Pre"] != conn_df["Post"]].copy()
edges    = edges.loc[edges["Pre"].isin(node_ids) & edges["Post"].isin(node_ids)]

edge_list = [{"s"  : r["Pre"],
              "t"  : r["Post"],
              "w"  : round(float(r["weight"]),      3),
              "nw" : round(float(r["norm_weight"]), 5)}
             for _, r in edges.iterrows()]

#------------------------------------------------------------------------------
# Neuropil outline
with open(outline_path, "r") as f:
    neuropil = json.load(f)

#------------------------------------------------------------------------------
# Package
data = {
    "nodes"    : nodes,
    "edges"    : edge_list,
    "k"        : k,
    "self_max" : round(self_max, 5),
    "outline"  : neuropil["outline"],
}

# Written as a JS global (not plain JSON) so index.html can load it with a
# <script> tag -- works from file:// as well as GitHub Pages.
out_path = "docs/viewer_data.js"
with open(out_path, "w") as f:
    f.write("// Generated by Scripts/viz/export_viewer_data.py -- do not edit.\n")
    f.write("const INLINE_DATA = ")
    json.dump(data, f, separators = (",", ":"))
    f.write(";\n")

print(f"Exported {len(nodes)} nodes, {len(edge_list)} edges → {out_path}")
print(f"  File size: {len(json.dumps(data, separators = (',', ':'))) / 1024:.0f} KB")

#------------------------------------------------------------------------------
print("हो गया दोस्तों!")
