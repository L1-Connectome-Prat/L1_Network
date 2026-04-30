"""
tSNE (sadly) visualization of the embedded acronyms. 

Signal Flow and Sensory Integration Analysis.

"""
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from functools import reduce
import os
import pickle as pkl

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_style("ticks")
import networkx as nx
from prats_helpers import place_panel_label

#------------------------------------------------------------------------------
folder = "Analysis_Outputs/Group_Signal-Flow"
home = os.environ["PROJECTS_HOME"]

#------------------------------------------------------------------------------
# Params
alpha_ppr = 0.85
max_iter  = 100
tol       = 1e-8
eps       = 1e-12

# Figure params
linewidth      = 0.75
scatter_size   = 6
scatter_alpha  = 0.75
linewidth_hist = 0.5
alpha_hist     = 0.35
title_fontsize = 7.5
label_fontsize = 7
tick_fontsize  = 6

# Panel Labels
label_kwargs = dict(fontsize = 16, va = "top", ha = "right")

#------------------------------------------------------------------------------
# Acronym and Group Sizes
neuron_df = pd.read_csv(f"{home}/L1_Skeletons/Data_Summary/L1_NeuronInformation_PK.csv")
neuron_df["VolkerClass"] = neuron_df["VolkerClass"] + "_" + neuron_df["Hemisphere"]
acronym_counts = dict(neuron_df.groupby("VolkerClass").size())

# Remove the NaN groups
neuron_df = neuron_df.loc[~neuron_df["ConnectivityGroup"].isna()]
neuron_df["ConnectivityGroup"] = neuron_df["ConnectivityGroup"] + "_" + neuron_df["Hemisphere"]
group_counts = dict(neuron_df.groupby("ConnectivityGroup").size())

#------------------------------------------------------------------------------
# tSNE 
#------------------------------------------------------------------------------
tsne = TSNE(
    n_components = 2,
    perplexity         = 35,
    early_exaggeration = 12,
    learning_rate      = "auto",
    max_iter           = 1000,
    random_state       = 42,
    init               = "pca",
    metric             = "euclidean")

# Matrix
clustered_matrix = pd.read_csv("Analysis_Outputs/Embedding/ClusteredMatrix_128_8_wLabels.csv")

# Feat
embedding = clustered_matrix.iloc[:, 1:-1].values.astype(np.float32)
acronyms  = clustered_matrix.iloc[:, 0].values
labels    = clustered_matrix.iloc[:, -1].values

# Run
embedding2d = tsne.fit_transform(embedding)

# Convert to DataFrame
tsne_df = pd.DataFrame({
    "Acronym" : acronyms,
    "Label"   : labels,
    "t-SNE1"  : embedding2d[:, 0],
    "t-SNE2"  : embedding2d[:, 1]
})

# Save it
tsne_df.to_parquet(f"{folder}/Acronyms_tSNE.parquet",
                   index = False,
                   engine = "fastparquet")

#------------------------------------------------------------------------------
# PERSONALIZED PAGE RANK
#------------------------------------------------------------------------------
def personalized_page_rank(
    conn_matrix : pd.DataFrame,
    sources     : set,
    raw_out     : dict,
    alpha       : float = alpha_ppr,
    max_iter    : int = max_iter,
    tol         : float = tol) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Executes a Personalized PageRank (PPR) random walk on a directed brain connectome.
 
    This function models signal flow by allowing a mathematical "random walker" to traverse 
    outgoing synaptic connections. To simulate specific sensory drives, the walker frequently 
    teleports back to a defined set of starting sensory neurons. The probability of teleporting 
    to a specific sensory neuron is weighted by its physical size (raw synaptic output), 
    meaning larger sensory organs pump proportionally more signal into the network.
 
    Parameters
    ----------
    conn_matrix : pandas.DataFrame
        Row-stochastic transition matrix representing the normalized connectome.
        Rows without outgoing connections must be entirely zeros.
    sources : set of str
        The specific labels of the sensory neurons where the signal originates.
    raw_out : dict of str to float
        A mapping of neuron labels to their absolute total synaptic output. 
        Used to weight the teleportation probabilities.
    alpha : float, optional
        The damping factor representing the probability of following a synapse 
        rather than teleporting back to the sensory source.
    max_iter : int, optional
        The maximum number of power iterations to run before forcing a stop.
    tol : float, optional
        The convergence threshold. Iteration stops when the distribution stabilizes.
 
    Returns
    -------
    score_df : pandas.DataFrame
        A dataframe containing the converged steady-state probability (PPR_score) 
        and the absolute signal volume (PPR_drive) for each neuron.
    p : numpy.ndarray
        The teleport personalization vector used during the walk.

    """
    nodes = conn_matrix.columns.tolist()
    M = conn_matrix.values.astype(float)
    n = M.shape[0]
 
    # Validate that the transition matrix properties are mathematically sound
    assert np.isfinite(M).all(), "conn_matrix contains NaN or inf."
    row_sums = M.sum(axis = 1)
    nonzero = row_sums > 0
    assert np.allclose(row_sums[nonzero], 1.0, atol = 1e-10), (
        "Non-dangling rows must sum to 1."
    )
    assert np.all(row_sums[~nonzero] == 0), (
        "Dangling rows must be all zeros."
    )
    assert 0.0 < alpha < 1.0, f"alpha must be in (0, 1); got {alpha}"
 
    # Build the teleport vector based on the physical size of the sensory sources
    p = np.zeros(n, dtype = float)
    present = [s for s in sources if s in nodes]
 
    if present:
        idxs = [nodes.index(s) for s in present]
        wts = np.array([raw_out.get(s, 1.0) for s in present],
                        dtype = float)
        # Normalize the weights so the teleport vector sums to exactly 1.0
        wts /= wts.sum() if wts.sum() > 0 else len(wts)
        p[idxs] = wts
    else:
        # Fallback to a uniform distribution if no sources are present in the matrix
        p[:] = 1.0 / n
 
    p /= p.sum()
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1.0), (
        "Teleport vector p is invalid."
    )
 
    # Execute power iteration to find the steady-state probabilities
    dangling = row_sums == 0
    score = p.copy()
 
    for t in range(int(max_iter)):
        # Collect probability mass from dead-end nodes and redistribute it to the sources
        dmass = score[dangling].sum()
        score_new = alpha * (M.T @ score + dmass * p) + (1 - alpha) * p
 
        ssum = score_new.sum()
        if not np.isclose(ssum, 1.0, atol = 1e-9):
            raise RuntimeError(
                f"Iteration {t}: score sum drifted to {ssum}. "
                "Check conn_matrix, p, and alpha."
            )
 
        if np.linalg.norm(score_new - score, ord = 1) < tol:
            score = score_new
            break
 
        score = score_new

    # Convert the steady-state probability into a physical drive by multiplying 
    # it by the total absolute volume of the starting sensory organs
    if raw_out is not None:
        modality_drive = sum(raw_out.get(s, 0.0) for s in sources)
        score_drive = score * modality_drive
 
    # Assemble the output dataframe and parse the group names
    score_df = pd.DataFrame({"Group": nodes, "PPR_score": score, "PPR_drive": score_drive})
    score_df[["Group_Num", "Hemisphere"]] = (
        score_df["Group"].str.rsplit("_", n = 1, expand = True)
    )
    try:
        score_df["Group_Num"] = score_df["Group_Num"].astype(int)
    except ValueError:
        pass
 
    return score_df, p

def ppr_mean(
    score_df : pd.DataFrame,
    modality : str = "Olf",
    eps      : float = 1e-12
    ) -> pd.DataFrame:
    """
    Consolidates hemisphere-specific PageRank scores into unified group-level metrics.
 
    This function averages the probabilities and drives of the left and right hemispheres.
    It also converts the raw probabilities into topological distances (synaptic hops)
    by applying a negative base-10 logarithm.
 
    Parameters
    ----------
    score_df : pandas.DataFrame
        Output of the personalized_page_rank function.
    modality : str, optional
        A string prefix used to label the output columns (e.g., "Olf" for Olfactory).
 
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the averaged drives and logarithmic distances.

    """
    # Mean across whichever hemispheres are present (1 or 2). Using sum/2
    # would halve the score of unilateral groups.
    agg_df = score_df.groupby("Group_Num")[["PPR_score", "PPR_drive"]].mean()
    agg_df = agg_df.reset_index()
    
    # Initialize output dictionary for clean DataFrame creation
    out_data = {
        "Group_Num": agg_df["Group_Num"],
        f"{modality}_Mean_Score" : agg_df["PPR_score"],
        f"{modality}_Mean_Drive" : agg_df["PPR_drive"],
        f"{modality}_Scaled_Distance" : -np.log10(agg_df["PPR_score"].to_numpy(dtype = float) + eps),
        f"{modality}_Scaled_Drive"    : np.log10(agg_df["PPR_drive"].to_numpy(dtype = float) + 1)
    }
 
    return pd.DataFrame(out_data)
 
def integration_index(matrix, eps = 1e-12):
    """
    Calculates multisensory integration using normalized Shannon entropy.
    
    This function analyzes the proportion of signal a node receives from each sensory 
    modality. A value of 0 indicates a pure, unimodal labeled line. A value of 1 
    indicates a perfectly balanced, multimodal mixing bowl.
    """
    x = matrix.to_numpy(dtype = float)
    row_sums = x.sum(axis = 1, keepdims = True)
    p = x / (row_sums + eps)
    H = -(p * np.log(p + eps)).sum(axis = 1)
    K = matrix.shape[1]
    return H / np.log(K + eps)

#--------------------------------------
def run(conn_df,
        drive_cols = ["Olf_Mean_Drive",
                      "Gust_Mean_Drive",
                      "Vis_Mean_Drive",
                      "Thermo_Mean_Drive"],
        alpha = alpha_ppr):
    """
    Wrapper function to execute the full signal flow and integration pipeline.

    This coordinates the extraction of sensory sources, executes the PageRank
    simulations for each modality, normalizes the topological distances, and
    computes the final biological metrics (Integration, Consensus Depth, and Enrichment).

    """
    # Calculate the PPR for each modality
    s_olf    = np.unique(conn_df.loc[conn_df["System_pre"] == "Olfactory-1", "Pre"])
    s_vis    = np.unique(conn_df.loc[conn_df["System_pre"] == "Visual-1", "Pre"])
    s_gust   = np.unique(conn_df.loc[conn_df["System_pre"] == "Gustatory-1", "Pre"])
    s_thermo = np.unique(conn_df.loc[conn_df["System_pre"] == "Thermosensory-1", "Pre"])

    # All
    s_all = np.concatenate([s_olf, s_vis, s_gust, s_thermo])

    # Convert conn matrix
    matrix = conn_df.pivot_table(index   = "Pre",
                                 columns = "Post",
                                 values  = "norm_weight",
                                 aggfunc = "sum",
                                 fill_value = 0)

    # Raw outputs
    raw_out = conn_df[["Pre", "total_out"]].drop_duplicates().reset_index(drop = True)
    raw_out = dict(zip(raw_out["Pre"], raw_out["total_out"]))

    # Sanity sort
    nodes = sorted((set(conn_df["Pre"]) | set(conn_df["Post"])))
    matrix = matrix.reindex(index = nodes, columns = nodes, fill_value = 0)

    # PPR for each modality
    olf_scores, _    = personalized_page_rank(matrix, s_olf,    raw_out = raw_out, alpha = alpha)
    vis_scores, _    = personalized_page_rank(matrix, s_vis,    raw_out = raw_out, alpha = alpha)
    gust_scores, _   = personalized_page_rank(matrix, s_gust,   raw_out = raw_out, alpha = alpha)
    thermo_scores, _ = personalized_page_rank(matrix, s_thermo, raw_out = raw_out, alpha = alpha)
    all_scores, p    = personalized_page_rank(matrix, s_all,    raw_out = raw_out, alpha = alpha)

    # Calculating the mean
    olf_mean    = ppr_mean(olf_scores,    modality = "Olf")
    vis_mean    = ppr_mean(vis_scores,    modality = "Vis")
    gust_mean   = ppr_mean(gust_scores,   modality = "Gust")
    thermo_mean = ppr_mean(thermo_scores, modality = "Thermo")
    all_mean    = ppr_mean(all_scores,    modality = "All")

    # Merge
    mean_scores = reduce(lambda L, R: pd.merge(L, R, on = "Group_Num", how = "outer"),
                     [olf_mean, gust_mean, vis_mean, thermo_mean, all_mean]).fillna(0.0)

    # Determine Primary Modality Only based on the Mean_Drive (signal volume)
    mean_scores["Primary_Modality"] = mean_scores[drive_cols].idxmax(axis = 1).str.split("_").str[0]

    # Integration Index
    mean_scores["Integration_Index"] = integration_index(mean_scores[drive_cols], eps = 1e-12)
    
    # The integration drive measures the multimodal bandwidth
    mean_scores["Integration_Drive"] = mean_scores["Integration_Index"] * mean_scores["All_Scaled_Drive"]

    # Zero-align the topological distances so every sensory stream starts exactly at a depth of zero
    for mod in ["Olf", "Gust", "Vis", "Thermo"]:
        dist_col = f"{mod}_Scaled_Distance"
        norm_col = f"{mod}_Normalized_Distance"
        mean_scores[norm_col] = mean_scores[dist_col] - mean_scores[dist_col].min()

    # Find the deepest topological point across all modalities to use as a global scaling factor
    global_max = max(mean_scores[f"{mod}_Normalized_Distance"].max()
                     for mod in ["Olf", "Gust", "Vis", "Thermo"])

    # Normalize by global max
    for mod in ["Olf", "Gust", "Vis", "Thermo"]:
        norm_col = f"{mod}_Normalized_Distance"
        mean_scores[norm_col] /= (global_max + 1e-12)

    # Extract the normalized baseline distance corresponding to the node's dominant modality
    mean_scores["Primary_Distance"] = mean_scores.apply(
        lambda row : row[row["Primary_Modality"] + "_Normalized_Distance"],
        axis = 1)

    # FINAL SCORE
    # Calculate the internal sensory recipe (proportions) for each node
    row_sums = mean_scores[drive_cols].sum(axis = 1) + 1e-12
    p_matrix = mean_scores[drive_cols].div(row_sums, axis = 0)

    # Calculate the Final Score by multiplying the specific sensory proportions 
    # by their respective normalized distances. This naturally places deep multimodal 
    # hubs at their appropriate topological depth without using heuristic penalties.
    dist_cols = [f"{mod}_Normalized_Distance" for mod in ["Olf", "Gust", "Vis", "Thermo"]]
    mean_scores["Final_Score"] = (p_matrix.values * mean_scores[dist_cols].values).sum(axis = 1)

    # Create mask for non-sensory nodes
    sources = np.unique([s.split("_")[0] for s in s_all])
    mask = ~mean_scores["Group_Num"].isin(sources)
    
    # MODALITY ENRICHMENT
    for col in drive_cols:
        # Isolate Modality
        mod_name = col.split("_")[0]

        # Determine the expected baseline
        # Sans-Sensory Nodes
        expected_mean = p_matrix.loc[mask, col].mean()

        # Calculate fold-change: (Observed Proportion - Expected Baseline) / Expected Baseline
        # Positive values denote signal enrichment; negative values denote signal depletion.
        mean_scores[f"{mod_name}_Enrichment"] = (p_matrix[col] - expected_mean) / (expected_mean + 1e-12)

    # Remove the index
    mean_scores.reset_index(drop = True, inplace = True)

    return mean_scores

#------------------------------------------------------------------------------
# ACRONYM LEVEL
# Merge System Information
sys_df = pd.read_csv("Analysis_Outputs/Embedding/Groups_Separated_128_8_Corrected.csv")
sys_df["Group"] = "g" + sys_df["Group"].astype(str).str.zfill(3)
sys_df.rename(columns = {"Class"  : "VolkerClass_pre",
                         "System" : "System_pre",
                         "Group"  : "ConnectivityGroup_pre"}, inplace = True)

#--------------------------------------
# Get the connectivity and merge
acronym_conn = pd.read_csv("Data_Connectivity/Connectivity_Acronym.csv")

# Merge
acronym_conn = acronym_conn.merge(sys_df[["VolkerClass_pre", "System_pre"]],
                                  on = "VolkerClass_pre",
                                  how = "left")

#--------------------------------------
# Scores
acronym_means = run(acronym_conn)

# Rename to Reflect the Acronym as its Unit 
acronym_means.rename(columns = {"Group_Num" : "Acronym"}, inplace = True)

# Save
acronym_means.to_csv(f"{folder}/Signal-Integration_Acronym.csv", index = False)
acronym_means.to_parquet(f"{folder}/Signal-Integration_Acronym.parquet", index = False,
                         engine = "pyarrow")

#------------------------------------------------------------------------------
# GROUP LEVEL
# Get the connectivity and merge
group_conn = pd.read_csv("Data_Connectivity/Connectivity_Group.csv")

# Merge
# Drop duplicates to ensure single entry. There are no conflicts (verified @KandimallaPrat)
group_conn = group_conn.merge(sys_df[["ConnectivityGroup_pre", "System_pre"]].drop_duplicates(),
                              on = "ConnectivityGroup_pre",
                              how = "left")

#--------------------------------------
# Running a sanity check on alphas for the groups
check_alphas = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
check_scores = {}

# Run through
for a in check_alphas:
    result = run(group_conn, alpha = a)
    check_scores[a] = result.set_index("Group_Num")["Final_Score"]

# Pairwise Spearman rank correlations
n_alpha = len(check_alphas)
rho_matrix = np.ones((n_alpha, n_alpha))
for i, ai in enumerate(check_alphas):
    for j, aj in enumerate(check_alphas):
        if i != j:
            rho_matrix[i, j], _ = spearmanr(check_scores[ai], check_scores[aj])

# Heatmap (lower triangle only since Spearman is symmetric)
rho_df = pd.DataFrame(rho_matrix,
                       index   = [f"{a:.2f}" for a in check_alphas],
                       columns = [f"{a:.2f}" for a in check_alphas])
mask_tri = np.triu(np.ones_like(rho_matrix, dtype = bool), k = 1)

fig, ax = plt.subplots(figsize = (4.5, 4))
sns.heatmap(rho_df,
            mask   = mask_tri,
            annot  = True,
            fmt    = ".3f",
            cmap   = "magma",
            vmin   = 0.5,
            vmax   = 1.0,
            square = True,
            linewidths = 0.5,
            annot_kws  = {"fontsize" : 7},
            cbar_kws   = {"label"  : r"Spearman $\rho$",
                          "shrink" : 0.8},
            ax = ax)
ax.set_xlabel(r"$\alpha$", fontsize = label_fontsize)
ax.set_ylabel(r"$\alpha$", fontsize = label_fontsize)
ax.tick_params(labelsize = tick_fontsize)

# Mark the selected alpha
sel_idx = check_alphas.index(alpha_ppr)
ax.add_patch(plt.Rectangle((sel_idx, sel_idx), 1, 1,
                            fill = False, edgecolor = "lime",
                            linewidth = 2, clip_on = False))

plt.savefig(f"{folder}/Figure_05_Supplementary_PPR-Alpha-Sensitivity.pdf",
            dpi = 1200,
            bbox_inches = "tight")
plt.close()

#--------------------------------------
# Scores
group_means = run(group_conn)

# Rename to Reflect the Acronym as its Unit 
group_means.rename(columns = {"Group_Num" : "Group"}, inplace = True)
                
# Save
group_means.to_csv(f"{folder}/Signal-Integration_Group.csv", index = False)
group_means.to_parquet(f"{folder}/Signal-Integration_Group.parquet", index = False,
                         engine = "pyarrow")

#------------------------------------------------------------------------------
# FIGURE FLOW DISTRIBUTIONS
sens = ["Olfactory", "Gustatory", "Visual", "Thermosensory", "Final Score"]
col  = sns.color_palette("tab10", 5)
cmap = dict(zip(sens, col))

# Fig
fig, ax = plt.subplots(figsize = (7.5, 12), ncols = 3, nrows = 4,
                       gridspec_kw = {"width_ratios" : [1, 1, 0.35]} )
plt.subplots_adjust(hspace = 0.45, wspace = 0.45)

# Acronym Level
drive = "Mean_Drive"
sns.ecdfplot(data      = acronym_means,
             x         = "Olf_" + drive,
             ax        = ax[0, 0],
             label     = "Olfactory",
             color     = cmap["Olfactory"],
             linewidth = linewidth)

sns.ecdfplot(data      = acronym_means,
             x         = "Gust_" + drive,
             ax        = ax[0, 0],
             label     = "Gustatory",
             color     = cmap["Gustatory"],
             linewidth = linewidth)

sns.ecdfplot(data      = acronym_means,
             x         = "Vis_" + drive,
             ax        = ax[0, 0],
             label     = "Visual",
             color     = cmap["Visual"],
             linewidth = linewidth)

sns.ecdfplot(data      = acronym_means,
             x         = "Thermo_" + drive,
             ax        = ax[0, 0],
             label     = "Thermosensory",
             color     = cmap["Thermosensory"],
             linewidth = linewidth)

# Group Level
sns.ecdfplot(data      = group_means,
             x         = "Olf_" + drive,
             ax        = ax[0, 1],
             label     = "Olfactory",
             color     = cmap["Olfactory"],
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Gust_" + drive,
             ax        = ax[0, 1],
             label     = "Gustatory",
             color     = cmap["Gustatory"],
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Vis_" + drive,
             ax        = ax[0, 1],
             label     = "Visual",
             color     = cmap["Visual"],
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Thermo_" + drive,
             ax        = ax[0, 1],
             label     = "Thermosensory",
             color     = cmap["Thermosensory"],
             linewidth = linewidth)

# Normalized Distances
distance = "Normalized_Distance"
# Acronym Level
sns.ecdfplot(data      = acronym_means,
             x         = "Olf_" + distance,
             ax        = ax[1, 0],
             label     = "Olfactory",
             color     = cmap["Olfactory"],
             linewidth = linewidth)

sns.ecdfplot(data      = acronym_means,
             x         = "Gust_" + distance,
             ax        = ax[1, 0],
             label     = "Gustatory",
             color     = cmap["Gustatory"],
             linewidth = linewidth)

sns.ecdfplot(data      = acronym_means,
             x         = "Vis_" + distance,
             ax        = ax[1, 0],
             label     = "Visual",
             color     = cmap["Visual"],
             linewidth = linewidth)

sns.ecdfplot(data      = acronym_means,
             x         = "Thermo_" + distance,
             ax        = ax[1, 0],
             label     = "Thermosensory",
             color     = cmap["Thermosensory"],
             linewidth = linewidth)

# Group Level
sns.ecdfplot(data      = group_means,
             x         = "Olf_" + distance,
             ax        = ax[1, 1],
             label     = "Olfactory",
             color     = cmap["Olfactory"],
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Gust_" + distance,
             ax        = ax[1, 1],
             label     = "Gustatory",
             color     = cmap["Gustatory"],
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Vis_" + distance,
             ax        = ax[1, 1],
             label     = "Visual",
             color     = cmap["Visual"],
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Thermo_" + distance,
             ax        = ax[1, 1],
             label     = "Thermosensory",
             color     = cmap["Thermosensory"],
             linewidth = linewidth)

# Integration Index
sns.ecdfplot(data      = acronym_means,
             x         = "Integration_Index",
             ax        = ax[2, 0],
             label     = "Classes",
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Integration_Index",
             ax        = ax[2, 0],
             label     = "Groups",
             linewidth = linewidth)

# Integration Drive
sns.ecdfplot(data      = acronym_means,
             x         = "Integration_Drive",
             ax        = ax[2, 1],
             label     = "Classes",
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Integration_Drive",
             ax        = ax[2, 1],
             label     = "Groups",
             linewidth = linewidth)
# Final Score
sns.ecdfplot(data      = acronym_means,
             x         = "Final_Score",
             ax        = ax[3, 0],
             label     = "Classes",
             linewidth = linewidth)

sns.ecdfplot(data      = group_means,
             x         = "Final_Score",
             ax        = ax[3, 0],
             label     = "Groups",
             linewidth = linewidth)

# Spines
for a in ax.flatten():
    # a.set_xlim(0, 0.05)
    sns.despine(ax = a)
    # Ticks
    a.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = tick_fontsize,
        length    = 1.5,
        width     = 0.5,
        pad       = 1.5)

# Close
for i in range(4):
    ax[i, 2].axis("off")
    # Normalize limits
    ax[i, 2].set_xlim(0, 1)
    ax[i, 2].set_ylim(0, 1)
    # Move it a little to the left
    pos = ax[i, 2].get_position()
    ax[i, 2].set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])  

# Close 
ax[3, 1].axis("off")  

# X-Labels
ax[0, 0].set_xlabel("PPR Drive", fontsize = label_fontsize, labelpad = 5)
ax[0, 1].set_xlabel("PPR Drive", fontsize = label_fontsize, labelpad = 5)
ax[1, 0].set_xlabel("Normalized Distance", fontsize = label_fontsize, labelpad = 5)
ax[1, 1].set_xlabel("Normalized Distance", fontsize = label_fontsize, labelpad = 5)
ax[2, 0].set_xlabel("Integration Index", fontsize = label_fontsize, labelpad = 5)
ax[2, 1].set_xlabel("Integration Drive", fontsize = label_fontsize, labelpad = 5)
ax[3, 0].set_xlabel("Final Score", fontsize = label_fontsize, labelpad = 5)

# Scale 
ax[0, 0].set_xscale("log")
ax[0, 1].set_xscale("log")
ax[0, 0].set_xlim(1e-8, 2000)
ax[0, 1].set_xlim(1e-8, 2000)
ax[1, 0].set_xlim(0, 1)
ax[1, 1].set_xlim(0, 1)
ax[2, 0].set_xlim(0, 1)
ax[2, 1].set_xlim(0, 2)
ax[3, 0].set_xlim(0, 1)

# Y-Labels
ax[0, 0].set_ylabel("Cumulative Proportion", fontsize = label_fontsize, labelpad = 5)
ax[0, 1].set_ylabel("", fontsize = label_fontsize, labelpad = 5)
ax[1, 0].set_ylabel("Cumulative Proportion", fontsize = label_fontsize, labelpad = 5)
ax[1, 1].set_ylabel("", fontsize = label_fontsize, labelpad = 5)
ax[2, 0].set_ylabel("Cumulative Proportion", fontsize = label_fontsize, labelpad = 5)
ax[2, 1].set_ylabel("", fontsize = label_fontsize, labelpad = 5)
ax[3, 0].set_ylabel("Cumulative Proportion", fontsize = label_fontsize, labelpad = 5)

# Titles
# Row 1: Absolute Volume
ax[0, 0].set_title("Sensory Signal Volume (Classes)", fontsize = title_fontsize, y = 1.02)
ax[0, 1].set_title("Sensory Signal Volume (Groups)", fontsize = title_fontsize, y = 1.02)

# Row 2: Pure Distance
ax[1, 0].set_title("Topological Distance (Classes)", fontsize = title_fontsize, y = 1.02)
ax[1, 1].set_title("Topological Distance (Groups)", fontsize = title_fontsize, y = 1.02)

# Row 3: Integration
ax[2, 0].set_title("Multisensory Integration Index", fontsize = title_fontsize, y = 1.02)
ax[2, 1].set_title("Effective Multimodal Bandwidth", fontsize = title_fontsize, y = 1.02)

# Row 4: The Final Metrics
ax[3, 0].set_title("Topological Layer Score", fontsize = title_fontsize, y = 1.02)

# Legends
ax[0, 1].legend(title = "Sensory Modality",
                loc = "upper left",
                bbox_to_anchor = (1.1, 1.1),
                title_fontsize = label_fontsize,
                fontsize = label_fontsize)

ax[0, 0].legend().set_visible(False)

ax[2, 1].legend(title = "Level",
                loc = "upper left",
                bbox_to_anchor = (1.1, 1.1),
                title_fontsize = label_fontsize,
                fontsize = label_fontsize)

# Row 1: Sensory Volume
ax[0, 2].text(x = 0, y = 0.5,
           s = r"$\text{PPR\_Drive} = \text{PPR\_Score} \times \sum \text{Source}_{\text{out}}$",
           fontsize = label_fontsize,
           ha = "left", va = "center")
ax[0, 2].text(x = 0, y = 0.35,
           s = r"Scaled Drive = $\log_{10}(\text{PPR\_Drive} + 1)$",
           fontsize = label_fontsize,
           ha = "left", va = "center")

# Row 2: Topological Distance
ax[1, 2].text(x = 0, y = 0.5,
           s = r"Scaled Distance ($D$) = $-\log_{10}(\text{PPR\_Score} + \epsilon)$",
           fontsize = label_fontsize,
           ha = "left", va = "center")
ax[1, 2].text(x = 0, y = 0.35,
           s = r"Normalized Distance ($D_k$) = $\frac{D - D_{\min}}{D_{\text{global\_max}}}$",
           fontsize = label_fontsize,
           ha = "left", va = "center")


# Row 3: Multisensory Integration
ax[2, 2].text(x = 0, y = 0.65,
           s = r"$p_{k} = \frac{\text{Drive}_{k}}{\sum \text{Drive}}$",
           fontsize = label_fontsize,
           ha = "left", va = "center")
ax[2, 2].text(x = 0, y = 0.5,
           s = r"Integration Index (II) = $\frac{-\sum p_{k} \log(p_{k})}{\log(K)}$",
           fontsize = label_fontsize,
           ha = "left", va = "center")
ax[2, 2].text(x = 0, y = 0.35,
           s = r"Integration Drive = II $\times$ Scaled Drive$_{\text{All}}$",
           fontsize = label_fontsize,
           ha = "left", va = "center")

# Row 4: Final Layer Synthesis
ax[3, 2].text(x = 0, y = 0.5,
           s = r"Final Score = $\sum_{k} (p_{k} \times D_{k})$",
           fontsize = label_fontsize,
           ha = "left", va = "center")

# Main Labels
place_panel_label(fig, ax[0, 0], "Ai", shx = -0.05, shy = 0.02,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[0, 1], "ii", shx = -0.03, shy = 0.02,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 0], "Bi", shx = -0.05, shy = 0.02,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 1], "ii", shx = -0.03, shy = 0.02,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[2, 0], "Ci", shx = -0.05, shy = 0.02,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[2, 1], "ii", shx = -0.03, shy = 0.02,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[3, 0], "D", shx = -0.05, shy = 0.02,
                  label_kwargs = label_kwargs)

# Save figure
plt.savefig(f"{folder}/Figure_05_Supplementary_Signal-Flow-Integration_ECDF.pdf",
            dpi = 1200,
            bbox_inches = "tight")

#------------------------------------------------------------------------------
# FIGURE STACKED BARS
senses = ["Olfactory", "Gustatory", "Visual", "Thermosensory"]
drive_cols = ["Olf_Mean_Drive", "Gust_Mean_Drive", "Vis_Mean_Drive", "Thermo_Mean_Drive"]

# Plotting DataFrame
bar_df = group_means[["Group"] + drive_cols].copy()

# Normalize rows to 1.0 (100%) to create proper stacked proportions
row_sums = bar_df[drive_cols].sum(axis = 1) + 1e-12
for col in drive_cols:
    bar_df[col] = bar_df[col] / row_sums

# Sort by Group Number
bar_df["Sort_Key"] = group_means["Group"]
bar_df = bar_df.sort_values(by = "Sort_Key").reset_index(drop = True)
bar_df = bar_df.drop(columns = ["Sort_Key"])

# Rename columns to match the cmap dictionary keys for easy plotting
bar_df.columns = ["Group"] + senses

#--------------------------------------
# Plot
fig, ax = plt.subplots(figsize = (8.5, 9), nrows = 3, ncols = 1)
plt.subplots_adjust(hspace = 0.4)

# Map your specific colors to the exact column names
color_list = [cmap[sense] for sense in senses]

# Chunks of 80
chunk_size = 80
for i, a in enumerate(ax):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    
    # Get the chunk
    chunk = bar_df.iloc[start_idx:end_idx]

    # Set Group as index so pandas plots the labels on the x-axis automatically
    chunk_plot = chunk.set_index("Group")
    
    # Plot the stacked bar
    chunk_plot.plot(kind    = "bar", 
                    stacked = True, 
                    ax      = a, 
                    color   = color_list, 
                    width   = 0.85, 
                    edgecolor = "none")
    
    # Clean up the aesthetics
    sns.despine(ax = a, top = True, right = True, left = True)
    # Ticks
    a.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = tick_fontsize,
        length    = 1.5,
        width     = 0.5,
        pad       = 1.5)
    a.tick_params(axis = "x", rotation = 90)
    # Limit
    a.set_xlim(-0.675, 79.675)
    
    # Tweaks
    a.set_ylim(0, 1.0)
    a.set_ylabel("Proportion of Drive", fontsize = label_fontsize)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # No X-label
    a.set_xlabel("")
    
    # Fix the Legend (Only put it on the top row to save space)
    if i == 0:
        a.legend(title           = "Sensory Modality", 
                  bbox_to_anchor = (1.01, 1), 
                  fontsize       = tick_fontsize,
                  title_fontsize = tick_fontsize,
                  loc            = "upper left", 
                  frameon        = False)
    else:
        a.get_legend().remove()

# Save figure
plt.savefig(f"{folder}/Figure_05_Supplementary_Signal-Drive-Stacked.pdf",
            dpi = 1200,
            bbox_inches = "tight")

#------------------------------------------------------------------------------
# Enrichment
enrichment = group_means[["Group",
                            "Olf_Enrichment", 
                            "Gust_Enrichment", 
                            "Vis_Enrichment", 
                            "Thermo_Enrichment"]]
enrichment.set_index("Group", inplace = True)
enrichment = enrichment.T

# Figure
fig, ax = plt.subplots(figsize = (8.5, 3), nrows = 3, ncols = 1)
plt.subplots_adjust(hspace = 0.4)

# Chunks of 80
chunk_size = 80
for i, a in enumerate(ax):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    
    # Get the chunk
    chunk = enrichment.iloc[:, start_idx:end_idx]
    
    # Heatmap
    sns.heatmap(chunk,
                ax = a, 
                cmap = "vlag",
                center = 0,  
                vmax = 2,
                vmin = -2,
                square = True,
                linewidth = 0.25,
                xticklabels = True,
                yticklabels = True,
                cbar_kws = dict(label  = "Enrichment",
                                shrink = 0.45,
                                aspect = 5,
                                pad    = 0.02,
                                extend = "both"))  

    # Axes cleanup
    cbar_axes = a.figure.axes[-1]
    cbar_axes.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = tick_fontsize - 2,
        length    = 0.5,
        width     = 0.15,
        pad       = 1.5)
    # Label
    cbar_axes.set_ylabel("Enrichment", fontsize = tick_fontsize - 2)
    
    # Ticks
    a.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = tick_fontsize - 1,
        length    = 1.5,
        width     = 0.5,
        pad       = 1.5)
    a.tick_params(axis = "x", rotation = 90)
    # Limit
    a.set_xlim(-0.675, 80.675)
    
    # Tweaks
    a.set_ylabel("", fontsize = label_fontsize)
    a.set_yticklabels(["Olfactory", "Gustatory", "Visual", "Thermosensory"],
                      fontsize = tick_fontsize - 1)

    # No X-label
    a.set_xlabel("")

# Save figure
plt.savefig(f"{folder}/Figure_05_Supplementary_Modality-Enrichment.pdf",
            dpi = 1200,
            bbox_inches = "tight")

#------------------------------------------------------------------------------
# t-SNEs
tsne_plot_df = pd.merge(tsne_df, acronym_means, on = "Acronym", how = "left")
tsne_plot_df["Label"] = tsne_plot_df["Label"].fillna("Other")

# Load the color scheme
with open(f"{folder}/Color_Scheme.pkl", "rb") as f:
    color_scheme = pkl.load(f)
order = list(color_scheme.keys())

# Panel Labels - Smaller here
label_kwargs = dict(fontsize = 12, va = "top", ha = "left")

# Plot
fig, ax = plt.subplots(figsize = (8.5, 5), nrows = 2, ncols = 4)
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

# Place simply by color coding to the labels
sns.scatterplot(
    data      = tsne_plot_df,
    x         = "t-SNE1",
    y         = "t-SNE2",
    hue       = "Label",
    hue_order = order,
    palette   = color_scheme,
    s         = 4,
    ax        = ax[0, 0],
    alpha     = 1,
    linewidth = 0.05)

# Tiny Black Dots for the comm
sns.scatterplot(tsne_plot_df.loc[tsne_plot_df["Label"].str.contains("com")],
                x   = "t-SNE1",
                y   = "t-SNE2",
                ax  = ax[0, 0],
                s   = 0.5,
                legend  = False,
                color = "black",
                linewidth = 0,
                alpha = 1,)

# Move the legend
sns.move_legend(ax[0, 0],
                loc = "upper left",
                bbox_to_anchor = (1.2, 1),
                ncol = 4,
                title = "",
                fontsize = tick_fontsize - 1)

# Each Scaled Sensory Drive
senses = ["Olfactory", "Gustatory", "Visual", "Thermosensory"]
drive_cols = ["Olf_Scaled_Drive", "Gust_Scaled_Drive", "Vis_Scaled_Drive", "Thermo_Scaled_Drive"]

# Max for scaling it all properly 
global_max_drive = tsne_plot_df[drive_cols].max().max()

for i, sense in enumerate(senses):
    # Subset
    plot_data = tsne_plot_df.sort_values(by = drive_cols[i])
    
    # Scatter
    sns.scatterplot(
        data      = plot_data,
        x         = "t-SNE1",
        y         = "t-SNE2",
        hue       = drive_cols[i],
        hue_norm  = (0, global_max_drive),
        palette   = "viridis",
        s         = 2,
        ax        = ax[1, i],
        alpha     = 1,
        linewidth = 0,
        legend    = False)

    # Title
    ax[1, i].set_title(sense, fontsize = label_fontsize, y = 1.02)


# Remove axes
for a in ax.flatten():
    a.axis("off")
    a.invert_yaxis()


# Main Labels
shx = -0.01
shy = 0.05
place_panel_label(fig, ax[0, 0], "A", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 0], "Bi", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 1], "ii", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 2], "iii", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)
place_panel_label(fig, ax[1, 3], "iv", shx = shx, shy = shy,
                  label_kwargs = label_kwargs)

# Teeny Tiny Axes
for a in [ax[0, 0], ax[1, 0]]:
    # Horizontal line (t-SNE 1)
    a.plot([0.0, 0.15], [0.0, 0.0],
            transform = a.transAxes,
            color     = 'black',
            lw        = 0.75,
            clip_on   = False)
    a.text(0.075, -0.02, "t-SNE 1",
            transform = a.transAxes,
            ha        = 'center',
            va        = 'top',
            fontsize  = tick_fontsize - 2)

    # Vertical line (t-SNE 2)
    a.plot([0.0, 0.0], [0.0, 0.15],
            transform = a.transAxes,
            color     = 'black',
            lw        = 0.75,
            clip_on   = False)
    a.text(-0.02, 0.075, "t-SNE 2",
            transform = a.transAxes,
            ha        = 'right',
            va        = 'center',
            rotation  = 90,
            fontsize  = tick_fontsize - 2)

# CBAR
cbar_ax = fig.add_axes([0.4, 0.02, 0.2, 0.02])
norm    = mcolors.Normalize(vmin = 0, vmax = global_max_drive)
sm      = cm.ScalarMappable(cmap = "viridis", norm = norm)
sm.set_array([])

cbar = fig.colorbar(sm, cax = cbar_ax, orientation = "horizontal")
cbar.set_label("Scaled Drive (Steady-State Signal)",
               fontsize = tick_fontsize,
               labelpad = 4)
cbar.ax.tick_params(labelsize = tick_fontsize - 1)
cbar.outline.set_visible(False)

# Save figure
plt.savefig(f"{folder}/Figure_05_Supplementary_Acronym-tSNEs.pdf",
            dpi = 1200,
            bbox_inches = "tight")

#------------------------------------------------------------------------------
# Status
print("हो गया दोस्तों!")