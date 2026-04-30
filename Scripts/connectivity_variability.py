"""
Connectivity variability across L1 neuron groupings.

Connectivity stereotypy was assessed at two grouping resolutions: fine
(Cluster, "Types") and coarse (Acronym, "Classes"). Analyses were
restricted to differentiated central brain neurons; sensory lineages were
excluded. The connectivity matrix W (N x N, presynaptic x postsynaptic,
duplicate edges summed) was used to define the cosine similarity of any
two neurons over their outgoing rows (output similarity) and incoming
columns (input similarity). The full input and output similarity matrices
were computed once and sliced for all downstream comparisons.

For each (Lineage, Hemisphere, Group) with k > 1 neurons, within-group
stereotypy was summarized as the mean off-diagonal cosine similarity,
separately for inputs and outputs. Two null distributions were generated
(n = 10,000 permutations each). The global shuffle randomly permuted all
weights across (pre, post) positions, destroying both the connectivity
backbone and group identity. The label shuffle preserved the empirical
backbone and only randomized neuron-to-group assignments within each
(Lineage, Hemisphere) block at fixed group sizes, isolating the
contribution of group identity. Significance was assessed by a
permutation test on the mean within-group similarity (one-sided, fraction
of shuffles >= observed) and visualized as the 95% ECDF band over
permutations against the true ECDF.

Within- versus across-group similarity was tested per (Lineage,
Hemisphere) by partitioning all neuron pairs in the block into within-
and across-group pairs. To control for the larger number of across-group
pairs, n_within pairs were drawn without replacement from the across-
group pool n = 10,000 times; the resulting distribution of resampled
across-group means was compared to the observed within-group means with
a one-sided permutation test, separately for inputs and outputs. Finally,
the empirical input and output within-group similarity distributions were
compared with a two-sample Kolmogorov-Smirnov test. All randomness was
seeded via a single numpy SeedSequence (master seed = 39335) for full
reproducibility.

Usage
    python connectivity_variability.py --level both           # reuses cache
    python connectivity_variability.py --level Cluster --redo
    python connectivity_variability.py --level Acronym --redo --n 10000

"""
import os

# Cap BLAS threads before importing numpy/sklearn -- we parallelize with
# threading, so each BLAS call should stay single-threaded to avoid
# oversubscription.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity

#------------------------------------------------------------------------------
# Paths
HOME         = os.environ["PROJECTS_HOME"]
# Paths
connectivity_parquet = "Data_Connectivity/Connectivity.parquet"
neuron_info_csv      = f"{HOME}/L1_Skeletons/Data_Summary/L1_NeuronInformation_PK.csv"
default_out          = "Analysis_Outputs/Variability"

#------------------------------------------------------------------------------
# Constants
hemispheres = ["Right", "Left"]

sensory = {"S-AN", "S-LNa", "S-LNad", "S-LNp", "S-MNa", "S-MNp", "S-MNp2",
           "S-PNa", "S-PNm", "S-PNp", "S-PaN", "BolwigsAxon"}

x_grid = np.linspace(0, 1.0, 301)

#------------------------------------------------------------------------------
# Plot params
linewidth      = 0.75
title_fontsize = 8.5
label_fontsize = 7
tick_fontsize  = 6
fill_alpha     = 0.25

#------------------------------------------------------------------------------
# Data loading
def load_inputs(level):
    """
    Load neuron info and build the (N x N) connectivity matrix.
    Rows are presynaptic, columns postsynaptic.

    """
    assert level in ("Cluster", "Acronym")

    neuron_df    = pd.read_csv(neuron_info_csv)
    all_lineages = set(neuron_df["Lineage"].dropna().unique())

    # Subset the neuron_df
    neuron_df = neuron_df.loc[neuron_df["Category"] == "Diff"]
    neuron_df = neuron_df.loc[~neuron_df["Cluster"].isna()].copy()

    if level == "Acronym":
        neuron_df["Acronym"] = neuron_df["Acronym"].fillna("NA")

    # Index map for the connectivity matrix
    neuron_df   = neuron_df[["bodyId", "Lineage", "Hemisphere", level]].copy()
    body_ids    = np.sort(neuron_df["bodyId"].unique())
    body_to_idx = {int(b) : i for i, b in enumerate(body_ids)}
    neuron_df["body_idx"] = neuron_df["bodyId"].map(body_to_idx).astype(np.int64)

    # Remove the sensory
    lineages = np.array(sorted(all_lineages - sensory))

    # Connectivity edge list
    conn = pd.read_parquet(connectivity_parquet,
                           columns = ["bodyId_pre", "bodyId_post", "weight"])
    conn = conn[conn["bodyId_pre"].isin(body_to_idx) &
                conn["bodyId_post"].isin(body_to_idx)]

    # Edge list may contain duplicates -- sum them
    n    = len(body_ids)
    W    = np.zeros((n, n), dtype = np.float32)
    rows = conn["bodyId_pre"].map(body_to_idx).to_numpy(dtype = np.int64)
    cols = conn["bodyId_post"].map(body_to_idx).to_numpy(dtype = np.int64)
    np.add.at(W, (rows, cols), conn["weight"].to_numpy(dtype = np.float32))

    return W, body_ids, neuron_df, lineages


#------------------------------------------------------------------------------
# Group bookkeeping
def build_groups(neuron_df, lineages, level):
    """
    Enumerate every (Lineage, Hemisphere, <level>) group and record the
    body_idx values for each.

    """
    rows, indices = [], []
    # Loop
    for lineage in lineages:
        for hemisphere in hemispheres:
            sub_df = neuron_df.loc[(neuron_df["Lineage"]    == lineage) &
                                   (neuron_df["Hemisphere"] == hemisphere)]
            if sub_df.empty:
                continue
            for grp, part in sub_df.groupby(level, sort = True):
                idx = part["body_idx"].to_numpy(dtype = np.int64)
                rows.append({"Lineage"     : lineage,
                             "Hemisphere"  : hemisphere,
                             level         : grp,
                             "Num_Neurons" : len(idx)})
                indices.append(idx)

    groups_df = pd.DataFrame(rows).reset_index(drop = True)
    return groups_df, indices

def group_means_from_full_sims(sim_in_full, sim_out_full, group_idx):
    """
    Mean off-diagonal within-group cosine similarity from precomputed
    N x N similarity matrices. Groups of size 0 -> nan, size 1 -> 1.

    """
    n_groups  = len(group_idx)
    in_means  = np.empty(n_groups, dtype = np.float32)
    out_means = np.empty(n_groups, dtype = np.float32)
    # Loop
    for g, idx in enumerate(group_idx):
        k = len(idx)
        if k == 0:
            in_means[g]  = np.nan
            out_means[g] = np.nan
        elif k == 1:
            in_means[g]  = 1.0
            out_means[g] = 1.0
        else:
            block_in  = sim_in_full[np.ix_(idx, idx)]
            block_out = sim_out_full[np.ix_(idx, idx)]
            # Off-diagonal mean -- matrix is symmetric so this is fine
            denom = k * (k - 1)
            in_means[g]  = (block_in.sum()  - np.trace(block_in))  / denom
            out_means[g] = (block_out.sum() - np.trace(block_out)) / denom

    return in_means, out_means

#------------------------------------------------------------------------------
# Shuffle workers
def run_global_shuffle(seed, W_flat, shape, group_idx):
    """
    One global-shuffle iteration. Shuffles all weights and computes the full
    N x N cosine similarity once -- much faster than per-group slicing.

    """
    rng  = np.random.default_rng(seed)
    flat = W_flat.copy()
    rng.shuffle(flat)
    Ws      = flat.reshape(shape)
    sim_out = cosine_similarity(Ws).astype(np.float32)
    sim_in  = cosine_similarity(Ws.T).astype(np.float32)
    return group_means_from_full_sims(sim_in, sim_out, group_idx)

def run_label_shuffle(seed, sim_in_full, sim_out_full,
                      block_positions, base_group_idx, group_block_id):
    """
    One label-shuffle iteration. Permutes neuron labels within each
    (Lineage, Hemisphere) block, preserving group sizes. Equivalent to
    permuting which neurons receive which group label.

    """
    rng = np.random.default_rng(seed)
    # Permute each block and split back into groups by their original sizes
    shuffled_group_idx = [None] * len(base_group_idx)
    for b, pool in enumerate(block_positions):
        perm_pool = rng.permutation(pool)
        # Groups in this block, in canonical order -- their sizes partition the pool
        sel    = np.where(group_block_id == b)[0]
        offset = 0
        for g in sel:
            k = len(base_group_idx[g])
            shuffled_group_idx[g] = perm_pool[offset:offset + k]
            offset += k

    return group_means_from_full_sims(sim_in_full, sim_out_full,
                                      shuffled_group_idx)

#------------------------------------------------------------------------------
# In-vs-across group resampling
def pairs_and_resample(sim_in_full, sim_out_full, neuron_df, lineages,
                       level, n_iter, master_seed):
    """
    For each (Lineage, Hemisphere), measure within-group and across-group
    pair similarities. Resample n_within pairs from across-group pairs
    n_iter times so the two distributions are size-matched.

    """
    ss          = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(len(lineages) * len(hemispheres))
    seed_iter   = iter(child_seeds)

    pair_rows    = []
    res_in_cols  = []
    res_out_cols = []

    # Loop
    for lineage in lineages:
        for hemisphere in hemispheres:
            seed   = next(seed_iter)
            sub_df = neuron_df.loc[(neuron_df["Lineage"]    == lineage) &
                                   (neuron_df["Hemisphere"] == hemisphere)]
            if sub_df.empty:
                continue
            idx    = sub_df["body_idx"].to_numpy()
            labels = sub_df[level].to_numpy()
            k      = len(idx)
            if k < 2:
                continue

            block_in  = sim_in_full[np.ix_(idx, idx)]
            block_out = sim_out_full[np.ix_(idx, idx)]

            # MASKS -----------------------
            same = labels[:, None] == labels[None, :]
            diag = ~np.eye(k, dtype = bool)

            within_mask  = diag & same
            between_mask = diag & ~same

            n_within  = int(within_mask.sum())
            n_between = int(between_mask.sum())

            within_in   = block_in[within_mask]
            within_out  = block_out[within_mask]
            between_in  = block_in[between_mask]
            between_out = block_out[between_mask]

            # OVERALL SIMILARITY MEANS-----
            pair_rows.append({
                "Lineage"                    : lineage,
                "Hemisphere"                 : hemisphere,
                "Num_Neurons"                : k,
                "Num_Within_Pairs"           : n_within,
                "Num_Between_Pairs"          : n_between,
                "Within_Cosine_Sim_Inputs"   : float(np.nanmean(within_in))  if n_within  else np.nan,
                "Between_Cosine_Sim_Inputs"  : float(np.nanmean(between_in)) if n_between else np.nan,
                "Within_Cosine_Sim_Outputs"  : float(np.nanmean(within_out)) if n_within  else np.nan,
                "Between_Cosine_Sim_Outputs" : float(np.nanmean(between_out)) if n_between else np.nan,
            })

            # Resample only when there are more across-group pairs than within
            if n_between > 0 and n_within > 0 and n_within < n_between:
                rng = np.random.default_rng(seed)
                # Without-replacement sample of n_within from n_between, n_iter times (argpartition trick)
                noise    = rng.random((n_iter, n_between), dtype = np.float32)
                take     = np.argpartition(noise, n_within - 1, axis = 1)[:, :n_within]
                mean_in  = between_in[take].mean(axis = 1).astype(np.float32)
                mean_out = between_out[take].mean(axis = 1).astype(np.float32)
            else:
                mean_in  = np.full(n_iter, np.nan, dtype = np.float32)
                mean_out = np.full(n_iter, np.nan, dtype = np.float32)

            res_in_cols.append(mean_in)
            res_out_cols.append(mean_out)

    pairs_df      = pd.DataFrame(pair_rows).reset_index(drop = True)
    resampled_in  = (np.stack(res_in_cols,  axis = 1) if res_in_cols
                     else np.zeros((n_iter, 0), dtype = np.float32))
    resampled_out = (np.stack(res_out_cols, axis = 1) if res_out_cols
                     else np.zeros((n_iter, 0), dtype = np.float32))
    return pairs_df, resampled_in, resampled_out


#------------------------------------------------------------------------------
# Top-level compute
def compute_level(level, n_iter, out_dir, redo = False,
                  master_seed = 39335, n_jobs = -1):
    """
    Run the full variability pipeline for one grouping level and save to
    out_dir/<level>/. When redo is False, the heavy shuffles + resampling are
    skipped if the cached files are already on disk (the true similarity is
    always cheap, so it is recomputed every call).

    """
    out_dir = f"{out_dir}/{level}"
    os.makedirs(out_dir, exist_ok = True)

    print(f"[{level}] Loading inputs...")
    W, body_ids, neuron_df, lineages = load_inputs(level)
    print(f"[{level}]   {len(body_ids)} neurons, {len(lineages)} lineages")

    print(f"[{level}] Precomputing full N x N cosine similarity...")
    sim_out_full = cosine_similarity(W).astype(np.float32)     # row sims
    sim_in_full  = cosine_similarity(W.T).astype(np.float32)   # col sims

    print(f"[{level}] Building group index...")
    groups_df, group_idx = build_groups(neuron_df, lineages, level)
    n_groups = len(groups_df)
    print(f"[{level}]   {n_groups} groups")
    groups_df.to_parquet(f"{out_dir}/groups_index.parquet")

    #-------------------------------------------------
    # True
    print(f"[{level}] True similarity...")
    true_in, true_out = group_means_from_full_sims(sim_in_full, sim_out_full,
                                                   group_idx)
    np.savez_compressed(f"{out_dir}/true.npz",
             input  = true_in.astype(np.float32),
             output = true_out.astype(np.float32))

    #-------------------------------------------------
    # Block bookkeeping for label shuffle.
    # Each (Lineage, Hemisphere) is a pool within which labels are shuffled,
    # preserving group sizes.
    block_positions = []
    group_block_id  = np.full(n_groups, -1, dtype = np.int64)
    block_index     = {}
    b               = 0
    # Same iteration order as build_groups
    for lineage in lineages:
        for hemisphere in hemispheres:
            sub_df = neuron_df.loc[(neuron_df["Lineage"]    == lineage) &
                                   (neuron_df["Hemisphere"] == hemisphere)]
            if sub_df.empty:
                continue
            block_positions.append(sub_df["body_idx"].to_numpy(dtype = np.int64))
            block_index[(lineage, hemisphere)] = b
            b += 1

    for g in range(n_groups):
        key = (groups_df.loc[g, "Lineage"], groups_df.loc[g, "Hemisphere"])
        group_block_id[g] = block_index[key]

    #-------------------------------------------------
    # Cached files for the heavy steps
    gl_path    = f"{out_dir}/global_shuffle.npz"
    lab_path   = f"{out_dir}/label_shuffle.npz"
    pairs_path = f"{out_dir}/pairs_index.parquet"
    res_path   = f"{out_dir}/invsout_resampled.npz"

    cached = (os.path.exists(gl_path)  and os.path.exists(lab_path) and
              os.path.exists(pairs_path) and os.path.exists(res_path))

    if not redo and cached:
        print(f"[{level}] Cached shuffles + resampling found -- skipping recomputation.")
        print(f"[{level}] Done. Saved to {out_dir}")
        return

    #-------------------------------------------------
    # Global shuffle
    print(f"[{level}] Global shuffle: {n_iter} runs (parallel)...")
    ss          = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(n_iter)
    W_flat      = W.ravel()
    shape       = W.shape

    # Threading -- BLAS releases the GIL and avoids pickling the large matrix
    gl = Parallel(n_jobs = n_jobs, backend = "threading", verbose = 5)(
        delayed(run_global_shuffle)(s, W_flat, shape, group_idx)
        for s in child_seeds)
    gl_in  = np.stack([r[0] for r in gl], axis = 0).astype(np.float32)
    gl_out = np.stack([r[1] for r in gl], axis = 0).astype(np.float32)
    np.savez_compressed(gl_path,
             input  = gl_in,
             output = gl_out)

    #-------------------------------------------------
    # Label shuffle
    print(f"[{level}] Label shuffle: {n_iter} runs (parallel)...")
    # Same seed sequence as above for reproducibility
    ss2          = np.random.SeedSequence(master_seed)
    child_seeds2 = ss2.spawn(n_iter)

    lab = Parallel(n_jobs = n_jobs, backend = "threading", verbose = 5)(
        delayed(run_label_shuffle)(s, sim_in_full, sim_out_full,
                                   block_positions, group_idx, group_block_id)
        for s in child_seeds2)
    lab_in  = np.stack([r[0] for r in lab], axis = 0).astype(np.float32)
    lab_out = np.stack([r[1] for r in lab], axis = 0).astype(np.float32)
    np.savez_compressed(lab_path,
             input  = lab_in,
             output = lab_out)

    #-------------------------------------------------
    # In-vs-across resampling
    print(f"[{level}] In-vs-across resampling...")
    pairs_df, res_in, res_out = pairs_and_resample(
        sim_in_full, sim_out_full, neuron_df, lineages, level,
        n_iter      = n_iter,
        master_seed = master_seed)
    pairs_df.to_parquet(pairs_path)
    np.savez_compressed(res_path,
             input  = res_in.astype(np.float32),
             output = res_out.astype(np.float32))

    print(f"[{level}] Done. Saved to {out_dir}")


#------------------------------------------------------------------------------
# Loading helpers (usable from a notebook)
def load_results(level, out_dir = None):
    """
    Load all cached results for one grouping level.

    """
    if out_dir is None:
        out_dir = default_out
    d      = f"{out_dir}/{level}"
    groups = pd.read_parquet(f"{d}/groups_index.parquet")
    pairs  = pd.read_parquet(f"{d}/pairs_index.parquet")

    def _np(name):
        z = np.load(f"{d}/{name}")
        return {"input" : z["input"], "output" : z["output"]}

    return {"groups"         : groups,
            "true"           : _np("true.npz"),
            "global_shuffle" : _np("global_shuffle.npz"),
            "label_shuffle"  : _np("label_shuffle.npz"),
            "pairs"          : pairs,
            "resampled"      : _np("invsout_resampled.npz")}


def as_long_df(values, groups, value_col):
    """
    Expand a (N_runs, N_groups) or (N_groups,) array into a long DataFrame.
    Useful for seaborn / older plotting code that expects the previous CSV format.

    """
    if values.ndim == 1:
        out = groups.copy()
        out[value_col] = values
        return out
    n_runs, n_groups      = values.shape
    rep_groups            = pd.concat([groups] * n_runs, ignore_index = True)
    rep_groups["Run"]     = np.repeat(np.arange(n_runs), n_groups)
    rep_groups[value_col] = values.ravel()
    return rep_groups


#------------------------------------------------------------------------------
# Stats helpers
def ecdf_band(values_matrix, mask = None, x_grid = x_grid):
    """
    Median + 95% CI ECDF across runs from a (N_runs, N_groups) matrix.
    `mask` is an optional boolean filter over groups (e.g. Num_Neurons > 1).

    """
    if mask is not None:
        values_matrix = values_matrix[:, mask]
    # Drop groups with NaN everywhere
    finite_any    = np.any(np.isfinite(values_matrix), axis = 0)
    values_matrix = values_matrix[:, finite_any]
    n_runs, n     = values_matrix.shape
    if n == 0:
        return None, None, None

    # Pre-allocate (fastest memory layout)
    sorted_per_run = np.sort(values_matrix, axis = 1)
    ecdf           = np.empty((n_runs, len(x_grid)), dtype = np.float64)
    for i in range(n_runs):
        ecdf[i] = np.searchsorted(sorted_per_run[i], x_grid, side = "right") / n

    return (np.median(ecdf,   axis = 0),
            np.quantile(ecdf, 0.025, axis = 0),
            np.quantile(ecdf, 0.975, axis = 0))


def perm_pvalue(shuffle_matrix, true_mean, mask = None):
    """
    Fraction of shuffle runs whose mean >= observed true mean.

    """
    if mask is not None:
        shuffle_matrix = shuffle_matrix[:, mask]
    per_run = np.nanmean(shuffle_matrix, axis = 1)
    return float(np.mean(per_run >= true_mean))

#------------------------------------------------------------------------------
# Plotting
def style_axis(ax, xlabel = "Cosine Similarity"):
    import seaborn as sns
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, labelpad = 10, fontsize = label_fontsize)
    ax.set_ylabel("")
    sns.despine(ax = ax)
    ax.tick_params(
        axis      = "both",
        which     = "major",
        color     = "black",
        labelsize = tick_fontsize,
        length    = 2,
        width     = 0.75,
        pad       = 1.5)


def plot_shuffle_ecdf(results, level, out_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ks_2samp
    sns.set_style("ticks")

    groups = results["groups"]
    keep   = (groups["Num_Neurons"].to_numpy() > 1)

    true_in  = results["true"]["input"]
    true_out = results["true"]["output"]
    gl_in    = results["global_shuffle"]["input"]
    gl_out   = results["global_shuffle"]["output"]
    lab_in   = results["label_shuffle"]["input"]
    lab_out  = results["label_shuffle"]["output"]

    # True means
    true_in_k     = true_in[keep]
    true_out_k    = true_out[keep]
    true_in_mean  = float(np.nanmean(true_in_k))
    true_out_mean = float(np.nanmean(true_out_k))

    # Bands
    gl_m_in,   gl_lo_in,   gl_hi_in   = ecdf_band(gl_in,  mask = keep)
    gl_m_out,  gl_lo_out,  gl_hi_out  = ecdf_band(gl_out, mask = keep)
    lab_m_in,  lab_lo_in,  lab_hi_in  = ecdf_band(lab_in,  mask = keep)
    lab_m_out, lab_lo_out, lab_hi_out = ecdf_band(lab_out, mask = keep)

    # Permutation p-values
    gl_in_p   = perm_pvalue(gl_in,  true_in_mean,  mask = keep)
    gl_out_p  = perm_pvalue(gl_out, true_out_mean, mask = keep)
    lab_in_p  = perm_pvalue(lab_in,  true_in_mean,  mask = keep)
    lab_out_p = perm_pvalue(lab_out, true_out_mean, mask = keep)

    n_runs        = gl_in.shape[0]
    shuffle_label = f"{level} Shuffle Median"

    fig, ax = plt.subplots(figsize = (8.5, 2.5), ncols = 3, sharey = True)
    plt.subplots_adjust(wspace = 0.15)

    #-------------------------------------------------------------
    # GLOBAL SHUFFLE
    ax[0].fill_between(x_grid, gl_lo_in,  gl_hi_in,  color = "grey", alpha = fill_alpha)
    ax[0].plot(x_grid, gl_m_in,
               color     = "grey",
               linewidth = linewidth,
               label     = "Global Shuffle Median")
    ax[1].fill_between(x_grid, gl_lo_out, gl_hi_out, color = "grey", alpha = fill_alpha)
    ax[1].plot(x_grid, gl_m_out,
               color     = "grey",
               linewidth = linewidth,
               label     = "Global Shuffle Median")

    #-------------------------------------------------------------
    # LABEL SHUFFLE
    ax[0].fill_between(x_grid, lab_lo_in,  lab_hi_in,  color = "magenta", alpha = fill_alpha)
    ax[0].plot(x_grid, lab_m_in,
               color     = "magenta",
               linewidth = linewidth,
               label     = shuffle_label)
    ax[1].fill_between(x_grid, lab_lo_out, lab_hi_out, color = "magenta", alpha = fill_alpha)
    ax[1].plot(x_grid, lab_m_out,
               color     = "magenta",
               linewidth = linewidth,
               label     = shuffle_label)

    #-------------------------------------------------------------
    # TRUE
    ti  = true_in_k[np.isfinite(true_in_k)]
    to_ = true_out_k[np.isfinite(true_out_k)]
    # Separated
    sns.ecdfplot(x = ti,  ax = ax[0], color = "green", linewidth = linewidth)
    sns.ecdfplot(x = to_, ax = ax[1], color = "red",   linewidth = linewidth)
    # United
    sns.ecdfplot(x = ti,  ax = ax[2], color = "green", linewidth = linewidth, label = "True Input")
    sns.ecdfplot(x = to_, ax = ax[2], color = "red",   linewidth = linewidth, label = "True Output")

    # Comparison
    stat, p_value = ks_2samp(ti, to_)
    ax[2].text(0.95, 0.1,
               f"KS-stat  : {stat:0.3f}\np-value : {p_value:0.3f}",
               va       = "center",
               ha       = "right",
               fontsize = label_fontsize)

    # Styling
    for a in ax:
        style_axis(a)
    ax[0].set_ylabel("Cumulative Proportion", labelpad = 10, fontsize = label_fontsize)
    ax[0].set_title("Inputs",                    y = 1.05, fontsize = title_fontsize)
    ax[1].set_title("Outputs",                   y = 1.05, fontsize = title_fontsize)
    ax[2].set_title("Input | Output Similarity", y = 1.05, fontsize = title_fontsize)

    # Legends
    ax[0].legend(loc = "upper left", bbox_to_anchor = (3.4, 0.85), fontsize = label_fontsize)
    ax[2].legend(loc = "upper left", bbox_to_anchor = (1.1, 0.65), fontsize = label_fontsize)

    #----------------------------------------------------------------------------
    # Permutation stats
    for a, p_lab, p_gl in [(ax[0], lab_in_p, gl_in_p),
                           (ax[1], lab_out_p, gl_out_p)]:
        a.text(0.95, 0.16,
               r"Permutation Test ($\mu$ comp.)",
               va = "center", color = "black", ha = "right", fontsize = label_fontsize)
        if p_lab <= 0:
            a.text(0.95, 0.10, f"p-value < {1 / n_runs:0.4f}",
                   va = "center", color = "magenta", ha = "right", fontsize = label_fontsize)
        else:
            a.text(0.95, 0.10, f"p = {p_lab:0.4f}",
                   va = "center", color = "magenta", ha = "right", fontsize = label_fontsize)
        if p_gl <= 0:
            a.text(0.95, 0.04, f"p-value < {1 / n_runs:0.4f}",
                   va = "center", color = "grey", ha = "right", fontsize = label_fontsize)
        else:
            a.text(0.95, 0.04, f"p = {p_gl:0.4f}",
                   va = "center", color = "grey", ha = "right", fontsize = label_fontsize)

    #----------------------------------------------------------------------------
    # Mean placement
    ax[0].text(0.35, 0.5,
               f"$\\mu$ = {true_in_mean:.3f}",
               color = "green", va = "center", ha = "left", fontsize = label_fontsize)
    ax[1].text(0.30, 0.5,
               f"$\\mu$ = {true_out_mean:.3f}",
               color = "red", va = "center", ha = "left", fontsize = label_fontsize)

    plt.savefig(out_path, bbox_inches = "tight", dpi = 1200)
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_in_vs_out_ecdf(results, level, out_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("ticks")

    pairs = results["pairs"]
    keep  = (pairs["Num_Within_Pairs"].to_numpy() > 0)

    res_in  = results["resampled"]["input"]
    res_out = results["resampled"]["output"]

    # Drop any all-NaN columns (pairs where resampling didn't apply)
    col_finite = np.any(np.isfinite(res_in), axis = 0)
    res_in     = res_in[:,  col_finite]
    res_out    = res_out[:, col_finite]

    comp_m_in,  comp_lo_in,  comp_hi_in  = ecdf_band(res_in)
    comp_m_out, comp_lo_out, comp_hi_out = ecdf_band(res_out)

    within_in_mean  = float(np.nanmean(pairs.loc[keep, "Within_Cosine_Sim_Inputs"]))
    within_out_mean = float(np.nanmean(pairs.loc[keep, "Within_Cosine_Sim_Outputs"]))

    in_p   = perm_pvalue(res_in,  within_in_mean)
    out_p  = perm_pvalue(res_out, within_out_mean)
    n_runs = res_in.shape[0] if res_in.size else 1

    fig, ax = plt.subplots(figsize = (4.5, 3.5))

    #-------------------------------------------------------------
    # True
    sns.ecdfplot(x = pairs.loc[keep, "Within_Cosine_Sim_Inputs"],
                 color     = "green",
                 ax        = ax,
                 linewidth = linewidth,
                 label     = "Inputs : Within")
    sns.ecdfplot(x = pairs.loc[keep, "Within_Cosine_Sim_Outputs"],
                 color     = "red",
                 ax        = ax,
                 linewidth = linewidth,
                 label     = "Outputs : Within")

    # Resampled
    if comp_m_in is not None:
        ax.fill_between(x_grid, comp_lo_in, comp_hi_in,
                        color = "#0BDA51", alpha = fill_alpha)
        ax.plot(x_grid, comp_m_in,
                color     = "#0BDA51",
                linewidth = linewidth,
                label     = "Inputs : Across")
    if comp_m_out is not None:
        ax.fill_between(x_grid, comp_lo_out, comp_hi_out,
                        color = "magenta", alpha = fill_alpha)
        ax.plot(x_grid, comp_m_out,
                color     = "magenta",
                linewidth = linewidth,
                label     = "Outputs : Across")

    style_axis(ax, xlabel = "Connectivity Cosine Similarity")
    ax.set_ylabel("Cumulative Proportion", labelpad = 10, fontsize = label_fontsize)

    #----------------------------------------------------------------------------
    # Permutation stats
    ax.text(1, 0.2,
            r"Permutation Test ($\mu$ comp.)",
            va = "center", color = "black", ha = "right", fontsize = label_fontsize)
    if in_p <= 0:
        ax.text(1, 0.12, f"p-value < {1 / n_runs:0.4f}",
                va = "center", color = "#0BDA51", ha = "right", fontsize = label_fontsize)
    else:
        ax.text(1, 0.12, f"p = {in_p:0.4f}",
                va = "center", color = "#0BDA51", ha = "right", fontsize = label_fontsize)
    if out_p <= 0:
        ax.text(1, 0.04, f"p-value < {1 / n_runs:0.4f}",
                va = "center", color = "magenta", ha = "right", fontsize = label_fontsize)
    else:
        ax.text(1, 0.04, f"p = {out_p:0.4f}",
                va = "center", color = "magenta", ha = "right", fontsize = label_fontsize)

    #----------------------------------------------------------------------------
    # Mean placement
    ax.text(0.35, 0.5,
            f"$\\mu$ = {within_in_mean:.3f}",
            color = "green", va = "center", ha = "left", fontsize = label_fontsize)
    ax.text(0.30, 0.425,
            f"$\\mu$ = {within_out_mean:.3f}",
            color = "red", va = "center", ha = "left", fontsize = label_fontsize)

    ax.legend(loc = "upper left", bbox_to_anchor = (1.02, 1.0), fontsize = label_fontsize)

    plt.savefig(out_path, bbox_inches = "tight", dpi = 1200)
    print(f"  Saved {out_path}")
    plt.close(fig)

#------------------------------------------------------------------------------
# CLI
def main():
    p = argparse.ArgumentParser(description = "Connectivity variability at Cluster / Acronym level")
    p.add_argument("--level", choices = ["Cluster", "Acronym", "both"], required = True)
    p.add_argument("--redo",  action  = "store_true",
                   help = "Recompute the shuffles + resampling. Plots are always produced.")
    p.add_argument("--n",     type    = int, default = 10000,
                   help = "Number of permutation runs (default 10000)")
    p.add_argument("--out",   default = default_out)
    p.add_argument("--seed",  type    = int, default = 39335)
    p.add_argument("--jobs",  type    = int, default = -1)

    args    = p.parse_args()
    levels  = ["Cluster", "Acronym"] if args.level == "both" else [args.level]
    out_dir = args.out

    # Compute (or reuse cache) and plot for every requested level
    for lvl in levels:
        compute_level(lvl,
                      n_iter      = args.n,
                      out_dir     = out_dir,
                      redo        = args.redo,
                      master_seed = args.seed,
                      n_jobs      = args.jobs)

        print(f"[{lvl}] Loading cached results...")
        r = load_results(lvl, out_dir = out_dir)
        plot_shuffle_ecdf(r,   lvl, f"{out_dir}/{lvl}/Shuffle_ECDF_{lvl}.pdf")
        plot_in_vs_out_ecdf(r, lvl, f"{out_dir}/{lvl}/Across_{lvl}_ECDF.pdf")

    print("हो गया दोस्तों!")

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
