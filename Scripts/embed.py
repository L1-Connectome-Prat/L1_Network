"""
Spectral embedding the Neuron Classes Connectivity, to identify clusters.

"""
from itertools import combinations
import pandas as pd
import numpy as np
from glob import glob

# Stats
from graspologic.embed import AdjacencySpectralEmbed
from matplotlib.gridspec import GridSpec
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Linkage
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
# Parameter Search
from sklearn.model_selection import ParameterGrid

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_style("whitegrid")

#-----------------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
import sys
sys.setrecursionlimit(1000)

#-----------------------------------------------------------------------------------------------------------------------
__colors__ = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
              '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
              '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
              '#ffd8b1', '#000075', '#a9a9a9']

re_embed = False
folder   = "Analysis_Outputs"
c_file   =  "Data_Connectivity/Connectivity.csv"
#-----------------------------------------------------------------------------------------------------------------------
def read_connectivity(path : str = c_file):
    """
    Just an easy way to load in the connectivity matrix.
    """
    # Loading in the connectivity
    conn_df = pd.read_csv(path)
    conn_df.fillna("NA", inplace = True)

    # Remove S-PaN
    # conn_df = conn_df.loc[~(conn_df["Class_pre"].str.contains("S-PaN")) &
    #                       ~(conn_df["Class_post"].str.contains("S-PaN"))]

    # Remove S-PaN
    conn_df = conn_df.loc[~(conn_df["Lineage_pre"].str.contains("S-PaN")) &
                          ~(conn_df["Lineage_post"].str.contains("S-PaN"))]

    # Isolate all the classes that we have
    # all_classes = sorted(set(conn_df["Class_pre"]) | set(conn_df["Class_post"]))
    all_classes = sorted(set(conn_df["VolkerClass_pre"]) | set(conn_df["VolkerClass_post"]))
    all_classes = [f"{a}_{h}" for a in all_classes for h in ["Left", "Right"]]

    # Manual Fix to Add in Zero connected
    zero_classes = {"S-LNad_NA_Left", "S-LNad_NA_Right",
                    "S-MNa_NA_Left", "S-MNa_NA_Right",
                    "S-PNm_NA_Left", "S-PNm_NA_Right"}
    # Add
    all_classes = set(all_classes) | zero_classes
    all_classes = sorted(list(all_classes))

    # Update the names with hemisphere
    conn_df["VolkerClass_pre"]  = conn_df["VolkerClass_pre"] + "_" + conn_df["Hemisphere_pre"]
    conn_df["VolkerClass_post"] = conn_df["VolkerClass_post"] + "_" + conn_df["Hemisphere_post"]

    # Convert to matrix
    # conn_matrix = pd.crosstab(index = conn_df.Class_pre, columns = conn_df.Class_post,
    #                           values = conn_df.weight, aggfunc = "sum"). fillna(0)
    conn_matrix = pd.crosstab(index = conn_df.VolkerClass_pre, columns = conn_df.VolkerClass_post,
                              values = conn_df.weight, aggfunc = "sum"). fillna(0)

    # Reindex
    conn_matrix = conn_matrix.reindex(columns = all_classes, index = all_classes).fillna(0)

    return conn_matrix, all_classes

#-----------------------------------------------------------------------------------------------------------------------
def compress_linkage(Z, mode = "sqrt"):
    """
    Monotonically compress merge distances for visualization only.
    Does not affect tree topology or leaf order.
    """
    Z_vis = Z.copy()
    if mode == "sqrt":
        Z_vis[:, 2] = np.sqrt(Z_vis[:, 2])
    elif mode == "log":
        Z_vis[:, 2] = np.log1p(Z_vis[:, 2])
    return Z_vis

#-----------------------------------------------------------------------------------------------------------------------
def split_hemispheres(matrix, embed):
    """
    Organizing the left and right components of the embedding into separate matrices.
    Returns
    -------
    left_df, right_df
        DataFrame with size (all_classes / 2) x (n_spectral_dims x 2)

    """
    # Convert to DataFrame
    embed_df = pd.DataFrame(embed, index = matrix.index)

    # Since we organized all the pairs correctly, the left and right should be alternating columns
    # Still we search, left and right separately (Sanity)
    left_df  = embed_df.loc[embed_df.index.str.contains("_Left")].sort_index()
    right_df = embed_df.loc[embed_df.index.str.contains("_Right")].sort_index()

    # Perform check that the shapes are equal
    if left_df.shape != right_df.shape:
        raise ValueError("Shape of left and right embedding do not match. Check your original all-to-all connectivity DataFrame.")

    # Check that the neuron classes line up
    left_split  = pd.DataFrame(left_df.index.str.rsplit("_", n = 1, expand = True).to_list())
    right_split = pd.DataFrame(right_df.index.str.rsplit("_", n = 1, expand = True).to_list())

    # Compare
    if list(left_split[0]) != list(right_split[0]):
        raise ValueError("Sequence of left and right embedding (neuron class sequences) do not match.")

    return left_df, right_df

#-----------------------------------------------------------------------------------------------------------------------
def measure_paired_distances(left_df, right_df):
    """
    To measure the paired distances between the left and right embeddings (can be used pre- and post- alignment).

    """
    # Gets the raw distances
    dists = np.linalg.norm(left_df.values - right_df.values, axis = 1)
    # Measure metrics
    min_dist, med_dist, max_dist = np.min(dists), np.median(dists), np.max(dists)
    # Print status
    print(f"L ⟷ R distances:  min = {min_dist:.3f}, median = {med_dist:.3f}, max = {max_dist:.3f}")

    return dists

#-----------------------------------------------------------------------------------------------------------------------
def align_hemispheres(left_df, right_df):
    """
    Procrustes alignment between left and right embeddings.
    """
    # Procrustes
    R, scale = orthogonal_procrustes(left_df.values, right_df.values)
    # Align the Left to the Right
    left_aligned = left_df.values @ R
    # Convert to DataFrame
    left_aligned = pd.DataFrame(left_aligned, index = left_df.index)

    # Calculate the distances
    dists = measure_paired_distances(left_aligned, right_df)

    return left_aligned, dists

#-----------------------------------------------------------------------------------------------------------------------
def average_embeddings(left_df, right_df):
    """
    Simply average the embeddings between. Assumes all checks and corrections have been applied previously.
    """
    # Average
    mean_embeddings = 0.5 * (left_df.values + right_df.values)

    # Get the index names
    left_pieces = pd.DataFrame(left_df.index.str.rsplit("_", n = 1, expand = True).to_list())[0].to_list()

    # Convert to DataFrame
    mean_embeddings = pd.DataFrame(mean_embeddings, index = left_pieces)

    return mean_embeddings

#-----------------------------------------------------------------------------------------------------------------------
def scale_matrix_by_count(matrix, count_path : str = f"{folder}/Raw/Class_Counts.csv"):
    """
    Dividing the connection weight by the number of neurons per class. This seems odd, size it can only be applied
    along one axis. The normalization is only by row or column - which might emphasize the role as sender vs receiver.
    This seemed to work when we were working with Ipsi vs Contra style rectangular matrix.

    """
    count_df = pd.read_csv(count_path)
    count_map = dict(zip(count_df["Name"], count_df["Count"]))

    # Map to the name
    count_array = matrix.index.to_series().map(count_map).fillna(1)
    # Divide
    scaled_matrix = matrix.div(count_array, axis = 0)

    return scaled_matrix

#-----------------------------------------------------------------------------------------------------------------------
def embed_and_reduce(matrix : pd.DataFrame,
                     spectral_dim   : int = 512,
                     svd_dim        : int = 32,
                     norm_mode      : str = "sqrt",
                     scale_by_count : bool = False,):
    """
    Scale the data, perform spectral embedding, and then drop the dimensionality.

    """
    # Check the options
    norm_mode = norm_mode.lower()
    if norm_mode not in ["raw", "sqrt", "log"]:
        raise ValueError(f"Invalid normalization mode: {norm_mode}")

    # Sanity check the embedding dimensions
    if spectral_dim >= matrix.shape[0]:
        raise ValueError("Spectral dimension is larger than the number of input neurons!")

    # Sanity check the SVD dimension
    if svd_dim >= spectral_dim:
        raise ValueError("SVD dimension is larger than the spectral dimension!")

    # Scale the weights if needed
    if scale_by_count:
        matrix = scale_matrix_by_count(matrix)

    # Get the values
    if norm_mode == "raw":
        values = matrix.values
    elif norm_mode == "sqrt":
        values = np.sqrt(matrix.values)
    elif norm_mode == "log":
        values = np.log(matrix.values)
    else:
        # Not needed, but just here for completeness
        raise ValueError(f"Invalid normalization mode: {norm_mode}")

    # Create embeddings
    ase = AdjacencySpectralEmbed(n_components = spectral_dim)

    # Embed
    U, V = ase.fit_transform(values)
    # Stack
    embed = np.hstack((U, V))

    # Split the hemispheres
    left_df, right_df = split_hemispheres(matrix, embed)

    # Measure the pairwise distances
    print("Pre-Alignment")
    raw_dists = measure_paired_distances(left_df, right_df)

    # Perform the alignment
    print("\nPost-Alignment")
    left_aligned, aligned_dists = align_hemispheres(left_df, right_df)

    # Average the embeddings
    mean_embeddings = average_embeddings(left_aligned, right_df)

    # Then run the SVD on this embedding
    svd = TruncatedSVD(n_components = svd_dim, random_state = 0)
    reduced = svd.fit_transform(mean_embeddings)

    # Return the reduced with indices as well
    reduced = pd.DataFrame(reduced, index = mean_embeddings.index)

    return mean_embeddings, reduced, ase, svd, raw_dists, aligned_dists

#-----------------------------------------------------------------------------------------------------------------------
def plot_singular_values(ase, svd, variance = True):
    """
    Plotting the singular values of the embedding and reduction.

    """
    # Create a figure
    fig, ax = plt.subplots(figsize = (5, 8), nrows = 2)
    plt.subplots_adjust(hspace = 0.45)

    # Spectral Embedding
    ase_vals = ase.singular_values_
    ase_comp = np.linspace(1, len(ase_vals), len(ase_vals))

    # SVD
    svd_vals = svd.singular_values_
    svd_comp = np.linspace(1, len(svd_vals), len(svd_vals))

    # If we want to instead plot the explained variance cumsum
    if variance:
        ase_vals = np.cumsum(ase_vals ** 2) / np.sum(ase_vals ** 2)
        svd_vals = np.cumsum(svd_vals ** 2) / np.sum(svd_vals ** 2)

    # Plot
    ax[0].plot(ase_comp, ase_vals, marker = "o", markersize = 2, linewidth = 1)
    ax[1].plot(svd_comp, svd_vals, marker = "o", markersize = 2, linewidth = 1)

    # Labels
    ax[0].set_xlabel("ASE Component")
    ax[1].set_xlabel("SVD Component")

    if variance:
        ax[0].set_ylabel("Cumulative Explained Variance")
        ax[1].set_ylabel("Cumulative Explained Variance")
    else:
        ax[0].set_ylabel("Singular Value")
        ax[1].set_ylabel("Singular Value")

    # Limits
    ax[0].set_xlim(-1, len(ase_vals) + 1)
    ax[1].set_xlim(-1, len(svd_vals) + 1)

    # Titles
    ax[0].set_title("Adjacency Spectral Embedding")
    ax[1].set_title("Singular Value Decomposition")

    plt.suptitle("Embedding and Reduction\nSingular Values", fontsize = 17, y = 1)

    return fig, ax

#-----------------------------------------------------------------------------------------------------------------------
def plot_distances(raw_dists, aligned_dists, binwidth = 0.5, binrange = (0, 25)):
    """
    Pre and post-alignment paired left-right distances.

    """
    fig, ax = plt.subplots(figsize = (5, 4))

    # Place the raw
    sns.histplot(raw_dists, binwidth = binwidth, binrange = binrange,
                 ax = ax, color = "red", alpha = 0.5, label = "Pre-Alignment")
    sns.histplot(aligned_dists, binwidth = binwidth, binrange = binrange,
                 ax = ax, color = "blue", alpha = 0.5, label = "Post-Alignment")

    # Legend
    ax.legend()

    # Labels
    ax.set_xlabel("Distances (AU)")
    ax.set_ylabel("Frequency")

    # Title
    ax.set_title("Pairwise Left ⟷ Right Distances")

    return fig, ax

#-----------------------------------------------------------------------------------------------------------------------
def plot_ase_energies(ase):
    """
    ASE Energy plot

    """
    fig, ax = plt.subplots(figsize = (5, 4))

    # Get the energies
    energies = (ase.singular_values_) ** 2
    # Calculate the cumsum
    energies_cumsum = np.cumsum(energies) / np.sum(energies)

    # Plot it
    ax.plot(np.arange(1, len(energies) + 1), energies_cumsum, marker = "o", markersize = 2, linewidth = 1)

    # Labels
    ax.set_xlabel("ASE Component")
    ax.set_ylabel("Cumulative Energy")

    # Title
    ax.set_title(f"ASE Cumulative Energy (n = {len(energies)})")

    return fig, ax

#-----------------------------------------------------------------------------------------------------------------------
def generate_hierarchy(reduced_matrix : pd.DataFrame,
                        method : str = "ward", metric : str = "euclidean"):
    """
    Run the hierarchical clustering.

    """
    # Only cluster along rows
    row_linkage = linkage(reduced_matrix, method = method, metric = metric, optimal_ordering = False)
    row_order = leaves_list(row_linkage)

    # Sort the Matrix
    clustered_matrix = reduced_matrix.iloc[row_order, :]

    return row_linkage, row_order, clustered_matrix

#-----------------------------------------------------------------------------------------------------------------------
node2cluster  = {}
cluster2color = {}

def build_node2cluster(row_linkage, leaf_labels, N):
    """
    Map every node to a cluster.
    """
    node2cluster = {i : leaf_labels[i] for i in range(N)}
    for i in range(row_linkage.shape[0]):
        node = N + i
        c1, c2 = int(row_linkage[i, 0]), int(row_linkage[i, 1])
        lab1, lab2 = node2cluster[c1], node2cluster[c2]
        node2cluster[node] = lab1 if (lab1 == lab2) else None
    return node2cluster

def build_cluster_palette(leaf_labels):
    """
    Based on out custom cmap.
    """
    unique = sorted(np.unique(leaf_labels))
    pal = sns.color_palette(__colors__, n_colors = len(unique))
    color_dict = dict(zip(unique, pal))
    return color_dict

def link_color_func(node_id):
    lab = node2cluster.get(node_id)
    coltup = cluster2color.get(lab, (0, 0, 0))
    color  = mcolors.to_hex(coltup)
    return color

#-----------------------------------------------------------------------------------------------------------------------
def cluster_from_linkage(Z : np.ndarray, N : int = 636, min_size : int = 1, max_size : int = 4):
    """
    Use the linkage map to cut into appropriate connectivity modules.

    """
    # Map parent-child IDs
    children = {}
    for i in range(Z.shape[0]):
        parent_id = N + i
        left_id   = int(Z[i, 0])
        right_id  = int(Z[i, 1])
        children[parent_id] = (left_id, right_id)

    # Compute the leaf indices under each node
    leaf_sets = {}

    def gather_leaves(node_id):
        if node_id < N:
            # It is already a leaf, original point
            leaf_sets[node_id] = np.array([node_id], dtype = int)
            return leaf_sets[node_id]

        # Are both children in leaf_sets
        left_c, right_c = children[node_id]
        if left_c not in leaf_sets:
            gather_leaves(left_c)
        if right_c not in leaf_sets:
            gather_leaves(right_c)
        # Now both are present
        left_leaves  = leaf_sets.get(left_c)
        right_leaves = leaf_sets.get(right_c)

        # Concatenate
        all_leaves = np.concatenate((left_leaves, right_leaves))
        leaf_sets[node_id] = np.sort(all_leaves)

        return leaf_sets[node_id]

    # Gather leaves from every internal ode
    for i in range(Z.shape[0]):
        node = N + i
        if node not in leaf_sets:
            gather_leaves(node)

    # Recursively traverse the tree
    labels = np.full(N, -1, dtype = int)
    next_cluster_id = 0

    def recurse_node(node_id):
        """
        Recusrively assign leaves under node_id to one or more final groups
        """
        nonlocal next_cluster_id
        leaves = leaf_sets[node_id]
        m = leaves.shape[0]

        # If cluster is small enough
        if m <= max_size:
            cid = next_cluster_id
            labels[leaves] = cid
            next_cluster_id += 1
            return

        # Otherwise we split
        left_c, right_c = children[node_id]
        leaves_left  = leaf_sets[left_c]
        leaves_right = leaf_sets[right_c]
        # Size
        left_size  = leaves_left.shape[0]
        right_size = leaves_right.shape[0]

        # If neither is too big to cut
        if left_size <= max_size and right_size < min_size:
            cid = next_cluster_id
            labels[leaves] = cid
            next_cluster_id += 1
            return

        # If ONLY Right is big enough to cut
        if left_size <= max_size < right_size:
            cid = next_cluster_id
            labels[leaves_left] = cid
            next_cluster_id += 1
            # Split the right
            recurse_node(right_c)
            return

        # If ONLY Left is big enough to cut
        if left_size > max_size >= right_size:
            cid = next_cluster_id
            labels[leaves_right] = cid
            next_cluster_id += 1
            # Split the right
            recurse_node(left_c)
            return

        # Both children are large enough to cut
        recurse_node(left_c)
        recurse_node(right_c)
        return

    # The "root" of the tree is node 2N-2
    root = (2 * N) - 2
    recurse_node(root)

    return labels

#-----------------------------------------------------------------------------------------------------------------------
def plot_clustermap(clustered_matrix : pd.DataFrame,
                    row_linkage : np.ndarray,
                    labels : np.ndarray,
                    vmin : float = -1,
                    vmax : float = 1,
                    cmap : str   = "vlag",
                    figsize : tuple = (12, 85)):
    """
    Helper to plot the clustermap for the reduced matrix. The SNS clustermap function is a little weird (I don't like it,
    so we custom cluster the matrix and work with it.

    """
    global node2cluster, cluster2color
    node2cluster  = build_node2cluster(row_linkage, labels, clustered_matrix.shape[0])
    cluster2color = build_cluster_palette(labels)
    # Create Figure
    fig = plt.figure(figsize = figsize)
    gs  = GridSpec(nrows = 1, ncols = 2, width_ratios = [0.6, 0.4], wspace = 0.01)

    # Add the dendrogram axis
    ax_dendro = fig.add_subplot(gs[0, 0])

    # For visualization
    Z_vis = compress_linkage(row_linkage, mode = "sqrt")

    # Place the dendrogram
    with plt.rc_context({"lines.linewidth" : 0.5}):
        dendro = dendrogram(Z_vis,
                            orientation = "left",
                            ax = ax_dendro,
                            no_labels = False,
                            #labels = labels,
                            link_color_func = link_color_func,
                            color_threshold = 0)
    # Remove the grid
    ax_dendro.set_axis_off()
    ax_dendro.grid(False)

    # Place the heatmap
    ax_clust = fig.add_subplot(gs[0, 1])
    sns.heatmap(clustered_matrix,
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
                ax   = ax_clust,
                cbar_kws = {"label" : "Value", "shrink" : 0.03, "aspect" : 10, "pad" : 0.45},
                linewidths  = 1,
                xticklabels = True,
                yticklabels = True,)

    # Labels
    ax_clust.set_xlabel("\nEmbedding Dimension", fontsize = 20)
    ax_clust.set_ylabel("Neuron Class", fontsize = 20)
    ax_clust.invert_yaxis()

    # Fixing items
    ax_clust.yaxis.tick_right()
    ax_clust.yaxis.set_label_position("right")
    # ax_dendro.set_xlim(10, 0)
    ax_dendro.set_xlim(Z_vis[:, 2].max() * 1.05, 0)
    plt.setp(ax_clust.get_xticklabels(), rotation = 0, fontsize = 8)
    plt.setp(ax_clust.get_yticklabels(), rotation = 0, fontsize = 8)
    plt.setp(ax_dendro.get_yticklabels(), visible = False)
    plt.setp(ax_dendro.get_xticklabels(), visible = False)

    # Title
    plt.suptitle("Clustered Neuron Classes", fontsize = 25, y = 0.895)

    return fig, (ax_dendro, ax_clust)
    #return fig, (ax_dendro)

#-----------------------------------------------------------------------------------------------------------------------
def match_groups(reduced_matrix : pd.DataFrame, labels : np.ndarray):
    group_df = pd.DataFrame(zip(reduced_matrix.index.to_list(), labels))
    #group_df.sort_values(by = [0, 1], inplace = True)
    group_df.columns = ["Class", "Group"]
    return group_df

#-----------------------------------------------------------------------------------------------------------------------
def run(conn_matrix, spectral_dim, svd_dim, folder : str = f"{folder}/Embedding_Search", save = False,
        separate_sensory = False):
    """
    Helper to run param search. There is likely a better nested way to do this, since we don't need to repeat the
    same embedding when choosing the SVD dimension. However, it is fast enough that I am too lazy to implement it.
    Plus, it is just easier to implement the final version like this.
    """
    # Embed
    mean_embeddings, reduced, ase, svd, raw_dists, aligned_dists = embed_and_reduce(conn_matrix,
                                                                                    spectral_dim = spectral_dim,
                                                                                    svd_dim = svd_dim)

    # Plot the pre- and post-alignment distances
    fig, ax = plot_distances(raw_dists, aligned_dists)
    fig.savefig(f"{folder}/AlignmentDistancesPairwise_{spectral_dim}_{svd_dim}.pdf",
                bbox_inches="tight")

    # Get the ASE energies
    # This is a very sad way of over-writing it each time, but oh well
    fig, ax = plot_ase_energies(ase)
    fig.savefig(f"{folder}/ASE-Energy_{spectral_dim}.pdf", bbox_inches = "tight")

    # Plot the singular values
    fig, ax = plot_singular_values(ase, svd)
    fig.savefig(f"{folder}/SingularValues_{spectral_dim}_{svd_dim}.pdf", bbox_inches = "tight")

    #----------------------------------
    # Separate the sensory classes if needed
    if separate_sensory:
        classes = reduced.index.tolist()
        sensory_classes = [sens for sens in classes if sens.startswith("S-") or sens.startswith("BolwigsAxon")]
        brain_classes   = [neu for neu in classes if neu not in sensory_classes]

        # Subset
        brain_reduced   = reduced.loc[brain_classes]
        sensory_reduced = reduced.loc[sensory_classes]

        # CENTRAL---------------------
        # Generate Linkage
        row_linkage, row_order, clustered_matrix = generate_hierarchy(brain_reduced)
        # Make the clusters
        labels = cluster_from_linkage(row_linkage, min_size = 2, max_size = 6, N = brain_reduced.shape[0])

        # Plot the clustermap
        fig, ax = plot_clustermap(clustered_matrix, row_linkage, labels)
        fig.savefig(f"{folder}/Clustermap_Central_{spectral_dim}_{svd_dim}.pdf", bbox_inches="tight")

        # Match the labels to the classes
        group_dict = dict(zip(brain_reduced.index.tolist(), labels))
        group_df = pd.DataFrame({"Class": clustered_matrix.index.tolist(), })
        group_df["Group"] = group_df["Class"].map(group_dict)
        # Save it
        group_df.to_parquet(f"{folder}/Groups_Central_{spectral_dim}_{svd_dim}.parquet", index = False,
                            engine = "pyarrow")

        if save:
            clustered_matrix.columns = clustered_matrix.columns.astype(str)
            clustered_matrix.to_parquet(f"{folder}/ClusteredMatrix_Central_{spectral_dim}_{svd_dim}.parquet",
                                        engine = "pyarrow")

        # SENSORY----------------------
        # Generate Linkage
        row_linkage, row_order, clustered_matrix = generate_hierarchy(sensory_reduced)
        # Make the clusters
        labels = cluster_from_linkage(row_linkage, min_size = 2, max_size = 6, N = sensory_reduced.shape[0])

        # Plot the clustermap
        fig, ax = plot_clustermap(clustered_matrix, row_linkage, labels, figsize = (12, 9))
        ax[0].set_xlim(4, 0)
        fig.savefig(f"{folder}/Clustermap_Sensory_{spectral_dim}_{svd_dim}.pdf", bbox_inches = "tight")

        # Match the labels to the classes
        sensory_group_dict = dict(zip(sensory_reduced.index.tolist(), labels))
        sensory_group_df = pd.DataFrame({"Class": clustered_matrix.index.tolist(), })
        sensory_group_df["Group"] = sensory_group_df["Class"].map(sensory_group_dict)

        # Save it
        sensory_group_df.to_parquet(f"{folder}/Groups_Sensory_{spectral_dim}_{svd_dim}.parquet", index = False,
                                    engine = "pyarrow")

        if save:
            #clustered_matrix.columns = clustered_matrix.columns.astype(str)
            clustered_matrix.to_parquet(f"{folder}/ClusteredMatrix_Sensory_{spectral_dim}_{svd_dim}.parquet",
                                        engine = "pyarrow")

    #----------------------------------
    # OVERALL
    # Generate Linkage
    row_linkage, row_order, clustered_matrix = generate_hierarchy(reduced)
    # Make the clusters
    labels = cluster_from_linkage(row_linkage, min_size = 2, max_size = 6, N = reduced.shape[0])

    # Plot the clustermap
    fig, ax = plot_clustermap(clustered_matrix, row_linkage, labels)
    fig.savefig(f"{folder}/Clustermap_{spectral_dim}_{svd_dim}.pdf", bbox_inches = "tight")


    # Match the labels to the classes
    group_dict = dict(zip(reduced.index.tolist(), labels))
    group_df   = pd.DataFrame({"Class" : clustered_matrix.index.tolist(),})
    group_df["Group"] = group_df["Class"].map(group_dict)
    # Save it
    group_df.to_parquet(f"{folder}/Groups_{spectral_dim}_{svd_dim}.parquet", index = False,
                        engine = "pyarrow")

    if save:
        #clustered_matrix.columns = clustered_matrix.columns.astype(str)
        clustered_matrix.to_parquet(f"{folder}/ClusteredMatrix_{spectral_dim}_{svd_dim}.parquet",
                                    engine = "pyarrow")

    # Close
    plt.close("all")

#-----------------------------------------------------------------------------------------------------------------------
def ari_compilation(assignments : dict):
    """
    Use all the assignments to calculate the ARI.

    """
    # Get the keys
    keys = list(assignments.keys())
    n    = len(keys)

    # Create the storage matrices
    ari_mat = np.zeros((n, n))
    nmi_mat = np.zeros((n, n))

    # Loop over every pairwise combination
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            # Get the labels
            labels_i = assignments[ki]
            labels_j = assignments[kj]
            # Place into the correct positions
            ari_mat[i, j] = adjusted_rand_score(labels_i, labels_j)
            nmi_mat[i, j] = normalized_mutual_info_score(labels_i, labels_j)

    # Wrap into DataFrames
    ari_df = pd.DataFrame(ari_mat, index = keys, columns = keys)
    nmi_df = pd.DataFrame(nmi_mat, index = keys, columns = keys)

    return ari_df, nmi_df

#-----------------------------------------------------------------------------------------------------------------------
def metric_vs_dims(ari_df, metric = "ARI"):
    """
    Create a plot for the ARI vs the ASE and SVD dimensions.
    """
    # Create a figure
    fig, ax = plt.subplots(figsize = (5, 8), nrows = 2)
    plt.subplots_adjust(hspace = 0.45)

    # Check the matrix
    if not list(ari_df.index.values) == list(ari_df.columns):
        raise ValueError("Something is weird about the ARI DataFrame.")

    # Matrix
    ari_matrix = ari_df.values

    # Extract the dimension pairse
    dimension_pairs = ari_df.index.values.tolist()

    # Convert the dimension pairs to a DataFrame
    dim_df = pd.DataFrame(dimension_pairs, columns = ["ASE", "SVD"])

    # Get the unique ASE and SVD dims
    ase_dims = np.unique(dim_df["ASE"])
    svd_dims = np.unique(dim_df["SVD"])

    #--------------------------------------------
    # Lists to store
    used_svd_dims = []
    mean_ari_list = []
    sem_ari_list  = []

    # Loop over each value
    for svd_dim in svd_dims:
        indices = dim_df[dim_df["SVD"] == svd_dim].index
        if len(indices) < 2:
            # Skip fewer than 2 configs
            continue

        # Extract the submatrix
        sub_matrix = ari_matrix[np.ix_(indices, indices)]
        # Mask diagonal to exclude self comparisons
        mask = ~np.eye(len(indices), dtype = bool)
        aris = sub_matrix[mask]

        # Metrics
        mean_ari = np.mean(aris)
        sem_ari  = np.std(aris) / np.sqrt(len(aris))

        # Append
        used_svd_dims.append(svd_dim)
        mean_ari_list.append(mean_ari)
        sem_ari_list.append(sem_ari)

    # Plot the values
    ax[1].errorbar(used_svd_dims, mean_ari_list, yerr = sem_ari_list, fmt = "-o", capsize = 2, elinewidth = 2, markersize = 2, linewidth = 1)

    #--------------------------------------------
    # Lists to store
    used_ase_dims = []
    mean_ari_list = []
    sem_ari_list  = []

    # Loop over each value
    for ase_dim in ase_dims:
        indices = dim_df[dim_df["ASE"] == ase_dim].index
        if len(indices) < 2:
            # Skip fewer than 2 configs
            continue

        # Extract the submatrix
        sub_matrix = ari_matrix[np.ix_(indices, indices)]
        # Mask diagonal to exclude self comparisons
        mask = ~np.eye(len(indices), dtype = bool)
        aris = sub_matrix[mask]

        # Metrics
        mean_ari = np.mean(aris)
        sem_ari  = np.std(aris) / np.sqrt(len(aris))

        # Append
        used_ase_dims.append(ase_dim)
        mean_ari_list.append(mean_ari)
        sem_ari_list.append(sem_ari)

    # Plot the values
    ax[0].errorbar(used_ase_dims, mean_ari_list, yerr = sem_ari_list, fmt = "-o", capsize = 2, elinewidth = 2, markersize = 2, linewidth = 1)

    #--------------------------------------------
    # Limits
    ax[0].set_ylim(0.3, 0.7)
    ax[1].set_ylim(0.4, 0.8)

    # Titles
    ax[0].set_title("Adjacency Spectral Embedding")
    ax[1].set_title("Singular Value Decomposition")

    # Labels
    ax[0].set_ylabel(f"Mean {metric}");
    ax[1].set_ylabel(f"Mean {metric}");
    ax[0].set_xlabel("ASE Dimensionality");
    ax[1].set_xlabel("SVD Dimensionality");

    # Labels
    plt.suptitle("ARI Stability Across Dimension Choice", y = 1.05, fontsize = 17)

    return fig, ax

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Read in and organize the connectivity matrix
    conn_matrix, all_classes = read_connectivity()

    #-------------------------------------------------------------------------------------------------------------------
    if re_embed:
        #---------------------------------------------------------------------------------------------------------------
        screen_spectral_dims = np.linspace(24, 512, 26).astype(int)
        screen_svd_dims      = np.linspace(2, 24, 23).astype(int)
        param_grid = {
            "spectral_dim" : screen_spectral_dims,
            "svd_dim"      : screen_svd_dims}

        # Loop over each
        for params in ParameterGrid(param_grid):
            print("")
            # Check and run
            if params["spectral_dim"] > params["svd_dim"]:
                print(params)
                run(conn_matrix, params["spectral_dim"], params["svd_dim"])

        #---------------------------------------------------------------------------------------------------------------
        # Comparison
        pattern     = f"{folder}/Embedding_Search/Groups_*_*.parquet"
        assignments = {}
        files       = glob(pattern)

        # Loop
        for f in files:
            df = pd.read_parquet(f)
            # Split name
            pieces = f.split("/")[-1].replace("Groups_", "").replace(".parquet", "")
            spectral_dim, svd_dim = map(int, pieces.split("_"))
            # Same ordering
            df = df.sort_values("Class").reset_index(drop = True)
            assignments[(spectral_dim, svd_dim)] = np.array(df["Group"])

        # Calculate the Adjusted Rank Indices
        print("\nCalculating ARI and NMI")
        ari_df, nmi_df = ari_compilation(assignments)

        # Save these
        ari_df.to_parquet(f"{folder}/Embedding_Eval/Eval_ARI.parquet",
                          engine = "pyarrow")
        nmi_df.to_parquet(f"{folder}/Embedding_Eval/Eval_NMI.parquet",
                          engine = "pyarrow")

        #---------------------------------------------------------------------------------------------------------------
        # Thinking about clustering stability with respect to changes in the SVD and ASE dimensions
        fig = sns.clustermap(ari_df, cmap = "magma", vmin = 0.5, xticklabels = True, yticklabels = True, figsize = (25, 25))
        plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), fontsize = 6)
        plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), fontsize = 6)
        # Save
        fig.savefig(f"{folder}/Embedding_Eval/GridSearch_ARI.pdf", bbox_inches = "tight")

        # Then Look at the ARI variability along the dimensions
        fig, ax = metric_vs_dims(ari_df, metric = "ARI")
        plt.savefig(f"{folder}/Embedding_Eval/Variability_ARI.pdf", bbox_inches = "tight")

        # And the NMI
        fig, ax = metric_vs_dims(nmi_df, metric = "NMI")
        ax[0].set_ylim(0.7, 0.9)
        ax[1].set_ylim(0.8, 1)
        plt.savefig(f"{folder}/Embedding_Eval/Variability_NMI.pdf", bbox_inches = "tight")
        plt.close("all")

    #-------------------------------------------------------------------------------------------------------------------
    # We have settled on ASE : 128 and SVD : 8
    run(conn_matrix, 128, 8, f"{folder}/Embedding", save = True, separate_sensory = True)
    #-------------------------------------------------------------------------------------------------------------------
    # Status
    print("हो गया दोस्तों!")
