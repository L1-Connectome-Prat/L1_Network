# L1 Connectome Network Analysis

Connectivity-based grouping, signal-flow characterization, and multisensory
integration in the first-instar *Drosophila* larval (L1) connectome. This
repository accompanies a larger study on L1 neural organization.

**[▶ Interactive Connectome Viewer](https://l1-connectome-prat.github.io/L1_Network/)** — hover any group to highlight its connections; click to pin; zoom/pan freely.

## Overview

The pipeline takes the complete synaptic connectivity of ~3,000
differentiated L1 neurons, groups them into ~200 connectivity-based modules
via spectral embedding, and characterizes how sensory signals propagate
through these modules using Personalized PageRank. Core outputs:

- **Connectivity Groups** — neuron classes clustered by shared connectivity
  profiles (ASE + hierarchical clustering)
- **Connectivity Variability** — per-grouping stereotypy quantified against
  global / label-shuffle nulls and within-vs-across-group resampling
- **Signal Flow Metrics** — per-group flux balance, partner-breadth entropy,
  and functional role assignments
- **Sensory Integration Scores** — topological distance, integration index,
  and modality enrichment for each group
- **Layered Network Architecture** — groups organized into 11 processing
  layers (L0 sensory sources through L10), with feedforward / feedback /
  lateral decomposition

## Pipeline

Scripts are intended to run sequentially; each builds on prior outputs.

| # | Script | Purpose |
|---|--------|---------|
| 1 | `connectivity_fetch.py` | Fetch all-to-all synaptic connectivity from CATMAID; aggregate at bodyId / Cluster / Lineage / Acronym / ConnectivityGroup levels; compute output-normalized weights. |
| 2 | `embed.py` | All-to-all matrix → ASE → Procrustes-aligned hemisphere average → TruncatedSVD → Ward hierarchical clustering with custom tree-cutting (group sizes 2–6). Final params: ASE=128, SVD=8. |
| 3 | `embed_anatomy_atlas.py` | Supplementary atlas: 2D SWC projections of every group against the CNS neuropil. |
| 4 | `embed_group_positions.py` | Per-group anatomical centroids (dendrite / axon / full neuron) via `navis.split_axon_dendrite`; symmetrized to template brain. |
| 5 | `connectivity_variability.py` | Within-group cosine similarity at Cluster (fine) and Acronym (coarse) resolution, with global-shuffle and label-shuffle nulls (n=10,000) plus within-vs-across pair resampling. |
| 6 | `embed_signal_flow.py` | Per-group flux balance (`f_in`, `f_out`, `Flow_Ratio`), partner entropies, effective partner counts; KMeans (k=7) → seven functional roles. |
| 7 | `embed_signal_integration.py` | Personalized PageRank from each sensory modality (Olf / Gust / Vis / Thermo) at Acronym and Group level. Produces drives, scaled distances, integration index, modality enrichment, final score. α = 0.85. |
| 8 | `network_layer_selection.py` | Bins groups into k discrete layers via Jenks natural breaks on Final_Score; selects k=10 (+ L0 sources) by silhouette / max-layer-size / within-layer CV. Also computes feedforward / feedback / lateral weight per layer. |
| 9 | `network_layer_connectivity.py` | Anatomical layered network diagram: nodes coloured by compartment, sized by neuron count, edges weighted by normalized output. Whole-network overview + per-layer decomposition. |
| 10 | `compartment_signal_distribution.py` | Per-compartment heatmap of layer membership and per-compartment distributions of layer / Integration_Drive / Final_Score. |
| 11 | `viz/export_viewer_data.py` | Bundle layer assignments + signal-flow metrics + group positions + connectivity into `docs/viewer_data.json`, fetched at load time by `docs/index.html` (served via GitHub Pages). |

Shared utilities:
- `_compartment_palette.py` — compartment colour scheme (`COMP_HEX`,
  `COMP_RGB`, `COMP_ORDER`) used by the layer / compartment figures.

## Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| PPR Drive | `PPR_score · Σ source_out` | Physical signal volume reaching each node |
| Scaled Distance | `−log10(PPR_score + ε)` | Topological depth (synaptic hops) from source |
| Primary Modality | `argmax(Drive_k)` | Dominant sensory input |
| Integration Index | `−Σ pₖ log pₖ / log K` | Normalized Shannon entropy (0 = unimodal, 1 = fully mixed) |
| Final Score | `Σ pₖ · Dₖ` | Proportion-weighted topological depth |
| Modality Enrichment | `(pₖ − p_expected) / p_expected` | Fold-change vs. non-sensory baseline |
| Flow Ratio | `(w_out − w_in) / (w_out + w_in)` | Net sender (+) vs. receiver (−) |

The PPR layering primarily reflects direct synaptic distance; the chosen
α=0.85 sits at the transition between source-dominated and
topology-dominated regimes (Spearman ρ > 0.96 across α ∈ [0.65, 0.80],
degrades above 0.90).

## Directory Structure

```
L1_Network/
  Scripts/                            # Analysis scripts (+ viz/ subfolder)
  Notebooks/                          # Variability + viewer notebooks
  Data_Connectivity/                  # Raw and aggregated connectivity matrices
  Analysis_Outputs/
    Embedding/                        # Final embedding + group assignments
    Embedding_Search/                 # Parameter search
    Embedding_Eval/                   # ARI / NMI stability
    Group_Anatomy/                    # Spatial positions + atlas figures
    Group_Signal-Flow/                # PPR results, integration, functional roles
    Group_Layer-Network/              # Layer assignments + diagrams
    Variability/                      # Shuffle nulls + resampling outputs
  docs/                               # GitHub Pages root (interactive viewer)
```

## Reproducing the Variability Outputs

`Analysis_Outputs/Variability/**/*.npz` are gitignored — they're large
shuffle-iteration arrays that anyone can regenerate from the script:

```bash
python Scripts/connectivity_variability.py --level both --redo --n 10000
```

## Dependencies

- **Connectome tools** — `pymaid`, `navis`, `flybrains`, `connectome_analysis`
- **Embedding** — `graspologic` (ASE)
- **ML / Stats** — `scikit-learn`, `scipy`, `jenkspy`
- **Data** — `pandas`, `numpy`, `networkx`, `pyarrow`
- **Visualization** — `matplotlib`, `seaborn`
- **Custom** — `prats_helpers` (panel label placement)
