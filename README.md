# From Neuron Classes to Circuits

This repository accompanies Section 5 of the first-instar (L1) *Drosophila* connectome.

**[▶ Interactive Network Viewer](https://l1-connectome-prat.github.io/L1_Network/)**

## Aim

Neuron classes are the functional vocabulary of the larval brain. The class-by-class weight matrix that connects them is too fine for a circuit: the edges are numerous, and the matrix supplies no intermediate scale at which a computation can be named.

Classes that share a connectivity signature are gathered into connectivity groups. A shared signature is common upstream inputs, common downstream targets, recurrent partners that themselves share a signature, or some combination of these. Each group is a candidate computational module. Shared inputs co-localize dendrites and shared outputs co-localize axons, so each group has a geometric profile and a direction of signal flow.

Sensory signals are followed from four entries:

- Olfactory (Olf)
- Gustatory (Gust)
- Visual (Vis)
- Thermosensory (Thermo)

Groups ordered by topological distance from those entries fall into layers, and the layers are mapped back onto neuropil compartments.

The assignment used from this point on is `Analysis_Outputs/Embedding/Groups_Separated_128_8_Corrected.csv`.

## Signal-Flow Measures

| Column | Definition |
| --- | --- |
| `Role` | `Sensory_Broadcast`, `Focused_Relay`, `Broad_Relay`, `Focused_Convergence`, `Broad_Convergence`, `Recurrent_Processor`, `Signal_Sink` |
| `*_Mean_Drive` | PageRank score × total output of that modality's sources |
| `*_Scaled_Distance` | −log10(PageRank score) |
| `Primary_Modality` | Largest drive: `Olf`, `Gust`, `Vis`, `Thermo` |
| `Integration_Index` | Entropy of the four drives / log(4) (0 unimodal, 1 equal mix) |
| `Integration_Drive` | `Integration_Index` × log10(total drive + 1) |
| `Final_Score` | Drive-weighted normalized depth (layer cut) |
| `*_Enrichment` | Fold change vs the non-sensory mixture |
| `Flow_Ratio` | (output − input) / (output + input), self-weight excluded |
| `Neff_inputs`, `Neff_outputs` | exp(entropy of partner weights) |

## Results

Lineage-derived classes partition into 233 connectivity groups, 2.7 classes per group. 

The architecture favors integration. Convergence groups number 131 and relay groups number 79.

| Role | Groups |
| --- | --- |
| ***Broad convergence*** | 74 |
| ***Focused convergence*** | 57 |
| Broad relay | 66 |
| Focused relay | 13 |
| Sensory broadcast | 15 |
| Recurrent processor | 4 |
| Signal sink | 4 |

Sensory broadcasters, recurrent processors, and signal sinks are rare, and are confined to the sensory and premotor periphery.

Propagation of the four modalities by Personalized PageRank (α = 0.85) shows that sensory identity erodes within a few steps. The integration index rises from near zero at the sensory entries and approaches a plateau near 0.74 in the deeper network. Gustatory input, the largest peripheral stream, is the primary drive of most groups.

| Modality | Groups |
| --- | --- |
| ***Gustatory*** | 176 |
| Olfactory | 49 |
| Visual | 6 |
| Thermosensory | 2 |

Partition of this depth axis at its natural breaks yields eleven layers, L0–L10. Superficial layers are feed-forward and predominantly ipsilateral. Intermediate layers, principally L3–L6, mediate interhemispheric exchange. Deeper layers remain coupled to the layers that supply them; a substantial fraction of the apparent feedback is the return arm of reciprocal loops that span several layers. Most neuropil compartments occupy a narrow band of these layers and therefore participate at a characteristic stage of processing. The tritocerebrum is an interesting exception, combining early gustatory preprocessing with late, premotor signals.

## Guide to the Repository

| Item | Path |
| --- | --- |
| Figure 5 | `Notebooks/Figure_05_Embedding-Example.ipynb`, `Figure_Book/Figure_05_Embedding-Example.pdf` |
| Lateral accessory lobe | `Notebooks/Figure_05_LAL-Network.ipynb`, `Figure_Book/Figure_05_LAL-Network.pdf` |
| Connectome viewer | `docs/index.html`, https://l1-connectome-prat.github.io/L1_Network/ |
| Group assignment | `Analysis_Outputs/Embedding/Groups_Separated_128_8_Corrected.csv` |
| Projection lengths | `Analysis_Outputs/Group_Anatomy/Group_Dendrite-to-Axon_Distance.csv` |
| Functional roles | `Analysis_Outputs/Group_Signal-Flow/Group_Functional-Roles.parquet` |
| Sensory integration | `Analysis_Outputs/Group_Signal-Flow/Signal-Integration_Group.csv` |
| Layer assignment | `Analysis_Outputs/Group_Layer-Network/Signal-Integration_Group_Layers-Assignment_k-10.parquet` |
| Group connectivity | `Data_Connectivity/Connectivity_Group.csv` |

Group identifiers in the paper (`g001`) match `Group` in the signal-flow, anatomical, and connectivity tables. `Groups_Separated_128_8_Corrected.csv` stores the same identifiers as integers, and the figure notebooks zero-pad them on load. `Groups_Central_128_8.parquet` and `Groups_Sensory_128_8.parquet` number their groups independently. `Groups_128_8.parquet` is the joint clustering.

The layer diagrams and the viewer read the committed placement in `Analysis_Outputs/Group_Layer-Network/Group_Hybrid-Placement.parquet`.

## Analysis Pipeline

Class labels come from `L1_NeuronInformation_PK.csv` in `L1_Skeletons`. `PROJECTS_HOME` points at the directory that contains `L1_Network`, `L1_Skeletons`, `L1_Compartment-Atlas`, `L1_Lineages`, and `L1_Neuron-Typing`. The fetch scripts open a CATMAID session through pymaid. Notebooks and the viewer then read those tables, plus `Synapse-by-Neuron_level1.parquet` in `L1_Neuron-Typing` and the committed `Group_Hybrid-Placement.parquet`.

**Connectivity.** `Scripts/connectivity_fetch.py` fetches the differentiated synaptic graph from the L1 CATMAID volume through pymaid, aggregates it by body, cluster, lineage, class, and connectivity group, and divides each edge by the presynaptic partner's output. Tables and heatmaps go to `Data_Connectivity/`. Later scripts read `Connectivity_Group.csv`. Run the script from the repository root.

```
python Scripts/connectivity_fetch.py
```

**Fragmented synapses.** `Scripts/connectivity_fetch_fragmented.py` labels each synaptic link with its axon or dendrite and the compartment of the connector. An existing axon-dendrite tag is kept; otherwise the split follows `L1_Neuron-Typing`. It writes the Level 1 and Level 2 tables and `Connectivity_Fragmented_Links.parquet` under `Data_Connectivity/`. The LAL notebook reads the Level 1 table. Run the script from the repository root.

```
python Scripts/connectivity_fetch_fragmented.py
```

**Groups.** `Scripts/embed.py` reads `Data_Connectivity/Connectivity.csv`, embeds it by adjacency spectral embedding (128 dimensions), aligns the hemispheres by orthogonal Procrustes, averages them, and reduces the result to 8 dimensions. Ward clustering then cuts groups of 2–6 classes, with sensory classes held apart from central classes. It writes `Analysis_Outputs/Embedding/`, including `ClusteredMatrix_Central_128_8.parquet`. The dimension grid and its adjusted Rand index and normalized mutual information are written when `re_embed` is set. Later analyses read `Groups_Separated_128_8_Corrected.csv`. Run the script from the repository root.

```
python Scripts/embed.py
```

**Positions.** `Scripts/embed_group_positions.py` separates axon from dendrite on the differentiated skeletons in `L1_Skeletons` and records dendritic, axonal, and whole-neuron centroids for each group and hemisphere. Projection length is that dendritic-to-axonal distance, in micrometers, and `Q` is the right-hemisphere quartile. Groups that innervate the ventral nerve cord are written out as well. Products go to `Analysis_Outputs/Group_Anatomy/`. Run the script from the repository root.

```
python Scripts/embed_group_positions.py
```

**Atlas.** `Scripts/embed_anatomy_atlas.py` draws each group against the neuropil, with lineage colors from `L1_Lineages`. The atlas is `Figure_05_Supplementary_GroupAnatomy_01.pdf` through `_03.pdf` in `Analysis_Outputs/Group_Anatomy/`. Run the script from the repository root.

```
python Scripts/embed_anatomy_atlas.py
```

**Roles.** `Scripts/embed_signal_flow.py` records the fraction of weight in, out, and returning, and the entropy of the input and output partners. K-means (k = 7) assigns the roles. Figures are drawn from `Group_Functional-Roles_2026-04-03.parquet` in `Analysis_Outputs/Group_Signal-Flow/`. Run the script from the repository root.

```
python Scripts/embed_signal_flow.py
```

**Synapse counts.** `Notebooks/Figure_05_Signal-Abundance.ipynb` is run from `Notebooks/` and writes `Analysis_Outputs/Group_Signal-Flow/Figure_05_Supplementary_Synapse-Counts-Flow.pdf`.

**Sensory propagation.** `Scripts/embed_signal_integration.py` computes Personalized PageRank from each modality's entry groups, at class and group resolution, with α = 0.85. Rank order is stable through the range used here and degrades above 0.90. It writes `Signal-Integration_Group.csv`, `Signal-Integration_Acronym.csv`, and `Figure_05_Supplementary_PPR-Alpha-Sensitivity.pdf` in `Analysis_Outputs/Group_Signal-Flow/`. Run the script from the repository root.

```
python Scripts/embed_signal_integration.py
```

**Layers.** `Scripts/network_layer_selection.py` assigns the groups in `Source_Groups.npy` to layer 0 and partitions the rest on `Final_Score` by Jenks natural breaks into ten layers. It writes the assignment, `Layer_FF-FB-Lateral.parquet`, and `Layer_Connectivity-Matrix.parquet` under `Analysis_Outputs/Group_Layer-Network/`. Run the script from the repository root.

```
python Scripts/network_layer_selection.py
```

**Layer diagram.** `Scripts/network_layer_connectivity.py` draws this graph on the neuropil, nodes colored by compartment, positions from `Group_Hybrid-Placement.parquet`. It writes `Figure_05_Compartment-Symmetric-Network.pdf` and `Figure_05_Supplementary_Layering-Compartment.pdf`. Run the script from the repository root.

```
python Scripts/network_layer_connectivity.py
```

**Compartments.** `Scripts/compartment_signal_distribution.py` reports layer, integration, and depth for each compartment from the same placement, and writes `Figure_05_Supplementary_Compartment-Layer-Heatmap.pdf`. Run the script from the repository root.

```
python Scripts/compartment_signal_distribution.py
```

**Viewer.** `Scripts/viz/export_viewer_data.py` assembles layer assignments, roles, positions, and connectivity into `docs/viewer_data.js`, which `docs/index.html` loads. Run the script from the repository root after a table has been revised.

```
python Scripts/viz/export_viewer_data.py
```

**Figure 5.** `Notebooks/Figure_05_Embedding-Example.ipynb` is run from `Notebooks/`. It reads the corrected group assignment, the tables above, and `Synapse-by-Neuron_level1.parquet` in `L1_Neuron-Typing`, and writes `Figure_Book/Figure_05_Embedding-Example.pdf`. `Notebooks/Figure_05_LAL-Network.ipynb` is run from `Notebooks/` as well.

## Related Analysis

Section 4 measured the stereotypy of these groupings. That analysis is kept here so that its supporting quantities remain with the groups they concern.

`Scripts/connectivity_variability.py` compares neurons within a group with neurons that share only a lineage, separately for inputs and for outputs, and at two resolutions: cluster, the finer grouping, and acronym, the class grouping. Each comparison is set against two null distributions of 10,000 permutations, a permutation of all synaptic weights and a permutation of group labels within each lineage and hemisphere, and against a resampling of across-group pairs equal in number to the within-group pairs. `Notebooks/Figure_04_Variability.ipynb` is run from `Notebooks/` and reads the cached arrays in `Analysis_Outputs/Variability/`.

The caches may be regenerated in about 40 minutes on 32 cores. Run the script from the repository root. The procedure is seeded, with master seed 39335. `Num_Within_Pairs` and `Num_Between_Pairs` count ordered pairs. The similarity matrices are symmetric, so the reported means match the means over unique pairs.

```
python Scripts/connectivity_variability.py --level both --redo --n 10000
```

## Requirements

Tables and figures use pandas, NumPy, matplotlib, seaborn, and SciPy. Parquet files are written with PyArrow. Adjacency spectral embedding uses graspologic. Clustering, the singular-value reduction, and the embedding stability scores use scikit-learn. Natural breaks use jenkspy. Graphs use networkx. Skeleton geometry and brain transforms use navis and flybrains. The variability permutations use joblib. Panel labels use `place_panel_label` from `prats_helpers`. CATMAID access and STL import use pymaid and `connectome_analysis`.

`PROJECTS_HOME` is required. The checkout is expected at `$PROJECTS_HOME/L1_Network`. External inputs are:

- `$PROJECTS_HOME/L1_Skeletons` (the neuron table and the SWC collections)
- `$PROJECTS_HOME/L1_Compartment-Atlas` (compartment meshes and the ventral nerve cord mesh)
- `$PROJECTS_HOME/L1_Lineages` (the lineage color map read by the anatomical atlas)
- `$PROJECTS_HOME/L1_Neuron-Typing` (the compartment-resolved synapse table read by the Figure 5 notebook)

The code is released under the GNU General Public License, version 3.

## Repository Layout

```
L1_Network/
  Notebooks/                 # Figure notebooks. Begin with Figure_05_Embedding-Example.
  Figure_Book/               # Exports of the principal Section 5 figures
  Scripts/                   # Analysis scripts, in the order above; viewer export in viz/
  Data_Connectivity/         # Connectivity from body to group, and the fragmented synapse tables
  Analysis_Outputs/          # Group assignment, anatomy, signal flow, layers, and the Section 4 caches
    Embedding/               # Group assignments and the 8-dimensional embedding
    Embedding_Search/        # Grid over adjacency-spectral and singular-value dimension
    Embedding_Eval/          # Stability of the partition across that grid
    Group_Anatomy/           # Centroids, projection lengths, and atlas figures
    Group_Signal-Flow/       # Roles, PageRank scores, integration, and modality enrichment
    Group_Layer-Network/     # Layer assignments, interlayer weight, and compartment summaries
    Variability/             # Section 4 permutation caches and empirical distribution functions
  docs/                      # Interactive viewer, served by GitHub Pages
  LICENSE                    # GNU GPL v3
```
