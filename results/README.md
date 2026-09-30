# Results

Derived outputs of the pipeline: every number and figure on the results page comes from a file
in this folder. The per-residue embeddings these were computed from (~380 GB) are not tracked;
`scripts/02_extract.py` regenerates them. See [`DATA.md`](../DATA.md) for sizes and licences.

Folders marked **page** are the source of a number or figure on the results page. Most are read by
`site/build_data.py` at build time; a few reach the page through the archived sections in `site/`
(see `site/README.md`). Where a folder
has a `params.json`, it records the exact command, arguments, git commit and library versions
that produced it; older folders predate that convention and are described here instead.

## Agreement between models (`scripts/04_convergence.py`)

| Folder | What it holds | Status |
|---|---|---|
| `ladder_ci/` | Every ladder pair, raw and with amino-acid identity subtracted; CKA, SVCCA, mutual k-NN; 50 whole-chain subsamples of 8,000 residues, 95% percentile intervals | **page** · headline numbers |
| `ladder_ci_null/` | The same run with a scrambled-residue null on every subsample (`--with-null`); reproduces `ladder_ci/` on 70 of 72 rows exactly and within 0.001 on the other two (see its README) | **page** (calibration) |
| `aa_control/` | Agreement before and after subtracting the per-amino-acid mean, all pairs | **page** |
| `convergence_controls/` | SVCCA against its permutation null and at matched widths | **page** |
| `convergence/` | Layer × layer grids, ESM-2 35M × ProteinMPNN, 30,000 residues; supervised (κ) grids; significance | **page** (heatmaps) |
| `convergence650/` | Same, ESM-2 650M × ProteinMPNN | supporting |
| `convergence_esmif1/` | ESM-2 650M × ESM-IF1 | supporting |
| `convergence_if_mpnn/` | ESM-IF1 × ProteinMPNN (two structure models) | supporting |
| `conv_carp_esm/`, `conv_carp_mpnn/`, `conv_carp_esmif1/` | CARP-38M against ESM-2 35M, ProteinMPNN and ESM-IF1 | supporting |
| `conv_esm1v_mpnn/`, `conv_esm1v_mpnn_30k/` | ESM-1v × ProteinMPNN at 20,000 and 30,000 residues | supporting |
| `conv_seed_pair/`, `conv_seed_pair_30k/` | ESM-1v seed 1 × seed 2, the ceiling, at 20,000 and 30,000 residues | supporting |
| `conv_rand_rand/`, `conv_randesm_trained/`, `conv_trained_randmpnn/` | Untrained floors: both untrained, untrained sequence model, untrained structure model | supporting |

The layer grids are single runs on the full residue budget. Headline agreement scores come from
`ladder_ci/`, which supersedes the single-run values at the same pairs. These measures are biased
upward at small sample sizes, so a grid at 30,000 residues and a ladder estimate at 8,000 are not
directly comparable.

## Structure of each model's own space (`scripts/03_geometry.py`)

| Folder | What it holds | Status |
|---|---|---|
| `umap6/` | 2-D maps of all six models, residue and chain level, with coordinates; `residue_maps_web.png` is the web-scale render used as the README figure | **page** (maps) |
| `umap/` | The earlier three-model version | superseded by `umap6/` |
| `analysis/`, `analysis650/` | Per-layer k-NN purity and local variance ratio by property (depth law) | supporting |
| `geometry/` | Unsupervised cluster structure against CATH labels | supporting |
| `health/` | Effective rank, components for 90% variance, condition number and mean cosine per layer | supporting |
| `clustering/` | Density clusters per model | supporting |

## What each model encodes (`scripts/05_probes.py`)

| Folder | What it holds | Status |
|---|---|---|
| `probes_ci/` | Linear and gradient-boosted probes for 25 properties at every layer, 5 independent refits | **page** |
| `probes_physchem/` | Baseline from amino-acid physicochemical descriptors alone | **page** |
| `probes_comp_rerun/` | Baseline from amino-acid composition alone | supporting |
| `probes/`, `probes650/` | Single-refit probes for ESM-2 35M, ProteinMPNN and ESM-2 650M | superseded by `probes_ci/` |

## Stitching (`scripts/06_stitching.py`)

| Folder | What it holds | Status |
|---|---|---|
| `stitch_grid_ci/proteinmpnn_ridge/` | Every ProteinMPNN layer fed into every ESM-2 layer through a linear connector, 5 repeats on independent chain samples; per-cell intervals on stitched − donor and a verdict | **page** |
| `stitch_grid_ci/carp_ridge/` | The same test from CARP's last layer: a within-sequence positive control | **page** |
| `stitch_grid_ci/proteinmpnn_mlp/` | ProteinMPNN's last layer at four entry depths through a one-hidden-layer network, 3 repeats | **page** |
| `stitch_grid/` | The original single run of the grid | superseded by `stitch_grid_ci/` |
| `depth_stitch/` | Per-property stitching at each injection depth | supporting |
| `functional_grids/` | How well each layer of one model linearly predicts each layer of the other | supporting |
| `predictivity_matrix/` | The same between models' last layers, every pair | supporting |
| `stitching_esmif1/` | Stitching in both directions with ESM-IF1 | supporting |
