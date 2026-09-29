# Replicating the study

Everything needed to go from an empty machine to the numbers on the results page. The short path
(rebuild the page from the tracked results) takes a minute; the long path (recompute the results
from structures) takes days on a CPU machine and a few hundred GB of disk.

- [Prerequisites](#prerequisites)
- [Short path: rebuild the page from `results/`](#short-path-rebuild-the-page-from-results)
- [Long path: recompute everything](#long-path-recompute-everything)
- [Pipeline reference](#pipeline-reference)
- [Reproducibility and the interval conventions](#reproducibility-and-the-interval-conventions)
- [The machine this was developed on](#the-machine-this-was-developed-on)

## Prerequisites

- **OS:** Linux (developed on Ubuntu). macOS should work for the sequence models.
- **Python 3.12**, pinned in `.python-version`.
- **[uv](https://docs.astral.sh/uv/)**, which manages Python, the environment and every package from
  the lockfile: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Hardware:** CPU only; no GPU is needed anywhere. A GPU speeds up embedding extraction, and
  nothing depends on it. The full set of per-residue embeddings is ~380 GB, so the long path needs
  a large data disk.

## Short path: rebuild the page from `results/`

```bash
git clone https://github.com/Natalija-Stepurko/seq-structure-convergence.git
cd seq-structure-convergence
pip install numpy          # the page builder needs nothing else
site/build.sh              # -> site/build/index.html, then audits every headline number
```

`site/build.sh --fresh-figures` also re-renders the three map images from
`results/umap6/coords.npz` (needs `matplotlib`). The test suite for the metric library runs with
`pip install numpy scikit-learn pytest && pytest`.

## Long path: recompute everything

### 1. Main environment

```bash
uv sync     # reads pyproject.toml + uv.lock, creates .venv (Python 3.12)
```

Run stages through it with `uv run python scripts/...`. On a machine with a small root disk, point
the environment and the model caches at a larger disk before `uv sync`:

```bash
export UV_PROJECT_ENVIRONMENT=/big-disk/.venv-ssc   # this project's own environment
export UV_CACHE_DIR=/big-disk/.uv-cache
export HF_HOME=/big-disk/.hf-cache                  # Hugging Face weights
export TORCH_HOME=/big-disk/.torch-hub              # torch.hub weights (ESM-2, ESM-1v, ESM-IF1)
```

Set `UV_PROJECT_ENVIRONMENT` explicitly if anything on the machine already sets it globally, or
`uv sync` installs into that other environment.

### 2. Model weights

All models are public; none needs a token or a licence acceptance.

- **ESM-2** (`esm2_t12_35M_UR50D`, `esm2_t33_650M_UR50D`), **ESM-1v** (two seeded checkpoints) and
  **ESM-IF1** (`esm_if1_gvp4_t16_142M_UR50`) download on first use through `fair-esm` and
  `torch.hub`, into `TORCH_HOME`. The 650M weights are ~2.5 GB.
- **ProteinMPNN**: the model code is vendored in `scripts/vendor/proteinmpnn/` (MIT, Dauparas et al.
  2022). Fetch the 7 MB vanilla weights once:
  ```bash
  mkdir -p "$TORCH_HOME/proteinmpnn"
  curl -L -o "$TORCH_HOME/proteinmpnn/v_48_020.pt" \
    https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt
  ```
- **CARP-38M** downloads through the `sequence-models` package.

### 3. Model environments

ESM-IF1 and CARP have dependencies that conflict with the main environment, so each has its own.
These are the package sets the results were produced with.

**ESM-IF1** needs the compiled PyTorch Geometric stack, which has wheels only for specific torch
versions:

```bash
uv venv --python 3.12 /big-disk/.venv-if
PY=/big-disk/.venv-if/bin/python
uv pip install --python $PY --index-url https://download.pytorch.org/whl/cpu torch==2.4.0
uv pip install --python $PY torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
uv pip install --python $PY torch_geometric==2.8.0.post1 fair-esm==2.0.0 biotite==0.39.0 \
    "numpy<2" scipy
```

**CARP**:

```bash
uv venv --python 3.12 /big-disk/.venv-carp
PY=/big-disk/.venv-carp/bin/python
uv pip install --python $PY --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
uv pip install --python $PY sequence-models==1.8.0 numpy scipy
```

Run the matching extraction with that environment's interpreter, e.g.
`/big-disk/.venv-if/bin/python scripts/02_extract.py esmif1 ...`. The scripts find the shared
metric library in `src/` on their own, so nothing needs installing into these environments.

### 4. Run the pipeline

Every stage takes explicit `--*-dir` flags and never writes bulk data into the repository. The
defaults point at `/ssc/...`, the paths on the development machine; pass your own.

```bash
S=/data/ssc/structures   R=/data/ssc/results
uv run python scripts/01_dataset.py fetch      --structures-dir $S --limit 5000
uv run python scripts/01_dataset.py annotate   --structures-dir $S
uv run python scripts/01_dataset.py targets    --structures-dir $S
uv run python scripts/02_extract.py esm        --structures-dir $S --results-dir $R/esm
uv run python scripts/02_extract.py struct     --structures-dir $S --results-dir $R/proteinmpnn
uv run python scripts/03_geometry.py depth-law --results-dir $R
uv run python scripts/04_convergence.py ladder-ci --results-root $R --structures-dir $S \
    --n-resamples 50 --resample-size 8000 --max-residues 20000 --out-dir $R/ladder_ci
uv run python scripts/05_probes.py repeats     --n-repeats 5 --seed0 42 --out-dir $R/probes_ci/esm \
    --passthrough --results-dir $R/esm --model-name esm --structures-dir $S
uv run python scripts/06_stitching.py grid     --n-chains 400 --per-chain 25 --properties ss3 burial rsa \
    --structures-dir $S --esm-dir $R/esm --struct-dir $R/proteinmpnn --out-dir $R/stitch_grid
```

Every subcommand lists its flags with `--help`. Where an output folder in `results/` has a
`params.json`, it records the exact command that produced it; `results/README.md` describes the
rest. For a quick end-to-end check, run stage 01 with `--limit 6` into a scratch directory first.

## Pipeline reference

Six entry points in `scripts/`, each grouping the stages that answer one kind of question,
selected by subcommand. Every stage is resume-safe (outputs are guarded by existence checks, so a
re-run fills gaps only) and writes `params.json` beside its outputs: the resolved arguments, the
exact command, the git commit, library versions and a UTC timestamp.

| Entry point | Subcommand | What it does |
|---|---|---|
| **`01_dataset.py`** | `fetch` | CATH S35 chains and PDB structures → per-chain `npz` (sequence, backbone coordinates, 3-state secondary structure, relative solvent accessibility) + `index.jsonl` |
| | `annotate` | UniProt chain labels via the SIFTS mapping → `annotations.jsonl` |
| | `targets` | per-residue binding, active and PTM sites; per-chain-normalised B-factor |
| **`02_extract.py`** | `esm` | ESM-2 (or ESM-1v) per-residue embeddings at every layer |
| | `struct` | ProteinMPNN encoder embeddings at every layer (never sees the sequence) |
| | `esmif1` | ESM-IF1, the second structure model |
| | `carp` | CARP-38M, the second sequence model (convolutional) |
| | `random` | untrained copies of both main models, the floor |
| **`03_geometry.py`** | `depth-law` | per-layer k-NN purity and local variance ratio |
| | `overlap` | unsupervised geometry against the property labels |
| | `health` | effective rank, conditioning, collapse |
| | `clusters` | density clusters against every categorical label |
| | `umap` | 2-D maps of every model, coloured by each property |
| **`04_convergence.py`** | `grids` | layer × layer CKA, SVCCA and mutual k-NN with a residue-permutation null |
| | `supervised` | Cohen's κ between the two models' probe predictions |
| | `significance` | resampled intervals and an empirical p-value for the peak |
| | `svcca-controls` | SVCCA's permutation null and width matching |
| | `functional` | per-property agreement grids |
| | `aa-control` | every pair re-measured with amino-acid identity subtracted |
| | `ladder-ci` | the full ladder, raw and with amino-acid identity subtracted, with whole-chain intervals; `--with-null` adds a scrambled-residue null on every subsample, and `--peaks-from` reuses a previous run's peak layers so its estimates reproduce exactly |
| **`05_probes.py`** | `probe` | linear and XGBoost probes per layer × property, chain-grouped splits |
| | `composition` | composition and physicochemical baselines the probes must beat |
| | `repeats` | `probe` refitted on independent splits, aggregated to intervals |
| **`06_stitching.py`** | `stitch` | stitching through each model's own frozen head |
| | `predictivity` | linear predictivity for one pair, both directions |
| | `matrix` | linear predictivity across every pair |
| | `depth` | per-property stitching at each injection depth |
| | `grid` | every donor layer × every injection depth, all four conditions |
| | `grid-ci` | the same grid repeated on independent chain samples, with paired intervals on stitched − donor; `--connector mlp` for a non-linear connector, `--donor-dir` for any donor model |

`scripts/export_benchmark.py` writes the residue-aligned embedding benchmark published on Hugging Face: every model arm at every layer on one fixed subsample of whole chains, with labels and reference scores.

Shared code lives in `src/ssc/metrics.py`: CKA, SVCCA (with the cached per-matrix split used for
layer grids), mutual k-NN, k-NN purity, local variance ratio and the provenance recorder. It is
covered by `tests/test_metrics.py`.

## Reproducibility and the interval conventions

- **Pinned dependencies.** `uv.lock` fixes exact versions; `uv sync` reproduces the environment.
- **Seeds** for splits, subsamples and probes are fixed and exposed as flags.
- **Redundancy control.** The chain set is CATH S35 (at most 35% sequence identity between
  chains) and probe splits are grouped by chain, so accuracy reflects generalisation across
  proteins, not memorised homology.
- **Provenance.** Two grids computed at different residue budgets are not comparable, and nothing
  in the arrays or the figures would reveal the difference; `params.json` records it.

Three conventions matter when comparing these numbers with another study:

- **Intervals come from whole-chain subsampling without replacement.** Residues within a chain are
  correlated, so resampling residues independently understates the spread. Chains are drawn without
  replacement because a duplicated chain makes a residue its own nearest neighbour in both models,
  which mutual k-NN scores as automatic agreement.
- **CKA, SVCCA and mutual k-NN are all biased upward at smaller sample sizes, and unequally.**
  Going from 20,190 residues to 8,000 raises CKA ~2%, SVCCA ~5% and mutual k-NN ~26%. Every
  subsample is drawn to the same fixed budget and the estimate is reported at that budget. A score is
  not comparable across studies that used different sample sizes.
- **SVCCA is capped at 5,000 rows** inside `svcca_reduce`, so its effective sample size is smaller
  than that of the other two measures on the same call.

## The machine this was developed on

An Azure CPU VM with 8 cores and 165 GB RAM. Two NVMe disks: an 8 TB persistent disk holding the
repository, structures and embeddings, and a 440 GB ephemeral disk holding the environments and
model caches (a reboot keeps it; deallocating the VM wipes it, and `uv sync` rebuilds it). The
environment variables above were set in `~/.bashrc`, with `TORCHDYNAMO_DISABLE=1` added to avoid
TorchDynamo compile errors on this CPU setup.
