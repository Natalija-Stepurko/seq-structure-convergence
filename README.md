# seq-structure-convergence

**Do a sequence-trained and a structure-trained protein foundation model converge to the
same internal representation of protein space — despite never sharing a training signal?**

This project takes two protein foundation models built on *fundamentally different* learning
signals and asks whether their internal representations nonetheless organise protein space the
same way — a cross-modality test of the [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)
across the sequence/structure divide.

- **Sequence arm — [ESM-2](https://www.science.org/doi/10.1126/science.ade2574).** A transformer
  trained *only* on masked amino-acid prediction. It never sees 3D coordinates, yet is known to
  encode structure in its representations (Rives et al. 2021; Rao et al. 2021; Lin et al. 2023).
- **Structure arm — [ESM-IF1](https://www.science.org/doi/10.1126/science.add2187) / [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187).**
  A GVP / message-passing network trained *on 3D structure* (to recover sequence). It never
  optimises a sequence-only objective, and implicitly learns structural energetics
  (cf. AlphaFold: Roney & Ovchinnikov 2022).

The two are trained on opposite signals and use different architectures. If their internal maps
of protein space converge — measured with CKA/SVCCA/mutual-kNN, layer by layer — that is
evidence for a shared, modality-independent representation of proteins.

> This repository is the biology counterpart to a materials-science study of foundation
> interatomic potentials (ORB-v3 / UMA). It reuses that project's analysis protocol
> (PCA/UMAP, k-NN purity, LVR, CKA/SVCCA, linear + XGBoost probes) but is an **entirely
> separate project**. See [`paper/PROJECT_PLAN.md`](paper/PROJECT_PLAN.md) for the full design,
> including the literature positioning and novelty analysis.

---

## The question, precisely

1. **Convergence.** Do ESM-2 (sequence) and ESM-IF1/ProteinMPNN (structure) place the same
   proteins/residues near each other in latent space? Where in each network is convergence
   strongest (early, middle, late)?
2. **What is shared vs. modality-specific.** Which properties (secondary structure, burial,
   contacts, fold class, function) are organised identically by both, and which only one
   captures?
3. **Depth organisation.** Does each model show the local→global depth split (residue-local
   biophysics early and linear; whole-protein properties late and non-linear)?
4. **Geometry vs. decodability.** Where does unsupervised geometry (k-NN/LVR) agree or disagree
   with supervised probe accuracy?

---

## Repository layout

```
seq-structure-convergence/
  README.md                     # this file
  CLAUDE.md                     # uv conventions + this instance's paths
  pyproject.toml                # dependencies (managed with uv)
  uv.lock                       # locked dependency versions (reproducible install)
  .python-version               # 3.12

  scripts/                      # the pipeline (01 → 05) — see "Pipeline" below
  paper/
    PROJECT_PLAN.md             # full experimental design (models, data, labels, stages, novelty)
  notebooks/                    # exploratory notebooks
  results/                      # git-ignored; heavy outputs live off-repo (see "Where data is saved")
```

---

## Prerequisites

- **OS:** Linux (developed on Ubuntu). macOS should work for the sequence arm.
- **Python:** 3.12 (pinned in `.python-version`).
- **[uv](https://docs.astral.sh/uv/):** the only hard requirement for the environment — it manages
  Python, the virtual environment, and all packages from the lockfile.
- **git**, and (for pushing) an SSH key with access to the repository.
- **Hardware:** the pipeline is **CPU-only by design** — no GPU is required (this project was
  developed on an 8-core / 165 GB-RAM CPU instance). A GPU will speed up embedding extraction but
  nothing depends on it. Bulk embedding data can reach hundreds of GB at full scale, so plan for a
  large data disk (see "Where data is saved").

---

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# binary lands at ~/.local/bin/uv — open a new shell or `source ~/.bashrc`
```

### 2. Clone

```bash
git clone git@github.com:Natalija-Stepurko/seq-structure-convergence.git
cd seq-structure-convergence
```

### 3. Create the environment

```bash
uv sync          # reads pyproject.toml + uv.lock, creates .venv (Python 3.12)
```

Run everything through the environment with `uv run …` (e.g. `uv run python scripts/01_fetch_proteins.py`).

> **Small root disk?** If your home/root partition is small, point uv's environment and caches at a
> larger disk *before* `uv sync` (this is what our instance does — see
> [This instance's environment](#this-instances-environment)):
> ```bash
> export UV_PROJECT_ENVIRONMENT=/big-disk/.venv-ssc   # venv off the root disk
> export UV_CACHE_DIR=/big-disk/.uv-cache             # uv download cache
> export HF_HOME=/big-disk/.hf-cache                  # HuggingFace weights (transformers)
> export TORCH_HOME=/big-disk/.torch-hub              # torch.hub weights (fair-esm: ESM-2/ESM-IF1)
> ```

### 4. Model weights

Both models are **public — no token or license acceptance required** (unlike the materials
project's HF-gated UMA weights):

- **ESM-2** (`esm2_t33_650M_UR50D`, and `esm2_t12_35M_UR50D` for fast iteration) and **ESM-IF1**
  (`esm_if1_gvp4_t16_142M_UR50`) download automatically on first use via `fair-esm`, which uses
  **`torch.hub`** — so set **`TORCH_HOME`** to a large disk (not the default `~/.cache/torch` on a
  small root partition; the 650M weights are ~2.5 GB):
  ```bash
  export TORCH_HOME=/scratch/.torch-hub
  ```
- **ProteinMPNN** (the current structure arm) — model code is vendored under
  `scripts/vendor/proteinmpnn/` (MIT, Dauparas 2022); fetch the ~7 MB vanilla weights once:
  ```bash
  mkdir -p /scratch/.torch-hub/proteinmpnn
  curl -L -o /scratch/.torch-hub/proteinmpnn/v_48_020.pt \
    https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt
  ```

### 5. Structure-arm dependencies

**ProteinMPNN needs nothing beyond the base install** (torch + numpy) — its code is vendored, so
the base `uv sync` is sufficient for the structure arm.

**ESM-IF1** (the intended *primary* structure arm) is **deferred**: it needs the `torch-geometric` +
`torch-scatter/sparse/cluster` compiled stack, which has no wheels for the current bleeding-edge
torch and won't build cleanly. Enabling it requires pinning torch to a PyG-supported version
(~2.4–2.6) — a separate change tracked in `paper/PROJECT_PLAN.md`. ProteinMPNN is a fully valid
structure arm in the meantime (and remains the robustness cross-check afterwards).

---

## Where data is saved

Scripts never write bulk data into the repo. Every stage takes explicit `--*-dir` flags, and all
heavy outputs are **git-ignored** (`results/`, `structures/`, `data/`). The defaults point at a
large, **persistent** disk (on this instance, under `/ssc` → the 8 TB NVMe):

| What | Default path | Persistence |
|---|---|---|
| Downloaded structures + `index.jsonl` manifest (stage 01) | `/ssc/structures/` | persistent data disk |
| Per-layer embeddings `.pt` (stage 02) | `/ssc/results/{esm,esmif1,proteinmpnn}/` | persistent data disk |
| Analysis outputs / figures (stages 03–05) | `/ssc/results/<model>/plots/` | persistent data disk |
| Model weights | `HF_HOME` (e.g. `/scratch/.hf-cache`) | rebuildable — re-downloaded |
| Virtual environment | `UV_PROJECT_ENVIRONMENT` (e.g. `/scratch/.venv-ssc`) | rebuildable — `uv sync` |

Rule of thumb: **data you want to keep** → persistent data disk; **rebuildable things** (venv,
weights) → scratch/ephemeral disk. Redirect anything with the corresponding flag, e.g.
`--structures-dir /my/path`.

---

## Pipeline

> **Status:** the environment and design are in place; the numbered scripts are being implemented.
> **Stages 01, 02a, 02b, 03 and 04 are implemented**; 05/06 are being built. This section documents the interface and is
> kept in sync as each stage lands. See [`paper/PROJECT_PLAN.md`](paper/PROJECT_PLAN.md) for the
> rationale behind each stage.

Numbered, resumable scripts under `scripts/`, mirroring the materials pipeline. Every stage is
idempotent/resume-safe — outputs are guarded by existence checks, so re-running fills gaps only.

| Stage | Script | Role |
|---|---|---|
| 01 ✅ | `01_fetch_proteins.py` | Non-redundant protein chains from **CATH** (one per S35 cluster) + structure from the RCSB PDB; computes sequence, backbone N/CA/C/O coords, per-residue 3-state SSE + relative SASA, and per-chain CATH C/A/T/H — all via **biotite** (no external DSSP). Writes `proteins/<id>.npz` + `index.jsonl` |
| 02a ✅ | `02_extract_embeddings_esm.py` | ESM-2 all-layer per-residue embeddings (embedding + each block) → `<id>.pt` (fp16); optional predicted contact map. Weights via `torch.hub` (`TORCH_HOME`) |
| 02b ✅ | `02_extract_embeddings_struct.py` | **ProteinMPNN** encoder per-layer node embeddings (sequence-agnostic, structure-only) → `<id>.pt` (fp16). ESM-IF1 deferred (needs a torch pin for the PyG stack) |
| 03 ✅ | `03_analyze_embeddings.py` | Per-model, per-layer **k-NN purity** (local SSE/burial vs global CATH fold) + **LVR** of RSA → the depth law → `metrics.csv`, `depth_law.png` |
| 04 ✅ | `04_convergence.py` | **The core.** Residue-aligned **CKA / SVCCA / mutual-kNN between ESM-2 and ProteinMPNN, layer × layer**, with a permutation baseline → `grids.npz`, `convergence.png`, `summary.txt` |
| 05 ✅ | `05_property_prediction.py` | Linear + XGBoost probes per layer (chain-grouped splits): SSE / burial / RSA (residue) + CATH class (pooled); XGB−linear gap; CATH **data-efficiency learning curve** vs an AA-composition baseline → `metrics.csv`, `probe_curves.png`, `learning_curve.png` |
| 06 ✅ | `06_significance.py` | Robustness of the convergence peak: repeated residue-resamples → peak-CKA vs permutation-baseline means, 95% CIs, modal peak layer-pair, empirical p-value → `summary.txt`, `significance.png` |

Typical run (from the repo root, environment active):

```bash
uv run python scripts/01_fetch_proteins.py        --structures-dir /ssc/structures --limit 5000
uv run python scripts/02_extract_embeddings_esm.py --structures-dir /ssc/structures --results-dir /ssc/results/esm
uv run python scripts/02_extract_embeddings_struct.py --structures-dir /ssc/structures --results-dir /ssc/results/proteinmpnn
uv run python scripts/03_analyze_embeddings.py    --results-dir /ssc/results
uv run python scripts/04_convergence.py           --results-dir /ssc/results
uv run python scripts/05_property_prediction.py   --results-dir /ssc/results/esm --model-name esm --structures-dir /ssc/structures
uv run python scripts/06_significance.py           --esm-dir /ssc/results/esm --struct-dir /ssc/results/proteinmpnn --structures-dir /ssc/structures
```

### Testing on a small subset

Start with a small subset and small models (`esm2_t12_35M`, ProteinMPNN) to validate the full
pipeline before scaling — mirroring the materials project's test-subset workflow. Stage 01 takes a
`--limit`, e.g. a 6-chain smoke test:

```bash
uv run python scripts/01_fetch_proteins.py --structures-dir structures_test --limit 6
```

---

## Reproducibility

- **Pinned dependencies.** `uv.lock` fixes exact versions; `uv sync` reproduces the environment
  byte-for-byte. Python is pinned to 3.12.
- **Deterministic seeds.** Probes, PCA/UMAP subsampling, and train/test splits are seeded; seeds
  are exposed as CLI flags.
- **Redundancy control.** Use identity-clustered / held-out-superfamily splits (e.g. CATH-S40) so
  probe accuracy reflects generalisation, not memorised homology.
- **CPU numerics.** Runs are CPU-only by default; results are independent of GPU availability.

---

## This instance's environment

The above is enough to replicate anywhere. For reference, the specific machine this was developed
on (an Azure CPU VM) is configured as:

- **Two NVMe disks:** `/data` (8 TB, persistent) and `/scratch` (440 GB, ephemeral — a reboot
  keeps it, a *deallocate* wipes it; rebuild with `uv sync`).
- **Working path `/ssc`:** a top-level [bind mount](https://man7.org/linux/man-pages/man8/mount.8.html)
  of `/data/.ssc` (backing store on the 8 TB disk), persisted in `/etc/fstab`
  (`/data/.ssc /ssc none bind,nofail 0 0`). This keeps the project at a clean top-level path,
  physically on the big disk, and separate from a co-located materials project at `/data/project`.
- **Environment variables** (in `~/.bashrc`, set before the first `uv` run):
  ```bash
  export UV_PROJECT_ENVIRONMENT=/scratch/.venv-ssc   # dedicated venv (do NOT share the materials /scratch/.venv)
  export UV_CACHE_DIR=/scratch/.uv-cache
  export HF_HOME=/scratch/.hf-cache                   # HuggingFace (transformers) weights
  export TORCH_HOME=/scratch/.torch-hub               # torch.hub (fair-esm) weights — ESM-2 / ESM-IF1
  export TORCHDYNAMO_DISABLE=1                        # avoids TorchDynamo compile errors on this CPU setup
  ```
  See [`CLAUDE.md`](CLAUDE.md) for the venv-isolation rationale.

---

## License

[MIT](LICENSE). All model/code dependencies used here (ESM-2, ESM-IF1, ProteinMPNN,
🤗 Transformers) are permissively licensed.
