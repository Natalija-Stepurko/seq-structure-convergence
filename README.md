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
  A GVP/message-passing network trained *on 3D structure* (to recover sequence). It never
  optimises a sequence-only objective, and implicitly learns structural energetics
  (cf. AlphaFold: Roney & Ovchinnikov 2022).

The two are trained on opposite signals and use different architectures. If their internal maps
of protein space converge — measured with CKA/SVCCA/mutual-kNN, layer by layer — that is
evidence for a shared, modality-independent representation of proteins.

> This repository is the biology counterpart to a materials-science study of foundation
> interatomic potentials (ORB-v3 / UMA). It reuses that project's analysis protocol
> (PCA/UMAP, k-NN purity, LVR, CKA/SVCCA, linear + XGBoost probes) but is an **entirely
> separate project**. See [`paper/PROJECT_PLAN.md`](paper/PROJECT_PLAN.md) for the full design
> and [`paper/literature.md`](paper/literature.md) for the literature positioning.

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

## Pipeline (planned)

Numbered, resumable scripts under `scripts/`, mirroring the materials pipeline. One model arm
each where the extraction differs.

| Stage | Script | Role |
|---|---|---|
| 01 | `01_fetch_proteins.py` | Pull a non-redundant protein set with known structure (CATH / PDB cull / AFDB) + per-residue and per-chain labels → `index.jsonl` manifest |
| 02a | `02_extract_embeddings_esm.py` | ESM-2 all-layer per-residue + attention extraction → `.pt` |
| 02b | `02_extract_embeddings_struct.py` | ESM-IF1 / ProteinMPNN all-layer per-residue extraction → `.pt` |
| 03 | `03_analyze_embeddings.py` | Per-layer PCA/UMAP + k-NN purity + LVR (per-residue and pooled) |
| 04 | `04_convergence.py` | **CKA / SVCCA / mutual-kNN between the two models, layer × layer**; embedding-health geometry; HDBSCAN |
| 05 | `05_property_prediction.py` | Linear + XGBoost probes per layer × pooling × property; learning curves |

See [`paper/PROJECT_PLAN.md`](paper/PROJECT_PLAN.md) for datasets, label sources, and the
model-pair rationale.

---

## Setup

This instance is **CPU-only**. `uv` manages the environment, into a **dedicated venv** so it
does not collide with the materials project (see [`CLAUDE.md`](CLAUDE.md)).

```bash
cd /ssc
export UV_PROJECT_ENVIRONMENT=/scratch/.venv-ssc   # dedicated venv — important
uv sync                                            # creates /scratch/.venv-ssc (Python 3.12)
uv run python scripts/01_fetch_proteins.py --help
```

Model weights (ESM-2, ESM-IF1) download on first run into the shared `HF_HOME=/scratch/.hf-cache`.

---

## License

See [`LICENSE`](LICENSE). All model/code dependencies used here (ESM-2, ESM-IF1, ProteinMPNN,
🤗 Transformers) are permissively licensed, so this repository is free to adopt a permissive
license of its own.
