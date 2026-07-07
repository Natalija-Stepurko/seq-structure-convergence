# Project plan — sequence↔structure representation convergence in protein foundation models

*Focused design for this repository. Literature positioning lives in [`literature.md`](literature.md).*

## 1. The claim under test

Two protein foundation models are trained on **opposite signals** and use **different
architectures**:

- **ESM-2** (sequence): a transformer trained only on masked amino-acid prediction — never sees 3D
  coordinates.
- **ESM-IF1 / ProteinMPNN** (structure): a GVP / message-passing GNN trained on 3D structure to
  recover sequence — never optimises a sequence-only objective.

**Hypothesis:** despite this, their internal representations converge to a shared, modality-
independent map of protein space (a cross-modality instance of the Platonic Representation
Hypothesis, Huh et al. 2024). The novelty is that convergence is tested **across the
sequence/structure divide**, layer-resolved, not within one modality.

**Falsifiable predictions**
- P1 — CKA/SVCCA/mutual-kNN between the two models is well above architecture-matched random
  baselines, and rises then plateaus (or peaks mid-network) with depth.
- P2 — Convergence is strongest for properties both modalities must encode (secondary structure,
  contacts, burial) and weakest for modality-specific ones (raw evolutionary/coevolutionary
  signal in ESM; fine backbone geometry in the structure model).
- P3 — Both models independently show the local→global depth split (residue-local early/linear;
  whole-protein late/non-linear).

## 2. Model pair (CPU-feasible)

| Arm | Model | Signal | Architecture | Layer axis |
|---|---|---|---|---|
| Sequence | ESM-2 (`esm2_t33_650M_UR50D`; `t12_35M` for the test subset) | Masked LM | Transformer | embed → 33 (or 12) blocks |
| Structure (primary) | ESM-IF1 (`esm_if1_gvp4_t16_142M_UR50`) | 3D structure → sequence | GVP-Transformer | encoder GVP-GNN + transformer blocks |
| Structure (secondary) | ProteinMPNN | 3D structure → sequence | Message-passing GNN | 3 enc + 3 dec layers |

**Why this pair on CPU.** AlphaFold's Evoformer needs MSAs and a GPU — impractical here. ESM-IF1
and ProteinMPNN are lightweight, structure-trained, and architecturally distinct from ESM-2,
giving a *sharper* cross-modality contrast than AlphaFold at a fraction of the cost. ProteinMPNN
is the fallback / robustness check for the structure arm.

> ESM-IF1's inverse-folding module needs extra geometry deps (`torch-geometric`, `torch-scatter`,
> `torch-sparse`, `torch-cluster`, `biotite`). These are added in stage 02b so the base install
> stays clean. ProteinMPNN is dependency-light (torch + numpy).

## 3. Dataset

- **Protein set:** a non-redundant set with known 3D structure so every emergent target has ground
  truth — **CATH S40** (≈30–35% identity clusters) and/or a PDB cull, scaled with AlphaFold DB.
  Target ~50–100k chains; start with a **5k test subset** (mirroring the materials
  `create_test_subset.py` pattern).
- **Redundancy control:** identity-clustered / held-out-superfamily splits so probe accuracy is
  not memorised homology.

## 4. Labels (locality-split, mirroring materials extensive-vs-global axis)

| Locality | Property | Source |
|---|---|---|
| Residue-local (expect early/linear) | Secondary structure (Q3/Q8), RSA/burial, backbone torsion, disorder | DSSP / NetSurfP / DisProt |
| Pairwise | Residue–residue contacts | PDB (Rao 2021 target) |
| Whole-protein (expect late/non-linear) | CATH fold class, subcellular localisation, EC/GO function, SCOP superfamily | CATH / DeepLoc / SwissProt / SCOP |
| Biophysical (regression) | Stability ΔΔG, Tm | FireProtDB / Tsuboyama 2023 / Meltome |

## 5. Pipeline stages

| Stage | Script | Output |
|---|---|---|
| 01 | `01_fetch_proteins.py` | CATH/PDB/AFDB chains + labels → `index.jsonl`, validated |
| 02a | `02_extract_embeddings_esm.py` | ESM-2 all-layer per-residue + attention `.pt` |
| 02b | `02_extract_embeddings_struct.py` | ESM-IF1 / ProteinMPNN all-layer per-residue `.pt` |
| 03 | `03_analyze_embeddings.py` | Per-layer PCA/UMAP + k-NN purity + LVR (per-residue + pooled) |
| 04 | `04_convergence.py` | **CKA / SVCCA / mutual-kNN, layer × layer, across the two models**; anisotropy/rank/collapse; HDBSCAN |
| 05 | `05_property_prediction.py` | Linear + XGBoost probes per layer × pooling × property; parity/confusion; learning curves vs from-scratch |

Stage 04 is the scientific heart and the one genuinely new piece vs. the materials pipeline: a
**cross-model** CKA/SVCCA/mutual-kNN grid requires aligning the two models' residue sets on the
same proteins, then comparing every ESM layer against every structure-model layer.

## 6. Analyses (the paper spine)

1. **Cross-modality convergence map** — layer×layer CKA/SVCCA/mutual-kNN heatmap; where the two
   models agree most, and whether it's mid-network (P1).
2. **Shared vs. modality-specific content** — which labels both models organise identically vs.
   only one (P2).
3. **Local→global depth split** in each model independently (P3).
4. **Geometry vs. decodability** — divergence between k-NN/LVR and probe accuracy.
5. **Pooling bottleneck** — per-residue vs mean-pooled for residue-level vs whole-protein targets.
6. **Embedding health** — anisotropy/rank/collapse as intrinsic reliability signals.

## 7. Two-week skeleton

1. `01`+`02a` for ESM-2 (35M) on a 5k CATH-S40 subset; per-residue DSSP labels; confirm all-layer
   extraction.
2. `02b` ProteinMPNN (lightest structure arm) on the same subset.
3. `03` on both; validate the depth split on **secondary structure (local) vs fold class (global)**.
4. `04` first cross-model CKA/SVCCA number.
5. Add ESM-IF1; scale toward the full set; `05` probes + learning curves; write up against the
   anchors in `literature.md`.

## 8. Risks & caveats

- **Not a new phenomenon** — the emergence itself is long-established; the contribution is the
  cross-modality, layer-resolved convergence protocol on a matched pair. Frame accordingly.
- **Structure arm needs input structures** — ESM-IF1/ProteinMPNN consume 3D coordinates, so the
  dataset must ship coordinates, not just sequences.
- **Residue-set alignment** between the two models is a real engineering step for stage 04.
- **CPU throughput** — ESM-2 650M is slow on CPU; use 35M/150M for iteration, 650M for final runs.
- **Label coverage skew** — ΔΔG/Tm/localisation labels are sparse and biased; report per-property
  coverage as the materials pipeline does.
- **Anisotropy** — PLM embeddings are anisotropic; the stage-04 geometry diagnostics are
  load-bearing, not decorative.
