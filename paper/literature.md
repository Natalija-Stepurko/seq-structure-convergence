# Do foundation models spontaneously learn what they were never trained on? — the biology analogue, and a project plan

*Companion to [`literature.md`](literature.md). Compiled 2026-07-07 via `~/search_papers.py` (arXiv + Semantic Scholar) and web search. As with `literature.md`, citation counts and some metadata are as reported on the retrieval date and should be re-verified before any submission; arXiv IDs / DOIs are given where confirmed.*

---

## 0. The question, and how it maps from materials to biology

The scientific question driving the ORB/UMA project is:

> **Do foundation interatomic potentials — trained only on energies, forces and stresses — spontaneously develop internal representations of properties they never saw in their training signal (band gap, crystal system, magnetism, …), and if so, *where in the network* does that information live, and is that organisation *shared across independently built architectures*?**

The bio analogue keeps the shape of the question and swaps the substrate. The single cleanest structural-biology port is:

> **Do protein foundation models — trained only on masked amino-acid prediction (sequence statistics) — spontaneously develop internal representations of 3D structure, function, and biophysics they never saw in their training signal, and where in the network does each property live, and is that organisation shared across architectures?**

The parallel is close to exact, and there is even a mirror-image version. In materials the training signal is *energetic/physical* and the emergent, un-trained targets are *structural/chemical*. In protein **language** models the training signal is *sequence* and the emergent targets are *structural/physical* — the same phenomenon read in the opposite direction. And in the protein **structure** models (AlphaFold) the training signal is *structure* and the emergent target is an *energy function* — which is almost literally the ORB/UMA setup transplanted into biology.

| ORB / UMA (materials) | Protein-model analogue (biology) |
|---|---|
| Training signal: energy, forces, stress | Seq models (ESM): masked amino-acid token; Struct models (AlphaFold): 3D coordinates |
| Emergent, un-trained targets: band gap, crystal system, magnetism, density, bulk modulus | Seq→ secondary/tertiary structure, contacts, burial/SASA, disorder, binding sites, localisation, EC/GO function, stability (ΔΔG), fitness. Struct→ a physical folding energy |
| Local/extensive (density) vs global/electronic (band gap) | Residue-local (secondary structure, burial) vs global/whole-protein (fold class, localisation, function) |
| Layer axis: encoder → 5 GNN blocks → backbone | Layer axis: embedding → N transformer blocks → head |
| Cross-architecture convergence (Li & Walsh "Platonic" MLIPs) | Cross-architecture convergence of PLMs / the Platonic Representation Hypothesis |
| Geometry (k-NN/LVR/UMAP) vs supervised decodability | Same split, same tools |
| Pooling bottleneck (per-atom vs mean-pooled) | Per-residue vs mean-pooled (residue-level vs whole-protein targets) |
| Embedding health (CKA/SVCCA, anisotropy, collapse) | Identical diagnostics; PLM anisotropy/collapse is a known issue |

---

## 1. Verdict: how well is this question answered in the bio / structural-biology space?

**Very well — arguably better than in materials.** The "does a model trained on one signal spontaneously encode another" question is not just answered in structural biology; it is one of the *founding results* of the modern protein-ML field and has a large, mature, still-growing interpretability literature attached to it. The same three-part decomposition used in `literature.md` applies, and the conclusion is stronger:

| Sub-claim | Status in biology | Anchor evidence |
|---|---|---|
| 1. Models encode properties beyond their training target | **Established as a field-defining result** | Rives 2021 (PNAS); Rao 2021 (ICLR); Lin 2023 (Science); Vig 2021 (ICLR); Roney & Ovchinnikov 2022 (PRL) |
| 2. Less labelled data needed vs. from scratch (transfer / few-shot) | **Established; standard practice** | TAPE (Rao 2019); FLIP (Dallago 2021); PEER (Xu 2022); Metalic (2024) |
| 3. "Where in the network / layer-resolved / cross-architecture convergence" | **Actively studied, not yet saturated** | Vig 2021; "Layer by Layer" (2502.02013); PEPE (2025); Platonic Hypothesis (Huh 2024); Gujral 2025 (PNAS) |

So, exactly as the materials `literature.md` concluded, any "nobody has looked at this" framing is indefensible in biology. **The defensible contribution in the bio port is narrow and methodological**, not the phenomenon itself — a *unified, layer-resolved, cross-architecture, geometry-vs-decodability* protocol applied to a *specific pair* of protein foundation models. That is the same residual-novelty posture the materials paper already adopts.

---

## 2. The literature, by theme (with the materials parallel drawn explicitly)

### 2.1 The founding "emergence" results — sequence → structure & function

This is the direct biological equivalent of "ORB learns band gap." It is not controversial; it is the reason the field exists.

- **Rives, Meier, Sercu, Goyal, … Fergus (2021).** *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* **PNAS** 118(15):e2016239118. DOI:10.1073/pnas.2016239118.
  → The canonical statement of the exact phenomenon: a transformer (ESM-1b) trained **only** on masked amino-acid prediction develops internal representations from which secondary structure, tertiary contacts, remote homology, and mutational effects are linearly recoverable — none of which is in the training loss. This *is* the ORB/UMA claim, in proteins, published five years before this project. Title alone is the thesis.

- **Rao, Meier, Sercu, Ovchinnikov, Rives (2021).** *Transformer protein language models are unsupervised structure learners.* **ICLR 2021.** bioRxiv 2020.12.15.422761.
  → 3D residue–residue contacts fall out of the **attention maps** with a sparse logistic regression; specific attention heads specialise for specific contact types. The cleanest "the geometry is already in there, unsupervised" result — the analogue of ORB's k-NN/LVR structure appearing without supervision.

- **Lin, Akin, Rao, … Rives (2023).** *Evolutionary-scale prediction of atomic-level protein structure with a language model.* **Science** 379(6637):1123–1130. DOI:10.1126/science.ade2574.
  → ESM-2 / ESMFold: scaling the masked-LM objective to 15 B parameters makes an **atomic-resolution** picture of structure emerge in the representations, enabling structure prediction from a single sequence. Emergence as a function of scale — a scaling-law version of the ORB claim.

- **Roney & Ovchinnikov (2022).** *State-of-the-Art Estimation of Protein Model Accuracy Using AlphaFold.* **Phys. Rev. Lett.** 129, 238101. DOI:10.1103/PhysRevLett.129.238101.
  → **The mirror image of the ORB/UMA setup.** AlphaFold, trained on *structure*, has implicitly learned a physical **energy function** it was never given: it ranks candidate structures consistently with folding physics and can locate low-energy conformations without co-evolutionary input. Materials: trained on energy → learns structure. Biology: trained on structure → learns energy. Same phenomenon, run backwards. (See also *An Evaluation of Biomolecular Energetics Learned by AlphaFold*, bioRxiv 2025.06.30.662466.)

### 2.2 Layer-resolved probing — *where* in the network the information lives

This is the direct equivalent of the project's central "local→global depth split."

- **Vig, Madani, Varshney, Xiong, Socher, Rajani (2021).** *BERTology Meets Biology: Interpreting Attention in Protein Language Models.* **ICLR 2021.** arXiv:2006.15222.
  → Attention captures folding contacts and binding sites, and — critically for us — **targets progressively more complex biophysical properties with increasing layer depth**, consistently across BERT/ALBERT/XLNet. This is the protein statement of "extensive/local early, global/electronic late," already published.

- **"Layer by Layer: Uncovering Hidden Representations in Language Models"** (2025). arXiv:2502.02013.
  → General-LM evidence (already cited in the materials draft) that **intermediate layers beat the final layer**, mid-network optimum. The generic version of "using the deepest layer can *hurt*."

- **PEPE: scalable extraction of multi-modal protein language model representations** (2025). bioRxiv 2025.10.13.680902.
  → Documents that PLM embedding quality depends strongly on **which layer, which pooling, which padding** — the "over-reliance on the final layer" oversight. Directly motivates our per-layer × pooling sweep.

- **Layer Probing Improves Kinase Functional Prediction with Protein Language Models** (2025). arXiv:2512.00376.
  → A worked example that *the best layer is task-dependent* in PLMs — the protein version of "best extraction layer depends on the target's locality."

### 2.3 Cross-architecture convergence — the "Platonic" analogue

Direct equivalent of Li & Walsh's Platonic-MLIP result that anchors the project's convergence claim.

- **Huh, Cheung, Wang, Isola (2024).** *The Platonic Representation Hypothesis.* **ICML 2024.** arXiv:2405.07987.
  → The general claim that independently trained networks converge to a shared representation, measurable by **CKA / mutual-kNN** — the very metrics the project already uses (04_quality_control). Provides the theoretical frame for an ESM-vs-structure-model convergence test, and the counter-literature (e.g. *Back into Plato's Cave*, arXiv:2604.18572) supplies the honest caveats.

- **Reverse Distillation: Consistently Scaling Protein Language Model Representations** (2026). arXiv:2603.07710.
  → PLMs within a family scale *poorly* and share nested subspaces — a convergence/redundancy result specific to proteins, relevant to whether "bigger = more emergent" holds.

### 2.4 Mechanistic interpretability — what the features actually are

Goes a step beyond the project (which is probe/geometry-based) and shows where the bio field is heading — useful for positioning and for an optional extension.

- **Gujral, Bafna, Alm, Berger (2025).** *Sparse autoencoders uncover biologically interpretable features in protein language model representations.* **PNAS.** DOI:10.1073/pnas.2506316122.
  → SAEs pull **monosemantic, GO-alignable** features out of ESM representations — mechanistic evidence that the emergent content is structured, not incidental. The natural "phase 2" beyond linear/XGBoost probes.
- **InterProt / ESM sparse-autoencoder** efforts and **Transcoder-based Circuit Analysis for Interpretable Single-Cell Foundation Models** (2025, arXiv:2509.14723) show the same interpretability turn in the adjacent single-cell space.

### 2.5 Data efficiency & few-shot — transfer from frozen embeddings (sub-claim 2)

The "less data than from scratch" leg, already standard practice in proteins.

- **TAPE — Rao et al. (2019).** *Evaluating Protein Transfer Learning.* NeurIPS 2019. — the founding benchmark that frozen/transferred PLM features beat from-scratch models on structure/function tasks.
- **FLIP — Dallago et al. (2021)** (fitness landscapes) and **PEER — Xu et al. (2022, NeurIPS)** (multi-task protein benchmark) — standardised suites where frozen-embedding probes are the baseline.
- **ProteinGym — Notin et al. (2023, NeurIPS)** — large mutational-effect benchmark; zero-/few-shot PLM scoring.
- **Metalic: Meta-Learning In-Context with Protein Language Models** (2024). arXiv:2410.08355. — few-shot fitness prediction directly on PLM representations.

### 2.6 Adjacent substrates (same question, other biomolecules)

- **Genomics LMs.** *Nucleotide Transformer* (Dalla-Torre et al., **Nature Methods** 2024, DOI:10.1038/s41592-024-02523-z) and *Enformer* (Avsec et al., **Nature Methods** 2021) learn cis-regulatory / chromatin structure **without functional labels** — the DNA version of the claim. Probing studies exist (e.g. *Evaluating the representational power of pre-trained DNA language models for regulatory genomics*, PMC10925287).
- **Single-cell foundation models.** *scGPT* (Cui et al., **Nature Methods** 2024) and *Geneformer* (Theodoris et al., **Nature** 2023) trained by masked-expression modelling; *Multi-Dimensional Spectral Geometry of Biological Knowledge in Single-Cell Transformer Representations* (2026, arXiv:2602.22247) is a near-verbatim geometry-of-emergent-knowledge study in that space.

---

## 3. What is genuinely novel for a bio port (the defensible residual)

As in materials, claim novelty of the **analysis**, not the phenomenon. No single bio paper above does *all* of the following together on one matched pair of models — which is exactly the gap the materials project fills for MLIPs:

1. A **layer-resolved** dissection of a *specific* protein foundation model — how each property is assembled across embedding → blocks → head — rather than "final-layer embedding as a black box."
2. A **local→global depth-specialisation** law for proteins: residue-local biophysics (secondary structure, burial/SASA, disorder) peaking early and stored *linearly*; whole-protein/global properties (fold class, subcellular localisation, function/EC) peaking late and *non-linearly* (the XGB−linear gap signal). Testable prediction: best layer depends on target locality, mirroring the materials result.
3. An explicit **geometry (k-NN / LVR / UMAP) vs. supervised decodability** separation — showing where unsupervised organisation and probe accuracy diverge.
4. A **pooling-bottleneck** characterisation: per-residue vs mean-pooled representations for residue-level vs whole-protein targets (the direct analogue of per-atom vs pooled nodes/edges).
5. **Cross-architecture convergence** measured with CKA/SVCCA/mutual-kNN between a **sequence** model and a **structure** model — a Platonic-style test across the sequence/structure divide, which is *stronger* than the within-modality MLIP convergence the materials paper reports.
6. **Embedding-health diagnostics** (anisotropy, effective rank, collapse, conditioning) proposed as **intrinsic reliability signals** for PLMs — reusing the project's 04-stage machinery essentially unchanged.

---

## 4. Project plan — a direct port of the ORB/UMA pipeline to proteins

The design principle: **reuse the existing 01→05 architecture and most of `qc_common.py` verbatim**, changing only (a) the data source, (b) the two models, (c) the property/label set, and (d) the "atom/edge" → "residue/contact" vocabulary. The scientific protocol (03: PCA/UMAP/k-NN/LVR; 04: CKA/SVCCA/HDBSCAN/geometry; 05: linear + XGBoost probes) transfers with almost no change.

### 4.1 The matched model pair (the "ORB vs UMA" of biology)

Pick two **architecturally distinct** foundation models so the convergence question is non-trivial:

- **Model A — sequence transformer:** **ESM-2** (e.g. `esm2_t33_650M_UR50D`; scale up to 3B if compute allows). Non-structural training signal (masked LM). This is the "trained on sequence, does it learn structure?" arm. Layer axis = embedding → 33 transformer blocks.
- **Model B — a structurally-informed / different-inductive-bias model.** Best options, in order of contrast:
  - **SaProt** or **ESM-3** (structure-token-aware) — tests "structure-aware vs pure-sequence."
  - or **ProtT5** (Elnaggar et al., encoder-decoder, different objective) — cleaner "different architecture, same modality" contrast, the closest match to ORB-vs-UMA's "two ways of encoding the same thing."
  - Optional third arm: a **GNN/GVP structure encoder** or the AlphaFold Evoformer/structure-module trunk, to run the *mirror* (structure→energy) direction and connect to Roney & Ovchinnikov.

Recommended minimal pair for a first pass: **ESM-2 650M vs ProtT5-XL** (both sequence-only, sharply different architectures) — the tightest analogue of ORB (non-equivariant vector) vs UMA (equivariant decomposition). Add SaProt/ESM-3 as the structure-aware arm in phase 2.

### 4.2 Dataset (the "154k Materials-Project structures" analogue)

- **Structures/sequences:** a curated, non-redundant protein set with known 3D structure so every emergent target has ground truth — e.g. **CATH S40 / S95** or a PDB cull (~30% sequence-identity clustered), plus AlphaFold DB structures to scale. Target ~100–150k chains to match the materials scale; start with a 5–10k test subset exactly as `create_test_subset.py` does.
- **Why known-structure:** the entire point is comparing emergent representations against labels the model never trained on, so labels must exist per residue and per chain.

### 4.3 The property/label set (the "12 MP properties" analogue)

Split by locality, mirroring the materials extensive/local vs global/electronic axis:

| Locality | Property (task) | Label source |
|---|---|---|
| **Residue-local (expect early / linear)** | Secondary structure (Q3/Q8) | DSSP / NetSurfP |
| | Relative solvent accessibility / burial | DSSP |
| | Backbone torsion / local geometry | DSSP |
| | Intrinsic disorder | DisProt / NetSurfP |
| | Residue–residue **contacts** | PDB (the Rao 2021 target) |
| **Whole-protein / global (expect late / non-linear)** | Fold class / architecture (CATH) | CATH |
| | Subcellular localisation | DeepLoc |
| | Enzyme class (EC top level) / GO function | SwissProt / GO |
| | Remote homology (SCOP superfamily) | SCOP |
| **Biophysical / regression** | Stability ΔΔG on mutation | FireProtDB / mega-scale ΔΔG (Tsuboyama 2023) |
| | Fluorescence / fitness | FLIP, ProteinGym |
| | Thermostability (Tm) | Meltome / FLIP |

This directly reproduces the project's most important finding structure: a locality axis along which the *best extraction layer* should shift, plus a linear-vs-nonlinear (XGB−linear gap) signature.

### 4.4 Pipeline stages — one-to-one with the existing scripts

| Stage | Materials script | Protein port | Notes |
|---|---|---|---|
| 01 | `01_fetch_structures.py` (MP API) | `01_fetch_proteins.py` | Pull CATH/PDB/AFDB chains + labels → per-chain manifest (`index.jsonl`), same validation pattern. |
| 02 | `02_extract_embeddings_{orb,fairchem}.py` | `02_extract_embeddings_{esm,prott5}.py` | **Forward-hook all-layer extraction** — identical idea; per-**residue** (= "nodes") and per-**contact/attention** (= "edges") tensors → `.pt`. Attention maps are the protein "edge" analogue (Rao 2021). |
| 03 | `03_analyze_embeddings_*` | `03_analyze_embeddings_*` | **PCA/UMAP + k-NN purity + LVR** across layers, per-residue and pooled. Reuse almost verbatim; labels are per-residue (SS, burial) and per-chain (fold, localisation). |
| 04 | `04_quality_control_*` + `qc_common.py` | same | **CKA/SVCCA between ESM and ProtT5** (the Platonic test), HDBSCAN, effective rank/anisotropy/collapse, k-stability. `qc_common.py` transfers with near-zero change. |
| 05 | `05_property_prediction_*` | same | **Linear + XGBoost probes** per layer × pooling × property; parity/confusion/per-class-F1; the XGB−linear gap; **learning curves** (probe accuracy vs training-set size vs from-scratch) to test sub-claim 2 directly. |
| — | — | **06 (new, optional)** | **Sparse-autoencoder / mechanistic** pass à la Gujral 2025 — extract monosemantic features and align to GO/SS. The "phase 2" beyond probes. |

### 4.5 Analyses that carry over unchanged (the paper's spine)

1. **Both models are strong, transferable descriptors** of properties they never trained on (§2.2 analogue) → best-layer probe table across the property set.
2. **Local→global depth split** (§2.3) → residue-local properties peak early and linearly; global/functional peak late and non-linearly; best layer tracks locality.
3. **Geometry vs decodability** (§2.4) → cases where k-NN/LVR and probe R²/accuracy diverge.
4. **Pooling bottleneck** → per-residue > mean-pooled for residue-level targets; the reverse for whole-protein targets.
5. **Cross-architecture convergence** → CKA/SVCCA/mutual-kNN ESM↔ProtT5 (↔structure model), a sequence/structure Platonic test.
6. **Embedding health** → anisotropy/collapse/rank as intrinsic reliability signals.

### 4.6 Concrete first steps (a two-week skeleton)

1. Stand up `01`+`02` for **ESM-2 650M** on a **5k CATH-S40 test subset** (mirror `create_test_subset.py`/`run_test_analysis.py`); confirm all-layer residue + attention `.pt` extraction and per-residue DSSP labels.
2. Port `03` (PCA/UMAP/k-NN/LVR) — validate the local→global depth split on **secondary structure (local) vs fold class (global)** as the two anchor properties.
3. Add **ProtT5** as model B; run `04` CKA/SVCCA to get a first convergence number.
4. Scale to the full ~100k set; run `05` probes + learning curves; write results against the anchors in §2 above.
5. (Optional) `06` sparse-autoencoder pass for the interpretability extension.

### 4.7 Risks & honest caveats

- **Not a new phenomenon.** As in materials, the emergence itself is long-established (Rives 2021). Frame the contribution as the *unified layer-resolved, cross-architecture, geometry-vs-decodability protocol on a specific matched pair* — and cite the anchors, don't claim a void.
- **Label coverage skew** (the §2.9 materials caveat recurs): ΔΔG / Tm / localisation labels are far sparser and more biased than SS/burial. Report per-property coverage, exactly as the ORB edge-census/coverage table does.
- **Data leakage / redundancy.** Proteins are highly redundant; use identity-clustered splits (CATH-S40, time-split, or held-out superfamilies) so probe accuracy isn't memorised homology — the analogue of not double-counting materials.
- **Attention-as-edges is a modelling choice.** ESM contacts live in attention (Rao 2021), not in a node/edge graph like ORB; the "edge" analogue is attention maps, and ProtT5 attention differs — document this as the one place the port is not literal.
- **Anisotropy.** PLM embeddings are known to be anisotropic; the 04 geometry diagnostics are not just decorative here — they may be load-bearing for interpreting probe results.

---

## 5. One-paragraph answer (for the user)

The question the ORB/UMA project asks — *do models trained on one narrow signal spontaneously encode properties they never saw, and where does that live?* — is, in structural biology, not an open question but a **founding result and an active interpretability subfield**. Protein language models trained only on masked-residue prediction demonstrably encode 3D structure, contacts, secondary structure, binding sites and function (Rives 2021 PNAS; Rao 2021 ICLR; Lin 2023 Science; Vig 2021 ICLR), the depth-organisation of that information has been probed (Vig 2021; layer-selection work), independently trained models converge (Platonic Representation Hypothesis, Huh 2024), and the emergent features have been mechanistically decomposed (Gujral 2025 PNAS). AlphaFold even provides the exact *mirror* of the materials setup — a model trained on structure that spontaneously learned an energy function (Roney & Ovchinnikov 2022 PRL). So a bio port cannot claim novelty of the phenomenon; its defensible contribution is exactly the materials paper's: a **single, unified, layer-resolved, cross-architecture protocol** — PCA/UMAP/k-NN/LVR geometry, CKA/SVCCA convergence, and linear+XGBoost decodability with a local→global depth law and a pooling-bottleneck analysis — applied to a **matched pair of protein foundation models (proposed: ESM-2 vs ProtT5, plus a structure-aware arm)** over a large known-structure protein set, reusing the existing 01→05 pipeline nearly verbatim. Section 4 above is the concrete plan.

---

### Provenance
References located via `~/search_papers.py` (arXiv + Semantic Scholar) and web search on 2026-07-07. Verify author lists, venues, DOIs and citation counts before submission — Semantic Scholar metadata is occasionally truncated, and a few venue/year fields here are from secondary sources.
