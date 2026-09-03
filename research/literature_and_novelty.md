# Do foundation models spontaneously learn what they were never trained on? — the evidence in structural biology

*Compiled 2026-07-07 via arXiv and Semantic Scholar search. Citation counts and some metadata are as reported on the retrieval date and should be re-verified before any submission; arXiv IDs / DOIs are given where confirmed.*

---

## 0. The question

The question this review surveys is:

> **Do protein foundation models — trained only on masked amino-acid prediction (sequence statistics) — spontaneously develop internal representations of 3D structure, function, and biophysics they never saw in their training signal, and where in the network does each property live, and is that organisation shared across architectures?**

The question has a mirror-image form worth keeping in view. In protein **language** models the training signal is *sequence* and the emergent targets are *structural/physical*. In the protein **structure** models (AlphaFold) the training signal is *structure* and the emergent target is an *energy function* — the same phenomenon read in the opposite direction.

| Axis | In protein foundation models |
|---|---|
| Training signal | Sequence models (ESM): masked amino-acid token. Structure models (AlphaFold): 3D coordinates |
| Emergent, un-trained targets | Seq→ secondary/tertiary structure, contacts, burial/SASA, disorder, binding sites, localisation, EC/GO function, stability (ΔΔG), fitness. Struct→ a physical folding energy |
| Locality split | Residue-local (secondary structure, burial) vs global/whole-protein (fold class, localisation, function) |
| Layer axis | Embedding → N transformer blocks → head |
| Cross-architecture convergence | Convergence of PLMs / the Platonic Representation Hypothesis |
| Geometry vs supervised decodability | k-NN / LVR / UMAP organisation against probe accuracy |
| Pooling bottleneck | Per-residue vs mean-pooled (residue-level vs whole-protein targets) |
| Embedding health | CKA/SVCCA, anisotropy, collapse; PLM anisotropy/collapse is a known issue |

---

## 1. Verdict: how well is this question answered in the bio / structural-biology space?

**Very well.** The "does a model trained on one signal spontaneously encode another" question is not just answered in structural biology; it is one of the *founding results* of the modern protein-ML field and has a large, mature, still-growing interpretability literature attached to it. A three-part decomposition of the claim makes the state of the evidence clear:

| Sub-claim | Status in biology | Anchor evidence |
|---|---|---|
| 1. Models encode properties beyond their training target | **Established as a field-defining result** | Rives 2021 (PNAS); Rao 2021 (ICLR); Lin 2023 (Science); Vig 2021 (ICLR); Roney & Ovchinnikov 2022 (PRL) |
| 2. Less labelled data needed vs. from scratch (transfer / few-shot) | **Established; standard practice** | TAPE (Rao 2019); FLIP (Dallago 2021); PEER (Xu 2022); Metalic (2024) |
| 3. "Where in the network / layer-resolved / cross-architecture convergence" | **Actively studied, not yet saturated** | Vig 2021; "Layer by Layer" (2502.02013); PEPE (2025); Platonic Hypothesis (Huh 2024); Gujral 2025 (PNAS) |

Any "nobody has looked at this" framing is therefore indefensible in biology. **The defensible contribution is narrow and methodological**, not the phenomenon itself — a *unified, layer-resolved, cross-architecture, geometry-vs-decodability* protocol applied to a *specific pair* of protein foundation models.

---

## 2. The literature, by theme

### 2.1 The founding "emergence" results — sequence → structure & function

This is not controversial; it is the reason the field exists.

- **Rives, Meier, Sercu, Goyal, … Fergus (2021).** *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* **PNAS** 118(15):e2016239118. DOI:10.1073/pnas.2016239118.
  → The canonical statement of the exact phenomenon: a transformer (ESM-1b) trained **only** on masked amino-acid prediction develops internal representations from which secondary structure, tertiary contacts, remote homology, and mutational effects are linearly recoverable — none of which is in the training loss. The title alone is the thesis.

- **Rao, Meier, Sercu, Ovchinnikov, Rives (2021).** *Transformer protein language models are unsupervised structure learners.* **ICLR 2021.** bioRxiv 2020.12.15.422761.
  → 3D residue–residue contacts fall out of the **attention maps** with a sparse logistic regression; specific attention heads specialise for specific contact types. The cleanest "the geometry is already in there, unsupervised" result.

- **Lin, Akin, Rao, … Rives (2023).** *Evolutionary-scale prediction of atomic-level protein structure with a language model.* **Science** 379(6637):1123–1130. DOI:10.1126/science.ade2574.
  → ESM-2 / ESMFold: scaling the masked-LM objective to 15 B parameters makes an **atomic-resolution** picture of structure emerge in the representations, enabling structure prediction from a single sequence. Emergence as a function of scale.

- **Roney & Ovchinnikov (2022).** *State-of-the-Art Estimation of Protein Model Accuracy Using AlphaFold.* **Phys. Rev. Lett.** 129, 238101. DOI:10.1103/PhysRevLett.129.238101.
  → **The mirror image of the language-model setup.** AlphaFold, trained on *structure*, has implicitly learned a physical **energy function** it was never given: it ranks candidate structures consistently with folding physics and can locate low-energy conformations without co-evolutionary input. Trained on sequence → learns structure; trained on structure → learns energy. The same phenomenon, run backwards. (See also *An Evaluation of Biomolecular Energetics Learned by AlphaFold*, bioRxiv 2025.06.30.662466.)

### 2.2 Layer-resolved probing — *where* in the network the information lives

This is the "local→global depth split" question.

- **Vig, Madani, Varshney, Xiong, Socher, Rajani (2021).** *BERTology Meets Biology: Interpreting Attention in Protein Language Models.* **ICLR 2021.** arXiv:2006.15222.
  → Attention captures folding contacts and binding sites, and — critically for us — **targets progressively more complex biophysical properties with increasing layer depth**, consistently across BERT/ALBERT/XLNet. This is the protein statement of "local early, global late," already published.

- **"Layer by Layer: Uncovering Hidden Representations in Language Models"** (2025). arXiv:2502.02013.
  → General-LM evidence that **intermediate layers beat the final layer**, mid-network optimum. The generic version of "using the deepest layer can *hurt*."

- **PEPE: scalable extraction of multi-modal protein language model representations** (2025). bioRxiv 2025.10.13.680902.
  → Documents that PLM embedding quality depends strongly on **which layer, which pooling, which padding** — the "over-reliance on the final layer" oversight. Directly motivates our per-layer × pooling sweep.

- **Layer Probing Improves Kinase Functional Prediction with Protein Language Models** (2025). arXiv:2512.00376.
  → A worked example that *the best layer is task-dependent* in PLMs — the protein version of "best extraction layer depends on the target's locality."

### 2.3 Cross-architecture convergence — the "Platonic" analogue

The same convergence claim, established outside biology: independently built interatomic potentials are reported to arrive at a shared representation.

- **Huh, Cheung, Wang, Isola (2024).** *The Platonic Representation Hypothesis.* **ICML 2024.** arXiv:2405.07987.
  → The general claim that independently trained networks converge to a shared representation, measurable by **CKA / mutual-kNN** — the metrics used throughout this review. Provides the theoretical frame for an ESM-vs-structure-model convergence test, and the counter-literature (e.g. *Back into Plato's Cave*, arXiv:2604.18572) supplies the honest caveats.

- **Reverse Distillation: Consistently Scaling Protein Language Model Representations** (2026). arXiv:2603.07710.
  → PLMs within a family scale *poorly* and share nested subspaces — a convergence/redundancy result specific to proteins, relevant to whether "bigger = more emergent" holds.

### 2.4 Mechanistic interpretability — what the features actually are

Goes a step beyond probe- and geometry-based analysis and shows where the bio field is heading — useful for positioning and for an optional extension.

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

## 3. What is genuinely novel (the defensible residual)

*Updated after the study was run. The original version of this section was written before any
results existed; it is superseded by what follows.*

Claim novelty of the **analysis**, not the phenomenon. Emergence in protein language models is a
founding result (§1), so the contribution is the protocol and the controls:

1. **Cross-modality convergence with a full control ladder.** Convergence measured between a
   sequence-trained and a structure-trained model across four architecture families (transformer,
   dilated CNN, message-passing GNN, GVP-transformer), bracketed by within-modality ceilings,
   randomly-initialised floors, residue-permutation nulls and dimension matching. Most convergence
   work reports a single number for a single pair.
2. **Convergence is metric-dependent, and the spread is the finding.** CKA, SVCCA and mutual k-NN
   agree on the *ordering* of pairs but disagree on magnitude by roughly fivefold. Read as a
   fraction of the same-modality ceiling, cross-modality convergence is 27–36 % by CKA, 40–60 % by
   SVCCA and 10–18 % by mutual k-NN: the models share a **linear subspace**, not a common geometry
   and not a common neighbourhood structure.
3. **A mechanism for the dissociation.** Both representations are strongly anisotropic, which
   inflates subspace overlap while destroying neighbourhood overlap — so the metric spread in (2) is
   explained rather than merely reported.
4. **A methodological warning with teeth.** The same anisotropy makes SVCCA's permutation null large
   and width-dependent (0.29 at 480-d, 0.48 at 1280-d). An apparent growth of convergence with model
   scale disappears once the null is subtracted. Cross-model or cross-scale SVCCA comparisons that
   omit a permutation null and dimension matching can report representation *width* as
   representational *convergence*.
5. **A sensitivity control for the metrics themselves.** Two replicates of one architecture differing
   only in random seed reach mutual k-NN 0.78, establishing that the neighbourhood measure can
   detect correspondence when it exists — so its near-zero cross-modality value is an absence, not a
   pessimistic metric. This is the sensitivity/specificity standard urged by Ding et al. (2021),
   applied rather than cited.
6. **Functional criteria alongside structural ones.** Structural similarity measures are criticised
   for ignoring behaviour, so convergence is checked by model stitching — mapping a structure model's
   representation through a fitted linear connector into the sequence model's own frozen output head
   — and by linear predictivity across every pair in both directions.
7. **An emergent-vs-compositional separation.** An amino-acid-composition baseline distinguishes
   properties genuinely encoded by the network from those recoverable from residue counts alone.

---

## 4. What was built

### 4.1 Models

| Role | Model | Representation |
|---|---|---|
| Sequence arm | **ESM-2** `esm2_t12_35M` (13 layers, 480-d) and `esm2_t33_650M` (34, 1280-d) | Masked-LM transformer; never sees coordinates |
| Sequence arm, second architecture | **CARP-38M** | Dilated-CNN masked LM, parameter-matched to ESM-2 35M |
| Structure arm | **ProteinMPNN encoder** (3 layers, 128-d) | Message-passing GNN on a k = 48 backbone graph; zero node init, rotation-invariant edges, sequence-agnostic input |
| Structure arm, second architecture | **ESM-IF1** (512-d) | GVP-transformer inverse folding |
| Metric-sensitivity control | **ESM-1v** seeds 1 and 2 | Same architecture, same data, different random seed |
| Floors | Randomly-initialised ESM-2 and ProteinMPNN | Untrained counterparts of both arms |

### 4.2 Dataset

4,898 non-redundant single-domain chains — one representative per **CATH S35** sequence cluster, with
experimental structures from the RCSB PDB. 1,374,765 residues; chain length min 40, median 241, max
992; 2,253 Bacteria, 2,058 Eukaryota, 256 Archaea, 211 Viruses. 4,781 chains (97.6 %) map to a
UniProt accession through the SIFTS flatfile. Both models emit one vector per residue for the *same*
residues, so every layer pair is compared with exact correspondence and no alignment assumption.

### 4.3 Properties

All properties are evaluated on the same protein set, so no comparison is confounded by a change of
substrate. Extrinsic properties requiring a different set (binding affinity, dynamics, allostery)
were deliberately excluded for this reason.

| Locality | Property | Source |
|---|---|---|
| **Residue-level** | Secondary structure (3-state) | biotite `annotate_sse` |
| | Relative solvent accessibility / burial | biotite `sasa` |
| | B-factor (per-chain normalised) | PDB |
| | Amino-acid identity | sequence |
| | Binding site, active site, PTM site | UniProt sequence features, aligned to chain |
| **Chain-level** | CATH class / architecture / topology / homologous superfamily | CATH |
| | EC top-level class; enzyme vs non-enzyme | UniProt |
| | Subcellular localisation; taxonomic kingdom | UniProt |
| | Pfam family; PTM flags (phospho, glyco) | UniProt |
| | Functional protein classes (transport, DNA/RNA-binding, kinase, ribosomal, metal-binding, membrane, structural/cytoskeleton, immune) | UniProt keywords |

### 4.4 Pipeline stages

Six entry points, each grouping the stages that answer one kind of question, selected by
subcommand. Every stage writes a `params.json` beside its outputs recording the resolved
arguments, the command, the git commit and library versions.

| Entry point | Subcommands | What it does |
|---|---|---|
| `01_dataset.py` | `fetch`, `annotate`, `targets` | CATH S35 chains and structures; UniProt/SIFTS chain labels; per-residue site and B-factor targets |
| `02_extract.py` | `esm`, `struct`, `esmif1`, `carp`, `random` | Forward-hook all-layer per-residue extraction for every model arm, including untrained counterparts |
| `03_geometry.py` | `depth-law`, `overlap`, `health`, `clusters`, `umap` | Per-model representation structure: k-NN purity and LVR, geometry–label overlap, effective rank and anisotropy, density clusters, projections |
| `04_convergence.py` | `grids`, `supervised`, `significance`, `svcca-controls`, `functional` | Layer × layer CKA/SVCCA/mutual-kNN against a permutation null; prediction agreement; resampled CIs; the SVCCA null and dimension matching |
| `05_probes.py` | `probe`, `composition` | Linear + XGBoost probes per layer × property with chain-grouped splits, against an amino-acid-composition baseline |
| `06_stitching.py` | `stitch`, `predictivity`, `matrix`, `depth` | Functional tests: stitching through each model's own frozen head, and linear predictivity across every pair |

### 4.5 Headline results

- Cross-modality CKA **0.214–0.285** against a permutation null of 0.001–0.005; within-structure
  0.517; within-sequence 0.796; untrained × untrained 0.038.
- Metric spread of roughly fivefold as described in §3.2, with mutual k-NN near zero across modality.
- Seed-pair ceiling: CKA 0.832, SVCCA 0.760, mutual k-NN 0.782.
- Stitching a structure representation into ESM's own frozen `lm_head` recovers **35.8 %** of
  residues from ProteinMPNN and **52.3 %** from ESM-IF1, against a 16.5 % untrained floor and a
  96.9 % ceiling.

### 4.6 Risks & honest caveats

- **Not a new phenomenon.** Emergence itself is long-established (Rives 2021). The contribution is
  the unified layer-resolved, cross-architecture, geometry-vs-decodability protocol and its controls
  — cite the anchors, don't claim a void.
- **Label coverage skew.** Site-level labels (active site 0.09 %, PTM site 0.11 % of residues) are
  far sparser than secondary structure or burial. Report per-property coverage and macro-F1 rather
  than a single headline accuracy, and exclude labels too sparse to score.
- **Data leakage / redundancy.** Proteins are highly redundant; chain-grouped, identity-clustered
  splits (CATH S35) keep probe accuracy from measuring memorised homology.
- **Anisotropy is load-bearing, not decorative.** It is the mechanism behind both the metric
  dissociation and the inflated SVCCA null, so the geometry diagnostics carry interpretive weight.
- **A shared prediction target.** The two arms have opposite *input* modalities but both are trained
  to predict amino-acid identity — ESM-2 by masking, ProteinMPNN by inverse folding. That shared
  target is a competing explanation for convergence and must be addressed rather than elided.

---

## 5. One-paragraph answer

The question — *do models trained on one narrow signal spontaneously encode properties they never saw, and where does that live?* — is, in structural biology, not an open question but a **founding result and an active interpretability subfield**. Protein language models trained only on masked-residue prediction demonstrably encode 3D structure, contacts, secondary structure, binding sites and function (Rives 2021 PNAS; Rao 2021 ICLR; Lin 2023 Science; Vig 2021 ICLR), the depth-organisation of that information has been probed (Vig 2021; layer-selection work), independently trained models converge (Platonic Representation Hypothesis, Huh 2024), and the emergent features have been mechanistically decomposed (Gujral 2025 PNAS). AlphaFold even provides the exact *mirror* of the language-model setup — a model trained on structure that spontaneously learned an energy function (Roney & Ovchinnikov 2022 PRL). So a study here cannot claim novelty of the phenomenon; its defensible contribution is methodological: a **single, unified, layer-resolved, cross-architecture protocol** — PCA/UMAP/k-NN/LVR geometry, CKA/SVCCA/mutual-kNN convergence against permutation nulls, and linear+XGBoost decodability — applied across the **sequence/structure divide** (ESM-2 and CARP against ProteinMPNN and ESM-IF1) on a matched set of 4,898 known-structure chains, with within-modality ceilings, untrained floors and a same-architecture different-seed control. Section 4 above describes what was built and what it found.

---

## 6. Direct-convergence prior-work check (2026-07-07 update — for the seq↔structure project)

After choosing the **cross-modality convergence** angle (ESM-2 sequence vs ESM-IF1/ProteinMPNN
structure), a focused search was run to confirm nobody has already measured representational
convergence *between a sequence-trained and a structure-trained protein model*. Result: the nearest
works stop short of exactly this, so the gap is real.

| Nearest prior work | What it actually does | Why our study differs |
|---|---|---|
| **Two Stages of Folding: Convergent Mechanisms in AI Protein Folding Trunks** (Lu, Brinkmann, Belinkov, Bau, Wendler — arXiv:2602.06020) | CKA + linear alignment/steering across **structure-prediction trunks only** (ESMFold, OpenFold, Boltz-1); finds a two-stage biochemical→spatial mechanism and cross-model linear interchangeability of pairwise states | All three models are **structure predictors (one modality)**. We cross the **sequence↔structure divide** — models trained on *opposite* signals (masked-LM vs structure→sequence). |
| **When Does Structure Help? The Information Bonus of AlphaFold2 Representations over Protein Language Models** (arXiv:2606.04228) | Information-theoretic probing of the **downstream task gain** of AF2 features over PLM features (PDBbind, AllosteryDB) | Explicitly **does not** do representational-alignment (CKA/SVCCA/RSA). We do the layer-resolved convergence map it omits. |
| **Platonic Representation Hypothesis** (Huh et al. 2024, ICML — arXiv:2405.07987) | Cross-domain representational convergence, general; CKA / mutual-kNN | Not proteins, not seq↔structure. We instantiate it for the protein sequence/structure divide. |
| PLM emergence (Rives 2021; Rao 2021; Lin 2023) | Structure **emerges from sequence** within a **single** model | Single-model emergence, not a two-model convergence *measurement*. |
| Layer probing of PLMs (Vig 2021; PEPE 2025) | Depth analysis of **PLMs only** | Single-modality; no structure-model counterpart, no convergence grid. |
| Sequence+structure fusion (INFUSSE, arXiv:2502.17294; ProtST) | **Combine** the two modalities to boost task accuracy | Fusion for performance ≠ measuring whether the two *independently* converge. |

**Conclusion (novelty statement for the paper).** A **layer-resolved, bidirectional
representational-convergence map** between a sequence-MLM (ESM-2) and a structure-trained
inverse-folding model (ESM-IF1 / ProteinMPNN) — attributing *where* (which layer pairs) and *for
which properties* the two converge, alongside a per-model local→global depth law and a
geometry-vs-decodability analysis — is not present in the literature. The emergence phenomenon is
old; the **cross-modality convergence measurement between opposite training signals** is the new,
publishable contribution. Cite 2602.06020 / 2606.04228 / Huh 2024 as the nearest neighbours we extend.

---

## 7. Cross-*architecture* convergence — is transformer↔GNN convergence surprising? (2026-08-04)

Motivating observation: our two models are not just trained on opposite signals, they are
**architecturally very different** — a Transformer (attention, sequence) vs a message-passing GNN
(geometry). Is cross-architecture representational convergence expected or surprising? The literature
cuts in favour of "surprising".

**Convergence is an established general idea — but mostly for similar families / same modality / at
the output level:**
- **Platonic Representation Hypothesis** (Huh et al., ICML 2024, arXiv:2405.07987) — models across
  architectures/datasets/modalities converge (CKA / mutual-kNN). Umbrella claim, largely same-modality.
- **Model stitching** — Lenc & Vedaldi 2015; **Bansal, Nakkiran & Barak 2021** ("Revisiting Model
  Stitching", arXiv:2106.07682); Hernandez et al. 2023 (arXiv:2303.11277). Strongest "different nets
  are compatible" evidence: splice bottom-of-A into top-of-B with a **single linear layer**, lose
  only 2–5% accuracy. But mostly CNN↔CNN / same-modality.
- **CKA** (Kornblith et al., ICML 2019) — the standard cross-architecture comparison tool.
- **Relative representations / latent communication** (Moschella et al. 2023; arXiv:2406.11014) —
  zero-shot stitching across models via relative embeddings.

**But when *genuinely* different architectures are compared, studies usually emphasise DIFFERENCE:**
- **"Do Vision Transformers See Like CNNs?"** (Raghu et al., NeurIPS 2021) — ViT vs CNN internal
  representations are **structurally different** (ViT uniform/global-early; CNN hierarchical). Same
  task, different mechanisms.
- The few **GNN-vs-Transformer** CKA comparisons report transformers have **lower** similarity to
  GCNs — i.e. divergence.

**Nuance to use in framing** — convergence is often **topological** (same nearest neighbours) rather
than **geometric** (same distances): "Back into Plato's Cave" (arXiv:2604.18572). This maps onto our
data (moderate CKA ~0.27, weak mutual-kNN ~0.02) — frame our result as **partial, geometric-leaning**
convergence, not full alignment.

**Takeaway for the paper.** Cite the convergence literature (Platonic, stitching) as the backdrop,
but note that **transformer↔message-passing-GNN, opposite-signal, in proteins is a real gap**, and the
architecture-comparison literature (ViT≠CNN, transformer≠GCN) makes a *positive* convergence result
**more** noteworthy: "prior work on very different architectures leads one to expect divergence; we
instead find partial convergence." This is a stronger novelty hook than the cross-modality angle alone.

---

### Provenance
References located via `~/search_papers.py` (arXiv + Semantic Scholar) and web search on 2026-07-07
(§0–5), a convergence-specific search on 2026-07-07 (§6), and a cross-architecture search on
2026-08-04 (§7). Verify author lists, venues, DOIs and citation counts before submission — Semantic
Scholar metadata is occasionally truncated, and a few venue/year fields here are from secondary sources.

---

## 8. What counts as "true" convergence? Metric critiques (2026-08-06)

Prompted by the observation that CKA is considered lenient. The field does **not** treat any single
similarity number as evidence of convergence; there is an explicit critical literature.

**The metrics disagree, and that is a known problem.**
- **Ding, Denain & Steinhardt (2021), "Grounding Representation Similarity with Statistical
  Testing"** (NeurIPS 2021; arXiv:2108.01661). The canonical reference. CCA and CKA "often disagree
  on fundamental observations, such as whether deep networks differing only in random initialization
  learn similar representations." Proposes the standard a metric must meet: **sensitivity** to
  changes that affect functional behaviour, and **specificity** against changes that do not. Compares
  CKA, three CCA variants and orthogonal Procrustes against probing accuracy and robustness.
- **"Reliability of CKA as a Similarity Measure"** (ICLR 2023, openreview 8HRvyxc606) and
  **"Deceiving the CKA Similarity Measure in Deep Learning"** — CKA can be manipulated / gives
  unreliable verdicts; it is sensitive to a small number of high-variance directions.
- Structure-based metrics (CKA, SVCCA) are criticised for **overestimating similarity from spurious
  feature correlations** and for being **agnostic to functional behaviour and invariances**. CCA/CKA
  "do not distinguish features that are learned and relevant for downstream tasks from spurious
  features"; e.g. appending 1000 useless random coordinates changes CKA without changing the
  representation meaningfully.

**What the field treats as stronger evidence.**
- **Model stitching** (Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021, arXiv:2106.07682;
  Hernandez et al. 2023) — *functional* similarity: can representation A be linearly mapped into
  model B and still work? Stitching "obtains results that align more closely with intuitions that
  well-performing networks learn similar representations", and unlike CKA can say one representation
  is *better*, not merely *far*. Recent work (arXiv:2505.20142, "Grounding Functional Similarity by
  Invariance-Aware Model Stitching") shows invariance-aware CKA can deviate substantially from
  stitching-based functional similarity.
- **Mutual k-NN** is the alignment measure adopted for the Platonic Representation Hypothesis
  (Huh 2024) — a local, neighbourhood-level criterion, stricter than global CKA/SVCCA.
- **Downstream/probing agreement** (Ding 2021's functional-behaviour axis) — our Cohen's-κ
  cross-model prediction agreement is an instance of this.

**Implication for our paper.** Reporting a single metric is not defensible; the honest presentation
is the **full three-metric matrix plus a functional check**, which we have (CKA, SVCCA, mutual k-NN,
and supervised κ agreement), each against a permutation null and against within-modality ceilings and
an untrained floor. Our own data show why: on the same pairs, SVCCA reads ~5× more convergence than
mutual k-NN as a fraction of the ceiling, and untrained networks reach SVCCA 0.163 while their
mutual k-NN is 0.004 — a paper reporting SVCCA alone could mistake an untrained pair for a partially
converged one. This makes the metric-dependence claim a *documented field problem* we quantify,
rather than an idiosyncrasy of our setup.

---

## 9. Functional-similarity measurement: what worked, what is circular (2026-08-07)

Prompted by the metric critique (§8): structural measures are agnostic to function, so we added
functional tests. Two lessons, both worth keeping.

**Naming.** What we first called "cross-prediction" is a ridge map from model A's embeddings to
model B's, scored by held-out R². That name collides with **cross-prediction-powered inference**
(Zrnic & Candès), a distinct statistical method for valid inference from ML-imputed labels. The
correct name for what we do is **linear predictivity** (as used in NeuroAI / Brain-Score), or
"linear regression similarity". Renamed throughout.

**Stitching is only well-defined in one direction here, and the reason is general.**
A stitch feeds donor model A's representation through a fitted linear connector into receiver model
B's *own frozen head*, and asks whether B still performs *B's task*.

- **struct → seq (valid).** ProteinMPNN's node state → ESM's own `lm_head`, task = residue identity.
  ProteinMPNN is *sequence-agnostic*, so it cannot contain the target. Result: 0.358 residue
  recovery vs 0.144 for an untrained ProteinMPNN, against a 0.982 self-ceiling (ESM reading its own
  representation, trivially high). A genuine functional stitch.
- **seq → struct (circular, discarded).** ESM node state → ProteinMPNN's frozen decoder, task =
  sequence recovery. ESM's representation *trivially contains the sequence* (its own `lm_head`
  recovers 98 % of its input), so the connector hands the decoder the answer. Diagnostic: an
  **untrained** ESM scored *above* ProteinMPNN's own encoder (0.569 vs 0.458) — a random
  transformer's residual stream still carries token identity. No fix within this direction.

**General rule worth stating in the paper:** a stitch is only interpretable when the donor
representation does not already determine the receiver's target, and when the receiver's head
consumes the same object the donor produces (ProteinMPNN's decoder additionally consumes an *edge*
state that a sequence model has no analogue for). Both conditions should be checked before reporting
a stitching number; an untrained-donor control detects the failure.

**Implementation notes (bugs found).** (i) ProteinMPNN's encoder updates *both* h_V and h_E; feeding
the decoder the raw `W_e(E)` edges instead of the encoder-updated h_E collapses recovery to chance
(0.067 vs 0.476). (ii) Teacher-forced decoding leaks neighbours' true residues, so a single-shot
decode (no position treated as already decoded) is required; even then the circularity above
remains.
