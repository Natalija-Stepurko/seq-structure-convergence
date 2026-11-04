# Future work: where this study could go next

Five directions that follow from the finding that about half of the measured agreement between
ESM-2 and ProteinMPNN is the amino acid both are trained to predict. They are recorded here to
return to; none has been started. Each lists what this repository already provides.

What the study already shows that matters for all five: the two models are good at different
things. On the residue-level probes in `results/probes_ci/` (best layer, mean of 5 refits),
ProteinMPNN leads on structural targets and ESM-2 on functional ones:

| Target | ESM-2 35M | ProteinMPNN encoder |
|---|---|---|
| Relative solvent accessibility (linear R²) | 0.44 | **0.83** |
| Burial (linear accuracy) | 0.77 | **0.90** |
| Secondary structure, 3 states (linear accuracy) | 0.80 | **0.89** |
| B-factor, a flexibility proxy (linear R²) | 0.20 | **0.29** |
| Binding site (linear F1) | **0.82** | 0.79 |
| Active site (linear F1) | **0.61** | 0.53 |
| PTM site (linear F1) | **0.60** | 0.45 |

---

## 1. Does disagreement between ESM-2 and ProteinMPNN predict mutation effects?

Take a protein with leucine at some position. ESM-2 may find valine perfectly plausible from the
patterns in natural sequences; ProteinMPNN may give valine a low probability because it does not
fit the local structural environment. Does that disagreement say something useful about the
mutation?

**Approach.** For every possible substitution, compute two scores.

From ESM-2:

$$\Delta_{\mathrm{seq}} = \log P_{\mathrm{ESM2}}(a_{\mathrm{mut}} \mid \mathrm{context}) - \log P_{\mathrm{ESM2}}(a_{\mathrm{WT}} \mid \mathrm{context})$$

From ProteinMPNN:

$$\Delta_{\mathrm{struct}} = \log P_{\mathrm{MPNN}}(a_{\mathrm{mut}} \mid \mathrm{structure}) - \log P_{\mathrm{MPNN}}(a_{\mathrm{WT}} \mid \mathrm{structure})$$

Quantify their disagreement, for example

$$D = \lvert \Delta_{\mathrm{seq}} - \Delta_{\mathrm{struct}} \rvert$$

A larger value means the two models judge the mutation more differently. The two scores sit on
different scales, so calibrate or standardise them before reading $D$. Compare against measured
mutation effects, for example the deep mutational scans in ProteinGym.

**What would be new.** Not combining the two predictions, but asking whether the disagreement
itself carries information beyond either score. Does it systematically pick out:

- functionally important residues;
- buried versus exposed positions;
- regions where evolution conserves residues for reasons other than structural stability;
- mutations that are particularly hard to predict?

The result that matters is whether disagreement improves prediction on held-out proteins, not
whether the two models give different answers.

**Starting point here.** The study extracts embeddings, not output probabilities; both models'
heads would need to be run for every substitution. The per-residue solvent accessibility, burial,
secondary structure and site labels in `scripts/01_dataset.py` give the strata for the analysis.

---

## 2. Can deconfounded representations be combined more effectively?

Much of the measured agreement between ESM-2 and ProteinMPNN is amino-acid identity. If that
shared information is removed, are the remaining representations more useful together?

**Approach.** Compare three ways of combining the two models' per-residue embeddings, with the same
downstream predictor and evaluation:

| Fusion | Features |
|---|---|
| Raw | ESM-2 and ProteinMPNN embeddings, concatenated |
| Residual | Both embeddings with each amino acid's mean subtracted (identity-centred), concatenated |
| Hybrid | One-hot amino-acid identity plus both residuals |

Tasks: mutation-effect prediction, residue flexibility, functional-site classification. Compare
against ESM-2 alone, ProteinMPNN alone, and models with matched parameter counts.

The hybrid arm matters because removing amino-acid identity may discard useful predictive
information. The test is whether residualising separates complementary signals, not whether it
simply removes one of them.

One property of the design to account for: for a linear predictor, the hybrid features are an
invertible linear reparameterisation of [one-hot identity, raw ESM-2, raw ProteinMPNN], so an
unregularised linear model fits the two equally. Differences can only come from regularisation,
from the predictor being non-linear, or from limited training data. The comparison is most
informative with non-linear predictors and across training-set sizes.

**What would be new.** The study so far asks whether the models encode similar information. This
asks whether accounting for the shared information makes their representations more
complementary and easier to combine, a direct continuation of the convergence result.

**Starting point here.** Everything except mutation effects exists already: per-residue embeddings
for both models, the amino-acid subtraction (`_aa_residualise` in `scripts/04_convergence.py`),
chain-grouped train/test splits and the residue targets in
`scripts/05_probes.py`. The table at the top of this file is the single-model baseline.

**Novelty check (October 2026).** Each ingredient exists; the combination does not, as far as a
search of the protein-ML and multimodal-learning literature found.

| Prior work | What it does | How close |
|---|---|---|
| [PLM-eXplain](https://arxiv.org/abs/2504.07156) (van Eck et al., *Bioinformatics* 2026) | Splits ESM-2 embeddings into an interpretable subspace (amino-acid type, secondary structure, solvent accessibility) and a residual; downstream performance is maintained, not improved | Closest: separates amino-acid identity from the rest, but in one model and with no fusion |
| [SPURS](https://www.nature.com/articles/s41467-025-67609-4) (Nature Communications 2025) | Wires ProteinMPNN's per-residue representations into ESM-2 through an adapter to predict ΔΔG | Fuses the same two models; no removal of shared information |
| [ESM-GearNet](https://arxiv.org/abs/2303.06275) (Zhang et al., 2023) | ESM-2 plus structure encoders under serial, parallel and cross fusion | Fusion strategies compared; redundancy not examined |
| [When does structure help?](https://arxiv.org/abs/2606.04228) (Chauhan, 2026) | Linear "information bonus" of AlphaFold2 representations over ESM-2; structure helps for allosteric sites, not binding or flexibility | Asks when structure adds information; nothing removed before comparing |
| [MISA](https://github.com/declare-lab/MISA) (Hazarika et al., ACM MM 2020) | Learns modality-invariant and modality-specific subspaces before fusing | Shared/private separation before fusion, in multimodal sentiment analysis |
| [Partial information decomposition](https://arxiv.org/abs/2302.12247) (Liang et al., NeurIPS 2023) | Estimates redundancy, unique information and synergy between modalities for a task | A direct way to measure whether residualising makes the two more complementary |
| [LEACE](https://arxiv.org/abs/2306.03819) (Belrose et al., NeurIPS 2023) | Closed-form linear concept erasure; a concept is linearly undetectable exactly when every class has the same mean | Subtracting each amino acid's mean is this condition met by construction: a simple linear erasure of amino-acid identity |

Verdict: fusing a sequence model with a structure model is established, and so is separating
shared from modality-specific information before fusion. No work found removes the **shared
training target** of the two models before fusing them and asks whether that makes them more
complementary. The framing is new; the expected effect is uncertain, because of the linear
equivalence noted above. A null result is still informative here: it would say the redundancy
inflates similarity scores without hurting fusion.

**Design, fixed before any fusion run (16 October 2026).**

- *Representations.* One layer per model, chosen by a rule applied to the probe results already
  published (`results/probes_ci`, `results/probes650`): the layer that scores best on the most of
  the seven residue targets. ESM-2 35M layer 11 (best on 4 of 7), ProteinMPNN encoder layer 3
  (6 of 7), the untrained ProteinMPNN at the same layer, ESM-2 650M layer 32 (4 of 7). The rule
  uses earlier results on the same chains, so it flatters every arm alike, single and fused.
- *Residues and splits.* Exactly as `05_probes.py`: 8 residues per chain plus every binding-,
  active- and PTM-site residue, 25% of chains held out, five refits with seeds 42–46, each drawing
  new residues and a new chain-level split. Every arm in a refit sees the same residues and split.
- *Residualising.* Each amino acid's mean vector is computed on the training residues only and
  subtracted from training and test residues alike, so nothing about the test set leaks into it.
- *Arms.* ESM-2 alone; ProteinMPNN alone; one-hot amino acid alone; raw fusion [ESM-2,
  ProteinMPNN]; raw fusion plus one-hot, the control the hybrid has to beat; residual fusion
  [ESM-2 residual, ProteinMPNN residual]; hybrid fusion [one-hot, both residuals]. Size controls:
  ESM-2 fused with the untrained ProteinMPNN (same dimensions, no learned structure) and
  ESM-2 650M alone (about 19 times the parameters, one model).
- *Predictors.* Linear: ridge regression and logistic regression with the penalty chosen by
  3-fold cross-validation on the training chains. Gradient-boosted trees: XGBoost, 80 trees of
  depth 4, as in `05_probes.py`. A one-hidden-layer MLP of 256 units with early stopping. Features
  are standardised for the linear model and the MLP. The linear predictor doubles as a check:
  hybrid and raw-plus-one-hot hold the same information and should tie.
- *Training size.* 1%, 10% and 100% of the training chains, against the same test chains. A target
  is skipped at a size where the training residues hold fewer than 10 positives, and the skip is
  recorded.
- *Tasks and scores.* The seven residue targets: solvent accessibility and B-factor (R²), burial
  and three-state secondary structure (accuracy), binding, active and PTM sites (macro F1, with
  balanced class weights).
- *Measures.* For every fused arm, its gain over **ESM-2 alone** and its gain over **ProteinMPNN
  alone**, each reported separately, plus its gain over raw-plus-one-hot. Gains are paired: the
  difference is taken within each refit, where both arms see the same residues and split, and
  reported as the mean over the five refits with a 95% t-interval. Redundancy is measured by
  variance partitioning of the linear held-out scores: unique to ESM-2 = score(fused) −
  score(ProteinMPNN), unique to ProteinMPNN = score(fused) − score(ESM-2), shared =
  score(ESM-2) + score(ProteinMPNN) − score(fused), computed for raw and for residual fusion. It is
  the linear counterpart of the redundancy and unique-information terms of Liang et al.; their
  neural estimators are left for later.
- *Decision rule.* Residual or hybrid fusion counts as better than raw-plus-one-hot for a predictor
  if the paired 95% interval of its gain lies above zero on at least three of the seven targets at
  full training size. The same rule, applied at 1% and 10%, answers whether residualising helps
  when data is scarce. Any other comparison reported is descriptive.

Cost: no new extraction for phase 1. Loading one layer per model and about 2,800 probe fits take
a few hours of CPU.

**Result, phase 1 (run 16 October 2026; `scripts/08_fusion.py`, `results/fusion/`).**

- *Combining helps, and where it helps follows each model's blind spot.* With the linear predictor
  at full training size, raw fusion beats ESM-2 alone and ProteinMPNN alone on all seven targets,
  every one of the 14 paired gains with its interval above zero. Over ESM-2 alone the gains are
  largest on structure (solvent accessibility +0.400, burial +0.131); over ProteinMPNN alone on
  functional sites (PTM sites +0.154, active sites +0.094). On structural targets the gains over
  ProteinMPNN shrink to +0.007 at most with boosted trees, and turn negative with the MLP (solvent
  accessibility −0.028, B-factor −0.063), which fits the larger input less well.
- *The subtraction does not make the two easier to combine.* By the rule fixed in advance, neither
  residual nor hybrid fusion beats raw fusion plus one-hot for any predictor at any training size.
  Residual fusion is significantly worse on most targets: removing amino-acid identity removes
  information the targets use. Hybrid fusion ties raw-plus-one-hot with the linear predictor, as the
  linear equivalence predicts, and is never better by the rule.
- *Size controls.* ESM-2 with an untrained ProteinMPNN recovers most of the fusion gain on
  structural targets (solvent accessibility 0.812 against 0.841, linear), so much of what
  ProteinMPNN adds is the backbone geometry itself. ESM-2 650M alone beats the fusion on functional
  sites (active sites 0.717 against 0.632) and loses on structure (solvent accessibility 0.567).

**Specificity check: point mutants (`scripts/09_mutants.py`, `results/mutants/`).** Nine point
mutants of T4 lysozyme (L99A, L99I, K16R, K16E) and ubiquitin (V26A, V26I, K48R, K48E, I44A),
embedded with ESM-2 35M; means over blocks 1–12, in units of within-amino-acid spread:

- At the mutated residue the embedding moves 0.39–0.74; the subtraction removes 39–66% of that (the
  amino-acid swap itself) and leaves 0.17–0.33, the residue's response to its new context.
- The change is local: residues within 8 Å move 2–5 times more than residues over 16 Å away, most
  around the buried sites (4.7–5.2 times around L99 in T4 lysozyme).
- Radical substitutions move it more than conservative ones at the same site in all four pairs, as
  is and after the subtraction; the contrast is larger after it (ratio 1.3–1.7 against 1.2–1.4).
  Four pairs make this descriptive.
- ProteinMPNN's encoder does not change at all (exactly zero): it reads only the backbone, which a
  point mutation on the same structure leaves untouched. Testing it needs a mutant structure.

**Phase 2 design, fixed before running (4 November 2026).**

*Per-protein fusion.* The same nine arms on whole proteins: each model's chosen layer (ESM-2 35M
layer 11, ProteinMPNN encoder 3) averaged over the protein's residues. Removing each amino acid's
mean (fitted on training proteins' residues only) before averaging removes the part of the average
explained by the protein's amino-acid composition, so the identity arm is the 20-value composition
vector, the input of the counting baseline. Arms: ESM-2; ProteinMPNN; composition; raw fusion; raw
fusion plus composition; residual fusion; hybrid (composition plus both residuals); ESM-2 with an
untrained ProteinMPNN; ESM-2 650M (layer 32). Targets: the 18 protein-level properties of
`05_probes.py`, with its class filters (classes under 40 proteins dropped, at most 12 classes,
unknown and other excluded). Macro-F1. Same predictors, training sizes (a target is skipped at a
size where any class has fewer than 2 training proteins), refits and chain-level splits. Decision
rule: residual or hybrid fusion is better than raw fusion plus composition for a predictor if the
paired 95% interval of its gain is above zero on at least 6 of the 18 targets at full size.

*Predictions on the point mutants.* Predictors trained once (seed 42, full training set, linear and
boosted trees) on the residue targets for ESM-2 alone (raw and residual), ProteinMPNN alone, and
raw, residual and hybrid fusion, and on the protein-level targets for the same arms, are applied to
wild type and each mutant. Reported, descriptively: the change in predicted value at the mutated
residue, within 8 Å, and beyond 16 Å, and whether any protein-level prediction changes class.
Ground truth for one mutant: T4 lysozyme L99A has a crystal structure (1L90) on the cysteine-free
background whose structure is 1L63, so the measured per-residue change in solvent accessibility and
secondary structure is compared with the predicted change (Pearson r over residues; agreement of
secondary-structure changes), as is and after the subtraction.

**Result, predictions on the point mutants (`09_mutants.py predict`).** Predictors trained on the
dataset and applied to wild type and the nine mutants, as the design above fixed:

- *Predicted properties barely move, and only locally.* Averaged over the mutants, the predicted
  solvent accessibility of the mutated residue changes by 0.02–0.07 (on a 0–1 scale) and that of
  residues within 8 Å by 0.002–0.02; class predictions change at the site or within 8 Å in a handful
  of cases out of hundreds (4–15 per arm). ProteinMPNN alone never changes, since its input does not.
- *Fusing with ProteinMPNN damps the change for structural properties* (burial at the site, linear:
  0.064 for ESM-2 alone, 0.016 fused): half the input is a backbone the mutation leaves unchanged.
- *The subtraction mostly matters where amino-acid identity carries the answer.* For PTM sites the
  predicted change at a mutated lysine falls from 0.36 to 0.10 (ESM-2, linear) once identity is
  removed; for structural properties raw and residual predictions move about equally.
- *Protein-level predictions rarely change class* (0–9 of 162 per arm), almost all for L99A and on
  weakly separated targets such as kingdom and subcellular location.
- *Against the crystal structures, the predictions fail for L99A.* The measured change between 1L63
  and 1L90 is confined to residue 99, which becomes more exposed (+0.14 relative solvent
  accessibility, the cavity); the nine residues within 8 Å change by 0.004 on average, less than
  distant residues. Most arms predict residue 99 becomes slightly more buried (−0.02 to −0.08); only
  boosted trees on the residual and hybrid features predict an increase, a quarter of the measured
  size. Giving ProteinMPNN the real mutant backbone does not fix it. With one informative residue
  this is an anecdote, not a test; correlations within 8 Å are dominated by that residue and are
  not reported as evidence.

---

## 3. Can ESM-2 improve the sequences ProteinMPNN generates?

The most directly applicable design project:

1. Take an experimentally determined protein backbone.
2. Generate hundreds of candidate sequences for it with ProteinMPNN.
3. Score each sequence with ESM-2.
4. Keep the sequences that score well under both models.
5. Check their predicted structures and, ideally, measure folding, stability or activity.

ProteinMPNN enforces structural compatibility; ESM-2 adds what it learned from natural sequences.

**Prior work.** The combination itself is not new. Hybrid scores built from ESM-2 pseudo-likelihood
and ProteinMPNN, alongside ESMFold and AlphaFold2, improved folding-stability prediction in
[Stable de novo protein design via joint conformational landscape and sequence optimization](https://www.nature.com/articles/s41467-025-66526-w)
(Nature Communications, 2025).

**What would be new.** Whether the models' agreement or disagreement can select diverse,
structurally compatible sequences without losing desirable biological properties.

---

## 4. What do disagreements reveal about protein function?

Perhaps the most biologically interesting direction. ProteinMPNN learns which amino acids fit a
three-dimensional structure; ESM-2 also learns evolutionary constraints that can reflect
requirements beyond keeping the fold.

Consider an active-site residue in an enzyme. Several amino acids may fit structurally, but only
one supports catalysis. ProteinMPNN could rate several substitutions as plausible while ESM-2
strongly prefers the conserved residue.

**Approach.** Test whether such disagreements are enriched at known functional sites. Compare
disagreement across catalytic residues, protein cores, solvent-exposed surfaces, interfaces and
flexible regions, controlling for amino-acid identity and residue conservation.

**The question.** Can we identify functional constraints that local structural compatibility does
not explain? This connects to experimental structural biology.

**Starting point here.** Active-, binding- and PTM-site labels and B-factors per residue
(`01_dataset.py targets`), solvent accessibility and burial from the structures, and the
amino-acid control from this study.

---

## 5. Does combining both models really improve mutation-effect prediction?

This has clear precedents, which makes it a chance for rigorous benchmarking. SPURS combines
ESM-2 and ProteinMPNN representations through adapter layers to predict stability changes (ΔΔG):
[Generalizable and scalable protein stability prediction with rewired protein generative models](https://www.nature.com/articles/s41467-025-67609-4)
(Nature Communications, 2025; code at [luo-group/SPURS](https://github.com/luo-group/SPURS)). It
shows that fusing the two is feasible and useful in some settings.

**The question worth asking.** When is the combination actually necessary? Does adding structural
information to ESM-2 consistently improve mutation prediction, or mainly for particular residues,
structures or proteins? Test whether the improvement depends on structural confidence, solvent
accessibility, secondary structure or evolutionary conservation.

**What would be new.** A study of the limits of complementarity, not another combined model.

---

## Literature search notes (16 October 2026)

**How the search was done.** Web searches over the protein machine-learning literature (arXiv,
bioRxiv, journals) and the general multimodal-learning literature, reading abstracts and, where
needed, methods. Only papers whose existence was confirmed are cited. This is a targeted search,
not a systematic review: a paper outside these search terms may exist. A first pass by a
delegated search agent returned only one confirmed paper (PLM-eXplain) before its report was
withheld, so the searches were repeated by hand.

**Coverage by idea.**

| Idea | Searched | What already exists | Not found |
|---|---|---|---|
| 1. Disagreement as a mutation-effect signal | Not yet | — | — |
| 2. Fusing amino-acid-centred representations | Yes, in depth (see idea 2) | Sequence–structure fusion; shared/specific separation before fusion; splitting amino-acid identity out of one model's embeddings; linear concept erasure | Removing the two models' shared training target before fusing them |
| 3. ESM-2 re-ranking of ProteinMPNN designs | Precedent confirmed only | Hybrid ESM-2 / ProteinMPNN scores for stability ([Nature Communications 2025](https://www.nature.com/articles/s41467-025-66526-w)) | Not searched further |
| 4. Disagreement at functional sites | Not yet | — | — |
| 5. When combining the two is needed | Precedent confirmed only | ESM-2 + ProteinMPNN fusion for ΔΔG ([SPURS](https://www.nature.com/articles/s41467-025-67609-4)); when AlphaFold2 representations add to ESM-2 ([Chauhan 2026](https://arxiv.org/abs/2606.04228)) | Not searched further |

**Idea 2, area by area.**

- *Fusing a protein language model with a structure model* is established. SPURS adapts
  ProteinMPNN's per-residue representations into ESM-2 for stability change
  ([Nature Communications 2025](https://www.nature.com/articles/s41467-025-67609-4); code at
  [luo-group/SPURS](https://github.com/luo-group/SPURS)). ESM-GearNet compares serial, parallel and
  cross fusion of ESM-2 with structure encoders ([Zhang et al. 2023](https://arxiv.org/abs/2303.06275)).
  S-PLM aligns ESM-2 with a structure encoder by contrastive learning
  ([bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.08.06.552203)). None removes
  information the two share before fusing.
- *Combining embeddings from models with different priors* is common and often helps, but the gain
  depends on the task. Ensembles of ProtT5, ESM-2 and Ankh embeddings improved O-GlcNAc site
  prediction ([PMC10650050](https://pmc.ncbi.nlm.nih.gov/articles/PMC10650050/)); concatenating
  ProstT5 structure-aware and sequence embeddings raised scores only numerically, not significantly
  ([NAR Genomics and Bioinformatics 2024](https://academic.oup.com/nargab/article/6/4/lqae150/7901286)).
- *Asking when structure adds to sequence.* Chauhan measures the linear "information bonus" of
  frozen AlphaFold2 representations over frozen ESM-2: it is positive for allosteric sites and
  negative for binding affinity and flexibility, and residue-level splits inflate flexibility
  scores by 27–39% against protein-level splits ([arXiv 2606.04228](https://arxiv.org/abs/2606.04228)).
  Nothing is removed before the comparison.
- *Separating amino-acid identity from a protein embedding.* PLM-eXplain factors ESM-2
  embeddings into an interpretable subspace (including amino-acid type, secondary structure and
  solvent accessibility) and a residual subspace, and reports that downstream performance is
  maintained ([van Eck et al., Bioinformatics 2026](https://arxiv.org/abs/2504.07156)). This is the
  closest work found; it treats one model and does not fuse.
- *Shared and modality-specific information before fusion*, outside proteins. MISA learns
  modality-invariant and modality-specific subspaces before fusing, for sentiment analysis
  ([Hazarika et al., ACM Multimedia 2020](https://github.com/declare-lab/MISA)). Liang et al.
  estimate redundancy, unique information and synergy between modalities for a task
  ([NeurIPS 2023](https://arxiv.org/abs/2302.12247)).
- *Linear concept erasure.* LEACE removes a concept so that no linear classifier can detect it,
  and shows this holds exactly when every class has the same mean
  ([Belrose et al., NeurIPS 2023](https://arxiv.org/abs/2306.03819)). Subtracting each amino acid's
  mean meets that condition by construction, so the subtraction used in this study is a simple
  linear erasure of amino-acid identity.

**Gap.** No work found removes the training target that a sequence model and a structure model
share before fusing them, or asks whether doing so makes them more complementary.
