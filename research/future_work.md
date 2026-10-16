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

**Proposed design.**

- *Representations:* ESM-2 35M at its best probe layer and the ProteinMPNN encoder's last layer,
  on the 4,898 chains, with the chain-grouped splits and 5 refits of `05_probes.py`.
- *Arms:* each model alone; one-hot amino acid alone; raw fusion; raw fusion plus one-hot (the
  control the hybrid must be compared against, since the two are linearly equivalent); residual
  fusion; hybrid fusion. Size controls: ESM-2 fused with an *untrained* ProteinMPNN (same
  dimensions, no learned structure) and ESM-2 650M alone (more parameters, one model).
- *Predictors:* ridge or logistic regression with cross-validated penalty, gradient-boosted trees
  and a small MLP. The linear predictor doubles as a check: hybrid and raw-plus-one-hot should tie.
- *Training size:* 1%, 10% and 100% of training chains; if residualising helps, it is most likely
  to show at small sizes.
- *Tasks:* the residue targets already built (solvent accessibility, burial, secondary structure,
  B-factor, binding, active and PTM sites); mutation effects on a ProteinGym subset as a second
  phase, since it needs both models run on every substitution.
- *Measures:* score per arm with chain-level intervals; fusion gain over the better single model;
  and the redundancy and unique-information estimates of Liang et al. before and after
  residualising, which measure "more complementary" directly.
- *Decision rule, set before running:* residual or hybrid fusion counts as better if its gain over
  raw-plus-one-hot exceeds both arms' refit intervals on at least three of the seven tasks for the
  same predictor.

Cost: no new extraction for phase 1; CPU probes on existing embeddings, roughly a day.

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
