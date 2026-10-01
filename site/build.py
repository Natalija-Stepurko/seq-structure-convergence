"""Build the results page from the pipeline outputs in results/.

Reads build/data.json (written by build_data.py), the chart helpers in charts.py and two archived
inputs in archive/: the three map images (pngs_opt.json) and the HTML fragments carried over from
the earlier page (fragments.json). Writes build/index.html, the page GitHub Pages serves.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
ARCHIVE = HERE / "archive"
D = json.load(open(BUILD / "data.json"))
FRAGMENTS = json.load(open(ARCHIVE / "fragments.json"))
# The three maps, losslessly re-encoded with PIL (optimize=True, exact palette): ~10% smaller and
# pixel-identical to the renders from results/umap6/coords.npz. Index 0 is Figure B. Pass
# --fresh-figures to use a new render from make_figures.py.
_FIG = BUILD / "figures.json" if "--fresh-figures" in sys.argv else ARCHIVE / "pngs_opt.json"
_OPT = json.load(open(_FIG))
FIGB, PNG = _OPT[0], [None, _OPT[1], _OPT[2]]

sys.path.insert(0, str(HERE))
from charts import (depth_bars, ladder_svg, null_strips, width_bars, layer_panels,
                    stitch_bars, depth_verdicts, depth_bars_partial, LADDER_ORDER, METRIC_LABEL)

REPO = "https://github.com/Natalija-Stepurko/seq-structure-convergence"
DATASET = "https://huggingface.co/datasets/NatalijaStepurko/seq-structure-convergence"
EP = D["every_pair"]
MAIN = "ESM-2 35M x ProteinMPNN"
DP = D["depth_partial"]
DPTOP = max(DP, key=lambda r: r["partial"])
ST = D["stitching"]
# how many of the shallowest entry depths lose in every case
LEAD = next((i for i, r in enumerate(ST["by_depth"])
             if r["worse"] < r["worse"] + r["no clear difference"] + r["better"]), len(ST["by_depth"]))
share = lambda k: 100 * D["calibration"][k]["null"] / D["calibration"][k]["obs"]


def block(key, tag, stop=None):
    """One inner block of the earlier page, as archived in archive/fragments.json."""
    return FRAGMENTS["|".join([key, tag, stop or ""])]


def v(pair, mode, metric="cka"):
    return EP[pair][mode][metric]["v"]


# ════════════════════════════════════════════════════════════════════ page CSS
CSS_EXTRA = """
/* charts */
text.ct{font-family:var(--mono);font-size:12.5px;font-weight:600;fill:var(--ink)}
text.ctsm{font-family:var(--mono);font-size:11px;font-weight:600;fill:var(--ink)}
text.cs{font-family:var(--mono);font-size:9px;fill:var(--ink-3)}
text.cax{font-family:var(--mono);font-size:9px;fill:var(--ink-3)}
text.clab{font-family:var(--serif);font-size:12px;fill:var(--ink)}
text.clabsm{font-family:var(--serif);font-size:11px;fill:var(--ink-2)}
text.ctag{font-family:var(--mono);font-size:8.5px;fill:var(--ink-3)}
text.cval{font-family:var(--mono);font-size:11px;fill:var(--ink);font-variant-numeric:tabular-nums}
text.cbad{font-family:var(--mono);font-size:9.5px;fill:var(--str);font-weight:600}
text.cgood{font-family:var(--mono);font-size:10px;fill:var(--chrome);font-weight:600}
.figwrap{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:18px 20px;
         overflow-x:auto}
.figwrap svg{width:100%;height:auto;display:block;min-width:520px}
.figcap{font-size:.88rem;color:var(--ink-2);line-height:1.5;margin-top:10px}
.minirow{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
.minipanel{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:14px;
           overflow-x:auto}
.minipanel svg{width:100%;height:auto;display:block;min-width:210px}
.minipanel.wide svg{min-width:320px}
/* result box */
/* exactly three columns: auto-fit fitted a fourth, empty cell at desktop width */
.resultbox{display:grid;grid-template-columns:repeat(3,1fr);gap:0;
           background:var(--panel);border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.rb{padding:20px 22px;border-right:1px solid var(--rule)}
.rb:last-child{border-right:0}
.rb h3{font-size:1.06rem;font-weight:600;margin:0 0 8px}
.rb .rbn{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:1.5rem;
         font-weight:500;letter-spacing:-.02em;display:block;margin-bottom:6px}
.rb p{font-size:.86rem;color:var(--ink-2);line-height:1.5}
.rbfoot{grid-column:1/-1;border-top:1px solid var(--rule);padding:14px 22px;font-size:.86rem;
        color:var(--ink-2);line-height:1.5;background:var(--ground)}
/* what is new */
.whatsnew{display:flex;flex-direction:column;gap:10px;background:var(--panel);
          border:1px solid var(--rule);border-left:3px solid var(--chrome);
          border-radius:0 3px 3px 0;padding:18px 22px}
.whatsnew ol{margin:0;padding-left:1.25em;display:flex;flex-direction:column;gap:8px}
.whatsnew li{font-size:.92rem;color:var(--ink-2);line-height:1.5}
/* contents */
.toc{display:flex;flex-wrap:wrap;gap:6px 14px;font-family:var(--mono);font-size:.68rem;
     padding:14px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.toc a{color:var(--ink-3);text-decoration:none;white-space:nowrap}
.toc a:hover,.toc a:focus-visible{color:var(--chrome)}
/* definitions */
.defs{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
.def{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:16px 18px}
.def b{display:block;font-family:var(--mono);font-size:.72rem;letter-spacing:.06em;
       text-transform:uppercase;color:var(--chrome);margin-bottom:6px}
.def span{font-size:.88rem;color:var(--ink-2);line-height:1.5}
/* reference model list */
.refmodels{display:grid;gap:10px;background:var(--panel);border:1px solid var(--rule);
           border-radius:3px;padding:18px 22px}
.refmodels div{display:grid;grid-template-columns:minmax(8rem,10rem) 1fr;gap:6px 16px;
               font-size:.9rem;color:var(--ink-2);line-height:1.5}
.refmodels dt{font-family:var(--mono);font-size:.8rem;color:var(--ink);font-weight:600}
@media(max-width:620px){.refmodels div{grid-template-columns:1fr;gap:2px}}
/* details */
details{background:var(--panel);border:1px solid var(--rule);border-radius:3px}
details[open]{padding-bottom:16px}
summary{cursor:pointer;padding:14px 18px;font-family:var(--mono);font-size:.76rem;
        letter-spacing:.04em;color:var(--ink-2);list-style:none;display:flex;gap:9px;
        align-items:center}
summary::-webkit-details-marker{display:none}
summary::before{content:"+";font-size:.95rem;color:var(--chrome);font-weight:600}
details[open] summary::before{content:"−"}
summary:hover{color:var(--chrome)}
details > *:not(summary){margin:0 18px}
/* controls */
.ctrls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding-bottom:14px}
.ctrlgrp{display:flex;align-items:center;gap:8px}
.ctrlgrp > b{font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;text-transform:uppercase;
             color:var(--ink-3);font-weight:600}
.seg{display:flex;border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.seg button{font-family:var(--mono);font-size:.7rem;padding:5px 10px;border:0;cursor:pointer;
            background:var(--panel);color:var(--ink-2);border-right:1px solid var(--rule)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"]{background:var(--chrome);color:#fff}
.seg button:hover:not([aria-pressed="true"]){background:var(--rule-2)}
/* references */
.refs{display:flex;flex-direction:column;gap:7px;font-size:.84rem;color:var(--ink-2);
      line-height:1.5}
.refs p{text-indent:-1.3em;padding-left:1.3em}
.callouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
.callout{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:16px 18px}
.callout b{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:1.15rem;
           display:block;margin-bottom:5px}
.callout span{font-size:.86rem;color:var(--ink-2);line-height:1.45}
@media(max-width:700px){.resultbox{grid-template-columns:1fr}.rb{border-right:0;border-bottom:1px solid var(--rule)}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


# ════════════════════════════════════════════════════════════════════ sections
def header():
    ds = D["dataset"]
    return f"""<header class="page" id="top">
  <div class="kicker">Two protein models, compared layer by layer ·
    {ds['domains']:,} protein domains · {ds['residues']:,} residues</div>
  <h1>Do two models given<br>opposite halves of a protein<br>learn the same biology?</h1>
  <p class="lede">A protein is a chain of amino acids that folds into a shape, and the shape is what
  lets it work. This page compares two machine-learning models that were each shown only one half
  of that: <b>ESM-2</b> reads the chain of letters and never sees a coordinate; <b>ProteinMPNN</b>
  reads the 3-D backbone and never sees a letter. If sequence and shape are two views of one
  biology, the two should end up describing proteins the same way. They do only partly — and about
  half of what standard measurements report as agreement between them turns out to be the one
  thing both were trained to output: the amino acid at each position.</p>

  <div class="resultbox">
    <div class="rb"><h3>Some shared directions, not many</h3>
      <span class="rbn">{v(MAIN,'partial'):.3f}</span>
      <p>Each model turns every residue into a list of numbers. Lay out the same
      {ds['residues']/1e6:.1f} million residues both ways and the two layouts share some of the same
      axes. On this measure two copies of one model score
      {v('ESM-1v s1 x ESM-1v s2','partial'):.3f} and an untrained network scores
      {v('untrained seq x trained str','partial'):.3f}; ESM-2 against ProteinMPNN scores
      {v(MAIN,'partial'):.3f}.</p></div>
    <div class="rb"><h3>Not the same map</h3>
      <span class="rbn">{v(MAIN,'partial','mutual_knn'):.3f}</span>
      <p>Ask each model which ten residues are most like a given one, and the two lists almost
      never overlap: {v(MAIN,'partial','mutual_knn'):.3f} of neighbours in common, against
      {v('ESM-1v s1 x ESM-1v s2','partial','mutual_knn'):.3f} for two copies of one model. The two
      arrange proteins by different principles — ESM-2 by chemical identity, ProteinMPNN by physical
      environment.</p></div>
    <div class="rb"><h3>Not interchangeable</h3>
      <span class="rbn">{ST['worse']} / {ST['total']}</span>
      <p>Hand ProteinMPNN's description of a residue to ESM-2 and let ESM-2's later layers finish
      the job. The result is clearly worse than ProteinMPNN alone in {ST['worse']} of {ST['total']}
      tests, including every test that enters ESM-2's first {LEAD} layers. Untrained layers pass the
      description through unchanged; trained ones lose part of it.</p></div>
    <div class="rbfoot"><b>Why the raw scores overstate it.</b> Both models are trained to output the
    amino acid at each position, so they agree that position 47 is a leucine before they agree on
    anything else. That shared answer accounts for about half of the strongest raw agreement score,
    and one common measure returns {share('svcca'):.0f}% of its value even on scrambled data. The
    agreement scores above are read with the shared answer subtracted.</div>
  </div>

  <div class="whatsnew">
    <span class="eyebrow">What is new</span>
    <ol>
      <li>Both models are trained to guess the amino acid at each position, so they agree on that
      before they agree on anything else. This page finds where in the network that shared answer
      sits — at ESM-2's very first layer — subtracts it, and checks the subtraction against two
      references: two copies of one model, whose agreement is certainly real and must survive, and
      an untrained network, whose agreement cannot be real and must vanish.</li>
      <li>Every score is read against a ladder of reference points — the same model trained twice,
      two models that share an input, untrained networks, and each measure's own score on scrambled
      data. One standard measure turns out to be mostly its scrambled score, and to grow with the
      size of the model.</li>
      <li>Agreement is tested by use as well as by geometry: ProteinMPNN's description is fed into
      ESM-2 at every layer and scored against ProteinMPNN alone.</li>
    </ol>
  </div>


  <a class="repo" href="{REPO}">
    <svg viewBox="0 0 16 16" aria-hidden="true" width="17" height="17"><path fill="currentColor"
      d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
      0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
      1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
      0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0
      1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87
      3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16
      8c0-4.42-3.58-8-8-8Z"/></svg>
    <span><b>All the code is public.</b> Every number on this page is produced by the pipeline in
    this repository — extraction, the controls, and the figures.</span>
    <span class="repo-path">Natalija-Stepurko/seq-structure-convergence</span>
  </a>
  <a class="repo" href="{DATASET}">
    <svg viewBox="0 0 16 16" aria-hidden="true" width="17" height="17"><g fill="none"
      stroke="currentColor" stroke-width="1.4"><ellipse cx="8" cy="3.5" rx="5.5" ry="2"/>
      <path d="M2.5 3.5v9c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2v-9"/>
      <path d="M2.5 8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2"/></g></svg>
    <span><b>The embeddings are public too.</b> All nine models at every layer: {D['dataset_hf']['residues']:,}
    residues from {D['dataset_hf']['chains']} proteins aligned row for row, and one averaged vector for
    each of {D['dataset_hf']['proteins']:,} proteins, with labels and reference scores. A test bed for
    measures of representation similarity.</span>
    <span class="repo-path">Hugging Face dataset</span>
  </a>
</header>"""


def s_question():
    return f"""<div class="page sec" id="question">
  <div class="stack prose">
    <span class="eyebrow">The question</span>
    <h2>If sequence and shape are two views of one biology, two models should agree</h2>
    <p class="prose">Everything a protein does is supposed to follow from its shape, and its shape
    from its sequence. That chain of reasoning is the foundation of structural biology, and it
    implies something testable about machine models: a model that has only ever read sequences and a
    model that has only ever read coordinates are looking at two views of the same object. Given
    enough of each, they should arrive at the same internal picture of proteins. This is the protein
    version of a broader claim — that large models trained on different data converge on a shared
    representation of the world (Huh et al., 2024).</p>
    <p class="prose">It matters practically. It is the assumption behind using one model's output in
    place of the other's, behind combining them, and behind treating either as a general-purpose
    description of a protein. If it holds, the two are interchangeable in ways that would save a
    great deal of work. If it holds only partly, it matters exactly where the limit falls.</p>
  </div>
  <div class="defs">
    <div class="def"><b>Agreement</b><span>A score between two models' descriptions of the same
      residues, from one of three measures. A number, not a claim.</span></div>
    <div class="def"><b>Convergence</b><span>The claim that agreement is high for a reason other
      than shared inputs or a shared training target. What this page is testing.</span></div>
    <div class="def"><b>The answer key</b><span>The amino-acid identity at each position, which both
      model families are trained to output. The artefact this page removes.</span></div>
  </div>
</div>"""


def s_models():
    ds = D["dataset"]
    return f"""<div class="page sec" id="models">
  <div class="stack prose">
    <span class="eyebrow">What was compared</span>
    <h2>One pair is the experiment; the rest set the scale</h2>
    <p class="prose">Everything detailed below — the maps, the layer grids, the stitching, the
    property tests — is ESM-2 35M (sequence) against the ProteinMPNN encoder (structure). They are
    matched in the way that matters here: both are small, both produce one description per residue
    for the same residues, and they read opposite halves of a protein.</p>
  </div>
  {block("Two models carry the argument", '<div class="models">', '<div class="scroll">')}
  <div class="refmodels">
    <div><dt>CARP-38M</dt><dd>sequence: a sequence model of a completely different design — stacked
      convolutions, not attention — to test whether anything depends on the design.</dd></div>
    <div><dt>ESM-IF1</dt><dd>structure: a second, larger structure model.</dd></div>
    <div><dt>ESM-2 650M</dt><dd>sequence: the same model 18× larger, used only to ask whether
      agreement grows with size.</dd></div>
    <div><dt>ESM-1v, two copies</dt><dd>sequence: the ceiling — the one protein language model
      released as several independently seeded checkpoints, so two copies differ only by their
      random start, which is the one honest answer to “how alike can two models even be?”</dd></div>
    <div><dt>Untrained copies</dt><dd>of both main models: same wiring, no learning — the floor.</dd></div>
  </div>
  <p class="prose">All comparisons use one fixed set of {ds['domains']:,} non-redundant protein
  domains with experimentally solved structures (CATH S35), {ds['residues']:,} residues in total.
  Both models describe every one of those residues, so each comparison is residue-for-residue. Every
  agreement score on this page is an estimate over {ds['subsamples']} independent subsamples of
  {ds['budget']:,} residues, drawn whole chain at a time, with a 95% interval, because residues
  within a chain are not independent; probe scores carry the spread across {ds['refits']} independent
  refits instead.</p>
  <details><summary>The reference models as a table</summary>
    {block("Two models carry the argument", '<div class="scroll">').replace("depends on the machinery", "depends on the design")}
  </details>
</div>"""


def s_measures():
    c = D["calibration"]
    return f"""<div class="page sec" id="measures">
  <div class="stack prose">
    <span class="eyebrow">How agreement was scored</span>
    <h2>Three ways to ask “do they agree?” — and one of them mostly answers on its own</h2>
    <p class="prose">Two taxonomists can use the same traits to organise species and still shelve
    them differently. These three measures separate exactly that. CKA is centred kernel alignment
    (Kornblith et al., 2019); SVCCA is singular-vector canonical correlation analysis (Raghu et al.,
    2017); mutual k-NN compares k nearest neighbours, as used by Huh et al. (2024).</p>
  </div>
  {block("Three ways of asking", '<div class="mxs">')}
  <div class="stack prose">
    <h3 class="subhead">Calibrating each measure on scrambled residues</h3>
    <p class="prose">No similarity score means anything on its own. Scramble which residue is which
    and every correspondence between the two models is destroyed; whatever a measure still reports
    is what it returns for unrelated data. For the main pair: same pattern of resemblance reports
    {c['cka']['obs']:.3f}, of which {share('cka'):.0f}% is null; same main directions reports
    {c['svcca']['obs']:.3f}, of which {share('svcca'):.0f}% is null; same neighbours reports
    {c['knn']['obs']:.3f}, of which {share('knn'):.0f}% is null. Two
    high-dimensional clouds of numbers always share some direction, whether or not they are related,
    so most of the headline “same main directions” score is noise floor. The other two measures are
    nearly clean.</p>
  </div>
  <div class="figwrap">{null_strips(D)}
    <p class="figcap">Each bar is the mean over the same {c['cka']['subsamples']} whole-chain
    subsamples of {c['cka']['n']:,} residues used for every pair further down; the grey part is the
    same subsamples with the residues scrambled.</p>
  </div>
  <div class="stack prose">
    <p class="prose">It gets worse when models of different sizes are compared, because the
    scrambled score grows with the number of dimensions. The bigger model looks far more convergent
    — {D['width'][2]['obs']:.3f} against {D['width'][1]['obs']:.3f} — and is not: read against its
    own scrambled score it is identical to the small one. Scores like this cannot be compared
    between models of different widths without a null. This is the same width effect that Gröger,
    Wen &amp; Brbić (2026) report for vision–language models, found here independently in
    proteins.</p>
  </div>
  <div class="figwrap">{width_bars(D)}
    <p class="figcap">From a separate run on a different residue sample, which is why the main pair
    reads {D['width'][1]['obs']:.3f} here and {c['svcca']['obs']:.3f} above; each score is read only
    against its own null. Gaps are computed before rounding: the displayed
    {D['width'][1]['obs']:.3f} − {D['width'][1]['null']:.3f} gives
    +{D['width'][1]['obs']-D['width'][1]['null']:.3f}, not the +0.117 a reader would get from the
    rounded figures.</p>
  </div>
</div>"""


def s_f1():
    a = D["aa_lookup"]
    dp = {r["layer"]: r["v"] for r in D["depth_cka"]}
    return f"""<div class="page sec" id="f1">
  <div class="stack prose">
    <span class="eyebrow">The answer key</span>
    <h2>The strongest agreement between the two families was the amino-acid code both are trained
    to output</h2>
    <p class="prose">Both families are trained to output the amino acid at each position — the
    sequence model by filling in masked residues, the structure model by saying which amino acids
    would fold into a given backbone. So two models can look like they agree simply by both knowing
    that position 47 holds a leucine, without sharing any understanding of proteins at all.</p>
    <p class="prose">This is not hypothetical. The single strongest agreement measured between a
    sequence model and a structure model sits at the sequence model's very first layer — the step
    that looks each letter up in a table, before any learned processing has happened. Against a bare
    label naming the amino acid, that layer scores {a['esm_svcca']:.3f} on same main directions and
    {a['esm_knn']:.3f} on same neighbours: at that depth the representation is nothing but the
    amino-acid code, so agreement measured there is the shared training target, not shared biology.
    Deeper in the model, where the learned biology should be, raw agreement with the structure model
    falls — from {dp['emb']:.3f} at the first layer to {dp['b7']:.3f} in the middle — and only rises
    again toward the output, {dp['b12']:.3f} at the last layer, where the model returns to predicting
    amino acids. The layers doing real work agree with the structure model less than the raw letter
    code does.</p>
  </div>
  <div class="figwrap">{depth_bars(D)}
    <p class="figcap">Each bar is one layer of the sequence model, scored against the structure
    model's last encoder layer. Teal marks the lookup layer, orange the middle minimum.</p>
  </div>
  <figure class="bigfig">
    <img src="{FIGB}" alt="Residue maps for the sequence and structure models, coloured by amino
      acid, local shape, burial and solvent accessibility. The sequence model splits into about
      twenty islands; the structure model is one connected cloud."/>
    <figcaption><b>The problem, drawn.</b> Every dot is one residue, placed near the residues each
    model describes similarly — the same 8,000 residues in both rows. The sequence model breaks
    residue space into about twenty islands, one per amino acid; colour the islands by anything
    structural and the colours are mixed within each island. That information is there, but it is
    not what sets the geography. The structure model, which never sees a letter, does the opposite:
    amino acid is scattered noise, and buried-versus-exposed splits the map almost in half. The two
    models organise protein space by different principles — one by chemical identity, one by
    physical environment.</figcaption>
  </figure>
  <div class="stack prose">
    <h3 class="subhead">Removing the answer key</h3>
    <p class="prose">Each model turns every residue into a list of numbers. To remove the shared
    answer key, take every leucine in the dataset, average its description, and subtract that average
    from each individual leucine — then repeat for all twenty amino acids. What is left is how
    <em>this</em> leucine differs from a typical leucine: its surroundings, its role. If a model knew
    only the letter, subtracting would leave exactly zero and its apparent agreement would vanish
    entirely; that is the intended behaviour. If a model knows real biology — where the residue sits,
    what surrounds it, whether it is buried, what it does — all of that survives untouched.</p>
  </div>
  {block("What we subtract", '<div class="subfigs">', '<div class="steps">')}
  <div class="stack prose">
    <p class="prose">Two checks show the subtraction does what it should. The same model trained
    twice — agreement that is certainly real — barely moves:
    {v('ESM-1v s1 x ESM-1v s2','raw'):.3f} → {v('ESM-1v s1 x ESM-1v s2','partial'):.3f}. An untrained
    sequence network against a trained structure model — agreement that cannot be about biology,
    because one side has learned nothing — scores a respectable-looking
    {v('untrained seq x trained str','raw'):.3f} as measured and collapses to
    {v('untrained seq x trained str','partial'):.3f} once the answer key is gone. Real agreement is
    left standing; manufactured agreement is removed. Every result below is reported both ways: as
    measured, and with the answer key removed.</p>
  </div>
  <div class="callouts">
    <div class="callout"><b>{v('ESM-1v s1 x ESM-1v s2','raw'):.3f} → {v('ESM-1v s1 x ESM-1v s2','partial'):.3f}</b>
      <span>the same model trained twice — unchanged</span></div>
    <div class="callout"><b>{v('untrained seq x trained str','raw'):.3f} → {v('untrained seq x trained str','partial'):.3f}</b>
      <span>untrained sequence against trained structure — collapses</span></div>
    <div class="callout"><b>{v(MAIN,'raw'):.3f} → {v(MAIN,'partial'):.3f}</b>
      <span>the main pair — about half of the strongest measure was the shared answer key</span></div>
  </div>
  <div class="stack prose">
    <p class="prose">Layer by layer, the subtraction redraws the first chart. ESM-2's lookup layer
    drops from {DP[0]['raw']:.3f} to nothing: it held only the letter code. The early blocks keep a
    small part of their agreement ({DP[1]['raw']:.3f} → {DP[1]['partial']:.3f} at block 1), and what
    remains grows with depth, highest at {DPTOP['label'].replace('b', 'block ')}
    ({DPTOP['raw']:.3f} → {DPTOP['partial']:.3f}). Before the subtraction the input layer led; after
    it, the agreement that is left sits where ESM-2 has done the most processing.</p>
  </div>
  <div class="figwrap">{depth_bars_partial(D)}
    <p class="figcap">Solid bars: each layer of ESM-2 against ProteinMPNN's last encoder layer, with
    every residue's amino-acid average subtracted from both; lines are 95% intervals. Outlines: the
    same layers as measured. Both on the same {D['depth_partial_basis']['subsamples']} whole-chain
    subsamples of {D['depth_partial_basis']['budget']:,} residues. The first chart in this section
    is a single run on 30,000 residues, which is why its values differ slightly.</p>
  </div>
</div>"""


def ladder_details():
    head = ("<tr><th>Pair</th>"
            + "".join(f'<th>{n}<br><span class="dim">as measured</span></th>'
                      f'<th>↳ answer key removed</th>' for n in
                      ("Same pattern", "Same directions", "Same neighbours"))
            + "</tr>")
    rows = []
    for k, lab, tag in LADDER_ORDER:
        if k not in EP:
            continue
        tds = []
        for m in ("cka", "svcca", "mutual_knn"):
            r, p = EP[k]["raw"][m], EP[k]["partial"][m]
            tds.append(f'<td>{r["v"]:.3f}<small>{r["lo"]:.3f}–{r["hi"]:.3f}</small></td>'
                       f'<td>{p["v"]:.3f}<small>{p["lo"]:.3f}–{p["hi"]:.3f}</small></td>')
        cls = "hl" if tag == "ceiling" else ("ctrl" if tag == "floor" else "")
        rows.append(f'<tr class="{cls}"><td>{lab} <span class="tag">{tag}</span></td>'
                    + "".join(tds) + "</tr>")
    return (f'<div class="scroll"><table class="ci"><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def s_f2():
    return f"""<div class="page sec" id="f2">
  <div class="stack prose">
    <span class="eyebrow">What survives · every pair, before and after</span>
    <h2>With the answer key removed, every sequence–structure pair still sits well above the floor —
    and well below two models that share an input</h2>
    <p class="prose">A score of 0.13 means nothing on its own until you know what the same
    measurement gives for two models that certainly agree and for two that certainly do not. So every
    pair is read against a ladder. At the top, the same model trained twice:
    {v('ESM-1v s1 x ESM-1v s2','partial'):.3f}, unchanged by the subtraction. Next, two models
    sharing an input: two sequence models (CARP-38M × ESM-2 35M) {v('CARP x ESM-2','partial'):.3f}
    and two structure models (ESM-IF1 × ProteinMPNN) {v('ESM-IF1 x ProteinMPNN','partial'):.3f} after
    the subtraction ({v('CARP x ESM-2','raw'):.3f} and {v('ESM-IF1 x ProteinMPNN','raw'):.3f} as
    measured). At the bottom, the floors: an untrained sequence network against a trained structure
    model {v('untrained seq x trained str','partial'):.3f}, the reverse
    {v('trained seq x untrained str','partial'):.3f}, two untrained networks
    {v('untrained x untrained','partial'):.3f}.</p>
    <p class="prose">Every sequence-against-structure pair lands between those rungs. The main pair
    goes from {v(MAIN,'raw'):.3f} as measured to {v(MAIN,'partial'):.3f} with the answer key removed;
    the other five cross-modal pairs land between {v('ESM-1v x ProteinMPNN','partial'):.3f} and
    {v('ESM-2 650M x ESM-IF1','partial'):.3f} after removal, the largest pair highest. All six sit
    clearly above every floor and clearly below the same-input rungs. Same main directions barely
    moves for any pair ({v(MAIN,'raw','svcca'):.3f} → {v(MAIN,'partial','svcca'):.3f} for the main
    pair) — as the calibration showed, most of that score was noise floor to begin with, so there was
    little answer key in it to remove.</p>
  </div>
  <div class="figwrap" id="ladderwrap">
    <div class="ctrls">
      <div class="ctrlgrp"><b>Measure</b><div class="seg" id="seg-measure" role="group"
        aria-label="Choose measure">
        <button type="button" data-v="cka" aria-pressed="true">CKA</button>
        <button type="button" data-v="svcca" aria-pressed="false">SVCCA</button>
        <button type="button" data-v="mutual_knn" aria-pressed="false">mutual k-NN</button>
      </div></div>
      <div class="ctrlgrp"><b>Show</b><div class="seg" id="seg-show" role="group"
        aria-label="Choose state">
        <button type="button" data-v="both" aria-pressed="true">both</button>
        <button type="button" data-v="as measured" aria-pressed="false">as measured</button>
        <button type="button" data-v="answer key removed" aria-pressed="false">key removed</button>
      </div></div>
    </div>
    <div id="ladderhost">{ladder_svg(D, 'cka', 'both')}</div>
    <p class="figcap" id="laddercap">ESM-2 35M × ProteinMPNN: {v(MAIN,'raw'):.3f} as measured,
    {v(MAIN,'partial'):.3f} with the answer key removed.</p>
  </div>
  <details><summary>Every pair, every measure, with intervals</summary>{ladder_details()}</details>
</div>"""


def s_f3():
    mx = max(v(k, "partial", "mutual_knn") for k, _, t in LADDER_ORDER
             if t == "opposite inputs" and k in EP)
    return f"""<div class="page sec" id="f3">
  <div class="stack prose">
    <span class="eyebrow">No shared map · neighbours</span>
    <h2>They do not arrange proteins the same way</h2>
    <p class="prose">Sharing axes is weaker than sharing a map. The neighbour measure asks, for each
    residue, how many of its ten closest neighbours in one model are also among its ten closest in
    the other. Two copies of the same model agree on
    {v('ESM-1v s1 x ESM-1v s2','partial','mutual_knn'):.3f} of each other's neighbours after the
    subtraction ({v('ESM-1v s1 x ESM-1v s2','raw','mutual_knn'):.3f} as measured); two sequence
    models on {v('CARP x ESM-2','partial','mutual_knn'):.3f}; two structure models on
    {v('ESM-IF1 x ProteinMPNN','partial','mutual_knn'):.3f}. No sequence–structure pair exceeds
    {mx:.2f}, and the main pair scores {v(MAIN,'partial','mutual_knn'):.3f}. The two families line up
    along some of the same directions, but they do not put the same residues next to each other —
    they are not two drawings of one map.</p>
    <p class="prose">Two sequence models are the one pair whose neighbour agreement <em>rises</em>
    after the subtraction ({v('CARP x ESM-2','raw','mutual_knn'):.3f} →
    {v('CARP x ESM-2','partial','mutual_knn'):.3f}): removing the twenty islands stops amino-acid identity from
    dominating the neighbour lists, so neighbours are compared within an amino acid. Cross-modal pairs do not move
    ({v(MAIN,'raw','mutual_knn'):.3f} → {v(MAIN,'partial','mutual_knn'):.3f}), which supports the
    reading that the subtraction removes the islands and leaves the biology intact.</p>
  </div>
  <div class="figwrap">{ladder_svg(D, 'mutual_knn', 'both')}
    <p class="figcap">The same ladder as in “What survives”, on the neighbour measure. The ceiling is far to the
    right; every sequence–structure pair is crowded against the axis.</p>
  </div>
</div>"""


def s_f4():
    dp = {r["layer"]: r["v"] for r in D["depth_cka"]}
    sv = [r["v"] for r in D["depth_svcca"]]
    kn = [r["v"] for r in D["depth_knn"]]
    return f"""<div class="page sec" id="f4">
  <div class="stack prose">
    <span class="eyebrow">Layer by layer</span>
    <h2>Agreement is highest where the sequence model has learned least</h2>
    <p class="prose">The same pair of models, every layer of one against every layer of the other, as
    measured. Read along the sequence model's depth: same pattern of resemblance is highest at the
    embedding layer ({dp['emb']:.3f} against the structure model's last encoder layer), falls through
    the middle blocks to {dp['b7']:.3f} at block 7, climbs back to {dp['b11']:.3f} at block 11 and
    dips to {dp['b12']:.3f} at the output. Same main directions is nearly flat
    ({min(sv):.2f}–{max(sv):.2f}) and mostly null. Same neighbours never leaves the floor
    ({min(kn):.3f}–{max(kn):.3f}). Read along the structure model's depth: its last encoder layer is
    the most similar to every sequence layer, on every measure. The three panels disagree because the
    measures do.</p>
  </div>
  <div class="minirow">{layer_panels(D)}</div>
  <p class="figcap">All three panels are <b>as measured</b> — the answer key has not been removed
  here, which is why the embedding layer leads.</p>
  <details><summary>All 39 layer pairs, all three measures</summary>
    {block("Layer by layer", '<div class="heats">')}
  </details>
</div>"""


def s_f5():
    st, ctl, mlp = D["stitching"], D["stitching_control"], D["stitching_mlp"]
    p = {x["key"]: x for x in st["props"]}
    return f"""<div class="page sec" id="f5">
  <div class="stack prose">
    <span class="eyebrow">Not interchangeable · stitching</span>
    <h2>The structure model's description does not survive the sequence model's layers intact</h2>
    <p class="prose">Similar is not the same as interchangeable. Stitching — taking one model's
    description of a residue and feeding it into the other model's remaining layers — tests the
    stronger claim directly: take the structure model's description of each residue, pass it through
    a single trained translation step into the sequence model at some layer, and let the sequence
    model's own remaining layers finish the job. Every entry point was tried — {st['cells']}
    combinations of structure-model layer and entry point — and each stitched model was scored on
    three properties: {st['total']} cases, each repeated on {st['repeats']} independent samples of
    proteins. The comparison that settles it is against the structure model <em>on its own</em>:
    scoring against the sequence model would flatter stitching, because the structure model is simply
    better at these properties to begin with (Lenc &amp; Vedaldi, 2015; Bansal et al., 2021).</p>
    <p class="prose">Passing the structure model's work through the sequence model's trained layers
    loses part of it. In {st['worse']} of {st['total']} cases the stitched model is clearly worse than
    the structure model alone, a difference that holds up across all {st['repeats']} repeats; in {st['unclear']}
    there is no clear difference, and in {st['better']} it is better. The losses are not spread
    evenly: every case that enters in the first {LEAD} layers is worse, and the cases with no clear
    difference sit at the deep entry points, where few of the sequence model's layers are left to
    act.</p>
    <p class="prose">What comes out is in between. Untrained layers pass the information through
    unchanged ({p['ss3']['donor']:.3f} → {p['ss3']['rand']:.3f} for local shape,
    {p['burial']['donor']:.3f} → {p['burial']['rand']:.3f} for buried or exposed,
    {p['rsa']['donor']:.3f} → {p['rsa']['rand']:.3f} for solvent accessibility). Trained layers keep
    more than the sequence model knows on its own and less than the structure model supplied: for
    solvent accessibility, {p['rsa']['trained']:.3f}, against {p['rsa']['native']:.3f} for the
    sequence model alone and {p['rsa']['donor']:.3f} for the structure model alone. The layers are
    tuned to inputs a structure model never produces.</p>
    <p class="prose">Two checks bound the result. The translation step is a single linear map, and on
    held-out proteins it captures only {st['r2_min']:.2f}–{st['r2_max']:.2f} of the variance (mean
    {st['r2_mean']:.3f}). A small neural network in its place fits slightly better (mean
    {mlp['r2_mean']:.3f}) and changes nothing: {mlp['worse']} of {mlp['total']} cases are still
    clearly worse. And the same test can pass. Fed into ESM-2 from CARP, a second sequence model, the
    linear map fits about {ctl['r2_mean'] / st['r2_mean']:.0f} times better (mean
    {ctl['r2_mean']:.3f}) and {ctl['unclear']} of {ctl['total']} cases show no clear loss. CARP starts
    at roughly ESM-2's level on these properties, which makes a no-loss result easier to reach; the
    structure model carries information the sequence model lacks, and that is what is lost. Folding
    trunks that share both input and target have likewise been shown to be functionally
    interchangeable (Lu et al., 2026).</p>
  </div>
  <div class="minirow">{stitch_bars(D)}</div>
  <p class="figcap">Each panel averages the {st['cells']} combinations of structure-model layer and
  entry point, over {st['repeats']} repeats.</p>
  <div class="figwrap">{depth_verdicts(D)}</div>
</div>"""


def s_f6():
    pc, ph = D["probe_chain"], D["physchem"]
    e, m = pc["esm"], pc["proteinmpnn"]
    return f"""<div class="page sec" id="f6">
  <div class="stack prose">
    <span class="eyebrow">Who knows what · property probes</span>
    <h2>Each model reads best what it was shown — except fold topology</h2>
    <p class="prose">Both models were tested on properties neither was trained to predict, with a
    probe on each layer and the best layer reported. The split is clean, and it runs along the
    boundary between geometry and biology. Everything geometric — solvent accessibility, buried or
    exposed, local shape, flexibility (B-factor) — is read better from the structure model: 0.816
    against 0.420 for solvent accessibility (R²), 0.892 against 0.761 for buried or exposed, 0.887
    against 0.786 for local shape, 0.276 against 0.180 for flexibility (R²). Everything functional
    and evolutionary is read better from the sequence model: enzyme class {e['ec_class']['v']:.3f}
    against {m['ec_class']['v']:.3f}, protein class {e['protein_class']['v']:.3f} against
    {m['protein_class']['v']:.3f}, kingdom {e['kingdom']['v']:.3f} against {m['kingdom']['v']:.3f},
    each with the best amino-acid-counting baseline well below the sequence model
    ({ph['ec_class']:.3f}, {ph['protein_class']:.3f}, {ph['kingdom']:.3f}).</p>
    <p class="prose">The exception is fold topology. It is a shape category — and the model that has
    never seen a shape predicts it better than the model that has seen nothing else:
    {e['cath_topol']['v']:.3f} against {m['cath_topol']['v']:.3f}, with a counting baseline of
    {ph['cath_topol']:.3f}. Evolutionary constraint, visible in sequence, carries information about
    fold at least as directly as this structure model's view of geometry does.</p>
    <p class="prose">Scores are macro-averaged F1 unless marked R²; intervals are the spread across
    five independent refits. The sequence model has 13 layers to pick its best from and the structure
    model 3, which favours sequence — the gaps are far larger than that can explain, but the absolute
    numbers are optimistic for both. Six further properties were tested and are excluded from these
    claims because their intervals overlap or their baselines do nothing but guess the commonest
    answer; they are shown in the full battery below.</p>
  </div>
  {block("Which model wins, and by how much", '<div class="dv-wrap">')}
  <p class="figcap">Which model reads each property better, and by how much (sequence minus
  structure). Bars left of centre: the structure model reads it better.</p>
  <details><summary>The headline tables with intervals</summary>
    {block("Each model wins at what it was shown", '<div class="twocol">')}
  </details>
  <details id="battery-details"><summary>All twenty-five properties, against an amino-acid-counting baseline</summary>
    <p class="figcap">The grey tick is what you score by counting amino acids alone; a model sitting
    on the tick has learned nothing beyond composition. The six properties excluded above are
    included here because they were measured.</p>
    <div class="ctrls">
      <div class="ctrlgrp"><b>Sort by</b><div class="seg" id="seg-bsort" role="group"
        aria-label="Sort properties">
        <button type="button" data-v="gap" aria-pressed="true">gap</button>
        <button type="button" data-v="seq" aria-pressed="false">sequence</button>
        <button type="button" data-v="str" aria-pressed="false">structure</button>
        <button type="button" data-v="name" aria-pressed="false">name</button>
      </div></div>
      <div class="ctrlgrp"><b>Show</b><div class="seg" id="seg-bshow" role="group"
        aria-label="Filter properties">
        <button type="button" data-v="all" aria-pressed="true">all</button>
        <button type="button" data-v="res" aria-pressed="false">per residue</button>
        <button type="button" data-v="chain" aria-pressed="false">per protein</button>
      </div></div>
    </div>
    {block("All twenty-five properties", '<div class="legend">', '<div class="stack prose"')}
  </details>
  <figure class="bigfig">
    <img src="{PNG[2]}" alt="Whole-protein maps for all six model arms, coloured by fold class,
      kingdom, enzyme status and protein class."/>
    <figcaption><b>The division of labour as geography.</b> Each dot is one protein, placed near the
    proteins the model describes similarly and coloured by a property it was never trained to
    predict. ProteinMPNN sorts proteins by broad fold class — the α, β and α/β regions are largely
    separate. Both sequence arms sort them by kingdom instead, splitting bacteria from eukaryotes
    cleanly, and by whether the protein is an enzyme — distinctions with no direct shape counterpart.
    Fine functional category is mixed in every arm, and those are the lowest scores in the
    battery.</figcaption>
  </figure>
  <details><summary>The same residues, as maps — every model arm</summary>
    <figure class="bigfig" style="border:0;padding:0">
      <img src="{PNG[1]}" alt="Residue maps for every model arm against six residue properties."/>
      <figcaption>The three ESM arms break into the same twenty amino-acid islands, so that pattern
      is not a matter of model size. CARP, also sequence-only and trained the same way, does not:
      its map is organised by local shape and burial, like the two structure models. Against a bare
      amino-acid label CARP is more aligned than ESM-2 on same main directions (0.937 against 0.830)
      and far less on same neighbours (0.052 against 0.916) — and a map like this is built from
      neighbours. Binding site is the one column where no model shows any organisation: contact
      residues are scattered through every map, which is why they have to be decoded by a probe and
      cannot be read off the geometry.</figcaption>
    </figure>
  </details>
</div>"""


def s_f7():
    return f"""<div class="page sec" id="f7">
  <div class="stack prose">
    <span class="eyebrow">Where it builds up · depth</span>
    <h2>Geometry is handed to the structure model; the sequence model has to build it</h2>
    <p class="prose">Left is each model's input, right is its output. The structure model already
    knows the geometry at its first layer — it is handed the coordinates, so there is nothing to
    build — and reaches its ceiling almost immediately (local shape 0.887 at the first encoder
    layer). The sequence model has to construct that knowledge, climbing steadily through twelve
    blocks and never quite arriving (local shape 0.556 at the embedding layer to 0.778 at block 11).
    For function it is the reverse: the sequence model rises far above the amino-acid-counting
    baseline — fold topology from 0.239 at the input to 0.700 at block 8, against a baseline of 0.23
    — while the structure model stays near it (0.400–0.481). This is also why the sequence model gets
    more chances to pick a best layer.</p>
  </div>
  {block("How each property builds up", '<div class="legend">')}
</div>"""


def s_beyond():
    c = D["calibration"]
    return f"""<div class="page sec" id="beyond">
  <div class="stack prose">
    <span class="eyebrow">Beyond proteins</span>
    <h2>Three artefacts, each on its own large enough to manufacture a convergence result</h2>
    <p class="prose">Of the headline agreement between a sequence and a structure model, {share('svcca'):.0f}% of one
    measure is what it reports on scrambled data; roughly half of another is the training target both
    models share; and an untrained network scores {v('untrained seq x trained str','raw'):.3f} — a
    respectable-looking number — until that target is removed, whereupon it collapses to
    {v('untrained seq x trained str','partial'):.3f}. Any one of the three would be enough to produce
    a convergence result where there is none.</p>
    <p class="prose">The controls that catch them are cheap. A scrambled-residue null for every
    measure, so a score is read as its distance above chance, not as a bare number (Gröger et al.,
    2026, make the same argument for vision and language). Dimension matching before comparing models
    of different widths. A ceiling from the same model trained twice, so “high” has a reference. And,
    wherever two models are trained toward the same target, a residualisation of that target before
    agreement is measured — the target-side counterpart of the input-side deconfounding proposed by
    Cui et al. (2022). The last of these is the one this study adds, and it changed the answer.</p>
  </div>
</div>"""


def s_limits():
    return """<div class="page sec" id="limits">
  <div class="stack prose">
    <span class="eyebrow">Limits</span>
    <h2>What this does not show</h2>
  </div>
  <div class="cards">
    <div class="card"><span class="tag">these are not AlphaFold</span>
      <p>The structure models here are ProteinMPNN and ESM-IF1, built to design sequences for a given
      backbone. No structure-prediction model is measured anywhere in this work, so nothing here says
      how AlphaFold-style models encode biology.</p></div>
    <div class="card"><span class="tag">unequal sizes</span>
      <p>ProteinMPNN's encoder is small — 3 layers against ESM-2's 12. Part of the property-prediction
      gap is size, not sequence versus shape. ESM-IF1, the larger structure model, narrows it without
      closing it.</p></div>
    <div class="card"><span class="tag">one kind of protein</span>
      <p>4,898 domains with experimentally solved structures. Everything here is crystallisable and
      domain-shaped; disordered and membrane proteins are under-represented.</p></div>
    <div class="card"><span class="tag">scores depend on sample size</span>
      <p>All three measures inflate on smaller samples, unequally — halving the sample raises the
      neighbour measure by about a quarter. Every number here is quoted at a stated sample size, and
      SVCCA is computed on at most 5,000 rows.</p></div>
    <div class="card"><span class="tag">the subtraction is linear</span>
      <p>Removing each amino acid's mean description removes the part of the answer key that is the
      same for every leucine. Anything the models encode about amino-acid identity in a non-linear
      way is not removed, so the “answer key removed” numbers are an upper bound on genuine
      agreement, not a floor.</p></div>
  </div>
</div>"""


REFERENCES = [
    "Bansal, Y., Nakkiran, P., Barak, B. (2021). Revisiting Model Stitching to Compare Neural "
    "Representations. <i>NeurIPS 2021</i>.",
    "Cui, T., Kumar, Y., Marttinen, P., Kaski, S. (2022). Deconfounded Representation Similarity for "
    "Comparison of Neural Networks. <i>NeurIPS 2022</i>. arXiv:2202.00095.",
    "Dauparas, J. et al. (2022). Robust deep learning–based protein sequence design using "
    "ProteinMPNN. <i>Science</i> 378, 49–56.",
    "Gröger, F., Wen, S., Brbić, M. (2026). Revisiting the Platonic Representation Hypothesis: An "
    "Aristotelian View. arXiv:2602.14486.",
    "Hsu, C. et al. (2022). Learning inverse folding from millions of predicted structures. "
    "<i>ICML 2022</i>. (ESM-IF1)",
    "Huh, M., Cheung, B., Wang, T., Isola, P. (2024). The Platonic Representation Hypothesis. "
    "<i>ICML 2024</i>. arXiv:2405.07987.",
    "Kornblith, S., Norouzi, M., Lee, H., Hinton, G. (2019). Similarity of Neural Network "
    "Representations Revisited. <i>ICML 2019</i>. (CKA)",
    "Lenc, K., Vedaldi, A. (2015). Understanding image representations by measuring their "
    "equivariance and equivalence. <i>CVPR 2015</i>.",
    "Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a "
    "language model. <i>Science</i> 379, 1123–1130. (ESM-2)",
    "Lu, K. et al. (2026). Two Stages of Folding: Convergent Mechanisms in AI Protein Folding "
    "Trunks. arXiv:2602.06020.",
    "Meier, J. et al. (2021). Language models enable zero-shot prediction of the effects of "
    "mutations on protein function. <i>NeurIPS 2021</i>. (ESM-1v)",
    "Raghu, M., Gilmer, J., Yosinski, J., Sohl-Dickstein, J. (2017). SVCCA: Singular Vector "
    "Canonical Correlation Analysis for Deep Learning Dynamics. <i>NeurIPS 2017</i>.",
    "Sillitoe, I. et al. (2021). CATH: increased structural coverage of functional space. "
    "<i>Nucleic Acids Research</i> 49, D266–D273.",
    "Yang, K. K., Fusi, N., Lu, A. X. (2024). Convolutions are competitive with transformers for "
    "protein sequence pretraining. <i>Cell Systems</i> 15, 286–294. (CARP)",
]


def s_data():
    # The dataset facts and the repo link already appear in "What was compared" and the header;
    # the foot carries only what has no other home: the reference list.
    refs = "".join(f"<p>{r}</p>" for r in REFERENCES)
    return f"""<div class="page sec" id="references">
  <div class="stack prose">
    <h3 class="subhead">References</h3>
  </div>
  <div class="refs">{refs}</div>
</div>"""


# ═══════════════════════════════════════════════════════════════════════ JS
def script():
    ladders = {m: {s: ladder_svg(D, m, s) for s in ("both", "as measured", "answer key removed")}
               for m in ("cka", "svcca", "mutual_knn")}
    caps = {m: (f"ESM-2 35M × ProteinMPNN: {v(MAIN,'raw',m):.3f} as measured, "
                f"{v(MAIN,'partial',m):.3f} with the answer key removed.")
            for m in ("cka", "svcca", "mutual_knn")}
    return f"""<script>
// Every figure is rendered server-side as static SVG; this only swaps between pre-rendered
// variants, so the page carries its findings with JavaScript disabled.
const LADDERS = {json.dumps(ladders)};
const LADDER_CAPS = {json.dumps(caps)};
(function () {{
  const host = document.getElementById('ladderhost');
  const cap = document.getElementById('laddercap');
  if (!host) return;
  let measure = 'cka', show = 'both';
  function render() {{
    host.innerHTML = LADDERS[measure][show];
    cap.textContent = LADDER_CAPS[measure];
  }}
  function wire(id, set) {{
    const grp = document.getElementById(id);
    if (!grp) return;
    grp.addEventListener('click', function (e) {{
      const b = e.target.closest('button');
      if (!b) return;
      grp.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      set(b.dataset.v);
      render();
    }});
  }}
  wire('seg-measure', x => {{ measure = x; }});
  wire('seg-show', x => {{ show = x; }});
}})();

// Battery sort/filter. Each row carries its two scores as bar widths (%), so the sort reads the
// DOM itself and keeps no second copy of the data: the rows and the numbers cannot drift apart.
(function () {{
  const det = document.getElementById('battery-details');
  if (!det) return;
  const groups = Array.from(det.querySelectorAll('.bat'));
  if (groups.length < 2) return;
  const heads = Array.from(det.querySelectorAll('h4'));
  const parse = row => {{
    const w = Array.from(row.querySelectorAll('.lane .lb')).map(b => parseFloat(b.style.width) || 0);
    return {{ el: row, name: (row.querySelector('.pbn') || {{}}).textContent || '',
             seq: w[0] || 0, str: w[1] || 0 }};
  }};
  const rowsBy = groups.map(g => Array.from(g.querySelectorAll('.pb')).map(parse));
  let sort = 'gap', show = 'all';
  const keyf = {{ gap: r => r.str - r.seq, seq: r => -r.seq, str: r => -r.str,
                 name: r => r.name.toLowerCase() }};
  function render() {{
    groups.forEach((g, gi) => {{
      const rows = rowsBy[gi].slice().sort((a, b) => {{
        const ka = keyf[sort](a), kb = keyf[sort](b);
        return ka < kb ? -1 : ka > kb ? 1 : 0;
      }});
      rows.forEach(r => g.appendChild(r.el));
      const on = show === 'all' || (show === 'res' && gi === 0) || (show === 'chain' && gi === 1);
      g.style.display = on ? '' : 'none';
      if (heads[gi]) heads[gi].style.display = on ? '' : 'none';
    }});
  }}
  function wire(id, set) {{
    const grp = document.getElementById(id);
    if (!grp) return;
    grp.addEventListener('click', e => {{
      const b = e.target.closest('button'); if (!b) return;
      grp.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      set(b.dataset.v); render();
    }});
  }}
  wire('seg-bsort', x => {{ sort = x; }});
  wire('seg-bshow', x => {{ show = x; }});
}})();
</script>"""


SECTIONS = [("question", "The question"), ("models", "Models"), ("measures", "Measures"),
            ("f1", "The answer key"), ("f2", "What survives"), ("f3", "No shared map"),
            ("f4", "Layer by layer"), ("f5", "Not interchangeable"), ("f6", "Who knows what"),
            ("f7", "Where it builds up"), ("beyond", "Beyond proteins"), ("limits", "Limits"),
            ("references", "References")]


def topnav():
    import html
    links = "".join(f'<a href="#{i}">{html.escape(lab)}</a>' for i, lab in SECTIONS)
    return (f'<nav class="topnav" aria-label="Sections">'
            f'<a class="brand" href="#top">Sequence vs structure</a>'
            f'<div class="navlinks">{links}</div>'
            f'<div class="navext"><a href="{REPO}">Code</a><a href="{DATASET}">Data</a></div>'
            f'</nav>')


NAV_JS = """<script>
(function () {
  // highlight the section being read: the last one whose top has passed under the bar.
  // Only the link strip is scrolled to follow it; the page's own scroll is never touched.
  const bar = document.querySelector('.topnav .navlinks');
  const links = [...document.querySelectorAll('.topnav .navlinks a')];
  const byId = Object.fromEntries(links.map(a => [a.getAttribute('href').slice(1), a]));
  const secs = Object.keys(byId).map(id => document.getElementById(id)).filter(Boolean);
  if (!bar || !secs.length) return;
  let current;
  const set = id => {
    if (id === current) return;
    current = id;
    links.forEach(a => a.classList.toggle('on', a === byId[id]));
    const a = byId[id];
    if (a) {
      const x = a.getBoundingClientRect().left - bar.getBoundingClientRect().left + bar.scrollLeft;
      bar.scrollTo({left: x - bar.clientWidth / 2 + a.offsetWidth / 2});
    }
  };
  const update = () => {
    let id = null;
    for (const s of secs) if (s.getBoundingClientRect().top <= 90) id = s.id;
    // at the very bottom the last sections cannot reach the bar; highlight the last one
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 2) id = secs[secs.length - 1].id;
    set(id);
  };
  let queued = false;
  addEventListener('scroll', () => {
    if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; update(); }); }
  }, {passive: true});
  addEventListener('resize', update);
  update();
})();
</script>"""

FAVICON = ("<link rel=\"icon\" href=\"data:image/svg+xml,"
           "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
           "%3Ccircle cx='12' cy='16' r='9' fill='%232D5BD1'/%3E"
           "%3Ccircle cx='20' cy='16' r='9' fill='%23C06014' fill-opacity='.85'/%3E%3C/svg%3E\">\n")

NAV_CSS = """
/* sticky section navbar */
html{scroll-padding-top:58px}
@media(prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
.topnav{position:sticky;top:0;z-index:50;display:flex;align-items:center;gap:18px;
        padding:0 max(16px,calc((100% - 1140px)/2 + 36px));height:48px;
        background:rgba(246,248,249,.94);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);
        border-bottom:1px solid var(--rule);font-family:var(--mono);font-size:.7rem}
.topnav a{color:var(--ink-3);text-decoration:none;white-space:nowrap}
.topnav a:hover,.topnav a:focus-visible{color:var(--chrome)}
.topnav .brand{color:var(--ink);font-weight:600;flex:none}
.navlinks{display:flex;gap:16px;overflow-x:auto;scrollbar-width:none;flex:1 1 auto;min-width:0;
          mask-image:linear-gradient(90deg,transparent 0,#000 12px,#000 calc(100% - 24px),transparent);
          -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 12px,#000 calc(100% - 24px),transparent);
          padding:0 12px}
.navlinks::-webkit-scrollbar{display:none}
.navlinks a{padding:14px 0 12px;border-bottom:2px solid transparent}
.navlinks a.on{color:var(--ink);border-bottom-color:var(--chrome)}
.navext{display:flex;gap:14px;flex:none}
.navext a{color:var(--chrome);font-weight:600}
@media(max-width:640px){.topnav .brand{display:none}.topnav{gap:10px}}
/* the win/loss chart's header row did not fit a phone screen and widened the page */
@media(max-width:640px){.dhead,.drow{grid-template-columns:minmax(6.5rem,9rem) minmax(0,1fr) 2.6rem;gap:10px}
  .dhead{letter-spacing:.02em}.dhead div{flex-wrap:wrap;column-gap:8px}}
"""


# ═══════════════════════════════════════════════════════════════════ assemble
BASE_CSS = open(ARCHIVE / "base_css.txt").read()
PAGE = ("<title>Do a sequence model and a structure model learn the same biology?</title>\n"
        f"<style>{BASE_CSS}{CSS_EXTRA}{NAV_CSS}</style>\n\n"
        + topnav() + "\n\n"
        + "\n\n".join([header(), s_question(), s_models(), s_measures(), s_f1(), s_f2(), s_f3(),
                       s_f4(), s_f5(), s_f6(), s_f7(), s_beyond(), s_limits(), s_data()])
        + "\n\n" + script() + "\n" + NAV_JS + "\n")

TITLE = "Do a sequence model and a structure model learn the same biology?"
STANDALONE = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
              '<meta name="description" content="Do a protein sequence model and a protein structure '
              'model learn the same biology? A layer-resolved comparison of ESM-2 and ProteinMPNN '
              'with controls for shared training targets and metric nulls.">\n'
              + FAVICON
              + PAGE.split("</style>", 1)[0] + "</style>\n</head>\n<body>\n"
              + PAGE.split("</style>", 1)[1] + "</body>\n</html>\n")
BUILD.mkdir(exist_ok=True)
(BUILD / "index.html").write_text(STANDALONE)


def check(p):
    bad = []
    body = p.split("</style>", 1)[1]
    depth = 0
    for t in re.finditer(r"<(/?)(div|section|figure|details|nav|header|table|tbody|thead|tr)\b[^>]*>", body):
        depth += -1 if t.group(1) else 1
        if depth < 0:
            bad.append("nesting negative"); break
    if depth:
        bad.append(f"unbalanced: {depth:+d}")
    for a in re.findall(r'href="#([\w-]+)"', p):
        if f'id="{a}"' not in p:
            bad.append(f"dangling anchor #{a}")
    return bad


errs = check(PAGE)
print(f"wrote {len(PAGE.splitlines())} lines · {len(PAGE)/1048576:.2f} MB")
print(f"  sections {PAGE.count('page sec')} · details {PAGE.count('<details>')} · "
      f"images {PAGE.count('src=\"data:image/png')}")
if errs:
    print("  ERRORS: " + "; ".join(errs)); raise SystemExit(1)
print("  structure OK")
