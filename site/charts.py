"""Hand-written SVG charts for the convergence page.

House rules: title inside the SVG in mono, bar charts for categorical data, only
the four colour tokens (sequence blue, structure orange, null grey, chrome teal), values printed
at bar ends where there are fewer than ~15 bars, and a static default that carries the finding
with JavaScript disabled.
"""
import json

SEQ, STR, NULL, CHROME = "#2D5BD1", "#C06014", "#6E7880", "#0E7C7B"
INK, INK3, RULE, RULE2 = "#12191F", "#6E7880", "#DDE4E7", "#EBF0F2"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ───────────────────────────────────────────────────── Figure A: agreement by depth
def depth_bars(D):
    """Raw CKA against the structure model's last encoder layer, by depth of the sequence model.

    The shape of this curve is the whole of Finding 1: highest at the lookup layer, lowest in the
    middle, rising again where the model returns to predicting amino acids.
    """
    rows = D["depth_cka"]
    w, h, pl, pr, pt, pb = 760, 300, 46, 18, 46, 52
    iw, ih = w - pl - pr, h - pt - pb
    vmax = 0.30
    bw = iw / len(rows) * 0.74
    gap = iw / len(rows)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Agreement with the structure model '
           f'by depth of the sequence model: highest at the first layer, lowest in the middle.">',
           f'<text x="0" y="14" class="ct">Raw agreement with the structure model, by depth of the '
           f'sequence model</text>',
           f'<text x="0" y="30" class="cs">same pattern of resemblance (CKA) against ProteinMPNN\'s '
           f'last encoder layer · as measured</text>']
    for gv in (0, 0.1, 0.2, 0.3):
        y = pt + (1 - gv / vmax) * ih
        out.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" stroke="{RULE2}"/>'
                   f'<text x="{pl-7}" y="{y+3.5:.1f}" class="cax" text-anchor="end">{gv:.1f}</text>')
    hi = ["emb", "b7", "b12"]
    for i, r in enumerate(rows):
        x = pl + i * gap + (gap - bw) / 2
        bh = r["v"] / vmax * ih
        y = pt + ih - bh
        first = r["layer"] == "emb"
        col = CHROME if first else (STR if r["layer"] in ("b7",) else SEQ)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" '
                   f'fill="{col}" opacity="{1 if r["layer"] in hi else .58}">'
                   f'<title>{esc(r["layer"])}: {r["v"]:.3f}</title></rect>')
        if r["layer"] in hi:
            out.append(f'<text x="{x+bw/2:.1f}" y="{y-5:.1f}" class="cval" text-anchor="middle">'
                       f'{r["v"]:.3f}</text>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{pt+ih+14:.1f}" class="cax" text-anchor="middle">'
                   f'{esc(r["layer"])}</text>')
    out.append(f'<text x="{pl}" y="{h-8}" class="cax">input — the lookup layer</text>')
    out.append(f'<text x="{pl+iw}" y="{h-8}" class="cax" text-anchor="end">output</text>')
    out.append("</svg>")
    return "".join(out)


def depth_bars_partial(D):
    """The same depth profile after the amino-acid subtraction, raw shown as an outline behind."""
    rows = D["depth_partial"]
    w, h, pl, pr, pt, pb = 760, 318, 46, 18, 62, 52
    iw, ih = w - pl - pr, h - pt - pb
    vmax = 0.30
    gap = iw / len(rows)
    bw = gap * 0.74
    top = max(rows, key=lambda r: r["partial"])
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Agreement with the structure model by '
           f'depth of the sequence model, with the amino-acid average subtracted: nothing left at the '
           f'first layer, highest near the output.">',
           '<text x="0" y="14" class="ct">After the subtraction: agreement with the structure model, '
           'by depth of the sequence model</text>',
           '<text x="0" y="30" class="cs">same pattern of resemblance (CKA) against ProteinMPNN\'s last '
           'encoder layer · amino-acid average subtracted</text>',
           f'<rect x="{pl}" y="40" width="10" height="10" rx="2" fill="{SEQ}"/>'
           f'<text x="{pl+15}" y="49" class="cs">answer key removed (95% interval)</text>'
           f'<rect x="{pl+250}" y="40" width="10" height="10" rx="2" fill="none" stroke="{NULL}" '
           f'stroke-dasharray="2 2"/><text x="{pl+265}" y="49" class="cs">as measured</text>']
    for gv in (0, 0.1, 0.2, 0.3):
        y = pt + (1 - gv / vmax) * ih
        out.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" stroke="{RULE2}"/>'
                   f'<text x="{pl-7}" y="{y+3.5:.1f}" class="cax" text-anchor="end">{gv:.1f}</text>')
    Y = lambda v: pt + ih - v / vmax * ih
    for i, r in enumerate(rows):
        x = pl + i * gap + (gap - bw) / 2
        out.append(f'<rect x="{x:.1f}" y="{Y(r["raw"]):.1f}" width="{bw:.1f}" '
                   f'height="{pt+ih-Y(r["raw"]):.1f}" rx="2" fill="none" stroke="{NULL}" '
                   f'stroke-dasharray="2 2"><title>{esc(r["label"])} as measured: {r["raw"]:.3f}</title></rect>')
        key = r["degenerate"] or r is top
        if r["partial"] > 0:
            out.append(f'<rect x="{x:.1f}" y="{Y(r["partial"]):.1f}" width="{bw:.1f}" '
                       f'height="{pt+ih-Y(r["partial"]):.1f}" rx="2" fill="{SEQ}" '
                       f'opacity="{1 if key else .62}"><title>{esc(r["label"])} answer key removed: '
                       f'{r["partial"]:.3f} [{r["lo"]:.3f}, {r["hi"]:.3f}]</title></rect>')
            xm = x + bw / 2
            out.append(f'<line x1="{xm:.1f}" y1="{Y(r["hi"]):.1f}" x2="{xm:.1f}" y2="{Y(r["lo"]):.1f}" '
                       f'stroke="{INK}" stroke-width="1.2"/>')
        if r["degenerate"]:
            out.append(f'<text x="{x+bw/2:.1f}" y="{Y(0)-6:.1f}" class="cval" text-anchor="middle">0</text>')
        elif r is top:
            out.append(f'<text x="{x+bw/2:.1f}" y="{Y(r["hi"])-5:.1f}" class="cval" '
                       f'text-anchor="middle">{r["partial"]:.3f}</text>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{pt+ih+14:.1f}" class="cax" text-anchor="middle">'
                   f'{esc(r["label"])}</text>')
    out.append(f'<text x="{pl}" y="{h-8}" class="cax">input — nothing left after the subtraction</text>')
    out.append(f'<text x="{pl+iw}" y="{h-8}" class="cax" text-anchor="end">output</text>')
    out.append("</svg>")
    return "".join(out)


# ─────────────────────────────────────────────────────────── the ladder (interactive)
LADDER_ORDER = [
    ("ESM-1v s1 x ESM-1v s2", "The same model trained twice (two seeds)", "ceiling"),
    ("CARP x ESM-2", "CARP-38M × ESM-2 35M", "same input"),
    ("ESM-IF1 x ProteinMPNN", "ESM-IF1 × ProteinMPNN", "same input"),
    ("ESM-2 650M x ESM-IF1", "ESM-2 650M × ESM-IF1", "opposite inputs"),
    ("ESM-2 650M x ProteinMPNN", "ESM-2 650M × ProteinMPNN", "opposite inputs"),
    ("CARP x ESM-IF1", "CARP-38M × ESM-IF1", "opposite inputs"),
    ("ESM-2 35M x ProteinMPNN", "ESM-2 35M × ProteinMPNN", "opposite inputs"),
    ("CARP x ProteinMPNN", "CARP-38M × ProteinMPNN", "opposite inputs"),
    ("ESM-1v x ProteinMPNN", "ESM-1v × ProteinMPNN", "opposite inputs"),
    ("untrained seq x trained str", "Untrained sequence × trained structure", "floor"),
    ("trained seq x untrained str", "Trained sequence × untrained structure", "floor"),
    ("untrained x untrained", "Two untrained networks", "floor"),
]
METRIC_LABEL = {"cka": "Same pattern of resemblance (CKA)",
                "svcca": "Same main directions (SVCCA)",
                "mutual_knn": "Same neighbours (mutual k-NN)"}


def ladder_svg(D, metric="cka", show="both", chart_id="ladder"):
    """Horizontal bars: answer-key-removed solid, as-measured as a ghost bar behind it."""
    rows = [(lab, tag, D["every_pair"][k]) for k, lab, tag in LADDER_ORDER if k in D["every_pair"]]
    w, rowh, pl, pr, pt = 760, 26, 232, 54, 50
    ih = rowh * len(rows)
    h = pt + ih + 34
    iw = w - pl - pr
    vmax = 1.0 if metric != "mutual_knn" else 0.9
    X = lambda v: pl + min(max(v, 0), vmax) / vmax * iw
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(METRIC_LABEL[metric])} for '
           f'every pair of models, as measured and with the answer key removed.">',
           f'<text x="0" y="14" class="ct">{esc(METRIC_LABEL[metric])} for every pair</text>',
           f'<text x="0" y="30" class="cs">ghost bar = as measured · solid bar = answer key removed '
           f'· whiskers = 95% interval</text>']
    for gv in [x / 10 for x in range(0, int(vmax * 10) + 1, 2)]:
        x = X(gv)
        out.append(f'<line x1="{x:.1f}" y1="{pt-6}" x2="{x:.1f}" y2="{pt+ih}" stroke="{RULE2}"/>'
                   f'<text x="{x:.1f}" y="{pt+ih+15}" class="cax" text-anchor="middle">{gv:.1f}</text>')
    for i, (lab, tag, d) in enumerate(rows):
        y = pt + i * rowh
        raw, par = d["raw"][metric], d["partial"][metric]
        out.append(f'<text x="0" y="{y+rowh/2+3.5:.1f}" class="clab">{esc(lab)}</text>')
        out.append(f'<text x="{pl-8}" y="{y+rowh/2+3.5:.1f}" class="ctag" text-anchor="end">'
                   f'{esc(tag)}</text>')
        bh = rowh - 11
        if show in ("both", "as measured"):
            out.append(f'<rect x="{pl}" y="{y+5.5:.1f}" width="{X(raw["v"])-pl:.1f}" height="{bh}" '
                       f'rx="2" fill="none" stroke="{NULL}" stroke-dasharray="3 2" opacity=".75">'
                       f'<title>{esc(lab)} — as measured {raw["v"]:.3f}</title></rect>')
        if show in ("both", "answer key removed"):
            col = CHROME if tag == "ceiling" else (NULL if tag == "floor" else
                                                   (SEQ if tag == "same input" else STR))
            out.append(f'<rect x="{pl}" y="{y+5.5:.1f}" width="{X(par["v"])-pl:.1f}" height="{bh}" '
                       f'rx="2" fill="{col}" opacity=".88">'
                       f'<title>{esc(lab)} — answer key removed {par["v"]:.3f} '
                       f'[{par["lo"]:.3f}, {par["hi"]:.3f}]</title></rect>')
            cy = y + rowh / 2
            out.append(f'<line x1="{X(par["lo"]):.1f}" y1="{cy:.1f}" x2="{X(par["hi"]):.1f}" '
                       f'y2="{cy:.1f}" stroke="{INK}" stroke-width="1.2" opacity=".55"/>')
        shown = par["v"] if show != "as measured" else raw["v"]
        out.append(f'<text x="{w-pr+8}" y="{y+rowh/2+3.5:.1f}" class="cval">{shown:.3f}</text>')
    out.append("</svg>")
    return "".join(out)


# ───────────────────────────────────────────────── scrambled-residue calibration
def null_strips(D):
    names = [("cka", "Same pattern of resemblance", "CKA"),
             ("svcca", "Same main directions", "SVCCA"),
             ("knn", "Same neighbours", "mutual k-NN")]
    w, rowh, pl, pr, pt = 760, 44, 224, 96, 46
    h = pt + rowh * 3 + 16
    iw = w - pl - pr
    vmax = 0.45
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="What each measure still reports on '
           f'scrambled residues.">',
           f'<text x="0" y="14" class="ct">What each measure still reports on scrambled residues</text>',
           f'<text x="0" y="30" class="cs">teal = the score · grey = the part it still returns when '
           f'residue correspondence is destroyed</text>']
    for i, (k, plain, tech) in enumerate(names):
        d = D["calibration"][k]
        y = pt + i * rowh
        share = 100 * d["null"] / d["obs"]
        out.append(f'<text x="0" y="{y+16:.1f}" class="clab">{esc(plain)}</text>'
                   f'<text x="0" y="{y+29:.1f}" class="ctag">{esc(tech)}</text>')
        out.append(f'<rect x="{pl}" y="{y+6}" width="{iw}" height="20" rx="2" fill="{RULE2}"/>')
        out.append(f'<rect x="{pl}" y="{y+6}" width="{d["obs"]/vmax*iw:.1f}" height="20" rx="2" '
                   f'fill="{CHROME}"><title>{esc(plain)}: {d["obs"]:.3f}</title></rect>')
        out.append(f'<rect x="{pl}" y="{y+6}" width="{d["null"]/vmax*iw:.1f}" height="20" rx="2" '
                   f'fill="{NULL}"><title>scrambled: {d["null"]:.3f}</title></rect>')
        out.append(f'<text x="{w-pr+8}" y="{y+20:.1f}" class="cval">{d["obs"]:.3f}</text>')
        cls = "cbad" if share > 40 else "cax"
        out.append(f'<text x="{w-pr+52}" y="{y+20:.1f}" class="{cls}">{share:.0f}% null</text>')
    out.append("</svg>")
    return "".join(out)


def width_bars(D):
    """SVCCA grows with width; the part above chance does not."""
    rows = D["width"]
    w, rowh, pl, pr, pt = 760, 42, 224, 104, 46
    h = pt + rowh * len(rows) + 16
    iw = w - pl - pr
    vmax = 0.70
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="SVCCA grows with width; the part '
           f'above chance does not.">',
           f'<text x="0" y="14" class="ct">SVCCA grows with width; the part above chance does not</text>',
           f'<text x="0" y="30" class="cs">teal = reported · grey = scrambled · the number at the '
           f'end is what is left above chance</text>']
    for i, r in enumerate(rows):
        y = pt + i * rowh
        out.append(f'<text x="0" y="{y+19:.1f}" class="clab">{esc(r["lab"])}</text>')
        out.append(f'<rect x="{pl}" y="{y+5}" width="{iw}" height="20" rx="2" fill="{RULE2}"/>')
        out.append(f'<rect x="{pl}" y="{y+5}" width="{r["obs"]/vmax*iw:.1f}" height="20" rx="2" '
                   f'fill="{CHROME}"><title>reported {r["obs"]:.3f}</title></rect>')
        out.append(f'<rect x="{pl}" y="{y+5}" width="{r["null"]/vmax*iw:.1f}" height="20" rx="2" '
                   f'fill="{NULL}"><title>scrambled {r["null"]:.3f}</title></rect>')
        out.append(f'<text x="{w-pr+8}" y="{y+19:.1f}" class="cval">{r["obs"]:.3f}</text>')
        out.append(f'<text x="{w-pr+52}" y="{y+19:.1f}" class="cgood">+{r["obs"]-r["null"]:.3f}</text>')
    out.append("</svg>")
    return "".join(out)


# ─────────────────────────────────────────────────────── layer-by-layer bar panels
def layer_panels(D):
    panels = [("depth_cka", "Same pattern of resemblance", 0.45, SEQ),
              ("depth_svcca", "Same main directions", 0.45, CHROME),
              ("depth_knn", "Same neighbours", 0.03, STR)]
    out = []
    for key, name, vmax, col in panels:
        rows = D[key]
        w, h, pl, pr, pt, pb = 250, 178, 34, 8, 40, 26
        iw, ih = w - pl - pr, h - pt - pb
        gap = iw / len(rows)
        bw = gap * 0.72
        sv = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(name)} by sequence-model '
              f'layer, against the structure model\'s last encoder layer.">',
              f'<text x="0" y="12" class="ctsm">{esc(name)}</text>',
              f'<text x="0" y="25" class="cs">scale 0–{vmax:g} · as measured</text>']
        for gv in (0, vmax / 2, vmax):
            y = pt + (1 - gv / vmax) * ih
            sv.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" stroke="{RULE2}"/>'
                      f'<text x="{pl-5}" y="{y+3:.1f}" class="cax" text-anchor="end">{gv:.2f}</text>')
        for i, r in enumerate(rows):
            x = pl + i * gap + (gap - bw) / 2
            bh = min(r["v"] / vmax, 1) * ih
            sv.append(f'<rect x="{x:.1f}" y="{pt+ih-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                      f'rx="1" fill="{col}" opacity=".8">'
                      f'<title>{esc(r["layer"])}: {r["v"]:.3f}</title></rect>')
        sv.append(f'<text x="{pl}" y="{h-7}" class="cax">emb</text>'
                  f'<text x="{pl+iw}" y="{h-7}" class="cax" text-anchor="end">b12</text>')
        sv.append("</svg>")
        out.append(f'<div class="minipanel">{"".join(sv)}</div>')
    return "".join(out)


# ──────────────────────────────────────────────────────────────────── stitching
def stitch_bars(D):
    conds = [("donor", "structure model alone", STR),
             ("rand", "→ untrained sequence layers", NULL),
             ("trained", "→ trained sequence layers", SEQ),
             ("native", "sequence model alone", "#B9C2C7")]
    out = []
    for p in D["stitching"]["props"]:
        w, rowh, pl, pr, pt = 380, 30, 176, 44, 42
        h = pt + rowh * 4 + 10
        iw = w - pl - pr
        sv = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Stitching result for '
              f'{esc(p["label"])}.">',
              f'<text x="0" y="13" class="ctsm">{esc(p["label"])}</text>',
              f'<text x="0" y="26" class="cs">mean over all {D["stitching"]["cells"]} layer × entry-point '
              f'combinations · higher is better</text>']
        for i, (k, lab, col) in enumerate(conds):
            y = pt + i * rowh
            v = p[k]
            sv.append(f'<text x="0" y="{y+16:.1f}" class="clabsm">{esc(lab)}</text>')
            sv.append(f'<rect x="{pl}" y="{y+4}" width="{iw}" height="17" rx="2" fill="{RULE2}"/>')
            sv.append(f'<rect x="{pl}" y="{y+4}" width="{v*iw:.1f}" height="17" rx="2" fill="{col}">'
                      f'<title>{esc(lab)}: {v:.3f}</title></rect>')
            sv.append(f'<text x="{w-pr+6}" y="{y+16:.1f}" class="cval">{v:.3f}</text>')
        sv.append("</svg>")
        out.append(f'<div class="minipanel wide">{"".join(sv)}</div>')
    return "".join(out)


def depth_verdicts(D):
    """Stitched vs the structure model alone, at each entry depth: worse / unclear / better."""
    rows = D["stitching"]["by_depth"]
    cols = [("worse", "structure model alone better", STR),
            ("no clear difference", "no clear difference", NULL),
            ("better", "stitched better", SEQ)]
    w, pl, pr, pt, rowh = 760, 150, 20, 58, 22
    iw = w - pl - pr
    n = max(sum(r[k] for k, _, _ in cols) for r in rows)
    h = pt + rowh * len(rows) + 24
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Stitching verdicts by entry depth.">',
           '<text x="0" y="14" class="ct">The deeper the entry point, the less is lost</text>',
           f'<text x="0" y="30" class="cs">each row: {n} cases (3 structure-model layers × 3 properties) '
           f'· a case counts only when the interval across {D["stitching"]["repeats"]} repeats excludes zero</text>']
    x = pl
    for k, lab, col in cols:
        out.append(f'<rect x="{x}" y="38" width="10" height="10" rx="2" fill="{col}"/>'
                   f'<text x="{x+15}" y="47" class="cs">{esc(lab)}</text>')
        x += 215
    for i, r in enumerate(rows):
        y = pt + i * rowh
        out.append(f'<text x="0" y="{y+14:.1f}" class="clabsm">enters at layer {r["layer"]}</text>')
        x = pl
        for k, lab, col in cols:
            v = r[k]
            if v:
                bw = iw * v / n
                out.append(f'<rect x="{x:.1f}" y="{y+3}" width="{bw:.1f}" height="15" fill="{col}">'
                           f'<title>layer {r["layer"]}: {v} {esc(lab)}</title></rect>')
                if bw > 16:
                    out.append(f'<text x="{x+bw/2:.1f}" y="{y+14.5:.1f}" class="aal" fill="#fff">{v}</text>')
                x += bw
    out.append("</svg>")
    return "".join(out)
