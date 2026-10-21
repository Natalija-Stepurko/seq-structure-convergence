"""Render the page's three map images from results/umap6/coords.npz, at web scale.

Writes build/figures.json in the order build.py expects: Figure B, the full residue-level grid,
the full whole-protein grid. `python build.py --fresh-figures` then uses these renders; by default
the page uses the archived copies in archive/maps.json.

Rendered at twice the 1140px page column, so they stay sharp on high-density screens, and stored
as WebP when that is smaller than PNG (it is, by about two thirds).

The pipeline's own figures are 6x3 and 4x3 panels at print resolution -- unreadable inlined in a
web page, and ~1 MB each before base64. These are the same coordinates, cut down to the panels
that carry the argument and styled to match the page.

The residue-level figure is the visual form of the shared-answer-key problem: the sequence model
partitions residue space into one island per amino acid, so what looks like structure is the
alphabet; the structure model partitions the same residues by environment instead.
"""
import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from PIL import Image

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "results/umap6/coords.npz"
Z = np.load(SRC, allow_pickle=True)
ARMS = [k[4:] for k in Z.keys() if k.startswith("res_") and k != "res_labels"]
print(f"  source: {SRC}\n  arms:   {', '.join(ARMS)}")
RES, CHAIN = Z["res_labels"].item(), Z["chain_labels"].item()
INK, INK3, RULE, GROUND = "#12191F", "#6E7880", "#DDE4E7", "#F6F8F9"
AA = "ACDEFGHIKLMNPQRSTVWY"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": RULE,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.linewidth": 0.8,
})


def panel(ax, xy, vals, kind, cmap, title=None, row_label=None, ncat=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(RULE)
    v = np.asarray(vals)
    if kind == "cat":
        # Categorical codes must be normalised against the class count, not against the data
        # range. `burial` encodes -1 as missing, so a plain -1..1 normalisation put BOTH real
        # classes (0 and 1) in the upper half of a two-colour map and the panel came out
        # uniformly one colour.
        codes = v.astype(float)
        keep = codes >= 0
        n = ncat if ncat else int(codes[keep].max()) + 1
        ax.scatter(xy[~keep, 0], xy[~keep, 1], c="#E4E9EB", s=1.2, linewidths=0, rasterized=True)
        ax.scatter(xy[keep, 0], xy[keep, 1], c=codes[keep], cmap=cmap, vmin=-0.5, vmax=n - 0.5,
                   s=1.6, alpha=.85, linewidths=0, rasterized=True)
    else:
        f = np.isfinite(v.astype(float))
        lo, hi = np.nanpercentile(v[f].astype(float), [2, 98])
        ax.scatter(xy[~f, 0], xy[~f, 1], c="#E4E9EB", s=1.2, linewidths=0, rasterized=True)
        ax.scatter(xy[f, 0], xy[f, 1], c=np.clip(v[f].astype(float), lo, hi), cmap=cmap,
                   s=1.6, alpha=.85, linewidths=0, rasterized=True)
    if title:
        ax.set_title(title, fontsize=10.5, pad=7, color=INK)
    if row_label:
        ax.set_ylabel(row_label, fontsize=10.5, labelpad=8, color=INK)


PAGE_PX = 1140          # the page column the maps are shown in


def to_data_uri(fig, width_in):
    """PNG at twice the page column width, then WebP (quality 92) if that is smaller."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=2 * PAGE_PX / width_in, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    png = buf.getvalue()
    im = Image.open(io.BytesIO(png)).convert("RGB")
    out = io.BytesIO(); im.save(out, format="WEBP", quality=92, method=6)
    webp = out.getvalue()
    raw, mime = (webp, "webp") if len(webp) < len(png) else (png, "png")
    print(f"  {im.size[0]}x{im.size[1]} px: png {len(png)/1024:.0f} KB, webp {len(webp)/1024:.0f} KB -> {mime}")
    return f"data:image/{mime};base64,{base64.b64encode(raw).decode()}", len(raw)


# ─────────────────────────────────────────────────────────── residue-level
aa_cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))
ss_cmap = ListedColormap(["#2D5BD1", "#C06014", "#0E7C7B"])
bu_cmap = ListedColormap(["#2D5BD1", "#C06014"])



# ───────────────────────────────────────────────────────────── chain-level
cath_cmap = ListedColormap(["#2D5BD1", "#C06014", "#0E7C7B", "#8B5FBF", "#B9C2C7"])
king_cmap = ListedColormap(["#2D5BD1", "#C06014", "#0E7C7B", "#C0392B", "#B9C2C7"])


MISSING = {"None", "none", "other", "unknown", "nan", ""}


def codes(v, drop_missing=True):
    """Map class strings to indices, sending unlabelled values to -1 so they render grey.

    `None` and `other` are not classes -- they are absent labels. Giving them a colour both
    wastes a slot in the palette and makes the legend claim a distinction that is not there.
    """
    vals = [str(x) for x in np.asarray(v).tolist()]
    real = sorted({x for x in vals if not (drop_missing and x in MISSING)})
    m = {x: i for i, x in enumerate(real)}
    return np.array([m.get(x, -1) for x in vals], float), real









# ═══════════════════════════════════════════════════ the FULL grids, with legends
# Every model arm against every property. A colour map is useless without a key, so each column
# carries its own legend strip: discrete swatches for categorical properties, a gradient bar with
# end values for continuous ones.

SS3 = ["helix", "strand", "coil"]
BURIAL = ["exposed", "buried"]
BIND = ["other", "contacts a ligand"]
bind_cmap = ListedColormap(["#C8D0D4", "#C0392B"])


def column_key(ax, prop, sub, kind, cmap, labels=None, vrange=None, ncol=None):
    """The key for one column, drawn above its top panel: name, one-line description, then
    swatches for a categorical property or a gradient with its end values for a continuous one.
    Anchored to the panel, so constrained layout reserves exactly the space it needs."""
    head = f"{prop}\n{sub}"
    if kind == "cat":
        n = len(labels)
        wide = sum(len(x) for x in labels) > 20
        cols = ncol or (1 if wide and n == 2 else (n if n <= 3 else (5 if n == 20 else 3)))   # fit the panel
        handles = [Line2D([], [], ls="", marker="o", ms=5.5, mew=0,
                          color=cmap(i / max(n - 1, 1) if n > 1 else 0.0)) for i in range(n)]
        leg = ax.legend(handles, labels, title=head, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                        ncol=cols, frameon=False, fontsize=7.4, title_fontsize=9.6,
                        handletextpad=0.25, columnspacing=0.9, labelspacing=0.35,
                        borderaxespad=0.2, borderpad=0.1)
        leg.get_title().set_color(INK)
        leg.get_title().set_multialignment("center")
        for t in leg.get_texts():
            t.set_color(INK3)
    else:
        cax = ax.inset_axes([0.12, 1.06, 0.76, 0.055])
        cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap=cmap)
        cax.set_xticks([0, 255], [f"{vrange[0]:.2g}", f"{vrange[1]:.2g}"], fontsize=7.4, color=INK3)
        cax.set_yticks([])
        cax.tick_params(length=0, pad=2)
        for sp in cax.spines.values():
            sp.set_visible(False)
        cax.set_title(head, fontsize=9.6, color=INK, pad=4, linespacing=1.3)


def pct(prop):
    v = np.asarray(RES[prop]).astype(float)
    f = np.isfinite(v)
    return tuple(np.nanpercentile(v[f], [2, 98]))


def grid_with_legends(rows, cols, width, panel_h, labels_src):
    """Every model arm (rows) against every property (columns), one key per column. Rendered at
    twice the page column width so it stays sharp on high-density screens."""
    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(width, panel_h * nr + 1.0), squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.03, hspace=0.04)
    for r, (key, rlab) in enumerate(rows):
        for c, (prop, kind, cmap, nc_cat, sub) in enumerate(cols):
            ax = axes[r, c]
            vals = RES[prop] if key.startswith("res_") else codes(CHAIN[prop])[0]
            panel(ax, Z[key], vals, kind, cmap, ncat=nc_cat, row_label=(rlab if c == 0 else None))
            if r == 0:
                column_key(ax, prop, sub, kind, cmap, labels=labels_src(prop),
                           vrange=(pct(prop) if kind == "num" else None))
    return to_data_uri(fig, width)


RES_COLS = [("amino acid", "cat", aa_cmap, 20, "which of the 20 amino acids"),
            ("secondary structure", "cat", ss_cmap, 3, "helix, strand or coil"),
            ("burial", "cat", bu_cmap, 2, "buried or exposed"),
            ("RSA", "num", plt.get_cmap("viridis"), None, "solvent exposure"),
            ("B-factor", "num", plt.get_cmap("viridis"), None, "flexibility"),
            ("binding site", "cat", bind_cmap, 2, "ligand contact")]

ROW_LABEL = {"esm": "ESM-2 35M\nsequence only", "esm650": "ESM-2 650M\nsequence only",
             "carp": "CARP-38M\nsequence only", "esm1v": "ESM-1v\nsequence only",
             "mpnn": "ProteinMPNN\nshape only", "esmif1": "ESM-IF1\nshape only"}
ROW_ORDER = ["esm", "esm650", "carp", "esm1v", "mpnn", "esmif1"]
present = [a for a in ROW_ORDER if a in ARMS]
RES_ROWS = [(f"res_{a}", ROW_LABEL[a]) for a in present]
RES_LABS = {"amino acid": list(AA), "secondary structure": SS3, "burial": BURIAL,
            "binding site": BIND, "RSA": None, "B-factor": None}

fullres_uri, fullres_bytes = grid_with_legends(RES_ROWS, RES_COLS, 11.4, 1.55, lambda p: RES_LABS[p])

# A ListedColormap longer than the class count skips entries -- with 4 kingdoms against a
# 5-colour list, Viruses landed on the grey reserved for "unlabelled". Slice each palette to
# exactly the number of classes present.
PALETTE = ["#2D5BD1", "#C06014", "#0E7C7B", "#8B5FBF", "#C0392B", "#7A8B99"]
exact = lambda n: ListedColormap(PALETTE[:n])

RAW = {p: codes(CHAIN[p])[1] for p in ["CATH class", "kingdom", "enzyme", "protein class"]}
CATH_NAMES = {"1": "α", "2": "β", "3": "α/β", "4": "few SS", "6": "special"}
CH_LABS = {
    "CATH class": [CATH_NAMES.get(c, c) for c in RAW["CATH class"]],
    "kingdom": RAW["kingdom"],
    "enzyme": ["not an enzyme", "enzyme"],
    "protein class": [c.replace("enzyme_ec", "EC").replace("_", " ").replace("dna ", "DNA ")
                      .replace("rna ", "RNA ") for c in RAW["protein class"]],
}
CH_COLS = [("CATH class", "cat", exact(len(CH_LABS["CATH class"])),
            len(CH_LABS["CATH class"]), "broad fold class"),
           ("kingdom", "cat", exact(len(CH_LABS["kingdom"])),
            len(CH_LABS["kingdom"]), "domain of life"),
           ("enzyme", "cat", exact(2), 2, "enzyme or not"),
           ("protein class", "cat",
            ListedColormap(plt.cm.tab20(np.linspace(0, 1, len(CH_LABS["protein class"])))),
            len(CH_LABS["protein class"]), f"{len(CH_LABS['protein class'])} functional categories")]
CH_ROWS = [(f"chain_{a}", ROW_LABEL[a]) for a in present]

fullchain_uri, fullchain_bytes = grid_with_legends(CH_ROWS, CH_COLS, 11.4, 2.1, lambda p: CH_LABS[p])




# Figure B for the "answer key" section: two models x four properties, so the caption can say
# "colour the islands by anything structural and the colours are mixed within each island" and
# the reader can check it. One property alone cannot carry that sentence.
FB_COLS = [("amino acid", "cat", aa_cmap, 20, "which of the 20 amino acids"),
           ("secondary structure", "cat", ss_cmap, 3, "helix, strand or coil"),
           ("burial", "cat", bu_cmap, 2, "buried or exposed"),
           ("RSA", "num", plt.get_cmap("viridis"), None, "solvent accessibility")]
FB_ROWS = [("res_esm", "ESM-2 35M\nsequence only"), ("res_mpnn", "ProteinMPNN\nshape only")]
FB_LABS = {"amino acid": list(AA), "secondary structure": SS3, "burial": BURIAL, "RSA": None}
res_uri, res_bytes = grid_with_legends(FB_ROWS, FB_COLS, 11.4, 2.5, lambda p: FB_LABS[p])
print(f"Figure B: {res_bytes/1024:.0f} KB -> {len(res_uri)/1024:.0f} KB base64")

(HERE / "build").mkdir(exist_ok=True)
json.dump([res_uri, fullres_uri, fullchain_uri], open(HERE / "build/figures.json", "w"))
print("wrote build/figures.json")
