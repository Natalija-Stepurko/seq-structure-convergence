"""Render the page's three map images from results/umap6/coords.npz, at web scale.

Writes build/figures.json in the order build.py expects: Figure B, the full residue-level grid,
the full whole-protein grid. `python build.py --fresh-figures` then uses these renders; by default
the page uses the archived, losslessly optimised copies in archive/pngs_opt.json.

Sized to 1320px against a 1140px page column: 150 dpi doubled the base64 payload
for no visible gain at the width the page actually renders.

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


def to_data_uri(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    raw = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}", len(raw)


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
from matplotlib.gridspec import GridSpec

SS3 = ["helix", "sheet", "coil"]
BURIAL = ["exposed", "buried"]
BIND = ["other", "contacts a ligand"]
bind_cmap = ListedColormap(["#C8D0D4", "#C0392B"])


def legend_strip(ax, kind, cmap, labels=None, vrange=None, rotate=False):
    """Draw the key for one column: swatches + names, or a gradient with its end values."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if kind == "cat":
        n = len(labels)
        for i, lab in enumerate(labels):
            ax.add_patch(plt.Rectangle((i / n, .55), 1 / n * .92, .42,
                                       transform=ax.transAxes, clip_on=False,
                                       facecolor=cmap(i / max(n - 1, 1) if n > 1 else 0.0),
                                       edgecolor="none"))
            ax.text((i + .46) / n, .44, lab, transform=ax.transAxes,
                    ha="center" if not rotate else "right", va="top",
                    fontsize=5.2 if n > 12 else (6.4 if n > 5 else 7.4), color=INK3,
                    rotation=90 if rotate else 0)
    else:
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", cmap=cmap, extent=(0, 1, .55, .97),
                  transform=ax.transAxes, clip_on=False)
        ax.text(0, .40, f"{vrange[0]:.2g}", transform=ax.transAxes, ha="left", va="top",
                fontsize=7.2, color=INK3)
        ax.text(1, .40, f"{vrange[1]:.2g}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.2, color=INK3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def pct(prop):
    v = np.asarray(RES[prop]).astype(float)
    f = np.isfinite(v)
    return tuple(np.nanpercentile(v[f], [2, 98]))


def grid_with_legends(rows, cols, title, figsize, dpi, labels_src, rotate_wide=False):
    nr, nc = len(rows), len(cols)
    fig = plt.figure(figsize=figsize)
    lead = 0.62 if rotate_wide else 0.30   # rotated class names need vertical room
    gs = GridSpec(nr + 1, nc, figure=fig, height_ratios=[lead] + [1] * nr,
                  hspace=0.16, wspace=0.09)
    for c, spec in enumerate(cols):
        prop, kind, cmap, nc_cat, sub = spec
        lax = fig.add_subplot(gs[0, c])
        labs = labels_src(prop)
        legend_strip(lax, kind, cmap, labels=labs,
                     vrange=(pct(prop) if kind == "num" else None),
                     rotate=bool(labs) and len(labs) > 5)
        lax.set_title(prop, fontsize=10, pad=14, color=INK)
        lax.text(.5, 1.06, sub, transform=lax.transAxes, ha="center", fontsize=7.2, color=INK3)
    for r, (key, rlab) in enumerate(rows):
        for c, (prop, kind, cmap, nc_cat, sub) in enumerate(cols):
            ax = fig.add_subplot(gs[r + 1, c])
            vals = RES[prop] if key.startswith("res_") else codes(CHAIN[prop])[0]
            panel(ax, Z[key], vals, kind, cmap, ncat=nc_cat,
                  row_label=(rlab if c == 0 else None))
    fig.suptitle(title, fontsize=12, y=.995, color=INK)
    return to_data_uri(fig, dpi=dpi)


RES_COLS = [("amino acid", "cat", aa_cmap, 20, "which of the 20 amino acids"),
            ("secondary structure", "cat", ss_cmap, 3, "local shape"),
            ("burial", "cat", bu_cmap, 2, "buried or on the surface"),
            ("RSA", "num", plt.get_cmap("viridis"), None, "how solvent-exposed"),
            ("B-factor", "num", plt.get_cmap("viridis"), None, "flexibility within the chain"),
            ("binding site", "cat", bind_cmap, 2, "contacts a ligand")]

ROW_LABEL = {"esm": "ESM-2 35M\nsequence only", "esm650": "ESM-2 650M\nsequence only",
             "carp": "CARP-38M\nsequence only", "esm1v": "ESM-1v\nsequence only",
             "mpnn": "ProteinMPNN\nshape only", "esmif1": "ESM-IF1\nshape only"}
ROW_ORDER = ["esm", "esm650", "carp", "esm1v", "mpnn", "esmif1"]
present = [a for a in ROW_ORDER if a in ARMS]
RES_ROWS = [(f"res_{a}", ROW_LABEL[a]) for a in present]
RES_LABS = {"amino acid": list(AA), "secondary structure": SS3, "burial": BURIAL,
            "binding site": BIND, "RSA": None, "B-factor": None}

fullres_uri, fullres_bytes = grid_with_legends(
    RES_ROWS, RES_COLS,
    "Residue-level maps — every model arm, every residue property, the same 8,000 residues",
    (14.0, 1.1 + 2.3 * len(RES_ROWS)), 88, lambda p: RES_LABS[p], rotate_wide=False)

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
    "protein class": [c.replace("enzyme_ec", "EC").replace("_", " ") for c in RAW["protein class"]],
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

fullchain_uri, fullchain_bytes = grid_with_legends(
    CH_ROWS, CH_COLS,
    "Whole-protein maps — every model arm, every whole-protein label, the same 2,000 proteins",
    (11.5, 2.2 + 2.4 * len(CH_ROWS)), 85, lambda p: CH_LABS[p], rotate_wide=True)




# Figure B for the "answer key" section: two models x four properties, so the caption can say
# "colour the islands by anything structural and the colours are mixed within each island" and
# the reader can check it. One property alone cannot carry that sentence.
FB_COLS = [("amino acid", "cat", aa_cmap, 20, "which of the 20 amino acids"),
           ("secondary structure", "cat", ss_cmap, 3, "local shape"),
           ("burial", "cat", bu_cmap, 2, "buried or exposed"),
           ("RSA", "num", plt.get_cmap("viridis"), None, "solvent accessibility")]
FB_ROWS = [("res_esm", "ESM-2 35M\nsequence only"), ("res_mpnn", "ProteinMPNN\nshape only")]
FB_LABS = {"amino acid": list(AA), "secondary structure": SS3, "burial": BURIAL, "RSA": None}
res_uri, res_bytes = grid_with_legends(
    FB_ROWS, FB_COLS,
    "The same 8,000 residues, as each model arranges them",
    (12.0, 5.6), 110, lambda p: FB_LABS[p], rotate_wide=False)
print(f"Figure B: {res_bytes/1024:.0f} KB png -> {len(res_uri)/1024:.0f} KB base64")

(HERE / "build").mkdir(exist_ok=True)
json.dump([res_uri, fullres_uri, fullchain_uri], open(HERE / "build/figures.json", "w"))
print("wrote build/figures.json")
