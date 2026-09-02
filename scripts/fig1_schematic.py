"""
fig1_schematic.py — Figure 1: study design schematic.

Four columns (DATA / MODELS / CONVERGENCE / ANALYSIS): rounded-rectangle panels,
colour-coded modalities, small inline mini-plots. Sequence-side and structure-side models carry two categorical hues
(validated for CVD separation, ΔE 77 protan) and are additionally distinguished by
position and text, so identity is never colour-alone.

Usage:
    uv run python scripts/fig1_schematic.py --out paper/neurips/figures/Fig1.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

SEQ = "#2E7EA8"      # sequence-side models
STR = "#C8641E"      # structure-side models
INK = "#1a1a1a"
MUT = "#6b6b6b"
LINE = "#3a3a3a"
SURF = "#fcfcfb"


def panel(ax, x, y, w, h, title=None, lw=1.1, ec=LINE, fc="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.014",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=1))
    if title:
        ax.text(x + w / 2, y + h + 0.016, title, ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=INK, zorder=3)


def box(ax, x, y, w, h, label, fc="white", ec=LINE, fs=8.2, bold=False, tc=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.009",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3)


def txt(ax, x, y, s, fs=8, color=INK, ha="left", va="center", style="normal", weight="normal"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, style=style,
            fontweight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=MUT, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
                                 linewidth=lw, color=color, zorder=2,
                                 shrinkA=0, shrinkB=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/neurips/figures/Fig1.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(16.5, 7.6))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ------------------------------------------------ column 1: DATA
    panel(ax, 0.012, 0.06, 0.175, 0.80, "DATA")
    txt(ax, 0.0995, 0.795, "4,898", fs=22, weight="bold", ha="center")
    txt(ax, 0.0995, 0.757, "non-redundant CATH domains", fs=8, color=MUT, ha="center")
    txt(ax, 0.0995, 0.735, "one per S35 cluster · all kingdoms", fs=7.2, color=MUT, ha="center")

    # little residue-graph glyph
    rng = np.random.default_rng(3)
    pts = np.array([[.055, .64], [.085, .68], [.115, .655], [.142, .69],
                    [.07, .60], [.105, .60], [.138, .615]])
    for i in range(len(pts) - 1):
        ax.plot([pts[i, 0], pts[i + 1, 0]], [pts[i, 1], pts[i + 1, 1]], color=STR, lw=1.1, zorder=2)
    for i, j in [(0, 4), (1, 5), (2, 6), (4, 5), (5, 6)]:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=STR, lw=0.7, alpha=.5, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=42, color=SEQ, zorder=3,
               edgecolors="white", linewidths=0.7)
    txt(ax, 0.0995, 0.565, "residues = nodes · backbone geometry = edges", fs=6.8, color=MUT, ha="center")
    txt(ax, 0.0995, 0.545, "1.37 M residues", fs=8, color=MUT, ha="center", style="italic")

    txt(ax, 0.024, 0.495, "Labels (all on the same chains)", fs=8.6, weight="bold")
    rows = [("per-residue", ["secondary structure", "burial · rel. solvent acc.",
                             "B-factor (per-chain z)", "binding / active / PTM site"]),
            ("per-chain", ["CATH class · arch. · topology", "EC class · enzyme · protein class",
                           "localisation · taxonomic kingdom"])]
    yy = 0.465
    for group, items in rows:
        txt(ax, 0.024, yy, group, fs=7.4, color=MUT, style="italic"); yy -= 0.030
        for it in items:
            ax.plot([0.030], [yy + 0.004], marker="o", ms=2.4, color=INK)
            txt(ax, 0.040, yy + 0.004, it, fs=7.4); yy -= 0.028
        yy -= 0.008

    box(ax, 0.024, 0.082, 0.152, 0.092,
        "chain-grouped 75/25 split\n"
        "(residues of one chain never\nspan train and test)\n"
        "experimental structures only",
        fc="#f2f2ef", ec=LINE, fs=7.6)

    # ------------------------------------------------ column 2: MODELS
    panel(ax, 0.203, 0.06, 0.30, 0.80, "MODELS — opposite training signals")
    txt(ax, 0.353, 0.815, "frozen · one forward pass · per-layer per-residue vectors",
        fs=7.6, color=MUT, ha="center", style="italic")

    # sequence side
    box(ax, 0.216, 0.735, 0.274, 0.032, "SEQUENCE  —  input: amino-acid sequence only",
        fc=SEQ, ec=SEQ, tc="white", bold=True, fs=8.4)
    for i, (name, sub) in enumerate([("ESM-2", "transformer · 13 / 34 layers"),
                                     ("CARP", "dilated CNN · 5 layers")]):
        yb = 0.672 - i * 0.058
        box(ax, 0.222, yb, 0.084, 0.044, name, ec=SEQ, bold=True, fs=9)
        txt(ax, 0.315, yb + 0.022, sub, fs=7.6, color=MUT)
    txt(ax, 0.222, 0.560, "objective: predict masked residues — never sees coordinates",
        fs=7.4, color=SEQ, style="italic")

    # structure side
    box(ax, 0.216, 0.487, 0.274, 0.032, "STRUCTURE  —  input: backbone coordinates only",
        fc=STR, ec=STR, tc="white", bold=True, fs=8.4)
    for i, (name, sub) in enumerate([("ProteinMPNN", "message-passing GNN · 3 layers"),
                                     ("ESM-IF1", "GVP-transformer · encoder")]):
        yb = 0.424 - i * 0.058
        box(ax, 0.222, yb, 0.084, 0.044, name, ec=STR, bold=True, fs=8.2)
        txt(ax, 0.315, yb + 0.022, sub, fs=7.6, color=MUT)
    txt(ax, 0.222, 0.312, "objective: recover sequence from structure — sequence-agnostic",
        fs=7.4, color=STR, style="italic")

    # the alignment statement
    arrow(ax, 0.353, 0.300, 0.353, 0.262, color=INK, lw=1.2)
    box(ax, 0.216, 0.196, 0.274, 0.062,
        "one vector per residue, same residues, same order\n"
        "→ exact correspondence, no alignment assumptions",
        fc="#f2f2ef", ec=LINE, fs=8.2, bold=True)
    box(ax, 0.216, 0.098, 0.274, 0.078,
        "controls: randomly-initialised counterparts\n"
        "of both architectures (4 trained/untrained combos)",
        fc="white", ec=MUT, fs=7.8)

    # ------------------------------------------------ column 3: CONVERGENCE
    panel(ax, 0.519, 0.06, 0.235, 0.80, "CONVERGENCE")
    txt(ax, 0.6365, 0.815, "every (layer × layer) pair, both models", fs=7.6,
        color=MUT, ha="center", style="italic")

    # mini heatmap glyph
    hm = np.array([[.15, .22, .30], [.20, .33, .48], [.28, .45, .72], [.24, .38, .60]])
    ax.imshow(hm, extent=[0.548, 0.664, 0.660, 0.782], cmap="viridis", aspect="auto", zorder=2)
    txt(ax, 0.606, 0.648, "structure layer →", fs=6.6, color=MUT, ha="center")
    ax.text(0.539, 0.721, "sequence layer →", fontsize=6.6, color=MUT, rotation=90,
            ha="center", va="center")

    metrics = [("CKA", "full representation"),
               ("SVCCA", "top-variance linear subspace"),
               ("mutual $k$-NN", "shared neighbourhoods")]
    yy = 0.600
    for m, d in metrics:
        box(ax, 0.532, yy, 0.096, 0.036, m, ec=LINE, bold=True, fs=8.6)
        txt(ax, 0.637, yy + 0.018, d, fs=7.4, color=MUT)
        yy -= 0.048
    txt(ax, 0.532, 0.428, "the three disagree — that gap is the result",
        fs=7.6, color=INK, style="italic", weight="bold")

    box(ax, 0.532, 0.316, 0.208, 0.094,
        "every value against a\nRESIDUE-PERMUTATION NULL\n"
        "+ SVCCA at matched PCA dimension",
        fc="#f2f2ef", ec=LINE, fs=8.0, bold=True)
    box(ax, 0.532, 0.180, 0.208, 0.118,
        "reference points, same estimator\n\n"
        "within-modality ceiling\nuntrained floor\npermutation chance",
        fc="white", ec=MUT, fs=7.8)
    box(ax, 0.532, 0.086, 0.208, 0.076,
        "supervised view:\nCohen's $\\kappa$ between the two\nmodels' probe predictions",
        fc="white", ec=MUT, fs=7.8)

    # ------------------------------------------------ column 4: ANALYSIS
    panel(ax, 0.770, 0.06, 0.218, 0.80, "ANALYSIS")

    txt(ax, 0.784, 0.800, "Is a property decodable?", fs=8.4, weight="bold")
    box(ax, 0.784, 0.700, 0.190, 0.078,
        "linear & XGBoost probes\nevery layer · chain-grouped split\nmacro-F1 / $R^2$",
        fc="white", ec=LINE, fs=7.8)
    box(ax, 0.784, 0.634, 0.190, 0.052,
        "vs amino-acid-composition null\n$\\Delta$ = emergent content",
        fc="#f2f2ef", ec=LINE, fs=7.8, bold=True)

    txt(ax, 0.784, 0.590, "Is the geometry healthy?", fs=8.4, weight="bold")
    # mini anisotropy sketch
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.828 + 0.030 * np.cos(th), 0.520 + 0.010 * np.sin(th), color=INK, lw=1.0, zorder=3)
    arrow(ax, 0.828, 0.520, 0.858, 0.520, color=STR, lw=1.4)
    arrow(ax, 0.828, 0.520, 0.828, 0.530, color=SEQ, lw=1.4)
    txt(ax, 0.872, 0.520, "anisotropy\n$\\lambda_{max}/\\lambda_{min}$", fs=7.2, color=MUT)
    txt(ax, 0.784, 0.482, "cosine similarity · PCA-90 · effective rank", fs=7.2, color=MUT)

    txt(ax, 0.784, 0.440, "Is it organised without labels?", fs=8.4, weight="bold")
    rng2 = np.random.default_rng(7)
    for c, cx in [(SEQ, 0.812), (STR, 0.878)]:
        p = rng2.normal((cx, 0.386), (0.013, 0.016), size=(60, 2))
        ax.scatter(p[:, 0], p[:, 1], s=1.6, color=c, alpha=0.65, zorder=3)
    txt(ax, 0.784, 0.324, "UMAP · HDBSCAN cluster–label ARI", fs=7.2, color=MUT)

    box(ax, 0.784, 0.196, 0.190, 0.112,
        "Reported with\n\n"
        "95 % CIs (residue resampling)\n"
        "empirical $p$ vs permutation\n"
        "bootstrap CIs on baselines",
        fc="white", ec=MUT, fs=7.8)
    box(ax, 0.784, 0.086, 0.190, 0.092,
        "CPU-only\n4,898 chains × 6 models\nall weights & data public",
        fc="#f2f2ef", ec=LINE, fs=7.8)

    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURF)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
