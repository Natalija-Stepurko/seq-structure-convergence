"""LinkedIn image (1200x1500): CKA between protein models before and after removing amino-acid identity.
Numbers from results/ladder_ci/ladder_ci.csv (the same source as the public page)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

R = {}
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for r in csv.DictReader(open(ROOT / "results/ladder_ci/ladder_ci.csv")):
    if r["metric"] == "cka":
        R[(r["pair"], r["mode"])] = (float(r["mean"]), float(r["lo"]), float(r["hi"]))

INK, MUTED, RULE, BG = "#16191D", "#525B65", "#DDE1E4", "#FFFFFF"
RAW, PART, SEQ, CTRL = "#9AA3AB", "#1F6F6B", "#1F6F6B", "#C2681A"
AA = {"Leu": "#2D5BD1", "Gly": "#C2681A", "Lys": "#7A4FB0"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
X0 = 0.055
fig = plt.figure(figsize=(8, 10), dpi=150)
fig.patch.set_facecolor(BG)

fig.text(X0, 0.955, "What happens when you subtract amino-acid", fontsize=20, fontweight="bold")
fig.text(X0, 0.924, "identity from protein embeddings?", fontsize=20, fontweight="bold")
fig.text(X0, 0.889, "A protein sequence model reads the sequence, a protein structure model the 3D backbone.",
         fontsize=11.0, color=MUTED)
fig.text(X0, 0.868, "Both are trained to predict the amino acid, so part of any agreement is that shared answer.",
         fontsize=11.0, color=MUTED)

# --- the subtraction, as a schematic -------------------------------------------------------------
fig.text(X0, 0.819, "Removing the shared answer", fontsize=14.5, fontweight="bold")
P = np.load(HERE / "aa_panels.npz")
AA = dict(zip(P["show"], ("#2D5BD1", "#C2681A", "#7A4FB0")))
NAME = {"L": "Leu", "G": "Gly", "K": "Lys"}
titles = ["1  Each residue is a vector\n    (colour = amino acid)",
          "2  Average every leucine,\n    every glycine, …",
          "3  Subtract it from each residue:\n    what makes this one different?"]
for k in range(3):
    ax = fig.add_axes([0.04 + k * 0.322, 0.597, 0.27, 0.165])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#F3F5F6")
    for sp in ax.spines.values():
        sp.set_color(RULE)
    pts = P["res"] if k == 2 else P["raw"]
    for a, col in AA.items():
        m = P["aa"] == a
        ax.scatter(pts[m, 0], pts[m, 1], s=4, color=col, alpha=.5, lw=0, zorder=2)
    if k == 1:
        for (a, col), c in zip(AA.items(), P["means"]):
            ax.scatter(*c, marker="*", s=300, color=col, ec="white", lw=1.3, zorder=3)
            ax.annotate(NAME[a], c, xytext=(9, 7), textcoords="offset points", fontsize=9.5, color=col,
                        fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.8))
    fig.text(0.04 + k * 0.322 + 0.005, 0.770, titles[k], fontsize=10.3, va="bottom",
             linespacing=1.25)
    if k < 2:
        fig.add_artist(FancyArrowPatch((0.04 + k * 0.322 + 0.274, 0.679), (0.04 + (k + 1) * 0.322 - 0.004, 0.679),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=13,
                                       color=MUTED, lw=1.3))

# --- the chart -------------------------------------------------------------------------------------
fig.text(X0, 0.553, "How much the two models agree (CKA)", fontsize=14.5, fontweight="bold")
fig.text(X0, 0.533, "0 = unrelated · 1 = the same representation", fontsize=10.3, color=MUTED)
rows = [("row", "ESM-1v s1 x ESM-1v s2", "ESM-1v, checkpoint 1 × checkpoint 2", "ref"),
        ("gap",),
        ("head", "Sequence × structure models"),
        ("row", "ESM-2 650M x ESM-IF1", "ESM-2 650M × ESM-IF1", "x"),
        ("row", "ESM-2 35M x ProteinMPNN", "ESM-2 35M × ProteinMPNN", "main"),
        ("row", "CARP x ESM-IF1", "CARP × ESM-IF1", "x"),
        ("row", "ESM-2 650M x ProteinMPNN", "ESM-2 650M × ProteinMPNN", "x"),
        ("row", "ESM-1v x ProteinMPNN", "ESM-1v × ProteinMPNN", "x"),
        ("row", "CARP x ProteinMPNN", "CARP × ProteinMPNN", "x"),
        ("gap",),
        ("head", "Untrained controls"),
        ("row", "untrained seq x trained str", "Untrained ESM-2 35M × ProteinMPNN", "x"),
        ("row", "trained seq x untrained str", "ESM-2 35M × untrained ProteinMPNN", "x")]
ax = fig.add_axes([0.40, 0.155, 0.555, 0.355])
yax = ax.get_yaxis_transform()
step = {"row": 1.0, "gap": 0.45, "head": 0.85}
y = sum(step[r[0]] for r in rows)
top = y
for r in rows:
    y -= step[r[0]]
    if r[0] == "gap":
        continue
    if r[0] == "head":
        ax.text(-0.03, y, r[1], transform=yax, ha="right", va="center", fontsize=10.6, color=PART,
                fontweight="bold")
        continue
    _, key, name, kind = r
    raw, par = R[(key, "raw")][0], R[(key, "partial")][0]
    col = INK if kind == "ref" else PART
    if abs(raw - par) > 0.02:
        ax.annotate("", xy=(par, y), xytext=(raw, y),
                    arrowprops=dict(arrowstyle="-|>", color=PART, lw=2.2 if kind == "main" else 1.6,
                                    mutation_scale=12, shrinkA=6, shrinkB=5))
    ax.scatter([raw], [y], s=70, color=RAW, zorder=3, lw=0)
    ax.scatter([par], [y], s=70, color=col, zorder=3, lw=0)
    ax.text(-0.03, y, name, transform=yax, ha="right", va="center", fontsize=10.6,
            fontweight="bold" if kind == "main" else "normal")
    w = "bold" if kind == "main" else "normal"
    if kind == "ref":
        ax.text(min(raw, par) - 0.035, y, f"{raw:.2f} → {par:.2f}", va="center", ha="right", fontsize=10.6,
                fontweight="bold", color=INK)
        continue
ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, top - 0.4)
ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="x", colors=MUTED, labelsize=9.5, length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(RULE)
ax.grid(axis="x", color=RULE, lw=0.8); ax.set_axisbelow(True)
# legend, one line under the plot
for x, col, lab in ((0.40, RAW, "Before subtraction"), (0.63, PART, "After subtraction")):
    fig.add_artist(plt.Line2D([x], [0.112], marker="o", ms=8.5, color=col, lw=0, transform=fig.transFigure))
    fig.text(x + 0.018, 0.112, lab, va="center", fontsize=10.2, color=MUTED)

fig.add_artist(plt.Line2D([X0, 1 - X0], [0.085, 0.085], color=RULE, lw=1.1))
fig.text(X0, 0.058, "Study and code: natalija-stepurko.github.io/seq-structure-convergence", fontsize=10.6, color=INK)
fig.text(X0, 0.034, "Natalija Stepurko · natalija-stepurko.com", fontsize=10.6, color=MUTED)
fig.savefig(HERE / "linkedin_aa_removal.png", dpi=150, facecolor=BG)
for r in rows:
    if r[0] == "row":
        a, b = R[(r[1], "raw")][0], R[(r[1], "partial")][0]
        print(f"{r[1]:<30} {a:.3f} -> {b:.3f}  ({(b - a) / a:+.0%})")
