"""
11_umap_property_grid.py — UMAP property grids, model side-by-side.

Qualitative view: embed the SAME residues (or chains) from each model with UMAP
and colour the identical points by each property. Placing the models side by side shows whether
the two organise protein space into visually corresponding structure — the geometric counterpart
of the quantitative convergence metrics.

Rows = models, columns = properties. Residue level: amino acid, secondary structure, burial,
relative solvent accessibility, B-factor (per-chain z), binding site. Chain level (mean-pooled):
CATH class, kingdom, enzyme, protein class.

Outputs (under --out-dir):
    umap_residue_grid.png, umap_chain_grid.png, coords.npz (2-D coords per model)

Usage:
    uv run python scripts/11_umap_property_grid.py --structures-dir /ssc/structures \\
        --models esm=/ssc/results/esm esm650=/ssc/results/esm650 mpnn=/ssc/results/proteinmpnn \\
        --out-dir /ssc/results/umap
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.decomposition import PCA

AA = "ACDEFGHIKLMNPQRSTVWY"; AA_IDX = {a: i for i, a in enumerate(AA)}
SS3_IDX = {"a": 0, "b": 1, "c": 2}
BURIAL_RSA = 0.25


def _collect(ids, dirs, prot, rest, manifest, annot, max_res, per_chain, seed):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in dirs}; pool = {k: [] for k in dirs}
    rl = {"amino acid": [], "secondary structure": [], "burial": [], "RSA": [],
          "B-factor": [], "binding site": []}
    cl = {"CATH class": [], "kingdom": [], "enzyme": [], "protein class": []}
    total = 0
    for cid in ids:
        if not all((Path(d) / f"{cid}.pt").exists() for d in dirs.values()):
            continue
        npz = np.load(prot / f"{cid}.npz", allow_pickle=True)
        seq = str(npz["seq"]); L = int(npz["ss3"].shape[0])
        E = {k: torch.load(Path(d) / f"{cid}.pt", weights_only=False)["layers"]
             .to(torch.float32).numpy() for k, d in dirs.items()}
        if any(v.shape[1] != L for v in E.values()):
            continue
        for k in dirs:
            pool[k].append(E[k][-1].mean(axis=0))
        m = manifest[cid]; a = annot.get(cid, {})
        cl["CATH class"].append(m.get("cath_class")); cl["kingdom"].append(a.get("kingdom"))
        cl["enzyme"].append(a.get("enzyme")); cl["protein class"].append(a.get("protein_class"))
        idx = rng.choice(L, min(per_chain, L), replace=False)
        for k in dirs:
            res[k].append(E[k][-1][idx])          # last layer, per-residue
        rl["amino acid"].append(np.array([AA_IDX.get(c, 20) for c in np.array(list(seq))[idx]]))
        rl["secondary structure"].append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        r = npz["rsa"][idx].astype(np.float64)
        rl["RSA"].append(r); rl["burial"].append(np.where(np.isfinite(r), (r < BURIAL_RSA).astype(int), -1))
        rt = rest / f"{cid}.npz"
        if rt.exists():
            t = np.load(rt); bf = t["bfactor"].astype(np.float64)
            bf = (bf - np.nanmean(bf)) / (np.nanstd(bf) + 1e-6)
            rl["B-factor"].append(bf[idx]); rl["binding site"].append(t["binding_site"][idx].astype(int))
        else:
            rl["B-factor"].append(np.full(len(idx), np.nan)); rl["binding site"].append(np.full(len(idx), -1))
        total += len(idx)
        if total >= max_res:
            break
    return ({k: np.concatenate(v, axis=0) for k, v in res.items()},
            {k: np.concatenate(v) for k, v in rl.items()},
            {k: np.stack(v) for k, v in pool.items()},
            {k: np.array(v, dtype=object) for k, v in cl.items()})


def _umap(X, seed):
    import umap
    Xp = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(Xp)


# display metadata: property -> (kind, category names). Anything unlisted is inferred.
PROP_META = {
    "amino acid": ("cat", list(AA) + ["X"]),
    "secondary structure": ("cat", ["helix", "sheet", "coil"]),
    "burial": ("cat", ["exposed", "buried"]),
    "binding site": ("cat", ["other", "binding"]),
    "enzyme": ("cat", ["non-enzyme", "enzyme"]),
    "CATH class": ("cat", None),   # numeric codes 1..6, labelled by value
    "RSA": ("cont", None),
    "B-factor": ("cont", None),
}


def _as_plottable(p, y):
    """-> (numeric values, is_categorical, category names or None). NaN = unlabelled."""
    kind, names = PROP_META.get(p, (None, None))
    if y.dtype == object:                      # strings (kingdom, protein class, ...)
        vals = np.array([None if v is None or (isinstance(v, float) and v != v) else str(v) for v in y],
                        dtype=object)
        cats = sorted({v for v in vals if v is not None})
        code = {c: k for k, c in enumerate(cats)}
        num = np.array([code[v] if v is not None else np.nan for v in vals], dtype=float)
        # numeric-coded flags stored as objects (e.g. enzyme 0/1) get their declared names
        if names is not None and len(cats) <= len(names) and all(c.isdigit() for c in cats):
            cats = [names[int(c)] for c in cats]
        return num, True, cats
    num = y.astype(float).copy()
    num[num < 0] = np.nan                      # -1 encodes "unlabelled" for our int targets
    if kind == "cont":
        return num, False, None
    uniq = np.unique(num[np.isfinite(num)])
    if kind == "cat" or len(uniq) <= 21:
        if names is None:
            names = [f"{int(u)}" for u in uniq]
            remap = {u: k for k, u in enumerate(uniq)}
            num = np.array([remap.get(v, np.nan) if np.isfinite(v) else np.nan for v in num])
        return num, True, names
    return num, False, None


COL_W = 3.3   # inches per column; used for figsize and for the label-fit test


def _grid(coords, labels, title, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    models = list(coords); props = list(labels)
    fig, axes = plt.subplots(len(models), len(props),
                             figsize=(COL_W * len(props), 3.2 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        xy = coords[m]
        for j, p in enumerate(props):
            ax = axes[i][j]
            num, is_cat, cats = _as_plottable(p, labels[p])
            ok = np.isfinite(num)
            ax.scatter(xy[~ok, 0], xy[~ok, 1], s=2, c="lightgrey", alpha=0.25)
            if is_cat:
                n = max(1, len(cats))
                base = plt.get_cmap("tab20" if n > 10 else "tab10")
                cmap = ListedColormap([base(k % base.N) for k in range(n)])
                norm = BoundaryNorm(np.arange(-0.5, n + 0.5), n)
                sc = ax.scatter(xy[ok, 0], xy[ok, 1], s=2, c=num[ok], cmap=cmap, norm=norm, alpha=0.7)
            else:
                # robust colour limits: a few extreme values (e.g. B-factor z up to +7) would
                # otherwise compress the bulk of the distribution into one flat colour
                lo, hi = np.nanpercentile(num[ok], [2, 98])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = np.nanmin(num[ok]), np.nanmax(num[ok])
                sc = ax.scatter(xy[ok, 0], xy[ok, 1], s=2, c=np.clip(num[ok], lo, hi),
                                cmap="viridis", alpha=0.7, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(m, fontsize=11, fontweight="bold")
            # one legend / colourbar per column, above the top row only
            if i == 0:
                cax = ax.inset_axes([0.0, 1.06, 1.0, 0.05])
                cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
                # move ticks to the top FIRST — doing this after styling would reset the labels
                cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
                pad = 16
                if is_cat:
                    cb.set_ticks(range(len(cats)))
                    cb.set_ticklabels([str(c) for c in cats])
                    longest = max((len(str(c)) for c in cats), default=1)
                    fs = 5.5 if len(cats) > 12 else (6.5 if len(cats) > 8 else 8)
                    # rotate only if the labels would actually collide: compare the widest
                    # label's width to the horizontal space one tick gets
                    per_tick_in = COL_W / max(1, len(cats))
                    text_w_in = longest * fs * 0.6 / 72.0
                    rot = 90 if text_w_in > 0.9 * per_tick_in else 0
                    cb.ax.tick_params(labelsize=fs, labelrotation=rot)
                    if rot:                       # rotated labels grow upward — clear the title
                        pad = 14 + int(longest * fs * 0.62)
                else:
                    cb.ax.tick_params(labelsize=7)
                cax.set_title(p, fontsize=11, pad=pad)
    fig.suptitle(title + "   (grey = unlabelled; continuous scales clipped to 2nd–98th percentile)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="UMAP property grids per model")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/umap")
    ap.add_argument("--max-residues", type=int, default=8000)
    ap.add_argument("--per-chain", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replot", action="store_true", help="re-render figures from cached coords.npz")
    args = ap.parse_args()

    dirs = dict(m.split("=", 1) for m in args.models)
    sdir = Path(args.structures_dir)
    manifest = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()}
    annot = {}
    if (sdir / "annotations.jsonl").exists():
        annot = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "annotations.jsonl").open() if l.strip()}
    ids = [i for i, r in manifest.items() if r.get("valid", True)]

    res, rl, pool, cl = _collect(ids, dirs, sdir / "proteins", sdir / "residue_targets",
                                 manifest, annot, args.max_residues, args.per_chain, args.seed)
    print(f"residues: {next(iter(res.values())).shape[0]:,}  chains: {next(iter(pool.values())).shape[0]:,}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = out / "coords.npz"
    if args.replot and cache.exists():        # re-render figures without recomputing UMAP
        z = np.load(cache, allow_pickle=True)
        rc = {m: z[f"res_{m}"] for m in dirs if f"res_{m}" in z}
        cc = {m: z[f"chain_{m}"] for m in dirs if f"chain_{m}" in z}
        rl = {k: v for k, v in z["res_labels"].item().items()}
        cl = {k: v for k, v in z["chain_labels"].item().items()}
        print("re-plotting from cached coords")
    else:
        rc = {m: _umap(v, args.seed) for m, v in res.items()}
        print("residue UMAPs done")
        cc = {m: _umap(v, args.seed) for m, v in pool.items()}
        print("chain UMAPs done")
        np.savez(cache, **{f"res_{m}": v for m, v in rc.items()},
                 **{f"chain_{m}": v for m, v in cc.items()},
                 res_labels=rl, chain_labels=cl)
    _grid(rc, rl, "Residue-level UMAP (last layer), same residues, coloured by property",
          out / "umap_residue_grid.png")
    _grid(cc, cl, "Chain-level UMAP (last layer, mean-pooled), same chains, coloured by property",
          out / "umap_chain_grid.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
