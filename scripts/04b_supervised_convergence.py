"""
04b_supervised_convergence.py — XGB model-to-model convergence heatmaps.

The supervised analog of stage 04's CKA grid. For a property, it asks: at which
(ESM-2 layer i, ProteinMPNN layer j) pairs do the two models produce the SAME functional
readout? For each layer it trains one XGBoost probe per model (n_esm + n_struct probes
total, not n_esm*n_struct), then fills an ESM×struct grid with the chance-corrected
agreement (Cohen's κ) between the two models' test-set predictions.

High κ = the two models — trained on opposite signals — converge on the same supervised
decision for that property at those depths.

Targets (residue- or chain-level; classification):
    binding_site, ss3            (residue)
    enzyme, cath_class           (chain, mean-pooled)

Residues/chains are aligned across models by construction (shared order).

Outputs (under --out-dir):
    supervised_convergence.png   one κ heatmap per property (+ each model's own accuracy)
    kappa_grids.npz              raw grids

Usage:
    uv run python scripts/04b_supervised_convergence.py \\
        --esm-dir /ssc/results/esm --struct-dir /ssc/results/proteinmpnn \\
        --structures-dir /ssc/structures --out-dir /ssc/results/convergence
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

SS3_IDX = {"a": 0, "b": 1, "c": 2}
MAX_CLASSES = 12
# (name, granularity, source, key)
TARGETS = [
    ("binding_site", "residue", "restgt", "binding_site"),
    ("ss3", "residue", "npz", "ss3"),
    ("enzyme", "chain", "annot", "enzyme"),
    ("cath_class", "chain", "manifest", "cath_class"),
]


def _layers(pt):
    return torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()


def _collect(ids, esm_dir, struct_dir, sdir, max_chains, per_chain, seed):
    rng = np.random.default_rng(seed)
    prot, rest = sdir / "proteins", sdir / "residue_targets"
    manifest = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()}
    annot = {}
    if (sdir / "annotations.jsonl").exists():
        annot = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "annotations.jsonl").open() if l.strip()}
    E_res, S_res, E_pool, S_pool, grp = [], [], [], [], []
    res_lab = {"binding_site": [], "ss3": []}
    ch_lab = {"enzyme": [], "cath_class": []}
    g = 0
    for cid in ids:
        pe, ps = esm_dir / f"{cid}.pt", struct_dir / f"{cid}.pt"
        if not (pe.exists() and ps.exists()) or cid not in manifest:
            continue
        npz = np.load(prot / f"{cid}.npz", allow_pickle=True)
        L = int(npz["ss3"].shape[0])
        Ee, Es = _layers(pe), _layers(ps)
        if Ee.shape[1] != L or Es.shape[1] != L:
            continue
        rt = np.load(rest / f"{cid}.npz") if (rest / f"{cid}.npz").exists() else None
        # residue sample; force-include binding sites
        force = np.where(rt["binding_site"] == 1)[0] if rt is not None else np.array([], int)
        pool = np.setdiff1d(np.arange(L), force)
        rand = rng.choice(pool, min(max(0, per_chain - len(force)), len(pool)), replace=False)
        idx = np.concatenate([force, rand]).astype(int)
        E_res.append(Ee[:, idx, :]); S_res.append(Es[:, idx, :])
        res_lab["ss3"].append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        res_lab["binding_site"].append(rt["binding_site"][idx].astype(int) if rt is not None
                                       else np.full(len(idx), -1))
        E_pool.append(Ee.mean(1)); S_pool.append(Es.mean(1))
        ch_lab["enzyme"].append(annot.get(cid, {}).get("enzyme"))
        ch_lab["cath_class"].append(manifest[cid].get("cath_class"))
        grp.append(np.full(len(idx), g)); g += 1
        if g >= max_chains:
            break
    if g < 20:
        sys.exit("ERROR: too few chains with both models' embeddings.")
    return {
        "E_res": np.concatenate(E_res, 1), "S_res": np.concatenate(S_res, 1),
        "E_pool": np.stack(E_pool, 1), "S_pool": np.stack(S_pool, 1),
        "grp": np.concatenate(grp), "n_chains": g,
        "res": {k: np.concatenate(v) for k, v in res_lab.items()},
        "chain": {k: np.array(v, dtype=object) for k, v in ch_lab.items()},
    }


def _fit_predict(Xtr, ytr, Xte, balanced):
    spw = 1.0
    if balanced and set(np.unique(ytr)) <= {0, 1}:
        pos = int(np.sum(ytr)); spw = max(1.0, (len(ytr) - pos) / max(1, pos))
    m = XGBClassifier(n_estimators=80, max_depth=4, tree_method="hist", n_jobs=4,
                      verbosity=0, scale_pos_weight=spw).fit(Xtr, ytr)
    return m.predict(Xte)


def _grid_for_target(Eemb, Semb, y, tr, te, balanced):
    """Return kappa grid [n_esm, n_struct], plus each model's own accuracy per layer."""
    ne, ns = Eemb.shape[0], Semb.shape[0]
    pe = [_fit_predict(Eemb[i][tr], y[tr], Eemb[i][te], balanced) for i in range(ne)]
    psr = [_fit_predict(Semb[j][tr], y[tr], Semb[j][te], balanced) for j in range(ns)]
    acc_e = [accuracy_score(y[te], p) for p in pe]
    acc_s = [accuracy_score(y[te], p) for p in psr]
    K = np.array([[cohen_kappa_score(pe[i], psr[j]) for j in range(ns)] for i in range(ne)])
    return K, acc_e, acc_s


def main() -> None:
    ap = argparse.ArgumentParser(description="XGB model-to-model convergence")
    ap.add_argument("--esm-dir", default="results/esm")
    ap.add_argument("--struct-dir", default="results/proteinmpnn")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/convergence")
    ap.add_argument("--max-chains", type=int, default=3000)
    ap.add_argument("--per-chain", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    print(f"Collecting aligned ESM+struct embeddings over <= {args.max_chains} chains ...")
    D = _collect(ids, Path(args.esm_dir), Path(args.struct_dir), sdir,
                 args.max_chains, args.per_chain, args.seed)
    C = D["n_chains"]
    rng = np.random.default_rng(args.seed)
    test_c = set(rng.choice(C, max(1, int(C * args.test_frac)), replace=False).tolist())
    res_te = np.isin(D["grp"], list(test_c)); ch_te = np.array([c in test_c for c in range(C)])
    ne, ns = D["E_res"].shape[0], D["S_res"].shape[0]
    e_lab = ["emb"] + [f"L{i}" for i in range(1, ne)]
    s_lab = [f"enc{j+1}" for j in range(ns)]

    grids = {}
    panels = []
    for name, gran, src, key in TARGETS:
        if gran == "residue":
            y_all = D["res"][key]; Eemb, Semb = D["E_res"], D["S_res"]
            mask = y_all >= 0 if key == "binding_site" else np.ones(len(y_all), bool)
            tr = mask & ~res_te; te = mask & res_te
            y = y_all
        else:
            y_raw = D["chain"][key]; Eemb, Semb = D["E_pool"], D["S_pool"]
            have = np.array([v is not None for v in y_raw])
            vals, cnt = np.unique(y_raw[have].astype(str), return_counts=True)
            keep = set(vals[np.argsort(-cnt)][:MAX_CLASSES])
            m = have & np.array([str(v) in keep for v in y_raw])
            y = np.full(len(y_raw), -1)
            y[m] = LabelEncoder().fit_transform(y_raw[m].astype(str))
            tr = m & ~ch_te; te = m & ch_te
        if te.sum() < 10 or len(np.unique(y[tr])) < 2:
            print(f"  [skip {name}]"); continue
        balanced = key in ("binding_site", "enzyme")
        K, ae, as_ = _grid_for_target(Eemb, Semb, y, tr, te, balanced)
        grids[name] = K
        panels.append((name, K, ae, as_))
        print(f"  {name}: peak κ={np.nanmax(K):.3f} at "
              f"ESM {e_lab[np.unravel_index(np.nanargmax(K), K.shape)[0]]} × "
              f"struct {s_lab[np.unravel_index(np.nanargmax(K), K.shape)[1]]}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "kappa_grids.npz", e_lab=e_lab, s_lab=s_lab, **grids)
    _plot(panels, e_lab, s_lab, out_dir / "supervised_convergence.png")
    print(f"-> {out_dir}/supervised_convergence.png")


def _plot(panels, e_lab, s_lab, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1, 4.6), squeeze=False)
    for ax, (name, K, ae, as_) in zip(axes[0], panels):
        im = ax.imshow(K, vmin=0, vmax=max(0.2, float(np.nanmax(K))), cmap="magma", aspect="auto")
        ax.set_xticks(range(len(s_lab)))
        ax.set_xticklabels([f"{l}\n{a:.2f}" for l, a in zip(s_lab, as_)], fontsize=7)
        ax.set_yticks(range(len(e_lab)))
        ax.set_yticklabels([f"{l} {a:.2f}" for l, a in zip(e_lab, ae)], fontsize=6)
        ax.set_xlabel("ProteinMPNN layer\n(struct; own acc)");
        if ax is axes[0][0]:
            ax.set_ylabel("ESM-2 layer (own acc)")
        ax.set_title(f"{name}\ncross-model agreement (κ)", fontsize=9)
        for a in range(K.shape[0]):
            for b in range(K.shape[1]):
                ax.text(b, a, f"{K[a,b]:.2f}", ha="center", va="center",
                        color="white" if K[a, b] < 0.5 * np.nanmax(K) else "black", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("XGB model-to-model convergence — where ESM-2 & ProteinMPNN make the same prediction",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
