"""
05_property_prediction.py — supervised probes on frozen embeddings.

Trains linear and XGBoost probes on each layer's frozen embeddings to decode labels the
model was never trained on, with chain-grouped train/test splits (no residue leakage).
Tests: transferability, the local→global depth law (P3), the linear-vs-nonlinear
(XGB−linear) gap, and — via a learning curve — data efficiency vs. a from-scratch
baseline (P5).

Targets:
    residue-level (per-residue embeddings): SSE (3-class acc), burial (binary acc),
        RSA (regression R²)
    chain-level (mean-pooled embeddings): CATH class (4-class acc)
Learning curve: CATH accuracy vs. training-set size — frozen embedding (best layer) vs.
    an amino-acid-composition baseline (a representation-free control).

Outputs (under --out-dir/<model-name>, default results/probes/<model-name>):
    metrics.csv, probe_curves.png, learning_curve.png (if enough chains)

Usage:
    uv run python scripts/05_property_prediction.py \\
        --results-dir /ssc/results/esm --model-name esm \\
        --structures-dir /ssc/structures --out-dir /ssc/results/probes
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
BURIAL_RSA = 0.25
SS3_IDX = {"a": 0, "b": 1, "c": 2}


def _collect(ids, res_dir, prot_dir, cath_by_id, max_chains, per_chain, seed):
    rng = np.random.default_rng(seed)
    res_emb, ss3, burial, rsa, grp = [], [], [], [], []
    pooled, cath, comp = [], [], []
    g = 0
    for cid in ids:
        pt = res_dir / f"{cid}.pt"
        if not pt.exists() or cid not in cath_by_id:
            continue
        npz = np.load(prot_dir / f"{cid}.npz", allow_pickle=True)
        seq = str(npz["seq"])
        L = int(npz["ss3"].shape[0])
        E = torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()
        if E.shape[1] != L:
            continue
        # chain-level pooled + AA composition baseline
        pooled.append(E.mean(axis=1))                        # [n_layers, D]
        cath.append(cath_by_id[cid])
        cc = np.zeros(20)
        for a in seq:
            if a in AA_IDX:
                cc[AA_IDX[a]] += 1
        comp.append(cc / max(1, len(seq)))
        # residue-level sample
        idx = rng.choice(L, per_chain, replace=False) if L > per_chain else np.arange(L)
        res_emb.append(E[:, idx, :])
        ss3.append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        r = npz["rsa"][idx].astype(np.float64)
        burial.append(np.where(np.isfinite(r), (r < BURIAL_RSA).astype(int), -1))
        rsa.append(r)
        grp.append(np.full(len(idx), g))
        g += 1
        if g >= max_chains:
            break
    if g < 4:
        sys.exit("ERROR: too few chains with embeddings + labels.")
    return {
        "res_emb": np.concatenate(res_emb, axis=1),   # [n_layers, N, D]
        "ss3": np.concatenate(ss3), "burial": np.concatenate(burial),
        "rsa": np.concatenate(rsa), "grp": np.concatenate(grp),
        "pooled": np.stack(pooled, axis=1),           # [n_layers, C, D]
        "cath": np.array(cath), "comp": np.array(comp), "n_chains": g,
    }


def _clf(Xtr, ytr, Xte, yte, linear=True):
    if linear:
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=300, C=1.0)
        m.fit(sc.transform(Xtr), ytr)
        return accuracy_score(yte, m.predict(sc.transform(Xte)))
    m = XGBClassifier(n_estimators=120, max_depth=4, tree_method="hist",
                      n_jobs=4, verbosity=0)
    m.fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))


def _reg(Xtr, ytr, Xte, yte, linear=True):
    if linear:
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
        return r2_score(yte, m.predict(sc.transform(Xte)))
    m = XGBRegressor(n_estimators=120, max_depth=4, tree_method="hist",
                     n_jobs=4, verbosity=0)
    m.fit(Xtr, ytr)
    return r2_score(yte, m.predict(Xte))


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen-embedding probes")
    ap.add_argument("--results-dir", default="results/esm")
    ap.add_argument("--model-name", default="esm")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/probes")
    ap.add_argument("--max-chains", type=int, default=4000)
    ap.add_argument("--per-chain", type=int, default=8, help="residues sampled per chain")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    prot_dir = Path(args.structures_dir) / "proteins"
    out_dir = Path(args.out_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cath_by_id, ids = {}, []
    for l in (Path(args.structures_dir) / "index.jsonl").open():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("valid", True):
            ids.append(r["id"])
            if r.get("cath_class") is not None:
                cath_by_id[r["id"]] = int(r["cath_class"])

    print(f"[{args.model_name}] collecting up to {args.max_chains:,} chains ...")
    D = _collect(ids, res_dir, prot_dir, cath_by_id, args.max_chains,
                 args.per_chain, args.seed)
    # CATH classes are non-contiguous (e.g. {1,2,3,4,6}); XGBoost needs 0..K-1.
    D["cath"] = LabelEncoder().fit_transform(D["cath"])
    n_layers = D["res_emb"].shape[0]
    C = D["n_chains"]
    print(f"  {C:,} chains, {D['res_emb'].shape[1]:,} residues, {n_layers} layers")

    rng = np.random.default_rng(args.seed)
    test_chains = set(rng.choice(C, max(1, int(C * args.test_frac)), replace=False).tolist())
    res_test = np.isin(D["grp"], list(test_chains))
    ch_test = np.array([c in test_chains for c in range(C)])
    do_chain = C >= 12   # CATH probe only meaningful with enough chains

    labels = ["emb"] + [f"L{i}" for i in range(1, n_layers)]
    if args.model_name.startswith("proteinmpnn"):
        labels = [f"enc{i+1}" for i in range(n_layers)]

    rows = []
    for i in range(n_layers):
        Xr = D["res_emb"][i]
        b = D["burial"] >= 0
        row = {"layer": labels[i]}
        for name, y, mask, fn in [
            ("ss3", D["ss3"], np.ones_like(res_test), _clf),
            ("burial", D["burial"], b, _clf),
            ("rsa", D["rsa"], np.isfinite(D["rsa"]), _reg),
        ]:
            tr = mask & ~res_test
            te = mask & res_test
            for lin, tag in [(True, "lin"), (False, "xgb")]:
                row[f"{name}_{tag}"] = round(fn(Xr[tr], y[tr], Xr[te], y[te], linear=lin), 4)
        if do_chain:
            Xp = D["pooled"][i]
            for lin, tag in [(True, "lin"), (False, "xgb")]:
                row[f"cath_{tag}"] = round(
                    _clf(Xp[~ch_test], D["cath"][~ch_test], Xp[ch_test], D["cath"][ch_test],
                         linear=lin), 4)
        rows.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    _plot_curves(rows, labels, args.model_name, out_dir / "probe_curves.png")
    if do_chain:
        _learning_curve(D, ch_test, labels, args.model_name, args.seed,
                        out_dir / "learning_curve.png")
    print(f"-> {out_dir}")


def _plot_curves(rows, labels, model_name, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = range(len(labels))
    keys = [k for k in rows[0] if k.endswith("_xgb")]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k in keys:
        base = k[:-4]
        ax.plot(x, [r[k] for r in rows], "o-", label=f"{base} (XGB)")
        ax.plot(x, [r.get(base + "_lin", np.nan) for r in rows], "--", alpha=0.5,
                color=ax.lines[-1].get_color(), label=f"{base} (linear)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("layer"); ax.set_ylabel("accuracy / R²")
    ax.set_title(f"{model_name}: probe performance vs depth (solid XGB, dashed linear)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def _learning_curve(D, ch_test, labels, model_name, seed, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # best layer = highest cath XGB acc on full train (use last layer as proxy pick)
    n_layers = D["pooled"].shape[0]
    accs_emb, accs_base, fracs = [], [], [0.1, 0.25, 0.5, 1.0]
    tr_idx = np.where(~ch_test)[0]
    te = ch_test
    rng = np.random.default_rng(seed)
    # pick layer with best full-train xgb acc
    best_l, best_a = 0, -1
    for i in range(n_layers):
        a = _clf(D["pooled"][i][~ch_test], D["cath"][~ch_test],
                 D["pooled"][i][te], D["cath"][te], linear=False)
        if a > best_a:
            best_a, best_l = a, i
    for fr in fracs:
        k = max(4, int(len(tr_idx) * fr))
        sub = rng.choice(tr_idx, k, replace=False)
        accs_emb.append(_clf(D["pooled"][best_l][sub], D["cath"][sub],
                             D["pooled"][best_l][te], D["cath"][te], linear=False))
        accs_base.append(_clf(D["comp"][sub], D["cath"][sub],
                              D["comp"][te], D["cath"][te], linear=False))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fracs, accs_emb, "o-", label=f"{model_name} embedding ({labels[best_l]})")
    ax.plot(fracs, accs_base, "s--", label="AA-composition baseline")
    ax.set_xlabel("training-set fraction"); ax.set_ylabel("CATH-class accuracy")
    ax.set_title(f"{model_name}: data efficiency (CATH class)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
