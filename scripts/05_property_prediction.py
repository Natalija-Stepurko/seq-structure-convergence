"""
05_property_prediction.py — supervised probes on frozen embeddings.

Linear and XGBoost probes on each layer's frozen embeddings, decoding labels the model was
never trained on, with chain-grouped train/test splits (no residue leakage). Tests
transferability, the depth law (P3), the XGB−linear (non-linear) gap, and data efficiency (P5).

Residue-level targets (per-residue embeddings):
    SSE (3-class acc), burial (binary acc), RSA (regression R²)
Chain-level targets (mean-pooled embeddings) — structural + the UniProt annotations from
stage 01b (function / localisation / taxonomy / family / PTM), each filtered to classes with
enough support and scored by accuracy and macro-F1:
    cath_class, cath_arch (fold/architecture)      [manifest]
    ec_class, enzyme (function)                    [annotations]
    localisation, kingdom                          [annotations]
    is_phospho, is_glyco (PTM proxies)             [annotations]
Learning curves: CATH-class and function (EC) accuracy vs training-set size, frozen
embedding (best layer) vs an amino-acid-composition baseline.

Outputs (under --out-dir/<model-name>):
    metrics.csv          per-layer residue-level metrics
    chain_metrics.csv    per (target, layer) chain-level acc + macro-F1 (lin & xgb)
    probe_curves.png     residue metrics vs depth
    chain_best_layer.png best-layer acc/F1 per chain target
    learning_curve.png   data efficiency

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
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
BURIAL_RSA = 0.25
SS3_IDX = {"a": 0, "b": 1, "c": 2}

# chain-level targets: name -> (source, key). source in {"manifest","annot"}
CHAIN_TARGETS = [
    # structural / fold-family (from CATH, already in manifest)
    ("cath_class", "manifest", "cath_class"),
    ("cath_arch", "manifest", "cath_arch"),
    ("cath_topol", "manifest", "cath_topol"),
    # function
    ("ec_class", "annot", "ec_class"),
    ("enzyme", "annot", "enzyme"),
    ("protein_class", "annot", "protein_class"),
    # localisation / taxonomy
    ("localisation", "annot", "localisation"),
    ("kingdom", "annot", "kingdom"),
    # functional protein classes (UniProt keyword one-vs-rest)
    ("is_transport", "annot", "is_transport"),
    ("is_dna_binding", "annot", "is_dna_binding"),
    ("is_rna_binding", "annot", "is_rna_binding"),
    ("is_kinase", "annot", "is_kinase"),
    ("is_ribosomal", "annot", "is_ribosomal"),
    ("is_membrane_protein", "annot", "is_membrane_protein"),
    ("is_structural", "annot", "is_structural"),
    ("is_immune", "annot", "is_immune"),
    # PTM proxies
    ("is_phospho", "annot", "is_phospho"),
    ("is_glyco", "annot", "is_glyco"),
]
MIN_CLASS_COUNT = 40   # drop chains in classes rarer than this before probing a target
MAX_CLASSES = 12       # cap high-cardinality targets to top classes (bounds XGB multiclass cost)
# label values that mean "no annotation" and must be excluded from a target's probe
EXCLUDE_VALUES = {"localisation": {"unknown", "other"}, "kingdom": {"other"}}


def _load_labels(structures_dir: Path):
    manifest = {}
    for l in (structures_dir / "index.jsonl").open():
        if l.strip():
            r = json.loads(l)
            if r.get("valid", True):
                manifest[r["id"]] = r
    annot = {}
    ann_path = structures_dir / "annotations.jsonl"
    if ann_path.exists():
        for l in ann_path.open():
            if l.strip():
                a = json.loads(l)
                annot[a["id"]] = a
    return manifest, annot


def _collect(ids, res_dir, prot_dir, manifest, annot, max_chains, per_chain, seed):
    rng = np.random.default_rng(seed)
    res_emb, ss3, burial, rsa, grp = [], [], [], [], []
    pooled, comp = [], []
    chain_lab = {name: [] for name, _, _ in CHAIN_TARGETS}
    g = 0
    for cid in ids:
        pt = res_dir / f"{cid}.pt"
        if not pt.exists():
            continue
        npz = np.load(prot_dir / f"{cid}.npz", allow_pickle=True)
        seq = str(npz["seq"]); L = int(npz["ss3"].shape[0])
        E = torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()
        if E.shape[1] != L:
            continue
        pooled.append(E.mean(axis=1))
        cc = np.zeros(20)
        for a in seq:
            if a in AA_IDX:
                cc[AA_IDX[a]] += 1
        comp.append(cc / max(1, len(seq)))
        for name, src, key in CHAIN_TARGETS:
            d = manifest[cid] if src == "manifest" else annot.get(cid, {})
            chain_lab[name].append(d.get(key, None))
        idx = rng.choice(L, per_chain, replace=False) if L > per_chain else np.arange(L)
        res_emb.append(E[:, idx, :])
        ss3.append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        r = npz["rsa"][idx].astype(np.float64)
        burial.append(np.where(np.isfinite(r), (r < BURIAL_RSA).astype(int), -1))
        rsa.append(r); grp.append(np.full(len(idx), g)); g += 1
        if g >= max_chains:
            break
    if g < 12:
        sys.exit("ERROR: too few chains with embeddings.")
    return {
        "res_emb": np.concatenate(res_emb, axis=1),
        "ss3": np.concatenate(ss3), "burial": np.concatenate(burial),
        "rsa": np.concatenate(rsa), "grp": np.concatenate(grp),
        "pooled": np.stack(pooled, axis=1),
        "chain_lab": {k: np.array(v, dtype=object) for k, v in chain_lab.items()},
        "comp": np.array(comp), "n_chains": g,
    }


def _clf(Xtr, ytr, Xte, yte, linear=True):
    if linear:
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
        pred = m.predict(sc.transform(Xte))
    else:
        m = XGBClassifier(n_estimators=80, max_depth=4, tree_method="hist",
                          n_jobs=4, verbosity=0).fit(Xtr, ytr)
        pred = m.predict(Xte)
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def _reg(Xtr, ytr, Xte, yte, linear=True):
    if linear:
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
        return r2_score(yte, m.predict(sc.transform(Xte)))
    m = XGBRegressor(n_estimators=80, max_depth=4, tree_method="hist",
                     n_jobs=4, verbosity=0).fit(Xtr, ytr)
    return r2_score(yte, m.predict(Xte))


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen-embedding probes")
    ap.add_argument("--results-dir", default="results/esm")
    ap.add_argument("--model-name", default="esm")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/probes")
    ap.add_argument("--max-chains", type=int, default=5000)
    ap.add_argument("--per-chain", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    prot_dir = Path(args.structures_dir) / "proteins"
    out_dir = Path(args.out_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, annot = _load_labels(Path(args.structures_dir))
    ids = list(manifest.keys())
    print(f"[{args.model_name}] {len(annot):,} chains have UniProt annotations; collecting ...")
    D = _collect(ids, res_dir, prot_dir, manifest, annot, args.max_chains,
                 args.per_chain, args.seed)
    n_layers = D["res_emb"].shape[0]; C = D["n_chains"]
    print(f"  {C:,} chains, {D['res_emb'].shape[1]:,} residues, {n_layers} layers")

    rng = np.random.default_rng(args.seed)
    test_chains = set(rng.choice(C, max(1, int(C * args.test_frac)), replace=False).tolist())
    res_test = np.isin(D["grp"], list(test_chains))
    ch_test = np.array([c in test_chains for c in range(C)])

    labels = ["emb"] + [f"L{i}" for i in range(1, n_layers)]
    if args.model_name.startswith("proteinmpnn"):
        labels = [f"enc{i+1}" for i in range(n_layers)]

    # ---- residue-level probes ----
    rows = []
    for i in range(n_layers):
        Xr = D["res_emb"][i]; b = D["burial"] >= 0
        row = {"layer": labels[i]}
        (row["ss3_lin_acc"], _) = _clf(Xr[~res_test], D["ss3"][~res_test],
                                       Xr[res_test], D["ss3"][res_test], linear=True)
        (row["ss3_xgb_acc"], _) = _clf(Xr[~res_test], D["ss3"][~res_test],
                                       Xr[res_test], D["ss3"][res_test], linear=False)
        trb = b & ~res_test; teb = b & res_test
        (row["burial_lin_acc"], _) = _clf(Xr[trb], D["burial"][trb], Xr[teb], D["burial"][teb], True)
        (row["burial_xgb_acc"], _) = _clf(Xr[trb], D["burial"][trb], Xr[teb], D["burial"][teb], False)
        fr = np.isfinite(D["rsa"])
        row["rsa_lin_r2"] = _reg(Xr[fr & ~res_test], D["rsa"][fr & ~res_test],
                                 Xr[fr & res_test], D["rsa"][fr & res_test], True)
        row["rsa_xgb_r2"] = _reg(Xr[fr & ~res_test], D["rsa"][fr & ~res_test],
                                 Xr[fr & res_test], D["rsa"][fr & res_test], False)
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- chain-level probes (structural + annotations) ----
    chain_rows = []
    for name, _, _ in CHAIN_TARGETS:
        y_raw = D["chain_lab"][name]
        excl = EXCLUDE_VALUES.get(name, set())
        have = np.array([v is not None and str(v) not in excl for v in y_raw])
        if have.sum() < 50:
            print(f"  [skip {name}] only {have.sum()} labelled chains"); continue
        yv = y_raw[have]
        vals, counts = np.unique(yv.astype(str), return_counts=True)
        ok = counts >= MIN_CLASS_COUNT
        vals, counts = vals[ok], counts[ok]
        # keep only the top-MAX_CLASSES most frequent classes (bounds multiclass XGB cost)
        keep_classes = set(vals[np.argsort(-counts)][:MAX_CLASSES])
        keep = have & np.array([str(v) in keep_classes for v in y_raw])
        if len(keep_classes) < 2 or keep.sum() < 60:
            print(f"  [skip {name}] <2 classes with >= {MIN_CLASS_COUNT} support"); continue
        y = LabelEncoder().fit_transform(y_raw[keep].astype(str))
        tr = keep & ~ch_test; te = keep & ch_test
        if len(set(y[np.isin(np.where(keep)[0], np.where(tr)[0])])) < 2:
            pass  # guard; proceed regardless
        ytr = LabelEncoder().fit(y_raw[keep].astype(str)).transform(y_raw[tr].astype(str))
        yte = LabelEncoder().fit(y_raw[keep].astype(str)).transform(y_raw[te].astype(str))
        print(f"  [{name}] {keep.sum()} chains, {len(keep_classes)} classes")
        for i in range(n_layers):
            Xp = D["pooled"][i]
            la, lf = _clf(Xp[tr], ytr, Xp[te], yte, linear=True)
            xa, xf = _clf(Xp[tr], ytr, Xp[te], yte, linear=False)
            chain_rows.append({"target": name, "layer": labels[i], "n_classes": len(keep_classes),
                               "lin_acc": round(la, 4), "lin_f1": round(lf, 4),
                               "xgb_acc": round(xa, 4), "xgb_f1": round(xf, 4)})
    if chain_rows:
        with (out_dir / "chain_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(chain_rows[0].keys()))
            w.writeheader(); w.writerows(chain_rows)

    _plot_residue(rows, labels, args.model_name, out_dir / "probe_curves.png")
    if chain_rows:
        _plot_chain_best(chain_rows, args.model_name, out_dir / "chain_best_layer.png")
        _plot_chain_heatmap(chain_rows, labels, args.model_name, out_dir / "chain_accuracy_heatmap.png")
        _learning_curve(D, ch_test, chain_rows, labels, args.model_name, args.seed,
                        out_dir / "learning_curve.png")
    print(f"-> {out_dir}")


def _plot_chain_heatmap(chain_rows, labels, model_name, out_png):
    """Property × layer probe-accuracy heatmaps (linear and XGB) — supervised analog of
    the k-NN purity-across-layers view."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = sorted({r["target"] for r in chain_rows},
                     key=lambda t: -max(r["xgb_acc"] for r in chain_rows if r["target"] == t))
    lay = [l for l in labels if any(r["layer"] == l for r in chain_rows)]
    def _mat(key):
        M = np.full((len(targets), len(lay)), np.nan)
        for r in chain_rows:
            M[targets.index(r["target"]), lay.index(r["layer"])] = r[key]
        return M
    fig, axes = plt.subplots(1, 2, figsize=(1.0 * len(lay) + 4, 0.5 * len(targets) + 2))
    for ax, (key, title) in zip(axes, [("lin_acc", "linear"), ("xgb_acc", "XGBoost")]):
        M = _mat(key)
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(lay))); ax.set_xticklabels(lay, rotation=45, ha="right")
        ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets)
        ax.set_title(f"{model_name}: {title} probe accuracy")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if np.isfinite(M[a, b]):
                    ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center",
                            color="white" if M[a, b] < 0.55 else "black", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def _learning_curve(D, ch_test, chain_rows, labels, model_name, seed, out_png):
    """Data efficiency: accuracy vs training-set size, embedding vs AA-composition baseline,
    for up to two targets (CATH class and, if present, function/EC class)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    targets = [t for t in ["cath_class", "ec_class"]
               if any(r["target"] == t for r in chain_rows)]
    if not targets:
        return
    fracs = [0.1, 0.25, 0.5, 1.0]
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), squeeze=False)
    for ax, tname in zip(axes[0], targets):
        y_raw = D["chain_lab"][tname]
        have = np.array([v is not None for v in y_raw])
        vals, counts = np.unique(y_raw[have].astype(str), return_counts=True)
        keep_cls = set(vals[counts >= MIN_CLASS_COUNT])
        keep = have & np.array([str(v) in keep_cls for v in y_raw])
        enc = LabelEncoder().fit(y_raw[keep].astype(str))
        best = max((r for r in chain_rows if r["target"] == tname), key=lambda r: r["xgb_acc"])
        li = labels.index(best["layer"])
        tr = np.where(keep & ~ch_test)[0]; te = keep & ch_test
        yte = enc.transform(y_raw[te].astype(str))
        emb_te = D["pooled"][li][te]; base_te = D["comp"][te]
        acc_emb, acc_base = [], []
        for fr in fracs:
            k = max(10, int(len(tr) * fr)); sub = rng.choice(tr, k, replace=False)
            ysub = enc.transform(y_raw[sub].astype(str))
            a1, _ = _clf(D["pooled"][li][sub], ysub, emb_te, yte, linear=False)
            a2, _ = _clf(D["comp"][sub], ysub, base_te, yte, linear=False)
            acc_emb.append(a1); acc_base.append(a2)
        ax.plot(fracs, acc_emb, "o-", label=f"{model_name} embedding ({best['layer']})")
        ax.plot(fracs, acc_base, "s--", label="AA-composition baseline")
        ax.set_xlabel("training-set fraction"); ax.set_ylabel("accuracy")
        ax.set_title(f"{tname} data efficiency"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def _plot_residue(rows, labels, model_name, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for base in ["ss3", "burial", "rsa"]:
        suf = "acc" if base != "rsa" else "r2"
        ax.plot(x, [r[f"{base}_xgb_{suf}"] for r in rows], "o-", label=f"{base} (XGB)")
        ax.plot(x, [r[f"{base}_lin_{suf}"] for r in rows], "--", alpha=0.5,
                color=ax.lines[-1].get_color(), label=f"{base} (linear)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("layer"); ax.set_ylabel("accuracy / R²")
    ax.set_title(f"{model_name}: residue-level probes vs depth"); ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def _plot_chain_best(chain_rows, model_name, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    targets = sorted({r["target"] for r in chain_rows})
    best_acc, best_f1 = [], []
    for t in targets:
        rs = [r for r in chain_rows if r["target"] == t]
        best = max(rs, key=lambda r: r["xgb_acc"])
        best_acc.append(best["xgb_acc"]); best_f1.append(best["xgb_f1"])
    y = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(targets) + 2))
    ax.barh(y - 0.2, best_acc, 0.4, label="accuracy")
    ax.barh(y + 0.2, best_f1, 0.4, label="macro-F1")
    ax.set_yticks(y); ax.set_yticklabels(targets); ax.set_xlabel("best-layer XGB score")
    ax.set_title(f"{model_name}: chain-level property decodability"); ax.legend()
    ax.grid(alpha=0.3, axis="x"); fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
