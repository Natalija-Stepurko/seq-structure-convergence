"""
05b_composition_baseline.py — amino-acid-composition-only baseline for whole-protein targets.

Separates genuinely emergent, representation-dependent content from what is trivially readable
from amino-acid composition. Runs the identical chain-grouped probe as stage 05, but on a 20-D
amino-acid-fraction feature vector instead of a model embedding — so it is model-independent.
Report Δ = (best embedding score) − (composition score) per target.

Uses the same chain set and split as stage 05 (chains with an ESM .pt present, seed 42) so the
composition scores are on the same test chains as the embedding probes.

Output (under --out-dir):
    composition_metrics.csv   per target: n_classes, comp_lin_acc/f1, comp_xgb_acc/f1

Usage:
    uv run python scripts/05b_composition_baseline.py --structures-dir /ssc/structures \\
        --ref-results-dir /ssc/results/esm --out-dir /ssc/results/probes
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# kept in sync with 05_property_prediction.py
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
MIN_CLASS_COUNT = 40
MAX_CLASSES = 12
EXCLUDE_VALUES = {"localisation": {"unknown", "other"}, "kingdom": {"other"}}
CHAIN_TARGETS = [
    ("cath_class", "manifest"), ("cath_arch", "manifest"), ("cath_topol", "manifest"),
    ("ec_class", "annot"), ("enzyme", "annot"), ("protein_class", "annot"),
    ("localisation", "annot"), ("kingdom", "annot"),
    ("is_transport", "annot"), ("is_dna_binding", "annot"), ("is_rna_binding", "annot"),
    ("is_kinase", "annot"), ("is_ribosomal", "annot"), ("is_membrane_protein", "annot"),
    ("is_structural", "annot"), ("is_immune", "annot"),
    ("is_phospho", "annot"), ("is_glyco", "annot"),
]


def _boot_ci(yte, pred, n_boot=1000, seed=42):
    """95% bootstrap CI on accuracy and macro-F1 by resampling test items."""
    rng = np.random.default_rng(seed)
    n = len(yte)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(yte[idx], pred[idx]))
        f1s.append(f1_score(yte[idx], pred[idx], average="macro"))
    return (np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5]))


def _clf(Xtr, ytr, Xte, yte, linear):
    if linear:
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
        p = m.predict(sc.transform(Xte))
    else:
        m = XGBClassifier(n_estimators=80, max_depth=4, tree_method="hist",
                          n_jobs=4, verbosity=0).fit(Xtr, ytr)
        p = m.predict(Xte)
    acc, f1 = accuracy_score(yte, p), f1_score(yte, p, average="macro")
    acc_ci, f1_ci = _boot_ci(np.asarray(yte), np.asarray(p))
    return acc, f1, acc_ci, f1_ci


def main() -> None:
    ap = argparse.ArgumentParser(description="AA-composition baseline for chain targets")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--ref-results-dir", default="results/esm",
                    help="embeddings dir defining the chain set/order (match stage 05)")
    ap.add_argument("--out-dir", default="results/probes")
    ap.add_argument("--max-chains", type=int, default=5000)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sdir = Path(args.structures_dir); prot = sdir / "proteins"
    ref = Path(args.ref_results_dir)
    manifest = {json.loads(l)["id"]: json.loads(l)
                for l in (sdir / "index.jsonl").open() if l.strip()}
    annot = {}
    if (sdir / "annotations.jsonl").exists():
        annot = {json.loads(l)["id"]: json.loads(l)
                 for l in (sdir / "annotations.jsonl").open() if l.strip()}

    comp, chain_lab, g = [], {n: [] for n, _ in CHAIN_TARGETS}, 0
    for cid, r in manifest.items():
        if not r.get("valid", True) or not (ref / f"{cid}.pt").exists():
            continue
        seq = str(np.load(prot / f"{cid}.npz", allow_pickle=True)["seq"])
        cc = np.zeros(20)
        for a in seq:
            if a in AA_IDX:
                cc[AA_IDX[a]] += 1
        comp.append(cc / max(1, len(seq)))
        for name, src in CHAIN_TARGETS:
            d = manifest[cid] if src == "manifest" else annot.get(cid, {})
            chain_lab[name].append(d.get(name))
        g += 1
        if g >= args.max_chains:
            break
    comp = np.array(comp); C = g
    print(f"composition baseline over {C:,} chains")

    rng = np.random.default_rng(args.seed)
    test = np.zeros(C, bool)
    test[rng.choice(C, max(1, int(C * args.test_frac)), replace=False)] = True

    rows = []
    for name, _ in CHAIN_TARGETS:
        y_raw = np.array(chain_lab[name], dtype=object)
        excl = EXCLUDE_VALUES.get(name, set())
        have = np.array([v is not None and str(v) not in excl for v in y_raw])
        vals, cnt = np.unique(y_raw[have].astype(str), return_counts=True)
        ok = cnt >= MIN_CLASS_COUNT; vals, cnt = vals[ok], cnt[ok]
        keep = set(vals[np.argsort(-cnt)][:MAX_CLASSES])
        m = have & np.array([str(v) in keep for v in y_raw])
        if len(keep) < 2 or m.sum() < 60:
            continue
        enc = LabelEncoder().fit(y_raw[m].astype(str))
        tr = m & ~test; te = m & test
        ytr, yte = enc.transform(y_raw[tr].astype(str)), enc.transform(y_raw[te].astype(str))
        la, lf, _, lf_ci = _clf(comp[tr], ytr, comp[te], yte, True)
        xa, xf, xa_ci, xf_ci = _clf(comp[tr], ytr, comp[te], yte, False)
        rows.append({"target": name, "n_classes": len(keep), "n_test": int(te.sum()),
                     "comp_lin_f1": round(lf, 4),
                     "comp_lin_f1_lo": round(lf_ci[0], 4), "comp_lin_f1_hi": round(lf_ci[1], 4),
                     "comp_xgb_acc": round(xa, 4),
                     "comp_xgb_acc_lo": round(xa_ci[0], 4), "comp_xgb_acc_hi": round(xa_ci[1], 4),
                     "comp_xgb_f1": round(xf, 4),
                     "comp_xgb_f1_lo": round(xf_ci[0], 4), "comp_xgb_f1_hi": round(xf_ci[1], 4)})
        print(f"  {name:20s} comp xgb F1={xf:.3f} [{xf_ci[0]:.3f}, {xf_ci[1]:.3f}] ({len(keep)} cls)")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "composition_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {out/'composition_metrics.csv'}")


if __name__ == "__main__":
    main()
