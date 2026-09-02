r"""
12_linear_stitch.py — functional similarity by linear cross-prediction ("stitching-lite").

Structural similarity measures (CKA, SVCCA) are criticised for being agnostic to functional
behaviour; the standard remedy is model stitching — can one model's representation be linearly
mapped into another's and still work? This stage implements a cheap, directional version:

  1. REPRESENTATION TRANSFER: ridge from model A's per-residue embedding to model B's, held-out R².
     "How much of B is linearly recoverable from A?" Reported both directions, since A->B and
     B->A need not agree — the asymmetry says which representation contains the other.

  2. FUNCTIONAL TRANSFER: train a probe on B's real embedding, then apply it to A-mapped-into-B
     (i.e. to \hat{B} = A W). Retained accuracy relative to the probe on real B is the functional
     criterion: a mapping that preserves the decision is a functionally adequate stitch.

Both are computed on residue-aligned embeddings at each model's best-converging layer, with a
residue-permutation control for (1).

Outputs (under --out-dir):
    stitch.csv / summary.txt

Usage:
    uv run python scripts/12_linear_stitch.py --structures-dir /ssc/structures \\
        --pairs esm=/ssc/results/esm mpnn=/ssc/results/proteinmpnn --out-dir /ssc/results/stitch
"""

import argparse
import csv
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SS3_IDX = {"a": 0, "b": 1, "c": 2}


def _collect(ids, dirs, prot, max_res, per_chain, seed):
    rng = np.random.default_rng(seed)
    emb = {k: [] for k in dirs}
    ss3 = []
    total = 0
    for cid in ids:
        if not all((Path(d) / f"{cid}.pt").exists() for d in dirs.values()):
            continue
        npz = np.load(prot / f"{cid}.npz", allow_pickle=True)
        L = int(npz["ss3"].shape[0])
        E = {k: torch.load(Path(d) / f"{cid}.pt", weights_only=False)["layers"]
             .to(torch.float32).numpy() for k, d in dirs.items()}
        if any(v.shape[1] != L for v in E.values()):
            continue
        idx = rng.choice(L, min(per_chain, L), replace=False)
        for k in dirs:
            emb[k].append(E[k][:, idx, :])
        ss3.append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        total += len(idx)
        if total >= max_res:
            break
    return ({k: np.concatenate(v, axis=1) for k, v in emb.items()},
            np.concatenate(ss3))


def _best_layers(A, B):
    """Pick the layer pair maximising linear cross-predictability (cheap proxy: last layers)."""
    return A.shape[0] - 1, B.shape[0] - 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear cross-prediction / stitching-lite")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--pairs", nargs="+", required=True, help="name=dir ...")
    ap.add_argument("--out-dir", default="results/stitch")
    ap.add_argument("--max-residues", type=int, default=20000)
    ap.add_argument("--per-chain", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dirs = dict(p.split("=", 1) for p in args.pairs)
    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    emb, ss3 = _collect(ids, dirs, sdir / "proteins", args.max_residues, args.per_chain, args.seed)
    n = next(iter(emb.values())).shape[1]
    print(f"{n:,} aligned residues across {len(dirs)} models")

    rng = np.random.default_rng(args.seed)
    te = np.zeros(n, bool); te[rng.choice(n, int(n * args.test_frac), replace=False)] = True
    tr = ~te

    rows = []
    for a, b in combinations(dirs, 2):
        ia, ib = _best_layers(emb[a], emb[b])
        A, B = emb[a][ia], emb[b][ib]
        As = StandardScaler().fit(A[tr]); Bs = StandardScaler().fit(B[tr])
        At, Bt = As.transform(A), Bs.transform(B)

        for src, dst, sa, da in [(At, Bt, a, b), (Bt, At, b, a)]:
            m = Ridge(alpha=args.alpha).fit(src[tr], dst[tr])
            pred = m.predict(src[te])
            r2 = r2_score(dst[te], pred, multioutput="variance_weighted")
            # permutation control: shuffle the residue correspondence
            perm = rng.permutation(tr.sum())
            mp = Ridge(alpha=args.alpha).fit(src[tr], dst[tr][perm])
            r2p = r2_score(dst[te], mp.predict(src[te]), multioutput="variance_weighted")

            # functional transfer: probe trained on real dst, applied to mapped src
            clf = XGBClassifier(n_estimators=80, max_depth=4, tree_method="hist",
                                n_jobs=4, verbosity=0).fit(dst[tr], ss3[tr])
            acc_real = accuracy_score(ss3[te], clf.predict(dst[te]))
            acc_map = accuracy_score(ss3[te], clf.predict(pred))
            rows.append({"source": sa, "target": da,
                         "transfer_r2": round(float(r2), 4),
                         "transfer_r2_perm": round(float(r2p), 4),
                         "probe_acc_real": round(float(acc_real), 4),
                         "probe_acc_mapped": round(float(acc_map), 4),
                         "retained": round(float(acc_map / acc_real), 4)})
            print(f"  {sa:>10s} -> {da:<10s} R²={r2:.3f} (perm {r2p:.3f})  "
                  f"probe {acc_real:.3f} -> {acc_map:.3f} ({acc_map/acc_real:.0%} retained)")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "stitch.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = ["Linear cross-prediction (stitching-lite): how much of the target representation is",
             "linearly recoverable from the source, and does a probe survive the mapping?", ""]
    for r in rows:
        lines.append(f"  {r['source']:>10s} -> {r['target']:<10s} "
                     f"transfer R²={r['transfer_r2']:.3f} (perm {r['transfer_r2_perm']:.3f})  "
                     f"probe {r['probe_acc_real']:.3f} -> {r['probe_acc_mapped']:.3f} "
                     f"({r['retained']:.0%} retained)")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
