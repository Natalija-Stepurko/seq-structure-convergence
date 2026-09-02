r"""15_predictivity_matrix.py — linear predictivity for every model pair (last layer each).

Adds a functional column to the structural convergence matrix. Ridge map A->B and B->A on
residue-aligned last-layer embeddings, held-out R^2, each against a residue-permutation control.
No circularity concern: this is representation-to-representation regression, not a task readout.
"""
import argparse, csv, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

PAIRS = [
    ("esm1v_s1","esm1v_s2","same arch, different seed"),
    ("carp","esm","within-sequence"),
    ("esmif1","proteinmpnn","within-structure"),
    ("esm650","proteinmpnn","cross-modality"),
    ("esm650","esmif1","cross-modality"),
    ("esm","proteinmpnn","cross-modality"),
    ("carp","proteinmpnn","cross-modality"),
    ("carp","esmif1","cross-modality"),
    ("rand_esm","rand_mpnn","untrained"),
    ("esm","rand_mpnn","one side untrained"),
    ("rand_esm","proteinmpnn","one side untrained"),
]

def load_last(p): return torch.load(p, weights_only=False)["layers"][-1].to(torch.float32).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--results-root", default="/ssc/results")
    ap.add_argument("--out-dir", default="/ssc/results/predictivity_matrix")
    ap.add_argument("--n-chains", type=int, default=500)
    ap.add_argument("--per-chain", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root = Path(a.results_root); sdir = Path(a.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir/"index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    rows = []
    for A, B, kind in PAIRS:
        da, db = root/A, root/B
        if not (da.exists() and db.exists()): print(f"  [skip {A}x{B}]"); continue
        rng = np.random.default_rng(a.seed)
        XA, XB = [], []
        for c in ids[:a.n_chains*3]:
            fa, fb = da/f"{c}.pt", db/f"{c}.pt"
            if not (fa.exists() and fb.exists()): continue
            ea, eb = load_last(fa), load_last(fb)
            if ea.shape[0] != eb.shape[0]: continue
            i = rng.choice(ea.shape[0], min(a.per_chain, ea.shape[0]), replace=False)
            XA.append(ea[i]); XB.append(eb[i])
            if sum(x.shape[0] for x in XA) >= a.n_chains*a.per_chain: break
        XA, XB = np.concatenate(XA), np.concatenate(XB)
        n = len(XA); te = np.zeros(n, bool); te[rng.choice(n, n//4, replace=False)] = True; tr = ~te
        out = {"pair": f"{A} x {B}", "type": kind, "n_res": n}
        for src, dst, tag in [(XA, XB, "fwd"), (XB, XA, "rev")]:
            m = Ridge(alpha=a.alpha).fit(src[tr], dst[tr])
            out[f"r2_{tag}"] = round(float(r2_score(dst[te], m.predict(src[te]),
                                                    multioutput="variance_weighted")), 4)
            pm = rng.permutation(tr.sum())
            mp = Ridge(alpha=a.alpha).fit(src[tr], dst[tr][pm])
            out[f"r2_{tag}_perm"] = round(float(r2_score(dst[te], mp.predict(src[te]),
                                                         multioutput="variance_weighted")), 4)
        rows.append(out)
        print(f"  {out['pair']:26s} {kind:26s} fwd {out['r2_fwd']:+.3f} (perm {out['r2_fwd_perm']:+.3f})"
              f"  rev {out['r2_rev']:+.3f} (perm {out['r2_rev_perm']:+.3f})  n={n}")
    o = Path(a.out_dir); o.mkdir(parents=True, exist_ok=True)
    with (o/"predictivity_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {o}")

if __name__ == "__main__": main()
