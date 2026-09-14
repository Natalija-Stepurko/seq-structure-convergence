"""Every similarity metric against its OWN permutation null, on one common sample.

The three metrics are usually quoted as if they were on a common scale. They are not. Shuffling
the residues of one model destroys all correspondence between the two representations, so a
metric that still scores high on shuffled data is reporting something other than convergence.

That null is negligible for CKA and mutual k-NN and very large for SVCCA, which means raw SVCCA
numbers are not comparable either across metrics or across models of different width. Measuring
all three on the same residues, at the same budget, at each pair's own peak layers, is the only
way to say how much agreement is actually there.

Writes results/null_corrected/null_corrected.csv.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/ssc/scripts")
import qc_common as qc

AA = "ACDEFGHIKLMNPQRSTVWY"

# label, dir A, dir B, what the comparison is for
PAIRS = [
    ("ESM-1v s1 x ESM-1v s2",     "esm1v_s1", "esm1v_s2",    "ceiling: same model, different seed"),
    ("CARP x ESM-2",              "carp",     "esm",         "within-sequence"),
    ("ESM-IF1 x ProteinMPNN",     "esmif1",   "proteinmpnn", "within-structure"),
    ("ESM-2 35M x ProteinMPNN",   "esm",      "proteinmpnn", "cross-modality (headline)"),
    ("ESM-2 650M x ProteinMPNN",  "esm650",   "proteinmpnn", "cross-modality"),
    ("untrained seq x trained str", "rand_esm", "proteinmpnn", "floor: donor carries no training"),
    ("untrained x untrained",     "rand_esm", "rand_mpnn",   "floor: neither trained"),
]


def load(p):
    return torch.load(p, weights_only=False)["layers"].to(torch.float32).numpy()


def collect(ids, da, db, prot, budget, seed):
    rng = np.random.default_rng(seed)
    A, B, keep = [], [], 0
    for cid in ids:
        pa, pb = da / f"{cid}.pt", db / f"{cid}.pt"
        if not (pa.exists() and pb.exists()):
            continue
        Xa, Xb = load(pa), load(pb)
        if Xa.shape[1] != Xb.shape[1]:
            continue
        L = Xa.shape[1]
        cap = max(1, min(L, budget // 50))
        idx = rng.choice(L, cap, replace=False) if L > cap else np.arange(L)
        A.append(Xa[:, idx, :]); B.append(Xb[:, idx, :]); keep += len(idx)
        if keep >= budget:
            break
    return np.concatenate(A, 1), np.concatenate(B, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--results-root", default="/ssc/results")
    ap.add_argument("--budget", type=int, default=15000)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--out-dir", default="/ssc/results/null_corrected")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    rng = np.random.default_rng(args.seed)
    rows = []

    for label, an, bn, kind in PAIRS:
        da, db = Path(args.results_root) / an, Path(args.results_root) / bn
        if not (da.exists() and db.exists()):
            print(f"  {label:<30} SKIP (missing embeddings)", flush=True)
            continue
        A, B = collect(ids, da, db, sdir / "proteins", args.budget, args.seed)
        n = A.shape[1]

        # prepare every layer once; see qc.svcca_reduce for why this is split out
        prep = {"cka": qc.column_center, "svcca": qc.svcca_reduce, "mutual_knn": lambda x: x}
        pairf = {"cka": qc.linear_cka, "svcca": qc.svcca_from_reduced, "mutual_knn": qc.mutual_knn}

        for metric in ("cka", "svcca", "mutual_knn"):
            Ap = [prep[metric](A[i].astype(np.float64)) for i in range(A.shape[0])]
            Bp = [prep[metric](B[j].astype(np.float64)) for j in range(B.shape[0])]
            best, at = -9.0, None
            for i in range(len(Ap)):
                for j in range(len(Bp)):
                    v = pairf[metric](Ap[i], Bp[j])
                    if np.isfinite(v) and v > best:
                        best, at = float(v), (i, j)
            i, j = at

            # NULL: permute the residues of ONE side. Every row still comes from a real protein
            # and the marginal distribution of each representation is untouched -- only the
            # correspondence between them is destroyed. Whatever the metric still reports is what
            # it scores for two unrelated representations of this shape and size.
            nulls = []
            Ai = A[i].astype(np.float64)
            Bj = B[j].astype(np.float64)
            for _ in range(args.n_perm):
                perm = rng.permutation(n)
                v = pairf[metric](prep[metric](Ai), prep[metric](Bj[perm]))
                if np.isfinite(v):
                    nulls.append(float(v))
            nm = statistics.fmean(nulls)
            nsd = statistics.stdev(nulls) if len(nulls) > 1 else 0.0
            rows.append({"pair": label, "kind": kind, "metric": metric,
                         "observed": round(best, 4), "null_mean": round(nm, 4),
                         "null_sd": round(nsd, 4), "above_null": round(best - nm, 4),
                         "null_share_pct": round(100 * nm / best, 1) if best > 0 else float("nan"),
                         "peak_at": f"A{i}xB{j}", "n_residues": int(n),
                         "dim_a": int(A.shape[2]), "dim_b": int(B.shape[2]),
                         "n_perm": len(nulls)})
            print(f"  {label:<28} {metric:<11} obs {best:.4f}  null {nm:.4f}"
                  f"  ABOVE NULL {best-nm:+.4f}  ({100*nm/max(best,1e-9):.0f}% of the raw score "
                  f"is null)", flush=True)
            with (out / "null_corrected.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    qc.record_params(out, args, extra={"n_pairs": len({r['pair'] for r in rows})})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
