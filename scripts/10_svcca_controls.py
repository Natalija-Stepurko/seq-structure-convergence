"""
10_svcca_controls.py — controls for the SVCCA scaling claim.

The headline result is that the shared linear subspace between the sequence model and the
structure model grows with sequence-model scale (SVCCA 0.40 -> 0.60 from ESM-2 35M to 650M).
Two confounds must be excluded before that can be attributed to convergence:

  C1  permutation baseline — SVCCA between residue-SHUFFLED representations. CCA can find
      correlated directions between unrelated high-dimensional data, so the null is not 0.
      (Our stage-04 grid baselines CKA only.)

  C2  dimension matching — 35M is 480-d and 650M is 1280-d. More dimensions give CCA more
      directions to correlate, inflating SVCCA independently of any real convergence. We
      therefore PCA-reduce BOTH representations to the same k before computing SVCCA, so the
      two scales are compared on equal footing.

For each (sequence model, structure model) pair we report, at the layer pair that peaks in the
raw grid: raw SVCCA, permuted SVCCA, and dimension-matched SVCCA (+ its permutation null) at
several k, with bootstrap CIs over residue resamples.

Outputs (under --out-dir):
    svcca_controls.csv / summary.txt / svcca_controls.png

Usage:
    uv run python scripts/10_svcca_controls.py --structures-dir /ssc/structures \\
        --pairs esm35=/ssc/results/esm esm650=/ssc/results/esm650 \\
        --struct-dir /ssc/results/proteinmpnn --out-dir /ssc/results/convergence_controls
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
import qc_common as qc


def _layers(pt):
    return torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()


def _collect_pair(ids, seq_dir, struct_dir, max_residues, per_chain, rng):
    S, T, total = [], [], 0
    for cid in ids:
        ps, pt = seq_dir / f"{cid}.pt", struct_dir / f"{cid}.pt"
        if not (ps.exists() and pt.exists()):
            continue
        A, B = _layers(ps), _layers(pt)
        if A.shape[1] != B.shape[1]:
            continue
        L = A.shape[1]
        idx = rng.choice(L, min(per_chain, L), replace=False)
        S.append(A[:, idx, :]); T.append(B[:, idx, :]); total += len(idx)
        if total >= max_residues:
            break
    if not S:
        sys.exit("ERROR: no aligned chains for this pair.")
    return np.concatenate(S, axis=1), np.concatenate(T, axis=1)


def _peak_layers(Sa, Tb, seed):
    """Find the (seq layer, struct layer) pair with max raw SVCCA."""
    best = (-1, 0, 0)
    for i in range(Sa.shape[0]):
        for j in range(Tb.shape[0]):
            v = qc.svcca(Sa[i], Tb[j], seed=seed)
            if v > best[0]:
                best = (v, i, j)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="SVCCA permutation + dimension-matched controls")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--pairs", nargs="+", required=True, help="name=seq_embeddings_dir ...")
    ap.add_argument("--struct-dir", required=True)
    ap.add_argument("--struct-name", default="mpnn")
    ap.add_argument("--out-dir", default="results/convergence_controls")
    ap.add_argument("--max-residues", type=int, default=20000)
    ap.add_argument("--per-chain", type=int, default=6)
    ap.add_argument("--dims", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--boot", type=int, default=8, help="residue resamples for CIs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    pairs = dict(p.split("=", 1) for p in args.pairs)

    rows, lines = [], ["SVCCA controls — permutation null and dimension matching", ""]
    for name, seq_dir in pairs.items():
        rng = np.random.default_rng(args.seed)
        Sa, Tb = _collect_pair(ids, Path(seq_dir), Path(args.struct_dir),
                               args.max_residues, args.per_chain, rng)
        raw, li, lj = _peak_layers(Sa, Tb, args.seed)
        A, B = Sa[li], Tb[lj]
        N = A.shape[1] if A.ndim > 1 else 0
        print(f"[{name} × {args.struct_name}] peak raw SVCCA={raw:.3f} at seq layer {li}, "
              f"struct layer {lj}; N={A.shape[0]:,} residues, dims {A.shape[1]}/{B.shape[1]}")
        lines.append(f"[{name} × {args.struct_name}]  peak layers: seq {li}, struct {lj}; "
                     f"dims {A.shape[1]}/{B.shape[1]}")

        # C1: permutation null at native dimensionality
        # SVCCA internally subsamples rows; varying its seed gives the spread of the estimate
        perms = [qc.svcca(A, B[rng.permutation(B.shape[0])], seed=args.seed + b)
                 for b in range(args.boot)]
        raws = [qc.svcca(A, B, seed=args.seed + b) for b in range(args.boot)]
        lines.append(f"  raw SVCCA          {np.mean(raws):.3f}  (permuted null {np.mean(perms):.3f} "
                     f"[{np.min(perms):.3f}, {np.max(perms):.3f}])   gap {np.mean(raws)-np.mean(perms):+.3f}")
        rows.append({"pair": name, "k": "native", "svcca": round(float(np.mean(raws)), 4),
                     "svcca_perm": round(float(np.mean(perms)), 4),
                     "gap": round(float(np.mean(raws) - np.mean(perms)), 4),
                     "dim_seq": A.shape[1], "dim_struct": B.shape[1]})

        # C2: dimension-matched — PCA both to the same k, then SVCCA (+ permutation null)
        for k in args.dims:
            kk = min(k, A.shape[1], B.shape[1], A.shape[0] - 1)
            Ak = PCA(n_components=kk, random_state=args.seed).fit_transform(A)
            Bk = PCA(n_components=kk, random_state=args.seed).fit_transform(B)
            v = [qc.svcca(Ak, Bk, seed=args.seed + b) for b in range(args.boot)]
            vp = [qc.svcca(Ak, Bk[rng.permutation(Bk.shape[0])], seed=args.seed + b)
                  for b in range(args.boot)]
            lines.append(f"  dim-matched k={kk:<4d}  {np.mean(v):.3f}  "
                         f"(permuted {np.mean(vp):.3f})   gap {np.mean(v)-np.mean(vp):+.3f}")
            rows.append({"pair": name, "k": kk, "svcca": round(float(np.mean(v)), 4),
                         "svcca_perm": round(float(np.mean(vp)), 4),
                         "gap": round(float(np.mean(v) - np.mean(vp)), 4),
                         "dim_seq": A.shape[1], "dim_struct": B.shape[1]})
            print(f"    k={kk}: svcca={np.mean(v):.3f} perm={np.mean(vp):.3f}")
        lines.append("")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "svcca_controls.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ks = [r["k"] for r in rows if r["pair"] == list(pairs)[0]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in pairs:
        sub = [r for r in rows if r["pair"] == name]
        x = range(len(sub))
        ax.plot(x, [r["svcca"] for r in sub], "o-", label=f"{name} SVCCA")
        ax.plot(x, [r["svcca_perm"] for r in sub], "s--", alpha=0.6, label=f"{name} permuted")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("PCA dimension (matched)"); ax.set_ylabel("SVCCA")
    ax.set_title("SVCCA controls: permutation null and dimension matching")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "svcca_controls.png", dpi=140); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
