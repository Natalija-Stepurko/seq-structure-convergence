"""
08_embedding_health.py — per-layer embedding-health / geometry QC.

Characterises each layer's representation geometry, because a
collapsed or highly anisotropic layer inflates subspace-overlap metrics (SVCCA) while degrading
distance/neighbourhood metrics (mutual k-NN) — which is exactly the dissociation at the centre of
this paper. Per layer, on sampled residue embeddings:

    mean_cosine   mean pairwise cosine similarity of raw vectors (collapse: -> 1 when all align)
    cond_number   covariance condition number  lambda_max / lambda_min  (anisotropy)
    pca90         number of principal components for 90% of variance (effective dimensionality)
    eff_rank      participation ratio (sum l)^2 / sum l^2 (smooth effective dimensionality)

Outputs (under --out-dir/<model-name>):
    health.csv    per-layer metrics
    health.png    effective dimensionality / anisotropy / cosine vs depth

Usage:
    uv run python scripts/08_embedding_health.py --results-dir /ssc/results/esm650 \\
        --model-name esm650 --structures-dir /ssc/structures --out-dir /ssc/results/health
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _collect(ids, res_dir, prot_dir, max_residues, per_chain, seed):
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    for cid in ids:
        pt = res_dir / f"{cid}.pt"
        if not pt.exists():
            continue
        E = torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()  # [n,L,D]
        L = E.shape[1]
        idx = rng.choice(L, min(per_chain, L), replace=False)
        parts.append(E[:, idx, :]); total += len(idx)
        if total >= max_residues:
            break
    if not parts:
        sys.exit("ERROR: no embeddings found.")
    return np.concatenate(parts, axis=1)   # [n_layers, N, D]


def _health(X):
    """X: [N, D] sampled embeddings for one layer."""
    N = X.shape[0]
    # mean pairwise cosine on a sub-sample (raw vectors)
    s = X[np.random.default_rng(0).choice(N, min(N, 2000), replace=False)]
    sn = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
    C = sn @ sn.T
    mean_cos = float((C.sum() - np.trace(C)) / (C.shape[0] * (C.shape[0] - 1)))
    # covariance spectrum
    Xc = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = (sv ** 2) / max(1, len(X) - 1)
    lam = lam[lam > 1e-12]
    cond = float(lam[0] / lam[-1]) if len(lam) else float("nan")
    ev = np.cumsum(lam) / lam.sum()
    pca90 = int(np.searchsorted(ev, 0.90) + 1)
    eff_rank = float((lam.sum() ** 2) / (lam ** 2).sum())
    return mean_cos, cond, pca90, eff_rank


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-layer embedding-health QC")
    ap.add_argument("--results-dir", default="results/esm")
    ap.add_argument("--model-name", default="esm")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/health")
    ap.add_argument("--max-residues", type=int, default=15000)
    ap.add_argument("--per-chain", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    emb = _collect(ids, Path(args.results_dir), sdir / "proteins",
                   args.max_residues, args.per_chain, args.seed)
    n_layers = emb.shape[0]
    labels = ["emb"] + [f"L{i}" for i in range(1, n_layers)]
    if args.model_name.startswith("proteinmpnn") or args.model_name == "mpnn":
        labels = [f"enc{i+1}" for i in range(n_layers)]

    rows = []
    for i in range(n_layers):
        mc, cond, p90, er = _health(emb[i])
        rows.append({"layer": labels[i], "mean_cosine": round(mc, 4),
                     "cond_number": round(cond, 1), "pca90": p90, "eff_rank": round(er, 1)})
        print(f"  {labels[i]:5s} cos={mc:.3f} cond={cond:.0f} pca90={p90} eff_rank={er:.1f}")

    out = Path(args.out_dir) / args.model_name; out.mkdir(parents=True, exist_ok=True)
    with (out / "health.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = range(n_layers)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
    a1.plot(x, [r["eff_rank"] for r in rows], "o-", label="eff. rank")
    a1.plot(x, [r["pca90"] for r in rows], "s--", label="PCA-90")
    a1.set_title(f"{args.model_name}: effective dimensionality"); a1.legend()
    a2.plot(x, [r["cond_number"] for r in rows], "o-", color="tab:red")
    a2.set_yscale("log"); a2.set_title(f"{args.model_name}: anisotropy (cov condition number)")
    a3.plot(x, [r["mean_cosine"] for r in rows], "o-", color="tab:green")
    a3.set_title(f"{args.model_name}: mean pairwise cosine (collapse)")
    for a in (a1, a2, a3):
        a.set_xticks(list(x)); a.set_xticklabels(labels, rotation=45, ha="right"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "health.png", dpi=140); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
