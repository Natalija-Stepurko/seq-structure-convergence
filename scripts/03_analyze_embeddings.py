"""
03_analyze_embeddings.py — per-model, per-layer representation geometry (the depth law).

For one model's embeddings, measures how well each layer organises protein space around
labels the model was never trained on, and how that changes with depth. Tests prediction
P3: residue-local biophysics (secondary structure, burial) is organised early; a
whole-protein label (CATH fold class) is organised late.

Per layer it computes (labels from stage 01):
    - k-NN purity of 3-state secondary structure   (residue-local)
    - k-NN purity of burial (RSA < 0.25)            (residue-local)
    - k-NN purity of CATH class                     (whole-protein, per-residue via chain)
    - LVR of relative SASA                          (continuous, residue-local)

Run once per model (point --results-dir / --model-name at each).

Outputs (under --out-dir/<model-name>, default results/analysis/<model-name>):
    metrics.csv        per-layer metric table
    depth_law.png      purity/LVR vs layer

Usage:
    uv run python scripts/03_analyze_embeddings.py \\
        --results-dir /ssc/results/esm --model-name esm \\
        --structures-dir /ssc/structures --out-dir /ssc/results/analysis
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import qc_common as qc

BURIAL_RSA = 0.25   # RSA below this = buried


def _collect(ids, res_dir, prot_dir, cath_by_id, max_residues, seed):
    rng = np.random.default_rng(seed)
    emb_parts, ss3, rsa, cath = [], [], [], []
    total = 0
    for cid in ids:
        pt = res_dir / f"{cid}.pt"
        if not pt.exists() or cid not in cath_by_id:
            continue
        npz = np.load(prot_dir / f"{cid}.npz", allow_pickle=True)
        L = int(npz["ss3"].shape[0])
        E = torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()  # [n,L,D]
        if E.shape[1] != L:
            continue
        cap = max(1, min(L, max_residues // 50))
        idx = rng.choice(L, cap, replace=False) if L > cap else np.arange(L)
        emb_parts.append(E[:, idx, :])
        ss3.append(npz["ss3"][idx])
        rsa.append(npz["rsa"][idx].astype(np.float64))
        cath.append(np.full(len(idx), cath_by_id[cid]))
        total += len(idx)
        if total >= max_residues:
            break
    if not emb_parts:
        sys.exit("ERROR: no chains with embeddings + labels found.")
    emb = np.concatenate(emb_parts, axis=1)     # [n_layers, N, D]
    return (emb, np.concatenate(ss3), np.concatenate(rsa), np.concatenate(cath))


def _plot(rows, model_name, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in rows]
    x = range(len(layers))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(x, [r["purity_ss3"] for r in rows], "o-", label="secondary structure (local)")
    ax1.plot(x, [r["purity_burial"] for r in rows], "s-", label="burial (local)")
    ax1.plot(x, [r["purity_cath"] for r in rows], "^-", label="CATH fold class (global)")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(layers, rotation=45, ha="right")
    ax1.set_xlabel("layer"); ax1.set_ylabel("chance-corrected k-NN purity")
    ax1.set_title(f"{model_name}: label organisation vs depth"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(x, [r["lvr_rsa"] for r in rows], "o-", color="tab:red")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(layers, rotation=45, ha="right")
    ax2.set_xlabel("layer"); ax2.set_ylabel("LVR of RSA (lower = better)")
    ax2.set_title(f"{model_name}: local RSA organisation vs depth"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-model per-layer geometry")
    ap.add_argument("--results-dir", default="results/esm")
    ap.add_argument("--model-name", default="esm")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--max-residues", type=int, default=15000)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    prot_dir = Path(args.structures_dir) / "proteins"
    out_dir = Path(args.out_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.structures_dir) / "index.jsonl"
    cath_by_id, ids = {}, []
    for l in manifest.open():
        if not l.strip():
            continue
        r = json.loads(l)
        if not r.get("valid", True):
            continue
        ids.append(r["id"])
        if r.get("cath_class") is not None:
            cath_by_id[r["id"]] = int(r["cath_class"])

    print(f"[{args.model_name}] collecting up to {args.max_residues:,} residues ...")
    emb, ss3, rsa, cath = _collect(ids, res_dir, prot_dir, cath_by_id,
                                   args.max_residues, args.seed)
    n_layers, N, D = emb.shape
    print(f"  N={N:,} residues, {n_layers} layers, dim {D}")

    burial = np.where(np.isfinite(rsa), (rsa < BURIAL_RSA).astype(int), -1)
    b_ok = burial >= 0

    labels = ["emb"] + [f"L{i}" for i in range(1, n_layers)]
    if args.model_name.startswith("proteinmpnn"):
        labels = [f"enc{i+1}" for i in range(n_layers)]

    rows = []
    for i in range(n_layers):
        X = emb[i]
        rows.append({
            "layer": labels[i],
            "purity_ss3": round(qc.knn_purity(X, ss3, k=args.k, seed=args.seed), 4),
            "purity_burial": round(qc.knn_purity(X[b_ok], burial[b_ok],
                                                 k=args.k, seed=args.seed), 4),
            "purity_cath": round(qc.knn_purity(X, cath, k=args.k, seed=args.seed), 4),
            "lvr_rsa": round(qc.lvr(X, rsa, k=args.k, seed=args.seed), 4),
        })
        print(f"  {labels[i]:5s}  ss3={rows[-1]['purity_ss3']:.3f}  "
              f"burial={rows[-1]['purity_burial']:.3f}  cath={rows[-1]['purity_cath']:.3f}  "
              f"lvr_rsa={rows[-1]['lvr_rsa']:.3f}")

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _plot(rows, args.model_name, out_dir / "depth_law.png")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
