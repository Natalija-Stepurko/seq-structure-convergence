"""
07_geometry_overlap.py — do the two models share UNSUPERVISED geometric organisation?

Convergence metrics (stage 04) showed a shared *linear subspace* (SVCCA) that grows with the
sequence model's scale, while neighbourhood-level mutual k-NN stays weak. This stage tests the
organisation side directly, at the chain level (mean-pooled last-layer embeddings):

    1. Per model: PCA(50) -> HDBSCAN clusters; agreement of clusters with CATH class (ARI/NMI)
       -> does each model organise chains by fold without supervision?
    2. Cross-model chain mutual-kNN and HDBSCAN-cluster ARI between (ESM, structure) and
       (ESM-650M, structure) -> is organisation shared, and does it grow with sequence-model scale?
    3. UMAP(2D) of each model coloured by CATH class -> qualitative side-by-side.

Compares whichever model embedding dirs are given via --models name=dir (repeatable).

Outputs (under --out-dir):
    geometry_overlap.png   UMAP panels coloured by CATH class
    summary.txt            per-model cluster-vs-CATH ARI/NMI; cross-model mutual-kNN and cluster ARI

Usage:
    uv run python scripts/07_geometry_overlap.py --structures-dir /ssc/structures \\
        --models esm=/ssc/results/esm esm650=/ssc/results/esm650 mpnn=/ssc/results/proteinmpnn \\
        --struct-ref mpnn --out-dir /ssc/results/geometry
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
import qc_common as qc


def _pooled_last(pt):
    d = torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()  # [n,L,D]
    return d[-1].mean(axis=0)   # last-layer mean-pooled [D]


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-model geometric-organisation overlap")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--models", nargs="+", required=True, help="name=dir ...")
    ap.add_argument("--struct-ref", default="mpnn", help="which model is the structure reference")
    ap.add_argument("--out-dir", default="results/geometry")
    ap.add_argument("--max-chains", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    dirs = {k: Path(v) for k, v in models.items()}
    sdir = Path(args.structures_dir)
    manifest = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()}

    # chains present for ALL models
    ids = [cid for cid, r in manifest.items() if r.get("valid", True)
           and all((d / f"{cid}.pt").exists() for d in dirs.values())][: args.max_chains]
    cath = np.array([manifest[c]["cath_class"] for c in ids])
    print(f"{len(ids):,} chains present in all {len(dirs)} models")

    emb = {k: np.stack([_pooled_last(d / f"{c}.pt") for c in ids]) for k, d in dirs.items()}
    pca = {k: PCA(n_components=min(50, v.shape[1]), random_state=args.seed).fit_transform(v)
           for k, v in emb.items()}

    import hdbscan
    clusters = {k: hdbscan.HDBSCAN(min_cluster_size=25).fit_predict(v) for k, v in pca.items()}

    lines = [f"Geometric-organisation overlap over {len(ids):,} chains (chain-level, last-layer pooled)", ""]
    lines.append("Per-model unsupervised organisation by CATH class (cluster vs CATH):")
    for k in dirs:
        cl = clusters[k]; m = cl >= 0
        ari = adjusted_rand_score(cath[m], cl[m]) if m.sum() > 10 else float("nan")
        nmi = normalized_mutual_info_score(cath[m], cl[m]) if m.sum() > 10 else float("nan")
        nclust = len(set(cl[m].tolist()))
        lines.append(f"  {k:8s}: {nclust} clusters, {100*(~m).mean():.0f}% noise, "
                     f"cluster-vs-CATH ARI={ari:.3f} NMI={nmi:.3f}")

    ref = args.struct_ref
    lines.append("")
    lines.append(f"Cross-model overlap vs structure reference '{ref}' "
                 f"(does organisation align, and grow with sequence-model scale?):")
    for k in dirs:
        if k == ref:
            continue
        mkn = qc.mutual_knn(pca[k], pca[ref], k=15, seed=args.seed)
        mk = (clusters[k] >= 0) & (clusters[ref] >= 0)
        cari = adjusted_rand_score(clusters[k][mk], clusters[ref][mk]) if mk.sum() > 10 else float("nan")
        lines.append(f"  {k:8s} vs {ref}: chain mutual-kNN={mkn:.3f}  cross-cluster ARI={cari:.3f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # UMAP panels coloured by CATH class
    try:
        import umap
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(dirs), figsize=(5 * len(dirs), 4.6), squeeze=False)
        for ax, k in zip(axes[0], dirs):
            xy = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed).fit_transform(pca[k])
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=cath, s=4, cmap="tab10", alpha=0.6)
            ax.set_title(f"{k} (last layer, by CATH class)"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.02, label="CATH class")
        fig.suptitle("Chain-level embedding geometry, coloured by CATH class")
        fig.savefig(out / "geometry_overlap.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"-> {out}/geometry_overlap.png")
    except Exception as exc:
        print(f"UMAP figure skipped: {exc}")


if __name__ == "__main__":
    main()
