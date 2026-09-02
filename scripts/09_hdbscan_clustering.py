"""
09_hdbscan_clustering.py — unsupervised HDBSCAN clustering vs every property.

Tests whether a model's embedding manifold forms density clusters that align with any label,
at every layer — the "is it a cluster atlas?" question. For each layer we PCA-reduce the
embeddings and run HDBSCAN, then score the clusters against every categorical property with
ARI and NMI (chance-corrected agreement). High agreement = the embedding organises by that
property without supervision; near-zero = the property is decodable but not clustered.

Residue-level labels: amino-acid identity, 3-state SSE, burial, binding site.
Chain-level labels (mean-pooled): CATH class/arch/topology, kingdom, enzyme, EC class,
    protein_class, localisation, and the functional-class flags.

Outputs (under --out-dir/<model-name>):
    residue_clustering.csv / chain_clustering.csv   (layer, property, n_clusters, pct_noise, ARI, NMI)
    clustering_ari.png                              property × layer ARI heatmaps (residue + chain)

Usage:
    uv run python scripts/09_hdbscan_clustering.py --results-dir /ssc/results/esm650 \\
        --model-name esm650 --structures-dir /ssc/structures --out-dir /ssc/results/clustering
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
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

AA = "ACDEFGHIKLMNPQRSTVWY"; AA_IDX = {a: i for i, a in enumerate(AA)}
SS3_IDX = {"a": 0, "b": 1, "c": 2}
BURIAL_RSA = 0.25
CHAIN_KEYS = ["cath_class", "cath_arch", "cath_topol", "kingdom", "enzyme", "ec_class",
              "protein_class", "localisation", "is_transport", "is_dna_binding",
              "is_kinase", "is_ribosomal", "is_membrane_protein", "is_structural", "is_immune"]


def _collect(ids, res_dir, prot_dir, rest_dir, manifest, annot, max_residues, per_chain, seed):
    rng = np.random.default_rng(seed)
    res_parts, pooled = [], []
    res_lab = {"aa": [], "ss3": [], "burial": [], "binding_site": []}
    ch_lab = {k: [] for k in CHAIN_KEYS}
    total = 0
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
        for k in CHAIN_KEYS:
            d = manifest[cid] if k.startswith("cath") else annot.get(cid, {})
            ch_lab[k].append(d.get(k))
        idx = rng.choice(L, min(per_chain, L), replace=False)
        res_parts.append(E[:, idx, :])
        res_lab["aa"].append(np.array([AA_IDX.get(a, 20) for a in np.array(list(seq))[idx]]))
        res_lab["ss3"].append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        r = npz["rsa"][idx].astype(np.float64)
        res_lab["burial"].append(np.where(np.isfinite(r), (r < BURIAL_RSA).astype(int), -1))
        rt = rest_dir / f"{cid}.npz"
        res_lab["binding_site"].append(np.load(rt)["binding_site"][idx].astype(int)
                                       if rt.exists() else np.full(len(idx), -1))
        total += len(idx)
        if total >= max_residues:
            break
    return (np.concatenate(res_parts, axis=1),
            {k: np.concatenate(v) for k, v in res_lab.items()},
            np.stack(pooled, axis=1),
            {k: np.array(v, dtype=object) for k, v in ch_lab.items()})


def _cluster_scores(X, labels_dict, pca_dim, min_cluster_size):
    import hdbscan
    Xr = PCA(n_components=min(pca_dim, X.shape[1]), random_state=0).fit_transform(X)
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(Xr)
    ok = cl >= 0
    n_clust = len(set(cl[ok].tolist())); pct_noise = float((~ok).mean())
    out = {}
    for name, y in labels_dict.items():
        y = np.asarray(y)
        valid = ok & np.array([v is not None and (not isinstance(v, float) or v == v) for v in y]) \
                & (y.astype(str) != "-1")
        if valid.sum() < 20 or n_clust < 2:
            out[name] = (n_clust, pct_noise, float("nan"), float("nan")); continue
        ys = y[valid].astype(str)
        out[name] = (n_clust, pct_noise,
                     adjusted_rand_score(ys, cl[valid]),
                     normalized_mutual_info_score(ys, cl[valid]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="HDBSCAN clustering vs every property")
    ap.add_argument("--results-dir", default="results/esm")
    ap.add_argument("--model-name", default="esm")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/clustering")
    ap.add_argument("--max-residues", type=int, default=12000)
    ap.add_argument("--per-chain", type=int, default=6)
    ap.add_argument("--pca", type=int, default=30)
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    manifest = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()}
    annot = {}
    if (sdir / "annotations.jsonl").exists():
        annot = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "annotations.jsonl").open() if l.strip()}
    ids = [i for i, r in manifest.items() if r.get("valid", True)]

    res_emb, res_lab, pool_emb, ch_lab = _collect(
        ids, Path(args.results_dir), sdir / "proteins", sdir / "residue_targets",
        manifest, annot, args.max_residues, args.per_chain, args.seed)
    n_layers = res_emb.shape[0]
    labels = ["emb"] + [f"L{i}" for i in range(1, n_layers)]
    if args.model_name.startswith("proteinmpnn") or args.model_name == "mpnn":
        labels = [f"enc{i+1}" for i in range(n_layers)]
    print(f"[{args.model_name}] {res_emb.shape[1]:,} residues, {pool_emb.shape[1]:,} chains, {n_layers} layers")

    out = Path(args.out_dir) / args.model_name; out.mkdir(parents=True, exist_ok=True)
    for level, emb, lab in [("residue", res_emb, res_lab), ("chain", pool_emb, ch_lab)]:
        rows = []
        for i in range(n_layers):
            sc = _cluster_scores(emb[i], lab, args.pca, args.min_cluster_size)
            for prop, (nc, noise, ari, nmi) in sc.items():
                rows.append({"layer": labels[i], "property": prop, "n_clusters": nc,
                             "pct_noise": round(noise, 3), "ARI": round(ari, 4), "NMI": round(nmi, 4)})
            best = max(sc.items(), key=lambda kv: (kv[1][2] if kv[1][2] == kv[1][2] else -9))
            print(f"  {labels[i]:5s} [{level}] {sc[list(sc)[0]][0]} clusters, "
                  f"{100*sc[list(sc)[0]][1]:.0f}% noise; best ARI: {best[0]}={best[1][2]:.3f}")
        with (out / f"{level}_clustering.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    _plot(out, labels, args.model_name)
    print(f"-> {out}")


def _plot(out, labels, model_name):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(0.5 * len(labels) + 12, 7))
    for ax, level in zip(axes, ["residue", "chain"]):
        rows = list(csv.DictReader((out / f"{level}_clustering.csv").open()))
        props = sorted({r["property"] for r in rows})
        M = np.full((len(props), len(labels)), np.nan)
        for r in rows:
            M[props.index(r["property"]), labels.index(r["layer"])] = float(r["ARI"])
        # ARI is extremely skewed (amino-acid identity ~1.0, everything else ~0.00x), so a
        # data-max scale renders every biological property flat black. Use a log-ish scale
        # (PowerNorm) so small-but-real ARI stays visible while 1.0 remains the top of the range.
        from matplotlib.colors import PowerNorm
        im = ax.imshow(M, aspect="auto", cmap="magma",
                       norm=PowerNorm(gamma=0.35, vmin=0, vmax=max(0.1, float(np.nanmax(M)))))
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(props))); ax.set_yticklabels(props, fontsize=7)
        ax.set_title(f"{model_name}: {level}-level cluster–property ARI")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out / "clustering_ari.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
