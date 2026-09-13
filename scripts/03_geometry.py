r"""
03_geometry.py -- Per-model representation structure: how ONE model organises protein space.

Subcommands (each keeps the exact flags it had as a standalone script):

    depth-law        per-layer k-NN purity and LVR -> the depth law
    overlap          overlap between unsupervised geometry and labels
    health           effective rank, anisotropy, collapse, conditioning
    clusters         HDBSCAN density clusters vs every categorical label
    umap             UMAP grids, models side by side, coloured by property

The subcommand token is removed from argv before the original parser runs, so every command
line that worked before still works, with the subcommand inserted after the script name:

    uv run python scripts/03_geometry.py depth-law --help

Provenance: every subcommand writes params.json beside its outputs (see qc_common.record_params).

Merged from:
    depth-law        was 03_analyze_embeddings.py
    overlap          was 07_geometry_overlap.py
    health           was 08_embedding_health.py
    clusters         was 09_hdbscan_clustering.py
    umap             was 11_umap_property_grid.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
import qc_common as qc
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings





# ======================================================================================
# depth-law  --  from 03_analyze_embeddings.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

BURIAL_RSA_depthlaw = 0.25   # RSA below this = buried

def _collect_depthlaw(ids, res_dir, prot_dir, restgt_dir, cath_by_id, max_residues, seed):
    rng = np.random.default_rng(seed)
    emb_parts, ss3, rsa, cath, bfac, binding = [], [], [], [], [], []
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
        rt = restgt_dir / f"{cid}.npz"
        if rt.exists():
            t = np.load(rt)
            bf = t["bfactor"].astype(np.float64)
            bf = (bf - np.nanmean(bf)) / (np.nanstd(bf) + 1e-6)   # per-chain relative flexibility
            bfac.append(bf[idx]); binding.append(t["binding_site"][idx].astype(int))
        else:
            bfac.append(np.full(len(idx), np.nan)); binding.append(np.full(len(idx), -1))
        total += len(idx)
        if total >= max_residues:
            break
    if not emb_parts:
        sys.exit("ERROR: no chains with embeddings + labels found.")
    emb = np.concatenate(emb_parts, axis=1)     # [n_layers, N, D]
    return (emb, np.concatenate(ss3), np.concatenate(rsa), np.concatenate(cath),
            np.concatenate(bfac), np.concatenate(binding))

def _plot_depthlaw(rows, model_name, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in rows]
    x = range(len(layers))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(x, [r["purity_ss3"] for r in rows], "o-", label="secondary structure (local)")
    ax1.plot(x, [r["purity_burial"] for r in rows], "s-", label="burial (local)")
    ax1.plot(x, [r["purity_binding"] for r in rows], "D-", label="binding site")
    ax1.plot(x, [r["purity_cath"] for r in rows], "^-", label="CATH fold class (global)")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(layers, rotation=45, ha="right")
    ax1.set_xlabel("layer"); ax1.set_ylabel("chance-corrected k-NN purity")
    ax1.set_title(f"{model_name}: label organisation vs depth"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(x, [r["lvr_rsa"] for r in rows], "o-", color="tab:red", label="RSA")
    ax2.plot(x, [r["lvr_bfactor"] for r in rows], "s-", color="tab:purple", label="B-factor (flexibility)")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(layers, rotation=45, ha="right")
    ax2.set_xlabel("layer"); ax2.set_ylabel("LVR (lower = better)")
    ax2.set_title(f"{model_name}: local continuous-property organisation"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)

def _main_depthlaw() -> None:
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
    qc.record_params(out_dir, args)   # provenance: exactly what produced these outputs

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
    emb, ss3, rsa, cath, bfac, binding = _collect_depthlaw(
        ids, res_dir, prot_dir, Path(args.structures_dir) / "residue_targets",
        cath_by_id, args.max_residues, args.seed)
    n_layers, N, D = emb.shape
    print(f"  N={N:,} residues, {n_layers} layers, dim {D}")

    burial = np.where(np.isfinite(rsa), (rsa < BURIAL_RSA_depthlaw).astype(int), -1)
    b_ok = burial >= 0
    bind_ok = binding >= 0

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
            "purity_binding": round(qc.knn_purity(X[bind_ok], binding[bind_ok],
                                                  k=args.k, seed=args.seed), 4),
            "purity_cath": round(qc.knn_purity(X, cath, k=args.k, seed=args.seed), 4),
            "lvr_rsa": round(qc.lvr(X, rsa, k=args.k, seed=args.seed), 4),
            "lvr_bfactor": round(qc.lvr(X, bfac, k=args.k, seed=args.seed), 4),
        })
        print(f"  {labels[i]:5s}  ss3={rows[-1]['purity_ss3']:.3f}  "
              f"burial={rows[-1]['purity_burial']:.3f}  bind={rows[-1]['purity_binding']:.3f}  "
              f"cath={rows[-1]['purity_cath']:.3f}  lvr_rsa={rows[-1]['lvr_rsa']:.3f}  "
              f"lvr_bfac={rows[-1]['lvr_bfactor']:.3f}")

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _plot_depthlaw(rows, args.model_name, out_dir / "depth_law.png")
    print(f"-> {out_dir}")

# ======================================================================================
# overlap  --  from 07_geometry_overlap.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

def _pooled_last(pt):
    d = torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()  # [n,L,D]
    return d[-1].mean(axis=0)   # last-layer mean-pooled [D]

def _main_overlap() -> None:
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
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
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

# ======================================================================================
# health  --  from 08_embedding_health.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

def _collect_health(ids, res_dir, prot_dir, max_residues, per_chain, seed):
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

def _main_health() -> None:
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
    emb = _collect_health(ids, Path(args.results_dir), sdir / "proteins",
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
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
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

# ======================================================================================
# clusters  --  from 09_hdbscan_clustering.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

AA = "ACDEFGHIKLMNPQRSTVWY"; AA_IDX = {a: i for i, a in enumerate(AA)}

AA = "ACDEFGHIKLMNPQRSTVWY"; AA_IDX = {a: i for i, a in enumerate(AA)}

SS3_IDX = {"a": 0, "b": 1, "c": 2}

BURIAL_RSA_clusters = 0.25

CHAIN_KEYS = ["cath_class", "cath_arch", "cath_topol", "kingdom", "enzyme", "ec_class",
              "protein_class", "localisation", "is_transport", "is_dna_binding",
              "is_kinase", "is_ribosomal", "is_membrane_protein", "is_structural", "is_immune"]

def _collect_clusters(ids, res_dir, prot_dir, rest_dir, manifest, annot, max_residues, per_chain, seed):
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
        res_lab["burial"].append(np.where(np.isfinite(r), (r < BURIAL_RSA_clusters).astype(int), -1))
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

def _main_clusters() -> None:
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

    res_emb, res_lab, pool_emb, ch_lab = _collect_clusters(
        ids, Path(args.results_dir), sdir / "proteins", sdir / "residue_targets",
        manifest, annot, args.max_residues, args.per_chain, args.seed)
    n_layers = res_emb.shape[0]
    labels = ["emb"] + [f"L{i}" for i in range(1, n_layers)]
    if args.model_name.startswith("proteinmpnn") or args.model_name == "mpnn":
        labels = [f"enc{i+1}" for i in range(n_layers)]
    print(f"[{args.model_name}] {res_emb.shape[1]:,} residues, {pool_emb.shape[1]:,} chains, {n_layers} layers")


    out = Path(args.out_dir) / args.model_name; out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
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
    _plot_clusters(out, labels, args.model_name)
    print(f"-> {out}")

def _plot_clusters(out, labels, model_name):
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

# ======================================================================================
# umap  --  from 11_umap_property_grid.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

BURIAL_RSA_umap = 0.25

def _collect_umap(ids, dirs, prot, rest, manifest, annot, max_res, per_chain, seed):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in dirs}; pool = {k: [] for k in dirs}
    rl = {"amino acid": [], "secondary structure": [], "burial": [], "RSA": [],
          "B-factor": [], "binding site": []}
    cl = {"CATH class": [], "kingdom": [], "enzyme": [], "protein class": []}
    total = 0
    for cid in ids:
        if not all((Path(d) / f"{cid}.pt").exists() for d in dirs.values()):
            continue
        npz = np.load(prot / f"{cid}.npz", allow_pickle=True)
        seq = str(npz["seq"]); L = int(npz["ss3"].shape[0])
        E = {k: torch.load(Path(d) / f"{cid}.pt", weights_only=False)["layers"]
             .to(torch.float32).numpy() for k, d in dirs.items()}
        if any(v.shape[1] != L for v in E.values()):
            continue
        for k in dirs:
            pool[k].append(E[k][-1].mean(axis=0))
        m = manifest[cid]; a = annot.get(cid, {})
        cl["CATH class"].append(m.get("cath_class")); cl["kingdom"].append(a.get("kingdom"))
        cl["enzyme"].append(a.get("enzyme")); cl["protein class"].append(a.get("protein_class"))
        idx = rng.choice(L, min(per_chain, L), replace=False)
        for k in dirs:
            res[k].append(E[k][-1][idx])          # last layer, per-residue
        rl["amino acid"].append(np.array([AA_IDX.get(c, 20) for c in np.array(list(seq))[idx]]))
        rl["secondary structure"].append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        r = npz["rsa"][idx].astype(np.float64)
        rl["RSA"].append(r); rl["burial"].append(np.where(np.isfinite(r), (r < BURIAL_RSA_umap).astype(int), -1))
        rt = rest / f"{cid}.npz"
        if rt.exists():
            t = np.load(rt); bf = t["bfactor"].astype(np.float64)
            bf = (bf - np.nanmean(bf)) / (np.nanstd(bf) + 1e-6)
            rl["B-factor"].append(bf[idx]); rl["binding site"].append(t["binding_site"][idx].astype(int))
        else:
            rl["B-factor"].append(np.full(len(idx), np.nan)); rl["binding site"].append(np.full(len(idx), -1))
        total += len(idx)
        if total >= max_res:
            break
    return ({k: np.concatenate(v, axis=0) for k, v in res.items()},
            {k: np.concatenate(v) for k, v in rl.items()},
            {k: np.stack(v) for k, v in pool.items()},
            {k: np.array(v, dtype=object) for k, v in cl.items()})

def _umap(X, seed):
    import umap
    Xp = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(Xp)

PROP_META = {
    "amino acid": ("cat", list(AA) + ["X"]),
    "secondary structure": ("cat", ["helix", "sheet", "coil"]),
    "burial": ("cat", ["exposed", "buried"]),
    "binding site": ("cat", ["other", "binding"]),
    "enzyme": ("cat", ["non-enzyme", "enzyme"]),
    "CATH class": ("cat", None),   # numeric codes 1..6, labelled by value
    "RSA": ("cont", None),
    "B-factor": ("cont", None),
}

def _as_plottable(p, y):
    """-> (numeric values, is_categorical, category names or None). NaN = unlabelled."""
    kind, names = PROP_META.get(p, (None, None))
    if y.dtype == object:                      # strings (kingdom, protein class, ...)
        vals = np.array([None if v is None or (isinstance(v, float) and v != v) else str(v) for v in y],
                        dtype=object)
        cats = sorted({v for v in vals if v is not None})
        code = {c: k for k, c in enumerate(cats)}
        num = np.array([code[v] if v is not None else np.nan for v in vals], dtype=float)
        # numeric-coded flags stored as objects (e.g. enzyme 0/1) get their declared names
        if names is not None and len(cats) <= len(names) and all(c.isdigit() for c in cats):
            cats = [names[int(c)] for c in cats]
        return num, True, cats
    num = y.astype(float).copy()
    num[num < 0] = np.nan                      # -1 encodes "unlabelled" for our int targets
    if kind == "cont":
        return num, False, None
    uniq = np.unique(num[np.isfinite(num)])
    if kind == "cat" or len(uniq) <= 21:
        if names is None:
            names = [f"{int(u)}" for u in uniq]
            remap = {u: k for k, u in enumerate(uniq)}
            num = np.array([remap.get(v, np.nan) if np.isfinite(v) else np.nan for v in num])
        return num, True, names
    return num, False, None

COL_W = 3.3   # inches per column; used for figsize and for the label-fit test

def _grid(coords, labels, title, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    models = list(coords); props = list(labels)
    fig, axes = plt.subplots(len(models), len(props),
                             figsize=(COL_W * len(props), 3.2 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        xy = coords[m]
        for j, p in enumerate(props):
            ax = axes[i][j]
            num, is_cat, cats = _as_plottable(p, labels[p])
            ok = np.isfinite(num)
            ax.scatter(xy[~ok, 0], xy[~ok, 1], s=2, c="lightgrey", alpha=0.25)
            if is_cat:
                n = max(1, len(cats))
                base = plt.get_cmap("tab20" if n > 10 else "tab10")
                cmap = ListedColormap([base(k % base.N) for k in range(n)])
                norm = BoundaryNorm(np.arange(-0.5, n + 0.5), n)
                sc = ax.scatter(xy[ok, 0], xy[ok, 1], s=2, c=num[ok], cmap=cmap, norm=norm, alpha=0.7)
            else:
                # robust colour limits: a few extreme values (e.g. B-factor z up to +7) would
                # otherwise compress the bulk of the distribution into one flat colour
                lo, hi = np.nanpercentile(num[ok], [2, 98])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = np.nanmin(num[ok]), np.nanmax(num[ok])
                sc = ax.scatter(xy[ok, 0], xy[ok, 1], s=2, c=np.clip(num[ok], lo, hi),
                                cmap="viridis", alpha=0.7, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(m, fontsize=11, fontweight="bold")
            # one legend / colourbar per column, above the top row only
            if i == 0:
                cax = ax.inset_axes([0.0, 1.06, 1.0, 0.05])
                cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
                # move ticks to the top FIRST — doing this after styling would reset the labels
                cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
                pad = 16
                if is_cat:
                    cb.set_ticks(range(len(cats)))
                    cb.set_ticklabels([str(c) for c in cats])
                    longest = max((len(str(c)) for c in cats), default=1)
                    fs = 5.5 if len(cats) > 12 else (6.5 if len(cats) > 8 else 8)
                    # rotate only if the labels would actually collide: compare the widest
                    # label's width to the horizontal space one tick gets
                    per_tick_in = COL_W / max(1, len(cats))
                    text_w_in = longest * fs * 0.6 / 72.0
                    rot = 90 if text_w_in > 0.9 * per_tick_in else 0
                    cb.ax.tick_params(labelsize=fs, labelrotation=rot)
                    if rot:                       # rotated labels grow upward — clear the title
                        pad = 14 + int(longest * fs * 0.62)
                else:
                    cb.ax.tick_params(labelsize=7)
                cax.set_title(p, fontsize=11, pad=pad)
    fig.suptitle(title + "   (grey = unlabelled; continuous scales clipped to 2nd–98th percentile)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)

def _main_umap() -> None:
    ap = argparse.ArgumentParser(description="UMAP property grids per model")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/umap")
    ap.add_argument("--max-residues", type=int, default=8000)
    ap.add_argument("--per-chain", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replot", action="store_true", help="re-render figures from cached coords.npz")
    args = ap.parse_args()

    dirs = dict(m.split("=", 1) for m in args.models)
    sdir = Path(args.structures_dir)
    manifest = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()}
    annot = {}
    if (sdir / "annotations.jsonl").exists():
        annot = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "annotations.jsonl").open() if l.strip()}
    ids = [i for i, r in manifest.items() if r.get("valid", True)]

    res, rl, pool, cl = _collect_umap(ids, dirs, sdir / "proteins", sdir / "residue_targets",
                                 manifest, annot, args.max_residues, args.per_chain, args.seed)
    print(f"residues: {next(iter(res.values())).shape[0]:,}  chains: {next(iter(pool.values())).shape[0]:,}")


    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
    cache = out / "coords.npz"
    if args.replot and cache.exists():        # re-render figures without recomputing UMAP
        z = np.load(cache, allow_pickle=True)
        rc = {m: z[f"res_{m}"] for m in dirs if f"res_{m}" in z}
        cc = {m: z[f"chain_{m}"] for m in dirs if f"chain_{m}" in z}
        rl = {k: v for k, v in z["res_labels"].item().items()}
        cl = {k: v for k, v in z["chain_labels"].item().items()}
        print("re-plotting from cached coords")
    else:
        rc = {m: _umap(v, args.seed) for m, v in res.items()}
        print("residue UMAPs done")
        cc = {m: _umap(v, args.seed) for m, v in pool.items()}
        print("chain UMAPs done")
        np.savez(cache, **{f"res_{m}": v for m, v in rc.items()},
                 **{f"chain_{m}": v for m, v in cc.items()},
                 res_labels=rl, chain_labels=cl)
    _grid(rc, rl, "Residue-level UMAP (last layer), same residues, coloured by property",
          out / "umap_residue_grid.png")
    _grid(cc, cl, "Chain-level UMAP (last layer, mean-pooled), same chains, coloured by property",
          out / "umap_chain_grid.png")
    print(f"-> {out}")


# ==========================================================================================
# dispatch
# ==========================================================================================

_SUBCOMMANDS = {
    "depth-law": _main_depthlaw,
    "overlap": _main_overlap,
    "health": _main_health,
    "clusters": _main_clusters,
    "umap": _main_umap,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in _SUBCOMMANDS:
        print(__doc__)
        print("subcommands: " + ", ".join(_SUBCOMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    cmd = sys.argv.pop(1)                 # the original main() then sees its original argv
    _SUBCOMMANDS[cmd]()


if __name__ == "__main__":
    main()
