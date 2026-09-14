r"""
04_convergence.py -- Between-model similarity and every control that makes it interpretable.

Subcommands (each keeps the exact flags it had as a standalone script):

    grids            layer x layer CKA/SVCCA/mutual-kNN + permutation null
    supervised       Cohen's kappa between the models' probe predictions
    significance     resampled CIs and empirical p-value for the peak
    svcca-controls   SVCCA permutation null and dimension matching
    functional       per-property convergence grids for annotation labels
    aa-control       convergence with the shared training target partialled out
    ladder-ci        confidence intervals for every pair, raw and AA-controlled

The subcommand token is removed from argv before the original parser runs, so every command
line that worked before still works, with the subcommand inserted after the script name:

    uv run python scripts/04_convergence.py grids --help

Provenance: every subcommand writes params.json beside its outputs (see qc_common.record_params).

Merged from:
    grids            was 04_convergence.py
    supervised       was 04b_supervised_convergence.py
    significance     was 06_significance.py
    svcca-controls   was 10_svcca_controls.py
    functional       was 14_functional_grids.py
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import qc_common as qc
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter
import csv
from sklearn.decomposition import PCA
import argparse, json, warnings
import numpy as np, torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score





# ======================================================================================
# grids  --  from 04_convergence.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

def _load_layers(pt_path: Path) -> np.ndarray:
    """Return [n_reps, L, D] float32."""
    d = torch.load(pt_path, weights_only=False)
    return d["layers"].to(torch.float32).numpy()

def _collect_aligned(ids, esm_dir, struct_dir, prot_dir, max_residues, seed):
    """Build residue-aligned per-layer matrices for both models over a residue budget."""
    rng = np.random.default_rng(seed)
    esm_parts: list[np.ndarray] = []
    st_parts: list[np.ndarray] = []
    total = 0
    used = 0
    for cid in ids:
        pe, ps = esm_dir / f"{cid}.pt", struct_dir / f"{cid}.pt"
        if not (pe.exists() and ps.exists()):
            continue
        E = _load_layers(pe)          # [nE, L, De]
        S = _load_layers(ps)          # [nS, L, Ds]
        if E.shape[1] != S.shape[1]:  # residue mismatch -> skip (should not happen)
            continue
        L = E.shape[1]
        # per-chain cap so no single big protein dominates the sample
        cap = max(1, min(L, max_residues // 50))
        idx = rng.choice(L, cap, replace=False) if L > cap else np.arange(L)
        esm_parts.append(E[:, idx, :])
        st_parts.append(S[:, idx, :])
        total += len(idx)
        used += 1
        if total >= max_residues:
            break
    if not esm_parts:
        sys.exit("ERROR: no chains with both ESM and structure embeddings found.")
    esm_all = np.concatenate(esm_parts, axis=1)   # [nE, N, De]
    st_all = np.concatenate(st_parts, axis=1)     # [nS, N, Ds]
    print(f"  aligned residues: N={esm_all.shape[1]:,} from {used:,} chains")
    return esm_all, st_all

def _grids(esm_all, st_all, seed):
    nE, nS = esm_all.shape[0], st_all.shape[0]
    # center once per layer for CKA
    esm_c = [qc.column_center(esm_all[i]) for i in range(nE)]
    st_c = [qc.column_center(st_all[j]) for j in range(nS)]
    cka = np.full((nE, nS), np.nan)
    svc = np.full((nE, nS), np.nan)
    mkn = np.full((nE, nS), np.nan)
    cka_perm = np.full((nE, nS), np.nan)
    perm = np.random.default_rng(seed).permutation(esm_all.shape[1])
    for i in range(nE):
        for j in range(nS):
            cka[i, j] = qc.linear_cka(esm_c[i], st_c[j])
            svc[i, j] = qc.svcca(esm_all[i], st_all[j], seed=seed)
            mkn[i, j] = qc.mutual_knn(esm_all[i], st_all[j], seed=seed)
            cka_perm[i, j] = qc.linear_cka(esm_c[i], st_c[j][perm])
    return {"cka": cka, "svcca": svc, "mutual_knn": mkn, "cka_permuted": cka_perm}

def _plot_grids(grids, esm_labels, st_labels, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("cka", "linear CKA"), ("svcca", "SVCCA"), ("mutual_knn", "mutual k-NN")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (key, title) in zip(axes, panels):
        M = grids[key]
        im = ax.imshow(M, vmin=0, vmax=max(0.2, float(np.nanmax(M))), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(st_labels))); ax.set_xticklabels(st_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(esm_labels))); ax.set_yticklabels(esm_labels)
        ax.set_xlabel("ProteinMPNN (structure) layer"); ax.set_ylabel("ESM-2 (sequence) layer")
        ax.set_title(title)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if np.isfinite(M[a, b]):
                    ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center",
                            color="white" if M[a, b] < 0.5 * np.nanmax(M) else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    base = np.nanmean(grids["cka_permuted"])
    fig.suptitle(f"Sequence↔structure representational convergence  "
                 f"(CKA permutation baseline ≈ {base:.3f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def _main_grids() -> None:
    ap = argparse.ArgumentParser(description="Cross-model convergence grids")
    ap.add_argument("--esm-dir", default="results/esm")
    ap.add_argument("--struct-dir", default="results/proteinmpnn")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/convergence")
    ap.add_argument("--max-residues", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    esm_dir, struct_dir = Path(args.esm_dir), Path(args.struct_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    qc.record_params(out_dir, args)   # provenance: exactly what produced these outputs
    prot_dir = Path(args.structures_dir) / "proteins"

    manifest = Path(args.structures_dir) / "index.jsonl"
    ids = [json.loads(l)["id"] for l in manifest.open() if l.strip()
           if json.loads(l).get("valid", True)]
    print(f"Convergence over {len(ids):,} candidate chains "
          f"(ESM-2 x ProteinMPNN), budget {args.max_residues:,} residues.")

    t0 = time.time()
    esm_all, st_all = _collect_aligned(ids, esm_dir, struct_dir, prot_dir,
                                       args.max_residues, args.seed)
    grids = _grids(esm_all, st_all, args.seed)
    # Re-record now that the realised sample is known: `--max-residues` is a *budget*, and the
    # number actually aligned can fall short of it. Comparing convergence values computed on
    # different realised samples is not valid, so the realised figure is what must be pinned.
    qc.record_params(out_dir, args, extra={"n_residues_used": int(esm_all.shape[1]),
                                           "n_esm_layers": int(esm_all.shape[0]),
                                           "n_struct_layers": int(st_all.shape[0])})

    nE, nS = esm_all.shape[0], st_all.shape[0]
    esm_labels = ["emb"] + [f"b{i}" for i in range(1, nE)]
    st_labels = [f"enc{j+1}" for j in range(nS)]

    np.savez(out_dir / "grids.npz", esm_labels=esm_labels, st_labels=st_labels, **grids)
    _plot_grids(grids, esm_labels, st_labels, out_dir / "convergence.png")

    # summary
    lines = [f"Sequence↔structure convergence  (N residues budget {args.max_residues:,})",
             f"ESM-2 layers: {esm_labels}", f"structure layers: {st_labels}", ""]
    for key in ("cka", "svcca", "mutual_knn"):
        M = grids[key]
        i, j = np.unravel_index(np.nanargmax(M), M.shape)
        lines.append(f"{key:12s} peak {M[i,j]:.3f} at ESM {esm_labels[i]} × struct {st_labels[j]}")
    lines.append(f"CKA permutation baseline (mean): {np.nanmean(grids['cka_permuted']):.3f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {out_dir}")

# ======================================================================================
# supervised  --  from 04b_supervised_convergence.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

SS3_IDX = {"a": 0, "b": 1, "c": 2}

MAX_CLASSES = 12

TARGETS = [
    ("binding_site", "residue", "restgt", "binding_site"),
    ("ss3", "residue", "npz", "ss3"),
    ("enzyme", "chain", "annot", "enzyme"),
    ("cath_class", "chain", "manifest", "cath_class"),
]

def _layers(pt):
    return torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()

def _collect(ids, esm_dir, struct_dir, sdir, max_chains, per_chain, seed):
    rng = np.random.default_rng(seed)
    prot, rest = sdir / "proteins", sdir / "residue_targets"
    manifest = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()}
    annot = {}
    if (sdir / "annotations.jsonl").exists():
        annot = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "annotations.jsonl").open() if l.strip()}
    E_res, S_res, E_pool, S_pool, grp = [], [], [], [], []
    res_lab = {"binding_site": [], "ss3": []}
    ch_lab = {"enzyme": [], "cath_class": []}
    g = 0
    for cid in ids:
        pe, ps = esm_dir / f"{cid}.pt", struct_dir / f"{cid}.pt"
        if not (pe.exists() and ps.exists()) or cid not in manifest:
            continue
        npz = np.load(prot / f"{cid}.npz", allow_pickle=True)
        L = int(npz["ss3"].shape[0])
        Ee, Es = _layers(pe), _layers(ps)
        if Ee.shape[1] != L or Es.shape[1] != L:
            continue
        rt = np.load(rest / f"{cid}.npz") if (rest / f"{cid}.npz").exists() else None
        # residue sample; force-include binding sites
        force = np.where(rt["binding_site"] == 1)[0] if rt is not None else np.array([], int)
        pool = np.setdiff1d(np.arange(L), force)
        rand = rng.choice(pool, min(max(0, per_chain - len(force)), len(pool)), replace=False)
        idx = np.concatenate([force, rand]).astype(int)
        E_res.append(Ee[:, idx, :]); S_res.append(Es[:, idx, :])
        res_lab["ss3"].append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        res_lab["binding_site"].append(rt["binding_site"][idx].astype(int) if rt is not None
                                       else np.full(len(idx), -1))
        E_pool.append(Ee.mean(1)); S_pool.append(Es.mean(1))
        ch_lab["enzyme"].append(annot.get(cid, {}).get("enzyme"))
        ch_lab["cath_class"].append(manifest[cid].get("cath_class"))
        grp.append(np.full(len(idx), g)); g += 1
        if g >= max_chains:
            break
    if g < 20:
        sys.exit("ERROR: too few chains with both models' embeddings.")
    return {
        "E_res": np.concatenate(E_res, 1), "S_res": np.concatenate(S_res, 1),
        "E_pool": np.stack(E_pool, 1), "S_pool": np.stack(S_pool, 1),
        "grp": np.concatenate(grp), "n_chains": g,
        "res": {k: np.concatenate(v) for k, v in res_lab.items()},
        "chain": {k: np.array(v, dtype=object) for k, v in ch_lab.items()},
    }

def _fit_predict(Xtr, ytr, Xte, balanced):
    spw = 1.0
    if balanced and set(np.unique(ytr)) <= {0, 1}:
        pos = int(np.sum(ytr)); spw = max(1.0, (len(ytr) - pos) / max(1, pos))
    m = XGBClassifier(n_estimators=80, max_depth=4, tree_method="hist", n_jobs=4,
                      verbosity=0, scale_pos_weight=spw).fit(Xtr, ytr)
    return m.predict(Xte)

def _grid_for_target(Eemb, Semb, y, tr, te, balanced):
    """Return kappa grid [n_esm, n_struct], plus each model's own accuracy per layer."""
    ne, ns = Eemb.shape[0], Semb.shape[0]
    pe = [_fit_predict(Eemb[i][tr], y[tr], Eemb[i][te], balanced) for i in range(ne)]
    psr = [_fit_predict(Semb[j][tr], y[tr], Semb[j][te], balanced) for j in range(ns)]
    acc_e = [accuracy_score(y[te], p) for p in pe]
    acc_s = [accuracy_score(y[te], p) for p in psr]
    K = np.array([[cohen_kappa_score(pe[i], psr[j]) for j in range(ns)] for i in range(ne)])
    return K, acc_e, acc_s

def _main_supervised() -> None:
    ap = argparse.ArgumentParser(description="XGB model-to-model convergence")
    ap.add_argument("--esm-dir", default="results/esm")
    ap.add_argument("--struct-dir", default="results/proteinmpnn")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/convergence")
    ap.add_argument("--max-chains", type=int, default=3000)
    ap.add_argument("--per-chain", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    print(f"Collecting aligned ESM+struct embeddings over <= {args.max_chains} chains ...")
    D = _collect(ids, Path(args.esm_dir), Path(args.struct_dir), sdir,
                 args.max_chains, args.per_chain, args.seed)
    C = D["n_chains"]
    rng = np.random.default_rng(args.seed)
    test_c = set(rng.choice(C, max(1, int(C * args.test_frac)), replace=False).tolist())
    res_te = np.isin(D["grp"], list(test_c)); ch_te = np.array([c in test_c for c in range(C)])
    ne, ns = D["E_res"].shape[0], D["S_res"].shape[0]
    e_lab = ["emb"] + [f"L{i}" for i in range(1, ne)]
    s_lab = [f"enc{j+1}" for j in range(ns)]

    grids = {}
    panels = []
    for name, gran, src, key in TARGETS:
        if gran == "residue":
            y_all = D["res"][key]; Eemb, Semb = D["E_res"], D["S_res"]
            mask = y_all >= 0 if key == "binding_site" else np.ones(len(y_all), bool)
            tr = mask & ~res_te; te = mask & res_te
            y = y_all
        else:
            y_raw = D["chain"][key]; Eemb, Semb = D["E_pool"], D["S_pool"]
            have = np.array([v is not None for v in y_raw])
            vals, cnt = np.unique(y_raw[have].astype(str), return_counts=True)
            keep = set(vals[np.argsort(-cnt)][:MAX_CLASSES])
            m = have & np.array([str(v) in keep for v in y_raw])
            y = np.full(len(y_raw), -1)
            y[m] = LabelEncoder().fit_transform(y_raw[m].astype(str))
            tr = m & ~ch_te; te = m & ch_te
        if te.sum() < 10 or len(np.unique(y[tr])) < 2:
            print(f"  [skip {name}]"); continue
        balanced = key in ("binding_site", "enzyme")
        K, ae, as_ = _grid_for_target(Eemb, Semb, y, tr, te, balanced)
        grids[name] = K
        panels.append((name, K, ae, as_))
        print(f"  {name}: peak κ={np.nanmax(K):.3f} at "
              f"ESM {e_lab[np.unravel_index(np.nanargmax(K), K.shape)[0]]} × "
              f"struct {s_lab[np.unravel_index(np.nanargmax(K), K.shape)[1]]}")


    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    qc.record_params(out_dir, args)   # provenance: exactly what produced these outputs
    np.savez(out_dir / "kappa_grids.npz", e_lab=e_lab, s_lab=s_lab, **grids)
    _plot_supervised(panels, e_lab, s_lab, out_dir / "supervised_convergence.png")
    print(f"-> {out_dir}/supervised_convergence.png")

def _plot_supervised(panels, e_lab, s_lab, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1, 4.6), squeeze=False)
    for ax, (name, K, ae, as_) in zip(axes[0], panels):
        im = ax.imshow(K, vmin=0, vmax=max(0.2, float(np.nanmax(K))), cmap="magma", aspect="auto")
        ax.set_xticks(range(len(s_lab)))
        ax.set_xticklabels([f"{l}\n{a:.2f}" for l, a in zip(s_lab, as_)], fontsize=7)
        ax.set_yticks(range(len(e_lab)))
        ax.set_yticklabels([f"{l} {a:.2f}" for l, a in zip(e_lab, ae)], fontsize=6)
        ax.set_xlabel("ProteinMPNN layer\n(struct; own acc)");
        if ax is axes[0][0]:
            ax.set_ylabel("ESM-2 layer (own acc)")
        ax.set_title(f"{name}\ncross-model agreement (κ)", fontsize=9)
        for a in range(K.shape[0]):
            for b in range(K.shape[1]):
                ax.text(b, a, f"{K[a,b]:.2f}", ha="center", va="center",
                        color="white" if K[a, b] < 0.5 * np.nanmax(K) else "black", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("XGB model-to-model convergence — where ESM-2 & ProteinMPNN make the same prediction",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=140); plt.close(fig)

# ======================================================================================
# significance  --  from 06_significance.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

def _one_repeat(ids, esm_dir, struct_dir, max_residues, rng):
    esm_parts, st_parts, total = [], [], 0
    order = list(ids); rng.shuffle(order)
    for cid in order:
        pe, ps = esm_dir / f"{cid}.pt", struct_dir / f"{cid}.pt"
        if not (pe.exists() and ps.exists()):
            continue
        E, S = _layers(pe), _layers(ps)
        if E.shape[1] != S.shape[1]:
            continue
        L = E.shape[1]
        cap = max(1, min(L, max_residues // 50))
        idx = rng.choice(L, cap, replace=False) if L > cap else np.arange(L)
        esm_parts.append(E[:, idx, :]); st_parts.append(S[:, idx, :])
        total += len(idx)
        if total >= max_residues:
            break
    esm_all = np.concatenate(esm_parts, axis=1)
    st_all = np.concatenate(st_parts, axis=1)
    esm_c = [qc.column_center(esm_all[i]) for i in range(esm_all.shape[0])]
    st_c = [qc.column_center(st_all[j]) for j in range(st_all.shape[0])]
    perm = rng.permutation(esm_all.shape[1])
    nE, nS = len(esm_c), len(st_c)
    cka = np.empty((nE, nS)); ckap = np.empty((nE, nS))
    for i in range(nE):
        for j in range(nS):
            cka[i, j] = qc.linear_cka(esm_c[i], st_c[j])
            ckap[i, j] = qc.linear_cka(esm_c[i], st_c[j][perm])
    ij = np.unravel_index(np.nanargmax(cka), cka.shape)
    return float(np.nanmax(cka)), (int(ij[0]), int(ij[1])), float(np.nanmax(ckap))

def _ci(x):
    x = np.asarray(x)
    return float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

def _main_significance() -> None:
    ap = argparse.ArgumentParser(description="Convergence significance")
    ap.add_argument("--esm-dir", default="results/esm")
    ap.add_argument("--struct-dir", default="results/proteinmpnn")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--out-dir", default="results/convergence/significance")
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--max-residues", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()


    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    qc.record_params(out_dir, args)   # provenance: exactly what produced these outputs
    ids = [json.loads(l)["id"] for l in (Path(args.structures_dir) / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]

    peaks, peaks_perm, locs = [], [], []
    for r in range(args.repeats):
        rng = np.random.default_rng(args.seed + r)
        pk, ij, pkp = _one_repeat(ids, Path(args.esm_dir), Path(args.struct_dir),
                                  args.max_residues, rng)
        peaks.append(pk); peaks_perm.append(pkp); locs.append(ij)
        print(f"  repeat {r+1}/{args.repeats}: peak CKA={pk:.3f} at {ij}  perm={pkp:.3f}")

    pm, plo, phi = _ci(peaks)
    qm, qlo, qhi = _ci(peaks_perm)
    modal, modal_n = Counter(locs).most_common(1)[0]
    pval = float(np.mean(np.asarray(peaks) <= np.asarray(peaks_perm)))

    lines = [
        f"Convergence significance over {args.repeats} residue resamples "
        f"(budget {args.max_residues:,})",
        f"  peak CKA:        mean {pm:.3f}  95% CI [{plo:.3f}, {phi:.3f}]",
        f"  permuted CKA:    mean {qm:.3f}  95% CI [{qlo:.3f}, {qhi:.3f}]",
        f"  separation:      {pm - qm:.3f}",
        f"  modal peak pair: ESM layer {modal[0]} × struct layer {modal[1]}  "
        f"({modal_n}/{args.repeats} repeats)",
        f"  empirical p (peak <= permuted): {pval:.3g}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([0, 1], [pm, qm], yerr=[[pm - plo, qm - qlo], [phi - pm, qhi - qm]],
           color=["tab:blue", "tab:gray"], capsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real peak CKA", "permutation baseline"])
    ax.set_ylabel("peak cross-model CKA")
    ax.set_title(f"Convergence robustness ({args.repeats} resamples; p={pval:.2g})")
    fig.tight_layout(); fig.savefig(out_dir / "significance.png", dpi=140); plt.close(fig)
    print(f"-> {out_dir}")

# ======================================================================================
# svcca-controls  --  from 10_svcca_controls.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

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

def _main_svccacontrols() -> None:
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
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
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

# ======================================================================================
# functional  --  from 14_functional_grids.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

def _load(p): return torch.load(p, weights_only=False)["layers"].to(torch.float32).numpy()

def _main_functional():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--esm-dir", default="/ssc/results/esm")
    ap.add_argument("--struct-dir", default="/ssc/results/proteinmpnn")
    ap.add_argument("--rand-struct-dir", default="/ssc/results/rand_mpnn")
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--out-dir", default="/ssc/results/functional_grids")
    ap.add_argument("--n-chains", type=int, default=150)
    ap.add_argument("--per-chain", type=int, default=12)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sdir = Path(a.structures_dir); prot = sdir / "proteins"
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    D = {"esm": Path(a.esm_dir), "str": Path(a.struct_dir), "rand": Path(a.rand_struct_dir)}
    use = [c for c in ids if all((d / f"{c}.pt").exists() for d in D.values())][: a.n_chains]
    rng = np.random.default_rng(a.seed)

    # residue-aligned sample for the predictivity grids
    E, S, R = [], [], []
    for c in use:
        L = int(np.load(prot / f"{c}.npz", allow_pickle=True)["ss3"].shape[0])
        e, s, r = _load(D["esm"]/f"{c}.pt"), _load(D["str"]/f"{c}.pt"), _load(D["rand"]/f"{c}.pt")
        if e.shape[1] != L or s.shape[1] != L: continue
        i = rng.choice(L, min(a.per_chain, L), replace=False)
        E.append(e[:, i, :]); S.append(s[:, i, :]); R.append(r[:, i, :])
    E, S, R = np.concatenate(E,1), np.concatenate(S,1), np.concatenate(R,1)
    n = E.shape[1]; te = np.zeros(n, bool); te[rng.choice(n, int(n*a.test_frac), replace=False)] = True
    tr = ~te
    print(f"{n:,} residues; ESM {E.shape[0]} layers, MPNN {S.shape[0]} layers")

    ne, ns = E.shape[0], S.shape[0]
    G_es = np.zeros((ne, ns)); G_se = np.zeros((ne, ns))
    for i in range(ne):
        for j in range(ns):
            m = Ridge(alpha=a.alpha).fit(E[i][tr], S[j][tr])
            G_es[i, j] = r2_score(S[j][te], m.predict(E[i][te]), multioutput="variance_weighted")
            m2 = Ridge(alpha=a.alpha).fit(S[j][tr], E[i][tr])
            G_se[i, j] = r2_score(E[i][te], m2.predict(S[i*0+j][te]*0 + S[j][te]),
                                  multioutput="variance_weighted")
        print(f"  esm L{i}: max R2 -> mpnn {G_es[i].max():.3f} | from mpnn {G_se[i].max():.3f}")

    # stitching sweep: each MPNN layer -> ESM final-layer slot -> ESM lm_head
    import esm as esmlib
    mdl, alph = getattr(esmlib.pretrained, a.esm_model)(); mdl.eval()
    tr_c = [c for k, c in enumerate(use) if k % 10 != 0]
    te_c = [c for k, c in enumerate(use) if k % 10 == 0]
    def st(cids, key, layer):
        return np.concatenate([_load(D[key]/f"{c}.pt")[layer] for c in cids], 0)
    stitch = {}
    for key, lab in [("str", "ProteinMPNN"), ("rand", "untrained MPNN")]:
        accs = []
        for j in range(ns):
            W = Ridge(alpha=a.alpha).fit(st(tr_c, key, j), st(tr_c, "esm", -1))
            cor = tot = 0
            for c in te_c:
                seq = str(np.load(prot/f"{c}.npz", allow_pickle=True)["seq"])[:1022]
                x = torch.from_numpy(W.predict(_load(D[key]/f"{c}.pt")[j][:len(seq)]).astype(np.float32))
                with torch.no_grad(): lg = mdl.lm_head(x[None])[0]
                tgt = torch.tensor([alph.get_idx(ch) for ch in seq])
                cor += int((lg.argmax(-1) == tgt).sum()); tot += len(seq)
            accs.append(cor/tot); print(f"  stitch {lab} enc{j+1} -> ESM lm_head: {cor/tot:.3f}")
        stitch[lab] = accs


    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, a)   # provenance: exactly what produced these outputs
    np.savez(out/"predictivity_grids.npz", esm_to_mpnn=G_es, mpnn_to_esm=G_se,
             stitch_real=stitch["ProteinMPNN"], stitch_rand=stitch["untrained MPNN"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    el = ["emb"]+[f"L{i}" for i in range(1, ne)]; sl = [f"enc{j+1}" for j in range(ns)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6),
                           gridspec_kw={"width_ratios":[1,1,0.8]})
    for k,(G,t) in enumerate([(G_es,"linear predictivity: ESM → MPNN"),
                              (G_se,"linear predictivity: MPNN → ESM")]):
        im = ax[k].imshow(G, aspect="auto", cmap="viridis", vmin=0)
        ax[k].set_xticks(range(ns)); ax[k].set_xticklabels(sl)
        ax[k].set_yticks(range(ne)); ax[k].set_yticklabels(el, fontsize=7)
        ax[k].set_title(t+"\n(held-out R²)", fontsize=10)
        for p in range(ne):
            for q in range(ns):
                ax[k].text(q,p,f"{G[p,q]:.2f}",ha="center",va="center",fontsize=6,
                           color="white" if G[p,q]<0.5*G.max() else "black")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    x = range(ns)
    ax[2].bar([i-0.2 for i in x], stitch["ProteinMPNN"], 0.4, label="ProteinMPNN")
    ax[2].bar([i+0.2 for i in x], stitch["untrained MPNN"], 0.4, label="untrained")
    ax[2].set_xticks(list(x)); ax[2].set_xticklabels(sl)
    ax[2].set_ylabel("residue readout"); ax[2].legend(fontsize=8)
    ax[2].set_title("stitch → ESM's own lm_head", fontsize=10); ax[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out/"functional_grids.png", dpi=140); plt.close(fig)
    print(f"-> {out}")




# ==========================================================================================
# aa-control  --  convergence with the shared training target partialled out
# ==========================================================================================
#
# Both model families are trained to predict amino-acid identity: the sequence models by
# masking it, the inverse-folding models by recovering it from backbone geometry. A shared
# representation of the residue's own amino acid is therefore a common cause, not evidence of
# a shared representation of protein space -- and it is large, because a sequence model's input
# layer is very nearly a pure amino-acid code (SVCCA 1.000 against a one-hot encoding).
#
# This subcommand reports every pair before and after removing that target. Because the one-hot
# design is full rank, projecting it out reduces to subtracting each residue's amino-acid class
# mean. Layers whose residual is degenerate (a pure amino-acid lookup) are excluded from BOTH
# columns so the two are computed over one layer set and remain comparable.

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def _aa_load_layers(p):
    return torch.load(p, weights_only=False)["layers"].to(torch.float32).numpy()


def _aa_collect(ids, a_dir, b_dir, prot_dir, max_residues, seed=42, return_groups=False):
    """Residue-aligned layer stacks for a model pair, plus the amino acid at each residue.

    With return_groups=True also returns a chain index per residue. Residues from one chain
    are strongly correlated, so any resampling that treats them as independent understates
    uncertainty; callers doing statistics must resample chains, not residues.
    """
    rng = np.random.default_rng(seed)
    A_parts, B_parts, aa_parts, grp_parts = [], [], [], []
    total = 0
    n_chain = 0
    for cid in ids:
        pa, pb = a_dir / f"{cid}.pt", b_dir / f"{cid}.pt"
        if not (pa.exists() and pb.exists()):
            continue
        A, B = _aa_load_layers(pa), _aa_load_layers(pb)
        if A.shape[1] != B.shape[1]:
            continue
        npz = np.load(prot_dir / f"{cid}.npz", allow_pickle=True)
        seq = str(npz["seq"])
        L = A.shape[1]
        if len(seq) != L:
            continue
        cap = max(1, min(L, max_residues // 50))
        idx = rng.choice(L, cap, replace=False) if L > cap else np.arange(L)
        A_parts.append(A[:, idx, :]); B_parts.append(B[:, idx, :])
        aa_parts.append(np.array([AA_IDX.get(seq[i], -1) for i in idx]))
        grp_parts.append(np.full(len(idx), n_chain)); n_chain += 1
        total += len(idx)
        if total >= max_residues:
            break
    A = np.concatenate(A_parts, axis=1); B = np.concatenate(B_parts, axis=1)
    aa = np.concatenate(aa_parts); grp = np.concatenate(grp_parts)
    keep = aa >= 0
    if return_groups:
        return A[:, keep, :], B[:, keep, :], aa[keep], grp[keep]
    return A[:, keep, :], B[:, keep, :], aa[keep]


def _aa_residualise(X, aa):
    """Remove E[X | amino acid] -- i.e. project out the one-hot amino-acid design."""
    R = X.astype(np.float64).copy()
    for a in np.unique(aa):
        m = aa == a
        R[m] -= R[m].mean(axis=0, keepdims=True)
    return R


def _aa_degenerate(X, tol=1e-9):
    """A residual with no variance left carries nothing beyond amino-acid identity."""
    return float(np.sum(X ** 2)) < tol


def _aa_grid_peaks(A, B, aa, partial):
    """Peak similarity over the layer grid.

    Residualised layers are computed ONCE per layer rather than once per pair. Layers whose
    residual is degenerate -- a pure amino-acid lookup, e.g. a sequence model's embedding --
    are skipped: with nothing left after the shared target is removed there is no subspace to
    compare, and SVCCA would divide by zero.
    """
    # Use the SAME layer set for raw and partial. A layer whose residual is degenerate is a
    # pure amino-acid lookup (a sequence model's embedding); its raw similarity is a comparison
    # of amino-acid codes, not of representations, and the raw CKA peak otherwise lands there --
    # which would make the raw and partial columns peak at different layers and not be comparable.
    Ares = [_aa_residualise(A[i], aa) for i in range(A.shape[0])]
    Bres = [_aa_residualise(B[j], aa) for j in range(B.shape[0])]
    Aok = [i for i in range(A.shape[0]) if not _aa_degenerate(Ares[i])]
    Bok = [j for j in range(B.shape[0]) if not _aa_degenerate(Bres[j])]
    Alay = {i: (Ares[i] if partial else A[i].astype(np.float64)) for i in Aok}
    Blay = {j: (Bres[j] if partial else B[j].astype(np.float64)) for j in Bok}
    best = {"cka": (-9.0, None), "svcca": (-9.0, None), "mutual_knn": (-9.0, None)}
    for i, Ai in Alay.items():
        Aic = qc.column_center(Ai)
        for j, Bj in Blay.items():
            for k, v in [("cka", qc.linear_cka(Aic, qc.column_center(Bj))),
                         ("svcca", qc.svcca(Ai, Bj)),
                         ("mutual_knn", qc.mutual_knn(Ai, Bj))]:
                if np.isfinite(v) and v > best[k][0]:
                    best[k] = (float(v), f"A{i}xB{j}")
    for k in best:
        if best[k][1] is None:
            best[k] = (float("nan"), "all layers degenerate")
    return best


_AA_PAIRS = [
    ("CARP x ESM-2",             "carp",        "esm",         "within-sequence"),
    ("ESM-IF1 x ProteinMPNN",    "esmif1",      "proteinmpnn", "within-structure"),
    ("ESM-2 650M x ESM-IF1",     "esm650",      "esmif1",      "cross-modality"),
    ("ESM-2 35M x ProteinMPNN",  "esm",         "proteinmpnn", "cross-modality"),
    ("CARP x ESM-IF1",           "carp",        "esmif1",      "cross-modality"),
    ("ESM-2 650M x ProteinMPNN", "esm650",      "proteinmpnn", "cross-modality"),
    ("CARP x ProteinMPNN",       "carp",        "proteinmpnn", "cross-modality"),
    ("ESM-1v x ProteinMPNN",     "esm1v_s1",    "proteinmpnn", "cross-modality"),
    ("untrained x untrained",    "rand_esm",    "rand_mpnn",   "control"),
    ("trained seq x untrained str","esm",       "rand_mpnn",   "control"),
    ("untrained seq x trained str","rand_esm",  "proteinmpnn", "control"),
    ("ESM-1v s1 x ESM-1v s2",    "esm1v_s1",    "esm1v_s2",    "same arch, diff seed"),
]


def _main_aacontrol() -> None:
    ap = argparse.ArgumentParser(
        description="Convergence before and after partialling out amino-acid identity")
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--results-root", default="/ssc/results")
    ap.add_argument("--max-residues", type=int, default=20000)
    ap.add_argument("--out-dir", default="/ssc/results/aa_control")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    out = Path(args.out_dir)
    qc.record_params(out, args)
    results = {}
    for label, a, b, kind in _AA_PAIRS:
        da, db = Path(args.results_root) / a, Path(args.results_root) / b
        if not (da.exists() and db.exists()):
            print(f"  {label:<30} SKIP (missing embeddings)"); continue
        A, B, aa = _aa_collect(ids, da, db, sdir / "proteins", args.max_residues, args.seed)
        raw = _aa_grid_peaks(A, B, aa, partial=False)
        par = _aa_grid_peaks(A, B, aa, partial=True)
        results[label] = {"kind": kind, "n_residues": int(len(aa)),
                          "raw": {k: v[0] for k, v in raw.items()},
                          "partial": {k: v[0] for k, v in par.items()}}
        print(f"  {label:<30} raw cka={raw['cka'][0]:.3f} svcca={raw['svcca'][0]:.3f} "
              f"knn={raw['mutual_knn'][0]:.3f}  |  partial cka={par['cka'][0]:.3f} "
              f"svcca={par['svcca'][0]:.3f} knn={par['mutual_knn'][0]:.3f}", flush=True)
        (out / "partial_convergence.json").write_text(json.dumps(results, indent=2) + "\n")
    qc.record_params(out, args, extra={"n_pairs": len(results)})
    print(f"-> {out}")


# ==========================================================================================
# dispatch
# ==========================================================================================

# ==========================================================================================
# ladder-ci  --  confidence intervals for every pair in the control ladder
# ==========================================================================================

def _main_ladderci() -> None:
    """Resample residues to put a confidence interval on every pair, raw and AA-controlled.

    Only one pair in the study was ever resampled; the rest of the ladder is single
    measurements. Since the central claims are comparisons BETWEEN rows of that ladder
    (cross-modality vs ceiling vs floor), rows without error bars cannot support them.

    Recomputing the full layer x layer grid per resample is not affordable -- one 34x34 grid
    takes hours -- so the peak layer pair is located once on the full sample and the metric
    is then recomputed at that fixed pair across resamples. The interval is therefore
    *conditional on the peak location*, which is stated in the output. Peak-location
    stability is measured separately by the `significance` subcommand.

    Outputs (under --out-dir): ladder_ci.csv, summary.txt
    """
    import argparse
    import csv
    import statistics

    ap = argparse.ArgumentParser(description="Confidence intervals for every ladder pair")
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--results-root", default="/ssc/results")
    ap.add_argument("--max-residues", type=int, default=20000)
    ap.add_argument("--n-resamples", type=int, default=25)
    ap.add_argument("--resample-size", type=int, default=8000,
                    help="residues drawn per resample; smaller keeps mutual k-NN affordable")
    ap.add_argument("--out-dir", default="/ssc/results/ladder_ci")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)

    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    rng = np.random.default_rng(args.seed)

    # (per-matrix prep, pairing) -- see the peak-search loop for why these are split
    METRICS = {"cka": (qc.column_center, qc.linear_cka),
               "svcca": (qc.svcca_reduce, qc.svcca_from_reduced),
               "mutual_knn": (lambda x: x, qc.mutual_knn)}
    rows = []
    for label, an, bn, kind in _AA_PAIRS:
        da, db = Path(args.results_root) / an, Path(args.results_root) / bn
        if not (da.exists() and db.exists()):
            print(f"  {label:<30} SKIP (missing embeddings)", flush=True)
            continue
        A, B, aa, grp = _aa_collect(ids, da, db, sdir / "proteins", args.max_residues,
                                    args.seed, return_groups=True)
        chains = np.unique(grp)
        by_chain = {c: np.flatnonzero(grp == c) for c in chains}

        for mode in ("raw", "partial"):
            Alay = [(_aa_residualise(A[i], aa) if mode == "partial" else A[i].astype(np.float64))
                    for i in range(A.shape[0])]
            Blay = [(_aa_residualise(B[j], aa) if mode == "partial" else B[j].astype(np.float64))
                    for j in range(B.shape[0])]
            # exclude layers that are pure amino-acid lookups, as elsewhere
            Ares = [_aa_residualise(A[i], aa) for i in range(A.shape[0])]
            Bres = [_aa_residualise(B[j], aa) for j in range(B.shape[0])]
            ok_a = [i for i in range(len(Alay)) if not _aa_degenerate(Ares[i])]
            ok_b = [j for j in range(len(Blay)) if not _aa_degenerate(Bres[j])]
            if not ok_a or not ok_b:
                continue

            for mname, (prep, pair_fn) in METRICS.items():
                # 1. locate the peak once, on the full sample.
                # Each layer is prepared ONCE rather than once per partner. Both CKA and SVCCA
                # split into a per-matrix step (centring; SVD-denoise) and a cheap pairing step,
                # and the per-matrix step is what costs -- SVCCA's SVD is ~9 s at 480 dimensions.
                # Preparing inside the double loop repeated it len(ok_b) and len(ok_a) times.
                Aprep = {i: prep(Alay[i]) for i in ok_a}
                Bprep = {j: prep(Blay[j]) for j in ok_b}
                best, at = -9.0, None
                for i in ok_a:
                    for j in ok_b:
                        v = pair_fn(Aprep[i], Bprep[j])
                        if np.isfinite(v) and v > best:
                            best, at = float(v), (i, j)
                del Aprep, Bprep
                if at is None:
                    continue
                i, j = at
                # 2. re-estimate at that fixed layer pair over independent CHAIN-level subsamples.
                #
                # Chains, not residues: residues within a chain are correlated, so resampling them
                # independently is pseudo-replication and yields intervals far too tight.
                #
                # WITHOUT replacement, unlike a textbook bootstrap. Drawing chains with replacement
                # duplicates whole chains, and a duplicated residue is its own nearest neighbour in
                # BOTH models -- mutual k-NN then scores those pairs as automatic agreement. Measured
                # against the no-duplicate estimate that inflated mutual k-NN by ~26%.
                #
                # All three metrics are also biased upward at small n, so every subsample is drawn to
                # the SAME fixed residue budget and the estimate is reported at that budget. Comparing
                # these numbers against a study using a different budget is not meaningful.
                vals = []
                for _ in range(args.n_resamples):
                    order = rng.permutation(len(chains))
                    take, n = [], 0
                    for c in order:                      # accumulate whole chains up to the budget
                        take.append(by_chain[chains[c]])
                        n += take[-1].size
                        if n >= args.resample_size:
                            break
                    idx = np.concatenate(take)[:args.resample_size]
                    # prep must be redone here: the rows differ every subsample, so a cached
                    # reduction from the peak search would not apply.
                    v = pair_fn(prep(Alay[i][idx]), prep(Blay[j][idx]))
                    if np.isfinite(v):
                        vals.append(float(v))
                if not vals:
                    continue
                m = statistics.fmean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                # PERCENTILE interval of the statistic. The previous 1.96*sd/sqrt(n) was the standard
                # error of the resampling MEAN -- it described how precisely the mean was known, not
                # how much the statistic varies, and was ~5x too narrow.
                lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
                rows.append({"pair": label, "type": kind, "mode": mode, "metric": mname,
                             "full_sample": round(best, 4), "mean": round(m, 4),
                             "lo": round(lo, 4), "hi": round(hi, 4),
                             "sd": round(sd, 4), "n_resamples": len(vals),
                             "peak_at": f"A{i}xB{j}", "n_chains": int(len(chains)),
                             "n_residues_per_subsample": int(args.resample_size),
                             "bootstrap": "chain-level subsample, no replacement, percentile CI"})
                print(f"  {label:<30} {mode:<8} {mname:<11} "
                      f"{m:.3f} [{lo:.3f}, {hi:.3f}]  sd {sd:.4f}  ({len(chains)} chains)", flush=True)
        with (out / "ladder_ci.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    lines = ["Convergence for every ladder pair, raw and amino-acid-controlled.",
             f"Point estimate and 95% percentile interval over {args.n_resamples} independent",
             f"CHAIN-level subsamples of {args.resample_size:,} residues, drawn without replacement.",
             "",
             "Read the estimate as 'the value at this residue budget'. CKA, SVCCA and mutual k-NN are",
             "all biased upward at smaller n (mutual k-NN worst), so these numbers are comparable",
             "ACROSS THE ROWS of this table and not against a study using a different budget.",
             "full_sample is the value on the whole collected sample, reported for reference only --",
             "it is a different quantity and will sit outside the interval.",
             "The peak layer pair is located once on the full sample; intervals are conditional on it.", ""]
    for r in rows:
        lines.append(f"  {r['pair']:<32} {r['mode']:<8} {r['metric']:<11} "
                     f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    qc.record_params(out, args, extra={"n_pairs": len({r["pair"] for r in rows})})
    print(f"-> {out}")


_SUBCOMMANDS = {
    "grids": _main_grids,
    "supervised": _main_supervised,
    "significance": _main_significance,
    "svcca-controls": _main_svccacontrols,
    "functional": _main_functional,
    "aa-control": _main_aacontrol,
    "ladder-ci": _main_ladderci,
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
