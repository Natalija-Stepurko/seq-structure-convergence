"""
04_convergence.py — cross-model representational convergence (the study's core).

Measures whether the sequence model (ESM-2) and the structure model (ProteinMPNN)
organise protein space the same way, layer by layer, on the *same residues*. For every
(ESM-2 layer i, structure layer j) pair it computes three representational-similarity
metrics and a permutation baseline:

    - linear CKA          (feature-space)
    - SVCCA               (subspace canonical correlation)
    - mutual k-NN         (Platonic-hypothesis neighbour agreement; Huh et al. 2024)

The two models are trained on opposite signals (masked-LM sequence vs. structure->sequence)
and have different architectures, so above-baseline similarity is evidence of a shared,
modality-independent representation. Residues are aligned by construction: both .pt files
for a chain share the same residue order (from the stage-01 npz).

Outputs (under --out-dir, default results/convergence):
    grids.npz            cka/svcca/mutual_knn [n_esm, n_struct] + permuted-CKA baseline
    convergence.png      the three heatmaps + peak-alignment summary
    summary.txt          peak (layer_esm, layer_struct) per metric, baseline

Usage:
    uv run python scripts/04_convergence.py \\
        --esm-dir /ssc/results/esm --struct-dir /ssc/results/proteinmpnn \\
        --structures-dir /ssc/structures --out-dir /ssc/results/convergence
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import qc_common as qc


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


def _plot(grids, esm_labels, st_labels, out_png):
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


def main() -> None:
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

    nE, nS = esm_all.shape[0], st_all.shape[0]
    esm_labels = ["emb"] + [f"b{i}" for i in range(1, nE)]
    st_labels = [f"enc{j+1}" for j in range(nS)]

    np.savez(out_dir / "grids.npz", esm_labels=esm_labels, st_labels=st_labels, **grids)
    _plot(grids, esm_labels, st_labels, out_dir / "convergence.png")

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


if __name__ == "__main__":
    main()
