"""
06_significance.py — robustness of the convergence result.

The headline claim (stage 04) is that ESM-2 and ProteinMPNN converge above chance. This
stage quantifies how robust the peak convergence is by resampling residues many times and
comparing the peak cross-model CKA against its permutation baseline over the repeats,
reporting means, 95% confidence intervals, the stability of the peak layer-pair, and an
empirical p-value (fraction of repeats where the real peak did not exceed the permuted one).

Outputs (under --out-dir, default results/convergence/significance):
    summary.txt          peak/baseline means, 95% CIs, modal peak layer-pair, p-value
    significance.png     per-repeat peak vs permuted-baseline with CIs

Usage:
    uv run python scripts/06_significance.py \\
        --esm-dir /ssc/results/esm --struct-dir /ssc/results/proteinmpnn \\
        --structures-dir /ssc/structures --out-dir /ssc/results/convergence/significance \\
        --repeats 25
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import qc_common as qc


def _layers(pt):
    return torch.load(pt, weights_only=False)["layers"].to(torch.float32).numpy()


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


def main() -> None:
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


if __name__ == "__main__":
    main()
