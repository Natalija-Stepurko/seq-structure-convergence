"""Export a residue-aligned embedding benchmark: every model arm, every layer, one fixed subsample.

The same residues are described by sequence models, structure models, two seeded copies of one
model and untrained copies, all stored at every layer and aligned row for row. That makes the
set a ready-made test bed for representation-similarity measures: the seeded pair is a ceiling,
the untrained copies are floors, and the sequence/structure pairs sit in between.

The subsample is whole chains, drawn in a seeded random order until the residue budget is met,
so chain-level resampling stays possible. `reference_scores.csv` gives CKA, SVCCA and mutual
k-NN for each ladder pair on exactly these residues, computed with `ssc.metrics`, raw and with
the per-amino-acid mean subtracted, each beside a scrambled-residue null.

Output layout (under --out-dir):
    embeddings/<arm>.safetensors   tensor "layers": float16 [n_layers, n_residues, dim]
    residues.csv                   one row per residue, same order as the tensors
    chains.csv                     one row per chain
    reference_scores.csv
    README.md is written separately (the dataset card)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssc import metrics as qc

# arm name -> (results folder, model, what it reads, trained?)
ARMS = {
    "esm2_35m":             ("esm",         "ESM-2 35M (esm2_t12_35M_UR50D)",       "sequence",  True),
    "esm2_650m":            ("esm650",      "ESM-2 650M (esm2_t33_650M_UR50D)",     "sequence",  True),
    "esm1v_seed1":          ("esm1v_s1",    "ESM-1v (esm1v_t33_650M_UR90S_1)",      "sequence",  True),
    "esm1v_seed2":          ("esm1v_s2",    "ESM-1v (esm1v_t33_650M_UR90S_2)",      "sequence",  True),
    "carp_38m":             ("carp",        "CARP-38M",                             "sequence",  True),
    "proteinmpnn":          ("proteinmpnn", "ProteinMPNN encoder (v_48_020)",       "structure", True),
    "esmif1":               ("esmif1",      "ESM-IF1 (esm_if1_gvp4_t16_142M_UR50)", "structure", True),
    "esm2_35m_untrained":   ("rand_esm",    "ESM-2 35M architecture, random weights", "sequence", False),
    "proteinmpnn_untrained": ("rand_mpnn",  "ProteinMPNN encoder, random weights",  "structure", False),
}
# ladder pair label -> (arm A, arm B), in the orientation ladder_ci.csv records peak_at
PAIRS = {
    "ESM-1v s1 x ESM-1v s2":       ("esm1v_seed1", "esm1v_seed2"),
    "CARP x ESM-2":                ("carp_38m", "esm2_35m"),
    "ESM-IF1 x ProteinMPNN":       ("esmif1", "proteinmpnn"),
    "ESM-2 35M x ProteinMPNN":     ("esm2_35m", "proteinmpnn"),
    "ESM-2 650M x ProteinMPNN":    ("esm2_650m", "proteinmpnn"),
    "ESM-2 650M x ESM-IF1":        ("esm2_650m", "esmif1"),
    "CARP x ESM-IF1":              ("carp_38m", "esmif1"),
    "CARP x ProteinMPNN":          ("carp_38m", "proteinmpnn"),
    "ESM-1v x ProteinMPNN":        ("esm1v_seed1", "proteinmpnn"),
    "untrained x untrained":       ("esm2_35m_untrained", "proteinmpnn_untrained"),
    "trained seq x untrained str": ("esm2_35m", "proteinmpnn_untrained"),
    "untrained seq x trained str": ("esm2_35m_untrained", "proteinmpnn"),
}
SS3 = {"a": "helix", "b": "sheet", "c": "coil"}
AA = "ACDEFGHIKLMNPQRSTVWY"
BURIAL_RSA = 0.25


def load(path):
    return torch.load(path, weights_only=False)["layers"].to(torch.float16)


def residualise(X, aa):
    R = X.astype(np.float64).copy()
    for a in np.unique(aa):
        m = aa == a
        R[m] -= R[m].mean(axis=0, keepdims=True)
    return R


CHAIN_ANNOT = ["uniprot", "kingdom", "enzyme", "ec_class", "protein_class", "localisation"]


def export_protein_level(args):
    """One vector per protein: the mean over its residues, at every layer of every arm.

    Mean pooling is what the study's protein-level probes use. Chains are those every arm
    describes in full, in index order; `proteins.csv` gives row order and labels.
    """
    sdir, R = Path(args.structures_dir), Path(args.results_root)
    out = Path(args.out_dir); (out / "embeddings_protein").mkdir(parents=True, exist_ok=True)
    index = [json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()]
    annot = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "annotations.jsonl").open() if l.strip()}
    rows = [r for r in index if r.get("valid", True)]
    keep, pooled = [], {a: [] for a in ARMS}
    for n, r in enumerate(rows):
        cid, L = r["id"], int(r["length"])
        paths = {a: R / d / f"{cid}.pt" for a, (d, *_) in ARMS.items()}
        if not all(p.exists() for p in paths.values()):
            continue
        means = {}
        for a, p in paths.items():
            X = torch.load(p, weights_only=False)["layers"]
            if X.shape[1] != L:
                break
            means[a] = X.to(torch.float32).mean(dim=1).to(torch.float16)
        else:
            keep.append(r)
            for a, m in means.items():
                pooled[a].append(m)
        if (n + 1) % 500 == 0:
            print(f"  {n + 1:,} / {len(rows):,} chains read, {len(keep):,} kept", flush=True)
    print(f"{len(keep):,} of {len(rows):,} chains described in full by every arm", flush=True)
    meta = {}
    for a, (d, name, reads, trained) in ARMS.items():
        X = torch.stack(pooled[a], dim=1).contiguous()           # [n_layers, n_proteins, dim]
        save_file({"layers": X}, out / "embeddings_protein" / f"{a}.safetensors",
                  metadata={"model": name, "reads": reads, "trained": str(trained),
                            "pooling": "mean over residues"})
        meta[a] = {"n_layers": X.shape[0], "dim": X.shape[2], "mb": round(X.numel() * 2 / 1e6, 1)}
        print(f"  {a:<24} {tuple(X.shape)}  {meta[a]['mb']:>7.1f} MB", flush=True)
    json.dump(meta, open(out / "arms_protein.json", "w"), indent=1)
    fields = ["chain_id", "pdb", "chain", "length", "cath_code", "cath_class", "cath_arch",
              "cath_topol", "cath_homol"] + CHAIN_ANNOT
    with (out / "proteins.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in keep:
            a = annot.get(r["id"], {})
            w.writerow({"chain_id": r["id"], "pdb": r["pdb"], "chain": r["chain"], "length": r["length"],
                        **{k: r[k] for k in ("cath_code", "cath_class", "cath_arch", "cath_topol", "cath_homol")},
                        **{k: a.get(k, "") for k in CHAIN_ANNOT}})
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--results-root", default="/ssc/results")
    ap.add_argument("--ladder-csv", default=str(Path(__file__).resolve().parents[1] / "results/ladder_ci/ladder_ci.csv"))
    ap.add_argument("--out-dir", default="/ssc/hf/seq-structure-convergence")
    ap.add_argument("--budget", type=int, default=10000, help="residues; whole chains until met")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-scores", action="store_true")
    ap.add_argument("--protein-level", action="store_true",
                    help="write only the protein-level set: one mean vector per protein, every arm "
                         "and layer, for every chain all arms describe in full")
    args = ap.parse_args()
    if args.protein_level:
        return export_protein_level(args)

    sdir, R = Path(args.structures_dir), Path(args.results_root)
    out = Path(args.out_dir); (out / "embeddings").mkdir(parents=True, exist_ok=True)
    index = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()}
    annot = {json.loads(l)["id"]: json.loads(l) for l in (sdir / "annotations.jsonl").open() if l.strip()}
    ids = [c for c, r in index.items() if r.get("valid", True)]

    # ---- choose whole chains in a seeded order, keeping only chains every arm describes in full
    order = np.random.default_rng(args.seed).permutation(len(ids))
    chosen, parts, total = [], {a: [] for a in ARMS}, 0
    for k in order:
        cid = ids[k]
        paths = {a: R / d / f"{cid}.pt" for a, (d, *_ ) in ARMS.items()}
        if not all(p.exists() for p in paths.values()):
            continue
        L = int(index[cid]["length"])
        tens = {a: load(p) for a, p in paths.items()}
        if any(t.shape[1] != L for t in tens.values()):
            continue
        chosen.append(cid)
        for a, t in tens.items():
            parts[a].append(t)
        total += L
        if total >= args.budget:
            break
    print(f"{len(chosen)} chains, {total:,} residues", flush=True)

    # ---- embeddings, one file per arm
    meta = {}
    for a, (d, name, reads, trained) in ARMS.items():
        X = torch.cat(parts[a], dim=1).contiguous()
        save_file({"layers": X}, out / "embeddings" / f"{a}.safetensors",
                  metadata={"model": name, "reads": reads, "trained": str(trained),
                            "layer_0": "input embedding" if a.startswith(("esm2", "esm1v")) else "first stored layer"})
        meta[a] = {"n_layers": X.shape[0], "dim": X.shape[2], "mb": round(X.numel() * 2 / 1e6, 1)}
        print(f"  {a:<24} layers={X.shape[0]:>2}  dim={X.shape[2]:>4}  {meta[a]['mb']:>7.1f} MB", flush=True)
    json.dump(meta, open(out / "arms.json", "w"), indent=1)

    # ---- residue and chain tables
    rows, aa_all = [], []
    for cid in chosen:
        z = np.load(sdir / "proteins" / f"{cid}.npz", allow_pickle=True)
        tpath = sdir / "residue_targets" / f"{cid}.npz"
        t = np.load(tpath, allow_pickle=True) if tpath.exists() else None
        seq = str(z["seq"])
        for i in range(len(seq)):
            rsa = float(z["rsa"][i])
            rows.append({
                "chain_id": cid, "position": i, "amino_acid": seq[i],
                "secondary_structure": SS3.get(str(z["ss3"][i]), ""),
                "rsa": round(rsa, 4) if np.isfinite(rsa) else "",
                "burial": ("buried" if rsa < BURIAL_RSA else "exposed") if np.isfinite(rsa) else "",
                "bfactor_norm": round(float(t["bfactor"][i]), 4) if t is not None and np.isfinite(t["bfactor"][i]) else "",
                "binding_site": int(t["binding_site"][i]) if t is not None else "",
                "active_site": int(t["active_site"][i]) if t is not None else "",
                "ptm_site": int(t["ptm_site"][i]) if t is not None else ""})
            aa_all.append(AA.find(seq[i]))
    with (out / "residues.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    with (out / "chains.csv").open("w", newline="") as f:
        fields = ["chain_id", "pdb", "chain", "length", "cath_code", "cath_class", "cath_arch",
                  "cath_topol", "cath_homol"] + CHAIN_ANNOT
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for cid in chosen:
            r, a = index[cid], annot.get(cid, {})
            w.writerow({"chain_id": cid, "pdb": r["pdb"], "chain": r["chain"], "length": r["length"],
                        **{k: r[k] for k in ("cath_code", "cath_class", "cath_arch", "cath_topol", "cath_homol")},
                        **{k: a.get(k, "") for k in CHAIN_ANNOT}})

    if args.skip_scores:
        return
    # ---- reference scores at the ladder's peak layer pairs
    aa = np.array(aa_all)
    keep = aa >= 0
    peaks = {(r["pair"], r["mode"], r["metric"]): r["peak_at"]
             for r in csv.DictReader(open(args.ladder_csv))}
    FUN = {"cka": lambda X, Y: qc.linear_cka(qc.column_center(X), qc.column_center(Y)),
           "svcca": qc.svcca, "mutual_knn": qc.mutual_knn}
    rng = np.random.default_rng(args.seed + 1)
    stacks = {a: torch.cat(parts[a], dim=1) for a in ARMS}
    out_rows = []
    for label, (a, b) in PAIRS.items():
        for mode in ("raw", "partial"):
            for metric, fn in FUN.items():
                at = peaks.get((label, mode, metric))
                if at is None:
                    continue
                i, j = (int(x) for x in at[1:].split("xB"))
                X = stacks[a][i].to(torch.float64).numpy()[keep]
                Y = stacks[b][j].to(torch.float64).numpy()[keep]
                if mode == "partial":
                    X, Y = residualise(X, aa[keep]), residualise(Y, aa[keep])
                v = fn(X, Y)
                vn = fn(X, Y[rng.permutation(len(Y))])
                out_rows.append({"pair": label, "arm_a": a, "arm_b": b, "mode": mode, "metric": metric,
                                 "layer_a": i, "layer_b": j, "score": round(float(v), 4),
                                 "scrambled": round(float(vn), 4), "n_residues": int(keep.sum())})
                print(f"  {label:<30} {mode:<8} {metric:<11} {v:.4f}  scrambled {vn:.4f}", flush=True)
    with (out / "reference_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
