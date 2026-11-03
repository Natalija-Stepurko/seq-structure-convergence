r"""
09_mutants.py -- Does a single point mutation move the embeddings, and where? (specificity check)

Two classic mutation systems, wild type against single point mutants:

    T4 lysozyme (PDB 2LZM)  L99 buried: L99A (the classic cavity mutant), L99I (conservative)
                            K16 exposed: K16R (conservative), K16E (charge reversal)
    ubiquitin   (PDB 1UBQ)  V26 buried: V26A, V26I;  K48 exposed: K48R, K48E;
                            I44A (the hydrophobic patch that binds ubiquitin receptors)

ESM-2 35M embeds wild type and mutant (all 13 representations). For every layer the change is
measured at the mutated residue and at the other residues, grouped by C-alpha distance from the
site, as is and after the amino-acid subtraction (each amino acid's mean vector, taken from the
study's own embeddings of the dataset, subtracted). Changes are in units of that layer's typical
within-amino-acid spread (root-mean-square distance of a residue from its amino acid's mean), so a
shift of 1.0 is as large as the usual difference between two residues of the same amino acid.

ProteinMPNN's encoder reads only backbone coordinates, so wild type and mutant on the same backbone
give the same encoder embedding by construction; the stage runs it anyway and checks the difference
is exactly zero.

    uv run python scripts/09_mutants.py --structures-dir structures --results-root results
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent / "vendor" / "proteinmpnn"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ssc import metrics as qc  # noqa: E402

PROTEINS = {
    "2LZM": ("A", "T4 lysozyme", ["L99A", "L99I", "K16R", "K16E"]),
    "1UBQ": ("A", "ubiquitin", ["V26A", "V26I", "K48R", "K48E", "I44A"]),
}
NEAR, MID = 8.0, 16.0           # Å, C-alpha distance bands around the mutated residue


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_stats(ids, results_root, structures, n_chains):
    """Per layer: each amino acid's mean ESM-2 vector and the RMS within-amino-acid spread,
    streamed over the first n_chains of the study's own embeddings."""
    sums = cnt = sq = None
    used = 0
    for cid in ids:
        pt = results_root / "esm" / f"{cid}.pt"
        if not pt.exists():
            continue
        E = torch.load(pt, weights_only=False)["layers"].to(torch.float64).numpy()   # [13, L, 480]
        seq = str(np.load(structures / "proteins" / f"{cid}.npz", allow_pickle=True)["seq"])
        if len(seq) != E.shape[1]:
            continue
        if sums is None:
            sums = {}; cnt = {}; sq = np.zeros(E.shape[0])
        for i, a in enumerate(seq):
            if a not in sums:
                sums[a] = np.zeros((E.shape[0], E.shape[2])); cnt[a] = 0
            sums[a] += E[:, i, :]; cnt[a] += 1
            sq += (E[:, i, :] ** 2).sum(axis=1)
        used += 1
        if used >= n_chains:
            break
    means = {a: sums[a] / cnt[a] for a in sums}
    n = sum(cnt.values())
    # E||h - mu_a||^2 = E||h||^2 - sum_a p_a ||mu_a||^2
    between = sum(cnt[a] * (means[a] ** 2).sum(axis=1) for a in means) / n
    spread = np.sqrt(sq / n - between)
    return means, spread, used, n


def esm_layers(model, alphabet, seq):
    _, _, tok = alphabet.get_batch_converter()([("x", seq)])
    with torch.no_grad():
        out = model(tok, repr_layers=list(range(model.num_layers + 1)))
    return torch.stack([out["representations"][k][0, 1:len(seq) + 1]
                        for k in range(model.num_layers + 1)]).double().numpy()


def run():
    ap = argparse.ArgumentParser(description="Point-mutant specificity of the embeddings")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results/mutants")
    ap.add_argument("--raw-dir", default="structures/raw")
    ap.add_argument("--reference-chains", type=int, default=500)
    ap.add_argument("--mpnn-weights", default=None)
    args = ap.parse_args()
    S, R, out = Path(args.structures_dir), Path(args.results_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_module("ds", Path(__file__).parent / "01_dataset.py")
    ex = load_module("ex", Path(__file__).parent / "02_extract.py")
    ids = [r["id"] for r in map(json.loads, (S / "index.jsonl").open()) if r and r.get("valid", True)]
    means, spread, used, n_res = reference_stats(ids, R, S, args.reference_chains)
    print(f"  amino-acid means and spread from {used} chains, {n_res:,} residues", flush=True)

    import esm
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()
    weights = Path(args.mpnn_weights) if args.mpnn_weights else \
        Path(torch.hub.get_dir()).parent / "proteinmpnn" / "v_48_020.pt"
    if not weights.exists():
        weights = Path.home() / ".cache/torch/proteinmpnn/v_48_020.pt"
    mpnn = ex._build_model(weights)

    rows, mpnn_check = [], {}
    for pdb, (chain, name, muts) in PROTEINS.items():
        f = ds._featurise_chain(pdb, chain, Path(args.raw_dir))
        seq, rid = f["seq"], [int(x) for x in f["res_ids"]]
        ca = f["coords_bb"][:, 1, :]
        wt_layers = esm_layers(model, alphabet, seq)
        enc_wt = ex._encode(mpnn, f["coords_bb"]).double().numpy()
        for m in muts:
            wt_aa, pos, mt_aa = m[0], int(m[1:-1]), m[-1]
            i = rid.index(pos)
            assert seq[i] == wt_aa, f"{pdb} {m}: residue {pos} is {seq[i]}, not {wt_aa}"
            mseq = seq[:i] + mt_aa + seq[i + 1:]
            mt_layers = esm_layers(model, alphabet, mseq)
            # ProteinMPNN: same backbone, so the encoder sees identical input
            enc_mt = ex._encode(mpnn, f["coords_bb"]).double().numpy()
            mpnn_check[f"{pdb}:{m}"] = float(np.abs(enc_mt - enc_wt).max())
            d = np.linalg.norm(ca - ca[i], axis=1)
            others = np.arange(len(seq)) != i
            bands = {"near": others & (d < NEAR), "mid": others & (d >= NEAR) & (d < MID),
                     "far": others & (d >= MID)}
            for k in range(1, wt_layers.shape[0]):
                # The embedding layer (k = 0) is a lookup: every leucine has the same vector, so the
                # within-amino-acid spread is zero and there is no unit to measure a shift in. It is
                # left out; a substitution simply swaps one lookup vector for another there.
                delta = mt_layers[k] - wt_layers[k]                       # [L, 480]
                raw_site = np.linalg.norm(delta[i])
                # subtraction: the mutated residue's reference mean changes with its amino acid
                res_site = np.linalg.norm(delta[i] - (means[mt_aa][k] - means[wt_aa][k]))
                per = np.linalg.norm(delta, axis=1)                      # other residues: same
                rows.append({"protein": name, "pdb": pdb, "mutation": m, "rsa": round(float(f["rsa"][i]), 3),
                             "layer": f"b{k}",
                             "spread": round(float(spread[k]), 4),
                             "site_raw": round(float(raw_site / spread[k]), 4),
                             "site_residual": round(float(res_site / spread[k]), 4),
                             **{f"{b}_mean": round(float(per[msk].mean() / spread[k]), 5) if msk.any() else ""
                                for b, msk in bands.items()},
                             "n_near": int(bands["near"].sum())})
            last = [r for r in rows if r["mutation"] == m and r["pdb"] == pdb][-1]
            print(f"  {name:<12} {m:<5} rsa {last['rsa']:.2f}  block 12: site {last['site_raw']:.2f} "
                  f"as is, {last['site_residual']:.2f} after subtraction; near {last['near_mean']:.3f}, "
                  f"far {last['far_mean']:.4f}", flush=True)
    with (out / "shifts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(
        {"reference_chains": used, "reference_residues": n_res, "bands_angstrom": [NEAR, MID],
         "embedding_layer": "left out: a lookup with zero within-amino-acid spread, so no unit",
         "spread_by_layer": [round(float(x), 6) for x in spread],
         "proteinmpnn_encoder_max_abs_change": mpnn_check}, indent=1) + "\n")
    qc.record_params(out, args, extra={"proteins": {k: v[2] for k, v in PROTEINS.items()}})
    print(f"  ProteinMPNN encoder, largest change over all mutants: {max(mpnn_check.values()):.1e}")
    print(f"-> {out}")


if __name__ == "__main__":
    run()
