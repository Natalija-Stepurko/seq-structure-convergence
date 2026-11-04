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

`predict` then asks whether the change matters downstream: property predictors trained on the
dataset (as in 08_fusion.py) are applied to wild type and each mutant, raw, after the subtraction
and fused, and the change in each prediction is measured at and around the mutated residue. For
T4 lysozyme L99A the predicted change in solvent accessibility is compared with the change measured
between the crystal structures 1L63 (cysteine-free background) and 1L90 (L99A on it).

    uv run python scripts/09_mutants.py shift   --structures-dir structures --results-root results
    uv run python scripts/09_mutants.py predict --structures-dir structures --results-root results
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


PRED_ARMS = ("esm", "esm_residual", "mpnn", "raw", "residual", "hybrid")
PRED_TASKS = (("rsa", "reg"), ("burial", "clf"), ("ss3", "clf"), ("bfactor", "reg"),
              ("binding_site", "site"), ("active_site", "site"), ("ptm_site", "site"))


def _fit(fx, pred, kind, Xtr, ytr, groups, seed=42):
    """A fitted (scaler, model) pair, trained exactly as 08_fusion.py trains it."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    w = compute_sample_weight("balanced", ytr) if kind == "site" else None
    kw = {} if w is None else {"sample_weight": w}
    if pred == "linear":
        model, _ = fx._cv_linear(kind, Xtr, ytr, groups, w)
        sc = StandardScaler().fit(Xtr)
        return sc, model.fit(sc.transform(Xtr), ytr, **kw)
    Model = fx.XGBRegressor if kind == "reg" else fx.XGBClassifier
    m = Model(n_estimators=80, max_depth=4, tree_method="hist", n_jobs=4, verbosity=0, random_state=seed)
    return None, m.fit(Xtr, ytr, **kw)


def _out(sm, kind, X):
    X = X if sm[0] is None else sm[0].transform(X)
    return sm[1].predict(X) if kind == "reg" else sm[1].predict_proba(X)


def _change(kind, a, b):
    """Per residue: absolute change for a regression, total-variation distance between the two
    predicted class distributions for a classifier (0 = same, 1 = completely different)."""
    return np.abs(a - b) if kind == "reg" else 0.5 * np.abs(a - b).sum(axis=1)


def _arms_for(seq_wt, seq_mt, E_wt, E_mt, M_wt, M_mt, means_e, means_m):
    from ssc import fusion as fu
    def res(X, seq, means):
        return np.stack([X[i] - means.get(a, means["_all"]) for i, a in enumerate(seq)])
    out = {}
    for tag, seq, E, M in (("wt", seq_wt, E_wt, M_wt), ("mt", seq_mt, E_mt, M_mt)):
        oh, er, mr = fu.onehot(np.array(list(seq))), res(E, seq, means_e), res(M, seq, means_m)
        out[tag] = {"esm": E, "esm_residual": er, "mpnn": M, "raw": np.hstack([E, M]),
                    "residual": np.hstack([er, mr]), "hybrid": np.hstack([oh, er, mr])}
    return out


def predict():
    ap = argparse.ArgumentParser(description="Do the mutants change the predicted properties?")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results/mutants")
    ap.add_argument("--raw-dir", default="structures/raw")
    ap.add_argument("--mpnn-weights", default=None)
    args = ap.parse_args()
    S, R, out = Path(args.structures_dir), Path(args.results_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_module("ds", Path(__file__).parent / "01_dataset.py")
    ex = load_module("ex", Path(__file__).parent / "02_extract.py")
    fx = load_module("fx", Path(__file__).parent / "08_fusion.py")

    # training data: 08_fusion.py's residues for refit seed 42, its training chains
    data, _ = fx.load(S, R, None)
    D = data[42]
    C = int(D["grp"].max()) + 1
    test_chains = np.random.default_rng(42).choice(C, max(1, int(C * fx.TEST_FRAC)), replace=False)
    train = ~np.isin(D["grp"], test_chains)
    means_e = {a: D["esm"][train & (D["aa"] == a)].mean(0) for a in np.unique(D["aa"][train])}
    means_m = {a: D["mpnn"][train & (D["aa"] == a)].mean(0) for a in np.unique(D["aa"][train])}
    means_e["_all"], means_m["_all"] = D["esm"][train].mean(0), D["mpnn"][train].mean(0)
    A = fx.arms(D, train)
    A["esm_residual"] = A["residual"][:, :D["esm"].shape[1]]
    models = {}
    for task, kind in PRED_TASKS:
        y = D[task]
        lab = train & (np.isfinite(y) if kind == "reg" else (y >= 0))
        for pred in ("linear", "xgb"):
            for arm in PRED_ARMS:
                models[(task, pred, arm)] = _fit(fx, pred, kind, A[arm][lab], y[lab], D["grp"][lab])
        print(f"  trained {task}", flush=True)

    import esm
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D(); model.eval()
    weights = Path(args.mpnn_weights) if args.mpnn_weights else \
        Path(torch.hub.get_dir()).parent / "proteinmpnn" / "v_48_020.pt"
    mpnn = ex._build_model(weights)
    L11, ENC3 = fx.LAYERS["esm"], fx.LAYERS["proteinmpnn"]

    rows = []
    for pdb, (chain, name, muts) in PROTEINS.items():
        f = ds._featurise_chain(pdb, chain, Path(args.raw_dir))
        seq, rid, ca = f["seq"], [int(x) for x in f["res_ids"]], f["coords_bb"][:, 1, :]
        E_wt = esm_layers(model, alphabet, seq)[L11].astype(np.float32)
        M = ex._encode(mpnn, f["coords_bb"])[ENC3].numpy()
        for m in muts:
            i = rid.index(int(m[1:-1])); assert seq[i] == m[0]
            mseq = seq[:i] + m[-1] + seq[i + 1:]
            E_mt = esm_layers(model, alphabet, mseq)[L11].astype(np.float32)
            X = _arms_for(seq, mseq, E_wt, E_mt, M, M, means_e, means_m)
            d = np.linalg.norm(ca - ca[i], axis=1); others = np.arange(len(seq)) != i
            near, far = others & (d < NEAR), others & (d >= MID)
            for task, kind in PRED_TASKS:
                for pred in ("linear", "xgb"):
                    for arm in PRED_ARMS:
                        sm = models[(task, pred, arm)]
                        a, b = _out(sm, kind, X["wt"][arm]), _out(sm, kind, X["mt"][arm])
                        ch = _change(kind, a, b)
                        flips = 0 if kind == "reg" else int((a.argmax(1) != b.argmax(1))[near | (np.arange(len(seq)) == i)].sum())
                        rows.append({"protein": name, "mutation": m, "task": task, "predictor": pred, "arm": arm,
                                     "site": round(float(ch[i]), 4), "near_mean": round(float(ch[near].mean()), 5),
                                     "far_mean": round(float(ch[far].mean()), 5), "flips_site_or_near": flips})
        print(f"  {name}: predictions done", flush=True)
    with (out / "predictions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # protein-level predictions: whole-protein averages, trained on 08_fusion.py's proteins (seed 42)
    P = fx.load_protein(S, R, None)
    Y = fx.protein_targets(P)
    Cp = len(P["counts"])
    test_p = np.random.default_rng(42).choice(Cp, max(1, int(Cp * fx.TEST_FRAC)), replace=False)
    train_p = ~np.isin(np.arange(Cp), test_p)
    AP = fx.protein_arms(P, train_p)
    AP["esm_residual"] = AP["residual"][:, :P["esm"].shape[1]]
    AP["hybrid"] = AP["hybrid"]                                   # composition + both residuals
    tot = P["counts"][train_p].sum(0)
    pm = {k: P[f"sum_{k}"][train_p].sum(0) / np.maximum(tot[:, None], 1) for k in ("esm", "mpnn")}

    def pooled(seq, E, Mx):
        codes = np.array([fx.AA_CODE.get(a, 20) for a in seq])
        frac = np.bincount(codes, minlength=21) / len(seq)
        er, mr = E.mean(0) - frac @ pm["esm"], Mx.mean(0) - frac @ pm["mpnn"]
        comp = frac[:20]
        e, mm = E.mean(0), Mx.mean(0)
        return {"esm": e, "esm_residual": er, "mpnn": mm, "raw": np.hstack([e, mm]),
                "residual": np.hstack([er, mr]), "hybrid": np.hstack([comp, er, mr])}

    pmodels = {}
    for task, y in Y.items():
        tr = train_p & (y >= 0)
        for pred in ("linear", "xgb"):
            for arm in PRED_ARMS:
                pmodels[(task, pred, arm)] = _fit(fx, pred, "pclf", AP[arm][tr], y[tr], np.arange(Cp)[tr])
    print(f"  trained {len(Y)} protein-level targets", flush=True)
    prow = []
    for pdb, (chain, name, muts) in PROTEINS.items():
        f = ds._featurise_chain(pdb, chain, Path(args.raw_dir))
        seq, rid = f["seq"], [int(x) for x in f["res_ids"]]
        E_wt = esm_layers(model, alphabet, seq)[L11].astype(np.float32)
        M = ex._encode(mpnn, f["coords_bb"])[ENC3].numpy()
        wt = pooled(seq, E_wt, M)
        for m in muts:
            i = rid.index(int(m[1:-1]))
            mseq = seq[:i] + m[-1] + seq[i + 1:]
            mt = pooled(mseq, esm_layers(model, alphabet, mseq)[L11].astype(np.float32), M)
            for task in Y:
                for pred in ("linear", "xgb"):
                    for arm in PRED_ARMS:
                        sm = pmodels[(task, pred, arm)]
                        a, b = _out(sm, "pclf", wt[arm][None]), _out(sm, "pclf", mt[arm][None])
                        prow.append({"protein": name, "mutation": m, "task": task, "predictor": pred, "arm": arm,
                                     "change": round(float(0.5 * np.abs(a - b).sum()), 5),
                                     "class_changed": int(a.argmax() != b.argmax())})
    with (out / "predictions_protein.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0])); w.writeheader(); w.writerows(prow)
    print(f"  protein-level: {sum(r['class_changed'] for r in prow)} class changes in {len(prow)} predictions",
          flush=True)

    # ground truth: T4 lysozyme L99A, crystal structures 1L63 (background) and 1L90 (L99A on it)
    g0, g1 = ds._featurise_chain("1L63", "A", Path(args.raw_dir)), ds._featurise_chain("1L90", "A", Path(args.raw_dir))
    common = sorted(set(int(x) for x in g0["res_ids"]) & set(int(x) for x in g1["res_ids"]))
    i0 = [list(g0["res_ids"]).index(r) for r in common]; i1 = [list(g1["res_ids"]).index(r) for r in common]
    seq0 = "".join(g0["seq"][k] for k in i0); seq1 = "".join(g1["seq"][k] for k in i1)
    k99 = common.index(99)
    assert seq0[k99] == "L" and seq1[k99] == "A", (seq0[k99], seq1[k99])
    assert sum(a != b for a, b in zip(seq0, seq1)) == 1, "the two structures should differ at residue 99 only"
    meas = g1["rsa"][i1] - g0["rsa"][i0]
    ss_changed = int((g1["ss3"][i1] != g0["ss3"][i0]).sum())
    ca0 = g0["coords_bb"][i0, 1, :]
    dist = np.linalg.norm(ca0 - ca0[k99], axis=1)
    E0 = esm_layers(model, alphabet, seq0)[L11].astype(np.float32)
    E1 = esm_layers(model, alphabet, seq1)[L11].astype(np.float32)
    M0 = ex._encode(mpnn, g0["coords_bb"][i0])[ENC3].numpy()
    M1 = ex._encode(mpnn, g1["coords_bb"][i1])[ENC3].numpy()     # the real mutant backbone
    truth = {"residues": len(common), "measured_abs_rsa_change_mean": round(float(np.nanmean(np.abs(meas))), 4),
             "measured_ss3_changes": ss_changed, "by_arm": {}}
    ok = np.isfinite(meas)
    for backbone, Mm in (("same backbone", M0), ("mutant crystal backbone", M1)):
        X = _arms_for(seq0, seq1, E0, E1, M0, Mm, means_e, means_m)
        for pred in ("linear", "xgb"):
            for arm in PRED_ARMS:
                sm = models[("rsa", pred, arm)]
                dp = _out(sm, "reg", X["mt"][arm]) - _out(sm, "reg", X["wt"][arm])
                if np.allclose(dp, 0):
                    r_all = r_near = float("nan")
                else:
                    r_all = float(np.corrcoef(dp[ok], meas[ok])[0, 1])
                    nn = ok & (dist < NEAR)
                    r_near = float(np.corrcoef(dp[nn], meas[nn])[0, 1])
                truth["by_arm"][f"{backbone}|{pred}|{arm}"] = {
                    "r_all": round(r_all, 3), "r_within_8A": round(r_near, 3),
                    "predicted_change_at_99": round(float(dp[k99]), 4),
                    "measured_change_at_99": round(float(meas[k99]), 4)}
    (out / "l99a_ground_truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    qc.record_params(out, args, filename="params_predict.json",
                     extra={"arms": PRED_ARMS, "tasks": [t for t, _ in PRED_TASKS], "train_seed": 42})
    print(json.dumps({k: v for k, v in truth.items() if k != "by_arm"}, indent=1))
    print(f"-> {out / 'predictions.csv'}, {out / 'predictions_protein.csv'}, {out / 'l99a_ground_truth.json'}")


_SUBCOMMANDS = {"shift": run, "predict": predict}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _SUBCOMMANDS:
        print(__doc__)
        raise SystemExit(2)
    _SUBCOMMANDS[sys.argv.pop(1)]()
