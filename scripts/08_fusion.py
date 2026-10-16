r"""
08_fusion.py -- Are amino-acid-centred representations easier to combine? (research/future_work.md, idea 2)

The design is fixed in research/future_work.md before this stage ran. In short: one layer per model
(ESM-2 35M L11, ProteinMPNN encoder 3, untrained ProteinMPNN encoder 3, ESM-2 650M L32); residues and
chain-level splits drawn as in 05_probes.py over five refits; nine feature arms; linear, boosted-tree
and MLP predictors; 1%, 10% and 100% of the training chains; seven residue targets.

Subcommands:

    run         fit every (refit, training size, target, predictor, arm) -> results/fusion/scores.csv
    summarise   paired gains, variance partitioning and the decision rule -> results/fusion/gains.csv,
                results/fusion/summary.json

    uv run python scripts/08_fusion.py run --structures-dir structures --results-root results
    uv run python scripts/08_fusion.py summarise

`run` writes one file per (refit, size) under results/fusion/parts/ and skips any that exist, so an
interrupted run resumes where it stopped.
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

THREADS = int(os.environ.get("SSC_THREADS", "4"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))      # one thread pool per worker, set before numpy loads

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge   # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold               # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor              # noqa: E402

from ssc import fusion as fu                                 # noqa: E402
from ssc import metrics as qc                                # noqa: E402

warnings.filterwarnings("ignore")

# The layer that scores best on the most residue targets in the published probe results
# (results/probes_ci, results/probes650): see research/future_work.md.
LAYERS = {"esm": 11, "proteinmpnn": 2, "rand_mpnn": 2, "esm650": 32}   # indices into the saved stacks
SEEDS = (42, 43, 44, 45, 46)
SIZES = (0.01, 0.10, 1.00)
PER_CHAIN, TEST_FRAC, BURIAL_RSA = 8, 0.25, 0.25
SS3_IDX = {"a": 0, "b": 1, "c": 2}
MIN_POSITIVES = 10
ARMS = ("esm", "mpnn", "aa", "raw", "raw_aa", "residual", "hybrid", "esm_untrained_mpnn", "esm650")
FUSED = ("raw", "raw_aa", "residual", "hybrid", "esm_untrained_mpnn")
PREDICTORS = ("linear", "xgb", "mlp")
# (target, kind, metric): reg -> R^2, clf -> accuracy, site -> macro F1 with balanced weights
TASKS = (("rsa", "reg", "r2"), ("bfactor", "reg", "r2"), ("burial", "clf", "accuracy"),
         ("ss3", "clf", "accuracy"), ("binding_site", "site", "f1"), ("active_site", "site", "f1"),
         ("ptm_site", "site", "f1"))
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
LOGIT_CS = (0.001, 0.01, 0.1, 1.0, 10.0)


# ==========================================================================================
# data
# ==========================================================================================

def _layer(path: Path, idx: int) -> np.ndarray:
    return torch.load(path, mmap=True, weights_only=False)["layers"][idx].to(torch.float32).numpy()


def load(structures: Path, results: Path, max_chains: int | None):
    """One pass over the chains: the chosen layer of each model and the residue labels, for every
    refit's residue draw. Draws follow 05_probes.py: 8 random residues per chain plus every
    binding-, active- and PTM-site residue, a separate random stream per refit."""
    ids = [r["id"] for r in map(json.loads, (structures / "index.jsonl").open())
           if r and r.get("valid", True)]
    rngs = {s: np.random.default_rng(s) for s in SEEDS}
    per = {s: {k: [] for k in ("esm", "mpnn", "rand", "esm650", "aa", "grp", "ss3", "burial", "rsa",
                               "bfactor", "binding_site", "active_site", "ptm_site")} for s in SEEDS}
    g, t0 = 0, time.time()
    for cid in ids:
        paths = {m: results / m / f"{cid}.pt" for m in LAYERS}
        if not all(p.exists() for p in paths.values()):
            continue
        npz = np.load(structures / "proteins" / f"{cid}.npz", allow_pickle=True)
        L = int(npz["ss3"].shape[0])
        feats = {m: _layer(p, LAYERS[m]) for m, p in paths.items()}
        if any(f.shape[0] != L for f in feats.values()):
            continue
        rt = structures / "residue_targets" / f"{cid}.npz"
        t = np.load(rt) if rt.exists() else None
        force = np.array([], int)
        if t is not None:
            force = np.unique(np.concatenate([np.where(t[k] == 1)[0]
                                              for k in ("binding_site", "active_site", "ptm_site")])).astype(int)
        pool = np.setdiff1d(np.arange(L), force)
        seq = np.array(list(str(npz["seq"])))
        if t is not None:
            bf = t["bfactor"].astype(np.float64)
            bf = (bf - np.nanmean(bf)) / (np.nanstd(bf) + 1e-6)     # flexibility within the chain
        for s in SEEDS:
            rand = rngs[s].choice(pool, min(max(0, PER_CHAIN - len(force)), len(pool)), replace=False)
            idx = np.concatenate([force, rand]).astype(int)
            d = per[s]
            d["esm"].append(feats["esm"][idx]); d["mpnn"].append(feats["proteinmpnn"][idx])
            d["rand"].append(feats["rand_mpnn"][idx]); d["esm650"].append(feats["esm650"][idx])
            d["aa"].append(seq[idx]); d["grp"].append(np.full(len(idx), g))
            d["ss3"].append(np.array([SS3_IDX.get(x, 2) for x in npz["ss3"][idx]]))
            r = npz["rsa"][idx].astype(np.float64)
            d["rsa"].append(r)
            d["burial"].append(np.where(np.isfinite(r), (r < BURIAL_RSA).astype(int), -1))
            for k in ("binding_site", "active_site", "ptm_site"):
                d[k].append(t[k][idx].astype(int) if t is not None else np.full(len(idx), -1))
            d["bfactor"].append(bf[idx] if t is not None else np.full(len(idx), np.nan))
        g += 1
        if g % 500 == 0:
            print(f"  loaded {g:,} chains ({time.time() - t0:,.0f}s)", flush=True)
        if max_chains and g >= max_chains:
            break
    return {s: {k: np.concatenate(v) for k, v in per[s].items()} for s in SEEDS}, g


def arms(D, train):
    """The nine feature sets. Amino-acid means for the residual arms are fitted on `train` only."""
    oh = fu.onehot(D["aa"])
    er = fu.residualise_on_train(D["esm"], D["aa"], train)
    mr = fu.residualise_on_train(D["mpnn"], D["aa"], train)
    return {"esm": D["esm"], "mpnn": D["mpnn"], "aa": oh,
            "raw": np.hstack([D["esm"], D["mpnn"]]),
            "raw_aa": np.hstack([oh, D["esm"], D["mpnn"]]),
            "residual": np.hstack([er, mr]),
            "hybrid": np.hstack([oh, er, mr]),
            "esm_untrained_mpnn": np.hstack([D["esm"], D["rand"]]),
            "esm650": D["esm650"]}


# ==========================================================================================
# predictors
# ==========================================================================================

def _score(kind, y, pred):
    if kind == "reg":
        return r2_score(y, pred)
    if kind == "clf":
        return accuracy_score(y, pred)
    return f1_score(y, pred, average="macro")


def _cv_linear(kind, X, y, groups, w):
    """Penalty by 3-fold cross-validation over training chains (folds never split a chain)."""
    grid = RIDGE_ALPHAS if kind == "reg" else LOGIT_CS
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))

    def make(p):
        return Ridge(alpha=p) if kind == "reg" else LogisticRegression(C=p, max_iter=500)

    best, best_score = grid[0], -np.inf
    for p in grid:
        scores = []
        for tr, va in folds:
            if kind != "reg" and len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            m = make(p).fit(sc.transform(X[tr]), y[tr], **({} if w is None else {"sample_weight": w[tr]}))
            scores.append(_score(kind, y[va], m.predict(sc.transform(X[va]))))
        if scores and np.mean(scores) > best_score:
            best, best_score = p, float(np.mean(scores))
    return make(best), best


def fit_predict(pred, kind, Xtr, ytr, Xte, gtr, seed):
    w = compute_sample_weight("balanced", ytr) if kind == "site" else None
    kw = {} if w is None else {"sample_weight": w}
    if pred == "linear":
        model, chosen = _cv_linear(kind, Xtr, ytr, gtr, w)
        sc = StandardScaler().fit(Xtr)
        return model.fit(sc.transform(Xtr), ytr, **kw).predict(sc.transform(Xte)), chosen
    if pred == "xgb":
        Model = XGBRegressor if kind == "reg" else XGBClassifier
        m = Model(n_estimators=80, max_depth=4, tree_method="hist", n_jobs=THREADS, verbosity=0,
                  random_state=seed)
        return m.fit(Xtr, ytr, **kw).predict(Xte), None
    sc = StandardScaler().fit(Xtr)
    if kind == "reg":
        mu, sd = float(ytr.mean()), float(ytr.std() + 1e-9)
        m = MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=seed)
        m.fit(sc.transform(Xtr), (ytr - mu) / sd)
        return m.predict(sc.transform(Xte)) * sd + mu, None
    m = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=seed)
    return m.fit(sc.transform(Xtr), ytr, **kw).predict(sc.transform(Xte)), None


# ==========================================================================================
# run
# ==========================================================================================

def run_part(D, seed, size, out_csv):
    C = int(D["grp"].max()) + 1
    rng = np.random.default_rng(seed)                                  # the split, as in 05_probes.py
    test_chains = rng.choice(C, max(1, int(C * TEST_FRAC)), replace=False)
    train_pool = np.setdiff1d(np.arange(C), test_chains)
    sub = np.random.default_rng(seed + 1000)                          # the training-size subsample
    n_keep = max(3, int(round(len(train_pool) * size)))
    train_chains = sub.choice(train_pool, min(n_keep, len(train_pool)), replace=False)
    train, test = np.isin(D["grp"], train_chains), np.isin(D["grp"], test_chains)
    A = arms(D, train)
    rows = []
    for task, kind, metric in TASKS:
        y = D[task]
        lab = np.isfinite(y) if kind == "reg" else (y >= 0)
        tr, te = train & lab, test & lab
        base = {"seed": seed, "size": size, "task": task, "metric": metric,
                "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_train_chains": len(train_chains)}
        skip = None
        if kind == "site" and int(y[tr].sum()) < MIN_POSITIVES:
            skip = f"{int(y[tr].sum())} training positives"
        elif kind != "reg" and len(np.unique(y[tr])) < 2:
            skip = "one class in training"
        for pred in PREDICTORS:
            for arm in ARMS:
                row = dict(base, predictor=pred, arm=arm, score="", penalty="", skipped=skip or "")
                if skip is None:
                    X = A[arm]
                    p, chosen = fit_predict(pred, kind, X[tr], y[tr], X[te], D["grp"][tr], seed)
                    row["score"] = round(float(_score(kind, y[te], p)), 5)
                    row["penalty"] = "" if chosen is None else chosen
                rows.append(row)
        print(f"    seed {seed} size {size:.2f} {task:<13} done", flush=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _main_run() -> None:
    ap = argparse.ArgumentParser(description="Fit every fusion arm")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results/fusion")
    ap.add_argument("--max-chains", type=int, default=None, help="smoke tests only")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out_dir); parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args, extra={"layers": LAYERS, "seeds": SEEDS, "sizes": SIZES,
                                       "per_chain": PER_CHAIN, "test_frac": TEST_FRAC, "threads": THREADS})
    t0 = time.time()
    data, n_chains = load(Path(args.structures_dir), Path(args.results_root), args.max_chains)
    print(f"  {n_chains:,} chains; residues per refit: "
          + ", ".join(f"{len(data[s]['grp']):,}" for s in SEEDS) + f" ({time.time() - t0:,.0f}s)", flush=True)
    jobs = [(s, z) for s in SEEDS for z in SIZES if not (parts / f"seed{s}_size{z:.2f}.csv").exists()]
    print(f"  {len(jobs)} parts to fit, {args.workers} workers x {THREADS} threads", flush=True)
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.workers, backend="loky")(
        delayed(run_part)(data[s], s, z, parts / f"seed{s}_size{z:.2f}.csv") for s, z in jobs)
    rows = []
    for p in sorted(parts.glob("seed*_size*.csv")):
        rows += list(csv.DictReader(p.open()))
    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    qc.record_params(out, args, extra={"layers": LAYERS, "seeds": SEEDS, "sizes": SIZES, "chains": n_chains,
                                       "rows": len(rows), "minutes": round((time.time() - t0) / 60, 1)})
    print(f"-> {out / 'scores.csv'} ({len(rows):,} rows, {(time.time() - t0) / 60:,.0f} min)")


# ==========================================================================================
# summarise
# ==========================================================================================

def _main_summarise() -> None:
    ap = argparse.ArgumentParser(description="Paired gains, variance partitioning, decision rule")
    ap.add_argument("--out-dir", default="results/fusion")
    args = ap.parse_args()
    out = Path(args.out_dir)
    S = {}
    for r in csv.DictReader((out / "scores.csv").open()):
        if r["score"] == "":
            continue
        S.setdefault((r["predictor"], float(r["size"]), r["task"], r["arm"]), {})[int(r["seed"])] = float(r["score"])

    def vec(key):
        d = S.get(key, {})
        return np.array([d.get(s, np.nan) for s in SEEDS])

    gains = []
    for pred in PREDICTORS:
        for size in SIZES:
            for task, _, metric in TASKS:
                for arm in FUSED + ("esm650",):
                    a = vec((pred, size, task, arm))
                    if not np.isfinite(a).any():
                        continue
                    for ref in ("esm", "mpnn", "raw_aa"):
                        if ref == arm:
                            continue
                        g = fu.paired_gain(a, vec((pred, size, task, ref)))
                        gains.append({"predictor": pred, "size": size, "task": task, "metric": metric,
                                      "arm": arm, "over": ref, **{k: round(v, 5) if isinstance(v, float) else v
                                                                   for k, v in g.items()}})
    with (out / "gains.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gains[0])); w.writeheader(); w.writerows(gains)

    # decision rule: residual or hybrid beats raw-plus-one-hot if the paired interval is above zero on
    # at least three of the seven targets for the same predictor
    rule = {}
    for pred in PREDICTORS:
        for size in SIZES:
            for arm in ("residual", "hybrid"):
                above = [g["task"] for g in gains if g["predictor"] == pred and g["size"] == size
                         and g["arm"] == arm and g["over"] == "raw_aa" and g["n"] >= 2 and g["lo"] > 0]
                below = [g["task"] for g in gains if g["predictor"] == pred and g["size"] == size
                         and g["arm"] == arm and g["over"] == "raw_aa" and g["n"] >= 2 and g["hi"] < 0]
                rule[f"{pred}|{size:.2f}|{arm}"] = {"above_zero": above, "below_zero": below,
                                                    "better": len(above) >= 3}
    # variance partitioning of the linear scores, raw and residual fusion, full training size
    vp = {}
    for task, _, _ in TASKS:
        e, m = vec(("linear", 1.0, task, "esm")), vec(("linear", 1.0, task, "mpnn"))
        for arm in ("raw", "residual"):
            f = vec(("linear", 1.0, task, arm))
            if np.isfinite(f).all() and np.isfinite(e).all() and np.isfinite(m).all():
                parts_ = fu.variance_partition(float(e.mean()), float(m.mean()), float(f.mean()))
                vp[f"{task}|{arm}"] = {k: round(v, 4) for k, v in parts_.items()}
    means = {f"{p}|{z:.2f}|{t}|{a}": round(float(np.nanmean(vec((p, z, t, a)))), 4)
             for p in PREDICTORS for z in SIZES for t, _, _ in TASKS for a in ARMS
             if np.isfinite(vec((p, z, t, a))).any()}
    summary = {"decision_rule": rule, "variance_partition_linear_full": vp, "mean_scores": means,
               "seeds": list(SEEDS), "sizes": list(SIZES), "layers": LAYERS}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(rule, indent=1))
    print(f"-> {out / 'gains.csv'}, {out / 'summary.json'}")


_SUBCOMMANDS = {"run": _main_run, "summarise": _main_summarise}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _SUBCOMMANDS:
        print(__doc__)
        raise SystemExit(2)
    _SUBCOMMANDS[sys.argv.pop(1)]()


if __name__ == "__main__":
    main()
