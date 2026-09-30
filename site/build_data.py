"""Assemble build/data.json, the object the page renders from, straight out of results/."""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
OUT = HERE / "build"
D = {}

lad = {}
for r in csv.DictReader(open(R / "ladder_ci/ladder_ci.csv")):
    d = lad.setdefault(r["pair"], {"raw": {}, "partial": {}, "kind": r["type"]})
    d[r["mode"]][r["metric"]] = {"v": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"])}
D["every_pair"] = lad

# Calibration: each measure against its scrambled-residue null, for the main pair, on the same 50
# whole-chain subsamples of 8,000 residues as every ladder row (ladder-ci --with-null).
cal = {(r["pair"], r["mode"], r["metric"]): r
       for r in csv.DictReader(open(R / "ladder_ci_null/ladder_ci.csv"))}
D["calibration"] = {
    k: {"obs": float(cal[("ESM-2 35M x ProteinMPNN", "raw", m)]["mean"]),
        "null": float(cal[("ESM-2 35M x ProteinMPNN", "raw", m)]["null_mean"]),
        "n": int(cal[("ESM-2 35M x ProteinMPNN", "raw", m)]["n_residues_per_subsample"]),
        "subsamples": int(cal[("ESM-2 35M x ProteinMPNN", "raw", m)]["n_resamples"]),
        "src": "results/ladder_ci_null"}
    for k, m in (("cka", "cka"), ("svcca", "svcca"), ("knn", "mutual_knn"))}

# SVCCA width series: a separate run (svcca-controls) on its own residue sample.
svc = {(r["pair"], r["k"]): r for r in csv.DictReader(open(R / "convergence_controls/svcca_controls.csv"))}
f = lambda pair, k, col: float(svc[(pair, k)][col])
# Gaps are computed before rounding: 0.4076 - 0.2912 = 0.1164, not 0.408 - 0.291 = 0.117.
D["width"] = [
    {"lab": "forced to 64 dimensions", "dim": 64,
     "obs": f("esm35", "64", "svcca"), "null": f("esm35", "64", "svcca_perm")},
    {"lab": "ESM-2 35M · 480 dimensions", "dim": 480,
     "obs": f("esm35", "native", "svcca"), "null": f("esm35", "native", "svcca_perm")},
    {"lab": "ESM-2 650M · 1,280 dimensions", "dim": 1280,
     "obs": f("esm650", "native", "svcca"), "null": f("esm650", "native", "svcca_perm")}]

z = np.load(R / "convergence/grids.npz", allow_pickle=True)
el = [str(x) for x in z["esm_labels"]]
for key, arr in [("depth_cka", "cka"), ("depth_svcca", "svcca"), ("depth_knn", "mutual_knn")]:
    D[key] = [{"layer": el[i], "v": round(float(z[arr][i, 2]), 4)} for i in range(len(el))]
D["layer_grids"] = {m: [[round(float(z[m][i, j]), 4) for j in range(z[m].shape[1])]
                        for i in range(z[m].shape[0])] for m in ("cka", "svcca", "mutual_knn")}
D["layer_labels"] = {"esm": el, "mpnn": [str(x) for x in z["st_labels"]]}

props = [("ss3", "local shape"), ("burial", "buried or exposed"), ("rsa", "solvent accessibility")]


def stitch_run(name):
    """One grid-ci run: condition means over cells, verdict counts overall and by entry depth."""
    cells = list(csv.DictReader(open(R / "stitch_grid_ci" / name / "cells.csv")))
    fl = lambda c, k: float(c[k])
    verdicts = {"worse": 0, "no clear difference": 0, "better": 0}
    depth = {}
    for c in cells:
        il = int(c["inject_layer"])
        for k, _ in props:
            v = c[f"{k}_verdict"]
            verdicts[v] += 1
            depth.setdefault(il, {"worse": 0, "no clear difference": 0, "better": 0})[v] += 1
    r2 = [fl(c, "connector_r2_mean") for c in cells]
    mean = lambda k: round(sum(fl(c, k) for c in cells) / len(cells), 3)
    return {
        "props": [{"key": k, "label": l, "donor": mean(f"{k}_donor_mean"),
                   "rand": mean(f"{k}_stitched_rand_mean"), "trained": mean(f"{k}_stitched_mean"),
                   "native": mean(f"{k}_native_mean")} for k, l in props],
        "worse": verdicts["worse"], "unclear": verdicts["no clear difference"],
        "better": verdicts["better"], "total": sum(verdicts.values()), "cells": len(cells),
        "repeats": int(cells[0]["n_repeats"]),
        "by_depth": [{"layer": il, **d} for il, d in sorted(depth.items())],
        "r2_mean": round(sum(r2) / len(r2), 3), "r2_min": round(min(r2), 3), "r2_max": round(max(r2), 3)}


D["stitching"] = stitch_run("proteinmpnn_ridge")
D["stitching_control"] = stitch_run("carp_ridge")
D["stitching_mlp"] = stitch_run("proteinmpnn_mlp")


def probes(m, fn, metric):
    b = {}
    for r in csv.DictReader(open(R / "probes_ci" / m / fn)):
        if r["metric"] != metric:
            continue
        t = r["target"]
        if t not in b or float(r["mean"]) > b[t]["v"]:
            b[t] = {"v": round(float(r["mean"]), 4), "sd": round(float(r["sd"]), 4), "layer": r["layer"]}
    return b


D["probe_chain"] = {m: probes(m, "chain_metrics_ci.csv", "xgb_f1") for m in ("esm", "proteinmpnn")}
D["probe_res"] = {m: {k: probes(m, "metrics_ci.csv", k) for k in ("xgb_acc", "xgb_f1", "xgb_r2")}
                  for m in ("esm", "proteinmpnn")}
D["physchem"] = {r["target"]: round(float(r["comp_xgb_f1"]), 4)
                 for r in csv.DictReader(open(R / "probes_physchem/composition_metrics.csv"))}

# The sequence model's lookup layer against a bare label naming the amino acid.
aa = json.load(open(R / "aa_control/aa_control.json"))["esm_vs_aa"]
D["aa_lookup"] = {"esm_svcca": round(aa["svcca"][0], 3), "esm_knn": round(aa["mutual_knn"][0], 3)}

chains = list(csv.DictReader(open(R / "dataset/chains.csv")))
lad0 = next(csv.DictReader(open(R / "ladder_ci/ladder_ci.csv")))
refits = {int(r["n"]) for r in csv.DictReader(open(R / "probes_ci/esm/chain_metrics_ci.csv"))}
assert len(refits) == 1, f"probe refit counts differ across rows: {refits}"
D["dataset"] = {"domains": len(chains), "residues": sum(int(c["length"]) for c in chains),
                "subsamples": int(lad0["n_resamples"]), "budget": int(lad0["n_residues_per_subsample"]),
                "refits": refits.pop()}

# Depth profile before and after the amino-acid subtraction, on the ladder's basis (depth-ci).
dep = {}
for r in csv.DictReader(open(R / "depth_ci/depth_ci.csv")):
    e = dep.setdefault(int(r["layer"]), {"label": r["label"]})
    if r["mode"] == "raw":
        e["raw"] = float(r["mean"])
    else:
        e.update(partial=float(r["mean"]), lo=float(r["lo"]), hi=float(r["hi"]),
                 degenerate=bool(int(r["degenerate"])))
D["depth_partial"] = [dep[k] for k in sorted(dep)]
D["depth_partial_basis"] = {"subsamples": int(r["n_resamples"]), "budget": int(r["n_residues_per_subsample"])}

OUT.mkdir(exist_ok=True)
json.dump(D, open(OUT / "data.json", "w"), indent=1)
print("DATA keys:", ", ".join(D))
print("  depth CKA:", [d["v"] for d in D["depth_cka"]])
print(f"  stitching: {D['stitching']['worse']} worse / {D['stitching']['unclear']} unclear / "
      f"{D['stitching']['better']} better of {D['stitching']['total']}")
print("  probe targets:", len(D["probe_chain"]["esm"]))
print("  dataset:", D["dataset"]["domains"], "domains,", D["dataset"]["residues"], "residues")
