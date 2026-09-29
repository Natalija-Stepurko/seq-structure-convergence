"""Assemble build/data.json, the object the page renders from, straight out of results/."""
import csv
import json
import re
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

# SVCCA controls: the main pair's score against its scrambled-residue null, and the width series.
svc = {(r["pair"], r["k"]): r for r in csv.DictReader(open(R / "convergence_controls/svcca_controls.csv"))}
f = lambda pair, k, col: float(svc[(pair, k)][col])

# CKA against its permuted null, from the significance run (25 residue resamples of 15,000).
sig = (R / "convergence/significance/summary.txt").read_text()
cka_obs = float(re.search(r"peak CKA:\s+mean ([0-9.]+)", sig).group(1))
cka_null = float(re.search(r"permuted CKA:\s+mean ([0-9.]+)", sig).group(1))

# The calibration and the ladder are measured at different residue budgets: the scrambled test at
# ~15,000 residues, the ladder at 8,000 drawn whole-chain. Both are valid; the captions say which,
# because these measures are sample-size sensitive.
D["calibration"] = {
    "cka":   {"obs": cka_obs, "null": cka_null, "n": 15000,
              "src": "results/convergence/significance, 25 residue resamples"},
    "svcca": {"obs": round(f("esm35", "native", "svcca"), 3), "null": round(f("esm35", "native", "svcca_perm"), 3),
              "n": 15000, "src": "results/convergence_controls"},
    # No tracked source: produced by a one-off script (20 scrambles of 15,148 residues) whose
    # output was not kept. site/audit.py reports these two values as unverified.
    "knn":   {"obs": 0.025, "null": 0.002, "n": 15148, "src": "scrambled-residue run, 20 scrambles"}}
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

g = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(R / "stitch_grid/stitch_grid.csv"))]
props = [("ss3", "local shape"), ("burial", "buried or exposed"), ("rsa", "solvent accessibility")]
D["stitching"] = {
    "props": [{"key": k, "label": l,
               "donor": round(sum(r[f"{k}_donor"] for r in g) / len(g), 3),
               "rand": round(sum(r[f"{k}_stitched_rand"] for r in g) / len(g), 3),
               "trained": round(sum(r[f"{k}_stitched"] for r in g) / len(g), 3),
               "native": round(sum(r[f"{k}_native"] for r in g) / len(g), 3)} for k, l in props],
    "worse": sum(1 for r in g for k, _ in props if r[f"{k}_stitched"] <= r[f"{k}_donor"]),
    "total": len(g) * 3, "cells": len(g)}


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

OUT.mkdir(exist_ok=True)
json.dump(D, open(OUT / "data.json", "w"), indent=1)
print("DATA keys:", ", ".join(D))
print("  depth CKA:", [d["v"] for d in D["depth_cka"]])
print(f"  stitching: {D['stitching']['worse']} of {D['stitching']['total']} worse than donor")
print("  probe targets:", len(D["probe_chain"]["esm"]))
print("  dataset:", D["dataset"]["domains"], "domains,", D["dataset"]["residues"], "residues")
