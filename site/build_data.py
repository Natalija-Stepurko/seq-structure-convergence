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

props = [("ss3", "secondary structure"), ("burial", "buried or exposed"), ("rsa", "solvent accessibility")]


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


REV_MAIN = ("binding_site", "active_site", "ptm_site")
REV_LABEL = {"binding_site": "binding site (ligand contact)", "active_site": "active site", "ptm_site": "PTM site",
             "ss3": "secondary structure", "burial": "buried or exposed", "rsa": "solvent accessibility"}


def stitch_reverse(name):
    """grid-ci-reverse: ESM-2 fed into ProteinMPNN. Verdicts count the site properties."""
    cells = list(csv.DictReader(open(R / "stitch_grid_ci" / name / "cells.csv")))
    props = [k for k in REV_LABEL if f"{k}_stitched_mean" in cells[0]]
    blank_v = lambda: {"worse": 0, "no clear difference": 0, "better": 0}
    vs = {"donor": blank_v(), "blank": blank_v()}
    depth = {}
    for c in cells:
        for k in REV_MAIN:
            for cmp in vs:
                vs[cmp][c[f"{k}_vs_{cmp}_verdict"]] += 1
            depth.setdefault(int(c["inject_layer"]), blank_v())[c[f"{k}_vs_donor_verdict"]] += 1
    mean = lambda k: round(sum(float(c[k]) for c in cells) / len(cells), 3)
    per = {k: {cmp: blank_v() for cmp in ("donor", "blank")} for k in props}
    for c in cells:
        for k in props:
            for cmp in ("donor", "blank"):
                per[k][cmp][c[f"{k}_vs_{cmp}_verdict"]] += 1
    return {
        "per_prop": per,
        "props": [{"key": k, "label": REV_LABEL[k], "donor": mean(f"{k}_donor_mean"),
                   "trained": mean(f"{k}_stitched_mean"), "rand": mean(f"{k}_stitched_rand_mean"),
                   "blank": mean(f"{k}_blank_mean"), "native": mean(f"{k}_native_mean")} for k in props],
        "vs_donor": vs["donor"], "vs_blank": vs["blank"], "total": len(cells) * len(REV_MAIN),
        "cells": len(cells), "repeats": int(cells[0]["n_repeats"]),
        "by_depth": [{"layer": d, **v} for d, v in sorted(depth.items())],
        "r2_mean": round(sum(float(c["connector_r2_mean"]) for c in cells) / len(cells), 3)}


D["stitching"] = stitch_run("proteinmpnn_ridge")
D["stitching_reverse"] = stitch_reverse("esm_to_proteinmpnn")
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

# Every property by depth, both models, from the five-refit probes (boosted trees): the full set
# behind "Where it builds up". Residue targets carry their own metric; protein-level targets are
# macro-F1, with the best counting baseline (composition alone, or composition plus length and
# physicochemical scales; linear or boosted trees) for reference.
DEPTH_RES = [("ss3", "secondary structure", "ss3_xgb_acc", "accuracy"),
             ("burial", "burial", "burial_xgb_acc", "accuracy"),
             ("rsa", "solvent accessibility", "rsa_xgb_r2", "R²"),
             ("bfactor", "B-factor (flexibility)", "bfactor_xgb_r2", "R²"),
             ("binding_site", "binding site", "binding_site_xgb_f1", "F1"),
             ("active_site", "active site", "active_site_xgb_f1", "F1"),
             ("ptm_site", "PTM site", "ptm_site_xgb_f1", "F1")]
DEPTH_CHAIN = [("cath_class", "fold class (CATH)"), ("cath_arch", "fold architecture"),
               ("cath_topol", "fold topology"), ("enzyme", "is it an enzyme"),
               ("ec_class", "enzyme class (EC)"), ("protein_class", "protein class"),
               ("is_kinase", "kinase"), ("is_transport", "transporter"),
               ("is_dna_binding", "DNA-binding"), ("is_rna_binding", "RNA-binding"),
               ("is_ribosomal", "ribosomal"), ("is_structural", "structural protein"),
               ("is_immune", "immune protein"), ("is_membrane_protein", "membrane protein"),
               ("is_phospho", "phosphorylated"), ("is_glyco", "glycosylated"),
               ("localisation", "where in the cell"), ("kingdom", "kingdom of life")]
ESM_ORDER = ["emb"] + [f"L{i}" for i in range(1, 13)]
MPNN_ORDER = ["enc1", "enc2", "enc3"]


def depth_series(m, fn, key, metric, order):
    rows = {r["layer"]: r for r in csv.DictReader(open(R / "probes_ci" / m / fn))
            if r["metric"] == metric and (key is None or r["target"] == key)}
    return [{"layer": l, "v": round(float(rows[l]["mean"]), 4), "lo": round(float(rows[l]["lo"]), 4),
             "hi": round(float(rows[l]["hi"]), 4)} for l in order if l in rows]


baseline = {}
for src in ("probes_comp_rerun", "probes_physchem"):
    for r in csv.DictReader(open(R / src / "composition_metrics.csv")):
        for col in ("comp_lin_f1", "comp_xgb_f1"):
            baseline[r["target"]] = max(baseline.get(r["target"], 0.0), float(r[col]))
D["depth_probes"] = (
    [{"key": k, "label": lab, "level": "residue", "metric": unit,
      "esm": depth_series("esm", "metrics_ci.csv", None, met, ESM_ORDER),
      "mpnn": depth_series("proteinmpnn", "metrics_ci.csv", None, met, MPNN_ORDER), "baseline": None}
     for k, lab, met, unit in DEPTH_RES]
    + [{"key": k, "label": lab, "level": "protein", "metric": "F1",
        "esm": depth_series("esm", "chain_metrics_ci.csv", k, "xgb_f1", ESM_ORDER),
        "mpnn": depth_series("proteinmpnn", "chain_metrics_ci.csv", k, "xgb_f1", MPNN_ORDER),
        "baseline": round(baseline[k], 4) if k in baseline else None}
       for k, lab in DEPTH_CHAIN])
for d in D["depth_probes"]:
    assert len(d["esm"]) == 13 and len(d["mpnn"]) == 3, (d["key"], len(d["esm"]), len(d["mpnn"]))

# The sequence model's lookup layer against a bare label naming the amino acid.
aa = json.load(open(R / "aa_control/aa_control.json"))["esm_vs_aa"]
D["aa_lookup"] = {"esm_svcca": round(aa["svcca"][0], 3), "esm_knn": round(aa["mutual_knn"][0], 3)}

chains = list(csv.DictReader(open(R / "dataset/chains.csv")))
lad0 = next(csv.DictReader(open(R / "ladder_ci/ladder_ci.csv")))
refits = {int(r["n"]) for r in csv.DictReader(open(R / "probes_ci/esm/chain_metrics_ci.csv"))}
assert len(refits) == 1, f"probe refit counts differ across rows: {refits}"
D["dataset"] = {"chains": len(chains), "residues": sum(int(c["length"]) for c in chains),
                "subsamples": int(lad0["n_resamples"]), "budget": int(lad0["n_residues_per_subsample"]),
                "refits": refits.pop()}

# What the chain set covers (scripts/01_dataset.py diversity).
div = json.load(open(R / "dataset/diversity.json"))
assert div["chains"] == D["dataset"]["chains"] and div["residues"] == D["dataset"]["residues"], \
    "diversity.json describes a different chain set from chains.csv"
D["diversity"] = div

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

# The published embedding subsample (scripts/07_hf_dataset.py): its size, for the link on the page.
hf = list(csv.DictReader(open(R / "dataset/hf_chains.csv")))
D["dataset_hf"] = {"residues": sum(int(r["length"]) for r in hf), "chains": len(hf),
                   "proteins": sum(1 for l in open(R / "dataset/hf_protein_ids.txt") if l.strip())}

OUT.mkdir(exist_ok=True)
json.dump(D, open(OUT / "data.json", "w"), indent=1)
print("DATA keys:", ", ".join(D))
print("  depth CKA:", [d["v"] for d in D["depth_cka"]])
print(f"  stitching: {D['stitching']['worse']} worse / {D['stitching']['unclear']} unclear / "
      f"{D['stitching']['better']} better of {D['stitching']['total']}")
print("  probe targets:", len(D["probe_chain"]["esm"]))
print("  dataset:", D["dataset"]["chains"], "chains,", D["dataset"]["residues"], "residues")
