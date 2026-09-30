"""Re-derive the page's quantitative claims from results/ and check each appears in the built page.

Three separate reporting bugs in this project had the same shape: a number that was correct in
isolation and wrong in the context it was printed in. This check reads the page as a reader sees
it (tags stripped) and exits non-zero if any claim is missing or a banned phrase appears, so a
build cannot pass on a stale or mistyped number.
"""
import csv
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
page = (HERE / "build/convergence.html").read_text()
body = page.split("</style>", 1)[1]
body = re.sub(r"<script\b.*?</script>", " ", body, flags=re.S)
text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body)))
ok, missing = 0, []


def claim(desc, value, fmt="{:.3f}"):
    global ok
    s = fmt.format(value)
    if s in text:
        ok += 1
        print(f"  ok    {desc:<58} {s}")
    else:
        missing.append(f"{desc} = {s}")
        print(f"  MISS  {desc:<58} {s}")


print("dataset")
chains = list(csv.DictReader(open(R / "dataset/chains.csv")))
claim("protein domains", len(chains), "{:,}")
claim("residues", sum(int(c["length"]) for c in chains), "{:,}")

print("agreement ladder (50 whole-chain subsamples of 8,000 residues)")
lad = {(r["pair"], r["mode"], r["metric"]): float(r["mean"])
       for r in csv.DictReader(open(R / "ladder_ci/ladder_ci.csv"))}
MAIN, CEIL = "ESM-2 35M x ProteinMPNN", "ESM-1v s1 x ESM-1v s2"
UNTRAINED = "untrained seq x trained str"
claim("main pair, CKA, raw", lad[(MAIN, "raw", "cka")])
claim("main pair, CKA, shared target removed", lad[(MAIN, "partial", "cka")])
claim("main pair, mutual k-NN, shared target removed", lad[(MAIN, "partial", "mutual_knn")])
claim("ceiling, CKA, raw", lad[(CEIL, "raw", "cka")])
claim("ceiling, mutual k-NN, shared target removed", lad[(CEIL, "partial", "mutual_knn")])
claim("untrained sequence model, CKA, raw", lad[(UNTRAINED, "raw", "cka")])
claim("untrained sequence model, CKA, shared target removed", lad[(UNTRAINED, "partial", "cka")])

print("calibration against scrambled residues (ladder rerun with nulls, main pair)")
cal = {(r["pair"], r["mode"], r["metric"]): r
       for r in csv.DictReader(open(R / "ladder_ci_null/ladder_ci.csv"))}
for m, name in (("cka", "CKA"), ("svcca", "SVCCA"), ("mutual_knn", "mutual k-NN")):
    r = cal[(MAIN, "raw", m)]
    claim(f"{name}, main pair", float(r["mean"]))
    claim(f"{name}, scrambled", float(r["null_mean"]))
    claim(f"{name}, share that is null", 100 * float(r["null_mean"]) / float(r["mean"]), "{:.0f}%")
svc = {(r["pair"], r["k"]): r for r in csv.DictReader(open(R / "convergence_controls/svcca_controls.csv"))}
claim("SVCCA width run, ESM-2 35M", float(svc[("esm35", "native")]["svcca"]))
claim("SVCCA width run, ESM-2 650M", float(svc[("esm650", "native")]["svcca"]))

print("stitching, 5 repeats per cell")
def stitch(name):
    return list(csv.DictReader(open(R / "stitch_grid_ci" / name / "cells.csv")))
props = ["ss3", "burial", "rsa"]
main_cells = stitch("proteinmpnn_ridge")
for cond in ("donor", "stitched_rand", "stitched"):
    for p in props:
        claim(f"{cond} / {p}, mean over {len(main_cells)} cells",
              sum(float(c[f"{p}_{cond}_mean"]) for c in main_cells) / len(main_cells))
claim("rsa, sequence model alone", sum(float(c["rsa_native_mean"]) for c in main_cells) / len(main_cells))
count = lambda cells, v: sum(1 for c in cells for p in props if c[f"{p}_verdict"] == v)
claim("clearly worse", count(main_cells, "worse"), "{} of " + str(3 * len(main_cells)))
claim("no clear difference", count(main_cells, "no clear difference"), "in {} there is no clear difference")
claim("clearly better", count(main_cells, "better"), "in {} it is better")
r2 = [float(c["connector_r2_mean"]) for c in main_cells]
claim("linear connector R2, mean", sum(r2) / len(r2))
mlp = stitch("proteinmpnn_mlp")
claim("MLP connector R2, mean", sum(float(c["connector_r2_mean"]) for c in mlp) / len(mlp))
claim("MLP, clearly worse", count(mlp, "worse"), "{} of " + str(3 * len(mlp)))
ctl = stitch("carp_ridge")
claim("CARP control connector R2, mean", sum(float(c["connector_r2_mean"]) for c in ctl) / len(ctl))
claim("CARP control, no clear loss", count(ctl, "no clear difference"), "{} of " + str(3 * len(ctl)))

print("probes: best layer per property, 5 refits")
def best(m):
    b = {}
    for r in csv.DictReader(open(R / "probes_ci" / m / "chain_metrics_ci.csv")):
        if r["metric"] == "xgb_f1" and (r["target"] not in b or float(r["mean"]) > b[r["target"]]):
            b[r["target"]] = float(r["mean"])
    return b
e, m = best("esm"), best("proteinmpnn")
for t in ("cath_topol", "ec_class", "kingdom"):
    claim(f"ESM-2 {t}", e[t])
    claim(f"ProteinMPNN {t}", m[t])

def best_res(model, metric):
    v = {}
    for r in csv.DictReader(open(R / "probes_ci" / model / "metrics_ci.csv")):
        if r["metric"] == metric:
            v[r["layer"]] = float(r["mean"])
    return max(v.values())
for prop, metric in (("rsa", "rsa_xgb_r2"), ("burial", "burial_xgb_acc"), ("ss3", "ss3_xgb_acc")):
    claim(f"ProteinMPNN {prop}", best_res("proteinmpnn", metric))
    claim(f"ESM-2 {prop}", best_res("esm", metric))

UNVERIFIED = []

banned = sorted({m.group(0).lower() for m in re.finditer(r"rather than|instead of|one-hot", text, re.I)})

print(f"\n{ok} verified, {len(missing)} missing, {len(UNVERIFIED)} without a tracked source")
for u in UNVERIFIED:
    print(f"  unverified: {u}")
if banned:
    print(f"  banned phrases on the page: {', '.join(banned)}")
if missing or banned:
    sys.exit(1)
