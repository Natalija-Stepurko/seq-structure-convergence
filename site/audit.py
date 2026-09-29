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

print("calibration against scrambled residues")
svc = {(r["pair"], r["k"]): r for r in csv.DictReader(open(R / "convergence_controls/svcca_controls.csv"))}
obs, null = float(svc[("esm35", "native")]["svcca"]), float(svc[("esm35", "native")]["svcca_perm"])
claim("SVCCA, main pair", obs)
claim("SVCCA, scrambled", null)
claim("SVCCA, share that is null", 100 * null / obs, "{:.0f}%")
claim("SVCCA, ESM-2 650M", float(svc[("esm650", "native")]["svcca"]))
sig = (R / "convergence/significance/summary.txt").read_text()
claim("CKA, main pair", float(re.search(r"peak CKA:\s+mean ([0-9.]+)", sig).group(1)))
claim("CKA, scrambled", float(re.search(r"permuted CKA:\s+mean ([0-9.]+)", sig).group(1)))

print("stitching grid")
grid = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(R / "stitch_grid/stitch_grid.csv"))]
props = ["ss3", "burial", "rsa"]
for cond in ("donor", "stitched_rand", "stitched"):
    for p in props:
        claim(f"{cond} / {p}, mean over {len(grid)} cells", sum(r[f"{p}_{cond}"] for r in grid) / len(grid))
worse = sum(1 for r in grid for p in props if r[f"{p}_stitched"] <= r[f"{p}_donor"])
claim("stitched no better than the structure model alone", worse, "{} of " + str(len(grid) * 3))

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

UNVERIFIED = ["mutual k-NN calibration 0.025 vs 0.002 on 15,148 residues: produced by a one-off "
              "script whose output was not kept; no file in results/ to check it against"]

banned = sorted({m.group(0).lower() for m in re.finditer(r"rather than|instead of|one-hot", text, re.I)})

print(f"\n{ok} verified, {len(missing)} missing, {len(UNVERIFIED)} without a tracked source")
for u in UNVERIFIED:
    print(f"  unverified: {u}")
if banned:
    print(f"  banned phrases on the page: {', '.join(banned)}")
if missing or banned:
    sys.exit(1)
