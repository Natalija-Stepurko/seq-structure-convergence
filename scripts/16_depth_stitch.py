r"""
16_depth_stitch.py — per-property stitching at depth: are the two models' computations composable?

Stitching into a model's final head is single-task (ESM's lm_head only names residues). Stitching at
an INTERMEDIATE layer is per-property: inject the donor representation at ESM layer d, let ESM's
remaining frozen, trained blocks process it, and read out a property from the result.

    ProteinMPNN enc3 --W--> [ESM layer d] --> ESM blocks d+1..N (frozen) --> probe(property)

The question is not "is the property in the donor" (that is probing) but "does the receiver's
LEARNED computation still operate usefully on the donor's representation". Four conditions make
that identifiable:

    stitched-trained    donor -> W -> ESM's trained blocks   -> probe
    stitched-untrained  donor -> W -> ESM's untrained blocks -> probe   (does training matter?)
    donor-direct        donor ------------------------------ -> probe   (do the blocks add anything?)
    native              ESM's own layer-d state -> same blocks -> probe (ceiling)

Properties: secondary structure, burial, relative solvent accessibility, B-factor, binding site and
amino-acid identity. Active-site and PTM-site residues are deliberately EXCLUDED — at 0.09 % and
0.11 % of residues, uniform per-chain sampling yields too few positives to score, and force-including
them would give the four conditions different residue distributions.

No circularity: ESM's blocks were not trained on any of these properties, so no head supplies the
answer — unlike stitching into a sequence-recovery decoder (see S7). Amino-acid identity is the one
property ESM's blocks *were* trained to predict, but the donor cannot supply it: ProteinMPNN never
sees sequence.

Outputs (under --out-dir): depth_stitch.csv / summary.txt

Usage:
    uv run python scripts/16_depth_stitch.py --inject-layers 4 8 --out-dir /ssc/results/depth_stitch
"""

import argparse
import copy
import csv
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from xgboost import XGBClassifier, XGBRegressor

SS3_IDX = {"a": 0, "b": 1, "c": 2}
BURIAL_RSA = 0.25
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def _emb(p, layer=-1):
    return torch.load(p, weights_only=False)["layers"][layer].to(torch.float32)


def _run_blocks(model, x_btE, start):
    """Run ESM blocks [start:] on a (B,T,E) hidden state, returning the final (B,T,E) state.

    Mirrors ESM2.forward from the point after block `start`: blocks operate on (T,B,E) and the
    final layer norm is applied, so the output matches the stored final-layer representation.
    """
    with torch.no_grad():
        x = x_btE.transpose(0, 1)                       # (B,T,E) -> (T,B,E)
        for layer in model.layers[start:]:
            x, _ = layer(x, self_attn_padding_mask=None, need_head_weights=False)
        x = model.emb_layer_norm_after(x)
        return x.transpose(0, 1)                        # -> (B,T,E)


def _score(X, y, tr, te, kind):
    """Held-out score, plus the degenerate-baseline it must beat.

    For a rare binary label an unweighted classifier predicts the majority class everywhere and
    scores macro-F1 = F1(negative)/2 (~0.49 at 97.5 % negatives) identically in every condition,
    which looks like a result and is not one. We rebalance with scale_pos_weight and return the
    constant-prediction baseline alongside the score so degeneracy is visible rather than inferred.
    """
    if kind == "reg":
        m = XGBRegressor(n_estimators=80, max_depth=4, tree_method="hist",
                         n_jobs=2, verbosity=0).fit(X[tr], y[tr])
        return round(float(r2_score(y[te], m.predict(X[te]))), 4), 0.0, int(tr.sum())
    kw = {}
    cls, cnt = np.unique(y[tr], return_counts=True)
    if len(cls) == 2:
        kw["scale_pos_weight"] = float(cnt[0] / max(1, cnt[1]))
    m = XGBClassifier(n_estimators=80, max_depth=4, tree_method="hist",
                      n_jobs=2, verbosity=0, **kw).fit(X[tr], y[tr])
    p = m.predict(X[te])
    maj = np.full(te.sum(), cls[np.argmax(cnt)])
    base = float(f1_score(y[te], maj, average="macro"))
    return (round(float(f1_score(y[te], p, average="macro")), 4),
            round(base, 4), int(cnt.min()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-property stitching at depth")
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--esm-dir", default="/ssc/results/esm")
    ap.add_argument("--struct-dir", default="/ssc/results/proteinmpnn")
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--inject-layers", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--out-dir", default="/ssc/results/depth_stitch")
    ap.add_argument("--n-chains", type=int, default=250)
    ap.add_argument("--per-chain", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sdir = Path(a.structures_dir); prot = sdir / "proteins"; rest = sdir / "residue_targets"
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    E_dir, S_dir = Path(a.esm_dir), Path(a.struct_dir)
    use = [c for c in ids if (E_dir / f"{c}.pt").exists() and (S_dir / f"{c}.pt").exists()][: a.n_chains]

    import esm as esmlib
    model, alphabet = getattr(esmlib.pretrained, a.esm_model)(); model.eval()
    # untrained twin: same architecture, random weights
    rnd = esmlib.model.esm2.ESM2(num_layers=model.num_layers, embed_dim=model.args.embed_dim
                                 if hasattr(model, "args") else model.embed_dim,
                                 attention_heads=20, alphabet=alphabet)
    rnd.eval()
    n_layers = model.num_layers
    print(f"{len(use)} chains, ESM has {n_layers} blocks; injecting at {a.inject_layers}")

    rng = np.random.default_rng(a.seed)
    rows = []
    for d in a.inject_layers:
        # ---- fit the connector: ProteinMPNN enc3 -> ESM layer d, on training chains only ----
        n_fit = int(len(use) * 0.6)
        fit_c, ev_c = use[:n_fit], use[n_fit:]
        A = np.concatenate([_emb(S_dir / f"{c}.pt").numpy() for c in fit_c])
        B = np.concatenate([_emb(E_dir / f"{c}.pt", d).numpy() for c in fit_c])
        W = Ridge(alpha=a.alpha).fit(A, B)
        # Connector quality bounds the whole experiment: if the map cannot reach ESM's layer-d
        # distribution, the "stitched" input is off-distribution for trained blocks (which are tuned
        # to that distribution) but harmless to random blocks, which alone can invert the comparison.
        print(f"  inject@L{d}: connector R² (in-sample) = {W.score(A, B):.3f}")

        # ---- build the four conditions on evaluation chains ----
        feats = {k: [] for k in ["stitched", "stitched_rand", "donor", "native"]}
        labs = {k: [] for k in ["ss3", "burial", "binding", "rsa", "bfactor", "aa"]}
        for c in ev_c:
            npz = np.load(prot / f"{c}.npz", allow_pickle=True)
            L = int(npz["ss3"].shape[0])
            don = _emb(S_dir / f"{c}.pt")                       # ProteinMPNN enc3
            nat = _emb(E_dir / f"{c}.pt", d)                    # ESM's own layer d
            if don.shape[0] != L or nat.shape[0] != L:
                continue
            inj = torch.from_numpy(W.predict(don.numpy()).astype(np.float32))
            out_st = _run_blocks(model, inj[None], d)[0]
            out_rd = _run_blocks(rnd, inj[None], d)[0]
            out_nt = _run_blocks(model, nat[None], d)[0]
            idx = rng.choice(L, min(a.per_chain, L), replace=False)
            feats["stitched"].append(out_st[idx].numpy())
            feats["stitched_rand"].append(out_rd[idx].numpy())
            feats["native"].append(out_nt[idx].numpy())
            feats["donor"].append(don[idx].numpy())
            labs["ss3"].append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
            r = npz["rsa"][idx].astype(np.float64)
            labs["burial"].append(np.where(np.isfinite(r), (r < BURIAL_RSA).astype(int), -1))
            labs["rsa"].append(np.where(np.isfinite(r), r, np.nan))
            seq = np.array(list(str(npz["seq"])))[idx]
            labs["aa"].append(np.array([AA_IDX.get(x, -1) for x in seq]))
            rt = rest / f"{c}.npz"
            if rt.exists():
                t = np.load(rt)
                labs["binding"].append(t["binding_site"][idx].astype(int))
                bf = t["bfactor"].astype(np.float64)
                bf = (bf - np.nanmean(bf)) / (np.nanstd(bf) + 1e-6)
                labs["bfactor"].append(bf[idx])
            else:
                labs["binding"].append(np.full(len(idx), -1))
                labs["bfactor"].append(np.full(len(idx), np.nan))
        F = {k: np.concatenate(v) for k, v in feats.items()}
        Y = {k: np.concatenate(v) for k, v in labs.items()}
        n = len(Y["ss3"]); te = np.zeros(n, bool)
        te[rng.choice(n, n // 3, replace=False)] = True; tr = ~te
        print(f"  inject@L{d}: {n:,} eval residues")

        for prop, kind in [("ss3", "clf"), ("burial", "clf"), ("binding", "clf"),
                           ("aa", "clf"), ("rsa", "reg"), ("bfactor", "reg")]:
            m_ = np.isfinite(Y[prop]) if kind == "reg" else (Y[prop] >= 0)
            for cond in ["native", "stitched", "stitched_rand", "donor"]:
                sc, base, sup = _score(F[cond], Y[prop], tr & m_, te & m_, kind)
                rows.append({"inject_layer": d, "property": prop, "condition": cond,
                             "score": sc, "majority_baseline": base, "min_class_support": sup})
            got = {r["condition"]: r["score"] for r in rows if r["inject_layer"] == d
                   and r["property"] == prop}
            b = [r for r in rows if r["inject_layer"] == d and r["property"] == prop][0]
            flag = "  <-- AT MAJORITY BASELINE" if kind == "clf" and all(
                abs(v - b["majority_baseline"]) < 0.01 for v in got.values()) else ""
            print(f"    {prop:8s} native {got['native']:.3f} | stitched {got['stitched']:.3f} | "
                  f"stitched-untrained {got['stitched_rand']:.3f} | donor-direct {got['donor']:.3f}"
                  f"   [base {b['majority_baseline']:.3f}, min-class n={b['min_class_support']}]{flag}")

    o = Path(a.out_dir); o.mkdir(parents=True, exist_ok=True)
    with (o / "depth_stitch.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = ["Per-property stitching at depth (macro-F1). 'stitched' = ProteinMPNN mapped into ESM",
             "layer d then processed by ESM's remaining trained blocks; 'stitched_rand' uses the same",
             "architecture untrained; 'donor' probes ProteinMPNN directly; 'native' is ESM's own state.", ""]
    for r in rows:
        lines.append(f"  L{r['inject_layer']:<3d} {r['property']:9s} {r['condition']:16s} {r['score']:.3f}")
    (o / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {o}")


if __name__ == "__main__":
    main()
