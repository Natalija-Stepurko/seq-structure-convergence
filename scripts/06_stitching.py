r"""
06_stitching.py -- Functional tests: is one model's representation usable by the other's machinery?

Subcommands (each keeps the exact flags it had as a standalone script):

    predictivity     linear predictivity for one pair, both directions
    stitch           stitching through each model's own frozen head
    matrix           linear predictivity across every pair
    depth            per-property stitching at an intermediate layer
    grid             stitching across donor layer x injection depth, with connector quality

The subcommand token is removed from argv before the original parser runs, so every command
line that worked before still works, with the subcommand inserted after the script name:

    uv run python scripts/06_stitching.py predictivity --help

Provenance: every subcommand writes params.json beside its outputs (see ssc.metrics.record_params).

Merged from:
    predictivity     was 12_linear_stitch.py
    stitch           was 13_model_stitching.py
    matrix           was 15_predictivity_matrix.py
    depth            was 16_depth_stitch.py
"""

import argparse
import csv
import json
import statistics
import sys
import warnings
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssc import metrics as qc
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import torch.nn.functional as F
import argparse, csv, json, warnings
import numpy as np, torch
from sklearn.metrics import r2_score
import copy
from sklearn.metrics import accuracy_score, f1_score, r2_score
from xgboost import XGBClassifier, XGBRegressor





# ======================================================================================
# predictivity  --  from 12_linear_stitch.py
# ======================================================================================


warnings.filterwarnings("ignore")

SS3_IDX = {"a": 0, "b": 1, "c": 2}

def _collect(ids, dirs, prot, max_res, per_chain, seed):
    rng = np.random.default_rng(seed)
    emb = {k: [] for k in dirs}
    ss3 = []
    total = 0
    for cid in ids:
        if not all((Path(d) / f"{cid}.pt").exists() for d in dirs.values()):
            continue
        npz = np.load(prot / f"{cid}.npz", allow_pickle=True)
        L = int(npz["ss3"].shape[0])
        E = {k: torch.load(Path(d) / f"{cid}.pt", weights_only=False)["layers"]
             .to(torch.float32).numpy() for k, d in dirs.items()}
        if any(v.shape[1] != L for v in E.values()):
            continue
        idx = rng.choice(L, min(per_chain, L), replace=False)
        for k in dirs:
            emb[k].append(E[k][:, idx, :])
        ss3.append(np.array([SS3_IDX.get(s, 2) for s in npz["ss3"][idx]]))
        total += len(idx)
        if total >= max_res:
            break
    return ({k: np.concatenate(v, axis=1) for k, v in emb.items()},
            np.concatenate(ss3))

def _best_layers(A, B):
    """Pick the layer pair maximising linear cross-predictability (cheap proxy: last layers)."""
    return A.shape[0] - 1, B.shape[0] - 1

def _main_predictivity() -> None:
    ap = argparse.ArgumentParser(description="Linear cross-prediction / stitching-lite")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--pairs", nargs="+", required=True, help="name=dir ...")
    ap.add_argument("--out-dir", default="results/stitch")
    ap.add_argument("--max-residues", type=int, default=20000)
    ap.add_argument("--per-chain", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dirs = dict(p.split("=", 1) for p in args.pairs)
    sdir = Path(args.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    emb, ss3 = _collect(ids, dirs, sdir / "proteins", args.max_residues, args.per_chain, args.seed)
    n = next(iter(emb.values())).shape[1]
    print(f"{n:,} aligned residues across {len(dirs)} models")

    rng = np.random.default_rng(args.seed)
    te = np.zeros(n, bool); te[rng.choice(n, int(n * args.test_frac), replace=False)] = True
    tr = ~te

    rows = []
    for a, b in combinations(dirs, 2):
        ia, ib = _best_layers(emb[a], emb[b])
        A, B = emb[a][ia], emb[b][ib]
        As = StandardScaler().fit(A[tr]); Bs = StandardScaler().fit(B[tr])
        At, Bt = As.transform(A), Bs.transform(B)

        for src, dst, sa, da in [(At, Bt, a, b), (Bt, At, b, a)]:
            m = Ridge(alpha=args.alpha).fit(src[tr], dst[tr])
            pred = m.predict(src[te])
            r2 = r2_score(dst[te], pred, multioutput="variance_weighted")
            # permutation control: shuffle the residue correspondence
            perm = rng.permutation(tr.sum())
            mp = Ridge(alpha=args.alpha).fit(src[tr], dst[tr][perm])
            r2p = r2_score(dst[te], mp.predict(src[te]), multioutput="variance_weighted")

            # functional transfer: probe trained on real dst, applied to mapped src
            clf = XGBClassifier(n_estimators=80, max_depth=4, tree_method="hist",
                                n_jobs=4, verbosity=0).fit(dst[tr], ss3[tr])
            acc_real = accuracy_score(ss3[te], clf.predict(dst[te]))
            acc_map = accuracy_score(ss3[te], clf.predict(pred))
            rows.append({"source": sa, "target": da,
                         "transfer_r2": round(float(r2), 4),
                         "transfer_r2_perm": round(float(r2p), 4),
                         "probe_acc_real": round(float(acc_real), 4),
                         "probe_acc_mapped": round(float(acc_map), 4),
                         "retained": round(float(acc_map / acc_real), 4)})
            print(f"  {sa:>10s} -> {da:<10s} R²={r2:.3f} (perm {r2p:.3f})  "
                  f"probe {acc_real:.3f} -> {acc_map:.3f} ({acc_map/acc_real:.0%} retained)")


    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
    with (out / "stitch.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = ["Linear cross-prediction (stitching-lite): how much of the target representation is",
             "linearly recoverable from the source, and does a probe survive the mapping?", ""]
    for r in rows:
        lines.append(f"  {r['source']:>10s} -> {r['target']:<10s} "
                     f"transfer R²={r['transfer_r2']:.3f} (perm {r['transfer_r2_perm']:.3f})  "
                     f"probe {r['probe_acc_real']:.3f} -> {r['probe_acc_mapped']:.3f} "
                     f"({r['retained']:.0%} retained)")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {out}")

# ======================================================================================
# stitch  --  from 13_model_stitching.py
# ======================================================================================


warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "proteinmpnn"))

HIDDEN, N_ENC = 128, 3

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"          # ProteinMPNN's 21-letter alphabet

def _load(pt):
    return torch.load(pt, weights_only=False)["layers"].to(torch.float32)

def _mpnn_model(weights):
    from protein_mpnn_utils import ProteinMPNN
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    m = ProteinMPNN(num_letters=21, node_features=HIDDEN, edge_features=HIDDEN, hidden_dim=HIDDEN,
                    num_encoder_layers=N_ENC, num_decoder_layers=3,
                    k_neighbors=ck["num_edges"], augment_eps=0.0)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m

def _mpnn_encode(model, coords_bb):
    """Run ProteinMPNN's encoder; return its node state, its UPDATED edge state and the graph."""
    from protein_mpnn_utils import gather_nodes

    L = coords_bb.shape[0]
    ca = coords_bb[:, 1, :]; ca_ok = np.isfinite(ca).all(axis=1)
    f = coords_bb.copy()
    for k in range(4):
        miss = ~np.isfinite(f[:, k, :]).all(axis=1)
        f[miss, k, :] = np.where(ca_ok[miss, None], ca[miss], 0.0)
    X = torch.from_numpy(np.nan_to_num(f, nan=0.0).astype(np.float32))[None]
    mask = torch.from_numpy(ca_ok.astype(np.float32))[None]
    with torch.no_grad():
        E, E_idx = model.features(X, mask, torch.arange(L)[None], torch.ones(1, L))
        h_V = torch.zeros((1, L, E.shape[-1])); h_E = model.W_e(E)
        ma = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        ma = mask.unsqueeze(-1) * ma
        for layer in model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, ma)   # NB: the encoder updates h_E too
    return h_V, h_E, E_idx, mask, ca_ok

def _mpnn_decode(model, h_V, h_E, E_idx, mask, seq_idx, teacher_forcing=False):
    """ProteinMPNN's frozen decoder on a (possibly substituted) node state h_V.

    Faithful to ProteinMPNN.forward from the point after the encoder: the encoder-updated edge
    state h_E, the graph E_idx, the random decoding order and the causal masks are untouched;
    only h_V is substituted, so the comparison isolates the node representation.

    teacher_forcing=False (default) runs a SINGLE-SHOT decode: every position is treated as
    undecoded, so no neighbouring residue's true identity reaches the decoder and the prediction
    depends only on (h_V, h_E). This is required here — under teacher forcing the true sequence of
    already-decoded neighbours alone predicts much of the sequence, and a degenerate injected h_V
    scores *higher* than the real encoder state by pushing the decoder onto that leak.
    """
    from protein_mpnn_utils import cat_neighbors_nodes

    L = h_V.shape[1]
    S = torch.from_numpy(seq_idx)[None]
    chain_M = torch.ones(1, L) * mask
    g = torch.Generator().manual_seed(0)
    randn = torch.randn(1, L, generator=g)
    with torch.no_grad():
        h_S = model.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        mask_1D = mask.view([1, L, 1, 1])
        if teacher_forcing:
            decoding_order = torch.argsort((chain_M + 0.0001) * torch.abs(randn))
            ms = E_idx.shape[1]
            pmr = F.one_hot(decoding_order, num_classes=ms).float()
            omb = torch.einsum('ij, biq, bjp->bqp',
                               (1 - torch.triu(torch.ones(ms, ms))), pmr, pmr)
            mask_attend = torch.gather(omb, 2, E_idx).unsqueeze(-1)
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)
        else:                       # single-shot: nothing is "already decoded"
            mask_bw = torch.zeros_like(mask_1D)
            mask_fw = mask_1D
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in model.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
        logits = model.W_out(h_V)
    return logits[0]

def _main_stitch() -> None:
    ap = argparse.ArgumentParser(description="Model stitching, both directions")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--esm-dir", required=True)
    ap.add_argument("--struct-dir", required=True)
    ap.add_argument("--rand-esm-dir", default=None)
    ap.add_argument("--rand-struct-dir", default=None)
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--mpnn-weights", default="/scratch/.torch-hub/proteinmpnn/v_48_020.pt")
    ap.add_argument("--out-dir", default="results/stitching")
    ap.add_argument("--n-chains", type=int, default=400)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-seq-to-struct", action="store_true",
                    help="Skip direction 1 (into ProteinMPNN's decoder). That decoder consumes "
                         "128-d ProteinMPNN node states, so the direction is only defined when the "
                         "structure donor IS ProteinMPNN; use this for any other donor (e.g. "
                         "ESM-IF1, 512-d). Direction 2 is donor-agnostic.")
    args = ap.parse_args()

    sdir = Path(args.structures_dir); prot = sdir / "proteins"
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    dirs = {"esm": Path(args.esm_dir), "str": Path(args.struct_dir)}
    if args.rand_esm_dir: dirs["rand_esm"] = Path(args.rand_esm_dir)
    if args.rand_struct_dir: dirs["rand_str"] = Path(args.rand_struct_dir)
    use = [c for c in ids if all((d / f"{c}.pt").exists() for d in dirs.values())][: args.n_chains]
    rng = np.random.default_rng(args.seed)
    te_ids = set(rng.choice(len(use), int(len(use) * args.test_frac), replace=False).tolist())
    tr_c = [c for i, c in enumerate(use) if i not in te_ids]
    te_c = [c for i, c in enumerate(use) if i in te_ids]
    print(f"{len(use)} chains ({len(tr_c)} fit / {len(te_c)} eval)")

    # ---- gather residue-aligned last-layer embeddings for fitting the connectors ----
    def stack(cids, key):
        return np.concatenate([_load(dirs[key] / f"{c}.pt")[-1].numpy() for c in cids], 0)
    E_tr, S_tr = stack(tr_c, "esm"), stack(tr_c, "str")
    maps = {}
    maps[("esm", "str")] = Ridge(alpha=args.alpha).fit(E_tr, S_tr)
    maps[("str", "esm")] = Ridge(alpha=args.alpha).fit(S_tr, E_tr)
    for rk, real, tgt in [("rand_esm", "esm", "str"), ("rand_str", "str", "esm")]:
        if rk in dirs:
            maps[(rk, tgt)] = Ridge(alpha=args.alpha).fit(stack(tr_c, rk), stack(tr_c, tgt))
    print("connectors fitted")

    import esm as esmlib
    esm_model, alphabet = getattr(esmlib.pretrained, args.esm_model)()
    esm_model.eval(); bc = alphabet.get_batch_converter()
    mpnn = _mpnn_model(Path(args.mpnn_weights))

    rows = []

    # ================= direction 1: seq -> MPNN's own decoder (sequence recovery) =========
    def mpnn_recovery(source):
        """source: 'native' | 'esm' | 'rand_esm' — what supplies the decoder's node state."""
        cor = tot = 0
        for c in te_c:
            npz = np.load(prot / f"{c}.npz", allow_pickle=True)
            seq = str(npz["seq"]); cb = npz["coords_bb"]
            sidx = np.array([ALPHABET.index(a) if a in ALPHABET else 20 for a in seq],
                            dtype=np.int64)
            h_V_nat, h_E, E_idx, mask, ok = _mpnn_encode(mpnn, cb)
            if source == "native":
                h_V = h_V_nat
            else:
                src = _load(dirs[source] / f"{c}.pt")[-1].numpy()
                h_V = torch.from_numpy(
                    maps[(source, "str")].predict(src).astype(np.float32))[None]
            logits = _mpnn_decode(mpnn, h_V, h_E, E_idx, mask, sidx)
            pred = logits.argmax(-1).numpy()
            sel = ok & (sidx < 20)
            cor += int((pred[sel] == sidx[sel]).sum()); tot += int(sel.sum())
        return cor / max(1, tot)

    for src, lab in [("native", "MPNN's own encoder (ceiling)"),
                     ("esm", "ESM stitched into MPNN decoder"),
                     ("rand_esm", "untrained ESM stitched in (floor)")]:
        if args.skip_seq_to_struct:
            break
        if src not in ("native",) and src not in dirs:
            continue
        acc = mpnn_recovery(src)
        rows.append({"direction": "seq→struct (MPNN decoder, sequence recovery)",
                     "source": lab, "accuracy": round(acc, 4)})
        print(f"  [MPNN decoder] {lab:38s} sequence recovery {acc:.3f}")

    # ================= direction 2: struct -> ESM's own lm_head (residue readout) =========
    def esm_readout(source):
        cor = tot = 0
        for c in te_c:
            seq = str(np.load(prot / f"{c}.npz", allow_pickle=True)["seq"])[:1022]
            tgt = torch.tensor([alphabet.get_idx(a) for a in seq])
            if source == "native":
                x = _load(dirs["esm"] / f"{c}.pt")[-1][: len(seq)]
            else:
                src = _load(dirs[source] / f"{c}.pt")[-1].numpy()[: len(seq)]
                x = torch.from_numpy(maps[(source, "esm")].predict(src).astype(np.float32))
            with torch.no_grad():
                logits = esm_model.lm_head(x[None])[0]
            pred = logits.argmax(-1)
            cor += int((pred == tgt).sum()); tot += len(seq)
        return cor / max(1, tot)

    for src, lab in [("native", "ESM's own final layer (ceiling)"),
                     ("str", "ProteinMPNN stitched into ESM lm_head"),
                     ("rand_str", "untrained MPNN stitched in (floor)")]:
        if src not in ("native",) and src not in dirs:
            continue
        acc = esm_readout(src)
        rows.append({"direction": "struct→seq (ESM lm_head, residue readout)",
                     "source": lab, "accuracy": round(acc, 4)})
        print(f"  [ESM lm_head]  {lab:38s} residue readout   {acc:.3f}")


    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
    with (out / "stitch_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = ["Model stitching: each direction feeds one model's representation through a fitted",
             "linear connector into the OTHER model's own frozen head, evaluated on that model's",
             f"own task. {len(tr_c)} chains fit the connector, {len(te_c)} held-out chains evaluated.", ""]
    for r in rows:
        lines.append(f"  {r['direction']}\n      {r['source']:42s} {r['accuracy']:.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {out}")

# ======================================================================================
# matrix  --  from 15_predictivity_matrix.py
# ======================================================================================


warnings.filterwarnings("ignore")

PAIRS = [
    ("esm1v_s1","esm1v_s2","same arch, different seed"),
    ("carp","esm","within-sequence"),
    ("esmif1","proteinmpnn","within-structure"),
    ("esm650","proteinmpnn","cross-modality"),
    ("esm650","esmif1","cross-modality"),
    ("esm","proteinmpnn","cross-modality"),
    ("carp","proteinmpnn","cross-modality"),
    ("carp","esmif1","cross-modality"),
    ("rand_esm","rand_mpnn","untrained"),
    ("esm","rand_mpnn","one side untrained"),
    ("rand_esm","proteinmpnn","one side untrained"),
]

def load_last(p): return torch.load(p, weights_only=False)["layers"][-1].to(torch.float32).numpy()

def _main_matrix():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--results-root", default="/ssc/results")
    ap.add_argument("--out-dir", default="/ssc/results/predictivity_matrix")
    ap.add_argument("--n-chains", type=int, default=500)
    ap.add_argument("--per-chain", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root = Path(a.results_root); sdir = Path(a.structures_dir)
    ids = [json.loads(l)["id"] for l in (sdir/"index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    rows = []
    for A, B, kind in PAIRS:
        da, db = root/A, root/B
        if not (da.exists() and db.exists()): print(f"  [skip {A}x{B}]"); continue
        rng = np.random.default_rng(a.seed)
        XA, XB = [], []
        for c in ids[:a.n_chains*3]:
            fa, fb = da/f"{c}.pt", db/f"{c}.pt"
            if not (fa.exists() and fb.exists()): continue
            ea, eb = load_last(fa), load_last(fb)
            if ea.shape[0] != eb.shape[0]: continue
            i = rng.choice(ea.shape[0], min(a.per_chain, ea.shape[0]), replace=False)
            XA.append(ea[i]); XB.append(eb[i])
            if sum(x.shape[0] for x in XA) >= a.n_chains*a.per_chain: break
        XA, XB = np.concatenate(XA), np.concatenate(XB)
        n = len(XA); te = np.zeros(n, bool); te[rng.choice(n, n//4, replace=False)] = True; tr = ~te
        out = {"pair": f"{A} x {B}", "type": kind, "n_res": n}
        for src, dst, tag in [(XA, XB, "fwd"), (XB, XA, "rev")]:
            m = Ridge(alpha=a.alpha).fit(src[tr], dst[tr])
            out[f"r2_{tag}"] = round(float(r2_score(dst[te], m.predict(src[te]),
                                                    multioutput="variance_weighted")), 4)
            pm = rng.permutation(tr.sum())
            mp = Ridge(alpha=a.alpha).fit(src[tr], dst[tr][pm])
            out[f"r2_{tag}_perm"] = round(float(r2_score(dst[te], mp.predict(src[te]),
                                                         multioutput="variance_weighted")), 4)
        rows.append(out)
        print(f"  {out['pair']:26s} {kind:26s} fwd {out['r2_fwd']:+.3f} (perm {out['r2_fwd_perm']:+.3f})"
              f"  rev {out['r2_rev']:+.3f} (perm {out['r2_rev_perm']:+.3f})  n={n}")
    o = Path(a.out_dir); o.mkdir(parents=True, exist_ok=True)
    qc.record_params(o, a)   # provenance: exactly what produced these outputs
    with (o/"predictivity_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {o}")

# ======================================================================================
# depth  --  from 16_depth_stitch.py
# ======================================================================================


warnings.filterwarnings("ignore")

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
    constant-prediction baseline alongside the score so degeneracy is visible.
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

def _main_depth() -> None:
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
    qc.record_params(o, a)   # provenance: exactly what produced these outputs
    with (o / "depth_stitch.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = ["Per-property stitching at depth (macro-F1). 'stitched' = ProteinMPNN mapped into ESM",
             "layer d then processed by ESM's remaining trained blocks; 'stitched_rand' uses the same",
             "architecture untrained; 'donor' probes ProteinMPNN directly; 'native' is ESM's own state.", ""]
    for r in rows:
        lines.append(f"  L{r['inject_layer']:<3d} {r['property']:9s} {r['condition']:16s} {r['score']:.3f}")
    (o / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {o}")


# ==========================================================================================
# dispatch
# ==========================================================================================

# ==========================================================================================
# grid  --  stitching across donor layer x injection depth
# ==========================================================================================

def _fit_connector(A, B, kind, alpha, seed):
    """The map from donor states to receiver states: ridge, or a small MLP to test whether a
    stronger map changes the answer."""
    if kind == "ridge":
        return Ridge(alpha=alpha).fit(A, B)
    if kind == "mlp":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        net = make_pipeline(StandardScaler(),
                            MLPRegressor(hidden_layer_sizes=(512,), early_stopping=True,
                                         max_iter=200, random_state=seed))
        return TransformedTargetRegressor(regressor=net, transformer=StandardScaler()).fit(A, B)
    raise ValueError(f"unknown connector {kind!r}")


def _grid_cells(model, rnd, S_dir, E_dir, prot, fit_c, ev_c, donor_layers, inject_layers,
                rng_for, per_chain, properties, connector="ridge", alpha=100.0, seed=0,
                on_row=None, label="donor"):
    """Score every (donor layer, injection layer) cell under the four stitching conditions.

    `rng_for(dl, il)` supplies the generator used for the residue draws and the probe split in
    that cell. `grid` passes one shared generator (the original sequential behaviour); `grid-ci`
    passes a generator seeded per cell, so a resumed run draws exactly what an uninterrupted
    one would.
    """
    rows = []
    for dl in donor_layers:
        A_fit = np.concatenate([_emb(S_dir / f"{c}.pt", dl).numpy() for c in fit_c])
        for il in inject_layers:
            rng = rng_for(dl, il)
            B_fit = np.concatenate([_emb(E_dir / f"{c}.pt", il).numpy() for c in fit_c])
            W = _fit_connector(A_fit, B_fit, connector, alpha, seed)

            feats = {"stitched": [], "native": [], "stitched_rand": [], "donor": []}
            labs = {k: [] for k in ("ss3", "burial", "rsa")}
            held_true, held_pred = [], []
            for c in ev_c:
                npz = np.load(prot / f"{c}.npz", allow_pickle=True)
                L = int(npz["ss3"].shape[0])
                don = _emb(S_dir / f"{c}.pt", dl)
                nat = _emb(E_dir / f"{c}.pt", il)
                if don.shape[0] != L or nat.shape[0] != L:
                    continue
                mapped = W.predict(don.numpy()).astype(np.float32)
                held_true.append(nat.numpy()); held_pred.append(mapped)
                inj = torch.from_numpy(mapped)
                idx = rng.choice(L, min(per_chain, L), replace=False)
                feats["stitched"].append(_run_blocks(model, inj[None], il)[0][idx].numpy())
                feats["native"].append(_run_blocks(model, nat[None], il)[0][idx].numpy())
                feats["stitched_rand"].append(_run_blocks(rnd, inj[None], il)[0][idx].numpy())
                feats["donor"].append(don[idx].numpy())          # donor with NO receiver blocks
                labs["ss3"].append(np.array([SS3_IDX.get(x, 2) for x in npz["ss3"][idx]]))
                r = npz["rsa"][idx].astype(np.float64)
                labs["burial"].append(np.where(np.isfinite(r), (r < BURIAL_RSA).astype(int), -1))
                labs["rsa"].append(np.where(np.isfinite(r), r, np.nan))
            if not held_true:
                continue
            conn_r2 = float(r2_score(np.concatenate(held_true), np.concatenate(held_pred),
                                     multioutput="variance_weighted"))
            F = {k: np.concatenate(v) for k, v in feats.items()}
            Y = {k: np.concatenate(v) for k, v in labs.items()}
            n = len(Y["ss3"]); te = np.zeros(n, bool)
            te[rng.choice(n, n // 3, replace=False)] = True; tr = ~te

            rec = {"donor_layer": dl + 1, "inject_layer": il,
                   "connector_r2_heldout": round(conn_r2, 4), "n_eval_residues": int(n)}
            for prop in properties:
                if prop not in Y:
                    continue
                kind = "reg" if prop == "rsa" else "clf"
                m = np.isfinite(Y[prop]) if kind == "reg" else (Y[prop] >= 0)
                for cond in ("stitched", "native", "stitched_rand", "donor"):
                    sc = _score(F[cond], Y[prop], tr & m, te & m, kind)
                    rec[f"{prop}_{cond}"] = sc[0] if isinstance(sc, tuple) else sc
            rows.append(rec)
            # Report stitched against DONOR-DIRECT, not against native. Native is the flattering
            # comparison -- the donor is simply better at these properties to begin with -- so
            # quoting it alone makes stitching look like it works when it does not.
            msg = "  ".join(f"{p} stitch={rec.get(f'{p}_stitched')} donor={rec.get(f'{p}_donor')}"
                            for p in properties)
            print(f"  {label} L{dl+1} -> ESM L{il:<2d}  connector R2={conn_r2:+.3f}   {msg}", flush=True)
            if on_row:
                on_row(rows)
    return rows


def _main_grid() -> None:
    """Stitch every donor layer into every receiver depth, and score what survives.

    Depth stitching so far used a fixed donor (the structure encoder's last layer) and two
    injection points. That leaves the obvious question unanswered: is there a depth at which
    the two models' computations actually compose, and does the best donor layer depend on
    where you inject?

    For each (donor layer, injection layer) this fits a linear connector on training chains,
    injects the mapped representation at that depth, lets the receiver's remaining frozen
    blocks run, and probes the result.

    Four conditions per cell make the result identifiable, and all four are necessary:

        stitched        donor -> connector -> receiver's TRAINED blocks -> probe
        stitched_rand   same, but the receiver's blocks are untrained   (did training matter?)
        donor           the donor probed directly, no receiver blocks   (did the blocks add anything?)
        native          the receiver's own state at that depth          (the ceiling)

    The `donor` condition is the one that cannot be omitted. The donor here is a structure
    model, which is already excellent at local geometry, so a linear map of it retains that
    signal and a probe will score highly on secondary structure no matter what the receiver
    does. Without donor-direct, a high `stitched` score reads as composability when it is
    merely the donor's own information surviving a linear transform.

    It also records the connector's own held-out R^2 at every cell, which is the control the
    earlier depth-stitch run lacked. That run found untrained receiver blocks beating trained
    ones -- explicable if the connector cannot reach the receiver's layer distribution, so
    the injected state is off-distribution for trained blocks and merely harmless to random
    ones. If connector quality is low everywhere, the stitching scores say more about the map
    than about composability, and the grid makes that visible.

    Outputs (under --out-dir): stitch_grid.csv, summary.txt
    """
    import argparse
    import csv
    import json
    import warnings
    from pathlib import Path

    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="Stitching grid: donor layer x injection depth")
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--esm-dir", default="/ssc/results/esm")
    ap.add_argument("--struct-dir", default="/ssc/results/proteinmpnn")
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--out-dir", default="/ssc/results/stitch_grid")
    ap.add_argument("--n-chains", type=int, default=400)
    ap.add_argument("--per-chain", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--properties", nargs="+", default=["ss3", "burial", "rsa"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)

    sdir = Path(args.structures_dir); prot = sdir / "proteins"
    rest = sdir / "residue_targets"
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    E_dir, S_dir = Path(args.esm_dir), Path(args.struct_dir)
    use = [c for c in ids if (E_dir / f"{c}.pt").exists() and (S_dir / f"{c}.pt").exists()][: args.n_chains]
    n_fit = int(len(use) * 0.6)
    fit_c, ev_c = use[:n_fit], use[n_fit:]

    import esm as esmlib
    model, alphabet = getattr(esmlib.pretrained, args.esm_model)(); model.eval()
    # untrained twin of the receiver: same architecture, random weights. Without it a high
    # stitched score cannot be attributed to the receiver having LEARNED anything. Seeded, so
    # the stitched_rand column reproduces; before this the initialisation varied run to run.
    torch.manual_seed(args.seed)
    rnd = esmlib.model.esm2.ESM2(
        num_layers=model.num_layers,
        embed_dim=getattr(model, "embed_dim", None) or model.args.embed_dim,
        attention_heads=20, alphabet=alphabet)
    rnd.eval()
    n_esm = model.num_layers
    n_don = torch.load(S_dir / f"{use[0]}.pt", weights_only=False)["layers"].shape[0]
    print(f"{len(use)} chains ({len(fit_c)} fit / {len(ev_c)} eval); "
          f"{n_don} donor layers x {n_esm} injection depths", flush=True)

    rng = np.random.default_rng(args.seed)

    def checkpoint(rows):
        with (out / "stitch_grid.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    rows = _grid_cells(model, rnd, S_dir, E_dir, prot, fit_c, ev_c,
                       donor_layers=range(n_don), inject_layers=range(1, n_esm + 1),
                       rng_for=lambda dl, il: rng, per_chain=args.per_chain,
                       properties=args.properties, connector="ridge", alpha=args.alpha,
                       on_row=checkpoint)

    best = max(rows, key=lambda r: r["connector_r2_heldout"]) if rows else None
    lines = ["Stitching grid: every donor layer into every injection depth.",
             "",
             "Four conditions per cell. The comparison that decides the question is",
             "stitched vs DONOR, not stitched vs native:",
             "  donor          the donor representation alone, no receiver blocks at all",
             "  stitched_rand  through the receiver's UNTRAINED blocks  (did training matter?)",
             "  stitched       through the receiver's TRAINED blocks",
             "  native         the receiver's own representation",
             "",
             "connector_r2_heldout is the linear map's own quality on held-out chains -- if it is",
             "low, the stitched scores describe the map more than the models' composability.", ""]
    if best:
        lines.append(f"best connector: donor enc{best['donor_layer']} -> ESM L{best['inject_layer']}"
                     f"  R2={best['connector_r2_heldout']:.3f}")

    # headline: receiver blocks only earn their keep if stitched beats donor-direct
    wins = sum(1 for r in rows for p in args.properties if r[f"{p}_stitched"] > r[f"{p}_donor"])
    total = len(rows) * len(args.properties)
    lines.append(f"stitched beat donor-direct in {wins} of {total} cells "
                 f"({'receiver blocks add nothing' if wins * 2 < total else 'see per-cell rows'})")
    for cond in ("donor", "stitched_rand", "stitched", "native"):
        means = " ".join(f"{p}={statistics.fmean(r[f'{p}_{cond}'] for r in rows):.3f}"
                         for p in args.properties)
        lines.append(f"  mean {cond:<14} {means}")
    lines.append("")
    for r in rows:
        lines.append(f"  enc{r['donor_layer']} -> L{r['inject_layer']:<2d} "
                     f"connR2={r['connector_r2_heldout']:+.3f} " +
                     " ".join(f"{p}[don={r[f'{p}_donor']} rnd={r[f'{p}_stitched_rand']} "
                              f"stch={r[f'{p}_stitched']} nat={r[f'{p}_native']}]"
                              for p in args.properties))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    qc.record_params(out, args, extra={"n_cells": len(rows)})
    print(f"-> {out}")


def _main_gridci() -> None:
    """The stitching grid with intervals: independent repeats on fresh chain samples.

    A single `grid` run scores each cell once on ~1,300 held-out residues, so differences of a
    few thousandths between stitched and donor-direct sit inside the noise. This subcommand repeats
    the grid on independent chain samples and reports, per cell and property, the paired
    difference stitched - donor as mean +- 1.96 sd across repeats (the refit convention used for
    the probes). A cell counts as worse or better only when that interval excludes zero.

    --connector mlp replaces the linear map with a one-hidden-layer network, which tests whether
    a stronger map changes the answer. --donor-dir takes any donor model, so a within-modality
    pair (CARP -> ESM-2) serves as a positive control for the protocol itself.

    Outputs (under --out-dir): repeats.csv (one row per repeat x cell), cells.csv (aggregated),
    summary.txt. Resumes from repeats.csv.
    """
    ap = argparse.ArgumentParser(description="Stitching grid with intervals across repeats")
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--esm-dir", default="/ssc/results/esm")
    ap.add_argument("--donor-dir", default="/ssc/results/proteinmpnn")
    ap.add_argument("--donor-name", default="ProteinMPNN")
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--out-dir", default="/ssc/results/stitch_grid_ci")
    ap.add_argument("--n-chains", type=int, default=400)
    ap.add_argument("--per-chain", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--properties", nargs="+", default=["ss3", "burial", "rsa"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--connector", choices=["ridge", "mlp"], default="ridge")
    ap.add_argument("--donor-layers", type=int, nargs="+", default=None,
                    help="1-based donor layers (default: all)")
    ap.add_argument("--inject-layers", type=int, nargs="+", default=None,
                    help="receiver injection depths (default: all)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)

    sdir = Path(args.structures_dir); prot = sdir / "proteins"
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    E_dir, S_dir = Path(args.esm_dir), Path(args.donor_dir)
    eligible = [c for c in ids if (E_dir / f"{c}.pt").exists() and (S_dir / f"{c}.pt").exists()]

    import esm as esmlib
    model, alphabet = getattr(esmlib.pretrained, args.esm_model)(); model.eval()

    def untrained(seed):
        # a fresh random receiver per repeat, seeded: the interval then also covers the
        # variation between random initialisations
        torch.manual_seed(seed)
        r = esmlib.model.esm2.ESM2(
            num_layers=model.num_layers,
            embed_dim=getattr(model, "embed_dim", None) or model.args.embed_dim,
            attention_heads=20, alphabet=alphabet)
        return r.eval()

    n_esm = model.num_layers
    n_don = torch.load(S_dir / f"{eligible[0]}.pt", weights_only=False)["layers"].shape[0]
    donor_layers = [l - 1 for l in args.donor_layers] if args.donor_layers else list(range(n_don))
    inject_layers = args.inject_layers or list(range(1, n_esm + 1))

    rep_path = out / "repeats.csv"
    done_rows = list(csv.DictReader(rep_path.open())) if rep_path.exists() else []
    done = {(int(r["repeat"]), int(r["donor_layer"]), int(r["inject_layer"])) for r in done_rows}
    all_rows = [{k: (float(v) if k not in ("repeat", "donor_layer", "inject_layer", "n_eval_residues")
                     else int(v)) for k, v in r.items()} for r in done_rows]
    print(f"{args.donor_name} -> {args.esm_model}: {len(eligible)} eligible chains; "
          f"{args.repeats} repeats x {len(donor_layers)} donor x {len(inject_layers)} depths; "
          f"connector={args.connector}; {len(done)} cells already done", flush=True)

    def save():
        with rep_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)

    for r in range(args.repeats):
        seed = args.seed0 + r
        pick = np.random.default_rng(seed).choice(len(eligible), min(args.n_chains, len(eligible)),
                                                  replace=False)
        use = [eligible[i] for i in pick]
        n_fit = int(len(use) * 0.6)
        fit_c, ev_c = use[:n_fit], use[n_fit:]
        rnd = untrained(seed)
        for dl in donor_layers:
            todo = [il for il in inject_layers if (r, dl + 1, il) not in done]
            if not todo:
                continue

            def keep(rows, r=r):
                all_rows.append({"repeat": r, **rows[-1]}); save()

            _grid_cells(model, rnd, S_dir, E_dir, prot, fit_c, ev_c,
                        donor_layers=[dl], inject_layers=todo,
                        rng_for=lambda d, i, s=seed: np.random.default_rng([s, d, i]),
                        per_chain=args.per_chain, properties=args.properties,
                        connector=args.connector, alpha=args.alpha, seed=seed,
                        on_row=keep, label=f"rep{r} {args.donor_name}")

    # ---- aggregate: paired stitched - donor per repeat, mean +- 1.96 sd across repeats
    conds = ("stitched", "native", "stitched_rand", "donor")
    cells, verdicts = [], {"worse": 0, "better": 0, "no clear difference": 0}
    by_depth = {}
    for dl in donor_layers:
        for il in inject_layers:
            reps = [x for x in all_rows if x["donor_layer"] == dl + 1 and x["inject_layer"] == il]
            if not reps:
                continue
            r2 = [x["connector_r2_heldout"] for x in reps]
            rec = {"donor_layer": dl + 1, "inject_layer": il, "n_repeats": len(reps),
                   "connector_r2_mean": round(statistics.fmean(r2), 4),
                   "connector_r2_sd": round(statistics.stdev(r2), 4) if len(r2) > 1 else 0.0}
            for prop in args.properties:
                for cond in conds:
                    v = [x[f"{prop}_{cond}"] for x in reps]
                    rec[f"{prop}_{cond}_mean"] = round(statistics.fmean(v), 4)
                    rec[f"{prop}_{cond}_sd"] = round(statistics.stdev(v), 4) if len(v) > 1 else 0.0
                for name, cond in (("delta", "stitched"), ("delta_rand", "stitched_rand")):
                    d = [x[f"{prop}_{cond}"] - x[f"{prop}_donor"] for x in reps]
                    m = statistics.fmean(d)
                    h = 1.96 * statistics.stdev(d) if len(d) > 1 else 0.0
                    rec[f"{prop}_{name}_mean"] = round(m, 4)
                    rec[f"{prop}_{name}_lo"] = round(m - h, 4)
                    rec[f"{prop}_{name}_hi"] = round(m + h, 4)
                lo, hi = rec[f"{prop}_delta_lo"], rec[f"{prop}_delta_hi"]
                v = "worse" if hi < 0 else "better" if lo > 0 else "no clear difference"
                rec[f"{prop}_verdict"] = v
                verdicts[v] += 1
                by_depth.setdefault(il, {"worse": 0, "better": 0, "no clear difference": 0})[v] += 1
            cells.append(rec)
    if not cells:
        return
    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys())); w.writeheader(); w.writerows(cells)

    total = sum(verdicts.values())
    r2m = [c["connector_r2_mean"] for c in cells]
    lines = [f"Stitching with intervals: {args.donor_name} -> {args.esm_model}, "
             f"connector = {args.connector}.",
             f"{cells[0]['n_repeats']} repeats on independent samples of {args.n_chains} chains; "
             f"{args.per_chain} residues per held-out chain.",
             "Per cell and property: stitched - donor-direct, paired within each repeat,",
             "mean +- 1.96 sd across repeats. A cell is worse or better only if that excludes zero.",
             "",
             f"cells x properties: {total}",
             *(f"  {k:<20} {v}" for k, v in verdicts.items()),
             "",
             "by injection depth (worse / no clear difference / better):",
             *(f"  L{il:<3} {c['worse']:>3} / {c['no clear difference']:>3} / {c['better']:>3}"
               for il, c in sorted(by_depth.items())),
             "",
             f"connector held-out R2: {min(r2m):.3f} to {max(r2m):.3f} "
             f"(mean {statistics.fmean(r2m):.3f})"]
    for prop in args.properties:
        lines.append("")
        lines.append(f"{prop}: mean over cells")
        for cond in conds:
            lines.append(f"  {cond:<14} {statistics.fmean(c[f'{prop}_{cond}_mean'] for c in cells):.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    qc.record_params(out, args, extra={"n_cells": len(cells), "verdicts": verdicts})
    print("\n".join(lines))
    print(f"-> {out}")


_SUBCOMMANDS = {
    "predictivity": _main_predictivity,
    "stitch": _main_stitch,
    "matrix": _main_matrix,
    "depth": _main_depth,
    "grid": _main_grid,
    "grid-ci": _main_gridci,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in _SUBCOMMANDS:
        print(__doc__)
        print("subcommands: " + ", ".join(_SUBCOMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    cmd = sys.argv.pop(1)                 # the original main() then sees its original argv
    _SUBCOMMANDS[cmd]()


if __name__ == "__main__":
    main()
