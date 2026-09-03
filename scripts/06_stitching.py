r"""
06_stitching.py -- Functional tests: is one model's representation usable by the other's machinery?

Subcommands (each keeps the exact flags it had as a standalone script):

    predictivity     linear predictivity for one pair, both directions
    stitch           stitching through each model's own frozen head
    matrix           linear predictivity across every pair
    depth            per-property stitching at an intermediate layer

The subcommand token is removed from argv before the original parser runs, so every command
line that worked before still works, with the subcommand inserted after the script name:

    uv run python scripts/06_stitching.py predictivity --help

Provenance: every subcommand writes params.json beside its outputs (see qc_common.record_params).

Merged from:
    predictivity     was 12_linear_stitch.py
    stitch           was 13_model_stitching.py
    matrix           was 15_predictivity_matrix.py
    depth            was 16_depth_stitch.py
"""

import argparse
import csv
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path
import qc_common as qc
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

sys.path.insert(0, str(Path(__file__).parent))

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


    qc.record_params(out, args)   # provenance: exactly what produced these outputs
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
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

sys.path.insert(0, str(Path(__file__).parent))

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


    qc.record_params(out, args)   # provenance: exactly what produced these outputs
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
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

sys.path.insert(0, str(Path(__file__).parent))

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

sys.path.insert(0, str(Path(__file__).parent))

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


    qc.record_params(o, a)   # provenance: exactly what produced these outputs
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


# ==========================================================================================
# dispatch
# ==========================================================================================

_SUBCOMMANDS = {
    "predictivity": _main_predictivity,
    "stitch": _main_stitch,
    "matrix": _main_matrix,
    "depth": _main_depth,
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
