r"""
13_model_stitching.py — model stitching, both directions, through each model's own frozen head.

Structural similarity (CKA/SVCCA/mutual k-NN, and linear cross-prediction) measures whether two
representations look alike. Stitching asks the functional question the similarity literature argues
is the right one [Lenc & Vedaldi 2015; Bansal et al. 2021]: if model A's representation is fed into
model B's OWN downstream machinery through a trained connector, does B still perform B's task?

Two directions, each using the receiving model's own frozen head and its own objective:

  seq -> struct :  ESM embedding --W--> ProteinMPNN encoder-state slot -> MPNN's frozen decoder
                   task: sequence recovery (MPNN's training objective)
                   ceiling = MPNN's real encoder state; floor = randomly-initialised ESM stitched in

  struct -> seq :  ProteinMPNN embedding --W--> ESM final-layer slot -> ESM's frozen lm_head
                   task: amino-acid identity readout (ESM's training objective)
                   ceiling = ESM's real final layer; floor = randomly-initialised MPNN stitched in

The connector W is a ridge map fitted on held-out-chain training residues only; both networks stay
frozen. Only the linear connector is fitted, exactly as in stitching.

Outputs (under --out-dir): stitch_results.csv / summary.txt

Usage:
    uv run python scripts/13_model_stitching.py --structures-dir /ssc/structures \
        --esm-dir /ssc/results/esm --struct-dir /ssc/results/proteinmpnn \
        --rand-esm-dir /ssc/results/rand_esm --rand-struct-dir /ssc/results/rand_mpnn \
        --out-dir /ssc/results/stitching
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

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


def main() -> None:
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
    with (out / "stitch_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = ["Model stitching: each direction feeds one model's representation through a fitted",
             "linear connector into the OTHER model's own frozen head, evaluated on that model's",
             f"own task. {len(tr_c)} chains fit the connector, {len(te_c)} held-out chains evaluated.", ""]
    for r in rows:
        lines.append(f"  {r['direction']}\n      {r['source']:42s} {r['accuracy']:.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
