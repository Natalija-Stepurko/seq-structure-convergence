r"""
02_extract.py -- All-layer per-residue embedding extraction, one arm per subcommand.

Subcommands (each keeps the exact flags it had as a standalone script):

    esm              ESM-2 all-layer per-residue embeddings
    struct           ProteinMPNN encoder per-layer node embeddings
    esmif1           ESM-IF1 (GVP-transformer) embeddings
    random           randomly-initialised counterparts (untrained floor)
    carp             CARP-38M (dilated CNN) embeddings

The subcommand token is removed from argv before the original parser runs, so every command
line that worked before still works, with the subcommand inserted after the script name:

    uv run python scripts/02_extract.py esm --help

Provenance: every subcommand writes params.json beside its outputs (see qc_common.record_params).

Merged from:
    esm              was 02_extract_embeddings_esm.py
    struct           was 02_extract_embeddings_struct.py
    esmif1           was 02c_extract_esmif1.py
    random           was 02d_extract_random_init.py
    carp             was 02e_extract_carp.py
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
import qc_common as qc
import numpy as np
import torch
import argparse, json, sys, time, warnings
import numpy as np, torch





# ======================================================================================
# esm  --  from 02_extract_embeddings_esm.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

os.environ.setdefault("TORCH_HOME", "/scratch/.torch-hub")

def _load_manifest_esm(structures_dir: Path) -> list[dict]:
    jsonl = structures_dir / "index.jsonl"
    if not jsonl.exists():
        sys.exit(f"ERROR: {jsonl} not found — run 01_fetch_proteins.py first.")
    recs = [json.loads(l) for l in jsonl.open() if l.strip()]
    recs = [r for r in recs if r.get("valid", True)]
    return recs

def _main_esm() -> None:
    ap = argparse.ArgumentParser(description="ESM-2 all-layer per-residue extraction")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-dir", default="results/esm")
    ap.add_argument("--model", default="esm2_t12_35M_UR50D",
                    help="fair-esm model name (e.g. esm2_t12_35M_UR50D, esm2_t33_650M_UR50D)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=1022,
                    help="truncate longer chains (ESM-2 positional limit ~1024 incl BOS/EOS)")
    ap.add_argument("--save-contacts", action="store_true",
                    help="also store the predicted [L,L] contact map (for a contact-recovery probe)")
    ap.add_argument("--num-threads", type=int, default=None, help="torch CPU threads")
    args = ap.parse_args()

    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    structures_dir = Path(args.structures_dir)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc.record_params(out_dir, args)   # provenance: exactly what produced these outputs
    prot_dir = structures_dir / "proteins"

    import esm
    print(f"Loading {args.model} (TORCH_HOME={os.environ['TORCH_HOME']}) ...")
    model, alphabet = getattr(esm.pretrained, args.model)()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    embed_dim = getattr(model, "embed_dim", None) or model.args.embed_dim
    rep_layers = list(range(n_layers + 1))            # 0 = embedding, then each block
    print(f"  {n_layers} blocks + embedding = {n_layers+1} representations, dim {embed_dim}")

    recs = _load_manifest_esm(structures_dir)
    if args.limit:
        recs = recs[: args.limit]
    todo = [r for r in recs if not (out_dir / f"{r['id']}.pt").exists()]
    print(f"{len(recs):,} valid chains, {len(todo):,} to extract this run.")

    t0 = time.time()
    n_ok, errors = 0, []
    for i, r in enumerate(todo, 1):
        cid = r["id"]
        try:
            npz = np.load(prot_dir / f"{cid}.npz", allow_pickle=True)
            seq = str(npz["seq"])
            truncated = len(seq) > args.max_len
            if truncated:
                seq = seq[: args.max_len]

            _, _, tokens = batch_converter([(cid, seq)])
            with torch.no_grad():
                out = model(tokens, repr_layers=rep_layers,
                            return_contacts=args.save_contacts)
            # [n_reps, L, D] fp16, BOS/EOS stripped
            layers = torch.stack(
                [out["representations"][l][0, 1:len(seq) + 1] for l in rep_layers]
            ).to(torch.float16).contiguous()

            payload = {
                "model": args.model, "id": cid, "length": len(seq),
                "truncated": truncated, "n_reps": n_layers + 1,
                "embed_dim": embed_dim, "layers": layers,
            }
            if args.save_contacts:
                payload["contacts"] = out["contacts"][0].to(torch.float16).contiguous()
            torch.save(payload, out_dir / f"{cid}.pt")
            n_ok += 1
        except Exception as exc:
            errors.append((cid, str(exc)))
        if i % 25 == 0 or i == len(todo):
            rate = i / (time.time() - t0 + 1e-9)
            print(f"  {i:,}/{len(todo):,}  ok={n_ok:,} err={len(errors):,}  ({rate:.1f} prot/s)")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min — ok={n_ok:,}, errors={len(errors):,}")
    if errors:
        (out_dir / "extract_errors.json").write_text(json.dumps(errors, indent=2))
        print(f"  errors -> {out_dir/'extract_errors.json'}")

# ======================================================================================
# struct  --  from 02_extract_embeddings_struct.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "proteinmpnn"))

HIDDEN = 128

N_ENCODER_LAYERS = 3

def _load_manifest_struct(structures_dir: Path) -> list[dict]:
    jsonl = structures_dir / "index.jsonl"
    if not jsonl.exists():
        sys.exit(f"ERROR: {jsonl} not found — run 01_fetch_proteins.py first.")
    return [r for r in (json.loads(l) for l in jsonl.open() if l.strip())
            if r.get("valid", True)]

def _build_model(weights: Path):
    from protein_mpnn_utils import ProteinMPNN
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    model = ProteinMPNN(
        num_letters=21, node_features=HIDDEN, edge_features=HIDDEN, hidden_dim=HIDDEN,
        num_encoder_layers=N_ENCODER_LAYERS, num_decoder_layers=3,
        k_neighbors=ck["num_edges"], augment_eps=0.0,
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model

def _encode(model, coords_bb: np.ndarray) -> torch.Tensor:
    """Run the (sequence-agnostic) ProteinMPNN encoder; return [n_enc, L, HIDDEN]."""
    from protein_mpnn_utils import gather_nodes

    L = coords_bb.shape[0]
    # Impute rare missing backbone atoms with the residue's CA (finite geometry); mask
    # residues whose CA itself is missing so they don't anchor the kNN graph.
    ca = coords_bb[:, 1, :]
    ca_ok = np.isfinite(ca).all(axis=1)
    filled = coords_bb.copy()
    for k in range(4):
        miss = ~np.isfinite(filled[:, k, :]).all(axis=1)
        filled[miss, k, :] = np.where(ca_ok[miss, None], ca[miss], 0.0)
    filled = np.nan_to_num(filled, nan=0.0).astype(np.float32)

    X = torch.from_numpy(filled)[None]                       # [1, L, 4, 3]
    mask = torch.from_numpy(ca_ok.astype(np.float32))[None]  # [1, L]
    residue_idx = torch.arange(L)[None]
    chain_enc = torch.ones(1, L)

    with torch.no_grad():
        E, E_idx = model.features(X, mask, residue_idx, chain_enc)
        h_V = torch.zeros((1, L, E.shape[-1]))
        h_E = model.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        reps = []
        for layer in model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            reps.append(h_V[0].clone())                       # [L, HIDDEN]
    return torch.stack(reps)                                  # [n_enc, L, HIDDEN]

def _main_struct() -> None:
    ap = argparse.ArgumentParser(description="Structure-arm (ProteinMPNN) extraction")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-dir", default="results/proteinmpnn")
    ap.add_argument("--weights", default="/scratch/.torch-hub/proteinmpnn/v_48_020.pt")
    ap.add_argument("--model-name", default="proteinmpnn_v_48_020")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-threads", type=int, default=None)
    args = ap.parse_args()

    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if not Path(args.weights).exists():
        sys.exit(f"ERROR: weights not found: {args.weights}\n  see the header for the fetch command.")

    structures_dir = Path(args.structures_dir)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc.record_params(out_dir, args)   # provenance: exactly what produced these outputs
    prot_dir = structures_dir / "proteins"

    print(f"Loading ProteinMPNN weights {args.weights} ...")
    model = _build_model(Path(args.weights))
    print(f"  encoder: {N_ENCODER_LAYERS} layers, dim {HIDDEN}")

    recs = _load_manifest_struct(structures_dir)
    if args.limit:
        recs = recs[: args.limit]
    todo = [r for r in recs if not (out_dir / f"{r['id']}.pt").exists()]
    print(f"{len(recs):,} valid chains, {len(todo):,} to extract this run.")

    t0 = time.time()
    n_ok, errors = 0, []
    for i, r in enumerate(todo, 1):
        cid = r["id"]
        try:
            npz = np.load(prot_dir / f"{cid}.npz", allow_pickle=True)
            layers = _encode(model, npz["coords_bb"]).to(torch.float16).contiguous()
            torch.save({
                "model": args.model_name, "id": cid, "length": layers.shape[1],
                "n_reps": N_ENCODER_LAYERS, "embed_dim": HIDDEN, "layers": layers,
            }, out_dir / f"{cid}.pt")
            n_ok += 1
        except Exception as exc:
            errors.append((cid, str(exc)))
        if i % 25 == 0 or i == len(todo):
            rate = i / (time.time() - t0 + 1e-9)
            print(f"  {i:,}/{len(todo):,}  ok={n_ok:,} err={len(errors):,}  ({rate:.1f} prot/s)")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min — ok={n_ok:,}, errors={len(errors):,}")
    if errors:
        (out_dir / "extract_errors.json").write_text(json.dumps(errors, indent=2))
        print(f"  errors -> {out_dir/'extract_errors.json'}")

# ======================================================================================
# esmif1  --  from 02c_extract_esmif1.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

def _main_esmif1() -> None:
    ap = argparse.ArgumentParser(description="ESM-IF1 encoder extraction")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-dir", default="results/esmif1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=1000)
    args = ap.parse_args()

    import esm
    import esm.inverse_folding
    print("loading esm_if1_gvp4_t16_142M_UR50 ...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()

    sdir = Path(args.structures_dir); prot = sdir / "proteins"
    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
    recs = [json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()]
    recs = [r for r in recs if r.get("valid", True)]
    if args.limit:
        recs = recs[: args.limit]
    todo = [r for r in recs if not (out / f"{r['id']}.pt").exists()]
    print(f"{len(recs):,} valid chains, {len(todo):,} to extract")

    t0 = time.time(); n_ok = 0; errors = []
    for i, r in enumerate(todo, 1):
        cid = r["id"]
        try:
            coords_bb = np.load(prot / f"{cid}.npz", allow_pickle=True)["coords_bb"]  # [L,4,3]
            coords = coords_bb[:, :3, :].astype(np.float32)   # N, CA, C
            if len(coords) > args.max_len:
                coords = coords[: args.max_len]
            with torch.no_grad():
                rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)  # [L, D]
            rep = torch.as_tensor(np.asarray(rep))[None].to(torch.float16).contiguous()  # [1, L, D]
            torch.save({"model": "esmif1", "id": cid, "length": rep.shape[1],
                        "n_reps": 1, "embed_dim": rep.shape[2], "layers": rep},
                       out / f"{cid}.pt")
            n_ok += 1
        except Exception as exc:
            errors.append((cid, str(exc)))
        if i % 25 == 0 or i == len(todo):
            print(f"  {i:,}/{len(todo):,}  ok={n_ok:,} err={len(errors):,}  "
                  f"({i/(time.time()-t0+1e-9):.1f}/s)")

    print(f"\nDone — ok={n_ok:,}, errors={len(errors):,}")
    if errors:
        (out / "extract_errors.json").write_text(json.dumps(errors, indent=2))
        print(f"  first error: {errors[0]}")

# ======================================================================================
# random  --  from 02d_extract_random_init.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "proteinmpnn"))

HIDDEN, N_ENC = 128, 3

def _main_random():
    ap = argparse.ArgumentParser(description="Random-init control extraction")
    ap.add_argument("--model", choices=["rand_esm", "rand_mpnn"], required=True)
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    sdir = Path(args.structures_dir); prot = sdir / "proteins"
    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
    recs = [json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()]
    recs = [r for r in recs if r.get("valid", True)]

    if args.model == "rand_esm":
        import esm
        alph = esm.data.Alphabet.from_architecture("ESM-1b")
        model = esm.model.esm2.ESM2(num_layers=12, embed_dim=480, attention_heads=20, alphabet=alph)
        model.eval(); bc = alph.get_batch_converter(); reps = list(range(13))
    else:
        from protein_mpnn_utils import ProteinMPNN, gather_nodes
        model = ProteinMPNN(num_letters=21, node_features=HIDDEN, edge_features=HIDDEN,
                            hidden_dim=HIDDEN, num_encoder_layers=N_ENC, num_decoder_layers=3,
                            k_neighbors=48, augment_eps=0.0)
        model.eval()

    todo = [r for r in recs if not (out / f"{r['id']}.pt").exists()]
    print(f"[{args.model}] {len(todo):,} to extract"); t0 = time.time(); ok = 0; err = []
    for i, r in enumerate(todo, 1):
        cid = r["id"]
        try:
            npz = np.load(prot / f"{cid}.npz", allow_pickle=True)
            if args.model == "rand_esm":
                seq = str(npz["seq"])[:1022]
                _, _, tok = bc([(cid, seq)])
                with torch.no_grad():
                    o = model(tok, repr_layers=reps)
                layers = torch.stack([o["representations"][l][0, 1:len(seq)+1] for l in reps])
            else:
                from protein_mpnn_utils import gather_nodes
                cb = npz["coords_bb"]; L = cb.shape[0]
                ca = cb[:, 1, :]; ca_ok = np.isfinite(ca).all(axis=1)
                f = cb.copy()
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
                    st = []
                    for layer in model.encoder_layers:
                        h_V, h_E = layer(h_V, h_E, E_idx, mask, ma); st.append(h_V[0].clone())
                layers = torch.stack(st)
            torch.save({"model": args.model, "id": cid, "length": layers.shape[1],
                        "n_reps": layers.shape[0], "embed_dim": layers.shape[2],
                        "layers": layers.to(torch.float16).contiguous()}, out / f"{cid}.pt")
            ok += 1
        except Exception as e:
            err.append((cid, str(e)))
        if i % 500 == 0 or i == len(todo):
            print(f"  {i:,}/{len(todo):,} ok={ok:,} err={len(err)} ({i/(time.time()-t0+1e-9):.1f}/s)")
    print(f"Done ok={ok:,} err={len(err):,}")
    if err: print("  first error:", err[0])

# ======================================================================================
# carp  --  from 02e_extract_carp.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

def _main_carp() -> None:
    ap = argparse.ArgumentParser(description="CARP (CNN protein LM) extraction")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-dir", default="results/carp")
    ap.add_argument("--model", default="carp_38M")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="layer indices to keep (default: evenly spaced across the stack)")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from sequence_models.pretrained import load_model_and_alphabet
    print(f"loading {args.model} ...")
    model, collater = load_model_and_alphabet(args.model)
    model.eval()
    n_blocks = len(getattr(model.model, "embedder").layers) if hasattr(model, "model") else 16
    layers = args.layers or sorted({0, n_blocks // 4, n_blocks // 2, 3 * n_blocks // 4, n_blocks})
    layers = [l for l in layers if 0 <= l <= n_blocks]
    print(f"  {n_blocks} blocks; extracting layers {layers}")

    sdir = Path(args.structures_dir); prot = sdir / "proteins"
    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    qc.record_params(out, args)   # provenance: exactly what produced these outputs
    recs = [json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()]
    recs = [r for r in recs if r.get("valid", True)]
    if args.limit:
        recs = recs[: args.limit]
    todo = [r for r in recs if not (out / f"{r['id']}.pt").exists()]
    print(f"{len(todo):,} to extract")

    t0 = time.time(); ok = 0; errors = []
    for i, r in enumerate(todo, 1):
        cid = r["id"]
        try:
            seq = str(np.load(prot / f"{cid}.npz", allow_pickle=True)["seq"])[: args.max_len]
            x = collater([[seq]])[0]
            with torch.no_grad():
                o = model(x, repr_layers=layers, logits=False)
            reps = o["representations"]
            stack = torch.stack([reps[l][0, : len(seq)] for l in sorted(reps.keys())])
            torch.save({"model": args.model, "id": cid, "length": stack.shape[1],
                        "n_reps": stack.shape[0], "embed_dim": stack.shape[2],
                        "layers": stack.to(torch.float16).contiguous()}, out / f"{cid}.pt")
            ok += 1
        except Exception as exc:
            errors.append((cid, str(exc)))
        if i % 250 == 0 or i == len(todo):
            print(f"  {i:,}/{len(todo):,} ok={ok:,} err={len(errors)} "
                  f"({i/(time.time()-t0+1e-9):.1f}/s)")
    print(f"Done ok={ok:,} errors={len(errors):,}")
    if errors:
        (out / "extract_errors.json").write_text(json.dumps(errors, indent=2))
        print("  first error:", errors[0])


# ==========================================================================================
# dispatch
# ==========================================================================================

_SUBCOMMANDS = {
    "esm": _main_esm,
    "struct": _main_struct,
    "esmif1": _main_esmif1,
    "random": _main_random,
    "carp": _main_carp,
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
