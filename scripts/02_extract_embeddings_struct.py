"""
02_extract_embeddings_struct.py — structure-arm per-residue embedding extraction.

The structure arm is a model trained on 3D coordinates. Its per-residue representation
is the counterpart to ESM-2's for the convergence study. We currently support:

    proteinmpnn   (default) — ProteinMPNN's ENCODER node embeddings. The encoder is
                  sequence-agnostic: it produces per-residue vectors purely from backbone
                  geometry (N,CA,C,O), so these are a clean "structure-only" representation.
                  Vendored under scripts/vendor/proteinmpnn/ (MIT, Dauparas 2022).

(ESM-IF1 is the intended primary structure arm but needs the torch-geometric compiled
stack, which has no wheels for the current torch; it is deferred until torch is pinned to
a PyG-supported version — see paper/PROJECT_PLAN.md.)

Weights: ProteinMPNN vanilla weights (~7 MB). Fetch once, e.g.:
    curl -L -o /scratch/.torch-hub/proteinmpnn/v_48_020.pt \\
      https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt

Outputs (under --results-dir, default ./results/proteinmpnn):
    <id>.pt   {model, id, length, n_reps, embed_dim, layers[fp16: n_reps, L, D]}

Resume: existing <id>.pt files are skipped.

Usage:
    uv run python scripts/02_extract_embeddings_struct.py \\
        --structures-dir /ssc/structures --results-dir /ssc/results/proteinmpnn \\
        --weights /scratch/.torch-hub/proteinmpnn/v_48_020.pt
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

# Vendored ProteinMPNN
sys.path.insert(0, str(Path(__file__).parent / "vendor" / "proteinmpnn"))

HIDDEN = 128
N_ENCODER_LAYERS = 3


def _load_manifest(structures_dir: Path) -> list[dict]:
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


def main() -> None:
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
    prot_dir = structures_dir / "proteins"

    print(f"Loading ProteinMPNN weights {args.weights} ...")
    model = _build_model(Path(args.weights))
    print(f"  encoder: {N_ENCODER_LAYERS} layers, dim {HIDDEN}")

    recs = _load_manifest(structures_dir)
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


if __name__ == "__main__":
    main()
