"""
02c_extract_esmif1.py — ESM-IF1 (GVP-GNN) structure-encoder embeddings.

A SECOND structure model for the convergence robustness check: ESM-IF1 is a GVP-Transformer
inverse-folding model (structure → sequence), architecturally distinct from both ESM-2 (sequence
Transformer) and ProteinMPNN (message-passing GNN). We extract its per-residue encoder output
from each chain's backbone (N, CA, C) coordinates, so the sequence↔structure convergence can be
tested against a structure model other than ProteinMPNN.

RUN WITH THE ISOLATED VENV (torch 2.4 + PyG):
    /scratch/.venv-if/bin/python scripts/02c_extract_esmif1.py --structures-dir /ssc/structures \\
        --results-dir /ssc/results/esmif1

Outputs (under --results-dir): <id>.pt  {model, id, length, n_reps, embed_dim, layers[fp16: 1,L,D]}
Resume: existing <id>.pt skipped.
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


def main() -> None:
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


if __name__ == "__main__":
    main()
