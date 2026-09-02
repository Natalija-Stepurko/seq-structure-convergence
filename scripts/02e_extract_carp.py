"""
02e_extract_carp.py — CARP (dilated-CNN protein language model) embeddings.

A THIRD sequence-side architecture family for the convergence study. CARP [Yang et al. 2024] is a
masked language model over protein sequence like ESM-2, but its backbone is a dilated
**convolutional** network rather than a transformer, so pairing it with the structure models tests
whether sequence↔structure convergence depends on self-attention. carp_38M is a close parameter
match to esm2_t12_35M, making the comparison architecture-controlled at fixed scale.

RUN WITH THE ISOLATED VENV:
    /scratch/.venv-carp/bin/python scripts/02e_extract_carp.py \\
        --structures-dir /ssc/structures --results-dir /ssc/results/carp

Outputs: <id>.pt  {model, id, length, n_reps, embed_dim, layers[fp16: n_reps, L, D]}
"""

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch


def main() -> None:
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


if __name__ == "__main__":
    main()
