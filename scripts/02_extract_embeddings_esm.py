"""
02_extract_embeddings_esm.py — ESM-2 all-layer per-residue embedding extraction.

For each protein chain in the stage-01 manifest, runs a single ESM-2 forward pass and
saves the per-residue representation at *every* layer (embedding + each transformer
block) as one .pt file. These node embeddings are the sequence arm's input to the
convergence (04) and probe (05) stages.

The sequence model never sees 3D coordinates — the whole point of the study is that
these representations nonetheless organise structural/biophysical information.

Weights: downloaded on first use via torch.hub. Set TORCH_HOME to a large disk so the
~2.5 GB esm2_t33_650M weights do not land on a small root partition:
    export TORCH_HOME=/scratch/.torch-hub

Outputs (under --results-dir, default ./results/esm):
    <id>.pt   {model, id, length, truncated, n_reps, embed_dim, layers[fp16: n_reps,L,D]}
    (optional --save-contacts) also stores the predicted contact map [L,L] fp16

Resume: existing <id>.pt files are skipped.

Usage:
    uv run python scripts/02_extract_embeddings_esm.py \
        --structures-dir /ssc/structures --results-dir /ssc/results/esm \
        --model esm2_t12_35M_UR50D
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Keep model weights off a small root disk unless the caller set TORCH_HOME already.
os.environ.setdefault("TORCH_HOME", "/scratch/.torch-hub")

import numpy as np
import torch


def _load_manifest(structures_dir: Path) -> list[dict]:
    jsonl = structures_dir / "index.jsonl"
    if not jsonl.exists():
        sys.exit(f"ERROR: {jsonl} not found — run 01_fetch_proteins.py first.")
    recs = [json.loads(l) for l in jsonl.open() if l.strip()]
    recs = [r for r in recs if r.get("valid", True)]
    return recs


def main() -> None:
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
    prot_dir = structures_dir / "proteins"

    import esm
    print(f"Loading {args.model} (TORCH_HOME={os.environ['TORCH_HOME']}) ...")
    model, alphabet = getattr(esm.pretrained, args.model)()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    rep_layers = list(range(n_layers + 1))            # 0 = embedding, then each block
    print(f"  {n_layers} blocks + embedding = {n_layers+1} representations, dim {model.embed_dim}")

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
                "embed_dim": model.embed_dim, "layers": layers,
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


if __name__ == "__main__":
    main()
