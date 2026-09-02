"""
02d_extract_random_init.py — random-initialised (untrained) control models.

Critical control for the convergence claim: are the shared directions between a sequence model
and a structure model a product of TRAINING, or merely of architecture plus the statistics of the
input? We build each architecture with random weights (no pretrained checkpoint) and run the same
extraction, so the identical convergence analysis can be applied to untrained counterparts.

    --model rand_esm    ESM-2 architecture (12 layers, 480-d), random weights
    --model rand_mpnn   ProteinMPNN encoder (3 layers, 128-d), random weights

Usage:
    uv run python scripts/02d_extract_random_init.py --model rand_esm \\
        --structures-dir /ssc/structures --results-dir /ssc/results/rand_esm
"""
import argparse, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "proteinmpnn"))
HIDDEN, N_ENC = 128, 3


def main():
    ap = argparse.ArgumentParser(description="Random-init control extraction")
    ap.add_argument("--model", choices=["rand_esm", "rand_mpnn"], required=True)
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    sdir = Path(args.structures_dir); prot = sdir / "proteins"
    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
