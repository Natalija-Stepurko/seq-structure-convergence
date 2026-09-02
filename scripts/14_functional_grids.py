r"""
14_functional_grids.py — layer x layer grids for the two functional criteria.

Replaces the hardcoded last-layer choice with a full sweep:

  LINEAR PREDICTIVITY   ridge map from ESM layer i -> ProteinMPNN layer j, held-out R^2,
                        both directions -> two [n_esm x n_mpnn] grids.

  STITCHING             ProteinMPNN layer j -> connector -> ESM's final-layer slot -> ESM's own
                        frozen lm_head; residue readout per j, with an untrained-MPNN floor.
                        (Only the struct->seq direction is interpretable; see S7.)

Outputs: predictivity_grids.npz, functional_grids.png, summary.txt
"""
import argparse, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def _load(p): return torch.load(p, weights_only=False)["layers"].to(torch.float32).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures-dir", default="/ssc/structures")
    ap.add_argument("--esm-dir", default="/ssc/results/esm")
    ap.add_argument("--struct-dir", default="/ssc/results/proteinmpnn")
    ap.add_argument("--rand-struct-dir", default="/ssc/results/rand_mpnn")
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--out-dir", default="/ssc/results/functional_grids")
    ap.add_argument("--n-chains", type=int, default=150)
    ap.add_argument("--per-chain", type=int, default=12)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sdir = Path(a.structures_dir); prot = sdir / "proteins"
    ids = [json.loads(l)["id"] for l in (sdir / "index.jsonl").open()
           if l.strip() and json.loads(l).get("valid", True)]
    D = {"esm": Path(a.esm_dir), "str": Path(a.struct_dir), "rand": Path(a.rand_struct_dir)}
    use = [c for c in ids if all((d / f"{c}.pt").exists() for d in D.values())][: a.n_chains]
    rng = np.random.default_rng(a.seed)

    # residue-aligned sample for the predictivity grids
    E, S, R = [], [], []
    for c in use:
        L = int(np.load(prot / f"{c}.npz", allow_pickle=True)["ss3"].shape[0])
        e, s, r = _load(D["esm"]/f"{c}.pt"), _load(D["str"]/f"{c}.pt"), _load(D["rand"]/f"{c}.pt")
        if e.shape[1] != L or s.shape[1] != L: continue
        i = rng.choice(L, min(a.per_chain, L), replace=False)
        E.append(e[:, i, :]); S.append(s[:, i, :]); R.append(r[:, i, :])
    E, S, R = np.concatenate(E,1), np.concatenate(S,1), np.concatenate(R,1)
    n = E.shape[1]; te = np.zeros(n, bool); te[rng.choice(n, int(n*a.test_frac), replace=False)] = True
    tr = ~te
    print(f"{n:,} residues; ESM {E.shape[0]} layers, MPNN {S.shape[0]} layers")

    ne, ns = E.shape[0], S.shape[0]
    G_es = np.zeros((ne, ns)); G_se = np.zeros((ne, ns))
    for i in range(ne):
        for j in range(ns):
            m = Ridge(alpha=a.alpha).fit(E[i][tr], S[j][tr])
            G_es[i, j] = r2_score(S[j][te], m.predict(E[i][te]), multioutput="variance_weighted")
            m2 = Ridge(alpha=a.alpha).fit(S[j][tr], E[i][tr])
            G_se[i, j] = r2_score(E[i][te], m2.predict(S[i*0+j][te]*0 + S[j][te]),
                                  multioutput="variance_weighted")
        print(f"  esm L{i}: max R2 -> mpnn {G_es[i].max():.3f} | from mpnn {G_se[i].max():.3f}")

    # stitching sweep: each MPNN layer -> ESM final-layer slot -> ESM lm_head
    import esm as esmlib
    mdl, alph = getattr(esmlib.pretrained, a.esm_model)(); mdl.eval()
    tr_c = [c for k, c in enumerate(use) if k % 10 != 0]
    te_c = [c for k, c in enumerate(use) if k % 10 == 0]
    def st(cids, key, layer):
        return np.concatenate([_load(D[key]/f"{c}.pt")[layer] for c in cids], 0)
    stitch = {}
    for key, lab in [("str", "ProteinMPNN"), ("rand", "untrained MPNN")]:
        accs = []
        for j in range(ns):
            W = Ridge(alpha=a.alpha).fit(st(tr_c, key, j), st(tr_c, "esm", -1))
            cor = tot = 0
            for c in te_c:
                seq = str(np.load(prot/f"{c}.npz", allow_pickle=True)["seq"])[:1022]
                x = torch.from_numpy(W.predict(_load(D[key]/f"{c}.pt")[j][:len(seq)]).astype(np.float32))
                with torch.no_grad(): lg = mdl.lm_head(x[None])[0]
                tgt = torch.tensor([alph.get_idx(ch) for ch in seq])
                cor += int((lg.argmax(-1) == tgt).sum()); tot += len(seq)
            accs.append(cor/tot); print(f"  stitch {lab} enc{j+1} -> ESM lm_head: {cor/tot:.3f}")
        stitch[lab] = accs

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out/"predictivity_grids.npz", esm_to_mpnn=G_es, mpnn_to_esm=G_se,
             stitch_real=stitch["ProteinMPNN"], stitch_rand=stitch["untrained MPNN"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    el = ["emb"]+[f"L{i}" for i in range(1, ne)]; sl = [f"enc{j+1}" for j in range(ns)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6),
                           gridspec_kw={"width_ratios":[1,1,0.8]})
    for k,(G,t) in enumerate([(G_es,"linear predictivity: ESM → MPNN"),
                              (G_se,"linear predictivity: MPNN → ESM")]):
        im = ax[k].imshow(G, aspect="auto", cmap="viridis", vmin=0)
        ax[k].set_xticks(range(ns)); ax[k].set_xticklabels(sl)
        ax[k].set_yticks(range(ne)); ax[k].set_yticklabels(el, fontsize=7)
        ax[k].set_title(t+"\n(held-out R²)", fontsize=10)
        for p in range(ne):
            for q in range(ns):
                ax[k].text(q,p,f"{G[p,q]:.2f}",ha="center",va="center",fontsize=6,
                           color="white" if G[p,q]<0.5*G.max() else "black")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    x = range(ns)
    ax[2].bar([i-0.2 for i in x], stitch["ProteinMPNN"], 0.4, label="ProteinMPNN")
    ax[2].bar([i+0.2 for i in x], stitch["untrained MPNN"], 0.4, label="untrained")
    ax[2].set_xticks(list(x)); ax[2].set_xticklabels(sl)
    ax[2].set_ylabel("residue readout"); ax[2].legend(fontsize=8)
    ax[2].set_title("stitch → ESM's own lm_head", fontsize=10); ax[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out/"functional_grids.png", dpi=140); plt.close(fig)
    print(f"-> {out}")

if __name__ == "__main__":
    main()
