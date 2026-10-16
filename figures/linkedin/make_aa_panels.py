"""Real ESM-2 35M residues (layer 6 of 12) for the LinkedIn image's three subtraction panels.
Same 78 chains as the CKA estimates; Leu, Gly and Lys shown; the subtraction uses all amino acids.

Needs the structures and ESM-2 embeddings (not tracked); writes aa_panels.npz beside this file,
which is tracked, so make_linkedin_aa.py redraws the image from tracked files alone."""
import json
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def residualise(X, aa):
    """Subtract each amino acid's mean vector (as _aa_residualise in scripts/04_convergence.py)."""
    R = X.astype(np.float64).copy()
    for a in np.unique(aa):
        R[aa == a] -= R[aa == a].mean(axis=0, keepdims=True)
    return R

LAYER, SHOW, PER_AA = 6, ("L", "G", "K"), 500
ids = [json.loads(l)["id"] for l in open(ROOT / "structures/index.jsonl")
       if l.strip() and json.loads(l).get("valid", True)][:78]
X, aa = [], []
for cid in ids:
    L = torch.load(ROOT / f"results/esm/{cid}.pt", weights_only=False)["layers"][LAYER].to(torch.float32).numpy()
    seq = str(np.load(ROOT / f"structures/proteins/{cid}.npz", allow_pickle=True)["seq"])
    assert len(seq) == L.shape[0]
    X.append(L); aa += list(seq)
X, aa = np.concatenate(X), np.array(aa)
R = residualise(X, aa)
rng = np.random.default_rng(0)
pick = np.concatenate([rng.choice(np.flatnonzero(aa == a), PER_AA, replace=False) for a in SHOW])

def pcs(M):
    Mc = M - M.mean(0)
    return Mc, np.linalg.svd(Mc, full_matrices=False)[2][:2]

Xc, Vx = pcs(X)
Rc, Vr = pcs(R)
means = np.stack([Xc[aa == a].mean(0) for a in SHOW]) @ Vx.T
np.savez(Path(__file__).with_name("aa_panels.npz"), raw=Xc[pick] @ Vx.T, res=Rc[pick] @ Vr.T, aa=aa[pick],
         means=means, show=np.array(SHOW), n_res=len(aa), share=float(((R - R.mean(0)) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()))
print(len(aa), "residues; share left", round(float(((R - R.mean(0)) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()), 3))
