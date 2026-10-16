"""
ssc.fusion -- helpers for the deconfounded-fusion experiment (scripts/08_fusion.py).

The design is fixed in research/future_work.md (idea 2): amino-acid means are fitted on training
residues only, gains are paired within a refit, and redundancy is measured by variance
partitioning of linear held-out scores.
"""

from __future__ import annotations

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

# Two-sided 95% Student t quantiles by degrees of freedom (refits - 1); the study uses five refits.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def onehot(aa: np.ndarray) -> np.ndarray:
    """One column per standard amino acid; non-standard residues get a zero row."""
    out = np.zeros((len(aa), len(AA)), dtype=np.float32)
    for i, a in enumerate(aa):
        j = AA_IDX.get(a)
        if j is not None:
            out[i, j] = 1.0
    return out


def residualise_on_train(X: np.ndarray, aa: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Subtract each amino acid's mean vector, fitted on the training rows only.

    Test rows are centred with the training means, so nothing about the test set enters the
    transformation. An amino acid absent from the training rows is centred with the overall
    training mean.
    """
    X = np.asarray(X, dtype=np.float64)
    fallback = X[train].mean(axis=0)
    out = X - fallback
    for a in np.unique(aa):
        rows = aa == a
        fit = rows & train
        if fit.any():
            out[rows] = X[rows] - X[fit].mean(axis=0)
    return out.astype(np.float32)


def paired_gain(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean of a - b over paired refits with a 95% t-interval.

    a and b hold one score per refit, aligned so that position k of both was computed on the same
    residues and split. Pairing removes the split-to-split variation both arms share.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m = float(d.mean())
    if n == 1:
        return {"mean": m, "lo": float("nan"), "hi": float("nan"), "n": 1}
    half = T975[n - 1] * float(d.std(ddof=1)) / np.sqrt(n)
    return {"mean": m, "lo": m - half, "hi": m + half, "n": n}


def variance_partition(s_a: float, s_b: float, s_ab: float) -> dict:
    """Commonality analysis of three held-out scores (A alone, B alone, A and B together).

    unique to A = s_ab - s_b, unique to B = s_ab - s_a, shared = s_a + s_b - s_ab. With R^2 these
    are the classical variance components; the shared term can go negative when the two inputs
    are complementary in a way neither shows alone.
    """
    return {"unique_a": s_ab - s_b, "unique_b": s_ab - s_a, "shared": s_a + s_b - s_ab}
