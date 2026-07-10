"""
qc_common.py — shared representation-similarity helpers.

linear_cka and svcca are ported from the materials ORB/UMA project's qc_common.py
(same implementations); mutual_knn is the Platonic-hypothesis metric (Huh et al. 2024).
All operate on 2-D arrays [n_samples, n_features]; the two inputs must share rows
(i.e. correspond to the same residues, in the same order).
"""

from __future__ import annotations

import numpy as np


def column_center(X: np.ndarray) -> np.ndarray:
    """Center each feature column (required before linear_cka)."""
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Feature-space linear CKA. X, Y must be column-centered, same number of rows."""
    xy = np.linalg.norm(Y.T @ X, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xy / (xx * yy)) if xx * yy > 0 else float("nan")


def svcca(Xa: np.ndarray, Xb: np.ndarray, var: float = 0.99,
          max_rows: int = 5000, seed: int = 42) -> float:
    """SVD-denoise each representation to `var` energy, then mean CCA correlation."""
    rng = np.random.default_rng(seed)
    n = Xa.shape[0]
    if n > max_rows and max_rows > 0:
        idx = rng.choice(n, max_rows, replace=False)
        Xa, Xb = Xa[idx], Xb[idx]

    def _reduce(X: np.ndarray) -> np.ndarray:
        Xc = X - X.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        k = int(np.searchsorted(energy, var) + 1)
        return U[:, :k] * S[:k]

    A, B = _reduce(Xa), _reduce(Xb)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def knn_purity(X: np.ndarray, labels: np.ndarray, k: int = 15,
               max_rows: int = 20000, seed: int = 42) -> float:
    """Chance-corrected k-NN label purity: mean fraction of same-label neighbours,
    rescaled so 0 = chance (sum of squared class frequencies) and 1 = perfect."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = X.shape[0]
    if n > max_rows and max_rows > 0:
        idx = rng.choice(n, max_rows, replace=False)
        X, labels = X[idx], labels[idx]
        n = max_rows
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neigh = nn.kneighbors(return_distance=False)[:, 1:]
    same = (labels[neigh] == labels[:, None]).mean()
    _, counts = np.unique(labels, return_counts=True)
    chance = float(((counts / counts.sum()) ** 2).sum())
    return float((same - chance) / (1 - chance)) if chance < 1 else float("nan")


def lvr(X: np.ndarray, y: np.ndarray, k: int = 15,
        max_rows: int = 20000, seed: int = 42) -> float:
    """Local variance ratio for a continuous target: mean within-neighbourhood variance
    of y divided by its global variance (lower = better local organisation)."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    n = X.shape[0]
    if n > max_rows and max_rows > 0:
        idx = rng.choice(n, max_rows, replace=False)
        X, y = X[idx], y[idx]
        n = max_rows
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neigh = nn.kneighbors(return_distance=False)[:, 1:]
    local_var = y[neigh].var(axis=1).mean()
    global_var = y.var()
    return float(local_var / global_var) if global_var > 0 else float("nan")


def mutual_knn(X: np.ndarray, Y: np.ndarray, k: int = 10,
               max_rows: int = 4000, seed: int = 42) -> float:
    """Mutual k-NN alignment (Huh et al. 2024): mean fraction of shared neighbours
    between the two representation spaces over the same points."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > max_rows and max_rows > 0:
        idx = rng.choice(n, max_rows, replace=False)
        X, Y = X[idx], Y[idx]
        n = max_rows
    k = min(k, n - 1)
    nx = NearestNeighbors(n_neighbors=k + 1).fit(X)
    ny = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    ix = nx.kneighbors(return_distance=False)[:, 1:]     # drop self
    iy = ny.kneighbors(return_distance=False)[:, 1:]
    shared = [len(set(a) & set(b)) / k for a, b in zip(ix, iy)]
    return float(np.mean(shared))
