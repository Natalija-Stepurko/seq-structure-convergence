"""
qc_common.py — shared representation-similarity helpers.

linear_cka follows Kornblith et al. (2019); svcca follows Raghu et al. (2017);
mutual_knn is the Platonic-hypothesis metric (Huh et al. 2024).
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


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def record_params(out_dir, args=None, extra=None, filename="params.json") -> dict:
    """Write `params.json` beside a stage's outputs, recording what produced them.

    Results whose configuration is not recorded cannot safely be compared across runs.
    A concrete example from this project: two convergence grids computed at different
    residue budgets are not directly comparable, and nothing in `grids.npz` or the
    figures would reveal the difference — the budget lived only in whichever shell
    command happened to launch the stage. This function removes that failure mode by
    making every stage self-documenting.

    Captures the resolved argument namespace, the exact command line, the git commit
    (flagged dirty if the tree has uncommitted changes), library versions whose
    numerics affect results, and a UTC timestamp.

    Args:
        out_dir:  directory the stage writes into; created if absent.
        args:     the argparse namespace (or any object with __dict__), optional.
        extra:    dict of stage-specific facts worth pinning — e.g. the number of
                  residues actually sampled, as opposed to the requested budget.
        filename: override when several stages share one output directory (the 01*
                  stages all write into `structures/`, so they use params_01*.json)
                  and would otherwise overwrite each other's record.

    Returns the dict it wrote, so callers may log or extend it.
    """
    import json
    import subprocess
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _jsonable(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return repr(v)

    def _git(*cmd, default=None):
        try:
            r = subprocess.run(("git", "-C", str(Path(__file__).resolve().parent)) + cmd,
                               capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else default
        except Exception:
            return default

    versions = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "sklearn", "torch", "xgboost", "umap", "biotite"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass

    payload = {
        "script": Path(sys.argv[0]).name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "args": {k: _jsonable(v) for k, v in vars(args).items()} if args is not None else {},
        "git_commit": _git("rev-parse", "HEAD", default="unknown"),
        "git_dirty": bool(_git("status", "--porcelain", default="")),
        "versions": versions,
    }
    if extra:
        payload["extra"] = {k: _jsonable(v) for k, v in extra.items()}

    (out / filename).write_text(json.dumps(payload, indent=2) + "\n")
    return payload
