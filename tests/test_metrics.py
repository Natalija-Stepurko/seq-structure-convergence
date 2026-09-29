"""Invariants the study's claims rest on.

Each test pins a property that a number on the results page depends on: CKA is a similarity
and must behave like one, the cached SVCCA split used for the layer grids must equal the direct
computation, SVCCA's silent row cap and its width-dependent null are real, and the neighbour
measures sit at chance on unrelated data.
"""
import argparse
import json

import numpy as np
import pytest

from ssc import metrics as qc

RNG = np.random.default_rng(0)


def _cloud(n=600, d=32, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


def _cka(X, Y):
    return qc.linear_cka(qc.column_center(X), qc.column_center(Y))


# --- CKA -----------------------------------------------------------------------------------

def test_cka_identity_is_one():
    X = _cloud()
    assert _cka(X, X) == pytest.approx(1.0, abs=1e-12)


def test_cka_invariant_to_rotation_and_scale():
    X = _cloud()
    Q, _ = np.linalg.qr(RNG.standard_normal((32, 32)))
    assert _cka(X, 3.7 * X @ Q) == pytest.approx(1.0, abs=1e-10)


def test_cka_near_zero_for_unrelated_data():
    assert _cka(_cloud(4000, 16, 1), _cloud(4000, 16, 2)) < 0.02


# --- SVCCA ---------------------------------------------------------------------------------

def test_svcca_cached_split_matches_direct():
    """The layer grids reduce each layer once and pair afterwards; that must change nothing."""
    A, B = _cloud(900, 24, 3), _cloud(900, 40, 4)
    direct = qc.svcca(A, B)
    cached = qc.svcca_from_reduced(qc.svcca_reduce(A), qc.svcca_reduce(B))
    assert cached == direct


def test_svcca_silently_caps_rows_at_5000():
    Q = qc.svcca_reduce(_cloud(8000, 8, 5))
    assert Q.shape[0] == 5000


def test_svcca_null_grows_with_width():
    """On unrelated data SVCCA is far from zero and rises with dimension.

    CKA also has a positive null that scales with width over sample size (about d/n), but at
    the same width it stays well below SVCCA's -- which is why the page reads SVCCA against
    its own scrambled score.
    """
    narrow = qc.svcca(_cloud(3000, 16, 6), _cloud(3000, 16, 7))
    wide = qc.svcca(_cloud(3000, 256, 8), _cloud(3000, 256, 9))
    assert wide > narrow + 0.05
    assert wide > 0.1
    assert _cka(_cloud(3000, 256, 8), _cloud(3000, 256, 9)) < wide / 2


def test_svcca_identity_is_one():
    X = _cloud(800, 12, 10)
    assert qc.svcca(X, X) == pytest.approx(1.0, abs=1e-9)


# --- neighbour measures --------------------------------------------------------------------

def test_mutual_knn_identity_and_chance():
    X = _cloud(1000, 16, 11)
    assert qc.mutual_knn(X, X, k=10) == pytest.approx(1.0)
    # unrelated spaces: expected overlap is about k / n
    assert qc.mutual_knn(X, _cloud(1000, 16, 12), k=10) < 0.05


def test_knn_purity_perfect_and_chance():
    labels = np.repeat(np.arange(4), 250)
    centres = np.eye(4, 8) * 50
    X = centres[labels] + _cloud(1000, 8, 13)
    assert qc.knn_purity(X, labels, k=15) > 0.95
    shuffled = np.random.default_rng(14).permutation(labels)
    assert abs(qc.knn_purity(X, shuffled, k=15)) < 0.05


def test_lvr_low_for_smooth_target_and_near_one_for_noise():
    X = _cloud(1500, 4, 15)
    y = X[:, 0]
    assert qc.lvr(X, y, k=15) < 0.2
    assert qc.lvr(X, np.random.default_rng(16).permutation(y), k=15) > 0.8


# --- provenance ----------------------------------------------------------------------------

def test_record_params_writes_command_and_args(tmp_path):
    args = argparse.Namespace(n=3, out=tmp_path)
    rec = qc.record_params(tmp_path, args=args, extra={"n_sampled": 7})
    on_disk = json.loads((tmp_path / "params.json").read_text())
    assert on_disk == rec
    assert on_disk["args"]["n"] == 3 and on_disk["args"]["out"] == str(tmp_path)
    assert on_disk["extra"]["n_sampled"] == 7
    assert "command" in on_disk and "versions" in on_disk
