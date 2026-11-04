"""The fusion experiment's helpers: no test leakage in the amino-acid centring, honest paired
intervals, and variance partitioning that adds up."""
import numpy as np

from ssc import fusion as fu


def test_onehot_marks_standard_residues_and_leaves_others_empty():
    oh = fu.onehot(np.array(list("ACX")))
    assert oh.shape == (3, 20)
    assert oh[0, fu.AA_IDX["A"]] == 1 and oh[1, fu.AA_IDX["C"]] == 1
    assert oh[2].sum() == 0


def test_residualise_uses_training_means_only():
    aa = np.array(list("AAAABBBB"))
    X = np.array([[0.], [2.], [100.], [100.], [10.], [12.], [-50.], [-50.]])
    train = np.array([True, True, False, False, True, True, False, False])
    R = fu.residualise_on_train(X, aa, train)
    # A's training mean is 1 and B's is 11; the test rows are centred with those, not their own
    np.testing.assert_allclose(R[:, 0], [-1, 1, 99, 99, -1, 1, -61, -61])


def test_residualise_changing_test_rows_does_not_move_training_rows():
    rng = np.random.default_rng(0)
    aa = rng.choice(list("ACDE"), 200)
    X = rng.normal(size=(200, 5))
    train = rng.random(200) < 0.7
    X2 = X.copy()
    X2[~train] += 1000.0
    np.testing.assert_allclose(fu.residualise_on_train(X, aa, train)[train],
                               fu.residualise_on_train(X2, aa, train)[train])


def test_amino_acid_missing_from_training_is_centred_with_the_overall_mean():
    aa = np.array(list("AAW"))
    X = np.array([[1.], [3.], [10.]])
    train = np.array([True, True, False])
    np.testing.assert_allclose(fu.residualise_on_train(X, aa, train)[:, 0], [-1, 1, 8])


def test_paired_gain_interval_reflects_the_paired_spread():
    a = np.array([0.80, 0.70, 0.75, 0.90, 0.85])
    b = a - 0.02                                   # same split noise, constant gap
    g = fu.paired_gain(a, b)
    assert abs(g["mean"] - 0.02) < 1e-12
    assert g["hi"] - g["lo"] < 1e-9                # no paired variation, no width
    g2 = fu.paired_gain([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert g2["lo"] < g2["mean"] < g2["hi"] and g2["n"] == 5


def test_variance_partition_adds_up():
    v = fu.variance_partition(0.4, 0.5, 0.7)
    assert abs(v["unique_a"] + v["unique_b"] + v["shared"] - 0.7) < 1e-12
    assert abs(v["shared"] - 0.2) < 1e-12
