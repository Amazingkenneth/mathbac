import numpy as np

from mathbac.eval import metrics


def test_auc_basic_cases():
    y = np.array([0, 0, 1, 1])
    assert metrics.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert metrics.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert metrics.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    # one inversion out of four positive-negative pairs
    assert metrics.roc_auc(y, np.array([0.1, 0.85, 0.8, 0.9])) == 0.75


def test_auc_matches_pairwise_definition_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) + 0.3 * y, 2)  # rounding creates ties
    pos, neg = s[y == 1], s[y == 0]
    expected = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert abs(metrics.roc_auc(y, s) - expected) < 1e-12


def test_accuracy_f1_and_per_fold():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1])
    assert metrics.accuracy(y, s) == 0.5
    assert metrics.f1(y, s) == 0.5
    r = metrics.per_fold(np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.2, 0.8]))
    assert r["auc"] == (0.5, 0.5)


def test_set_scores():
    r = metrics.set_scores({"1", "2", "5"}, {"1", "2", "3"}, {str(i) for i in range(1, 11)})
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 6)
    assert r["accuracy"] == 0.8
