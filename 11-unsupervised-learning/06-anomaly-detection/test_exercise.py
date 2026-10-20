"""Checks for exercise.py (anomaly detection).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scipy's Gaussian
density and scikit-learn's IsolationForest are oracles only (sklearn's score has the opposite
sign). Isolation trees are random, so their checks use exact expectations over many trees,
with tolerances of several standard errors. Tree growing runs under mlfs.testing.bounded.
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.ensemble import IsolationForest as SkIsolationForest

from mlfs.testing import bounded


def _auc(scores, y):
    """probability that a random anomaly (y = 1) outscores a random inlier (ties count half)."""
    s1, s0 = scores[y == 1], scores[y == 0]
    return float(np.mean((s1[:, None] > s0[None]) + 0.5 * (s1[:, None] == s0[None])))


def _with_outliers(n=300, n_out=15, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = np.r_[rng.standard_normal((n, d)), rng.uniform(-6, 6, (n_out, d))]
    y = np.r_[np.zeros(n), np.ones(n_out)].astype(int)
    far = np.linalg.norm(X, axis=1) > 3.0  # uniform draws that landed inside the bulk aren't really anomalies
    y = np.where(y == 1, far.astype(int), 0)
    return X, y


def _c(n):
    return 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n


def _paths(nb, tree, X):
    return bounded(nb.path_lengths, tree, X, what="path_lengths", hint="every point must reach a leaf within the tree's depth.")


def _scores(forest, X):
    return bounded(forest.score_samples, X, budget=2_000_000, what="IsolationForest.score_samples")


# ---------------------------------------------------------------- stage 1: density-style scores


def test_stage1_gaussian_scores(nb):
    rng = np.random.default_rng(1)
    Xtr = rng.multivariate_normal([1, -1, 0], [[2, 0.8, 0], [0.8, 1, 0.3], [0, 0.3, 0.5]], 400)
    X = rng.normal(0, 2, (30, 3))
    s = nb.gaussian_scores(Xtr, X, reg=0.0)
    mvn = multivariate_normal(Xtr.mean(0), np.cov(Xtr.T, bias=True))
    want = -2 * (mvn.logpdf(X) - mvn.logpdf(Xtr.mean(0)))
    assert np.shape(s) == (30,), f"one score per row of X, got shape {np.shape(s)}"
    assert np.allclose(s, want), (
        "gaussian_scores = (x - mean)^T Sigma^-1 (x - mean) with mean and Sigma (bias=True, the MLE) from X_train: "
        "the squared Mahalanobis distance, which is -2 log density up to a constant")
    assert np.allclose(nb.gaussian_scores(Xtr, X), s, rtol=1e-4), "the default reg=1e-6 barely changes well-conditioned scores"


def test_stage1_gaussian_handles_degenerate(nb):
    X = np.c_[np.arange(10.0), np.zeros(10)]  # second feature constant: singular covariance
    s = nb.gaussian_scores(X, X)
    assert np.all(np.isfinite(s)), "reg * I on the covariance keeps a constant feature from making Sigma singular"


def test_stage1_knn_scores(nb):
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    assert np.allclose(nb.knn_distance_scores(X, k=1), [1, 1, 1, 8]), (
        "k=1: each point's distance to its nearest OTHER point: (1, 1, 1, 8). don't count the point itself")
    assert np.allclose(nb.knn_distance_scores(X, k=2), [2, 1, 2, 9]), "k=2: the second-nearest other point"
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (80, 3))
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    assert np.allclose(nb.knn_distance_scores(X, k=5), np.sort(D, 1)[:, 4]), "the 5th smallest distance to another point"
    try:
        nb.knn_distance_scores(X[:4], k=4)
    except ValueError:
        pass
    else:
        raise AssertionError("with 4 points each has only 3 others: k=4 should raise ValueError")


# ---------------------------------------------------------------- stage 2: c(n)


def test_stage2_values(nb):
    c = nb.average_path_length
    assert c(1) == 0 and c(0) == 0, "c(1) = 0: a single point needs no splits"
    assert np.isclose(c(2), 1.0), "c(2) = 1: two points need exactly one split"
    assert np.isclose(c(256), _c(256)) and np.isclose(c(256), 10.2448, atol=1e-4), (
        f"c(256) = 2 (ln 255 + 0.5772) - 2 * 255/256 = 10.2448, got {c(256)}")
    got = c(np.array([1, 2, 3, 10, 1000]))
    want = np.array([0, 1, _c(3), _c(10), _c(1000)])
    assert np.shape(got) == (5,) and np.allclose(got, want), "average_path_length works elementwise on arrays"


# ---------------------------------------------------------------- stage 3: one isolation tree


def _tree(nb, X, depth, rng):
    return bounded(nb.isolation_tree, X, depth, rng, what="isolation_tree",
                   hint="stop splitting at max_depth, at nodes with <= 1 point, and when every feature is constant.")


def test_stage3_three_points(nb):
    X = np.array([[0.0], [1.0], [10.0]])
    rng = np.random.default_rng(3)
    H = np.array([_paths(nb, _tree(nb, X, 10, rng), X) for _ in range(3000)])
    # first split uniform on (0, 10): isolates 10 with prob 0.9 (depth 1), else 0 (depth 1)
    assert np.allclose(H.mean(0), [1.9, 2.0, 1.1], atol=0.04), (
        f"average path lengths over 3,000 trees should be (1.9, 2.0, 1.1) for the points (0, 1, 10), got "
        f"{np.round(H.mean(0), 3)}. choose the threshold uniformly between the node's min and max of the chosen feature")
    assert set(np.unique(H)) <= {1.0, 2.0}, "with unlimited depth every point ends alone in a leaf: path = its depth"


def test_stage3_depth_limit_and_constant_data(nb):
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (100, 3))
    h = _paths(nb, _tree(nb, X, 0, rng), X)
    assert np.allclose(h, _c(100)), f"max_depth=0: the root is a leaf holding 100 points, so every path is c(100) = {_c(100):.3f}"
    h = _paths(nb, _tree(nb, X, 2, rng), X)
    assert np.all((h == 1) | (h >= 2)) and np.all(h <= 2 + _c(100)), (
        "max_depth=2: a point stops at depth 2 (path 2 + c(leaf size)), unless a split isolated it earlier "
        "(alone in a depth-1 leaf: path exactly 1)")
    same = np.ones((50, 2))
    h = _paths(nb, _tree(nb, same, 8, rng), same)
    assert np.allclose(h, _c(50)), "identical points can't be split: the root stays a leaf and every path is c(50)"


def test_stage3_tree_is_consistent(nb):
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (64, 2))
    t = _tree(nb, X, 6, rng)
    h = _paths(nb, t, X)
    assert h.shape == (64,) and np.all(h <= 6 + _c(64)), "paths are at most max_depth + c(leaf size)"
    Xnew = rng.normal(0, 1, (10, 2)) * 5
    assert np.all(np.isfinite(_paths(nb, t, Xnew))), "new points (even far away) must reach a leaf"
    a = _paths(nb, _tree(nb, X, 6, np.random.default_rng(9)), X)
    b = _paths(nb, _tree(nb, X, 6, np.random.default_rng(9)), X)
    assert np.allclose(a, b), "the same generator state gives the same tree"


# ---------------------------------------------------------------- stage 4: the forest


def _forest(nb, X, **kw):
    f = nb.IsolationForest(**kw)
    return bounded(f.fit, X, budget=5_000_000, what="IsolationForest.fit")


def test_stage4_identical_points_score_half(nb):
    f = _forest(nb, np.ones((40, 3)), n_estimators=10, max_samples=16, rng=0)
    assert np.allclose(_scores(f, np.ones((5, 3))), 0.5), (
        "if no split is possible, E[h] = c(max_samples) and the score is 2^-1 = 0.5 exactly. "
        "normalize by c(max_samples_), the subsample size, not by n")


def test_stage4_finds_outliers(nb):
    X, y = _with_outliers(seed=6)
    f = _forest(nb, X, n_estimators=100, rng=0)
    s = _scores(f, X)
    assert np.all((s > 0) & (s <= 1)), "scores lie in (0, 1]"
    auc = _auc(s, y)
    sk = SkIsolationForest(n_estimators=100, random_state=0).fit(X)
    sk_auc = _auc(-sk.score_samples(X), y)
    assert auc > 0.95 and auc > sk_auc - 0.03, (
        f"scattered outliers should get the highest scores: AUC {auc:.3f} (scikit-learn, sign flipped: {sk_auc:.3f}). "
        "higher score = more anomalous: s = 2^(-mean path / c(psi))")
    sks = -sk.score_samples(X)
    corr = np.corrcoef(s, sks)[0, 1]
    assert corr > 0.95, f"your scores should rank points like scikit-learn's (sign flipped); correlation {corr:.3f}"


def test_stage4_threshold_and_predict(nb):
    X, _ = _with_outliers(seed=7)
    f = _forest(nb, X, n_estimators=50, contamination=0.05, rng=1)
    s = _scores(f, X)
    assert np.isclose(f.threshold_, np.quantile(s, 0.95)), "threshold_ = the (1 - contamination) quantile of the training scores"
    p = f.predict(X)
    assert set(np.unique(p)) <= {0, 1} and np.array_equal(p, (s > f.threshold_).astype(int)), (
        "predict returns 1 for anomalies (score above threshold_) and 0 for inliers")
    assert abs(p.mean() - 0.05) < 0.01, f"on the training data about 5% are flagged, got {p.mean():.3f}"
    assert f.get_params() == dict(n_estimators=50, max_samples=256, contamination=0.05, rng=1)


def test_stage4_subsampling(nb):
    X, _ = _with_outliers(n=100, n_out=5, seed=8)
    f = _forest(nb, X, n_estimators=5, max_samples=256, rng=2)
    assert f.max_samples_ == len(X), "max_samples larger than n uses all n points (max_samples_ = min(max_samples, n))"
    f = _forest(nb, X, n_estimators=5, max_samples=32, rng=2)
    assert f.max_samples_ == 32 and len(f.estimators_) == 5, "store the subsample size and the trees (estimators_)"
    f = _forest(nb, X, n_estimators=20, max_samples=100, rng=3)
    depths = [int(t["depth"].max()) for t in f.estimators_]
    assert max(depths) == 7, (
        f"trees on 100 points are limited to depth ceil(log2(100)) = 7 (and some reach it); got maximum depths {sorted(set(depths))}")
    assert all(t["size"][0] == 100 for t in f.estimators_), "every tree's root holds max_samples_ = 100 rows"
    X2 = np.r_[X, X + 100.0]  # 210 rows: duplicates can't sneak in if the subsample is drawn without replacement
    f2 = _forest(nb, X2, n_estimators=30, max_samples=len(X2), rng=5)
    leaves = [(t["feature"] < 0) & (t["size"] > 1) & (t["depth"] < int(np.ceil(np.log2(len(X2))))) for t in f2.estimators_]
    assert not any(l.any() for l in leaves), (
        "with max_samples = n and distinct points, a leaf above the depth limit holds a single point. a leaf with several "
        "points there means repeated rows: draw the subsample WITHOUT replacement")
    try:
        nb.IsolationForest(max_samples=1).fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError("max_samples=1 can't work (c(1) = 0, so every score is NaN): raise ValueError")
    a = _scores(_forest(nb, X, n_estimators=10, max_samples=32, rng=4), X)
    b = _scores(_forest(nb, X, n_estimators=10, max_samples=32, rng=4), X)
    assert np.allclose(a, b), "the same rng gives the same forest (draw everything from as_rng(self.rng))"
