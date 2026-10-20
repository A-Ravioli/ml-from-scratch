"""Checks for exercise.py (gradient boosting).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Binning and
histograms are checked against brute-force loops; split gains and leaf weights by hand (the
lesson's worked example) and against brute force; one histogram tree with lambda = 0 and h = 1
must equal a CART regression tree fitted to the bin codes (scikit-learn as an oracle); the
boosters against first principles (monotone training loss, Newton leaf values) and against
scikit-learn's HistGradientBoosting* on synthetic data. Fits run under mlfs.testing.bounded.
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as SkHGBC
from sklearn.ensemble import HistGradientBoostingRegressor as SkHGBR
from sklearn.tree import DecisionTreeRegressor as SkTree
from threadpoolctl import threadpool_limits

from mlfs.testing import bounded

_HINT = "every node popped from the stack must become a leaf or push two children; cap the depth"


def _fit(est, *args, budget=20_000_000):
    return bounded(est.fit, *args, budget=budget, what=f"{type(est).__name__}.fit", hint=_HINT)


def _reg_data(seed, n=400, d=4, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 2 * X[:, 0] + np.sin(3 * X[:, 1]) + X[:, 2] * X[:, 3] + noise * rng.normal(size=n)
    return X, y, rng


def _clf_data(seed, n=400, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    logit = 2 * X[:, 0] - X[:, 1] ** 2 + X[:, 2] * X[:, 3] + 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, rng


# ---------------------------------------------------------------- stage 1: binning


def test_stage1_few_distinct_values(nb):
    X = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [1.0, 5.0], [0.0, 5.0]])
    edges = nb.quantile_bin_edges(X, 256)
    assert len(edges) == 2 and np.allclose(edges[0], [0.5, 1.5]) and len(edges[1]) == 0, (
        f"with at most n_bins distinct values every value gets its own bin, with edges at the midpoints: "
        f"feature 0 (values 0, 1, 2) -> [0.5, 1.5]; a constant feature has no edges. got {edges}")
    codes = np.asarray(nb.bin_data(X, edges))
    assert codes.dtype == np.uint8 and codes.shape == (5, 2), f"codes must be uint8 (n, d), got {codes.dtype}"
    assert np.array_equal(codes[:, 0], [0, 1, 2, 1, 0]) and np.all(codes[:, 1] == 0), (
        f"code = number of edges strictly below x: values 0, 1, 2 -> codes 0, 1, 2. got {codes[:, 0].tolist()}")


def test_stage1_quantile_bins(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    X[:, 2] = np.where(rng.random(4000) < 0.6, 0.0, X[:, 2])  # a spike: 60% of the values are 0
    edges = nb.quantile_bin_edges(X, 8)
    assert all(len(e) <= 7 for e in edges) and len(edges[0]) == 7, (
        f"n_bins=8 means at most 7 edges per feature (8 bins); a continuous feature uses all 7. got {[len(e) for e in edges]}")
    codes = np.asarray(nb.bin_data(X, edges))
    counts = np.bincount(codes[:, 0], minlength=8)
    assert counts.min() > 400 and counts.max() < 600, (
        f"quantile edges put about n / n_bins = 500 rows in every bin, got counts {counts.tolist()}")
    assert len(edges[2]) < 7 and np.all(np.diff(edges[2]) > 0), (
        "repeated quantiles (the spike at 0) must collapse: edges are strictly increasing, so fewer bins")
    for j in range(3):
        for b in range(len(edges[j])):
            assert np.array_equal(codes[:, j] <= b, X[:, j] <= edges[j][b]), (
                "the defining property: code <= b exactly when x <= edges[b] (searchsorted with side='left')")


def test_stage1_new_data_and_limits(nb):
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(500, 2))
    edges = nb.quantile_bin_edges(X, 16)
    codes = np.asarray(nb.bin_data(np.array([[-5.0, 0.5], [5.0, 0.5]]), edges))
    assert codes[0, 0] == 0 and codes[1, 0] == len(edges[0]), (
        "values outside the training range go to the first and the last bin")
    full = np.asarray(nb.bin_data(rng.uniform(size=(5000, 2)), nb.quantile_bin_edges(rng.uniform(size=(5000, 2)), 256)))
    assert full.max() <= 255, "with n_bins=256 every code fits in a uint8"


# ---------------------------------------------------------------- stage 2: histograms and gains


def test_stage2_worked_example(nb):
    codes = np.array([[0], [0], [0], [1], [2], [2]], dtype=np.uint8)
    y = np.array([1.0, 1.0, 3.0, 5.0, 7.0, 7.0])
    g, h = 4.0 - y, np.ones(6)  # squared loss at F = mean(y) = 4
    G, H = nb.gradient_histograms(codes, g, h, 3)
    assert np.allclose(G, [[7.0, -1.0, -6.0]]) and np.allclose(H, [[3.0, 1.0, 2.0]]), (
        f"the lesson's worked example: G = [[7, -1, -6]], H = [[3, 1, 2]], got G = {np.asarray(G).tolist()}, "
        f"H = {np.asarray(H).tolist()}")
    f, b, gain = nb.best_hist_split(np.asarray(G), np.asarray(H), reg_lambda=1.0)
    assert (f, b) == (0, 0) and np.isclose(gain, 12.25), (
        f"split after bin 0: 1/2 (7^2/4 + 7^2/4 - 0^2/7) = 12.25 beats bin 1 (9.6). got ({f}, {b}, {gain})")
    assert np.isclose(nb.leaf_weight(7.0, 3.0, 1.0), -1.75), "leaf weight = -G / (H + lambda) = -7 / 4"
    _, _, g13 = nb.best_hist_split(np.asarray(G), np.asarray(H), reg_lambda=1.0, gamma=13.0)
    assert np.isclose(g13, -0.75), "gamma is subtracted from the gain: 12.25 - 13 = -0.75 (so: don't split)"


def test_stage2_histograms_match_loops(nb):
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 10, size=(300, 4)).astype(np.uint8)
    g, h = rng.normal(size=300), rng.uniform(0.1, 1.0, size=300)
    G, H = (np.asarray(a) for a in nb.gradient_histograms(codes, g, h, 12))
    assert G.shape == (4, 12) and H.shape == (4, 12), f"histograms are (d, n_bins) = (4, 12), got {G.shape}"
    for j in range(4):
        for b in range(12):
            m = codes[:, j] == b
            assert np.isclose(G[j, b], g[m].sum()) and np.isclose(H[j, b], h[m].sum()), (
                f"G[{j}, {b}] must be the sum of g over the rows whose feature {j} has code {b}")


def test_stage2_split_gain_and_best_split(nb):
    rng = np.random.default_rng(3)
    G_hist, H_hist = rng.normal(size=(3, 8)), rng.uniform(0.0, 2.0, size=(3, 8))
    G_hist -= G_hist.sum(1, keepdims=True) / 8 - G_hist[0].sum() / 8  # every feature has the same totals
    H_hist *= H_hist[0].sum() / H_hist.sum(1, keepdims=True)
    lam, gam, mcw = 0.7, 0.1, 1.5
    best, arg = -np.inf, (-1, -1)
    for j in range(3):
        for b in range(7):
            GL, HL = G_hist[j, : b + 1].sum(), H_hist[j, : b + 1].sum()
            GR, HR = G_hist[j].sum() - GL, H_hist[j].sum() - HL
            if HL < mcw or HR < mcw:
                continue
            gain = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - (GL + GR) ** 2 / (HL + HR + lam)) - gam
            assert np.isclose(nb.split_gain(GL, HL, GR, HR, lam, gam), gain), "split_gain formula"
            if gain > best:
                best, arg = gain, (j, b)
    f, b, gain = nb.best_hist_split(G_hist, H_hist, reg_lambda=lam, gamma=gam, min_child_weight=mcw)
    assert (f, b) == arg and np.isclose(gain, best), (
        f"brute force over every (feature, bin) with both children's hessian sums >= min_child_weight finds "
        f"{arg} with gain {best:.4f}; got ({f}, {b}, {gain:.4f}). left child = bins 0..b (a cumulative sum).")
    Gt = np.array([[-2.0, -2.0, -2.0, 6.0, 0.0], [-6.0, 6.0, 0.0, 0.0, 0.0]])
    Ht = np.array([[1.0, 1.0, 1.0, 1.0, 0.0], [3.0, 1.0, 0.0, 0.0, 0.0]])
    assert nb.best_hist_split(Gt, Ht, reg_lambda=0.0)[:2] == (0, 2), (
        "feature 0 after bin 2 and feature 1 after bin 0 make the same split (gain 24): ties go to the LOWEST "
        "FEATURE first, then the lowest bin (np.argmax over the (d, n_bins - 1) gains, row by row)")
    none = nb.best_hist_split(G_hist, H_hist, min_child_weight=1e9)
    assert none[0] == -1 and none[2] == -np.inf, "no valid split -> (-1, -1, -inf)"


def test_stage2_histograms_vectorized(nb):
    rng = np.random.default_rng(4)
    codes = rng.integers(0, 256, size=(20000, 8)).astype(np.uint8)
    g, h = rng.normal(size=20000), np.ones(20000)

    def best(fn):
        out = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
        return min(out)

    t_base = best(lambda: np.bincount(codes[:, 0], weights=g, minlength=256))
    t_yours = best(lambda: bounded(nb.gradient_histograms, codes, g, h, 256, budget=100_000,
                                   what="gradient_histograms", hint="no Python loop over bins or rows"))
    assert t_yours <= 60 * t_base + 0.01, (
        f"gradient_histograms took {1e3 * t_yours:.1f} ms on 20,000 x 8 with 256 bins, against {1e3 * t_base:.2f} ms "
        "for ONE bincount of one feature. use np.bincount (one call per feature, or one call on codes offset by "
        "feature * n_bins), not a loop over bins with boolean masks.")


# ---------------------------------------------------------------- stage 3: one tree


def test_stage3_equals_cart_without_regularization(nb):
    for seed in range(3):
        X, y, _ = _reg_data(seed, n=500)
        codes = np.asarray(nb.bin_data(X, nb.quantile_bin_edges(X, 32)))
        r = y - y.mean()
        tree = _fit(nb.HistGradientTree(max_depth=3, min_child_weight=1.0, reg_lambda=0.0, n_bins=32),
                    codes, -r, np.ones(len(y)))
        cart = SkTree(max_depth=3).fit(codes, r)
        assert np.allclose(tree.predict(codes), cart.predict(codes)), (
            "with reg_lambda=0, gamma=0 and h = 1, the gain is half the drop in squared error and the leaf value "
            "-G/H is the mean of -g, so one tree must equal a CART regression tree (depth 3) on the bin codes. "
            "check the gain, that left = code <= bin, and the leaf values.")


def test_stage3_regularization(nb):
    X, y, _ = _reg_data(5)
    codes = np.asarray(nb.bin_data(X, nb.quantile_bin_edges(X, 64)))
    g, h = -(y - y.mean()), np.ones(len(y))
    stump = _fit(nb.HistGradientTree(max_depth=0, reg_lambda=10.0, n_bins=64), codes, g, h)
    assert np.allclose(stump.predict(codes), -g.sum() / (h.sum() + 10.0)), "max_depth=0: one leaf, -G / (H + lambda)"
    flat = _fit(nb.HistGradientTree(max_depth=4, gamma=1e6, n_bins=64), codes, g, h)
    assert len(np.unique(flat.predict(codes))) == 1, "a huge gamma makes every split's gain negative: one leaf"
    t = _fit(nb.HistGradientTree(max_depth=6, min_child_weight=25.0, reg_lambda=1.0, n_bins=64), codes, g, h)
    leaf_of = t.predict(codes)
    sizes = [np.sum(leaf_of == v) for v in np.unique(leaf_of)]
    assert min(sizes) >= 25, f"min_child_weight=25 with h = 1 means every leaf holds >= 25 rows, got {min(sizes)}"
    for depth in (1, 2, 3):
        td = _fit(nb.HistGradientTree(max_depth=depth, n_bins=64), codes, g, h)
        n_leaves = int(np.sum(np.asarray(td.children_left_) < 0))
        assert n_leaves <= 2 ** depth and (depth > 1 or n_leaves == 2), (
            f"max_depth={depth} allows at most {2 ** depth} leaves (children_left_ == -1), got {n_leaves}: the root "
            "has depth 0, and a node at depth max_depth is a leaf")
    shrunk = _fit(nb.HistGradientTree(max_depth=2, reg_lambda=50.0, n_bins=64), codes, g, h).predict(codes)
    plain = _fit(nb.HistGradientTree(max_depth=2, reg_lambda=0.0, n_bins=64), codes, g, h).predict(codes)
    assert np.abs(shrunk).max() < np.abs(plain).max(), "a larger lambda shrinks the leaf values toward 0"


# ---------------------------------------------------------------- stage 4: regressor


def test_stage4_one_tree_and_staging(nb):
    X, y, _ = _reg_data(6)
    m = _fit(nb.GradientBoostingRegressor(n_estimators=1, learning_rate=1.0, max_depth=3, reg_lambda=0.0,
                                          n_bins=32), X, y)
    codes = np.asarray(nb.bin_data(X, nb.quantile_bin_edges(X, 32)))
    assert np.allclose(m.predict(X), SkTree(max_depth=3).fit(codes, y).predict(codes)), (
        "one tree with learning_rate=1 and no regularization, started at the mean: the prediction is the mean "
        "plus a least-squares tree on the residuals, i.e. a CART tree on the bin codes")
    assert np.isclose(m.init_, y.mean()), "init_ is the mean of y: the best constant under squared loss"
    m = _fit(nb.GradientBoostingRegressor(n_estimators=15, learning_rate=0.3, rng=0), X, y)
    assert np.allclose(m.predict(X[:40]), m.predict(X)[:40]), (
        "predicting 40 rows must give the same values as predicting all rows: bin new data with the TRAINING "
        "bin_edges_, never with edges computed from the data you're predicting")
    staged = [np.asarray(p) for p in m.staged_predict(X)]  # no copies: each stage must be a new array
    assert len(staged) == 15 and np.allclose(staged[-1], m.predict(X)), (
        "staged_predict yields one prediction per tree; the last equals predict")
    assert not np.allclose(staged[0], staged[-1]), (
        "the first stage (one tree) must differ from the last: yield a NEW array each time (F = F + ..., not "
        "F += ...), or every stored stage ends up equal to the final prediction")
    losses = [np.mean((p - y) ** 2) for p in staged]
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])), (
        f"with subsample=1 every tree lowers the training MSE (a shrunken Newton step on a quadratic), got "
        f"{np.round(losses[:5], 4).tolist()}... use g = F - y (prediction minus target), not y - F")
    Xn = np.vstack([X[:3], X[:3] * 100])
    assert np.all(np.isfinite(m.predict(Xn))), "points far outside the training range land in the edge bins"


def test_stage4_subsampling(nb):
    rows = np.asarray(nb.subsample_rows(100, 0.3, rng=0))
    assert len(rows) == 30 and len(np.unique(rows)) == 30 and np.all(np.diff(rows) > 0), (
        "subsample_rows(100, 0.3): 30 distinct sorted row indices (WITHOUT replacement)")
    assert np.array_equal(nb.subsample_rows(10, 1.0, rng=0), np.arange(10)), "fraction 1.0 -> every row"
    X, y, _ = _reg_data(7)
    a = _fit(nb.GradientBoostingRegressor(n_estimators=10, subsample=0.5, rng=3), X, y).predict(X)
    b = _fit(nb.GradientBoostingRegressor(n_estimators=10, subsample=0.5, rng=3), X, y).predict(X)
    c = _fit(nb.GradientBoostingRegressor(n_estimators=10, subsample=0.5, rng=4), X, y).predict(X)
    assert np.allclose(a, b) and not np.allclose(a, c), "subsampling uses the model's rng: same seed, same model"


def test_stage4_subsampling_updates_every_row(nb):
    X, y, _ = _reg_data(12, n=300)
    m = _fit(nb.GradientBoostingRegressor(n_estimators=6, learning_rate=0.5, max_depth=2, subsample=0.5, n_bins=32,
                                          rng=5), X, y)
    # rebuild the fit by hand from the spec: one generator, used only by subsample_rows, one draw per tree
    gen = np.random.default_rng(5)
    codes = np.asarray(nb.bin_data(X, nb.quantile_bin_edges(X, 32)))
    F = np.full(len(y), y.mean())
    for tree in m.estimators_:
        rows = np.asarray(nb.subsample_rows(len(y), 0.5, gen))
        want = nb.HistGradientTree(2, 1.0, 1.0, 0.0, 32).fit(codes[rows], (F - y)[rows], np.ones(len(rows)))
        assert np.allclose(tree.predict(codes), want.predict(codes)), (
            "each tree must be fitted on the subsampled rows' gradients g = F - y, where F has been updated for "
            "EVERY row by all previous trees (the subsample only decides which rows fit the next tree). updating "
            "only the subsampled rows' F leaves stale gradients behind.")
        F = F + 0.5 * want.predict(codes)
    assert np.allclose(m.predict(X), F), "the final predictions are init_ + learning_rate * the sum of all trees"


def test_stage4_matches_sklearn_exactly(nb):
    with threadpool_limits(1):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            n, d = int(rng.integers(150, 400)), int(rng.integers(1, 4))
            X = np.round(rng.normal(size=(n, d)) * 20) / 20  # < 256 distinct values: both bin exactly
            y = 2 * X[:, 0] + np.sin(3 * X[:, -1]) + 0.3 * rng.normal(size=n)
            # compared on training rows: where a node's rows leave a gap of empty bins, every threshold in the gap
            # gives the same partition (a tie), and the two libraries may place it differently, which only shows
            # on feature combinations the training set doesn't contain
            Xt = X
            lam, depth, lr = [0.0, 1.0, 5.0][seed], 1 + seed, [1.0, 0.5, 0.1][seed]
            ours = _fit(nb.GradientBoostingRegressor(n_estimators=15, learning_rate=lr, max_depth=depth,
                                                     reg_lambda=lam), X, y)
            sk = SkHGBR(max_iter=15, learning_rate=lr, max_depth=depth, max_leaf_nodes=None, min_samples_leaf=1,
                        l2_regularization=lam, early_stopping=False, max_bins=255).fit(X, y)
            gap = np.max(np.abs(ours.predict(Xt) - sk.predict(Xt))) / np.std(y)
            assert gap < 1e-6, (
                f"on features with fewer than 256 distinct values both boosters bin exactly, and with depth-limited "
                f"trees and min_samples_leaf=1 scikit-learn's HistGradientBoostingRegressor computes the same model "
                f"(up to its float32 gradients); yours differs by {gap:.2e} standard deviations "
                f"(lambda={lam}, depth={depth}, learning rate {lr}). check the gain, the leaf values and the shrinkage.")


def test_stage4_matches_sklearn_quality(nb):
    ours, theirs = [], []
    for seed in range(3):
        X, y, _ = _reg_data(10 + seed, n=600)
        Xt, yt, _ = _reg_data(100 + seed, n=2000)
        ours.append(_fit(nb.GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3), X, y)
                    .score(Xt, yt))
        with threadpool_limits(1):  # the oracle's OpenMP threads only fight each other on small data
            theirs.append(SkHGBR(max_iter=150, learning_rate=0.1, max_depth=3, min_samples_leaf=1,
                                 l2_regularization=1.0, early_stopping=False).fit(X, y).score(Xt, yt))
    assert np.median(ours) > np.median(theirs) - 0.03, (
        f"test R^2: yours {np.round(ours, 3).tolist()}, scikit-learn's HistGradientBoosting "
        f"{np.round(theirs, 3).tolist()} with the same depth, rate and trees. they should agree within noise.")


# ---------------------------------------------------------------- stage 5: classifier


def test_stage5_newton_leaves(nb):
    X, y, _ = _clf_data(8)
    labels = np.array(["neg", "pos"])[y]
    m = _fit(nb.GradientBoostingClassifier(n_estimators=1, learning_rate=1.0, max_depth=1, reg_lambda=2.0,
                                           n_bins=32), X, labels)
    p0 = y.mean()
    assert list(m.classes_) == ["neg", "pos"] and np.isclose(m.init_, np.log(p0 / (1 - p0))), (
        "init_ is the log-odds of the training rate of classes_[1]: the best constant under log loss")
    codes = np.asarray(nb.bin_data(X, m.bin_edges_))
    tree = m.estimators_[0]
    leaf = tree.predict(codes)
    g, h = p0 - y, np.full(len(y), p0 * (1 - p0))
    for v in np.unique(leaf):
        rows = leaf == v
        assert np.isclose(v, -g[rows].sum() / (h[rows].sum() + 2.0)), (
            "each leaf is the regularized Newton step -sum(g) / (sum(h) + lambda) with g = p - y and "
            "h = p (1 - p), p = sigmoid(F) = the current probability")
    f = np.asarray(m.decision_function(X))
    assert np.allclose(f, m.init_ + leaf), "decision_function = init_ + learning_rate * sum of the trees"
    p = np.asarray(m.predict_proba(X))
    assert p.shape == (len(y), 2) and np.allclose(p[:, 1], 1 / (1 + np.exp(-f))), "P(classes_[1]) = sigmoid(F)"


def test_stage5_matches_sklearn_exactly(nb):
    with threadpool_limits(1):
        for seed in range(2):
            rng = np.random.default_rng(10 + seed)
            n, d = 400, 2
            X = np.round(rng.normal(size=(n, d)) * 20) / 20
            t = (rng.random(n) < 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, -1] ** 2 + 0.5)))).astype(int)
            Xt = X  # training rows (see the regressor's check). depth 2: in the first round g takes only two values,
            # so deeper, smaller nodes have exact ties between splits, which the libraries may break differently
            ours = _fit(nb.GradientBoostingClassifier(n_estimators=15, learning_rate=0.5, max_depth=2,
                                                      reg_lambda=1.0, min_child_weight=1e-3), X, t)
            sk = SkHGBC(max_iter=15, learning_rate=0.5, max_depth=2, max_leaf_nodes=None, min_samples_leaf=1,
                        l2_regularization=1.0, early_stopping=False, max_bins=255).fit(X, t)
            gap = np.max(np.abs(ours.decision_function(Xt) - sk.decision_function(Xt)))
            assert gap < 1e-5, (
                f"with min_child_weight=1e-3 (scikit-learn's min_hessian_to_split) the log-odds should match "
                f"HistGradientBoostingClassifier's up to float32 rounding; the largest difference is {gap:.2e}. check "
                "init_ (the log-odds of the class-1 rate), g = p - t, h = p (1 - p) and the leaf values.")


def test_stage5_training_and_quality(nb):
    X, y, _ = _clf_data(9, n=600)
    m = _fit(nb.GradientBoostingClassifier(n_estimators=40, learning_rate=0.2, max_depth=2), X, y)
    staged = [np.asarray(f) for f in m.staged_decision_function(X)]  # each stage must be a new array
    assert len(staged) == 40 and np.allclose(staged[-1], m.decision_function(X)), "one staged score per tree"
    ll = [np.mean(np.log1p(np.exp(-(2 * y - 1) * f))) for f in staged]
    assert ll[-1] < ll[0] < np.log(2) and ll[-1] < 0.9 * ll[0], (
        f"training log loss should fall steadily: {ll[0]:.4f} after 1 tree, {ll[-1]:.4f} after 40")
    ours, theirs = [], []
    for seed in range(3):
        X, y, _ = _clf_data(20 + seed, n=600)
        Xt, yt, _ = _clf_data(200 + seed, n=3000)
        ours.append(_fit(nb.GradientBoostingClassifier(n_estimators=100, max_depth=3), X, y).score(Xt, yt))
        with threadpool_limits(1):
            theirs.append(SkHGBC(max_iter=100, max_depth=3, min_samples_leaf=1, l2_regularization=1.0,
                                 early_stopping=False).fit(X, y).score(Xt, yt))
    assert np.median(ours) > np.median(theirs) - 0.02, (
        f"test accuracy: yours {np.round(ours, 3).tolist()}, scikit-learn's {np.round(theirs, 3).tolist()}")
    try:
        nb.GradientBoostingClassifier(n_estimators=1).fit(X, np.arange(len(X)) % 3)
    except ValueError:
        pass
    else:
        raise AssertionError("the classifier is binary: three classes must raise ValueError")
