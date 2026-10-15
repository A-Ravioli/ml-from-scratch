"""Checks for exercise.py (decision trees).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Impurities are
checked by hand; `best_split` against a brute-force loop over every threshold; whole trees
against scikit-learn's CART (an oracle only: trees agree exactly when no two splits tie, so the
oracle tests use continuous sample weights and leaves of at least 10-20 rows, where ties don't
happen). Every fit runs under mlfs.testing.bounded, so a loop that never ends fails instead of
hanging.
"""
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SkClassifier
from sklearn.tree import DecisionTreeRegressor as SkRegressor

from mlfs.testing import bounded

_HINT = ("every node must be popped from the stack exactly once, and a node that isn't split must "
         "not be pushed again.")


def _fit(est, X, y, sample_weight=None, budget=3_000_000):
    return bounded(est.fit, X, y, sample_weight, budget=budget, what=f"{type(est).__name__}.fit", hint=_HINT)


def _class_data(seed, n=400, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] ** 2 + rng.normal(size=n) > 1).astype(int) + (X[:, 2] > 1)
    return X, y, rng


def _reg_data(seed, n=400, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 3 * X[:, 0] + np.sin(2 * X[:, 1]) + rng.normal(size=n)
    return X, y, rng


def _brute_force(X, stats, w, impurity, msl):
    """the best decrease by looping over every feature and every midpoint (slow and obvious)."""
    best = -np.inf
    total = stats.sum(0)
    parent = w.sum() * impurity(total)
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for t in (vals[:-1] + vals[1:]) / 2:
            m = X[:, j] <= t
            if m.sum() < msl or (~m).sum() < msl:
                continue
            dec = parent - w[m].sum() * impurity(stats[m].sum(0)) - w[~m].sum() * impurity(stats[~m].sum(0))
            best = max(best, dec)
    return best


# ---------------------------------------------------------------- stage 1: impurity


def test_stage1_gini_by_hand(nb):
    got = np.asarray(nb.gini(np.array([[3.0, 1.0], [2.0, 2.0], [4.0, 0.0], [0.0, 0.0]])))
    want = np.array([0.375, 0.5, 0.0, 0.0])
    assert got.shape == (4,) and np.allclose(got, want), (
        f"gini of class counts (3, 1), (2, 2), (4, 0), (0, 0) should be {want.tolist()}, got {np.round(got, 4).tolist()}. "
        "gini = 1 - sum_k p_k^2 with p = counts / counts.sum(); an empty node (total 0) has impurity 0, not nan.")
    w = np.asarray(nb.gini(np.array([1.5, 0.5])))
    assert np.isclose(w, 0.375), "counts may be weighted (non-integer): (1.5, 0.5) has the same proportions as (3, 1)"


def test_stage1_entropy_by_hand(nb):
    got = np.asarray(nb.entropy(np.array([[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [5.0, 0.0, 0.0, 0.0],
                                          [3.0, 1.0, 0.0, 0.0]])))
    want = np.array([1.0, 2.0, 0.0, -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))])
    assert got.shape == (4,) and np.all(np.isfinite(got)), (
        f"entropy must be finite for counts with zeros (0 log 0 = 0), got {got}")
    assert np.allclose(got, want), (
        f"entropy in bits of (2,2), (1,1,1,1), (5,0), (3,1) should be {np.round(want, 4).tolist()}, "
        f"got {np.round(got, 4).tolist()}. use log base 2, and skip the zero proportions.")


def test_stage1_squared_error(nb):
    y = np.array([1.0, 2.0, 3.0, 6.0])
    got = float(nb.squared_error(np.array([4.0, y.sum(), (y ** 2).sum()])))
    assert np.isclose(got, 3.5), (
        f"sums (4, 12, 50) for y = (1, 2, 3, 6) give a variance of 50/4 - 3^2 = 3.5, got {got}. "
        "divide by the total weight, not by n - 1.")
    rng = np.random.default_rng(0)
    y, w = rng.normal(size=(3, 50)), rng.uniform(0.1, 2.0, size=(3, 50))
    sums = np.stack([w.sum(1), (w * y).sum(1), (w * y * y).sum(1)], axis=-1)
    want = np.array([np.average((y[i] - np.average(y[i], weights=w[i])) ** 2, weights=w[i]) for i in range(3)])
    assert np.allclose(nb.squared_error(sums), want), "weighted variance: sum w (y - ybar_w)^2 / sum w, row by row"
    const = np.array([3.0, 3 * 0.7, 3 * 0.7 ** 2])  # three rows of y = 0.7: the formula leaves +1.7e-16
    assert float(nb.squared_error(const)) == 0.0, (
        "a constant target must have impurity exactly 0, but sum(w y^2)/W - (sum(w y)/W)^2 leaves a tiny rounding "
        "residue: clip values below 1e-12 * mean square to 0 (otherwise trees keep splitting pure nodes).")
    assert float(nb.squared_error(np.zeros(3))) == 0.0, "an empty node has impurity 0"


def test_stage1_vectorized_over_leading_axes(nb):
    counts = np.random.default_rng(1).integers(0, 5, size=(2, 3, 4)).astype(float)
    g, e = np.asarray(nb.gini(counts)), np.asarray(nb.entropy(counts))
    assert g.shape == (2, 3) and e.shape == (2, 3), (
        f"impurities reduce only the LAST axis: counts (2, 3, 4) -> (2, 3), got {g.shape} and {e.shape}. "
        "best_split will call them on (n-1, d, k) arrays of cumulative counts.")
    for i in range(2):
        for j in range(3):
            assert np.isclose(g[i, j], nb.gini(counts[i, j])) and np.isclose(e[i, j], nb.entropy(counts[i, j]))


# ---------------------------------------------------------------- stage 2: best split


def test_stage2_worked_example(nb):
    X = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 1.0], [4.0, 4.0], [5.0, 2.0], [6.0, 3.0]])
    y = np.array([0, 0, 1, 0, 1, 1])
    stats = np.eye(2)[y]
    f, t, dec = nb.best_split(X, stats, np.ones(6), nb.gini)
    assert np.isclose(dec, 3.0), (
        f"the lesson's worked example: the best weighted Gini decrease is 3.0 (feature 1 at 3.5: (1, 2, 3) are "
        f"all class 1 vs (4, 5, 6) all class 0, so 6 * 0.5 - 0 - 0), got {dec}. "
        "decrease = W imp(node) - W_L imp(left) - W_R imp(right), with W the summed weights (not fractions).")
    assert (f, t) == (1, 3.5), f"expected feature 1, threshold 3.5 (the midpoint of 3 and 4), got ({f}, {t})"
    f, t, dec = nb.best_split(X[:, :1], stats, np.ones(6), nb.gini)
    assert (f, t) == (0, 2.5) and np.isclose(dec, 1.5), (
        f"on feature 0 alone, thresholds 2.5 and 4.5 tie at a decrease of 1.5: ties go to the LOWEST threshold "
        f"(and then the lowest feature), got ({f}, {t}, {dec})")


def test_stage2_matches_brute_force(nb):
    rng = np.random.default_rng(2)
    for trial in range(4):
        X = np.round(rng.normal(size=(60, 3)), 1)  # rounding creates repeated values
        w = rng.uniform(0.2, 2.0, size=60)
        y = rng.integers(0, 3, size=60)
        stats = np.eye(3)[y] * w[:, None]
        for imp, msl in [(nb.gini, 1), (nb.entropy, 5)]:
            f, t, dec = nb.best_split(X, stats, w, imp, min_samples_leaf=msl)
            want = _brute_force(X, stats, w, imp, msl)
            assert np.isclose(dec, want), (
                f"best_split found a decrease of {dec:.5f}, but looping over every midpoint of every feature finds "
                f"{want:.5f} (impurity {imp.__name__}, min_samples_leaf={msl}). check the cumulative sums: the left "
                "child after sorted position i holds rows 0..i.")
            m = X[:, f] <= t
            assert m.sum() >= msl and (~m).sum() >= msl, f"the split leaves fewer than min_samples_leaf={msl} rows"
            redo = w.sum() * imp(stats.sum(0)) - w[m].sum() * imp(stats[m].sum(0)) - w[~m].sum() * imp(stats[~m].sum(0))
            assert np.isclose(redo, dec), "the returned (feature, threshold) must actually achieve the returned decrease"
            vals = np.unique(X[:, f])
            assert np.any(np.isclose(t, (vals[:-1] + vals[1:]) / 2)), (
                f"threshold {t} should be the midpoint between two consecutive DISTINCT values of feature {f}; "
                "never cut between two equal values (they can't be separated)")
        yr = rng.normal(size=60)
        s = np.column_stack([w, w * yr, w * yr * yr])
        f, t, dec = nb.best_split(X, s, w, nb.squared_error, min_samples_leaf=3)
        assert np.isclose(dec, _brute_force(X, s, w, nb.squared_error, 3)), "squared_error splits: same rule"


def test_stage2_ties_go_to_the_lowest_feature(nb):
    y = np.array([0, 0, 1, 1, 1, 1])
    a = np.arange(1.0, 7.0)          # perfect split after sorted position 1 (threshold 2.5)
    b = 7.0 - a                      # the same partition, but after sorted position 3 (threshold 4.5)
    f, t, dec = nb.best_split(np.column_stack([b, a]), np.eye(2)[y], np.ones(6), nb.gini)
    assert (f, t) == (0, 4.5) and np.isclose(dec, 6 * 4 / 9), (
        f"both features split the classes perfectly (decrease 2.667): ties go to the LOWEST FEATURE first, then the "
        f"lowest threshold, so expected (0, 4.5), got ({f}, {t}). lay the candidates out feature by feature "
        "(argmax of decrease.T), not position by position.")


def test_stage2_no_valid_split(nb):
    X = np.ones((5, 2))
    f, t, dec = nb.best_split(X, np.eye(2)[[0, 1, 0, 1, 1]], np.ones(5), nb.gini)
    assert f == -1 and dec == -np.inf and np.isnan(t), (
        f"constant features can't be split: expected (-1, nan, -inf), got ({f}, {t}, {dec})")
    X = np.arange(6.0)[:, None]
    f, *_ = nb.best_split(X, np.eye(2)[[0, 1, 0, 1, 1, 0]], np.ones(6), nb.gini, min_samples_leaf=4)
    assert f == -1, "6 rows can't make two children of at least 4 rows each: no valid split"
    X = np.column_stack([np.arange(6.0), [0, 0, 0, 1, 1, 1]])
    f, t, _ = nb.best_split(X, np.eye(2)[[0, 0, 0, 1, 1, 1]], np.ones(6), nb.gini, features=[0])
    assert f == 0 and t == 2.5, f"features=[0] restricts the search to feature 0, got feature {f}"


def test_stage2_vectorized(nb):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 10))
    y = rng.integers(0, 2, size=3000)
    stats, w = np.eye(2)[y], np.ones(3000)

    def baseline():  # what a vectorized search does anyway: sort each column, cumulative sums
        order = np.argsort(X, axis=0)
        np.cumsum(stats[order], axis=0)

    def best(fn):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    t_base = best(baseline)
    t_yours = best(lambda: bounded(nb.best_split, X, stats, w, nb.gini, budget=200_000, what="best_split",
                                   hint="loop over features if you like, but never over thresholds"))
    assert t_yours <= 25 * t_base + 0.01, (
        f"best_split took {1e3 * t_yours:.1f} ms on 3,000 x 10, against {1e3 * t_base:.1f} ms for sorting the "
        "columns and taking cumulative sums. evaluate every threshold at once with cumulative sums instead of "
        "re-summing each candidate child in a Python loop.")


# ---------------------------------------------------------------- stage 3: classifier


def test_stage3_fit_predict_basics(nb):
    X, y, _ = _class_data(0)
    labels = np.array(["ant", "bee", "cat"])[y]
    tree = nb.DecisionTreeClassifier()
    assert _fit(tree, X, labels) is tree, "fit must return self"
    assert list(tree.classes_) == ["ant", "bee", "cat"], f"classes_ should be the sorted labels, got {tree.classes_}"
    p = tree.predict_proba(X[:7])
    assert np.shape(p) == (7, 3) and np.allclose(np.sum(p, 1), 1), "predict_proba: (n, k), rows sum to 1"
    assert np.mean(tree.predict(X) == labels) == 1.0, (
        "a fully grown tree (max_depth=None) on distinct points must fit its training data perfectly: "
        "keep splitting until every leaf is pure")
    leaves = tree.apply(X)
    assert np.all(tree.children_left_[leaves] == -1), "apply must return the ids of LEAVES (children_left_ == -1)"
    assert tree.get_n_leaves() == len(np.unique(leaves)), "every leaf of a perfectly fitted tree holds training rows"


def test_stage3_matches_sklearn(nb):
    for seed in range(3):
        X, y, rng = _class_data(seed)
        w = rng.uniform(0.5, 1.5, size=len(y))
        for kw in [dict(max_depth=2), dict(max_depth=4, min_samples_leaf=20),
                   dict(criterion="entropy", min_samples_leaf=20, min_samples_split=60)]:
            ours = _fit(nb.DecisionTreeClassifier(**kw), X, y, w)
            sk = SkClassifier(random_state=0, **kw).fit(X, y, sample_weight=w)
            assert ours.feature_[0] == sk.tree_.feature[0] and np.isclose(ours.threshold_[0], sk.tree_.threshold[0]), (
                f"with {kw} your root splits feature {ours.feature_[0]} at {ours.threshold_[0]:.4f}; CART (sklearn) "
                f"splits feature {sk.tree_.feature[0]} at {sk.tree_.threshold[0]:.4f}. the best split maximizes the "
                "WEIGHTED impurity decrease, at the midpoint between two distinct values.")
            assert ours.get_depth() <= sk.get_depth() + 1 and abs(ours.get_n_leaves() - sk.get_n_leaves()) <= 2, (
                f"with {kw} your tree has {ours.get_n_leaves()} leaves and depth {ours.get_depth()}; CART has "
                f"{sk.get_n_leaves()} and {sk.get_depth()}. check the stopping rules: depth >= max_depth, fewer than "
                "min_samples_split rows, fewer than 2 * min_samples_leaf rows, or a pure node -> leaf.")
            # on the training rows the predictions agree even where two splits tie (same partition, other feature)
            assert np.allclose(ours.predict_proba(X), sk.predict_proba(X)), (
                f"with {kw} your predicted probabilities on the training rows differ from CART's. leaf probabilities "
                "are the WEIGHTED class fractions; a row goes left when x[feature] <= threshold.")


def test_stage3_stopping_rules(nb):
    X, y, _ = _class_data(4)
    stump = _fit(nb.DecisionTreeClassifier(max_depth=0), X, y)
    assert stump.get_depth() == 0 and stump.get_n_leaves() == 1, "max_depth=0 is a single leaf"
    assert np.allclose(stump.predict_proba(X[:1]), np.bincount(y) / len(y)), "a lone leaf predicts the class fractions"
    for depth in (1, 3):
        t = _fit(nb.DecisionTreeClassifier(max_depth=depth), X, y)
        assert t.get_depth() <= depth and t.get_n_leaves() <= 2 ** depth, f"max_depth={depth} not respected"
    t = _fit(nb.DecisionTreeClassifier(min_samples_leaf=15), X, y)
    sizes = np.bincount(t.apply(X))
    assert sizes[sizes > 0].min() >= 15, f"min_samples_leaf=15, but a leaf holds only {sizes[sizes > 0].min()} rows"
    pure = _fit(nb.DecisionTreeClassifier(), X, np.zeros(len(y), int))
    assert pure.get_n_leaves() == 1, "a node with a single class is pure: don't split it"


def test_stage3_max_features_and_rng(nb):
    X, y, _ = _class_data(5)
    a = _fit(nb.DecisionTreeClassifier(max_features=1, rng=7), X, y)
    b = _fit(nb.DecisionTreeClassifier(max_features=1, rng=7), X, y)
    assert np.array_equal(a.apply(X), b.apply(X)), "the same rng seed must give the same tree"
    used = {tuple(_fit(nb.DecisionTreeClassifier(max_features=1, max_depth=1, rng=s), X, y).feature_[:1])
            for s in range(12)}
    assert len(used) > 1, (
        "with max_features=1 the root's feature is drawn at random: 12 seeds should not all split on the same "
        "feature. draw the candidate features per node with the tree's generator (as_rng(self.rng)).")
    assert np.mean(a.predict(X) == y) == 1.0, (
        "max_features=1 still grows until the leaves are pure: if the drawn features can't split a node, try the "
        "others before giving up")
    sq = _fit(nb.DecisionTreeClassifier(max_features="sqrt", max_depth=2, rng=0), X, y)
    assert sq.get_depth() <= 2
    used = set(a.feature_[a.feature_ >= 0].tolist())
    assert len(used) > 2, (
        f"with max_features=1 each NODE draws its own random feature, so a big tree uses many features; yours uses "
        f"{sorted(used)}. draw from one generator created once per fit (gen = as_rng(self.rng)), not a fresh "
        "as_rng(self.rng) per node (which repeats the same draw).")
    Xc = np.column_stack([X[:, :1], np.zeros((len(y), 4))])  # four constant columns that can never split
    c = _fit(nb.DecisionTreeClassifier(max_features=1, rng=3), Xc, y)
    assert np.mean(c.predict(Xc) == y) == 1.0, (
        "with max_features=1 and four constant columns, most draws can't split a node. CART keeps searching the "
        "other features before making it a leaf; otherwise the tree stops early and can't fit the training data")


def test_stage3_deep_tree_without_recursion_limit(nb):
    n = 1500
    X = np.arange(n, dtype=float)[:, None]
    y = np.arange(n) % 2  # alternating labels: every split can only peel off one row
    try:
        tree = _fit(nb.DecisionTreeClassifier(), X, y, budget=20_000_000)
    except RecursionError:
        raise AssertionError(
            "fitting alternating labels on a line needs a tree about 1,500 levels deep, which overflows Python's "
            "recursion limit (1,000). grow the tree with an explicit stack (a list of nodes still to split) "
            "instead of recursion.") from None
    assert tree.get_depth() > 1000 and np.all(tree.predict(X) == y), (
        f"expected a perfect fit of depth > 1000, got depth {tree.get_depth()}")


# ---------------------------------------------------------------- stage 4: regressor


def test_stage4_matches_sklearn(nb):
    for seed in range(3):
        X, y, rng = _reg_data(seed)
        w = rng.uniform(0.2, 2.0, size=len(y))
        Xt = rng.normal(size=(300, X.shape[1]))
        for kw in [dict(min_samples_leaf=10), dict(max_depth=4, min_samples_leaf=10)]:
            ours = bounded(nb.DecisionTreeRegressor(**kw).fit, X, y, w, budget=3_000_000, what="fit", hint=_HINT)
            sk = SkRegressor(random_state=0, **kw).fit(X, y, sample_weight=w)
            assert ours.get_n_leaves() == sk.get_n_leaves(), (
                f"with {kw}: {ours.get_n_leaves()} leaves, CART has {sk.get_n_leaves()}")
            assert np.allclose(ours.predict(Xt), sk.predict(Xt)), (
                f"with {kw} your predictions differ from CART's: a leaf predicts the WEIGHTED mean of its targets, "
                "and splits minimize the weighted squared error")


def test_stage4_piecewise_constant_and_step(nb):
    X, y, _ = _reg_data(6)
    t = bounded(nb.DecisionTreeRegressor(max_depth=3).fit, X, y, budget=3_000_000, what="fit")
    assert len(np.unique(np.round(t.predict(X), 10))) == t.get_n_leaves() <= 8, (
        "a regression tree predicts one constant per leaf: as many distinct values as leaves")
    x = np.linspace(0, 1, 101)[:, None]
    step = np.where(x[:, 0] > 0.305, 5.0, 2.0)
    s = bounded(nb.DecisionTreeRegressor(max_depth=1).fit, x, step, budget=1_000_000, what="fit")
    assert s.feature_[0] == 0 and 0.30 < s.threshold_[0] < 0.31 and np.allclose(s.predict(x), step), (
        "a depth-1 tree on a step function must find the step exactly")
    assert np.isclose(s.score(x, step), 1.0), "score is R^2 (inherited from Regressor)"


def test_stage4_numerically_careful(nb):
    X, y, _ = _reg_data(7, n=200)
    a = bounded(nb.DecisionTreeRegressor(min_samples_leaf=5).fit, X, y, budget=3_000_000, what="fit")
    b = bounded(nb.DecisionTreeRegressor(min_samples_leaf=5).fit, X, y + 1e6, budget=3_000_000, what="fit")
    assert a.get_n_leaves() == b.get_n_leaves() and np.allclose(b.predict(X) - 1e6, a.predict(X), atol=1e-6), (
        "adding 1e6 to every target must shift the predictions by 1e6 and change nothing else. sums of y^2 near "
        "1e12 lose the variance to rounding: build the statistics from y minus its (weighted) mean, and add the "
        "mean back in predict.")
    c = bounded(nb.DecisionTreeRegressor().fit, X, np.full(200, 0.1), budget=1_000_000, what="fit")
    assert c.get_n_leaves() == 1, "a constant target is a single leaf"


# ---------------------------------------------------------------- stage 5: pruning + importances


def test_stage5_pruning_matches_sklearn(nb):
    X, y, rng = _reg_data(8)
    w = rng.uniform(0.2, 2.0, size=len(y))
    Xt = rng.normal(size=(300, X.shape[1]))
    for alpha in (0.005, 0.02, 0.1):
        ours = bounded(nb.DecisionTreeRegressor(min_samples_leaf=10, ccp_alpha=alpha).fit, X, y, w,
                       budget=3_000_000, what="fit with ccp_alpha", hint="each pruning pass must remove a node")
        sk = SkRegressor(random_state=0, min_samples_leaf=10, ccp_alpha=alpha).fit(X, y, sample_weight=w)
        assert ours.get_n_leaves() == sk.get_n_leaves(), (
            f"ccp_alpha={alpha}: your pruned tree has {ours.get_n_leaves()} leaves, minimal cost-complexity pruning "
            f"gives {sk.get_n_leaves()}. R(t) = (W_t / W_root) * impurity(t); repeatedly collapse the internal node "
            "with the smallest (R(t) - R(T_t)) / (leaves(T_t) - 1) while that value is <= ccp_alpha.")
        assert np.allclose(ours.predict(Xt), sk.predict(Xt)), f"ccp_alpha={alpha}: predictions differ after pruning"
    Xc, yc, rng = _class_data(9)
    wc = rng.uniform(0.5, 1.5, size=len(yc))
    ours = _fit(nb.DecisionTreeClassifier(min_samples_leaf=20, ccp_alpha=0.01), Xc, yc, wc)
    sk = SkClassifier(random_state=0, min_samples_leaf=20, ccp_alpha=0.01).fit(Xc, yc, sample_weight=wc)
    assert ours.get_n_leaves() == sk.get_n_leaves(), "classification pruning: leaf counts differ from CART's"


def test_stage5_pruning_is_monotone(nb):
    X, y, _ = _class_data(10)
    leaves = [_fit(nb.DecisionTreeClassifier(ccp_alpha=a), X, y).get_n_leaves() for a in (0.0, 0.002, 0.01, 0.05)]
    assert all(a >= b for a, b in zip(leaves, leaves[1:])) and leaves[0] > leaves[-1], (
        f"larger ccp_alpha must give smaller (or equal) trees, got leaves {leaves}")
    big = _fit(nb.DecisionTreeClassifier(ccp_alpha=10.0), X, y)
    assert big.get_n_leaves() == 1 and big.get_depth() == 0, "a huge ccp_alpha prunes everything back to the root"
    assert np.all(big.apply(X) == 0), "after pruning, the nodes are renumbered: the lone root is node 0"


def test_stage5_feature_importances(nb):
    X, y, rng = _reg_data(11)
    w = rng.uniform(0.2, 2.0, size=len(y))
    ours = bounded(nb.DecisionTreeRegressor(min_samples_leaf=10).fit, X, y, w, budget=3_000_000, what="fit")
    sk = SkRegressor(random_state=0, min_samples_leaf=10).fit(X, y, sample_weight=w)
    imp = np.asarray(ours.feature_importances_)
    assert imp.shape == (5,) and np.isclose(imp.sum(), 1.0), "feature_importances_: (d,), sums to 1"
    assert np.allclose(imp, sk.feature_importances_), (
        f"importances {np.round(imp, 3)} differ from CART's {np.round(sk.feature_importances_, 3)}: each split adds "
        "W_t imp(t) - W_L imp(L) - W_R imp(R) to its feature, then normalize")
    Xc, yc, _ = _class_data(12)
    Xc = np.column_stack([Xc, np.zeros(len(yc))])  # a constant feature can never be used
    t = _fit(nb.DecisionTreeClassifier(max_depth=3), Xc, yc)
    assert np.asarray(t.feature_importances_)[-1] == 0.0, "a feature no split uses has importance 0"
    lone = _fit(nb.DecisionTreeClassifier(max_depth=0), Xc, yc)
    assert np.allclose(lone.feature_importances_, 0.0), "a single-leaf tree has all-zero importances (not nan)"
