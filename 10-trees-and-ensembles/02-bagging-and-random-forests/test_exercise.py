"""Checks for exercise.py (bagging & random forests).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Exact checks use
the fact that a forest without bootstrap and without feature subsampling is n copies of one CART
tree, and recompute OOB estimates from the forest's own trees and samples. Statistical checks
aggregate over several seeds and compare with scikit-learn's forest as an oracle. Under pytest,
mlfs.trees' decision trees are the reference ones. Fits run under mlfs.testing.bounded.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor as SkForest

from mlfs.testing import bounded
from mlfs.trees import DecisionTreeClassifier, DecisionTreeRegressor


def _fit(est, X, y, budget=20_000_000):
    return bounded(est.fit, X, y, budget=budget, what=f"{type(est).__name__}.fit",
                   hint="the loop over trees should run exactly n_estimators times")


def _noisy_classes(seed, n=300, d=6, flip=0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
    return X, np.where(rng.random(n) < flip, 1 - y, y), rng


def _friedman(seed, n=300, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 8))
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
    return X, y + noise * rng.normal(size=n), rng


# ---------------------------------------------------------------- stage 1: bootstrap


def test_stage1_bootstrap_sample(nb):
    idx, oob = nb.bootstrap_sample(10, rng=0)
    idx, oob = np.asarray(idx), np.asarray(oob)
    assert idx.shape == (10,) and oob.shape == (10,) and oob.dtype == bool, (
        f"expected idx (10,) ints and a bool mask (10,), got {idx.shape}, {oob.shape} {oob.dtype}")
    assert idx.min() >= 0 and idx.max() < 10, "indices must lie in 0..n-1"
    assert np.array_equal(oob, ~np.isin(np.arange(10), idx)), "oob[i] is True exactly when row i was never drawn"
    a, b = nb.bootstrap_sample(50, rng=3), nb.bootstrap_sample(50, rng=3)
    assert np.array_equal(a[0], b[0]), "the same seed must give the same sample"
    gen = np.random.default_rng(4)
    c, d = nb.bootstrap_sample(50, rng=gen), nb.bootstrap_sample(50, rng=gen)
    assert not np.array_equal(c[0], d[0]), "passing one Generator twice must continue its stream (use as_rng)"


def test_stage1_worked_example(nb):
    idx, oob = nb.bootstrap_sample(6, rng=0)
    assert np.array_equal(idx, [5, 3, 3, 1, 1, 0]) and np.array_equal(np.flatnonzero(oob), [2, 4]), (
        f"the lesson's worked example: with rng=0 and n=6, rng.integers(0, n, size=n) draws rows "
        f"[5, 3, 3, 1, 1, 0], so rows 2 and 4 are out of bag. got idx {np.asarray(idx).tolist()}, "
        f"oob rows {np.flatnonzero(oob).tolist()}")


def test_stage1_out_of_bag_fraction(nb):
    fracs = [np.mean(nb.bootstrap_sample(2000, rng=s)[1]) for s in range(20)]
    assert abs(np.mean(fracs) - np.exp(-1)) < 0.01, (
        f"about (1 - 1/n)^n ~ 1/e = 0.368 of the rows are out of bag, got {np.mean(fracs):.3f}. draw WITH "
        "replacement (rng.integers(0, n, size=n)), not a permutation.")
    counts = np.bincount(nb.bootstrap_sample(5000, rng=1)[0], minlength=5000)
    assert counts.max() >= 4, "with replacement, some rows are drawn several times"


# ---------------------------------------------------------------- stage 2: classifier


def test_stage2_basics(nb):
    X, y, _ = _noisy_classes(0)
    labels = np.array(["no", "yes"])[y]
    rf = nb.RandomForestClassifier(n_estimators=15, rng=0)
    assert _fit(rf, X, labels) is rf, "fit must return self"
    assert list(rf.classes_) == ["no", "yes"] and len(rf.estimators_) == 15, (
        "classes_ = sorted labels, and estimators_ holds n_estimators fitted trees")
    p = rf.predict_proba(X[:9])
    assert np.shape(p) == (9, 2) and np.allclose(np.sum(p, 1), 1), "predict_proba: (n, k), rows sum to 1"
    assert all(t.max_features == "sqrt" for t in rf.estimators_), "pass max_features on to every tree"
    assert len(rf.estimators_samples_) == 15 and np.shape(rf.estimators_samples_[0]) == (len(y),), (
        "keep each tree's bootstrap indices (n,) in estimators_samples_")
    again = _fit(nb.RandomForestClassifier(n_estimators=15, rng=0), X, labels)
    assert np.allclose(again.predict_proba(X), rf.predict_proba(X)), "the same rng seed must give the same forest"


def test_stage2_no_randomness_means_one_tree(nb):
    X, y, _ = _noisy_classes(1)
    rf = _fit(nb.RandomForestClassifier(n_estimators=4, bootstrap=False, max_features=None, max_depth=3, rng=0), X, y)
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    Xt = np.random.default_rng(9).normal(size=(200, X.shape[1]))
    assert np.allclose(rf.predict_proba(Xt), tree.predict_proba(Xt)), (
        "without bootstrap and with max_features=None every tree sees the same data and every feature, so the "
        "forest must predict exactly like one tree. check that each tree is fitted on X[idx], y[idx], and that "
        "predict_proba AVERAGES the trees' probabilities.")
    assert all(np.array_equal(s, np.arange(len(y))) for s in rf.estimators_samples_), (
        "with bootstrap=False each tree's sample is every row once")
    rf1 = _fit(nb.RandomForestClassifier(n_estimators=6, bootstrap=False, max_features=1, rng=0), X, y)
    roots = {(int(t.feature_[0]), float(t.threshold_[0])) for t in rf1.estimators_}
    assert len(roots) > 1, (
        "with max_features=1 and no bootstrap, only the per-split feature draws make trees differ, yet all 6 trees "
        "have the same root: give every tree the forest's generator (rng=gen), not the same int seed")


def test_stage2_missing_class_in_bootstrap(nb):
    X, y, _ = _noisy_classes(2, n=200)
    y = y.copy()
    y[:2] = -1  # a rare class, and the FIRST in classes_: many bootstrap samples miss it
    rf = _fit(nb.RandomForestClassifier(n_estimators=25, rng=1), X, y)
    p = rf.predict_proba(X)
    assert np.shape(p) == (200, 3) and np.allclose(np.sum(p, 1), 1), (
        "a tree fitted on a bootstrap sample that misses a class only knows the classes it saw: place its "
        "columns into the forest's (n, k) array by class, don't stack them blindly")
    want = np.mean([np.column_stack([np.zeros(200), t.predict_proba(X)]) if len(t.classes_) == 2
                    else t.predict_proba(X) for t in rf.estimators_], axis=0)
    assert np.allclose(p, want), (
        "trees that never saw the rare class (column 0) must contribute 0 to column 0 and their own probabilities "
        "to the columns of the classes they did see")
    assert np.all(p[:2, 0] > 0), "the rare class's own rows should get some probability from the trees that saw them"


def test_stage2_beats_a_single_tree(nb):
    wins, gaps = 0, []
    for seed in range(5):
        X, y, rng = _noisy_classes(10 + seed, n=250)
        Xt, yt, _ = _noisy_classes(100 + seed, n=2000)
        tree = DecisionTreeClassifier().fit(X, y).score(Xt, yt)
        forest = _fit(nb.RandomForestClassifier(n_estimators=20, min_samples_leaf=2, rng=seed), X, y).score(Xt, yt)
        wins += forest > tree
        gaps.append(forest - tree)
    assert wins >= 4 and np.median(gaps) > 0.03, (
        f"a forest of 20 trees should beat one fully grown tree on noisy data (by ~6 points here); test accuracy "
        f"gains over 5 seeds: {np.round(gaps, 3).tolist()}. bootstrap the rows AND use max_features per split.")


# ---------------------------------------------------------------- stage 3: out of bag


def test_stage3_oob_recomputed(nb):
    X, y, _ = _noisy_classes(3)
    rf = _fit(nb.RandomForestClassifier(n_estimators=3, min_samples_leaf=5, oob_score=True, rng=2), X, y)  # few trees: some rows never OOB
    n = len(y)
    total, count = np.zeros((n, 2)), np.zeros(n)
    for tree, idx in zip(rf.estimators_, rf.estimators_samples_):
        oob = ~np.isin(np.arange(n), idx)
        p = np.zeros((oob.sum(), 2))
        p[:, tree.classes_.astype(int)] = tree.predict_proba(X[oob])
        total[oob] += p
        count[oob] += 1
    seen = count > 0
    got = np.asarray(rf.oob_decision_function_)
    assert got.shape == (n, 2), f"oob_decision_function_ should be (n, k), got {got.shape}"
    assert np.allclose(got[seen], total[seen] / count[seen, None]), (
        "row i's OOB probabilities average ONLY the trees whose bootstrap sample didn't contain i")
    assert np.all(np.isnan(got[~seen])), "rows that were in every sample have no OOB estimate: nan"
    want = np.mean(np.argmax(got[seen], 1) == y[seen])
    assert np.isclose(rf.oob_score_, want), "oob_score_ = accuracy of the OOB predictions, over rows that have one"


def test_stage3_oob_tracks_test_accuracy(nb):
    diffs, train_gap = [], []
    for seed in range(5):
        X, y, _ = _noisy_classes(20 + seed, n=250)
        Xt, yt, _ = _noisy_classes(200 + seed, n=3000)
        rf = _fit(nb.RandomForestClassifier(n_estimators=40, min_samples_leaf=2, oob_score=True, rng=seed), X, y)
        diffs.append(rf.oob_score_ - rf.score(Xt, yt))
        train_gap.append(rf.score(X, y) - rf.oob_score_)
    assert -0.08 < np.median(diffs) < 0.04, (
        f"the OOB accuracy should be close to the accuracy on fresh data (a little lower: each OOB prediction "
        f"uses only ~37% of the trees); OOB - test over 5 seeds: {np.round(diffs, 3).tolist()}")
    assert np.median(train_gap) > 0.1, (
        "the OOB accuracy must be much lower than the training accuracy: if they're equal, you're scoring trees on "
        "rows they were trained on")


def test_stage3_oob_requires_bootstrap(nb):
    X, y, _ = _noisy_classes(4, n=50)
    try:
        nb.RandomForestClassifier(n_estimators=2, bootstrap=False, oob_score=True).fit(X, y)
    except ValueError:
        return
    raise AssertionError("oob_score=True with bootstrap=False must raise ValueError: no row is ever out of bag")


# ---------------------------------------------------------------- stage 4: regressor + importances


def test_stage4_regressor_exact_and_oob(nb):
    X, y, _ = _friedman(5)
    rf = _fit(nb.RandomForestRegressor(n_estimators=3, bootstrap=False, max_features=1.0, rng=0), X, y)
    tree = DecisionTreeRegressor().fit(X, y)
    Xt = np.random.default_rng(1).uniform(size=(100, 8))
    assert np.allclose(rf.predict(Xt), tree.predict(Xt)), (
        "without bootstrap and with max_features=1.0 the regression forest is copies of one tree")
    rf = _fit(nb.RandomForestRegressor(n_estimators=3, oob_score=True, rng=1), X, y)  # few trees: some rows never OOB
    n = len(y)
    total, count = np.zeros(n), np.zeros(n)
    for tree, idx in zip(rf.estimators_, rf.estimators_samples_):
        oob = ~np.isin(np.arange(n), idx)
        total[oob] += tree.predict(X[oob])
        count[oob] += 1
    seen = count > 0
    assert np.allclose(np.asarray(rf.oob_prediction_)[seen], total[seen] / count[seen]), (
        "oob_prediction_[i] averages the predictions of the trees that didn't see row i")
    r2 = 1 - np.sum((y[seen] - total[seen] / count[seen]) ** 2) / np.sum((y[seen] - y[seen].mean()) ** 2)
    assert np.isclose(rf.oob_score_, r2), "oob_score_ is the R^2 of the OOB predictions (rows that have one)"


def test_stage4_regressor_matches_sklearn_quality(nb):
    ours, theirs = [], []
    for seed in range(3):
        X, y, _ = _friedman(30 + seed, n=250)
        Xt, yt, _ = _friedman(300 + seed, n=2000)
        kw = dict(n_estimators=25, max_features=0.5, min_samples_leaf=3)
        ours.append(_fit(nb.RandomForestRegressor(rng=seed, **kw), X, y).score(Xt, yt))
        theirs.append(SkForest(random_state=seed, **kw).fit(X, y).score(Xt, yt))
    assert np.median(ours) > np.median(theirs) - 0.03, (
        f"test R^2 of your forest {np.round(ours, 3).tolist()} vs scikit-learn's {np.round(theirs, 3).tolist()} "
        "with the same settings: they should be within noise of each other")


def test_stage4_feature_importances(nb):
    X, y, _ = _friedman(6)
    rf = _fit(nb.RandomForestRegressor(n_estimators=15, min_samples_leaf=3, rng=0), X, y)
    imp = np.asarray(rf.feature_importances_)
    want = np.mean([t.feature_importances_ for t in rf.estimators_], axis=0)
    assert imp.shape == (8,) and np.isclose(imp.sum(), 1) and np.allclose(imp, want / want.sum()), (
        "the forest's importances are the mean of its trees' importances, renormalized to sum 1")
    assert set(np.argsort(imp)[-4:]) >= {0, 1, 3}, (
        f"features 0, 1 and 3 drive the target; columns 5-7 are pure noise. got importances {np.round(imp, 3)}")
    Xc, yc, _ = _noisy_classes(7)
    rc = _fit(nb.RandomForestClassifier(n_estimators=10, rng=0), Xc, yc)
    assert np.isclose(np.sum(rc.feature_importances_), 1), "the classifier has feature_importances_ too"
