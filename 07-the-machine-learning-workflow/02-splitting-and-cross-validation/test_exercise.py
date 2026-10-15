"""Checks for exercise.py (splitting and cross-validation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Splitters are
checked against scikit-learn where the algorithms are the same, and by their defining
properties (partitions, balance, no overlap) where they aren't.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: a mean predictor, 3-fold CV, scored by -MSE
X_CV = np.arange(6.0).reshape(-1, 1)
Y_CV = np.array([1.0, 3.0, 2.0, 6.0, 5.0, 7.0])


def _b(fn, *args, **kwargs):
    """call fn under a line budget, so a loop that never ends fails instead of hanging."""
    return bounded(fn, *args, budget=400_000, what=getattr(fn, "__name__", "the function"),
                   hint="a split generator must stop after n_splits folds; every while loop needs a step that "
                        "moves it towards its exit.", **kwargs)


def _folds(splitter, X, y=None, groups=None):
    def collect():
        out = []
        for tr, te in splitter.split(X, y, groups):
            out.append((np.asarray(tr), np.asarray(te)))
            if len(out) > 50:
                raise AssertionError(f"{type(splitter).__name__}.split yielded more than 50 folds: does it ever stop?")
        return out
    return _b(collect)


def _check_partition(folds, n, label):
    tests = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(tests), np.arange(n)), (
        f"{label}: every row must be in exactly one test fold; the test folds cover "
        f"{len(np.unique(tests))} distinct rows ({len(tests)} in total) out of {n}")
    for tr, te in folds:
        assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(n)), (
            f"{label}: train and test of one fold must together be every row, with no overlap")
        assert tr.dtype.kind == "i" and te.dtype.kind == "i", f"{label}: indices must be integer arrays"


# ---------------------------------------------------------------- stage 1: train_test_split


def test_stage1_sizes_and_alignment(nb):
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10) * 10
    out = _b(nb.train_test_split, X, y, rng=0)
    assert len(out) == 4, "two arrays in -> four out: [X_train, X_test, y_train, y_test]"
    Xtr, Xte, ytr, yte = out
    assert len(Xte) == 3 and len(Xtr) == 7, (
        f"test_size=0.25 of 10 rows is ceil(2.5) = 3 test rows; got {len(Xte)} test and {len(Xtr)} train")
    assert np.array_equal(Xtr[:, 0] * 5, ytr) and np.array_equal(Xte[:, 0] * 5, yte), (
        "the arrays must be split by the SAME rows: X_train[i] must still go with y_train[i]")
    assert np.array_equal(np.sort(np.concatenate([ytr, yte])), y), "train and test together must be every row exactly once"
    Xtr, Xte = _b(nb.train_test_split, X, test_size=4, rng=0)
    assert len(Xte) == 4, "an int test_size is a count of test rows"


def test_stage1_shuffle_and_reproducibility(nb):
    y = np.arange(100)
    a_tr, a_te = _b(nb.train_test_split, y, rng=1)
    b_tr, b_te = _b(nb.train_test_split, y, rng=1)
    c_tr, c_te = _b(nb.train_test_split, y, rng=2)
    assert np.array_equal(a_te, b_te), "the same rng seed must give the same split"
    assert not np.array_equal(a_te, c_te), "different seeds should give different splits"
    assert not np.array_equal(np.sort(a_te), np.arange(75, 100)), "shuffle=True must pick random test rows"
    tr, te = _b(nb.train_test_split, y, shuffle=False, test_size=0.2)
    assert np.array_equal(tr, np.arange(80)) and np.array_equal(te, np.arange(80, 100)), (
        "shuffle=False: the first rows train, the LAST n_test rows test (like time-ordered data)")


@pytest.mark.parametrize("seed", range(4))
def test_stage1_stratify(nb, seed):
    y = np.array(["a"] * 70 + ["b"] * 25 + ["c"] * 5)
    X = np.arange(100)
    Xtr, Xte, ytr, yte = _b(nb.train_test_split, X, y, test_size=0.2, stratify=y, rng=seed)
    counts = {c: int(np.sum(yte == c)) for c in "abc"}
    assert counts == {"a": 14, "b": 5, "c": 1}, (
        f"a stratified 20% test set of 70/25/5 rows should hold exactly 14/5/1 of each class; got {counts}")
    assert np.array_equal(y[Xte], yte), "stratify must not break the row alignment"
    y2 = np.repeat([0, 1, 2], [7, 7, 7])
    _, te = _b(nb.train_test_split, y2, test_size=0.5, stratify=y2, rng=seed)
    c2 = np.bincount(te, minlength=3)
    assert len(te) == 11 and c2.max() - c2.min() <= 1, (
        f"ceil(0.5 * 21) = 11 test rows over 3 equal classes: each gets 3 or 4 (largest remainders); got {c2.tolist()}")


@pytest.mark.parametrize("bad", [0, 0.0, 1.0, 10, 1.5])
def test_stage1_rejects_bad_sizes(nb, bad):
    with pytest.raises(ValueError):
        _b(nb.train_test_split, np.arange(10), test_size=bad)
        pytest.fail(f"test_size={bad!r} on 10 rows leaves an empty train or test set: raise ValueError")


def test_stage1_rejects_mismatched_lengths(nb):
    with pytest.raises(ValueError):
        _b(nb.train_test_split, np.arange(10), np.arange(9))
        pytest.fail("arrays of different lengths can't be split by the same rows: raise ValueError")


# ---------------------------------------------------------------- stage 2: KFold


@pytest.mark.parametrize("n,k", [(10, 3), (7, 7), (12, 4), (5, 2)])
def test_stage2_kfold_matches_sklearn(nb, n, k):
    from sklearn.model_selection import KFold as SkKFold
    ours = _folds(nb.KFold(k), np.zeros((n, 1)))
    ref = list(SkKFold(k).split(np.zeros((n, 1))))
    assert len(ours) == k, f"KFold({k}) should yield {k} folds, got {len(ours)}"
    for f, ((tr, te), (rtr, rte)) in enumerate(zip(ours, ref)):
        assert np.array_equal(te, rte) and np.array_equal(tr, rtr), (
            f"n={n}, k={k}, fold {f}: expected test rows {rte.tolist()}, got {te.tolist()}. unshuffled folds are "
            "contiguous blocks, and the first n % k blocks are one row larger")
    assert nb.KFold(k).get_n_splits() == k


def test_stage2_worked_example(nb):
    tests = [te.tolist() for _, te in _folds(nb.KFold(3), np.zeros((7, 2)))]
    assert tests == [[0, 1, 2], [3, 4], [5, 6]], f"7 rows in 3 folds: sizes 3, 2, 2; got {tests}"


@pytest.mark.parametrize("seed", range(3))
def test_stage2_shuffled_kfold(nb, seed):
    n = 23
    folds = _folds(nb.KFold(4, shuffle=True, rng=seed), np.zeros((n, 1)))
    _check_partition(folds, n, "KFold(4, shuffle=True)")
    sizes = sorted(len(te) for _, te in folds)
    assert sizes == [5, 6, 6, 6], f"23 rows in 4 folds should give fold sizes 6, 6, 6, 5; got {sizes}"
    again = _folds(nb.KFold(4, shuffle=True, rng=seed), np.zeros((n, 1)))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again)), "an int rng must give the same folds every time"
    assert all(np.all(np.diff(te) > 0) and np.all(np.diff(tr) > 0) for tr, te in folds), (
        "both index arrays must be SORTED, also when shuffle=True (np.sort the test block)")
    plain = _folds(nb.KFold(4), np.zeros((n, 1)))
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(folds, plain)), "shuffle=True should change the folds"


@pytest.mark.parametrize("k", [1, 11])
def test_stage2_kfold_rejects_bad_k(nb, k):
    with pytest.raises(ValueError):
        _folds(nb.KFold(k), np.zeros((10, 1)))
        pytest.fail(f"KFold({k}) on 10 rows is impossible (need 2 <= k <= n): raise ValueError")


# ---------------------------------------------------------------- stage 3: StratifiedKFold, GroupKFold


@pytest.mark.parametrize("seed", range(4))
def test_stage3_stratified_balance(nb, seed):
    rng = np.random.default_rng(seed)
    y = rng.choice(["cat", "dog", "eel"], size=53, p=[0.6, 0.3, 0.1])
    for shuffle in (False, True):
        folds = _folds(nb.StratifiedKFold(5, shuffle=shuffle, rng=seed), np.zeros((53, 1)), y)
        _check_partition(folds, 53, f"StratifiedKFold(shuffle={shuffle})")
        for c in np.unique(y):
            per_fold = [int(np.sum(y[te] == c)) for _, te in folds]
            assert max(per_fold) - min(per_fold) <= 1, (
                f"class {c!r} ({np.sum(y == c)} rows) should be spread evenly: its count per fold may differ by at "
                f"most 1, got {per_fold}")
        sizes = [len(te) for _, te in folds]
        assert max(sizes) - min(sizes) <= 1, f"fold sizes should differ by at most 1, got {sizes}"


def test_stage3_stratified_needs_y_and_shuffles(nb):
    with pytest.raises(ValueError):
        _folds(nb.StratifiedKFold(3), np.zeros((9, 1)))
        pytest.fail("StratifiedKFold without y should raise ValueError")
    y = np.array([0, 1] * 20)
    a = _folds(nb.StratifiedKFold(4, shuffle=True, rng=0), np.zeros((40, 1)), y)
    b = _folds(nb.StratifiedKFold(4, shuffle=True, rng=0), np.zeros((40, 1)), y)
    c = _folds(nb.StratifiedKFold(4), np.zeros((40, 1)), y)
    assert all(np.array_equal(p[1], q[1]) for p, q in zip(a, b)), "an int rng must give the same folds every time"
    assert any(not np.array_equal(p[1], q[1]) for p, q in zip(a, c)), "shuffle=True should change the folds"


@pytest.mark.parametrize("seed", range(4))
def test_stage3_group_kfold_matches_sklearn(nb, seed):
    from sklearn.model_selection import GroupKFold as SkGroupKFold
    rng = np.random.default_rng(seed)
    groups = rng.choice([f"patient{i:02d}" for i in range(12)], size=60)
    ours = _folds(nb.GroupKFold(4), np.zeros((60, 1)), None, groups)
    ref = list(SkGroupKFold(4).split(np.zeros((60, 1)), None, groups))
    _check_partition(ours, 60, "GroupKFold")
    for tr, te in ours:
        assert not set(groups[tr]) & set(groups[te]), "a group must never be in both train and test"
    for f, ((tr, te), (_, rte)) in enumerate(zip(ours, ref)):
        assert np.array_equal(np.sort(te), rte), (
            f"fold {f}: expected the test rows {rte.tolist()} (largest group first, each into the fold with the fewest "
            f"rows, ties to the lowest fold), got {np.sort(te).tolist()}")


def test_stage3_group_kfold_errors(nb):
    with pytest.raises(ValueError):
        _folds(nb.GroupKFold(3), np.zeros((6, 1)))
        pytest.fail("GroupKFold without groups should raise ValueError")
    with pytest.raises(ValueError):
        _folds(nb.GroupKFold(4), np.zeros((6, 1)), None, [1, 1, 2, 2, 3, 3])
        pytest.fail("3 groups can't fill 4 folds: raise ValueError")


# ---------------------------------------------------------------- stage 4: TimeSeriesSplit


@pytest.mark.parametrize("n,k", [(12, 3), (10, 5), (23, 4), (6, 5)])
def test_stage4_time_series_matches_sklearn(nb, n, k):
    from sklearn.model_selection import TimeSeriesSplit as SkTSS
    ours = _folds(nb.TimeSeriesSplit(k), np.zeros((n, 1)))
    ref = list(SkTSS(k).split(np.zeros((n, 1))))
    assert len(ours) == k, f"TimeSeriesSplit({k}) should yield {k} splits, got {len(ours)}"
    for (tr, te), (rtr, rte) in zip(ours, ref):
        assert np.array_equal(tr, rtr) and np.array_equal(te, rte), (
            f"n={n}, k={k}: expected train 0..{rtr[-1]} and test {rte.tolist()}, got train of length {len(tr)} "
            f"and test {te.tolist()}. test_size = n // (k + 1), and the test blocks are the LAST k * test_size rows")
        assert tr.max() < te.min(), "every training row must come before every test row"
    assert nb.TimeSeriesSplit(k).get_n_splits() == k


def test_stage4_rejects_too_many_splits(nb):
    with pytest.raises(ValueError):
        _folds(nb.TimeSeriesSplit(5), np.zeros((5, 1)))
        pytest.fail("5 splits need at least 6 rows (one block to train on first): raise ValueError")


# ---------------------------------------------------------------- stage 5: cross_validate


class _MeanRegressor:
    """a minimal estimator (no mlfs.base): predicts the training mean + shift; score = -MSE."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def get_params(self):
        return {"shift": self.shift}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.seen_ = X[:, 0].copy()
        self.mean_ = (X[:, 0].mean() if y is None else np.mean(y)) + self.shift
        self.got_y_ = y is not None
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    def score(self, X, y=None):
        target = np.asarray(X)[:, 0] if y is None else np.asarray(y)
        return -float(np.mean((target - self.predict(X)) ** 2))


def test_stage5_worked_example(nb):
    est = _MeanRegressor()
    out = _b(nb.cross_validate, est, X_CV, Y_CV, cv=3)
    scores = np.asarray(out["test_score"])
    assert np.allclose(scores, [-10.0, -4.0, -10.0]), (
        f"with 3 contiguous folds the mean predictor's test MSEs are 10, 4, 10 (see the worked example), so the "
        f"scores (-MSE) are [-10, -4, -10]; got {scores.tolist()}")
    assert not hasattr(est, "mean_"), (
        "cross_validate must never fit the estimator you pass in: fit clone(estimator) in each fold")
    assert np.allclose(_b(nb.cross_val_score, _MeanRegressor(), X_CV, Y_CV, cv=3), scores), (
        "cross_val_score is cross_validate(...)['test_score']")


def test_stage5_clones_and_never_sees_test_rows(nb):
    rng = np.random.default_rng(0)
    X = rng.permutation(40).astype(float).reshape(-1, 1)
    y = rng.normal(size=40)
    splits = list(_folds(nb.KFold(5, shuffle=True, rng=1), X))
    out = _b(nb.cross_validate, _MeanRegressor(shift=0.5), X, y, cv=splits, return_estimators=True, return_train_score=True)
    ests = out["estimators"]
    assert len(ests) == 5 and len({id(e) for e in ests}) == 5, "one freshly cloned, fitted estimator per fold"
    for e, (tr, te) in zip(ests, splits):
        assert e.shift == 0.5, "clones keep the hyperparameters"
        assert not set(e.seen_) & set(X[te, 0]), "a fold's estimator must be fitted on its TRAINING rows only"
        assert np.isclose(e.mean_, y[tr].mean() + 0.5), "fit each clone on X[train], y[train]"
    train = np.asarray(out["train_score"])
    want = [-np.mean((y[tr] - y[tr].mean() - 0.5) ** 2) for tr, _ in splits]
    assert np.allclose(train, want), "train_score is the same scoring on the training rows of each fold"


def test_stage5_scoring_groups_and_unsupervised(nb):
    calls = []

    def scorer(est, X, y):
        calls.append((len(X), None if y is None else len(y)))
        return 42.0

    out = _b(nb.cross_validate, _MeanRegressor(), X_CV, Y_CV, cv=2, scoring=scorer)
    assert np.allclose(out["test_score"], 42.0) and calls == [(3, 3), (3, 3)], (
        "a callable scoring is called as scoring(fitted_estimator, X_test, y_test) once per fold")
    groups = np.array([0, 0, 1, 1, 2, 2])
    out = _b(nb.cross_validate, _MeanRegressor(), X_CV, Y_CV, cv=nb.GroupKFold(3), groups=groups, return_estimators=True)
    for e in out["estimators"]:
        assert len(e.seen_) == 4, "pass groups on to cv.split(X, y, groups)"
    out = _b(nb.cross_validate, _MeanRegressor(), X_CV, cv=3, return_estimators=True)
    assert all(not e.got_y_ for e in out["estimators"]), "with y=None (unsupervised), fit each clone with y=None"
    assert len(out["test_score"]) == 3, "cv=3 means KFold(3)"
