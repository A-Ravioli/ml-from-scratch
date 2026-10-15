"""Checks for exercise.py (hyperparameter search).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The estimators
below implement just enough of the protocol for clone and cross_validate, and record what
they were fitted on.
"""
import numpy as np
import pytest

from mlfs.testing import bounded


class _Shift:
    """predicts mean(y_train) + shift; `unused` does nothing. score = -MSE."""

    def __init__(self, shift=0.0, unused=0):
        self.shift = shift
        self.unused = unused

    def get_params(self):
        return {"shift": self.shift, "unused": self.unused}

    def set_params(self, **p):
        for k, v in p.items():
            if k not in ("shift", "unused"):
                raise ValueError(k)
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self.rows_ = np.asarray(X)[:, 0].copy()
        self.mean_ = float(np.mean(y)) + self.shift
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    def score(self, X, y):
        return -float(np.mean((np.asarray(y) - self.predict(X)) ** 2))


def _data(seed=0, n=24):
    rng = np.random.default_rng(seed)
    return np.arange(float(n)).reshape(-1, 1), 3.0 + rng.normal(size=n)


def _manual_cv(shift, X, y, k):
    out = []
    for f in range(k):
        te = np.arange(len(y))[f * len(y) // k:(f + 1) * len(y) // k]
        tr = np.setdiff1d(np.arange(len(y)), te)
        out.append(-np.mean((y[te] - y[tr].mean() - shift) ** 2))
    return np.array(out)


# ---------------------------------------------------------------- stage 1: ParameterGrid


GRIDS = [
    {"b": [1, 2], "a": ["x", "y", "z"]},
    [{"k": [1, 3, 5]}, {"k": [7], "weights": ["uniform", "distance"]}],
    {"alpha": (0.1,), "fit_intercept": [True, False], "degree": [1, 2, 3]},
]


@pytest.mark.parametrize("case", range(len(GRIDS)))
def test_stage1_matches_sklearn_order(nb, case):
    from sklearn.model_selection import ParameterGrid
    grid = GRIDS[case]
    ours = list(nb.ParameterGrid(grid))
    ref = list(ParameterGrid(grid))
    assert ours == ref, (
        f"expected {ref[:4]}... (keys sorted, last key varying fastest, dicts of a list one after another); "
        f"got {ours[:4]}...")
    assert len(nb.ParameterGrid(grid)) == len(ref), f"len should be {len(ref)}"


def test_stage1_worked_example_and_edge_cases(nb):
    got = list(nb.ParameterGrid({"degree": [1, 2], "alpha": [0.1, 1.0]}))
    assert got == [{"alpha": 0.1, "degree": 1}, {"alpha": 0.1, "degree": 2},
                   {"alpha": 1.0, "degree": 1}, {"alpha": 1.0, "degree": 2}], (
        f"keys sorted (alpha before degree), the last key varies fastest; got {got}")
    assert list(nb.ParameterGrid({})) == [{}] and len(nb.ParameterGrid({})) == 1, "an empty grid has one candidate: {}"
    with pytest.raises(ValueError):
        list(nb.ParameterGrid({"metric": "euclidean"}))
        pytest.fail("a string is not a list of values: raise ValueError (the given grids() helper does this)")


# ---------------------------------------------------------------- stage 2: GridSearchCV


def test_stage2_worked_example(nb):
    X, y = np.arange(6.0).reshape(-1, 1), np.array([1.0, 3.0, 2.0, 6.0, 5.0, 7.0])
    search = nb.GridSearchCV(_Shift(), {"shift": [-1.0, 0.0, 1.0]}, cv=3).fit(X, y)
    got = np.asarray(search.cv_results_["mean_test_score"])
    assert np.allclose(got, [-9.0, -8.0, -9.0]), (
        f"the lesson's table: mean fold MSEs 9, 8, 9 for shifts -1, 0, +1, so mean_test_score = [-9, -8, -9]; "
        f"got {got.tolist()}")
    assert search.best_params_ == {"shift": 0.0} and np.isclose(search.best_score_, -8.0)
    assert np.allclose(search.predict(X[:1]), 4.0), "the refit on all six points predicts their mean, 4"


def test_stage2_finds_the_best_and_records_results(nb):
    X, y = _data()
    shifts = [-1.0, -0.2, 0.0, 0.5]
    search = nb.GridSearchCV(_Shift(), {"shift": shifts}, cv=3)
    assert search.fit(X, y) is search, "fit returns self"
    res = search.cv_results_
    assert res["params"] == [{"shift": s} for s in shifts], "cv_results_['params'] lists the candidates in grid order"
    for i, s in enumerate(shifts):
        want = _manual_cv(s, X, y, 3)
        assert np.isclose(res["mean_test_score"][i], want.mean()) and np.isclose(res["std_test_score"][i], want.std()), (
            f"candidate shift={s}: mean_test_score should be the mean of its 3 fold scores ({want.mean():.4f}) and "
            "std_test_score their (ddof=0) std")
    assert search.best_params_ == {"shift": 0.0} and search.best_index_ == 2, (
        f"the unshifted predictor is best; got best_params_ = {search.best_params_}")
    assert np.isclose(search.best_score_, max(res["mean_test_score"])), "best_score_ is the best mean CV score"
    best = search.best_estimator_
    assert best.shift == 0.0 and np.isclose(best.mean_, y.mean()), (
        "best_estimator_ is a fresh clone with the best parameters, refitted on ALL the data")
    assert np.allclose(search.predict(X[:3]), y.mean()), "predict uses best_estimator_"
    assert np.isclose(search.score(X, y), -np.mean((y - y.mean()) ** 2)), "score without scoring is best_estimator_.score"


def test_stage2_clones_ties_scoring_and_refit(nb):
    X, y = _data(1)
    est = _Shift(shift=9.0)
    search = nb.GridSearchCV(est, {"unused": [5, 3, 4]}, cv=4).fit(X, y)
    assert est.shift == 9.0 and est.unused == 0 and not hasattr(est, "mean_"), (
        "never modify or fit the estimator you're given: clone it for every candidate and for the refit")
    assert search.best_params_ == {"unused": 5}, "candidates with equal scores: the FIRST one wins"
    calls = []

    def scorer(e, Xs, ys):
        calls.append(len(Xs))
        return float(e.shift)

    s2 = nb.GridSearchCV(_Shift(), {"shift": [1.0, 3.0, 2.0]}, cv=3, scoring=scorer).fit(X, y)
    assert s2.best_params_ == {"shift": 3.0} and len(calls) == 9, "pass scoring on to cross_validate for every candidate"
    assert s2.score(X, y) == 3.0, "score uses the scoring function when one was given"
    s3 = nb.GridSearchCV(_Shift(), {"shift": [0.0, 1.0]}, cv=3, refit=False).fit(X, y)
    assert not hasattr(s3, "best_estimator_") and s3.best_params_ == {"shift": 0.0}, "refit=False: no best_estimator_"
    with pytest.raises(TypeError):
        s3.predict(X)
        pytest.fail("predict without a refitted model should raise TypeError (the given best() helper does this)")


def test_stage2_same_folds_for_every_candidate(nb):
    from mlfs.model_selection import KFold
    X, y = _data(2, n=30)
    cv = KFold(5, shuffle=True, rng=np.random.default_rng(0))  # a Generator: every split() call differs
    search = bounded(nb.GridSearchCV(_Shift(), {"unused": [0, 1, 2, 3]}, cv=cv).fit, X, y,
                     budget=300_000, what="GridSearchCV.fit")
    means = search.cv_results_["mean_test_score"]
    assert np.allclose(means, means[0]), (
        f"four identical candidates must get identical scores, got {np.round(means, 4).tolist()}: compute the splits "
        "ONCE (list(cv.split(X, y))) and give every candidate the same list")


def test_stage2_get_params_and_clone(nb):
    from mlfs.base import clone
    search = nb.GridSearchCV(_Shift(), {"shift": [0.0]}, cv=3)
    params = search.get_params()
    assert set(params) == {"estimator", "param_grid", "cv", "scoring", "refit"}, (
        "GridSearchCV is an Estimator: __init__ stores its arguments under the same names")
    c = clone(search)
    assert type(c) is type(search) and c.estimator is not search.estimator, "clone(search) clones the inner estimator"


# ---------------------------------------------------------------- stage 3: RandomizedSearchCV


def _cands(search):
    return bounded(search.candidate_params, budget=300_000, what="candidate_params",
                   hint="draw exactly n_iter candidates, with a for loop over range(n_iter).")


def test_stage3_candidates(nb):
    seen_types = []

    def log_uniform(g):
        seen_types.append(type(g).__name__)
        return float(10 ** g.uniform(-3, 0))

    search = nb.RandomizedSearchCV(_Shift(), {"shift": log_uniform, "unused": [1, 2, 3]}, n_iter=500, rng=0)
    cands = _cands(search)
    assert len(cands) == 500 and all(set(c) == {"shift", "unused"} for c in cands), "n_iter dicts, one value per name"
    shifts = np.array([c["shift"] for c in cands])
    assert np.all((shifts >= 1e-3) & (shifts <= 1)) and set(seen_types) == {"Generator"}, (
        "a callable is called with the numpy Generator, value(gen)")
    assert abs(np.mean(np.log10(shifts)) + 1.5) < 0.17, "log-uniform on [1e-3, 1]: the mean of log10 should be near -1.5"
    counts = np.bincount([c["unused"] for c in cands], minlength=4)[1:]
    assert counts.min() > 130, f"a list is sampled uniformly: each of 3 values about 167 times in 500, got {counts.tolist()}"
    again = _cands(nb.RandomizedSearchCV(_Shift(), {"shift": log_uniform, "unused": [1, 2, 3]}, n_iter=500, rng=0))
    assert again == cands, "the same rng seed must give the same candidates"
    swapped = _cands(nb.RandomizedSearchCV(_Shift(), {"unused": [1, 2, 3], "shift": log_uniform}, n_iter=500, rng=0))
    assert swapped == cands, (
        "the order the names were written in must not matter: draw the names in SORTED order, so the same seed gives "
        "the same candidates")
    with pytest.raises(ValueError):
        _cands(nb.RandomizedSearchCV(_Shift(), {"shift": [0.0]}, n_iter=0))
        pytest.fail("n_iter=0 should raise ValueError")


def test_stage3_fit_uses_the_candidates(nb):
    X, y = _data(3)
    search = bounded(nb.RandomizedSearchCV(_Shift(), {"shift": [-2.0, 0.0, 2.0]}, n_iter=6, cv=3, rng=1).fit, X, y,
                     budget=1_000_000, what="RandomizedSearchCV.fit")
    params = search.cv_results_["params"]
    assert len(params) == 6, "cv_results_ has one entry per sampled candidate"
    assert params == _cands(nb.RandomizedSearchCV(_Shift(), {"shift": [-2.0, 0.0, 2.0]}, n_iter=6, rng=1)), (
        "fit evaluates exactly candidate_params(), in order")
    means = search.cv_results_["mean_test_score"]
    best = params[int(np.argmax(means))]
    assert search.best_params_ == best, f"best_params_ should be the best SAMPLED candidate, {best}; got {search.best_params_}"
    assert np.isclose(search.best_estimator_.mean_, y.mean() + best["shift"]), "and the refit uses it"


# ---------------------------------------------------------------- stage 4: nested cross-validation


def test_stage4_matches_a_manual_outer_loop(nb):
    from mlfs.model_selection import KFold
    X, y = _data(4, n=30)
    search = nb.GridSearchCV(_Shift(), {"shift": [-0.5, 0.0, 0.5]}, cv=3)
    scores = np.asarray(nb.nested_cross_validate(search, X, y, cv=5))
    assert scores.shape == (5,), f"one outer score per outer fold, got shape {scores.shape}"
    for f, (tr, te) in enumerate(KFold(5).split(X)):
        inner = nb.GridSearchCV(_Shift(), {"shift": [-0.5, 0.0, 0.5]}, cv=3).fit(X[tr], y[tr])
        assert np.isclose(scores[f], inner.score(X[te], y[te])), (
            f"outer fold {f}: run the search on the outer training rows only, then score its refitted winner on the "
            "outer test rows")
    assert not hasattr(search, "best_estimator_"), "the search you pass in stays unfitted (cross_validate clones it)"


class _Lucky(_Shift):
    """predicts the training mean plus a fixed random pattern chosen by `seed` (row i gets
    pattern[i]): every seed is equally useless on new data, but some fit a given sample by luck."""

    def __init__(self, seed=0):
        self.seed = seed

    def get_params(self):
        return {"seed": self.seed}

    def set_params(self, **p):
        self.seed = p.get("seed", self.seed)
        return self

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        pattern = np.random.default_rng(self.seed).normal(size=1000)
        return self.mean_ + 0.5 * pattern[np.asarray(X)[:, 0].astype(int)]


def test_stage4_is_less_optimistic_than_best_score(nb):
    from mlfs.model_selection import KFold
    X = np.arange(40.0).reshape(-1, 1)
    grid = {"seed": list(range(30))}
    best, nested, typical = [], [], []
    for d in range(10):
        y = np.random.default_rng(100 + d).normal(size=40)
        search = nb.GridSearchCV(_Lucky(), grid, cv=5).fit(X, y)
        best.append(search.best_score_)
        typical.append(np.mean(search.cv_results_["mean_test_score"]))
        outer = KFold(5, shuffle=True, rng=200 + d)  # different from the inner, unshuffled folds
        nested.append(np.mean(nb.nested_cross_validate(nb.GridSearchCV(_Lucky(), grid, cv=5), X, y, cv=outer)))
    best, nested, typical = np.mean(best), np.mean(nested), np.mean(typical)
    assert best > typical + 0.05 and nested < best - 0.1, (
        f"30 equally useless candidates, 10 datasets: the best CV score (mean {best:.3f}) is optimistic, above the "
        f"average candidate ({typical:.3f}); nested CV evaluates each choice on rows its search never saw, so it should "
        f"come out well below. got nested = {nested:.3f}")


class _Thresh:
    """a classifier with string labels: predicts classes_[1] where x0 > t."""

    def __init__(self, t=0.0):
        self.t = t

    def get_params(self):
        return {"t": self.t}

    def set_params(self, **p):
        self.t = p.get("t", self.t)
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X)[:, 0] - self.t)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


def test_stage4_classes_and_probability_scorers(nb):
    from mlfs.metrics import get_scorer
    from mlfs.model_selection import StratifiedKFold
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 1))
    y = np.where(X[:, 0] + 0.5 * rng.normal(size=80) > 0, "spam", "ham")
    search = nb.GridSearchCV(_Thresh(), {"t": [-1.0, 0.0, 1.0]}, cv=StratifiedKFold(4, shuffle=True, rng=0)).fit(X, y)
    assert list(getattr(search, "classes_", [])) == ["ham", "spam"], (
        "after a refit, the search exposes the winner's classes_, so probability scorers (roc_auc, neg_log_loss) can map "
        "labels to predict_proba's columns")
    scores = np.asarray(nb.nested_cross_validate(
        nb.GridSearchCV(_Thresh(), {"t": [-1.0, 0.0, 1.0]}, cv=StratifiedKFold(4, shuffle=True, rng=0)), X, y,
        cv=StratifiedKFold(4, shuffle=True, rng=1), scoring=get_scorer("roc_auc")))
    assert scores.shape == (4,) and np.all(scores > 0.5) and scores.mean() > 0.75, (
        f"nested CV with the roc_auc scorer should work on a search (it needs search.classes_ and search.predict_proba); "
        f"the feature is informative, so every outer AUC should be above 0.5, and their mean well above it. got {scores.tolist()}")
