"""Checks for exercise.py (what is learning? the Estimator protocol, baselines, ERM).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The toy
estimators below are built on YOUR base classes (nb.Estimator, nb.Classifier, ...), so
these checks exercise the protocol exactly the way chapters 08-13 will use it.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: a stump on six points
X_STUMP = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
Y_STUMP = np.array([0, 0, 1, 0, 1, 1])


# ---------------------------------------------------------------- stage 1: check_X_y


def test_stage1_converts_to_float_2d(nb):
    X, y = nb.check_X_y([[1, 2], [3, 4], [5, 6]], [0, 1, 0])
    assert isinstance(X, np.ndarray) and X.dtype == np.float64 and X.shape == (3, 2), (
        f"check_X_y should return X as a float64 ndarray of shape (n, d); got {type(X).__name__} "
        f"{getattr(X, 'dtype', None)} {getattr(X, 'shape', None)}")
    assert isinstance(y, np.ndarray) and y.shape == (3,), "y should come back as a 1-D ndarray"
    X1, _ = nb.check_X_y(np.array([1.0, 2.0, 3.0]))
    assert np.asarray(X1).shape == (3, 1), (
        f"a 1-D X is ONE feature measured on n samples: it should become a column (n, 1), got {np.asarray(X1).shape}")
    X2, y2 = nb.check_X_y(np.ones((4, 2)))
    assert y2 is None and np.asarray(X2).shape == (4, 2), "with y=None, return (X, None)"


def test_stage1_keeps_labels_and_flattens_columns(nb):
    _, y = nb.check_X_y(np.zeros((3, 1)), np.array(["ham", "spam", "ham"]))
    assert list(y) == ["ham", "spam", "ham"], "y keeps its dtype: class labels may be strings"
    _, y = nb.check_X_y(np.zeros((3, 1)), np.array([[1.0], [2.0], [3.0]]))
    assert np.asarray(y).shape == (3,), "a y column of shape (n, 1) should be flattened to (n,)"


@pytest.mark.parametrize("case", ["nan_X", "inf_X", "3d_X", "empty_X", "short_y", "2d_y", "nan_y"])
def test_stage1_rejects_bad_input(nb, case):
    X, y = np.ones((4, 2)), np.arange(4.0)
    if case == "nan_X":
        X[1, 1] = np.nan
    elif case == "inf_X":
        X[0, 0] = np.inf
    elif case == "3d_X":
        X = np.ones((4, 2, 2))
    elif case == "empty_X":
        X, y = np.ones((0, 2)), np.ones(0)
    elif case == "short_y":
        y = np.arange(3.0)
    elif case == "2d_y":
        y = np.ones((4, 2))
    elif case == "nan_y":
        y[2] = np.nan
    with pytest.raises(ValueError):
        nb.check_X_y(X, y)
        pytest.fail(f"check_X_y should raise ValueError for {case.replace('_', ' ')}, but it returned normally")


# ---------------------------------------------------------------- stage 2: get_params, set_params, clone


def _toys(nb):
    class Smoother(nb.Estimator):
        def __init__(self, alpha=1.0, *, fit_intercept=True):
            self.alpha = alpha
            self.fit_intercept = fit_intercept

        def fit(self, X, y=None):
            self.coef_ = np.ones(2)
            return self

    class Wrapper(nb.Estimator):
        def __init__(self, base, n_copies=3, tags=None, rng=None):
            self.base = base
            self.n_copies = n_copies
            self.tags = tags
            self.rng = rng

    class Forgetful(nb.Estimator):
        def __init__(self, alpha=1.0):
            pass

    class NoInit(nb.Estimator):
        pass

    return Smoother, Wrapper, Forgetful, NoInit


def test_stage2_get_params_reads_init_signature(nb):
    Smoother, Wrapper, Forgetful, NoInit = _toys(nb)
    s = Smoother(alpha=0.5)
    assert s.get_params() == {"alpha": 0.5, "fit_intercept": True}, (
        f"get_params should return every __init__ parameter (not self) with its CURRENT value; got {s.get_params()}")
    s.alpha = 7.0
    assert s.get_params()["alpha"] == 7.0, "get_params must read the attributes now (getattr), not remember the defaults"
    assert NoInit().get_params() == {}, "a class without its own __init__ has no hyperparameters: get_params() == {}"
    with pytest.raises(AttributeError):
        Forgetful().get_params()
        pytest.fail("Forgetful.__init__ takes alpha but never stores self.alpha: get_params should raise AttributeError")


def test_stage2_set_params(nb):
    Smoother, Wrapper, *_ = _toys(nb)
    s = Smoother()
    out = s.set_params(alpha=3.0, fit_intercept=False)
    assert out is s, "set_params returns self, so calls can be chained: clone(est).set_params(...).fit(X, y)"
    assert s.alpha == 3.0 and s.fit_intercept is False
    with pytest.raises(ValueError):
        s.set_params(alhpa=1.0)
        pytest.fail("a misspelled hyperparameter (alhpa) must raise ValueError, not silently create an attribute")
    w = Wrapper(Smoother())
    w.set_params(base__alpha=0.25, n_copies=5)
    assert w.base.alpha == 0.25 and w.n_copies == 5, (
        "set_params(base__alpha=0.25) should call self.base.set_params(alpha=0.25)")


def test_stage2_clone_is_fresh_and_unfitted(nb):
    Smoother, *_ = _toys(nb)
    s = Smoother(alpha=0.1).fit(None)
    c = nb.clone(s)
    assert type(c) is Smoother and c is not s, "clone returns a NEW object of the same class"
    assert c.get_params() == {"alpha": 0.1, "fit_intercept": True}, "the clone has the same hyperparameters"
    assert not hasattr(c, "coef_"), (
        "the clone must be unfitted: build it with type(est)(**params), never with copy.deepcopy(est), "
        "which would copy fitted attributes like coef_ too")


def test_stage2_clone_copies_nested_and_mutable_params(nb):
    Smoother, Wrapper, *_ = _toys(nb)
    inner = Smoother(alpha=2.0).fit(None)
    tags = ["a", "b"]
    w = Wrapper(inner, tags=tags, rng=np.random.default_rng(0))
    c = nb.clone(w)
    assert c.base is not inner and type(c.base) is Smoother and c.base.alpha == 2.0, (
        "a hyperparameter that is itself an estimator must be cloned recursively")
    assert not hasattr(c.base, "coef_"), "the nested estimator must come back unfitted"
    tags.append("c")
    assert c.tags == ["a", "b"], "mutable hyperparameters (lists, arrays) must be copied, not shared"
    assert c.rng is not w.rng and c.rng.random() == np.random.default_rng(0).random(), (
        "a Generator hyperparameter should be deep-copied: same state, separate object")
    steps = Wrapper([("first", Smoother(alpha=3.0).fit(None)), ("second", Smoother())])
    cs = nb.clone(steps)
    assert cs.base[0][0] == "first" and cs.base[0][1] is not steps.base[0][1] and cs.base[0][1].alpha == 3.0, (
        "estimators inside lists/tuples (like a pipeline's [(name, est), ...] steps) must be cloned too")
    assert not hasattr(cs.base[0][1], "coef_"), (
        "an estimator inside a list must be CLONED (unfitted), not deep-copied along with its fitted state")
    assert isinstance(cs.base, list) and isinstance(cs.base[0], tuple), "keep the container types (list of tuples)"


def test_stage2_clone_ignores_nested_keys(nb):
    Smoother, *_ = _toys(nb)

    class Deep(nb.Estimator):
        def __init__(self, inner):
            self.inner = inner

        def get_params(self):  # like Pipeline: also lists the sub-estimator's settings
            return {"inner": self.inner, "inner__alpha": self.inner.alpha}

    c = nb.clone(Deep(Smoother(alpha=4.0)))
    assert c.inner.alpha == 4.0, (
        "clone should pass only the keys without '__' to the constructor; 'inner__alpha' describes the sub-estimator")


# ---------------------------------------------------------------- stage 3: Regressor, Classifier, Transformer


def _fixed_regressor(nb, preds):
    class Fixed(nb.Regressor):
        def predict(self, X):
            return np.asarray(preds, dtype=float)
    return Fixed()


def test_stage3_r2_score(nb):
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    for preds in (y + 0.3 * rng.normal(size=20), np.full(20, y.mean()), -y, y):
        got = _fixed_regressor(nb, preds).score(np.zeros((20, 1)), y)
        assert np.isclose(got, r2_score(y, preds)), (
            f"R^2 = 1 - SS_res/SS_tot should be {r2_score(y, preds):.4f}, got {got}. predicting the mean gives 0, "
            "a perfect fit 1, and predicting -y gives -3")
    assert isinstance(got, float), "score should return a Python float"
    const = np.full(5, 2.0)
    assert _fixed_regressor(nb, const).score(np.zeros((5, 1)), const) == 1.0, (
        "constant y predicted perfectly: SS_tot = 0, so return 1.0 (don't divide by zero)")
    assert _fixed_regressor(nb, const + 1).score(np.zeros((5, 1)), const) == 0.0, (
        "constant y predicted imperfectly: return 0.0")


def _fixed_classifier(nb, proba, classes):
    class Fixed(nb.Classifier):
        def fit(self, X, y=None):
            self.classes_ = np.asarray(classes)
            return self

        def predict_proba(self, X):
            return np.asarray(proba, dtype=float)
    return Fixed().fit(None)


def test_stage3_classifier_predict_and_score(nb):
    proba = [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2]]
    clf = _fixed_classifier(nb, proba, ["bird", "cat", "dog"])
    pred = np.asarray(clf.predict(np.zeros((4, 1))))
    assert list(pred) == ["bird", "dog", "cat", "bird"], (
        f"predict = classes_[argmax predict_proba] row by row, ties to the first column; got {list(pred)}. "
        "return LABELS (here strings), not column indices")
    acc = clf.score(np.zeros((4, 1)), ["bird", "dog", "dog", "cat"])
    assert np.isclose(acc, 0.5) and isinstance(acc, float), f"accuracy should be 2/4 = 0.5 (a float), got {acc!r}"


def test_stage3_abstract_methods_raise(nb):
    class Bare(nb.Classifier):
        pass

    class BareT(nb.Transformer):
        pass

    for call, what in ((lambda: nb.Estimator().fit(np.zeros((2, 1))), "Estimator.fit"),
                       (lambda: nb.Estimator().predict(np.zeros((2, 1))), "Estimator.predict"),
                       (lambda: Bare().predict_proba(np.zeros((2, 1))), "Classifier.predict_proba"),
                       (lambda: BareT().transform(np.zeros((2, 1))), "Transformer.transform"),
                       (lambda: BareT().inverse_transform(np.zeros((2, 1))), "Transformer.inverse_transform")):
        with pytest.raises(NotImplementedError):
            call()
            pytest.fail(f"{what} on a subclass that doesn't override it should raise NotImplementedError")


def test_stage3_transformer_fit_transform(nb):
    seen = {}

    class Shift(nb.Transformer):
        def fit(self, X, y=None):
            seen["y"] = y
            self.mean_ = np.mean(X, axis=0)
            return self

        def transform(self, X):
            return np.asarray(X) - self.mean_

    X = np.array([[1.0, 2.0], [3.0, 6.0]])
    out = Shift().fit_transform(X, np.array([0, 1]))
    assert np.allclose(out, [[-1.0, -2.0], [1.0, 2.0]]), "fit_transform(X) = fit(X, y).transform(X)"
    assert seen.get("y") is not None, "fit_transform must pass y on to fit (a supervised transformer may need it)"


# ---------------------------------------------------------------- stage 4: dummy baselines


def test_stage4_dummy_regressor(nb):
    y = np.array([1.0, 2.0, 3.0, 10.0])
    X = np.zeros((4, 3))
    m = nb.DummyRegressor().fit(X, y)
    assert np.isclose(m.constant_, 4.0) and np.allclose(m.predict(np.zeros((2, 3))), [4.0, 4.0]), (
        f"strategy='mean' should predict mean(y) = 4.0 for every row; got constant_ = {getattr(m, 'constant_', None)}")
    assert np.isclose(m.score(X, y), 0.0), "the mean predictor has R^2 exactly 0 on its own training data"
    md = nb.DummyRegressor(strategy="median").fit(X, y)
    assert np.isclose(md.constant_, 2.5), "strategy='median' should predict median(y) = 2.5"
    assert np.asarray(m.predict(np.zeros((5, 3)))).shape == (5,), "predict returns one value per row of X"
    assert nb.DummyRegressor().get_params() == {"strategy": "mean"}, "__init__ stores strategy as self.strategy"
    with pytest.raises(ValueError):
        nb.DummyRegressor(strategy="mode").fit(X, y)
        pytest.fail("an unknown strategy should raise ValueError in fit")


def test_stage4_dummy_classifier_deterministic_strategies(nb):
    y = np.array(["spam", "ham", "ham", "eggs", "ham", "spam"])
    X = np.zeros((6, 2))
    clf = nb.DummyClassifier().fit(X, y)
    assert list(clf.classes_) == ["eggs", "ham", "spam"], "classes_ are the sorted distinct labels"
    assert np.allclose(clf.class_prior_, [1 / 6, 3 / 6, 2 / 6]), f"class_prior_ should be [1/6, 1/2, 1/3], got {clf.class_prior_}"
    assert np.allclose(clf.predict_proba(np.zeros((2, 2))), [[0, 1, 0], [0, 1, 0]]), (
        "most_frequent: a one-hot row on the most frequent class ('ham')")
    assert list(clf.predict(np.zeros((3, 2)))) == ["ham"] * 3
    tie = nb.DummyClassifier().fit(np.zeros((4, 1)), [2, 1, 2, 1])
    assert list(tie.predict(np.zeros((1, 1)))) == [1], "a tie between most frequent classes goes to the smallest label"
    pri = nb.DummyClassifier(strategy="prior").fit(X, y)
    assert np.allclose(pri.predict_proba(np.zeros((2, 2))), [[1 / 6, 1 / 2, 1 / 3]] * 2), "prior: class_prior_ in every row"
    assert list(pri.predict(np.zeros((1, 2)))) == ["ham"], "prior predicts the argmax: the most frequent class"
    uni = nb.DummyClassifier(strategy="uniform").fit(X, y)
    assert np.allclose(uni.predict_proba(np.zeros((2, 2))), 1 / 3), "uniform: 1/k everywhere"
    with pytest.raises(ValueError):
        nb.DummyClassifier(strategy="majority").fit(X, y)
        pytest.fail("an unknown strategy should raise ValueError in fit")
    with pytest.raises(ValueError):
        nb.DummyClassifier().fit(np.zeros((5, 2)), y)
        pytest.fail("len(X) != len(y) should raise ValueError")


def test_stage4_dummy_classifier_matches_sklearn(nb):
    from sklearn.dummy import DummyClassifier as SkDummy
    rng = np.random.default_rng(1)
    y = rng.choice([3, 5, 9], size=50, p=[0.2, 0.5, 0.3])
    X = rng.normal(size=(50, 2))
    for strategy in ("most_frequent", "prior", "uniform"):
        ours = nb.DummyClassifier(strategy=strategy).fit(X, y).predict_proba(X[:7])
        ref = SkDummy(strategy=strategy).fit(X, y).predict_proba(X[:7])
        assert np.allclose(ours, ref), f"predict_proba for strategy={strategy!r} should match sklearn's DummyClassifier"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stage4_dummy_classifier_random_strategies(nb, seed):
    y = np.array([0] * 70 + [1] * 30)
    X = np.zeros((100, 1))
    n = 4000
    strat = nb.DummyClassifier(strategy="stratified", rng=seed).fit(X, y)
    P = np.asarray(strat.predict_proba(np.zeros((n, 1))))
    assert P.shape == (n, 2) and np.allclose(P.sum(1), 1) and set(np.unique(P)) <= {0.0, 1.0}, (
        "stratified: each row is one-hot on a class drawn at random")
    frac = P[:, 1].mean()
    assert abs(frac - 0.3) < 0.04, f"stratified should draw class 1 about 30% of the time (its prior); got {frac:.3f}"
    again = nb.DummyClassifier(strategy="stratified", rng=seed).fit(X, y).predict_proba(np.zeros((n, 1)))
    assert np.array_equal(P, again), "with the same rng seed, refitting must reproduce the same draws (use as_rng)"
    uni = nb.DummyClassifier(strategy="uniform", rng=seed).fit(X, y)
    pred = np.asarray(uni.predict(np.zeros((n, 1))))
    assert abs(np.mean(pred == 1) - 0.5) < 0.04, (
        f"uniform: predict should guess each class half of the time, got class 1 in {np.mean(pred == 1):.0%} of rows. "
        "the argmax of a flat row always picks classes_[0]: draw a random label instead")


# ---------------------------------------------------------------- stage 5: memorization vs ERM


def test_stage5_lookup_table_memorizes(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.integers(0, 2, size=40)
    lut = nb.LookupTableClassifier().fit(X, y)
    assert lut.score(X, y) == 1.0, "the lookup table must get every (distinct) training row right"
    Xd = np.array([[0.0], [0.0], [0.0], [1.0], [2.0], [2.0], [3.0]])
    yd = np.array(["b", "b", "a", "c", "a", "c", "c"])
    lut = nb.LookupTableClassifier().fit(Xd, yd)
    pred = list(lut.predict(np.array([[0.0], [1.0], [2.0], [7.0]])))
    assert pred == ["b", "c", "a", "c"], (
        f"expected ['b', 'c', 'a', 'c']: row 0 was seen with b, b, a (majority b); row 1 with c; row 2 with a and c "
        f"(a tie: the smaller label, 'a'); row 7 was never seen, so use the most frequent training class ('c', 3 of 7). "
        f"got {pred}")
    assert np.asarray(lut.predict_proba(np.array([[1.0]]))).shape == (1, 3), "predict_proba has one column per class"


def test_stage5_stump_worked_example(nb):
    s = nb.StumpClassifier().fit(X_STUMP, Y_STUMP)
    assert (s.feature_, s.threshold_, s.sign_) == (0, 2.5, 1), (
        f"thresholds 2.5 and 4.5 both make 1 mistake on the worked example; the tie goes to the smaller threshold, "
        f"so expect (feature_, threshold_, sign_) = (0, 2.5, 1); got {(s.feature_, s.threshold_, s.sign_)}")
    assert np.isclose(s.train_error_, 1 / 6), f"train_error_ should be 1/6, got {s.train_error_}"
    assert list(s.predict(np.array([[0.0], [2.4], [2.6], [100.0]]))) == [0, 0, 1, 1]


def _brute_force_stump_error(X, z):
    n, d = X.shape
    best = n
    for j in range(d):
        v = np.unique(X[:, j])
        for t in np.concatenate([[-np.inf], (v[1:] + v[:-1]) / 2]):
            right = X[:, j] > t
            for pred in (right, ~right):
                best = min(best, int(np.sum(pred != z)))
    return best / n


@pytest.mark.parametrize("seed", range(5))
def test_stage5_stump_is_the_empirical_risk_minimizer(nb, seed):
    rng = np.random.default_rng(seed)
    n, d = 30, 3
    X = np.round(rng.normal(size=(n, d)), 1)  # rounding creates repeated values
    z = (X[:, seed % d] + 0.8 * rng.normal(size=n) > 0)
    y = np.where(z, "yes", "no")  # classes_ = ["no", "yes"]
    s = nb.StumpClassifier().fit(X, y)
    want = _brute_force_stump_error(X, z)
    assert np.isclose(s.train_error_, want), (
        f"the best stump makes {want:.3f} training error (checked over every feature, midpoint and sign); yours "
        f"reports {s.train_error_:.3f}. remember both signs, and that a threshold only makes sense between DISTINCT values")
    pred = np.asarray(s.predict(X))
    assert np.isclose(np.mean(pred != y), s.train_error_), "predict must agree with the stump you chose (train_error_)"
    flipped = nb.StumpClassifier().fit(-X, y)
    assert np.isclose(flipped.train_error_, want), "negating the features shouldn't change the best error (sign -1 covers it)"


def test_stage5_stump_rules_and_speed(nb):
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    s = nb.StumpClassifier().fit(X, [1, 1, 0, 0])
    assert (s.feature_, s.sign_) == (0, -1) and s.threshold_ == 1.5, (
        "two identical features tie: choose feature 0; the classes are on the LEFT here, so sign_ = -1 and threshold 1.5. "
        f"got feature_={s.feature_}, threshold_={s.threshold_}, sign_={s.sign_}")
    with pytest.raises(ValueError):
        nb.StumpClassifier().fit(X, [0, 1, 2, 2])
        pytest.fail("three classes should raise ValueError: a stump splits two")
    rng = np.random.default_rng(3)
    Xb = rng.normal(size=(2000, 2))
    yb = (Xb[:, 1] > 0.3).astype(int)
    s = bounded(nb.StumpClassifier().fit, Xb, yb, budget=300_000, what="StumpClassifier.fit",
                hint="score all thresholds of a feature at once with cumulative sums over the sorted labels, "
                     "instead of a Python loop over rows for every threshold.")
    assert s.feature_ == 1 and abs(s.threshold_ - 0.3) < 0.01 and s.train_error_ == 0.0
