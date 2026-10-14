"""Checks for exercise.py (features and preprocessing).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The
transformers are compared with scikit-learn's; the pipeline is checked with small
recording estimators defined here.
"""
import numpy as np
import pytest

# the lesson's worked example: two numeric features and one categorical one
X_NUM = np.array([[1.0, 100.0], [2.0, 300.0], [3.0, 200.0], [6.0, 400.0]])
COLORS = np.array([["red"], ["green"], ["blue"], ["green"]])


def _data(seed, n=40, d=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) * np.array([1.0, 50.0, 0.01])[:d] + np.array([0.0, 1000.0, -3.0])[:d]


# ---------------------------------------------------------------- stage 1: scalers


def test_stage1_worked_example(nb):
    s = nb.StandardScaler().fit(X_NUM)
    assert np.allclose(s.mean_, [3.0, 250.0]) and np.allclose(s.scale_, [np.sqrt(3.5), np.sqrt(12500.0)]), (
        f"mean_ should be [3, 250] and scale_ the population standard deviations [sqrt(3.5), sqrt(12500)]; "
        f"got {s.mean_}, {s.scale_}")
    z = s.transform(X_NUM)
    assert np.allclose(z[:, 0], np.array([-2.0, -1.0, 0.0, 3.0]) / np.sqrt(3.5)), "z = (x - mean) / std, column by column"
    m = nb.MinMaxScaler().fit(X_NUM)
    assert np.allclose(m.transform(X_NUM), [[0, 0], [0.2, 2 / 3], [0.4, 1 / 3], [1, 1]]), (
        "min-max maps the column minimum to 0 and maximum to 1: x0 -> (x0 - 1)/5, x1 -> (x1 - 100)/300")


@pytest.mark.parametrize("seed", range(3))
def test_stage1_match_sklearn(nb, seed):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    X, Xnew = _data(seed), _data(seed + 10, n=7)
    for wm in (True, False):
        for ws in (True, False):
            ours = nb.StandardScaler(with_mean=wm, with_std=ws).fit(X)
            ref = StandardScaler(with_mean=wm, with_std=ws).fit(X)
            assert np.allclose(ours.transform(Xnew), ref.transform(Xnew)), (
                f"StandardScaler(with_mean={wm}, with_std={ws}) should match sklearn on NEW data, using the statistics "
                "learned in fit")
            assert np.allclose(ours.inverse_transform(ours.transform(Xnew)), Xnew), "inverse_transform undoes transform"
    for fr in ((0.0, 1.0), (-1.0, 1.0)):
        ours = nb.MinMaxScaler(feature_range=fr).fit(X)
        ref = MinMaxScaler(feature_range=fr).fit(X)
        assert np.allclose(ours.transform(Xnew), ref.transform(Xnew)), (
            f"MinMaxScaler(feature_range={fr}) should match sklearn on new data (values outside the range are NOT clipped)")
        assert np.allclose(ours.inverse_transform(ours.transform(Xnew)), Xnew)


def test_stage1_constant_columns_and_errors(nb):
    X = np.column_stack([np.full(5, 7.0), np.arange(5.0)])
    z = nb.StandardScaler().fit(X).transform(X)
    assert np.all(np.isfinite(z)) and np.allclose(z[:, 0], 0.0), (
        "a constant column has std 0: use scale 1, so it becomes all zeros instead of NaN")
    tenths = np.full((7, 1), 0.1)  # its computed std is about 1e-17, not 0
    s01 = nb.StandardScaler().fit(tenths)
    out = s01.transform(np.array([[0.1], [0.2]]))
    assert np.allclose(out, [[0.0], [0.1]], atol=1e-12), (
        f"a column of 0.1s is constant, but rounding makes its std ~1e-17, not 0. test for constant columns with "
        f"X.max(0) == X.min(0) and use scale 1; got transform([[0.1], [0.2]]) = {np.asarray(out).ravel().tolist()}")
    mm = nb.MinMaxScaler(feature_range=(2.0, 3.0)).fit(X).transform(X)
    assert np.allclose(mm[:, 0], 2.0), "a constant column maps to the low end of feature_range"
    assert nb.StandardScaler().get_params() == {"with_mean": True, "with_std": True}
    with pytest.raises(ValueError):
        nb.StandardScaler().fit(X[:, :1]).transform(np.ones((2, 3)))
        pytest.fail("a scaler fitted on 1 column got 3: raise ValueError (broadcasting would silently apply the one "
                    "column's statistics to all three)")
    with pytest.raises(ValueError):
        nb.MinMaxScaler(feature_range=(1.0, 0.0)).fit(X)
        pytest.fail("feature_range (1, 0) is backwards: raise ValueError in fit")


# ---------------------------------------------------------------- stage 2: OneHotEncoder


def test_stage2_worked_example(nb):
    enc = nb.OneHotEncoder().fit(COLORS)
    assert [list(c) for c in enc.categories_] == [["blue", "green", "red"]], "categories_ are sorted per column"
    out = enc.transform(COLORS)
    assert np.allclose(out, [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]]) and out.dtype == float, (
        f"one column per category in sorted order (blue, green, red); got {np.asarray(out).tolist()}")
    dropped = nb.OneHotEncoder(drop="first").fit(COLORS).transform(COLORS)
    assert np.allclose(dropped, [[0, 1], [1, 0], [0, 0], [1, 0]]), "drop='first' removes the 'blue' column: blue = (0, 0)"


@pytest.mark.parametrize("drop", [None, "first"])
def test_stage2_match_sklearn(nb, drop):
    from sklearn.preprocessing import OneHotEncoder
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.choice(["a", "b", "c"], 30), rng.choice(["x", "y"], 30), rng.choice(["p", "q", "r", "s"], 30)])
    ours = nb.OneHotEncoder(drop=drop).fit(X)
    ref = OneHotEncoder(drop=drop, sparse_output=False).fit(X)
    assert np.allclose(ours.transform(X[:9]), ref.transform(X[:9])), f"transform (drop={drop}) should match sklearn"
    back = np.asarray(ours.inverse_transform(ours.transform(X[:9])))
    assert (back == X[:9]).all(), "inverse_transform(transform(X)) should give back the categories"
    Xn = np.column_stack([rng.integers(0, 4, 20), rng.integers(10, 12, 20)])
    assert np.allclose(nb.OneHotEncoder(drop=drop).fit(Xn).transform(Xn),
                       OneHotEncoder(drop=drop, sparse_output=False).fit(Xn).transform(Xn)), "numeric categories work too"


def test_stage2_unknown_categories(nb):
    test = np.array([["green"], ["purple"]])
    with pytest.raises(ValueError):
        nb.OneHotEncoder().fit(COLORS).transform(test)
        pytest.fail("an unseen category ('purple') should raise ValueError with handle_unknown='error'")
    out = nb.OneHotEncoder(handle_unknown="ignore").fit(COLORS).transform(test)
    assert np.allclose(out, [[0, 1, 0], [0, 0, 0]]), "handle_unknown='ignore' encodes an unseen category as all zeros"
    back = np.asarray(nb.OneHotEncoder(handle_unknown="ignore").fit(COLORS).inverse_transform(out))
    assert back[0, 0] == "green" and back[1, 0] is None, "inverse_transform maps an all-zero block to None"
    with pytest.raises(ValueError):
        nb.OneHotEncoder().fit(np.array(["red", "blue"]))
        pytest.fail("a 1-D X is ambiguous: raise ValueError (pass a column of shape (n, 1))")


# ---------------------------------------------------------------- stage 3: SimpleImputer


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
def test_stage3_match_sklearn(nb, strategy):
    from sklearn.impute import SimpleImputer
    rng = np.random.default_rng(1)
    X = rng.integers(0, 5, size=(30, 3)).astype(float)
    X[rng.random((30, 3)) < 0.25] = np.nan
    Xnew = rng.integers(0, 5, size=(6, 3)).astype(float)
    Xnew[rng.random((6, 3)) < 0.5] = np.nan
    kw = {"fill_value": -1.0} if strategy == "constant" else {}
    ours = nb.SimpleImputer(strategy=strategy, **kw).fit(X)
    ref = SimpleImputer(strategy=strategy, **kw).fit(X)
    assert np.allclose(ours.statistics_, np.asarray(ref.statistics_, dtype=float)), (
        f"statistics_ for strategy={strategy!r} should be {ref.statistics_}, got {ours.statistics_}")
    before = Xnew.copy()
    out = ours.transform(Xnew)
    assert np.allclose(out, ref.transform(Xnew)), "transform fills NEW data with the statistics learned in fit"
    assert np.array_equal(np.isnan(Xnew), np.isnan(before)), "transform must not modify its input: fill a copy"


def test_stage3_hand_cases(nb):
    X = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, 4.0], [1.0, 2.0], [3.0, 2.0]])
    mf = nb.SimpleImputer(strategy="most_frequent").fit(X)
    assert np.allclose(mf.statistics_, [1.0, 2.0]), (
        f"most_frequent ties (1 and 3 twice each; 2 and 4 twice each) go to the smallest value: [1, 2]; got {mf.statistics_}")
    assert np.allclose(nb.SimpleImputer().fit(X).transform(X)[1, 0], 2.0), "the mean ignores NaNs: (1 + 3 + 1 + 3) / 4 = 2"
    assert np.allclose(nb.SimpleImputer(strategy="constant").fit(X).statistics_, 0.0), "constant without fill_value uses 0"
    with pytest.raises(ValueError):
        nb.SimpleImputer().fit(np.array([[1.0, np.nan], [2.0, np.nan]]))
        pytest.fail("a column with no observed values has no mean: raise ValueError")
    with pytest.raises(ValueError):
        nb.SimpleImputer(strategy="mode").fit(X)
        pytest.fail("an unknown strategy should raise ValueError")


# ---------------------------------------------------------------- stage 4: PolynomialFeatures


def test_stage4_worked_example(nb):
    X = np.array([[2.0, 3.0]])
    out = nb.PolynomialFeatures(degree=2).fit(X).transform(X)
    assert np.allclose(out, [[1, 2, 3, 4, 6, 9]]), (
        f"degree 2 on (x0, x1) = (2, 3) gives (1, x0, x1, x0^2, x0 x1, x1^2) = (1, 2, 3, 4, 6, 9); got {np.asarray(out).tolist()}")


@pytest.mark.parametrize("degree,bias,inter", [(1, True, False), (2, False, False), (3, True, False), (3, True, True), (2, False, True)])
def test_stage4_match_sklearn(nb, degree, bias, inter):
    from sklearn.preprocessing import PolynomialFeatures
    X = _data(2, n=6)
    ours = nb.PolynomialFeatures(degree, include_bias=bias, interaction_only=inter).fit(X)
    ref = PolynomialFeatures(degree, include_bias=bias, interaction_only=inter).fit(X)
    assert np.array_equal(ours.powers_, ref.powers_), (
        f"powers_ (degree={degree}, include_bias={bias}, interaction_only={inter}) should list the monomials in sklearn's "
        "order: by total degree, then itertools.combinations_with_replacement order")
    assert np.allclose(ours.transform(X), ref.transform(X)), "transform should match sklearn's"


# ---------------------------------------------------------------- stage 5: Pipeline


class _Record:
    """a transformer that logs its calls and adds `add` to X."""

    def __init__(self, add=0.0, log=None):
        self.add = add
        self.log = log

    def get_params(self):
        return {"add": self.add, "log": self.log}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        self.log.append(("fit", self.add, None if y is None else len(y)))
        self.fitted_ = True
        return self

    def transform(self, X):
        self.log.append(("transform", self.add))
        return np.asarray(X, dtype=float) + self.add

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class _Model:
    classes_ = np.array([0, 1])

    def __init__(self, log=None):
        self.log = log

    def get_params(self):
        return {"log": self.log}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        self.seen_ = np.asarray(X).copy()
        self.log.append(("model fit", None if y is None else len(y)))
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.asarray(X)[:, 0]))
        return np.column_stack([1 - p, p])

    def score(self, X, y=None):
        return float(np.asarray(X)[:, 0].sum())


def test_stage5_fit_and_predict_flow(nb):
    log = []
    pipe = nb.Pipeline([("a", _Record(1.0, log)), ("b", _Record(10.0, log)), ("m", _Model(log))])
    X, y = np.zeros((3, 2)), np.array([0, 1, 0])
    assert pipe.fit(X, y) is pipe, "fit returns self"
    assert log == [("fit", 1.0, 3), ("transform", 1.0), ("fit", 10.0, 3), ("transform", 10.0), ("model fit", 3)], (
        f"fit must fit_transform each step in order (passing y), then fit the last step on the result; the calls were {log}")
    assert np.allclose(pipe.named_steps["m"].seen_, 11.0), "the model must be fitted on the TRANSFORMED X"
    log.clear()
    assert np.allclose(pipe.predict(np.ones((2, 2))), 12.0), "predict transforms with every earlier step, then predicts"
    assert log == [("transform", 1.0), ("transform", 10.0)], "predict must only transform (never refit) the earlier steps"
    proba = np.asarray(pipe.predict_proba(np.zeros((2, 2))))
    s11 = 1 / (1 + np.exp(-11.0))
    assert proba.shape == (2, 2) and np.allclose(proba, [[1 - s11, s11]] * 2), (
        "predict_proba must transform X with every earlier step (here +1 then +10, so the model sees 11) before calling "
        f"the last step's predict_proba: expected rows [1 - sigmoid(11), sigmoid(11)], got {proba.tolist()}")
    assert np.isclose(pipe.score(np.zeros((2, 2)), None), 22.0), "score passes the transformed X (and y) to the last step"


def test_stage5_transform_and_validation(nb):
    log = []
    pipe = nb.Pipeline([("a", _Record(1.0, log)), ("b", _Record(2.0, log))]).fit(np.zeros((2, 1)))
    assert np.allclose(pipe.transform(np.zeros((2, 1))), 3.0), "a pipeline of transformers transforms with every step"
    with pytest.raises(ValueError):
        nb.Pipeline([("a", _Record(1.0, log)), ("a", _Model(log))]).fit(np.zeros((2, 1)), np.zeros(2))
        pytest.fail("duplicate step names should raise ValueError in fit")
    with pytest.raises(TypeError):
        nb.Pipeline([("m", _Model(log)), ("m2", _Model(log))]).fit(np.zeros((2, 1)), np.zeros(2))
        pytest.fail("a middle step without transform should raise TypeError")


def test_stage5_fits_statistics_on_training_rows_only(nb):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 2)) * 5 + 3
    y = rng.integers(0, 2, 50)
    log = []
    pipe = nb.Pipeline([("scale", nb.StandardScaler()), ("m", _Model(log))]).fit(X[:30], y[:30])
    scaler = pipe.named_steps["scale"]
    assert np.allclose(scaler.mean_, X[:30].mean(0)), "the scaler inside the pipeline learns from the rows it's fitted on"
    out = pipe.predict(X[30:])
    assert np.allclose(out, (X[30:, 0] - X[:30, 0].mean()) / X[:30, 0].std()), (
        "new rows are scaled with the TRAINING statistics: that's what keeps a pipeline from leaking in cross-validation")


def test_stage5_make_pipeline_names(nb):
    p = nb.make_pipeline(nb.StandardScaler(), _Model([]))
    assert [n for n, _ in p.steps] == ["standardscaler", "_model"], f"names are lowercased class names, got {p.steps}"
    p2 = nb.make_pipeline(nb.StandardScaler(), nb.StandardScaler(), _Model([]))
    assert [n for n, _ in p2.steps] == ["standardscaler-1", "standardscaler-2", "_model"], (
        f"repeated names get suffixes -1, -2, ...; got {[n for n, _ in p2.steps]}")
    assert isinstance(p, nb.Pipeline)


# ---------------------------------------------------------------- stage 6: nested parameters


def test_stage6_get_params(nb):
    log = []
    pipe = nb.Pipeline([("scale", nb.StandardScaler(with_mean=False)), ("m", _Model(log))])
    params = pipe.get_params()
    assert params["steps"] is pipe.steps, "get_params()['steps'] is the steps list"
    assert params["scale__with_mean"] is False and params["scale__with_std"] is True and params["m__log"] is log, (
        f"every step's parameters appear as 'name__param'; got keys {sorted(params)}")
    inner = nb.Pipeline([("a", _Record(1.0, log)), ("b", _Record(2.0, log))])
    outer = nb.Pipeline([("pre", inner), ("m", _Model(log))])
    assert outer.get_params()["pre__b__add"] == 2.0, "nested pipelines nest their keys: 'pre__b__add'"


def test_stage6_set_params(nb):
    log = []
    steps = [("a", _Record(1.0, log)), ("m", _Model(log))]
    pipe = nb.Pipeline(steps)
    assert pipe.set_params(a__add=5.0) is pipe and pipe.named_steps["a"].add == 5.0, (
        "set_params(a__add=5.0) sets add on the step named 'a', and returns the pipeline")
    new = _Record(7.0, log)
    pipe.set_params(a=new)
    assert pipe.named_steps["a"] is new and [n for n, _ in pipe.steps] == ["a", "m"], (
        "set_params(a=estimator) replaces that step, keeping its name and position")
    assert steps[0][1] is not new, "replace a step by assigning a new list to self.steps, not by mutating the old list"
    inner = nb.Pipeline([("x", _Record(1.0, log)), ("y", _Record(2.0, log))])
    outer = nb.Pipeline([("pre", inner), ("m", _Model(log))])
    outer.set_params(pre__y__add=9.0)
    assert inner.named_steps["y"].add == 9.0, "a nested key is passed down: pre__y__add -> inner.set_params(y__add=...)"
    with pytest.raises(ValueError):
        pipe.set_params(b__add=1.0)
        pytest.fail("there's no step 'b': raise ValueError")


def test_stage6_clone_and_cross_validation_compatible(nb):
    from mlfs.base import clone
    log = []
    pipe = nb.Pipeline([("scale", nb.StandardScaler(with_std=False)), ("m", _Model(log))])
    pipe.fit(np.arange(8.0).reshape(4, 2), np.array([0, 1, 0, 1]))
    c = clone(pipe)
    assert type(c) is type(pipe) and c.steps[0][1] is not pipe.steps[0][1], "clone builds new step objects"
    assert c.get_params()["scale__with_std"] is False, "the clone keeps every step's hyperparameters"
    assert not hasattr(c.named_steps["scale"], "mean_"), "the clone's steps are unfitted"
