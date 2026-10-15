"""Checks for exercise.py (bias-variance and complexity).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The small
estimators below don't use mlfs.base: they implement just enough of the protocol
(get_params/set_params/fit/predict/score) for clone and cross_validate.
"""
import numpy as np
import pytest

from mlfs.testing import bounded


class _Mean:
    """predicts the mean of the training targets (+ shift); score = -MSE."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def get_params(self):
        return {"shift": self.shift}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self.n_fit_ = len(y)
        self.mean_ = float(np.mean(y)) + self.shift
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    def score(self, X, y):
        return -float(np.mean((np.asarray(y) - self.predict(X)) ** 2))


class _Line(_Mean):
    """least-squares line through the training points."""

    def fit(self, X, y):
        self.c_ = np.polyfit(np.asarray(X, dtype=float)[:, 0], y, 1)
        return self

    def predict(self, X):
        return np.polyval(self.c_, np.asarray(X, dtype=float)[:, 0])


def _two_points(sigma):
    """the lesson's worked example: f(x) = x observed at x = 0 and 1 with noise sigma."""
    def sample(gen):
        X = np.array([[0.0], [1.0]])
        return X, X[:, 0] + sigma * gen.normal(size=2)
    return sample


# ---------------------------------------------------------------- stage 1: PolynomialRegressor


def test_stage1_recovers_a_polynomial(nb):
    x = np.linspace(-1, 1, 15)
    y = 2 - x + 0.5 * x**3
    m = nb.PolynomialRegressor(degree=3).fit(x.reshape(-1, 1), y)
    assert np.allclose(m.coef_, [2, -1, 0, 0.5]), (
        f"fitting a degree-3 polynomial to exact cubic data should recover coef_ = [2, -1, 0, 0.5] (lowest power first); "
        f"got {np.round(m.coef_, 4).tolist()}")
    xt = np.array([[-0.3], [0.7]])
    assert np.allclose(m.predict(xt), 2 - xt[:, 0] + 0.5 * xt[:, 0] ** 3), "predict evaluates the polynomial"
    assert np.isclose(m.score(x.reshape(-1, 1), y), 1.0), "PolynomialRegressor is a Regressor: score is R^2"


def test_stage1_matches_polyfit_and_degree_zero(nb):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 25)
    y = np.sin(3 * x) + 0.1 * rng.normal(size=25)
    for d in (0, 1, 4, 7):
        m = nb.PolynomialRegressor(degree=d).fit(x, y)  # a 1-D X is one feature (check_X_y)
        assert np.asarray(m.coef_).shape == (d + 1,), f"degree {d} has {d + 1} coefficients"
        assert np.allclose(m.coef_, np.polyfit(x, y, d)[::-1], atol=1e-8), (
            f"degree {d}: coef_ should be the least-squares coefficients (np.polyfit reversed)")
    assert np.allclose(nb.PolynomialRegressor(degree=0).fit(x, y).predict(x[:3]), y.mean()), "degree 0 predicts the mean"
    assert nb.PolynomialRegressor(degree=5).get_params() == {"degree": 5}
    with pytest.raises(ValueError):
        nb.PolynomialRegressor().fit(np.ones((5, 2)), np.ones(5))
        pytest.fail("two features should raise ValueError: this regressor takes one")


# ---------------------------------------------------------------- stage 2: bias-variance decomposition


@pytest.mark.parametrize("seed", range(3))
def test_stage2_worked_example(nb, seed):
    const = nb.bias_variance_decomposition(_Mean(), _two_points(1.0), np.array([[0.0], [1.0]]),
                                           np.array([0.0, 1.0]), n_rounds=4000, noise_var=1.0, rng=seed)
    assert np.isclose(const["bias2"], 0.25, atol=0.03) and np.isclose(const["variance"], 0.5, atol=0.04), (
        f"the mean predictor on two points has bias^2 = 0.25 and variance = sigma^2/2 = 0.5 (worked example); got "
        f"bias2 = {const['bias2']:.3f}, variance = {const['variance']:.3f}")
    line = nb.bias_variance_decomposition(_Line(), _two_points(1.0), np.array([[0.0], [1.0]]), np.array([0.0, 1.0]),
                                          n_rounds=4000, noise_var=1.0, rng=seed)
    assert line["bias2"] < 0.02 and np.isclose(line["variance"], 1.0, atol=0.08), (
        f"the line through both points is unbiased with variance sigma^2 = 1; got bias2 = {line['bias2']:.3f}, "
        f"variance = {line['variance']:.3f}")
    assert const["expected_error"] < line["expected_error"], (
        "with sigma^2 = 1 the biased constant (1.75) beats the unbiased line (2.0): that's the trade-off")


def test_stage2_bookkeeping(nb):
    out = bounded(nb.bias_variance_decomposition, _Mean(), _two_points(0.5), np.array([[0.0], [0.5], [1.0]]),
                  np.array([0.0, 0.5, 1.0]), n_rounds=50, noise_var=0.25, rng=1, budget=200_000,
                  what="bias_variance_decomposition", hint="run exactly n_rounds rounds.")
    assert set(out) == {"bias2", "variance", "noise", "expected_error"}, f"return exactly these four keys, got {sorted(out)}"
    assert all(isinstance(v, float) for v in out.values()), "all four values should be Python floats"
    assert out["noise"] == 0.25 and np.isclose(out["expected_error"], out["bias2"] + out["variance"] + 0.25), (
        "noise is noise_var, and expected_error = bias2 + variance + noise")
    again = nb.bias_variance_decomposition(_Mean(), _two_points(0.5), np.array([[0.0], [0.5], [1.0]]),
                                           np.array([0.0, 0.5, 1.0]), n_rounds=50, noise_var=0.25, rng=1)
    assert out == again, "the same rng seed must reproduce the same numbers: draw everything from as_rng(rng)"
    est = _Mean(shift=0.0)
    nb.bias_variance_decomposition(est, _two_points(0.5), np.array([[0.0]]), np.array([0.0]), n_rounds=3, rng=0)
    assert not hasattr(est, "mean_"), "fit clone(estimator) in each round; the estimator you're given stays unfitted"


def test_stage2_constant_model_has_no_variance(nb):
    seen = []

    def sample(gen):
        seen.append(type(gen).__name__)
        return np.zeros((4, 1)), gen.normal(size=4)

    out = nb.bias_variance_decomposition(_Mean(shift=3.0), lambda g: (np.zeros((4, 1)), np.zeros(4)),
                                         np.array([[0.0], [1.0]]), np.array([1.0, 2.0]), n_rounds=5, rng=0)
    assert np.isclose(out["variance"], 0.0) and np.isclose(out["bias2"], (4 + 1) / 2), (
        "a model whose prediction never changes has zero variance; here it predicts 3 everywhere against f = (1, 2), "
        f"so bias2 = (2^2 + 1^2)/2 = 2.5; got {out}")
    nb.bias_variance_decomposition(_Mean(), sample, np.array([[0.0]]), np.array([0.0]), n_rounds=3, rng=0)
    assert seen == ["Generator"] * 3, "call sample_data(gen) once per round with a numpy Generator (as_rng(rng))"
    rounds = iter(range(3))

    def counting(gen):  # round r's training targets are all r, so the mean predictor predicts 0, 1, 2
        r = next(rounds)
        return np.zeros((2, 1)), np.full(2, float(r))

    out = nb.bias_variance_decomposition(_Mean(), counting, np.array([[0.0]]), np.array([1.0]), n_rounds=3)
    assert np.isclose(out["bias2"], 0.0) and np.isclose(out["variance"], 2 / 3), (
        f"predictions 0, 1, 2 around f = 1: bias2 = 0 and variance = mean((p - 1)^2) = 2/3 (population variance, "
        f"ddof=0, so that bias2 + variance is exactly the mean squared distance to f); got {out}")


# ---------------------------------------------------------------- stage 3: learning_curve


def test_stage3_sizes_and_prefixes(nb):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 1))
    y = rng.normal(size=20)
    sizes, tr, va = nb.learning_curve(_Mean(), X, y, train_sizes=(0.25, 0.6, 1.0), cv=4)
    sizes, tr, va = np.asarray(sizes), np.asarray(tr), np.asarray(va)
    assert sizes.tolist() == [3, 9, 15] and sizes.dtype.kind == "i", (
        "4 folds of 20 rows leave 15 training rows; fractions (0.25, 0.6, 1.0) give sizes int(3.75), 9, 15 = [3, 9, 15] "
        f"(round DOWN); got {sizes.tolist()}")
    assert tr.shape == va.shape == (3, 4), f"scores are (n_sizes, n_folds) = (3, 4), got {tr.shape} and {va.shape}"
    folds = [np.setdiff1d(np.arange(20), np.arange(5 * f, 5 * f + 5)) for f in range(4)]
    for j, train_idx in enumerate(folds):
        test_idx = np.arange(5 * j, 5 * j + 5)
        for i, s in enumerate([3, 9, 15]):
            sub = train_idx[:s]
            want_tr = -np.mean((y[sub] - y[sub].mean()) ** 2)
            want_va = -np.mean((y[test_idx] - y[sub].mean()) ** 2)
            assert np.isclose(tr[i, j], want_tr) and np.isclose(va[i, j], want_va), (
                f"fold {j}, size {s}: fit on the FIRST {s} training indices of the fold, score on them (train) and on "
                "the fold's test rows (validation)")


def test_stage3_uses_the_smallest_fold_and_scoring(nb):
    X = np.arange(10.0).reshape(-1, 1)
    y = np.arange(10.0)
    sizes, tr, va = nb.learning_curve(_Mean(), X, y, train_sizes=(0.5, 1.0), cv=3,
                                      scoring=lambda est, X, y: float(est.n_fit_))
    assert np.asarray(sizes).tolist() == [3, 6], (
        "10 rows in 3 folds: the training folds have 6, 7 and 7 rows, so n_max is the SMALLEST, 6, and the sizes are "
        f"[3, 6]; got {np.asarray(sizes).tolist()}")
    assert np.all(np.asarray(va)[0] == 3) and np.all(np.asarray(tr)[1] == 6), (
        "pass scoring on to cross_validate (here it reports how many rows each model was fitted on)")
    with pytest.raises(ValueError):
        nb.learning_curve(_Mean(), X, y, train_sizes=(0.0, 1.0), cv=3)
        pytest.fail("a training size of 0 should raise ValueError")


# ---------------------------------------------------------------- stage 4: validation_curve


def test_stage4_validation_curve(nb):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 1))
    y = rng.normal(size=12) + 1.0
    est = _Mean(shift=0.0)
    shifts = [-1.0, 0.0, 0.5]
    tr, va = nb.validation_curve(est, X, y, "shift", shifts, cv=3)
    tr, va = np.asarray(tr), np.asarray(va)
    assert tr.shape == va.shape == (3, 3), f"scores are (n_values, n_folds) = (3, 3), got {tr.shape}, {va.shape}"
    for i, s in enumerate(shifts):
        for f in range(3):
            test_idx = np.arange(4 * f, 4 * f + 4)
            train_idx = np.setdiff1d(np.arange(12), test_idx)
            pred = y[train_idx].mean() + s
            assert np.isclose(va[i, f], -np.mean((y[test_idx] - pred) ** 2)), (
                f"value shift={s}, fold {f}: the validation score should come from clone(estimator).set_params(shift={s}) "
                "fitted on that fold")
            assert np.isclose(tr[i, f], -np.mean((y[train_idx] - pred) ** 2)), "train scores on the fold's training rows"
    assert est.shift == 0.0 and not hasattr(est, "mean_"), "never modify or fit the estimator you're given: clone it"
    assert np.argmax(va.mean(axis=1)) == 1, "the unshifted mean predictor should have the best validation score"


def test_stage4_same_folds_for_every_value(nb):
    from mlfs.model_selection import KFold
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 1))
    y = rng.normal(size=30)
    gen = np.random.default_rng(5)  # a Generator: each call to split() would give different folds
    tr, va = nb.validation_curve(_Mean(), X, y, "shift", [0.0, 0.0, 0.0], cv=KFold(5, shuffle=True, rng=gen))
    va = np.asarray(va)
    assert np.allclose(va[0], va[1]) and np.allclose(va[0], va[2]), (
        "with three identical values the three rows must be identical: compute the splits ONCE "
        "(materialize_splits) so every value is scored on the same folds")
