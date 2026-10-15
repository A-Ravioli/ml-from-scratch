"""Checks for exercise.py (bounds in practice).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.

Bounds are checked against hand-computed values with the constants stated in the lesson, the
test-set bound against scipy's beta quantile and its defining binomial-tail property, the
sample-size calculator by its defining property (smallest n with bound <= epsilon) under
mlfs.testing.bounded (a linear scan up to 10^6 runs out of budget), and the report with a
deterministic counting model.
"""
import math

import numpy as np
from scipy import stats

from mlfs.testing import bounded


def _vc(n, d, delta, emp=0.0):
    g = 0.0 if d == 0 else (min(d * math.log(2 * math.e * n / d), 2 * n * math.log(2)) if 2 * n >= d else 2 * n * math.log(2))
    return emp + math.sqrt(8 / n * (g + math.log(4 / delta)))


# ---------------------------------------------------------------- stage 1: finite classes


def test_stage1_finite_class_bound(nb):
    got = nb.finite_class_bound(1000, 1000, 0.05, 0.1)
    want = 0.1 + math.sqrt((math.log(1000) + math.log(20)) / 2000)
    assert np.isclose(got, want), (
        f"n = 1000, |H| = 1000, delta = 0.05, empirical risk 0.1: 0.1 + sqrt((ln 1000 + ln 20) / 2000) = {want:.5f}; "
        f"got {got}. it's the one-sided Hoeffding deviation sqrt(ln(1/delta') / (2n)) with delta' = delta / |H|")
    assert np.isclose(nb.finite_class_bound(500, 1, 0.1), math.sqrt(math.log(10) / 1000)), (
        "one hypothesis: plain Hoeffding, sqrt(ln(1/delta) / (2n)); emp_risk defaults to 0")
    H = 2 ** (32 * 785)
    got = nb.finite_class_bound(60000, H, 0.05, 0.02)
    want = 0.02 + math.sqrt((32 * 785 * math.log(2) + math.log(20)) / 120000)
    assert np.isclose(got, want), (
        f"|H| = 2^(32 * 785) (a float32 model on 784 pixels): expected {want:.5f}, got {got}. use math.log on the Python "
        "int; np.log(H) overflows")
    assert isinstance(got, float), "return a Python float"


# ---------------------------------------------------------------- stage 2: VC


def test_stage2_vc_bound_values(nb):
    got = nb.vc_bound(10000, 10, 0.05)
    want = math.sqrt(8 / 10000 * (10 * math.log(2 * math.e * 10000 / 10) + math.log(80)))
    assert np.isclose(got, want), (
        f"vc_bound(10000, 10, 0.05) = sqrt((8/n)(d ln(2en/d) + ln(4/delta))) = {want:.5f}; got {got}. "
        "the 8, the 2en/d and the 4/delta all matter")
    assert np.isclose(nb.vc_bound(10000, 10, 0.05, 0.12), 0.12 + want), "add the empirical risk"
    assert np.isclose(nb.vc_bound(400, 0, 0.1), math.sqrt(8 / 400 * math.log(40))), (
        "VC dimension 0 (one hypothesis): d ln(2en/d) -> 0 as d -> 0; don't divide by zero")


def test_stage2_small_samples_and_monotonicity(nb):
    got = nb.vc_bound(10, 100, 0.05)
    assert math.isfinite(got) and got >= 1, (
        f"n = 10 points, VC dimension 100: Sauer's lemma needs 2n >= d; for 2n < d use ln Pi(2n) <= 2n ln 2 (all "
        f"labelings). the bound must be finite and vacuous (>= 1); got {got}")
    assert np.isclose(got, _vc(10, 100, 0.05)), f"expected {_vc(10, 100, 0.05):.4f} with ln Pi(2n) = 2n ln 2, got {got}"
    small = [nb.vc_bound(n, 10, 0.05) for n in range(1, 40)]
    assert all(a >= b for a, b in zip(small, small[1:])), (
        f"near 2n = d the two growth-function bounds cross: use ln Pi(2n) <= min(d ln(2en/d), 2n ln 2) for 2n >= d, so "
        f"the bound never grows with n; got {[round(v, 3) for v in small[:8]]} for n = 1..8, d = 10")
    vals = [nb.vc_bound(n, 20, 0.05) for n in (20, 100, 1000, 10**4, 10**5, 10**6)]
    assert all(a > b for a, b in zip(vals, vals[1:])), f"the bound must shrink as n grows (n >= d): {vals}"
    assert nb.vc_bound(1000, 5, 0.05) < nb.vc_bound(1000, 50, 0.05) < nb.vc_bound(1000, 50, 0.001)


# ---------------------------------------------------------------- stage 3: the test-set bound


def test_stage3_test_set_bound(nb):
    got = nb.test_set_bound(0, 100, 0.05)
    want = 1 - 0.05 ** (1 / 100)
    assert np.isclose(got, want, atol=1e-7), (
        f"0 errors in 100 test examples, delta = 0.05: the largest p with (1 - p)^100 >= 0.05 is {want:.5f}; got {got}")
    for k, n, delta in [(3, 50, 0.05), (17, 1000, 0.01), (250, 2000, 0.05), (1, 10, 0.2)]:
        got = nb.test_set_bound(k, n, delta)
        want = stats.beta.ppf(1 - delta, k + 1, n - k)
        assert np.isclose(got, want, atol=1e-6), (
            f"test_set_bound({k}, {n}, {delta}): expected {want:.6f}, got {got}. find the p at which "
            "P(Binomial(n, p) <= k) falls to delta (bisection on scipy.stats.binom.cdf, or the beta quantile)")
        assert np.isclose(stats.binom.cdf(k, n, got), delta, atol=1e-5), "at the bound, the binomial tail equals delta"
        hoeff = k / n + math.sqrt(math.log(1 / delta) / (2 * n))
        assert k / n < got <= hoeff + 1e-9, (
            f"the exact binomial bound lies above the observed rate {k / n:.4f} and below the Hoeffding bound {hoeff:.4f}; "
            f"got {got:.4f}")
    assert nb.test_set_bound(20, 20, 0.05) == 1.0, "every test example wrong: the bound is 1"


# ---------------------------------------------------------------- stage 4: how much data the VC bound needs


def test_stage4_vc_sample_size(nb):
    for d, eps, delta in [(10, 0.1, 0.05), (1, 0.3, 0.1), (50, 0.2, 0.01), (100, 0.05, 0.05)]:
        n = bounded(nb.vc_sample_size, d, eps, delta, what="vc_sample_size", budget=200_000,
                    hint="the answer can be in the millions: double n until the bound is <= epsilon, then bisect")
        assert isinstance(n, int), f"return an int, got {type(n).__name__}"
        assert _vc(n, d, delta) <= eps < _vc(n - 1, d, delta), (
            f"d = {d}, epsilon = {eps}, delta = {delta}: n must be the SMALLEST sample size with vc_bound(n) <= epsilon. "
            f"got n = {n}: bound(n) = {_vc(n, d, delta):.5f}, bound(n - 1) = {_vc(n - 1, d, delta):.5f}")
    assert nb.vc_sample_size(3, 6.0, 0.05) == 3, "if even n = d satisfies the bound, return max(d, 1)"
    assert nb.vc_sample_size(0, 0.1, 0.05) == math.ceil(8 * math.log(80) / 0.01), (
        "VC dimension 0: sqrt(8 ln(4/delta) / n) <= epsilon gives n = ceil(8 ln(80) / epsilon^2) = 3506")


# ---------------------------------------------------------------- stage 5: the report


class _Stump:
    """predicts 1[x[:, 0] >= t]; counts fits."""

    def __init__(self, t):
        self.t, self.fits = t, []

    def fit(self, X, y):
        self.fits.append((X, y))
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= self.t).astype(int)


def test_stage5_report(nb):
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.random((400, 2)), rng.random((300, 2))
    ytr = ((Xtr[:, 0] >= 0.5) ^ (rng.random(400) < 0.1)).astype(int)
    yte = ((Xte[:, 0] >= 0.5) ^ (rng.random(300) < 0.1)).astype(int)
    model = _Stump(0.55)
    rep = nb.generalization_report(model, Xtr, ytr, Xte, yte, vc_dim=1, n_params=1, delta=0.05)
    assert len(model.fits) == 1 and model.fits[0][0] is Xtr, "fit the model exactly once, on the training set"
    tr = float(np.mean(model.predict(Xtr) != ytr))
    k = int(np.sum(model.predict(Xte) != yte))
    want = {
        "n_train": 400, "n_test": 300, "train_error": tr, "test_error": k / 300, "gap": k / 300 - tr,
        "vc_bound": _vc(400, 1, 0.05, tr),
        "finite_bound": tr + math.sqrt((32 * math.log(2) + math.log(20)) / 800),
        "test_bound": stats.beta.ppf(0.95, k + 1, 300 - k),
    }
    missing = set(want) - set(rep)
    assert not missing, f"the report is missing {sorted(missing)}"
    for key, val in want.items():
        assert np.isclose(rep[key], val, atol=1e-6), (
            f"report[{key!r}] should be {val:.5f}, got {rep[key]}. vc_bound and finite_bound use the TRAINING error and "
            "n_train (|H| = 2^(bits * n_params)); test_bound uses the number of TEST errors and n_test")


def test_stage5_with_a_real_model(nb):
    from mlfs.linear import LogisticRegression
    rng = np.random.default_rng(1)
    w = rng.normal(size=5)
    w *= 3 / np.linalg.norm(w)
    X = rng.normal(size=(3000, 5))
    y = ((X @ w + 0.5 * rng.normal(size=3000)) > 0).astype(int)
    rep = nb.generalization_report(LogisticRegression(), X[:2000], y[:2000], X[2000:], y[2000:], vc_dim=6, n_params=6)
    assert rep["test_error"] < 0.2 and abs(rep["gap"]) < 0.05, f"logistic regression should fit this easily: {rep}"
    assert rep["test_error"] <= rep["test_bound"] < rep["test_error"] + 0.05, (
        "the test-set bound sits just above the test error")
    assert rep["vc_bound"] > rep["test_bound"] + 0.1, "the VC bound is far looser than the test-set bound"
