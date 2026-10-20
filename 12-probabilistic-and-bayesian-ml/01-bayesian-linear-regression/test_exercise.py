"""Checks for exercise.py (bayesian linear regression).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Oracles: the
lesson's worked example (by hand), the "function-space" (Woodbury) form of the posterior, and the
evidence as a multivariate normal density of y computed with SciPy. Randomized checks use
tolerances of several standard errors.
"""
import math
import tracemalloc

import numpy as np
from scipy.stats import multivariate_normal

from mlfs.testing import bounded


def _problem(seed, n=30, d=4, noise=0.5):
    rng = np.random.default_rng(seed)
    Phi = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = Phi @ w + noise * rng.normal(size=n)
    return Phi, y


def _woodbury(Phi, y, alpha, beta, m0=None, S0=None):
    """posterior via the n x n form: m = m0 + S0 Phi^T K^-1 (y - Phi m0), K = beta^-1 I + Phi S0 Phi^T."""
    d = Phi.shape[1]
    m0 = np.zeros(d) if m0 is None else m0
    S0 = np.eye(d) / alpha if S0 is None else S0
    K = np.eye(len(y)) / beta + Phi @ S0 @ Phi.T
    G = S0 @ Phi.T @ np.linalg.inv(K)
    return m0 + G @ (y - Phi @ m0), S0 - G @ Phi @ S0


def _evidence(Phi, y, alpha, beta):
    K = np.eye(len(y)) / beta + Phi @ Phi.T / alpha
    return multivariate_normal(np.zeros(len(y)), K).logpdf(y)


# ---------------------------------------------------------------- stage 1: basis functions


def test_stage1_polynomial_basis(nb):
    got = np.asarray(nb.polynomial_basis(np.array([0.0, 1.0, 2.0]), 2))
    want = np.array([[1, 0, 0], [1, 1, 1], [1, 2, 4]], dtype=float)
    assert got.shape == (3, 3) and np.allclose(got, want), (
        f"polynomial_basis([0, 1, 2], 2) should have columns x^0, x^1, x^2: {want.tolist()}; got {got.tolist()}")
    got0 = np.asarray(nb.polynomial_basis([0.5, -1.0], 0))
    assert got0.shape == (2, 1) and np.allclose(got0, 1), "degree 0 is a single constant column (lists work too)"


def test_stage1_gaussian_basis(nb):
    x = np.array([0.0, 1.0, 3.0])
    got = np.asarray(nb.gaussian_basis(x, [0.0, 2.0], 1.0))
    assert got.shape == (3, 3), f"3 inputs, a constant + 2 centers -> shape (3, 3); got {got.shape}"
    assert np.allclose(got[:, 0], 1), "the first column is the constant 1 (the model has no separate intercept)"
    want = np.exp(-np.array([[0, 4], [1, 1], [9, 1]]) / 2.0)
    assert np.allclose(got[:, 1:], want), (
        f"bump j is exp(-(x - c_j)^2 / (2 width^2)); expected {np.round(want, 4).tolist()}, "
        f"got {np.round(got[:, 1:], 4).tolist()}. did you forget the 2 or square the width?")
    wide = np.asarray(nb.gaussian_basis(x, [0.0], 2.0))[:, 1]
    assert np.allclose(wide, np.exp(-x**2 / 8.0)), "width is a standard deviation: exp(-x^2 / (2 * 2^2))"


# ---------------------------------------------------------------- stage 2: the posterior


def test_stage2_worked_example(nb):
    m, S = nb.posterior(np.array([[1.0], [2.0]]), np.array([1.0, 3.0]), 1.0, 1.0)
    assert np.shape(m) == (1,) and np.shape(S) == (1, 1), f"shapes (d,) and (d, d) with d = 1; got {np.shape(m)}, {np.shape(S)}"
    assert np.isclose(S[0, 0], 1 / 6) and np.isclose(m[0], 7 / 6), (
        f"lesson worked example: A = alpha + beta (1 + 4) = 6, so S = 1/6 and m = beta S (1*1 + 2*3) = 7/6; "
        f"got m = {m[0]:.4f}, S = {S[0, 0]:.4f}")


def test_stage2_matches_function_space_form(nb):
    for seed, (n, d, alpha, beta) in enumerate([(30, 4, 1.0, 4.0), (5, 8, 0.3, 10.0), (50, 2, 20.0, 0.5)]):
        Phi, y = _problem(seed, n, d)
        m, S = nb.posterior(Phi, y, alpha, beta)
        m2, S2 = _woodbury(Phi, y, alpha, beta)
        assert np.allclose(m, m2, atol=1e-8) and np.allclose(S, S2, atol=1e-8), (
            f"n={n}, d={d}, alpha={alpha}, beta={beta}: the posterior should be mean = beta S Phi^T y with "
            f"S = (alpha I + beta Phi^T Phi)^-1. largest error: mean {np.max(np.abs(m - m2)):.2e}, "
            f"cov {np.max(np.abs(S - S2)):.2e}. (n < d is fine: the prior makes A invertible)")
        assert np.allclose(S, S.T), "the posterior covariance must be symmetric"


def test_stage2_no_data_is_the_prior(nb):
    m, S = nb.posterior(np.zeros((0, 3)), np.zeros(0), 4.0, 1.0)
    assert np.allclose(m, 0) and np.allclose(S, np.eye(3) / 4.0), (
        "with no data the posterior is the prior N(0, alpha^-1 I): mean 0, cov I / 4 for alpha = 4")


def test_stage2_sequential_equals_batch(nb):
    Phi, y = _problem(3, 40, 3)
    alpha, beta = 2.0, 3.0
    m, S = np.zeros(3), np.eye(3) / alpha
    for sl in [slice(0, 1), slice(1, 15), slice(15, 40)]:
        m, S = nb.sequential_posterior(m, S, Phi[sl], y[sl], beta)
    mb, Sb = _woodbury(Phi, y, alpha, beta)
    assert np.allclose(m, mb, atol=1e-8) and np.allclose(S, Sb, atol=1e-8), (
        "updating one batch at a time (yesterday's posterior is today's prior) must give the same answer as "
        f"using all the data at once; mean error {np.max(np.abs(m - mb)):.2e}")
    rng = np.random.default_rng(7)
    B = rng.normal(size=(3, 3))
    S0 = B @ B.T + 0.5 * np.eye(3)
    m0 = rng.normal(size=3)
    got = nb.sequential_posterior(m0, S0, Phi, y, beta)
    want = _woodbury(Phi, y, None, beta, m0, S0)
    assert np.allclose(got[0], want[0], atol=1e-8) and np.allclose(got[1], want[1], atol=1e-8), (
        "with a general prior N(m0, S0): S_N^-1 = S0^-1 + beta Phi^T Phi and m_N = S_N (S0^-1 m0 + beta Phi^T y). "
        "don't assume m0 = 0 or S0 isotropic")


# ---------------------------------------------------------------- stage 3: the predictive distribution


def test_stage3_worked_example(nb):
    mu, var = nb.predictive(np.array([[3.0]]), np.array([7 / 6]), np.array([[1 / 6]]), 1.0)
    assert np.allclose(mu, [3.5]) and np.allclose(var, [2.5]), (
        f"lesson worked example at phi = 3: mean 3 * 7/6 = 3.5, variance 1/beta + 9 * 1/6 = 2.5; got "
        f"{np.round(mu, 4).tolist()}, {np.round(var, 4).tolist()}. forgot the noise term 1/beta?")


def test_stage3_oracle_and_shapes(nb):
    Phi, y = _problem(4, 25, 3)
    m, S = _woodbury(Phi, y, 1.0, 2.0)
    P = np.random.default_rng(5).normal(size=(7, 3))
    mu, var = nb.predictive(P, m, S, 2.0)
    assert np.shape(mu) == (7,) and np.shape(var) == (7,), f"one mean and one variance per row: (7,), (7,); got {np.shape(mu)}, {np.shape(var)}"
    want = np.array([0.5 + p @ S @ p for p in P])
    assert np.allclose(mu, P @ m) and np.allclose(var, want), (
        "var_i = 1/beta + phi_i^T S phi_i (only the diagonal of Phi S Phi^T, never the full n x n matrix)")


def test_stage3_uncertainty_grows_away_from_data(nb):
    rng = np.random.default_rng(6)
    x = rng.uniform(-1, 1, 20)
    Phi = np.stack([np.ones_like(x), x], axis=1)
    m, S = _woodbury(Phi, 2 * x + rng.normal(0, 0.1, 20), 1.0, 100.0)
    xs = np.array([0.0, 2.0, 5.0, 20.0])
    _, var = nb.predictive(np.stack([np.ones(4), xs], 1), m, S, 100.0)
    assert np.all(np.diff(var) > 0) and np.all(np.asarray(var) > 0.01 - 1e-12), (
        f"a straight-line model fitted on [-1, 1]: the predictive variance must grow as you move away "
        f"(and never drop below the noise 1/beta = 0.01); got {np.round(var, 4).tolist()}")


# ---------------------------------------------------------------- stage 4: the evidence


def test_stage4_worked_example(nb):
    got = nb.log_evidence(np.array([[1.0], [2.0]]), np.array([1.0, 3.0]), 1.0, 1.0)
    want = -11 / 12 - 0.5 * math.log(6) - math.log(2 * math.pi)
    assert np.isclose(got, want, atol=1e-9), (
        f"lesson worked example: y ~ N(0, [[2, 2], [2, 5]]), log p(y) = -11/12 - log(6)/2 - log(2 pi) = {want:.5f}; "
        f"got {got}")
    assert isinstance(got, float), "return a Python float"


def test_stage4_matches_gaussian_marginal(nb):
    for seed, (n, d, alpha, beta) in enumerate([(30, 4, 1.0, 4.0), (6, 9, 0.3, 10.0), (60, 2, 50.0, 0.2)]):
        Phi, y = _problem(10 + seed, n, d)
        got = nb.log_evidence(Phi, y, alpha, beta)
        want = _evidence(Phi, y, alpha, beta)
        assert np.isclose(got, want, atol=1e-7), (
            f"n={n}, d={d}, alpha={alpha}, beta={beta}: integrating w out, y ~ N(0, beta^-1 I + alpha^-1 Phi Phi^T), "
            f"so log p(y) = {want:.6f}; got {got:.6f}. check the d/2 log alpha and n/2 log beta terms")


def test_stage4_no_n_by_n_matrix(nb):
    rng = np.random.default_rng(8)
    Phi = rng.normal(size=(4000, 3))
    y = Phi @ np.ones(3) + rng.normal(size=4000)
    tracemalloc.start()
    try:
        nb.log_evidence(Phi, y, 1.0, 1.0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16e6, (
        f"log_evidence on n = 4000, d = 3 allocated {peak / 1e6:.0f} MB at its peak; an n x n covariance would be "
        "128 MB. use the d x d posterior: log p(y) = d/2 log a + n/2 log b - E(m) - 1/2 log|A| - n/2 log 2 pi")


# ---------------------------------------------------------------- stage 5: the estimator


def test_stage5_estimator_protocol(nb):
    from mlfs.base import Regressor, clone

    est = nb.BayesianLinearRegression(alpha=2.0, beta=3.0, optimize=True, max_iters=7)
    assert isinstance(est, Regressor), "BayesianLinearRegression subclasses mlfs.base.Regressor"
    assert est.get_params() == {"alpha": 2.0, "beta": 3.0, "optimize": True, "max_iters": 7}, (
        f"__init__ stores its arguments unchanged; get_params() returned {est.get_params()}")
    c = clone(est)
    assert c.get_params() == est.get_params() and not hasattr(c, "mean_"), "clone gives an unfitted copy"


def test_stage5_fit_predict(nb):
    rng = np.random.default_rng(20)
    Phi = rng.normal(size=(60, 3))
    y = Phi @ np.array([2.0, -1.5, 1.0]) + rng.normal(0, 0.3, 60)
    est = nb.BayesianLinearRegression(alpha=0.5, beta=10.0)
    assert est.fit(Phi, y) is est, "fit returns self"
    m, S = _woodbury(Phi, y, 0.5, 10.0)
    assert np.allclose(est.mean_, m) and np.allclose(est.cov_, S), "mean_ and cov_ are the posterior under (alpha, beta)"
    assert est.alpha_ == 0.5 and est.beta_ == 10.0, "without optimize, alpha_ and beta_ are the constructor's values"
    P = np.random.default_rng(1).normal(size=(5, 3))
    mu = est.predict(P)
    mu2, sd = est.predict(P, return_std=True)
    assert np.shape(mu) == (5,) and np.allclose(mu, P @ m) and np.allclose(mu2, mu), "predict returns the predictive mean"
    assert np.allclose(sd, np.sqrt(0.1 + np.einsum("ij,jk,ik->i", P, S, P))), (
        "return_std=True gives (mean, std) where std = sqrt(1/beta + phi^T S phi): include the noise")
    assert est.score(Phi, y) > 0.9, "R^2 on this nearly noise-free problem should be above 0.9"
    assert np.isclose(est.log_evidence(Phi, y), _evidence(Phi, y, 0.5, 10.0)), "log_evidence(X, y) at alpha_, beta_"


def test_stage5_sample_weights(nb):
    rng = np.random.default_rng(21)
    z = rng.normal(size=15)
    Phi = np.stack([z, z + 0.3 * rng.normal(size=15), -z + 0.3 * rng.normal(size=15)], axis=1)  # correlated columns
    y = Phi @ np.array([1.0, -0.5, 0.3]) + rng.normal(size=15)
    est = nb.BayesianLinearRegression(alpha=1.0, beta=1.0).fit(Phi, y)
    W = np.asarray(est.sample_weights(6000, rng=0))
    assert W.shape == (6000, 3), f"sample_weights(n) returns (n, d); got {W.shape}"
    se = np.sqrt(np.diag(est.cov_) / 6000)
    assert np.all(np.abs(W.mean(0) - est.mean_) < 5 * se), (
        f"the sample mean {np.round(W.mean(0), 3).tolist()} should be within 5 standard errors of mean_ "
        f"{np.round(est.mean_, 3).tolist()}")
    assert np.allclose(np.cov(W.T), est.cov_, rtol=0.12, atol=0.12 * np.max(np.diag(est.cov_))), (
        "the sample covariance should match cov_: draw mean_ + Z @ L.T with L the lower Cholesky factor of cov_ "
        "(Z @ L would give covariance L^T L, which differs from L L^T for a correlated posterior like this one)")
    W2 = np.asarray(est.sample_weights(3, rng=5))
    assert np.allclose(W2, est.sample_weights(3, rng=5)), "the same seed gives the same draws"


def test_stage5_evidence_maximization(nb):
    rng = np.random.default_rng(22)
    x = rng.uniform(-1, 1, 800)
    Phi = np.stack([np.ones_like(x), x, x**2], 1)
    y = Phi @ np.array([0.5, -1.0, 2.0]) + rng.normal(0, 0.3, 800)
    est = bounded(nb.BayesianLinearRegression(alpha=1.0, beta=1.0, optimize=True).fit, Phi, y,
                  what="fit(optimize=True)", hint="cap the fixed-point loop at max_iters.")
    a, b = est.alpha_, est.beta_
    assert 0.75 * (1 / 0.09) < b < 1.25 * (1 / 0.09), (
        f"with noise sd 0.3 the noise precision is about 1/0.09 = 11.1, and 800 points pin it down to a few "
        f"percent; got beta_ = {b:.3g}")
    h = 1e-4
    for name, f in [("log alpha", lambda t: _evidence(Phi, y, a * math.exp(t), b)),
                    ("log beta", lambda t: _evidence(Phi, y, a, b * math.exp(t)))]:
        slope = (f(h) - f(-h)) / (2 * h)
        assert abs(slope) < 1e-2, (
            f"at the fixed point the evidence is stationary: d log p(y) / d {name} should be ~0, got {slope:.3g}. "
            "gamma = sum lambda_i / (alpha + lambda_i) with lambda_i the eigenvalues of beta Phi^T Phi; "
            "alpha = gamma / m^T m; beta = (n - gamma) / ||y - Phi m||^2")
    assert np.isclose(est.log_evidence(Phi, y), _evidence(Phi, y, a, b)), "log_evidence(X, y) uses the fitted alpha_, beta_"
    assert est.log_evidence(Phi, y) > _evidence(Phi, y, 1.0, 1.0), "optimizing must increase the evidence"
    assert np.allclose(est.mean_, _woodbury(Phi, y, a, b)[0]), "mean_ is the posterior under the optimized alpha_, beta_"
    # many weak features, few points: the effective number of parameters gamma is well below d
    Phi2 = rng.normal(size=(25, 12))
    y2 = Phi2[:, 0] * 1.5 - Phi2[:, 1] + rng.normal(0, 0.5, 25)
    est2 = bounded(nb.BayesianLinearRegression(optimize=True, max_iters=1000).fit, Phi2, y2, what="fit(optimize=True)")
    for name, f in [("log alpha", lambda t: _evidence(Phi2, y2, est2.alpha_ * math.exp(t), est2.beta_)),
                    ("log beta", lambda t: _evidence(Phi2, y2, est2.alpha_, est2.beta_ * math.exp(t)))]:
        slope = (f(h) - f(-h)) / (2 * h)
        assert abs(slope) < 1e-2, (
            f"25 points, 12 features, only 2 of them used: d log p(y) / d {name} should be ~0 at the fixed point, got "
            f"{slope:.3g}. here gamma (the effective number of parameters) is well below d = 12, so using d instead "
            "of gamma = sum lambda_i / (alpha + lambda_i) gives the wrong fixed point")
    short = nb.BayesianLinearRegression(optimize=True, max_iters=2).fit(Phi, y)
    assert short.n_iter_ <= 2, f"stop after max_iters fixed-point updates; n_iter_ = {short.n_iter_}"
