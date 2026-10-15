"""Checks for exercise.py (gaussian processes).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scikit-learn's
GaussianProcessRegressor and SciPy are used only as oracles. Hyperparameter optimization runs
inside `bounded`, so a runaway loop fails with a message instead of hanging.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded


def _wiggle(n, seed, noise=0.2):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 5, size=(n, 1))
    return X, np.sin(2 * X[:, 0]) + noise * rng.normal(size=n) + 3.0


def _sk_kernel(kind, ls, var):
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    base = {"rbf": RBF(ls), "matern32": Matern(ls, nu=1.5), "matern52": Matern(ls, nu=2.5)}[kind]
    return ConstantKernel(var) * base


# ---------------------------------------------------------------- stage 1: kernels


def test_stage1_hand_values(nb):
    x, y = np.array([[0.0]]), np.array([[1.0]])
    cases = [("rbf", 2 * math.exp(-0.5)), ("matern32", 2 * (1 + math.sqrt(3)) * math.exp(-math.sqrt(3))),
             ("matern52", 2 * (1 + math.sqrt(5) + 5 / 3) * math.exp(-math.sqrt(5)))]
    for kind, want in cases:
        got = np.asarray(nb.stationary_kernel(x, y, kind=kind, lengthscale=1.0, variance=2.0))
        assert got.shape == (1, 1) and np.isclose(got[0, 0], want), (
            f"{kind} at distance 1 with lengthscale 1, variance 2 should be {want:.6f}, got {got.ravel()}")


@pytest.mark.parametrize("kind", ["rbf", "matern32", "matern52"])
def test_stage1_matches_sklearn(nb, kind):
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(8, 3)), rng.normal(size=(5, 3))
    got = np.asarray(nb.stationary_kernel(X, Y, kind=kind, lengthscale=0.7, variance=1.5))
    assert got.shape == (8, 5)
    assert np.allclose(got, _sk_kernel(kind, 0.7, 1.5)(X, Y)), (
        f"{kind}: compare your formula with idea 1 (r = ||x - y|| / lengthscale, times the variance)")
    K = np.asarray(nb.stationary_kernel(X, kind=kind, lengthscale=0.7, variance=1.5))
    assert np.allclose(np.diag(K), 1.5), "Y defaults to X, and k(x, x) is the variance"
    with pytest.raises(ValueError):
        nb.stationary_kernel(X, kind="matern12")
        pytest.fail("an unknown kind should raise ValueError")


# ---------------------------------------------------------------- stage 2: the posterior


def test_stage2_worked_example(nb):
    gp = nb.GaussianProcessRegressor(lengthscale=1.0, variance=1.0, noise=0.1, optimize=False)
    assert gp.fit(np.array([[0.0], [1.0]]), np.array([1.0, 3.0])) is gp, "fit returns self"
    assert np.isclose(gp.y_mean_, 2.0) and np.allclose(gp.y_train_, [-1.0, 1.0]), "the targets are centred by their mean"
    assert np.allclose(gp.alpha_, [-2.026472, 2.026472], atol=1e-5), (
        f"alpha_ = (K + 0.1 I)^-1 (-1, 1) = (-2.0265, 2.0265), got {gp.alpha_}")
    mean, std = gp.predict(np.array([[0.5], [0.0]]), return_std=True)
    assert np.allclose(mean, [2.0, 1.202658], atol=1e-5), (
        f"the lesson's posterior means: 2 at x = 0.5 (by symmetry) and 1.2027 at x = 0. got {mean}. don't forget "
        "to add y_mean_ back")
    assert np.isclose(std[0], 0.295417, atol=1e-5), f"the posterior std at 0.5 is 0.2954, got {std[0]}"


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_stage2_matches_sklearn(nb, kind):
    from sklearn.gaussian_process import GaussianProcessRegressor
    X, y = _wiggle(25, 2)
    Q = np.linspace(-1, 6, 30)[:, None]
    gp = nb.GaussianProcessRegressor(kernel=kind, lengthscale=0.8, variance=0.6, noise=0.05, optimize=False).fit(X, y)
    ref = GaussianProcessRegressor(_sk_kernel(kind, 0.8, 0.6), alpha=0.05 + 1e-8, optimizer=None).fit(X, y - y.mean())
    m_ref, s_ref = ref.predict(Q, return_std=True)
    m, s = gp.predict(Q, return_std=True)
    assert np.allclose(m, m_ref + y.mean(), atol=1e-6), f"{kind}: the posterior mean is k*^T (K + noise I)^-1 y + y_mean"
    assert np.allclose(s, s_ref, atol=1e-6), (
        f"{kind}: the posterior std of f is sqrt(k(x, x) - v^T v) with v = L^-1 k* (it excludes the observation noise)")
    _, cov = gp.predict(Q, return_cov=True)
    assert np.allclose(cov, ref.predict(Q, return_cov=True)[1], atol=1e-6), "return_cov gives the full posterior covariance"
    assert np.allclose(gp.predict(Q), m), "without return_std / return_cov, predict returns the mean alone"


def test_stage2_prior_and_far_away(nb):
    gp = nb.GaussianProcessRegressor(lengthscale=0.5, variance=4.0, optimize=False)
    mean, std = gp.predict(np.array([[0.0], [3.0]]), return_std=True)
    assert np.allclose(mean, 0.0) and np.allclose(std, 2.0), (
        "an unfitted GP predicts its prior: mean 0 and std sqrt(variance) everywhere")
    X, y = _wiggle(20, 3)
    gp.fit(X, y)
    m, s = gp.predict(np.array([[100.0]]), return_std=True)
    assert np.isclose(m[0], y.mean()) and np.isclose(s[0], 2.0), (
        "far from the data the posterior returns to the prior: the training mean of y and std sqrt(variance)")


# ---------------------------------------------------------------- stage 3: sampling


def test_stage3_prior_samples(nb):
    gp = nb.GaussianProcessRegressor(kernel="matern32", lengthscale=1.0, variance=2.0, optimize=False)
    X = np.array([[0.0], [0.5], [1.0], [3.0]])
    S = np.asarray(gp.sample_y(X, 30000, rng=0))
    assert S.shape == (30000, 4), f"sample_y(X, n_samples) returns (n_samples, m) = (30000, 4), got {S.shape}"
    K = nb.stationary_kernel(X, kind="matern32", lengthscale=1.0, variance=2.0)
    assert np.allclose(np.cov(S.T), K, atol=0.15) and np.allclose(S.mean(axis=0), 0, atol=0.1), (
        "an unfitted GP's samples are N(0, K): their sample covariance should match the kernel matrix")
    assert np.allclose(gp.sample_y(X, 3, rng=5), gp.sample_y(X, 3, rng=5)), "the same rng gives the same samples"
    assert np.asarray(gp.sample_y(X, rng=1)).shape == (1, 4), "n_samples defaults to 1"


def test_stage3_posterior_samples(nb):
    X, y = _wiggle(12, 4, noise=0.0)  # consistent (noise-free) targets, so the posterior can pass through them
    gp = nb.GaussianProcessRegressor(lengthscale=0.7, variance=1.0, noise=1e-6, optimize=False).fit(X, y)
    S = np.asarray(gp.sample_y(X, 50, rng=0))
    assert np.allclose(S, y, atol=0.05), (
        "with noise-free data and a tiny noise variance, every posterior sample passes (almost) through the observations")
    Q = np.linspace(0, 5, 40)[:, None]
    S = np.asarray(gp.sample_y(Q, 4000, rng=1))
    mean, std = gp.predict(Q, return_std=True)
    assert np.allclose(S.mean(axis=0), mean, atol=0.05) and np.allclose(S.std(axis=0), std, atol=0.05), (
        "posterior samples are N(mean, cov) from predict(return_cov=True): their mean and std must match")


# ---------------------------------------------------------------- stage 4: the evidence


def test_stage4_log_marginal_likelihood(nb):
    from scipy.stats import multivariate_normal
    X, y = _wiggle(15, 5)
    gp = nb.GaussianProcessRegressor(kernel="matern52", lengthscale=1.3, variance=0.8, noise=0.1, optimize=False).fit(X, y)
    yc = y - y.mean()
    for theta in [np.log([1.3, 0.8, 0.1]), np.array([-1.0, 0.5, -3.0]), np.array([0.7, -2.0, -0.5])]:
        ls, var, noise = np.exp(theta)
        cov = nb.stationary_kernel(X, kind="matern52", lengthscale=ls, variance=var) + (noise + 1e-8) * np.eye(15)
        want = multivariate_normal(np.zeros(15), cov).logpdf(yc)
        got = gp.log_marginal_likelihood(theta)
        assert np.isclose(got, want), (
            f"theta = {theta.round(2)}: log N(y_c; 0, K + (noise + JITTER) I) = {want:.6f}, got {got}. "
            "-0.5 y^T alpha - sum log diag(L) - n/2 log(2 pi)")
    assert np.isclose(gp.log_marginal_likelihood(), gp.log_marginal_likelihood(np.log([1.3, 0.8, 0.1]))), (
        "theta=None uses the fitted theta_")


@pytest.mark.parametrize("kind", ["rbf", "matern32", "matern52"])
def test_stage4_gradient_matches_finite_differences(nb, kind):
    X, y = _wiggle(18, 6)
    gp = nb.GaussianProcessRegressor(kernel=kind, optimize=False).fit(X, y)
    for theta in [np.array([0.2, -0.3, -2.0]), np.array([-0.8, 0.6, -1.0])]:
        g = np.asarray(gp.lml_gradient(theta))
        fd = np.array([(gp.log_marginal_likelihood(theta + h) - gp.log_marginal_likelihood(theta - h)) / 2e-5
                       for h in 1e-5 * np.eye(3)])
        assert g.shape == (3,) and np.allclose(g, fd, rtol=1e-4, atol=1e-5), (
            f"{kind}, theta = {theta}: analytic gradient {g.round(5)} vs finite differences {fd.round(5)}. "
            "dLML/dtheta_j = 0.5 tr((a a^T - K_y^-1) dK_y/dtheta_j), with dK_y/dlog(noise) = noise I")


# ---------------------------------------------------------------- stage 5: learning the hyperparameters


def test_stage5_default_fit_optimizes(nb):
    X, y = _wiggle(30, 9)
    gp = bounded(nb.GaussianProcessRegressor().fit, X, y, budget=3_000_000, what="GaussianProcessRegressor().fit")
    theta0 = np.log([1.0, 1.0, 0.01])
    assert gp.log_marginal_likelihood() > gp.log_marginal_likelihood(theta0) + 1.0, (
        "with the defaults (optimize=True, n_restarts=0) fit must run L-BFGS from theta0 itself and improve on it")
    assert np.linalg.norm(gp.lml_gradient(gp.theta_)) < 1e-2, "and stop at a stationary point (gradient ~ 0)"


def test_stage5_matches_sklearn_optimum(nb):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import WhiteKernel
    X, y = _wiggle(30, 7)
    gp = bounded(nb.GaussianProcessRegressor(n_restarts=10, rng=0).fit, X, y, budget=12_000_000,
                 what="GaussianProcessRegressor.fit with optimize=True",
                 hint="pass max_iters to lbfgs, and loop over exactly 1 + n_restarts starts.")
    ref = GaussianProcessRegressor(_sk_kernel("rbf", 1.0, 1.0) + WhiteKernel(0.01), n_restarts_optimizer=5,
                                   random_state=0).fit(X, y - y.mean())
    best = ref.log_marginal_likelihood_value_
    assert gp.log_marginal_likelihood() >= best - 0.5, (
        f"your optimized log marginal likelihood {gp.log_marginal_likelihood():.4f} is more than 0.5 below the optimum {best:.4f}. "
        "minimize -LML with lbfgs, using -lml_gradient as the gradient")
    assert np.allclose(gp.theta_, np.log([gp.lengthscale_, gp.variance_, gp.noise_])), "theta_ = log of the fitted values"
    assert np.linalg.norm(gp.lml_gradient(gp.theta_)) < 1e-2, "at the optimum the gradient is (nearly) zero"
    assert gp.lengthscale == 1.0 and gp.noise == 0.01, "fit must not change the hyperparameters passed to __init__"


def test_stage5_restarts_never_hurt(nb):
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 10, size=(25, 1))
    y = np.sin(3 * X[:, 0]) + 0.3 * rng.normal(size=25)
    y = (y - y.mean()) / y.std()
    single = nb.GaussianProcessRegressor(lengthscale=5.0, noise=0.5, rng=0).fit(X, y)
    multi = bounded(nb.GaussianProcessRegressor(lengthscale=5.0, noise=0.5, n_restarts=40, rng=0).fit, X, y,
                    budget=40_000_000, what="fit with 40 restarts")
    assert multi.log_marginal_likelihood() >= single.log_marginal_likelihood() - 1e-8, (
        "restarts keep the best result over every start, including theta0's: they can't make it worse")
    short = nb.GaussianProcessRegressor(lengthscale=0.3, noise=0.01, rng=0).fit(X, y)
    best = max(single.log_marginal_likelihood(), short.log_marginal_likelihood())
    if single.log_marginal_likelihood() < best - 4:  # the single run got clearly stuck (it does on the shipped data)
        assert multi.log_marginal_likelihood() >= best - 2.0, (
            f"from a long length scale, a single L-BFGS run gets stuck at 'everything is noise' (LML "
            f"{single.log_marginal_likelihood():.2f}); 40 restarts spread over theta0 +- 3 should find the better "
            f"optimum ({best:.2f}), got {multi.log_marginal_likelihood():.2f}. run lbfgs from every start and keep the best")


# ---------------------------------------------------------------- stage 6: expected improvement


def test_stage6_expected_improvement(nb):
    from scipy.stats import norm
    got = nb.expected_improvement(1.01, 1.0, 1.0, xi=0.01)
    assert np.isclose(got, norm.pdf(0)), f"mu = best + xi, sigma = 1: EI = sigma phi(0) = 0.39894, got {got}"
    mu = np.array([0.0, 0.5, 2.0, -1.0])
    sigma = np.array([1.0, 0.2, 0.5, 2.0])
    imp = mu - 0.8 - 0.05
    want = imp * norm.cdf(imp / sigma) + sigma * norm.pdf(imp / sigma)
    got = np.asarray(nb.expected_improvement(mu, sigma, 0.8, xi=0.05))
    assert got.shape == (4,) and np.allclose(got, want), f"(mu - best - xi) Phi(z) + sigma phi(z); want {want}, got {got}"
    z = np.asarray(nb.expected_improvement(np.array([3.0, 0.0]), np.array([0.0, 0.0]), 1.0))
    assert np.allclose(z, 0.0), (
        "the contract's convention: EI = 0 where sigma = 0 (at evaluated points of a noise-free GP, mu <= best, so it "
        "agrees with the sigma -> 0 limit max(mu - best - xi, 0) there). guard the division instead of dividing by 0")
    grid = np.asarray(nb.expected_improvement(np.linspace(-2, 2, 9), 0.5, 0.0))
    assert np.all(np.diff(grid) > 0) and np.all(grid >= 0), "EI is nonnegative and increases with the mean"
    wide = np.asarray(nb.expected_improvement(0.0, np.array([0.1, 0.5, 2.0]), 0.0))
    assert np.all(np.diff(wide) > 0), "at a fixed mean, more uncertainty means more expected improvement"


def test_stage6_far_below_the_incumbent(nb):
    from scipy.special import ndtr  # an accurate normal CDF in the far tail (it uses erfc internally)
    zs = np.array([-5.0, -7.5, -10.0, -20.0])
    got = np.asarray(nb.expected_improvement(zs, 1.0, 0.0, xi=0.0))
    assert np.all(got >= 0), f"EI is an expectation of a nonnegative quantity: it can't be negative, got {got}"
    assert np.all(np.asarray(nb.expected_improvement(np.linspace(-40, 0, 401), 1.0, 0.0, xi=0.0)) >= 0), "EI >= 0 everywhere"
    for z, g in zip(zs, got):
        want = z * ndtr(z) + np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        assert abs(g - want) <= 1e-6 * want, (
            f"z = {z}: EI = {want:.6e}, got {g:.6e}. Phi(z) = (1 + erf(z/sqrt2))/2 cancels catastrophically for very "
            "negative z; use Phi(z) = erfc(-z/sqrt2)/2 (and clip the result at 0)")
