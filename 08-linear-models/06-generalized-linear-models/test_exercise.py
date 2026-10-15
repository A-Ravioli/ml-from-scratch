"""Checks for exercise.py (generalized linear models).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Oracles are
finite differences, Newton's method written out below, and scikit-learn's PoissonRegressor,
LogisticRegression, Ridge, mean_poisson_deviance and d2_tweedie_score.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: x = (0, 1), counts y = (1, 3); the fit is exact: mu = exp(x log 3)
A_WE = np.array([[1.0, 0.0], [1.0, 1.0]])
Y_WE = np.array([1.0, 3.0])


def _families(nb):
    return {"gaussian": nb.Gaussian(), "bernoulli": nb.Bernoulli(), "poisson": nb.Poisson()}


def _data(family, seed, n=200, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    eta = 0.3 + X @ (0.5 * rng.normal(size=d))
    if family == "gaussian":
        y = eta + 0.5 * rng.normal(size=n)
    elif family == "bernoulli":
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    return X, y


def _num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


# ---------------------------------------------------------------- stage 1: families


def test_stage1_canonical_links(nb):
    eta = np.linspace(-3, 3, 13)
    for name, fam in _families(nb).items():
        mu = np.asarray(fam.inverse_link(eta))
        assert np.allclose(fam.link(mu), eta), f"{name}: link must undo inverse_link (link(inverse_link(eta)) = eta)"
        dmu = (np.asarray(fam.inverse_link(eta + 1e-6)) - np.asarray(fam.inverse_link(eta - 1e-6))) / 2e-6
        assert np.allclose(dmu, fam.variance(mu), rtol=1e-5), (
            f"{name}: for a canonical link, d mu / d eta = V(mu). got derivative {np.round(dmu[:3], 4)} vs variance "
            f"{np.round(np.asarray(fam.variance(mu))[:3], 4)}")
    assert np.all(np.isfinite(nb.Bernoulli().inverse_link(np.array([-1e4, 1e4])))), "use a stable sigmoid"


def test_stage1_deviance_values(nb):
    assert np.isclose(nb.Gaussian().deviance(np.array([3.0]), np.array([1.0]))[0], 4.0), "gaussian: (y - mu)^2"
    p = np.asarray(nb.Poisson().deviance(np.array([2.0, 0.0]), np.array([1.0, 0.5])))
    assert np.allclose(p, [2 * (2 * np.log(2) - 1), 1.0]), (
        f"poisson: d(2, 1) = 2 [2 log 2 - (2 - 1)] = 0.7726, and d(0, 0.5) = 2 * 0.5 = 1 (0 log 0 = 0); got {p}")
    b = np.asarray(nb.Bernoulli().deviance(np.array([1.0, 0.0]), np.array([0.8, 0.8])))
    assert np.allclose(b, [-2 * np.log(0.8), -2 * np.log(0.2)]), (
        f"bernoulli: d(1, mu) = -2 log mu and d(0, mu) = -2 log(1 - mu); got {b}")


def test_stage1_match_sklearn(nb):
    from sklearn.metrics import log_loss, mean_poisson_deviance

    rng = np.random.default_rng(0)
    y = rng.poisson(2.0, 50).astype(float)
    mu = rng.uniform(0.5, 4.0, 50)
    assert np.isclose(np.mean(nb.Poisson().deviance(y, mu)), mean_poisson_deviance(y, mu)), (
        "the mean poisson deviance should match sklearn's mean_poisson_deviance")
    yb = (rng.random(50) < 0.4).astype(float)
    pb = rng.uniform(0.05, 0.95, 50)
    assert np.isclose(np.mean(nb.Bernoulli().deviance(yb, pb)), 2 * log_loss(yb, pb)), (
        "the bernoulli deviance is twice the log loss")
    assert np.all(np.asarray(nb.Poisson().deviance(y, mu)) >= 0), "deviances are never negative"
    assert np.allclose(nb.Poisson().deviance(y[y > 0], y[y > 0]), 0.0), "the deviance is 0 at mu = y"


# ---------------------------------------------------------------- stage 2: loss, gradient, one IRLS step


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "poisson"])
def test_stage2_gradient_check(nb, family):
    X, y = _data(family, 1)
    A = np.column_stack([np.ones(len(X)), X])
    pen = np.array([0.0, 0.2, 0.2, 0.2])
    fam = _families(nb)[family]
    theta = 0.3 * np.random.default_rng(2).normal(size=4)
    got = np.asarray(nb.glm_gradient(theta, A, y, fam, pen))
    want = _num_grad(lambda t: nb.glm_loss(t, A, y, fam, pen), theta)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-7), (
        f"{family}: glm_gradient disagrees with finite differences of glm_loss (half the mean deviance + penalty): "
        f"{np.round(got, 5)} vs {np.round(want, 5)}")


def test_stage2_worked_example(nb):
    theta0 = np.array([np.log(2.0), 0.0])  # mu = 2 for both points: the mean count
    got = np.asarray(nb.irls_step(theta0, A_WE, Y_WE, nb.Poisson(), np.zeros(2)))
    assert np.allclose(got, [np.log(2.0) - 0.5, 1.0]), (
        f"weights V(mu) = 2 for both points and working responses z = eta + (y - mu)/mu = log 2 + (-0.5, 0.5): "
        f"the weighted least-squares line through two points passes through both, so theta = (log 2 - 0.5, 1); "
        f"got {got}")


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "poisson"])
def test_stage2_irls_step_is_newton(nb, family):
    X, y = _data(family, 3)
    A = np.column_stack([np.ones(len(X)), X])
    pen = np.array([0.0, 0.1, 0.1, 0.1])
    fam = _families(nb)[family]
    theta = 0.2 * np.random.default_rng(4).normal(size=4)
    mu = np.asarray(fam.inverse_link(A @ theta))
    H = (A.T * np.asarray(fam.variance(mu))) @ A / len(y) + np.diag(pen)
    newton = theta - np.linalg.solve(H, np.asarray(nb.glm_gradient(theta, A, y, fam, pen)))
    got = np.asarray(nb.irls_step(theta, A, y, fam, pen))
    assert np.allclose(got, newton, atol=1e-9), (
        f"{family}: with a canonical link, one IRLS step is exactly one Newton step on glm_loss. expected "
        f"{np.round(newton, 5)}, got {np.round(got, 5)}. solve (A^T W A + n diag(penalty)) theta = A^T (W eta + y - mu)")
    if family == "gaussian":
        exact = np.linalg.solve(A.T @ A + len(y) * np.diag(pen), A.T @ y)
        assert np.allclose(got, exact), "for the gaussian family one step from anywhere is the exact (ridge) solution"


def test_stage2_tiny_variance_is_safe(nb):
    A = np.column_stack([np.ones(4), [-800.0, -799.0, 799.0, 800.0]])  # sigmoid(+-800) is exactly 1 / 0
    y = np.array([0.0, 0.0, 1.0, 1.0])
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            out = np.asarray(nb.irls_step(np.array([0.0, 1.0]), A, y, nb.Bernoulli(), np.array([0.0, 0.1])))
        except FloatingPointError as e:
            pytest.fail(f"irls_step divided by a tiny variance ({e}): build the right-hand side as A^T (W eta + y - mu)")
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------- stage 3: the estimator


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_stage3_poisson_matches_sklearn(nb, alpha):
    from sklearn.linear_model import PoissonRegressor

    X, y = _data("poisson", 5)
    m = bounded(nb.GLM("poisson", alpha=alpha).fit, X, y, what="GLM.fit",
                hint="stop after max_iters IRLS steps")
    ref = PoissonRegressor(alpha=alpha, tol=1e-12, max_iter=10_000).fit(X, y)
    assert np.allclose(m.coef_, ref.coef_, atol=1e-6) and np.isclose(m.intercept_, ref.intercept_, atol=1e-6), (
        f"alpha={alpha}: expected coef_ {np.round(ref.coef_, 4)}, intercept_ {ref.intercept_:.4f}; got "
        f"{np.round(m.coef_, 4)}, {m.intercept_:.4f}. the objective is half the mean deviance + (alpha/2)||w||^2")
    assert np.allclose(m.predict(X[:5]), ref.predict(X[:5]), rtol=1e-6), "predict returns the mean mu = exp(eta)"
    assert m.converged_ and m.n_iter_ <= 15, f"IRLS should converge in a handful of steps, took {m.n_iter_}"


def test_stage3_other_families_are_old_friends(nb):
    from sklearn.linear_model import LogisticRegression, Ridge

    X, y = _data("gaussian", 6)
    g = bounded(nb.GLM("gaussian", alpha=0.05).fit, X, y, what="GLM.fit")
    r = Ridge(alpha=len(y) * 0.05).fit(X, y)
    assert np.allclose(g.coef_, r.coef_, atol=1e-8), "family='gaussian' is ridge regression with alpha_ridge = n alpha"
    Xb, yb = _data("bernoulli", 7)
    b = bounded(nb.GLM("bernoulli", alpha=0.05).fit, Xb, yb, what="GLM.fit")
    lr = LogisticRegression(C=1 / (len(yb) * 0.05), tol=1e-12, max_iter=10_000).fit(Xb, yb)
    assert np.allclose(b.coef_, lr.coef_[0], atol=1e-6), "family='bernoulli' is topic 04's logistic regression"
    assert np.all((b.predict(Xb) > 0) & (b.predict(Xb) < 1)), "the bernoulli mean is a probability"


def test_stage3_worked_example_and_protocol(nb):
    m = bounded(nb.GLM("poisson").fit, A_WE[:, 1:], Y_WE, what="GLM.fit")
    assert np.isclose(m.intercept_, 0.0, atol=1e-6) and np.allclose(m.coef_, [np.log(3.0)], atol=1e-6), (
        f"two points, two parameters: the fit is exact, mu = (1, 3), so intercept 0 and slope log 3; got "
        f"{m.intercept_}, {m.coef_}")
    assert isinstance(m.intercept_, float) and np.asarray(m.coef_).shape == (1,)
    assert nb.GLM().get_params() == {"family": "poisson", "alpha": 0.0, "max_iters": 100, "tol": 1e-08,
                                     "fit_intercept": True}
    X, y = _data("poisson", 8)
    m = nb.GLM().fit(X, y)
    assert np.all(m.predict(X) > 0), "poisson means are positive"
    assert -1 < m.score(X, y) <= 1, "score is R^2 (from Regressor)"
    m0 = nb.GLM(fit_intercept=False).fit(X, y)
    assert m0.intercept_ == 0.0
    for bad_family in ("gamma", "normal"):
        with pytest.raises(ValueError):
            nb.GLM(bad_family).fit(X, y)
            pytest.fail(f"family={bad_family!r} should raise ValueError")
    with pytest.raises(ValueError):
        nb.GLM("poisson").fit(X, y - 5)
        pytest.fail("negative counts should raise ValueError for the poisson family")
    with pytest.raises(ValueError):
        nb.GLM("bernoulli").fit(X, y)
        pytest.fail("counts above 1 should raise ValueError for the bernoulli family")


def test_stage3_steps_never_increase_the_loss(nb):
    # strong effects and few rows: a full IRLS step from the intercept-only start overshoots here
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 2)) * rng.uniform(0.5, 3)
    w = rng.normal(size=2) * rng.uniform(0.5, 2)
    y = rng.poisson(np.exp(np.clip(0.5 + X @ w, -20, 8))).astype(float)

    def loss(mu):  # mean Poisson negative log-likelihood, up to a constant
        return float(np.mean(mu - y * np.log(mu)))

    losses = [loss(np.full(len(y), y.mean()))]
    for k in range(1, 9):
        m = bounded(nb.GLM("poisson", max_iters=k).fit, X, y, what="GLM.fit")
        with np.errstate(all="ignore"):
            losses.append(loss(np.asarray(m.predict(X))))
    assert all(np.isfinite(losses)) and all(b <= a + 1e-9 * (1 + abs(a)) for a, b in zip(losses, losses[1:])), (
        f"after 0, 1, 2, ... IRLS steps the loss should never go up, but got {np.round(losses, 4).tolist()}. on this "
        "data a full step from the start overshoots: halve the step until the loss doesn't increase")
    assert losses[-1] < losses[0], "IRLS should improve on the intercept-only start"


def test_stage3_separable_bernoulli_does_not_hang(nb):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(float)
    m = bounded(nb.GLM("bernoulli", max_iters=40).fit, X, y, what="GLM.fit",
                hint="on separable data the loss keeps falling: only max_iters can stop it")
    assert m.n_iter_ <= 40 and np.all(np.isfinite(m.coef_))


# ---------------------------------------------------------------- stage 4: goodness of fit


def test_stage4_d2_score(nb):
    from sklearn.metrics import d2_tweedie_score, r2_score

    X, y = _data("poisson", 10)
    mu = nb.GLM().fit(X, y).predict(X)
    got = nb.d2_score(y, mu, nb.Poisson())
    assert isinstance(got, float) and np.isclose(got, d2_tweedie_score(y, mu, power=1)), (
        "the fraction of poisson deviance explained should match sklearn's d2_tweedie_score(power=1)")
    yg, mg = np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.5, 3.5])  # mean(mg) != mean(yg) on purpose
    assert np.isclose(nb.d2_score(yg, mg, nb.Gaussian()), r2_score(yg, mg)), "for the gaussian family, D^2 is R^2"
    assert np.isclose(nb.d2_score(y, np.full_like(y, y.mean()), nb.Poisson()), 0.0), "predicting the mean gives 0"
    shifted = np.full_like(y, 1.5 * y.mean())
    want = 1 - np.sum(nb.Poisson().deviance(y, shifted)) / np.sum(nb.Poisson().deviance(y, np.full_like(y, y.mean())))
    assert np.isclose(nb.d2_score(y, shifted, nb.Poisson()), want) and want < 0, (
        "the null model predicts mean(y), not the mean of your predictions: a constant 1.5 * mean(y) must score < 0")


def test_stage4_pearson_dispersion(nb):
    y, mu = np.array([0.0, 2.0, 5.0]), np.array([1.0, 2.0, 4.0])
    got = nb.pearson_dispersion(y, mu, nb.Poisson(), 1)
    assert np.isclose(got, (1.0 + 0.0 + 0.25) / 2), (
        f"sum (y - mu)^2 / V(mu) = 1/1 + 0/2 + 1/4 = 1.25, divided by n - p = 2: 0.625; got {got}")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(3000, 2))
    mu_true = np.exp(1.0 + X @ np.array([0.4, -0.3]))
    y_pois = rng.poisson(mu_true).astype(float)
    y_nb = rng.poisson(mu_true * rng.gamma(2.0, 0.5, size=3000)).astype(float)  # overdispersed: var = mu + mu^2/2
    phi_p = nb.pearson_dispersion(y_pois, nb.GLM().fit(X, y_pois).predict(X), nb.Poisson(), 3)
    phi_nb = nb.pearson_dispersion(y_nb, nb.GLM().fit(X, y_nb).predict(X), nb.Poisson(), 3)
    assert abs(phi_p - 1) < 0.1 and phi_nb > 1.5, (
        f"real poisson data should give a dispersion near 1 (got {phi_p:.2f}); gamma-mixed counts, whose variance "
        f"is larger than their mean, should give clearly more (got {phi_nb:.2f})")
    with pytest.raises(ValueError):
        nb.pearson_dispersion(y, mu, nb.Poisson(), 3)
        pytest.fail("n_params >= n leaves no degrees of freedom: raise ValueError")
