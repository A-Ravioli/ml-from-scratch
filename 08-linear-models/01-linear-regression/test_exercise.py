"""Checks for exercise.py (linear regression).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Oracles are
np.linalg.lstsq and scikit-learn's LinearRegression; the lesson's worked example is checked
by hand.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: four points, best line y = 1.1 + 1.1 x
X_WE = np.array([[0.0], [1.0], [2.0], [3.0]])
Y_WE = np.array([1.0, 3.0, 2.0, 5.0])
A_WE = np.column_stack([np.ones(4), X_WE[:, 0]])  # with a column of ones for the intercept


def _problem(seed, n=60, d=4, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    return X, X @ w + 0.7 + noise * rng.normal(size=n), w


def _num_grad(f, w, h=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g


# ---------------------------------------------------------------- stage 1: loss and gradient


def test_stage1_worked_example(nb):
    loss0 = nb.mse_loss(np.zeros(2), A_WE, Y_WE)
    assert isinstance(loss0, float) and np.isclose(loss0, 39 / 8), (
        f"at w = 0 the residuals are -y, so L = (1 + 9 + 4 + 25) / (2 * 4) = 4.875; got {loss0}. "
        "if you got 9.75 you divided by n instead of 2n; 39 means you forgot to divide at all")
    g0 = np.asarray(nb.mse_gradient(np.zeros(2), A_WE, Y_WE))
    assert g0.shape == (2,) and np.allclose(g0, [-2.75, -5.5]), (
        f"at w = 0 the gradient is -X^T y / n = -[11, 22] / 4 = [-2.75, -5.5]; got {g0}")
    w_star = np.array([1.1, 1.1])
    assert np.isclose(nb.mse_loss(w_star, A_WE, Y_WE), 2.7 / 8), "at the optimum RSS = 2.7, so L = 2.7 / 8"
    assert np.allclose(nb.mse_gradient(w_star, A_WE, Y_WE), 0.0, atol=1e-12), (
        "at the least-squares solution (1.1, 1.1) the gradient must be exactly zero (the normal equations)")


@pytest.mark.parametrize("seed", range(3))
def test_stage1_gradient_check(nb, seed):
    X, y, _ = _problem(seed)
    w = np.random.default_rng(seed + 100).normal(size=X.shape[1])
    got = np.asarray(nb.mse_gradient(w, X, y))
    want = _num_grad(lambda v: nb.mse_loss(v, X, y), w)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-7), (
        f"mse_gradient disagrees with a finite-difference derivative of your mse_loss: {got} vs {want}. "
        "the two must use the same scaling (1/(2n) in the loss gives 1/n in the gradient)")


# ---------------------------------------------------------------- stage 2: exact solvers


@pytest.mark.parametrize("seed", range(3))
def test_stage2_match_lstsq(nb, seed):
    X, y, _ = _problem(seed, n=40, d=5)
    want = np.linalg.lstsq(X, y, rcond=None)[0]
    for name in ("solve_normal_equations", "solve_qr"):
        got = np.asarray(getattr(nb, name)(X, y))
        assert got.shape == (5,) and np.allclose(got, want, atol=1e-10), (
            f"{name} should give the least-squares solution {np.round(want, 4)}, got {np.round(got, 4)}")


def test_stage2_qr_is_stable(nb):
    # the Läuchli matrix: cond(X) ~ 1.4e8, so cond(X^T X) ~ 2e16 and 1 + eps^2 == 1 in floating point
    eps = 1e-8
    X = np.array([[1.0, 1.0], [eps, 0.0], [0.0, eps]])
    y = X @ np.array([1.0, 2.0])
    got = np.asarray(nb.solve_qr(X, y))
    assert np.allclose(got, [1.0, 2.0], rtol=1e-6), (
        f"solve_qr should recover w = [1, 2] exactly-ish on this ill-conditioned X, got {got}. QR works with X itself "
        "(cond ~1e8); if you formed X^T X anywhere, its condition number squared to ~1e16 and the digits are gone")
    with pytest.raises(np.linalg.LinAlgError):
        nb.solve_normal_equations(X, y)
        pytest.fail("in floating point X^T X = [[1, 1], [1, 1]] here (1 + 1e-16 rounds to 1): it's singular, and "
                    "cholesky must raise LinAlgError. don't catch it, and don't add a fudge term")


def test_stage2_inputs_untouched(nb):
    X, y, _ = _problem(5, n=20, d=3)
    X0, y0 = X.copy(), y.copy()
    nb.solve_normal_equations(X, y)
    nb.solve_qr(X, y)
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "the solvers must not modify X or y"


# ---------------------------------------------------------------- stage 3: LinearRegression


@pytest.mark.parametrize("solver", ["normal", "qr"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_stage3_match_sklearn(nb, solver, fit_intercept):
    from sklearn.linear_model import LinearRegression

    X, y, _ = _problem(7, n=50, d=3, noise=0.5)
    ours = nb.LinearRegression(solver=solver, fit_intercept=fit_intercept).fit(X, y)
    ref = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    assert np.asarray(ours.coef_).shape == (3,), f"coef_ should have shape (d,) = (3,), got {np.shape(ours.coef_)}"
    assert isinstance(ours.intercept_, float), f"intercept_ should be a Python float, got {type(ours.intercept_)}"
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-8) and np.isclose(ours.intercept_, ref.intercept_, atol=1e-8), (
        f"solver={solver!r}, fit_intercept={fit_intercept}: expected coef_ {np.round(ref.coef_, 4)} and intercept_ "
        f"{ref.intercept_:.4f}, got {np.round(ours.coef_, 4)} and {ours.intercept_:.4f}")
    Xn = np.random.default_rng(8).normal(size=(6, 3))
    assert np.allclose(ours.predict(Xn), ref.predict(Xn), atol=1e-8), "predict is X @ coef_ + intercept_"


def test_stage3_worked_example_and_protocol(nb):
    m = nb.LinearRegression().fit(X_WE[:, 0], Y_WE)  # a 1-D X is one feature (check_X_y)
    assert np.allclose(m.coef_, [1.1]) and np.isclose(m.intercept_, 1.1), (
        f"the lesson's four points give slope 1.1 and intercept 1.1; got {m.coef_}, {m.intercept_}")
    assert np.isclose(m.score(X_WE, Y_WE), 1 - 2.7 / 8.75), "score is R^2 (from Regressor): 1 - RSS/TSS = 1 - 2.7/8.75"
    assert nb.LinearRegression().get_params() == {
        "fit_intercept": True, "solver": "normal", "lr": 0.01, "max_iters": 1000, "tol": 1e-06}, (
        "__init__ stores its five keyword arguments unchanged")
    with pytest.raises(ValueError):
        nb.LinearRegression(solver="magic").fit(X_WE, Y_WE)
        pytest.fail("an unknown solver should raise ValueError in fit")
    with pytest.raises(ValueError):
        m.predict(np.ones((2, 3)))
        pytest.fail("fitted on 1 feature, predicting with 3 should raise ValueError")


@pytest.mark.parametrize("solver", ["normal", "qr"])
def test_stage3_centering_handles_offsets(nb, solver):
    # features around 10,000: a ones column next to them makes X^T X hopelessly ill-conditioned
    rng = np.random.default_rng(11)
    X = 1e4 + rng.normal(size=(80, 2))
    y = 3.0 + X @ np.array([2.0, -1.0]) + 0.01 * rng.normal(size=80)
    xm, ym = X.mean(0), y.mean()
    want = np.linalg.lstsq(X - xm, y - ym, rcond=None)[0]
    m = nb.LinearRegression(solver=solver).fit(X, y)
    assert np.allclose(m.coef_, want, rtol=1e-7), (
        f"with features near 10,000 the coefficients should be {want}, got {m.coef_}. center X and y before solving "
        "(and recover intercept_ = y_mean - x_mean @ coef_): a column of ones next to columns near 1e4 makes the "
        "normal equations lose about 16 digits")
    assert np.isclose(m.intercept_, ym - xm @ want, rtol=1e-6)


# ---------------------------------------------------------------- stage 4: gradient descent


def test_stage4_solve_gd_converges(nb):
    X, y, _ = _problem(3, n=80, d=3)
    want = np.linalg.lstsq(X, y, rcond=None)[0]
    res = bounded(nb.solve_gd, X, y, lr=0.3, max_iters=5000, tol=1e-10, what="solve_gd",
                  hint="gradient_descent stops after max_iters steps: pass max_iters through")
    assert res.converged and np.allclose(res.x, want, atol=1e-8), (
        f"with lr = 0.3 on well-conditioned data, gradient descent should converge to {np.round(want, 5)}; "
        f"got {np.round(res.x, 5)} (converged={res.converged})")
    assert 0 < res.n_iters < 5000 and np.linalg.norm(nb.mse_gradient(res.x, X, y)) <= 1e-10
    assert res.history[0] == pytest.approx(nb.mse_loss(np.zeros(3), X, y)), "the default start is w0 = zeros"


def test_stage4_caps_and_divergence(nb):
    X, y, _ = _problem(4, n=50, d=3)
    short = bounded(nb.solve_gd, X, y, lr=0.01, max_iters=5, tol=1e-12, what="solve_gd")
    assert short.n_iters == 5 and not short.converged, "with max_iters=5 it takes exactly 5 steps and hasn't converged"
    big = bounded(nb.solve_gd, X, y, lr=50.0, max_iters=10_000, tol=1e-12, what="solve_gd",
                  hint="with a step size this large the loss explodes: gradient_descent stops at a non-finite loss")
    assert not big.converged, "lr = 50 diverges: it must not report convergence"


def test_stage4_linear_regression_gd(nb):
    from sklearn.linear_model import LinearRegression

    X, y, _ = _problem(5, n=100, d=3)
    X = X * np.array([1.0, 0.5, 2.0]) + np.array([5.0, -3.0, 0.0])
    m = bounded(nb.LinearRegression(solver="gd", lr=0.1, max_iters=20_000, tol=1e-9).fit, X, y,
                what="LinearRegression(solver='gd').fit")
    ref = LinearRegression().fit(X, y)
    assert np.allclose(m.coef_, ref.coef_, atol=1e-6) and np.isclose(m.intercept_, ref.intercept_, atol=1e-6), (
        f"solver='gd' should reach the least-squares solution {np.round(ref.coef_, 4)} (+ {ref.intercept_:.4f}); got "
        f"{np.round(m.coef_, 4)} (+ {m.intercept_:.4f}). center first, run solve_gd on the centered data, then "
        "recover the intercept as for the other solvers")
    assert 0 < m.n_iter_ < 20_000, f"n_iter_ should record the gradient steps taken, got {getattr(m, 'n_iter_', None)}"


# ---------------------------------------------------------------- stage 5: maximum likelihood


def test_stage5_worked_example(nb):
    pred = 1.1 + 1.1 * X_WE[:, 0]
    assert np.isclose(nb.noise_variance(Y_WE, pred), 0.675), "the MLE of the noise variance is RSS / n = 2.7 / 4"
    assert np.isclose(nb.noise_variance(Y_WE, pred, n_params=2), 1.35), (
        "with 2 fitted parameters the unbiased estimate is RSS / (n - 2) = 2.7 / 2")
    nll = nb.gaussian_nll(Y_WE, pred, 0.675)
    assert isinstance(nll, float) and np.isclose(nll, 0.5 * np.log(2 * np.pi * 0.675) + 0.5), (
        f"at sigma2 = RSS/n the mean NLL is 0.5 log(2 pi sigma2) + 1/2; got {nll}")
    with pytest.raises(ValueError):
        nb.noise_variance(Y_WE, pred, n_params=4)
        pytest.fail("n_params >= n leaves no degrees of freedom: raise ValueError")


def test_stage5_mle_is_least_squares(nb):
    X, y, _ = _problem(6, n=40, d=2, noise=0.3)
    w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = nb.noise_variance(y, X @ w_ls)
    grid = s2 * np.linspace(0.5, 1.5, 101)
    nlls = [nb.gaussian_nll(y, X @ w_ls, g) for g in grid]
    assert np.argmin(nlls) == 50, "for fixed predictions the NLL is minimized over sigma2 at RSS / n"
    rng = np.random.default_rng(0)
    for _ in range(5):
        w = w_ls + 0.05 * rng.normal(size=2)
        assert nb.gaussian_nll(y, X @ w, s2) > nb.gaussian_nll(y, X @ w_ls, s2), (
            "for a fixed sigma2, moving w away from least squares must increase the NLL: the NLL is the "
            "squared error, rescaled and shifted")


def test_stage5_unbiased_on_average(nb):
    rng = np.random.default_rng(12)
    n, p, sigma2 = 10, 4, 2.0
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    mle, unb = [], []
    for _ in range(4000):
        y = X @ np.ones(p) + np.sqrt(sigma2) * rng.normal(size=n)
        pred = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        mle.append(nb.noise_variance(y, pred))
        unb.append(nb.noise_variance(y, pred, n_params=p))
    assert abs(np.mean(unb) - sigma2) < 0.1, f"RSS/(n - p) should average sigma2 = 2 over many datasets, got {np.mean(unb):.3f}"
    assert abs(np.mean(mle) - sigma2 * (n - p) / n) < 0.06, (
        f"RSS/n should average sigma2 (n - p)/n = 1.2 (biased low), got {np.mean(mle):.3f}")


# ---------------------------------------------------------------- stage 6: basis expansions


def test_stage6_values(nb):
    x = np.array([-1.0, 0.5, 2.0])
    P = np.asarray(nb.polynomial_basis(x, 3))
    assert P.shape == (3, 3) and np.allclose(P, [[-1, 1, -1], [0.5, 0.25, 0.125], [2, 4, 8]]), (
        f"columns x, x^2, x^3 (no constant column); got {P.tolist()}")
    assert np.asarray(nb.polynomial_basis(x, 0)).shape == (3, 0), "degree 0 has no columns"
    R = np.asarray(nb.rbf_basis(x, np.array([0.0, 2.0]), 1.0))
    want = np.exp(-np.array([[1, 9], [0.25, 2.25], [4, 0]]) / 2.0)
    assert R.shape == (3, 2) and np.allclose(R, want), (
        f"entry (i, j) is exp(-(x_i - c_j)^2 / (2 width^2)); got {np.round(R, 4).tolist()}")


def test_stage6_fits_a_cubic(nb):
    rng = np.random.default_rng(13)
    x = rng.uniform(-2, 2, 30)
    y = 1.0 - 2.0 * x + 0.5 * x**3
    m = nb.LinearRegression(solver="qr").fit(nb.polynomial_basis(x, 3), y)
    assert np.allclose(m.coef_, [-2.0, 0.0, 0.5], atol=1e-8) and np.isclose(m.intercept_, 1.0), (
        f"a cubic is linear in (x, x^2, x^3): LinearRegression on polynomial_basis(x, 3) should recover "
        f"coef_ [-2, 0, 0.5] and intercept 1; got {np.round(m.coef_, 4)}, {m.intercept_:.4f}")
