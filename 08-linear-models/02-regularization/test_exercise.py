"""Checks for exercise.py (regularization).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Oracles are
scikit-learn's Ridge, Lasso, ElasticNet and lasso_path (with tight tolerances), plus the
optimality (KKT) conditions of the lasso, which any correct solver must satisfy.
"""
import sys

import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: one feature, centered, x = (-1, 0, 1), y = (-2, 1, 1) -> centered y
X_WE = np.array([[-1.0], [0.0], [1.0]])
Y_WE = np.array([-2.0, 1.0, 1.0])


def _problem(seed, n=60, d=6, sparse=True, corr=0.3, noise=0.5):
    rng = np.random.default_rng(seed)
    C = corr * np.ones((d, d)) + (1 - corr) * np.eye(d)
    X = rng.normal(size=(n, d)) @ np.linalg.cholesky(C).T
    w = np.zeros(d)
    w[: max(2, d // 3)] = rng.choice([-1.0, 1.0], max(2, d // 3)) * rng.uniform(1, 3, max(2, d // 3))
    if not sparse:
        w = rng.normal(size=d)
    return X, X @ w + 2.0 + noise * rng.normal(size=n)


def _kkt_violation(X, y, w, l1, l2):
    """largest violation of the elastic-net optimality conditions (no intercept)."""
    n = len(y)
    g = X.T @ (y - X @ w) / n - l2 * w  # must equal l1 * sign(w_j) where w_j != 0, |.| <= l1 where w_j == 0
    nz = w != 0
    v_nz = np.abs(g[nz] - l1 * np.sign(w[nz])) if nz.any() else np.zeros(1)
    v_z = np.maximum(np.abs(g[~nz]) - l1, 0.0) if (~nz).any() else np.zeros(1)
    return float(max(v_nz.max(), v_z.max()))


# ---------------------------------------------------------------- stage 1: Ridge


def test_stage1_worked_example(nb):
    m = nb.Ridge(alpha=1.0).fit(X_WE, Y_WE)
    assert np.allclose(m.coef_, [1.0]) and np.isclose(m.intercept_, 0.0), (
        f"x is centered with sum x^2 = 2 and sum x y = 3, so w = 3 / (2 + alpha) = 1 for alpha = 1, and the "
        f"intercept is mean(y) - 0 = 0; got coef_ {m.coef_}, intercept_ {m.intercept_}")
    m0 = nb.Ridge(alpha=0.0).fit(X_WE, Y_WE)
    assert np.allclose(m0.coef_, [1.5]), "alpha = 0 is ordinary least squares: w = 3 / 2"


@pytest.mark.parametrize("alpha", [0.01, 1.0, 30.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_stage1_match_sklearn(nb, alpha, fit_intercept):
    from sklearn.linear_model import Ridge

    X, y = _problem(1, sparse=False)
    ours = nb.Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    ref = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="cholesky").fit(X, y)
    assert np.asarray(ours.coef_).shape == (6,) and isinstance(ours.intercept_, float), (
        "coef_ has shape (d,) and intercept_ is a Python float")
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-9) and np.isclose(ours.intercept_, ref.intercept_, atol=1e-9), (
        f"alpha={alpha}, fit_intercept={fit_intercept}: expected coef_ {np.round(ref.coef_, 4)}, got "
        f"{np.round(ours.coef_, 4)}. the penalty is alpha * ||w||^2 on the SUM of squared errors (not the mean), "
        "and the intercept is not penalized: center, solve, recover the intercept")
    Xn = np.random.default_rng(2).normal(size=(4, 6))
    assert np.allclose(ours.predict(Xn), ref.predict(Xn), atol=1e-9)


def test_stage1_intercept_is_not_shrunk(nb):
    X, y = _problem(3, sparse=False)
    a = nb.Ridge(alpha=1e8).fit(X, y)
    assert np.allclose(a.coef_, 0.0, atol=1e-5) and np.isclose(a.intercept_, y.mean(), atol=1e-4), (
        "with an enormous alpha every coefficient goes to 0 but the intercept must stay at mean(y): it isn't "
        f"penalized. got coef_ {np.round(a.coef_, 6)}, intercept_ {a.intercept_:.4f} vs mean(y) {y.mean():.4f}")
    b = nb.Ridge(alpha=5.0).fit(X, y + 100.0)
    c = nb.Ridge(alpha=5.0).fit(X, y)
    assert np.allclose(b.coef_, c.coef_) and np.isclose(b.intercept_ - c.intercept_, 100.0), (
        "shifting y by 100 should only shift intercept_ by 100: a penalized intercept would be shrunk instead")
    assert nb.Ridge().get_params() == {"alpha": 1.0, "fit_intercept": True}
    with pytest.raises(ValueError):
        nb.Ridge(alpha=-1.0).fit(X, y)
        pytest.fail("a negative alpha should raise ValueError")


# ---------------------------------------------------------------- stage 2: the ridge path


def test_stage2_path_matches_ridge(nb):
    from sklearn.linear_model import Ridge

    X, y = _problem(4, n=40, d=5, sparse=False)
    alphas = [0.0, 0.1, 1.0, 10.0, 1000.0]
    coefs = np.asarray(nb.ridge_path(X, y, alphas))
    assert coefs.shape == (5, 5), f"ridge_path returns (len(alphas), d) = (5, 5), got {coefs.shape}"
    for a, row in zip(alphas, coefs):
        ref = np.linalg.lstsq(X, y, rcond=None)[0] if a == 0 else Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_
        assert np.allclose(row, ref, atol=1e-9), (
            f"alpha={a}: expected {np.round(ref, 4)}, got {np.round(row, 4)}. with X = U diag(s) V^T the ridge "
            "solution is V diag(s / (s^2 + alpha)) U^T y (no intercept)")
    norms = np.linalg.norm(coefs, axis=1)
    assert np.all(np.diff(norms) < 0), "the coefficient norm shrinks monotonically as alpha grows"


def test_stage2_rank_deficient_alpha_zero(nb):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(10, 3))
    X = np.column_stack([X, X[:, 0] + X[:, 1]])  # rank 3
    y = rng.normal(size=10)
    w = np.asarray(nb.ridge_path(X, y, [0.0]))[0]
    assert np.all(np.isfinite(w)) and np.allclose(w, np.linalg.pinv(X) @ y, atol=1e-8), (
        "for alpha = 0 and a rank-deficient X, s / (s^2 + 0) blows up for the zero singular value (which comes out of "
        "the SVD as ~1e-16, not 0): treat s <= 1e-12 * s.max() as zero and use a factor 0 there. that gives the "
        "minimum-norm least-squares solution (the pseudoinverse)")


def test_stage2_dof(nb):
    X, _ = _problem(6, n=30, d=4, sparse=False)
    alphas = np.array([0.0, 0.5, 5.0, 50.0, 1e6])
    dof = np.asarray(nb.ridge_dof(X, alphas))
    assert dof.shape == (5,) and np.isclose(dof[0], 4.0), f"at alpha = 0 the dof is the rank, 4; got {dof}"
    for a, got in zip(alphas, dof):
        H = X @ np.linalg.solve(X.T @ X + a * np.eye(4), X.T)
        assert np.isclose(got, np.trace(H), rtol=1e-8), (
            f"alpha={a}: the dof equals the trace of the ridge hat matrix X (X^T X + alpha I)^-1 X^T = {np.trace(H):.4f}; "
            f"got {got:.4f}")
    assert np.all(np.diff(dof) < 0) and dof[-1] < 0.01, "the dof falls from d towards 0 as alpha grows"


# ---------------------------------------------------------------- stage 3: coordinate descent


def test_stage3_soft_threshold(nb):
    z = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 2.0])
    got = np.asarray(nb.soft_threshold(z, 1.0))
    assert np.allclose(got, [-2.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (
        f"soft_threshold(z, 1) moves z towards 0 by 1 and stops at 0: expected [-2, 0, 0, 0, 0, 1], got {got}")
    assert np.isclose(nb.soft_threshold(-0.25, 0.0), -0.25), "with t = 0 nothing changes"


@pytest.mark.parametrize("l1,l2", [(0.1, 0.0), (0.5, 0.0), (0.2, 0.3)])
def test_stage3_match_sklearn_and_kkt(nb, l1, l2):
    from sklearn.linear_model import ElasticNet

    X, y = _problem(7)
    yc = y - y.mean()
    Xc = X - X.mean(0)
    w, sweeps = bounded(nb.coordinate_descent, Xc, yc, l1, l2, max_iters=10_000, tol=1e-12,
                        what="coordinate_descent", hint="stop after max_iters sweeps")
    alpha = l1 + l2
    ref = ElasticNet(alpha=alpha, l1_ratio=l1 / alpha, fit_intercept=False, tol=1e-14, max_iter=100_000).fit(Xc, yc)
    assert np.allclose(w, ref.coef_, atol=1e-7), (
        f"l1={l1}, l2={l2}: expected {np.round(ref.coef_, 5)}, got {np.round(w, 5)}. the update is "
        "w_j = S(X_j^T r / n + z_j w_j, l1) / (z_j + l2) with r the CURRENT residual and z_j = ||X_j||^2 / n")
    viol = _kkt_violation(Xc, yc, np.asarray(w), l1, l2)
    assert viol < 1e-8, f"the optimality conditions are violated by {viol:.2e}: the solution isn't converged"
    assert 1 <= sweeps < 10_000, f"n_sweeps should count the sweeps done, got {sweeps}"


def test_stage3_caps_zero_columns_and_inputs(nb):
    X, y = _problem(8)
    X = np.column_stack([X, np.zeros(len(y))])
    X0, y0 = X.copy(), y.copy()
    w, sweeps = bounded(nb.coordinate_descent, X, y - y.mean(), 0.05, max_iters=3, tol=0.0,
                        what="coordinate_descent", hint="with tol = 0 only max_iters can stop the loop")
    assert sweeps == 3, f"with tol = 0 it must run exactly max_iters = 3 sweeps, got {sweeps}"
    assert w[-1] == 0.0 and np.all(np.isfinite(w)), "an all-zero column gets w_j = 0 (z_j = 0: don't divide by it)"
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "coordinate_descent must not modify X or y"
    w0 = np.ones(X.shape[1])
    w1, _ = nb.coordinate_descent(X, y - y.mean(), 0.05, w0=w0, max_iters=1, tol=0.0)
    assert np.array_equal(w0, np.ones(X.shape[1])), "copy w0: the caller's array must not change"


# ---------------------------------------------------------------- stage 4: Lasso and ElasticNet


@pytest.mark.parametrize("alpha", [0.02, 0.2, 1.0])
def test_stage4_lasso_matches_sklearn(nb, alpha):
    from sklearn.linear_model import Lasso

    X, y = _problem(9)
    ours = bounded(nb.Lasso(alpha=alpha, max_iters=10_000, tol=1e-12).fit, X, y, what="Lasso.fit")
    ref = Lasso(alpha=alpha, tol=1e-14, max_iter=100_000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-7) and np.isclose(ours.intercept_, ref.intercept_, atol=1e-7), (
        f"alpha={alpha}: expected coef_ {np.round(ref.coef_, 4)} and intercept_ {ref.intercept_:.4f}; got "
        f"{np.round(ours.coef_, 4)} and {ours.intercept_:.4f}. the loss is (1/2n)||r||^2 + alpha ||w||_1 with an "
        "unpenalized intercept: center first")
    assert np.array_equal(ours.coef_ == 0, ref.coef_ == 0), "the same coefficients should be EXACTLY zero"
    assert isinstance(ours.intercept_, float) and ours.n_iter_ >= 1


@pytest.mark.parametrize("l1_ratio", [0.0, 0.3, 1.0])
def test_stage4_elastic_net_matches_sklearn(nb, l1_ratio):
    from sklearn.linear_model import ElasticNet

    X, y = _problem(10)
    ours = bounded(nb.ElasticNet(alpha=0.3, l1_ratio=l1_ratio, max_iters=10_000, tol=1e-12).fit, X, y,
                   what="ElasticNet.fit")
    ref = ElasticNet(alpha=0.3, l1_ratio=l1_ratio, tol=1e-14, max_iter=100_000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-7) and np.isclose(ours.intercept_, ref.intercept_, atol=1e-7), (
        f"l1_ratio={l1_ratio}: expected {np.round(ref.coef_, 4)}, got {np.round(ours.coef_, 4)}. "
        "l1 = alpha * l1_ratio, l2 = alpha * (1 - l1_ratio)")


def test_stage4_worked_example(nb):
    m = nb.Lasso(alpha=0.5).fit(X_WE, Y_WE)
    assert np.allclose(m.coef_, [0.75]) and np.isclose(m.intercept_, 0.0), (
        f"with z = sum x^2 / n = 2/3 and rho = sum x y / n = 1, the lasso gives S(1, 0.5) / (2/3) = 0.75; got {m.coef_}")
    assert np.all(nb.Lasso(alpha=1.0).fit(X_WE, Y_WE).coef_ == 0.0), "at alpha = rho = 1 the coefficient is exactly 0"
    e = nb.ElasticNet(alpha=0.5, l1_ratio=0.5).fit(X_WE, Y_WE)
    assert np.allclose(e.coef_, [0.75 / (2 / 3 + 0.25)]), (
        f"elastic net: l1 = l2 = 0.25, so w = S(1, 0.25) / (2/3 + 0.25) = 0.818; got {e.coef_}")


def test_stage4_protocol_and_sparsity(nb):
    X, y = _problem(11)
    big = nb.Lasso(alpha=100.0).fit(X, y)
    assert np.all(big.coef_ == 0.0) and np.isclose(big.intercept_, y.mean()), (
        "a huge alpha zeroes every coefficient exactly; the intercept stays at mean(y)")
    assert nb.Lasso().get_params() == {"alpha": 1.0, "fit_intercept": True, "max_iters": 1000, "tol": 1e-06}
    assert nb.ElasticNet().get_params() == {
        "alpha": 1.0, "l1_ratio": 0.5, "fit_intercept": True, "max_iters": 1000, "tol": 1e-06}
    with pytest.raises(ValueError):
        nb.ElasticNet(l1_ratio=1.5).fit(X, y)
        pytest.fail("l1_ratio must be in [0, 1]")
    with pytest.raises(ValueError):
        nb.Lasso(alpha=-0.1).fit(X, y)
        pytest.fail("a negative alpha should raise ValueError")
    with pytest.raises(ValueError):
        nb.Lasso(alpha=0.1).fit(X, y).predict(X[:, :3])
        pytest.fail("predicting with a different number of features should raise ValueError")


# ---------------------------------------------------------------- stage 5: the lasso path


def test_stage5_alpha_max(nb):
    assert np.isclose(nb.lasso_alpha_max(X_WE, Y_WE), 1.0), "the worked example: max |x^T y| / n = 3 / 3 = 1"
    X, y = _problem(12)
    Xc, yc = X - X.mean(0), y - y.mean()
    amax = nb.lasso_alpha_max(Xc, yc)
    assert isinstance(amax, float) and np.isclose(amax, np.max(np.abs(Xc.T @ yc)) / len(yc)), (
        "alpha_max = max_j |X_j^T y| / n: the largest correlation of a feature with the target")
    at, below = nb.lasso_path(Xc, yc, [1.0001 * amax, 0.9999 * amax], tol=1e-12, max_iters=10_000)
    assert np.all(at == 0.0), f"at (just above) alpha_max every coefficient is exactly zero, got {at}"
    assert np.count_nonzero(below) == 1, (
        f"just below alpha_max exactly one coefficient (the feature with the largest |X_j^T y|) becomes nonzero, got {below}")


def test_stage5_path_matches_sklearn(nb):
    from sklearn.linear_model import Lasso

    X, y = _problem(13, d=8)
    Xc, yc = X - X.mean(0), y - y.mean()
    alphas = np.geomspace(1e-3, 1.0, 12)  # ascending on purpose
    ours = np.asarray(bounded(nb.lasso_path, Xc, yc, alphas, max_iters=10_000, tol=1e-12, budget=5_000_000,
                              what="lasso_path"))
    ref = np.array([Lasso(alpha=a, fit_intercept=False, tol=1e-14, max_iter=100_000).fit(Xc, yc).coef_
                    for a in alphas])
    assert ours.shape == (12, 8), f"coefs should be (len(alphas), d) = (12, 8), got {ours.shape}"
    assert np.allclose(ours, ref, atol=1e-7), (
        "each row should be the lasso solution for its alpha, with row i belonging to alphas[i] in the order you "
        "were GIVEN them (these were ascending). solve from the largest alpha down, but store each row in its place")


def _lines_executed(fn, *args, **kwargs):
    """how much Python work a call does: the number of executed lines (NumPy calls count as one)."""
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line":
            count += 1
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn(*args, **kwargs)
    finally:
        sys.settrace(old)
    return count


def test_stage5_warm_start_pays_off(nb):
    X, y = _problem(14, d=10, corr=0.5)
    Xc, yc = X - X.mean(0), y - y.mean()
    # 20 almost identical alphas: after the first solve, a warm start is already at the next solution
    alphas = 0.05 * nb.lasso_alpha_max(Xc, yc) * (1 + 1e-9 * np.arange(20))
    bounded(nb.lasso_path, Xc, yc, alphas, max_iters=5000, tol=1e-8, budget=20_000_000, what="lasso_path")
    warm = _lines_executed(nb.lasso_path, Xc, yc, alphas, max_iters=5000, tol=1e-8)
    cold = sum(_lines_executed(nb.coordinate_descent, Xc, yc, a, max_iters=5000, tol=1e-8) for a in alphas)
    assert warm < 0.3 * cold, (
        f"along 20 nearly identical alphas, a warm-started path should cost little more than ONE solve, but yours "
        f"did {warm / cold:.0%} of the work of 20 cold starts. pass the previous alpha's solution as w0")
