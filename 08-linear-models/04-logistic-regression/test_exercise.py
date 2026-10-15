"""Checks for exercise.py (logistic regression).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Oracles are
finite differences (for the gradient and Hessian) and scikit-learn's LogisticRegression with
C = 1 / (n * alpha) (its loss is C * sum NLL + ||w||^2 / 2, ours is mean NLL + alpha/2 ||w||^2).
"""
import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: x = (-2, -1, 1, 2), y = (0, 1, 0, 1), with an intercept
X_WE = np.array([[-2.0], [-1.0], [1.0], [2.0]])
Y_WE = np.array([0.0, 1.0, 0.0, 1.0])
A_WE = np.column_stack([np.ones(4), X_WE[:, 0]])


def _problem(seed, n=120, d=3, scale=1.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = scale * rng.normal(size=d)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w + 0.3)))).astype(float)
    return X, y


def _design(X, alpha=0.0):
    A = np.column_stack([np.ones(len(X)), X])
    pen = np.concatenate([[0.0], np.full(X.shape[1], alpha)])
    return A, pen


def _num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def _sklearn(X, y, alpha):
    from sklearn.linear_model import LogisticRegression

    C = np.inf if alpha == 0 else 1.0 / (len(y) * alpha)
    return LogisticRegression(C=C, tol=1e-12, max_iter=100_000).fit(X, y)


# ---------------------------------------------------------------- stage 1: loss and gradient


def test_stage1_worked_example(nb):
    pen = np.zeros(2)
    loss = nb.logistic_loss(np.zeros(2), A_WE, Y_WE, pen)
    assert isinstance(loss, float) and np.isclose(loss, np.log(2)), (
        f"at theta = 0 every probability is 1/2, so the mean cross-entropy is log 2 = 0.6931; got {loss}")
    g = np.asarray(nb.logistic_gradient(np.zeros(2), A_WE, Y_WE, pen))
    assert g.shape == (2,) and np.allclose(g, [0.0, -0.25]), (
        f"the gradient is A^T (sigmoid(A theta) - y) / n = A^T (0.5 - y) / 4 = (0, -0.25) at theta = 0; got {g}")
    pen = np.array([0.0, 2.0])
    theta = np.array([0.5, 1.0])
    base = nb.logistic_loss(theta, A_WE, Y_WE, np.zeros(2))
    assert np.isclose(nb.logistic_loss(theta, A_WE, Y_WE, pen), base + 0.5 * 2.0 * 1.0), (
        "the penalty adds 0.5 * sum(penalty * theta^2): here 0.5 * 2 * 1^2 = 1 (the intercept's penalty is 0)")


@pytest.mark.parametrize("seed", range(3))
def test_stage1_gradient_check(nb, seed):
    X, y = _problem(seed)
    A, pen = _design(X, alpha=0.3)
    theta = np.random.default_rng(seed + 50).normal(size=A.shape[1])
    got = np.asarray(nb.logistic_gradient(theta, A, y, pen))
    want = _num_grad(lambda t: nb.logistic_loss(t, A, y, pen), theta)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-7), (
        f"logistic_gradient disagrees with finite differences of your logistic_loss: {np.round(got, 6)} vs "
        f"{np.round(want, 6)}. remember the 1/n and the penalty term penalty * theta")


def test_stage1_stable_for_huge_scores(nb):
    A = np.array([[1.0, 5000.0], [1.0, -5000.0]])
    y = np.array([0.0, 1.0])
    theta = np.array([0.0, 2.0])  # scores +-1e4, both badly wrong
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            loss = nb.logistic_loss(theta, A, y, np.zeros(2))
            g = np.asarray(nb.logistic_gradient(theta, A, y, np.zeros(2)))
        except FloatingPointError as e:
            pytest.fail(f"overflow for scores of +-1e4 ({e}): use log_sigmoid / sigmoid from mlfs.utils, never "
                        "np.log(1 + np.exp(z))")
    assert np.isclose(loss, 1e4, rtol=1e-6), (
        f"each point is wrong by a score of 1e4, so each costs log(1 + e^1e4) ~ 1e4; got {loss}")
    assert np.all(np.isfinite(g))


# ---------------------------------------------------------------- stage 2: the Hessian


def test_stage2_worked_example(nb):
    H = np.asarray(nb.logistic_hessian(np.zeros(2), A_WE, np.zeros(2)))
    assert H.shape == (2, 2) and np.allclose(H, [[0.25, 0.0], [0.0, 0.625]]), (
        f"at theta = 0, s (1 - s) = 1/4 for every row, so H = A^T A / (4 n) = [[4, 0], [0, 10]] / 16; got {H}")


@pytest.mark.parametrize("seed", range(3))
def test_stage2_matches_finite_differences(nb, seed):
    X, y = _problem(seed + 10)
    A, pen = _design(X, alpha=0.2)
    theta = np.random.default_rng(seed).normal(size=A.shape[1])
    H = np.asarray(nb.logistic_hessian(theta, A, pen))
    cols = [_num_grad(lambda t: nb.logistic_gradient(t, A, y, pen)[i], theta) for i in range(A.shape[1])]
    assert np.allclose(H, np.array(cols), rtol=1e-5, atol=1e-7), (
        "the Hessian should match finite differences of your gradient: A^T diag(s(1-s)) A / n + diag(penalty)")
    assert np.allclose(H, H.T) and np.all(np.linalg.eigvalsh(H) > 0), "the Hessian is symmetric positive definite here"


def test_stage2_no_n_by_n_matrix(nb):
    import tracemalloc

    rng = np.random.default_rng(0)
    A = np.column_stack([np.ones(4000), rng.normal(size=(4000, 3))])
    tracemalloc.start()
    nb.logistic_hessian(np.zeros(4), A, np.zeros(4))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 4000 * 4000 * 8 / 10, (
        f"logistic_hessian peaked at {peak / 1e6:.0f} MB: np.diag(weights) builds an (n, n) matrix (128 MB here). "
        "scale the rows of A by the weights instead: (A.T * w) @ A")


# ---------------------------------------------------------------- stage 3: the estimator, by gradient descent


def test_stage3_gd_matches_sklearn(nb):
    X, y = _problem(20)
    m = bounded(nb.LogisticRegression(solver="gd", alpha=0.05, lr=1.0, max_iters=5000, tol=1e-9).fit, X, y,
                what="LogisticRegression(solver='gd').fit", budget=5_000_000)
    ref = _sklearn(X, y, 0.05)
    assert np.allclose(m.coef_, ref.coef_[0], atol=1e-6) and np.isclose(m.intercept_, ref.intercept_[0], atol=1e-6), (
        f"expected coef_ {np.round(ref.coef_[0], 4)}, intercept_ {ref.intercept_[0]:.4f}; got {np.round(m.coef_, 4)}, "
        f"{m.intercept_:.4f}. the objective is mean NLL + (alpha/2) ||w||^2, intercept unpenalized")
    assert m.converged_ and 0 < m.n_iter_ < 5000, "n_iter_ and converged_ come from gradient_descent's result"


def test_stage3_probabilities_and_protocol(nb):
    X, y = _problem(21)
    labels = np.where(y > 0, "yes", "no")
    m = nb.LogisticRegression(solver="gd", alpha=0.1, lr=1.0, max_iters=2000).fit(X, labels)
    P = np.asarray(m.predict_proba(X[:7]))
    assert list(m.classes_) == ["no", "yes"] and P.shape == (7, 2) and np.allclose(P.sum(axis=1), 1.0), (
        "predict_proba returns (n, 2) rows summing to 1, column j for classes_[j] (sorted labels)")
    z = np.asarray(m.decision_function(X[:7]))
    assert np.allclose(P[:, 1], 1 / (1 + np.exp(-z))), "column 1 is sigmoid(decision_function): P(classes_[1])"
    assert set(m.predict(X)) <= {"no", "yes"}, "predict (from Classifier) returns the original labels"
    assert np.asarray(m.coef_).shape == (3,) and isinstance(m.intercept_, float)
    assert nb.LogisticRegression().get_params() == {
        "alpha": 0.0, "solver": "irls", "lr": 0.1, "max_iters": 100, "tol": 1e-08, "fit_intercept": True}
    m0 = nb.LogisticRegression(solver="gd", fit_intercept=False, alpha=0.1, lr=1.0).fit(X, y)
    assert m0.intercept_ == 0.0
    big = nb.LogisticRegression(solver="gd", alpha=0.1, lr=1.0).fit(X, y)
    Pbig = np.asarray(big.predict_proba(X * 1e4))
    assert np.all(np.isfinite(Pbig)), "predict_proba must not overflow for huge scores"
    z = np.asarray(big.decision_function(X))
    Xs = X * (300.0 / np.max(np.abs(z - big.intercept_)))  # scores up to ~300: sigma(-z) ~ 1e-130, representable
    zs = np.asarray(big.decision_function(Xs))
    Ps = np.asarray(big.predict_proba(Xs))
    far = zs > 40
    assert far.any() and np.all(Ps[far, 0] > 0) and np.allclose(np.log(Ps[far, 0]), -zs[far], rtol=1e-6), (
        "for a score z = 100, P(classes_[0]) = sigma(-z) ~ 4e-44 is a perfectly good float, but 1 - sigmoid(z) "
        "rounds to exactly 0. compute column 0 as sigmoid(-z)")
    short = nb.LogisticRegression(solver="gd", lr=1.0, max_iters=3, tol=1e-12).fit(X, y)
    assert short.n_iter_ == 3 and short.converged_ is False, (
        f"3 gradient steps can't reach tol = 1e-12: n_iter_ should be 3 and converged_ False, got {short.n_iter_}, "
        f"{short.converged_}")
    for bad in (dict(solver="newton"), dict(alpha=-1.0)):
        with pytest.raises(ValueError):
            nb.LogisticRegression(**bad).fit(X, y)
            pytest.fail(f"{bad} should raise ValueError")
    with pytest.raises(ValueError):
        nb.LogisticRegression(solver="gd").fit(X[:9], [0, 1, 2] * 3)
        pytest.fail("three classes should raise ValueError: this model is binary")


# ---------------------------------------------------------------- stage 4: IRLS / Newton


def test_stage4_first_step_is_least_squares(nb):
    theta, steps, conv = nb.irls(A_WE, Y_WE, np.zeros(2), max_iters=1)
    assert steps == 1 and np.allclose(theta, [0.0, 0.4]), (
        f"from theta = 0 one Newton step is least squares on the targets 4 (y - 1/2) = (-2, 2, -2, 2): "
        f"intercept 0, slope 4 / 10 = 0.4. got theta = {theta} after {steps} steps")


@pytest.mark.parametrize("alpha", [0.0, 0.01, 1.0])
def test_stage4_matches_sklearn(nb, alpha):
    X, y = _problem(30 + int(alpha * 100))
    m = bounded(nb.LogisticRegression(alpha=alpha).fit, X, y, what="LogisticRegression.fit",
                hint="irls must stop after max_iters Newton steps")
    ref = _sklearn(X, y, alpha)
    assert np.allclose(m.coef_, ref.coef_[0], atol=1e-6) and np.isclose(m.intercept_, ref.intercept_[0], atol=1e-6), (
        f"alpha={alpha}: expected coef_ {np.round(ref.coef_[0], 4)}, got {np.round(m.coef_, 4)}")
    assert m.converged_ and m.n_iter_ <= 12, (
        f"Newton's method converges quadratically: on this problem it needs well under 12 steps, took {m.n_iter_} "
        f"(converged_={m.converged_})")


def test_stage4_reports_non_convergence(nb):
    X, y = _problem(42)
    A, pen = _design(X, alpha=0.01)
    theta, steps, conv = nb.irls(A, y, pen, max_iters=2, tol=1e-12)
    assert steps == 2 and conv is False, (
        f"2 Newton steps can't reach tol = 1e-12: expected (theta, 2, False), got steps {steps}, converged {conv}. "
        "converged must say whether ||gradient|| <= tol at the returned theta")


def _leverage_problem():
    # two rows 30 times further out than the rest: pure Newton from theta = 0 overshoots and diverges
    rng = np.random.default_rng(281)
    X = rng.normal(size=(70, 2)) * np.array([10.0, 3.0])
    X[:2] *= 30
    z = 2 * (X @ np.array([0.1, -0.2])) + 1.0
    y = (rng.random(70) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y


def test_stage4_step_halving_on_high_leverage_rows(nb):
    X, y = _leverage_problem()
    m = bounded(nb.LogisticRegression().fit, X, y, what="LogisticRegression.fit")
    ref = _sklearn(X, y, 0.0)
    assert m.converged_ and np.allclose(m.coef_, ref.coef_[0], atol=1e-5), (
        f"expected coef_ {np.round(ref.coef_[0], 4)} with converged_ True, got {np.round(m.coef_, 4)} "
        f"(converged_={m.converged_}). on this data a full Newton step from theta = 0 overshoots (two rows have huge "
        "leverage): halve the step until the loss doesn't increase")
    A, pen = _design(X)
    losses = [nb.logistic_loss(nb.irls(A, y, pen, max_iters=k, tol=0.0)[0], A, y, pen) for k in range(0, 12)]
    assert all(np.isfinite(losses)) and all(b <= a + 1e-9 for a, b in zip(losses, losses[1:])), (
        f"the loss after 0, 1, 2, ... IRLS steps must never go up; got {np.round(losses, 5).tolist()}")


def test_stage4_converges_to_a_tight_tolerance(nb):
    # strict "f_new < f" stalls here near the optimum: rounding makes good steps look like no decrease
    rng = np.random.default_rng(1)
    n, d = int(rng.integers(50, 2000)), int(rng.integers(1, 10))  # 972 rows, 5 features
    X = rng.normal(size=(n, d)) * rng.uniform(0.1, 30, d)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ (rng.normal(size=d) / X.std(0)) + rng.normal())))).astype(float)
    m = bounded(nb.LogisticRegression(tol=1e-10).fit, X, y, what="LogisticRegression.fit", budget=5_000_000)
    assert m.converged_ and m.n_iter_ <= 15, (
        f"IRLS should reach ||gradient|| <= 1e-10 here in a few steps; got converged_={m.converged_} after "
        f"{m.n_iter_} steps. near the optimum the loss can't resolve the tiny real decrease of a Newton step: a "
        "strict `f_new < f` rejects every step and stalls. accept f_new <= f + 1e-12 (1 + |f|)")


def test_stage4_separable_data_does_not_hang(nb):
    rng = np.random.default_rng(40)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] - X[:, 1] > 0).astype(float)  # perfectly separable
    A, pen = _design(X)
    theta, steps, conv = bounded(nb.irls, A, y, pen, max_iters=50, tol=1e-14, what="irls",
                                 hint="on separable data the loss keeps falling forever: only max_iters can stop it")
    assert steps <= 50 and np.all(np.isfinite(theta)), (
        f"irls on separable data must stop within max_iters with finite weights, got {steps} steps, theta {theta}")
    assert nb.logistic_loss(theta, A, y, pen) < 1e-3, "on separable data the loss heads towards 0"
    m = bounded(nb.LogisticRegression(alpha=0.01).fit, X, y, what="LogisticRegression.fit")
    assert m.converged_ and np.linalg.norm(m.coef_) < 50, "with alpha > 0 the optimum is finite and IRLS converges"


def test_stage4_monotone_and_fast(nb):
    X, y = _problem(41, n=300, d=5, scale=1.5)
    A, pen = _design(X, alpha=0.01)
    losses = []
    for k in range(1, 9):
        theta, steps, _ = nb.irls(A, y, pen, max_iters=k, tol=0.0)
        losses.append(nb.logistic_loss(theta, A, y, pen))
    assert all(b <= a + 1e-15 for a, b in zip(losses, losses[1:])), (
        f"each Newton step (with step halving) must not increase the loss; got {np.round(losses, 6)}")
    gaps = np.array(losses[:-1]) - losses[-1]
    assert gaps[5] < 1e-8, (
        f"after 6 Newton steps the loss should be within 1e-8 of its minimum (quadratic convergence); the gaps were "
        f"{gaps}")
