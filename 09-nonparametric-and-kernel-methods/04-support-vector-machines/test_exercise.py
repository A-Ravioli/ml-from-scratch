"""Checks for exercise.py (support vector machines).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scikit-learn's
SVC (libsvm) is used only as an oracle for the optimal dual objective. SMO runs inside
`bounded`, so a loop that never stops fails with a message instead of hanging.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

X2 = np.array([[0.0, 0.0], [2.0, 0.0]])  # the lesson's two-point example
Y2 = np.array([-1.0, 1.0])


def _circles(n, seed, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 + noise * rng.normal(size=n) > 1.4).astype(int)
    return X, y


def _fit(nb, X, y, **kw):
    return bounded(nb.SVC(**kw).fit, X, y, budget=8_000_000, what="SVC.fit",
                   hint="the outer loop must stop after max_passes sweeps without a change, or after max_iters sweeps.")


# ---------------------------------------------------------------- stage 1: primal


def test_stage1_hinge(nb):
    got = np.asarray(nb.hinge_loss(np.array([1.0, -1.0, 1.0, -1.0]), np.array([2.0, 0.5, -1.0, -3.0])))
    assert np.allclose(got, [0.0, 1.5, 2.0, 0.0]), (
        f"max(0, 1 - y f): y=1, f=2 is beyond the margin (0); y=-1, f=0.5 is on the wrong side (1.5); y=1, f=-1 "
        f"(2); y=-1, f=-3 (0). got {got}")


def test_stage1_primal_objective(nb):
    got = nb.primal_objective(np.array([1.0, 0.0]), -1.0, X2, Y2, 10.0)
    assert np.isclose(got, 0.5), f"w = (1, 0), b = -1 separates the two points with margin: 0.5 ||w||^2 + 0 = 0.5, got {got}"
    got = nb.primal_objective(np.array([0.5, 0.0]), -0.5, X2, Y2, 10.0)
    assert np.isclose(got, 0.125 + 10 * (0.5 + 0.5)), (
        f"w = (0.5, 0), b = -0.5: both points have y f = 0.5, so each pays hinge 0.5: 0.125 + 10 * 1 = 10.125, got {got}")


# ---------------------------------------------------------------- stage 2: the dual and KKT


def test_stage2_dual_objective_and_zero_gap(nb):
    K = X2 @ X2.T
    got = nb.dual_objective(np.array([0.5, 0.5]), K, Y2)
    assert np.isclose(got, 0.5), (
        f"sum alpha - 0.5 (alpha y)^T K (alpha y) = 1 - 0.5 * 1 = 0.5, got {got}. at the optimum it equals the primal "
        "objective (zero duality gap)")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 3)), np.where(rng.random(20) < 0.5, -1.0, 1.0)
    a = rng.uniform(0, 1, 20)
    v = a * y
    assert np.isclose(nb.dual_objective(a, X @ X.T, y), a.sum() - 0.5 * np.sum((X.T @ v) ** 2)), (
        "with a linear kernel the quadratic term is 0.5 ||sum_i alpha_i y_i x_i||^2")


def test_stage2_kkt_cases(nb):
    C = 1.0
    alpha = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    y = np.ones(7)
    f = np.array([1.5, 0.5, 0.5, 1.5, 1.0, 1.2, 1.0005])
    got = np.asarray(nb.kkt_violations(alpha, y, f, C, tol=1e-3))
    want = [False, True, False, True, False, True, False]
    assert got.dtype == bool and got.tolist() == want, (
        f"expected {want}, got {got.tolist()}. the rules (with margins m = y f): alpha = 0 needs m >= 1; alpha = C needs "
        "m <= 1; 0 < alpha < C needs m = 1. each within tol (the last example is 0.0005 off: fine)")


# ---------------------------------------------------------------- stage 3: the SMO step


def test_stage3_worked_example(nb):
    K = X2 @ X2.T
    E = (K @ np.zeros(2)) - Y2  # f = 0 at the start, so E = -y
    out = nb.smo_pair_update(0, 1, np.zeros(2), 0.0, K, Y2, 10.0, E)
    assert out is not None, "this pair can make progress: the update should return (alpha, b)"
    alpha, b = out
    assert np.allclose(alpha, [0.5, 0.5]) and np.isclose(b, -1.0), (
        f"the lesson's example: eta = -4, alpha_2 = 0 - 1 * (1 - (-1)) / (-4) = 0.5, alpha_1 = 0.5, b = -1. "
        f"got alpha = {alpha}, b = {b}")


def test_stage3_clipping_and_bias(nb):
    K = X2 @ X2.T
    start = np.zeros(2)
    alpha, b = nb.smo_pair_update(0, 1, start, 0.0, K, Y2, 0.2, -Y2)
    assert np.allclose(alpha, [0.2, 0.2]), f"with C = 0.2 the unclipped 0.5 must be clipped to H = 0.2, got {alpha}"
    assert np.isclose(b, -0.4), (
        f"both alphas end at the bound C, so b is the average of b1 = -1 and b2 = 0.2: -0.4. got {b}")
    assert np.array_equal(start, [0.0, 0.0]), "smo_pair_update must not modify the alpha array it was given"


def test_stage3_bias_from_the_free_variable(nb):
    X = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    y = np.array([-1.0, 1.0, 1.0])
    K = X @ X.T
    alpha = np.array([0.2, 0.1, 0.1])
    E = K @ (alpha * y) - y
    out = nb.smo_pair_update(0, 1, alpha, 0.0, K, y, 0.5, E)
    assert out is not None, "this pair can make progress"
    new, b = out
    assert np.allclose(new, [0.5, 0.4, 0.1]), (
        f"the lesson's second example: eta = -2, the unclipped alpha_2 = 0.95 is clipped to H = 0.4, alpha_1 = 0.5; got {new}")
    assert np.isclose(b, -0.2), (
        f"alpha_1 ends at the bound C = 0.5 and alpha_2 = 0.4 is free, so b = b2 = 0 - E_2 - y_1 d1 K_12 - y_2 d2 K_22 = "
        f"0.7 + 0.6 - 1.5 = -0.2 (b1 would be -1.3, their average -0.75); got {b}")


def test_stage3_no_progress_cases(nb):
    K = X2 @ X2.T
    y_same = np.array([1.0, 1.0])
    assert nb.smo_pair_update(0, 1, np.zeros(2), 0.0, K, y_same, 1.0, -y_same) is None, (
        "same labels and both alphas 0: L = H = 0, nothing can move: return None")
    assert nb.smo_pair_update(1, 1, np.zeros(2), 0.0, K, Y2, 1.0, -Y2) is None, "i == j is not a pair: return None"
    Kz = np.ones((2, 2))  # identical points: eta = 0
    assert nb.smo_pair_update(0, 1, np.zeros(2), 0.0, Kz, Y2, 1.0, -Y2) is None, (
        "eta = 2 K_ij - K_ii - K_jj = 0 (two identical points): return None")


def test_stage3_increases_the_dual(nb):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(15, 2))
    y = np.where(rng.random(15) < 0.5, -1.0, 1.0)
    K = np.exp(-((X[:, None] - X[None]) ** 2).sum(-1))
    C = 1.0
    alpha, b = np.zeros(15), 0.0
    for _ in range(60):
        i, j = rng.choice(15, size=2, replace=False)
        E = K @ (alpha * y) + b - y
        out = nb.smo_pair_update(i, j, alpha, b, K, y, C, E)
        if out is None:
            continue
        new, b = out
        before, after = nb.dual_objective(alpha, K, y), nb.dual_objective(new, K, y)
        assert after >= before - 1e-10, f"an SMO step maximizes the dual along a line: it went from {before} to {after}"
        assert np.all(new >= -1e-12) and np.all(new <= C + 1e-12), "alphas must stay in [0, C] (clip to [L, H])"
        assert np.isclose(new @ y, alpha @ y), "the step keeps sum alpha_i y_i unchanged"
        alpha = new


# ---------------------------------------------------------------- stage 4: SVC


def test_stage4_worked_example(nb):
    m = _fit(nb, X2, np.array([0, 1]), C=10.0, kernel="linear")
    w = m.dual_coef_ @ m.support_vectors_
    assert np.allclose(w, [1.0, 0.0], atol=1e-6) and np.isclose(m.intercept_, -1.0, atol=1e-6), (
        f"two points at (0, 0) and (2, 0): the max-margin boundary is x1 = 1, with w = (1, 0) and b = -1. got "
        f"w = sum dual_coef_ * support vectors = {w}, intercept_ = {m.intercept_}")
    assert np.allclose(m.decision_function(np.array([[1.0, 5.0], [3.0, 0.0]])), [0.0, 2.0], atol=1e-6)
    assert m.n_iter_ <= m.max_passes + 5, (
        f"one pair update solves this problem, so SMO should stop after max_passes = {m.max_passes} sweeps without a "
        f"change; it ran {m.n_iter_} sweeps. reset the passes counter only when a sweep changed something")


def test_stage4_final_bias_hand_checks(nb):
    got = nb.final_bias(np.array([0.5, 0.5, 0.5]), np.array([1.0, 1.0, -1.0]), np.array([1.0, 1.0, -4.0]), 1.0, 9.0)
    assert np.isclose(got, 1.0), (
        f"three free support vectors with y - f0 = 0, 0, 3: b is their mean, 1.0, got {got}")
    got = nb.final_bias(np.zeros(2), np.array([1.0, -1.0]), np.array([3.0, -3.0]), 1.0, 9.0)
    assert np.isclose(got, 0.0), (
        f"no free SVs: alpha = 0 with y = +1 needs b >= -2 and with y = -1 needs b <= 2, so b = 0 (the midpoint); got {got}")


def test_stage4_matches_the_optimum(nb):
    from sklearn.svm import SVC
    for seed, (C, g) in enumerate([(1.0, 1.0), (10.0, 0.5), (0.1, 2.0)]):
        X, y = _circles(120, seed)
        m = _fit(nb, X, y, C=C, gamma=g, rng=0)
        ref = SVC(C=C, gamma=g, tol=1e-6).fit(X, y)
        ys = np.where(y == 1, 1.0, -1.0)
        K = np.exp(-g * ((X[:, None] - X[None]) ** 2).sum(-1))
        a_ref = np.zeros(len(X))
        a_ref[ref.support_] = np.abs(ref.dual_coef_[0])
        best = a_ref.sum() - 0.5 * (a_ref * ys) @ K @ (a_ref * ys)
        v = m.alpha_ * ys
        ours = m.alpha_.sum() - 0.5 * v @ K @ v
        assert ours >= best - 0.005 * abs(best), (
            f"C={C}, gamma={g}: your dual objective {ours:.4f} is more than 0.5% below the optimum {best:.4f}. does SMO "
            "stop too early (check the KKT test and the passes counter) or update E incorrectly?")
        Q = np.random.default_rng(100 + seed).normal(size=(400, 2))
        agree = np.mean(m.predict(Q) == ref.predict(Q))
        assert agree >= 0.97, f"C={C}, gamma={g}: predictions agree with the exact optimum on only {agree:.1%} of points"


def test_stage4_attributes_and_constraints(nb):
    X, y = _circles(100, 7)
    labels = np.array(["inner", "outer"])[y]
    m = _fit(nb, X, labels, C=2.0, gamma=1.0, rng=0)
    assert list(m.classes_) == ["inner", "outer"], "classes_ holds the two sorted labels"
    a = m.alpha_
    assert a.shape == (100,) and np.all(a >= -1e-12) and np.all(a <= 2.0 + 1e-12), "0 <= alpha_i <= C for all i"
    ys = np.where(y == 1, 1.0, -1.0)
    assert abs(a @ ys) < 1e-8, f"the dual constraint sum alpha_i y_i = 0 must hold (got {a @ ys:.2e})"
    assert np.array_equal(m.support_, np.flatnonzero(a > 1e-8)), "support_ lists the indices with alpha_i > 0"
    assert np.allclose(m.support_vectors_, X[m.support_]) and np.allclose(m.dual_coef_, (a * ys)[m.support_]), (
        "support_vectors_ = X[support_] and dual_coef_ = alpha_i y_i on the support vectors (y = +1 for classes_[1])")
    f = m.decision_function(X)
    free = (a > 2e-8) & (a < 2.0 - 2e-8)  # free: strictly inside the box (final_bias uses 1e-8 * C)
    gap = np.mean(f[free] - ys[free])
    assert free.any() and abs(gap) < 1e-6, (
        f"free support vectors sit on the margin, f(x_i) = y_i, and the intercept is chosen so they do on average: "
        f"mean(f - y) over them should be 0, got {gap:.2e}. after the loop, set intercept_ with the given final_bias, "
        "not the b of the last pair update")
    viol = np.mean(nb.kkt_violations(a, ys, f, 2.0, tol=0.01))
    assert viol < 0.05, (
        f"at the end, KKT should hold (to 0.01) almost everywhere, but {viol:.0%} of the examples violate it. look at "
        "their alphas: values like 1e-17 (or C - 1e-17) are round-off from alpha_i + y_i y_j (alpha_j - alpha_j_new); "
        "snap alpha_i to 0 or C when it's within 1e-12 * C of the bound")
    pred = m.predict(X)
    assert np.array_equal(pred, np.where(f > 0, "outer", "inner")), "predict: classes_[1] where decision_function > 0"
    with pytest.raises(NotImplementedError):
        m.predict_proba(X[:2])
        pytest.fail("an SVM gives no probabilities: predict_proba raises NotImplementedError")


def test_stage4_kernels_caps_and_errors(nb):
    X, y = _circles(60, 8)
    lin = _fit(nb, X, y, kernel="linear", C=1.0, rng=0)
    custom = _fit(nb, X, y, kernel=lambda A, B=None: A @ (A if B is None else B).T, C=1.0, rng=0)
    assert np.allclose(lin.decision_function(X), custom.decision_function(X), atol=1e-6), (
        "a callable kernel must work like the named one (build the Gram matrix with get_kernel)")
    capped = nb.SVC(C=1.0, max_iters=2, rng=0).fit(X, y)
    assert capped.n_iter_ == 2, (
        f"this problem needs many sweeps, so max_iters=2 stops SMO after exactly 2 (n_iter_ counts sweeps); got "
        f"n_iter_ = {capped.n_iter_}")
    a, b = _fit(nb, X, y, rng=3), _fit(nb, X, y, rng=3)
    assert np.allclose(a.alpha_, b.alpha_), "the same rng gives the same solution"
    with pytest.raises(ValueError):
        nb.SVC().fit(X, np.arange(60) % 3)
        pytest.fail("three classes should raise ValueError: this SVC is binary")
