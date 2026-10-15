"""Checks for exercise.py (momentum & acceleration).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The hand examples
are 1-D quadratics f = 2 x^2 (so grad = 4 x), where every iterate can be computed on paper; the
rate checks use diagonal quadratics, where the theory is exact. Every grad handed to your code is
wrapped in a call budget, so a loop that ignores max_iters fails instead of hanging.
"""
import numpy as np


class _Budget:
    def __init__(self, fn, limit, what="grad"):
        self.fn, self.limit, self.what, self.calls = fn, limit, what, 0

    def __call__(self, x):
        self.calls += 1
        assert self.calls <= self.limit, (
            f"{self.what} was called more than {self.limit} times: does your loop respect max_iters?")
        return self.fn(x)


def _quad(eigs):
    A = np.diag(np.asarray(eigs, dtype=float))
    return A, (lambda x: float(0.5 * x @ A @ x)), (lambda x: A @ x)


F1 = lambda x: float(2 * x[0] ** 2)  # noqa: E731
G1 = lambda x: 4 * x  # noqa: E731


def _path(method, x0, n, **kw):
    seen = [np.array(x0, dtype=float)]
    method(F1, _Budget(G1, 2 * n + 5), np.array(x0, dtype=float), max_iters=n, tol=0.0,
           callback=lambda x, k: seen.append(np.array(x)), **kw)
    return np.array(seen)[:, 0]


def _iters_to(method, A, f, g, x0, rel, **kw):
    tol = rel * np.linalg.norm(g(x0))
    return method(f, _Budget(g, 100_000), x0, tol=tol, max_iters=50_000, **kw)


# ---------------------------------------------------------------- stage 1: heavy ball


def test_stage1_hand_iterates(nb):
    got = _path(nb.heavy_ball, [1.0], 3, lr=0.1, momentum=0.5)
    want = [1.0, 0.6, 0.16, -0.124]
    assert np.allclose(got, want), (
        f"on f = 2x^2 (grad 4x) with lr 0.1 and momentum 0.5 from 1: x1 = 1 - 0.4 = 0.6 (no velocity yet), "
        f"x2 = 0.6 - 0.24 + 0.5 (0.6 - 1) = 0.16, x3 = 0.16 - 0.064 + 0.5 (0.16 - 0.6) = -0.124. expected {want}, "
        f"got {np.round(got, 6).tolist()}")


def test_stage1_zero_momentum_is_gradient_descent(nb):
    got = _path(nb.heavy_ball, [1.0], 4, lr=0.1, momentum=0.0)
    assert np.allclose(got, 0.6 ** np.arange(5)), (
        f"with momentum 0 heavy ball is gradient descent: x_k = 0.6^k; got {np.round(got, 6).tolist()}")


def test_stage1_loop_conventions(nb):
    A, f, g = _quad([1.0, 10.0])
    x0 = np.array([1.0, 1.0])
    gb = _Budget(g, 1_000)
    kept = []
    res = nb.heavy_ball(f, gb, x0, lr=0.05, momentum=0.5, tol=1e-8, max_iters=900, callback=lambda x, k: kept.append((x, k)))
    assert res.converged and np.allclose(res.x, 0, atol=1e-7), f"should converge to 0; got {res}"
    assert len(res.history) == res.n_iters + 1 and np.isclose(res.history[0], 5.5), (
        "history is [f(x0), f(x1), ...]: length n_iters + 1, starting with f(x0) = 5.5")
    assert gb.calls <= res.n_iters + 1, f"one grad call per iterate: {gb.calls} calls for {res.n_iters} steps"
    assert [k for _, k in kept] == list(range(1, res.n_iters + 1)), "callback(x, k) after every step, k = 1, 2, ..."
    assert not np.allclose(kept[0][0], kept[1][0]), (
        "the arrays passed to callback must be different iterates: build a new array each step (no in-place update)")
    assert x0.tolist() == [1.0, 1.0], "heavy_ball must not modify x0"
    assert type(res).__name__ == "OptimResult", "return an OptimResult (import it from mlfs.optim)"


def test_stage1_max_iters_and_divergence(nb):
    A, f, g = _quad([1.0, 10.0])
    res = nb.heavy_ball(f, _Budget(g, 12), np.ones(2), lr=0.001, momentum=0.5, max_iters=10)
    assert res.n_iters == 10 and not res.converged, f"stop after max_iters = 10 steps, unconverged; got {res.n_iters}"
    with np.errstate(all="ignore"):
        res = nb.heavy_ball(f, _Budget(g, 5_000), np.ones(2), lr=0.5, momentum=0.9, max_iters=100_000)
    bad = [i for i, v in enumerate(res.history) if not np.isfinite(v)]
    assert not res.converged and bad == [len(res.history) - 1], (
        "lr 0.5 diverges on this quadratic (L = 10): stop as soon as f is not finite, with converged False")


def test_stage1_acceleration_on_ill_conditioned_quadratic(nb):
    A, f, g = _quad(np.geomspace(1.0, 100.0, 8))
    x0 = np.ones(8)
    gd = nb.heavy_ball(f, _Budget(g, 50_000), x0, lr=1 / 100, momentum=0.0, tol=1e-6 * np.linalg.norm(g(x0)), max_iters=20_000)
    hb = _iters_to(nb.heavy_ball, A, f, g, x0, 1e-6, lr=4 / 121, momentum=(9 / 11) ** 2)
    assert hb.converged and hb.n_iters < 120 and gd.n_iters > 800, (
        f"kappa = 100: gradient descent (momentum 0, lr 1/L) needs {gd.n_iters} steps to shrink the gradient by 1e6; "
        f"heavy ball with lr = 4/(sqrt(L) + sqrt(mu))^2, momentum = ((sqrt(kappa) - 1)/(sqrt(kappa) + 1))^2 should need "
        f"well under 120; got {hb.n_iters} (converged={hb.converged})")


# ---------------------------------------------------------------- stage 2: Nesterov


def test_stage2_constant_momentum_hand_iterates(nb):
    got = _path(nb.nesterov, [1.0], 3, lr=0.1, momentum=0.5)
    want = [1.0, 0.6, 0.24, 0.036]
    assert np.allclose(got, want), (
        f"with constant momentum 0.5: x1 = 0.6 (y = x0), then y = 0.6 + 0.5 (0.6 - 1) = 0.4 and x2 = 0.4 - 0.1 * 1.6 "
        f"= 0.24, then y = 0.24 + 0.5 (0.24 - 0.6) = 0.06 and x3 = 0.06 * 0.6 = 0.036. the gradient is taken at the "
        f"look-ahead point y, not at x. expected {want}, got {np.round(got, 6).tolist()}")


def test_stage2_schedule_hand_iterates(nb):
    got = _path(nb.nesterov, [1.0], 3, lr=0.1)
    want = [1.0, 0.6, 0.3, 0.108]
    assert np.allclose(got, want), (
        f"momentum=None uses beta_k = (k - 1)/(k + 2): 0, 1/4, 2/5 for k = 1, 2, 3. so x1 = 0.6, y = 0.6 + 0.25 (-0.4) "
        f"= 0.5, x2 = 0.5 * 0.6 = 0.3, y = 0.3 + 0.4 (0.3 - 0.6) = 0.18, x3 = 0.18 * 0.6 = 0.108. expected {want}, "
        f"got {np.round(got, 6).tolist()}")


def test_stage2_zero_momentum_is_not_the_schedule(nb):
    got = _path(nb.nesterov, [1.0], 4, lr=0.1, momentum=0.0)
    assert np.allclose(got, 0.6 ** np.arange(5)), (
        f"momentum=0.0 is a constant momentum of 0 (plain gradient descent, x_k = 0.6^k), not 'use the schedule': "
        f"test `momentum is None`, not `not momentum`. got {np.round(got, 6).tolist()}")


def test_stage2_conventions_and_grad_budget(nb):
    A, f, g = _quad([1.0, 10.0])
    gb = _Budget(g, 2_000)
    res = nb.nesterov(f, gb, np.array([1.0, 1.0]), lr=0.1, momentum=0.5, tol=1e-8, max_iters=900)
    assert res.converged and np.allclose(res.x, 0, atol=1e-7), f"should converge to 0; got {res}"
    assert len(res.history) == res.n_iters + 1, "history has n_iters + 1 entries"
    assert gb.calls <= 2 * res.n_iters + 1, (
        f"at most two grad calls per step (at y, and at x_k for the stopping test) plus one at x0: "
        f"{gb.calls} calls for {res.n_iters} steps")
    res = nb.nesterov(f, _Budget(g, 25), np.ones(2), lr=0.001, max_iters=10)
    assert res.n_iters == 10 and not res.converged, "stop after max_iters steps"


def test_stage2_accelerates_on_the_worst_case(nb):
    n = 101
    T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    A, b = T / 4, np.eye(n)[0] / 4  # L = 1
    f = lambda x: float(0.5 * x @ A @ x - b @ x)
    g = lambda x: A @ x - b
    f_star = f(np.linalg.solve(A, b))
    nest = nb.nesterov(f, _Budget(g, 50_000), np.zeros(n), lr=1.0, max_iters=50, tol=0.0)
    gd = nb.nesterov(f, _Budget(g, 50_000), np.zeros(n), lr=1.0, momentum=0.0, max_iters=50, tol=0.0)
    r_n, r_g = nest.fun - f_star, gd.fun - f_star
    assert r_n < 0.4 * r_g, (
        f"on Nesterov's worst-case quadratic (convex, not strongly), 50 accelerated steps (schedule) should beat 50 "
        f"gradient steps by a wide margin: f - f* = {r_n:.4g} vs {r_g:.4g}")


# ---------------------------------------------------------------- stage 3: rates on quadratics


def _iteration_matrix_radius(eigs, lr, beta, nesterov):
    n = len(eigs)
    A = np.diag(eigs)
    I = np.eye(n)
    if nesterov:
        top = np.hstack([(1 + beta) * (I - lr * A), -beta * (I - lr * A)])
    else:
        top = np.hstack([(1 + beta) * I - lr * A, -beta * I])
    M = np.vstack([top, np.hstack([I, np.zeros((n, n))])])
    return np.max(np.abs(np.linalg.eigvals(M)))


def test_stage3_hand_radii(nb):
    hb = nb.momentum_spectral_radius([4.0], 0.1, 0.5)
    ne = nb.momentum_spectral_radius([4.0], 0.1, 0.5, nesterov=True)
    gd = nb.momentum_spectral_radius([4.0], 0.1, 0.0)
    assert np.isclose(hb, np.sqrt(0.5)), (
        f"heavy ball on lam = 4, lr 0.1, momentum 0.5: z^2 - 1.1 z + 0.5 has complex roots of size sqrt(0.5) = 0.7071; "
        f"got {hb}")
    assert np.isclose(ne, np.sqrt(0.3)), (
        f"nesterov: a = 1.5 * 0.6 = 0.9, b = 0.5 * 0.6 = 0.3, complex roots of size sqrt(0.3) = 0.5477; got {ne}")
    assert np.isclose(gd, 0.6), (
        f"with momentum 0 the roots are 1 - lr lam = 0.6 and 0: radius 0.6 (plain gradient descent); got {gd}")
    assert isinstance(hb, float), "return a float"


def test_stage3_matches_iteration_matrix(nb):
    rng = np.random.default_rng(0)
    for trial in range(6):
        eigs = rng.uniform(0.1, 10.0, size=5)
        lr = rng.uniform(0.01, 0.3)
        beta = rng.uniform(0.0, 0.95)
        for nest in (False, True):
            want = _iteration_matrix_radius(eigs, lr, beta, nest)
            got = nb.momentum_spectral_radius(eigs, lr, beta, nesterov=nest)
            assert np.isclose(got, want, rtol=1e-6), (
                f"{'nesterov' if nest else 'heavy ball'}, eigs {np.round(eigs, 3)}, lr {lr:.3f}, momentum {beta:.3f}: "
                f"the spectral radius of the 2n x 2n iteration matrix is {want:.6f}, got {got:.6f}. take the larger "
                "|root| for EVERY eigenvalue (real or complex roots), then the max over eigenvalues.")


def test_stage3_tuned_parameters(nb):
    lr, beta = nb.tuned_momentum(100.0, 1.0)
    assert np.isclose(lr, 4 / 121) and np.isclose(beta, (9 / 11) ** 2), (
        f"heavy ball, L = 100, mu = 1: lr = 4/(10 + 1)^2 = 0.03306, momentum = (9/11)^2 = 0.6694; got ({lr}, {beta})")
    lr, beta = nb.tuned_momentum(100.0, 1.0, nesterov=True)
    assert np.isclose(lr, 0.01) and np.isclose(beta, 9 / 11), (
        f"nesterov, L = 100, mu = 1: lr = 1/L = 0.01, momentum = (10 - 1)/(10 + 1) = 0.8182; got ({lr}, {beta})")
    eigs = np.linspace(1.0, 100.0, 50)
    r_hb = nb.momentum_spectral_radius(eigs, *nb.tuned_momentum(100.0, 1.0))
    r_ne = nb.momentum_spectral_radius(eigs, *nb.tuned_momentum(100.0, 1.0, nesterov=True), nesterov=True)
    assert np.isclose(r_hb, 9 / 11, rtol=1e-6) and np.isclose(r_ne, 0.9, rtol=1e-6), (
        f"tuned on [1, 100], heavy ball's rate is (sqrt(k) - 1)/(sqrt(k) + 1) = 0.8182 and nesterov's is "
        f"1 - 1/sqrt(k) = 0.9 (compare 0.99 for gradient descent with lr 1/L); got {r_hb:.6f}, {r_ne:.6f}")


def test_stage3_prediction_matches_your_optimizer(nb):
    eigs = np.array([1.0, 3.0, 10.0, 30.0])
    A, f, g = _quad(eigs)
    lr, beta = 0.02, 0.6
    seen = []
    nb.heavy_ball(f, _Budget(g, 50_000), np.ones(4), lr=lr, momentum=beta, max_iters=150, tol=0.0, callback=lambda x, k: seen.append(np.linalg.norm(x)))
    measured = (seen[149] / seen[99]) ** (1 / 50)
    predicted = nb.momentum_spectral_radius(eigs, lr, beta)
    assert abs(measured - predicted) < 0.01, (
        f"the measured per-step contraction of your heavy_ball (steps 100 to 150) is {measured:.4f}, the predicted "
        f"spectral radius {predicted:.4f}: they should agree")


# ---------------------------------------------------------------- stage 4: the lower bound


def test_stage4_worst_case_structure(nb):
    A, b = nb.worst_case_quadratic(5, L=8.0)
    T = 2 * np.eye(5) - np.eye(5, k=1) - np.eye(5, k=-1)
    assert np.asarray(A).shape == (5, 5) and np.allclose(A, 2.0 * T), (
        f"A = (L/4) T with T tridiagonal (2 on the diagonal, -1 beside it): for L = 8 that's 2 T; got\n{np.asarray(A)}")
    assert np.allclose(b, [2.0, 0, 0, 0, 0]), f"b = (L/4) e_1 = (2, 0, 0, 0, 0); got {b}"
    x_star = np.linalg.solve(A, b)
    assert np.allclose(x_star, 1 - np.arange(1, 6) / 6), "its minimizer should be x*_i = 1 - i/(n + 1)"
    assert np.linalg.eigvalsh(A).max() <= 8.0, "the function must be L-smooth: largest eigenvalue <= L"


def test_stage4_bound_value(nb):
    got = nb.first_order_lower_bound(2.0, 3.0, 5)
    assert np.isclose(got, 3 * 2 * 9 / (32 * 36)), f"3 L R^2 / (32 (k + 1)^2) with L 2, R 3, k 5 = 0.046875; got {got}"


def test_stage4_iterates_live_in_the_krylov_subspace(nb):
    A, b = nb.worst_case_quadratic(12, L=1.0)
    A, b = np.asarray(A), np.asarray(b)
    f = lambda x: float(0.5 * x @ A @ x - b @ x)
    g = lambda x: A @ x - b
    for name, method, kw in (("heavy_ball", nb.heavy_ball, dict(lr=1.0, momentum=0.5)),
                             ("nesterov", nb.nesterov, dict(lr=1.0))):
        xs = []
        method(f, _Budget(g, 20), np.zeros(12), max_iters=6, tol=0.0, callback=lambda x, k: xs.append(np.array(x)), **kw)
        for k, x in enumerate(xs, start=1):
            assert np.all(x[k:] == 0) and x[k - 1] != 0, (
                f"{name}: on the tridiagonal worst case from 0, iterate x_{k} should have exactly its first {k} "
                f"coordinates nonzero (each gradient reaches one coordinate further); got {np.round(x, 4)}")


def test_stage4_no_method_beats_the_bound(nb):
    for k in (3, 10, 25):
        n = 2 * k + 1
        A, b = nb.worst_case_quadratic(n, L=1.0)
        A, b = np.asarray(A), np.asarray(b)
        f = lambda x: float(0.5 * x @ A @ x - b @ x)
        g = lambda x: A @ x - b
        x_star = np.linalg.solve(A, b)
        bound = nb.first_order_lower_bound(1.0, np.linalg.norm(x_star), k)
        for name, res in (("nesterov", nb.nesterov(f, _Budget(g, 50_000), np.zeros(n), lr=1.0, max_iters=k, tol=0.0)),
                          ("heavy_ball", nb.heavy_ball(f, _Budget(g, 50_000), np.zeros(n), lr=1.0, momentum=0.8, max_iters=k, tol=0.0))):
            gap = res.fun - f(x_star)
            assert gap >= bound * (1 - 1e-9), (
                f"{name} after k = {k} steps on the n = {n} worst case: f - f* = {gap:.4g}, below the lower bound "
                f"{bound:.4g}. that's impossible for a first-order method: check the bound's formula.")
