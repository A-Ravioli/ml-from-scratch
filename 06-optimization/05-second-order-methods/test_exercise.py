"""Checks for exercise.py (second-order methods).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Newton is checked
on hand-computable 1-D problems, on quadratics (one step) and on a double well where pure Newton
heads for a maximum. BFGS and L-BFGS are checked against the secant condition, against a dense
re-implementation of the update, and by behaviour on Rosenbrock and ill-conditioned quadratics.
Every f/grad/hess handed to your code is wrapped in a call budget, so loops can't run away.
"""
import numpy as np
import pytest


class _Budget:
    def __init__(self, fn, limit, what="grad"):
        self.fn, self.limit, self.what, self.calls = fn, limit, what, 0

    def __call__(self, x):
        self.calls += 1
        assert self.calls <= self.limit, (
            f"{self.what} was called more than {self.limit} times: does your loop respect max_iters, and does it stop "
            "when the line search returns 0?")
        return self.fn(x)


def _rosen(v):
    return float((1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2)


def _rosen_grad(v):
    return np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2), 200 * (v[1] - v[0] ** 2)])


def _rosen_hess(v):
    return np.array([[2 - 400 * (v[1] - 3 * v[0] ** 2), -400 * v[0]], [-400 * v[0], 200.0]])


def _quad(eigs, seed=0, b=True):
    rng = np.random.default_rng(seed)
    n = len(eigs)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = Q @ np.diag(eigs) @ Q.T
    bb = rng.normal(size=n) if b else np.zeros(n)
    return A, bb, (lambda x: float(0.5 * x @ A @ x - bb @ x)), (lambda x: A @ x - bb)


def _pairs(n, m, seed):
    """m random pairs (s, y) with s . y > 0, from a random SPD matrix."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    B = M @ M.T + n * np.eye(n)
    S = [rng.normal(size=n) for _ in range(m)]
    return S, [B @ s + 0.1 * rng.normal(size=n) for s in S]


def _dense_bfgs(H, s, y):
    rho = 1 / (y @ s)
    I = np.eye(len(s))
    return (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)


class _Count:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, x):
        self.calls += 1
        return self.fn(x)


def _rigged(x0):
    """f = ||x||^2 + 1 everywhere except exactly at x0: every trial step of a line search looks worse, so
    the very first line search must fail."""
    x0 = np.array(x0, dtype=float)
    return (lambda x: float(x @ x) + (0.0 if np.array_equal(x, x0) else 1.0)), (lambda x: 2 * x)


# ---------------------------------------------------------------- stage 1: Newton


def test_stage1_hand_iterates(nb):
    f = lambda v: float(np.exp(v[0]) - 2 * v[0])
    g = lambda v: np.array([np.exp(v[0]) - 2])
    h = lambda v: np.array([[np.exp(v[0])]])
    seen = []
    res = nb.newton(f, _Budget(g, 50), _Budget(h, 50, "hess"), np.array([0.0]), tol=1e-12,
                    callback=lambda x, k: seen.append(float(x[0])))
    assert np.allclose(seen[:2], [1.0, 2 / np.e]), (
        f"Newton on e^x - 2x from 0: x1 = 0 - (1 - 2)/1 = 1, x2 = 1 - (e - 2)/e = 2/e = 0.7358 (both full steps pass "
        f"the Armijo test); got {np.round(seen[:2], 6).tolist()}")
    assert res.converged and abs(res.x[0] - np.log(2)) < 1e-12 and res.n_iters <= 6, (
        f"it should reach the minimizer ln 2 = 0.693147 to 1e-12 in at most 6 steps (quadratic convergence); got "
        f"x = {res.x}, {res.n_iters} steps")
    e = np.abs(np.array(seen[:4]) - np.log(2))
    assert e[2] < 2 * e[1] ** 2 and e[3] < 2 * e[2] ** 2, (
        f"the errors {e} should roughly square at every step (Newton's quadratic convergence)")


def test_stage1_quadratic_in_one_step(nb):
    A, b, f, g = _quad([1.0, 10.0, 100.0, 1000.0])
    res = nb.newton(f, g, lambda x: A, np.zeros(4), tol=1e-8)
    assert res.n_iters == 1 and np.allclose(res.x, np.linalg.solve(A, b)), (
        f"on a quadratic, the Newton step solves A d = -g exactly: one step to the minimizer, whatever the condition "
        f"number (here 1000); got {res.n_iters} steps, x = {res.x}")


def test_stage1_damping(nb):
    A, b, f, g = _quad([1.0, 4.0])
    x0 = np.array([1.0, -1.0])
    seen = []
    nb.newton(f, g, lambda x: A, x0, damping=3.0, line_search=False, max_iters=1, callback=lambda x, k: seen.append(np.array(x)))
    want = x0 - np.linalg.solve(A + 3.0 * np.eye(2), g(x0))
    assert np.allclose(seen[0], want), (
        f"with damping 3 the step solves (H + 3 I) d = -g: expected {want}, got {seen[0]}")


def test_stage1_pure_newton_finds_a_maximum(nb):
    f = lambda v: float(v[0] ** 4 / 4 - v[0] ** 2 / 2)
    g = lambda v: np.array([v[0] ** 3 - v[0]])
    h = lambda v: np.array([[3 * v[0] ** 2 - 1]])
    pure = nb.newton(f, _Budget(g, 200), h, np.array([0.2]), line_search=False, max_iters=50)
    safe = nb.newton(f, _Budget(g, 200), h, np.array([0.2]), max_iters=50)
    assert abs(pure.x[0]) < 1e-6, (
        f"pure Newton (line_search=False) jumps to the stationary point of the local quadratic model. at 0.2 the "
        f"curvature is negative, so that's the MAXIMUM at 0; it should converge there, got {pure.x}")
    assert abs(abs(safe.x[0]) - 1) < 1e-6 and safe.fun < -0.2, (
        f"with line_search=True, a Newton direction with g . d >= 0 isn't a descent direction: fall back to -g. that "
        f"walks downhill to a minimum at +-1 (f = -0.25); got x = {safe.x}, f = {safe.fun}")


def test_stage1_singular_hessian_falls_back(nb):
    f = lambda v: float(v[0] ** 4 + v[1] ** 2)
    g = lambda v: np.array([4 * v[0] ** 3, 2 * v[1]])
    h = lambda v: np.array([[12 * v[0] ** 2, 0.0], [0.0, 2.0]])
    res = nb.newton(f, _Budget(g, 300), h, np.array([0.0, 1.0]), max_iters=100)
    assert res.converged and np.allclose(res.x, 0, atol=1e-8), (
        f"at (0, 1) the Hessian diag(0, 2) is singular, so the solve fails (np.linalg.LinAlgError): fall back to "
        f"d = -g instead of crashing. got {res}")


def test_stage1_stops_when_the_line_search_fails(nb):
    f, g = _rigged([0.5, 0.5])
    gb = _Budget(g, 50)
    res = nb.newton(f, gb, lambda x: 2 * np.eye(2), np.array([0.5, 0.5]), tol=0.0, max_iters=100_000)
    assert res.n_iters == 0 and not res.converged and np.allclose(res.x, [0.5, 0.5]), (
        f"here every trial point looks worse than x0, so backtracking_line_search returns 0.0 on the first step: "
        f"stop right there (n_iters 0, converged False, x = x0) instead of looping; got {res}")


def test_stage1_any_shape(nb):
    T = np.arange(6.0).reshape(2, 3) / 10
    f = lambda W: float(np.sum((W - T) ** 2) + np.sum(W**4))
    g = lambda W: 2 * (W - T) + 4 * W**3
    def h(W):
        assert np.shape(W) == (2, 3), f"hess must receive x in x0's shape (2, 3), got {np.shape(W)}"
        return np.diag((2 + 12 * W**2).ravel())

    seen = []
    res = nb.newton(f, g, h, np.zeros((2, 3)), tol=1e-10, callback=lambda x, k: seen.append(np.shape(x)))
    assert res.x.shape == (2, 3) and np.allclose(g(res.x), 0, atol=1e-9), (
        f"x0 may be a matrix: work on the flattened vector (hess returns the (6, 6) Hessian of the flattened x), but "
        f"give f, grad, hess and callback x in x0's shape and return it in that shape; got x of shape "
        f"{np.shape(res.x)}")
    assert set(seen) == {(2, 3)}, f"callback must receive x in x0's shape, got shapes {set(seen)}"


def test_stage1_conventions(nb):
    gb, hb = _Budget(_rosen_grad, 200), _Budget(_rosen_hess, 200, "hess")
    x0 = np.array([-1.2, 1.0])
    ks = []
    res = nb.newton(_rosen, gb, hb, x0, callback=lambda x, k: ks.append(k))
    assert res.converged and np.allclose(res.x, [1, 1], atol=1e-8) and res.n_iters <= 30, (
        f"damped Newton should solve Rosenbrock from (-1.2, 1) in at most 30 steps; got {res}")
    assert all(b <= a for a, b in zip(res.history, res.history[1:])), "with the line search, f never increases"
    assert ks == list(range(1, res.n_iters + 1)) and len(res.history) == res.n_iters + 1, "callback and history conventions"
    assert hb.calls <= res.n_iters, f"one Hessian per step: {hb.calls} calls for {res.n_iters} steps"
    assert x0.tolist() == [-1.2, 1.0] and type(res).__name__ == "OptimResult", "copy x0; return an OptimResult"
    res = nb.newton(_rosen, _Budget(_rosen_grad, 10), _rosen_hess, x0, max_iters=3)
    assert res.n_iters == 3 and not res.converged, "stop after max_iters steps"


# ---------------------------------------------------------------- stage 2: Wolfe line search and BFGS


def test_stage2_wolfe_bisects_and_expands(nb):
    f = lambda x: float(x @ x)
    g = lambda x: 2 * x
    a, f_new, g_new = nb.wolfe_line_search(f, g, np.array([1.0]), 1.0, np.array([2.0]), np.array([-2.0]))
    assert np.isclose(a, 0.5) and np.isclose(f_new, 0.0) and np.allclose(g_new, [0.0]), (
        f"f = x^2 at 1 along d = -2: alpha = 1 lands on -1 with f = 1 (Armijo fails, so hi = 1); the bisection "
        f"alpha = 0.5 lands on 0 with slope 0 >= 0.9 * (-4): accept, returning (0.5, 0.0, [0.0]); got ({a}, {f_new}, "
        f"{g_new})")
    a, f_new, g_new = nb.wolfe_line_search(f, g, np.array([1.0]), 1.0, np.array([2.0]), np.array([-0.01]))
    assert np.isclose(a, 16.0) and np.isclose(f_new, 0.84**2) and np.allclose(g_new, [1.68]), (
        f"along the tiny direction d = -0.01, every step up to alpha = 8 passes Armijo but fails the curvature "
        f"condition (the slope is still steep, so the step is too short): double alpha (1, 2, 4, 8, 16) until "
        f"grad . d >= 0.9 * slope, at alpha = 16; got ({a}, {f_new}, {g_new}). plain backtracking would return 1.")


def test_stage2_wolfe_guarantees_curvature_and_fails_cleanly(nb):
    A, b, f, g = _quad([0.5, 3.0, 40.0], seed=10)
    rng = np.random.default_rng(11)
    for _ in range(10):
        x = rng.normal(size=3)
        d = -g(x) * rng.uniform(0.01, 3.0)
        a, f_new, g_new = nb.wolfe_line_search(f, g, x, f(x), g(x), d)
        s, y = a * d, g_new - g(x)
        assert a > 0 and s @ y > 0 and np.isclose(f_new, f(x + a * d)), (
            "an accepted Wolfe step must return f and grad at x + alpha d, and give s . y > 0 (that's why BFGS uses it)")
    fb = _Budget(f, 30, "f")
    x = np.ones(3)
    a, f_new, g_new = nb.wolfe_line_search(fb, g, x, f(x), g(x), g(x), max_iters=12)
    assert a == 0.0 and f_new is None and g_new is None and fb.calls == 12, (
        f"along an ascent direction nothing passes: return (0.0, None, None) after exactly max_iters = 12 trials; got "
        f"({a}, {f_new}, {g_new}) after {fb.calls} calls of f")


def test_stage2_wolfe_bisects_between_brackets(nb):
    # f = x^2 for x >= 0 and a steep wall 100 x^2 for x < 0; from 1 along d = -0.1, with c2 = 0.1 the acceptable
    # steps are about [9, 10]: 1, 2, 4, 8 are too short, 16 and 12 overshoot into the wall, the midpoint 10 works
    f = lambda x: float(x[0] ** 2 if x[0] >= 0 else 100 * x[0] ** 2)
    g = lambda x: np.array([2 * x[0] if x[0] >= 0 else 200 * x[0]])
    a, f_new, g_new = nb.wolfe_line_search(f, g, np.array([1.0]), 1.0, np.array([2.0]), np.array([-0.1]), c2=0.1)
    assert np.isclose(a, 10.0), (
        f"trials 1, 2, 4, 8 fail the curvature test (lo = 8), 16 fails Armijo (hi = 16), then bisect the BRACKET: "
        f"(lo + hi)/2 = 12 fails Armijo (hi = 12), (8 + 12)/2 = 10 is accepted. expected 10, got {a}. the next trial "
        "is (lo + hi) / 2, not hi / 2.")


def test_stage2_bfgs_first_iterates(nb):
    A = np.diag([1.0, 4.0])
    fc, gc = _Count(lambda x: float(0.5 * x @ A @ x)), _Count(lambda x: A @ x)
    xs = []
    nb.bfgs(fc, gc, np.array([1.0, 1.0]), max_iters=2, tol=0.0, callback=lambda x, k: xs.append(np.array(x)))
    assert np.allclose(xs[0], [0.5, -1.0]), f"step 1 is -g with the Wolfe step 0.5: x1 = (0.5, -1); got {xs[0]}"
    assert np.allclose(xs[1], [0.543071, -0.033942], atol=1e-6), (
        f"step 2: s = (-0.5, -2), y = (-0.5, -8), and BEFORE the first update Hinv is rescaled to (s.y)/(y.y) I = "
        f"0.2529 I, then BFGS-updated; with the Wolfe step that gives x2 = (0.543071, -0.033942). got {xs[1]}"
        + (" (that's x2 without the rescale)" if np.allclose(xs[1], [-0.034083, 0.002130], atol=1e-5) else ""))
    # count the calls a correct bfgs makes: f(x0), grad(x0), then only what the two line searches use
    rf, rg = _Count(fc.fn), _Count(gc.fn)
    x, H = np.array([1.0, 1.0]), np.eye(2)
    fx, gx = rf(x), rg(x)
    for k in range(2):
        d = -H @ gx
        a, fx, g_new = nb.wolfe_line_search(rf, rg, x, fx, gx, d)
        s_, y_ = a * d, g_new - gx
        if k == 0:
            H = (s_ @ y_) / (y_ @ y_) * np.eye(2)
        H = _dense_bfgs(H, s_, y_)
        x, gx = x + s_, g_new
    assert (fc.calls, gc.calls) == (rf.calls, rg.calls), (
        f"two BFGS steps should cost exactly f(x0), grad(x0) and the line searches' own calls: {rf.calls} f and "
        f"{rg.calls} grad calls; yours made {fc.calls} and {gc.calls}. reuse f_new and g_new from wolfe_line_search.")


def test_stage2_stops_when_the_line_search_fails(nb):
    f, g = _rigged([0.5, 0.5])
    res = nb.bfgs(_Budget(f, 200, "f"), _Budget(g, 200), np.array([0.5, 0.5]), tol=0.0, max_iters=100_000)
    assert res.n_iters == 0 and not res.converged and np.allclose(res.x, [0.5, 0.5]), (
        f"every trial point looks worse than x0, so wolfe_line_search returns alpha = 0: stop (n_iters 0, converged "
        f"False) rather than taking some tiny step; got {res}")


def test_stage2_update_secant_and_symmetry(nb):
    rng = np.random.default_rng(1)
    S, Y = _pairs(5, 1, 2)
    s, y = S[0], Y[0]
    M = rng.normal(size=(5, 5))
    H = M @ M.T + np.eye(5)
    H_before = H.copy()
    H1 = np.asarray(nb.bfgs_update(H, s, y))
    assert np.allclose(H1 @ y, s), "the updated matrix must satisfy the secant condition H_new y = s"
    assert np.allclose(H1, H1.T) and np.linalg.eigvalsh(H1).min() > 0, "it must stay symmetric positive definite"
    assert np.allclose(H1, _dense_bfgs(H, s, y)), (
        "the BFGS inverse update is (I - rho s y^T) H (I - rho y s^T) + rho s s^T with rho = 1/(y . s)")
    assert np.array_equal(H, H_before), "bfgs_update must return a new matrix and leave Hinv unchanged"


def test_stage2_update_is_inverse_of_hessian_update(nb):
    S, Y = _pairs(4, 1, 3)
    s, y = S[0], Y[0]
    B = np.diag([1.0, 2.0, 3.0, 4.0])
    B1 = B - np.outer(B @ s, B @ s) / (s @ B @ s) + np.outer(y, y) / (y @ s)
    H1 = np.asarray(nb.bfgs_update(np.linalg.inv(B), s, y))
    assert np.allclose(H1, np.linalg.inv(B1)), (
        "updating the INVERSE with your formula must give the inverse of the BFGS update of the Hessian approximation "
        "B (Sherman-Morrison-Woodbury): check rho and the order of the outer products")


def test_stage2_bfgs_rosenbrock(nb):
    gb = _Budget(_rosen_grad, 400)
    res = nb.bfgs(_Budget(_rosen, 2_000, "f"), gb, np.array([-1.2, 1.0]))
    assert res.converged and np.allclose(res.x, [1, 1], atol=1e-5) and res.n_iters <= 60, (
        f"BFGS should solve Rosenbrock from (-1.2, 1) to ||grad|| <= 1e-6 in at most 60 steps (gradient descent "
        f"needs thousands); got {res}")
    assert all(b <= a for a, b in zip(res.history, res.history[1:])), "every accepted Wolfe step decreases f"
    assert gb.calls <= 2 * res.n_iters + 1, (
        f"{gb.calls} grad calls for {res.n_iters} steps: reuse f_new and g_new from the line search instead of "
        "evaluating them again")


def test_stage2_bfgs_superlinear_on_quadratic(nb):
    A, b, f, g = _quad(np.geomspace(1, 1000, 10), seed=4)
    res = nb.bfgs(f, _Budget(g, 500), np.zeros(10), tol=1e-7)
    x_star = np.linalg.solve(A, b)
    assert res.converged and np.allclose(res.x, x_star, atol=1e-8) and res.n_iters <= 70, (
        f"on a 10-D quadratic with condition number 1000, BFGS needs a few dozen steps (gradient descent: thousands); "
        f"got {res.n_iters} steps, converged={res.converged}")


def test_stage2_nonconvex_does_not_break(nb):
    f = lambda v: float(np.sum(v**4 / 4 - v**2 / 2) + 0.1 * v[0] * v[1])
    g = lambda v: v**3 - v + 0.1 * v[::-1]
    res = nb.bfgs(f, _Budget(g, 1_000), np.array([0.3, -0.05]), max_iters=500)
    assert res.converged and np.all(np.isfinite(res.x)) and res.fun < -0.4, (
        f"on a nonconvex double well, skip updates that fail the curvature condition (s . y <= 0) and reset to -g "
        f"when the direction isn't a descent direction: BFGS should still reach a minimum (f < -0.4); got {res}")


# ---------------------------------------------------------------- stage 3: L-BFGS


def test_stage3_two_loop_matches_dense_bfgs(nb):
    S, Y = _pairs(6, 4, 5)
    g = np.random.default_rng(6).normal(size=6)
    g_before = g.copy()
    gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
    H = gamma * np.eye(6)
    for s, y in zip(S, Y):
        H = _dense_bfgs(H, s, y)
    got = np.asarray(nb.two_loop(g, S, Y))
    assert np.allclose(got, H @ g), (
        f"two_loop must equal H g, where H starts at gamma I (gamma from the NEWEST pair) and is BFGS-updated with each "
        f"pair, oldest first: expected {np.round(H @ g, 5)}, got {np.round(got, 5)}. the first loop goes newest -> "
        "oldest, the second oldest -> newest, reusing each pair's alpha.")
    assert np.array_equal(g, g_before), "two_loop must not modify g"


def test_stage3_two_loop_edge_cases(nb):
    g = np.array([1.0, -2.0, 3.0])
    got = np.asarray(nb.two_loop(g, [], []))
    assert np.allclose(got, g), f"with no pairs, H = I and two_loop returns g itself (gamma = 1); got {got}"
    S, Y = _pairs(3, 1, 7)
    assert np.allclose(np.asarray(nb.two_loop(Y[0], S, Y)), S[0]), "with pairs, H y_newest = s_newest (the secant condition)"


def test_stage3_lbfgs_problems(nb):
    gb = _Budget(_rosen_grad, 300)
    res = nb.lbfgs(_rosen, gb, np.array([-1.2, 1.0]))
    assert res.converged and np.allclose(res.x, [1, 1], atol=1e-5) and res.n_iters <= 60, (
        f"L-BFGS should solve Rosenbrock from (-1.2, 1) in at most 60 steps; got {res}. if it crawls with tiny steps, "
        "check that you use the Wolfe line search (Armijo alone lets s . y go negative, and the history stops "
        "learning)")
    A, b, f, g = _quad(np.geomspace(1, 1e4, 200), seed=8)
    res = nb.lbfgs(f, _Budget(g, 3_000), np.zeros(200), history_size=10, tol=1e-4, max_iters=2_000)
    assert res.converged and res.n_iters <= 1_000, (
        f"on a 200-D quadratic with condition number 1e4, L-BFGS with 10 pairs should converge in well under 1,000 "
        f"steps (gradient descent would need ~1e5); got {res.n_iters} steps, converged={res.converged}")


def test_stage3_lbfgs_uses_exactly_the_last_pairs(nb):
    A, b, f, g = _quad([1.0, 3.0, 10.0, 30.0, 100.0], seed=12)
    xs = []
    nb.lbfgs(f, g, np.zeros(5), history_size=2, max_iters=8, tol=0.0, callback=lambda x, k: xs.append(np.array(x)))
    x, S, Y = np.zeros(5), [], []
    want = []
    fx, gx = f(x), g(x)
    for _ in range(len(xs)):
        d = -np.asarray(nb.two_loop(gx, S[-2:], Y[-2:]))
        a, fx, g_new = nb.wolfe_line_search(f, g, x, fx, gx, d)
        S.append(a * d)
        Y.append(g_new - gx)
        x, gx = x + a * d, g_new
        want.append(x)
    assert len(xs) == 8 and np.allclose(xs, want, atol=1e-10), (
        "with history_size=2, each direction must use exactly the LAST TWO (s, y) pairs (oldest first): your "
        f"iterates differ from that from step {int(np.argmax(np.any(~np.isclose(xs, want, atol=1e-10), axis=1))) + 1} on. "
        "keep a deque(maxlen=history_size).")


def test_stage3_stops_when_the_line_search_fails(nb):
    f, g = _rigged([0.5, 0.5])
    res = nb.lbfgs(_Budget(f, 200, "f"), _Budget(g, 200), np.array([0.5, 0.5]), tol=0.0, max_iters=100_000)
    assert res.n_iters == 0 and not res.converged, f"wolfe_line_search returns alpha = 0 here: stop; got {res}"


def test_stage3_history_size_matters(nb):
    A, b, f, g = _quad(np.geomspace(1, 1e3, 50), seed=9)
    short = nb.lbfgs(f, _Budget(g, 8_000), np.zeros(50), history_size=1, tol=1e-5, max_iters=3_000)
    long = nb.lbfgs(f, _Budget(g, 8_000), np.zeros(50), history_size=20, tol=1e-5, max_iters=3_000)
    assert long.converged and long.n_iters < short.n_iters, (
        f"more history should help on an ill-conditioned quadratic: history_size 20 took {long.n_iters} steps, "
        f"history_size 1 took {short.n_iters}. keep only the LAST history_size pairs (a deque with maxlen).")


# ---------------------------------------------------------------- stage 4: minimize


def test_stage4_dispatches_and_forwards(nb):
    from mlfs import optim

    for method, opts in (("gd", dict(lr=0.002)), ("heavy_ball", dict(lr=0.002, momentum=0.8)), ("nesterov", dict(lr=0.002)),
                         ("bfgs", {}), ("lbfgs", dict(history_size=3))):
        seen = []
        res = nb.minimize(_rosen, _Budget(_rosen_grad, 1_000), np.array([-1.2, 1.0]), method=method, max_iters=150,
                          tol=1e-9, callback=lambda x, k: seen.append(k), **opts)
        direct = getattr(optim, {"gd": "gradient_descent"}.get(method, method))(
            _rosen, _rosen_grad, np.array([-1.2, 1.0]), max_iters=150, tol=1e-9, **opts)
        assert np.allclose(res.x, direct.x) and res.n_iters == direct.n_iters, (
            f"minimize(method={method!r}, ...) must call {method} with max_iters, tol and callback passed through and "
            f"the options {opts} forwarded: it gave x = {res.x} after {res.n_iters} steps, the direct call x = "
            f"{direct.x} after {direct.n_iters}")
        assert seen == list(range(1, res.n_iters + 1)), f"{method}: pass the callback through"


def test_stage4_matrix_shaped_x0(nb):
    T = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    f = lambda W: float(np.sum((W - T) ** 2))
    g = lambda W: 2 * (W - T)
    for method, kw in (("gd", dict(lr=0.25)), ("heavy_ball", dict(lr=0.1, momentum=0.5)), ("nesterov", dict(lr=0.25)),
                       ("newton", dict(hess=lambda W: 2 * np.eye(6))), ("bfgs", {}), ("lbfgs", {})):
        shapes = set()
        res = nb.minimize(f, _Budget(g, 5_000), np.zeros((2, 3)), method=method, tol=1e-8,
                          callback=lambda x, k: shapes.add(np.shape(x)), **kw)
        assert shapes == {(2, 3)}, f"method={method!r}: callback must receive x in x0's shape (2, 3); got {shapes}"
        assert np.shape(res.x) == (2, 3) and np.allclose(res.x, T, atol=1e-6), (
            f"method={method!r}: x0 of shape (2, 3) (a weight matrix) must work and come back in that shape; got "
            f"shape {np.shape(res.x)}")


def test_stage4_newton_and_default(nb):
    res = nb.minimize(_rosen, _rosen_grad, np.array([-1.2, 1.0]), method="newton", hess=_rosen_hess, tol=1e-10)
    assert res.converged and np.allclose(res.x, [1, 1]), f"method='newton' with hess should solve Rosenbrock; got {res}"
    with pytest.raises(ValueError):
        nb.minimize(_rosen, _rosen_grad, np.array([-1.2, 1.0]), method="newton")
    with pytest.raises(ValueError):
        nb.minimize(_rosen, _rosen_grad, np.array([-1.2, 1.0]), method="adam")
    res = nb.minimize(_rosen, _Budget(_rosen_grad, 1_000), np.array([-1.2, 1.0]))
    assert res.converged and np.allclose(res.x, [1, 1], atol=1e-5), "the default method is L-BFGS, which should converge"
