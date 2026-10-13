"""Checks for capstone.py (an autodiff-powered Newton solver).

All problems are synthetic and deterministic: polynomial systems, Rosenbrock, a double well,
and hand-made error sequences. Newton iterates are compared with a reference Newton that uses
hand-derived (exact) Jacobians, so a finite-difference Jacobian is caught by its ~1e-9 drift.
Under pytest, `mlfs.autodiff` is the reference implementation.
"""
import numpy as np
import pytest

ROOT_WALLIS = 2.0945514815423265


def _circle(v):
    return np.array([v[0] ** 2 + v[1] ** 2 - 4.0, v[0] * v[1] - 1.0], dtype=object)


def _circle_J(v):
    return np.array([[2 * v[0], 2 * v[1]], [v[1], v[0]]])


def _rosen(v):
    return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2


def _orders(e):
    r = np.log(e[1:] / e[:-1])
    return r[1:] / r[:-1]


class _Counter:
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, x):
        self.calls += 1
        return self.f(x)


# ---------------------------------------------------------------- stage 1: Newton for systems


def test_stage1_babylonian_iterates(nb):
    x, its = nb.newton_root(lambda v: v * v - 2.0, np.array([1.0]))
    its = np.asarray(its, dtype=float)
    assert its.ndim == 2 and its.shape[1] == 1, f"iterates should have shape (k + 1, n) = (k + 1, 1); got {its.shape}"
    want = [1.0, 1.5, 17 / 12, 577 / 408, 665857 / 470832]
    assert np.allclose(its[:5, 0], want, rtol=1e-14, atol=0), (
        f"Newton on x^2 - 2 from 1 visits 1, 1.5, 1.41667, 1.4142157, 1.41421356237469 (the iterates must start "
        f"with x0); got {its[:5, 0].tolist()}")
    assert np.allclose(np.asarray(x), its[-1]) and abs(float(np.asarray(x)[0]) ** 2 - 2) <= 1e-10, (
        "x_final should be the last iterate, with |F(x)| <= tol")


def test_stage1_system_matches_exact_newton(nb):
    x0 = np.array([2.0, 0.5])
    x, its = nb.newton_root(_circle, x0, tol=1e-12)
    its = np.asarray(its, dtype=float)
    ref = [x0]
    for _ in range(3):
        v = ref[-1]
        ref.append(v - np.linalg.solve(_circle_J(v), np.asarray(_circle(v), dtype=float)))
    assert np.allclose(its[:4], np.array(ref), rtol=0, atol=1e-13), (
        "your first Newton iterates differ from Newton with the exact Jacobian [[2x, 2y], [y, x]] by up to "
        f"{np.abs(its[:4] - np.array(ref)).max():.1e}. use mlfs.autodiff.jacobian (exact), not finite differences, "
        "and solve J s = -F(x).")
    assert np.linalg.norm(np.asarray(_circle(np.asarray(x)), dtype=float)) <= 1e-12, "the final point should solve the system"
    assert len(its) <= 8, f"from (2, 0.5) Newton should converge in a handful of steps; took {len(its) - 1}"


def test_stage1_quadratic_convergence(nb):
    _, its = nb.newton_root(lambda v: v**3 - 2 * v - 5, np.array([3.0]), tol=1e-15)
    e = np.abs(np.asarray(its, dtype=float)[:, 0] - ROOT_WALLIS)
    e = e[e > 1e-14]
    q = _orders(e)
    assert len(q) >= 2 and abs(q[-1] - 2) < 0.15, f"Newton at a simple root is quadratic; order estimates {np.round(q, 3)}"


def test_stage1_stopping_rules(nb):
    _, its = nb.newton_root(lambda v: v - 3.0, np.array([3.0]))
    assert np.asarray(its).shape == (1, 1), (
        f"if x0 already satisfies ||F(x0)|| <= tol, return without stepping: iterates shape (1, 1); got {np.asarray(its).shape}")
    calls = [0]

    def no_root(v):
        calls[0] += 1
        if calls[0] > 500:
            raise AssertionError("F was called more than 500 times for max_iters = 7: does your loop respect max_iters?")
        return v * v + 1.0

    _, its = nb.newton_root(no_root, np.array([0.5]), max_iters=7)
    assert np.asarray(its).shape == (8, 1), (
        f"x^2 + 1 has no real root: after max_iters = 7 steps there are 8 iterates; got {np.asarray(its).shape[0]}")
    _, its = nb.newton_root(lambda v: v * v - 2.0, np.array([1.0]), tol=1e-3)
    assert np.asarray(its).shape == (4, 1), (
        "with tol = 1e-3, stop at the first iterate with |x^2 - 2| <= 1e-3: that's 577/408, the 4th iterate "
        f"(|F| = 6e-6; 17/12 has |F| = 7e-3); got {np.asarray(its).shape[0]} iterates")


# ---------------------------------------------------------------- stage 2: exact Hessians


def test_stage2_rosenbrock_exact(nb):
    x = np.array([-1.2, 1.0])
    H = np.asarray(nb.hessian(_rosen, x), dtype=float)
    want = np.array([[2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]], [-400 * x[0], 200.0]])
    assert H.shape == (2, 2), f"expected a (2, 2) Hessian, got {H.shape}"
    assert np.allclose(H, want, rtol=1e-13, atol=0), (
        f"the Rosenbrock Hessian at (-1.2, 1) is {want.tolist()}; got {H.tolist()} (max error "
        f"{np.abs(H - want).max():.1e}). nested dual numbers give it EXACTLY; an error near 1e-6 means finite differences.")


def test_stage2_transcendental_and_symmetric(nb):
    def f(v):
        return np.exp(v[0] * v[1]) + np.sin(v[1]) * v[2] ** 2 + np.log(1 + v[0] ** 2) * np.tanh(v[2])

    x = np.array([0.3, -0.7, 1.1])
    a, b, c = x
    e = np.exp(a * b)
    want = np.array([
        [b * b * e + np.tanh(c) * (2 - 2 * a * a) / (1 + a * a) ** 2, e + a * b * e, 2 * a / (1 + a * a) * (1 - np.tanh(c) ** 2)],
        [0.0, a * a * e - np.sin(b) * c * c, 2 * c * np.cos(b)],
        [0.0, 0.0, 2 * np.sin(b) - 2 * np.log(1 + a * a) * np.tanh(c) * (1 - np.tanh(c) ** 2)],
    ])
    want = np.triu(want) + np.triu(want, 1).T
    counter = _Counter(f)
    H = np.asarray(nb.hessian(counter, x), dtype=float)
    assert np.allclose(H, want, rtol=1e-12, atol=1e-14), f"Hessian off by up to {np.abs(H - want).max():.1e}"
    assert np.array_equal(H, H.T), "the Hessian must be exactly symmetric: compute each pair once, store it twice"
    assert counter.calls <= 9, f"one evaluation per pair (i, j) is enough: at most n^2 = 9 calls, you used {counter.calls}"


def test_stage2_constant_and_quadratic(nb):
    A = np.array([[2.0, 1.0, 0.0], [3.0, -1.0, 2.0], [0.5, 0.0, 4.0]])
    H = np.asarray(nb.hessian(lambda v: v @ A @ v + 7.0, np.array([1.0, -2.0, 0.5])), dtype=float)
    assert np.allclose(H, A + A.T, rtol=1e-14), "the Hessian of x^T A x is A + A^T, for any square A"
    H = np.asarray(nb.hessian(lambda v: 3.0, np.zeros(2)), dtype=float)
    assert np.array_equal(H, np.zeros((2, 2))), "a function that ignores its input has a zero Hessian (handle a plain-number result)"


# ---------------------------------------------------------------- stage 3: damped Newton minimization


def test_stage3_quadratic_in_one_step(nb):
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, -2.0])
    f = lambda v: 0.5 * v @ A @ v - b @ v
    x, its = nb.newton_minimize(f, np.array([10.0, -7.0]))
    its = np.asarray(its, dtype=float)
    assert np.allclose(its[1], np.linalg.solve(A, b), atol=1e-10), (
        "on a convex quadratic, one full Newton step (t = 1, no damping) lands exactly on the minimum A^-1 b")
    assert len(its) == 2, f"...and then the gradient is ~0, so it stops: 2 iterates expected, got {len(its)}"


def test_stage3_rosenbrock(nb):
    x, its = nb.newton_minimize(_rosen, np.array([-1.2, 1.0]), tol=1e-10)
    its = np.asarray(its, dtype=float)
    assert np.allclose(np.asarray(x), [1.0, 1.0], atol=1e-8), f"Rosenbrock's minimum is (1, 1); you ended at {x}"
    assert len(its) - 1 <= 40, f"damped Newton should solve Rosenbrock from (-1.2, 1) in about 20 steps; took {len(its) - 1}"
    vals = np.array([_rosen(v) for v in its])
    assert np.all(np.diff(vals) <= 1e-12), (
        "f increased at some step: the backtracking line search must guarantee f(x + t d) <= f(x) + 1e-4 t g.d")


def test_stage3_avoids_the_saddle(nb):
    # double well: minima at (+-1, 0), a saddle at (0, 0). pure Newton from (0.3, 1) goes to the saddle.
    f = lambda v: (v[0] ** 2 - 1) ** 2 / 4 + v[1] ** 2 / 2
    x, its = nb.newton_minimize(f, np.array([0.3, 1.0]))
    x = np.asarray(x, dtype=float)
    assert np.allclose(np.abs(x), [1.0, 0.0], atol=1e-8), (
        f"starting at (0.3, 1), where the Hessian is indefinite, damped Newton must still reach a MINIMUM (+-1, 0); "
        f"you ended at {np.round(x, 6)}" + (" (the saddle: make H positive definite before solving)" if abs(x[0]) < 1e-3 else ""))
    vals = np.array([f(v) for v in np.asarray(its, dtype=float)])
    assert np.all(np.diff(vals) <= 1e-12), "f must never increase"


def test_stage3_armijo_not_just_decrease(nb):
    # f = sqrt(1 + x^2): from x0 just below 1, the full Newton step lands at about -x0^3, which DOES decrease f,
    # but only by ~1e-5, far less than the Armijo condition asks for (1e-4 * |g.d| ~ 1.4e-4). so t is halved once.
    f = lambda v: np.sqrt(1 + v[0] ** 2)
    x0 = 0.99999
    _, its = nb.newton_minimize(f, np.array([x0]), max_iters=1)
    x1 = float(np.asarray(its, dtype=float)[1, 0])
    want = x0 * (1 - x0**2) / 2  # x0 + 0.5 * d with d = -x0 (1 + x0^2)
    assert np.isclose(x1, want, atol=1e-9), (
        f"from x0 = 0.99999 the first iterate should be x0 + d/2 = {want:.3e} (t halved once by the Armijo test "
        f"f(x + t d) <= f(x) + 1e-4 t g.d); got {x1:.6f}"
        + (". you accepted t = 1 because f merely decreased: require SUFFICIENT decrease." if x1 < -0.9 else ""))


def test_stage3_hessian_fix_is_bounded(nb):
    calls = {"n": 0}

    def f(v):
        calls["n"] += 1
        if calls["n"] > 5000:
            raise AssertionError("newton_minimize called f more than 5000 times: is a loop (the Hessian fix or the "
                                 "line search) missing its cap?")
        return v[0] ** 4 - v[0] ** 2 + v[1] ** 2

    x, its = nb.newton_minimize(f, np.array([0.1, 0.5]), max_iters=50)
    assert np.allclose(np.abs(np.asarray(x, dtype=float)), [np.sqrt(0.5), 0.0], atol=1e-7), (
        f"x^4 - x^2 + y^2 has minima at (+-1/sqrt 2, 0); from (0.1, 0.5) you ended at {x}")


def test_stage3_stops_when_line_search_fails(nb):
    calls = {"n": 0}

    def f(v):
        calls["n"] += 1
        if calls["n"] > 2000:
            raise AssertionError(
                "newton_minimize called f more than 2000 times: when the line search finds no t with "
                "sufficient decrease, stop and return instead of taking the step anyway and trying again")
        smooth = (v[0] - 1.0) ** 2 + 0.5 * v[1] ** 2
        # rigged: plain-float trial points (the line search) always look worse than the current point,
        # while Dual evaluations (gradient, Hessian) see the smooth function. no t can pass Armijo.
        is_dual = isinstance(v, np.ndarray) and v.dtype == object
        return smooth if is_dual else smooth + 1e6

    x0 = np.array([3.0, -2.0])
    x, its = nb.newton_minimize(f, x0, tol=1e-12, max_iters=500)
    assert calls["n"] < 500, (
        f"newton_minimize used {calls['n']} evaluations of f although no trial step ever decreased it. "
        "if the 60 halvings all fail the Armijo test, stop: return without stepping")
    assert np.allclose(np.asarray(x, dtype=float), x0), (
        f"the line search never found a decrease, so x must stay at x0 = {x0}; you moved to {x}")


# ---------------------------------------------------------------- stage 4: order estimates


def test_stage4_clean_sequences(nb):
    assert np.isclose(nb.estimate_order(0.5 ** np.arange(1, 15)), 1.0), "e_k = 0.5^k is linear: order 1"
    assert np.isclose(nb.estimate_order(0.5 ** (2.0 ** np.arange(5))), 2.0), "e_(k+1) = e_k^2 is quadratic: order 2"
    e = [0.5, 0.4]
    for _ in range(8):
        e.append(0.8 * e[-1] * e[-2])  # secant-like recursion: order -> golden ratio
    got = nb.estimate_order(np.array(e))
    assert abs(got - 1.618) < 0.05, f"e_(k+1) = C e_k e_(k-1) converges with order (1 + sqrt 5)/2 = 1.618; got {got:.4f}"


def test_stage4_ignores_the_rounding_floor(nb):
    e = np.array([1e-1, 1e-2, 1e-4, 1e-8, 3e-16, 2e-16, 4e-16])
    got = nb.estimate_order(e)
    assert np.isclose(got, 2.0), (
        f"the last three errors are rounding noise: drop errors <= floor (1e-13) before estimating; the clean part "
        f"1e-1, 1e-2, 1e-4, 1e-8 is quadratic, so the answer is 2; got {got}")


def test_stage4_median_of_last_three(nb):
    # build errors whose consecutive order estimates are exactly [1.5, 1.0, 2.0, 9.0, 1.9]
    orders = [1.5, 1.0, 2.0, 9.0, 1.9]
    r = [np.log(0.9)]  # log(e_1 / e_0)
    for q in orders:
        r.append(q * r[-1])
    e = 1.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    got = nb.estimate_order(e)
    hint = ""
    if np.isclose(got, 1.9):
        hint = " 1.9 is the last estimate alone (or the median of all five): use the median of the LAST THREE."
    elif np.isclose(got, np.mean(orders)):
        hint = " that's the mean of all the estimates: one wild estimate (9.0) drags a mean around; use a median."
    assert np.isclose(got, 2.0, rtol=1e-9), (
        f"these errors have order estimates {orders}; the median of the last three (2.0, 9.0, 1.9) is 2.0; "
        f"got {got}.{hint}")


def test_stage4_floor_is_inclusive(nb):
    got = nb.estimate_order(np.array([1e-1, 1e-2, 1e-3, 1e-13]))
    assert np.isclose(got, 1.0), (
        f"errors <= floor (1e-13) are dropped, so only 1e-1, 1e-2, 1e-3 remain: one estimate, 1.0; got {got}")


def test_stage4_too_few_errors(nb):
    with pytest.raises(ValueError):
        nb.estimate_order(np.array([1e-2, 1e-4, 1e-15, 1e-16]))
