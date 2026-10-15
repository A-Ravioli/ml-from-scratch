"""Checks for exercise.py (gradient descent).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Most checks use
quadratics f(x) = x^T A x / 2 - b^T x, whose iterates can be computed by hand. Every f and grad
handed to your code is wrapped in a call budget, so a loop that ignores max_iters (or a line
search that never stops) fails with a message instead of hanging.
"""
import numpy as np


class _Budget:
    """wraps a function, counts calls, and fails (instead of hanging) past `limit` calls."""

    def __init__(self, fn, limit, what="grad"):
        self.fn, self.limit, self.what, self.calls = fn, limit, what, 0

    def __call__(self, x):
        self.calls += 1
        assert self.calls <= self.limit, (
            f"{self.what} was called more than {self.limit} times: does your loop respect max_iters "
            "(and does the line search stop after its own max_iters)?")
        return self.fn(x)


def _quad(A, b=None):
    A = np.asarray(A, dtype=float)
    b = np.zeros(len(A)) if b is None else np.asarray(b, dtype=float)
    return (lambda x: float(0.5 * x @ A @ x - b @ x)), (lambda x: A @ x - b)


A2 = np.diag([2.0, 10.0])  # f = x1^2 + 5 x2^2


# ---------------------------------------------------------------- stage 1: the record and the loop


def test_stage1_optimresult_is_a_plain_record(nb):
    r = nb.OptimResult([1, 2], np.float32(0.5), np.int64(3), np.bool_(True), np.array([2.0, 1.0, 0.5, 0.5]))
    assert isinstance(r.x, np.ndarray) and r.x.dtype == np.float64 and r.x.tolist() == [1.0, 2.0], (
        f"x should be stored as a float64 ndarray; got {type(r.x).__name__} {getattr(r.x, 'dtype', '')}")
    assert type(r.fun) is float and r.fun == 0.5, f"fun should be a Python float, got {type(r.fun).__name__}"
    assert type(r.n_iters) is int and r.n_iters == 3, f"n_iters should be a Python int, got {type(r.n_iters).__name__}"
    assert r.converged is True, f"converged should be a Python bool (True), got {r.converged!r}"
    assert isinstance(r.history, list) and all(type(v) is float for v in r.history), (
        f"history should be a list of Python floats, got {type(r.history).__name__}")


def test_stage1_fixed_step_hand_iterates(nb):
    f, g = _quad(A2)
    x0 = np.array([1.0, 1.0])
    seen = []
    res = nb.gradient_descent(f, _Budget(g, 20_000), x0, lr=0.1, max_iters=3, tol=0.0, callback=lambda x, k: seen.append((np.array(x), k)))
    # x1 = (1 - 0.2, 1 - 1.0) = (0.8, 0), then x_k = (0.8^k, 0)
    assert np.allclose(res.x, [0.512, 0.0]), (
        f"with lr = 0.1, x <- x - 0.1 (2 x1, 10 x2): (1, 1) -> (0.8, 0) -> (0.64, 0) -> (0.512, 0); got {res.x}")
    assert np.allclose(res.history, [6.0, 0.64, 0.4096, 0.262144]), (
        f"history is f at every iterate starting with f(x0) = 6: [6, 0.64, 0.4096, 0.262144]; got {res.history}")
    assert res.n_iters == 3 and not res.converged, (
        f"3 steps were taken and the gradient isn't 0 (tol = 0): n_iters 3, converged False; got {res.n_iters}, "
        f"{res.converged}")
    assert np.isclose(res.fun, f(res.x)), "fun must be f at the returned x"
    assert [k for _, k in seen] == [1, 2, 3], f"callback(x, k) after each step with k = 1, 2, 3; got {[k for _, k in seen]}"
    assert np.allclose(seen[0][0], [0.8, 0.0]), f"the callback gets the NEW iterate x_k; first call got {seen[0][0]}"
    assert x0.tolist() == [1.0, 1.0], "gradient_descent must not modify x0"


def test_stage1_iterates_are_new_arrays(nb):
    f, g = _quad(A2)
    kept = []
    res = nb.gradient_descent(f, _Budget(g, 20_000), np.array([1.0, 1.0]), lr=0.1, max_iters=3, tol=0.0, callback=lambda x, k: kept.append(x))
    assert np.allclose(kept[0], [0.8, 0.0]) and np.allclose(kept[1], [0.64, 0.0]), (
        f"a caller that keeps the arrays passed to callback (without copying) should see three different iterates; "
        f"got {[np.round(x, 4).tolist() for x in kept]}. compute x = x - lr * g (a new array) instead of updating x "
        "in place with x -= ...")
    assert np.allclose(res.x, [0.512, 0.0])


def test_stage1_stops_at_tolerance(nb):
    f, g = _quad(A2)
    gb = _Budget(g, 500)
    res = nb.gradient_descent(f, gb, np.array([1.0, 1.0]), lr=0.1, max_iters=10_000, tol=1e-6)
    # x_k = (0.8^k, 0), so the gradient is (2 * 0.8^k, 0): ||g|| <= 1e-6 first at k = 66
    assert res.converged and res.n_iters == 66, (
        f"x_k = (0.8^k, 0), so the gradient is (2 * 0.8^k, 0); ||g|| <= 1e-6 first at k = 66. expected converged "
        f"after 66 steps, got converged={res.converged}, n_iters={res.n_iters}")
    assert len(res.history) == res.n_iters + 1, (
        f"history holds f(x0) plus one value per step: length n_iters + 1 = {res.n_iters + 1}, got {len(res.history)}")
    assert gb.calls <= res.n_iters + 1, (
        f"grad was called {gb.calls} times for {res.n_iters} steps: call it once per iterate (n_iters + 1 times)")


def test_stage1_already_converged_and_max_iters(nb):
    f, g = _quad(A2)
    res = nb.gradient_descent(f, _Budget(g, 20_000), np.zeros(2), lr=0.1)
    assert res.converged and res.n_iters == 0 and res.history == [0.0], (
        f"starting at the minimum: no steps, converged, history [f(x0)] = [0.0]; got {res.n_iters} steps, "
        f"converged={res.converged}, history={res.history}")
    gb = _Budget(g, 12)
    res = nb.gradient_descent(f, gb, np.array([1.0, 1.0]), lr=0.01, max_iters=10)
    assert res.n_iters == 10 and not res.converged and len(res.history) == 11, (
        f"lr = 0.01 is slow: after max_iters = 10 steps, stop with converged False (n_iters 10, 11 history values); "
        f"got n_iters={res.n_iters}, converged={res.converged}, len(history)={len(res.history)}")


def test_stage1_divergence_stops(nb):
    f, g = _quad(A2)
    gb = _Budget(g, 2_000)
    with np.errstate(all="ignore"):
        res = nb.gradient_descent(f, gb, np.array([1.0, 1.0]), lr=0.3, max_iters=100_000)
    assert not res.converged, "lr = 0.3 > 2/L = 0.2 diverges: converged must be False"
    bad = [k for k, v in enumerate(res.history) if not np.isfinite(v)]
    assert bad == [len(res.history) - 1], (
        f"stop as soon as f(x) is inf or nan: the history should end with exactly one non-finite value; non-finite "
        f"entries at positions {bad[:5]}{'...' if len(bad) > 5 else ''} of {len(res.history)}")
    assert res.n_iters < 2_000, (
        f"once f(x) is inf or nan there is nothing left to do: stop early (took {res.n_iters} steps)")


def test_stage1_matrix_shaped_parameters(nb):
    target = np.array([[1.0, -2.0], [0.5, 3.0]])
    f = lambda W: float(np.sum((W - target) ** 2))
    g = lambda W: 2 * (W - target)
    res = nb.gradient_descent(f, _Budget(g, 20_000), np.zeros((2, 2)), lr=0.25, tol=1e-10)
    assert res.x.shape == (2, 2) and np.allclose(res.x, target), (
        f"x may be any shape (here a 2 x 2 matrix): keep its shape, and use the 2-norm of the flattened gradient "
        f"(np.linalg.norm). expected {target.tolist()}, got {np.round(res.x, 4).tolist()}")


# ---------------------------------------------------------------- stage 2: Armijo backtracking


def test_stage2_worked_example(nb):
    f, g = _quad(A2)
    fb = _Budget(f, 100, "f")
    x = np.array([1.0, 1.0])
    gx = g(x)
    alpha = nb.backtracking_line_search(fb, x, f(x), gx, -gx)
    assert np.isclose(alpha, 0.125), (
        f"f = x1^2 + 5 x2^2 at (1, 1): f = 6, g = (2, 10), slope g.d = -104. alpha = 1, 0.5, 0.25 give f = 406, 80, "
        f"11.5 (too big); alpha = 0.125 gives 0.875 <= 6 - 1e-4 * 0.125 * 104. expected 0.125, got {alpha}")
    assert fb.calls == 4, (
        f"one call of f per trial (1, 0.5, 0.25, 0.125): 4 calls, got {fb.calls}. f(x) is passed in as fx: "
        "don't recompute it.")
    assert isinstance(alpha, float), "return a float"


def test_stage2_parameters_matter(nb):
    f, g = _quad(A2)
    x = np.array([1.0, 1.0])
    gx = g(x)
    a = nb.backtracking_line_search(f, x, f(x), gx, -gx, alpha0=0.05)
    assert np.isclose(a, 0.05), f"alpha0 = 0.05 already satisfies Armijo: return it unchanged, got {a}"
    a = nb.backtracking_line_search(f, x, f(x), gx, -gx, rho=0.3)
    assert np.isclose(a, 0.09), (
        f"with rho = 0.3 the trials are 1, 0.3, 0.09: f(1 - 0.18, 1 - 0.9) = 0.6724 + 0.05 passes, 0.3 gives "
        f"0.16 + 20 = 20.16 > 6. expected 0.09, got {a}")
    a = nb.backtracking_line_search(f, x, f(x), gx, -gx, c=0.9)
    assert np.isclose(a, 0.015625), (
        f"with c = 0.9 the condition f(x + a d) <= 6 - 0.9 * 104 a is much stricter: 0.125, 0.0625 and 0.03125 fail "
        f"(f = 0.875, 1.47, 3.24 against 6 - 93.6 a = -5.7, 0.15, 3.08), and 0.015625 passes (4.50 <= 4.54). "
        f"expected 0.015625, got {a}")


def test_stage2_failure_returns_zero(nb):
    f, g = _quad(A2)
    x = np.array([1.0, 1.0])
    gx = g(x)
    fb = _Budget(f, 1_000, "f")
    a = nb.backtracking_line_search(fb, x, f(x), gx, gx, max_iters=20)  # an ASCENT direction
    assert a == 0.0, f"along +g, f increases for every small step: no trial passes, so return 0.0; got {a}"
    assert fb.calls == 20, f"try exactly max_iters = 20 step sizes, then give up; f was called {fb.calls} times"


def test_stage2_survives_overflow_and_any_shape(nb):
    def f(x):
        with np.errstate(over="ignore"):
            return float(np.sum(np.exp(10 * x)))  # inf for big steps

    x = np.zeros((2, 2))
    gx = 10 * np.ones((2, 2))
    a = nb.backtracking_line_search(f, x, f(x), gx, -gx * 100, alpha0=1.0)
    want = 1.0
    while f(x - want * gx * 100) > f(x) - 1e-4 * want * 100 * np.sum(gx * gx):
        want *= 0.5
    assert np.isclose(a, want), (
        f"x and the direction may be matrices: the slope is the sum of gx * direction over all entries. a trial "
        f"where f is inf or nan simply fails the test. expected {want}, got {a}")


# ---------------------------------------------------------------- stage 3: gradient descent with a line search


def test_stage3_first_step_uses_backtracking(nb):
    f, g = _quad(A2)
    seen = []
    nb.gradient_descent(f, _Budget(g, 20_000), np.array([1.0, 1.0]), lr=1.0, line_search=True, max_iters=1,
                        callback=lambda x, k: seen.append(np.array(x)))
    assert len(seen) == 1 and np.allclose(seen[0], [0.75, -0.25]), (
        f"with line_search=True and lr = 1.0, the first step backtracks from alpha0 = lr to alpha = 0.125: "
        f"(1, 1) - 0.125 (2, 10) = (0.75, -0.25); got {seen[0] if seen else 'no callback call'}")


def test_stage3_alpha0_is_lr(nb):
    f, g = _quad(A2)
    seen = []
    nb.gradient_descent(f, _Budget(g, 20_000), np.array([1.0, 1.0]), lr=0.05, line_search=True, max_iters=1,
                        callback=lambda x, k: seen.append(np.array(x)))
    assert np.allclose(seen[0], [0.9, 0.5]), (
        f"the line search starts from alpha0 = lr. with lr = 0.05 the first trial already passes Armijo, so the step "
        f"is (1, 1) - 0.05 (2, 10) = (0.9, 0.5); got {seen[0]}. (starting from alpha0 = 1 gives (0.75, -0.25).)")


def test_stage3_line_search_rescues_a_big_lr(nb):
    f, g = _quad(A2)
    gb = _Budget(g, 5_000)
    with np.errstate(all="ignore"):
        fixed = nb.gradient_descent(f, _Budget(g, 20_000), np.array([1.0, 1.0]), lr=1.0, max_iters=50)
    res = nb.gradient_descent(f, gb, np.array([1.0, 1.0]), lr=1.0, line_search=True, max_iters=2_000, tol=1e-8)
    assert not fixed.converged, "with a fixed lr = 1 (far above 2/L = 0.2) gradient descent diverges"
    assert res.converged and np.allclose(res.x, 0, atol=1e-8), (
        f"with the line search, lr = 1 is just the first trial step: it should converge to 0; got converged="
        f"{res.converged}, x={res.x}")
    assert all(b <= a for a, b in zip(res.history, res.history[1:])), (
        "every accepted Armijo step decreases f: the history must be non-increasing")


def test_stage3_stops_when_the_line_search_fails(nb):
    # |x| is not smooth: the gradient is +-1 right up to 0, so tiny |x| leaves no acceptable step
    fb = _Budget(lambda x: float(abs(x[0])), 50_000, "f")
    res = nb.gradient_descent(fb, lambda x: np.sign(x), np.array([0.7]), lr=1.0, line_search=True,
                              max_iters=100_000, tol=1e-12)
    assert not res.converged and res.n_iters < 100_000, (
        f"on |x| the gradient never gets small, so tol is never met. once |x| is below ~1e-15 no step passes "
        f"Armijo and backtracking_line_search returns 0.0: stop there (converged False). got n_iters={res.n_iters}, "
        f"converged={res.converged}")
    assert abs(res.x[0]) < 1e-6, f"before stopping it should get very close to 0; got {res.x}"


def test_stage3_rosenbrock_progress(nb):
    f = lambda v: float((1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2)
    g = lambda v: np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2), 200 * (v[1] - v[0] ** 2)])
    res = nb.gradient_descent(f, _Budget(g, 3_001), np.array([-1.2, 1.0]), lr=1.0, line_search=True, max_iters=3_000)
    assert res.fun < 1e-3 and all(b <= a for a, b in zip(res.history, res.history[1:])), (
        f"on Rosenbrock from (-1.2, 1), 3,000 line-search steps get f below 1e-3 with f never increasing; got "
        f"f = {res.fun:.3g}")


# ---------------------------------------------------------------- stage 4: exact line search on quadratics


def test_stage4_exact_step_value(nb):
    a = nb.exact_step(A2, np.array([2.0, 10.0]))
    assert np.isclose(a, 104 / 1008), (
        f"alpha* = g.g / g.Ag = 104 / (2 * 4 + 10 * 100) = 0.10317; got {a}")
    assert nb.exact_step(A2, np.zeros(2)) == 0.0, "a zero gradient means no step: return 0.0 (not nan)"


def test_stage4_worked_example_zigzag(nb):
    A = np.diag([1.0, 10.0])
    xs = np.asarray(nb.steepest_descent_quadratic(A, np.zeros(2), np.array([10.0, 1.0]), 4), dtype=float)
    want = np.array([(9 / 11) ** k * np.array([10.0, (-1.0) ** k]) for k in range(5)])
    assert xs.shape == (5, 2), f"4 steps give 5 iterates including x0: shape (5, 2), got {xs.shape}"
    assert np.allclose(xs, want), (
        f"from (10, 1) on diag(1, 10) the exact step is 2/11 every time and x_k = (9/11)^k (10, (-1)^k): "
        f"expected {np.round(want, 4).tolist()}, got {np.round(xs, 4).tolist()}")


def test_stage4_orthogonal_gradients_and_kantorovich(nb):
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    eigs = np.array([1.0, 2.0, 5.0, 10.0, 25.0])
    A = Q @ np.diag(eigs) @ Q.T
    b = rng.normal(size=5)
    x0 = rng.normal(size=5)
    xs = np.asarray(nb.steepest_descent_quadratic(A, b, x0, 30), dtype=float)
    G = xs @ A - b
    cos = np.array([G[k] @ G[k + 1] / (np.linalg.norm(G[k]) * np.linalg.norm(G[k + 1])) for k in range(10)])
    assert np.allclose(cos, 0, atol=1e-8), (
        f"with an exact line search, each new gradient is orthogonal to the previous one (that's the zig-zag); "
        f"cosines between successive gradients: {np.round(cos, 6)}")
    x_star = np.linalg.solve(A, b)
    f = lambda x: 0.5 * x @ A @ x - b @ x
    gaps = np.array([f(x) - f(x_star) for x in xs[:12]])
    bound = ((25 - 1) / (25 + 1)) ** 2
    assert np.all(gaps[1:] <= bound * gaps[:-1] * (1 + 1e-9)), (
        f"Kantorovich: each step shrinks f - f* by at least ((kappa - 1)/(kappa + 1))^2 = {bound:.4f}; ratios "
        f"{np.round(gaps[1:] / gaps[:-1], 4)}")
    assert x0.tolist() == xs[0].tolist(), "the first iterate is x0, and x0 must not be modified"
