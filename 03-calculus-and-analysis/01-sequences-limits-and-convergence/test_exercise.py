"""Checks for exercise.py (sequences, limits & convergence).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Error sequences
are built here from exact formulas, so a bug in your estimator can't hide behind a bug in a
root finder.
"""
import numpy as np
import pytest

ROOT_WALLIS = 2.0945514815423265  # the real root of x^3 - 2x - 5 (Wallis's example)


def _wallis(x):
    return x**3 - 2 * x - 5


def _wallis_d(x):
    return 3 * x**2 - 2


def _capped(f, limit=2000):
    """wrap f so that a loop missing its iteration cap fails with a message instead of hanging."""
    calls = [0]

    def g(x):
        calls[0] += 1
        if calls[0] > limit:
            raise AssertionError(f"f was called more than {limit} times: does your loop respect max_iters?")
        return f(x)

    return g


def _orders(e):
    r = np.log(e[1:] / e[:-1])
    return r[1:] / r[:-1]


# ---------------------------------------------------------------- stage 1: epsilon-N


def test_stage1_hand_example(nb):
    a = np.array([0.9, 0.05, 0.4, 0.2, 0.08, 0.03, 0.01])
    got = nb.tail_index(a, 0.0, 0.1)
    assert got == 4, (
        f"a = {a.tolist()}, limit 0, eps 0.1: every term from index 4 on is within 0.1, but index 3 "
        f"(0.2) is not, so N = 4. got {got}. "
        + ("1 is the FIRST index inside the band; the sequence wanders back out at indices 2 and 3. "
           "N must work for every later term." if got == 1 else
           "3 is the last index OUTSIDE the band; N is the one after it." if got == 3 else ""))


def test_stage1_strict_inequality(nb):
    a = np.array([1.0, 0.5, 0.25, 0.125])
    got = nb.tail_index(a, 0.0, 0.5)
    assert got == 2, (
        f"with eps = 0.5 the term 0.5 is NOT within eps (|a_n - L| < eps is strict), so N = 2; got {got}")


def test_stage1_edge_cases(nb):
    a = np.array([0.01, -0.02, 0.005])
    assert nb.tail_index(a, 0.0, 0.1) == 0, "every term is within eps, so N = 0"
    b = np.array([0.01, 0.02, 0.5])
    got = nb.tail_index(b, 0.0, 0.1)
    assert got == 3, (
        f"the LAST term (0.5) is outside the band, so no N within the array works: return K = 3; got {got}")
    assert isinstance(nb.tail_index(a, 0.0, 0.1), (int, np.integer)), "return an int"


def test_stage1_one_over_n(nb):
    n = np.arange(1, 10001)
    a = 3.0 + (-1.0) ** n / n
    for eps, want in ((0.15, 6), (0.012, 83), (0.00105, 952)):
        got = nb.tail_index(a, 3.0, eps)
        # |a_n - 3| = 1/n < eps  <=>  n > 1/eps; array index i holds n = i + 1
        assert got == want, (
            f"a_n = 3 + (-1)^n / n, eps = {eps}: |a_n - 3| = 1/n < eps exactly when n > {1 / eps:.2f}, "
            f"i.e. from n = {want + 1}, which is array index {want} (index i holds n = i + 1); got {got}")


# ---------------------------------------------------------------- stage 2: Cauchy diameters


def test_stage2_hand_example(nb):
    a = np.array([3.0, 1.0, 2.0, 1.5])
    got = np.asarray(nb.cauchy_diameters(a), dtype=float)
    want = np.array([2.0, 1.0, 0.5, 0.0])
    assert got.shape == (4,), f"one diameter per starting index N: expected shape (4,), got {got.shape}"
    assert np.allclose(got, want), (
        f"for a = {a.tolist()}: d[0] = max - min of all terms = 3 - 1 = 2, d[1] = 2 - 1 = 1, "
        f"d[2] = 2 - 1.5 = 0.5, d[3] = 0. expected {want.tolist()}, got {got.tolist()}"
        + (". you returned |a[N+1] - a[N]| (adjacent gaps); the diameter takes EVERY pair m, n >= N."
           if np.allclose(got[:3], np.abs(np.diff(a))) else ""))


def test_stage2_matches_brute_force(nb):
    a = np.random.default_rng(0).normal(size=300).cumsum() * 0.1
    got = np.asarray(nb.cauchy_diameters(a), dtype=float)
    want = np.array([a[N:].max() - a[N:].min() for N in range(len(a))])
    assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape}"
    bad = int(np.argmax(np.abs(got - want)))
    assert np.allclose(got, want), (
        f"on a random walk, d[{bad}] should be max(a[{bad}:]) - min(a[{bad}:]) = {want[bad]:.6f}, got "
        f"{got[bad]:.6f}. if only the end of the array is right, check that you reversed the running "
        "max/min back into the original order.")
    assert np.all(np.diff(got) <= 1e-15), "tail diameters can only shrink as N grows"


def test_stage2_harmonic_is_not_cauchy(nb):
    # partial sums of 1/n: consecutive terms get arbitrarily close, but the tails keep spreading
    n = np.arange(1, 20001)
    H = np.cumsum(1.0 / n)
    d = np.asarray(nb.cauchy_diameters(H), dtype=float)
    assert np.isclose(d[99], H[-1] - H[99]), (
        "for the increasing harmonic partial sums the tail diameter from index 99 is H[-1] - H[99] "
        f"= {H[-1] - H[99]:.4f} (about log(20000/100)), got {d[99]:.4f}. adjacent gaps would give 1/101.")


# ---------------------------------------------------------------- stage 3: order and rate


def test_stage3_linear(nb):
    e = 0.5 ** np.arange(1, 12)
    q = np.asarray(nb.order_estimates(e), dtype=float)
    assert q.shape == (9,), f"n = 11 errors give n - 2 = 9 order estimates; got shape {q.shape}"
    assert np.allclose(q, 1.0), f"e_k = 0.5^k converges linearly: every estimate should be 1, got {q[:3]}..."
    C = np.asarray(nb.rate_estimates(e, 1.0), dtype=float)
    assert C.shape == (10,) and np.allclose(C, 0.5), (
        f"rate_estimates(e, q=1) should be e_(k+1) / e_k = 0.5, shape (10,); got shape {C.shape}, values {C[:3]}")


def test_stage3_quadratic_and_cubic(nb):
    e2 = 0.5 ** (2.0 ** np.arange(6))  # e_{k+1} = e_k^2
    q = np.asarray(nb.order_estimates(e2), dtype=float)
    hint = ""
    if np.allclose(q, 0.5):
        hint = " you got 1/2: the ratio is upside down. q = log(e_(k+2)/e_(k+1)) / log(e_(k+1)/e_k)."
    assert np.allclose(q, 2.0), f"e_(k+1) = e_k^2 is quadratic: every estimate should be 2, got {q}.{hint}"
    assert np.allclose(nb.rate_estimates(e2, 2.0), 1.0), "e_(k+1) / e_k^2 = 1 for e_(k+1) = e_k^2"
    e3 = 0.9 ** (3.0 ** np.arange(5))
    assert np.allclose(nb.order_estimates(e3), 3.0), "e_(k+1) = e_k^3 is cubic: estimates should be 3"


def test_stage3_rate_constant(nb):
    # e_{k+1} = 0.3 e_k^2 exactly
    e = [0.1]
    for _ in range(4):
        e.append(0.3 * e[-1] ** 2)
    e = np.array(e)
    C = np.asarray(nb.rate_estimates(e, 2.0), dtype=float)
    assert np.allclose(C, 0.3), (
        f"for e_(k+1) = 0.3 e_k^2 the rate estimates e_(k+1) / e_k^2 should all be 0.3, got {np.round(C, 4)}")


def test_stage3_rejects_zero_errors(nb):
    with pytest.raises(ValueError):
        nb.order_estimates(np.array([0.1, 0.01, 0.0, 0.0]))


def test_stage3_newton_worked_example(nb):
    xs = np.array([1.0, 1.5, 17 / 12, 577 / 408, 665857 / 470832])
    e = np.abs(xs - np.sqrt(2))
    q = np.asarray(nb.order_estimates(e), dtype=float)
    assert np.allclose(q, [2.2575, 1.9839, 1.9998], atol=2e-4), (
        f"Newton's iterates for sqrt(2) from x0 = 1 should give order estimates about "
        f"[2.26, 1.98, 2.00] (lesson, worked example); got {np.round(q, 4).tolist()}")
    C = np.asarray(nb.rate_estimates(e, 2.0), dtype=float)
    assert np.isclose(C[-1], 1 / (2 * np.sqrt(2)), rtol=1e-3), (
        f"the last quadratic rate estimate should approach |f''/(2f')| = 1/(2 sqrt 2) = 0.3536, got {C[-1]:.4f}")


# ---------------------------------------------------------------- stage 4: root finders


def test_stage4_bisection_steps_and_accuracy(nb):
    mids = np.asarray(nb.bisection(lambda x: x * x - 2.0, 1.0, 3.0, tol=1e-3), dtype=float)
    assert mids[0] == 2.0 and mids[1] == 1.5 and mids[2] == 1.25, (
        f"bisection on [1, 3] for x^2 - 2 visits midpoints 2, 1.5, 1.25, ...; got {mids[:3].tolist()}")
    # half-width after recording midpoint j is 2 / 2^j; the first j with 2 / 2^j <= 1e-3 is 11
    assert len(mids) == 11, (
        f"with tol = 1e-3 on an interval of width 2, stop after the 11th midpoint (half-width 2/2^11 "
        f"< 1e-3); got {len(mids)} midpoints")
    root = np.asarray(nb.bisection(_wallis, 2.0, 3.0), dtype=float)[-1]
    assert abs(root - ROOT_WALLIS) < 1e-11, f"bisection should find the root 2.0945514815... of x^3 - 2x - 5, got {root!r}"


def test_stage4_bisection_details(nb):
    with pytest.raises(ValueError):
        nb.bisection(lambda x: x * x + 1.0, -1.0, 1.0)
    mids = np.asarray(nb.bisection(lambda x: x - 1.0, 0.0, 2.0))
    assert mids.tolist() == [1.0], f"f(1) = 0 exactly at the first midpoint: stop there; got {mids.tolist()}"
    for a, b, root in ((1.0, 3.0, 1.0), (-1.0, 1.0, 1.0)):
        got = np.asarray(nb.bisection(_capped(lambda x: x - 1.0), a, b))
        assert got.tolist() == [root], (
            f"on [{a}, {b}] the endpoint {root} is already an exact root of x - 1: return [{root}]; got "
            f"{got[:4].tolist()}{'...' if len(got) > 4 else ''}. (with f(a) == 0, 'f(a) f(m) < 0' is never true, "
            "so plain bisection walks off to the other end.)")
    calls = []

    def f(x):
        calls.append(x)
        return np.cos(x) - x

    mids = np.asarray(nb.bisection(f, 0.0, 1.0, tol=1e-8))
    assert len(calls) <= len(mids) + 2, (
        f"bisection needs ONE new evaluation of f per step (plus f(a), f(b) at the start); you used "
        f"{len(calls)} evaluations for {len(mids)} midpoints. remember f at the endpoint you keep.")


def test_stage4_newton_worked_example(nb):
    xs = np.asarray(nb.newton_1d(lambda x: x * x - 2.0, lambda x: 2.0 * x, 1.0), dtype=float)
    want = [1.0, 1.5, 17 / 12, 577 / 408, 665857 / 470832]
    assert len(xs) >= 5 and np.allclose(xs[:5], want, rtol=1e-14, atol=0), (
        f"Newton on x^2 - 2 from x0 = 1 visits 1, 1.5, 1.41667, 1.4142157, 1.41421356237469; "
        f"got {xs[:5].tolist()}. the array should start with x0.")
    assert abs(xs[-1] ** 2 - 2) <= 1e-12 and abs(xs[-2] ** 2 - 2) > 1e-12, (
        "stop at the FIRST iterate with |f(x)| <= tol, without taking another step")


def test_stage4_newton_stopping(nb):
    xs = np.asarray(nb.newton_1d(lambda x: x - 3.0, lambda x: 1.0, 0.0), dtype=float)
    assert xs.tolist() == [0.0, 3.0], f"for f(x) = x - 3 from 0, one step lands exactly on 3: expected [0, 3], got {xs.tolist()}"
    xs = np.asarray(nb.newton_1d(_capped(lambda x: x * x + 1.0), lambda x: 2.0 * x, 0.5, max_iters=5))
    assert len(xs) == 6, f"x^2 + 1 has no real root: after max_iters = 5 steps there are 6 iterates, got {len(xs)}"


def test_stage4_secant(nb):
    xs = np.asarray(nb.secant(lambda x: x * x - 2.0, 1.0, 2.0), dtype=float)
    assert np.allclose(xs[:4], [1.0, 2.0, 4 / 3, 1.4]), (
        f"secant on x^2 - 2 from x0 = 1, x1 = 2 gives 1, 2, 4/3, 1.4, ...; got {np.round(xs[:4], 6).tolist()}")
    assert abs(xs[-1] - np.sqrt(2)) < 1e-12, f"secant should converge to sqrt(2), ended at {xs[-1]!r}"
    calls = []

    def counted(x):
        calls.append(x)
        return _wallis(x)

    xs = np.asarray(nb.secant(_capped(counted), 3.0, 2.5, tol=1e-15), dtype=float)
    assert len(calls) <= len(xs) + 1, (
        f"the secant method needs ONE new evaluation of f per step (remember f at the previous iterate); you used "
        f"{len(calls)} evaluations for {len(xs)} iterates. that's what makes it faster than Newton per evaluation.")
    try:
        flat = np.asarray(nb.secant(_capped(lambda x: x * x - 1.0), -2.0, 2.0), dtype=float)
    except ZeroDivisionError:
        raise AssertionError("f(-2) == f(2) for x^2 - 1, so the secant slope is 0/0: stop (return [-2, 2]) "
                             "instead of dividing by zero") from None
    assert flat.tolist() == [-2.0, 2.0], f"when f(x_k) == f(x_(k-1)) the secant is undefined: stop; got {flat.tolist()}"
    e = np.abs(xs - ROOT_WALLIS)
    e = e[e > 1e-15]
    q = _orders(e)
    assert len(q) >= 3 and 1.5 < np.median(q[-3:]) < 1.75, (
        f"the secant method's order of convergence is the golden ratio 1.618; your iterates give "
        f"order estimates {np.round(q, 3).tolist()}")
