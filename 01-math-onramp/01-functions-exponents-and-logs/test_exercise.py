"""Checks for exercise.py (functions, exponents & logs).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. NumPy's own
np.exp and np.log are the oracles: your versions should agree with them to many digits.
"""
import numpy as np
import pytest


def _capped(f, limit=10_000):
    """wrap f so that a bisection that never stops fails with a message instead of hanging."""
    calls = []

    def g(x):
        calls.append(x)
        if len(calls) > limit:
            raise AssertionError(
                f"bisect_inverse didn't stop after {limit:,} evaluations of f. a float can't be halved "
                "forever: cap the loop at 200 halvings (for _ in range(200): ...).")
        return f(x)

    g.calls = calls
    return g


def _rel_err(got, want):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    return float(np.max(np.abs(got - want) / np.abs(want)))


# ---------------------------------------------------------------- stage 1: composition and inverses


def test_stage1_compose_order(nb):
    add_one = lambda x: x + 1
    double = lambda x: 2 * x
    h = nb.compose(double, add_one)
    assert callable(h), "compose(f, g) should return a FUNCTION (define an inner function and return it)"
    got = h(5)
    assert got == 12, (
        f"compose(double, add_one)(5) should apply add_one FIRST, then double: 2 * (5 + 1) = 12; got {got}. "
        + ("11 means you applied them in the wrong order: f(g(x)) runs g first." if got == 11 else ""))
    got2 = nb.compose(np.sqrt, add_one)(8.0)
    assert np.isclose(got2, 3.0), f"compose(np.sqrt, add_one)(8.0) = sqrt(9) = 3, got {got2}"


def test_stage1_compose_works_on_arrays(nb):
    h = nb.compose(np.sin, lambda x: x**2)
    x = np.array([0.0, 0.5, 1.0, 2.0])
    got = np.asarray(h(x))
    assert np.allclose(got, np.sin(x**2)), f"compose(np.sin, square)(x) should be sin(x^2) = {np.sin(x**2)}, got {got}"


def test_stage1_bisect_inverse_cube_root(nb):
    got = nb.bisect_inverse(_capped(lambda x: x**3), 27.0, 0.0, 10.0)
    assert isinstance(got, float), f"bisect_inverse should return a float, got {type(got).__name__}"
    assert abs(got - 3.0) < 1e-11, (
        f"x^3 = 27 has the solution x = 3, got {got!r}. if your answer is far off, check which end of "
        "the bracket you move: when f(mid) < y the answer is to the right, so lo = mid.")


def test_stage1_bisect_inverse_undoes_exp(nb):
    for y in (0.5, 1.0, 10.0, 100.0):
        got = nb.bisect_inverse(_capped(np.exp), y, -5.0, 5.0)
        assert abs(got - np.log(y)) < 1e-11, (
            f"bisect_inverse(np.exp, {y}, -5, 5) should find log({y}) = {np.log(y):.12f}, got {got:.12f}")


def test_stage1_bisect_inverse_respects_tol(nb):
    f = _capped(lambda x: x)
    calls = f.calls
    got = nb.bisect_inverse(f, 0.3, 0.0, 1.0, tol=1e-3)
    assert abs(got - 0.3) <= 1e-3, f"with tol = 1e-3 the answer must be within 1e-3 of 0.3, got {got}"
    assert len(calls) < 40, (
        f"with tol = 1e-3 on a bracket of width 1, about 10 halvings are enough, but f was called {len(calls)} "
        "times. stop once hi - lo <= tol.")
    got = nb.bisect_inverse(_capped(lambda x: x), 1e6 + 0.25, 0.0, 2e6)  # a float can't shrink this bracket to 1e-12
    assert abs(got - (1e6 + 0.25)) < 1e-6, (
        f"bisect_inverse(identity, 1e6 + 0.25, 0, 2e6) should give 1000000.25, got {got}. (and it must "
        "stop: cap the loop at 200 halvings.)")


def test_stage1_bisect_inverse_rejects_bad_bracket(nb):
    with pytest.raises(ValueError):
        nb.bisect_inverse(_capped(lambda x: x**2), 50.0, 0.0, 5.0)  # 50 > f(5) = 25: no solution in [0, 5]
    with pytest.raises(ValueError):
        nb.bisect_inverse(_capped(lambda x: x**2), -1.0, 0.0, 5.0)  # -1 < f(0) = 0


# ---------------------------------------------------------------- stage 2: the exponential series


def test_stage2_worked_example(nb):
    got = float(nb.exp_series(1.0, 4))
    assert np.isclose(got, 8 / 3, rtol=1e-14), (
        f"exp_series(1.0, 4) = 1 + 1 + 1/2 + 1/6 = 2.6667 (FOUR terms, k = 0..3), got {got}. "
        + ("2.7083 or 2.5 means you added one term too many or too few." if np.isclose(got, 65 / 24) or np.isclose(got, 2.5) else ""))


def test_stage2_uses_factorials(nb):
    got = float(nb.exp_series(2.0, 4))
    want = 1 + 2 + 4 / 2 + 8 / 6
    assert np.isclose(got, want, rtol=1e-14), (
        f"exp_series(2.0, 4) = 1 + 2 + 2^2/2! + 2^3/3! = {want:.6f}, got {got:.6f}. "
        + ("7.6667 means term k is x^k / k instead of x^k / k!: update term = term * x / k so the "
           "divisions pile up into a factorial." if np.isclose(got, 1 + 2 + 2 + 8 / 3) else ""))
    assert np.isclose(float(nb.exp_series(-3.0, 1)), 1.0), "with ONE term the sum is just the k = 0 term, 1"


def test_stage2_arrays_and_convergence(nb):
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    got = np.asarray(nb.exp_series(x, 30))
    assert got.shape == x.shape, f"exp_series of a {x.shape} array should have shape {x.shape}, got {got.shape}"
    assert np.allclose(got, np.exp(x), rtol=1e-14), (
        f"30 terms of the series should match np.exp to ~15 digits for small x: expected {np.exp(x)}, got {got}")
    g5 = np.asarray(nb.exp_series(np.array([0.1, 0.2]), 3))
    assert np.allclose(g5, 1 + np.array([0.1, 0.2]) + np.array([0.1, 0.2]) ** 2 / 2, rtol=1e-14), (
        "with 3 terms, every entry should be 1 + x + x^2/2: work entry by entry, with array arithmetic")


# ---------------------------------------------------------------- stage 3: exp with range reduction


def test_stage3_small_x_is_accurate(nb):
    x = np.linspace(-1, 1, 41)
    err = _rel_err(nb.exp(x), np.exp(x))
    assert err < 1e-14, f"for |x| <= 1, exp should agree with np.exp to ~1e-15 relative error, got {err:.1e}"


def test_stage3_large_positive_x(nb):
    x = np.array([10.0, 37.3, 50.0, 100.5, 300.0, 700.0])
    err = _rel_err(nb.exp(x), np.exp(x))
    assert err < 1e-12, (
        f"exp(x) for x up to 700 should have relative error < 1e-12, got {err:.1e}. 20 terms of the series "
        "are nowhere near enough for x = 50: split x = k + r with k = np.round(x), and use e^k * e^r.")


def test_stage3_large_negative_x(nb):
    x = np.array([-5.0, -20.0, -30.0, -99.7, -700.0])
    err = _rel_err(nb.exp(x), np.exp(x))
    assert err < 1e-12, (
        f"exp(x) for negative x should have relative error < 1e-12, got {err:.1e}. the raw series for x = -30 "
        "adds huge terms of alternating sign that cancel, and rounding destroys the answer. range reduction "
        "keeps |r| <= 1/2, where every term is tiny.")


def test_stage3_exponent_rules_hold(nb):
    a, b = 3.7, -1.2
    got = float(nb.exp(a + b))
    want = float(nb.exp(a)) * float(nb.exp(b))
    assert np.isclose(got, want, rtol=1e-13), f"e^(a+b) should equal e^a * e^b: {got} vs {want}"
    assert np.isclose(float(nb.exp(0.0)), 1.0, rtol=1e-15), "e^0 = 1"
    assert np.isclose(float(nb.exp(1.0)), np.e, rtol=1e-15), f"e^1 = e = {np.e}, got {float(nb.exp(1.0))}"
    x = np.array([[0.3, -2.5], [12.0, 4.5]])
    got = np.asarray(nb.exp(x))
    assert got.shape == (2, 2), f"exp of a (2, 2) array should be (2, 2), got {got.shape}"


# ---------------------------------------------------------------- stage 4: Newton's method


def test_stage4_worked_example(nb):
    xs = np.asarray(nb.newton_log_steps(2.0, 1.0, 2), dtype=float)
    assert xs.shape == (3,), (
        f"newton_log_steps(2.0, 1.0, 2) returns [x0, x1, x2], shape (3,), got shape {xs.shape}")
    assert xs[0] == 1.0, f"the first entry is the starting guess x0 = 1.0, got {xs[0]}"
    want1 = 2 / np.e
    assert np.isclose(xs[1], want1, rtol=1e-12), (
        f"x1 = x0 - 1 + y e^(-x0) = 1 - 1 + 2/e = {want1:.10f}, got {xs[1]:.10f}. check the update "
        "x_new = x - 1 + y * exp(-x).")
    want2 = want1 - 1 + 2 * np.exp(-want1)
    assert np.isclose(xs[2], want2, rtol=1e-12), f"x2 should be {want2:.10f}, got {xs[2]:.10f}"


def test_stage4_converges_quadratically(nb):
    xs = np.asarray(nb.newton_log_steps(10.0, 2.0, 6), dtype=float)
    errs = np.abs(xs - np.log(10.0))
    assert errs[-1] < 1e-14, f"6 Newton steps from x0 = 2 should reach log(10) to ~1e-15, final error {errs[-1]:.1e}"
    # quadratic convergence: each error is about (previous error)^2 / 2
    for e_prev, e_next in zip(errs[:3], errs[1:4]):
        assert e_next <= 0.6 * e_prev**2 + 1e-15, (
            f"Newton's errors should roughly square at every step (digits double), but went "
            f"{e_prev:.2e} -> {e_next:.2e}. errors: {np.array2string(errs, precision=2)}")


def test_stage4_zero_steps_and_far_start(nb):
    xs = np.asarray(nb.newton_log_steps(3.0, 0.5, 0), dtype=float)
    assert xs.shape == (1,) and xs[0] == 0.5, f"with n_steps = 0 the result is just [x0] = [0.5], got {xs}"
    xs = np.asarray(nb.newton_log_steps(5.0, 6.0, 12), dtype=float)
    assert abs(xs[-1] - np.log(5.0)) < 1e-12, (
        f"starting too HIGH (x0 = 6 for log 5 = 1.609), Newton creeps down by about 1 per step and then "
        f"converges fast; after 12 steps expected {np.log(5.0):.12f}, got {xs[-1]:.12f}")


# ---------------------------------------------------------------- stage 5: log


def test_stage5_matches_numpy_over_a_huge_range(nb):
    y = np.logspace(-300, 300, 241)
    got = np.asarray(nb.log(y))
    assert got.shape == y.shape, f"log of a {y.shape} array should have shape {y.shape}, got {got.shape}"
    err = np.max(np.abs(got - np.log(y)) / np.maximum(1.0, np.abs(np.log(y))))
    assert err < 1e-13, (
        f"log(y) for y from 1e-300 to 1e300 should match np.log to ~1e-15 (relative), worst error {err:.1e}. "
        "Newton from x = m - 1 only works because frexp put m in [0.5, 1): check that you used m (not y) "
        "inside the Newton loop, and added p * log(2).")


def test_stage5_near_one_and_identities(nb):
    y = np.array([0.5, 0.75, 0.999, 1.0, 1.001, 1.5, 2.0])
    got = np.asarray(nb.log(y))
    assert np.allclose(got, np.log(y), rtol=0, atol=1e-14), (
        f"log near 1 should match np.log to ~1e-15 absolute: expected {np.log(y)}, got {got}")
    a, b = 123.4, 0.0567
    assert np.isclose(float(nb.log(a * b)), float(nb.log(a)) + float(nb.log(b)), rtol=1e-13), (
        "log(ab) should equal log(a) + log(b)")
    assert np.isclose(float(nb.log(2.0**40)), 40 * np.log(2.0), rtol=1e-14), "log(2^40) = 40 log 2"


def test_stage5_undoes_exp(nb):
    x = np.array([-50.0, -3.3, 0.1, 7.0, 99.0])
    got = np.asarray(nb.log(nb.exp(x)))
    assert np.allclose(got, x, rtol=1e-12, atol=1e-14), f"log(exp(x)) should give x back: {x} -> {got}"


def test_stage5_rejects_non_positive(nb):
    for bad in (0.0, -1.0, np.array([1.0, -2.0])):
        with pytest.raises(ValueError):
            nb.log(bad)
