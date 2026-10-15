"""Checks for exercise.py (integrals).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Exact integrals
come from antiderivatives (the fundamental theorem), so the checks don't depend on any of your
code being right.
"""
import numpy as np
import pytest

SQ = lambda x: x**2


# ---------------------------------------------------------------- stage 1: Riemann sums


def test_stage1_worked_example(nb):
    want = {"left": 14 / 64, "right": 30 / 64, "midpoint": 84 / 256}
    for rule, w in want.items():
        got = nb.riemann_sum(SQ, 0.0, 1.0, 4, rule=rule)
        assert np.isclose(got, w, rtol=1e-12), (
            f"riemann_sum(x^2, 0, 1, n=4, rule={rule!r}) should be {w} (see the lesson's worked example), got {got}. "
            "with h = 1/4 the left ends are 0, 1/4, 2/4, 3/4; the right ends 1/4, ..., 1; the midpoints 1/8, 3/8, 5/8, 7/8.")


def test_stage1_other_interval(nb):
    # on [1, 3] the width is h = 2/n, not 1/n
    got = nb.riemann_sum(SQ, 1.0, 3.0, 2, rule="midpoint")
    want = 1.0 * (1.5**2 + 2.5**2)  # h = 1, midpoints 1.5 and 2.5
    assert np.isclose(got, want), (
        f"riemann_sum(x^2, 1, 3, n=2, rule='midpoint'): h = (3 - 1) / 2 = 1, midpoints 1.5 and 2.5, area "
        f"{want}; got {got}. use h = (b - a) / n and start the points at a.")
    exact = (3**3 - 1) / 3
    got = nb.riemann_sum(SQ, 1.0, 3.0, 1000, rule="left")
    assert abs(got - exact) < 0.01, f"with 1000 strips the left sum on [1, 3] should be within 0.01 of {exact:.4f}, got {got:.4f}"


def test_stage1_converges_and_brackets(nb):
    exact = np.e - 1  # area under e^x on [0, 1]
    left = nb.riemann_sum(np.exp, 0.0, 1.0, 50, rule="left")
    right = nb.riemann_sum(np.exp, 0.0, 1.0, 50, rule="right")
    mid = nb.riemann_sum(np.exp, 0.0, 1.0, 50, rule="midpoint")
    assert left < exact < right, (
        f"for an increasing function, left rectangles sit under the curve and right rectangles over it: "
        f"expected left < {exact:.6f} < right, got left = {left:.6f}, right = {right:.6f}")
    assert abs(mid - exact) < abs(left - exact) / 20, (
        f"the midpoint rule should be far more accurate than the left rule: errors {abs(mid - exact):.1e} vs {abs(left - exact):.1e}")


def test_stage1_rejects_unknown_rule(nb):
    with pytest.raises(ValueError):
        nb.riemann_sum(SQ, 0.0, 1.0, 4, rule="top")


# ---------------------------------------------------------------- stage 2: trapezoid rule


def test_stage2_worked_example(nb):
    got = nb.trapezoid(SQ, 0.0, 1.0, 4)
    assert isinstance(got, float), f"trapezoid should return a float, got {type(got).__name__}"
    assert np.isclose(got, 0.34375, rtol=1e-12), (
        f"trapezoid(x^2, 0, 1, n=4) = (1/4)(0/2 + 1/16 + 4/16 + 9/16 + 1/2) = 0.34375, got {got}. "
        + ("0.46875 means the end points weren't halved." if np.isclose(got, 0.46875) else
           "check that you use n + 1 points (np.linspace(a, b, n + 1)) and halve the two ends."))


def test_stage2_exact_for_straight_lines(nb):
    got = nb.trapezoid(lambda x: 3 * x - 1, -2.0, 5.0, 7)
    want = 1.5 * (25 - 4) - 7  # antiderivative 1.5 x^2 - x
    assert np.isclose(got, want, rtol=1e-12), (
        f"the trapezoid rule is exact for straight lines: the area under 3x - 1 on [-2, 5] is {want}, got {got}")


def test_stage2_default_n_and_error_rate(nb):
    exact = 2.0  # area under sin on [0, pi]
    got = nb.trapezoid(np.sin, 0.0, np.pi)
    assert abs(got - exact) < 2e-4 and abs(got - exact) > 1e-5, (
        f"with the default n = 100, the trapezoid rule for sin on [0, pi] should have an error near 1.6e-4, "
        f"got {abs(got - exact):.1e}. is your default n = 100?")
    e1 = abs(nb.trapezoid(np.sin, 0.0, np.pi, 20) - exact)
    e2 = abs(nb.trapezoid(np.sin, 0.0, np.pi, 40) - exact)
    assert 3.8 < e1 / e2 < 4.2, (
        f"doubling n should cut the trapezoid error by about 4 (error ∝ h^2); got a ratio of {e1 / e2:.2f}")


def test_stage2_reversed_interval(nb):
    a, b = 0.5, 2.0
    fwd = nb.trapezoid(np.exp, a, b, 30)
    rev = nb.trapezoid(np.exp, b, a, 30)
    assert np.isclose(rev, -fwd, rtol=1e-12), (
        f"integrating from b to a gives minus the area from a to b: expected {-fwd}, got {rev}. "
        "use h = (b - a) / n without abs(): the sign takes care of itself.")


# ---------------------------------------------------------------- stage 3: Simpson's rule


def test_stage3_worked_example(nb):
    got = nb.simpson(lambda x: x**4, 0.0, 1.0, 2)
    assert np.isclose(got, 1.25 / 6, rtol=1e-12), (
        f"simpson(x^4, 0, 1, n=2) = (0.5 / 3)(0 + 4 * 0.5^4 + 1) = 0.208333, got {got}. "
        + ("check h / 3 (not h / 6)." if np.isclose(got, 1.25 / 12) else ""))


def test_stage3_weights(nb):
    # an f that is 1 at exactly one grid point and 0 at all others reveals that point's weight
    n, h = 6, 1.0
    for k, w in enumerate([1, 4, 2, 4, 2, 4, 1]):
        f = lambda x, k=k: (np.abs(x - k) < 1e-9).astype(float)
        got = nb.simpson(f, 0.0, 6.0, n) * 3 / h
        assert np.isclose(got, w), (
            f"with n = 6, grid point {k} should have Simpson weight {w} (pattern 1, 4, 2, 4, 2, 4, 1), "
            f"got {got:g}. odd-numbered points get 4, inner even-numbered points 2, the ends 1.")


def test_stage3_exact_for_cubics(nb):
    f = lambda x: 2 * x**3 - x**2 + 5
    F = lambda x: x**4 / 2 - x**3 / 3 + 5 * x
    got = nb.simpson(f, -1.0, 2.0, 2)
    assert np.isclose(got, F(2.0) - F(-1.0), rtol=1e-12), (
        f"Simpson's rule is exact for cubics, even with n = 2: expected {F(2.0) - F(-1.0)}, got {got}")


def test_stage3_error_rate_and_default(nb):
    exact = np.e - 1
    e1 = abs(nb.simpson(np.exp, 0.0, 1.0, 8) - exact)
    e2 = abs(nb.simpson(np.exp, 0.0, 1.0, 16) - exact)
    assert 14 < e1 / e2 < 18, f"doubling n should cut Simpson's error by about 16 (∝ h^4); got a ratio of {e1 / e2:.2f}"
    assert abs(nb.simpson(np.exp, 0.0, 1.0) - exact) < 1e-10, "with the default n = 100 the error for e^x on [0, 1] is ~1e-11"


def test_stage3_rejects_odd_n(nb):
    with pytest.raises(ValueError):
        nb.simpson(SQ, 0.0, 1.0, 5)


# ---------------------------------------------------------------- stage 4: the running area


def test_stage4_worked_example(nb):
    xs, F = (np.asarray(v) for v in nb.cumulative_trapezoid(lambda x: 2 * x, 0.0, 1.0, 2))
    assert xs.shape == (3,) and F.shape == (3,), (
        f"with n = 2 strips there are 3 grid points, so xs and F have shape (3,); got {xs.shape} and {F.shape}")
    assert np.allclose(xs, [0.0, 0.5, 1.0]), f"xs should be np.linspace(0, 1, 3) = [0, 0.5, 1], got {xs}"
    assert np.allclose(F, [0.0, 0.25, 1.0]), (
        f"F for f = 2x should be [0, 0.25, 1.0] (no area yet, then the area up to 0.5, then up to 1), got {F}")


def test_stage4_matches_trapezoid_and_antiderivative(nb):
    xs, F = (np.asarray(v) for v in nb.cumulative_trapezoid(np.cos, 0.0, np.pi, 200))
    assert F[0] == 0.0, f"the running area starts at F[0] = 0, got {F[0]}"
    full = 0.5 * (np.pi / 200) * (np.cos(xs[:-1]) + np.cos(xs[1:])).sum()
    assert np.isclose(F[-1], full, rtol=1e-12, atol=1e-14), (
        f"F[-1] (all the area) should equal the trapezoid rule with the same n: {full}, got {F[-1]}")
    err = np.abs(F - np.sin(xs)).max()
    assert err < 1e-4, (
        f"the running area of cos from 0 is sin(x) (fundamental theorem); your F differs from sin by up to {err:.1e}")


def test_stage4_slope_of_running_area_is_f(nb):
    f = lambda x: np.exp(-x) * x
    xs, F = (np.asarray(v) for v in nb.cumulative_trapezoid(f, 0.2, 3.0, 400))
    slope = np.diff(F) / np.diff(xs)
    mid_f = 0.5 * (f(xs[:-1]) + f(xs[1:]))
    assert np.allclose(slope, mid_f, rtol=1e-10), (
        "the slope of the running area across each strip should be the average height of f there: this is "
        "the fundamental theorem of calculus, (area so far)' = f. check that F[k] - F[k-1] is strip k's area.")


# ---------------------------------------------------------------- stage 5: convergence order


def test_stage5_worked_example(nb):
    got = nb.convergence_order([10, 20, 40], [1e-2, 2.5e-3, 6.25e-4])
    assert isinstance(got, float), f"convergence_order should return a float, got {type(got).__name__}"
    assert np.isclose(got, 2.0, rtol=1e-10), (
        f"each doubling of n divides the error by 4 = 2^2, so the order is 2.0, got {got}. "
        + ("-2 means you returned the slope itself; the order is MINUS the slope." if np.isclose(got, -2.0) else ""))


def test_stage5_least_squares_through_all_points(nb):
    ns = np.array([4, 8, 16, 32, 64])
    errors = np.array([3e-2, 4e-3, 9e-4, 4e-5, 5e-6])
    u, v = np.log(ns), np.log(errors)
    want = -np.polyfit(u, v, 1)[0]
    got = nb.convergence_order(ns, errors)
    assert np.isclose(got, want, rtol=1e-10), (
        f"for noisy errors the order is minus the least-squares slope through ALL points: {want:.4f}, got {got:.4f}. "
        "using only the first and last points (or the first two) gives a different number.")


def test_stage5_measures_real_rules(nb):
    ns = np.array([8, 16, 32, 64])
    exact = np.e - 1
    trap = [abs(0.5 * (1 / n) * (np.exp(np.linspace(0, 1, n + 1))[:-1] + np.exp(np.linspace(0, 1, n + 1))[1:]).sum() - exact) for n in ns]
    p = nb.convergence_order(ns, trap)
    assert abs(p - 2) < 0.05, f"the trapezoid rule's measured order on e^x should be 2.00, got {p:.3f}"
