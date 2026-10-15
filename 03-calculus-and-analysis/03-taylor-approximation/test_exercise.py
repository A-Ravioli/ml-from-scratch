"""Checks for exercise.py (taylor approximation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Most expected
values are exact (polynomials, closed-form remainders); the multivariate models are checked
against hand-computed gradients and Hessians.
"""
import math

import numpy as np
import pytest


# ---------------------------------------------------------------- stage 1: 1-D Taylor polynomials


def test_stage1_worked_example(nb):
    got = nb.taylor_poly_1d([1.0, 1.0, 1.0], 0.0, 0.5)
    assert np.isclose(got, 1.625, rtol=1e-15), (
        f"the degree-2 Taylor polynomial of e^x at 0 is 1 + x + x^2/2, which is 1.625 at x = 0.5; got {got}"
        + (". 1.75 means you forgot to divide by k!." if np.isclose(got, 1.75) else ""))
    assert type(got) is float, (
        f"for a float x, return a Python float (got {type(got).__name__}); convert a 0-d result with float(...)")


def test_stage1_sine_and_vectorized(nb):
    derivs = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0]  # sin at 0, up to degree 5
    x = np.array([[0.1, 0.5], [1.0, -2.0]])
    got = np.asarray(nb.taylor_poly_1d(derivs, 0.0, x), dtype=float)
    want = x - x**3 / 6 + x**5 / 120
    assert got.shape == (2, 2), f"x has shape (2, 2), so the result must too; got {got.shape}"
    assert np.allclose(got, want, rtol=1e-14), f"x - x^3/6 + x^5/120 at {x.tolist()} is {np.round(want, 8).tolist()}, got {np.round(got, 8).tolist()}"


def test_stage1_center_matters(nb):
    # x^3 around x0 = 1: f = 1, f' = 3, f'' = 6, f''' = 6. the cubic Taylor polynomial IS x^3.
    got = nb.taylor_poly_1d([1.0, 3.0, 6.0, 6.0], 1.0, 2.5)
    assert np.isclose(got, 2.5**3, rtol=1e-14), (
        f"x^3 expanded around x0 = 1 to degree 3 reproduces x^3 exactly: 15.625 at x = 2.5; got {got}. "
        "use powers of (x - x0), not x.")
    got = nb.taylor_poly_1d([7.0], 3.0, np.array([0.0, 10.0]))
    assert np.allclose(got, [7.0, 7.0]), "a degree-0 Taylor polynomial is the constant f(x0)"


# ---------------------------------------------------------------- stage 2: remainders and orders


def test_stage2_lagrange_worked_example(nb):
    got = nb.lagrange_bound(np.exp(0.5), 0.0, 0.5, 2)
    want = np.exp(0.5) * 0.5**3 / 6
    assert np.isclose(got, want, rtol=1e-14), (
        f"for e^x, degree 2, on [0, 0.5]: M = e^0.5 and the bound is M |x|^3 / 3! = {want:.6f}; got {got}"
        + (". you used |x|^2 / 2!: the remainder of a degree-k polynomial involves the (k+1)-th derivative."
           if np.isclose(got, np.exp(0.5) * 0.25 / 2) else ""))
    actual = np.exp(0.5) - 1.625
    assert actual <= got, "the true error 0.0237 must be below the bound"


def test_stage2_lagrange_vectorized(nb):
    x = np.array([-1.0, 0.0, 2.0])
    got = np.asarray(nb.lagrange_bound(1.0, 0.5, x, 3), dtype=float)
    want = np.abs(x - 0.5) ** 4 / 24
    assert got.shape == (3,) and np.allclose(got, want), f"expected {want.tolist()}, got {got.tolist()}"
    left = nb.lagrange_bound(2.0, 0.5, -1.0, 2)  # x < x0 and an odd power: the distance needs an absolute value
    assert np.isclose(left, 2.0 * 1.5**3 / 6), (
        f"for x = -1 < x0 = 0.5, order 2: the bound is M |x - x0|^3 / 3! = {2.0 * 1.5**3 / 6}, a POSITIVE number; got {left}. "
        "use the absolute value |x - x0|.")


def test_stage2_bound_holds_for_sine(nb):
    x = np.linspace(-3, 3, 61)
    err = np.abs(np.sin(x) - (x - x**3 / 6))  # the degree-3 Taylor polynomial of sin at 0
    bound = np.asarray(nb.lagrange_bound(1.0, 0.0, x, 3), dtype=float)
    assert np.all(err <= bound + 1e-15), "with |sin^(4)| <= 1, the degree-3 bound |x|^4/24 must cover the true error"
    assert np.isclose(bound[-1], 81 / 24), f"at x = 3 the bound is 3^4 / 4! = 3.375, got {bound[-1]}"


def test_stage2_loglog_slope(nb):
    hs = np.logspace(-3, -1, 9)
    got = nb.loglog_slope(hs, 3.0 * hs**2.5)
    assert np.isclose(got, 2.5, rtol=1e-12), f"errors = 3 h^2.5 have log-log slope exactly 2.5; got {got}"
    wobble = 1 + 0.3 * np.sin(np.arange(9) * 1.7)  # deterministic noise
    errs = hs**2 * wobble
    want = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    got = nb.loglog_slope(hs, errs)
    endpoints = np.log(errs[-1] / errs[0]) / np.log(hs[-1] / hs[0])
    assert np.isclose(got, want, rtol=1e-10), (
        f"the least-squares slope through all 9 points is {want:.6f}, got {got:.6f}"
        + (" (that's the slope through the two endpoints only: fit all the points)" if np.isclose(got, endpoints) else ""))


# ---------------------------------------------------------------- stage 3: quadratic models


def test_stage3_values_and_center(nb):
    x0 = np.array([1.0, -2.0])
    g = np.array([3.0, 0.5])
    H = np.array([[2.0, 1.0], [1.0, 4.0]])
    m = nb.quadratic_model(x0, 5.0, g, H)
    assert callable(m), "quadratic_model returns a function m(x)"
    assert np.isclose(m(x0), 5.0), "at the center, the model equals f0"
    x = np.array([1.5, -1.0])
    d = x - x0
    want = 5.0 + g @ d + 0.5 * d @ H @ d
    got = m(x)
    assert np.isclose(got, want, rtol=1e-14), (
        f"m(x) = f0 + g.d + 1/2 d^T H d with d = x - x0 = {d.tolist()} gives {want}; got {got}"
        + (" (you dropped the 1/2)" if np.isclose(got, 5.0 + g @ d + d @ H @ d) else "")
        + (" (use d = x - x0, not x)" if np.isclose(got, 5.0 + g @ x + 0.5 * x @ H @ x) else ""))
    assert isinstance(got, float), "for a single point (n,), return a float"


def test_stage3_batch(nb):
    rng = np.random.default_rng(0)
    x0, g = rng.normal(size=3), rng.normal(size=3)
    B = rng.normal(size=(3, 3))
    H = B + B.T
    m = nb.quadratic_model(x0, -1.0, g, H)
    X = rng.normal(size=(7, 3))
    got = np.asarray(m(X), dtype=float)
    want = np.array([-1.0 + g @ (x - x0) + 0.5 * (x - x0) @ H @ (x - x0) for x in X])
    assert got.shape == (7,), f"a batch of 7 points (7, 3) gives 7 values (7,); got shape {got.shape}"
    assert np.allclose(got, want, rtol=1e-13), "batch values differ from evaluating the points one at a time"


def test_stage3_linear_model(nb):
    m = nb.quadratic_model(np.zeros(2), 1.0, np.array([2.0, -1.0]), np.zeros((2, 2)))
    assert np.isclose(m(np.array([3.0, 4.0])), 1.0 + 6.0 - 4.0), "with H = 0 the model is the tangent plane f0 + g.(x - x0)"


# ---------------------------------------------------------------- stage 4: Taylor models of any f


def test_stage4_worked_example(nb):
    f = lambda v: np.exp(v[0]) * np.sin(v[1])
    x0 = np.array([0.0, np.pi / 2])
    m = nb.taylor_model(f, x0)
    x = np.array([0.1, np.pi / 2 + 0.1])
    assert np.isclose(m(x), 1.1, atol=1e-7), (
        f"e^x sin y at (0, pi/2) has f = 1, grad = (1, 0), H = [[1, 0], [0, -1]], so the quadratic model at "
        f"(0.1, pi/2 + 0.1) is 1 + 0.1 + (0.01 - 0.01)/2 = 1.1; got {m(x)}")
    m1 = nb.taylor_model(f, x0, order=1)
    assert np.isclose(m1(x), 1.1, atol=1e-7) and np.isclose(m1(np.array([0.2, np.pi / 2])), 1.2, atol=1e-7), (
        "the order-1 model is the tangent plane 1 + x")
    m0 = nb.taylor_model(f, x0, order=0)
    assert np.isclose(m0(x), 1.0), "the order-0 model is the constant f(x0)"


def test_stage4_exact_for_quadratics_and_batches(nb):
    A = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, -0.5], [0.0, -0.5, 1.0]])
    b = np.array([1.0, -2.0, 0.5])
    f = lambda v: 0.5 * v @ A @ v + b @ v + 4.0
    m = nb.taylor_model(f, np.array([0.3, -0.2, 1.0]))
    X = np.random.default_rng(1).normal(size=(5, 3)) * 3
    got = np.asarray(m(X), dtype=float)
    want = np.array([f(x) for x in X])
    assert got.shape == (5,) and np.allclose(got, want, rtol=1e-6, atol=1e-6), (
        "the second-order model of a quadratic IS the quadratic, everywhere (up to finite-difference error); "
        f"max difference {np.abs(got - want).max():.2e}")
    with pytest.raises(ValueError):
        nb.taylor_model(f, np.zeros(3), order=3)


def test_stage4_error_shrinks_like_r_cubed(nb):
    f = lambda v: np.log(1 + np.exp(v[0] - v[1])) + np.sin(v[0] * v[1]) + v[1] ** 4
    x0 = np.array([0.4, 0.3])
    u = np.array([0.6, 0.8])
    rs = np.array([0.08, 0.04, 0.02, 0.01])
    for order, want in ((1, 2.0), (2, 3.0)):
        m = nb.taylor_model(f, x0, order=order)
        errs = np.array([abs(f(x0 + r * u) - m(x0 + r * u)) for r in rs])
        slope = np.polyfit(np.log(rs), np.log(errs), 1)[0]
        assert abs(slope - want) < 0.2, (
            f"the order-{order} model's error should shrink like r^{order + 1} along a ray; measured slope {slope:.2f}")


# ---------------------------------------------------------------- stage 5: trusting the model


def _cubic_case():
    f = lambda v: v[0] ** 2 + v[0] ** 3 + v[1] ** 2
    model = lambda v: v[0] ** 2 + v[1] ** 2  # its exact quadratic model at 0
    return f, model


def test_stage5_agreement_ratio(nb):
    f, model = _cubic_case()
    for r in (0.1, 0.5, 2.0):
        got = nb.agreement_ratio(f, model, np.zeros(2), np.array([r, 0.0]))
        assert np.isclose(got, 1 + r, rtol=1e-12), (
            f"for f = x^2 + x^3 + y^2 and its model x^2 + y^2 at 0, the ratio at (r, 0) is "
            f"(0 - (r^2 + r^3)) / (0 - r^2) = 1 + r = {1 + r}; got {got}"
            + (" (you divided predicted by actual)" if np.isclose(got, 1 / (1 + r)) else ""))


def test_stage5_model_offset(nb):
    f = lambda v: v @ v
    model = lambda v: 5.0 + 2.0 * (v @ v)  # a model that doesn't pass through f(x0): only DIFFERENCES matter
    got = nb.agreement_ratio(f, model, np.zeros(2), np.array([1.0, 0.0]))
    assert np.isclose(got, 0.5), (
        f"actual reduction f(x0) - f(x) = -1, predicted reduction model(x0) - model(x) = -2: the ratio is 0.5; got {got}. "
        "both reductions are measured on their own function (use model(x0), not f(x0), in the denominator).")


def test_stage5_trust_radius_hand_example(nb):
    f, model = _cubic_case()
    radii = np.array([0.1, 0.2, 0.26, 0.4])
    got = nb.trust_radius(f, model, np.zeros(2), np.array([3.0, 0.0]), radii)
    assert np.isclose(got, 0.2), (
        f"along the x-axis the ratio is 1 + r, so |ratio - 1| <= 0.25 holds for r = 0.1 and 0.2 but not 0.26: "
        f"the answer is 0.2; got {got}. (the direction (3, 0) must be normalized to (1, 0) first.)")
    got = nb.trust_radius(f, model, np.zeros(2), np.array([1.0, 0.0]), np.array([0.125, 0.25, 0.5]))
    assert np.isclose(got, 0.25), (
        f"at r = 0.25 the ratio is exactly 1.25, so |ratio - 1| = 0.25 <= tol = 0.25 still counts as trustworthy "
        f"(the test is <=, failure is > tol): the answer is 0.25; got {got}")
    got = nb.trust_radius(f, model, np.zeros(2), np.array([-1.0, 0.0]), np.array([0.1, 0.3, 0.37]), tol=0.35)
    assert np.isclose(got, 0.3), f"along -x the ratio is 1 - r; with tol = 0.35, r = 0.37 fails and the answer is 0.3, got {got}"


def test_stage5_stops_at_first_failure(nb):
    def rho(r):
        return 2.0 if 0.25 <= r < 0.45 else 1.0

    model = lambda v: v @ v
    f = lambda v: (v @ v) * rho(np.linalg.norm(v))  # ratio is exactly rho(|x|)
    radii = np.array([0.1, 0.2, 0.3, 0.5, 0.6])
    got = nb.trust_radius(f, model, np.zeros(2), np.array([1.0, 1.0]), radii)
    assert np.isclose(got, 0.2), (
        f"the ratio fails at r = 0.3 and passes again at 0.5: the trust radius stops at the FIRST failure, "
        f"so it's 0.2; got {got}")
    got = nb.trust_radius(f, model, np.zeros(2), np.array([1.0, 0.0]), np.array([0.3, 0.6]))
    assert got == 0.0, f"if the first radius already fails, return 0.0; got {got}"
