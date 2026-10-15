"""Checks for exercise.py (derivatives).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The rule tests
compare your derivative functions against this file's own finite differences (not yours), so
a bug in your `derivative` can't hide a bug in a rule.
"""
import numpy as np
import pytest


def _fd(f, x, h=1e-6):
    """Reference central difference."""
    x = np.asarray(x, dtype=float)
    return (f(x + h) - f(x - h)) / (2 * h)


POINTS = np.array([-1.7, -0.6, 0.3, 0.9, 2.2])  # random-ish points: never 0, 1, or symmetric pairs


# ---------------------------------------------------------------- stage 1: secant slopes


def test_stage1_worked_example(nb):
    got = nb.secant_slope(lambda x: x**2, 1.0, 3.0)
    assert np.isclose(got, 4.0), (
        f"the line through (1, 1) and (3, 9) rises 8 over a run of 2, slope 4.0; got {got}. "
        + ("-4 means you computed (f(a) - f(b)) / (b - a): rise and run must go the same way."
           if np.isclose(got, -4.0) else ""))


def test_stage1_order_of_points_does_not_matter(nb):
    f = lambda x: x**3 - 2 * x
    s1 = nb.secant_slope(f, -0.5, 2.0)
    s2 = nb.secant_slope(f, 2.0, -0.5)
    want = (f(2.0) - f(-0.5)) / 2.5
    assert np.isclose(s1, want) and np.isclose(s2, want), (
        f"the slope of a line doesn't depend on which point you call a and which b: expected {want} "
        f"both ways, got {s1} and {s2}")


def test_stage1_arrays(nb):
    a = np.array([0.0, 1.0, 2.0])
    b = a + 0.5
    got = np.asarray(nb.secant_slope(np.exp, a, b))
    want = (np.exp(b) - np.exp(a)) / 0.5
    assert got.shape == (3,) and np.allclose(got, want), (
        f"with arrays a and b, compute every slope at once (plain array arithmetic): expected {want}, got {got}")


# ---------------------------------------------------------------- stage 2: finite differences


def test_stage2_worked_example_all_methods(nb):
    f = lambda x: x**2
    for method, want in (("forward", 6.1), ("backward", 5.9), ("central", 6.0)):
        got = float(nb.derivative(f, 3.0, h=0.1, method=method))
        assert np.isclose(got, want, atol=1e-10), (
            f"derivative(x^2, 3.0, h=0.1, method={method!r}) should be {want}, got {got:.6f}. "
            "forward uses the points x and x + h, backward x - h and x, central x - h and x + h.")


def test_stage2_default_is_central_and_accurate(nb):
    got = np.asarray(nb.derivative(np.sin, POINTS))
    assert got.shape == POINTS.shape, (
        f"derivative at an array of {POINTS.shape} points should return shape {POINTS.shape}, got {got.shape}")
    err = np.abs(got - np.cos(POINTS)).max()
    assert err < 1e-9, (
        f"the default (central, h = 1e-5) derivative of sin should match cos to ~1e-11; worst error {err:.1e}. "
        + ("an error near 1e-6 means your default method or h isn't central with h = 1e-5." if err > 1e-7 else ""))


def test_stage2_uses_h(nb):
    f = lambda x: x**3
    got = float(nb.derivative(f, 2.0, h=0.5))
    want = (2.5**3 - 1.5**3) / 1.0  # = 12.25: central differences of x^3 are 3x^2 + h^2
    assert np.isclose(got, want), f"derivative(x^3, 2.0, h=0.5) (central) should be {want}, got {got}. use the h you're given."
    gf = float(nb.derivative(f, 2.0, h=0.5, method="forward"))
    assert np.isclose(gf, (2.5**3 - 8.0) / 0.5), f"forward with h=0.5 should be {(2.5**3 - 8.0) / 0.5}, got {gf}"


def test_stage2_forward_and_backward_errors_are_first_order(nb):
    # on e^x at 0, forward overshoots by about h/2 and backward undershoots by about h/2
    h = 1e-3
    ef = float(nb.derivative(np.exp, 0.0, h=h, method="forward")) - 1.0
    eb = float(nb.derivative(np.exp, 0.0, h=h, method="backward")) - 1.0
    assert np.isclose(ef, h / 2, rtol=0.01) and np.isclose(eb, -h / 2, rtol=0.01), (
        f"for e^x at 0 with h = 1e-3, forward should be too big by about h/2 = 5e-4 and backward too "
        f"small by about 5e-4; got errors {ef:.2e} and {eb:.2e}")


def test_stage2_rejects_unknown_method(nb):
    with pytest.raises(ValueError):
        nb.derivative(np.sin, 1.0, method="sideways")


# ---------------------------------------------------------------- stage 3: rules


def test_stage3_product_rule(nb):
    f, df = np.sin, np.cos
    g = lambda x: x**3 + 1
    dg = lambda x: 3 * x**2
    dp = nb.product_rule(f, df, g, dg)
    assert callable(dp), "product_rule should return a FUNCTION of x"
    got = np.asarray(dp(POINTS))
    want = _fd(lambda x: f(x) * g(x), POINTS)
    assert np.allclose(got, want, atol=1e-7), (
        f"(f g)' should match finite differences {np.round(want, 5)}, got {np.round(got, 5)}. "
        "it's f' g + f g', not f' g' (the derivative of a product is NOT the product of the derivatives).")


def test_stage3_chain_rule(nb):
    f, df = np.sin, np.cos
    g = lambda x: x**2 + 0.5 * x
    dg = lambda x: 2 * x + 0.5
    dc = nb.chain_rule(f, df, g, dg)
    assert callable(dc), "chain_rule should return a FUNCTION of x"
    got = np.asarray(dc(POINTS))
    want = _fd(lambda x: f(g(x)), POINTS)
    if np.allclose(got, df(POINTS) * dg(POINTS), atol=1e-7):
        raise AssertionError(
            "your chain rule gives f'(x) * g'(x). the outer derivative must be evaluated at the INNER "
            "value: f'(g(x)) * g'(x).")
    assert np.allclose(got, want, atol=1e-7), (
        f"(f(g(x)))' should match finite differences {np.round(want, 5)}, got {np.round(got, 5)}. "
        "remember the inner slope: f'(g(x)) * g'(x).")


def test_stage3_rules_combine(nb):
    # d/dx [x^2 * e^(sin x)]: a product whose second factor is a composition
    sq, dsq = (lambda x: x**2), (lambda x: 2 * x)
    inner = nb.chain_rule(np.exp, np.exp, np.sin, np.cos)          # derivative of e^(sin x)
    dp = nb.product_rule(sq, dsq, lambda x: np.exp(np.sin(x)), inner)
    got = np.asarray(dp(POINTS))
    want = _fd(lambda x: x**2 * np.exp(np.sin(x)), POINTS)
    assert np.allclose(got, want, atol=1e-6), (
        f"product_rule and chain_rule should combine: d/dx x^2 e^(sin x) = {np.round(want, 5)}, got {np.round(got, 5)}")


# ---------------------------------------------------------------- stage 4: the derivative checker


def test_stage4_correct_derivative_passes(nb):
    f = lambda x: np.exp(np.sin(x)) * x
    df = lambda x: np.exp(np.sin(x)) * (np.cos(x) * x + 1)
    err = nb.check_derivative(f, df, np.linspace(-3, 3, 13))
    assert isinstance(err, float), f"check_derivative should return a float, got {type(err).__name__}"
    assert err < 1e-8, f"a correct derivative should give an error around 1e-10 or less, got {err:.2e}"


def test_stage4_wrong_derivative_fails(nb):
    f = lambda x: np.sin(x**2)
    buggy = lambda x: np.cos(x**2)  # forgot the chain rule's 2x
    err = nb.check_derivative(f, buggy, POINTS)
    assert err > 0.1, f"cos(x^2) is a WRONG derivative of sin(x^2); the checker should report a big error, got {err:.2e}"


def test_stage4_exact_value_max_and_central(nb):
    # with h = 0.1, the central difference of x^3 is exactly 3x^2 + h^2 = 3x^2 + 0.01, so the
    # errors at x = 0, 0.1, 2 are 0.01/1, 0.01/1 and 0.01/12.01: the max is 0.01
    got = nb.check_derivative(lambda x: x**3, lambda x: 3 * x**2, [0.0, 0.1, 2.0], h=0.1)
    hint = ""
    if np.isclose(got, np.mean([0.01, 0.01, 0.01 / 12.01])):
        hint = "you returned the MEAN error; return the largest one."
    elif got > 0.04:
        hint = "use CENTRAL differences (a forward difference has a much bigger error)."
    elif np.isclose(got, 0.01 / 12.01, rtol=1e-4):
        hint = ("divide each point's error by max(1, |fd(x)|) at THAT point, not by one maximum over all the "
                "points.")
    elif got < 1e-6:
        hint = "pass h through to derivative (with the default h = 1e-5 the error would be tiny)."
    assert np.isclose(got, 0.01, rtol=1e-6), (
        f"check_derivative(x^3, 3x^2, [0, 0.1, 2], h=0.1) should be exactly 0.01, got {got:.6g}. {hint}")


def test_stage4_relative_to_finite_difference_with_floor(nb):
    # df = 0 everywhere, f = 5x^3: at x = 2 the finite difference is 60, so the error is 60/60 = 1
    got = nb.check_derivative(lambda x: 5 * x**3, lambda x: 0 * x, [2.0])
    assert np.isclose(got, 1.0, rtol=1e-6), (
        f"for df = 0 against fd = 60 the error is |0 - 60| / max(1, |60|) = 1, got {got:.6g}. "
        "divide by max(1, |fd|), the finite difference, not by |df|.")
    # near-zero derivatives: the floor of 1 stops the error from exploding
    got = nb.check_derivative(lambda x: 1e-9 * x, lambda x: 2e-9 + 0 * x, [0.5])
    assert got < 1e-8, (
        f"for tiny derivatives (1e-9 vs 2e-9) the max(1, |fd|) floor makes the error ~1e-9, got {got:.3g}: "
        "without the floor, two tiny numbers look 100% different.")


# ---------------------------------------------------------------- stage 5: second derivative


def test_stage5_worked_example(nb):
    got = float(nb.second_derivative(lambda x: x**4, 1.0, h=0.1))
    assert np.isclose(got, 12.02, atol=1e-9), (
        f"(1.1^4 - 2 * 1^4 + 0.9^4) / 0.1^2 = 12.02, got {got:.6f}. "
        + ("6.01 means you divided by 2h^2." if np.isclose(got, 6.01) else
           "12.08 means you stepped by 2h (a central difference of central differences): use x - h, x, x + h."
           if np.isclose(got, 12.08) else
           "check the formula (f(x+h) - 2f(x) + f(x-h)) / h^2."))


def test_stage5_default_h_is_accurate(nb):
    got = np.asarray(nb.second_derivative(np.sin, POINTS))
    assert got.shape == POINTS.shape, f"expected shape {POINTS.shape}, got {got.shape}"
    err = np.abs(got + np.sin(POINTS)).max()
    assert err < 2e-7, (
        f"(sin)'' = -sin; with the default h = 1e-4 the error should be ~1e-8, got {err:.1e}. "
        "a much smaller h makes rounding errors explode, because you divide by h^2.")


def test_stage5_exact_on_quadratics(nb):
    got = np.asarray(nb.second_derivative(lambda x: 3 * x**2 - x + 7, POINTS, h=0.5))
    assert np.allclose(got, 6.0, atol=1e-10), (
        f"the second difference is exact for quadratics: (3x^2 - x + 7)'' = 6 for any h, got {got}")
