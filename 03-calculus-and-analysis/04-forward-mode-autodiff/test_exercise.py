"""Checks for exercise.py (forward-mode autodiff).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Derivatives are
compared with exact formulas at tolerances (1e-13 relative) that finite differences can't
reach, and the gradient tests count how many times your code calls f.
"""
import math

import numpy as np


def _parts(d):
    return float(d.real), float(d.dual)


def _check(d, real, dual, label, hint="", rtol=1e-13):
    assert hasattr(d, "real") and hasattr(d, "dual"), f"{label} should return a Dual, got {d!r}"
    r, du = _parts(d)
    assert np.isclose(r, real, rtol=rtol, atol=1e-300) and np.isclose(du, dual, rtol=rtol, atol=1e-15), (
        f"{label}: expected Dual({real!r}, {dual!r}), got Dual({r!r}, {du!r}). {hint}")


class _Counter:
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, x):
        self.calls += 1
        return self.f(x)


# ---------------------------------------------------------------- stage 1: arithmetic


def test_stage1_attributes(nb):
    d = nb.Dual(3.0, 1.5)
    assert d.real == 3.0 and d.dual == 1.5, "Dual(3.0, 1.5) should have .real == 3.0 and .dual == 1.5"
    c = nb.Dual(2.0)
    assert c.real == 2.0 and c.dual == 0.0, "Dual(2.0) is a constant: its dual part defaults to 0.0"


def test_stage1_add_sub_neg(nb):
    D = nb.Dual
    _check(D(3.0, 1.0) + D(2.0, 5.0), 5.0, 6.0, "Dual(3, 1) + Dual(2, 5)", "add the parts separately.")
    _check(D(3.0, 1.0) - D(2.0, 5.0), 1.0, -4.0, "Dual(3, 1) - Dual(2, 5)")
    _check(D(3.0, 1.0) + 2.0, 5.0, 1.0, "Dual(3, 1) + 2", "a plain number is a Dual with dual part 0.")
    _check(2.0 + D(3.0, 1.0), 5.0, 1.0, "2 + Dual(3, 1)", "define __radd__ for a number on the left.")
    _check(D(3.0, 1.0) - 2.0, 1.0, 1.0, "Dual(3, 1) - 2")
    _check(2.0 - D(3.0, 1.0), -1.0, -1.0, "2 - Dual(3, 1)",
           "__rsub__ computes other - self: the dual part is NEGATED.")
    _check(-D(3.0, 1.0), -3.0, -1.0, "-Dual(3, 1)")


def test_stage1_product_rule(nb):
    D = nb.Dual
    _check(D(3.0, 1.0) * D(2.0, 5.0), 6.0, 17.0, "Dual(3, 1) * Dual(2, 5)",
           "(a + b eps)(c + d eps) = ac + (ad + bc) eps, because eps^2 = 0. "
           + "(5 means you multiplied the dual parts: that's the eps^2 term, which vanishes.)")
    _check(D(3.0, 1.0) * 4.0, 12.0, 4.0, "Dual(3, 1) * 4")
    _check(4.0 * D(3.0, 1.0), 12.0, 4.0, "4 * Dual(3, 1)", "define __rmul__.")


def test_stage1_quotient_rule(nb):
    D = nb.Dual
    _check(D(3.0, 1.0) / D(2.0, 5.0), 1.5, -3.25, "Dual(3, 1) / Dual(2, 5)",
           "(a + b eps)/(c + d eps) = a/c + (bc - ad)/c^2 eps = 1.5 + (2 - 15)/4 eps.")
    _check(D(3.0, 1.0) / 2.0, 1.5, 0.5, "Dual(3, 1) / 2")
    _check(2.0 / D(2.0, 5.0), 1.0, -2.5, "2 / Dual(2, 5)", "__rtruediv__: c / (a + b eps) = c/a - (c b / a^2) eps.")


def test_stage1_comparisons_use_real_parts(nb):
    D = nb.Dual
    assert D(1.0, 5.0) < D(2.0, -3.0) and D(2.0, -3.0) > D(1.0, 5.0), "compare by the real parts only"
    assert D(2.0, 7.0) <= 2.0 and D(2.0, 7.0) >= 2.0 and D(2.0, 1.0) == 2.0, (
        "comparisons with plain numbers use the real part: Dual(2, 7) <= 2, >= 2 and == 2 are all True")
    assert not (D(3.0, 0.0) < 1.0) and 1.0 < D(3.0, 0.0), "Dual(3, 0) < 1 is False, and 1 < Dual(3, 0) is True"


def test_stage1_operands_unchanged_and_numpy(nb):
    D = nb.Dual
    a, b = D(1.0, 2.0), D(3.0, 4.0)
    _ = a + b, a * b, a - 1.0, 2.0 / a
    assert _parts(a) == (1.0, 2.0) and _parts(b) == (3.0, 4.0), (
        "arithmetic must return NEW Duals and leave its operands alone")
    _check(np.float64(2.0) * D(3.0, 1.0), 6.0, 2.0, "np.float64(2) * Dual(3, 1)")
    x = np.array([D(1.0, 1.0), D(2.0, 0.0)], dtype=object)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = A @ x
    _check(y[0], 5.0, 1.0, "(A @ x)[0] for an object array x of Duals")
    _check(x @ A @ x, 27.0, 12.0, "x^T A x on an object array of Duals",
           "NumPy calls your +, * and reflected operators entry by entry.")


def test_stage1_dual_times_numpy_array(nb):
    D = nb.Dual
    try:
        y = D(2.0, 1.0) * np.array([1.0, 3.0])
        z = D(2.0, 1.0) - np.array([1.0, 3.0])
    except TypeError as e:
        raise AssertionError(
            f"Dual(2, 1) * np.array([1., 3.]) raised {type(e).__name__}: {e}. when `other` is a NumPy array, return "
            "NotImplemented from __add__/__sub__/__mul__/__truediv__, so NumPy applies the operation entry by entry.") from None
    ok = isinstance(y, np.ndarray) and y.shape == (2,) and all(hasattr(v, "dual") for v in y)
    assert ok, (
        f"Dual(2, 1) * np.array([1., 3.]) should be an object ARRAY of two Duals, [Dual(2, 1), Dual(6, 3)]; got {y!r}. "
        "a Dual whose parts are arrays looks fine until code like np.sum((x[0] - mu)**2) breaks later: in each "
        "binary method, `if isinstance(other, np.ndarray): return NotImplemented`.")
    _check(y[1], 6.0, 3.0, "(Dual(2, 1) * np.array([1., 3.]))[1]")
    _check(z[1], -1.0, 1.0, "(Dual(2, 1) - np.array([1., 3.]))[1]")
    mu = np.array([0.5, -1.0, 2.0])
    w = (D(1.0, 1.0) - mu) * (D(1.0, 1.0) - mu)
    _check(np.sum(w), float(np.sum((1 - mu) ** 2)), float(np.sum(2 * (1 - mu))), "np.sum((x - mu) * (x - mu))",
           "the derivative of sum_i (x - mu_i)^2 is sum_i 2 (x - mu_i).")


def test_stage1_polynomial_derivative(nb):
    D = nb.Dual

    def f(x):
        return 3 * x * x * x - 2 * x / (1 + x) + 7

    y = f(D(2.0, 1.0))
    _check(y, 24 - 4 / 3 + 7, 36 - 2 / 9, "f(Dual(2, 1)) for f(x) = 3x^3 - 2x/(1 + x) + 7",
           "f'(x) = 9x^2 - 2/(1 + x)^2 = 35.777...: if the value is right but the derivative isn't, recheck "
           "the product and quotient rules.")


# ---------------------------------------------------------------- stage 2: elementary functions


def test_stage2_elementary_functions_chain_rule(nb):
    D = nb.Dual
    a, b = 0.7, 2.0  # dual part 2: every rule must multiply f'(a) by b
    cases = {
        "exp": (np.exp(a), np.exp(a) * b),
        "log": (np.log(a), b / a),
        "sin": (np.sin(a), np.cos(a) * b),
        "cos": (np.cos(a), -np.sin(a) * b),
        "tanh": (np.tanh(a), (1 - np.tanh(a) ** 2) * b),
        "sqrt": (np.sqrt(a), b / (2 * np.sqrt(a))),
        "log1p": (np.log1p(a), b / (1 + a)),
        "expm1": (np.expm1(a), np.exp(a) * b),
    }
    for name, (real, dual) in cases.items():
        got = getattr(D(a, b), name)()
        _check(got, real, dual, f"Dual(0.7, 2.0).{name}()",
               "f(a + b eps) = f(a) + f'(a) * b eps: did you forget the factor b (the incoming dual part)?")


def test_stage2_numpy_functions_dispatch(nb):
    D = nb.Dual
    _check(np.sin(D(1.0, 1.0)), np.sin(1.0), np.cos(1.0), "np.sin(Dual(1, 1))",
           "np.sin on a Dual calls its .sin() method: name the method exactly `sin`.")
    x = np.array([D(0.5, 1.0), D(1.5, 0.0)], dtype=object)
    y = np.exp(x)
    _check(y[0], np.exp(0.5), np.exp(0.5), "np.exp(object array)[0]")


def test_stage2_powers(nb):
    D = nb.Dual
    _check(D(2.0, 1.0) ** 3, 8.0, 12.0, "Dual(2, 1) ** 3", "d/dx x^3 = 3x^2.")
    _check(D(4.0, 1.0) ** 0.5, 2.0, 0.25, "Dual(4, 1) ** 0.5")
    _check(D(2.0, 1.0) ** -1, 0.5, -0.25, "Dual(2, 1) ** -1")
    _check(D(0.0, 1.0) ** 2, 0.0, 0.0, "Dual(0, 1) ** 2")
    try:
        zero_pow = D(0.0, 1.0) ** 0
    except ZeroDivisionError:
        raise AssertionError("Dual(0, 1) ** 0 raised ZeroDivisionError: the formula p a^(p-1) b computes 0.0 ** -1. "
                             "x^0 is the constant 1: handle p == 0 separately and return Dual(1, 0).") from None
    _check(zero_pow, 1.0, 0.0, "Dual(0, 1) ** 0", "x^0 is the constant 1: its derivative is 0.")
    try:
        arr = D(2.0, 1.0) ** np.array([1.0, 3.0])
    except (TypeError, ValueError) as e:
        raise AssertionError(f"Dual ** ndarray raised {type(e).__name__}: return NotImplemented for an array exponent "
                             "so NumPy handles it entry by entry") from None
    assert isinstance(arr, np.ndarray) and arr.shape == (2,), f"Dual(2, 1) ** np.array([1., 3.]) should be an object array; got {arr!r}"
    _check(arr[1], 8.0, 12.0, "(Dual(2, 1) ** np.array([1., 3.]))[1]")
    _check(D(2.0, 0.0) ** D(3.0, 1.0), 8.0, 8 * np.log(2.0), "Dual(2, 0) ** Dual(3, 1)",
           "for a Dual exponent use x^p = exp(p log x): d/dp 2^p = 2^p log 2.")
    _check(D(2.0, 1.0) ** D(3.0, 0.0), 8.0, 12.0, "Dual(2, 1) ** Dual(3, 0)")
    _check(2.0 ** D(3.0, 1.0), 8.0, 8 * np.log(2.0), "2 ** Dual(3, 1)", "define __rpow__: c^x = c^a + c^a log(c) b eps.")


def test_stage2_abs(nb):
    D = nb.Dual
    _check(abs(D(-2.0, 3.0)), 2.0, -3.0, "abs(Dual(-2, 3))", "for a negative real part, abs flips the sign of BOTH parts.")
    _check(abs(D(2.0, 3.0)), 2.0, 3.0, "abs(Dual(2, 3))")


def test_stage2_nested_duals_give_second_derivatives(nb):
    D = nb.Dual
    x0 = 0.8
    # x = x0 + e1 + e2 with two independent infinitesimals: the e1 e2 part of f(x) is f''(x0)
    x = D(D(x0, 1.0), D(1.0, 0.0))
    try:
        y = x.sin() * x.exp() + x ** 3 / (1 + x * x) + (2 * x).sqrt().log() + x.tanh().cos()
        got = float(y.dual.dual)
    except (TypeError, AttributeError) as e:
        raise AssertionError(
            "a Dual whose parts are Duals (nesting) broke your code: inside the methods, use np.exp/np.log/... "
            f"and plain arithmetic on .real and .dual, never math.exp or float(). ({type(e).__name__}: {e})") from None

    def f(t):
        return math.sin(t) * math.exp(t) + t**3 / (1 + t * t) + 0.5 * math.log(2 * t) + math.cos(math.tanh(t))

    h = 1e-3  # a fourth-order second difference: accurate to about 1e-10 here
    ref = (-f(x0 + 2 * h) + 16 * f(x0 + h) - 30 * f(x0) + 16 * f(x0 - h) - f(x0 - 2 * h)) / (12 * h * h)
    assert np.isclose(got, ref, rtol=1e-8), (
        f"evaluating f on Dual(Dual(x, 1), Dual(1, 0)) should put f''(x) = {ref:.10f} in .dual.dual; got {got:.10f}")


# ---------------------------------------------------------------- stage 3: derivatives and gradients


def test_stage3_dual_parts(nb):
    D = nb.Dual
    assert nb.dual_parts(D(2.0, 3.0)) == (2.0, 3.0), "dual_parts(Dual(2, 3)) should be (2.0, 3.0)"
    assert nb.dual_parts(5.0) == (5.0, 0.0), "a plain number has dual part 0: dual_parts(5.0) == (5.0, 0.0)"
    arr = np.array([[D(1.0, 2.0), 3.0], [D(4.0, -1.0), D(0.0, 0.5)]], dtype=object)
    r, d = nb.dual_parts(arr)
    r, d = np.asarray(r), np.asarray(d)
    assert r.shape == (2, 2) and d.shape == (2, 2) and r.dtype == float and d.dtype == float, (
        f"for a (2, 2) object array, return two FLOAT arrays of shape (2, 2); got {r.shape} {r.dtype}, {d.shape} {d.dtype}")
    assert np.array_equal(r, [[1.0, 3.0], [4.0, 0.0]]) and np.array_equal(d, [[2.0, 0.0], [-1.0, 0.5]]), (
        f"wrong parts: real {r.tolist()}, dual {d.tolist()}")


def test_stage3_derivative_is_exact(nb):
    f = lambda x: np.exp(np.sin(x)) * x**2 / (1 + np.sqrt(x))
    x0 = 1.3
    s = np.sqrt(x0)
    u = np.exp(np.sin(x0))
    want = (u * np.cos(x0) * x0**2 + u * 2 * x0) / (1 + s) - u * x0**2 / (1 + s) ** 2 / (2 * s)
    got = nb.derivative(f, x0)
    assert isinstance(got, float), f"derivative should return a Python float, got {type(got).__name__}"
    assert np.isclose(got, want, rtol=1e-13, atol=0), (
        f"expected f'(1.3) = {want!r}, got {got!r} (relative error {abs(got - want) / abs(want):.1e}). "
        "forward mode is exact to rounding: errors around 1e-10 mean you used finite differences.")
    assert nb.derivative(lambda x: 3.0, 2.0) == 0.0, "the derivative of a constant function is 0"


def test_stage3_worked_example(nb):
    f = lambda x: np.log(x[0]) + x[0] * x[1] - np.sin(x[1])
    val, g = nb.value_and_grad(f, np.array([2.0, 5.0]))
    want = np.array([0.5 + 5.0, 2.0 - np.cos(5.0)])
    assert np.isclose(val, np.log(2) + 10 - np.sin(5.0), rtol=1e-14), (
        f"f(2, 5) = log 2 + 10 - sin 5 = 11.6521, got {val}")
    assert isinstance(val, float), "value_and_grad's value should be a float"
    g = np.asarray(g)
    assert g.shape == (2,) and g.dtype == float and np.allclose(g, want, rtol=1e-14, atol=0), (
        f"the gradient of log(x1) + x1 x2 - sin(x2) at (2, 5) is (1/x1 + x2, x1 - cos x2) = {np.round(want, 6)}; "
        f"got {g!r}. return a float array.")


def test_stage3_grad_uses_n_passes_and_is_exact(nb):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    f = _Counter(lambda x: x @ A @ x + np.sum(np.tanh(x)) * x[0])
    x = rng.normal(size=6)
    g = np.asarray(nb.grad(f, x))
    t = np.tanh(x)
    want = (A + A.T) @ x + (1 - t**2) * x[0]
    want[0] += np.sum(t)
    assert g.shape == (6,) and np.allclose(g, want, rtol=1e-12, atol=1e-13), (
        f"grad is off by up to {np.abs(g - want).max():.1e}; forward mode should match the exact gradient to ~1e-15")
    assert f.calls == 6, (
        f"grad should call f exactly n = 6 times (one forward pass per seed e_i); it called f {f.calls} times. "
        + ("2n calls suggests central differences, not dual numbers." if f.calls == 12 else ""))
    f.calls = 0
    nb.value_and_grad(f, x)
    assert f.calls == 6, f"value_and_grad should also use n = 6 calls (take the value from the first pass), used {f.calls}"


def test_stage3_grad_details(nb):
    g = np.asarray(nb.grad(lambda x: x[0] ** 2 + 3 * x[1], np.array([3, 1])))
    assert np.allclose(g, [6.0, 3.0]), f"works for an integer x too: expected [6, 3], got {g}"
    g = np.asarray(nb.grad(lambda x: 4.0, np.array([1.0, 2.0])))
    assert np.array_equal(g, [0.0, 0.0]), "a function that ignores x (returns a plain float) has gradient 0"
    relu_sum = lambda x: np.sum(np.maximum(x, 0.0))
    g = np.asarray(nb.grad(relu_sum, np.array([-1.0, 2.0, 0.5])))
    assert np.allclose(g, [0.0, 1.0, 1.0]), f"np.maximum works through comparisons by real part: expected [0, 1, 1], got {g}"


# ---------------------------------------------------------------- stage 4: JVPs and Jacobians


def _F(x):
    return np.array([x[0] * x[1], np.sin(x[0]) + x[1] ** 2, np.exp(x[1])], dtype=object)


def _JF(x):
    return np.array([[x[1], x[0]], [np.cos(x[0]), 2 * x[1]], [0.0, np.exp(x[1])]])


def test_stage4_jvp_one_pass(nb):
    x, v = np.array([0.7, -1.3]), np.array([2.0, 0.5])
    F = _Counter(_F)
    out = nb.jvp(F, x, v)
    assert F.calls == 1, f"jvp must evaluate f exactly ONCE (seed the dual parts with v); it called f {F.calls} times"
    val, tangent = (np.asarray(o, dtype=float) for o in out)
    assert np.allclose(val, [x[0] * x[1], np.sin(x[0]) + x[1] ** 2, np.exp(x[1])], rtol=1e-14), "the first output is f(x)"
    assert tangent.shape == (3,) and np.allclose(tangent, _JF(x) @ v, rtol=1e-14, atol=0), (
        f"the second output is J(x) @ v = {np.round(_JF(x) @ v, 6)}, got {tangent}")


def test_stage4_jvp_scalar_function(nb):
    f = lambda x: np.sum(x**3)
    x, v = np.array([1.0, 2.0, -1.0]), np.array([0.5, -1.0, 2.0])
    val, d = nb.jvp(f, x, v)
    assert np.isclose(float(val), 8.0) and np.isclose(float(d), (3 * x**2) @ v, rtol=1e-14, atol=0), (
        f"for scalar f, jvp returns (f(x), grad f . v) = (8, {(3 * x**2) @ v}); got ({val}, {d})")


def test_stage4_jacobian_non_square(nb):
    x = np.array([0.7, -1.3])
    F = _Counter(_F)
    J = np.asarray(nb.jacobian(F, x), dtype=float)
    assert J.shape == (3, 2), (
        f"F maps R^2 -> R^3, so its Jacobian is (outputs, inputs) = (3, 2); got {J.shape}"
        + (". you stacked the columns as rows." if J.shape == (2, 3) else ""))
    assert np.allclose(J, _JF(x), rtol=1e-14, atol=1e-15), f"expected {np.round(_JF(x), 6).tolist()}, got {np.round(J, 6).tolist()}"
    assert F.calls == 2, f"forward mode builds the Jacobian one column per pass: n = 2 calls, you used {F.calls}"


def test_stage4_jacobian_scalar_and_wide(nb):
    J = np.asarray(nb.jacobian(lambda x: x @ x, np.array([1.0, -2.0, 0.5])), dtype=float)
    assert J.shape == (1, 3) and np.allclose(J, [[2.0, -4.0, 1.0]]), f"a scalar f has a (1, n) Jacobian; got {J!r}"
    # one input, many outputs: still one pass per input
    t = np.linspace(0, 1, 50)
    curve = _Counter(lambda x: np.sin(x[0] * t))
    J = np.asarray(nb.jacobian(curve, np.array([2.0])), dtype=float)
    assert J.shape == (50, 1) and np.allclose(J[:, 0], t * np.cos(2.0 * t), rtol=1e-14, atol=1e-15), (
        "the Jacobian of x -> sin(x t) (50 outputs, 1 input) is the column t cos(x t)")
    assert curve.calls == 1, f"one input means ONE forward pass, however many outputs; used {curve.calls}"
