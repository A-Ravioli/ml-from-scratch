"""Checks for exercise.py (multivariable calculus).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Derivatives are
compared with exact formulas worked out by hand, at random, non-symmetric, non-square test
points, so a transposed Jacobian or an unsymmetrized Hessian can't slip through.
"""
import numpy as np
import pytest


def _cubic(v):
    """the worked example: f(x, y) = x^3 - 3x + y^2."""
    return v[0] ** 3 - 3 * v[0] + v[1] ** 2


def _rosen(v):
    return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2


# ---------------------------------------------------------------- stage 1: directional derivatives


def test_stage1_worked_example(nb):
    got = nb.directional_derivative(_cubic, np.array([2.0, 1.0]), np.array([1.0, 1.0]))
    assert np.isclose(got, 11.0, atol=1e-6), (
        f"for f = x^3 - 3x + y^2 at (2, 1), grad f = (3x^2 - 3, 2y) = (9, 2), so the derivative along "
        f"v = (1, 1) is 9 + 2 = 11; got {got}"
        + (". 7.78 = 11 / sqrt(2) means you normalized v: the directional derivative along v uses v as given."
           if np.isclose(got, 11 / np.sqrt(2), atol=1e-4) else ""))


def test_stage1_linear_in_v_and_matches_gradient(nb):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    f = lambda v: np.sin(v[0]) * v[1] + v @ A @ v + np.exp(v[2] - v[3])
    x, v = rng.normal(size=4), rng.normal(size=4)
    g = np.array([np.cos(x[0]) * x[1], np.sin(x[0]), np.exp(x[2] - x[3]), -np.exp(x[2] - x[3])]) + (A + A.T) @ x
    d1 = nb.directional_derivative(f, x, v)
    assert np.isclose(d1, g @ v, rtol=1e-7), f"the derivative along v should equal grad f(x) . v = {g @ v:.8f}, got {d1:.8f}"
    d3 = nb.directional_derivative(f, x, 3 * v)
    assert np.isclose(d3, 3 * d1, rtol=1e-7), "the directional derivative is linear in v: tripling v must triple it"
    d = nb.directional_derivative(lambda w: w[0] ** 3, np.array([1.0]), np.array([1.0]), h=0.1)
    assert np.isclose(d, 3.01, atol=1e-12), (
        f"with h = 0.1 the central difference of x^3 at 1 along v = 1 is (1.1^3 - 0.9^3) / 0.2 = 3.01; got {d}. "
        "use a central difference with the given h.")


def test_stage1_steepest_direction(nb):
    x = np.array([2.0, 1.0])
    u, slope = nb.steepest_direction(_cubic, x)
    u = np.asarray(u, dtype=float)
    want = np.array([9.0, 2.0]) / np.hypot(9, 2)
    assert u.shape == (2,) and np.allclose(u, want, atol=1e-7), (
        f"the steepest ascent direction is grad f / ||grad f|| = (9, 2) / {np.hypot(9, 2):.4f} = {np.round(want, 5)}, "
        f"got {np.round(u, 5)}")
    assert np.isclose(slope, np.hypot(9, 2), rtol=1e-7), f"the slope along it is ||grad f|| = {np.hypot(9, 2):.6f}, got {slope}"
    # no random unit direction is steeper
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(200, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    slopes = dirs @ np.array([9.0, 2.0])
    assert slopes.max() <= slope + 1e-6, "some direction is steeper than your 'steepest' one"


def test_stage1_flat_point_raises(nb):
    with pytest.raises(ValueError):
        nb.steepest_direction(lambda v: v @ v, np.zeros(3))


# ---------------------------------------------------------------- stage 2: Jacobians


def test_stage2_polar_worked_example(nb):
    F = lambda p: np.array([p[0] * np.cos(p[1]), p[0] * np.sin(p[1])])
    J = np.asarray(nb.numerical_jacobian(F, np.array([2.0, np.pi / 2])), dtype=float)
    want = np.array([[0.0, -2.0], [1.0, 0.0]])
    assert J.shape == (2, 2) and np.allclose(J, want, atol=1e-8), (
        f"the polar map (r cos t, r sin t) at r = 2, t = pi/2 has Jacobian [[cos t, -r sin t], [sin t, r cos t]] "
        f"= [[0, -2], [1, 0]]; got {np.round(J, 6).tolist()}"
        + (" (that's the transpose: row i holds the derivatives of OUTPUT i)" if np.allclose(J, want.T, atol=1e-6) else ""))


def test_stage2_non_square(nb):
    F = lambda v: np.array([v[0] * v[1], np.sin(v[0]) + v[1] ** 2, np.exp(v[1])])
    x = np.array([0.7, -1.3])
    J = np.asarray(nb.numerical_jacobian(F, x), dtype=float)
    want = np.array([[x[1], x[0]], [np.cos(x[0]), 2 * x[1]], [0.0, np.exp(x[1])]])
    assert J.shape == (3, 2), (
        f"F maps R^2 -> R^3, so its Jacobian has shape (outputs, inputs) = (3, 2); got {J.shape}")
    assert np.allclose(J, want, rtol=1e-7, atol=1e-9), f"expected {np.round(want, 6).tolist()}, got {np.round(J, 6).tolist()}"


def test_stage2_linear_map_and_h(nb):
    A = np.random.default_rng(2).normal(size=(5, 3))
    J = np.asarray(nb.numerical_jacobian(lambda v: A @ v, np.array([1, 2, 3])), dtype=float)
    assert np.allclose(J, A, atol=1e-8), (
        "the Jacobian of x -> A x is A itself (here for an INTEGER x = [1, 2, 3]: perturb a float copy)")
    J = np.asarray(nb.numerical_jacobian(lambda v: v**3, np.array([1.0, 2.0]), h=0.1), dtype=float)
    assert np.allclose(J, np.diag([3.01, 12.01]), atol=1e-12), (
        f"with h = 0.1 the central difference of x^3 is 3x^2 + h^2 = [3.01, 12.01] on the diagonal; got "
        f"{np.round(J, 6).tolist()}. pass h through and use central differences.")


def test_stage2_scalar_output_and_no_mutation(nb):
    x = np.array([0.5, -0.25, 2.0])
    x0 = x.copy()
    J = np.asarray(nb.numerical_jacobian(lambda v: float(v @ v), x), dtype=float)
    assert J.shape == (1, 3) and np.allclose(J, 2 * x[None, :], atol=1e-8), (
        f"a scalar-valued f has a (1, n) Jacobian: the gradient as a row. expected {(2 * x).tolist()} in shape (1, 3), "
        f"got shape {J.shape}")
    assert np.array_equal(x, x0), "numerical_jacobian must not modify x"


def test_stage2_chain_rule(nb):
    g = lambda u: np.array([np.sin(u[0]) * u[1], u[0] ** 2 - u[1], np.exp(0.3 * u[1])])
    F = lambda v: np.array([v[0] * v[1] + v[2], v[0] - np.cos(v[2])])
    x = np.array([0.4, -0.8, 1.1])
    Jgf = np.asarray(nb.numerical_jacobian(lambda v: g(F(v)), x), dtype=float)
    prod = np.asarray(nb.numerical_jacobian(g, F(x)), dtype=float) @ np.asarray(nb.numerical_jacobian(F, x), dtype=float)
    assert Jgf.shape == (3, 3) and np.allclose(Jgf, prod, rtol=1e-6, atol=1e-8), (
        "the chain rule says J_(g∘F)(x) = J_g(F(x)) @ J_F(x); your Jacobians don't satisfy it")


# ---------------------------------------------------------------- stage 3: Hessians


def test_stage3_quadratic_nonsymmetric(nb):
    A = np.random.default_rng(3).normal(size=(4, 4))
    x = np.random.default_rng(4).normal(size=4)
    H = np.asarray(nb.numerical_hessian(lambda v: v @ A @ v, x), dtype=float)
    want = A + A.T
    assert H.shape == (4, 4), f"the Hessian of a function of 4 variables is (4, 4), got {H.shape}"
    assert np.allclose(H, want, atol=1e-6), (
        f"the Hessian of x^T A x is A + A^T (constant), for any square A; yours differs by up to "
        f"{np.abs(H - want).max():.2e}")


def test_stage3_exactly_symmetric(nb):
    f = lambda v: np.exp(v[0] * v[1]) + np.sin(v[0] + 2 * v[2]) * v[1] ** 2
    H = np.asarray(nb.numerical_hessian(f, np.array([0.3, -0.7, 1.2])), dtype=float)
    assert np.array_equal(H, H.T), (
        f"the Hessian must be exactly symmetric (max |H - H^T| = {np.abs(H - H.T).max():.2e}). "
        "compute each off-diagonal pair once and set H[i, j] = H[j, i], or average H with its transpose.")


def test_stage3_rosenbrock_worked_values(nb):
    H = np.asarray(nb.numerical_hessian(_rosen, np.array([1.0, 1.0])), dtype=float)
    want = np.array([[802.0, -400.0], [-400.0, 200.0]])
    assert np.allclose(H, want, atol=1e-3), (
        f"the Rosenbrock function (1-x)^2 + 100(y-x^2)^2 has Hessian [[802, -400], [-400, 200]] at its minimum "
        f"(1, 1); got {np.round(H, 4).tolist()}")


def test_stage3_accuracy_with_large_values(nb):
    # f has values ~20 and a known Hessian: exp(sum v) * ones
    x = np.array([1.0, 1.0, 1.0])
    H = np.asarray(nb.numerical_hessian(lambda v: np.exp(np.sum(v)), x), dtype=float)
    want = np.exp(3.0) * np.ones((3, 3))
    err = np.abs(H - want).max() / np.exp(3.0)
    assert err < 1e-5, (
        f"relative error {err:.1e} on the Hessian of exp(x + y + z): too big. second differences divide "
        "by h^2, so rounding error grows like eps |f| / h^2: use the default h = 1e-4 (not 1e-6), and "
        "difference f directly rather than differencing a numerical gradient.")


def test_stage3_cross_term_and_no_mutation(nb):
    x = np.array([2, -1])  # integer input
    H = np.asarray(nb.numerical_hessian(lambda v: v[0] ** 2 * v[1] + 3 * v[0] * v[1] ** 2, x), dtype=float)
    want = np.array([[2 * x[1], 2 * x[0] + 6 * x[1]], [2 * x[0] + 6 * x[1], 6 * x[0]]], dtype=float)
    assert np.allclose(H, want, atol=1e-5), (
        f"for f = x^2 y + 3 x y^2 at (2, -1) the Hessian is [[2y, 2x + 6y], [2x + 6y, 6x]] = {want.tolist()}; "
        f"got {np.round(H, 5).tolist()}. (the input here is an INTEGER array: work on a float copy)")
    assert x.tolist() == [2, -1], "numerical_hessian must not modify x"


# ---------------------------------------------------------------- stage 4: critical points


def test_stage4_worked_example(nb):
    assert nb.classify_critical_point(_cubic, np.array([1.0, 0.0])) == "minimum", (
        "f = x^3 - 3x + y^2 has Hessian diag(6x, 2); at (1, 0) that's diag(6, 2): a minimum")
    assert nb.classify_critical_point(_cubic, np.array([-1.0, 0.0])) == "saddle", (
        "at (-1, 0) the Hessian is diag(-6, 2): one negative and one positive eigenvalue, a saddle")


def test_stage4_maximum_in_three_dimensions(nb):
    f = lambda v: -(v[0] ** 2) - 2 * v[1] ** 2 - 3 * v[2] ** 2 + v[0] * v[1]
    got = nb.classify_critical_point(f, np.zeros(3))
    assert got == "maximum", (
        f"every Hessian eigenvalue is negative at 0, so it's a maximum; got {got!r}. "
        "(in 3-D det(H) < 0 does NOT mean saddle: use the signs of the eigenvalues)")


def test_stage4_rotated_saddle(nb):
    f = lambda v: v[0] * v[1]  # Hessian [[0, 1], [1, 0]]: zero diagonal, eigenvalues +-1
    got = nb.classify_critical_point(f, np.zeros(2))
    assert got == "saddle", (
        f"f = xy has Hessian [[0, 1], [1, 0]] with eigenvalues +1 and -1: a saddle; got {got!r}. "
        "looking only at the diagonal of H misses this.")


def test_stage4_threshold_is_relative(nb):
    f = lambda v: 1e6 * v[0] ** 2 + 1e-3 * v[1] ** 2  # curvatures 2e6 and 2e-3: a ratio of 1e9
    got = nb.classify_critical_point(f, np.zeros(2))
    assert got == "degenerate", (
        f"the eigenvalues are 2e6 and 2e-3; relative to the Hessian's scale (s = tol * max(1, max|lambda|) = 2), "
        f"2e-3 counts as zero, so the answer is 'degenerate'; got {got!r}. scale the threshold by the largest "
        "|eigenvalue| instead of comparing with tol directly.")


def test_stage4_degenerate_cases(nb):
    got = nb.classify_critical_point(lambda v: v[0] ** 4 + v[1] ** 2, np.zeros(2))
    assert got == "degenerate", (
        f"x^4 + y^2 at 0 has Hessian diag(0, 2): a zero eigenvalue, so the second-derivative test can't "
        f"decide (it IS a minimum, but x^3 + y^2 has the same Hessian and isn't); got {got!r}")
    got = nb.classify_critical_point(lambda v: v[0] ** 3 - 3 * v[0] * v[1] ** 2, np.zeros(2))
    assert got == "degenerate", f"the monkey saddle x^3 - 3xy^2 has a zero Hessian at 0: 'degenerate'; got {got!r}"
    got = nb.classify_critical_point(lambda v: v[0] ** 2 - v[1] ** 2 + v[2] ** 4, np.zeros(3))
    assert got == "saddle", (
        f"x^2 - y^2 + z^4 has eigenvalues 2, -2, 0: mixed signs already prove a saddle, whatever the zero "
        f"eigenvalue does; got {got!r}")
