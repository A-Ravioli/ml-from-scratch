"""Checks for exercise.py (function spaces).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Inner products,
Legendre polynomials and coefficients are compared with closed forms and with NumPy's own
Legendre module (used here only as an oracle).
"""
import numpy as np
from numpy.polynomial import legendre as npleg


def _P(k, x):
    c = np.zeros(k + 1)
    c[k] = 1.0
    return npleg.legval(x, c)


# ---------------------------------------------------------------- stage 1: inner products


def test_stage1_orthogonality_and_lengths(nb):
    x1 = lambda t: t
    x2 = lambda t: t**2
    assert abs(nb.l2_inner(x1, x2, -1.0, 1.0)) < 1e-12, "x and x^2 are orthogonal on [-1, 1] (odd times even)"
    got = nb.l2_inner(x1, x1, -1.0, 1.0)
    assert np.isclose(got, 2 / 3, rtol=1e-6), f"<x, x> on [-1, 1] = integral of x^2 = 2/3; got {got}"
    assert abs(nb.l2_inner(np.sin, np.cos, -np.pi, np.pi)) < 1e-12, "sin and cos are orthogonal on [-pi, pi]"
    got = nb.l2_norm(lambda t: np.ones_like(t), 0.0, 2.0)
    assert np.isclose(got, np.sqrt(2), rtol=1e-12), (
        f"the constant 1 on [0, 2] has length sqrt(integral of 1) = sqrt(2); got {got}"
        + (". forgetting the cell width h gives sqrt(n)." if np.isclose(got, 64.0) else ""))


def test_stage1_midpoint_rule_with_given_n(nb):
    got = nb.l2_inner(lambda t: t**2, lambda t: np.ones_like(t), 0.0, 1.0, n=2)
    assert np.isclose(got, 0.3125, rtol=1e-14), (
        f"with n = 2 cells on [0, 1], the midpoints are 0.25 and 0.75, so <x^2, 1> ≈ (0.0625 + 0.5625)/2 = 0.3125; "
        f"got {got}. use the midpoint rule with the given n.")
    got = nb.l2_norm(lambda t: t**2, 0.0, 1.0, n=2)
    want = np.sqrt((0.25**4 + 0.75**4) / 2)
    assert np.isclose(got, want, rtol=1e-14), f"l2_norm must pass n on to l2_inner: with n = 2, ||x^2|| on [0, 1] ≈ {want:.7f}; got {got}"


def test_stage1_cauchy_schwarz(nb):
    f = lambda t: np.exp(t) * np.sin(3 * t)
    g = lambda t: 1 + t - t**3
    lhs = abs(nb.l2_inner(f, g, -1.0, 2.0))
    rhs = nb.l2_norm(f, -1.0, 2.0) * nb.l2_norm(g, -1.0, 2.0)
    assert lhs <= rhs, f"Cauchy-Schwarz: |<f, g>| <= ||f|| ||g||, but {lhs} > {rhs}"
    assert np.isclose(nb.l2_norm(lambda t: 3 * f(t), -1.0, 2.0), 3 * nb.l2_norm(f, -1.0, 2.0)), "||3 f|| = 3 ||f||"


# ---------------------------------------------------------------- stage 2: Legendre polynomials


def test_stage2_low_degrees(nb):
    x = np.linspace(-1, 1, 11)
    assert np.allclose(nb.legendre(0, x), 1.0) and np.asarray(nb.legendre(0, x)).shape == x.shape, (
        "P_0 = 1, as an array shaped like x")
    assert np.allclose(nb.legendre(1, x), x), "P_1 = x"
    assert np.allclose(nb.legendre(2, x), (3 * x**2 - 1) / 2), "P_2 = (3x^2 - 1)/2"
    got = nb.legendre(3, x)
    assert np.allclose(got, (5 * x**3 - 3 * x) / 2), (
        "P_3 = (5x^3 - 3x)/2. the recurrence is (j + 1) P_(j+1) = (2j + 1) x P_j - j P_(j-1): check the "
        "coefficients and which j you start from.")


def test_stage2_high_degree_and_endpoints(nb):
    x = np.random.default_rng(0).uniform(-1, 1, size=(4, 5))
    for k in (5, 10, 25):
        got = np.asarray(nb.legendre(k, x))
        assert got.shape == (4, 5), f"legendre(k, x) should have x's shape (4, 5); got {got.shape}"
        assert np.allclose(got, _P(k, x), atol=1e-12), f"P_{k} disagrees with the reference values"
    ends = np.array([1.0, -1.0])
    assert np.allclose(nb.legendre(7, ends), [1.0, -1.0]) and np.allclose(nb.legendre(8, ends), [1.0, 1.0]), (
        "P_k(1) = 1 and P_k(-1) = (-1)^k")


def test_stage2_orthogonal(nb):
    x = -1 + (np.arange(20000) + 0.5) / 10000
    w = 2 / 20000
    for j, k in ((2, 5), (3, 3), (4, 6), (6, 6)):
        ip = np.sum(nb.legendre(j, x) * nb.legendre(k, x)) * w
        want = 2 / (2 * k + 1) if j == k else 0.0
        assert np.isclose(ip, want, atol=1e-7), f"<P_{j}, P_{k}> should be {want:.6f}, got {ip:.6f}"


# ---------------------------------------------------------------- stage 3: Legendre projections


def test_stage3_cubic_exactly(nb):
    c = np.asarray(nb.legendre_coefficients(lambda t: t**3, 5), dtype=float)
    want = np.array([0.0, 0.6, 0.0, 0.4, 0.0, 0.0])
    assert c.shape == (6,), f"K = 5 gives 6 coefficients c_0..c_5; got shape {c.shape}"
    hint = ""
    if np.allclose(c, [0, 0.4, 0, 0.4 * 2 / 7, 0, 0], atol=1e-4):
        hint = " you returned <f, P_k>: divide by <P_k, P_k> = 2/(2k + 1)."
    # the midpoint rule with n = 4096 cells has error ~2e-6 on these coefficients, so 2e-5 is a safe bound
    assert np.allclose(c, want, atol=2e-5), (
        f"x^3 = (3/5) P_1 + (2/5) P_3, so the coefficients are {want.tolist()}; got {np.round(c, 6).tolist()}.{hint}")


def test_stage3_series_rebuilds_polynomials(nb):
    p = lambda t: 2 - t + 4 * t**2 - 0.5 * t**4
    c = nb.legendre_coefficients(p, 4)
    x = np.linspace(-1, 1, 9)
    got = np.asarray(nb.legendre_series(c, x))
    assert got.shape == x.shape and np.allclose(got, p(x), atol=1e-4), (
        "a polynomial of degree 4 is exactly its own degree-4 Legendre projection: legendre_series(c, x) should "
        "rebuild it")
    assert np.allclose(nb.legendre_series([1.0, 2.0, 3.0], x), 1 + 2 * x + 3 * (3 * x**2 - 1) / 2), (
        "legendre_series(c, x) = sum_k c[k] P_k(x)")


def test_stage3_best_approximation_and_bessel(nb):
    f = lambda t: np.exp(t) * np.cos(2 * t)
    x = -1 + (np.arange(4000) + 0.5) / 2000
    norm2 = np.sum(f(x) ** 2) * (2 / 4000)
    errs = []
    for K in (2, 4, 6):
        c = np.asarray(nb.legendre_coefficients(f, K), dtype=float)
        err = np.sqrt(np.sum((f(x) - nb.legendre_series(c, x)) ** 2) * (2 / 4000))
        taylorish = np.polynomial.Polynomial.fit(x[::40], f(x[::40]), K)(x)  # another degree-K polynomial
        other = np.sqrt(np.sum((f(x) - taylorish) ** 2) * (2 / 4000))
        assert err <= other + 1e-9, (
            f"degree {K}: the Legendre projection is the BEST degree-{K} approximation in the L2 norm, but another "
            f"polynomial does better ({other:.2e} < {err:.2e})")
        bessel = np.sum(c**2 * 2 / (2 * np.arange(K + 1) + 1))
        assert bessel <= norm2 + 1e-9, "Bessel's inequality: sum c_k^2 ||P_k||^2 <= ||f||^2"
        errs.append(err)
    assert errs[0] > errs[1] > errs[2], "projecting onto more polynomials must reduce the L2 error"


# ---------------------------------------------------------------- stage 4: Fourier projections


def test_stage4_trig_polynomial_exactly(nb):
    f = lambda t: 3 + 2 * np.cos(t) - np.sin(3 * t)
    a, b = (np.asarray(v, dtype=float) for v in nb.fourier_coefficients(f, 3))  # K = 3: the top frequency is used
    assert a.shape == (4,) and b.shape == (4,), f"K = 3 gives arrays of 4 coefficients each (k = 0..3); got {a.shape}, {b.shape}"
    hint = " with the convention f ≈ a0/2 + ..., a0 is TWICE the mean of f." if np.isclose(a[0], 3.0) else ""
    assert np.allclose(a, [6, 2, 0, 0], atol=1e-12) and np.allclose(b, [0, 0, 0, -1], atol=1e-12), (
        f"3 + 2 cos x - sin 3x has a = [6, 2, 0, 0] and b = [0, 0, 0, -1] (a0/2 = 3); got "
        f"a = {np.round(a, 6).tolist()}, b = {np.round(b, 6).tolist()}.{hint}")
    x = np.linspace(-3, 3, 7)
    assert np.allclose(nb.fourier_series(a, b, x), f(x), atol=1e-12), (
        "fourier_series should rebuild f from its coefficients, using EVERY frequency up to len(a) - 1 "
        "(here the last one, k = 3, carries the -sin 3x term)")


def test_stage4_square_wave(nb):
    sq = lambda t: np.sign(t)
    a, b = (np.asarray(v, dtype=float) for v in nb.fourier_coefficients(sq, 7))
    want_b = np.array([0, 4 / np.pi, 0, 4 / (3 * np.pi), 0, 4 / (5 * np.pi), 0, 4 / (7 * np.pi)])
    assert np.allclose(a, 0, atol=1e-10) and np.allclose(b, want_b, atol=1e-6), (
        f"the square wave sign(x) has b_k = 4/(pi k) for odd k and every other coefficient 0; got "
        f"b = {np.round(b, 5).tolist()}"
        + (" (that's off by a factor 2: the coefficients use 1/pi, not 1/(2 pi))" if np.allclose(b, want_b / 2, atol=1e-5) else ""))


def test_stage4_gibbs_overshoot(nb):
    a, b = nb.fourier_coefficients(np.sign, 101, n=8192)
    x = np.linspace(1e-4, 0.2, 4000)
    peak = float(np.max(nb.fourier_series(a, b, x)))
    assert 1.17 < peak < 1.19, (
        f"partial Fourier sums of a jump overshoot by about 9% of the jump: near 0 the degree-101 sum of sign(x) "
        f"peaks at about 1.179; got {peak:.4f}")
