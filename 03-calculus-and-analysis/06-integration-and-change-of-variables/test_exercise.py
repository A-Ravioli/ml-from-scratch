"""Checks for exercise.py (integration & change of variables).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Integrals are
compared with closed forms; Monte Carlo checks use fixed seeds, exact identities (constant
integrands) and coverage over many seeds; densities are compared with the Gaussian formula.
"""
import math

import numpy as np


def _std_normal_logpdf(z):
    z = np.atleast_1d(z)
    return float(-0.5 * z @ z - 0.5 * z.size * np.log(2 * np.pi))


# ---------------------------------------------------------------- stage 1: grid integration


def test_stage1_midpoint_exact_error(nb):
    n = 10
    got = nb.midpoint_rule(lambda X: X[:, 0] ** 2, 0.0, 1.0, n)
    want = 1 / 3 - 1 / (12 * n**2)  # the midpoint rule's exact answer for x^2
    hint = ""
    if np.isclose(got, 1 / 3 - 1 / (2 * n) + 1 / (6 * n**2)):
        hint = " you used LEFT endpoints; the midpoint of cell i is low + (i + 1/2) h."
    assert np.isclose(got, want, rtol=1e-13), (
        f"the midpoint rule with n = {n} on x^2 over [0, 1] gives exactly 1/3 - 1/(12 n^2) = {want:.10f}; "
        f"got {got!r}.{hint}")
    assert isinstance(got, float), "return a float"


def test_stage1_two_dimensional_box(nb):
    got = nb.midpoint_rule(lambda X: X[:, 0] * X[:, 1], [0.0, 0.0], [1.0, 2.0], 7)
    assert np.isclose(got, 1.0, rtol=1e-13), (
        f"the integral of x y over [0, 1] x [0, 2] is (1/2)(2) = 1, and the midpoint rule is exact for it; "
        f"got {got}. the box has different side lengths: each axis needs its own cell width.")
    got = nb.midpoint_rule(lambda X: np.exp(-np.sum(X**2, axis=1)), [-5.0, -5.0], [5.0, 5.0], 80)
    assert np.isclose(got, np.pi, rtol=1e-6), f"the integral of exp(-x^2 - y^2) over the plane is pi; got {got}"


def test_stage1_three_dimensions_and_vectorized_calls(nb):
    shapes = []

    def f(X):
        shapes.append(np.shape(X))
        return np.exp(-np.sum(X**2, axis=1))

    got = nb.midpoint_rule(f, [-5.0] * 3, [5.0] * 3, 30)
    assert np.isclose(got, np.pi**1.5, rtol=1e-5), f"in 3-D the Gaussian integral is pi^(3/2) = {np.pi ** 1.5:.6f}; got {got}"
    assert all(len(s) == 2 and s[1] == 3 for s in shapes), (
        f"f must be called with arrays of points of shape (m, 3), got shapes {sorted(set(shapes))[:3]}")
    assert sum(s[0] for s in shapes) == 30**3, f"n = 30 cells per axis in 3-D is 27000 points; f saw {sum(s[0] for s in shapes)}"


# ---------------------------------------------------------------- stage 2: Monte Carlo


def test_stage2_constant_integrand_and_volume(nb):
    est, se = nb.monte_carlo_integrate(lambda X: np.full(len(X), 2.5), [0.0, 1.0], [2.0, 4.0], 100, rng=0)
    assert np.isclose(est, 2.5 * 6.0) and np.isclose(se, 0.0, atol=1e-12), (
        f"a constant 2.5 over a box of volume 2 x 3 = 6 integrates to exactly 15 with zero standard error; "
        f"got ({est}, {se}). multiply the mean by the box volume.")


def test_stage2_standard_error_formula(nb):
    n = 40000
    est, se = nb.monte_carlo_integrate(lambda X: (X[:, 0] < 0.3).astype(float), [0.0], [1.0], n, rng=1)
    want_se = np.sqrt(0.3 * 0.7 / n)
    assert abs(est - 0.3) < 4 * want_se, f"the integral of 1[x < 0.3] over [0, 1] is 0.3; got {est}"
    assert np.isclose(se, want_se, rtol=0.03), (
        f"for a 0/1 integrand with p = 0.3 the standard error is sqrt(p(1-p)/n) = {want_se:.2e}; got {se:.2e}"
        + (". you divided by n instead of sqrt(n)." if np.isclose(se, want_se / np.sqrt(n), rtol=0.05) else ""))


def test_stage2_seeds_and_generators(nb):
    f = lambda X: np.sin(X[:, 0]) * X[:, 1]
    a = nb.monte_carlo_integrate(f, [0, 0], [1, 1], 500, rng=7)
    b = nb.monte_carlo_integrate(f, [0, 0], [1, 1], 500, rng=7)
    c = nb.monte_carlo_integrate(f, [0, 0], [1, 1], 500, rng=8)
    assert a == b, "the same integer seed must give the same result"
    assert a != c, "different seeds should give different results"
    g = nb.monte_carlo_integrate(f, [0, 0], [1, 1], 500, rng=np.random.default_rng(7))
    assert np.allclose(g, a), "rng may be a np.random.Generator: normalize it with np.random.default_rng(rng)"


def test_stage2_coverage(nb):
    truth = (1 - math.cos(2.0)) * 0.5 * 3.0  # integral of sin(x) y over [0, 2] x [0, 1], times 3
    f = lambda X: 3.0 * np.sin(X[:, 0]) * X[:, 1]
    hits = 0
    for seed in range(200):
        est, se = nb.monte_carlo_integrate(f, [0.0, 0.0], [2.0, 1.0], 200, rng=seed)
        hits += abs(est - truth) <= 2 * se
    assert 0.88 <= hits / 200 <= 0.995, (
        f"estimate ± 2 standard errors should contain the true value about 95% of the time; it did {hits / 200:.0%} "
        "of the time over 200 seeds. check the standard error (ddof=1, the volume factor, sqrt(n)).")


# ---------------------------------------------------------------- stage 3: Jacobian determinants


def test_stage3_linear_map_with_reflection(nb):
    A = np.array([[0.5, -1.0, 3.0], [2.0, 1.0, 0.0], [1.0, 0.0, 1.0]])  # det A = -0.5: a reflection
    got = nb.log_abs_det_jacobian(lambda z: A @ z + 1.0, np.array([0.3, -0.2, 0.5]))
    want = np.log(abs(np.linalg.det(A)))
    assert np.isfinite(got) and np.isclose(got, want, rtol=1e-7), (
        f"for T(z) = A z + b the Jacobian is A, det A = {np.linalg.det(A):.3f} (negative: a reflection), so "
        f"log|det| = {want:.6f}; got {got}. use np.linalg.slogdet and its absolute log-determinant.")


def test_stage3_polar_and_exp(nb):
    polar = lambda p: np.array([p[0] * np.cos(p[1]), p[0] * np.sin(p[1])])
    got = nb.log_abs_det_jacobian(polar, np.array([2.0, 0.7]))
    assert np.isclose(got, np.log(2.0), rtol=1e-7), f"the polar map has |det J| = r, so log|det| = log 2 at r = 2; got {got}"
    z = np.array([0.5, -1.0, 2.0, 0.1])
    got = nb.log_abs_det_jacobian(np.exp, z)
    assert np.isclose(got, z.sum(), rtol=1e-7), (
        f"elementwise exp has a diagonal Jacobian diag(e^z), so log|det| = sum(z) = {z.sum()}; got {got}")


def test_stage3_large_stretch_no_overflow(nb):
    got = nb.log_abs_det_jacobian(lambda z: 1e30 * z, np.ones(20))
    assert np.isclose(got, 20 * np.log(1e30), rtol=1e-9), (
        f"scaling 20 coordinates by 1e30 has det 1e600, which overflows a float, but log|det| = {20 * np.log(1e30):.4f} "
        f"is fine; got {got}. use slogdet, not log(abs(det)).")


# ---------------------------------------------------------------- stage 4: pushforward densities


def test_stage4_lognormal_worked_example(nb):
    log_pz = _std_normal_logpdf
    got1 = nb.pushforward_log_density(log_pz, np.log, np.array([1.0]))
    gote = nb.pushforward_log_density(log_pz, np.log, np.array([np.e]))
    want1, wante = -0.5 * np.log(2 * np.pi), -0.5 * np.log(2 * np.pi) - 0.5 - 1.0
    assert np.isclose(got1, want1, rtol=1e-8), f"X = e^Z has p_X(1) = phi(0) / 1, log = {want1:.6f}; got {got1}"
    hint = ""
    if np.isclose(gote, wante + 2.0, rtol=1e-6):
        hint = " you ADDED log|det J_T| at T^-1(x); the density needs the Jacobian of T^-1 at x, i.e. minus that."
    elif np.isclose(gote, wante + 1.0, rtol=1e-6):
        hint = " you left out the Jacobian term: a density is per unit volume, and T stretches volume."
    assert np.isclose(gote, wante, rtol=1e-8), (
        f"p_X(e) = phi(log e) * |d log x / dx| = phi(1) / e, so log p_X(e) = {wante:.6f}; got {gote}.{hint}")


def test_stage4_affine_gaussian(nb):
    A = np.array([[2.0, 0.0], [1.0, 0.5]])
    b = np.array([1.0, -1.0])
    Ainv = np.linalg.inv(A)
    S = A @ A.T
    x = np.array([[0.3, 0.2], [2.0, -1.5], [-1.0, 0.0]])
    got = np.asarray(nb.pushforward_log_density(_std_normal_logpdf, lambda v: Ainv @ (v - b), x), dtype=float)
    diff = x - b
    want = -0.5 * np.einsum("ij,jk,ik->i", diff, np.linalg.inv(S), diff) - np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(S))
    assert got.shape == (3,), f"a batch of 3 points (3, 2) should give 3 log-densities, got shape {got.shape}"
    assert np.allclose(got, want, rtol=1e-7), (
        f"X = A Z + b with Z ~ N(0, I) is N(b, A A^T): expected log-densities {np.round(want, 6).tolist()}, "
        f"got {np.round(got, 6).tolist()}")
