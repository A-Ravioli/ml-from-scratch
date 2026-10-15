"""Checks for exercise.py (the multivariate gaussian).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import math

import numpy as np
import pytest
from scipy import stats

# the lesson's worked example
MU2 = np.array([0.0, 1.0])
S2 = np.array([[4.0, 2.0], [2.0, 3.0]])

# a 4-D Gaussian whose Cholesky factor is far from symmetric (so L vs L.T bugs show)
MU4 = np.array([1.0, -2.0, 0.5, 3.0])
_A = np.array([[2.0, 0.0, 0.0, 0.0], [1.5, 1.0, 0.0, 0.0], [-1.0, 0.8, 0.7, 0.0], [0.5, -0.6, 1.2, 0.9]])
S4 = _A @ _A.T


def _mvn(nb, mu=MU4, S=S4):
    return nb.MultivariateNormal(mu, S)


# ---------------------------------------------------------------- stage 1: construction and sampling


def test_stage1_attributes(nb):
    mu, S = MU4.copy(), S4.copy()
    g = nb.MultivariateNormal(mu, S)
    assert g.dim == 4, f"dim should be 4, got {g.dim}"
    assert np.allclose(g.mean, MU4) and np.allclose(g.cov, S4), "store mean and cov as given"
    mu[0], S[0, 0] = 99.0, 99.0
    assert g.mean[0] == 1.0 and g.cov[0, 0] == S4[0, 0], (
        "changing the caller's arrays changed your distribution: store copies (np.array(..., dtype=float))")
    L = np.asarray(g.chol)
    assert np.allclose(L, np.tril(L)) and np.allclose(L @ L.T, S4), (
        "chol should be the LOWER Cholesky factor with chol @ chol.T == cov (use mlfs.linalg.cholesky)")


def test_stage1_validates(nb):
    with pytest.raises(np.linalg.LinAlgError):
        nb.MultivariateNormal([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
        pytest.fail("[[1, 2], [2, 1]] is not positive definite: cholesky should raise LinAlgError")
    with pytest.raises(ValueError):
        nb.MultivariateNormal([0.0, 0.0, 0.0], np.eye(2))
        pytest.fail("a 3-vector mean with a 2x2 covariance should raise ValueError")


def test_stage1_sample_shape_and_reproducibility(nb):
    g = _mvn(nb)
    x = np.asarray(g.sample(5, rng=0))
    assert x.shape == (5, 4), f"sample(5) should return (5, 4), got {x.shape}"
    assert np.asarray(g.sample(rng=0)).shape == (1, 4), "sample() with the default n=1 should return (1, 4)"
    assert np.array_equal(x, np.asarray(g.sample(5, rng=0))), "the same seed must give the same draws"
    r = np.random.default_rng(1)
    assert not np.array_equal(np.asarray(g.sample(5, rng=r)), np.asarray(g.sample(5, rng=r))), (
        "two calls with the same Generator should give different draws")


def test_stage1_sample_moments(nb):
    g = _mvn(nb)
    x = np.asarray(g.sample(200_000, rng=2))
    m, C = x.mean(axis=0), np.cov(x, rowvar=False)
    assert np.abs(m - MU4).max() < 0.03, (
        f"the sample mean of 200,000 draws should be within 0.03 of {MU4.tolist()}, got {np.round(m, 3).tolist()}")
    err = np.abs(C - S4).max()
    if err > 0.08:
        hint = ""
        if np.abs(C - _A.T @ _A).max() < 0.08:
            hint = " your draws have covariance L^T L: for a batch Z (n, d) use Z @ L.T, not Z @ L."
        elif np.abs(C - np.diag(np.diag(S4))).max() < 0.08:
            hint = " your coordinates are uncorrelated: you scaled by the standard deviations only."
        raise AssertionError(
            f"the sample covariance should match cov to within 0.08, but differs by {err:.3f}.{hint} "
            f"expected\n{np.round(S4, 2)}\ngot\n{np.round(C, 2)}")


# ---------------------------------------------------------------- stage 2: densities


def test_stage2_worked_example(nb):
    g = nb.MultivariateNormal(MU2, S2)
    got = g.logpdf(np.array([1.0, 2.0]))
    want = -0.5 * (3 / 8 + math.log(8) + 2 * math.log(2 * math.pi))
    assert isinstance(got, float), f"logpdf of a single point (d,) should return a Python float, got {type(got).__name__}"
    assert abs(got - want) < 1e-12, (
        f"the lesson's worked example: the quadratic form is 3/8 and det S = 8, so logpdf(1, 2) = "
        f"-1/2 (3/8 + ln 8 + 2 ln 2π) = {want:.6f}; got {got:.6f}. "
        + ("you're off by log det S: log det S = 2 sum(log diag L), not sum(log diag L)."
           if abs(got - (want + 0.5 * math.log(8) - 0.25 * math.log(8))) < 1e-9 else ""))
    assert abs(g.pdf(np.array([1.0, 2.0])) - math.exp(want)) < 1e-14, "pdf should be exp(logpdf)"


def test_stage2_matches_scipy_batch(nb):
    g = _mvn(nb)
    X = np.random.default_rng(3).normal(size=(50, 4)) * 2 + MU4
    got = np.asarray(g.logpdf(X))
    want = stats.multivariate_normal(MU4, S4).logpdf(X)
    assert got.shape == (50,), f"logpdf of (50, 4) points should return (50,), got {got.shape}"
    assert np.allclose(got, want, atol=1e-10), (
        f"logpdf disagrees with scipy by up to {np.abs(got - want).max():.3g}. solve L z = x - mu with a "
        "triangular solve; the quadratic form is |z|^2.")
    assert np.allclose(np.asarray(g.pdf(X)), np.exp(want)), "pdf should match exp(scipy's logpdf)"


def test_stage2_one_dimensional_and_far_away(nb):
    g = nb.MultivariateNormal([2.0], [[9.0]])
    got = g.logpdf(np.array([5.0]))
    assert abs(got - stats.norm(2.0, 3.0).logpdf(5.0)) < 1e-12, "a 1-D MultivariateNormal is N(2, 9): sd 3"
    far = g.logpdf(np.array([2.0 + 3.0 * 60]))
    assert np.isfinite(far) and abs(far - stats.norm(2.0, 3.0).logpdf(182.0)) < 1e-9, (
        f"60 standard deviations out, logpdf should be about -1802, not -inf; got {far}. compute the log "
        "density directly, don't take log(pdf).")


def test_stage2_entropy(nb):
    for mu, S in ((MU2, S2), (MU4, S4)):
        got = nb.MultivariateNormal(mu, S).entropy()
        want = stats.multivariate_normal(mu, S).entropy()
        assert abs(got - want) < 1e-10, f"entropy should be d/2 (1 + ln 2π) + 1/2 ln det S = {want:.6f}, got {got:.6f}"


def test_stage2_pdf_integrates_to_one(nb):
    g = nb.MultivariateNormal(MU2, S2)
    t = np.linspace(-12, 12, 241)
    xx, yy = np.meshgrid(t, t + 1.0)
    p = np.asarray(g.pdf(np.column_stack([xx.ravel(), yy.ravel()])))
    total = p.sum() * (t[1] - t[0]) ** 2
    assert abs(total - 1) < 1e-3, f"the density should integrate to 1 over the plane; a grid sum gives {total:.4f}"


# ---------------------------------------------------------------- stage 3: marginals


def test_stage3_marginal_is_selection(nb):
    g = _mvn(nb)
    m = g.marginal([2, 0])
    assert m.dim == 2, f"marginal([2, 0]) is 2-dimensional, got dim {m.dim}"
    assert np.allclose(m.mean, MU4[[2, 0]]), (
        f"marginal([2, 0]) should have mean (mu_2, mu_0) = {MU4[[2, 0]].tolist()} IN THAT ORDER, got {np.asarray(m.mean).tolist()}")
    assert np.allclose(m.cov, S4[np.ix_([2, 0], [2, 0])]), (
        "the marginal covariance is the submatrix cov[idx][:, idx] (rows AND columns), keeping idx's order")
    one = g.marginal([3])
    assert one.dim == 1 and np.allclose(one.cov, [[S4[3, 3]]]), "marginal([3]) is a 1-D Gaussian with variance cov[3, 3]"


def test_stage3_marginal_matches_samples(nb):
    g = _mvn(nb)
    m = g.marginal([1, 3])
    x = np.asarray(g.sample(100_000, rng=4))[:, [1, 3]]
    assert np.allclose(x.mean(axis=0), m.mean, atol=0.03) and np.allclose(np.cov(x, rowvar=False), m.cov, atol=0.08), (
        "dropping coordinates from joint samples gives samples of the marginal: their moments should agree")


# ---------------------------------------------------------------- stage 4: conditionals


def test_stage4_worked_example(nb):
    c = nb.MultivariateNormal(MU2, S2).condition([0], [2.0])
    assert c.dim == 1, f"conditioning a 2-D Gaussian on one coordinate leaves a 1-D one, got dim {c.dim}"
    assert np.allclose(c.mean, [2.0]) and np.allclose(c.cov, [[2.0]]), (
        f"given x0 = 2: mean = 1 + (2/4)(2 - 0) = 2 and variance = 3 - 2^2/4 = 2. got mean "
        f"{np.asarray(c.mean).tolist()}, cov {np.asarray(c.cov).tolist()}")


def test_stage4_matches_formula_and_keeps_order(nb):
    g = _mvn(nb)
    a, vals = [3, 1], np.array([2.5, -1.0])
    c = g.condition(a, vals)
    b = [0, 2]
    Saa, Sba, Sbb = S4[np.ix_(a, a)], S4[np.ix_(b, a)], S4[np.ix_(b, b)]
    mean = MU4[b] + Sba @ np.linalg.solve(Saa, vals - MU4[a])
    cov = Sbb - Sba @ np.linalg.solve(Saa, Sba.T)
    assert c.dim == 2, f"conditioning 4 coordinates on 2 leaves 2, got dim {c.dim}"
    assert np.allclose(c.mean, mean) and np.allclose(c.cov, cov), (
        f"condition([3, 1], ...) should describe the remaining coordinates (0, 2) in their ORIGINAL order. "
        f"expected mean {np.round(mean, 4).tolist()} and cov {np.round(cov, 4).tolist()}, got "
        f"{np.round(np.asarray(c.mean), 4).tolist()} and {np.round(np.asarray(c.cov), 4).tolist()}. "
        "note values[k] belongs to coordinate idx[k].")
    C = np.asarray(c.cov)
    assert np.array_equal(C, C.T), (
        f"the conditional covariance should be EXACTLY symmetric, but C - C.T has entries up to "
        f"{np.abs(C - C.T).max():.1e}. rounding in S_bb - S_ba S_aa^-1 S_ab breaks symmetry in the last bits: "
        "return (C + C.T) / 2.")


def test_stage4_exactly_symmetric_in_six_dimensions(nb):
    A = np.random.default_rng(0).normal(size=(6, 6))
    g = nb.MultivariateNormal(np.zeros(6), A @ A.T + 0.5 * np.eye(6))
    C = np.asarray(g.condition([4, 1], [0.3, -0.2]).cov)
    assert np.array_equal(C, C.T), (
        f"the conditional covariance (4 x 4 here) should be EXACTLY symmetric, but C - C.T has entries up to "
        f"{np.abs(C - C.T).max():.1e}. rounding in S_bb - S_ba S_aa^-1 S_ab makes the two triangles differ in the "
        "last bits, and later Cholesky factorizations of it can fail: return (C + C.T) / 2.")


def test_stage4_conditioning_is_slicing(nb):
    """p(x_b | x_a = v) = p(x_b, v) / p_a(v): the joint density along a slice, renormalized."""
    g = _mvn(nb)
    a, v = [1], np.array([-1.0])
    c = g.condition(a, v)
    rng = np.random.default_rng(5)
    xb = rng.normal(size=(20, 3)) + np.asarray(c.mean)
    full = np.empty((20, 4))
    full[:, [0, 2, 3]] = xb
    full[:, 1] = v[0]
    lhs = np.asarray(c.logpdf(xb))
    rhs = np.asarray(g.logpdf(full)) - g.marginal(a).logpdf(v)
    assert np.allclose(lhs, rhs, atol=1e-9), (
        f"log p(x_b | x_a) should equal log p(x_b, x_a) - log p(x_a) at every point; they differ by up to "
        f"{np.abs(lhs - rhs).max():.3g}.")


def test_stage4_conditional_variance_ignores_the_value(nb):
    g = _mvn(nb)
    c1, c2 = g.condition([0, 2], [0.0, 0.0]), g.condition([0, 2], [10.0, -7.0])
    assert np.allclose(c1.cov, c2.cov), "the conditional covariance doesn't depend on the observed values"
    assert not np.allclose(c1.mean, c2.mean), "but the conditional mean does"


def test_stage4_all_observed_raises(nb):
    with pytest.raises(ValueError):
        nb.MultivariateNormal(MU2, S2).condition([0, 1], [0.0, 0.0])
        pytest.fail("conditioning on every coordinate leaves nothing: raise ValueError")
