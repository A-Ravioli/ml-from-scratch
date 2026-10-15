"""Checks for exercise.py (exponential families & Fisher information).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.

Stages 3-5 build their families here, from your ExponentialFamily class, so they don't depend
on your stage-2 constructors. Densities are checked against scipy.stats; moments and Fisher
information against closed forms and Monte Carlo (fixed seeds, ~5 standard errors).
"""
import math

import numpy as np
from scipy import stats

from mlfs.testing import bounded


def _counted(A, limit=100_000):
    """wrap a log-partition function so that a runaway loop fails instead of hanging."""
    calls = [0]

    def wrapped(e):
        calls[0] += 1
        if calls[0] > limit:
            raise AssertionError(f"A was evaluated more than {limit:,} times: a loop in fit never ends. cap the Newton "
                                 "iterations at max_iters and the step halvings at 50.")
        return A(e)
    return wrapped


def _fit(fam, x, **kwargs):
    return bounded(fam.fit, x, what="fit",
                   hint="at most max_iters Newton steps, and at most 50 halvings per step.", **kwargs)


def _gauss(nb, eta0=(0.0, -0.5)):
    return nb.ExponentialFamily(
        lambda x: np.stack([np.asarray(x, float), np.asarray(x, float) ** 2], axis=1),
        _counted(lambda e: -e[0] ** 2 / (4 * e[1]) - 0.5 * np.log(-2 * e[1])),
        lambda x: np.full(len(x), -0.5 * math.log(2 * math.pi)),
        list(eta0))


def _bern(nb):
    return nb.ExponentialFamily(lambda x: np.asarray(x, float)[:, None], _counted(lambda e: np.log1p(np.exp(e[0]))),
                                lambda x: np.zeros(len(x)), [0.0])


def _pois(nb):
    return nb.ExponentialFamily(lambda x: np.asarray(x, float)[:, None], _counted(lambda e: np.exp(e[0])),
                                lambda x: -np.array([math.lgamma(v + 1) for v in x]), [0.0])


def _expo(nb):
    return nb.ExponentialFamily(lambda x: np.asarray(x, float)[:, None], _counted(lambda e: -np.log(-e[0])),
                                lambda x: np.zeros(len(x)), [-1.0])


def _nat(mu, s2):
    return np.array([mu / s2, -1 / (2 * s2)])


# ---------------------------------------------------------------- stage 1: the canonical form


def test_stage1_attributes(nb):
    T, A, h = (lambda x: x[:, None]), (lambda e: e[0] ** 2), (lambda x: np.zeros(len(x)))
    fam = nb.ExponentialFamily(T, A, h, [0.5])
    assert fam.T is T and fam.A is A and fam.log_h is h, "store the functions as attributes T, A and log_h"
    assert isinstance(fam.eta0, np.ndarray) and fam.eta0.dtype == float and fam.eta0.tolist() == [0.5], (
        "store eta0 as a float array of shape (k,)")
    assert fam.dim == 1 and _gauss(nb).dim == 2, "dim is the number of natural parameters, len(eta0)"


def test_stage1_log_prob_gaussian(nb):
    x = np.array([-1.0, 0.0, 2.5])
    for mu, s2 in ((0.0, 1.0), (1.5, 4.0), (-2.0, 0.25)):
        got = np.asarray(_gauss(nb).log_prob(x, _nat(mu, s2)))
        want = stats.norm(mu, math.sqrt(s2)).logpdf(x)
        assert got.shape == (3,), f"log_prob returns one value per data point: shape (3,), got {got.shape}"
        assert np.allclose(got, want, atol=1e-12), (
            f"N({mu}, {s2}) in natural parameters {np.round(_nat(mu, s2), 4).tolist()}: log densities should be "
            f"{np.round(want, 5).tolist()}, got {np.round(got, 5).tolist()}. log p = log_h(x) + T(x) @ eta - A(eta).")


def test_stage1_log_prob_discrete(nb):
    k = np.array([0, 1, 4, 7])
    got = np.asarray(_pois(nb).log_prob(k, [math.log(3.0)]))
    assert np.allclose(got, stats.poisson(3.0).logpmf(k), atol=1e-12), "Poisson(3) as an exponential family, eta = log 3"
    got = np.asarray(_bern(nb).log_prob(np.array([0, 1]), [math.log(0.2 / 0.8)]))
    assert np.allclose(got, np.log([0.8, 0.2]), atol=1e-12), "Bernoulli(0.2), eta = logit(0.2)"


# ---------------------------------------------------------------- stage 2: four families


def test_stage2_match_scipy(nb):
    cases = [
        ("bernoulli_family", np.array([0, 1, 1]), [math.log(0.3 / 0.7)], stats.bernoulli(0.3).logpmf),
        ("poisson_family", np.array([0, 2, 5, 11]), [math.log(4.5)], stats.poisson(4.5).logpmf),
        ("exponential_family", np.array([0.1, 1.0, 3.3]), [-2.0], stats.expon(scale=0.5).logpdf),
        ("gaussian_family", np.array([-3.0, 0.4, 2.0]), _nat(1.0, 2.0), stats.norm(1.0, math.sqrt(2.0)).logpdf),
    ]
    for name, x, eta, ref in cases:
        fam = getattr(nb, name)()
        got = np.asarray(fam.log_prob(x, eta))
        assert np.allclose(got, ref(x), atol=1e-10), (
            f"{name}: at eta = {np.round(eta, 4).tolist()} the log densities should be {np.round(ref(x), 5).tolist()}, "
            f"got {np.round(got, 5).tolist()}")


def test_stage2_starting_points(nb):
    for name, want in (("bernoulli_family", [0.0]), ("poisson_family", [0.0]),
                       ("exponential_family", [-1.0]), ("gaussian_family", [0.0, -0.5])):
        fam = getattr(nb, name)()
        assert np.allclose(fam.eta0, want), f"{name}: eta0 should be {want}, got {fam.eta0}"
        assert np.isfinite(fam.A(np.asarray(fam.eta0, float))), f"{name}: A(eta0) must be finite (a valid parameter)"


def test_stage2_normalized(nb):
    x = np.linspace(-12, 12, 48_001)
    eta = _nat(0.7, 1.7)
    dens = np.exp(np.asarray(nb.gaussian_family().log_prob(x, eta)))
    assert abs(np.sum(dens) * (x[1] - x[0]) - 1) < 1e-6, (
        "the Gaussian family's density must integrate to 1: A(eta) is the log of the normalizing constant")
    k = np.arange(0, 80)
    assert abs(np.exp(np.asarray(nb.poisson_family().log_prob(k, [math.log(6.0)]))).sum() - 1) < 1e-10, (
        "the Poisson family's probabilities must sum to 1")


# ---------------------------------------------------------------- stage 3: derivatives of A


def test_stage3_mean_params(nb):
    m = np.asarray(_bern(nb).mean_params([0.4]))
    assert m.shape == (1,) and abs(m[0] - 1 / (1 + math.exp(-0.4))) < 1e-12, (
        f"for the Bernoulli, grad A(eta) = sigmoid(eta) = P(x = 1); at eta = 0.4 that's {1 / (1 + math.exp(-0.4)):.6f}, got {m}")
    m = np.asarray(_gauss(nb).mean_params(_nat(1.5, 2.0)))
    assert np.allclose(m, [1.5, 1.5**2 + 2.0], atol=1e-10), (
        f"for N(1.5, 2), E[T(X)] = (E X, E X^2) = (1.5, 4.25), got {np.round(m, 6).tolist()}")


def test_stage3_fisher_closed_forms(nb):
    for eta in (-1.0, 0.0, 2.0):
        p = 1 / (1 + math.exp(-eta))
        I = np.asarray(_bern(nb).fisher_information([eta]))
        assert I.shape == (1, 1) and abs(I[0, 0] - p * (1 - p)) < 1e-8, (
            f"Bernoulli Fisher information in the natural parameter is Var(X) = p(1 - p) = {p * (1 - p):.6f}, got {I}")
    I = np.asarray(_pois(nb).fisher_information([math.log(3.0)]))
    assert abs(I[0, 0] - 3.0) < 1e-7, f"Poisson: I(eta) = Var(X) = lambda = 3, got {I}"
    mu, s2 = 1.0, 2.0
    want = np.array([[s2, 2 * mu * s2], [2 * mu * s2, 4 * mu**2 * s2 + 2 * s2**2]])  # Cov(X, X^2)
    I = np.asarray(_gauss(nb).fisher_information(_nat(mu, s2)))
    assert I.shape == (2, 2) and np.allclose(I, want, atol=1e-6), (
        f"Gaussian: I(eta) = Cov(T(X)) = Cov((X, X^2)) = {want.tolist()}, got {np.round(I, 5).tolist()}")
    assert np.array_equal(I, I.T), "the Fisher information must be exactly symmetric: symmetrize (J + J^T) / 2"


def test_stage3_fisher_is_covariance_of_T(nb):
    fam = _gauss(nb)
    eta = _nat(-0.5, 0.8)
    x = np.random.default_rng(0).normal(-0.5, math.sqrt(0.8), 200_000)
    T = np.stack([x, x**2], axis=1)
    assert np.allclose(np.asarray(fam.fisher_information(eta)), np.cov(T.T), rtol=0.03), (
        "the Hessian of A should equal the covariance of the sufficient statistics (Monte Carlo check)")
    assert np.allclose(np.asarray(fam.mean_params(eta)), T.mean(axis=0), atol=0.01), "grad A should equal E[T(X)]"


# ---------------------------------------------------------------- stage 4: maximum likelihood


def test_stage4_matches_closed_forms(nb):
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 1.5, 500)
    eta = np.asarray(_fit(_gauss(nb), x))
    mu, s2 = x.mean(), x.var()
    assert eta.shape == (2,) and np.allclose(eta, _nat(mu, s2), atol=1e-8), (
        f"the Gaussian MLE in natural parameters is (mu/s2, -1/(2 s2)) with mu = x̄ and s2 = the ddof-0 variance: "
        f"{np.round(_nat(mu, s2), 6).tolist()}, got {np.round(eta, 6).tolist()}")
    b = rng.integers(0, 2, 300)
    assert abs(_fit(_bern(nb), b)[0] - math.log(b.mean() / (1 - b.mean()))) < 1e-8, "the Bernoulli MLE is logit(mean)"
    k = rng.poisson(0.3, 400)
    assert abs(_fit(_pois(nb), k)[0] - math.log(k.mean())) < 1e-8, "the Poisson MLE is log(mean)"
    e = rng.exponential(5.0, 300)
    assert abs(_fit(_expo(nb), e)[0] + 1 / e.mean()) < 1e-8, "the exponential MLE is eta = -1/mean (rate 1/mean)"


def test_stage4_far_start_needs_damping(nb):
    x = np.random.default_rng(2).normal(30.0, 0.5, 200)  # far from the standard-normal start
    eta = np.asarray(_fit(_gauss(nb), x))
    assert np.allclose(eta, _nat(x.mean(), x.var()), rtol=1e-6), (
        "starting from N(0, 1) with data near 30, a full Newton step leaves the valid region (eta2 >= 0, where "
        "A is undefined). halve the step until the average log-likelihood is finite and doesn't decrease.")
    e = np.random.default_rng(3).exponential(0.01, 100)
    assert abs(_fit(_expo(nb), e, eta0=[-0.001])[0] + 1 / e.mean()) < 1e-6 * (1 / e.mean()), (
        "fit(x, eta0=...) should start from the given eta0")


def test_stage4_never_accepts_a_worse_step(nb):
    x = np.array([0.0, 1.0] * 50)  # the MLE is eta = 0
    eta = np.asarray(_fit(_bern(nb), x, eta0=[-10.0]))
    assert abs(eta[0]) < 1e-6, (
        f"from eta0 = -10 the first Newton step is about +11,000: far past the optimum, where A overflows or the "
        f"log-likelihood is far LOWER. halve the step until the average log-likelihood is finite AND at least as large "
        f"as before; then Newton converges to 0. got {eta}")


def test_stage4_iteration_count(nb):
    oracle, e = [], 0.0
    for _ in range(3):  # undamped Newton for all-ones Bernoulli data: every full step increases the likelihood
        e = e + 1 / (1 / (1 + math.exp(-e)))
        oracle.append(e)
    for k in (1, 3):
        got = np.asarray(_fit(_bern(nb), np.ones(20), max_iters=k))
        assert abs(got[0] - oracle[k - 1]) < 1e-5, (
            f"with max_iters={k}, fit should take exactly {k} Newton step(s) from eta0 = 0: eta = {oracle[k - 1]:.5f}, "
            f"got {got}")


def test_stage4_respects_max_iters(nb):
    calls = {"n": 0}

    def A(e):
        calls["n"] += 1
        if calls["n"] > 20_000:
            raise AssertionError("fit evaluated A more than 20,000 times: it must stop after max_iters iterations")
        return np.log1p(np.exp(e[0]))

    fam = nb.ExponentialFamily(lambda x: np.asarray(x, float)[:, None], A, lambda x: np.zeros(len(x)), [0.0])
    eta = np.asarray(_fit(fam, np.ones(50), max_iters=25))  # all ones: the MLE is at eta = +infinity
    assert eta.shape == (1,) and eta[0] > 5, (
        "for all-ones Bernoulli data the likelihood keeps increasing as eta grows: after 25 Newton steps eta should "
        f"be large and positive, got {eta}")
    calls["n"] = 0
    _fit(fam, np.array([0.0, 1.0, 1.0]), max_iters=10, tol=-1.0)  # a negative tol is never met: must still stop
    start = np.array([0.37])
    got = np.asarray(_fit(fam, np.array([0.0, 1.0, 1.0]), eta0=start, max_iters=0))
    assert np.allclose(got, start), f"with max_iters=0, fit should return its starting point eta0 = {start}, got {got}"


# ---------------------------------------------------------------- stage 5: score and Cramér-Rao


def test_stage5_score(nb):
    fam = _pois(nb)
    s = np.asarray(fam.score(np.array([0, 2, 5]), [math.log(2.0)]))
    assert s.shape == (3, 1) and np.allclose(s[:, 0], [-2.0, 0.0, 3.0]), (
        f"Poisson score at lambda = 2: T(x) - grad A = x - 2, so (-2, 0, 3) for x = (0, 2, 5); got {s.ravel()}")
    eta = _nat(1.0, 0.5)
    x = np.random.default_rng(4).normal(1.0, math.sqrt(0.5), 100_000)
    S = np.asarray(_gauss(nb).score(x, eta))
    assert S.shape == (100_000, 2), f"score returns (n, k), got {S.shape}"
    assert np.all(np.abs(S.mean(axis=0)) < 5 * S.std(axis=0) / math.sqrt(100_000)), (
        "the score has mean zero at the true parameter")
    assert np.allclose(np.cov(S.T), np.asarray(_gauss(nb).fisher_information(eta)), rtol=0.04), (
        "the covariance of the score is the Fisher information")


def test_stage5_cramer_rao(nb):
    p = 0.2
    eta = math.log(p / (1 - p))
    crb = np.asarray(_bern(nb).cramer_rao_bound([eta], 50))
    assert crb.shape == (1, 1) and abs(crb[0, 0] - 1 / (50 * p * (1 - p))) < 1e-7, (
        f"for n = 50 Bernoulli(0.2) draws, no unbiased estimator of the log-odds has variance below "
        f"1 / (n p (1 - p)) = {1 / (50 * p * (1 - p)):.5f}; got {crb}. the bound is inverse(I) / n, not I / n.")
    eta2 = _nat(0.5, 2.0)
    crb = np.asarray(_gauss(nb).cramer_rao_bound(eta2, 10))
    I = np.array([[2.0, 2.0], [2.0, 4 * 0.25 * 2 + 8.0]])
    assert np.allclose(crb, np.linalg.inv(I) / 10, atol=1e-7), (
        f"2-D: the bound is the matrix inverse of the Fisher information, divided by n: expected "
        f"{np.round(np.linalg.inv(I) / 10, 5).tolist()}, got {np.round(crb, 5).tolist()}")
