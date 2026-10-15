"""Checks for exercise.py (variational inference).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Oracles: the ELBO of
q against itself (known exactly for any draws), the exact evidence of the one-component mixture
(where CAVI is exact), finite differences of a deterministic ELBO estimate for the gradients, and
the known optimum of mean-field and full-covariance VI on a Gaussian target (the mean-field
variances are 1 / diag(precision)). Stochastic fits are checked with tolerances wide enough for
Adam's leftover noise. Every loop runs under mlfs.testing.bounded.
"""
import math

import numpy as np
from scipy.stats import multivariate_normal

from mlfs.testing import bounded

COV = np.array([[1.0, 0.8], [0.8, 1.0]])
PREC = np.linalg.inv(COV)
MU = np.array([1.0, -0.5])


def _gauss_lp(z):
    d = z - MU
    return -0.5 * (PREC[0, 0] * d[0] * d[0] + 2 * PREC[0, 1] * d[0] * d[1] + PREC[1, 1] * d[1] * d[1])


def _gauss_grad(z):
    return -PREC @ (np.asarray(z, float) - MU)


def _quartic_lp(z):
    return -0.25 * np.sum(z**4) - 0.5 * z[0] * z[1] + np.sum(z)


def _quartic_grad(z):
    return -z**3 - 0.5 * z[::-1] + 1.0


def _elbo(lp, mean, log_std, eps):
    z = mean + np.exp(log_std) * eps
    return np.mean([lp(s) for s in z]) + np.sum(log_std) + 0.5 * len(mean) * (1 + math.log(2 * math.pi))


# ---------------------------------------------------------------- stage 1: the ELBO


def test_stage1_q_against_itself(nb):
    rng = np.random.default_rng(0)
    mean, log_std = np.array([0.3, -1.0, 2.0]), np.array([0.1, -0.5, 0.4])
    std = np.exp(log_std)
    lq = lambda z: float(np.sum(-0.5 * ((z - mean) / std) ** 2 - np.log(std) - 0.5 * math.log(2 * math.pi)))
    eps = rng.standard_normal((50, 3))
    got = nb.elbo_diag(lq, mean, log_std, eps)
    want = 0.5 * (3 - np.mean(np.sum(eps**2, axis=1)))
    assert np.isclose(got, want, atol=1e-10), (
        f"with the target equal to q itself, the ELBO estimate is mean_s log q(z_s) + H(q) = (d - mean |eps|^2) / 2 "
        f"= {want:.6f} exactly for these draws (its expectation is 0 = -KL(q || q)); got {got}. the entropy of a "
        "diagonal Gaussian is sum(log_std) + d/2 (1 + log 2 pi)")
    assert isinstance(got, float), "return a Python float"


def test_stage1_oracle_and_constant(nb):
    rng = np.random.default_rng(1)
    mean, log_std = np.array([0.5, -0.2]), np.array([-0.3, 0.2])
    eps = rng.standard_normal((20, 2))
    got = nb.elbo_diag(_quartic_lp, mean, log_std, eps)
    assert np.isclose(got, _elbo(_quartic_lp, mean, log_std, eps)), (
        "elbo_diag: z_s = mean + exp(log_std) * eps_s, then average log_prob(z_s) over the draws and add the entropy")
    shifted = nb.elbo_diag(lambda z: _quartic_lp(z) + 7.0, mean, log_std, eps)
    assert np.isclose(shifted - got, 7.0), "an unnormalized target shifts the ELBO by its log normalizer, nothing else"


# ---------------------------------------------------------------- stage 2: CAVI for a Gaussian mixture


def _oracle_elbo(x, m, s2, phi, sig2):
    n, K = phi.shape
    t1 = np.sum(-0.5 * np.log(2 * np.pi * sig2) - (s2 + m**2) / (2 * sig2))
    t3 = np.sum(phi * (-0.5 * np.log(2 * np.pi) - 0.5 * (x[:, None] ** 2 - 2 * x[:, None] * m + s2 + m**2)))
    h = np.sum(0.5 * np.log(2 * np.pi * np.e * s2)) - np.sum(np.where(phi > 0, phi * np.log(np.where(phi > 0, phi, 1)), 0))
    return t1 - n * math.log(K) + t3 + h


def test_stage2_worked_example(nb):
    out = bounded(nb.cavi_gmm, np.array([1.0, 2.0, 3.0]), 1, prior_var=1.0, rng=0, what="cavi_gmm")
    want = -2.5 - 0.5 * math.log(4) - 1.5 * math.log(2 * math.pi)
    assert np.allclose(out["m"], [1.5]) and np.allclose(out["s2"], [0.25]) and np.isclose(out["elbo"][-1], want), (
        f"lesson worked example: x = (1, 2, 3), one component, prior_var = 1: s2 = 1/(1 + 3) = 0.25, m = 0.25 * 6 = "
        f"1.5, and the ELBO is the log evidence -5/2 - log(4)/2 - 3/2 log(2 pi) = {want:.4f}; got m = {out['m']}, "
        f"s2 = {out['s2']}, ELBO = {out['elbo'][-1]:.4f}")


def test_stage2_one_component_is_exact(nb):
    x = np.random.default_rng(2).normal(1.5, 1.0, size=30)
    out = bounded(nb.cavi_gmm, x, 1, prior_var=4.0, rng=0, what="cavi_gmm")
    prec = 1 / 4.0 + 30
    assert np.allclose(out["m"], [x.sum() / prec]) and np.allclose(out["s2"], [1 / prec]), (
        f"with one component, q(mu) is the exact conjugate posterior: mean sum(x) / (1/prior_var + n) = "
        f"{x.sum() / prec:.5f}, variance 1 / (1/prior_var + n) = {1 / prec:.5f}; got m = {out['m']}, s2 = {out['s2']}")
    evidence = multivariate_normal(np.zeros(30), np.eye(30) + 4.0 * np.ones((30, 30))).logpdf(x)
    assert np.isclose(out["elbo"][-1], evidence, atol=1e-8), (
        f"and since q is the exact posterior, the ELBO equals the log evidence log p(x) = {evidence:.6f}; got "
        f"{out['elbo'][-1]:.6f}. check the constants in the ELBO (the prior's -1/2 log(2 pi prior_var), the "
        "likelihood's -1/2 log 2 pi, the entropy 1/2 log(2 pi e s2))")


def test_stage2_mixture(nb):
    rng = np.random.default_rng(3)
    true = np.array([-4.0, 0.0, 5.0])
    x = np.concatenate([rng.normal(mu, 1.0, 100) for mu in true])
    sample_means = x.reshape(3, 100).mean(axis=1)
    runs = [bounded(nb.cavi_gmm, x, 3, rng=s, what="cavi_gmm") for s in range(5)]
    for out in runs:
        e = np.asarray(out["elbo"])
        assert np.all(np.diff(e) >= -1e-8), (
            f"CAVI maximizes the ELBO one coordinate block at a time, so the ELBO never decreases; got a drop of "
            f"{np.min(np.diff(e)):.3g}")
        assert out["phi"].shape == (300, 3) and np.allclose(out["phi"].sum(1), 1), "phi is (n, K) with rows summing to 1"
        assert np.isclose(e[-1], _oracle_elbo(x, out["m"], out["s2"], out["phi"], 10.0), atol=1e-6), (
            "the last ELBO value should be the ELBO of the returned (m, s2, phi)")
        assert len(e) <= 100, "at most max_iters iterations"
    best = max(runs, key=lambda o: o["elbo"][-1])
    assert np.allclose(np.sort(best["m"]), sample_means, atol=0.2), (
        f"three well-separated clusters around -4, 0, 5 (100 points each, sample means "
        f"{np.round(sample_means, 2).tolist()}): the best of 5 restarts should put the means there; got "
        f"{np.round(np.sort(best['m']), 2).tolist()}")
    assert np.all(best["s2"] < 0.02), "each mean is estimated from ~100 points: s2 ~ 1/100"
    short = bounded(nb.cavi_gmm, x, 3, max_iters=2, rng=0, what="cavi_gmm")
    assert len(short["elbo"]) == 2, "max_iters=2 means exactly 2 iterations (one ELBO value each)"
    loose = bounded(nb.cavi_gmm, x, 3, tol=1e10, rng=0, what="cavi_gmm")
    assert len(loose["elbo"]) == 2, (
        "with an enormous tol the second iteration's ELBO change is already below it: stop after 2 iterations "
        "(stop when |elbo[-1] - elbo[-2]| < tol)")


def test_stage2_fixed_point(nb):
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.normal(0.0, 1.0, 300), rng.normal(3.0, 1.0, 60)])  # two overlapping, unequal clusters
    out = bounded(nb.cavi_gmm, x, 2, max_iters=2000, tol=1e-13, rng=1, what="cavi_gmm", budget=2_000_000)
    m, s2, phi = out["m"], out["s2"], out["phi"]
    nk = phi.sum(0)
    assert np.allclose(s2, 1 / (1 / 10.0 + nk), rtol=1e-6) and np.allclose(m, s2 * (phi * x[:, None]).sum(0), rtol=1e-6), (
        "at convergence q(mu_k) must be the CAVI update for the final phi: s2_k = 1 / (1/prior_var + sum_i phi_ik), "
        "m_k = s2_k sum_i phi_ik x_i")
    lg = x[:, None] * m - 0.5 * (s2 + m**2)
    want = np.exp(lg - lg.max(1, keepdims=True))
    want /= want.sum(1, keepdims=True)
    assert np.allclose(phi, want, atol=2e-5), (
        "and phi must be the CAVI update for the final q(mu): phi_ik proportional to exp(x_i m_k - (s2_k + m_k^2) / 2). "
        "the s2_k term (the uncertainty in mu_k: E[mu_k^2] = m_k^2 + s2_k) shifts the ambiguous points by ~3e-4 here")


# ---------------------------------------------------------------- stage 3: the reparameterization gradient


def test_stage3_matches_finite_differences(nb):
    rng = np.random.default_rng(4)
    mean, log_std = np.array([0.4, -0.3]), np.array([-0.2, 0.1])
    eps = rng.standard_normal((25, 2))
    gm, gs = nb.elbo_grad_diag(_quartic_grad, mean, log_std, eps)
    h = 1e-6
    fd_m = np.array([(_elbo(_quartic_lp, mean + h * e, log_std, eps) - _elbo(_quartic_lp, mean - h * e, log_std, eps))
                     / (2 * h) for e in np.eye(2)])
    fd_s = np.array([(_elbo(_quartic_lp, mean, log_std + h * e, eps) - _elbo(_quartic_lp, mean, log_std - h * e, eps))
                     / (2 * h) for e in np.eye(2)])
    assert np.allclose(gm, fd_m, atol=1e-6), (
        f"d ELBO / d mean = mean_s grad log p(z_s); finite differences say {np.round(fd_m, 5).tolist()}, got "
        f"{np.round(gm, 5).tolist()}")
    assert np.allclose(gs, fd_s, atol=1e-6), (
        f"d ELBO / d log_std = mean_s [grad log p(z_s) * eps_s * std] + 1 (the +1 is the entropy's derivative); "
        f"finite differences say {np.round(fd_s, 5).tolist()}, got {np.round(gs, 5).tolist()}")


# ---------------------------------------------------------------- stage 4: reparameterized VI


def test_stage4_mean_field_optimum(nb):
    q, hist = bounded(nb.reparam_vi, _gauss_lp, 2, grad_log_prob=_gauss_grad, lr=0.01, n_iters=2000, rng=5,
                      what="reparam_vi", budget=4_000_000)
    assert len(hist) == 2000, f"one ELBO estimate per iteration: 2000 values; got {len(hist)}"
    assert np.allclose(q.mean, MU, atol=0.12), f"the mean-field optimum has the target's mean {MU.tolist()}; got {q.mean}"
    sd = np.sqrt(np.diag(q.cov))
    want = np.sqrt(1 / np.diag(PREC))
    assert np.allclose(sd, want, rtol=0.15), (
        f"mean-field VI on a Gaussian with correlation 0.8 converges to standard deviations 1 / sqrt(diag(precision)) "
        f"= {np.round(want, 3).tolist()} (smaller than the true marginal sds of 1: VI underestimates variance); got "
        f"{np.round(sd, 3).tolist()}")
    assert np.allclose(q.cov, np.diag(np.diag(q.cov))), "the diagonal family has zero off-diagonal covariance"
    log_z = math.log(2 * math.pi) + 0.5 * math.log(np.linalg.det(COV))  # the target is unnormalized
    best = log_z - 0.5 * math.log(np.prod(np.diag(PREC)) * np.linalg.det(COV))  # log Z - KL(q* || p)
    late = np.mean(hist[-200:])
    assert late > np.mean(hist[:10]) + 0.5, "the ELBO estimates should increase over training"
    assert abs(late - best) < 0.15 and late < log_z, (
        f"at the mean-field optimum the ELBO is log Z - KL(q* || p) = {best:.3f} (log Z = {log_z:.3f}: the ELBO is a "
        f"LOWER bound). the average of your last 200 ELBO estimates is {late:.3f}")


def test_stage4_autodiff_default(nb):
    kw = dict(lr=0.05, n_iters=40, n_samples=4, rng=6)
    q1, h1 = bounded(nb.reparam_vi, _gauss_lp, 2, grad_log_prob=_gauss_grad, what="reparam_vi", **kw)
    q2, h2 = bounded(nb.reparam_vi, _gauss_lp, 2, what="reparam_vi", budget=3_000_000, **kw)
    assert np.allclose(h1, h2, atol=1e-9) and np.allclose(q1.mean, q2.mean, atol=1e-9), (
        "with grad_log_prob=None, reparam_vi must use mlfs.autodiff.grad on log_prob: with the same seed the run must "
        "match the one with the exact gradient")


# ---------------------------------------------------------------- stage 5: full covariance


def test_stage5_full_gradient(nb):
    rng = np.random.default_rng(7)
    mean = np.array([0.2, -0.4])
    L = np.array([[0.9, 0.0], [0.3, 0.7]])
    eps = rng.standard_normal((25, 2))

    def elbo_full(m, L_):
        z = m + eps @ L_.T
        return np.mean([_quartic_lp(s) for s in z]) + np.sum(np.log(np.diag(L_))) + (1 + math.log(2 * math.pi))

    gm, gL = nb.elbo_grad_full(_quartic_grad, mean, L, eps)
    h = 1e-6
    fd_m = np.array([(elbo_full(mean + h * e, L) - elbo_full(mean - h * e, L)) / (2 * h) for e in np.eye(2)])
    assert np.allclose(gm, fd_m, atol=1e-6), "d ELBO / d mean = mean_s grad log p(z_s)"
    for i, j in [(0, 0), (1, 0), (1, 1)]:
        E = np.zeros((2, 2))
        E[i, j] = h
        fd = (elbo_full(mean, L + E) - elbo_full(mean, L - E)) / (2 * h)
        assert np.isclose(gL[i, j], fd, atol=1e-6), (
            f"d ELBO / d L[{i},{j}]: finite differences say {fd:.6f}, got {gL[i, j]:.6f}. with z = mean + L eps, "
            "the gradient is mean_s grad log p(z_s) eps_s^T (lower triangle), plus 1 / L_ii on the diagonal (entropy)")
    assert np.isclose(gL[0, 1], 0.0), "the upper triangle of L isn't a parameter: its gradient is 0"


def test_stage5_full_covariance_fit(nb):
    q, hist = bounded(nb.reparam_vi, _gauss_lp, 2, grad_log_prob=_gauss_grad, family="full", lr=0.01, n_iters=2000,
                      rng=8, what="reparam_vi", budget=4_000_000)
    assert np.allclose(q.mean, MU, atol=0.12), f"mean {MU.tolist()}; got {q.mean}"
    corr = q.cov[0, 1] / np.sqrt(q.cov[0, 0] * q.cov[1, 1])
    assert np.allclose(np.diag(q.cov), 1.0, rtol=0.25) and abs(corr - 0.8) < 0.1, (
        f"a full-covariance Gaussian family contains the target, so VI should recover its covariance "
        f"{COV.tolist()} (variances 1, correlation 0.8; Adam's leftover noise allows some slack); got "
        f"{np.round(q.cov, 3).tolist()}, correlation {corr:.3f}")
    try:
        nb.reparam_vi(_gauss_lp, 2, grad_log_prob=_gauss_grad, family="banana", n_iters=1)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown family should raise ValueError")
