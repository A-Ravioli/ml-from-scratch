"""Checks for exercise.py (stability, margins & PAC-Bayes).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.

The binary KL and its inverse are checked against hand values, scipy root finding and their
defining properties. The PAC-Bayes bound is checked by hand and by a coverage simulation (with
KL = 0 it must cover the true risk in at least 1 - delta of repeated samples). The Gibbs risk
is checked against Monte Carlo over Gaussian weights, the KL against the full-covariance
formula, and the stability estimate against a predictor whose leave-one-out change is known
exactly. Loops that could run forever go through mlfs.testing.bounded.
"""
import math

import numpy as np
from scipy.optimize import brentq

from mlfs.testing import bounded


def _kl(q, p):
    t = 0.0
    if q > 0:
        t += q * math.log(q / p)
    if q < 1:
        t += (1 - q) * math.log((1 - q) / (1 - p))
    return t


def _klinv(q, c):
    if c <= 0:
        return q
    return brentq(lambda p: _kl(q, p) - c, q + 1e-15, 1 - 1e-15, xtol=1e-14)


# ---------------------------------------------------------------- stage 1: binary KL


def test_stage1_values(nb):
    got = nb.kl_bernoulli(0.1, 0.2)
    want = 0.1 * math.log(0.5) + 0.9 * math.log(0.9 / 0.8)
    assert np.isclose(got, want), f"kl(0.1 || 0.2) = 0.1 ln(0.1/0.2) + 0.9 ln(0.9/0.8) = {want:.6f}; got {got}"
    assert np.isclose(nb.kl_bernoulli(0.0, 0.3), -math.log(0.7)), "kl(0 || p) = -ln(1 - p): the 0 ln 0 term is 0"
    assert np.isclose(nb.kl_bernoulli(1.0, 0.3), -math.log(0.3)), "kl(1 || p) = -ln p"
    for q in (0.0, 0.37, 1.0):
        assert nb.kl_bernoulli(q, q) == 0.0 or abs(nb.kl_bernoulli(q, q)) < 1e-15, f"kl(q || q) = 0 (q = {q})"
    assert nb.kl_bernoulli(0.5, 0.0) == math.inf and nb.kl_bernoulli(0.5, 1.0) == math.inf, (
        "kl(q || p) is infinite when p = 0 or 1 but q isn't: return math.inf, without a ZeroDivisionError or a warning")


def test_stage1_properties(nb):
    from mlfs.stats import kl_divergence
    rng = np.random.default_rng(0)
    for _ in range(30):
        q, p = rng.random(2) * 0.98 + 0.01
        got = nb.kl_bernoulli(q, p)
        assert got >= 0 and np.isclose(got, kl_divergence(np.array([q, 1 - q]), np.array([p, 1 - p]))), (
            f"kl({q:.3f} || {p:.3f}) should equal the KL divergence of the two-outcome distributions; got {got}")
        assert got >= 2 * (q - p) ** 2 - 1e-12, "Pinsker's inequality kl(q || p) >= 2 (q - p)^2 must hold"
    assert nb.kl_bernoulli(0.1, 0.3) < nb.kl_bernoulli(0.1, 0.5) < nb.kl_bernoulli(0.1, 0.9), (
        "for p > q, kl(q || p) increases with p")


# ---------------------------------------------------------------- stage 2: the inverse


def test_stage2_matches_root_finding(nb):
    for q, c in [(0.05, 0.01), (0.2, 0.1), (0.0, 0.5), (0.5, 0.02), (0.01, 2.0), (0.9, 0.001)]:
        got = bounded(nb.kl_inverse, q, c, what="kl_inverse", budget=100_000,
                      hint="bisection on [q, 1] halves the bracket each step; stop when it's narrower than tol")
        want = _klinv(q, c)
        assert abs(got - want) < 1e-8, (
            f"kl_inverse({q}, {c}): the largest p >= q with kl(q || p) <= c is {want:.10f}; got {got}. bisect on [q, 1]: "
            "if kl(q || mid) <= c the answer is above mid")
        assert got >= q, "the inverse is at least q (kl(q || q) = 0 <= c)"
    assert np.isclose(nb.kl_inverse(0.0, 0.5), 1 - math.exp(-0.5), atol=1e-9), "q = 0: kl(0 || p) = -ln(1 - p), so p = 1 - e^-c"
    assert nb.kl_inverse(0.3, 0.0) == 0.3, "c = 0: nothing but p = q satisfies kl(q || p) <= 0"
    assert nb.kl_inverse(1.0, 0.1) == 1.0 and nb.kl_inverse(0.2, 50.0) == 1.0, "the answer can't exceed 1"


def test_stage2_tolerance_and_monotonicity(nb):
    rough = nb.kl_inverse(0.1, 0.05, tol=1e-3)
    exact = _klinv(0.1, 0.05)
    assert abs(rough - exact) <= 1e-3, f"with tol = 1e-3 the answer must be within 1e-3 of {exact:.6f}; got {rough}"
    assert rough >= exact - 1e-12, (
        "return the UPPER end of the final bracket, so a coarse tolerance makes the bound looser, never invalid")
    vals = [nb.kl_inverse(0.1, c) for c in (0.001, 0.01, 0.1, 1.0)]
    assert all(a < b for a, b in zip(vals, vals[1:])), f"the inverse grows with c: {vals}"


# ---------------------------------------------------------------- stage 3: the PAC-Bayes bound


def test_stage3_values(nb):
    got = nb.pac_bayes_bound(0.05, 10.0, 1000, 0.05)
    c = (10.0 + math.log(2 * math.sqrt(1000) / 0.05)) / 1000
    want = _klinv(0.05, c)
    assert abs(got - want) < 1e-8, (
        f"pac_bayes_bound(0.05, KL = 10, n = 1000, delta = 0.05) = kl_inverse(0.05, (10 + ln(2 sqrt(1000) / 0.05)) / 1000) = "
        f"{want:.6f}; got {got}")
    mc = nb.mcallester_bound(0.05, 10.0, 1000, 0.05)
    assert np.isclose(mc, 0.05 + math.sqrt(c / 2)), "McAllester's form: emp_risk + sqrt((KL + ln(2 sqrt(n)/delta)) / (2n))"
    assert got < mc, f"the kl form ({got:.4f}) is tighter than its Pinsker relaxation ({mc:.4f}) at small empirical risk"


def test_stage3_coverage(nb):
    # KL = 0 (posterior = prior = one fixed classifier): the bound must cover the true risk w.p. >= 1 - delta
    rng = np.random.default_rng(1)
    n, p_true, delta, reps = 200, 0.1, 0.1, 3000
    errs = rng.binomial(n, p_true, size=reps)
    cover = np.mean([nb.pac_bayes_bound(k / n, 0.0, n, delta) >= p_true for k in errs])
    assert cover >= 1 - delta - 3 * math.sqrt(delta * (1 - delta) / reps), (
        f"with KL = 0 the bound is a confidence bound for one classifier: it must cover the true risk 0.1 in at least "
        f"90% of repetitions; it covered {cover:.3f}")


# ---------------------------------------------------------------- stage 4: Gibbs risk and Gaussian KL


def test_stage4_gibbs_error(nb):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 5))
    w = rng.normal(size=5)
    y = np.where(X @ w + 0.5 * rng.normal(size=200) > 0, 1.0, -1.0)
    sigma = 0.8
    got = nb.gibbs_error(w, sigma, X, y)
    W = w + sigma * rng.normal(size=(4000, 5))
    errs = np.mean(np.sign(X @ W.T) != y[:, None], axis=0)
    assert abs(got - errs.mean()) < 4.5 * errs.std() / math.sqrt(4000), (
        f"the Gibbs classifier's expected error should match a simulation over 4,000 weight draws ({errs.mean():.4f}); "
        f"got {got:.4f}. per example, P(y (w + sigma eps) . x < 0) = Phi(-y w.x / (sigma ||x||))")
    det = float(np.mean(np.sign(X @ w) != y))
    assert abs(nb.gibbs_error(w, 1e-9, X, y) - det) < 1e-3, "as sigma -> 0 the Gibbs error becomes the plain 0-1 error"
    assert abs(nb.gibbs_error(w, 1e6, X, y) - 0.5) < 1e-4, "as sigma -> infinity every prediction is a coin flip"
    assert np.isclose(nb.gibbs_error(3 * w, 3 * sigma, X, y), got), "scaling w and sigma together changes nothing"
    Z = np.vstack([np.zeros(5), X[:3]])
    want_z = (0.5 + np.sum([0.5 * math.erfc(y[i] * (X[i] @ w) / (sigma * np.linalg.norm(X[i])) / math.sqrt(2))
                            for i in range(3)])) / 4
    assert np.isclose(nb.gibbs_error(w, sigma, Z, np.r_[1.0, y[:3]]), want_z), (
        "a row with x = 0 gets the prediction sign(0): a coin flip, contributing error 1/2 (and no division by zero)")


def test_stage4_gaussian_kl(nb):
    from mlfs.stats import gaussian_kl
    rng = np.random.default_rng(3)
    for d in (1, 4, 30):
        mq, mp = rng.normal(size=d), rng.normal(size=d)
        sq, sp = rng.uniform(0.2, 2, size=2)
        got = nb.isotropic_gaussian_kl(mq, sq, mp, sp)
        want = gaussian_kl(mq, sq**2 * np.eye(d), mp, sp**2 * np.eye(d))
        assert np.isclose(got, want), (
            f"d = {d}: KL(N(mu_q, {sq:.2f}^2 I) || N(mu_p, {sp:.2f}^2 I)) should be {want:.5f}, got {got}. "
            "(d/2)(s_q^2/s_p^2 - 1 - ln(s_q^2/s_p^2)) + ||mu_q - mu_p||^2 / (2 s_p^2)")
    assert np.isclose(nb.isotropic_gaussian_kl(np.array([2.0, 0, 0]), 1.0, np.zeros(3), 1.0), 2.0), (
        "equal variances: KL = ||mu_q - mu_p||^2 / (2 sigma^2) = 4 / 2 = 2")


# ---------------------------------------------------------------- stage 5: a bound for a linear classifier


def test_stage5_pac_bayes_linear(nb):
    rng = np.random.default_rng(4)
    X = np.hstack([rng.normal(size=(2000, 6)), np.ones((2000, 1))])
    w_true = np.r_[rng.normal(size=6), 0.3]
    y = np.where(X @ w_true > 0, 1.0, -1.0)
    flip = rng.random(2000) < 0.05
    y[flip] *= -1
    w = 2.0 * w_true / np.linalg.norm(w_true)
    sig = [0.05, 0.1, 0.2, 0.4, 0.8]
    out = nb.pac_bayes_linear(w, X, y, sigmas=sig, delta=0.05)
    for key in ("sigma", "emp_risk", "kl", "bound", "bounds"):
        assert key in out, f"the result is a dict with at least sigma, emp_risk, kl, bound, bounds; missing {key!r}"
    bounds = []
    for s in sig:
        e = nb.gibbs_error(w, s, X, y)
        k = nb.isotropic_gaussian_kl(w, s, np.zeros(7), s)
        bounds.append(_klinv(e, (k + math.log(2 * math.sqrt(2000) / (0.05 / 5))) / 2000))
    i = int(np.argmin(bounds))
    assert np.allclose(out["bounds"], bounds, atol=1e-7), (
        f"bounds per sigma should be {np.round(bounds, 5).tolist()}, got {np.round(out['bounds'], 5).tolist()}. for each "
        "s: Gibbs error of N(w, s^2 I), KL to the prior N(0, s^2 I) (prior_sigma=None means 'the same s'), and "
        "pac_bayes_bound with delta / len(sigmas) (a union bound over the grid)")
    assert np.isclose(out["sigma"], sig[i]) and np.isclose(out["bound"], bounds[i]), "report the sigma with the smallest bound"
    assert 0.05 < out["bound"] < 0.3, f"this easy problem should get a non-vacuous bound, got {out['bound']:.3f}"
    w0 = w + 0.1 * rng.normal(size=7)
    tight = nb.pac_bayes_linear(w, X, y, sigmas=sig, prior_mean=w0, prior_sigma=0.2)
    want = _klinv(nb.gibbs_error(w, tight["sigma"], X, y),
                  (nb.isotropic_gaussian_kl(w, tight["sigma"], w0, 0.2) + math.log(2 * math.sqrt(2000) / 0.01)) / 2000)
    assert np.isclose(tight["bound"], want, atol=1e-7), "prior_mean and prior_sigma set the prior N(prior_mean, prior_sigma^2 I)"
    assert tight["bound"] < out["bound"], "a prior centred near w has a smaller KL and a tighter bound"


# ---------------------------------------------------------------- stage 6: margins and stability


def test_stage6_margin_bound(nb):
    m = np.array([2.0, 1.0, 0.5, -0.5, 3.0, 0.25, 1.5, 0.75])
    got = nb.linear_margin_bound(m, 1.0, 2.0, 1.5, 0.1)
    ramp = np.clip(1 - m, 0, 1).mean()
    want = ramp + 2 * 2.0 * 1.5 / math.sqrt(8) + math.sqrt(math.log(10) / 16)
    assert np.isclose(got, want), (
        f"ramp loss {ramp:.4f} (1 for margins <= 0, 1 - m/gamma in between, 0 above gamma) + 2BR/(gamma sqrt n) + "
        f"sqrt(ln(1/delta)/(2n)) = {want:.4f}; got {got}")
    big = np.full(10000, 5.0)
    assert nb.linear_margin_bound(big, 2.0, 1.0, 1.0, 0.05) < nb.linear_margin_bound(big, 0.5, 1.0, 1.0, 0.05), (
        "with every margin large, a larger gamma means a smaller complexity term and a tighter bound")


def test_stage6_loo_stability(nb):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 2))
    y = rng.normal(size=50)
    calls = []

    def mean_model(Xa, ya, Xe):
        calls.append(len(ya))
        return np.full(len(Xe), ya.mean())

    got = bounded(nb.loo_stability, mean_model, X, y, X[:5], n_remove=50, rng=0, what="loo_stability")
    want = np.max(np.abs(y - y.mean())) / 49  # removing y_i moves the mean by (mean - y_i) / (n - 1)
    assert np.isclose(got, want), (
        f"for the 'predict the mean' model, removing point i changes every prediction by |y_i - mean| / (n - 1); the "
        f"largest over all 50 points is {want:.5f}; got {got}")
    assert calls[0] == 50 and sorted(calls[1:]) == [49] * 50, "fit once on S, then once per removed point on S without it"
    calls.clear()
    nb.loo_stability(mean_model, X, y, X[:5], n_remove=7, rng=1)
    assert len(calls) == 8, f"n_remove = 7 removals plus the full fit = 8 calls; got {len(calls)}"
    j = int(np.argsort(np.abs(y - y.mean()))[len(y) // 2])  # a middling point, NOT the most influential one
    got = nb.loo_stability(mean_model, X, y, X[:5], indices=[j])
    assert np.isclose(got, abs(y[j] - y.mean()) / 49), (
        f"with indices=[{j}] remove exactly that training point (n_remove and rng are ignored): the change is "
        f"|y_j - mean| / 49 = {abs(y[j] - y.mean()) / 49:.5f}; got {got}")

    removed = []

    def nn1(Xa, ya, Xe):  # 1-NN regression: predictions depend on the evaluation point
        if len(ya) < len(y):
            removed.append(next(i for i in range(len(y)) if i >= len(ya) or not np.array_equal(Xa[i], X[i])))
        d2 = ((np.asarray(Xe)[:, None, :] - np.asarray(Xa)[None, :, :]) ** 2).sum(-1)
        return np.asarray(ya)[np.argmin(d2, axis=1)]

    Xe = rng.normal(size=(30, 2))
    got = nb.loo_stability(nn1, X, y, Xe, n_remove=50, rng=3)
    full = nn1(X, y, Xe)
    want = max(np.max(np.abs(nn1(np.delete(X, i, 0), np.delete(y, i), Xe) - full)) for i in range(50))
    removed.clear()
    assert np.isclose(got, want), (
        f"the stability is the MAX over evaluation points (and over removals) of |f_S(x) - f_S\\i(x)|: removing one point "
        f"changes 1-NN's prediction only where it was the nearest neighbour; expected {want:.4f}, got {got:.4f}")
    nb.loo_stability(nn1, X, y, Xe, n_remove=30, rng=4)
    assert len(removed) == 30 and len(set(removed)) == 30, (
        f"remove n_remove DISTINCT training points (replace=False); removed {sorted(removed)}")

    def ridge(lam):
        def fp(Xa, ya, Xe):
            w = np.linalg.solve(Xa.T @ Xa + lam * len(ya) * np.eye(Xa.shape[1]), Xa.T @ ya)
            return Xe @ w
        return fp

    Xr = rng.normal(size=(400, 3))
    yr = Xr @ np.array([1.0, -1.0, 0.5]) + rng.normal(size=400)
    Xe = rng.normal(size=(100, 3))
    s_small = nb.loo_stability(ridge(0.01), Xr, yr, Xe, n_remove=30, rng=2)
    s_big = nb.loo_stability(ridge(10.0), Xr, yr, Xe, n_remove=30, rng=2)
    assert s_big < s_small, f"stronger regularization makes ridge more stable: {s_big:.5f} should be < {s_small:.5f}"
