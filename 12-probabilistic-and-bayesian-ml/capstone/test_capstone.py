"""Checks for capstone.py (a tiny probabilistic programming language).

Small, seeded setups only. Distributions are checked against scipy.stats and for Dual-friendliness;
the Model against hand-written log densities (Jacobian included) and a normalization integral;
the one-pass gradient against finite differences, counting model evaluations; inference against
EXACT posteriors (a conjugate normal mean, a scale parameter by quadrature, the eight schools by
2-D quadrature) with tolerances of 4.5 Monte Carlo standard errors from the reported ESS. Cheap
calls run under mlfs.testing.bounded; samplers get a capped model that fails after a generous
number of log-density and gradient evaluations, so a runaway loop fails instead of hanging.
"""
import math

import numpy as np
import pytest
from scipy import stats

from mlfs.autodiff import derivative

Y8 = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
S8 = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def _toy(nb):
    def model(t, y):
        mu = t.sample("mu", nb.Normal(0.0, 10.0))
        s = t.sample("s", nb.HalfNormal(2.0))
        t.observe(nb.Normal(mu, s), y)

    return model


Y_TOY = np.array([1.2, 0.4, 2.1, 1.7, 0.9])


class _Capped:
    """wraps a Model; raises AssertionError once log_prob / grad / value_and_grad are called more than `limit` times."""

    def __init__(self, model, limit, what):
        self._m, self._limit, self._what, self.calls = model, int(limit), what, 0

    def __getattr__(self, name):
        return getattr(self._m, name)

    def _tick(self):
        self.calls += 1
        if self.calls > self._limit:
            raise AssertionError(
                f"{self._what} evaluated the model more than {self._limit:,} times, several times what this run "
                "needs: a loop that doesn't stop, or far too many evaluations per step")

    def log_prob(self, z):
        self._tick()
        return self._m.log_prob(z)

    def grad(self, z):
        self._tick()
        return self._m.grad(z)

    def value_and_grad(self, z):
        self._tick()
        return self._m.value_and_grad(z)


def _hmc_budget(chains, n, burn, L):
    return 4 * chains * (n + burn) * (L + 3)


# ---------------------------------------------------------------- stage 1: distributions


def test_stage1_log_densities(nb):
    x = np.array([0.1, 0.7, 2.5, 6.0])
    got = np.asarray(nb.Normal(np.array([0.0, 1.0, -1.0, 2.0]), 1.5).log_prob(x))
    assert np.allclose(got, stats.norm.logpdf(x, [0.0, 1.0, -1.0, 2.0], 1.5)), (
        "Normal(loc, scale).log_prob(x) = -((x - loc)/scale)^2 / 2 - log(scale) - log(2 pi)/2, elementwise, "
        "broadcasting loc and scale")
    assert np.allclose(nb.HalfNormal(2.0).log_prob(x), stats.halfnorm.logpdf(x, scale=2.0)), (
        "HalfNormal(scale): twice the N(0, scale) density on x > 0, so log 2 + log N(x; 0, scale)")
    assert np.allclose(nb.HalfCauchy(5.0).log_prob(x), stats.halfcauchy.logpdf(x, scale=5.0)), (
        "HalfCauchy(scale): log(2 / (pi scale)) - log(1 + (x/scale)^2); use np.log1p")
    assert nb.Normal(0, 1).support == "real" and nb.HalfNormal(1).support == "positive" and \
        nb.HalfCauchy(1).support == "positive", "support is 'real' for Normal, 'positive' for the half distributions"


def test_stage1_dual_friendly(nb):
    for dist, h in [(lambda v: nb.Normal(0.5, 2.0).log_prob(v), 1e-6), (lambda v: nb.HalfCauchy(3.0).log_prob(v), 1e-6),
                    (lambda v: nb.HalfNormal(1.5).log_prob(v), 1e-6), (lambda v: nb.Normal(0.3, v).log_prob(1.7), 1e-6),
                    (lambda v: nb.Normal(v, 1.3).log_prob(0.2), 1e-6), (lambda v: nb.HalfCauchy(v).log_prob(2.0), 1e-6)]:
        x = 1.3
        d = derivative(dist, x)
        fd = (float(dist(x + h)) - float(dist(x - h))) / (2 * h)
        assert np.isclose(d, fd, rtol=1e-5), (
            "log_prob must work on mlfs.autodiff.Dual numbers (in x AND in the parameters): use arithmetic, ** and "
            "np.log / np.log1p (which dispatch to Dual methods), never math.log or float() on the inputs. "
            f"autodiff gave {d}, finite differences {fd}")


# ---------------------------------------------------------------- stage 2: the model


def test_stage2_layout_and_unpack(nb):
    m = nb.Model(_toy(nb), Y_TOY)
    assert m.names == ["mu", "s"] and m.dim == 2, f"sites in the order they're sampled: ['mu', 's'], dim 2; got {m.names}, {m.dim}"
    assert m.supports == {"mu": "real", "s": "positive"} and m.shapes == {"mu": (), "s": ()}
    v = m.unpack(np.array([0.3, math.log(2.0)]))
    assert np.isclose(v["mu"], 0.3) and np.isclose(v["s"], 2.0), (
        f"unpack maps unconstrained z to constrained values: mu = z[0], s = exp(z[1]); got {v}")

    def vec(t):
        a = t.sample("a", nb.HalfNormal(1.0), shape=3)
        b = t.sample("b", nb.Normal(0.0, 1.0))
        t.deterministic("c", a * b)

    m2 = nb.Model(vec)
    assert m2.dim == 4 and m2.shapes["a"] == (3,), "a vector site of shape 3 takes 3 coordinates of z"
    v2 = m2.unpack(np.array([0.0, 1.0, -1.0, 2.0]))
    assert np.allclose(v2["a"], np.exp([0.0, 1.0, -1.0])) and np.isclose(v2["b"], 2.0), "the layout follows the site order"
    assert np.allclose(v2["c"], 2.0 * np.exp([0.0, 1.0, -1.0])), "deterministic sites are reported by unpack"

    def dup(t):
        t.sample("x", nb.Normal(0.0, 1.0))
        t.sample("x", nb.Normal(0.0, 1.0))

    with pytest.raises(ValueError):
        nb.Model(dup)


def test_stage2_log_prob_with_jacobian(nb):
    m = nb.Model(_toy(nb), Y_TOY)
    z = np.array([0.8, -0.4])
    s = math.exp(-0.4)
    want = (stats.norm.logpdf(0.8, 0, 10) + stats.halfnorm.logpdf(s, scale=2.0) + (-0.4)
            + stats.norm.logpdf(Y_TOY, 0.8, s).sum())
    got = m.log_prob(z)
    assert np.isclose(got, want), (
        f"log_prob(z) = log p(mu) + log p(s) + log|ds/dz| + sum log p(y | mu, s) with s = exp(z[1]) and log|ds/dz| "
        f"= z[1]; expected {want:.6f}, got {got}")

    def scale_only(t):
        t.sample("s", nb.HalfCauchy(2.0))

    m2 = nb.Model(scale_only)
    u = np.linspace(-40, 40, 40001)
    total = np.trapezoid(np.exp([m2.log_prob(np.array([v])) for v in u]), u)
    assert np.isclose(total, 1.0, atol=2e-3), (
        f"a model with only s ~ HalfCauchy(2) must give a density in u = log s that integrates to 1 (you need the "
        f"Jacobian term + u); got {total:.4f}")


# ---------------------------------------------------------------- stage 3: one-pass gradients


def test_stage3_preflight_vector_tangents(nb):
    from mlfs.autodiff import Dual  # yours from chapter 03 in the notebook (the reference under pytest)

    a, v = 1.3, np.array([1.0, 0.0, -2.0])
    escape = ("Model.grad seeds Dual(z_i, e_i) with a unit VECTOR as the dual part, so every Dual operation must work "
              "elementwise on an ndarray .dual (the contract allows it). fix your chapter 03 Dual (never call float() on "
              "the parts, never test `if self.dual == 0`), or run this notebook with MLFS_REFERENCE=autodiff to use the "
              "reference Dual.")
    try:
        x = Dual(a, v.copy())
        y = np.log1p(x * x) + np.exp(x) / x - x**2 + 3.0 * x - 1.0 + 2.0 / x
        arr = np.sum(np.array([x, 2.0 * x], dtype=object) * np.array([1.0, 0.5]))
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{escape} (it failed with {type(e).__name__}: {e})") from None
    dfdx = 2 * a / (1 + a * a) + np.exp(a) * (a - 1) / a**2 - 2 * a + 3.0 - 2.0 / a**2
    assert np.allclose(np.asarray(y.dual, dtype=float), dfdx * v), f"{escape} (wrong tangent for a composite expression)"
    assert np.allclose(np.asarray(arr.dual, dtype=float), 2 * v), f"{escape} (wrong tangent through an object array)"


def test_stage3_gradient(nb):
    def schools(t, y, sigma):  # the centered eight-schools model (you'll write your own in stage 5)
        mu = t.sample("mu", nb.Normal(0.0, 5.0))
        tau = t.sample("tau", nb.HalfCauchy(5.0))
        theta = t.sample("theta", nb.Normal(mu, tau), shape=len(y))
        t.observe(nb.Normal(theta, sigma), y)

    for fn, args in [(_toy(nb), (Y_TOY,)), (schools, (Y8, S8))]:
        m = nb.Model(fn, *args)
        z = np.random.default_rng(1).uniform(-1, 1, m.dim)
        val, g = m.value_and_grad(z)
        assert np.isclose(val, m.log_prob(z)), "value_and_grad returns log_prob(z) as its value"
        h = 1e-6
        fd = np.array([(m.log_prob(z + h * e) - m.log_prob(z - h * e)) / (2 * h) for e in np.eye(m.dim)])
        assert np.shape(g) == (m.dim,) and np.allclose(g, fd, rtol=1e-5, atol=1e-6), (
            f"Model.grad disagrees with finite differences (largest error {np.max(np.abs(g - fd)):.2e})")
        assert np.allclose(m.grad(z), g), "grad(z) is value_and_grad(z)[1]"


def test_stage3_one_pass(nb):
    calls = [0]

    def model(t):
        calls[0] += 1
        x = t.sample("x", nb.Normal(0.0, 1.0), shape=6)
        s = t.sample("s", nb.HalfNormal(1.0))
        t.observe(nb.Normal(x, s), np.arange(6.0))

    m = nb.Model(model)
    calls[0] = 0
    g = m.grad(np.linspace(-1, 1, 7))
    assert calls[0] == 1, (
        f"a 7-parameter gradient should cost ONE run of the model: seed z_i with Dual(z_i, e_i), where e_i is the "
        f"i-th unit VECTOR, and read the whole gradient off the result's dual part. you ran the model {calls[0]} times")
    assert np.all(np.isfinite(g)) and np.shape(g) == (7,)


# ---------------------------------------------------------------- stage 4: inference and summaries


def _conj(nb):
    y = np.random.default_rng(7).normal(1.0, 1.0, 20)

    def model(t):
        mu = t.sample("mu", nb.Normal(0.0, 10.0))
        t.observe(nb.Normal(mu, 1.0), y)

    prec = 1 / 100 + 20
    return model, y.sum() / prec, 1 / math.sqrt(prec)


def test_stage4_hmc_conjugate(nb):
    model, mean, sd = _conj(nb)
    capped = _Capped(nb.Model(model), _hmc_budget(2, 400, 100, 10), "sample_hmc")
    draws, acc = nb.sample_hmc(capped, 400, n_chains=2, step_size=0.1, n_leapfrog=10, burn_in=100, rng=0)
    assert capped.calls >= 2 * (400 + 100) * 11, (
        f"2 chains x (100 burn-in + 400 kept) iterations x (10 + 1) gradients per trajectory need at least 11,000 "
        f"gradient "
        f"evaluations; the model was evaluated {capped.calls} times. pass burn_in (and n_leapfrog) on to hmc")
    assert np.shape(draws["mu"]) == (2, 400) and np.shape(acc) == (2,), (
        "sample_hmc returns draws as name -> (chains, n_samples, *shape) and one acceptance rate per chain")
    s = nb.summarize(draws)
    assert set(s["mu"]) >= {"mean", "sd", "q5", "q95", "ess", "r_hat"}, f"summary fields; got {sorted(s['mu'])}"
    se = sd / math.sqrt(s["mu"]["ess"])
    assert abs(s["mu"]["mean"] - mean) < 4.5 * se, (
        f"normal mean with a N(0, 10^2) prior and 20 observations: exact posterior N({mean:.4f}, {sd:.4f}^2). "
        f"HMC mean {s['mu']['mean']:.4f}, Monte Carlo error {se:.4f}")
    assert abs(s["mu"]["sd"] / sd - 1) < 0.2, f"posterior sd should be {sd:.4f}; got {s['mu']['sd']:.4f}"
    assert s["mu"]["r_hat"] < 1.05, f"two healthy chains: R-hat ~ 1; got {s['mu']['r_hat']:.3f}"
    assert s["mu"]["q5"] < mean < s["mu"]["q95"], "the 90% interval should contain the posterior mean"


def test_stage4_positive_parameter(nb):
    y = np.random.default_rng(8).normal(0.0, 0.3, 15)

    def model(t):
        s = t.sample("s", nb.HalfNormal(2.0))
        t.observe(nb.Normal(0.0, s), y)

    grid = np.linspace(1e-3, 8, 20001)
    lp = stats.halfnorm.logpdf(grid, scale=2.0) + stats.norm.logpdf(y[:, None], 0, grid).sum(0)
    w = np.exp(lp - lp.max())
    w /= np.trapezoid(w, grid)
    mean = np.trapezoid(w * grid, grid)
    sd = math.sqrt(np.trapezoid(w * grid**2, grid) - mean**2)
    capped = _Capped(nb.Model(model), _hmc_budget(2, 500, 100, 8), "sample_hmc")
    draws, acc = nb.sample_hmc(capped, 500, n_chains=2, step_size=0.08, n_leapfrog=8, burn_in=100, rng=1)
    assert np.all(np.asarray(acc) > 0.5), f"step 0.08 in log-scale space should be accepted most of the time; got {acc}"
    s = nb.summarize(draws)["s"]
    se = sd / math.sqrt(s["ess"])
    assert np.all(np.asarray(draws["s"]) > 0), "draws of a positive parameter are reported on the constrained scale"
    assert abs(s["mean"] - mean) < 4.5 * se, (
        f"scale of a zero-mean normal with a HalfNormal(2) prior: the exact posterior mean (by quadrature) is "
        f"{mean:.4f}; HMC gives {s['mean']:.4f} (Monte Carlo error {se:.4f}). check the log-Jacobian term")


def test_stage4_vi_and_vector_summaries(nb):
    model, mean, sd = _conj(nb)
    capped = _Capped(nb.Model(model), 4 * 1000 * 16 * 2, "fit_vi")
    draws, q, hist = nb.fit_vi(capped, n_iters=1000, n_samples=8, n_draws=500, rng=2)
    assert np.shape(draws["mu"]) == (1, 500) and len(hist) == 1000, "fit_vi: draws (1, n_draws), one ELBO per iteration"
    assert abs(float(q.mean[0]) - mean) < 0.06 and abs(math.sqrt(q.cov[0, 0]) / sd - 1) < 0.25, (
        f"on a Gaussian posterior VI is exact: q should be close to N({mean:.3f}, {sd:.3f}^2); got mean "
        f"{float(q.mean[0]):.3f}, sd {math.sqrt(q.cov[0, 0]):.3f}")
    th = np.random.default_rng(3).normal(size=(2, 50, 3)).cumsum(axis=1)
    b = np.random.default_rng(4).normal(size=(2, 50))
    s = nb.summarize({"theta": th, "b": b})
    assert set(s) == {"theta[0]", "theta[1]", "theta[2]", "b"}, f"vector sites are summarized per element: got {sorted(s)}"
    from mlfs.infer import effective_sample_size, r_hat
    want_ess = effective_sample_size(th[0, :, 1]) + effective_sample_size(th[1, :, 1])
    assert np.isclose(s["theta[1]"]["ess"], want_ess) and np.isclose(s["theta[1]"]["r_hat"], r_hat(th[:, :, 1])), (
        "ess is the SUM of the per-chain effective sample sizes, r_hat the split R-hat across the chains")
    assert np.isclose(s["theta[1]"]["sd"], th[:, :, 1].std(ddof=1)) and np.isclose(s["b"]["q5"], np.quantile(b, 0.05)), (
        "sd (ddof=1) and quantiles are over all chains pooled")


# ---------------------------------------------------------------- stage 5: eight schools


def _exact_2d():
    mu = np.linspace(-25, 35, 801)
    u = np.linspace(-10, 6, 801)
    tau = np.exp(u)
    M, T = np.meshgrid(mu, tau, indexing="ij")
    ll = stats.norm.logpdf(Y8[None, None, :], M[..., None], np.sqrt(S8**2 + T[..., None] ** 2)).sum(-1)
    lp = ll + stats.norm.logpdf(M, 0, 5) + stats.halfcauchy.logpdf(T, scale=5) + np.log(T)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    return float((w * M).sum()), float((w * T).sum()), float((w * np.log(T)).sum())


def test_stage5_models(nb):
    c = nb.Model(nb.eight_schools, Y8, S8)
    n = nb.Model(nb.eight_schools_noncentered, Y8, S8)
    assert c.names == ["mu", "tau", "theta"] and c.dim == 10, f"centered: mu, tau, theta (8): dim 10; got {c.names}"
    assert n.names == ["mu", "tau", "eta"] and n.dim == 10, f"non-centered: mu, tau, eta (8): dim 10; got {n.names}"
    z = np.random.default_rng(4).uniform(-1, 1, 10)
    tau = math.exp(z[1])
    want = (stats.norm.logpdf(z[0], 0, 5) + stats.halfcauchy.logpdf(tau, scale=5) + z[1]
            + stats.norm.logpdf(z[2:], z[0], tau).sum() + stats.norm.logpdf(Y8, z[2:], S8).sum())
    assert np.isclose(c.log_prob(z), want), "centered log density: mu ~ N(0, 5), tau ~ HalfCauchy(5), theta ~ N(mu, tau)"
    v = n.unpack(z)
    assert np.allclose(v["theta"], z[0] + tau * z[2:]), "non-centered: theta = mu + tau * eta, reported by unpack"
    want_n = (stats.norm.logpdf(z[0], 0, 5) + stats.halfcauchy.logpdf(tau, scale=5) + z[1]
              + stats.norm.logpdf(z[2:]).sum() + stats.norm.logpdf(Y8, z[0] + tau * z[2:], S8).sum())
    assert np.isclose(n.log_prob(z), want_n), "non-centered log density: eta ~ N(0, 1), y ~ N(mu + tau eta, sigma)"


def test_stage5_exact(nb):
    ex = nb.eight_schools_exact(Y8, S8)
    mu2, tau2, lt2 = _exact_2d()
    assert abs(ex["mu_mean"] - mu2) < 0.02 and abs(ex["tau_mean"] - tau2) < 0.03 and abs(ex["log_tau_mean"] - lt2) < 0.01, (
        f"a brute-force 2-D quadrature over (mu, log tau) gives E[mu] = {mu2:.3f}, E[tau] = {tau2:.3f}, "
        f"E[log tau] = {lt2:.3f}; got {ex['mu_mean']:.3f}, {ex['tau_mean']:.3f}, {ex['log_tau_mean']:.3f}")
    assert 3.2 < ex["mu_sd"] < 3.45 and 3.0 < ex["tau_sd"] < 3.45, "posterior sds: about 3.3 for mu, 3.2 for tau"
    th = np.asarray(ex["theta_mean"])
    assert th.shape == (8,) and th[0] > ex["mu_mean"] > th[2], (
        "partial pooling: each school's effect is shrunk from its estimate y_j towards the common mean, so school 1 "
        "(y = 28) stays above E[mu] and school 3 (y = -3) below it")
    assert np.all(np.abs(th - ex["mu_mean"]) < np.abs(Y8 - ex["mu_mean"])), "every theta_j lies between y_j and E[mu]"


@pytest.mark.slow
def test_stage5_noncentered_hmc_matches_exact(nb):
    ex = nb.eight_schools_exact(Y8, S8)
    capped = _Capped(nb.Model(nb.eight_schools_noncentered, Y8, S8), _hmc_budget(2, 400, 150, 12), "sample_hmc")
    draws, acc = nb.sample_hmc(capped, 400, n_chains=2, step_size=0.25, n_leapfrog=12, burn_in=150, rng=5)
    s = nb.summarize(draws)
    for key, mean, sd in [("mu", ex["mu_mean"], ex["mu_sd"]), ("tau", ex["tau_mean"], ex["tau_sd"])]:
        se = sd / math.sqrt(s[key]["ess"])
        assert abs(s[key]["mean"] - mean) < 4.5 * se, (
            f"non-centered eight schools: posterior mean of {key} {s[key]['mean']:.3f}, exact {mean:.3f} (Monte Carlo "
            f"error {se:.3f}, ESS {s[key]['ess']:.0f})")
    assert s["theta[0]"]["mean"] > s["theta[2]"]["mean"], "school 1 is estimated above school 3"
