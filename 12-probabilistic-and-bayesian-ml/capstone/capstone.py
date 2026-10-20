import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.autodiff import Dual  # yours, from chapter 03
    from mlfs.infer import effective_sample_size, hmc, leapfrog, r_hat, reparam_vi  # yours, from this chapter
    from mlfs.utils import as_rng  # yours, from chapter 00

    # the eight schools (Rubin, 1981; Gelman et al., Bayesian Data Analysis, section 5.5): the estimated
    # effect of SAT coaching in each school, and its standard error.
    Y_SCHOOLS = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    SIGMA_SCHOOLS = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    N_CHAINS = 2 if SMALL else 4
    N_DRAWS = 300 if SMALL else 2000
    BURN_IN = 200 if SMALL else 1000


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 12 · capstone: a tiny probabilistic programming language

    read [`brief.md`](brief.md) first. you'll build a PPL in a few hundred lines: write a
    model as an ordinary Python function that *samples* its parameters and *observes* its
    data; the library turns it into an unconstrained log density with exact gradients, runs
    your HMC or VI on it, and summarizes the posterior. then you use it on the classic
    hierarchical model, the **eight schools**, and meet the funnel.

    | stage | you build |
    |---|---|
    | 1 | `Normal`, `HalfNormal`, `HalfCauchy`: Dual-friendly log densities, with a support |
    | 2 | `Trace`, `Model`: run a model function to find its parameters, then as a log density on $\mathbb R^d$ |
    | 3 | `Model.grad`: the whole gradient in ONE forward pass, with vector-valued dual parts |
    | 4 | `sample_hmc`, `fit_vi`, `summarize`: inference and posterior summaries with ESS and $\hat R$ |
    | 5 | `eight_schools`, `eight_schools_noncentered`, `eight_schools_exact` |

    when they're green, scroll down: the experiments run on your library, and the write-up
    prompts are at the end.
    """)
    return


@app.class_definition
class Normal:
    """N(loc, scale) on the real line; loc and scale may be numbers, arrays or Duals."""

    support = "real"

    def __init__(self, loc, scale):
        """stage 1: store loc and scale."""
        raise NotImplementedError("stage1: implement Normal.__init__")

    def log_prob(self, x):
        """stage 1: elementwise log density at x, written only with Dual-friendly operations."""
        raise NotImplementedError("stage1: implement Normal.log_prob")


@app.class_definition
class HalfNormal:
    """the normal N(0, scale) folded onto x > 0."""

    support = "positive"

    def __init__(self, scale):
        """stage 1: store scale."""
        raise NotImplementedError("stage1: implement HalfNormal.__init__")

    def log_prob(self, x):
        """stage 1: elementwise log density for x > 0: log 2 + log N(x; 0, scale)."""
        raise NotImplementedError("stage1: implement HalfNormal.log_prob")


@app.class_definition
class HalfCauchy:
    """the Cauchy(0, scale) folded onto x > 0: heavy-tailed, a popular prior for scales."""

    support = "positive"

    def __init__(self, scale):
        """stage 1: store scale."""
        raise NotImplementedError("stage1: implement HalfCauchy.__init__")

    def log_prob(self, x):
        """stage 1: elementwise log density for x > 0: log(2 / (pi scale)) - log1p((x / scale)^2)."""
        raise NotImplementedError("stage1: implement HalfCauchy.log_prob")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", Normal=Normal, HalfNormal=HalfNormal, HalfCauchy=HalfCauchy)
    return


@app.class_definition
class Trace:
    """the object a model function receives as `t`. in setup mode (z is None) it records the sites;
    in evaluation mode it hands out values from z and accumulates the log density."""

    def __init__(self, z=None, layout=None):
        """stage 2: z (d,) or None for setup; layout: name -> (start, shape, support)."""
        raise NotImplementedError("stage2: implement Trace.__init__")

    def sample(self, name, dist, shape=()):
        """stage 2: a latent site. setup: record (shape, support), return ones; eval: value from z (+ log density, Jacobian)."""
        raise NotImplementedError("stage2: implement Trace.sample")

    def observe(self, dist, value):
        """stage 2: an observed site: add sum(dist.log_prob(value)) to the log density (eval mode only)."""
        raise NotImplementedError("stage2: implement Trace.observe")

    def deterministic(self, name, value):
        """stage 2: record a derived quantity (e.g. theta = mu + tau * eta) so unpack can report it."""
        raise NotImplementedError("stage2: implement Trace.deterministic")


@app.class_definition
class Model:
    """a model function fn(t, *args, **kwargs) turned into an unconstrained log density on R^dim."""

    def __init__(self, fn, *args, **kwargs):
        """stage 2: run fn once in setup mode; store names (in order), shapes, supports, layout and dim."""
        raise NotImplementedError("stage2: implement Model.__init__")

    def log_prob(self, z):
        """stage 2: log p(data, params(z)) + log |Jacobian| for unconstrained z (d,) (floats or Duals)."""
        raise NotImplementedError("stage2: implement Model.log_prob")

    def unpack(self, z):
        """stage 2: dict name -> constrained value (float or array) for every sample and deterministic site."""
        raise NotImplementedError("stage2: implement Model.unpack")

    def init(self, rng=None):
        """stage 2: a starting point, uniform on (-2, 2) in unconstrained space (Stan's default)."""
        raise NotImplementedError("stage2: implement Model.init")

    def value_and_grad(self, z):
        """stage 3: (log_prob(z), grad (d,)) in ONE evaluation: seed z_i with the dual part e_i (a vector)."""
        raise NotImplementedError("stage3: implement Model.value_and_grad")

    def grad(self, z):
        """stage 3: the gradient of log_prob at z, shape (d,)."""
        raise NotImplementedError("stage3: implement Model.grad")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", Model=Model, Trace=Trace, Normal=Normal, HalfNormal=HalfNormal,
               HalfCauchy=HalfCauchy)
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", Model=Model, Normal=Normal, HalfNormal=HalfNormal, HalfCauchy=HalfCauchy)
    return


@app.function
def to_draws(model, raw):
    """unconstrained draws (chains, n, d) -> dict name -> constrained array (chains, n, *shape)."""
    raw = np.asarray(raw, dtype=float)
    C, n, _ = raw.shape
    first = model.unpack(raw[0, 0])
    out = {k: np.empty((C, n) + np.shape(v)) for k, v in first.items()}
    for c in range(C):
        for i in range(n):
            for k, v in model.unpack(raw[c, i]).items():
                out[k][c, i] = v
    return out


@app.function
def sample_hmc(model, n_samples, *, n_chains=4, step_size=0.1, n_leapfrog=20, burn_in=500, rng=None):
    """stage 4: n_chains HMC chains from model.init(rng); returns (draws dict name -> (chains, n, *shape), accept rates (chains,))."""
    raise NotImplementedError("stage4: implement sample_hmc")


@app.function
def fit_vi(model, *, family="diagonal", n_iters=2000, lr=0.01, n_samples=16, n_draws=1000, rng=None):
    """stage 4: reparam_vi on the model; returns (draws dict name -> (1, n_draws, *shape), q, elbo history)."""
    raise NotImplementedError("stage4: implement fit_vi")


@app.function
def summarize(draws):
    """stage 4: dict 'name' or 'name[i]' -> dict(mean, sd, q5, q95, ess, r_hat) for every scalar quantity."""
    raise NotImplementedError("stage4: implement summarize")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", Model=Model, Normal=Normal, HalfNormal=HalfNormal, HalfCauchy=HalfCauchy,
               sample_hmc=sample_hmc, fit_vi=fit_vi, summarize=summarize)
    return


@app.function
def eight_schools(t, y, sigma):
    """stage 5: the centered model: mu ~ N(0, 5), tau ~ HalfCauchy(5), theta ~ N(mu, tau) (8,), y ~ N(theta, sigma)."""
    raise NotImplementedError("stage5: implement eight_schools")


@app.function
def eight_schools_noncentered(t, y, sigma):
    """stage 5: the same model with theta = mu + tau * eta, eta ~ N(0, 1) (8,); theta recorded as a deterministic."""
    raise NotImplementedError("stage5: implement eight_schools_noncentered")


@app.function
def eight_schools_exact(y, sigma, *, mu_sd=5.0, tau_scale=5.0, n_grid=4000):
    """stage 5: exact posterior summaries by 1-D quadrature over log tau (mu and theta integrated analytically)."""
    raise NotImplementedError("stage5: implement eight_schools_exact")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", Model=Model, Normal=Normal, HalfCauchy=HalfCauchy, sample_hmc=sample_hmc,
               summarize=summarize, eight_schools=eight_schools, eight_schools_noncentered=eight_schools_noncentered,
               eight_schools_exact=eight_schools_exact)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A: the exact answer, and two parameterizations

    first the ground truth from `eight_schools_exact`. then your HMC on both
    parameterizations with the same budget (step size 0.2, 15 leapfrog steps), and VI on the
    non-centered one. a good PPL user always asks: do the samplers agree with each other,
    with the diagnostics, and (here, unusually) with the exact answer?
    """)
    return


@app.cell
def _():
    exact = eight_schools_exact(Y_SCHOOLS, SIGMA_SCHOOLS)
    centered = Model(eight_schools, Y_SCHOOLS, SIGMA_SCHOOLS)
    noncentered = Model(eight_schools_noncentered, Y_SCHOOLS, SIGMA_SCHOOLS)
    _t0 = time.perf_counter()
    draws_c, acc_c = sample_hmc(centered, N_DRAWS, n_chains=N_CHAINS, step_size=0.2, n_leapfrog=15, burn_in=BURN_IN,
                                rng=0)
    _t1 = time.perf_counter()
    draws_n, acc_n = sample_hmc(noncentered, N_DRAWS, n_chains=N_CHAINS, step_size=0.2, n_leapfrog=15,
                                burn_in=BURN_IN, rng=1)
    _t2 = time.perf_counter()
    draws_vi, q_vi, elbo_vi = fit_vi(noncentered, n_iters=1000 if SMALL else 4000, n_draws=2000, rng=2)
    timings = (_t1 - _t0, _t2 - _t1, time.perf_counter() - _t2)
    return acc_c, acc_n, centered, draws_c, draws_n, draws_vi, exact, noncentered, timings


@app.cell
def _(acc_c, acc_n, draws_c, draws_n, draws_vi, exact, timings):
    _sc, _sn, _sv = summarize(draws_c), summarize(draws_n), summarize(draws_vi)
    _log_c = np.log(draws_c["tau"]).mean()
    _log_n = np.log(draws_n["tau"]).mean()
    _log_v = np.log(draws_vi["tau"]).mean()

    def _row(label, s, key, exact_mean, exact_sd):
        sd_txt = "–" if exact_sd is None else f"{exact_sd:.2f}"
        return (f"| {label} | {key} | {s[key]['mean']:.2f} ± {s[key]['sd'] / np.sqrt(max(s[key]['ess'], 1)):.2f} | "
                f"{s[key]['sd']:.2f} | {s[key]['ess']:.0f} | {s[key]['r_hat']:.3f} | {exact_mean:.2f} | {sd_txt} |")

    _rows = []
    for _lab, _s in [("centered HMC", _sc), ("non-centered HMC", _sn), ("non-centered VI", _sv)]:
        _rows.append(_row(_lab, _s, "mu", exact["mu_mean"], exact["mu_sd"]))
        _rows.append(_row(_lab, _s, "tau", exact["tau_mean"], exact["tau_sd"]))
        _rows.append(_row(_lab, _s, "theta[0]", exact["theta_mean"][0], None))
    mo.md(
        f"config: {'small' if SMALL else 'full'} ({N_CHAINS} chains x {N_DRAWS} draws after {BURN_IN} burn-in). "
        f"time: centered {timings[0]:.1f} s, non-centered {timings[1]:.1f} s, VI {timings[2]:.1f} s. "
        f"HMC acceptance: centered {np.round(acc_c, 2).tolist()}, non-centered {np.round(acc_n, 2).tolist()}.\n\n"
        "| run | quantity | mean ± MC error | sd | ESS | R-hat | exact mean | exact sd |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\nE[log tau]: exact **{exact['log_tau_mean']:.3f}**, centered **{_log_c:.3f}**, "
        f"non-centered **{_log_n:.3f}**, VI **{_log_v:.3f}**.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B: the funnel

    the joint posterior of $\log\tau$ and one school's effect $\theta_1$, from each sampler,
    over the exact marginal density of $\log\tau$. and a divergence map: we take 600 states
    from the *non-centered* chains (which sample the true posterior), map them to centered
    coordinates ($\theta_j = \mu + \tau\eta_j$), draw a momentum, integrate one centered
    trajectory with the sampler's settings (step 0.2, 15 steps), and flag it as **divergent**
    if the energy error exceeds 100. replaying from the centered chains' own states would
    mostly measure wherever those chains happened to get stuck. how does the divergence rate
    depend on $\log\tau$?
    """)
    return


@app.cell
def _(centered, draws_c, draws_n, exact):
    # your analysis. (1) scatter theta_1 against log tau for the centered and the non-centered draws (the funnel).
    # (2) divergence map: take a few hundred TRUE posterior states from the non-centered draws, map them to centered
    # coordinates (mu, log tau, theta = mu + tau * eta), draw a momentum, run mlfs.infer.leapfrog with the sampler's
    # step size and number of steps on centered.log_prob / centered.grad, and flag the start as divergent if the
    # energy error exceeds 100. report the divergence rate per log tau bin (< -1, -1..0, 0..1, > 1) and mark the
    # divergent starts on the scatter plot. (don't replay from the centered chains' own states: stuck chains would
    # dominate the count.)
    # (3) plot the histograms of log tau from both samplers over exact["log_tau_grid"] / exact["log_tau_density"],
    # and compare P(tau < 1) with the exact value.
    mo.md(f"experiment B isn't written yet. the exact E[log tau] to compare with is {exact['log_tau_mean']:.3f}.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer each prompt with numbers from the experiments above (full config: run with
    `MLFS_OFFLINE` unset). see [`rubric.md`](rubric.md).

    1. **the library.** how does your `Model` find the parameters of a model function, how does
       it handle the constraint $\tau > 0$, and why is the Jacobian term needed? how much faster
       is your one-pass `grad` than `mlfs.autodiff.grad` on the eight-schools model, and why?
    2. **correctness.** how do you know your HMC is right? compare the non-centered posterior
       summaries with `eight_schools_exact`, with Monte Carlo error bars from the ESS.
    3. **the funnel.** describe the geometry of the centered posterior in $(\theta_j, \log\tau)$.
       why does a fixed step size fail there? which diagnostics flagged the problem (ESS, $\hat R$,
       divergences, the $\log\tau$ histogram), and which didn't? what's the bias in
       $\mathbb E[\log\tau]$ and $P(\tau < 1)$?
    4. **the fix.** why does the non-centered parameterization remove the funnel? when would
       the centered one be the better choice (think about a school with $\sigma_j = 1$)?
    5. **VI versus MCMC.** how do VI's posterior means and standard deviations for $\mu$ and $\tau$
       compare with the exact values? relate what you see to topic 07's lab.
    """)
    return


if __name__ == "__main__":
    app.run()
