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
    # 12 · capstone: a tiny probabilistic programming language (reference solution)

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
        self.loc = loc
        self.scale = scale

    def log_prob(self, x):
        """stage 1: elementwise log density at x, written only with Dual-friendly operations."""
        z = (x - self.loc) / self.scale
        return -0.5 * z * z - np.log(self.scale) - 0.5 * math.log(2 * math.pi)


@app.class_definition
class HalfNormal:
    """the normal N(0, scale) folded onto x > 0."""

    support = "positive"

    def __init__(self, scale):
        """stage 1: store scale."""
        self.scale = scale

    def log_prob(self, x):
        """stage 1: elementwise log density for x > 0: log 2 + log N(x; 0, scale)."""
        z = x / self.scale
        return -0.5 * z * z - np.log(self.scale) + 0.5 * math.log(2 / math.pi)


@app.class_definition
class HalfCauchy:
    """the Cauchy(0, scale) folded onto x > 0: heavy-tailed, a popular prior for scales."""

    support = "positive"

    def __init__(self, scale):
        """stage 1: store scale."""
        self.scale = scale

    def log_prob(self, x):
        """stage 1: elementwise log density for x > 0: log(2 / (pi scale)) - log1p((x / scale)^2)."""
        z = x / self.scale
        return math.log(2 / math.pi) - np.log(self.scale) - np.log1p(z * z)


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
        self.z = z
        self.layout = layout
        self.sites = {}
        self.values = {}
        self.log_prob = 0.0

    def sample(self, name, dist, shape=()):
        """stage 2: a latent site. setup: record (shape, support), return ones; eval: value from z (+ log density, Jacobian)."""
        shape = tuple(np.atleast_1d(shape).astype(int)) if np.size(shape) else ()
        if self.z is None:
            if name in self.sites:
                raise ValueError(f"site {name!r} is sampled twice")
            self.sites[name] = (shape, dist.support)
            value = np.ones(shape) if shape else 1.0
        else:
            start, shape, support = self.layout[name]
            size = int(np.prod(shape)) if shape else 1
            u = self.z[start:start + size]
            if support == "positive":
                value = np.exp(u)
                self.log_prob = self.log_prob + np.sum(u)  # log |d exp(u) / du| = u
            else:
                value = u
            value = value.reshape(shape) if shape else value[0]
            self.log_prob = self.log_prob + np.sum(dist.log_prob(value))
        self.values[name] = value
        return value

    def observe(self, dist, value):
        """stage 2: an observed site: add sum(dist.log_prob(value)) to the log density (eval mode only)."""
        if self.z is not None:
            self.log_prob = self.log_prob + np.sum(dist.log_prob(np.asarray(value, dtype=float)))

    def deterministic(self, name, value):
        """stage 2: record a derived quantity (e.g. theta = mu + tau * eta) so unpack can report it."""
        self.values[name] = value
        return value


@app.class_definition
class Model:
    """a model function fn(t, *args, **kwargs) turned into an unconstrained log density on R^dim."""

    def __init__(self, fn, *args, **kwargs):
        """stage 2: run fn once in setup mode; store names (in order), shapes, supports, layout and dim."""
        self.fn, self.args, self.kwargs = fn, args, kwargs
        t = Trace()
        fn(t, *args, **kwargs)
        self.names = list(t.sites)
        self.shapes = {n: t.sites[n][0] for n in self.names}
        self.supports = {n: t.sites[n][1] for n in self.names}
        self.layout = {}
        start = 0
        for n in self.names:
            self.layout[n] = (start, self.shapes[n], self.supports[n])
            start += int(np.prod(self.shapes[n])) if self.shapes[n] else 1
        self.dim = start

    def log_prob(self, z):
        """stage 2: log p(data, params(z)) + log |Jacobian| for unconstrained z (d,) (floats or Duals)."""
        t = Trace(z, self.layout)
        self.fn(t, *self.args, **self.kwargs)
        return t.log_prob

    def unpack(self, z):
        """stage 2: dict name -> constrained value (float or array) for every sample and deterministic site."""
        t = Trace(np.asarray(z, dtype=float), self.layout)
        self.fn(t, *self.args, **self.kwargs)
        return {k: (float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=float)) for k, v in t.values.items()}

    def init(self, rng=None):
        """stage 2: a starting point, uniform on (-2, 2) in unconstrained space (Stan's default)."""
        return as_rng(rng).uniform(-2.0, 2.0, self.dim)

    def value_and_grad(self, z):
        """stage 3: (log_prob(z), grad (d,)) in ONE evaluation: seed z_i with the dual part e_i (a vector)."""
        z = np.asarray(z, dtype=float)
        d = z.size
        zd = np.empty(d, dtype=object)
        for i in range(d):
            e = np.zeros(d)
            e[i] = 1.0
            zd[i] = Dual(float(z[i]), e)
        out = self.log_prob(zd)
        if not isinstance(out, Dual):
            return float(out), np.zeros(d)
        return float(out.real), np.broadcast_to(np.asarray(out.dual, dtype=float), (d,)).copy()

    def grad(self, z):
        """stage 3: the gradient of log_prob at z, shape (d,)."""
        return self.value_and_grad(z)[1]


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
    rng = as_rng(rng)
    raw, acc = [], []
    for _ in range(n_chains):
        s, a = hmc(model.log_prob, model.init(rng), n_samples, grad_log_prob=model.grad, step_size=step_size,
                   n_leapfrog=n_leapfrog, burn_in=burn_in, rng=rng)
        raw.append(s)
        acc.append(a)
    return to_draws(model, np.array(raw)), np.array(acc)


@app.function
def fit_vi(model, *, family="diagonal", n_iters=2000, lr=0.01, n_samples=16, n_draws=1000, rng=None):
    """stage 4: reparam_vi on the model; returns (draws dict name -> (1, n_draws, *shape), q, elbo history)."""
    rng = as_rng(rng)
    q, hist = reparam_vi(model.log_prob, model.dim, grad_log_prob=model.grad, family=family, n_samples=n_samples,
                         lr=lr, n_iters=n_iters, rng=rng)
    raw = q.sample(n_draws, rng=rng)[None]
    return to_draws(model, raw), q, hist


@app.function
def summarize(draws):
    """stage 4: dict 'name' or 'name[i]' -> dict(mean, sd, q5, q95, ess, r_hat) for every scalar quantity."""
    out = {}
    for name, arr in draws.items():
        arr = np.asarray(arr, dtype=float)
        C, n = arr.shape[:2]
        flat = arr.reshape(C, n, -1)
        for j in range(flat.shape[2]):
            key = name if arr.ndim == 2 else f"{name}[{j}]"
            x = flat[:, :, j]
            out[key] = {
                "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                "q5": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95)),
                "ess": float(sum(effective_sample_size(x[c]) for c in range(C))),
                "r_hat": float(r_hat(x)),
            }
    return out


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", Model=Model, Normal=Normal, HalfNormal=HalfNormal, HalfCauchy=HalfCauchy,
               sample_hmc=sample_hmc, fit_vi=fit_vi, summarize=summarize)
    return


@app.function
def eight_schools(t, y, sigma):
    """stage 5: the centered model: mu ~ N(0, 5), tau ~ HalfCauchy(5), theta ~ N(mu, tau) (8,), y ~ N(theta, sigma)."""
    mu = t.sample("mu", Normal(0.0, 5.0))
    tau = t.sample("tau", HalfCauchy(5.0))
    theta = t.sample("theta", Normal(mu, tau), shape=len(y))
    t.observe(Normal(theta, sigma), y)


@app.function
def eight_schools_noncentered(t, y, sigma):
    """stage 5: the same model with theta = mu + tau * eta, eta ~ N(0, 1) (8,); theta recorded as a deterministic."""
    mu = t.sample("mu", Normal(0.0, 5.0))
    tau = t.sample("tau", HalfCauchy(5.0))
    eta = t.sample("eta", Normal(0.0, 1.0), shape=len(y))
    theta = t.deterministic("theta", mu + tau * eta)
    t.observe(Normal(theta, sigma), y)


@app.function
def eight_schools_exact(y, sigma, *, mu_sd=5.0, tau_scale=5.0, n_grid=4000):
    """stage 5: exact posterior summaries by 1-D quadrature over log tau (mu and theta integrated analytically)."""
    y = np.asarray(y, dtype=float)
    s2 = np.asarray(sigma, dtype=float) ** 2
    u = np.linspace(-12.0, 8.0, n_grid)  # log tau
    tau = np.exp(u)
    D = s2[None, :] + tau[:, None] ** 2  # (G, J): marginal variances of y_j given tau, mu integrated below
    inv = 1.0 / D
    a = 1.0 + mu_sd**2 * inv.sum(1)
    quad = (y**2 * inv).sum(1) - mu_sd**2 * (y * inv).sum(1) ** 2 / a  # Sherman-Morrison
    logdet = np.log(D).sum(1) + np.log(a)
    log_lik = -0.5 * (quad + logdet)
    log_prior = math.log(2 / (math.pi * tau_scale)) - np.log1p((tau / tau_scale) ** 2) + u  # + Jacobian of tau = e^u
    lw = log_lik + log_prior
    w = np.exp(lw - lw.max())
    w /= np.trapezoid(w, u)
    prec = 1.0 / mu_sd**2 + inv.sum(1)
    mu_hat = (y * inv).sum(1) / prec  # E[mu | tau, y]
    E = lambda f: float(np.trapezoid(w * f, u))
    mu_mean = E(mu_hat)
    mu_var = E(1.0 / prec + mu_hat**2) - mu_mean**2
    tau_mean = E(tau)
    theta_hat = (y[None, :] / s2 + mu_hat[:, None] / tau[:, None] ** 2) / (1 / s2 + 1 / tau[:, None] ** 2)
    return {
        "mu_mean": mu_mean, "mu_sd": math.sqrt(mu_var),
        "tau_mean": tau_mean, "tau_sd": math.sqrt(E(tau**2) - tau_mean**2),
        "log_tau_mean": E(u), "log_tau_sd": math.sqrt(E(u**2) - E(u) ** 2),
        "theta_mean": np.array([E(theta_hat[:, j]) for j in range(len(y))]),
        "log_tau_grid": u, "log_tau_density": w,
    }


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
    _rng = np.random.default_rng(5)
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    for _ax, _d, _title in [(_axes[0], draws_c, "centered"), (_axes[1], draws_n, "non-centered")]:
        _ax.scatter(_d["theta"][..., 0].ravel(), np.log(_d["tau"]).ravel(), s=3, alpha=0.4)
        _ax.set(xlabel="θ₁", ylabel="log τ", title=f"{_title} HMC", ylim=(-6, 4), xlim=(-25, 45))
    # divergence map: start centered trajectories from TRUE posterior states (non-centered draws, mapped)
    _mu, _lt_n, _th = draws_n["mu"].ravel(), np.log(draws_n["tau"]).ravel(), draws_n["theta"].reshape(-1, 8)
    _pick = _rng.choice(len(_mu), size=min(600, len(_mu)), replace=False)
    _z = np.column_stack([_mu[_pick], _lt_n[_pick], _th[_pick]])
    _div = []
    for _q in _z:
        _p = _rng.standard_normal(centered.dim)
        _h0 = -centered.log_prob(_q) + 0.5 * _p @ _p
        with np.errstate(all="ignore"):
            _q1, _p1 = leapfrog(_q, _p, centered.grad, 0.2, 15)
            _h1 = -centered.log_prob(_q1) + 0.5 * _p1 @ _p1 if np.all(np.isfinite(_q1)) else np.inf
        _div.append(not np.isfinite(_h1) or _h1 - _h0 > 100)
    _div = np.array(_div)
    _axes[1].scatter(_z[_div, 2], _z[_div, 1], s=12, c="C3", label="a centered trajectory from here diverges")
    _axes[1].legend(fontsize=7, loc="lower right")
    _bins = [(-np.inf, -1), (-1, 0), (0, 1), (1, np.inf)]
    _rates = []
    for _lo, _hi in _bins:
        _in = (_z[:, 1] >= _lo) & (_z[:, 1] < _hi)
        _rates.append(f"| [{_lo}, {_hi}) | {_in.sum()} | {_div[_in].mean() if _in.any() else float('nan'):.0%} |")
    _u = exact["log_tau_grid"]
    _axes[2].plot(_u, exact["log_tau_density"], "k", label="exact")
    for _d, _lab in [(draws_c, "centered"), (draws_n, "non-centered")]:
        _axes[2].hist(np.log(_d["tau"]).ravel(), bins=60, range=(-6, 4), density=True, histtype="step", label=_lab)
    _axes[2].set(xlim=(-6, 4), xlabel="log τ", ylabel="density", title="posterior of log τ")
    _axes[2].legend(fontsize=8)
    _fig.tight_layout()
    _lt = np.log(draws_c["tau"].ravel())
    mo.vstack([_fig, mo.md(
        f"centered trajectories started from {len(_z)} posterior states: **{_div.sum()}** diverge "
        f"({_div.mean():.0%}). median log τ of the divergent starts: "
        f"{np.median(_z[_div, 1]) if _div.any() else float('nan'):.2f}, of all starts: {np.median(_z[:, 1]):.2f}.\n\n"
        "| log τ | starts | divergent |\n|---|---|---|\n" + "\n".join(_rates)
        + f"\n\nexact P(τ < 1) = {np.trapezoid(exact['log_tau_density'][_u < 0], _u[_u < 0]):.3f}; "
        f"centered HMC: {np.mean(_lt < 0):.3f}; non-centered HMC: {np.mean(np.log(draws_n['tau']) < 0):.3f}.")])
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
