# 12 · capstone: a tiny probabilistic programming language

## goal

Stan, PyMC and NumPyro let you write down a statistical model the way you'd write it on a whiteboard, then hand you posterior samples. underneath, they all do the same few things: find the model's parameters, map the constrained ones (scales, probabilities) to an unconstrained space with the right Jacobian, differentiate the log density, run Hamiltonian Monte Carlo or variational inference, and summarize the result with diagnostics you can trust. in this capstone you build that pipeline, small but complete, out of the pieces of this chapter (and chapter 03's dual numbers), and then use it on the most famous small hierarchical model there is.

the **eight schools** (Rubin, 1981; Gelman et al., *Bayesian Data Analysis*, §5.5): an educational testing service ran SAT coaching programs in eight high schools. each school reports an estimated effect $y_j$ and its standard error $\sigma_j$:

| school | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| $y_j$ | 28 | 8 | −3 | 7 | −1 | 1 | 18 | 12 |
| $\sigma_j$ | 15 | 10 | 16 | 11 | 9 | 11 | 10 | 18 |

do the programs have one common effect, eight unrelated ones, or something in between? the hierarchical model says the school effects $\theta_j$ come from a common population, and lets the data decide how much they differ:

$$\mu \sim \mathcal N(0, 5^2), \qquad \tau \sim \text{HalfCauchy}(5), \qquad \theta_j \sim \mathcal N(\mu, \tau^2), \qquad y_j \sim \mathcal N(\theta_j, \sigma_j^2).$$

it's small (10 parameters) and its posterior is known almost exactly, yet it defeats a naive HMC sampler, because of a **funnel**: when $\tau$ is small, the $\theta_j$ are squeezed together, so the posterior has a wide mouth at large $\tau$ and a narrow neck at small $\tau$. no single step size suits both. the cure is a reparameterization, $\theta_j = \mu + \tau\eta_j$ with $\eta_j \sim \mathcal N(0, 1)$, and seeing *why* it works is the point of this capstone.

## data

the eight $(y_j, \sigma_j)$ pairs above are embedded in the notebook (`Y_SCHOOLS`, `SIGMA_SCHOOLS`); nothing is downloaded. the tests use small synthetic models with known posteriors and the same eight schools.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 2 chains × 300 draws after 200 burn-in; VI 1,000 steps | about 30 seconds |
| **full** (your write-up) | default | 4 chains × 2,000 draws after 1,000 burn-in; VI 4,000 steps | about 2 minutes (measured: centered HMC 50 s, non-centered 43 s, VI 17 s, on a shared machine) |

the notebook picks the config from `datasets.is_offline()`. both samplers use step size 0.2 and 15 leapfrog steps; seeds are fixed in the notebook.

## deliverables

1. `capstone.py` with the required code implemented and its checks green:
   - **distributions** `Normal(loc, scale)`, `HalfNormal(scale)`, `HalfCauchy(scale)` with an elementwise `log_prob(x)` written only with Dual-friendly operations (so autodiff works through the parameters too), and a class attribute `support` (`"real"` or `"positive"`).
   - **`Trace`**: the object a model function `fn(t, *args)` receives. `t.sample(name, dist, shape=())` declares a latent parameter; `t.observe(dist, value)` adds a likelihood term; `t.deterministic(name, value)` records a derived quantity. in setup mode (no `z`) it records each site's shape and support and returns placeholder ones; in evaluation mode it slices its values out of the unconstrained vector `z`, maps positive sites through $\exp$, and accumulates the log density plus the log-Jacobian $\sum u$.
   - **`Model(fn, *args, **kwargs)`**: runs `fn` once to find `names`, `shapes`, `supports`, a `layout` and `dim`; `log_prob(z)`, `unpack(z)` (constrained values of every site, deterministic ones included), `init(rng)` (uniform on $(-2, 2)$, Stan's default), and `value_and_grad(z)` / `grad(z)` computed in **one** run of the model by seeding each coordinate with a dual number whose tangent is a unit *vector* (`Dual(z_i, e_i)`): forward mode with vector tangents gives the whole gradient in a single pass. the `mlfs.autodiff.Dual` contract allows an ndarray `.dual`, and a stage-3 preflight check tests that *your* chapter 03 `Dual` handles one through the operations the PPL uses; if it doesn't, fix it or run the notebook with `MLFS_REFERENCE=autodiff` to use the reference `Dual`.
   - **inference**: `sample_hmc(model, n_samples, *, n_chains=4, step_size=0.1, n_leapfrog=20, burn_in=500, rng=None)` → `(draws, accept_rates)`, with `draws` a dict name → array `(chains, n_samples, *shape)` on the constrained scale (via `mlfs.infer.hmc` and your `Model.grad`); `fit_vi(model, *, family='diagonal', n_iters=2000, lr=0.01, n_samples=16, n_draws=1000, rng=None)` → `(draws, q, elbo_history)` (via `mlfs.infer.reparam_vi`); `summarize(draws)` → dict `"name"` or `"name[i]"` → `{mean, sd, q5, q95, ess, r_hat}` (ESS summed over chains, split $\hat R$ across chains, both from `mlfs.infer`).
   - **eight schools**: `eight_schools(t, y, sigma)` (centered), `eight_schools_noncentered(t, y, sigma)` (with `theta` as a deterministic site), and `eight_schools_exact(y, sigma, *, mu_sd=5.0, tau_scale=5.0, n_grid=4000)`: the exact posterior by one-dimensional quadrature over $\log\tau$, integrating $\mu$ and the $\theta_j$ analytically (given $\tau$, $y_j \sim \mathcal N(\mu, \sigma_j^2 + \tau^2)$ and $\mu$ is conjugate). it returns the posterior means and sds of $\mu$, $\tau$ and $\log\tau$, the posterior means of the $\theta_j$, and the density of $\log\tau$ on its grid.
2. the experiments: experiment A (provided) runs both parameterizations and VI and compares them with the exact answer; experiment B (yours) shows the funnel: a scatter of $\theta_1$ against $\log\tau$ for both samplers, a divergence map (centered trajectories started from true posterior states, the non-centered draws mapped to centered coordinates, with the divergence rate per $\log\tau$ bin), and the histograms of $\log\tau$ over the exact density.
3. the write-up at the end of the notebook, answering its five prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| distributions | match `scipy.stats` log densities; autodiff derivatives (in $x$ and in the parameters) match finite differences |
| `Model` | log density with the exact Jacobian (a HalfCauchy-only model integrates to 1 in $\log$ space); one model evaluation per gradient |
| inference | conjugate normal mean and a scale parameter (by quadrature) recovered by HMC within 4.5 Monte Carlo standard errors; VI within 0.06 of the exact mean |
| eight schools | `eight_schools_exact` agrees with a brute-force 2-D quadrature; non-centered HMC agrees with it within 4.5 Monte Carlo standard errors |

for reference, the exact posterior: $\mathbb E[\mu] = 4.40$ (sd 3.32), $\mathbb E[\tau] = 3.60$ (sd 3.22), $\mathbb E[\log\tau] = 0.80$ (sd 1.17), $\mathbb E[\theta_A] = 6.21$, and $P(\tau < 1) = 0.200$. with the **full config**, the reference solution measured (mean ± Monte Carlo error, ESS):

| sampler | $\mathbb E[\mu]$ | $\mathbb E[\tau]$ | $\mathbb E[\log\tau]$ | ESS of $\tau$ | $\hat R$ of $\mu$ | acceptance |
|---|---|---|---|---|---|---|
| centered HMC | 1.73 ± 0.41 | 2.35 ± 0.36 | −0.18 | 110 | 1.58 | 0.86, **0.00, 0.00**, 0.92 |
| non-centered HMC | 4.38 ± 0.08 | 3.54 ± 0.04 | 0.78 | 6,414 | 1.000 | 0.99 (all chains) |
| non-centered VI (diagonal) | 4.49 ± 0.07 | 3.06 ± 0.06 | 0.83 | (independent draws) | | |

$P(\tau < 1)$ from the same full run: centered **0.563**, non-centered **0.201** (exact 0.200). two of the four centered chains got stuck in the neck of the funnel ($\log\tau \approx -1.8$ and $-1.0$) and never accepted a single proposal after warm-up, so the centered run *over*-represents small $\tau$. the small config's run shows a milder version of the same thing: $\mathbb E[\log\tau] = 0.65$ (exact 0.80) and $P(\tau < 1) = 0.30$ (exact 0.20), with a stretch where a chain stalls near $\log\tau \approx -0.5$, while it never gets below $\log\tau \approx -1.5$ ($\hat R(\tau) = 1.46$, ESS 18). either way the centered answer is wrong and depends on luck, and $\hat R$ flags it; the non-centered answer agrees with the exact posterior to within its error bars. VI gets $\mu$ right but its sd of $\tau$ is 2.86 against the exact 3.22 (and of $\mu$ 2.93 against 3.32): the familiar underestimation of spread.

**where the centered sampler diverges.** starting centered trajectories (step 0.2, 15 steps) from 600 *true* posterior states (non-centered draws mapped to centered coordinates), the full run measured these divergence rates (energy error above 100):

| $\log\tau$ | starts | divergent |
|---|---|---|
| below $-1$ | 52 | **77%** |
| $-1$ to 0 | 86 | 12% |
| 0 to 1 | 174 | 0% |
| above 1 | 288 | 0% |

8% of all starts diverge, with median $\log\tau = -1.63$ for the divergent ones against 0.93 overall: divergences are a map of the neck. (the small config gives the same picture: 69%, 1%, 0%, 0%.)

## the funnel, in one paragraph

in the centered model, given $\tau$ the effects satisfy $\theta_j - \mu \sim \mathcal N(0, \tau^2)$, so the log posterior curves like $1/\tau^2$ in the $\theta$ directions: at $\tau = 10$ a leapfrog step of 0.2 is tiny, at $\tau = 0.1$ it's ten times the width of the distribution, past the stability limit of topic 06. HMC either can't enter the neck (its trajectories diverge there, so it under-samples small $\tau$ and overestimates the spread) or, once it's in, can't leave (every proposal diverges, so the chain freezes). in the non-centered model the sampler works with $\eta_j = (\theta_j - \mu)/\tau$, whose prior is $\mathcal N(0, 1)$ whatever $\tau$ is: the funnel lives in the prior, and the reparameterization removes it. with very informative data (tiny $\sigma_j$) the likelihood pins each $\theta_j$ down and the roles reverse: then the centered model is the easy one.

## stretch goals

- **step-size adaptation.** implement dual averaging (Hoffman & Gelman, 2014, §3.2) during warm-up to target an acceptance rate of 0.8, and see what step size the centered model ends up with, and whether it helps.
- **a mass matrix from warm-up.** estimate each coordinate's posterior variance from the warm-up draws and pass it as `inv_mass`.
- **more distributions.** add `Exponential`, `Beta` (with a logit transform for the $(0, 1)$ support and its Jacobian) and a `Bernoulli` likelihood, and fit a small logistic regression with your PPL.
- **when centering wins.** rerun both parameterizations with every $\sigma_j$ divided by 20, and explain the result.
- **full-covariance VI.** fit the non-centered model with `family='full'` and compare the spread of $\tau$ with the diagonal fit and the exact value.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Rubin (1981), "Estimation in parallel randomized experiments", *Journal of Educational Statistics* 6(4). the eight schools. [doi:10.3102/10769986006004377](https://doi.org/10.3102/10769986006004377)
- Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), *Bayesian Data Analysis*, 3rd ed., CRC Press, §5.5 (the eight schools) and ch. 12. [stat.columbia.edu/~gelman/book](http://www.stat.columbia.edu/~gelman/book/)
- Papaspiliopoulos, Roberts & Sköld (2007), "A general framework for the parametrization of hierarchical models", *Statistical Science* 22(1). centered versus non-centered. [doi:10.1214/088342307000000014](https://doi.org/10.1214/088342307000000014)
- Betancourt & Girolami (2015), "Hamiltonian Monte Carlo for hierarchical models", in *Current Trends in Bayesian Methodology with Applications*, CRC Press. the funnel and divergences. [arXiv:1312.0906](https://arxiv.org/abs/1312.0906)
- Carpenter et al. (2017), "Stan: a probabilistic programming language", *Journal of Statistical Software* 76(1). [doi:10.18637/jss.v076.i01](https://doi.org/10.18637/jss.v076.i01)
- Bingham et al. (2019), "Pyro: deep universal probabilistic programming", *JMLR* 20. effect handlers: the "run the model with a tracer" design you build here. [arXiv:1810.09538](https://arxiv.org/abs/1810.09538)
