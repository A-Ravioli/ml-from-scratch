import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import logsumexp  # yours, from ch 00 · topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # maximum likelihood & MAP · build (reference solution)

    work through the stages in order. you'll write log-likelihoods for three common families,
    find their maximizers in closed form, then turn the crank of Bayes' rule: conjugate
    updates for a coin (beta–bernoulli) and a Gaussian mean (normal–normal), and finally a
    grid posterior that works for *any* one-parameter model.

    | stage | you build |
    |---|---|
    | 1 | `bernoulli_loglik`, `normal_loglik`, `poisson_loglik`: how well a parameter explains the data |
    | 2 | `mle_bernoulli`, `mle_normal`, `mle_poisson`, `mle_exponential`: the maximizers, in closed form |
    | 3 | `beta_bernoulli_posterior`, `beta_mean`, `beta_mode`: conjugate updating for a coin |
    | 4 | `normal_posterior`: a Gaussian mean with a Gaussian prior |
    | 5 | `grid_posterior`, `credible_interval`: Bayes' rule numerically, for any model |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def bernoulli_loglik(x, p: float) -> float:
    """stage 1: log P(x | p) for i.i.d. Bernoulli(p) data x (n,) of 0s and 1s.

    k log p + (n - k) log(1 - p) with k = number of ones, using 0 * log 0 = 0, so p = 1 with
    all-ones data gives 0.0 and p = 1 with any zero gives -inf. returns a Python float.
    """
    x = np.asarray(x)
    k, n = int(np.sum(x)), len(x)
    out = 0.0
    if k > 0:
        out += k * math.log(p) if p > 0 else -math.inf
    if n - k > 0:
        out += (n - k) * math.log1p(-p) if p < 1 else -math.inf
    return float(out)


@app.function
def normal_loglik(x, mu: float, sigma2: float) -> float:
    """stage 1: log density of i.i.d. N(mu, sigma2) data x (n,): sum of -(x - mu)^2 / (2 sigma2) - log(2 pi sigma2) / 2."""
    x = np.asarray(x, dtype=float)
    return float(-0.5 * np.sum((x - mu) ** 2) / sigma2 - 0.5 * len(x) * math.log(2 * math.pi * sigma2))


@app.function
def poisson_loglik(x, lam: float) -> float:
    """stage 1: log P(x | lam) for i.i.d. Poisson(lam) counts x (n,): sum of x log lam - lam - log(x!).

    log(x!) = math.lgamma(x + 1). lam = 0 is allowed when every count is 0 (then the result is 0.0).
    """
    x = np.asarray(x)
    s = int(np.sum(x))
    log_fact = sum(math.lgamma(int(v) + 1) for v in x)
    if lam == 0:
        return 0.0 if s == 0 else -math.inf
    return float(s * math.log(lam) - len(x) * lam - log_fact)


@app.cell(hide_code=True)
def _():
    checks.run(1, bernoulli_loglik=bernoulli_loglik, normal_loglik=normal_loglik, poisson_loglik=poisson_loglik)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def mle_bernoulli(x) -> float:
    """stage 2: the maximum-likelihood p for Bernoulli data x (n,): the fraction of ones."""
    return float(np.mean(np.asarray(x, dtype=float)))


@app.function
def mle_normal(x) -> tuple:
    """stage 2: the maximum-likelihood (mu, sigma2) for N(mu, sigma2) data x (n,). sigma2 divides by n."""
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    return mu, float(np.mean((x - mu) ** 2))


@app.function
def mle_poisson(x) -> float:
    """stage 2: the maximum-likelihood rate lam for Poisson counts x (n,)."""
    return float(np.mean(np.asarray(x, dtype=float)))


@app.function
def mle_exponential(x) -> float:
    """stage 2: the maximum-likelihood rate for Exponential(rate) data x (n,) (density rate * exp(-rate x))."""
    return float(1.0 / np.mean(np.asarray(x, dtype=float)))


@app.cell(hide_code=True)
def _():
    checks.run(2, mle_bernoulli=mle_bernoulli, mle_normal=mle_normal, mle_poisson=mle_poisson,
               mle_exponential=mle_exponential)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def beta_bernoulli_posterior(x, a: float, b: float) -> tuple:
    """stage 3: the Beta posterior after observing coin flips x (n,) of 0s/1s under a Beta(a, b) prior.

    returns (a + number of ones, b + number of zeros) as floats.
    """
    x = np.asarray(x)
    k = float(np.sum(x))
    return float(a + k), float(b + len(x) - k)


@app.function
def beta_mean(a: float, b: float) -> float:
    """stage 3: the mean a / (a + b) of Beta(a, b), which is also the posterior predictive P(next flip = 1)."""
    return float(a / (a + b))


@app.function
def beta_mode(a: float, b: float) -> float:
    """stage 3: the mode (a - 1) / (a + b - 2) of Beta(a, b): the MAP estimate.

    defined for a >= 1, b >= 1 with a + b > 2; raise ValueError otherwise (the density is then
    flat, or unbounded at an end point).
    """
    if a < 1 or b < 1 or a + b <= 2:
        raise ValueError(f"Beta({a}, {b}) has no unique interior-or-endpoint mode by this formula")
    return float((a - 1) / (a + b - 2))


@app.cell(hide_code=True)
def _():
    checks.run(3, beta_bernoulli_posterior=beta_bernoulli_posterior, beta_mean=beta_mean, beta_mode=beta_mode)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def normal_posterior(x, sigma2: float, mu0: float, tau2: float) -> tuple:
    """stage 4: the posterior of a Gaussian mean mu given data x (n,) ~ N(mu, sigma2) (sigma2 known)
    and the prior mu ~ N(mu0, tau2).

    returns (posterior mean, posterior variance): precisions add, 1/var = 1/tau2 + n/sigma2, and
    the mean is the precision-weighted average of mu0 and the sample mean.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    prec = 1.0 / tau2 + n / sigma2
    mean = (mu0 / tau2 + np.sum(x) / sigma2) / prec
    return float(mean), float(1.0 / prec)


@app.cell(hide_code=True)
def _():
    checks.run(4, normal_posterior=normal_posterior)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def grid_posterior(log_lik, log_prior, grid) -> np.ndarray:
    """stage 5: Bayes' rule on a grid of parameter values. returns probabilities (m,) summing to 1.

    log_lik(grid) and log_prior(grid) each map the grid (m,) to log values (m,). the posterior
    mass at grid[j] is proportional to exp(log_lik + log_prior); normalize in log space (with
    logsumexp) so that log-likelihoods like -10^5 don't underflow to 0/0.
    """
    grid = np.asarray(grid, dtype=float)
    logp = np.asarray(log_lik(grid), dtype=float) + np.asarray(log_prior(grid), dtype=float)
    return np.exp(logp - logsumexp(logp))


@app.function
def credible_interval(grid, post, level: float = 0.95) -> tuple:
    """stage 5: the equal-tailed credible interval of a grid posterior post (m,) on grid (m,), ascending.

    lo is the first grid value where the cumulative probability reaches (1 - level) / 2, hi the first
    where it reaches (1 + level) / 2. returns (lo, hi) as floats.
    """
    grid = np.asarray(grid, dtype=float)
    cdf = np.cumsum(post)
    tail = (1 - level) / 2
    lo = grid[np.searchsorted(cdf, tail)]
    hi = grid[min(np.searchsorted(cdf, 1 - tail), len(grid) - 1)]
    return float(lo), float(hi)


@app.cell(hide_code=True)
def _():
    checks.run(5, grid_posterior=grid_posterior, credible_interval=credible_interval)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`) and find out how much a
    prior is worth, and why maximum likelihood can be infinitely wrong.
    """)
    return


if __name__ == "__main__":
    app.run()
