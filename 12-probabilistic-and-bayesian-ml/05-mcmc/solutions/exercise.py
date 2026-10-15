import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # markov chain monte carlo · build (reference solution)

    work through the stages in order. a sampler here gets an **unnormalized** log density
    `log_prob(x)` (x a float array of shape (d,)) and returns a chain of correlated draws
    whose long-run distribution is the target. the diagnostics tell you how much those draws
    are worth. everything becomes part of `mlfs.infer`.

    | stage | you build |
    |---|---|
    | 1 | `metropolis_hastings` → `mlfs.infer`: random-walk Metropolis with burn-in and thinning |
    | 2 | `gibbs_sampler` → `mlfs.infer`: one coordinate at a time, from its full conditional |
    | 3 | `autocorrelation` → `mlfs.infer`: the ACF of a chain, via the FFT |
    | 4 | `effective_sample_size` → `mlfs.infer`: how many independent draws the chain is worth |
    | 5 | `r_hat` → `mlfs.infer`: did several chains find the same distribution? |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def metropolis_hastings(log_prob, x0, n_samples, *, proposal_scale=1.0, burn_in=0, thin=1, rng=None):
    """stage 1: random-walk Metropolis; returns (samples (n_samples, d) after burn-in and thinning, acceptance rate)."""
    rng = as_rng(rng)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    d = x.size
    lp = log_prob(x)
    total = burn_in + n_samples * thin
    out = np.empty((n_samples, d))
    accepted = 0
    k = 0
    for it in range(total):
        prop = x + proposal_scale * rng.standard_normal(d)
        lp_prop = log_prob(prop)
        if math.log(max(rng.random(), 1e-300)) < lp_prop - lp:
            x, lp = prop, lp_prop
            accepted += 1
        if it >= burn_in and (it - burn_in + 1) % thin == 0:
            out[k] = x
            k += 1
    return out, accepted / max(total, 1)


@app.cell(hide_code=True)
def _():
    checks.run(1, metropolis_hastings=metropolis_hastings)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gibbs_sampler(conditionals, x0, n_samples, *, burn_in=0, thin=1, rng=None):
    """stage 2: systematic-scan Gibbs; conditionals[i](x, rng) draws coordinate i given the current x; returns (n_samples, d)."""
    rng = as_rng(rng)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    d = x.size
    total = burn_in + n_samples * thin
    out = np.empty((n_samples, d))
    k = 0
    for it in range(total):
        for i in range(d):
            x[i] = conditionals[i](x, rng)
        if it >= burn_in and (it - burn_in + 1) % thin == 0:
            out[k] = x
            k += 1
    return out


@app.cell(hide_code=True)
def _():
    checks.run(2, gibbs_sampler=gibbs_sampler)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def autocorrelation(x, max_lag=None):
    """stage 3: normalized autocorrelation rho_0..rho_max_lag of a 1-D chain (rho_0 = 1; all ones if constant), via the FFT."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if max_lag is None:
        max_lag = n - 1
    y = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()  # zero-pad to avoid circular wrap-around
    f = np.fft.rfft(y, size)
    acov = np.fft.irfft(f * np.conj(f), size)[: max_lag + 1]
    if acov[0] <= 0:  # a chain that never moves is perfectly correlated with itself
        return np.ones(max_lag + 1)
    return acov / acov[0]


@app.cell(hide_code=True)
def _():
    checks.run(3, autocorrelation=autocorrelation)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def effective_sample_size(chain):
    """stage 4: Geyer's initial-positive-sequence ESS of a chain (n,) -> float, or per column of (n, d) -> (d,)."""
    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 2:
        return np.array([effective_sample_size(chain[:, j]) for j in range(chain.shape[1])])
    n = chain.size
    rho = autocorrelation(chain)
    tau = -1.0
    for m in range(n // 2):
        if 2 * m + 1 >= n:
            break
        pair = rho[2 * m] + rho[2 * m + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
    tau = max(tau, 1.0 / math.log10(n))
    return float(n / tau)


@app.cell(hide_code=True)
def _():
    checks.run(4, effective_sample_size=effective_sample_size, autocorrelation=autocorrelation)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def r_hat(chains):
    """stage 5: split R-hat of chains (m, n) -> float, or per coordinate of (m, n, d) -> (d,)."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 3:
        return np.array([r_hat(chains[:, :, j]) for j in range(chains.shape[2])])
    half = chains.shape[1] // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)  # (2m, half)
    W = np.mean(np.var(split, axis=1, ddof=1))
    B = half * np.var(np.mean(split, axis=1), ddof=1)
    var_plus = (half - 1) / half * W + B / half
    return float(np.sqrt(var_plus / W))


@app.cell(hide_code=True)
def _():
    checks.run(5, r_hat=r_hat)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your samplers and diagnostics are part of `mlfs.infer` (try
    `from mlfs.infer import metropolis_hastings` in a scratch cell). then open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
