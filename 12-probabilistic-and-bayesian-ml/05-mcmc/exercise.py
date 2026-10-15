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
    # markov chain monte carlo · build

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
    raise NotImplementedError("stage1: implement metropolis_hastings")


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
    raise NotImplementedError("stage2: implement gibbs_sampler")


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
    raise NotImplementedError("stage3: implement autocorrelation")


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
    raise NotImplementedError("stage4: implement effective_sample_size")


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
    raise NotImplementedError("stage5: implement r_hat")


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
