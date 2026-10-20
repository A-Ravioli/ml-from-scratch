import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # estimation · build (reference solution)

    work through the stages in order. you're building an **estimator-evaluation harness**: a
    small simulator that takes any estimator, runs it on thousands of fresh datasets drawn from
    a distribution whose true parameter you know, and reports its bias, variance and mean
    squared error. it's the tool statisticians use to decide which estimator to trust, and
    you'll use it in the lab to catch a biased estimator beating an unbiased one.

    two conventions run through the whole topic. a **sampler** is a function
    `sampler(size, rng)` returning i.i.d. draws with shape `size`, as in ch 04. an
    **estimator** works on many datasets at once: it takes an array `(n_reps, n)`, one dataset
    per row, and returns `(n_reps,)` estimates, one per row (for example
    `lambda x: x.mean(axis=-1)`).

    | stage | you build |
    |---|---|
    | 1 | `bias_variance_mse`: score a pile of estimates against the truth |
    | 2 | `simulate`, `evaluate`: the harness, with a Monte Carlo error bar on the MSE |
    | 3 | `theta_mom`, `theta_max`, `theta_unbiased_max`: three estimators of a uniform's upper end |
    | 4 | `scaled_variance`, `scaled_variance_mse`: dividing by $n - 1$, $n$, or $n + 1$ |
    | 5 | `mse_curve`, `convergence_rate`: consistency, and how fast |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def bias_variance_mse(estimates, theta: float) -> dict:
    """stage 1: summarize estimates (n_reps,) of a true value theta.

    returns a dict with float entries
      "bias":     mean(estimates) - theta
      "variance": the spread of the estimates around THEIR OWN mean (divide by n_reps, ddof=0)
      "mse":      mean((estimates - theta) ** 2)
    with these conventions, mse == bias**2 + variance exactly (up to rounding).
    """
    est = np.asarray(estimates, dtype=float)
    center = est.mean()
    return {
        "bias": float(center - theta),
        "variance": float(np.mean((est - center) ** 2)),
        "mse": float(np.mean((est - theta) ** 2)),
    }


@app.cell(hide_code=True)
def _():
    checks.run(1, bias_variance_mse=bias_variance_mse)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def simulate(estimator, sampler, n: int, n_reps: int, *, rng=None) -> np.ndarray:
    """stage 2: run estimator on n_reps fresh datasets of size n. returns float (n_reps,).

    normalize rng with as_rng, draw ONE array sampler((n_reps, n), rng) (one dataset per row),
    and apply estimator to it: estimator maps (n_reps, n) -> (n_reps,).
    """
    rng = as_rng(rng)
    data = np.asarray(sampler((n_reps, n), rng), dtype=float)
    est = np.asarray(estimator(data), dtype=float)
    if est.shape != (n_reps,):
        raise ValueError(f"estimator returned shape {est.shape}, expected ({n_reps},): one estimate per row")
    return est


@app.function
def evaluate(estimator, sampler, theta: float, n: int, n_reps: int, *, rng=None) -> dict:
    """stage 2: bias, variance and MSE of estimator at sample size n, from n_reps simulated datasets.

    returns bias_variance_mse(simulate(...), theta) plus one more float entry:
      "mse_se": the Monte Carlo standard error of the "mse" entry,
                std((estimates - theta) ** 2, ddof=1) / sqrt(n_reps)
    """
    est = simulate(estimator, sampler, n, n_reps, rng=rng)
    out = bias_variance_mse(est, theta)
    sq = (est - theta) ** 2
    out["mse_se"] = float(np.std(sq, ddof=1) / np.sqrt(n_reps))
    return out


@app.cell(hide_code=True)
def _():
    checks.run(2, simulate=simulate, evaluate=evaluate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def theta_mom(x) -> np.ndarray:
    """stage 3: method-of-moments estimate of theta for Uniform(0, theta): 2 * mean, along the LAST axis.

    x (n,) -> float; x (n_reps, n) -> (n_reps,).
    """
    return 2.0 * np.mean(np.asarray(x, dtype=float), axis=-1)


@app.function
def theta_max(x) -> np.ndarray:
    """stage 3: the maximum-likelihood estimate of theta for Uniform(0, theta): the sample max, along the last axis."""
    return np.max(np.asarray(x, dtype=float), axis=-1)


@app.function
def theta_unbiased_max(x) -> np.ndarray:
    """stage 3: the bias-corrected max, (n + 1) / n * max, along the last axis (n = x.shape[-1])."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    return (n + 1) / n * np.max(x, axis=-1)


@app.cell(hide_code=True)
def _():
    checks.run(3, theta_mom=theta_mom, theta_max=theta_max, theta_unbiased_max=theta_unbiased_max)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def scaled_variance(x, c) -> np.ndarray:
    """stage 4: sum of squared deviations from the sample mean, divided by c, along the last axis.

    c = n - 1 is the unbiased sample variance, c = n the MLE. x (n,) -> float; (n_reps, n) -> (n_reps,).
    """
    x = np.asarray(x, dtype=float)
    dev = x - x.mean(axis=-1, keepdims=True)
    return np.sum(dev**2, axis=-1) / c


@app.function
def scaled_variance_mse(n: int, c, sigma2: float = 1.0):
    """stage 4: the exact MSE of scaled_variance(x, c) for n i.i.d. N(mu, sigma2) draws.

    the sum of squared deviations is sigma2 times a chi-squared with n - 1 degrees of freedom
    (mean n - 1, variance 2(n - 1)). returns bias**2 + variance; c may be an array.
    """
    c = np.asarray(c, dtype=float)
    bias = sigma2 * ((n - 1) / c - 1.0)
    var = 2.0 * sigma2**2 * (n - 1) / c**2
    out = bias**2 + var
    return float(out) if out.ndim == 0 else out


@app.cell(hide_code=True)
def _():
    checks.run(4, scaled_variance=scaled_variance, scaled_variance_mse=scaled_variance_mse)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def mse_curve(estimator, sampler, theta: float, ns, n_reps: int, *, rng=None) -> np.ndarray:
    """stage 5: the Monte Carlo MSE of estimator at each sample size in ns. returns (len(ns),).

    use your evaluate (or simulate) once per n, with ONE generator (as_rng(rng)) shared by all n.
    """
    rng = as_rng(rng)
    return np.array([evaluate(estimator, sampler, theta, int(n), n_reps, rng=rng)["mse"] for n in ns])


@app.function
def convergence_rate(ns, values) -> float:
    """stage 5: the slope of the least-squares line through (log n, log value).

    if value ~ C n^(-r), this returns about -r: -1 for an MSE that falls like 1/n.
    """
    log_n = np.log(np.asarray(ns, dtype=float))
    log_v = np.log(np.asarray(values, dtype=float))
    slope, _ = np.polyfit(log_n, log_v, 1)
    return float(slope)


@app.cell(hide_code=True)
def _():
    checks.run(5, mse_curve=mse_curve, convergence_rate=convergence_rate)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`) and use your harness to
    catch biased estimators beating unbiased ones, including one that shouldn't be able to.
    """)
    return


if __name__ == "__main__":
    app.run()
