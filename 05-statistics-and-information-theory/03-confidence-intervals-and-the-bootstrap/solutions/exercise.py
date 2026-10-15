import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from statistics import NormalDist

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # confidence intervals & the bootstrap · build (reference solution)

    work through the stages in order. you'll build the classic normal-theory interval, a
    simulator that measures how often an interval *actually* contains the truth, and then the
    bootstrap: confidence intervals for any statistic, computed by resampling your own data.
    `bootstrap_ci` becomes part of `mlfs.stats`, and from chapter 07 on it puts the error bar
    on every number you report.

    `NormalDist` (from Python's standard library) gives you the standard normal CDF
    `NormalDist().cdf(z)` and its inverse `NormalDist().inv_cdf(p)`, which you built by hand
    in ch 04.

    | stage | you build |
    |---|---|
    | 1 | `normal_ci`: mean ± z · standard error |
    | 2 | `coverage`: how often an interval method contains the truth, and how wide it is |
    | 3 | `bootstrap_replicates`, `bootstrap_ci` (percentile): resample, recompute, read off quantiles |
    | 4 | `jackknife`, `bca_interval`, `bootstrap_ci(method="bca")`: correcting the percentile interval's bias and skew |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def normal_ci(x, alpha: float = 0.05) -> tuple:
    """stage 1: the normal-theory confidence interval for the mean of x (n,), n >= 2.

    xbar ± z * s / sqrt(n), with s the sample standard deviation (ddof=1) and
    z = NormalDist().inv_cdf(1 - alpha / 2) (1.96 for alpha = 0.05). returns (lo, hi) floats.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    half = z * np.std(x, ddof=1) / np.sqrt(n)
    m = float(np.mean(x))
    return m - float(half), m + float(half)


@app.cell(hide_code=True)
def _():
    checks.run(1, normal_ci=normal_ci)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def coverage(ci_fn, sampler, theta: float, n: int, n_reps: int, *, rng=None) -> tuple:
    """stage 2: estimate the actual coverage and the mean width of an interval method.

    draw data = sampler((n_reps, n), as_rng(rng)) (one dataset per row), call ci_fn(row) -> (lo, hi)
    on every row, and return (fraction of intervals with lo <= theta <= hi, mean of hi - lo), as floats.
    """
    rng = as_rng(rng)
    data = np.asarray(sampler((n_reps, n), rng), dtype=float)
    hits, widths = 0, 0.0
    for row in data:
        lo, hi = ci_fn(row)
        hits += lo <= theta <= hi
        widths += hi - lo
    return float(hits / n_reps), float(widths / n_reps)


@app.cell(hide_code=True)
def _():
    checks.run(2, coverage=coverage, normal_ci=normal_ci)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def bootstrap_replicates(data, statistic, n_boot: int, *, rng=None) -> np.ndarray:
    """stage 3: n_boot bootstrap replicates of statistic. returns float (n_boot,).

    data is one array (resampled along axis 0) or a tuple of equal-length arrays, resampled
    JOINTLY: each replicate draws one set of n row indices with replacement and applies it to
    every array. statistic(*resampled) -> float (one argument per array).
    """
    rng = as_rng(rng)
    arrays = data if isinstance(data, tuple) else (data,)
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays in data must have the same length")
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, size=n)  # one index set per replicate: memory O(n), not O(n_boot * n)
        out[b] = statistic(*(a[i] for a in arrays))
    return out


@app.function
def bootstrap_ci(data, statistic, *, n_boot=10000, alpha=0.05, method='percentile', rng=None) -> tuple:
    """stage 3: a bootstrap confidence interval for statistic(*data). returns (lo, hi) floats.

    data: one array, or a tuple of equal-length arrays resampled jointly (see bootstrap_replicates).
    replicates = bootstrap_replicates(data, statistic, n_boot, rng=rng), called once.
    method "percentile": the alpha/2 and 1 - alpha/2 quantiles (np.quantile) of the replicates.
    method "bca": bca_interval(statistic(*data), replicates, jackknife(data, statistic), alpha) (stage 4).
    any other method raises ValueError.
    """
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}")
    reps = bootstrap_replicates(data, statistic, n_boot, rng=rng)
    if method == "percentile":
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)
    arrays = data if isinstance(data, tuple) else (data,)
    theta_hat = float(statistic(*(np.asarray(a) for a in arrays)))
    return bca_interval(theta_hat, reps, jackknife(data, statistic), alpha)


@app.cell(hide_code=True)
def _():
    checks.run(3, bootstrap_replicates=bootstrap_replicates, bootstrap_ci=bootstrap_ci)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def jackknife(data, statistic) -> np.ndarray:
    """stage 4: the leave-one-out values of statistic. returns float (n,): entry i is statistic
    computed with row i removed (from every array, if data is a tuple)."""
    arrays = data if isinstance(data, tuple) else (data,)
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    return np.array([statistic(*(np.delete(a, i, axis=0) for a in arrays)) for i in range(n)], dtype=float)


@app.function
def bca_interval(theta_hat: float, replicates, jack, alpha: float = 0.05) -> tuple:
    """stage 4: the BCa interval from the replicates (B,) and the jackknife values (n,).

    bias correction: z0 = Phi^-1(p0), with p0 = (#{reps < theta_hat} + #{reps <= theta_hat}) / (2B)
    (ties count half), clipped to [1/(2B), 1 - 1/(2B)].
    acceleration: a = sum(d^3) / (6 * sum(d^2)^1.5) with d = mean(jack) - jack (a = 0 if all d are 0).
    for z = Phi^-1(alpha/2) and Phi^-1(1 - alpha/2): level = Phi(z0 + (z0 + z) / (1 - a (z0 + z))).
    returns the replicates' np.quantile at those two levels, (lo, hi), as floats.
    """
    reps = np.asarray(replicates, dtype=float)
    B = len(reps)
    p0 = (np.sum(reps < theta_hat) + np.sum(reps <= theta_hat)) / (2 * B)
    p0 = min(max(p0, 1 / (2 * B)), 1 - 1 / (2 * B))
    nd = NormalDist()
    z0 = nd.inv_cdf(p0)
    d = np.mean(jack) - np.asarray(jack, dtype=float)
    denom = 6 * np.sum(d**2) ** 1.5
    a = float(np.sum(d**3) / denom) if denom > 0 else 0.0
    levels = []
    for z in (nd.inv_cdf(alpha / 2), nd.inv_cdf(1 - alpha / 2)):
        levels.append(nd.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    lo, hi = np.quantile(reps, levels)
    return float(lo), float(hi)


@app.cell(hide_code=True)
def _():
    checks.run(4, jackknife=jackknife, bca_interval=bca_interval, bootstrap_ci=bootstrap_ci,
               bootstrap_replicates=bootstrap_replicates)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`) and measure how often your
    "95%" intervals really contain the truth.
    """)
    return


if __name__ == "__main__":
    app.run()
