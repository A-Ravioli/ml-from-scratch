import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # limit theorems · build

    work through the stages in order. you're building a small simulator for the two great
    limit theorems: the law of large numbers (averages settle down) and the central limit
    theorem (the way they settle down is Gaussian). a "sampler" here is any function
    `sampler(size, rng)` that returns an array of i.i.d. draws with shape `size`, for example
    `lambda size, rng: rng.exponential(1.0, size)` (NumPy's samplers are fair game now that
    you've built your own in topic 02).

    | stage | you build |
    |---|---|
    | 1 | `running_mean`, `sample_means`: one path of averages, and many averages at once |
    | 2 | `normal_cdf`, `ks_distance`: how far a sample is from a standard normal |
    | 3 | `clt_distances`, `berry_esseen_bound`: the CLT, measured and bounded |
    | 4 | `deviation_probability`: the weak law, measured |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def running_mean(x) -> np.ndarray:
    """stage 1: the running averages of x (n,): out[k] = mean of x[0], ..., x[k]. shape (n,), float."""
    raise NotImplementedError("stage1: implement running_mean")


@app.function
def sample_means(sampler, n: int, n_reps: int, *, rng=None) -> np.ndarray:
    """stage 1: n_reps independent sample means, each of n i.i.d. draws. returns (n_reps,).

    draw one (n_reps, n) array with sampler((n_reps, n), rng) and average each row.
    rng is normalized with np.random.default_rng(rng) before it's passed to the sampler.
    """
    raise NotImplementedError("stage1: implement sample_means")


@app.cell(hide_code=True)
def _():
    checks.run(1, running_mean=running_mean, sample_means=sample_means)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def normal_cdf(z):
    """stage 2: the standard normal CDF Phi(z) = (1 + erf(z / sqrt 2)) / 2, elementwise.

    z is a number or an array; returns a float or an array of the same shape. (math.erf works
    on one number at a time.)
    """
    raise NotImplementedError("stage2: implement normal_cdf")


@app.function
def ks_distance(z) -> float:
    """stage 2: the Kolmogorov–Smirnov distance sup_t |F_n(t) - Phi(t)| between the empirical CDF
    F_n of the sample z (n,) and the standard normal CDF.

    F_n jumps by 1/n at each sorted point z_(1) <= ... <= z_(n), so the sup is attained just
    after or just before a jump: max over i of max(i/n - Phi(z_(i)), Phi(z_(i)) - (i-1)/n),
    with i = 1..n. returns a Python float.
    """
    raise NotImplementedError("stage2: implement ks_distance")


@app.cell(hide_code=True)
def _():
    checks.run(2, normal_cdf=normal_cdf, ks_distance=ks_distance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def clt_distances(sampler, mu: float, sigma: float, ns, n_reps: int, *, rng=None) -> np.ndarray:
    """stage 3: for each n in ns, how far the standardized sample mean is from N(0, 1).

    for each n: draw n_reps sample means (your sample_means), standardize them as
    z = sqrt(n) (mean - mu) / sigma, and record ks_distance(z). returns an array (len(ns),).
    use ONE generator (np.random.default_rng(rng)) for all n, so the result is reproducible.
    """
    raise NotImplementedError("stage3: implement clt_distances")


@app.function
def berry_esseen_bound(n, sigma: float, rho: float, C: float = 0.4748):
    """stage 3: the Berry–Esseen bound C rho / (sigma^3 sqrt(n)) on the KS distance between the
    standardized mean of n i.i.d. draws and N(0, 1). rho = E|X - mu|^3. n may be an array.
    """
    raise NotImplementedError("stage3: implement berry_esseen_bound")


@app.cell(hide_code=True)
def _():
    checks.run(3, clt_distances=clt_distances, berry_esseen_bound=berry_esseen_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def deviation_probability(sampler, mu: float, eps: float, ns, n_reps: int, *, rng=None) -> np.ndarray:
    """stage 4: Monte Carlo estimates of P(|mean of n draws - mu| > eps) (strictly greater), one per n in ns.

    for each n, draw n_reps sample means (your sample_means) and return the fraction whose
    distance from mu exceeds eps. returns an array (len(ns),). one generator for all n.
    """
    raise NotImplementedError("stage4: implement deviation_probability")


@app.cell(hide_code=True)
def _():
    checks.run(4, deviation_probability=deviation_probability, clt_distances=clt_distances)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`) and find out where these
    theorems stop working.
    """)
    return


if __name__ == "__main__":
    app.run()
