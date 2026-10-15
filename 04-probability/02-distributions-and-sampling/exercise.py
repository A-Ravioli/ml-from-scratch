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
    # distributions & sampling · build

    work through the stages in order. the rule for this whole notebook: **the only source of
    randomness is `rng.random()`**, uniform numbers in $[0, 1)$. every other distribution you
    build by transforming uniforms. (`rng` is a `np.random.Generator`; each function accepts
    `rng=None`, an int seed, or a Generator, and normalizes it with `np.random.default_rng(rng)`.)
    everything you write here becomes `mlfs.sampling`, which later chapters use to initialize
    networks, simulate data, and sample from models.

    | stage | you build |
    |---|---|
    | 1 | `uniform`, `inverse_cdf_sample`, `exponential`: the inverse-CDF method |
    | 2 | `bernoulli`, `categorical`, `poisson`: discrete distributions |
    | 3 | `box_muller`, `normal`: Gaussians from two uniforms |
    | 4 | `rejection_sample`: sample from any density you can bound |
    | 5 | `gamma`, `beta`: Marsaglia–Tsang, and a Beta from two Gammas |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def uniform(low: float = 0.0, high: float = 1.0, size=None, *, rng=None):
    """stage 1: Uniform(low, high) draws, by rescaling rng.random().

    size: None (return one Python float), an int, or a shape tuple (return a float array of
    that shape). low < high are scalars. raises ValueError unless low < high.
    """
    raise NotImplementedError("stage1: implement uniform")


@app.function
def inverse_cdf_sample(inv_cdf, size=None, *, rng=None):
    """stage 1: draws from the distribution whose inverse CDF is `inv_cdf`.

    feed uniforms in the OPEN interval (0, 1) through inv_cdf (vectorized: it takes and returns
    arrays). rng.random() lies in [0, 1) and can return exactly 0.0: replace any exact 0 by
    2^-54 (half the smallest positive value rng.random() returns) before calling inv_cdf.
    size None -> a Python float, else an array of shape `size`.
    """
    raise NotImplementedError("stage1: implement inverse_cdf_sample")


@app.function
def exponential(rate: float = 1.0, size=None, *, rng=None):
    """stage 1: Exponential(rate) draws (mean 1 / rate) by the inverse-CDF method.

    the CDF is F(x) = 1 - exp(-rate x) for x >= 0. raises ValueError unless rate > 0.
    size None -> float, else an array of shape `size`.
    """
    raise NotImplementedError("stage1: implement exponential")


@app.cell(hide_code=True)
def _():
    checks.run(1, uniform=uniform, inverse_cdf_sample=inverse_cdf_sample, exponential=exponential)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def bernoulli(p: float, size=None, *, rng=None):
    """stage 2: Bernoulli(p) draws: 1 with probability p, else 0, as ints.

    size None -> a Python int, else an int array of shape `size`. raises ValueError unless 0 <= p <= 1.
    """
    raise NotImplementedError("stage2: implement bernoulli")


@app.function
def categorical(probs, size=None, *, rng=None):
    """stage 2: draws of a category index in {0, ..., k-1} with P(i) = probs[i].

    probs: (k,) non-negative, summing to 1 (within 1e-8), else ValueError. a category with
    probability 0 must never be drawn. size None -> a Python int, else an int array of shape `size`.
    """
    raise NotImplementedError("stage2: implement categorical")


@app.function
def poisson(lam: float, size=None, *, rng=None):
    """stage 2: Poisson(lam) draws (counts; mean and variance both lam), as ints.

    lam <= 30: Knuth's method (count how many uniforms you can multiply before the product
    drops to exp(-lam) or below). lam > 30: categorical on the pmf table for k = 0 .. K with
    K = ceil(lam + 12 sqrt(lam) + 20) (the tail beyond K has probability < 1e-30;
    the table has O(lam) entries, built once per call).
    size None -> a Python int, else an int array of shape `size`. raises ValueError if lam < 0.
    """
    raise NotImplementedError("stage2: implement poisson")


@app.cell(hide_code=True)
def _():
    checks.run(2, bernoulli=bernoulli, categorical=categorical, poisson=poisson)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def box_muller(u1, u2):
    """stage 3: the Box–Muller transform of uniforms u1 in (0, 1] and u2 in [0, 1).

    returns (z1, z2) = (r cos(2 pi u2), r sin(2 pi u2)) with r = sqrt(-2 log u1): two
    independent standard normals. works elementwise on arrays of equal shape.
    """
    raise NotImplementedError("stage3: implement box_muller")


@app.function
def normal(loc: float = 0.0, scale: float = 1.0, size=None, *, rng=None):
    """stage 3: Normal(loc, scale^2) draws via Box–Muller (scale is the standard deviation).

    use BOTH outputs of each Box–Muller pair: n draws need ceil(n / 2) pairs. u1 must avoid 0
    (1 - rng.random() lies in (0, 1]). raises ValueError unless scale > 0.
    size None -> float, else an array of shape `size`.
    """
    raise NotImplementedError("stage3: implement normal")


@app.cell(hide_code=True)
def _():
    checks.run(3, box_muller=box_muller, normal=normal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def rejection_sample(log_target, sample_proposal, log_proposal, log_M: float, size: int, *, rng=None):
    """stage 4: exactly `size` draws from the density proportional to exp(log_target).

    proposals x come from `sample_proposal(n, rng)` (n draws, first axis n) and are accepted
    with probability exp(log_target(x) - log_M - log_proposal(x)), using one rng.random() each.
    log_target and log_proposal are vectorized. keep proposing in batches until `size` draws are
    accepted; return them in the order they were accepted, shape (size, ...).
    raises ValueError if some proposal has log_target(x) > log_M + log_proposal(x) (M too small:
    the envelope doesn't cover the target, so the samples would be wrong).
    """
    raise NotImplementedError("stage4: implement rejection_sample")


@app.cell(hide_code=True)
def _():
    checks.run(4, rejection_sample=rejection_sample)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def gamma(shape: float, scale: float = 1.0, size=None, *, rng=None):
    """stage 5: Gamma(shape, scale) draws (mean shape * scale, variance shape * scale^2).

    shape >= 1: Marsaglia–Tsang. with d = shape - 1/3 and c = 1 / sqrt(9 d), propose
    v = (1 + c z)^3 with z from YOUR normal; reject if v <= 0; accept if
    log(u) < z^2 / 2 + d - d v + d log v (u from YOUR uniform); the draw is d v.
    shape < 1: draw g ~ Gamma(shape + 1) and return g * u^(1 / shape) (u uniform in (0, 1)).
    multiply by scale at the end. raises ValueError unless shape > 0 and scale > 0.
    size None -> float, else an array of shape `size`.
    """
    raise NotImplementedError("stage5: implement gamma")


@app.function
def beta(a: float, b: float, size=None, *, rng=None):
    """stage 5: Beta(a, b) draws in (0, 1) (mean a / (a + b)) as X / (X + Y) with
    X ~ Gamma(a), Y ~ Gamma(b) independent, from your gamma. raises ValueError unless a, b > 0.
    size None -> float, else an array of shape `size`.
    """
    raise NotImplementedError("stage5: implement beta")


@app.cell(hide_code=True)
def _():
    checks.run(5, gamma=gamma, beta=beta, normal=normal, uniform=uniform)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your code is `mlfs.sampling` (try
    `from mlfs.sampling import normal` in a scratch cell). then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
