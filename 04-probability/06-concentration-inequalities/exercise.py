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
    # concentration inequalities · build

    work through the stages in order. each function is a **bound**: a guaranteed upper limit
    on the probability that a random quantity strays far from its mean, valid for every
    distribution satisfying the stated assumptions. probabilities never exceed 1, so every
    bound here is clipped to 1. every function works elementwise when `t` (or `n`) is an array.

    | stage | you build |
    |---|---|
    | 1 | `markov_bound`, `chebyshev_bound`: bounds from one or two moments |
    | 2 | `hoeffding_bound`, `hoeffding_sample_size`: exponential bounds for bounded variables |
    | 3 | `bernstein_bound`: exploit a small variance |
    | 4 | `chernoff_bernoulli_bound`, `empirical_tail`: the tightest exponential bound for coins, and the truth to compare with |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def markov_bound(mean, t):
    """stage 1: Markov's inequality for a NON-NEGATIVE X with E[X] = mean: P(X >= t) <= mean / t.

    t > 0 (number or array). returns min(1, mean / t), elementwise.
    """
    raise NotImplementedError("stage1: implement markov_bound")


@app.function
def chebyshev_bound(var, t):
    """stage 1: Chebyshev's inequality: P(|X - E[X]| >= t) <= var / t^2.

    var = Var(X), t > 0 (number or array). returns min(1, var / t^2), elementwise.
    """
    raise NotImplementedError("stage1: implement chebyshev_bound")


@app.cell(hide_code=True)
def _():
    checks.run(1, markov_bound=markov_bound, chebyshev_bound=chebyshev_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def hoeffding_bound(n, t, a: float = 0.0, b: float = 1.0):
    """stage 2: Hoeffding's inequality for the mean of n independent draws in [a, b]:

    P(|mean - mu| >= t) <= 2 exp(-2 n t^2 / (b - a)^2). returns min(1, ...), elementwise in n and t.
    """
    raise NotImplementedError("stage2: implement hoeffding_bound")


@app.function
def hoeffding_sample_size(t: float, delta: float, a: float = 0.0, b: float = 1.0) -> int:
    """stage 2: the smallest n for which hoeffding_bound(n, t, a, b) <= delta, as a Python int.

    solve 2 exp(-2 n t^2 / (b - a)^2) <= delta for n and round UP.
    """
    raise NotImplementedError("stage2: implement hoeffding_sample_size")


@app.cell(hide_code=True)
def _():
    checks.run(2, hoeffding_bound=hoeffding_bound, hoeffding_sample_size=hoeffding_sample_size)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def bernstein_bound(n, t, var: float, c: float):
    """stage 3: Bernstein's inequality for the mean of n independent draws with variance var and
    |X - mu| <= c almost surely:

    P(|mean - mu| >= t) <= 2 exp(-n t^2 / (2 var + 2 c t / 3)). returns min(1, ...), elementwise.
    """
    raise NotImplementedError("stage3: implement bernstein_bound")


@app.cell(hide_code=True)
def _():
    checks.run(3, bernstein_bound=bernstein_bound, hoeffding_bound=hoeffding_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def chernoff_bernoulli_bound(n, p: float, t):
    """stage 4: the Chernoff bound for the mean of n independent Bernoulli(p) draws (ONE-sided):

    P(mean >= p + t) <= exp(-n KL(p + t || p)), with KL(q || p) = q log(q/p) + (1-q) log((1-q)/(1-p)).
    0 < p < 1 and t > 0. when p + t > 1 the event is impossible: return 0. when p + t == 1,
    KL = log(1/p) (use 0 log 0 = 0). elementwise in n and t.
    """
    raise NotImplementedError("stage4: implement chernoff_bernoulli_bound")


@app.function
def empirical_tail(samples, mu: float, t, two_sided: bool = True):
    """stage 4: the fraction of samples (m,) with |sample - mu| >= t (two_sided=True) or
    sample - mu >= t (two_sided=False), for each t (a number or an array of thresholds).

    returns a float for a scalar t, else an array shaped like t.
    """
    raise NotImplementedError("stage4: implement empirical_tail")


@app.cell(hide_code=True)
def _():
    checks.run(4, chernoff_bernoulli_bound=chernoff_bernoulli_bound, empirical_tail=empirical_tail,
               hoeffding_bound=hoeffding_bound)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`) and find out how loose each
    bound is in practice.
    """)
    return


if __name__ == "__main__":
    app.run()
