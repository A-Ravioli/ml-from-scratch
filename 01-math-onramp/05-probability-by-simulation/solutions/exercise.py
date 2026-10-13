import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # probability by simulation · build (reference solution)

    some probabilities you can count exactly; most you can't. but you can almost always
    *simulate*: let the computer play the game a hundred thousand times and count. in this
    notebook you do both, and you learn the most important habit in experimental science:
    every estimate comes with an **error bar**. `mc_estimate` becomes part of `mlfs.numerics`.

    every function has step-by-step comments inside it: replace its last line (the one that
    raises "not implemented") with code that follows the steps. randomness always comes from
    a NumPy random generator: functions take `rng` (None, a seed like 0, or a
    `np.random.Generator`) and turn it into a generator with `np.random.default_rng(rng)`,
    so the same seed always gives the same results.

    | stage | you build |
    |---|---|
    | 1 | `dice_sum_pmf`, `roll_dice`: exact probabilities by counting, and simulated dice |
    | 2 | `empirical_pmf`, `expectation`, `variance`: frequencies and averages |
    | 3 | `mean_and_se`: an estimate and its standard error |
    | 4 | `mc_estimate`: Monte Carlo estimates of any expectation, with error bars |
    | 5 | `estimate_pi`: $\pi$ by throwing darts |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def dice_sum_pmf(n_dice=2):
    """stage 1: the exact distribution of the sum of n_dice fair six-sided dice, by counting.

    list every possible outcome (6 ** n_dice equally likely tuples like (3, 5)), count how
    many give each sum, and divide by the number of outcomes.
    returns (sums, probs): sums = all possible sums in increasing order (int array, from
    n_dice to 6 * n_dice), probs = their probabilities (float array, adds up to 1).
    example: dice_sum_pmf(2) gives sums [2, ..., 12] and probs [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1] / 36
    """
    # step 1: sums = np.arange(n_dice, 6 * n_dice + 1), and counts = np.zeros(len(sums)).
    # step 2: for outcome in itertools.product(range(1, 7), repeat=n_dice):   (every tuple of faces)
    #             s = sum(outcome); add 1 to the count for s, which sits at index s - n_dice.
    # step 3: probs = counts / 6 ** n_dice.  return sums, probs.
    sums = np.arange(n_dice, 6 * n_dice + 1)
    counts = np.zeros(len(sums))
    for outcome in itertools.product(range(1, 7), repeat=n_dice):
        counts[sum(outcome) - n_dice] += 1
    return sums, counts / 6**n_dice


@app.function
def roll_dice(n, n_dice=2, *, rng=None):
    """stage 1: simulate n throws of n_dice fair six-sided dice.

    returns an int array of shape (n, n_dice), every entry in {1, 2, 3, 4, 5, 6}.
    rng: None, an int seed, or a np.random.Generator. always draw through
    rng = np.random.default_rng(rng), never through np.random's global functions.
    careful: rng.integers(low, high) never returns `high` itself.
    """
    # step 1: rng = np.random.default_rng(rng).
    # step 2: return rng.integers(1, 7, size=(n, n_dice))   (7 is excluded, so 1..6).
    rng = np.random.default_rng(rng)
    return rng.integers(1, 7, size=(n, n_dice))


@app.cell(hide_code=True)
def _():
    checks.run(1, dice_sum_pmf=dice_sum_pmf, roll_dice=roll_dice)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def empirical_pmf(samples, values):
    """stage 2: the fraction of samples equal to each of the given values.

    samples: 1-D array of observed outcomes (n of them). values: 1-D array of the outcomes to
    count. returns a float array shaped like values: entry k = (number of samples equal to
    values[k]) / n.
    example: empirical_pmf([1, 2, 2, 3, 3, 3], [1, 2, 3, 4]) = [1/6, 2/6, 3/6, 0]
    """
    # step 1: samples = np.asarray(samples) and values = np.asarray(values).
    # step 2: for each value v, count np.sum(samples == v) and divide by len(samples).
    # step 3: return those fractions as a float array.
    samples = np.asarray(samples)
    values = np.asarray(values)
    return np.array([np.sum(samples == v) for v in values], dtype=float) / len(samples)


@app.function
def expectation(values, probs):
    """stage 2: the expected value (probability-weighted average) of a discrete distribution.

    E[X] = sum over k of values[k] * probs[k]. values, probs: 1-D arrays of the same length,
    probs adding up to 1. returns a float.
    example: a fair die, expectation([1, ..., 6], [1/6] * 6) = 3.5
    """
    # step 1: return float(np.sum(np.asarray(values, dtype=float) * np.asarray(probs, dtype=float))).
    return float(np.sum(np.asarray(values, dtype=float) * np.asarray(probs, dtype=float)))


@app.function
def variance(values, probs):
    """stage 2: the variance of a discrete distribution: the expected squared distance from the mean.

    Var[X] = sum over k of probs[k] * (values[k] - E[X])^2, with E[X] from your expectation.
    returns a float.
    example: a fair die has variance 35/12 = 2.9167
    """
    # step 1: mu = expectation(values, probs).
    # step 2: return the probability-weighted average of (values - mu) ** 2, as a float.
    values = np.asarray(values, dtype=float)
    mu = expectation(values, probs)
    return float(np.sum(np.asarray(probs, dtype=float) * (values - mu) ** 2))


@app.cell(hide_code=True)
def _():
    checks.run(2, empirical_pmf=empirical_pmf, expectation=expectation, variance=variance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def mean_and_se(x):
    """stage 3: the sample mean of x and its standard error. returns (mean, se), two floats.

    x: 1-D array of n >= 2 independent draws. the standard error estimates how far the sample
    mean typically is from the true mean:  se = s / sqrt(n),  where s is the SAMPLE standard
    deviation, s = sqrt( sum((x - mean)^2) / (n - 1) )  (note n - 1: np.std(x, ddof=1)).
    example: mean_and_se([1, 2, 3, 4]) = (2.5, 0.6455)
    """
    # step 1: x = np.asarray(x, dtype=float); n = len(x).
    # step 2: mean = x.mean().
    # step 3: s = np.std(x, ddof=1)   (ddof=1 divides by n - 1 instead of n).
    # step 4: return float(mean), float(s / np.sqrt(n)).
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(x.mean()), float(np.std(x, ddof=1) / np.sqrt(n))


@app.cell(hide_code=True)
def _():
    checks.run(3, mean_and_se=mean_and_se)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def mc_estimate(f, sample, n=10000, *, rng=None):
    """stage 4: a Monte Carlo estimate of E[f(X)], with its standard error.

    sample(n, rng) is a function that returns n random draws of X, using the generator rng.
    f maps an array of draws to an array of n values. returns (estimate, se), two floats:
    the mean of f over the draws, and its standard error (your mean_and_se).
    rng: None, an int seed, or a np.random.Generator; normalize it with
    np.random.default_rng(rng) and pass the GENERATOR on to sample.
    example: mc_estimate(lambda u: u**2, lambda n, rng: rng.random(n), rng=0) is about (0.333, 0.003)
    """
    # step 1: rng = np.random.default_rng(rng).
    # step 2: draws = sample(n, rng).
    # step 3: values = f(draws).
    # step 4: return mean_and_se(values).
    rng = np.random.default_rng(rng)
    draws = sample(n, rng)
    return mean_and_se(f(draws))


@app.cell(hide_code=True)
def _():
    checks.run(4, mc_estimate=mc_estimate, mean_and_se=mean_and_se)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def estimate_pi(n=10000, *, rng=None):
    """stage 5: estimate pi by throwing n darts uniformly at the square [-1, 1] x [-1, 1].

    the circle of radius 1 inside the square covers pi/4 of its area, so
    pi = 4 * P(dart lands in the circle) = E[ 4 * (x^2 + y^2 <= 1) ].
    use your mc_estimate with
      sample: n darts as an (n, 2) array, uniform on [-1, 1]: rng.uniform(-1, 1, size=(n, 2))
      f:      4.0 where x^2 + y^2 <= 1, else 0.0
    returns (estimate, se), two floats.
    """
    # step 1: define sample(n, rng) returning rng.uniform(-1.0, 1.0, size=(n, 2)).
    # step 2: define f(points) returning 4.0 * (points[:, 0]**2 + points[:, 1]**2 <= 1).
    # step 3: return mc_estimate(f, sample, n, rng=rng).
    def sample(m, gen):
        return gen.uniform(-1.0, 1.0, size=(m, 2))

    def f(points):
        return 4.0 * (points[:, 0] ** 2 + points[:, 1] ** 2 <= 1.0)

    return mc_estimate(f, sample, n, rng=rng)


@app.cell(hide_code=True)
def _():
    checks.run(5, estimate_pi=estimate_pi, mc_estimate=mc_estimate, mean_and_se=mean_and_se)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. you'll solve the
    birthday problem by simulation, find out what it costs to get one more digit of $\pi$,
    and catch error bars lying.
    """)
    return


if __name__ == "__main__":
    app.run()
