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
    # hypothesis testing · build

    work through the stages in order. you'll build a **permutation test**, the most general
    and assumption-light hypothesis test there is: to ask whether a difference could be
    chance, it rebuilds the world where it *is* chance by shuffling labels. then you'll
    measure its error rates by simulation, and control them when you run many tests at once.
    `permutation_test`, `bonferroni` and `benjamini_hochberg` become part of `mlfs.stats`.

    | stage | you build |
    |---|---|
    | 1 | `p_value`: how extreme an observed statistic is, against samples from the null |
    | 2 | `permutation_null`: the null distribution by shuffling (unpaired) or sign-flipping (paired) |
    | 3 | `permutation_test`, `rejection_rate`: the test itself, and its size and power by simulation |
    | 4 | `bonferroni`, `benjamini_hochberg`: many tests at once |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def p_value(observed: float, null_stats, alternative: str = 'two-sided') -> float:
    """stage 1: the Monte Carlo p-value (1 + #as-extreme) / (1 + B) of observed against null_stats (B,).

    "as extreme": null >= observed ('greater'), null <= observed ('less'), |null| >= |observed|
    ('two-sided': only meaningful for a statistic centered at 0 under the null, like a difference of means).
    compare with a tolerance tol = 1e-12 * max(1, |observed|) in the direction that
    counts near-ties as extreme (e.g. null >= observed - tol), so rounding can't break a tie.
    any other alternative raises ValueError.
    """
    raise NotImplementedError("stage1: implement p_value")


@app.cell(hide_code=True)
def _():
    checks.run(1, p_value=p_value)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def mean_difference(x, y) -> float:
    """the default test statistic: mean(x) - mean(y)."""
    return float(np.mean(x) - np.mean(y))


@app.function
def permutation_null(x, y, statistic=None, *, n_permutations=10000, paired=False, rng=None) -> np.ndarray:
    """stage 2: n_permutations draws of statistic(x', y') under the null hypothesis. returns float (B,).

    statistic defaults to mean_difference. unpaired: pool x and y, shuffle (rng.permutation), and split
    into the first len(x) values (x') and the rest (y'). paired (x, y equal length, else ValueError):
    independently for each i, swap x[i] and y[i] with probability 1/2 (for the mean difference that
    flips the sign of x[i] - y[i]).
    """
    raise NotImplementedError("stage2: implement permutation_null")


@app.cell(hide_code=True)
def _():
    checks.run(2, permutation_null=permutation_null)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def permutation_test(x, y, statistic=None, *, n_permutations=10000, alternative='two-sided', paired=False, rng=None) -> tuple:
    """stage 3: a permutation test of "x and y come from the same distribution" (unpaired) or
    "within each pair, x and y are exchangeable" (paired).

    returns (observed, p): observed = statistic(x, y) (default mean_difference), and p = your p_value
    of observed against permutation_null(...) with the same arguments. alternative='two-sided' counts
    |T| >= |t_obs|, so it needs a statistic centered at 0 under H0 (a difference, not a ratio); otherwise
    use 'greater' or 'less'.
    """
    raise NotImplementedError("stage3: implement permutation_test")


@app.function
def rejection_rate(test, simulate, n_sims: int, alpha: float = 0.05, *, rng=None) -> float:
    """stage 3: the fraction of n_sims simulated experiments in which test rejects at level alpha.

    each experiment: (x, y) = simulate(rng), then p = test(x, y, rng); count p <= alpha. one generator
    (as_rng(rng)) is shared by every call. under the null this estimates the size (should be <= alpha);
    under an alternative, the power.
    """
    raise NotImplementedError("stage3: implement rejection_rate")


@app.cell(hide_code=True)
def _():
    checks.run(3, permutation_test=permutation_test, rejection_rate=rejection_rate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def bonferroni(p_values, alpha=0.05) -> np.ndarray:
    """stage 4: reject hypothesis i when p_i <= alpha / m (m = number of p-values). returns a bool mask (m,).

    no p-values -> an empty mask; any NaN p-value raises ValueError.
    """
    raise NotImplementedError("stage4: implement bonferroni")


@app.function
def benjamini_hochberg(p_values, alpha=0.05) -> np.ndarray:
    """stage 4: the Benjamini-Hochberg step-up procedure. returns a bool mask (m,) in the original order.

    sort the p-values, find the LARGEST k with p_(k) <= k alpha / m, and reject the k hypotheses
    with the smallest p-values (none if no such k). no p-values -> an empty mask; NaN raises ValueError.
    """
    raise NotImplementedError("stage4: implement benjamini_hochberg")


@app.cell(hide_code=True)
def _():
    checks.run(4, bonferroni=bonferroni, benjamini_hochberg=benjamini_hochberg)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`) and build a machine for
    "discovering" effects in pure noise.
    """)
    return


if __name__ == "__main__":
    app.run()
