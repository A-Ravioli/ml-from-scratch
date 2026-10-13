import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def zscores_a(x):
    """candidate A: z-scores (x - mean) / std."""
    return (x - x.mean()) / x.std()


@app.function
def zscores_b(x):
    """candidate B."""
    return (x - x.mean()) / x.std(ddof=1)


@app.function
def zscores_c(x):
    """candidate C."""
    return (x - x[1:].mean()) / x.std()


@app.function
def distances_unclipped(X, Y):
    """pairwise distances via ||x||^2 - 2 x.y + ||y||^2, WITHOUT clipping at 0 (planted bug)."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.sum(X * X, 1)[:, None] - 2 * X @ Y.T + np.sum(Y * Y, 1)[None, :])


@app.function
def train_sgd(batch_size, seed, *, epochs=40, lr=0.01):
    """linear regression by minibatch SGD on a random 60/340 train/test split; returns test MSE.

    every random choice (the split, the initial weights, the shuffling) comes from as_rng(seed).
    """
    rng = ex.as_rng(seed)
    data_rng = np.random.default_rng(100)  # the dataset itself is fixed
    X = data_rng.normal(size=(400, 10))
    y = X @ data_rng.normal(size=10) + 2.0 * data_rng.normal(size=400)
    perm = rng.permutation(400)
    train, test = perm[:60], perm[60:]
    w = 0.1 * rng.normal(size=10)
    for _ in range(epochs):
        order = rng.permutation(train)
        for start in range(0, len(order), batch_size):
            b = order[start:start + batch_size]
            w -= lr * 2 * X[b].T @ (X[b] @ w - y[b]) / len(b)
    return float(np.mean((X[test] @ w - y[test]) ** 2))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # testing, debugging & reproducibility · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: spot the bug

    three colleagues each wrote a function to turn 100 exam scores into z-scores,
    $(x_i - \bar x)/\sigma$, with $\sigma$ the standard deviation (NumPy's default, dividing
    by $n$). here are their outputs, plotted on top of each other. at most one is correct.
    """)
    return


@app.cell
def _():
    scores = np.random.default_rng(0).normal(70, 12, size=100).round()
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    for _f, _name, _style in ((zscores_a, "A", "-"), (zscores_b, "B", "--"), (zscores_c, "C", ":")):
        _ax.plot(_f(scores), _style, lw=1.5, label=_name)
    _ax.set(xlabel="student", ylabel="z-score")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return (scores,)


@app.cell
def _():
    p1 = ui.predict("from the plot alone: which of A, B and C are buggy?")
    p1
    return (p1,)


@app.cell
def _(p1, scores):
    ui.gate(p1)
    _mean = sum(scores) / len(scores)  # a trusted reference, computed a different way
    _sd = (sum((s - _mean) ** 2 for s in scores) / len(scores)) ** 0.5
    _reference = np.array([(s - _mean) / _sd for s in scores])
    _out = []
    for _f, _name in ((zscores_a, "A"), (zscores_b, "B"), (zscores_c, "C")):
        try:
            ex.assert_close(_f(scores), _reference, name=f"candidate {_name}")
            _out.append(f"**candidate {_name}**: passes")
        except AssertionError as _e:
            _out.append(f"**candidate {_name}**: fails\n\n```\n{_e}\n```")
    mo.md("\n\n".join(_out))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** read the source of the three candidates at the top of this notebook. what
    exactly is each bug, and how big an error does it cause (relative to the z-scores)? why
    is it invisible in the plot? what did `assert_close` tell you that the plot couldn't?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a bug that hides from random tests

    a colleague's `pairwise_distances` uses the fast formula
    $\|x\|^2 - 2x\cdot y + \|y\|^2$ but forgot to clip at 0 before the square root (look at
    `distances_unclipped` above). we test the property "no `nan` anywhere off the diagonal"
    with your `find_counterexample`, using two input generators:

    - **plain**: 2 to 19 random points in 3-D, standard normal coordinates;
    - **edge cases**: the same, but 30% of the time the last point duplicates the first, and
      30% of the time every coordinate is shifted by $10^k$ for a random $k$ from 1 to 8.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 1,000 trials each, does the plain generator find the bug? how many trials does "
        "the edge-case generator need?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _no_nan(X):
        _D = distances_unclipped(X, X)
        np.fill_diagonal(_D, 0.0)
        return bool(np.all(np.isfinite(_D)))

    def _plain(rng):
        return rng.normal(size=(int(rng.integers(2, 20)), 3))

    _counter = {"n": 0}

    def _edgy(rng):
        _counter["n"] += 1
        _X = rng.normal(size=(int(rng.integers(2, 20)), 3))
        if rng.random() < 0.3:
            _X[-1] = _X[0]
        if rng.random() < 0.3:
            _X = _X + 10.0 ** int(rng.integers(1, 9))
        return _X

    _found_plain = ex.find_counterexample(_no_nan, _plain, n_trials=1000, rng=0)
    _found_edgy = ex.find_counterexample(_no_nan, _edgy, n_trials=1000, rng=0)
    if _found_edgy is None:
        _edgy_text = "nothing in 1,000 trials"
    else:
        _D = distances_unclipped(_found_edgy, _found_edgy)
        _edgy_text = (f"a counterexample on trial **{_counter['n']}**: {len(_found_edgy)} points, "
                      f"coordinates around {np.abs(_found_edgy).max():.3g}, with "
                      f"{int(np.isnan(_D).sum() - np.isnan(np.diag(_D)).sum())} nan distances off the diagonal")
    mo.md(f"""
    - **plain generator:** {"nothing found in 1,000 trials" if _found_plain is None else "found one"}
    - **edge-case generator:** {_edgy_text}

    run it again with the same seed (`rng=0`) and you get exactly the same counterexample.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what has to be true of an input for this bug to produce a `nan`? why can
    standard normal points essentially never trigger it, no matter how many you draw? what
    would you add to the generator of your own random tests from now on?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: is the change really better?

    you train a small linear regression with minibatch SGD and try a change: batch size 5
    instead of 10. every random choice (the train/test split, the initial weights, the
    order of the minibatches) comes from one seed through your `as_rng`. pick a seed with
    the slider, and compare the test error of the two settings on that seed.
    """)
    return


@app.cell
def _():
    seed = mo.ui.slider(0, 19, value=4, step=1, label="seed")
    seed
    return (seed,)


@app.cell
def _(seed):
    _a, _b = train_sgd(10, seed.value), train_sgd(5, seed.value)
    _winner = "batch size 5" if _b < _a else "batch size 10"
    mo.md(f"seed {seed.value}: test MSE with batch size 10 = **{_a:.3f}**, with batch size 5 = **{_b:.3f}**. "
          f"**{_winner}** wins by {abs(_a - _b) / max(_a, _b):.1%}.")
    return


@app.cell
def _():
    p3 = ui.predict(
        "across 20 seeds, in what fraction does batch size 5 beat batch size 10? how does the "
        "difference between the settings compare with the spread across seeds?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _seeds = np.arange(20)
    _res = {bs: np.array([train_sgd(bs, s) for s in _seeds]) for bs in (20, 10, 5)}
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    for _i, (_bs, _v) in enumerate(_res.items()):
        _ax.scatter(np.full(20, _i) + np.random.default_rng(_i).uniform(-0.12, 0.12, 20), _v, s=18, alpha=0.7)
        _ax.errorbar(_i + 0.3, _v.mean(), yerr=_v.std(ddof=1), fmt="ko", capsize=4)
    _ax.set_xticks(range(3), [f"batch {b}" for b in _res])
    _ax.set(ylabel="test MSE (one dot per seed)")
    _w_5_10 = np.mean(_res[5] < _res[10])
    _w_10_20 = np.mean(_res[10] < _res[20])
    mo.vstack([_fig, mo.md(
        "| batch size | mean test MSE | std across seeds |\n|---|---|---|\n"
        + "\n".join(f"| {b} | {v.mean():.3f} | {v.std(ddof=1):.3f} |" for b, v in _res.items())
        + f"\n\nbatch 5 beats batch 10 on **{_w_5_10:.0%}** of seeds; batch 10 beats batch 20 on "
        f"**{_w_10_20:.0%}** of seeds (same seed = same split, same initial weights).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** if you had run only your slider's seed, what would you have concluded?
    which of the two comparisons (5 vs 10, 10 vs 20) is a real effect, and how can you tell?
    where does most of the seed-to-seed variation come from here? (hint: what does the seed
    control besides the weights?)
    """)
    return


if __name__ == "__main__":
    app.run()
