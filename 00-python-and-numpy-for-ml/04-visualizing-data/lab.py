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

    # Anscombe's quartet (Anscombe, 1973)
    ANSCOMBE_X = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
    ANSCOMBE = {
        "I": (ANSCOMBE_X, np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])),
        "II": (ANSCOMBE_X, np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])),
        "III": (ANSCOMBE_X, np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])),
        "IV": (np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float),
               np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # visualizing data · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    the theme: **one dataset, four plots, four different stories.**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how many peaks?

    here are 300 measurements (say, the reaction times of 300 people, standardized). before
    plotting anything: you will see histograms with 4, 12 and 150 bins, all drawn with your
    `histogram`, and a slider to try your own.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how many peaks (modes) will each of the three histograms show? which one shows the "
        "truth about the data?"
    )
    p1
    return (p1,)


@app.cell
def _():
    n_bins = mo.ui.slider(2, 150, value=25, step=1, label="bins (your choice)")
    n_bins
    return (n_bins,)


@app.cell
def _(n_bins, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _side = _rng.random(300) < 0.5
    _data = np.where(_side, -1.0, 1.0) + 0.55 * _rng.normal(size=300)
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 2.8))
    for _ax, _b in zip(_axes, (4, 12, 150, n_bins.value)):
        _counts, _edges = ex.histogram(_data, _b)
        _ax.stairs(_counts, _edges, fill=True, alpha=0.8)
        _ax.set_title(f"{_b} bins", fontsize=9)
        _ax.set_yticks([])
    _axes[-1].set_title(f"{n_bins.value} bins (slider)", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the data really comes from two overlapping groups. which histograms show
    that, and which hide it? for the 150-bin histogram, how many values does a typical bin
    expect, and how much should its count fluctuate? what would you do in practice to decide
    whether a peak is real?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: which model is better?

    two models trained for 2,000 steps. here are their training losses, drawn with your
    `plot_curves` on an ordinary linear y axis. tick the box to switch to a log y axis.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "looking at the linear plot only: which model would you ship, A or B? by roughly how "
        "much does the better one beat the other at step 2,000?"
    )
    p2
    return (p2,)


@app.cell
def _():
    log_axis = mo.ui.checkbox(label="log y axis", value=False)
    log_axis
    return (log_axis,)


@app.cell
def _(log_axis, p2):
    ui.gate(p2)
    _t = np.arange(2001)
    _noise = np.random.default_rng(1).normal(size=(2, 2001))
    _a = (0.03 + 0.97 * np.exp(-_t / 80)) * np.exp(0.03 * _noise[0])
    _b = (1.5 / (1 + _t / 20)) * np.exp(0.03 * _noise[1])
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    ex.plot_curves({"model A": _a, "model B": _b}, ax=_ax, ylabel="training loss", logy=log_axis.value,
                   title="log y axis" if log_axis.value else "linear y axis")
    mo.vstack([_fig, mo.md(f"final losses (mean of the last 100 steps): A **{_a[-100:].mean():.4f}**, "
                           f"B **{_b[-100:].mean():.4f}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does the linear axis hide, and why? on the log axis, describe the
    shape of each curve. which model is still improving at step 2,000, and what would you
    expect if you trained both for 10,000 steps?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: four datasets, one set of statistics

    four small datasets, I to IV, with 11 points each. here are their summary statistics:
    the means and variances of $x$ and $y$, the correlation between them, and the
    least-squares line $y = a + bx$.
    """)
    return


@app.cell
def _():
    _rows = []
    for _name, (_x, _y) in ANSCOMBE.items():
        _b = np.cov(_x, _y)[0, 1] / np.var(_x, ddof=1)
        _a = _y.mean() - _b * _x.mean()
        _rows.append(f"| {_name} | {_x.mean():.2f} | {np.var(_x, ddof=1):.2f} | {_y.mean():.2f} | "
                     f"{np.var(_y, ddof=1):.2f} | {np.corrcoef(_x, _y)[0, 1]:.3f} | y = {_a:.2f} + {_b:.3f} x |")
    mo.md("| dataset | mean x | var x | mean y | var y | correlation | fitted line |\n"
          "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _():
    p3 = ui.predict(
        "from the table alone: how different do you expect the four scatter plots to look? "
        "sketch what you think dataset I looks like."
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3), sharex=True, sharey=True)
    _line_x = np.array([2.0, 20.0])
    for _ax, (_name, (_x, _y)) in zip(_axes, ANSCOMBE.items()):
        _b = np.cov(_x, _y)[0, 1] / np.var(_x, ddof=1)
        _a = _y.mean() - _b * _x.mean()
        _ax.scatter(_x, _y, s=25, edgecolors="k")
        _ax.plot(_line_x, _a + _b * _line_x, "C1-", lw=1)
        _ax.set_title(f"dataset {_name}", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for each dataset, say in one sentence what's going on, and whether the
    fitted line is a sensible model. which single point controls the correlation in dataset
    IV? what does this say about reporting a correlation (or an accuracy) without a plot?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 4: a perfect classifier?

    two classes of noisy 2-D points that overlap. a **k-nearest-neighbour** classifier
    predicts the majority label among the $k$ training points closest to the query. with
    $k = 1$ it predicts the label of the single closest training point, so it gets every
    training point right. your `plot_decision_boundary` shows what it does everywhere else.
    """)
    return


@app.cell
def _():
    p4 = ui.predict(
        "with k = 1 (100% training accuracy), what does the decision boundary look like? "
        "what changes when you raise k to 25?"
    )
    p4
    return (p4,)


@app.cell
def _():
    k_nn = mo.ui.slider(1, 51, value=1, step=2, label="k (neighbours)")
    k_nn
    return (k_nn,)


@app.cell
def _(k_nn, p4):
    ui.gate(p4)
    _rng = np.random.default_rng(3)
    _y = np.repeat([0, 1], 80)
    _X = np.where(_y[:, None] == 0, [-0.6, 0.0], [0.6, 0.0]) + _rng.normal(size=(160, 2))
    _Xt = np.where(np.repeat([0, 1], 500)[:, None] == 0, [-0.6, 0.0], [0.6, 0.0]) + _rng.normal(size=(1000, 2))
    _yt = np.repeat([0, 1], 500)

    def _knn(P, k=k_nn.value):
        _d = np.sum(P**2, 1)[:, None] - 2 * P @ _X.T + np.sum(_X**2, 1)[None, :]
        _nearest = np.argsort(_d, axis=1)[:, :k]
        return (_y[_nearest].mean(axis=1) > 0.5).astype(int)

    _fig, _ax = plt.subplots(figsize=(5.5, 4.5))
    ex.plot_decision_boundary(_knn, _X, _y, ax=_ax, resolution=150)
    _ax.set_title(f"k = {k_nn.value}", fontsize=9)
    mo.vstack([_fig, mo.md(f"training accuracy **{np.mean(_knn(_X) == _y):.3f}** · accuracy on 1,000 new "
                           f"points from the same distribution **{np.mean(_knn(_Xt) == _yt):.3f}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does $k = 1$ score perfectly on the training data but worse on new
    points? point to the "islands" in the plot: what training points created them? which
    plot would you trust, and what does that suggest about judging a model by its training
    accuracy? (chapters 07 and 09 return to this.)
    """)
    return


if __name__ == "__main__":
    app.run()
