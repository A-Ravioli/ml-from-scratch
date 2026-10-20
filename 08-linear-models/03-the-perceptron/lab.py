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
def separable_with_gap(gap, rng, n=600, offset=0.1):
    """n points drawn uniformly in the unit disc, labelled by the line x1 + x2 = offset, with
    every point closer than `gap` to the line removed (so that line has margin >= gap).
    the same rng seed gives the same points, so a smaller gap only adds points."""
    w = np.array([1.0, 1.0]) / np.sqrt(2)
    b = -offset / np.sqrt(2)
    r, a = np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(a), r * np.sin(a)])
    X = X[np.abs(X @ w + b) >= gap]
    return X, np.where(X @ w + b > 0, 1, -1), w, b


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the perceptron · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the margin and the number of mistakes

    600 random points in the unit disc, labelled by a line, with every point closer than
    $\gamma$ to the line removed (so that line separates them with margin at least $\gamma$). for each
    $\gamma$ you train your `Perceptron` with 10 different shuffling seeds, count the updates
    it makes before an error-free epoch, and compare with your `mistake_bound` for the
    true line. as a control, the same experiment with the line moved to pass through the
    origin ($x_1 + x_2 = 0$, so the best separator needs no intercept).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "halving the margin quadruples Novikoff's bound. does it quadruple the number of updates "
        "the perceptron actually makes? how close to the bound does it get?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _gaps = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    _rows, _med, _bounds, _med0 = [], [], [], []
    for _g in _gaps:
        _X0, _t0, _, _ = separable_with_gap(_g, np.random.default_rng(0), offset=0.0)
        _med0.append(float(np.median([ex.Perceptron(max_epochs=10_000, rng=_s).fit(_X0, _t0).n_updates_
                                      for _s in range(10)])))
        _X, _t, _w, _b = separable_with_gap(_g, np.random.default_rng(0))
        _bound = ex.mistake_bound(_X, _t, _w, _b)
        _ups = [ex.Perceptron(max_epochs=10_000, rng=_s).fit(_X, _t).n_updates_ for _s in range(10)]
        _med.append(float(np.median(_ups)))
        _bounds.append(_bound)
        _rows.append(f"| {_g} | {len(_X)} | {ex.margin(_X, _t, _w, _b):.4f} | {_bound:,.0f} | {int(np.min(_ups))} – "
                     f"{int(np.max(_ups))} (median {np.median(_ups):.0f}) | {_med0[-1]:.0f} |")
    _fig, _ax = plt.subplots(figsize=(6.2, 3.8))
    _ax.loglog(_gaps, _bounds, "k--", label=r"bound $(R/\gamma)^2$")
    _ax.loglog(_gaps, _med, "o-", label="updates (median of 10 orders)")
    _ax.loglog(_gaps, _med0, "s:", label="control: line through the origin")
    _ax.set(xlabel=r"margin $\gamma$", ylabel="updates")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| gap | points | margin of the true line | bound | updates (10 orders) | control: median updates |\n|---|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the bound grow like $1/\gamma^2$, and does the measured count grow
    at the same rate? the bound is for the worst possible order of the points: why is the
    actual count so far below it? notice that the bound doesn't depend on the number of points
    or (directly) on the dimension: what does? and why does the control, whose best line passes
    through the origin, have no jump between gaps 0.1 and 0.05?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: when the data aren't separable

    500 training points labelled by a line, then 15% of the labels flipped at random, so no
    line gets them all right; 2,000 test points from the same distribution (also with flipped
    labels, so the best possible test accuracy is about 85%). you train your `Perceptron` for
    1, 2, ..., 30 epochs (same seed, so each run extends the previous one) and record the test
    accuracy of the final weights and of the averaged weights.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "on non-separable data the perceptron never stops updating. how much will the test "
        "accuracy of its final weights jump around from one epoch to the next? and the averaged "
        "perceptron's?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _w = np.array([1.0, -2.0, 0.5, 1.5, 0.0])

    def _data(n):
        _X = _rng.normal(size=(n, 5))
        _t = np.where(_X @ _w > 0, 1, -1)
        _t[_rng.random(n) < 0.15] *= -1
        return _X, _t

    _X, _t = _data(500)
    _Xt, _tt = _data(2000)
    _epochs = np.arange(1, 31)
    _last, _avg = [], []
    for _e in _epochs:
        _last.append(ex.Perceptron(max_epochs=int(_e), rng=7).fit(_X, _t).score(_Xt, _tt))
        _avg.append(ex.Perceptron(max_epochs=int(_e), averaged=True, rng=7).fit(_X, _t).score(_Xt, _tt))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.plot(_epochs, _last, "o-", label="final weights")
    _ax.plot(_epochs, _avg, "s-", label="averaged weights")
    _ax.axhline(np.mean(np.where(_Xt @ _w > 0, 1, -1) == _tt), color="gray", ls=":", label="the true line")
    _ax.set(xlabel="epochs", ylabel="test accuracy", ylim=(0.4, 0.95))
    _ax.legend(loc="upper left", fontsize=8, ncol=3)
    _fig.tight_layout()
    _l, _a = np.array(_last[9:]), np.array(_avg[9:])
    mo.vstack([_fig, mo.md(
        f"epochs 10–30: final weights range **{_l.min():.3f}–{_l.max():.3f}** (std {_l.std():.3f}); averaged "
        f"**{_a.min():.3f}–{_a.max():.3f}** (std {_a.std():.3f}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with flipped labels, every epoch contains mistakes, and each mistake moves
    the weights by a whole data point. what does that do to the final weights, and why does
    the epoch you happen to stop at matter so much? why does averaging over all the iterates
    help (think of the iterates as a noisy sequence bouncing around a good solution)? what
    would you use to *choose* when to stop, if you used the plain perceptron?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: XOR, and a feature that fixes it

    four points: $(0, 0)$ and $(1, 1)$ in one class, $(0, 1)$ and $(1, 0)$ in the other (plus
    small noise, 50 copies each). you train your `Perceptron` on the two coordinates, and
    then on three features, $(x_1, x_2, x_1x_2)$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how many epochs will the perceptron need on XOR with 2 features? and with the product "
        "feature added? what training accuracy will each reach?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _base = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    _X = np.repeat(_base, 50, axis=0) + 0.1 * _rng.normal(size=(200, 2))
    _t = np.repeat([-1, -1, 1, 1], 50)
    _m2 = ex.Perceptron(max_epochs=200, rng=0).fit(_X, _t)
    _X3 = np.column_stack([_X, _X[:, 0] * _X[:, 1]])
    _m3 = ex.Perceptron(max_epochs=200, rng=0).fit(_X3, _t)
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4))
    _g = np.linspace(-0.4, 1.4, 200)
    _G1, _G2 = np.meshgrid(_g, _g)
    _P = np.column_stack([_G1.ravel(), _G2.ravel()])
    for _ax, _m, _feat, _title in (
        (_axes[0], _m2, _P, "features $(x_1, x_2)$"),
        (_axes[1], _m3, np.column_stack([_P, _P[:, 0] * _P[:, 1]]), "features $(x_1, x_2, x_1 x_2)$"),
    ):
        _ax.contourf(_G1, _G2, (_m.decision_function(_feat) > 0).reshape(_G1.shape), alpha=0.25, cmap="coolwarm")
        _ax.scatter(_X[:, 0], _X[:, 1], c=_t, cmap="coolwarm", s=8)
        _ax.set(title=f"{_title}: {_m.n_iter_} epochs, train acc {_m.score(_X if _m is _m2 else _X3, _t):.2f}",
                aspect="equal")
        _ax.title.set_fontsize(9)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"2 features: **{_m2.n_iter_}** epochs (the cap), last-epoch mistakes {_m2.mistakes_[-1]}, training accuracy "
        f"{_m2.score(_X, _t):.2f}. with $x_1x_2$: **{_m3.n_iter_}** epochs, training accuracy {_m3.score(_X3, _t):.2f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can no line separate XOR (try writing the four inequalities a line
    would have to satisfy)? what does the boundary look like in the original two coordinates
    once $x_1x_2$ is a feature, and why is it not a line there? this is the idea behind
    multilayer networks (learn the features) and kernels (use very many fixed ones), chapters
    14 and 09.
    """)
    return


if __name__ == "__main__":
    app.run()
