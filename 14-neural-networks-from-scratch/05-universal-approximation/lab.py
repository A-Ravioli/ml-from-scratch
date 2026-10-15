import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.nn import Linear, ReLU, Sequential, predict  # yours, from topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    GRID = np.linspace(0.0, 1.0, 10_000)[1:-1]

    def relu_mlp(widths, seed):
        """a ReLU network 1 -> widths... -> 1 with your Linear layers' default init."""
        layers, d_in = [], 1
        for j, w in enumerate(widths):
            layers += [Linear(d_in, w, rng=100 * seed + j), ReLU()]
            d_in = w
        layers.append(Linear(d_in, 1, rng=100 * seed + 99))
        return Sequential(*layers)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # universal approximation · lab

    every experiment here runs **your** code: `relu_interpolant`, `sawtooth_net`,
    `count_linear_regions`, `fit_function` and `max_error` from `exercise.py`, on your
    `mlfs.nn` layers. for each one: write down a prediction, lock it in, run it, then explain
    what happened. compare your explanations with `solutions/lab_notes.md` afterwards. the
    whole lab takes about half a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: depth wins on the sawtooth

    the target is your `sawtooth_net(d)`: the tent map composed $d$ times, with $2^d$
    linear pieces, computed *exactly* by a width-2 network with $2d$ hidden units. the
    challenger is a **shallow** network: your `relu_interpolant` through $k$ evenly spaced
    samples of the target (one hidden layer of $k$ units). we plot the shallow network's max
    error against $k$, and the hidden units each architecture needs for an exact fit as $d$
    grows.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for the depth-5 sawtooth (32 pieces), how many hidden units does the shallow network need before its "
        "max error drops below 0.01? how does that compare with the deep network's 10?"
    )
    p1
    return (p1,)


@app.cell
def _():
    d1 = mo.ui.slider(2, 7, value=5, label="target depth d")
    d1
    return (d1,)


@app.cell
def _(d1, p1):
    ui.gate(p1)
    _d = d1.value
    _target = ex.sawtooth_net(_d)

    def _f(x):
        return predict(_target, np.asarray(x, dtype=float)[:, None])[:, 0]

    _ks = list(range(2, 2 ** (_d + 1) + 2))
    _errs = [ex.max_error(ex.relu_interpolant(np.linspace(0, 1, k), _f(np.linspace(0, 1, k))), _f) for k in _ks]
    _first = next((k for k, e in zip(_ks, _errs) if e < 0.01), None)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _xs = np.linspace(0, 1, 2001)
    for _k, _lw in [(2 ** (_d - 1) + 1, 2), (2 ** _d + 1, 4)]:
        _net = ex.relu_interpolant(np.linspace(0, 1, _k), _f(np.linspace(0, 1, _k)))
        _axes[0].plot(_xs, predict(_net, _xs[:, None])[:, 0], lw=_lw, alpha=0.7, label=f"shallow, {_k} units")
    _axes[0].plot(_xs, _f(_xs), "k--", lw=1, label=f"target: deep, {2 * _d} units")
    _axes[0].set(xlabel="x", title=f"sawtooth of depth {_d} ({2 ** _d} pieces)")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_ks, _errs, ".-", label="shallow (interpolant through k samples)")
    _axes[1].plot([2 * _d], [0.0], "r*", ms=14, label=f"deep: {2 * _d} units, error 0")
    _axes[1].set(xlabel="hidden units", ylabel="max error on [0, 1]")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {d} | {2 ** d} | {2 * d} | {2 ** d + 1} |" for d in range(1, 11)]
    mo.vstack([_fig,
               mo.md(f"first shallow width with max error < 0.01: **{_first} units** (deep network: {2 * _d}).\n\n"
                     "| depth d | pieces | deep units (exact) | shallow units (exact, evenly spaced knots) |\n"
                     "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the shallow error stay large until $k = 2^d + 1$ and then drop to
    zero? (where are the knots relative to the sawtooth's peaks?) count the linear pieces a
    one-hidden-layer network with $k$ units can have, and the pieces of the target: why must a
    shallow network be exponentially wide? where does the deep network get its pieces from?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: do random deep networks have many pieces?

    48 hidden ReLU units, arranged as 1 layer of 48, 2 of 24, 3 of 16, 4 of 12, 6 of 8 or 12
    of 4, with your `Linear`'s default initialization. we count each random network's linear
    pieces on $[-3, 3]$ with your `count_linear_regions` (10 networks per shape).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the sawtooth shows that depth CAN multiply the number of pieces. for random networks with the same 48 "
        "units, how will the typical number of pieces change as the same units are stacked deeper?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _xs = np.linspace(-3, 3, 30_001)[1:-1]
    _shapes = [[48], [24] * 2, [16] * 3, [12] * 4, [8] * 6, [4] * 12]
    _means, _rows = [], []
    for _w in _shapes:
        _counts = [ex.count_linear_regions(relu_mlp(_w, s), _xs) for s in range(10)]
        _means.append(np.mean(_counts))
        _rows.append(f"| {len(_w)} × {_w[0]} | {np.mean(_counts):.1f} | {min(_counts)}–{max(_counts)} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.plot([len(w) for w in _shapes], _means, "o-", label="random networks (mean of 10)")
    _ax.axhline(49, color="gray", ls=":", label="most a 1-layer net with 48 units can have (49)")
    _ax.set(xlabel="number of hidden layers (48 units in total)", ylabel="linear pieces on [-3, 3]")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| layers × width | mean pieces | range |\n|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the sawtooth gets $2^d$ pieces from $2d$ units; what is special about its
    weights that random weights don't have? (think about where each layer's kinks land
    relative to the previous layer's pieces.) what happens to many units in a deep, narrow
    random network (look for units that are never on)? what does this say about reading
    "depth gives exponentially many pieces" as a statement about *typical* networks?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: can gradient descent find the deep solution?

    two networks with the same 73 parameters, a shallow one (24 units) and a deep, narrow one
    (4 layers of 4), learn the sawtooth of depth 2 (4 pieces) and of depth 3 (8 pieces) with
    your `fit_function` (Adam, learning rate 0.01, 1,500 full-batch steps on 256 points),
    from 5 random initializations each. a fit counts as a success if its max error is below
    0.1. (both networks can represent both targets exactly.)
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the deep architecture represents sawtooths exponentially more efficiently. which network will learn them "
        "more reliably from random initializations?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rows = []
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    _xs = np.linspace(0, 1, 1001)
    for _ax, _d in zip(_axes, [2, 3]):
        _target = ex.sawtooth_net(_d)

        def _f(x, t=_target):
            return predict(t, np.asarray(x, dtype=float)[:, None])[:, 0]

        _ax.plot(_xs, _f(_xs), "k", lw=2, label="target")
        for _name, _w in [("shallow 1 × 24", [24]), ("deep 4 × 4", [4] * 4)]:
            _errs, _mses, _best, _best_err = [], [], None, np.inf
            for _s in range(5):
                _m = relu_mlp(_w, _s)
                _h = ex.fit_function(_m, _f, n_points=256, epochs=1500, batch_size=256, lr=0.01, rng=_s)
                _e = ex.max_error(_m, _f)
                _errs.append(_e)
                _mses.append(_h["train_loss"][-1])
                if _e < _best_err:
                    _best, _best_err = _m, _e
            _ax.plot(_xs, predict(_best, _xs[:, None])[:, 0], label=f"{_name} (best of 5)")
            _rows.append(f"| {_d} ({2 ** _d} pieces) | {_name} | {sum(e < 0.1 for e in _errs)}/5 | "
                         f"{np.median(_mses):.4f} | {min(_errs):.3f} |")
        _ax.set(xlabel="x", title=f"sawtooth of depth {_d}")
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| target | network | successes (max error < 0.1) | median final MSE | best max error |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows)
                           + "\n\n(predicting the mean, 1/2, everywhere gives MSE 1/12 = 0.0833 and max error 0.5.)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how many of the deep runs end up predicting (nearly) a constant, and what
    MSE does that give? look at experiment 2 again: what happens to the units of a deep,
    narrow ReLU network at initialization, and why can gradient descent not revive a unit that
    is off for every input? what's the difference between a network being able to *represent*
    a function and gradient descent being able to *find* it?
    """)
    return


if __name__ == "__main__":
    app.run()
