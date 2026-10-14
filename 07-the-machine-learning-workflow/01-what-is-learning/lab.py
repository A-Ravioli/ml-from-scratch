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
def noisy_linear_data(n, noise, rng):
    """2-D inputs; label 1 when x0 + 0.5 x1 > 0, then a fraction `noise` of labels flipped."""
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    flip = rng.random(n) < noise
    return X, np.where(flip, 1 - y, y)


@app.function
def binary_data(n, d, rng, noise=0.1):
    """d yes/no features; label = majority vote of the first 3, with a fraction `noise` flipped."""
    X = rng.integers(0, 2, size=(n, d)).astype(float)
    y = (X[:, :3].sum(axis=1) >= 2).astype(int)
    flip = rng.random(n) < noise
    return X, np.where(flip, 1 - y, y)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # what is learning? · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: memorizer, stump, and baselines

    200 training points in 2-D. the true rule is a line ($x_0 + 0.5x_1 > 0$), and a fraction
    of the labels is flipped at random (the slider). four models, all fitted with your code:
    the two simplest dummies, the lookup table, and the stump. we score each on its own
    training set and on 5,000 fresh points from the same distribution.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with 15% of labels flipped, rank the four models by TRAINING accuracy, then by TEST "
        "accuracy. give a rough number for each."
    )
    p1
    return (p1,)


@app.cell
def _():
    noise1 = mo.ui.slider(0.0, 0.45, value=0.15, step=0.05, label="fraction of labels flipped")
    noise1
    return (noise1,)


@app.cell
def _(noise1, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _Xtr, _ytr = noisy_linear_data(200, noise1.value, _rng)
    _Xte, _yte = noisy_linear_data(5000, noise1.value, _rng)
    _models = {
        "dummy (most frequent)": ex.DummyClassifier(),
        "dummy (stratified)": ex.DummyClassifier(strategy="stratified", rng=0),
        "lookup table": ex.LookupTableClassifier(),
        "stump": ex.StumpClassifier(),
    }
    _rows, _tr, _te = [], [], []
    for _name, _m in _models.items():
        _m.fit(_Xtr, _ytr)
        _tr.append(_m.score(_Xtr, _ytr))
        _te.append(_m.score(_Xte, _yte))
        _rows.append(f"| {_name} | {_tr[-1]:.3f} | {_te[-1]:.3f} |")
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _pos = np.arange(len(_models))
    _ax.bar(_pos - 0.2, _tr, 0.4, label="train")
    _ax.bar(_pos + 0.2, _te, 0.4, label="test (5,000 fresh points)")
    _ax.axhline(1 - noise1.value, color="k", ls=":", lw=1, label="best possible (1 − noise)")
    _ax.set_xticks(_pos, list(_models), fontsize=9)
    _ax.set(ylabel="accuracy", ylim=(0, 1.05))
    _ax.legend(fontsize=8, loc="upper left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| model | train accuracy | test accuracy |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the lookup table is perfect on the training set. what does it predict for a
    test point, and why does that make its test accuracy equal to a dummy's? the stump uses
    only one feature, while the true rule uses both: why is it still close to the best
    possible? set the noise to 0.45: what happens to the gap between train and test for each
    model?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: when does memorizing work?

    now the inputs are $d$ yes/no features, so there are only $2^d$ possible rows. the label is
    a majority vote of the first three features, with 10% of labels flipped (so the best
    possible test accuracy is 0.90). we train the lookup table on $n$ examples and measure test
    accuracy on 5,000 fresh rows, for several $d$, averaged over 5 seeds. the right panel shows
    the fraction of test rows the table has already seen.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with d = 3 (8 possible rows), d = 10 (1,024) and d = 16 (65,536), how many training "
        "examples does the lookup table need to get close to 0.90 test accuracy?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _ns = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    _ds = [3, 6, 10, 16]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _lines = []
    for _d in _ds:
        _acc, _seen = [], []
        for _n in _ns:
            _a, _s = [], []
            for _seed in range(5):
                _rng = np.random.default_rng(100 * _seed + _d)
                _Xtr, _ytr = binary_data(int(_n), _d, _rng)
                _Xte, _yte = binary_data(5000, _d, _rng)
                _m = ex.LookupTableClassifier().fit(_Xtr, _ytr)
                _a.append(_m.score(_Xte, _yte))
                _s.append(np.mean([tuple(r) in _m.table_ for r in _Xte]))
            _acc.append(np.mean(_a))
            _seen.append(np.mean(_s))
        _a1.semilogx(_ns, _acc, "o-", ms=4, label=f"d = {_d} ({2**_d:,} rows)")
        _a2.semilogx(_ns, _seen, "o-", ms=4, label=f"d = {_d}")
        _lines.append(f"| {_d} | {2**_d:,} | " + " | ".join(f"{a:.2f}" for a in _acc) + " |")
    _stump = []
    for _n in _ns:
        _rng = np.random.default_rng(7)
        _Xtr, _ytr = binary_data(int(_n), 16, _rng)
        _Xte, _yte = binary_data(5000, 16, _rng)
        _stump.append(ex.StumpClassifier().fit(_Xtr, _ytr).score(_Xte, _yte))
    _a1.semilogx(_ns, _stump, "k--", label="stump, d = 16")
    _a1.axhline(0.9, color="gray", lw=0.8, ls=":")
    _a1.set(xlabel="training examples n", ylabel="test accuracy", ylim=(0.45, 0.95), title="lookup table")
    _a2.set(xlabel="training examples n", ylabel="fraction of test rows seen in training", ylim=(-0.02, 1.02))
    _a1.legend(fontsize=8)
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _hdr = "| d | possible rows | " + " | ".join(f"n={n:,}" for n in _ns) + " |\n|---|---|" + "---|" * len(_ns)
    mo.vstack([_fig, mo.md(_hdr + "\n" + "\n".join(_lines))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for $d = 3$ the lookup table is about as good as anything can be. why? what
    is it estimating for each row? why does it need roughly $2^d$ examples (times a constant)
    to work, and what does that say about memorizing continuous inputs, where every row is new?
    the stump (dashed) is far from optimal at $d = 16$: what can't it represent?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: ERM on pure noise

    the labels are now fair coin flips, independent of every feature, so **no** classifier can
    beat 50% on new data. we fit your `StumpClassifier` (exact ERM over all stumps) to $n$
    examples with $d$ random features, and record its training accuracy and its accuracy on
    2,000 fresh examples. the dashed line is the finite-class bound from idea 5 (with
    $\delta = 0.05$ and $|\mathcal H| = 2d(n+1)$): how far above 50% the training accuracy can
    be pushed by chance.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with n = 50 coin-flip labels and d = 200 random features, what training accuracy does "
        "the best stump reach? and what test accuracy?"
    )
    p3
    return (p3,)


@app.cell
def _():
    n3 = mo.ui.slider(20, 400, value=50, step=10, label="training examples n")
    n3
    return (n3,)


@app.cell
def _(n3, p3):
    ui.gate(p3)
    _n = n3.value
    _ds = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    _tr, _te = [], []
    for _d in _ds:
        _a, _b = [], []
        for _seed in range(20):
            _rng = np.random.default_rng(1000 * _seed + int(_d))
            _X = _rng.normal(size=(_n, _d))
            _y = _rng.integers(0, 2, size=_n)
            _m = ex.StumpClassifier().fit(_X, _y)
            _a.append(_m.score(_X, _y))
            _Xn = _rng.normal(size=(2000, _d))
            _b.append(_m.score(_Xn, _rng.integers(0, 2, size=2000)))
        _tr.append(np.mean(_a))
        _te.append(np.mean(_b))
    _bound = 0.5 + np.sqrt((np.log(2 * _ds * (_n + 1)) + np.log(2 / 0.05)) / (2 * _n))
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.semilogx(_ds, _tr, "o-", label="training accuracy (mean of 20 runs)")
    _ax.semilogx(_ds, _te, "s-", label="accuracy on fresh data")
    _ax.semilogx(_ds, np.minimum(_bound, 1.0), "k--", lw=1, label="0.5 + finite-class bound")
    _ax.axhline(0.5, color="gray", lw=0.8)
    _ax.set(xlabel="number of random features d", ylabel="accuracy", ylim=(0.4, 1.0), title=f"coin-flip labels, n = {_n}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = " | ".join(f"{a:.2f}" for a in _tr)
    mo.vstack([_fig, mo.md("| d | " + " | ".join(str(d) for d in _ds) + " |\n|---|" + "---|" * len(_ds)
                            + f"\n| train | {_rows} |\n| fresh | " + " | ".join(f"{b:.2f}" for b in _te) + " |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** nothing here is learnable, yet the training accuracy climbs with $d$. where
    does it come from? why is the bound above the measured curve, and why does it have
    roughly the right *shape* ($\sqrt{\log d}$)? if you reported the stump's training accuracy
    at $d = 200$ as "the model's accuracy", what mistake would you be making, and how will
    topic 02 stop you from making it?
    """)
    return


if __name__ == "__main__":
    app.run()
