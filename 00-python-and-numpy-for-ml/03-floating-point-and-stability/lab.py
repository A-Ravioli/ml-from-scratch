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
def naive_softmax(x, dtype=np.float64):
    """softmax straight from the formula, in the given dtype (for comparison only)."""
    with np.errstate(all="ignore"):
        e = np.exp(np.asarray(x, dtype=dtype))
        return e / e.sum()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # floating point & stability · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: where does naive softmax break?

    we take three logits $c + (0, 1, 2)$ and slide the offset $c$ from $-1200$ to $1200$. the
    softmax is the same for every $c$ (shift invariance), about $(0.09, 0.24, 0.67)$. we
    compute it with the formula as written, `exp(x) / exp(x).sum()`, in `float64` and in
    `float32` (the format most neural networks train in), and compare with your `softmax`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for which offsets c does the naive formula return nan in float64? in float32? "
        "does it break only for large positive c, or for large negative c too?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _cs = np.linspace(-1200, 1200, 4801)
    _base = np.array([0.0, 1.0, 2.0])
    _good = ex.softmax(_base)
    _err = {}
    for _dt in (np.float64, np.float32):
        _e = []
        for _c in _cs:
            _p = naive_softmax(_c + _base, _dt).astype(float)
            _e.append(np.nan if not np.all(np.isfinite(_p)) else np.abs(_p - _good).max())
        _err[_dt.__name__] = np.array(_e)
    _mine = np.array([np.abs(ex.softmax(_c + _base) - _good).max() for _c in _cs[::40]])
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    for _k, _v in _err.items():
        _ok = np.isfinite(_v)
        _ax.plot(_cs, np.where(_ok, 0, 1) + (0.02 if _k == "float32" else 0), label=f"naive, {_k}: 1 = nan")
    _ax.set(xlabel="offset c (logits are c, c+1, c+2)", ylabel="returns nan?", yticks=[0, 1])
    _ax.legend(fontsize=8, loc="center")
    _fig.tight_layout()
    _ranges = {}
    for _k, _v in _err.items():
        _ok = _cs[np.isfinite(_v)]
        _ranges[_k] = (_ok.min(), _ok.max())
    mo.vstack([_fig, mo.md(
        f"naive float64 works for c from **{_ranges['float64'][0]:.1f}** to **{_ranges['float64'][1]:.1f}**; "
        f"naive float32 only from **{_ranges['float32'][0]:.1f}** to **{_ranges['float32'][1]:.1f}**. "
        f"your `softmax`: worst error over the whole range **{_mine.max():.1e}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compute $\log$ of the largest `float64` ($\approx 1.8 \times 10^{308}$) and
    of the largest `float32` ($\approx 3.4 \times 10^{38}$). what are they, and where do they
    show up in the plot? what goes wrong at the negative end, where nothing overflows? why
    does subtracting the max fix both ends at once?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the variance of big numbers

    the variance of $x_1, \dots, x_n$ has two textbook formulas that are equal in exact
    arithmetic:

    $$\text{one pass: } \overline{x^2} - \bar x^{\,2} \qquad\qquad \text{two pass: } \overline{(x - \bar x)^2}.$$

    the one-pass version is popular because it needs only one sweep through the data (keep
    running sums of $x$ and $x^2$). we take $n = 100{,}000$ measurements with true variance 1,
    shifted by an offset $c$ (think of timestamps in nanoseconds, or prices in cents), and
    compute both.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the true variance is 1 for every offset. at c = 1e8, what do the one-pass and "
        "two-pass formulas return?"
    )
    p2
    return (p2,)


@app.cell
def _():
    offset_exp = mo.ui.slider(0, 10, value=8, step=1, label="show the details for c = 10^k, k =")
    offset_exp
    return (offset_exp,)


@app.cell
def _(offset_exp, p2):
    ui.gate(p2)
    _z = np.random.default_rng(0).normal(size=100_000)
    _z = (_z - _z.mean()) / _z.std()  # exactly mean 0, variance 1 (to rounding)
    _ks = np.arange(0, 11)
    _one, _two = [], []
    for _k in _ks:
        _x = _z + 10.0 ** _k
        _one.append(np.mean(_x * _x) - np.mean(_x) ** 2)
        _two.append(np.mean((_x - np.mean(_x)) ** 2))
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.semilogy(_ks, np.abs(np.array(_one) - 1) + 1e-17, "o-", label="one pass: |error|")
    _ax.semilogy(_ks, np.abs(np.array(_two) - 1) + 1e-17, "s-", label="two pass: |error|")
    _ax.axhline(1.0, color="gray", ls=":", lw=1)
    _ax.set(xlabel="k (offset c = 10^k)", ylabel="|computed variance - 1|", ylim=(1e-17, 1e10))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _k = offset_exp.value
    _sq = (10.0 ** _k) ** 2
    _s, _m, _e = ex.decompose(_sq)
    _rows = "\n".join(f"| 1e{k} | {o:.6g} | {t:.6g} |" for k, o, t in zip(_ks, _one, _two))
    mo.vstack([_fig, mo.md(
        f"at c = 1e{_k}, each $x^2$ is about {_sq:.3g} = {_m:.4f} × 2^{_e}, so (by your `decompose`) "
        f"neighbouring floats there are 2^({_e} - 52) = **{2.0 ** (_e - 52):.3g}** apart.\n\n"
        "| c | one pass | two pass |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compare the spacing of floats near $c^2$ with the answer you're trying to
    compute (1). which subtraction in the one-pass formula cancels? why does the two-pass
    formula survive even though it also subtracts (it computes $x - \bar x$)? can a variance
    come out negative, and what would `np.sqrt` do with it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the loss of a confident mistake

    a three-class classifier outputs logits $(m, 0, 0)$, but the true class is the second one.
    the bigger the margin $m$, the more confidently wrong it is. its cross-entropy loss is
    $-\log p_{\text{true}}$. we compute it three ways: from the naive softmax, from **your**
    stable `softmax` followed by `np.log`, and from your `log_softmax`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "as the margin m grows from 0 to 1,000, what happens to each of the three losses? "
        "is a stable softmax followed by np.log good enough?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _ms = np.linspace(0, 1000, 2001)
    _naive, _stable_log, _logsm = [], [], []
    with np.errstate(all="ignore"):
        for _m in _ms:
            _z = np.array([_m, 0.0, 0.0])
            _naive.append(-np.log(naive_softmax(_z)[1]))
            _stable_log.append(-np.log(ex.softmax(_z)[1]))
            _logsm.append(-ex.log_softmax(_z)[1])
    _naive, _stable_log, _logsm = map(np.array, (_naive, _stable_log, _logsm))
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.plot(_ms, np.where(np.isfinite(_naive), _naive, np.nan), lw=4, alpha=0.5, label="naive softmax, then log")
    _ax.plot(_ms, np.where(np.isfinite(_stable_log), _stable_log, np.nan), lw=2, label="your softmax, then log")
    _ax.plot(_ms, _logsm, "k--", lw=1, label="your log_softmax")
    _ax.set(xlabel="margin m of the wrong class", ylabel="loss = -log p(true class)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()

    def _first_bad(v):
        _bad = ~np.isfinite(v)
        return f"{_ms[np.argmax(_bad)]:.1f}" if _bad.any() else "never"

    mo.vstack([_fig, mo.md(
        f"first margin where the loss stops being a finite number: naive softmax **{_first_bad(_naive)}** "
        f"({_naive[-1]} at m = 1000); your softmax + log "
        f"**{_first_bad(_stable_log)}** ({_stable_log[-1]} at m = 1000); your log_softmax "
        f"**{_first_bad(_logsm)}** ({_logsm[-1]:.6g} at m = 1000).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the naive version break at a smaller margin than your
    softmax-then-log? what is $p_{\text{true}}$ at the point where each one breaks, and which
    number can't be represented there? why does `log_softmax` never break? during training, the gradient of this loss pushes the model to fix exactly
    these confident mistakes: what happens to training if their loss is `inf`?
    """)
    return


if __name__ == "__main__":
    app.run()
