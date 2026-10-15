import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def best_time(fn, *args, budget=0.2):
    """Smallest wall-clock time of fn(*args) over repeated runs within about `budget` seconds."""
    best, spent, runs = np.inf, 0.0, 0
    while spent < budget or runs < 2:
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        best, spent, runs = min(best, dt), spent + dt, runs + 1
        if dt > budget:
            break
    return best


@app.function
def distances_double_loop(X, Y):
    """Every distance with two Python loops (for comparison only)."""
    out = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            out[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    return out


@app.function
def distances_row_loop(X, Y):
    """One Python loop over the rows of X, vectorized over Y (for comparison only)."""
    return np.array([np.sqrt(np.sum((Y - x) ** 2, axis=1)) for x in X])


@app.function
def distances_broadcast(X, Y):
    """Every distance from the (n, m, d) array of differences (your pairwise_differences)."""
    return np.sqrt(np.sum(ex.pairwise_differences(X, Y) ** 2, axis=2))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # numpy arrays & vectorization · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. timings depend on your machine: look at ratios
    and trends.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how much does vectorization buy?

    we compute all distances between $n$ random points and $n$ others in $d$ dimensions, two
    ways: with a double Python loop (one distance per iteration), and with your
    `pairwise_distances`. the loop's time is measured up to $n = 300$ and extrapolated beyond
    (it grows like $n^2$), so you don't have to wait.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at n = 300 points (90,000 pairs) in 10 dimensions, how many times faster is your "
        "pairwise_distances than the double loop? does the speedup grow, shrink or stay the "
        "same as n grows to 3,000?"
    )
    p1
    return (p1,)


@app.cell
def _():
    dim = mo.ui.slider(2, 200, value=10, step=1, label="dimension d")
    dim
    return (dim,)


@app.cell
def _(dim, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _ns = [10, 30, 100, 300, 1000, 3000]
    _loop, _vec = [], []
    for _n in _ns:
        _X, _Y = _rng.normal(size=(_n, dim.value)), _rng.normal(size=(_n, dim.value))
        _vec.append(best_time(ex.pairwise_distances, _X, _Y, budget=0.1))
        if _n <= 300:
            _loop.append(best_time(distances_double_loop, _X, _Y, budget=0.05))
        else:  # extrapolate: the loop does n^2 identical iterations
            _loop.append(_loop[_ns.index(300)] * (_n / 300) ** 2)
    _loop, _vec = np.array(_loop), np.array(_vec)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    _a1.loglog(_ns, _loop, "o-", label="double loop (dashed: extrapolated)")
    _a1.loglog(_ns[3:], _loop[3:], "o--", color="C0")
    _a1.loglog(_ns, _vec, "s-", label="your pairwise_distances")
    _a1.set(xlabel="n (points in each set)", ylabel="time (s)", title=f"all n² distances, d = {dim.value}")
    _a1.legend(fontsize=8)
    _a2.semilogx(_ns, _loop / _vec, "o-")
    _a2.set(xlabel="n", ylabel="speedup (loop ÷ yours)", title="speedup", yscale="log")
    _fig.tight_layout()
    _rows = "\n".join(f"| {n:,} | {l:.4g} | {v:.4g} | {l / v:,.0f}× |" for n, l, v in zip(_ns, _loop, _vec))
    mo.vstack([_fig, mo.md("| n | loop (s) | yours (s) | speedup |\n|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the loop's cost per pair is roughly constant. why isn't yours? why is the
    speedup small for tiny $n$? move the dimension slider: does a bigger $d$ help or hurt the
    loop's relative performance, and why?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: three loop-free ways to compute distances

    all three methods below avoid a loop over *pairs*:

    - **broadcast**: build the full $(n, m, d)$ array of differences with your
      `pairwise_differences`, square, sum over the last axis, square root.
    - **row loop**: loop over the $n$ rows of $X$ in Python; for each, compute its distance to
      all of $Y$ at once.
    - **expansion**: your `pairwise_distances`, with $\|x\|^2 - 2x\cdot y + \|y\|^2$ and one
      matrix product.

    $n = m = 500$, and the dimension $d$ varies. next to each time is the size of the largest
    array the method creates.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which of the three is fastest at d = 128? is the fully vectorized 'broadcast' method "
        "faster or slower than the 'row loop', which still has a Python loop?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _n = 500
    _ds = [2, 8, 32, 128]
    _methods = {"broadcast": distances_broadcast, "row loop": distances_row_loop,
                "expansion (yours)": ex.pairwise_distances}
    _mem = {"broadcast": lambda d: _n * _n * d * 8, "row loop": lambda d: _n * d * 8,
            "expansion (yours)": lambda d: _n * _n * 8}
    _t = {k: [] for k in _methods}
    for _d in _ds:
        _X, _Y = _rng.normal(size=(_n, _d)), _rng.normal(size=(_n, _d))
        for _k, _f in _methods.items():
            _t[_k].append(best_time(_f, _X, _Y, budget=0.15))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    for _k in _methods:
        _ax.loglog(_ds, _t[_k], "o-", label=_k)
    _ax.set(xlabel="dimension d", ylabel="time (s)", title=f"{_n} × {_n} distances")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(
        f"| {d} | " + " | ".join(f"{_t[k][i] * 1e3:.2f} ms ({_mem[k](d) / 1e6:,.1f} MB)" for k in _methods) + " |"
        for i, d in enumerate(_ds))
    mo.vstack([_fig, mo.md("| d | " + " | ".join(_methods) + " |\n|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** all three do about the same number of multiply-adds. so where does the
    time go in the broadcast method? why can a method with a Python loop beat one without?
    and what is special about the matrix product inside the expansion method?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the fast formula, far from the origin

    distances don't change if you shift every point by the same amount. so we take 200 random
    points in 3-D (typical distances between them are around 2.5), shift them all by a
    constant $c$ added to every coordinate, and compare your `pairwise_distances` with the
    true distances (computed from the unshifted points' differences, which don't suffer any
    cancellation).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "as the shift c grows from 1 to 100,000,000, how does the error of your "
        "pairwise_distances change? at what c does it become useless?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _Z = _rng.normal(size=(200, 3))
    _cs = 10.0 ** np.arange(0, 9)
    _err_raw, _err_centered = [], []
    for _c in _cs:
        _X = _Z + _c
        _true = np.sqrt(np.sum(ex.pairwise_differences(_X - _c, _X - _c) ** 2, axis=2))
        _off = ~np.eye(len(_X), dtype=bool)
        _raw = ex.pairwise_distances(_X)
        _cen = ex.pairwise_distances(_X - _X.mean(axis=0))
        _err_raw.append(np.max(np.abs(_raw - _true)[_off] / _true[_off]))
        _err_centered.append(np.max(np.abs(_cen - _true)[_off] / _true[_off]) + 1e-17)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _ax.loglog(_cs, _err_raw, "o-", label="your pairwise_distances(X)")
    _ax.loglog(_cs, _err_centered, "s-", label="pairwise_distances(X - X.mean(axis=0))")
    _ax.axhline(1.0, color="gray", ls=":", lw=1)
    _ax.set(xlabel="shift c added to every coordinate", ylabel="worst relative error",
            title="same points, shifted", ylim=(1e-17, 1e2))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {c:.0e} | {r:.1e} | {e:.1e} |" for c, r, e in zip(_cs, _err_raw, _err_centered))
    mo.vstack([_fig, mo.md("| shift c | error, raw | error, centered first |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with $c = 10^8$, how big are $\|x\|^2$ and $2\,x\cdot y$, and how big is
    their difference? floating point keeps about 16 significant digits (topic 03 explains
    exactly why): how many of them are left for the answer? why does subtracting the mean
    first fix it, and why is that allowed?
    """)
    return


if __name__ == "__main__":
    app.run()
