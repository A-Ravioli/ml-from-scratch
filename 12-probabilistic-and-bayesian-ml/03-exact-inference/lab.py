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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # exact inference · lab

    every experiment here runs **your** code from `exercise.py` (and your `Factor` from topic
    02). for each one: write down a prediction, lock it in, run it, then explain what
    happened. compare your explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.function
def grid_mrf(n, *, seed=0):
    """an n x n grid of binary variables with random pairwise factors (an Ising-like MRF)."""
    rng = np.random.default_rng(seed)
    name = lambda r, c: f"g{r:02d}{c:02d}"
    fs = []
    for r in range(n):
        for c in range(n):
            fs.append(ex.Factor((name(r, c),), (2,), rng.uniform(0.5, 1.5, 2)))
            if c + 1 < n:
                fs.append(ex.Factor((name(r, c), name(r, c + 1)), (2, 2), np.exp(rng.normal(0, 0.5, (2, 2)))))
            if r + 1 < n:
                fs.append(ex.Factor((name(r, c), name(r + 1, c)), (2, 2), np.exp(rng.normal(0, 0.5, (2, 2)))))
    return fs, [name(r, c) for r in range(n) for c in range(n)]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the order is everything

    a $6 \times 6$ grid of binary variables with random pairwise factors (36 variables, 60
    edges; the full joint would have $2^{36} \approx 7\times10^{10}$ entries). we ask your
    `elimination_cost` what the largest intermediate table would be for 300 random
    elimination orders, for row-by-row order, and for your `min_degree_order`. then we time
    your `variable_elimination` with the min-degree order and with the cheapest and costliest
    random orders that are still small enough to run.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the min-degree order's largest table has 2^7 = 128 entries. how big is a typical random order's? the worst "
        "of 300 random orders?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _fs, _names = grid_mrf(6)
    _rng = np.random.default_rng(1)
    _orders = [list(_rng.permutation(_names)) for _ in range(300)]
    _costs = np.array([ex.elimination_cost(_fs, o) for o in _orders])
    _md = ex.min_degree_order(_fs, _names)
    _c_md = ex.elimination_cost(_fs, _md)
    _c_row = ex.elimination_cost(_fs, _names)
    _q = _names[14]
    _timings = {}
    _runnable = [i for i in np.argsort(_costs) if _costs[i] <= 2**18]
    for _label, _order in [("min-degree", _md), ("best random", _orders[_runnable[0]]),
                           ("worst runnable random", _orders[_runnable[-1]])]:
        _best = np.inf
        for _ in range(3):  # best of 3 timings
            _t = time.perf_counter()
            _res = ex.variable_elimination(_fs, _q, order=[v for v in _order if v != _q])
            _best = min(_best, time.perf_counter() - _t)
        _timings[_label] = (_best, ex.elimination_cost(_fs, _order), _res.values[1])
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _ax.hist(np.log2(_costs), bins=np.arange(4, 26) - 0.5, color="C7", label="300 random orders")
    _ax.axvline(np.log2(_c_md), color="C2", lw=2, label=f"min-degree (2^{np.log2(_c_md):.0f})")
    _ax.axvline(np.log2(_c_row), color="C0", lw=2, ls="--", label=f"row by row (2^{np.log2(_c_row):.0f})")
    _ax.set(xlabel="log2(entries of the largest table)", ylabel="orders")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {k} | 2^{np.log2(c):.0f} | {t * 1000:.1f} ms | {p:.6f} |" for k, (t, c, p) in _timings.items())
    mo.vstack([_fig, mo.md(
        f"random orders: median 2^{np.log2(np.median(_costs)):.1f}, worst 2^{np.log2(_costs.max()):.0f} entries.\n\n"
        "| order | largest table | time | P(" + _q + " = 1) |\n|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** every order gives the same answer (check the last column), so where does
    the difference come from? what happens to the neighbours of a variable when you eliminate
    it, and why does eliminating a variable in the middle of the grid early hurt so much?
    the best possible order on an $n \times n$ grid costs about $2^{n+1}$: what does that say
    about exact inference on a $100 \times 100$ image?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: brute force versus elimination

    a chain $x_0 - x_1 - \dots - x_{n-1}$ of binary variables with random pairwise factors.
    we compute $P(x_{n-1})$ two ways: by brute force (multiply every factor into the joint,
    then sum, which is what topic 02's `joint_query` does) and with your
    `variable_elimination`, for chain lengths up to 22, and time both (elimination: best of 3 runs).
    """)
    return


@app.cell
def _():
    p2 = ui.predict("at n = 22, how many times slower is brute force than variable elimination? 10x? 1,000x? more?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(2)
    _ns = list(range(4, 23, 2))
    _tb, _tv = [], []
    for _n in _ns:
        _fs = [ex.Factor(("x0",), (2,), [0.5, 0.5])]
        _fs += [ex.Factor((f"x{i}", f"x{i + 1}"), (2, 2), _rng.uniform(0.1, 1, (2, 2))) for i in range(_n - 1)]
        _t = time.perf_counter()
        _joint = _fs[0]
        for _f in _fs[1:]:
            _joint = _joint * _f
        _pb = _joint.marginalize([f"x{i}" for i in range(_n - 1)]).normalize().values
        _tb.append(time.perf_counter() - _t)
        _best = np.inf
        for _ in range(3):  # best of 3: elimination is fast enough for timer noise to matter
            _t = time.perf_counter()
            _pv = ex.variable_elimination(_fs, f"x{_n - 1}").values
            _best = min(_best, time.perf_counter() - _t)
        _tv.append(_best)
        assert np.allclose(_pb, _pv)
    _big = np.array(_ns) >= 14
    _sb = np.polyfit(np.array(_ns)[_big], np.log(np.array(_tb)[_big]), 1)[0]
    _sv = np.polyfit(np.array(_ns)[_big], np.log(np.array(_tv)[_big]), 1)[0]
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.semilogy(_ns, _tb, "o-", label="brute force (full joint)")
    _ax.semilogy(_ns, _tv, "s-", label="variable elimination")
    _ax.set(xlabel="chain length n", ylabel="seconds")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"at n = {_ns[-1]}: brute force {_tb[-1] * 1000:.0f} ms, elimination {_tv[-1] * 1000:.1f} ms, "
                           f"ratio **{_tb[-1] / _tv[-1]:.0f}x**. fitted growth per extra variable over n = 14..22 "
                           f"(the slope of log time): brute force **x{np.exp(_sb):.2f}**, elimination **x{np.exp(_sv):.2f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does each extra variable cost each method? at what $n$ would brute
    force need more memory than your laptop has (8 bytes per entry)? elimination on a chain
    is the forward algorithm of an HMM (topic 04): which "message" does it pass along the
    chain?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: all the marginals at once

    a random tree of 80 variables with 3 states each (pairwise factors plus some unary
    evidence-like factors). we want the marginal of *every* variable. option A: call your
    `variable_elimination` 80 times. option B: call your `sum_product` once.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("how many times faster is one sum_product call than 80 variable_elimination calls?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _n = 80
    _fs = [ex.Factor((f"t{i}",), (3,), _rng.uniform(0.2, 1, 3)) for i in range(_n) if _rng.random() < 0.3]
    _fs += [ex.Factor((f"t{int(_rng.integers(i))}", f"t{i}"), (3, 3), _rng.uniform(0.1, 1, (3, 3))) for i in range(1, _n)]
    _t = time.perf_counter()
    _ve = {f"t{i}": ex.variable_elimination(_fs, f"t{i}").values for i in range(_n)}
    _t_ve = time.perf_counter() - _t
    _t_bp = np.inf
    for _ in range(3):  # best of 3: one call is only a few milliseconds
        _t = time.perf_counter()
        _bp = ex.sum_product(_fs)
        _t_bp = min(_t_bp, time.perf_counter() - _t)
    _err = max(np.max(np.abs(_ve[k] - _bp[k])) for k in _ve)
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.barh(["80 x variable_elimination", "1 x sum_product"], [_t_ve, _t_bp], color=["C1", "C2"])
    _ax.set(xlabel="seconds")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"80 elimination runs: **{_t_ve:.2f} s**; one sum_product: **{_t_bp * 1000:.0f} ms** "
                           f"(**{_t_ve / _t_bp:.0f}x** faster). largest disagreement between them: {_err:.1e}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a marginal by elimination on a tree costs $O(n)$ table operations, so 80 of
    them cost $O(n^2)$. how many messages does sum-product compute in total, and why is that
    enough for every marginal? which messages would two elimination runs for neighbouring
    variables compute in common? (that shared work is exactly what the messages cache.)
    """)
    return


if __name__ == "__main__":
    app.run()
