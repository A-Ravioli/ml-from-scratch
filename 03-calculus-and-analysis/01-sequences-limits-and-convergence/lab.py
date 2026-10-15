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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # sequences, limits & convergence · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a race between three root finders

    the cubic $f(x) = x^3 - 2x - 5$ has one real root, $x^* \approx 2.0945514815$. we run your
    `bisection` on $[2, 3]$, your `newton_1d` from $x_0 = 3$, and your `secant` from
    $x_0 = 3, x_1 = 2.5$, all with `tol = 1e-15`, and feed their errors to your
    `order_estimates`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how many iterations does each method need to reach an error below 1e-12? what order "
        "of convergence will each one show? (bisection and Newton are in the lesson; guess the secant's.)"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _root = 2.0945514815423265
    _f = lambda x: x**3 - 2 * x - 5
    _runs = {
        "bisection": ex.bisection(_f, 2.0, 3.0, tol=1e-15),
        "newton": ex.newton_1d(_f, lambda x: 3 * x**2 - 2, 3.0, tol=1e-15),
        "secant": ex.secant(_f, 3.0, 2.5, tol=1e-15),
    }
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _rows = []
    for _name, _xs in _runs.items():
        _e = np.abs(np.asarray(_xs) - _root)
        _ax.semilogy(np.arange(len(_e)), np.maximum(_e, 1e-17), "o-", ms=3, label=_name)
        _pos = _e[_e > 1e-14]  # rounding floor: errors near 1e-16 are noise
        _q = ex.order_estimates(_pos) if len(_pos) >= 3 else np.array([np.nan])
        _hit = np.flatnonzero(_e < 1e-12)
        _n = int(_hit[0]) if _hit.size else None
        _ratio = np.exp(np.mean(np.log(_pos[1:] / _pos[:-1])))  # geometric-mean e_(k+1)/e_k
        _rows.append(f"| {_name} | {_n} | {_ratio:.3f} | {np.median(_q[-3:]):.3f} | "
                     f"{', '.join(f'{v:.2f}' for v in _q[-5:])} |")
    _ax.set(xlabel="iteration k", ylabel="error |x_k - x*|", title="error per iteration")
    _ax.legend()
    mo.vstack([_fig, mo.md(
        "| method | iterations to error < 1e-12 | average ratio e_(k+1)/e_k | order (median of last 3 estimates) | last order estimates |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are bisection's order estimates so jumpy, even though it "halves the
    error" every step? what quantity *does* halve exactly? the secant method uses no
    derivative and only one new function evaluation per step, while Newton uses two (f and
    f'). per function evaluation, which one is faster?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: Newton at a repeated root

    now the root is repeated: $f(x) = (x - 1)^m (x + 2)$ has a root of multiplicity $m$ at
    $x^* = 1$. we run your `newton_1d` from $x_0 = 2.5$ with the exact derivative (up to 200
    steps) and measure the order and the linear rate $e_{k+1}/e_k$ with your estimators. move
    the slider to change $m$. the dropdown chooses how $f$ is *computed*: in factored form, or
    expanded into a polynomial $x^{m+1} + \dots$ the way you'd usually meet it.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for a double root (m = 2), is Newton still quadratic? if not, what order and what "
        "rate do you expect? how accurately can it locate the root?"
    )
    p2
    return (p2,)


@app.cell
def _():
    mult = mo.ui.slider(1, 5, value=2, label="multiplicity m")
    form = mo.ui.dropdown(["factored", "expanded"], value="factored", label="how f is computed")
    mo.hstack([mult, form], justify="start")
    return form, mult


@app.cell
def _(form, mult, p2):
    ui.gate(p2)
    _m = mult.value
    if form.value == "factored":
        _f = lambda x: (x - 1.0) ** _m * (x + 2.0)
        _df = lambda x: _m * (x - 1.0) ** (_m - 1) * (x + 2.0) + (x - 1.0) ** _m
    else:
        _c_f = np.poly([1.0] * _m + [-2.0])  # coefficients of (x - 1)^m (x + 2), multiplied out
        _c_df = np.polyder(_c_f)
        _f = lambda x: np.polyval(_c_f, x)
        _df = lambda x: np.polyval(_c_df, x)
    with np.errstate(all="ignore"):
        _xs = ex.newton_1d(_f, _df, 2.5, tol=0.0, max_iters=200)
    _e = np.abs(_xs - 1.0)
    _e = _e[np.isfinite(_e)]
    _e = _e[: int(np.argmax(_e < 1e-15))] if np.any(_e < 1e-15) else _e
    _e = _e[_e > 0]
    _q = ex.order_estimates(_e)
    _c = ex.rate_estimates(_e, 1.0)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    _a1.semilogy(_e, "o-", ms=3)
    _a1.set(xlabel="iteration k", ylabel="error", title=f"Newton, multiplicity m = {_m} ({form.value})")
    _a2.plot(_c, "s-", ms=3, label="rate e_(k+1)/e_k")
    _a2.plot(np.clip(_q, 0, 3), "o-", ms=3, label="order estimate q_k")
    _a2.axhline((_m - 1) / _m, color="gray", ls=":", label="(m - 1)/m")
    _a2.set(xlabel="k", ylim=(0, 3))
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _mid = len(_c) // 2
    mo.vstack([_fig, mo.md(
        f"iterations run: **{len(_xs) - 1}** · best error reached: **{_e.min():.1e}** · "
        f"rate estimate mid-run: **{_c[_mid]:.4f}** · median order estimate: **{np.median(_q):.3f}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** write $f(x) = (x - 1)^m g(x)$ and compute the Newton step near $x = 1$.
    why does the error shrink by the factor $(m-1)/m$ each step? then switch to the expanded
    form: why does the best error now stall far above $10^{-16}$, near $10^{-16/m}$? (hint:
    how big is $f$ when $|x - 1| = 10^{-6}$ and $m = 3$, compared with the rounding error in
    summing terms of size about 1?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: can a computer tell that a sequence converges?

    three sequences of partial sums: the harmonic series $\sum 1/n$ (diverges), $\sum 1/n^2$
    (converges to $\pi^2/6$), and the alternating harmonic series $\sum (-1)^{n+1}/n$
    (converges to $\log 2$). for each, we look at the gaps between consecutive terms and at
    your `cauchy_diameters` of the first $K$ terms, then grow the window $K$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the consecutive gaps of all three sequences go to zero. from K = 10^6 computed terms, "
        "can you tell which sequence diverges? what happens to the tail diameter from index 100 as K grows?"
    )
    p3
    return (p3,)


@app.cell
def _():
    window = mo.ui.dropdown(["1000", "10000", "100000", "1000000"], value="10000", label="window K (terms computed)")
    window
    return (window,)


@app.cell
def _(p3, window):
    ui.gate(p3)
    _K = int(window.value)
    _n = np.arange(1, _K + 1)
    _seqs = {
        "harmonic Σ 1/n": np.cumsum(1.0 / _n),
        "Σ 1/n²": np.cumsum(1.0 / _n**2),
        "alternating Σ (-1)^(n+1)/n": np.cumsum((-1.0) ** (_n + 1) / _n),
    }
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _rows = []
    for _name, _s in _seqs.items():
        _d = ex.cauchy_diameters(_s)
        _ax.loglog(_n[: _K // 2], np.maximum(_d[: _K // 2], 1e-17), label=_name)
        _rows.append(f"| {_name} | {abs(_s[-1] - _s[-2]):.1e} | {_d[99]:.4f} | {_d[999] if _K > 1000 else float('nan'):.4f} |")
    _ax.set(xlabel="tail start N", ylabel="diameter of a[N:K]", title=f"tail diameters within the first K = {_K:,} terms")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "| sequence | last gap |a_K - a_(K-1)| | diameter from N = 100 | diameter from N = 1000 |\n"
        "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compute the harmonic tail diameter from $N = 100$ by hand as a function of
    $K$. why does it grow without bound as $K \to \infty$, while the other two stay below
    about $1/N$? what does this say about using "the last few terms barely change" as a
    stopping rule?
    """)
    return


if __name__ == "__main__":
    app.run()
