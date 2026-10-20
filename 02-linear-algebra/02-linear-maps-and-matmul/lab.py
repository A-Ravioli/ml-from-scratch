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
def best_time(fn, *args, budget=0.3):
    """Smallest wall-clock time of fn(*args) over repeated runs within about `budget` seconds."""
    best, spent = np.inf, 0.0
    while spent < budget:
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        best, spent = min(best, dt), spent + dt
        if dt > budget:
            break
    return best


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # linear maps & matmul · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: your matmuls against BLAS

    all your matmuls do the same $2n^3$ floating point operations. we time each one on random
    $n \times n$ matrices, along with `np.matmul` (which calls an optimized BLAS library), and
    report the speed in GFLOP/s (billions of floating point operations per second).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at n = 128, how many times faster is np.matmul than your triple loop? than your "
        "outer-product version? which of your four versions is fastest?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _ns = [16, 32, 64, 128, 256]
    _methods = {"entries (3 loops)": ex.matmul_entries, "columns": ex.matmul_columns,
                "rows": ex.matmul_rows, "outer products": ex.matmul_outer,
                "blocked (32)": ex.matmul_blocked, "np.matmul (BLAS)": np.matmul}
    _gflops = {name: [] for name in _methods}
    _times = {name: [] for name in _methods}
    for _n in _ns:
        _A, _B = _rng.normal(size=(_n, _n)), _rng.normal(size=(_n, _n))
        for _name, _f in _methods.items():
            if _name.startswith("entries") and _n > 128:
                _t = np.nan  # too slow to be worth waiting for
            else:
                _t = best_time(_f, _A, _B, budget=0.2)
            _times[_name].append(_t)
            _gflops[_name].append(2 * _n**3 / _t / 1e9)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _name, _g in _gflops.items():
        _ax.loglog(_ns, _g, "o-", label=_name)
    _ax.set(xlabel="n", ylabel="GFLOP/s", title="same arithmetic, very different speed")
    _ax.legend(fontsize=8)
    _i = _ns.index(128)
    _ratio = {name: _times[name][_i] / _times["np.matmul (BLAS)"][_i] for name in _methods}
    _rows = "\n".join(f"| {name} | {_times[name][_i] * 1e3:.3f} ms | {_ratio[name]:,.0f}× |" for name in _methods)
    mo.vstack([_fig, mo.md("at n = 128:\n\n| method | time | slower than BLAS by |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the triple loop runs $n^3$ Python-level iterations. how many Python-level
    iterations do the columns, outer-product and blocked versions run? that explains most
    of the gaps between your versions, but not the last factor between `blocked` and BLAS.
    what else does an optimized library do? (think: CPU vector instructions, cache reuse,
    multiple cores.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: where you put the parentheses

    matrix multiplication is associative: $(AB)x = A(Bx)$. for $n \times n$ matrices $A$, $B$ and
    a vector $x$, we time both orders (using `@`, now that you've earned it).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for n = 1500, how much faster is A @ (B @ x) than (A @ B) @ x? do they give the same answer?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _rows = []
    for _n in (250, 500, 1000, 1500):
        _A, _B, _x = _rng.normal(size=(_n, _n)), _rng.normal(size=(_n, _n)), _rng.normal(size=_n)
        _t_left = best_time(lambda A=_A, B=_B, x=_x: (A @ B) @ x, budget=0.3)
        _t_right = best_time(lambda A=_A, B=_B, x=_x: A @ (B @ x), budget=0.3)
        _y1, _y2 = (_A @ _B) @ _x, _A @ (_B @ _x)
        _diff = np.abs(_y1 - _y2).max() / np.abs(_y1).max()
        _rows.append(f"| {_n} | {2 * _n**3 + 2 * _n**2:.1e} | {4 * _n**2:.1e} | {_t_left * 1e3:.2f} ms | "
                     f"{_t_right * 1e3:.3f} ms | {_t_left / _t_right:.0f}× | {_diff:.1e} |")
    mo.md("| n | flops (AB)x | flops A(Bx) | time (AB)x | time A(Bx) | speedup | relative difference |\n"
          "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** count the flops of each order. the measured speedup is smaller than the
    flop ratio: why might a matrix-vector product run at a lower GFLOP/s than a matrix-matrix
    product? where does this matter in ML? (think of computing $J^\top v$ through a deep
    network: chapter 14.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: composing maps of the plane

    $R$ rotates the plane by θ, and $S$ shears it horizontally by $s$: $(x, y) \mapsto (x + sy, y)$.
    we get their matrices with your `matrix_of`, compose them with your `matmul_entries`, and
    draw what each composite does to the unit square and the unit circle. the third panel
    shows a *projection* $P$ onto a line, where the null space appears.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "do 'shear then rotate' (R S) and 'rotate then shear' (S R) do the same thing to the "
        "square? for the projection P, which vectors land on 0, and what does the circle become?"
    )
    p3
    return (p3,)


@app.cell
def _():
    theta = mo.ui.slider(0, 180, value=45, step=5, label="rotation θ (degrees)")
    shear = mo.ui.slider(-2.0, 2.0, value=1.0, step=0.25, label="shear s")
    mo.hstack([theta, shear])
    return shear, theta


@app.cell
def _(p3, shear, theta):
    ui.gate(p3)
    _th = np.deg2rad(theta.value)
    _rot = lambda v: np.array([np.cos(_th) * v[0] - np.sin(_th) * v[1], np.sin(_th) * v[0] + np.cos(_th) * v[1]])
    _shr = lambda v: np.array([v[0] + shear.value * v[1], v[1]])
    _R, _S = ex.matrix_of(_rot, 2), ex.matrix_of(_shr, 2)
    _RS, _SR = ex.matmul_entries(_R, _S), ex.matmul_entries(_S, _R)
    _u = np.array([np.cos(_th), np.sin(_th)])
    _P = ex.matrix_of(lambda v: (v @ _u) * _u, 2)  # orthogonal projection onto the line through u
    _sq = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    _t = np.linspace(0, 2 * np.pi, 200)
    _circ = np.vstack([np.cos(_t), np.sin(_t)])
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4))
    for _ax, _M, _title in ((_axes[0], _RS, "R S: shear, then rotate"),
                            (_axes[1], _SR, "S R: rotate, then shear"),
                            (_axes[2], _P, "P: project onto the line through u")):
        _ax.plot(*_sq, "k:", lw=1)
        _ax.plot(*_circ, ":", color="gray", lw=1)
        _ax.fill(*(_M @ _sq), alpha=0.3)
        _ax.plot(*(_M @ _circ), "C3", lw=2)
        _ax.set_title(_title, fontsize=10)
        _ax.set_aspect("equal")
        _ax.set(xlim=(-3, 3), ylim=(-3, 3))
        _ax.grid(alpha=0.3)
    _k = np.array([-_u[1], _u[0]])
    _axes[2].plot([-3 * _k[0], 3 * _k[0]], [-3 * _k[1], 3 * _k[1]], "C2--", label="null space of P")
    _axes[2].legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"R S = {np.round(_RS, 3).tolist()}, S R = {np.round(_SR, 3).tolist()}. "
        f"det(R S) = det(S R) = {np.linalg.det(_RS):.3f}; det(P) = {np.linalg.det(_P):.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do $RS$ and $SR$ differ, yet have the same determinant (area scaling)?
    for $P$, the whole plane collapses onto a line: by rank–nullity, what's the dimension of
    its null space, and which direction is it?
    """)
    return


if __name__ == "__main__":
    app.run()
