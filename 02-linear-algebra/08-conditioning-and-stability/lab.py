import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.linalg import solve, svd  # yours, from topics 03 and 06

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def hilbert(n):
    """The n x n Hilbert matrix H[i, j] = 1 / (i + j + 1): famously ill-conditioned."""
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # conditioning & stability · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: Hilbert systems lose digits

    for $n = 2, \dots, 14$ we set $x_\text{true} = (1, \dots, 1)$, compute $b = Hx_\text{true}$,
    and solve $Hx = b$ with your topic-03 `solve`. we record the condition number (your
    `cond`), the forward error $\|x - x_\text{true}\|/\|x_\text{true}\|$, and the backward error
    (your `backward_error`).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how many correct digits does x have when cond(H) = 1e10? at n = 14, is solve's answer "
        "garbage, and if so, is solve to blame?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _ns = np.arange(2, 15)
    _conds, _fwd, _bwd = [], [], []
    for _n in _ns:
        _H = hilbert(_n)
        _x_true = np.ones(_n)
        _b = _H @ _x_true
        _x = solve(_H, _b)
        _conds.append(ex.cond(_H))
        _fwd.append(np.linalg.norm(_x - _x_true) / np.linalg.norm(_x_true) + 1e-18)
        _bwd.append(ex.backward_error(_H, _x, _b) + 1e-18)
    _conds = np.array(_conds)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.semilogy(_ns, _conds, "k^-", ms=4, label="cond(H)")
    _ax.semilogy(_ns, _fwd, "o-", label="forward error")
    _ax.semilogy(_ns, _bwd, "s-", label="backward error")
    _ax.semilogy(_ns, 1.1e-16 * _conds, "k:", label=r"$\varepsilon \cdot$ cond")
    _ax.set(xlabel="n", ylabel="size", ylim=(1e-18, 1e20))
    _ax.legend(fontsize=8)
    _rows = "\n".join(f"| {n} | {c:.1e} | {f:.1e} | {max(0.0, -np.log10(f)):.1f} | {e:.1e} |"
                      for n, c, f, e in zip(_ns, _conds, _fwd, _bwd))
    mo.vstack([_fig, mo.md("| n | cond | forward error | correct digits | backward error |\n"
                           "|---|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how do correct digits relate to $\log_{10}\text{cond}$? the forward error
    reaches 100%, but the backward error never budges from about $10^{-17}$. what does that
    say about `solve`, and about the *problem*? (state the rule
    forward ≲ cond × backward in words.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: estimating the condition number cheaply

    your `cond` runs a full SVD. your `cond_estimate` reuses one LU factorization and does a few
    dozen triangular solves. we compare both on random $n \times n$ matrices.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "at n = 160, how much faster is cond_estimate than cond? how many digits of the true "
        "condition number does the estimate get?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _rows = []
    for _n in (20, 40, 80, 160):
        _A = _rng.normal(size=(_n, _n))
        _t0 = time.perf_counter()
        _c = ex.cond(_A)
        _t1 = time.perf_counter()
        _e = ex.cond_estimate(_A, rng=1)
        _t2 = time.perf_counter()
        _rows.append(f"| {_n} | {_c:.6e} | {_e:.6e} | {abs(_e - _c) / _c:.1e} | "
                     f"{_t1 - _t0:.3f} s | {_t2 - _t1:.3f} s | {(_t1 - _t0) / (_t2 - _t1):.1f}× |")
    mo.md("| n | cond (SVD) | cond_estimate | relative difference | SVD time | estimate time | speedup |\n"
          "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** count the work: what's the cost of an SVD, of one LU, and of one
    iteration of each loop in `cond_estimate`? when is "the right order of magnitude" all you
    need from a condition number? (LAPACK's estimators settle for about a factor of 3.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the worst-case direction

    the bound $\frac{\|\delta x\|}{\|x\|} \le \kappa\,\frac{\|\delta b\|}{\|b\|}$ holds for every
    right-hand side and perturbation. but is it ever *reached*? take a $10 \times 10$ matrix with
    $\kappa = 10^6$, choose the direction of $b$ and of the perturbation $\delta b$ (size $10^{-8}$),
    and measure the **amplification** (relative change in $x$) ÷ (relative change in $b$).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "which choice of (b, δb) directions gives amplification κ = 1e6? what amplification "
        "does a random b with a random δb give?"
    )
    p3
    return (p3,)


@app.cell
def _():
    b_dir = mo.ui.dropdown(["u_max (top singular vector)", "u_min (bottom singular vector)", "random"],
                           value="u_max (top singular vector)", label="direction of b")
    db_dir = mo.ui.dropdown(["u_max (top singular vector)", "u_min (bottom singular vector)", "random"],
                            value="u_min (bottom singular vector)", label="direction of δb")
    mo.hstack([b_dir, db_dir])
    return b_dir, db_dir


@app.cell
def _(b_dir, db_dir, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _Q1, _ = np.linalg.qr(_rng.normal(size=(10, 10)))
    _Q2, _ = np.linalg.qr(_rng.normal(size=(10, 10)))
    _A = (_Q1 * np.logspace(0, -6, 10)) @ _Q2.T
    _U, _s, _Vt = svd(_A)

    def _direction(choice, seed):
        if choice.startswith("u_max"):
            return _U[:, 0]
        if choice.startswith("u_min"):
            return _U[:, -1]
        _v = np.random.default_rng(seed).normal(size=10)
        return _v / np.linalg.norm(_v)

    _b = _direction(b_dir.value, 3)
    _db = 1e-8 * _direction(db_dir.value, 4)
    _x = solve(_A, _b)
    _dx = solve(_A, _b + _db) - _x
    _amp = (np.linalg.norm(_dx) / np.linalg.norm(_x)) / (np.linalg.norm(_db) / np.linalg.norm(_b))
    _rand = []
    for _k in range(200):
        _r = np.random.default_rng(100 + _k)
        _bb = _r.normal(size=10)
        _dd = 1e-8 * _r.normal(size=10)
        _xx = solve(_A, _bb)
        _rand.append((np.linalg.norm(solve(_A, _bb + _dd) - _xx) / np.linalg.norm(_xx))
                     / (np.linalg.norm(_dd) / np.linalg.norm(_bb)))
    mo.md(f"cond(A) = **{ex.cond(_A):.3e}**. amplification for your choice: **{_amp:.3e}**.\n\n"
          f"over 200 random (b, δb) pairs: median **{np.median(_rand):.2e}**, max **{np.max(_rand):.2e}**.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** write $b$ and $\delta b$ in the basis $u_1, \dots, u_n$. then
    $x = \sum (u_i^\top b/\sigma_i)\,v_i$. which combination makes $\|x\|$ as small as possible
    and $\|\delta x\|$ as large as possible? why do random pairs land far below κ, and why
    is that no comfort when your data isn't random?
    """)
    return


if __name__ == "__main__":
    app.run()
