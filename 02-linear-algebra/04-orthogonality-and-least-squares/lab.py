import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.linalg import solve  # yours, from topic 03

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def with_condition(m, n, kappa, rng):
    """An (m, n) matrix with singular values logspaced from 1 to 1/kappa, built with YOUR qr."""
    U, _ = ex.qr(rng.normal(size=(m, n)))
    V, _ = ex.qr(rng.normal(size=(n, n)))
    return (U * np.logspace(0, -np.log10(kappa), n)) @ V.T


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # orthogonality & least squares · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: three ways to orthogonalize

    classical Gram–Schmidt, modified Gram–Schmidt and Householder QR all produce $A = QR$
    with orthonormal $Q$, *in exact arithmetic*. we measure how orthonormal the computed $Q$
    really is, $\max|Q^\top Q - I|$, as the matrix gets closer to rank-deficient (larger
    condition number $\kappa$, the ratio of largest to smallest singular value).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "as κ grows from 1 to 1e12, how does the orthogonality error of each method grow? "
        "sketch three lines on a log-log plot."
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _kappas = np.logspace(0, 12, 13)
    _errs = {"classical GS": [], "modified GS": [], "Householder": []}
    for _k in _kappas:
        _A = with_condition(60, 15, _k, _rng)
        for _name, _f in (("classical GS", lambda M: ex.gram_schmidt(M, modified=False)),
                          ("modified GS", lambda M: ex.gram_schmidt(M, modified=True)),
                          ("Householder", ex.qr)):
            _Q, _ = _f(_A)
            _errs[_name].append(np.abs(_Q.T @ _Q - np.eye(15)).max() + 1e-18)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _name, _e in _errs.items():
        _ax.loglog(_kappas, _e, "o-", ms=4, label=_name)
    _ax.loglog(_kappas, 1e-16 * _kappas, "k:", lw=1, label=r"$\varepsilon\,\kappa$")
    _ax.loglog(_kappas, 1e-16 * _kappas**2, "k--", lw=1, label=r"$\varepsilon\,\kappa^2$")
    _ax.set(xlabel="condition number κ", ylabel=r"max $|Q^\top Q - I|$", ylim=(1e-17, 10))
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** match each method to a slope. the two Gram–Schmidts do *exactly* the same
    arithmetic operations in a different order: why does the order matter? and why doesn't
    Householder care about κ at all?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: normal equations or QR?

    fit a polynomial of degree $d$ to 40 points on $[0, 1]$ whose values come *exactly* from a
    known polynomial. no noise, so the right answer is known. we compare your `lstsq` (QR)
    with solving the normal equations $A^\top A x = A^\top b$ using your topic-03 `solve`.
    (the dotted guide lines need $\kappa(A)$, measured here with NumPy's SVD. you'll build your
    own SVD in topic 06.)
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "up to what degree does each method recover the coefficients to 6 correct digits?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _t = np.linspace(0, 1, 40)
    _rng = np.random.default_rng(1)
    _degs = np.arange(1, 16)
    _e_qr, _e_ne, _conds = [], [], []
    for _d in _degs:
        _A = np.vander(_t, _d + 1, increasing=True)
        _x_true = _rng.normal(size=_d + 1)
        _b = _A @ _x_true
        _x_qr = ex.lstsq(_A, _b)
        try:
            _x_ne = solve(_A.T @ _A, _A.T @ _b)
        except np.linalg.LinAlgError:
            _x_ne = np.full_like(_x_true, np.nan)
        _e_qr.append(np.linalg.norm(_x_qr - _x_true) / np.linalg.norm(_x_true) + 1e-18)
        _e_ne.append(np.linalg.norm(_x_ne - _x_true) / np.linalg.norm(_x_true) + 1e-18)
        _s = np.linalg.svd(_A, compute_uv=False)
        _conds.append(_s[0] / _s[-1])
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.semilogy(_degs, _e_qr, "o-", label="QR (your lstsq)")
    _ax.semilogy(_degs, _e_ne, "s-", label="normal equations")
    _ax.semilogy(_degs, 1e-16 * np.array(_conds), "k:", label=r"$\varepsilon\,\kappa(A)$")
    _ax.semilogy(_degs, 1e-16 * np.array(_conds) ** 2, "k--", label=r"$\varepsilon\,\kappa(A)^2$")
    _ax.axhline(1e-6, color="gray", lw=0.8)
    _ax.set(xlabel="polynomial degree", ylabel="relative error in coefficients", ylim=(1e-17, 1e3))
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what is the condition number of $A^\top A$ in terms of that of $A$? use it
    to explain the gap between the two curves. when (if ever) would you still use the normal
    equations?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: one point, two places

    fit a line to 21 evenly spaced points near $y = 1 + 0.5t$ on $t \in [-1, 1]$. then grab
    **one** point and push it up by `shift`. the slider picks *which* point.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "you push one point up by 3 units. how much does the fitted SLOPE change if it's the "
        "middle point (t = 0)? the last point (t = 1)?"
    )
    p3
    return (p3,)


@app.cell
def _():
    which = mo.ui.slider(0, 20, value=20, label="which point to move (index, 10 = middle)")
    shift = mo.ui.slider(-5.0, 5.0, value=3.0, step=0.5, label="shift")
    mo.vstack([which, shift])
    return shift, which


@app.cell
def _(p3, shift, which):
    ui.gate(p3)
    _t = np.linspace(-1, 1, 21)
    _rng = np.random.default_rng(2)
    _y = 1 + 0.5 * _t + 0.1 * _rng.normal(size=21)
    _A = np.column_stack([np.ones_like(_t), _t])
    _x0 = ex.lstsq(_A, _y)
    _y2 = _y.copy()
    _y2[which.value] += shift.value
    _x1 = ex.lstsq(_A, _y2)
    _slope_change = [ex.lstsq(_A, _y + shift.value * np.eye(21)[_i])[1] - _x0[1] for _i in range(21)]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    _a1.plot(_t, _y2, "o", ms=4)
    _a1.plot(_t[which.value], _y2[which.value], "o", color="C3", ms=8)
    _a1.plot(_t, _A @ _x0, "--", color="gray", label=f"before: slope {_x0[1]:.2f}")
    _a1.plot(_t, _A @ _x1, color="C3", label=f"after: slope {_x1[1]:.2f}")
    _a1.legend()
    _a1.set(xlabel="t", ylabel="y", title="the fit")
    _a2.bar(_t, _slope_change, width=0.08)
    _a2.set(xlabel="t of the moved point", ylabel="change in slope", title=f"moving each point by {shift.value}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the slope change is a *linear* function of where the moved point sits.
    derive it: with centered $t$ (mean zero), the least-squares slope is
    $\sum t_i y_i / \sum t_i^2$. what does that say about points far from the center
    (*high leverage*) in real data?
    """)
    return


if __name__ == "__main__":
    app.run()
