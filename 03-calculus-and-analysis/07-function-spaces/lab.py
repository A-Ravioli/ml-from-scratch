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
    # function spaces · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the Gibbs phenomenon

    the square wave $\operatorname{sign}(x)$ on $[-\pi, \pi]$ jumps from $-1$ to $+1$ at 0. we
    project it onto sines and cosines up to frequency $K$ with your `fourier_coefficients` and
    rebuild it with your `fourier_series`. slide $K$ up and watch the overshoot near the jump.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "near the jump the partial sums overshoot 1. as K grows from 5 to 200, what happens to the "
        "height of the overshoot? and to the L2 error?"
    )
    p1
    return (p1,)


@app.cell
def _():
    K = mo.ui.slider(5, 201, value=25, step=4, label="frequencies K")
    K
    return (K,)


@app.cell
def _(K, p1):
    ui.gate(p1)
    _x = np.linspace(-np.pi, np.pi, 6001)
    _a, _b = ex.fourier_coefficients(np.sign, K.value, n=16384)
    _S = ex.fourier_series(_a, _b, _x)
    _rows = []
    for _k in (5, 25, 101, 201):
        _ak, _bk = ex.fourier_coefficients(np.sign, _k, n=16384)
        _fine = np.linspace(1e-5, 0.5, 20000)
        _peak = np.max(ex.fourier_series(_ak, _bk, _fine))
        _l2 = ex.l2_norm(lambda t, a=_ak, b=_bk: np.sign(t) - ex.fourier_series(a, b, t), -np.pi, np.pi, n=16384)
        _rows.append(f"| {_k} | {_peak:.4f} | {_l2:.4f} |")
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    _a1.plot(_x, np.sign(_x), "k", lw=1)
    _a1.plot(_x, _S, "C3", lw=1)
    _a1.set(title=f"square wave and its Fourier sum, K = {K.value}")
    _zoom = np.abs(_x) < 0.6
    _a2.plot(_x[_zoom], np.sign(_x[_zoom]), "k", lw=1)
    _a2.plot(_x[_zoom], _S[_zoom], "C3", lw=1)
    _a2.axhline(1.179, color="gray", ls=":", lw=0.8)  # the limiting peak height
    _a2.set(title="zoom on the jump", ylim=(-1.4, 1.4))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| K | peak of the partial sum | L2 error |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the L2 error goes to zero but the peak doesn't come down: how can both be
    true? (where is the overshoot, and how wide is it, as $K$ grows?) what does this say about
    the difference between convergence "in $L^2$" and convergence "at every point"? why would
    an image compressed by dropping high frequencies (JPEG) show ringing next to sharp edges?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: smoothness and the decay of coefficients

    four functions on $[-\pi, \pi]$: the square wave, a triangle wave $|x|$, the smooth
    periodic function $e^{\cos x}$, and the straight line $f(x) = x$. we plot the size of
    their Fourier coefficients $\sqrt{a_k^2 + b_k^2}$ against $k$ on log-log axes (your
    `fourier_coefficients`), and also project $f(x) = x/\pi$ onto Legendre polynomials on
    $[-1, 1]$ (your `legendre_coefficients`).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "rank the four functions by how fast their Fourier coefficients shrink. where does the "
        "perfectly smooth straight line f(x) = x land?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _funcs = {
        "square wave sign(x)": np.sign,
        "triangle abs(x)": np.abs,
        "smooth exp(cos x)": lambda t: np.exp(np.cos(t)),
        "straight line x": lambda t: t,
    }
    _K = 64
    _ks = np.arange(1, _K + 1)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _rows = []
    for _name, _f in _funcs.items():
        _a, _b = ex.fourier_coefficients(_f, _K, n=16384)
        _mag = np.hypot(_a[1:], _b[1:])
        _nz = _mag > 1e-13
        _ax.loglog(_ks[_nz], _mag[_nz], "o", ms=3, label=_name)
        _use = _nz & (_ks >= 5) & (_ks <= 40)
        _slope = np.polyfit(np.log(_ks[_use]), np.log(_mag[_use]), 1)[0]
        _rows.append(f"| {_name} | {_mag[0]:.3f} | {_mag[10]:.1e} | {_mag[_nz][-1]:.1e} (k = {_ks[_nz][-1]}) | {_slope:.2f} |")
    _ax.set(xlabel="frequency k", ylabel="|coefficient|", ylim=(1e-17, 10))
    _ax.legend(fontsize=8)
    _c = ex.legendre_coefficients(lambda t: t, 8, n=16384)
    mo.vstack([_fig, mo.md(
        "| function | k = 1 | k = 11 | last above 1e-13 | log-log slope, 5 ≤ k ≤ 40 |\n|---|---|---|---|---|\n"
        + "\n".join(_rows)
        + f"\n\nLegendre coefficients of x on [−1, 1] (rounded to 5 digits): **{(np.round(_c, 5) + 0.0).tolist()}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a Fourier series sees $f$ as a *periodic* function. draw the periodic
    extension of $f(x) = x$ from $[-\pi, \pi]$: what does it look like at $\pm\pi$? use that to
    explain why the line's coefficients decay like the square wave's, while the triangle's
    decay faster. what's the general rule connecting smoothness (of the periodic extension) to
    the decay rate? why does the Legendre basis have no trouble with $x$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: projection versus interpolation

    Runge's function $f(x) = 1/(1 + 25x^2)$ on $[-1, 1]$ is smooth and harmless-looking. two
    ways to approximate it by a polynomial of degree $K$: your **Legendre projection**
    (`legendre_coefficients` + `legendre_series`), and **interpolation** through $K + 1$
    equally spaced points (the polynomial that passes exactly through the samples). we
    compare their maximum errors on $[-1, 1]$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "as the degree K grows from 4 to 24, which approximation improves? does the interpolating "
        "polynomial, which matches f exactly at K + 1 points, get better too?"
    )
    p3
    return (p3,)


@app.cell
def _():
    degree = mo.ui.slider(2, 24, value=14, step=2, label="degree K")
    degree
    return (degree,)


@app.cell
def _(degree, p3):
    ui.gate(p3)
    _f = lambda t: 1 / (1 + 25 * t**2)
    _x = np.linspace(-1, 1, 2001)

    def _interp(K):
        nodes = np.linspace(-1, 1, K + 1)
        V = np.vander(nodes, K + 1)
        coef = np.linalg.solve(V, _f(nodes))
        return np.polyval(coef, _x), nodes

    _rows = []
    for _k in (4, 8, 12, 16, 20, 24):
        _proj = ex.legendre_series(ex.legendre_coefficients(_f, _k, n=8192), _x)
        _int, _ = _interp(_k)
        _rows.append(f"| {_k} | {np.abs(_proj - _f(_x)).max():.2e} | {np.abs(_int - _f(_x)).max():.2e} |")
    _K = degree.value
    _proj = ex.legendre_series(ex.legendre_coefficients(_f, _K, n=8192), _x)
    _int, _nodes = _interp(_K)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.plot(_x, _f(_x), "k", lw=2, label="1/(1 + 25x²)")
    _ax.plot(_x, _proj, "C0", label=f"Legendre projection, degree {_K}")
    _ax.plot(_x, _int, "C3", label=f"interpolation, {_K + 1} equispaced points")
    _ax.plot(_nodes, _f(_nodes), "C3o", ms=3)
    _ax.set(ylim=(-0.5, 1.5))
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| K | max error, projection | max error, interpolation |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the projection minimizes the L2 error over all degree-$K$ polynomials, by
    construction. interpolation instead forces agreement at $K + 1$ points. why can forcing
    agreement at more points make things *worse* between them, near the ends of the interval?
    (think about what a high-degree polynomial must do to pass through equispaced samples of a
    function with complex poles at $\pm i/5$, as in topic 03's experiment 1.) what does this
    suggest about fitting high-capacity models by matching training points exactly?
    """)
    return


if __name__ == "__main__":
    app.run()
