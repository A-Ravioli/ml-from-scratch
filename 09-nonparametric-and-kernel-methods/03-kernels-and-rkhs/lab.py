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
def tensor_features(X, p):
    """the explicit map (x; 1)^{tensor p}: every product of p coordinates of (x_1, ..., x_d, 1), duplicates
    included, so that tensor_features(x) . tensor_features(y) = (x . y + 1)^p exactly. shape (n, (d+1)^p)."""
    X1 = np.hstack([X, np.ones((len(X), 1))])
    Z = np.ones((len(X), 1))
    for _ in range(p):
        Z = (Z[:, :, None] * X1[:, None, :]).reshape(len(X), -1)
    return Z


@app.function
def best_time(fn, repeat=5):
    """best wall-clock time of `repeat` calls, in seconds."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # kernels & RKHS · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: explicit features versus the kernel trick

    600 points in $d = 10$ dimensions. for degree $p = 1, \dots, 4$ we build the Gram matrix of
    the polynomial kernel $(x^\top y + 1)^p$ two ways: explicitly, with the feature map
    $\varphi(x) = (x; 1)^{\otimes p}$ (all products of $p$ coordinates of $(x, 1)$) and
    $K = \Phi\Phi^\top$, and with your `polynomial_kernel`. we check they agree, and measure
    the memory of $\Phi$ and the time of each (best of 5, on this machine).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at degree 4 in 10 dimensions, how many times slower is the explicit route than your kernel: "
        "about the same, 10x, 100x, 1000x?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _X = np.random.default_rng(0).normal(size=(600, 10)) / np.sqrt(10)
    _rows = []
    for _p in [1, 2, 3, 4]:
        _Phi = tensor_features(_X, _p)
        _err = np.max(np.abs(_Phi @ _Phi.T - ex.polynomial_kernel(_X, degree=_p, gamma=1.0, coef0=1.0)))
        _t_exp = best_time(lambda: (lambda F: F @ F.T)(tensor_features(_X, _p)))  # noqa: B023
        _t_ker = best_time(lambda: ex.polynomial_kernel(_X, degree=_p, gamma=1.0, coef0=1.0))  # noqa: B023
        _rows.append((_p, _Phi.shape[1], _Phi.nbytes / 1e6, _t_exp, _t_ker, _err))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _ps = [r[0] for r in _rows]
    _pp = np.arange(1, 9)
    _a1.semilogy(_ps, [r[1] for r in _rows], "o-", label="explicit features $(d+1)^p$ (built)")
    _a1.semilogy(_pp, 11.0**_pp, "C0:", label="$(d+1)^p$, continued")
    _a1.semilogy(_pp, [10] * len(_pp), "k-", lw=1, label="the kernel's input: d = 10 numbers")
    _a1.set(xlabel="degree p", ylabel="numbers stored per point", xticks=_pp)
    _a1.legend(fontsize=8)
    _a2.semilogy(_ps, [r[3] * 1e3 for r in _rows], "o-", label="explicit: build Φ, then Φ Φᵀ")
    _a2.semilogy(_ps, [r[4] * 1e3 for r in _rows], "s-", label="your polynomial_kernel")
    _a2.set(xlabel="degree p", ylabel="time for a 600 × 600 Gram matrix (ms)", xticks=_ps)
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _table = "| p | features | memory of Φ | explicit | kernel | ratio | max difference |\n|---|---|---|---|---|---|---|\n" + "\n".join(
        f"| {p} | {D:,} | {mb:.1f} MB | {te * 1e3:.1f} ms | {tk * 1e3:.2f} ms | {te / tk:.1f}× | {e:.1e} |"
        for p, D, mb, te, tk, e in _rows)
    mo.vstack([_fig, mo.md(_table)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** both routes give the same matrix to round-off. what does each one cost, as a
    function of $n$, $d$ and $p$? at what point would the explicit route stop fitting in
    memory? and what would "explicit features" even mean for the RBF kernel?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: which similarities are kernels?

    eight functions of two points that all look like reasonable similarity measures. for 150
    random points (standard normal) in 1-D and in 2-D we build each Gram matrix, and ask your
    `is_psd`. we also report the smallest eigenvalue divided by the largest.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which of these fail: tanh(x·y − 1), exp(−‖x−y‖⁴), the triangle max(0, 1 − ‖x−y‖/2), cos(‖x−y‖), "
        "1/(1 + ‖x−y‖²)? does the answer depend on the dimension?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _dist(X):
        return np.sqrt(np.clip(((X[:, None] - X[None]) ** 2).sum(-1), 0, None))

    _cands = {
        "rbf  exp(−‖x−y‖²)": lambda X: ex.rbf_kernel(X, gamma=1.0),
        "laplacian  exp(−‖x−y‖₁)": lambda X: ex.laplacian_kernel(X, gamma=1.0),
        "exponential  exp(−‖x−y‖)": lambda X: np.exp(-_dist(X)),
        "Cauchy  1/(1+‖x−y‖²)": lambda X: 1 / (1 + _dist(X) ** 2),
        "triangle  max(0, 1−‖x−y‖/2)": lambda X: np.clip(1 - _dist(X) / 2, 0, None),
        "exp(−‖x−y‖⁴)": lambda X: np.exp(-_dist(X) ** 4),
        "cos(‖x−y‖)": lambda X: np.cos(_dist(X)),
        "sigmoid  tanh(x·y − 1)": lambda X: np.tanh(X @ X.T - 1),
    }
    _res = {}
    for _d in (1, 2):
        _X = np.random.default_rng(10 + _d).normal(size=(150, _d))
        for _name, _f in _cands.items():
            _K = _f(_X)
            _w = np.linalg.eigvalsh(_K)
            _res[(_name, _d)] = (ex.is_psd(_K), _w[0] / np.abs(_w).max())
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _names = list(_cands)
    _y = np.arange(len(_names))
    for _d, _off in ((1, -0.18), (2, 0.18)):
        _vals = [_res[(n, _d)][1] for n in _names]
        _ax.barh(_y + _off, _vals, height=0.34, label=f"d = {_d}")
    _ax.axvline(0, color="k", lw=0.8)
    _ax.set_yticks(_y, _names, fontsize=8)
    _ax.invert_yaxis()
    _ax.set_xscale("symlog", linthresh=1e-4)
    _ax.set(xlabel="smallest eigenvalue / largest |eigenvalue|  (symmetric log scale)", xlim=(-1.5, 0.05))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _table = "| similarity | PSD in 1-D | PSD in 2-D |\n|---|---|---|\n" + "\n".join(
        f"| {n} | {'✓' if _res[(n, 1)][0] else '✗'} ({_res[(n, 1)][1]:.1e}) | "
        f"{'✓' if _res[(n, 2)][0] else '✗'} ({_res[(n, 2)][1]:.1e}) |" for n in _names)
    mo.vstack([_fig, mo.md(_table)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the triangle and $\cos\|x - y\|$ are kernels on the line. what changes in
    the plane? why is $\exp(-\|x - y\|^4)$ not a kernel when $\exp(-\|x - y\|^2)$ is? a
    function passed on these 150 points: is it a kernel?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: interpolating with RBF bumps

    12 noisy samples of $\sin x$ on $[0, 10]$. your `interpolate` finds the minimum-norm
    interpolant for an RBF kernel with width $\gamma$, `kernel_expansion` draws it, and
    `rkhs_norm` measures it. we also report the condition number of $K$. move the slider
    (it sets $\log_{10}\gamma$).
    """)
    return


@app.cell
def _():
    log_gamma = mo.ui.slider(-2.5, 1.5, step=0.25, value=-0.5, label="log10 gamma")
    log_gamma
    return (log_gamma,)


@app.cell
def _():
    p3 = ui.predict(
        "every interpolant passes through all 12 points. which gamma gives the interpolant with the "
        "SMALLEST RKHS norm: a tiny one (very smooth bumps), a medium one, or a huge one (spikes)?"
    )
    p3
    return (p3,)


@app.cell
def _(log_gamma, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _x = np.sort(_rng.uniform(0, 10, 12))[:, None]
    _y = np.sin(_x[:, 0]) + 0.3 * _rng.normal(size=12)
    _grid = np.linspace(-0.5, 10.5, 600)[:, None]
    _gammas = 10.0 ** np.arange(-2.5, 1.51, 0.25)
    _norms, _conds = [], []
    for _g in _gammas:
        _K = ex.rbf_kernel(_x, gamma=_g)
        _a = ex.interpolate(_x, _y, lambda A, B=None, g=_g: ex.rbf_kernel(A, B, gamma=g))
        _norms.append(ex.rkhs_norm(_a, _K))
        _conds.append(np.linalg.cond(_K))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4))
    for _g, _c in [(0.01, "C0"), (10 ** log_gamma.value, "C3"), (10.0, "C2")]:
        _k = lambda A, B=None, g=_g: ex.rbf_kernel(A, B, gamma=g)  # noqa: E731
        _f = ex.kernel_expansion(ex.interpolate(_x, _y, _k), _x, _grid, _k)
        _a1.plot(_grid[:, 0], np.clip(_f, -4, 4), color=_c, lw=1.3, label=f"γ = {_g:.3g}")
    _a1.plot(_x[:, 0], _y, "ko", ms=5, label="data")
    _a1.plot(_grid[:, 0], np.sin(_grid[:, 0]), "k:", lw=1, label="sin x")
    _a1.set(xlabel="x", ylim=(-4, 4), title="minimum-norm interpolants (clipped to ±4)")
    _a1.legend(fontsize=7, loc="lower left")
    _a2.loglog(_gammas, _norms, "o-", label="RKHS norm of the interpolant")
    _a2.loglog(_gammas, _conds, "s--", label="condition number of K")
    _a2.axvline(10 ** log_gamma.value, color="C3", lw=1)
    _a2.set(xlabel="γ")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _i = int(np.argmin(np.abs(np.log10(_gammas) - log_gamma.value)))
    mo.vstack([_fig, mo.md(
        f"γ = {_gammas[_i]:.3g}: norm **{_norms[_i]:.3g}**, cond(K) **{_conds[_i]:.3g}**. "
        f"γ = 0.00316: norm {_norms[0]:.3g}, cond {_conds[0]:.3g} · γ = 31.6: norm {_norms[-1]:.3g} "
        f"(‖y‖ = {np.linalg.norm(_y):.3g}), cond {_conds[-1]:.3g}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the spiky interpolant have a small norm, and what does its norm
    approach as $\gamma \to \infty$? why do norms computed with *different* $\gamma$ not
    measure the same thing? what makes the small-$\gamma$ interpolant swing so far, and what
    would you change to get a curve close to $\sin x$ (topic 05)?
    """)
    return


if __name__ == "__main__":
    app.run()
