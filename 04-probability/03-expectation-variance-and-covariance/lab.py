import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def shortcut_float32(x):
    """the shortcut formula E[X^2] - E[X]^2 evaluated entirely in float32 (for comparison)."""
    x = np.asarray(x, dtype=np.float32)
    return float(np.mean(x * x, dtype=np.float32) - np.mean(x, dtype=np.float32) ** 2)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # expectation, variance & covariance · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: when does the textbook formula break?

    10,000 draws from $\mathcal{N}(\mu, 1)$: the true variance is 1 whatever the offset $\mu$.
    we compute it with your `naive_variance` (the shortcut), your `two_pass_variance`, your
    `RunningStats` (fed in batches of 100), and the shortcut formula in float32, the precision
    most GPU training runs in.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at what offset μ does each method's relative error reach 1% (float64 shortcut, float32 "
        "shortcut, two-pass, RunningStats)? when does the shortcut first return a negative variance?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _z = np.random.default_rng(0).normal(size=10_000)
    _true_var = float(np.var(_z))
    _offsets = np.logspace(0, 12, 49)
    _errs = {"shortcut (float64)": [], "shortcut (float32)": [], "two-pass": [], "RunningStats": []}
    _first_neg = {}
    for _mu in _offsets:
        _x = _mu + _z
        _rs = ex.RunningStats()
        for _chunk in np.array_split(_x, 100):
            _rs.update(_chunk)
        _vals = {"shortcut (float64)": ex.naive_variance(_x), "shortcut (float32)": shortcut_float32(_x),
                 "two-pass": ex.two_pass_variance(_x), "RunningStats": float(_rs.var)}
        for _k, _v in _vals.items():
            _errs[_k].append(abs(_v - _true_var) / _true_var + 1e-17)
            if _v < 0 and _k not in _first_neg:
                _first_neg[_k] = _mu
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    for _k, _e in _errs.items():
        _ax.loglog(_offsets, _e, ".-", label=_k)
    _ax.axhline(0.01, color="gray", lw=0.6, ls=":")
    _ax.set(xlabel="offset μ (the data are μ + N(0, 1))", ylabel="relative error of the variance",
            ylim=(1e-17, 1e9))
    _ax.legend(fontsize=8)
    _neg = ", ".join(f"{_k}: μ ≈ {_v:.0e}" for _k, _v in _first_neg.items()) or "none"
    mo.vstack([_fig, mo.md(f"first offset with a **negative** variance: {_neg}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the shortcut's error grows like $\varepsilon\mu^2/\sigma^2$. use that to
    predict where float64 and float32 cross 1% ($\varepsilon \approx 1.1 \times 10^{-16}$ and
    $6 \times 10^{-8}$) and compare with the plot. why do two-pass and `RunningStats` stay
    accurate, and why does even their error creep up at the far right?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: can you trust a Monte Carlo error bar?

    we estimate $\mathbb{E}[e^{cZ}]$ for $Z \sim \mathcal{N}(0, 1)$, whose true value is
    $e^{c^2/2}$, with your `mc_estimate`, and repeat the whole estimate 2,000 times for each
    sample size $n$. two questions: does the error shrink like $1/\sqrt n$, and how often does
    the 95% interval $\hat\mu \pm 1.96\,\widehat{\text{se}}$ actually contain the truth? the slider
    sets $c$: larger $c$ means a more skewed, heavier-tailed $e^{cZ}$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with c = 2 and n = 100 samples, what fraction of your '95%' intervals contain the true value?"
    )
    p2
    return (p2,)


@app.cell
def _():
    c_skew = mo.ui.slider(0.25, 2.5, value=2.0, step=0.25, label="c in E[exp(cZ)]", show_value=True)
    c_skew
    return (c_skew,)


@app.cell
def _(c_skew, p2):
    ui.gate(p2)
    _c = c_skew.value
    _truth = math.exp(_c**2 / 2)
    _rng = np.random.default_rng(1)
    _ns = [10, 30, 100, 300, 1000, 3000]
    _rows, _rmse = [], []
    for _n in _ns:
        _draws = np.exp(_c * _rng.normal(size=(2000, _n)))
        _est = [ex.mc_estimate(_row) for _row in _draws]
        _m = np.array([_e[0] for _e in _est])
        _se = np.array([_e[1] for _e in _est])
        _cover = np.mean(np.abs(_m - _truth) <= 1.96 * _se)
        _below = np.mean(_m < _truth)
        _rmse.append(np.sqrt(np.mean((_m - _truth) ** 2)))
        _rows.append(f"| {_n} | {_rmse[-1]:.4f} | {np.median(_se):.4f} | {_cover:.3f} | {_below:.3f} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.loglog(_ns, _rmse, "o-", label="actual RMS error")
    _ax.loglog(_ns, _rmse[0] * np.sqrt(_ns[0] / np.array(_ns)), "k:", label="1/√n reference")
    _ax.set(xlabel="n", ylabel="error of the estimate")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"true value e^(c²/2) = **{_truth:.4f}**.\n\n"
        "| n | RMS error | median reported SE | coverage of ±1.96 SE | fraction of estimates below truth |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the estimator is unbiased, so why do most estimates fall *below* the truth
    when $c$ is large? why are the intervals too short exactly then? (think about what a sample
    of 100 draws of $e^{2Z}$ usually doesn't contain.) what does this mean for reporting
    "mean ± 2 SE" on a heavy-tailed metric?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the eigenvalues of a sample covariance

    draw $n$ points from $\mathcal{N}(0, I_{100})$: the true covariance matrix is the identity,
    so all 100 true eigenvalues are exactly 1. compute the sample covariance with your
    `covariance` and look at *its* eigenvalues.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with n = 200 samples in d = 100 dimensions, what are the smallest and largest eigenvalues "
        "of the sample covariance?"
    )
    p3
    return (p3,)


@app.cell
def _():
    n_samples = mo.ui.dropdown(options=["120", "200", "500", "2000", "20000"], value="200",
                               label="number of samples n (d = 100)")
    n_samples
    return (n_samples,)


@app.cell
def _(n_samples, p3):
    ui.gate(p3)
    _d, _n = 100, int(n_samples.value)
    _X = np.random.default_rng(2).normal(size=(_n, _d))
    _eig = np.linalg.eigvalsh(ex.covariance(_X))
    _q = _d / _n
    _lo, _hi = (1 - math.sqrt(_q)) ** 2, (1 + math.sqrt(_q)) ** 2
    _grid = np.linspace(_lo, _hi, 400)[1:-1]
    _mp = np.sqrt((_hi - _grid) * (_grid - _lo)) / (2 * math.pi * _q * _grid)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.hist(_eig, bins=30, density=True, alpha=0.6, label="eigenvalues of your sample covariance")
    _ax.plot(_grid, _mp, "k-", lw=1, label="Marchenko–Pastur prediction")
    _ax.axvline(1.0, color="C3", ls="--", label="every true eigenvalue = 1")
    _ax.set(xlabel="eigenvalue", ylabel="density")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"n = {_n}, d = {_d}: smallest eigenvalue **{_eig.min():.3f}**, largest **{_eig.max():.3f}**, "
        f"mean {_eig.mean():.3f}. Marchenko–Pastur edges (1 ± √(d/n))² = [{_lo:.3f}, {_hi:.3f}].")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** every entry of the sample covariance is an unbiased estimate of the right
    number, so why are its eigenvalues so wrong? why is their *mean* still right (hint: the
    trace)? what would PCA on this data "discover", and how many samples per dimension do you
    need before the spread is small?
    """)
    return


if __name__ == "__main__":
    app.run()
