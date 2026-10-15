import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()


@app.function
def bimodal_sample(n, rng):
    """n draws from 0.5 N(-2, 0.5^2) + 0.5 N(2, 0.5^2)."""
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    return 2.0 * sign + 0.5 * rng.normal(size=n)


@app.function
def bimodal_pdf(x):
    """the true density of bimodal_sample."""
    return 0.5 * norm.pdf(x, -2, 0.5) + 0.5 * norm.pdf(x, 2, 0.5)


@app.function
def count_peaks(y):
    """number of strict local maxima of a sampled curve, treating flat tops as one peak."""
    y = np.asarray(y)
    keep = np.concatenate([[True], np.diff(y) != 0])
    y = y[keep]
    return int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # density estimation · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    {"**offline mode:** experiment 3's California data is replaced by a synthetic Gaussian blob; run without `MLFS_OFFLINE` for the real map." if OFFLINE else ""}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: same data, same bins, different origin

    39 points from a mixture of three bumps. your `histogram_density` draws a histogram with a
    fixed bin width, for 20 bin origins between 0 and one bin width. we count the peaks of
    each histogram and show the two most different ones.
    """)
    return


@app.cell
def _():
    width = mo.ui.slider(0.3, 1.5, step=0.1, value=0.6, label="bin width")
    width
    return (width,)


@app.cell
def _():
    p1 = ui.predict("with the bin width fixed, how much can the number of peaks change when only the origin moves?")
    p1
    return (p1,)


@app.cell
def _(p1, width):
    ui.gate(p1)
    _rng = np.random.default_rng(4)
    _x = np.concatenate([_rng.normal(-1.5, 0.6, 16), _rng.normal(0.8, 0.5, 13), _rng.normal(2.6, 0.5, 10)])
    _h = width.value
    _grid = np.linspace(-4.5, 5, 2000)
    _origins = np.linspace(0, _h, 20, endpoint=False)
    _dens = [ex.histogram_density(_x, _grid, _h, o) for o in _origins]
    _peaks = np.array([count_peaks(d) for d in _dens])
    _lo, _hi = int(np.argmin(_peaks)), int(np.argmax(_peaks))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for _ax, _i in zip(_axes, (_lo, _hi)):
        _ax.fill_between(_grid, _dens[_i], step="mid", alpha=0.5)
        _ax.plot(_x, np.full_like(_x, -0.01), "|k", ms=8)
        _ax.set(title=f"origin {_origins[_i]:.2f}: {_peaks[_i]} peaks", xlabel="x")
    _axes[0].set_ylabel("density")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"bin width {_h:.1f}: over 20 origins the number of peaks ranges from **{_peaks.min()}** to "
                           f"**{_peaks.max()}** (counts: {np.bincount(_peaks).tolist()} for 0, 1, 2, ... peaks)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the data and the bin width didn't change. what did? which features of a
    histogram can you trust, and how would a kernel density estimate avoid the problem?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the bandwidth trade-off, measured

    the true density is an equal mixture of $\mathcal N(-2, 0.5^2)$ and $\mathcal N(2, 0.5^2)$.
    we draw 40 samples of 200 points, fit your `KernelDensity` for each bandwidth on a grid, and
    measure, over $x \in [-5, 5]$: the integrated squared bias of the average estimate, the
    integrated variance across samples, and their sum (the MISE). we also record what your
    `silverman_bandwidth` and your `select_bandwidth` (5-fold CV) choose on each sample.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the two modes are 4 standard deviations apart. will Silverman's rule pick a bandwidth near the "
        "MISE-optimal one? will cross-validation?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(2)
    _grid = np.linspace(-5, 5, 400)
    _dx = _grid[1] - _grid[0]
    _hs = np.geomspace(0.05, 2.0, 18)
    _hs_cv = np.geomspace(0.05, 2.0, 40)
    _true = bimodal_pdf(_grid)
    _est = np.empty((40, len(_hs), len(_grid)))
    _silver, _cv = [], []
    for _r in range(40):
        _x = bimodal_sample(200, _rng)[:, None]
        for _j, _h in enumerate(_hs):
            _est[_r, _j] = np.exp(ex.KernelDensity(_h).fit(_x).score_samples(_grid[:, None]))
        _silver.append(ex.silverman_bandwidth(_x))
        _cv.append(ex.select_bandwidth(_x, _hs_cv, cv=5, rng=_r)[0])
    _bias2 = np.sum((_est.mean(axis=0) - _true) ** 2, axis=1) * _dx
    _var = np.sum(_est.var(axis=0), axis=1) * _dx
    _mise = _bias2 + _var
    _best = _hs[np.argmin(_mise)]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    _a1.loglog(_hs, _bias2, "o-", label="integrated bias²")
    _a1.loglog(_hs, _var, "s-", label="integrated variance")
    _a1.loglog(_hs, _mise, "k-", lw=2, label="MISE")
    _a1.axvline(np.median(_silver), color="C3", ls="--", label=f"Silverman (median {np.median(_silver):.2f})")
    _a1.axvline(np.median(_cv), color="C2", ls=":", lw=2, label=f"CV (median {np.median(_cv):.2f})")
    _a1.set(xlabel="bandwidth h", ylabel="integrated error", title="40 samples of n = 200")
    _a1.legend(fontsize=7)
    _x0 = bimodal_sample(200, np.random.default_rng(99))[:, None]
    _a2.plot(_grid, _true, "k-", lw=2, label="true density")
    for _h, _name, _c in [(ex.silverman_bandwidth(_x0), "Silverman", "C3"),
                          (ex.select_bandwidth(_x0, _hs_cv, rng=0)[0], "CV", "C2"), (0.05, "h = 0.05", "C7")]:
        _a2.plot(_grid, np.exp(ex.KernelDensity(_h).fit(_x0).score_samples(_grid[:, None])), color=_c,
                 lw=1.2, label=f"{_name} (h = {_h:.2f})")
    _a2.set(xlabel="x", ylabel="density", title="one sample")
    _a2.legend(fontsize=7)
    _fig.tight_layout()
    _imise = lambda h: float(np.interp(np.log(h), np.log(_hs), _mise))  # noqa: E731
    mo.vstack([_fig, mo.md(
        f"MISE-optimal h ≈ **{_best:.2f}** (MISE {_mise.min():.4f}). Silverman: median **{np.median(_silver):.2f}** "
        f"(MISE ≈ {_imise(np.median(_silver)):.4f}). CV: median **{np.median(_cv):.2f}**, middle half "
        f"{np.percentile(_cv, 25):.2f}–{np.percentile(_cv, 75):.2f} (MISE ≈ {_imise(np.median(_cv)):.4f}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the bias rise and the variance fall with $h$, and at what rates
    (compare the slopes with idea 5)? why does Silverman's rule overshoot on this density: what
    did it assume about $p''$? what does cross-validation pay for being adaptive?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: where Californians live

    the (longitude, latitude) of California housing districts, a real 2-D density. we choose
    the bandwidth with your `select_bandwidth` on 3,000 districts, compare its held-out
    log-likelihood on 3,000 *other* districts with Silverman's rule, draw the density map, and
    sample 2,000 new "districts" from your KDE.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "Silverman's rule assumes one Gaussian blob. how will its bandwidth compare with the CV choice "
        "on a map of a state whose people live in a few cities? (in degrees: about the same? 2x? 10x?)"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X = datasets.california_housing()[0][:, :2]
    _rng = np.random.default_rng(3)
    _perm = _rng.permutation(len(_X))
    _m = min(3000, len(_X) // 2)  # the offline stand-in has only 1,000 rows
    _Xa, _Xb = _X[_perm[:_m]], _X[_perm[_m:2 * _m]]
    _hs = np.geomspace(0.01, 1.0, 15)
    _h_cv, _scores = ex.select_bandwidth(_Xa, _hs, cv=5, rng=0)
    _h_s = ex.silverman_bandwidth(_Xa)
    _ll = {_name: float(np.mean(ex.KernelDensity(_h).fit(_Xa).score_samples(_Xb)))
           for _name, _h in [("cv", _h_cv), ("silverman", _h_s)]}
    _kde = ex.KernelDensity(_h_cv).fit(_Xa)
    _lo, _hi = _X.min(axis=0) - 0.5, _X.max(axis=0) + 0.5
    _gx, _gy = np.meshgrid(np.linspace(_lo[0], _hi[0], 120), np.linspace(_lo[1], _hi[1], 120))
    _G = np.column_stack([_gx.ravel(), _gy.ravel()])
    _logp = np.concatenate([_kde.score_samples(_G[_i:_i + 1500]) for _i in range(0, len(_G), 1500)])
    _S = _kde.sample(2000, rng=4)
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(13, 4.2))
    _a1.semilogx(_hs, _scores, "o-")
    _a1.axvline(_h_cv, color="C2", ls=":", label=f"CV: h = {_h_cv:.3f}")
    _a1.axvline(_h_s, color="C3", ls="--", label=f"Silverman: h = {_h_s:.3f}")
    _a1.set(xlabel="bandwidth (degrees)", ylabel="mean held-out log-likelihood", title="5-fold CV on 3,000 districts")
    _a1.legend(fontsize=7)
    _a2.contourf(_gx, _gy, np.maximum(_logp, _logp.max() - 10).reshape(_gx.shape), levels=30, cmap="viridis")
    _a2.set(xlabel="longitude", ylabel="latitude", title="log density (CV bandwidth)", aspect="equal")
    _a3.plot(_Xb[:2000, 0], _Xb[:2000, 1], ".", ms=1.5, alpha=0.5, label="real (held-out) districts")
    _a3.plot(_S[:, 0], _S[:, 1], ".", ms=1.5, alpha=0.5, color="C1", label="sampled from your KDE")
    _a3.set(xlabel="longitude", title="2,000 real vs 2,000 sampled", aspect="equal")
    _a3.legend(fontsize=7, markerscale=6)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"CV bandwidth **{_h_cv:.3f}°**, Silverman **{_h_s:.3f}°** ({_h_s / _h_cv:.1f}× wider). held-out mean "
        f"log-likelihood: CV **{_ll['cv']:.3f}**, Silverman **{_ll['silverman']:.3f}** nats per district "
        f"(a gain of {_ll['cv'] - _ll['silverman']:.2f} nats, i.e. {np.exp(_ll['cv'] - _ll['silverman']):.1f}× the "
        "density at a typical real district).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does a single global bandwidth struggle with this map (think about
    downtown Los Angeles and the Mojave desert)? why do some sampled districts fall in the
    Pacific? what would a bandwidth that varies with the local density of points do?
    """)
    return


if __name__ == "__main__":
    app.run()
