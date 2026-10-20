import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.kernels import KernelRidge  # yours, from topic 05
    from mlfs.preprocessing import StandardScaler  # yours, from ch 07 · topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # gaussian processes · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    {"**offline mode:** experiment 3's California data is a synthetic stand-in with a linear target; run without `MLFS_OFFLINE` for the real numbers." if OFFLINE else ""}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what do the kernels believe?

    top row: 5 samples from your GP's *prior* under each kernel. bottom row: the same GP after
    observing 6 points (noise variance $10^{-4}$, hyperparameters fixed): 5 posterior samples,
    the mean, and ±2 standard deviations. move the slider to change the length scale.
    """)
    return


@app.cell
def _():
    lengthscale = mo.ui.slider(0.2, 3.0, step=0.1, value=1.0, label="length scale ℓ")
    lengthscale
    return (lengthscale,)


@app.cell
def _():
    p1 = ui.predict(
        "the three kernels have the same length scale and variance. how will their prior samples differ? "
        "and between two observations 3 units apart, how wide will the posterior band get?"
    )
    p1
    return (p1,)


@app.cell
def _(lengthscale, p1):
    ui.gate(p1)
    _grid = np.linspace(0, 10, 300)[:, None]
    _X = np.array([[1.0], [2.5], [3.0], [6.0], [8.0], [8.5]])
    _y = np.sin(_X[:, 0]) + 0.3 * _X[:, 0] - 1.5
    _fig, _axes = plt.subplots(2, 3, figsize=(12.5, 6), sharex=True, sharey=True)
    _widths = {}
    for _c, _kind in enumerate(["rbf", "matern52", "matern32"]):
        _gp = ex.GaussianProcessRegressor(kernel=_kind, lengthscale=lengthscale.value, variance=1.0, noise=1e-4,
                                          optimize=False)
        _axes[0, _c].plot(_grid[:, 0], _gp.sample_y(_grid, 5, rng=0).T, lw=1)
        _axes[0, _c].set_title(f"{_kind} prior", fontsize=10)
        _gp.fit(_X, _y)
        _m, _s = _gp.predict(_grid, return_std=True)
        _axes[1, _c].fill_between(_grid[:, 0], _m - 2 * _s, _m + 2 * _s, color="C0", alpha=0.15)
        _axes[1, _c].plot(_grid[:, 0], _gp.sample_y(_grid, 5, rng=1).T, lw=0.8, alpha=0.8)
        _axes[1, _c].plot(_grid[:, 0], _m, "k", lw=1.5)
        _axes[1, _c].plot(_X[:, 0], _y, "ko", ms=5)
        _axes[1, _c].set_title(f"{_kind} posterior", fontsize=10)
        _widths[_kind] = float(_gp.predict(np.array([[4.5]]), return_std=True)[1][0])
    _axes[0, 0].set_ylim(-3.5, 3.5)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"posterior std at x = 4.5 (midway between the observations at 3 and 6), ℓ = {lengthscale.value:.1f}: " +
        ", ".join(f"{k} **{v:.3f}**" for k, v in _widths.items()) + " (prior std 1).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are the Matérn-3/2 samples so much rougher than the RBF samples with the
    same length scale? what happens to the band at the observations, between them, and beyond
    the data? at which ℓ do the three posteriors agree most, and why?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: two explanations of the same data

    25 noisy samples of $\sin 3x$ on $[0, 10]$, standardized (so signal plus noise variance is
    about 1). we map your `log_marginal_likelihood` over the length scale and the noise
    variance $\sigma_n^2$, with the signal variance set to $\sigma_f^2 = 1 - \sigma_n^2$, and fit with `optimize=True` from two starts: a long length scale ($\ell = 5$, noise
    0.5) and a short one ($\ell = 0.3$, noise 0.01). then once more from the long start, with 6
    restarts.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "starting from ℓ = 5 and noise 0.5, will L-BFGS find the sine wave? how much better (in log "
        "marginal likelihood) is the better of the two answers?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(4)
    _X = _rng.uniform(0, 10, size=(25, 1))
    _y = np.sin(3 * _X[:, 0]) + 0.3 * _rng.normal(size=25)
    _y = (_y - _y.mean()) / _y.std()
    _long = ex.GaussianProcessRegressor(lengthscale=5.0, noise=0.5, rng=0).fit(_X, _y)
    _short = ex.GaussianProcessRegressor(lengthscale=0.3, noise=0.01, rng=0).fit(_X, _y)
    _multi = ex.GaussianProcessRegressor(lengthscale=5.0, noise=0.5, n_restarts=6, rng=0).fit(_X, _y)
    _ls = np.linspace(np.log(0.05), np.log(50), 60)
    _ns = np.linspace(0.005, 0.995, 50)
    _Z = np.array([[_long.log_marginal_likelihood(np.array([a, np.log(1 - b), np.log(b)])) for a in _ls] for b in _ns])
    _grid = np.linspace(-0.5, 10.5, 400)[:, None]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    _cs = _a1.contourf(np.exp(_ls), _ns, np.maximum(_Z, _Z.max() - 40), levels=30, cmap="viridis")
    _fig.colorbar(_cs, ax=_a1, label="log marginal likelihood")
    for _gp, _mk, _name in [(_long, "rs", "from ℓ=5"), (_short, "w^", "from ℓ=0.3")]:
        _frac = _gp.noise_ / (_gp.noise_ + _gp.variance_)
        _a1.plot(min(_gp.lengthscale_, 45.0), _frac, _mk, ms=9, mec="k",
                 label=f"{_name}: LML {_gp.log_marginal_likelihood():.2f}" + (" (ℓ off the chart)" if _gp.lengthscale_ > 50 else ""))
    _a1.set(xscale="log", xlabel="length scale ℓ", ylabel="noise share σₙ² / (σ_f² + σₙ²)",
            title="log marginal likelihood, σ_f² + σₙ² = 1")
    _a1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    _a2.plot(_X[:, 0], _y, "ko", ms=4)
    for _gp, _c, _name in [(_long, "C3", "from ℓ=5"), (_short, "C0", "from ℓ=0.3")]:
        _m, _s = _gp.predict(_grid, return_std=True)
        _a2.plot(_grid[:, 0], _m, color=_c, label=f"{_name}: ℓ={_gp.lengthscale_:.3g}, σ_f²={_gp.variance_:.2g}, "
                                                  f"σₙ²={_gp.noise_:.2g}")
        _a2.fill_between(_grid[:, 0], _m - 2 * _s, _m + 2 * _s, color=_c, alpha=0.1)
    _a2.set(xlabel="x", title="the two fitted GPs")
    _a2.legend(fontsize=7)
    _fig.tight_layout()
    _gap = _short.log_marginal_likelihood() - _long.log_marginal_likelihood()
    mo.vstack([_fig, mo.md(
        f"from ℓ = 5: LML **{_long.log_marginal_likelihood():.2f}** (ℓ = {_long.lengthscale_:.3g}, signal variance "
        f"{_long.variance_:.2g}, noise {_long.noise_:.2g}). from ℓ = 0.3: LML **{_short.log_marginal_likelihood():.2f}** "
        f"(ℓ = {_short.lengthscale_:.3g}, noise {_short.noise_:.2g}). difference **{_gap:.2f}** nats, a likelihood ratio "
        f"of about {np.exp(_gap):,.0f}. from ℓ = 5 with 6 restarts: LML **{_multi.log_marginal_likelihood():.2f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** describe the two answers in words. why is "everything is noise" a local
    maximum of the evidence at all? why does a gradient method started at $\ell = 5$ never see
    the sine wave? what do restarts buy, and what do they cost?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: honest error bars on real data

    California housing: 500 training districts (standardized features, target in \$100,000)
    and 2,000 test districts. your GP with a Matérn-5/2 kernel learns its hyperparameters (3
    restarts). we check its 95% predictive intervals, mean ± 1.96 $\sqrt{\sigma_*^2 + \sigma_n^2}$,
    on the test set, against a GP with hand-picked hyperparameters (ℓ = 1, σ_f² = 1,
    σₙ² = 0.01), and compare the learned GP's mean with your `KernelRidge` (RBF kernels on both
    sides, $\gamma = 1/(2\ell^2)$ and $\alpha = \sigma_n^2/\sigma_f^2$).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what fraction of test districts will fall inside the learned GP's 95% intervals? and inside the "
        "hand-picked GP's?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X, _y, _ = datasets.california_housing()
    _perm = np.random.default_rng(5).permutation(len(_X))
    _ntr = min(500, len(_X) // 2)
    _tr, _te = _perm[:_ntr], _perm[_ntr:_ntr + 2000]
    _sc = StandardScaler().fit(_X[_tr])
    _Xtr, _Xte, _ytr, _yte = _sc.transform(_X[_tr]), _sc.transform(_X[_te]), _y[_tr], _y[_te]

    def _report(gp):
        m, s = gp.predict(_Xte, return_std=True)
        half = 1.96 * np.sqrt(s**2 + gp.noise_)
        return (float(np.sqrt(np.mean((m - _yte) ** 2))), float(np.mean(np.abs(m - _yte) <= half)),
                float(np.mean(2 * half)))

    _learned = ex.GaussianProcessRegressor(kernel="matern52", n_restarts=3, rng=0).fit(_Xtr, _ytr)
    _fixed = ex.GaussianProcessRegressor(kernel="matern52", lengthscale=1.0, variance=1.0, noise=0.01,
                                         optimize=False).fit(_Xtr, _ytr)
    _rows = {"learned": _report(_learned), "hand-picked": _report(_fixed)}
    _rbf = ex.GaussianProcessRegressor(kernel="rbf", n_restarts=3, rng=0).fit(_Xtr, _ytr)
    _krr = KernelRidge(_rbf.noise_ / _rbf.variance_, gamma=1 / (2 * _rbf.lengthscale_**2)).fit(_Xtr, _ytr - _ytr.mean())
    _gap = float(np.max(np.abs(_krr.predict(_Xte) + _ytr.mean() - _rbf.predict(_Xte))))
    _m, _s = _learned.predict(_Xte, return_std=True)
    _order = np.argsort(_s)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    _z = (_yte - _m) / np.sqrt(_s**2 + _learned.noise_)
    _a1.hist(_z, bins=50, density=True, alpha=0.6, label="test residual / predictive std")
    _u = np.linspace(-5, 5, 200)
    _a1.plot(_u, np.exp(-_u**2 / 2) / np.sqrt(2 * np.pi), "k", lw=1, label="N(0, 1)")
    _a1.set(xlim=(-5, 5), xlabel="standardized residual", title="learned GP")
    _a1.legend(fontsize=8)
    _bins = np.array_split(_order, 5)
    _a2.plot([np.mean(_s[b]) for b in _bins], [np.sqrt(np.mean((_m[b] - _yte[b]) ** 2)) for b in _bins], "o-",
             label="RMSE in each fifth of test points, by σ*")
    _a2.plot([np.mean(_s[b]) for b in _bins], [np.sqrt(np.mean(_s[b] ** 2 + _learned.noise_)) for b in _bins], "s--",
             label="predicted: sqrt(σ*² + σₙ²)")
    _a2.set(xlabel="mean posterior std σ* in the group", ylabel="error ($100k)")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"learned (ℓ = {_learned.lengthscale_:.2f}, σ_f² = {_learned.variance_:.2f}, σₙ² = {_learned.noise_:.3f}): "
        f"RMSE **{_rows['learned'][0]:.3f}**, 95% interval coverage **{_rows['learned'][1]:.1%}**, mean width "
        f"{_rows['learned'][2]:.2f}. hand-picked: RMSE **{_rows['hand-picked'][0]:.3f}**, coverage "
        f"**{_rows['hand-picked'][1]:.1%}**, mean width {_rows['hand-picked'][2]:.2f}. RBF GP mean vs your KernelRidge: "
        f"largest difference **{_gap:.1e}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are the hand-picked intervals so wrong, and in which direction? the
    learned intervals cover about the right fraction overall: look at the histogram's tails
    and at the right panel. which districts does the GP get wrong *with confidence*? why do
    the GP mean and kernel ridge agree to round-off?
    """)
    return


if __name__ == "__main__":
    app.run()
