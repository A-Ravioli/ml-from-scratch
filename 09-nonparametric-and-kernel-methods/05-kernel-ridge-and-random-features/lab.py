import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.kernels import rbf_kernel  # yours, from topic 03
    from mlfs.preprocessing import StandardScaler  # yours, from ch 07 · topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    GAMMA, ALPHA = 0.2, 0.1  # kernel width and ridge strength for standardized California features


@app.function
def housing_split(n_train, n_test, seed=0):
    """standardized California features (scaler fitted on the training rows) and centred targets."""
    X, y, _ = datasets.california_housing()
    perm = np.random.default_rng(seed).permutation(len(X))
    n_train = min(n_train, len(X) // 2)
    tr, te = perm[:n_train], perm[n_train:n_train + n_test]
    scaler = StandardScaler().fit(X[tr])
    y_mean = y[tr].mean()
    X_te = scaler.transform(X[te]) if len(te) else X[te]
    return scaler.transform(X[tr]), y[tr] - y_mean, X_te, y[te] - y_mean


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # kernel ridge & random features · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. experiments 1 and 2 use California housing with
    standardized features, an RBF kernel with $\gamma = {GAMMA}$ and ridge strength
    $\alpha = {ALPHA}$ (chosen by a quick validation run).

    {"**offline mode:** California housing is a 1,000-row synthetic stand-in with a linear target; run without `MLFS_OFFLINE` for the real numbers." if OFFLINE else ""}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: approximating a real Gram matrix

    1,500 districts. for $D$ from 10 to 1,000 we compute your `gram_error` for
    `RandomFourierFeatures(D)` and `Nystroem(D)` (median of 3 seeds), and compare with the best
    possible rank-$D$ approximation of $K$ (from its eigenvalues).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with D = 100 features, how will random Fourier features and Nyström compare: about the same, "
        "Nyström 2× better, 10× better, 100× better?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _X = housing_split(1500, 0, seed=1)[0]
    _K = rbf_kernel(_X, gamma=GAMMA)
    _lam = np.sort(np.linalg.eigvalsh(_K))[::-1]
    _Ds = [10, 20, 50, 100, 200, 500, 1000]
    _rff, _ny, _best = [], [], []
    for _D in _Ds:
        _rff.append(np.median([ex.gram_error(ex.RandomFourierFeatures(_D, gamma=GAMMA, rng=s).fit_transform(_X), _K)
                               for s in range(3)]))
        _ny.append(np.median([ex.gram_error(ex.Nystroem(_D, gamma=GAMMA, rng=s).fit_transform(_X), _K)
                              for s in range(3)]))
        _best.append(np.sqrt(np.sum(_lam[_D:] ** 2) / np.sum(_lam**2)))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    _a1.loglog(_Ds, _rff, "o-", label="random Fourier features")
    _a1.loglog(_Ds, _ny, "s-", label="Nyström")
    _a1.loglog(_Ds, _best, "k--", lw=1, label="best rank-D (eigenvalues)")
    _a1.loglog(_Ds, _rff[0] * np.sqrt(_Ds[0] / np.array(_Ds)), "C0:", lw=1, label=r"$\propto 1/\sqrt{D}$")
    _a1.set(xlabel="number of features D", ylabel="‖ZZᵀ − K‖_F / ‖K‖_F", title="1,500 districts, RBF γ = 0.2")
    _a1.legend(fontsize=7)
    _a2.semilogy(np.arange(1, 301), _lam[:300] / _lam[0])
    _a2.set(xlabel="k", ylabel="λ_k / λ_1", title="eigenvalues of K")
    _fig.tight_layout()
    _i = _Ds.index(100)
    mo.vstack([_fig, mo.md(
        f"D = 100: random features **{_rff[_i]:.3f}**, Nyström **{_ny[_i]:.4f}** ({_rff[_i] / _ny[_i]:.0f}× smaller), "
        f"best possible {_best[_i]:.4f}. D = 1000: random features {_rff[-1]:.3f}, Nyström {_ny[-1]:.1e}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the random-feature error fall like $1/\sqrt D$ whatever the data,
    while Nyström follows the eigenvalue curve? what is it about this $K$'s spectrum that makes
    a few hundred landmarks enough?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does a better Gram matrix mean a better model?

    4,000 training districts, 2,000 test districts. exact `KernelRidge` against `primal_ridge`
    on $D$ random Fourier or Nyström features, all with the same $\gamma$ and $\alpha$. we
    report test RMSE (in \$100,000) and fit time (wall clock, this machine).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "how many Nyström features do you need to match exact kernel ridge's test error to within 1%? "
        "and random Fourier features?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _Xtr, _ytr, _Xte, _yte = housing_split(4000, 2000, seed=0)
    _rmse = lambda pred: float(np.sqrt(np.mean((pred - _yte) ** 2)))  # noqa: E731
    _t0 = time.perf_counter()
    _exact = ex.KernelRidge(ALPHA, gamma=GAMMA).fit(_Xtr, _ytr)
    _t_exact = time.perf_counter() - _t0
    _r_exact = _rmse(_exact.predict(_Xte))
    _Ds = [10, 30, 100, 300, 1000, 2000]
    _res = {"random Fourier": [], "Nyström": []}
    for _name, _make in [("random Fourier", lambda D: ex.RandomFourierFeatures(D, gamma=GAMMA, rng=0)),
                         ("Nyström", lambda D: ex.Nystroem(D, gamma=GAMMA, rng=0))]:
        for _D in _Ds:
            _t0 = time.perf_counter()
            _f = _make(_D).fit(_Xtr)
            _w = ex.primal_ridge(_f.transform(_Xtr), _ytr, ALPHA)
            _t = time.perf_counter() - _t0
            _res[_name].append((_rmse(_f.transform(_Xte) @ _w), _t))
    _base = _rmse(np.zeros_like(_yte))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    for _name, _r in _res.items():
        _a1.semilogx(_Ds, [r[0] for r in _r], "o-", label=_name)
        _a2.loglog(_Ds, [r[1] for r in _r], "o-", label=_name)
    _a1.axhline(_r_exact, color="k", ls="--", lw=1, label=f"exact KRR ({_r_exact:.3f})")
    _a1.set(xlabel="D", ylabel="test RMSE ($100k)", title=f"predicting the mean: {_base:.3f}")
    _a1.legend(fontsize=8)
    _a2.axhline(_t_exact, color="k", ls="--", lw=1, label=f"exact KRR ({_t_exact:.2f} s)")
    _a2.set(xlabel="D", ylabel="fit time (s)")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _fmt = lambda name: ", ".join(f"D={D}: {r:.3f}" for D, (r, _) in zip(_Ds, _res[name]))  # noqa: E731
    mo.vstack([_fig, mo.md(
        f"exact KRR: **{_r_exact:.3f}** (mean baseline {_base:.3f}). random Fourier: {_fmt('random Fourier')}. "
        f"Nyström: {_fmt('Nyström')}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how do the test errors compare with experiment 1's Gram errors: does a
    10× better approximation give a 10× better model? where do the fit-time curves cross the
    exact solver, and how would that change at $n = 40{,}000$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: from interpolation to over-smoothing

    40 noisy samples of $\sin(2x)$ on $[0, 3]$, an RBF kernel with $\gamma = 5$. move the
    slider to change $\log_{10}\alpha$. the right panel shows the test error and the effective
    degrees of freedom $\mathrm{df}(\alpha) = \mathrm{tr}\,K(K + \alpha I)^{-1}$ for every $\alpha$.
    """)
    return


@app.cell
def _():
    log_alpha = mo.ui.slider(-6, 2, step=0.5, value=-1, label="log10 alpha")
    log_alpha
    return (log_alpha,)


@app.cell
def _():
    p3 = ui.predict(
        "40 points, noise sd 0.3. how many effective degrees of freedom will the best α use: about 40 "
        "(one per point), about 10, about 2?"
    )
    p3
    return (p3,)


@app.cell
def _(log_alpha, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(4)
    _x = np.sort(_rng.uniform(0, 3, 40))[:, None]
    _y = np.sin(2 * _x[:, 0]) + 0.3 * _rng.normal(size=40)
    _xt = np.linspace(0, 3, 400)[:, None]
    _yt = np.sin(2 * _xt[:, 0])
    _lam = np.linalg.eigvalsh(rbf_kernel(_x, gamma=5.0))
    _alphas = 10.0 ** np.arange(-6, 2.01, 0.5)
    _df = [float(np.sum(_lam / (_lam + a))) for a in _alphas]
    _err = [float(np.sqrt(np.mean((ex.KernelRidge(a, gamma=5.0).fit(_x, _y).predict(_xt) - _yt) ** 2))) for a in _alphas]
    _a = 10.0 ** log_alpha.value
    _fit = ex.KernelRidge(_a, gamma=5.0).fit(_x, _y).predict(_xt)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    _a1.plot(_x[:, 0], _y, "ko", ms=4, label="data")
    _a1.plot(_xt[:, 0], _yt, "k:", lw=1, label="sin 2x")
    _a1.plot(_xt[:, 0], _fit, "C3", lw=1.5, label=f"KRR, α = {_a:.2g}")
    _a1.set(xlabel="x", ylim=(-2, 2))
    _a1.legend(fontsize=8)
    _a2.loglog(_alphas, _err, "o-", label="RMSE against sin 2x")
    _a2.set(xlabel="α", ylabel="RMSE (log scale)")
    _b = _a2.twinx()
    _b.semilogx(_alphas, _df, "s--", color="C1", label="df(α)")
    _b.set_ylabel("effective degrees of freedom", color="C1")
    _a2.axvline(_a, color="C3", lw=1)
    _a2.legend(fontsize=8, loc="upper center")
    _fig.tight_layout()
    _ib = int(np.argmin(_err))
    mo.vstack([_fig, mo.md(
        f"α = {_a:.2g}: df = **{np.sum(_lam / (_lam + _a)):.1f}**, RMSE {np.sqrt(np.mean((_fit - _yt) ** 2)):.3f}. "
        f"best α on this grid: {_alphas[_ib]:.2g} with df = **{_df[_ib]:.1f}** and RMSE **{_err[_ib]:.3f}**; "
        f"α = 1e-6: df {_df[0]:.1f}, RMSE {_err[0]:.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does df fall towards 0 as $\alpha$ grows, and why is it well below 40
    even at $\alpha = 10^{-6}$ (look at the eigenvalues of $K$)? what does the fit look like at
    each end? why does the best model use far fewer degrees of freedom than there are data
    points, and how does the noise level enter?
    """)
    return


if __name__ == "__main__":
    app.run()
