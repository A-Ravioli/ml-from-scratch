import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.kernels import KNeighborsClassifier  # yours, from ch 09 · topic 01
    from mlfs.linear import LogisticRegression  # yours, from ch 08 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # rademacher complexity · lab

    every experiment here runs **your** estimators and bounds from `exercise.py`. for each one:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how complexity grows with the number of hypotheses

    on $n = 100$ points, finite classes of $N$ **random** sign vectors ($N = 1$ to $10^4$),
    and the class of all two-sided **thresholds** on 100 points of the line (200 distinct
    labelings). for each class: your `empirical_rademacher` (2,000 draws) and your
    `massart_bound`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "going from 10 random hypotheses to 10,000, by what factor does the Rademacher complexity grow? "
        "are 200 thresholds more or less complex than 200 random hypotheses?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _n = 100
    _rng = np.random.default_rng(0)
    _Ns = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    _pool = np.sign(_rng.normal(size=(max(_Ns), _n)))
    _est = [ex.empirical_rademacher(_pool[:N], n_draws=2000, rng=k) for k, N in enumerate(_Ns)]
    _mas = [ex.massart_bound(_pool[:N]) for N in _Ns]
    _x = np.sort(_rng.random(_n))
    _cuts = np.concatenate([[-1.0], (_x[:-1] + _x[1:]) / 2, [2.0]])
    _thr = np.array([np.where(_x >= c, 1.0, -1.0) for c in _cuts])
    _thr = np.unique(np.vstack([_thr, -_thr]), axis=0)
    _thr_est, _thr_mas = ex.empirical_rademacher(_thr, n_draws=2000, rng=99), ex.massart_bound(_thr)
    _rand200 = ex.empirical_rademacher(_pool[: len(_thr)], n_draws=2000, rng=98)
    _fig, _ax = plt.subplots(figsize=(7, 4.3))
    _ax.semilogx(_Ns, _est, "ko-", label="random hypotheses: Monte Carlo")
    _ax.semilogx(_Ns, _mas, "C3--", label="Massart bound √(2 ln N / n)")
    _ax.semilogx([len(_thr)], [_thr_est], "C0s", ms=9, label=f"{len(_thr)} thresholds: Monte Carlo")
    _ax.semilogx([len(_thr)], [_thr_mas], "C0x", ms=9, label="thresholds: Massart")
    _ax.set(xlabel="N (number of distinct hypotheses)", ylabel="empirical Rademacher complexity (n = 100)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {N:,} | {e:.4f} | {m:.4f} | {e / m:.2f} |" if N > 1 else f"| 1 | {e:.4f} | 0 | – |"
                      for N, e, m in zip(_Ns, _est, _mas))
    mo.vstack([_fig, mo.md(
        "| N | Monte Carlo | Massart | ratio |\n|---|---|---|---|\n" + _rows
        + f"\n\n{len(_thr)} two-sided thresholds: **{_thr_est:.4f}** (Massart {_thr_mas:.4f}); "
        f"{len(_thr)} random hypotheses: **{_rand200:.4f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the complexity of random hypotheses grow like $\sqrt{\ln N}$, and
    why is Massart's lemma nearly tight for them (think of the maximum of $N$ nearly
    independent Gaussians)? thresholds have as many labelings as the 200 random hypotheses
    but a much smaller complexity: what about their *structure* makes them worse at fitting
    noise? what does that say about bounds that only count hypotheses?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: complexity against norm constraints

    linear predictors $x \mapsto w^\top x$ with $\|w\|_2 \le 1$ or $\|w\|_1 \le 1$, on $n$ points
    with random $\pm1$ features in dimension $d$ from 1 to 10,000. your `norm_ball_rademacher`
    (300 draws) for both balls, with the Jensen bound $\sqrt{d/n}$ for the L2 ball and the
    Massart-style bound $\sqrt{2\ln(2d)/n}$ for the L1 ball (its sup is attained at one of the
    $2d$ vertices $\pm e_j$).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "from d = 10 to d = 10,000 features, how much does the complexity of each ball grow?"
    )
    n2 = mo.ui.dropdown(options=["50", "100", "400"], value="100", label="n (sample size)")
    mo.vstack([p2, n2])
    return n2, p2


@app.cell
def _(n2, p2):
    ui.gate(p2)
    _n = int(n2.value)
    _ds = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    _rng = np.random.default_rng(1)
    _l2, _l1 = [], []
    for _k, _d in enumerate(_ds):
        _X = np.sign(_rng.normal(size=(_n, _d)))
        _l2.append(ex.norm_ball_rademacher(_X, 1.0, norm=2, n_draws=300, rng=_k))
        _l1.append(ex.norm_ball_rademacher(_X, 1.0, norm=1, n_draws=300, rng=_k))
    _dd = np.array(_ds, dtype=float)
    _fig, _ax = plt.subplots(figsize=(7, 4.3))
    _ax.loglog(_ds, _l2, "C3o-", label="‖w‖₂ ≤ 1: Monte Carlo")
    _ax.loglog(_dd, np.sqrt(_dd / _n), "C3:", label="Jensen bound √(d/n)")
    _ax.loglog(_ds, _l1, "C0s-", label="‖w‖₁ ≤ 1: Monte Carlo")
    _ax.loglog(_dd, np.sqrt(2 * np.log(2 * _dd) / _n), "C0:", label="√(2 ln(2d)/n)")
    _ax.set(xlabel="d (number of features)", ylabel="empirical Rademacher complexity", title=f"n = {_n}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {d:,} | {a:.3f} | {math.sqrt(d / _n):.3f} | {b:.3f} | {math.sqrt(2 * math.log(2 * d) / _n):.3f} |"
                      for d, a, b in zip(_ds, _l2, _l1))
    mo.vstack([_fig, mo.md("| d | L2 ball | √(d/n) | L1 ball | √(2 ln 2d / n) |\n|---|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the L2 ball's complexity grows like $\sqrt d$ and passes 1 around $d = n$
    (with predictions compared to a margin of 1, a bound built on it then says nothing); the
    L1 ball's grows like $\sqrt{\ln d}$. why the difference? (which dual norm of
    $\sum_i\sigma_ix_i$ does each one measure, and how big is each norm of a random $\pm1$
    vector?) what does this suggest about learning with many more features than examples, and
    about the lasso? at $d = 1$ the two balls coincide: why is their common value
    $\mathbb E|\sum_i\sigma_i|/n \approx \sqrt{2/(\pi n)}$, below the Jensen bound $1/\sqrt n$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: fitting random labels

    real data: 1,000 MNIST training digits (offline: the synthetic stand-in), projected onto
    their top $k$ principal components (the 1,000 centred digits span 593 dimensions, so $k$ goes up to 593). for each $k$, logistic regression (your
    `LogisticRegression`, $\alpha = 10^{-7}$, nearly unregularized) is trained on **random** $\pm1$ labels and your
    `rademacher_via_erm` records how well it fits them (10 draws). then the same model is
    trained on the real task, even vs odd digits, and tested on 2,000 held-out digits. the last
    row is 1-nearest-neighbour on all 784 pixels. the "bound" column is
    $\text{train error} + \hat{\mathfrak R} + 3\sqrt{\ln(2/\delta)/(2n)}$ with $\delta = 0.05$:
    your `rademacher_bound` applied to the 0-1 loss class, whose complexity is half the
    classifier class's (see the lesson), with the random-label correlation standing in for
    $\hat{\mathfrak R}$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with 1,000 training digits, at what number of components k will logistic regression fit random labels "
        "perfectly? 1-NN fits them perfectly: does that make its test error bad?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _d = datasets.mnist()
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1).astype(float) / 255.0
    _y = _d["y_train"]
    _rng = np.random.default_rng(2)
    _perm = _rng.permutation(len(_X))
    _n = 1000
    _tr, _te = _perm[:_n], _perm[_n:_n + 2000]
    _mu = _X[_tr].mean(axis=0)
    _, _, _Vt = np.linalg.svd(_X[_tr] - _mu, full_matrices=False)
    _ytr, _yte = (_y[_tr] % 2).astype(int), (_y[_te] % 2).astype(int)
    _rows, _res = [], []
    for _k in [2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 593]:
        _Z = (_X - _mu) @ _Vt[:_k].T
        _Z /= _Z[_tr].std()
        _Ztr, _Zte = _Z[_tr], _Z[_te]

        def _fp(Z, s):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(alpha=1e-7).fit(Z, (s > 0).astype(int))
            return np.where(m.predict(Z) == 1, 1.0, -1.0)

        _rad = ex.rademacher_via_erm(_fp, _Ztr, n_draws=10, rng=_k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _m = LogisticRegression(alpha=1e-7).fit(_Ztr, _ytr)
        _tre, _tee = float(np.mean(_m.predict(_Ztr) != _ytr)), float(np.mean(_m.predict(_Zte) != _yte))
        _b = ex.rademacher_bound(_tre, _rad / 2, _n, 0.05)
        _res.append((_k, _rad, _tre, _tee, _b))
        _rows.append(f"| logistic, k = {_k} | {_rad:.3f} | {_tre:.3f} | {_tee:.3f} | {_tee - _tre:+.3f} | {_b:.3f} |")
    _knn = KNeighborsClassifier(1)
    _rad_nn = ex.rademacher_via_erm(lambda Z, s: _knn.fit(Z, s).predict(Z), _X[_tr], n_draws=3, rng=0)
    _knn.fit(_X[_tr], _ytr)
    _tee_nn = float(np.mean(_knn.predict(_X[_te]) != _yte))
    _b_nn = ex.rademacher_bound(0.0, _rad_nn / 2, _n, 0.05)
    _rows.append(f"| 1-NN, 784 pixels | {_rad_nn:.3f} | 0.000 | {_tee_nn:.3f} | {_tee_nn:+.3f} | {_b_nn:.3f} |")
    _fig, _ax = plt.subplots(figsize=(7, 4.3))
    _ks = [r[0] for r in _res]
    _ax.semilogx(_ks, [r[1] for r in _res], "ko-", label="random-label correlation (≈ R̂)")
    _ax.semilogx(_ks, [r[3] for r in _res], "C0s-", label="test error, real labels")
    _ax.semilogx(_ks, [r[2] for r in _res], "C0^:", label="train error, real labels")
    _ax.semilogx(_ks, [min(r[4], 1.5) for r in _res], "C3--", label="Rademacher bound (clipped at 1.5)")
    _ax.axhline(1.0, color="0.6", lw=0.8)
    _ax.axvline(_n / 2 - 1, color="0.6", ls=":", lw=1)
    _ax.text(_n / 2 - 1, 0.02, " k + 1 = n/2", fontsize=8, color="0.4")
    _ax.set(xlabel="k (principal components)", ylabel="value", title=f"logistic regression, n = {_n}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| model | random-label correlation | train error | test error | gap | bound |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at which $k$ does logistic regression start fitting random labels well, and
    why near $k \approx n$? follow the bound column: where is it informative, and where does it
    pass 1? the real-label gap grows with $k$ too, but much less than the random-label
    correlation: why can a model that fits noise perfectly still generalize on real labels?
    1-NN is the extreme case. is this random-label correlation a valid upper bound on the
    class's Rademacher complexity, or only a lower one (logistic regression isn't an exact
    0-1 ERM)?
    """)
    return


if __name__ == "__main__":
    app.run()
