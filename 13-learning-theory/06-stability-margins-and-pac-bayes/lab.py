import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from mlfs import datasets, notebooks, ui
    from mlfs.kernels import KNeighborsRegressor  # yours, from ch 09 · topic 01
    from mlfs.linear import LogisticRegression, Ridge  # yours, from ch 08
    from mlfs.theory import vc_bound  # yours, from topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # stability, margins & PAC-Bayes · lab

    every experiment here runs **your** code from `exercise.py` (and your `vc_bound` from
    topic 04). for each one: write down a prediction, lock it in, run it, then explain what
    happened. compare your explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a non-vacuous bound, actually computed

    MNIST, odd vs even digits (offline: the synthetic stand-in): logistic regression on all
    784 pixels plus a bias, trained on the 60,000 training images. the PAC-Bayes bound is for
    the **stochastic** classifier $\text{sign}(w'^\top x)$ with $w' \sim \mathcal N(w, \sigma^2 I)$,
    computed with your `pac_bayes_linear` ($\delta = 0.05$, a union bound over 25 values of
    $\sigma$ fixed in advance), in two versions:

    - **prior at 0**: $P = \mathcal N(0, \sigma^2 I)$, bound computed on all 60,000 images;
    - **data-dependent prior**: $P = \mathcal N(w_0, \sigma^2 I)$, where $w_0$ is a logistic
      regression trained on the *first* 30,000 images only, and the bound is computed on the
      other 30,000 (the prior must not see the data the bound is computed on).

    next to them: your VC bound for $d = 785$, and the test-set bound on the 10,000 test
    images (exact binomial, as in topic 04).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the VC bound for this model is about 0.89. what will the PAC-Bayes bound be: still vacuous (above 0.5), "
        "around 0.2, or within 0.01 of the test error?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _d = datasets.mnist()
    _add1 = lambda A: np.hstack([A.reshape(len(A), -1) / 255.0, np.ones((len(A), 1))])  # noqa: E731
    _X, _Xt = _add1(_d["X_train"]), _add1(_d["X_test"])
    _y, _yt = np.where(_d["y_train"] % 2 == 1, 1.0, -1.0), np.where(_d["y_test"] % 2 == 1, 1.0, -1.0)
    _n, _half = len(_y), len(_y) // 2

    def _fit(A, b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(alpha=1e-4).fit(A[:, :-1], (b > 0).astype(int))
        return np.append(m.coef_, m.intercept_)

    _w, _w0 = _fit(_X, _y), _fit(_X[:_half], _y[:_half])
    _sig = np.geomspace(0.003, 1.0, 25)
    _pb0 = ex.pac_bayes_linear(_w, _X, _y, sigmas=_sig, delta=0.05)
    _pbd = ex.pac_bayes_linear(_w, _X[_half:], _y[_half:], sigmas=_sig, prior_mean=_w0, delta=0.05)
    _tr = float(np.mean(np.sign(_X @ _w) != _y))
    _k = int(np.sum(np.sign(_Xt @ _w) != _yt))
    _te = _k / len(_yt)
    _tsb = float(stats.beta.ppf(0.95, _k + 1, len(_yt) - _k))
    _vcb = vc_bound(_n, _w.size, 0.05, _tr)
    _rows = [
        f"| deterministic $w$ | train error {_tr:.4f} | test error {_te:.4f} | – |",
        f"| VC bound (d = {_w.size}) | – | – | **{_vcb:.3f}** |",
        f"| PAC-Bayes, prior at 0 (σ = {_pb0['sigma']:.3f}, KL = {_pb0['kl']:.0f}) | Gibbs train {_pb0['emp_risk']:.4f} | "
        f"Gibbs test {ex.gibbs_error(_w, _pb0['sigma'], _Xt, _yt):.4f} | **{_pb0['bound']:.3f}** |",
        f"| PAC-Bayes, data-dependent prior (σ = {_pbd['sigma']:.3f}, KL = {_pbd['kl']:.0f}) | Gibbs on the bound half "
        f"{_pbd['emp_risk']:.4f} | Gibbs test {ex.gibbs_error(_w, _pbd['sigma'], _Xt, _yt):.4f} | **{_pbd['bound']:.3f}** |",
        f"| test-set bound (10,000 images) | – | test error {_te:.4f} | **{_tsb:.4f}** |",
    ]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for _r, _lab, _c in [(_pb0, "prior at 0", "C3"), (_pbd, "data-dependent prior", "C0")]:
        _a1.semilogx(_sig, _r["bounds"], "-", color=_c, label=f"bound, {_lab}")
        _a1.semilogx(_sig, _r["emp_risks"], ":", color=_c, label=f"Gibbs empirical risk, {_lab}")
    _a1.axhline(_te, color="k", lw=0.8, label="test error of w")
    _a1.set(xlabel="σ (posterior = prior width)", ylabel="error / bound", ylim=(0, 0.6))
    _a1.legend(fontsize=7)
    _names = ["test error", "test-set\nbound", "PAC-Bayes\ndata prior", "PAC-Bayes\nprior 0", "VC"]
    _vals = [_te, _tsb, _pbd["bound"], _pb0["bound"], _vcb]
    _a2.bar(_names, _vals, color=["0.3", "C0", "C0", "C3", "C1"])
    _a2.axhline(0.5, color="0.5", ls=":", lw=1)
    for _i, _v in enumerate(_vals):
        _a2.text(_i, _v + 0.01, f"{_v:.3f}", ha="center", fontsize=8)
    _a2.set(ylabel="error rate / bound", ylim=(0, 1.0), title="odd vs even, n = 60,000")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| | empirical | test | bound |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the PAC-Bayes bound informative when the VC bound isn't, for the
    *same* weight vector? (what does the KL term measure that the VC dimension doesn't?) in
    the left panel, which term drives the bound up at small $\sigma$, and which at large
    $\sigma$? the bound is about the stochastic Gibbs classifier, whose test error is higher
    than $w$'s: how would you turn it into a statement about $w$ itself? why does a prior
    trained on half the data help so much, even though the bound then uses only half the
    data?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: why invert the kl?

    for a fixed complexity budget $c = (\text{KL} + \ln(2\sqrt n/\delta))/n$, three bounds as a
    function of the empirical risk $\hat L$: your `pac_bayes_bound` ($\text{kl}^{-1}(\hat L, c)$),
    your `mcallester_bound` ($\hat L + \sqrt{c/2}$, which follows from Pinsker's inequality),
    and the "Bernstein-like" $\hat L + \sqrt{2\hat Lc} + 2c$ for comparison. the slider sets $c$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with c = 0.01 and an empirical risk of 0, what do the kl bound and McAllester's bound give? which is tighter at "
        "an empirical risk of 0.5?"
    )
    c2 = mo.ui.slider(0.001, 0.1, value=0.01, step=0.001, label="c = (KL + ln(2√n/δ)) / n")
    mo.vstack([p2, c2])
    return c2, p2


@app.cell
def _(c2, p2):
    ui.gate(p2)
    _c = float(c2.value)
    _q = np.linspace(0, 0.6, 121)
    _n = 10000
    _kl_part = _c * _n - math.log(2 * math.sqrt(_n) / 0.05)  # the KL that gives this c at n = 10,000, delta = 0.05
    _klb = np.array([ex.pac_bayes_bound(q, max(_kl_part, 0.0), _n, 0.05) for q in _q])
    _cc = (max(_kl_part, 0.0) + math.log(2 * math.sqrt(_n) / 0.05)) / _n
    _mca = _q + math.sqrt(_cc / 2)
    _ber = _q + np.sqrt(2 * _q * _cc) + 2 * _cc
    _fig, _ax = plt.subplots(figsize=(7, 4.3))
    _ax.plot(_q, _klb, "C0", label="kl⁻¹(L̂, c)  (your pac_bayes_bound)")
    _ax.plot(_q, _mca, "C3--", label="L̂ + √(c/2)  (McAllester / Pinsker)")
    _ax.plot(_q, _ber, "C2:", label="L̂ + √(2L̂c) + 2c")
    _ax.plot(_q, _q, color="0.6", lw=0.8, label="L̂ (no slack)")
    _ax.set(xlabel="empirical (Gibbs) risk L̂", ylabel="bound on the true risk", title=f"c = {_cc:.4f}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _pick = [0.0, 0.01, 0.05, 0.2, 0.5]
    _rows = "\n".join(f"| {q} | {ex.pac_bayes_bound(q, max(_kl_part, 0.0), _n, 0.05):.4f} | {q + math.sqrt(_cc / 2):.4f} | "
                      f"{q + math.sqrt(2 * q * _cc) + 2 * _cc:.4f} |" for q in _pick)
    mo.vstack([_fig, mo.md("| L̂ | kl inverse | McAllester | Bernstein-like |\n|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at $\hat L = 0$ the kl bound is about $c$ while McAllester's is
    $\sqrt{c/2}$: why does the slack scale *linearly* in $c$ when the empirical risk is zero
    (think of a coin with tiny bias, and its variance)? near $\hat L = 0.5$ the two nearly
    agree: why? which regime do the bounds of experiment 1 live in, and how much did the kl
    inversion buy there?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: stability, measured

    California housing regression (standardized features and target). we measure your
    `loo_stability`, the largest change in prediction on 500 held-out districts when one of 20
    random training districts is removed, for training sets of $n = 100$ to $10{,}000$:
    ridge regression with penalty $\alpha = \lambda n$ for $\lambda \in \{10^{-3}, 10^{-1}\}$
    (so the regularization per example is fixed), and 1-nearest-neighbour regression. for 1-NN
    we also remove, instead of random points, the training districts that are the nearest
    neighbours of 20 evaluation districts (`indices=`): uniform stability is a supremum over
    *every* removal, and random removals may miss the ones that matter.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how does ridge's leave-one-out stability change when n grows tenfold? and 1-NN's?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X, _y, _ = datasets.california_housing()
    _X = (_X - _X.mean(0)) / _X.std(0)
    _y = (_y - _y.mean()) / _y.std()
    _perm = np.random.default_rng(0).permutation(len(_X))
    _ev = _perm[:500]
    _pool = _perm[500:]
    _ns = [n for n in [100, 300, 1000, 3000, 10000] if n <= len(_pool)]

    def _ridge(lam):
        def fp(A, b, E):
            return Ridge(alpha=lam * len(b)).fit(A, b).predict(E)
        return fp

    def _nn(A, b, E):
        return KNeighborsRegressor(1).fit(A, b).predict(E)

    _models = {"ridge, λ = 0.001": _ridge(1e-3), "ridge, λ = 0.1": _ridge(1e-1), "1-NN (random removals)": _nn,
               "1-NN (removing neighbours)": _nn}
    _res = {m: [] for m in _models}
    for _m, _fp in _models.items():
        for _k, _n in enumerate(_ns):
            _Xn, _yn = _X[_pool[:_n]], _y[_pool[:_n]]
            if _m == "1-NN (removing neighbours)":
                # the training points that ARE the nearest neighbours of 20 evaluation districts
                _d2 = ((_X[_ev[:20], None, :] - _Xn[None, :, :]) ** 2).sum(-1)
                _idx = np.unique(np.argmin(_d2, axis=1))
                _res[_m].append(ex.loo_stability(_fp, _Xn, _yn, _X[_ev], indices=_idx))
            else:
                _res[_m].append(ex.loo_stability(_fp, _Xn, _yn, _X[_ev], n_remove=20, rng=_k))
    _fig, _ax = plt.subplots(figsize=(7, 4.3))
    for (_m, _v), _c in zip(_res.items(), ["C0", "C2", "C3", "C1"]):
        _ax.loglog(_ns, np.maximum(_v, 1e-4), "o-", color=_c, label=_m)
    _nn_ = np.array(_ns, dtype=float)
    _ax.loglog(_nn_, _res["ridge, λ = 0.1"][0] * _nn_[0] / _nn_, "k:", lw=1, label="slope −1 (∝ 1/n)")
    _ax.set(xlabel="n (training districts)", ylabel="max |Δ prediction| when one point is removed")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _slopes = {m: np.polyfit(np.log(_ns), np.log(np.maximum(v, 1e-12)), 1)[0] for m, v in _res.items() if min(v) > 0}
    _rows = "\n".join(f"| {n:,} | " + " | ".join(f"{_res[m][i]:.2e}" for m in _models) + " |" for i, n in enumerate(_ns))
    mo.vstack([_fig, mo.md("| n | " + " | ".join(_models) + " |\n|---|---|---|---|---|\n" + _rows
                           + "\n\nlog-log slopes (curves with no zero): " + ", ".join(f"{m}: **{s:.2f}**" for m, s in _slopes.items())
                           + ". a measured change of exactly 0 is drawn at 10⁻⁴.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** ridge's stability shrinks like $1/n$, with a constant set by $\lambda$:
    derive the $1/(\lambda n)$ scaling from the strong convexity of the regularized
    objective. the stability bound of Bousquet & Elisseeff needs $\beta = o(1/\sqrt n)$ to say
    anything: which models qualify? 1-NN doesn't get more stable with more data at all: why
    (what happens to the prediction at a query point whose nearest neighbour is removed)?
    does that mean 1-NN doesn't generalize?
    """)
    return


if __name__ == "__main__":
    app.run()
