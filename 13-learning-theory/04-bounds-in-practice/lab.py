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
    from mlfs.preprocessing import StandardScaler  # yours, from ch 07 · topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bounds in practice · lab

    every experiment here runs **your** bounds and report from `exercise.py`, on real data
    (cached under `~/.cache/mlfs`; offline, the synthetic stand-ins). for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. all bounds use $\delta = 0.05$; each holds on its own
    with probability 95%.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a learning curve with its bounds

    California housing, binarized: is a district's median house value above the median of
    the training districts? logistic regression on the 8 standardized features (VC dimension
    9, 9 float32 parameters), trained on $n$ districts for six values of $n$ up to 16,000 and
    tested on 4,000 others, through your `generalization_report`. the last row is a 15-nearest-
    neighbour classifier, whose VC dimension is infinite.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with 16,000 training districts and a model with VC dimension 9: how far above the measured test error "
        "will the VC bound be? the 32-bit counting bound? the test-set bound?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _X, _y, _ = datasets.california_housing()
    _perm = np.random.default_rng(0).permutation(len(_X))
    _n_te = min(4000, len(_X) // 4)  # 4,000 held-out districts (fewer for the offline stand-in)
    _te = _perm[:_n_te]
    _pool = _perm[_n_te:]
    _sizes = [s for s in [100, 300, 1000, 3000, 10000, 16000] if s <= len(_pool)]
    _rows, _reps = [], []
    for _n in _sizes:
        _tr = _pool[:_n]
        _sc = StandardScaler().fit(_X[_tr])
        _thr = np.median(_y[_tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _r = ex.generalization_report(LogisticRegression(alpha=1e-4), _sc.transform(_X[_tr]), (_y[_tr] > _thr).astype(int),
                                          _sc.transform(_X[_te]), (_y[_te] > _thr).astype(int), vc_dim=9, n_params=9)
        _reps.append(_r)
        _rows.append(f"| logistic, n = {_n:,} | {_r['train_error']:.3f} | {_r['test_error']:.3f} | {_r['gap']:+.3f} | "
                     f"{_r['vc_bound']:.3f} | {_r['finite_bound']:.3f} | {_r['test_bound']:.3f} |")
    _n = _sizes[-1]
    _tr = _pool[:_n]
    _sc = StandardScaler().fit(_X[_tr])
    _thr = np.median(_y[_tr])
    _knn = KNeighborsClassifier(15).fit(_sc.transform(_X[_tr]), (_y[_tr] > _thr).astype(int))
    _yte = (_y[_te] > _thr).astype(int)
    _k_te = int(np.sum(_knn.predict(_sc.transform(_X[_te])) != _yte))
    _k_tr = float(np.mean(_knn.predict(_sc.transform(_X[_tr[:_n_te]])) != (_y[_tr[:_n_te]] > _thr)))
    _rows.append(f"| 15-NN, n = {_n:,} | {_k_tr:.3f}* | {_k_te / len(_te):.3f} | {_k_te / len(_te) - _k_tr:+.3f} | ∞ (VC = ∞) | – | "
                 f"{ex.test_set_bound(_k_te, len(_te), 0.05):.3f} |")
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    _ax.semilogx(_sizes, [r["test_error"] for r in _reps], "ko-", label="test error (measured)")
    _ax.semilogx(_sizes, [r["train_error"] for r in _reps], "k^:", label="train error")
    _ax.semilogx(_sizes, [min(r["vc_bound"], 3) for r in _reps], "C3s--", label="VC bound")
    _ax.semilogx(_sizes, [min(r["finite_bound"], 3) for r in _reps], "C1d--", label="32-bit counting bound")
    _ax.semilogx(_sizes, [r["test_bound"] for r in _reps], "C0o-", label="test-set bound (4,000 held out)")
    _ax.axhline(1.0, color="0.5", lw=0.8)
    _ax.set(xlabel="training districts n", ylabel="error rate / bound", ylim=(0, 2.0), title="logistic regression, δ = 0.05")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _need = {e: ex.vc_sample_size(9, e, 0.05) for e in (0.2, 0.1, 0.05)}
    mo.vstack([_fig, mo.md(
        "| model | train error | test error | gap | VC bound | 32-bit bound | test-set bound |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(_rows)
        + "\n\n\\* 15-NN's training error is resubstitution error on 4,000 of its training districts."
        + "\n\nyour `vc_sample_size(9, ε, 0.05)`: " + ", ".join(f"ε = {e}: **{v:,}**" for e, v in _need.items())
        + " districts for the VC complexity term alone.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the measured gap shrinks to almost nothing by $n = 1000$, while the VC bound
    is still vacuous: which constants and terms in $\sqrt{(8/n)(d\ln(2en/d) + \ln(4/\delta))}$
    are responsible, and where does the slack come from (the union bound over a ghost sample,
    Sauer, the worst-case distribution)? why is the crude 32-bit counting bound *tighter* than
    the VC bound here? the test-set bound is tight at every $n$: what does it assume that the
    others don't, and what does it cost you?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: count the bits

    MNIST, even vs odd digits: logistic regression on all 784 pixels (785 parameters), trained
    on the 60,000 training images (offline: the synthetic stand-in). the counting bound treats
    a model stored in $b$ bits per parameter as one of $2^{785b}$ hypotheses, but only if it
    *really* is: so we round the learned weights to a fixed grid of $2^b$ levels on
    $[-2, 2]$ (chosen before seeing the data) and measure the **quantized** model's errors. the
    dropdown sets the bits for the figure's highlighted point.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the VC bound for 785 parameters on 60,000 images is about 0.8 above the training error. how many bits per weight "
        "do you need before the counting bound is tighter, and how few bits does the classifier tolerate?"
    )
    bits2 = mo.ui.dropdown(options=["2", "3", "4", "6", "8", "16", "32"], value="4", label="bits per weight")
    mo.vstack([p2, bits2])
    return bits2, p2


@app.cell
def _(bits2, p2):
    ui.gate(p2)
    _d = datasets.mnist()
    _Xtr = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    _Xte = _d["X_test"].reshape(len(_d["X_test"]), -1) / 255.0
    _ytr, _yte = _d["y_train"] % 2, _d["y_test"] % 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _m = LogisticRegression(alpha=1e-4).fit(_Xtr, _ytr)
    _w = np.append(_m.coef_, _m.intercept_)
    _n, _P = len(_ytr), _w.size

    def _errors(w):
        tr = float(np.mean((_Xtr @ w[:-1] + w[-1] >= 0) != _ytr))
        k = int(np.sum((_Xte @ w[:-1] + w[-1] >= 0) != _yte))
        return tr, k

    _rows, _res = [], {}
    for _b in [2, 3, 4, 6, 8, 16, 32]:
        if _b >= 16:
            _wq = _w.astype(np.float16 if _b == 16 else np.float32).astype(float)  # round to the counted precision
        else:
            _levels = np.linspace(-2, 2, 2**_b)
            _wq = _levels[np.argmin(np.abs(np.clip(_w, -2, 2)[:, None] - _levels[None, :]), axis=1)]
        _tr, _k = _errors(_wq)
        _fb = ex.finite_class_bound(_n, 2 ** (_b * _P), 0.05, _tr)
        _res[_b] = (_tr, _k / len(_yte), _fb, ex.test_set_bound(_k, len(_yte), 0.05))
        _rows.append(f"| {_b} | {_tr:.4f} | {_k / len(_yte):.4f} | {_fb:.3f} | {_res[_b][3]:.4f} |")
    _tr32 = _res[32][0]
    _vcb = ex.vc_bound(_n, _P, 0.05, _tr32)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.3))
    _bs = sorted(_res)
    _ax.semilogx(_bs, [_res[b][1] for b in _bs], "ko-", label="test error of the quantized model")
    _ax.semilogx(_bs, [_res[b][2] for b in _bs], "C1d--", label="counting bound, 785 b bits")
    _ax.axhline(_vcb, color="C3", ls="--", label=f"VC bound (d = 785): {_vcb:.3f}")
    _ax.semilogx(_bs, [_res[b][3] for b in _bs], "C0o-", label="test-set bound (10,000 held out)")
    _sel = int(bits2.value)
    _ax.plot([_sel], [_res[_sel][2]], "o", ms=14, mfc="none", mec="C1", mew=2)
    _ax.set_xticks(_bs, [str(b) for b in _bs])
    _ax.minorticks_off()
    _ax.set(xlabel="bits per weight b", ylabel="error rate / bound", ylim=(0, 1.0), title=f"MNIST even vs odd, n = {_n:,}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| bits | train error | test error | counting bound | test-set bound |\n|---|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\nVC bound for the unquantized model: **{_vcb:.3f}**. at {_sel} bits the counting bound is "
        f"**{_res[_sel][2]:.3f}** against a measured test error of {_res[_sel][1]:.4f}. if you pick the bit width "
        f"*after* looking at these 7 bounds, pay for the choice with a union bound (δ/7): "
        f"{ex.finite_class_bound(_n, 2 ** (_sel * _P), 0.05 / 7, _res[_sel][0]):.3f} at {_sel} bits.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the counting bound get *tighter* with fewer bits, and why does it
    beat the VC bound below 32 bits, even though the VC bound is for all real-valued weights?
    how few bits can the classifier lose before its error rises, and what does that say about
    how much information the learned weights actually contain? why must the quantization grid
    be fixed before looking at the data for the bound to be valid?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: coin flips that look like classifiers

    the test-set bound is valid for **one** classifier fixed before the test set is seen. here
    $k$ "classifiers" guess every label by a fair coin flip, so each has true error exactly
    0.5. we evaluate all $k$ on 500 held-out examples, pick the one with the fewest errors, and
    compute your `test_set_bound` for it twice: naively with $\delta = 0.05$, and with the
    union-bound correction $\delta/k$. repeated 300 times per $k$: how often does the true error
    0.5 lie above the "95%" bound?
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with k = 1,000 coin-flip classifiers, what test error will the best one show on 500 examples, and how often "
        "will its naive 95% bound fail to cover the true error 0.5?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _n, _reps = 500, 300
    _rows, _naive_fail, _union_fail, _ks = [], [], [], [1, 10, 100, 1000, 10000]
    for _k in _ks:
        _best = _rng.binomial(_n, 0.5, size=(_reps, _k)).min(axis=1)
        _nb = np.array([ex.test_set_bound(int(e), _n, 0.05) for e in _best])
        _ub = np.array([ex.test_set_bound(int(e), _n, 0.05 / _k) for e in _best])
        _naive_fail.append(float(np.mean(_nb < 0.5)))
        _union_fail.append(float(np.mean(_ub < 0.5)))
        _rows.append(f"| {_k:,} | {np.mean(_best) / _n:.3f} | {np.mean(_nb):.3f} | {_naive_fail[-1]:.2f} | {np.mean(_ub):.3f} | "
                     f"{_union_fail[-1]:.2f} |")
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.semilogx(_ks, _naive_fail, "C3o-", label="naive bound (δ = 0.05) misses 0.5")
    _ax.semilogx(_ks, _union_fail, "C0s-", label="union bound (δ/k) misses 0.5")
    _ax.axhline(0.05, color="0.5", ls=":", label="promised failure rate 0.05")
    _ax.set(xlabel="k (classifiers compared on the same test set)", ylabel="fraction of 300 repetitions", ylim=(-0.03, 1.03))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| k | best test error (mean) | naive bound (mean) | naive misses | δ/k bound (mean) | δ/k misses |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the best of 1,000 coins look like a real classifier, and why does
    the naive bound fail almost every time? the $\delta/k$ correction restores the guarantee:
    how much wider does it make the bound at $k = 10{,}000$, and why so little? what does this
    mean for a test set you've used to compare dozens of models, or for a public leaderboard?
    """)
    return


if __name__ == "__main__":
    app.run()
