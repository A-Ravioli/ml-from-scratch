import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from functools import partial

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.datasets import california_housing, mnist
    from mlfs.kernels import SVC, rbf_kernel  # yours, from ch 09
    from mlfs.model_selection import StratifiedKFold  # yours, from ch 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # quantum kernels · lab

    every experiment here runs **your** kernels from `exercise.py` (on your topic 30 simulator)
    through **your** ch 09 SVM. for each one: write down a prediction, lock it in, run it, then
    explain what happened. compare your explanations with `solutions/lab_notes.md` afterwards.
    experiments 2 and 3 are the block's project: *when (if ever) do quantum kernels beat
    classical ones?* the whole lab takes about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    ex.qsim_load.callout()  # shown only if your topic 30 simulator isn't finished
    return


@app.cell
def _():
    def precomputed(G):
        """a kernel for SVC that looks entries up in a precomputed Gram matrix G; the 'features' are row indices."""

        def k(A, B=None):
            B = A if B is None else B
            return G[np.ix_(A[:, 0].astype(int), B[:, 0].astype(int))]

        return k

    def svc_accuracy(G, y, train, test, C=1.0):
        """fit your SVC on the train rows of G, return test accuracy."""
        idx = np.arange(len(y), dtype=float)[:, None]
        model = SVC(C=C, kernel=precomputed(G), rng=0).fit(idx[train], y[train])
        return float(np.mean(model.predict(idx[test]) == y[test]))

    def cv_select(grams, y, train, k=3):
        """index of the Gram matrix with the best k-fold CV accuracy on the training rows."""
        folds = list(StratifiedKFold(k, shuffle=True, rng=0).split(train[:, None], y[train]))
        scores = [np.mean([svc_accuracy(G, y, train[a], train[b]) for a, b in folds]) for G in grams]
        return int(np.argmax(scores))

    def splits(m, n_splits, n_train):
        return [np.split(np.random.default_rng(s).permutation(m), [n_train]) for s in range(n_splits)]

    return cv_select, precomputed, splits, svc_accuracy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: kernel concentration

    40 random points $x \sim U(0, 2\pi)^n$, the ZZ feature map with 2 reps, and the bandwidth
    `scale` in front of $x$. left: the variance of the off-diagonal kernel entries as the number
    of qubits grows. right: how well a Gram matrix estimated from a finite number of shots per
    entry (your `estimate_kernel`) matches the exact one, measured by the correlation between
    their off-diagonal entries (the diagonal is 1 in both, so it carries no information).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at scale 1, what happens to the off-diagonal kernel values as you go from 2 to 10 qubits? with 1,000 "
        "shots per entry, how much of the 12-qubit kernel survives the shot noise?"
    )
    p1
    return (p1,)


@app.cell
def _():
    shots = mo.ui.dropdown({"100": 100, "1,000": 1000, "10,000": 10000}, value="1,000", label="shots per kernel entry")
    shots
    return (shots,)


@app.cell
def _(p1, shots):
    ui.gate(p1)
    _ns = [2, 4, 6, 8, 10, 12]
    _scales = [1.0, 0.1, 0.03, 0.01]
    _rng = np.random.default_rng(0)
    _var = {s: [] for s in _scales}
    _mean = {s: [] for s in _scales}
    _align = {s: [] for s in _scales}
    for _n in _ns:
        _X = _rng.uniform(0, 2 * np.pi, size=(40, _n))
        for _s in _scales:
            _fm = partial(ex.zz_feature_map, scale=_s)
            _K = ex.fidelity_kernel(_X, feature_map=_fm)
            _off = _K[np.triu_indices(40, 1)]
            _var[_s].append(_off.var())
            _mean[_s].append(_off.mean())
            _E = ex.estimate_kernel(_X, shots.value, feature_map=_fm, rng=1)
            _align[_s].append(np.corrcoef(_E[np.triu_indices(40, 1)], _off)[0, 1])
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for _s in _scales:
        _a1.semilogy(_ns, _var[_s], "o-", label=f"scale {_s}")
        _a2.plot(_ns, _align[_s], "o-", label=f"scale {_s}")
    _a1.set(xlabel="qubits n", ylabel="variance of off-diagonal K")
    _a2.set(xlabel="qubits n", ylabel=f"corr(estimate, exact), {shots.value:,} shots", ylim=(0, 1.05))
    _a1.legend(fontsize=8)
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {s} | " + " | ".join(f"{m:.3g} / {v:.1e}" for m, v in zip(_mean[s], _var[s])) + " |"
                      for s in _scales)
    _corr = "\n".join(f"| {s} | " + " | ".join(f"{c:.2f}" for c in _align[s]) + " |" for s in _scales)
    _head = "| scale | " + " | ".join(f"n = {n}" for n in _ns) + " |\n|---|" + "---|" * len(_ns) + "\n"
    mo.vstack([_fig, mo.md("off-diagonal kernel: mean / variance\n\n" + _head + _rows
                           + f"\n\ncorrelation of the {shots.value:,}-shot estimate with the exact kernel\n\n"
                           + _head + _corr)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at scale 1, the off-diagonal entries shrink towards 0 and their spread
    shrinks faster still. if $|\phi(x)\rangle$ behaved like a random state in $2^n$ dimensions,
    what would $\mathbb E|\langle\phi(x)|\phi(y)\rangle|^2$ be? why does a kernel with all
    off-diagonal entries near 0 make an SVM memorize instead of generalize, even with exact
    values? with shots, why does the correlation collapse once the variance drops below
    about $p/\text{shots}$ (the binomial noise of an entry near $p$)? how does the bandwidth fix it, and why does the bandwidth you need
    shrink as $n$ grows?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the project data

    **MNIST 3 vs 5**: the first 3,000 threes and fives of the training set, projected onto their
    top 6 principal components (6 qubits) and standardized; 300 of them are used.
    **California housing**: the 8 features standardized (and clipped at ±3), label = price above
    the median (8 qubits); 300 districts. kernels:

    - **ZZ** (quantum): your `zz_feature_map`, 2 reps, scale $s \in \{0.01, 0.02, 0.05, 0.1, 0.2, 0.4\}$;
    - **product** (quantum circuit, classical kernel): your `product_feature_map`, scale $\in \{0.25, 0.5, 1, 2\}$;
    - **RBF**: `rbf_kernel`, $\gamma \in \{0.01, 0.03, 0.1, 0.3, 1\}$;
    - **linear**: $x^\top y$.

    every SVM is your ch 09 `SVC` with $C = 1$; "CV" means the hyperparameter is chosen by 3-fold
    cross-validation on the training half only. 5 random splits of 150 train / 150 test.
    """)
    return


@app.cell
def _():
    def _mnist_pca(n_comp=6, m=300):
        d = mnist()
        X = d["X_train"].reshape(len(d["X_train"]), -1).astype(float) / 255
        sel = np.flatnonzero((d["y_train"] == 3) | (d["y_train"] == 5))[:3000]
        X, y = X[sel], np.where(d["y_train"][sel] == 3, 1, -1)
        Xc = X - X.mean(0)
        Vt = np.linalg.svd(Xc, full_matrices=False)[2]
        Z = Xc @ Vt[:n_comp].T
        Z = (Z - Z.mean(0)) / Z.std(0)
        pick = np.random.default_rng(0).choice(len(Z), m, replace=False)
        return Z[pick], y[pick]

    def _housing(m=300):
        X, price, _ = california_housing()
        pick = np.random.default_rng(0).choice(len(X), m, replace=False)
        X = (X - X.mean(0)) / X.std(0)
        X = np.clip(X[pick], -3, 3)
        return X, np.where(price[pick] > np.median(price), 1, -1)

    zz_scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    prod_scales = [0.25, 0.5, 1.0, 2.0]
    gammas = [0.01, 0.03, 0.1, 0.3, 1.0]

    def grams(X):
        return {
            "ZZ": [ex.fidelity_kernel(X, feature_map=partial(ex.zz_feature_map, scale=s)) for s in zz_scales],
            "product": [ex.fidelity_kernel(X, feature_map=partial(ex.product_feature_map, scale=s)) for s in prod_scales],
            "RBF": [rbf_kernel(X, gamma=g) for g in gammas],
            "linear": [X @ X.T],
        }

    data = {"MNIST 3 vs 5": _mnist_pca(), "California housing": _housing()}
    gram = {name: grams(X) for name, (X, _) in data.items()}
    return data, gammas, gram, prod_scales, zz_scales


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2 (project, part 1): a dataset built for the quantum kernel

    Huang et al. (2021) showed how to construct labels on which a given quantum kernel beats a
    given classical one by as much as possible: your `engineered_labels`. we take the MNIST
    points (real inputs), the ZZ kernel at scale 0.2, and the RBF kernel at $\gamma = 0.1$, and
    relabel the points. then every kernel family tries to learn the new labels.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "on labels engineered for the ZZ kernel at scale 0.2, what test accuracy will that kernel reach? the RBF "
        "with its gamma chosen by CV? and the *same* ZZ circuit at scale 0.1 or 0.4?"
    )
    p2
    return (p2,)


@app.cell
def _(cv_select, data, gammas, gram, p2, prod_scales, splits, svc_accuracy, zz_scales):
    ui.gate(p2)
    _G = gram["MNIST 3 vs 5"]
    _Kq, _Kc = _G["ZZ"][zz_scales.index(0.2)], _G["RBF"][gammas.index(0.1)]
    _g = ex.geometric_difference(_Kc, _Kq, reg=1e-3)
    _, _lab = ex.engineered_labels(_Kc, _Kq, reg=1e-3)
    _y_real = data["MNIST 3 vs 5"][1]
    _perm = np.random.default_rng(5).permutation(_lab)  # same labels, structure destroyed
    _res = {}
    _zz_cv = []
    for _tr, _te in splits(len(_lab), 5, 150):
        for _s, _K in zip(zz_scales, _G["ZZ"]):
            _res.setdefault(f"ZZ, scale {_s}", []).append(svc_accuracy(_K, _lab, _tr, _te))
        for _fam in ("ZZ", "RBF", "product"):
            _i = cv_select(_G[_fam], _lab, _tr)
            if _fam == "ZZ":
                _zz_cv.append(zz_scales[_i])
            _res.setdefault(f"{_fam} (CV)", []).append(svc_accuracy(_G[_fam][_i], _lab, _tr, _te))
        _res.setdefault("linear", []).append(svc_accuracy(_G["linear"][0], _lab, _tr, _te))
        _i = cv_select(_G["RBF"], _perm, _tr)
        _res.setdefault("control: RBF (CV) on permuted labels", []).append(svc_accuracy(_G["RBF"][_i], _perm, _tr, _te))
        _major = 1 if np.mean(_lab[_tr] == 1) >= 0.5 else -1
        _res.setdefault("control: majority class of the training half", []).append(float(np.mean(_lab[_te] == _major)))
    # leave-one-out vote weighted by the classical kernel: how the labels look to K_C's notion of "nearby"
    _W = _Kc - np.diag(np.diag(_Kc))
    _loo = float(np.mean(np.sign(_W @ _lab) == _lab))
    _names = list(_res)
    _m = np.array([np.mean(_res[k]) for k in _names])
    _se = np.array([np.std(_res[k], ddof=1) / np.sqrt(len(_res[k])) for k in _names])
    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _colors = ["C3" if k in ("ZZ, scale 0.2", "ZZ (CV)") else ("C1" if k.startswith("ZZ") else
               ("0.6" if k.startswith("control") else "C0")) for k in _names]
    _ax.bar(range(len(_names)), _m, yerr=2 * _se, color=_colors, capsize=3)
    _ax.axhline(0.5, color="k", lw=1, ls=":")
    _ax.set(xticks=range(len(_names)), ylabel="test accuracy (± 2 s.e., 5 splits)", ylim=(0, 1))
    _ax.set_xticklabels([k.replace("control: ", "ctrl: ").replace(" of the training half", "") for k in _names],
                        rotation=30, ha="right", fontsize=7)
    _fig.tight_layout()
    _agree = np.mean(_lab == _y_real)
    mo.vstack([_fig, mo.md(
        f"geometric difference $g$(RBF γ=0.1 ‖ ZZ scale 0.2) = **{_g:.1f}** · engineered labels agree with the "
        f"real 3-vs-5 labels on {_agree:.0%} of the points · CV's ZZ scale per split: {_zz_cv} · RBF (γ=0.1) "
        f"kernel-weighted leave-one-out vote on all 300 points: {_loo:.3f}\n\n"
        "(our $g$ regularizes as $(K_C + \\lambda I)^{-1}$, a simplification of Huang et al.'s regularized geometric "
        "difference, so its value isn't comparable with the paper's numbers.)\n\n"
        "| kernel | test accuracy (mean ± s.e.) |\n|---|---|\n"
        + "\n".join(f"| {k} | {m:.3f} ± {s:.3f} |" for k, m, s in zip(_names, _m, _se)))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the quantum kernel wins by a landslide. but look at the neighbouring
    bars: what does it mean that the same circuit at scale 0.1 or 0.4 does no better than
    chance? what do these labels look like to every other kernel, and why (think about what
    the top eigenvector of $\sqrt{K_Q}K_C^{-1}\sqrt{K_Q}$ is made to be)? the engineered labels
    barely agree with the digit identity: would you expect a real-world task to have labels
    like this? the RBF scores *below* the permuted-label control, and the kernel-weighted
    leave-one-out vote is far below 0.5: what does that say about where these labels sit in
    $K_C$'s eigenbasis?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3 (project, part 2): real labels

    the same kernel families, each with its hyperparameter chosen by CV, on the real labels:
    digit 3 vs 5, and expensive vs cheap districts. we also report each family's best
    alignment with the labels, $A(K, yy^\top)$, over its hyperparameters (on the full 300 points).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "on real MNIST 3 vs 5 and California housing, will the CV-tuned ZZ quantum kernel beat, match, or lose to "
        "the CV-tuned RBF kernel? by how much?"
    )
    p3
    return (p3,)


@app.cell
def _(cv_select, data, gammas, gram, p3, prod_scales, splits, svc_accuracy, zz_scales):
    ui.gate(p3)
    _grid = {"ZZ": zz_scales, "product": prod_scales, "RBF": gammas, "linear": ["-"]}
    _out = []
    _bars = {}
    _per_split = {}
    for _name, (_X, _y) in data.items():
        _G = gram[_name]
        for _fam in ("ZZ", "product", "RBF", "linear"):
            _accs, _picked = [], []
            for _tr, _te in splits(len(_y), 5, 150):
                _i = cv_select(_G[_fam], _y, _tr) if len(_G[_fam]) > 1 else 0
                _picked.append(_grid[_fam][_i])
                _accs.append(svc_accuracy(_G[_fam][_i], _y, _tr, _te))
            _align = max(ex.kernel_alignment(K, np.outer(_y, _y)) for K in _G[_fam])
            _m, _se = np.mean(_accs), np.std(_accs, ddof=1) / np.sqrt(len(_accs))
            _bars[(_name, _fam)] = (_m, _se)
            _per_split[(_name, _fam)] = np.array(_accs)
            _out.append(f"| {_name} | {_fam} | {', '.join(str(v) for v in _picked)} | {_m:.3f} ± {_se:.3f} | {_align:.3f} |")
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _fams = ["ZZ", "product", "RBF", "linear"]
    for _j, _name in enumerate(data):
        _mm = [_bars[(_name, f)][0] for f in _fams]
        _ss = [_bars[(_name, f)][1] for f in _fams]
        _ax.bar(np.arange(4) + 0.4 * _j, _mm, 0.4, yerr=2 * np.array(_ss), capsize=3, label=_name)
    _ax.set(xticks=np.arange(4) + 0.2, ylabel="test accuracy (± 2 s.e., 5 splits)", ylim=(0.5, 1))
    _ax.set_xticklabels(["ZZ (quantum)", "product", "RBF", "linear"])
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _paired = []
    for _name in data:
        for _fam in ("ZZ", "product", "linear"):
            _d = _per_split[(_name, _fam)] - _per_split[(_name, "RBF")]
            _paired.append(f"| {_name} | {_fam} − RBF | {_d.mean():+.3f} ± {_d.std(ddof=1) / np.sqrt(len(_d)):.3f} | "
                           f"{', '.join(f'{v:+.3f}' for v in _d)} |")
    mo.vstack([_fig, mo.md("| data | kernel | CV's choice per split | test accuracy (mean ± s.e.) | best alignment with labels |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_out)
                           + "\n\npaired differences (same splits)\n\n| data | difference | mean ± s.e. | per split |\n"
                           "|---|---|---|---|\n" + "\n".join(_paired))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which kernel wins on real labels, and is the gap bigger than the
    uncertainty? which ZZ scale does CV pick, and what does that say about how "quantum" the
    winning ZZ kernel is (compare its off-diagonal values with experiment 1, and with the
    product kernel)? putting experiments 2 and 3 together, write the project's answer in
    three sentences: when can a quantum kernel beat a classical one, how would you know in
    advance, and why does an advantage on one dataset not transfer to another?
    """)
    return


if __name__ == "__main__":
    app.run()
