import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.unsupervised import KMeans  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gaussian mixtures & EM · lab

    every experiment here runs **your** `GaussianMixture` (and your `KMeans` from topic 01).
    for each one: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes
    about half a minute.
    """)
    return


@app.cell
def _():
    def ari(a, b):
        """adjusted Rand index from the contingency table (you build it properly in topic 05)."""
        _, ia = np.unique(a, return_inverse=True)
        _, ib = np.unique(b, return_inverse=True)
        C = np.zeros((ia.max() + 1, ib.max() + 1))
        np.add.at(C, (ia, ib), 1)
        comb = lambda m: (m * (m - 1) / 2).sum()  # noqa: E731
        idx, ra, rb, tot = comb(C), comb(C.sum(1)), comb(C.sum(0)), comb(np.array([len(a)]))
        exp = ra * rb / tot
        return float((idx - exp) / (0.5 * (ra + rb) - exp))

    def ellipses(ax, g, color="k"):
        """2-sigma ellipses of a fitted 2-D mixture."""
        t = np.linspace(0, 2 * np.pi, 100)
        for j in range(len(g.weights_)):
            c = g.covariances_[j]
            cov = c if g.covariance_type == "full" else np.diag(c) if g.covariance_type == "diag" else c * np.eye(2)
            w, V = np.linalg.eigh(cov)
            pts = (V * 2 * np.sqrt(w)) @ np.vstack([np.cos(t), np.sin(t)])
            ax.plot(g.means_[j, 0] + pts[0], g.means_[j, 1] + pts[1], color=color, lw=1.2)

    return ari, ellipses


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: monotone, but where does it end?

    six elongated Gaussian blobs arranged in a ring (600 points). we fit a 6-component GMM
    in two ways. **random starts**: `init="random"` (each point's starting responsibilities
    are a random distribution over components) with the default `tol=1e-6`, and the same
    three seeds again with `tol=0` and 1,500 iterations, to see what happens if you don't
    stop. **k-means starts**: `init="kmeans"` with 30 different seeds. we plot the mean
    log-likelihood per iteration.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with random starts and the default tol, at what iteration will EM stop, and how close to the best "
        "log-likelihood? will the 30 k-means-started fits all end at the same value?"
    )
    p1
    return (p1,)


@app.cell
def _(ellipses, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    _C = np.c_[4 * np.cos(_ang), 4 * np.sin(_ang)]
    X1 = np.concatenate([_c + _rng.standard_normal((100, 2)) * [1.0, 0.5] for _c in _C])
    _stopped = [ex.GaussianMixture(6, init="random", rng=_s).fit(X1) for _s in range(3)]
    _long = [ex.GaussianMixture(6, init="random", tol=0.0, max_iters=1500, rng=_s).fit(X1) for _s in range(3)]
    _km = [ex.GaussianMixture(6, init="kmeans", max_iters=500, rng=_s).fit(X1) for _s in range(30)]
    _finals = np.array([_g.lower_bound_history_[-1] for _g in _km])
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for _s, _g in enumerate(_long):
        _axes[0].plot(np.arange(1, len(_g.lower_bound_history_) + 1), _g.lower_bound_history_, label=f"random start, seed {_s}")
    for _g in _stopped:
        _axes[0].plot(len(_g.lower_bound_history_), _g.lower_bound_history_[-1], "kx", ms=10)
    for _g in _km:
        _axes[0].plot(np.arange(1, len(_g.lower_bound_history_) + 1), _g.lower_bound_history_, color="0.6", lw=0.6)
    _axes[0].set(xscale="log", xlabel="EM iteration", ylabel="mean log-likelihood",
                 title="random starts (colour; x = stopped by tol), k-means starts (grey)")
    _axes[0].legend(fontsize=7)
    for _ax, _g, _title in [(_axes[1], _stopped[0], "random start, stopped by tol"),
                            (_axes[2], _km[int(np.argmin(_finals))], "worst k-means-started fit")]:
        _ax.scatter(X1[:, 0], X1[:, 1], s=4, c="0.6")
        ellipses(_ax, _g, "C3")
        _ax.set_aspect("equal")
        _ax.set_title(f"{_title}: {_g.lower_bound_history_[-1]:.3f}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "random starts with the default `tol`: stopped after **"
        + ", ".join(str(_g.n_iter_) for _g in _stopped) + "** iterations at mean log-likelihood **"
        + ", ".join(f"{_g.lower_bound_history_[-1]:.3f}" for _g in _stopped) + "**. the same seeds run for 1,500 iterations end at **"
        + ", ".join(f"{_g.lower_bound_history_[-1]:.3f}" for _g in _long) + "**. k-means starts (30 seeds): final values from **"
        f"{_finals.min():.3f}** to **{_finals.max():.3f}**, {np.sum(np.isclose(_finals, _finals.max(), atol=1e-3))} of 30 at the best.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what do the six components look like when the random-start fit stops
    (right after the first M-step, what are the means)? why is the improvement per iteration
    so tiny for so long, and then suddenly large? why did `tol` declare convergence? the
    k-means-started fits are all monotone too: what's different about the worst one?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: stretched, tilted clusters

    three long, thin, parallel clusters (standard deviations 4 along and 0.4 across), tilted
    by 34°. we compare your `KMeans(3)` with your `GaussianMixture(3)` using each covariance
    type, by ARI against the true clusters (the median over 5 seeds).
    """)
    return


@app.cell
def _():
    p2 = ui.predict("rank k-means and the spherical, diag and full GMMs by ARI.")
    p2
    return (p2,)


@app.cell
def _(ari, ellipses, p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _X = np.concatenate([np.c_[_rng.normal(0, 4, 200), _rng.normal(_o, 0.4, 200)] for _o in (0, 2.5, 5)])
    _y = np.repeat([0, 1, 2], 200)
    _R = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    X2 = _X @ _R.T
    _scores = {"k-means": np.median([ari(_y, KMeans(3, rng=_s).fit(X2).labels_) for _s in range(5)])}
    _fits = {}
    for _ct in ("spherical", "diag", "full"):
        _fits[_ct] = [ex.GaussianMixture(3, covariance_type=_ct, rng=_s).fit(X2) for _s in range(5)]
        _scores[_ct] = np.median([ari(_y, _g.predict(X2)) for _g in _fits[_ct]])
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 3.8))
    _axes[0].scatter(X2[:, 0], X2[:, 1], c=KMeans(3, rng=0).fit(X2).labels_, cmap="tab10", s=4)
    _axes[0].set_title(f"k-means: ARI {_scores['k-means']:.2f}")
    for _ax, _ct in zip(_axes[1:], ("spherical", "diag", "full")):
        _g = _fits[_ct][0]
        _ax.scatter(X2[:, 0], X2[:, 1], c=_g.predict(X2), cmap="tab10", s=4)
        ellipses(_ax, _g)
        _ax.set_title(f"GMM {_ct}: ARI {_scores[_ct]:.2f}")
    for _ax in _axes:
        _ax.set_aspect("equal")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("median ARI over 5 seeds: " + ", ".join(f"{_k} **{_v:.3f}**" for _k, _v in _scores.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what shape is each model allowed to give a cluster, and which shapes fit
    these data? the diagonal model can stretch along the axes: why doesn't that help here
    (what would happen if the clusters weren't tilted)? when would you still prefer the
    spherical or diag model despite this?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: choosing k

    four tilted Gaussian blobs: 400 training points, and 2,000 fresh points from the same
    distribution for testing. for $k = 1, \dots, 8$ we fit your full-covariance
    `GaussianMixture` and record the training mean log-likelihood, the held-out mean
    log-likelihood, and the BIC on the training set. then we add **six copies of one point**
    (in the empty middle of the data) to the training set and do it again.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "which k does BIC pick on the clean data? and after adding six identical points to 400?"
    )
    n_dup = mo.ui.slider(0, 20, value=6, label="copies of the duplicated point (after you've answered)")
    mo.vstack([p3, n_dup])
    return n_dup, p3


@app.cell
def _(n_dup, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(1)
    _means = np.array([[0, 0], [5, 0], [0, 5], [5, 5]])
    _A = np.array([[1.0, 0.3], [0.0, 0.6]])

    def _draw(n):
        lab = _rng.integers(0, 4, n)
        return _means[lab] + _rng.standard_normal((n, 2)) @ _A

    _Xtr, _Xte = _draw(400), _draw(2000)
    _Xdup = np.vstack([_Xtr, np.tile([[2.5, 2.5]], (n_dup.value, 1))])
    _dn = f"+ {n_dup.value} duplicates"
    _ks = list(range(1, 9))
    _rows, _res = [], {}
    for _name, _Xfit in [("clean", _Xtr), (_dn, _Xdup)]:
        _gs = [ex.GaussianMixture(_k, rng=0).fit(_Xfit) for _k in _ks]
        _res[_name] = dict(train=[_g.score(_Xfit) for _g in _gs], test=[_g.score(_Xte) for _g in _gs],
                           bic=[_g.bic(_Xfit) for _g in _gs],
                           minvar=[min(np.linalg.eigvalsh(_c).min() for _c in _g.covariances_) for _g in _gs])
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _name, _ls in [("clean", "-"), (_dn, "--")]:
        _axes[0].plot(_ks, _res[_name]["train"], "o" + _ls, color="C0", label=f"train, {_name}")
        _axes[0].plot(_ks, _res[_name]["test"], "s" + _ls, color="C1", label=f"held out, {_name}")
        _axes[1].plot(_ks, _res[_name]["bic"], "o" + _ls, color="C2", label=f"BIC, {_name}")
    _axes[0].set(xlabel="k", ylabel="mean log-likelihood")
    _axes[1].set(xlabel="k", ylabel="BIC (lower is better)")
    for _ax in _axes:
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _table = "| k | " + " | ".join(map(str, _ks)) + " |\n|---|" + "---|" * len(_ks) + "\n"
    for _name in ("clean", _dn):
        _table += f"| BIC, {_name} | " + " | ".join(f"{_v:.0f}" for _v in _res[_name]["bic"]) + " |\n"
        _table += f"| held-out, {_name} | " + " | ".join(f"{_v:.3f}" for _v in _res[_name]["test"]) + " |\n"
        _table += f"| smallest variance, {_name} | " + " | ".join(f"{_v:.1e}" for _v in _res[_name]["minvar"]) + " |\n"
    _pick = {n: _ks[int(np.argmin(_res[n]["bic"]))] for n in _res}
    _hold = {n: _ks[int(np.argmax(_res[n]["test"]))] for n in _res}
    mo.vstack([_fig, mo.md(_table + f"\nBIC picks **k = {_pick['clean']}** on clean data and **k = {_pick[_dn]}** "
                           f"with the duplicates; held-out likelihood picks **{_hold['clean']}** and **{_hold[_dn]}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the training log-likelihood keep (roughly) rising with $k$
    while the held-out one peaks? what happened to one component once the duplicates were
    added (look at the smallest variance), and why does that make BIC's first term so much
    better? which criterion is fooled, which isn't, and what does on paper 4 have to do with it?
    """)
    return


if __name__ == "__main__":
    app.run()
