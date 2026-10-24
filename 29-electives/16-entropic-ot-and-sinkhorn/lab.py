import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from mlfs import notebooks, ui
    from mlfs.datasets import make_moons

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    def _ot15_smoke(m):  # raises NotImplementedError if a topic 15 stage this lab needs isn't done yet
        m.kantorovich_lp(np.array([1.0]), np.array([1.0]), np.zeros((1, 1)))
        m.barycentric_map(np.ones((1, 1)), np.zeros((1, 2)))

    # your topic 15 exact solver and barycentric map: the reference under pytest/verify (or MLFS_REFERENCE=ot15), and
    # whenever a topic 15 stage it needs is unfinished
    prev = notebooks.load_topic("15-wasserstein-distances", key="ot15", smoke=_ot15_smoke)
    ot15 = prev.module


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # entropic OT & sinkhorn · lab

    every experiment here runs **your** code: your Sinkhorn (both versions), objectives and
    divergence from this topic, and your exact LP and barycentric map from topic 15. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare with `solutions/lab_notes.md` afterwards. the lab takes about 20–25 seconds on one CPU
    core. all data are synthetic.
    """)
    return


@app.cell
def _():
    prev.callout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how $\varepsilon$ trades blur for speed

    60 source points (a Gaussian blob) and 60 target points (two moons), uniform weights,
    squared Euclidean cost, rescaled so the **median cost is 1**; $\varepsilon$ is therefore in
    units of a typical cost. for each $\varepsilon$ we run your `sinkhorn_log` and measure:

    - **iterations** to reach a row-marginal error below $10^{-6}$ (by doubling `max_iters`
      from 1, so the count is exact to within a factor of 2), and the time that took;
    - **partners per point**, $e^{H(P) - H(a)}$: how many targets each source point
      effectively spreads its mass over (1 for a permutation, 60 for the independent coupling);
    - the **cost gap** $\langle P_\varepsilon, C\rangle - W_2^2$ against your exact LP;
    - what your **naive** `sinkhorn` returns with the same number of iterations: the same plan, or not.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "going from eps = 0.1 to eps = 0.001 (in units of the median cost): how many times more iterations does "
        "Sinkhorn need? at which eps does your naive version break, and does it break with NaN?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _g = np.random.default_rng(0)
    Xs = _g.normal(size=(60, 2)) * 0.4 + np.array([0.6, -2.2])
    Ys = make_moons(60, noise=0.05, rng=1)[0] * 1.2
    a1 = np.full(60, 1 / 60)
    C1 = ((Xs[:, None] - Ys[None]) ** 2).sum(-1)
    C1 = C1 / np.median(C1)
    P_lp, w_lp = ot15.kantorovich_lp(a1, a1, C1)

    def iterations_to(a, b, C, eps, tol=1e-6, cap=2 ** 15):
        k, t_total = 1, 0.0
        while True:
            t0 = time.perf_counter()
            P, f, g = ex.sinkhorn_log(a, b, C, eps, max_iters=k, tol=0.0)
            t_total = time.perf_counter() - t0
            if np.abs(P.sum(1) - a).sum() <= tol or k >= cap:
                return k, t_total, P, np.abs(P.sum(1) - a).sum() <= tol
            k *= 2

    eps_grid = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    plans = {}
    _rows = []
    for _eps in eps_grid:
        _k, _t, _P, _ok = iterations_to(a1, a1, C1, _eps)
        plans[_eps] = _P
        _Pp = _P[_P > 0]
        _partners = np.exp(-(_Pp * np.log(_Pp)).sum() - np.log(60))
        with np.errstate(all="ignore"):
            _Pn = ex.sinkhorn(a1, a1, C1, _eps, max_iters=_k, tol=0.0)
        if not np.all(np.isfinite(_Pn)):
            _naive = "NaN"
        elif np.abs(_Pn - _P).sum() < 1e-6:
            _naive = "same plan"
        else:
            _naive = f"finite, off by {np.abs(_Pn - _P).sum():.2f} (L1)"
        _iters = f"{_k // 2 + 1}–{_k}" if _ok else f"> {_k} (not converged)"
        _rows.append(f"| {_eps:g} | {_iters} | {_t * 1000:.0f} | {_partners:.1f} | {np.sum(_P * C1) - w_lp:.4f} | {_naive} |")
    plan_eps = mo.ui.dropdown([str(e) for e in eps_grid], value="0.01", label="plan to draw: eps")
    mo.vstack([mo.md(f"exact $W_2^2$ (your LP) = {w_lp:.4f}\n\n"
                     "| eps | iterations | ms | partners per point | cost gap | naive `sinkhorn` |\n"
                     "|---|---|---|---|---|---|\n" + "\n".join(_rows)), plan_eps])
    return C1, P_lp, Xs, Ys, a1, plan_eps, plans


@app.cell
def _(C1, P_lp, Xs, Ys, a1, plan_eps, plans):
    _eps = float(plan_eps.value)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5))
    for _ax, _P, _title in [(_axes[0], plans[_eps], f"entropic plan, eps = {_eps:g}"), (_axes[1], P_lp, "exact plan (LP)")]:
        _i, _j = np.nonzero(_P > 1e-4 / 60)
        for _ii, _jj in zip(_i, _j):
            _ax.plot([Xs[_ii, 0], Ys[_jj, 0]], [Xs[_ii, 1], Ys[_jj, 1]], color="k",
                     alpha=float(min(1.0, _P[_ii, _jj] * 60)) * 0.6, lw=0.6)
        _T = ot15.barycentric_map(_P, Ys)
        _ax.scatter(*Xs.T, s=12, label="source")
        _ax.scatter(*Ys.T, s=12, label="target")
        _ax.scatter(*_T.T, s=12, marker="x", label="barycentric map T(x)")
        _ax.set(title=_title, aspect="equal")
    _axes[0].legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the iteration count grow roughly like $1/\varepsilon$ (what does
    one Sinkhorn step do to the potentials, and how far must they move)? why does the naive
    version fail at small $\varepsilon$ (look at the size of $C/\varepsilon$), and why does
    it give *exactly* the same plans as the log-domain version until then? pick $\varepsilon = 1$ in the dropdown: where do the crosses (the barycentric
    map) land, and why does a blurry plan pull every image towards the target's mean? which
    $\varepsilon$ would you use to *match* points, and which to compute a smooth *loss*?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: Sinkhorn against the exact LP

    $n$ uniform points in the unit square on each side, squared Euclidean cost (median about
    0.17). we compare your topic 15 `kantorovich_lp` (HiGHS) with your `sinkhorn_log` at
    $\varepsilon = 0.01$ and $0.002$ (tolerance $10^{-6}$, at most 3,000 iterations), on time
    and on two estimates of $W_2^2$: the transport part $\langle P_\varepsilon, C\rangle$ of
    the entropic plan, and (at $\varepsilon = 0.01$) your debiased `sinkhorn_divergence`,
    which runs three Sinkhorn problems.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "at n = 400, which is faster: the exact LP, or Sinkhorn at eps = 0.002? and which is closer to the exact "
        "W2^2: the entropic plan's cost ⟨P, C⟩, or the debiased Sinkhorn divergence?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _g = np.random.default_rng(2)
    _rows = []
    race = {}
    for _n in (50, 100, 200, 400):
        _X, _Y = _g.random((_n, 2)), _g.random((_n, 2))
        _a = np.full(_n, 1 / _n)
        _C = ((_X[:, None] - _Y[None]) ** 2).sum(-1)
        _t0 = time.perf_counter()
        _, _w = ot15.kantorovich_lp(_a, _a, _C)
        _t_lp = time.perf_counter() - _t0
        _r, _c = linear_sum_assignment(_C)
        assert np.isclose(_w, _C[_r, _c].mean())
        _cells = [f"{_n}", f"{_w:.5f}", f"{_t_lp * 1000:.0f}"]
        for _eps, _debias in ((0.01, True), (0.002, False)):
            _t0 = time.perf_counter()
            _P, _, _ = ex.sinkhorn_log(_a, _a, _C, _eps, max_iters=3000, tol=1e-6)
            _t_sk = time.perf_counter() - _t0
            _cells += [f"{_t_sk * 1000:.0f}", f"{100 * (np.sum(_P * _C) / _w - 1):+.1f}%"]
            if _debias:
                _S = ex.sinkhorn_divergence(_X, _Y, _eps, max_iters=3000, tol=1e-6)
                _cells.append(f"{100 * (_S / _w - 1):+.1f}%")
            race[_n, _eps] = (_t_sk, np.sum(_P * _C))
        race[_n, "lp"] = (_t_lp, _w)
        _rows.append("| " + " | ".join(_cells) + " |")
    mo.md("| n | exact W2² | LP ms | Sinkhorn ms (eps 0.01) | ⟨P, C⟩ error | S error | Sinkhorn ms (eps 0.002) | "
          "⟨P, C⟩ error |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does Sinkhorn's cost per iteration grow like $n^2$ while the LP's total
    cost grows faster? why can the small-$\varepsilon$ Sinkhorn lose the race anyway? the
    entropic plan's cost $\langle P_\varepsilon, C\rangle$ is always *above* $W_2^2$: why
    (on paper 2)? the debiased divergence can land *below*: what do its two self-terms
    subtract? which estimate would you trust for a training loss, and which for a reported
    number?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: fitting a point cloud: biased vs debiased

    150 points start as a small blob at the origin; the target is 150 points on a ring of
    radius 2. we move the points by gradient descent on a loss, using the envelope theorem
    (lesson, idea 4): for the squared cost, the gradient of $\mathrm{OT}_\varepsilon(X, Y)$
    with respect to $x_i$, divided by its mass $a_i$, is $2\big(x_i - T_{XY}(x_i)\big)$, where
    $T_{XY}$ is the barycentric map of the entropic plan, and the debiased divergence adds
    $-2\big(x_i - T_{XX}(x_i)\big)$ from its self-term. so the two flows are

    - **biased**, minimizing $\mathrm{OT}_\varepsilon(X, Y)$: $x_i \leftarrow x_i - \eta\cdot2\big(x_i - T_{XY}(x_i)\big)$;
    - **debiased**, minimizing $S_\varepsilon(X, Y)$: $x_i \leftarrow x_i - \eta\cdot2\big(T_{XX}(x_i) - T_{XY}(x_i)\big)$.

    $\eta = 0.25$, 60 steps, each with your `sinkhorn_log` (300 iterations) and your topic 15
    `barycentric_map`. we report the mean radius of the fitted cloud (the ring's is 2) and the
    exact $W_2$ to the target.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with eps = 1 (in squared distance; the ring's diameter is 4), where does the biased flow put the points? "
        "and the debiased one?"
    )
    p3
    return (p3,)


@app.cell
def _():
    eps3 = mo.ui.dropdown(["0.1", "0.3", "1.0", "3.0"], value="1.0", label="eps for experiment 3")
    eps3
    return (eps3,)


@app.cell
def _(eps3, p3):
    ui.gate(p3)
    _eps = float(eps3.value)
    _g = np.random.default_rng(3)
    _ang = 2 * np.pi * (np.arange(150) + 0.5) / 150
    Yr = 2 * np.c_[np.cos(_ang), np.sin(_ang)] + 0.05 * _g.normal(size=(150, 2))
    X0 = 0.2 * _g.normal(size=(150, 2))
    _a = np.full(150, 1 / 150)

    def bary(X, Y):
        C = ((X[:, None] - Y[None]) ** 2).sum(-1)
        P, _, _ = ex.sinkhorn_log(_a, _a, C, _eps, max_iters=300, tol=1e-9)
        return ot15.barycentric_map(P, Y)

    def flow(debiased, steps=60, lr=0.25):
        X = X0.copy()
        for _ in range(steps):
            grad = 2 * (X - bary(X, Yr))
            if debiased:
                grad -= 2 * (X - bary(X, X))
            X = X - lr * grad
        return X

    def exact_w2(X, Y):
        C = ((X[:, None] - Y[None]) ** 2).sum(-1)
        r, c = linear_sum_assignment(C)
        return np.sqrt(C[r, c].mean())

    _t0 = time.perf_counter()
    fits = {"biased OT_eps": flow(False), "debiased S_eps": flow(True)}
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.4))
    _rows = []
    for _ax, (_name, _X) in zip(_axes, fits.items()):
        _ax.scatter(*Yr.T, s=8, c="C1", label="target ring")
        _ax.scatter(*_X.T, s=8, c="C0", label="fitted points")
        _ax.set(title=f"{_name}, eps = {_eps:g}", aspect="equal", xlim=(-2.6, 2.6), ylim=(-2.6, 2.6))
        _rows.append(f"| {_name} | {np.linalg.norm(_X, axis=1).mean():.3f} | {exact_w2(_X, Yr):.3f} |")
    _axes[0].legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| loss | mean radius (target: 2) | exact W2 to the ring |\n|---|---|---|\n"
                           + "\n".join(_rows) + f"\n\nboth flows took {time.perf_counter() - _t0:.1f} s.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the biased flow moves each point to $T_{XY}(x_i)$, an *average* of the
    targets it is coupled to: with a blurry plan, where do those averages lie for a ring?
    (that's why $\mathrm{OT}_\varepsilon(X, Y)$ is minimized by a shrunken copy of $Y$, not by
    $Y$ itself.) how does the self-term $T_{XX}$ cancel the shrinkage? switch $\varepsilon$
    to 0.1 and 3.0: how do the two flows change? this is why Feydy et al. (2019) recommend
    the debiased divergence as a loss, and why topic 17's Sinkhorn loss is debiased by
    default.
    """)
    return


if __name__ == "__main__":
    app.run()
