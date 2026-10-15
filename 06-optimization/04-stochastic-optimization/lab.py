import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # stochastic optimization · lab

    every experiment here runs **your** optimizers from `exercise.py`, through the same
    `zero_grad → set .grad → step` loop you'll use for the rest of the course. for each one:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.function
def train(make_opt, w0, grad_fn, n_steps):
    """run n_steps of the standard loop with exact gradients; returns the iterates (n_steps + 1, d)."""
    p = ex.Param(w0)
    opt = make_opt([p])
    path = [p.data.copy()]
    for _ in range(n_steps):
        opt.zero_grad()
        p.grad = grad_fn(p.data)
        opt.step()
        path.append(p.data.copy())
    return np.array(path)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: when Adam beats SGD, and when it doesn't

    a 20-dimensional quadratic $f(w) = \tfrac12 w^\top Aw$ with eigenvalues from 1 to $10^6$
    ($\kappa = 10^6$, like features measured in wildly different units), with exact gradients, in two versions: **aligned** ($A$ diagonal, so
    the ill-conditioning is along the coordinate axes) and **rotated** ($A = QDQ^\top$ for a
    random rotation $Q$: the same eigenvalues, the same problem up to a change of basis). we
    run your `SGD` (momentum 0.9) and your `Adam` for 300 steps, each with the best learning
    rate from a small grid, and compare the final loss.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "on the aligned problem, which wins after 300 steps, and by how much? does rotating the problem change "
        "SGD's result? Adam's?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _D = np.geomspace(1.0, 1e6, 20)
    _Q, _ = np.linalg.qr(_rng.normal(size=(20, 20)))
    _w0 = _rng.normal(size=20)
    _problems = {"aligned": np.diag(_D), "rotated": _Q @ np.diag(_D) @ _Q.T}
    _grids = {"SGD + momentum": (lambda lr: (lambda ps: ex.SGD(ps, lr=lr, momentum=0.9)), [1e-7, 3e-7, 1e-6, 2e-6, 3e-6]),
              "Adam": (lambda lr: (lambda ps: ex.Adam(ps, lr=lr)), [1e-3, 3e-3, 1e-2, 3e-2, 1e-1])}
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    _rows = []
    for _ax, (_pname, _A) in zip(_axes, _problems.items()):
        _f = lambda w, A=_A: 0.5 * np.einsum("...i,ij,...j->...", w, A, w)
        _g = lambda w, A=_A: A @ w
        for _oname, (_make, _lrs) in _grids.items():
            _best = None
            for _lr in _lrs:
                with np.errstate(all="ignore"):
                    _vals = _f(train(_make(_lr), _w0, _g, 300))
                if np.isfinite(_vals[-1]) and (_best is None or _vals[-1] < _best[1][-1]):
                    _best = (_lr, _vals)
            _ax.semilogy(_best[1], label=f"{_oname} (lr {_best[0]:g})")
            _rows.append(f"| {_pname} | {_oname} | {_best[0]:g} | {_best[1][-1]:.2e} |")
        _ax.set(title=f"{_pname} quadratic, κ = 10⁶", xlabel="step")
        _ax.legend(fontsize=8)
    _axes[0].set_ylabel("f(w) − f*")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| problem | optimizer | best lr | f after 300 steps |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does Adam's division by $\sqrt{\hat v}$ do to each coordinate of the
    aligned problem (what is the gradient's size along an axis with curvature $a_i$)? why
    does that stop working after a rotation? why is SGD's result identical (up to rounding)
    on both problems? what does this tell you about *where* Adam's advantage comes from, and
    about features on very different scales?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the noise floor

    least squares, $f(w) = \frac1{2n}\|Xw - y\|^2$, on $n = 1{,}000$ synthetic examples with 5
    features, trained by your `SGD` (no momentum) on minibatches from your
    `minibatch_indices`, at a constant learning rate, for 30 epochs. we measure the average
    excess loss $f(w) - f^*$ over every step of the last 15 epochs (and 4 random seeds), and
    compare it with your
    `sgd_noise_floor`, fed with the Hessian's eigenvalues and the minibatch gradient noise at
    $w^*$ along each eigenvector. one extra row draws every batch of 64 independently instead of
    reshuffling once per epoch, and the last row uses a decaying rate $\eta_k = \eta_0/(1 + k/200)$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with a constant learning rate, how does the final excess loss depend on the batch size (1, 4, 16, 64)? "
        "if you halve the learning rate? does the decaying schedule get below all of them?"
    )
    p2
    return (p2,)


@app.cell
def _():
    lr_slider = mo.ui.slider(0.005, 0.1, value=0.05, step=0.005, label="constant learning rate η")
    lr_slider
    return (lr_slider,)


@app.cell
def _(lr_slider, p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _n, _d = 1000, 5
    _X = _rng.normal(size=(_n, _d)) * np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    _y = _X @ np.array([1.0, -2.0, 0.5, 3.0, -1.0]) + 0.5 * _rng.normal(size=_n)
    _w_star = np.linalg.lstsq(_X, _y, rcond=None)[0]
    _f = lambda w: 0.5 * np.mean((_X @ w - _y) ** 2)
    _f_star = _f(_w_star)
    _H = _X.T @ _X / _n
    _a, _V = np.linalg.eigh(_H)
    _G = _X * (_X @ _w_star - _y)[:, None]  # per-example gradients at w*
    _Sigma = np.cov(_G.T, bias=True)
    _eta = lr_slider.value

    def _run(B, sched, epochs=30, seed=0, independent=False):
        """returns (excess loss at the end of each epoch, mean excess loss over every step of the second half).
        independent=True draws every batch afresh (n // B batches per 'epoch') instead of reshuffling once."""
        p = ex.Param(np.zeros(_d))
        opt = ex.SGD([p], lr=_eta)
        rng = np.random.default_rng(seed)
        hist, tail, k = [], [], 0
        for ep in range(epochs):
            batches = ([rng.choice(_n, B, replace=False) for _ in range(_n // B)] if independent
                       else ex.minibatch_indices(_n, B, rng=rng))
            for idx in batches:
                if sched:
                    opt.lr = _eta / (1 + k / 200)
                opt.zero_grad()
                p.grad = _X[idx].T @ (_X[idx] @ p.data - _y[idx]) / len(idx)
                opt.step()
                k += 1
                if ep >= epochs // 2:
                    r = p.data - _w_star
                    tail.append(0.5 * r @ _H @ r)
            hist.append(_f(p.data) - _f_star)
        return np.array(hist), float(np.mean(tail))

    _rows = []
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _B in (1, 4, 16, 64):
        _runs = [_run(_B, False, seed=_sd) for _sd in range(4)]
        _h = _runs[0][0]
        _fpc = (_n - _B) / ((_n - 1) * _B)
        _s = _fpc * np.einsum("ij,jk,ki->i", _V.T, _Sigma, _V)
        _pred = ex.sgd_noise_floor(_eta, _a, _s)
        _meas = float(np.mean([r[1] for r in _runs]))
        _ax.semilogy(np.arange(1, 31), _h, label=f"batch {_B}")
        _rows.append(f"| {_B} | constant {_eta:g} | {_meas:.2e} | {_pred:.2e} |")
    _ind = float(np.mean([_run(64, False, seed=_sd, independent=True)[1] for _sd in range(4)]))
    _rows.append(f"| 64, each batch drawn independently | constant {_eta:g} | {_ind:.2e} | (same as above) |")
    _hd, _tail_d = _run(1, True)
    _ax.semilogy(np.arange(1, 31), _hd, "k--", label="batch 1, decaying")
    _rows.append(f"| 1 | {_eta:g} / (1 + k/200) | {_tail_d:.2e} | → 0 |")
    _ax.set(xlabel="epoch", ylabel="f(w) − f* (end of epoch)", title=f"SGD on least squares, η = {_eta:g}")
    _ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| batch | learning rate | measured mean excess loss (second half, 4 seeds) | your predicted floor |\n"
        "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does the floor scale with the batch size and the learning rate, and
    why (on paper 3 and the variance formula)? the batch-64 run takes 64 times fewer steps
    per epoch: at equal *epochs*, which batch size is ahead, and is that a fair comparison?
    why does the decaying schedule break through the floor, and what does it cost early on?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: same training loss, different weights

    linear regression with **more weights than examples**: $n = 30$ examples, $d = 100$
    features, so infinitely many weight vectors fit the data exactly. we train from $w = 0$
    with full-batch gradients for 20,000 steps using your `SGD` (momentum 0.9), `Adam` and
    `Adagrad`, until the training error is essentially zero, and then measure the error on
    2,000 fresh test examples. two ground truths: **sparse** (5 nonzero weights) and
    **dense** (all 100 nonzero). the reference point is the minimum-norm solution
    $X^+y$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "all three optimizers reach (almost) zero training error. do they find the same weights? "
        "which generalizes best, and is it the same for the sparse and the dense truth?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _n, _d = 30, 100
    _rows = []
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for _ax, _kind in zip(_axes, ("sparse", "dense")):
        _X = _rng.normal(size=(_n, _d))
        _ws = np.zeros(_d)
        if _kind == "sparse":
            _ws[:5] = 3 * _rng.normal(size=5)
        else:
            _ws = _rng.normal(size=_d)
        _y = _X @ _ws
        _Xt = _rng.normal(size=(2000, _d))
        _w_mn = np.linalg.pinv(_X) @ _y
        _g = lambda w, X=_X, y=_y: X.T @ (X @ w - y) / len(y)
        _te_mn = np.mean((_Xt @ (_w_mn - _ws)) ** 2)
        _rows.append(f"| {_kind} | min-norm X⁺y | 0 | {_te_mn:.1f} | {np.linalg.norm(_w_mn):.2f} | "
                     f"{np.abs(_w_mn).max():.2f} | 0 |")
        for _name, _make in (("SGD + momentum", lambda ps: ex.SGD(ps, lr=0.01, momentum=0.9)),
                             ("Adam", lambda ps: ex.Adam(ps, lr=0.01)),
                             ("Adagrad", lambda ps: ex.Adagrad(ps, lr=0.1))):
            _w = train(_make, np.zeros(_d), _g, 20_000)[-1]
            _tr = np.mean((_X @ _w - _y) ** 2)
            _te = np.mean((_Xt @ (_w - _ws)) ** 2)
            _rows.append(f"| {_kind} | {_name} | {_tr:.1e} | {_te:.1f} | {np.linalg.norm(_w):.2f} | "
                         f"{np.abs(_w).max():.2f} | {np.linalg.norm(_w - _w_mn):.2f} |")
            if _name != "Adagrad":
                _ax.scatter(_w_mn, _w, s=10, label=_name)
        _lim = np.abs(_w_mn).max() * 1.2
        _ax.plot([-_lim, _lim], [-_lim, _lim], "k:", lw=0.8)
        _ax.set(xlabel="min-norm solution, weight j", ylabel="trained weight j", title=f"{_kind} truth")
        _ax.set_aspect("equal")
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| truth | weights from | train MSE | test MSE | ‖w‖₂ | max abs weight | ‖w − X⁺y‖ |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does gradient descent from 0 land exactly on the minimum-norm
    solution? (where do the gradients $X^\top(Xw - y)$ live?) why doesn't Adam's iterate stay
    in that subspace? compare the norms and the largest weights: what kind of solution do the
    adaptive methods prefer? the training loss can't tell these solutions apart: what does
    this say about choosing an optimizer by its training curve?
    """)
    return


if __name__ == "__main__":
    app.run()
