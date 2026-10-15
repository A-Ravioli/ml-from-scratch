import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)  # small models: one thread is fastest, and timings are steadier


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # natural gradient · lab

    every experiment here runs **your** natural-gradient code from `exercise.py`. for each
    one: write down a prediction, lock it in, run it, then explain what happened. compare
    your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about a
    minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: three descriptions of one Gaussian

    fit $\mathcal N(\mu, \sigma^2)$ by maximum likelihood to 200 points drawn from
    $\mathcal N(2, 0.3^2)$, starting from $\mathcal N(0, 3^2)$. the model is the same in all
    three runs; only its coordinates differ: $(\mu, \sigma)$, $(\mu, \log\sigma)$ and
    $(\mu, \lambda)$ with the precision $\lambda = 1/\sigma^2$. each coordinate system gets
    60 steps of plain gradient descent and 60 steps of your `natural_gradient_descent`, all
    with the same step size, and every path is drawn in the same $(\mu, \sigma)$ picture.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with step size 0.1: do the three gradient-descent paths end up in the same place? do the three natural-"
        "gradient paths? what changes at step size 0.5?"
    )
    p1
    return (p1,)


@app.cell
def _():
    lr1 = mo.ui.dropdown({"0.02": 0.02, "0.1": 0.1, "0.5": 0.5}, value="0.1", label="step size (both methods)")
    lr1
    return (lr1,)


@app.cell
def _(lr1, p1):
    ui.gate(p1)
    _g = torch.Generator().manual_seed(0)
    _x = 2.0 + 0.3 * torch.randn(200, generator=_g)

    def _nll(mu, s):
        return torch.log(s) + ((_x - mu) ** 2).mean() / (2 * s * s)

    # each entry: coordinates -> (mu, sigma), (mu, sigma) -> coordinates, the Fisher matrix in these coordinates
    _coords = {
        "(μ, σ)": (lambda t: (t[0], t[1]), lambda mu, s: torch.stack([mu, s]),
                   lambda t: torch.diag(torch.stack([1 / t[1] ** 2, 2 / t[1] ** 2]))),
        "(μ, log σ)": (lambda t: (t[0], torch.exp(t[1])), lambda mu, s: torch.stack([mu, torch.log(s)]),
                       lambda t: torch.diag(torch.stack([torch.exp(-2 * t[1]), torch.tensor(2.0)]))),
        "(μ, 1/σ²)": (lambda t: (t[0], t[1] ** -0.5), lambda mu, s: torch.stack([mu, s ** -2]),
                      lambda t: torch.diag(torch.stack([t[1], 0.5 / t[1] ** 2]))),
    }

    def _grad(to_ms):
        def g(t):
            t = t.detach().clone().requires_grad_(True)
            (gr,) = torch.autograd.grad(_nll(*to_ms(t)), t)
            return gr
        return g

    _lr, _n = float(lr1.value), 60
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    _rows = []
    for _i, (_name, (_to, _from, _fis)) in enumerate(_coords.items()):
        _t0 = _from(torch.tensor(0.0), torch.tensor(3.0))
        _t, _gd = _t0.clone(), [_to(_t0)]
        for _ in range(_n):
            _t = _t - _lr * _grad(_to)(_t)
            _ms = _to(_t)
            if not (torch.isfinite(_t).all() and _ms[1] > 0):
                break
            _gd.append(_ms)
        _ng = [_to(t) for t in ex.natural_gradient_descent(_grad(_to), _fis, _t0, lr=_lr, n_steps=_n)]
        for _ax, _path in [(_axes[0], _gd), (_axes[1], _ng)]:
            _a = np.array([[float(m), float(s)] for m, s in _path])
            _ax.plot(_a[:, 0], _a[:, 1], "o-", ms=2.5, lw=1, color=f"C{_i}", label=_name)
        _end = "diverged (σ ≤ 0 or nan) at step " + str(len(_gd)) if len(_gd) <= _n else \
            f"({float(_gd[-1][0]):.3f}, {float(_gd[-1][1]):.3f})"
        _rows.append(f"| {_name} | {_end} | ({float(_ng[10][0]):.3f}, {float(_ng[10][1]):.3f}) | "
                     f"({float(_ng[-1][0]):.3f}, {float(_ng[-1][1]):.3f}) |")
    for _ax, _title in [(_axes[0], "gradient descent"), (_axes[1], "natural gradient descent (yours)")]:
        _ax.plot([float(_x.mean())], [float(_x.std(unbiased=False))], "k*", ms=14, label="MLE")
        _ax.set(yscale="log", xlabel="μ", title=f"{_title}, step size {_lr}", xlim=(-1, 4), ylim=(0.1, 6))
        _ax.legend(fontsize=8)
    _axes[0].set_ylabel("σ (log scale)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"MLE: μ̂ = {float(_x.mean()):.3f}, σ̂ = {float(_x.std(unbiased=False)):.3f}\n\n"
                           "| coordinates | GD after 60 steps (μ, σ) | NGD after 10 steps | NGD after 60 steps |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** gradient descent's path depends on the coordinates; what in its update
    rule makes that inevitable (think about units: what are the units of $\partial L/\partial\sigma$
    and of $\sigma$)? the natural-gradient paths nearly coincide but not exactly, and the gap
    grows with the step size: which part of the method is exactly invariant, and which isn't?
    at step size 0.5 one gradient-descent run produces a negative $\sigma$; why can't the
    natural-gradient run do that so easily?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the empirical Fisher as a preconditioner

    linear regression $y = wx + b + \varepsilon$ on 100 points with $x \sim \mathcal N(2, 1)$
    (uncentred inputs make the loss an elongated valley), fitted by least squares, i.e. a
    Gaussian likelihood with unit noise. its Fisher is constant, $F = \frac1N\sum_n\phi_n\phi_n^\top$
    with $\phi_n = (x_n, 1)$; its empirical Fisher is $\frac1N\sum_n r_n^2\phi_n\phi_n^\top$, weighted by
    the squared residuals. from $(w, b) = (-2, 4)$ we run your `natural_gradient_descent` with
    each of them (step size 1, damping $10^{-8}$) next to gradient descent with its best step
    size. the data's real noise level is a knob: at 1 the model is right about the noise.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with the true noise level 1: how many steps does natural gradient with the Fisher need to reach the "
        "least-squares solution? with the empirical Fisher? and what happens to the empirical-Fisher run when "
        "the real noise level is 0.3?"
    )
    p2
    return (p2,)


@app.cell
def _():
    noise2 = mo.ui.dropdown({"0.3": 0.3, "1 (correct)": 1.0, "3": 3.0}, value="1 (correct)", label="real noise std")
    noise2
    return (noise2,)


@app.cell
def _(noise2, p2):
    ui.gate(p2)
    _N = 100
    _g = torch.Generator().manual_seed(0)
    _x = 2.0 + torch.randn(_N, generator=_g)
    _Phi = torch.stack([_x, torch.ones(_N)], 1)
    _yy = _Phi @ torch.tensor([1.5, -1.0]) + float(noise2.value) * torch.randn(_N, generator=_g)
    _wstar = torch.linalg.lstsq(_Phi, _yy.unsqueeze(1)).solution.squeeze(1)

    def _loss(w):
        return 0.5 * float(((_yy - _Phi @ w) ** 2).mean())

    def _grad(w):
        return -(_Phi.T @ (_yy - _Phi @ w)) / _N

    _F = _Phi.T @ _Phi / _N

    def _ef(w):
        r = _yy - _Phi @ w
        return (_Phi * r[:, None] ** 2).T @ _Phi / _N

    _w0 = torch.tensor([-2.0, 4.0])
    _best = _loss(_wstar)
    _n = 40
    _runs = {"NGD, Fisher": ex.natural_gradient_descent(_grad, lambda w: _F, _w0, lr=1.0, n_steps=_n, damping=1e-8),
             "NGD, empirical Fisher": ex.natural_gradient_descent(_grad, _ef, _w0, lr=1.0, n_steps=_n, damping=1e-8)}
    _gd_best = None
    for _lr in [0.05, 0.1, 0.2, 0.3]:  # gradient descent: keep the step size with the lowest final loss
        _w, _p = _w0.clone(), [_w0.clone()]
        for _ in range(_n):
            _w = _w - _lr * _grad(_w)
            _p.append(_w.clone())
        _p = torch.stack(_p)
        if torch.isfinite(_p).all() and (_gd_best is None or _loss(_p[-1]) < _loss(_gd_best[1][-1])):
            _gd_best = (_lr, _p)
    _runs[f"GD, best step size ({_gd_best[0]})"] = _gd_best[1]
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.4))
    _W, _B = np.meshgrid(np.linspace(-3, 4, 120), np.linspace(-6, 6, 120))
    _Z = np.array([[_loss(torch.tensor([a, b])) for a, b in zip(ra, rb)] for ra, rb in zip(_W, _B)])
    _axes[0].contour(_W, _B, np.log(_Z - _best + 1e-3), levels=20, cmap="Greys", linewidths=0.7)
    _rows = []
    for _i, (_name, _p) in enumerate(_runs.items()):
        _a = _p.numpy()
        _ok = np.isfinite(_a).all(1) & (np.abs(_a) < 1e3).all(1)
        _axes[0].plot(_a[_ok, 0], _a[_ok, 1], "o-", ms=3, lw=1, color=f"C{_i}", label=_name)
        _ex = np.array([_loss(w) - _best if torch.isfinite(w).all() else np.nan for w in _p])
        _axes[1].plot(np.arange(_n + 1), np.maximum(_ex, 1e-16), color=f"C{_i}", label=_name)
        _hit = np.flatnonzero(_ex < 1e-6)
        _rows.append(f"| {_name} | {_ex[1]:.3g} | {_ex[5]:.3g} | {_ex[_n]:.3g} | "
                     f"{_hit[0] if len(_hit) else 'not within ' + str(_n)} |")
    _axes[0].plot(*_wstar.numpy(), "k*", ms=14)
    _axes[0].set(xlabel="w", ylabel="b", xlim=(-3, 4), ylim=(-6, 6), title="paths over the loss")
    _axes[1].set(yscale="log", xlabel="step", ylabel="loss − least-squares loss", title="convergence")
    for _ax in _axes:
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| method | excess loss after 1 step | after 5 | after 40 | first step below 1e-6 |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does natural gradient with the Fisher solve this problem in one step
    (what is it, for a linear-Gaussian model)? the empirical Fisher scales each direction by
    the squared residuals: why does that make its early steps tiny, and why does it suddenly
    catch up near the solution when the noise level is right? set the real noise to 0.3 and
    then 3: near the solution the empirical Fisher is about $\sigma^2_{\text{real}} F$; use
    that to predict, and then explain, what each run does.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: natural gradient on a real network

    an MLP (784 → 32 tanh → 10, 25,450 parameters) on 1,000 MNIST training images,
    full-batch, three seeds per method: SGD (step size 0.5), Adam (its best step size of
    $10^{-3}, 3\cdot10^{-3}, 10^{-2}, 3\cdot10^{-2}$, chosen on seed 0) and your `ngd_step`
    (step size 1, damping $10^{-2}$, conjugate-gradient iterations from the dropdown), 40
    steps each. we record the training loss against steps and against wall-clock time, the
    steps and seconds each run needs to reach a training loss of 0.2, and the accuracy on
    1,000 test images at the end. (offline, a synthetic stand-in replaces
    MNIST.)
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how many steps does natural gradient need to reach the training loss Adam reaches in 40? and in wall-clock "
        "time, which one gets there first? which has the best test accuracy?"
    )
    p3
    return (p3,)


@app.cell
def _():
    cg3 = mo.ui.dropdown({"3": 3, "10": 10, "30": 30}, value="10", label="CG iterations per natural-gradient step")
    cg3
    return (cg3,)


@app.cell
def _(cg3, p3):
    ui.gate(p3)
    _d = datasets.mnist()
    _mean, _std = _d["X_train"][:1000].mean(), _d["X_train"][:1000].std()
    _X = torch.tensor((_d["X_train"][:1000].reshape(1000, -1) - _mean) / _std)
    _y = torch.tensor(_d["y_train"][:1000])
    _Xv = torch.tensor((_d["X_test"][:1000].reshape(-1, 784) - _mean) / _std)
    _yv = torch.tensor(_d["y_test"][:1000])

    def _make(seed):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(784, 32), nn.Tanh(), nn.Linear(32, 10))

    def _run(kind, seed, lr, n=40):
        m = _make(seed)
        opt = None if kind == "ngd" else (torch.optim.Adam if kind == "adam" else torch.optim.SGD)(m.parameters(), lr=lr)
        losses, times, t0 = [], [0.0], time.perf_counter()
        for _ in range(n):
            if kind == "ngd":
                losses.append(ex.ngd_step(m, _X, _y, lr=lr, damping=1e-2, cg_iters=int(cg3.value)))
            else:
                opt.zero_grad()
                loss = nn.functional.cross_entropy(m(_X), _y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            times.append(time.perf_counter() - t0)
        with torch.no_grad():
            losses.append(nn.functional.cross_entropy(m(_X), _y).item())
            acc = (m(_Xv).argmax(1) == _yv).double().mean().item()
        return np.array(losses), np.array(times), acc

    _run("sgd", 0, 0.1, n=2)  # warm-up, so the first timed run doesn't pay for lazy initialisation
    _adam_lr = min([1e-3, 3e-3, 1e-2, 3e-2], key=lambda lr: _run("adam", 0, lr)[0][-1])
    _methods = {"SGD (0.5)": ("sgd", 0.5), f"Adam ({_adam_lr:g})": ("adam", _adam_lr), "NGD (yours)": ("ngd", 1.0)}
    _res = {name: [_run(k, s, lr) for s in range(3)] for name, (k, lr) in _methods.items()}
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for _i, (_name, _rs) in enumerate(_res.items()):
        for _j, (_L, _T, _) in enumerate(_rs):
            _axes[0].plot(np.arange(len(_L)), _L, color=f"C{_i}", alpha=0.8, label=_name if _j == 0 else None)
            _axes[1].plot(_T, _L, color=f"C{_i}", alpha=0.8, label=_name if _j == 0 else None)
    _axes[0].set(yscale="log", xlabel="step", ylabel="training loss (full batch)")
    _axes[1].set(yscale="log", xlabel="wall-clock seconds (one CPU thread)")
    for _ax in _axes:
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _target = 0.2
    _rows = []
    for _name, _rs in _res.items():
        _steps = [int(np.argmax(L <= _target)) if (L <= _target).any() else None for L, _, _ in _rs]
        _secs = [T[s] if s is not None else None for (L, T, _), s in zip(_rs, _steps)]
        _fmt = lambda v, f: ", ".join("—" if x is None else f.format(x) for x in v)  # noqa: E731
        _rows.append(f"| {_name} | {np.median([r[0][-1] for r in _rs]):.4f} | {_fmt(_steps, '{}')} | "
                     f"{_fmt(_secs, '{:.3f}')} | {np.median([r[1][-1] / 40 * 1e3 for r in _rs]):.1f} | "
                     f"{np.mean([r[2] for r in _rs]):.3f} ± {np.std([r[2] for r in _rs], ddof=1) / np.sqrt(3):.3f} |")
    mo.vstack([_fig, mo.md(f"target training loss: **{_target}**\n\n"
                           "| method | final loss (median) | steps to target (per seed) | seconds to target (per seed) | "
                           "ms per step (median) | test accuracy (mean ± s.e.) |\n|---|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each natural-gradient step costs roughly $1 + 3k$ backward-sized passes
    for $k$ CG iterations; compare that with the measured cost per step. why does it need so
    many fewer steps? the wall-clock comparison is closer than the step comparison: what would
    tip it either way (bigger networks, minibatches, more CG iterations)? and look at the test
    accuracy: what does driving the training loss to nearly zero on 1,000 images buy you?
    """)
    return


if __name__ == "__main__":
    app.run()
