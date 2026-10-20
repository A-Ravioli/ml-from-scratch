import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    WIDTHS = (16, 64, 256, 1024, 4096, 16384)
    STEPS = 6000


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # infinite width: NTK & mean field · lab

    every experiment uses **your** `NTKNet`, `empirical_ntk`, `relu_ntk`,
    `kernel_regression_predict`, `ntk_gd_predictions`, `gd_train` and `relative_change` from
    `exercise.py`. the data is deliberately tiny so that the networks can be very wide: inputs on
    the circle, $x = (\cos\phi, \sin\phi, 1)/\sqrt2$ (the constant third coordinate plays the
    role of a bias; without it the ReLU NTK on the circle is nearly blind to some frequencies, and
    gradient descent stalls on them: a 4,096-unit network reached a training loss of 2.0
    instead of 0.0007 in 3,000 steps), 16 training angles (one at a random place in each sixteenth of the circle) and the target
    $y = \sin 3\phi + \tfrac12\cos\phi$. on the circle the ReLU NTK depends only on the angle
    between two inputs, so every kernel here is a function of one variable you can plot. the lab
    takes about a minute on a laptop CPU.
    """)
    return


@app.cell
def _():
    _rng = np.random.default_rng(0)
    phi_train = (np.arange(16) + _rng.uniform(0.1, 0.9, 16)) * 2 * np.pi / 16  # one random angle per sixteenth
    phi_test = np.linspace(0, 2 * np.pi, 200)

    def circle(phi):
        return torch.tensor(np.stack([np.cos(phi), np.sin(phi), np.ones_like(phi)], 1) / np.sqrt(2),
                            dtype=torch.float32)

    def target(phi):
        return np.sin(3 * phi) + 0.5 * np.cos(phi)

    x_train, x_test = circle(phi_train), circle(phi_test)
    y_train = torch.tensor(target(phi_train), dtype=torch.float32)
    return phi_test, phi_train, target, x_test, x_train, y_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the kernel of a random network, as the width grows

    at initialization, compute the empirical NTK of `NTKNet(3, width)` on the 16 training
    points, for widths from 16 to 16,384 (three random networks per width), and compare it with
    the infinite-width formula `relu_ntk`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the empirical NTK is an average over the width's hidden units. by how much does its relative error against "
        "the infinite-width kernel shrink when the width grows 4x?"
    )
    p1
    return (p1,)


@app.cell
def _(p1, x_train):
    ui.gate(p1)
    _t0 = time.perf_counter()
    K_inf = ex.relu_ntk(x_train.double())
    kernel_err = {}
    for _w in WIDTHS:
        _errs = []
        for _seed in range(3):
            torch.manual_seed(_seed)
            _K = ex.empirical_ntk(ex.NTKNet(3, _w), x_train).double()
            _errs.append(float((_K - K_inf).norm() / K_inf.norm()))
        kernel_err[_w] = _errs
    _ws = np.array(WIDTHS)
    _mean = np.array([np.mean(kernel_err[w]) for w in WIDTHS])
    _slope = np.polyfit(np.log(_ws), np.log(_mean), 1)[0]
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for _w in WIDTHS:
        _axes[0].plot([_w] * 3, kernel_err[_w], "ko", ms=3)
    _axes[0].plot(_ws, _mean[0] * (_ws / _ws[0]) ** -0.5, "r--", label=r"$\propto \mathrm{width}^{-1/2}$")
    _axes[0].set(xscale="log", yscale="log", xlabel="width", ylabel="relative error of the NTK")
    _axes[0].legend()
    _gap = np.linspace(0, np.pi, 100)
    _pts = torch.tensor(np.stack([np.cos(_gap), np.sin(_gap), np.ones_like(_gap)], 1) / np.sqrt(2))
    _axes[1].plot(_gap, ex.relu_ntk(_pts[:1], _pts)[0].numpy(), "r-", label="relu_ntk (infinite width)")
    for _w, _c in ((16, "C0"), (1024, "C2")):
        torch.manual_seed(0)
        _axes[1].plot(_gap, ex.empirical_ntk(ex.NTKNet(3, _w), _pts[:1].float(), _pts.float())[0].numpy(), _c,
                      lw=1, label=f"width {_w}")
    _axes[1].set(xlabel=r"angle $\phi - \phi'$ between the two inputs", ylabel=r"$\Theta(x, x')$")
    _axes[1].legend(fontsize=8)
    mo.vstack([_fig, mo.md("| width | relative error (3 networks) |\n|---|---|\n"
                           + "\n".join(f"| {w} | {', '.join(f'{e:.3f}' for e in kernel_err[w])} |" for w in WIDTHS)
                           + f"\n\nfitted slope on the log-log plot: **{_slope:.2f}** · {time.perf_counter() - _t0:.0f} s")])
    return (K_inf,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each entry of the empirical NTK is an average of `width` independent terms,
    one per hidden unit. what does the central limit theorem say about how fast such an average
    approaches its mean? the kernel at width 16 isn't just off by a constant: it *jumps* as the
    angle changes. which of the NTK's two terms contains the step function
    $\mathbb 1[w_j \cdot x > 0]$, and how many jumps can 16 units produce over half a circle?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does a trained wide network do kernel regression?

    train `NTKNet(3, width)` on the 16 points with full-batch gradient descent (your `gd_train`,
    learning rate $1/\lambda_{\max}$ of the infinite-width kernel, 6,000 steps, by which time the
    training loss is below 0.006, and below $10^{-3}$ for every network of width 1,024 or more). separately, *without training*, compute what the network
    linearized at its initialization would predict after the same 6,000 steps, from its
    empirical NTK and its initial outputs (your `ntk_gd_predictions`). compare the two on 200
    test angles, and measure how far the weights moved.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "a network with 16,384 hidden units and 49,152 weights fits 16 points. by what fraction of their initial size "
        "do its weights change during training? does the width-64 network learn the same function?"
    )
    p2
    return (p2,)


@app.cell
def _(K_inf, p2, phi_test, phi_train, target, x_test, x_train, y_train):
    ui.gate(p2)
    _t0 = time.perf_counter()
    lr = 1.0 / float(torch.linalg.eigvalsh(K_inf).max())
    lazy = {}
    for _w in (64, 1024, 16384):
        for _seed in range(3):
            torch.manual_seed(_seed)
            _net = ex.NTKNet(3, _w)
            _init = {k: v.detach().clone() for k, v in _net.named_parameters()}
            _K = ex.empirical_ntk(_net, x_train).double()
            _Kx = ex.empirical_ntk(_net, x_test, x_train).double()
            with torch.no_grad():
                _f0, _f0x = _net(x_train).double(), _net(x_test).double()
            _, _lin = ex.ntk_gd_predictions(_K, _Kx, y_train.double(), _f0, _f0x, lr=lr, steps=STEPS)
            _losses = ex.gd_train(_net, x_train, y_train, lr=lr, steps=STEPS)
            with torch.no_grad():
                _fx = _net(x_test).double()
            lazy[(_w, _seed)] = dict(
                gap=float((_fx - _lin).norm() / _lin.norm()), moved=ex.relative_change(_net, _init),
                loss=_losses[-1], net_pred=_fx.numpy(), lin_pred=np.asarray(_lin), f0=_f0x.numpy())
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for _w, _c in ((64, "C0"), (16384, "C2")):
        _axes[0].plot(phi_test, lazy[(_w, 0)]["net_pred"], _c, label=f"trained, width {_w}")
        _axes[0].plot(phi_test, lazy[(_w, 0)]["lin_pred"], _c, ls=":", label=f"linearized, width {_w}")
    _axes[0].plot(phi_test, target(phi_test), "k", lw=0.8, alpha=0.4, label="target")
    _axes[0].plot(phi_train, y_train.numpy(), "ko", ms=4)
    _axes[0].set(xlabel=r"$\phi$", ylabel="prediction")
    _axes[0].legend(fontsize=7)
    for _key, _mk, _lab in (("gap", "o-", "trained vs linearized (relative distance)"), ("moved", "s--", r"$\|\theta - \theta_0\| / \|\theta_0\|$")):
        _axes[1].plot([64, 1024, 16384], [np.mean([lazy[(w, s)][_key] for s in range(3)]) for w in (64, 1024, 16384)],
                      _mk, label=_lab)
    _axes[1].set(xscale="log", yscale="log", xlabel="width")
    _axes[1].legend(fontsize=8)
    _rows = [f"| {w} | {', '.join(f'{lazy[(w, s)]['loss']:.1e}' for s in range(3))} | "
             f"{', '.join(f'{lazy[(w, s)]['gap']:.3f}' for s in range(3))} | "
             f"{', '.join(f'{lazy[(w, s)]['moved']:.4f}' for s in range(3))} |" for w in (64, 1024, 16384)]
    mo.vstack([_fig, mo.md(f"learning rate {lr:.3f}\n\n| width | final train loss | trained vs linearized | "
                           "relative weight change |\n|---|---|---|---|\n" + "\n".join(_rows)
                           + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return lazy, lr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the widest network moves its weights by about 4% and lands on
    the prediction of a kernel method computed before training started. why does a smaller
    relative weight change make the linear (first-order Taylor) model more accurate? the trained
    predictions are not the infinite-width kernel regression `relu_ntk` would give: they include
    the network's initial function $f_0$, a random function that never goes away. where does
    it show up between the training points?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: lazy versus rich, at the same width

    the same task, width 1,024, two parameterizations: NTK ($f = a \cdot \text{relu}(Wx) /
    \sqrt{m}$) and mean-field ($f = a \cdot \text{relu}(Wx) / m$). the mean-field network's
    tangent kernel is $m$ times smaller, so it needs a learning rate about $m$ times larger to
    move at a comparable speed in function space; it gets $m/3$ times the NTK network's rate (the
    full $m$-fold rate is unstable once its features start to move). we measure how much the *kernel itself* changes during
    training, $\lVert K_T - K_0\rVert / \lVert K_0\rVert$, and the fraction of hidden units whose
    input weights rotated by more than 10 degrees.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "both networks fit the 16 points. which one changes its tangent kernel (its features) during training, "
        "and by roughly how much?"
    )
    p3
    return (p3,)


@app.cell
def _(lr, p3, phi_test, phi_train, x_test, x_train, y_train):
    ui.gate(p3)
    _t0 = time.perf_counter()
    rich = {}
    _m = 1024
    for _param, _lr in (("ntk", lr), ("mf", lr * _m / 3)):
        for _seed in range(3):
            torch.manual_seed(_seed)
            _net = ex.NTKNet(3, _m, parameterization=_param)
            _W0 = _net.W.detach().clone()
            _K0 = ex.empirical_ntk(_net, x_train)
            _losses = ex.gd_train(_net, x_train, y_train, lr=_lr, steps=STEPS)
            _K1 = ex.empirical_ntk(_net, x_train)
            _cos = torch.nn.functional.cosine_similarity(_net.W.detach(), _W0, dim=1).clamp(-1, 1)
            with torch.no_grad():
                _pred = _net(x_test).numpy()
            rich[(_param, _seed)] = dict(loss=_losses[-1], dK=float((_K1 - _K0).norm() / _K0.norm()),
                                         rotated=float((torch.rad2deg(torch.arccos(_cos)) > 10).float().mean()),
                                         pred=_pred)
    _fig, _ax = plt.subplots(figsize=(5, 3.2))
    for _param, _c in (("ntk", "C0"), ("mf", "C3")):
        _ax.plot(phi_test, rich[(_param, 0)]["pred"], _c, label=f"{_param} (width {_m})")
    _ax.plot(phi_train, y_train.numpy(), "ko", ms=4)
    _ax.set(xlabel=r"$\phi$", ylabel="trained prediction")
    _ax.legend(fontsize=8)
    _rows = [f"| {p} | {', '.join(f'{rich[(p, s)]['loss']:.1e}' for s in range(3))} | "
             f"{', '.join(f'{rich[(p, s)]['dK']:.3f}' for s in range(3))} | "
             f"{', '.join(f'{rich[(p, s)]['rotated']:.2f}' for s in range(3))} |" for p in ("ntk", "mf")]
    mo.vstack([_fig, mo.md("| parameterization | final train loss | kernel change | units rotated > 10° |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows) + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return (rich,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in the mean-field network the kernel changes by an amount of order one and
    many hidden units turn towards new directions: it *learns features*, which a kernel method,
    by definition, can't. in the NTK network nothing moves. the two scalings differ only in a
    constant per width; why does dividing by $m$ instead of $\sqrt m$ (and compensating with the
    learning rate) change which terms of the Taylor expansion matter? which of the two do you
    think describes the networks that work well in practice, and what would that mean for
    explanations of deep learning based on the NTK?
    """)
    return


if __name__ == "__main__":
    app.run()
