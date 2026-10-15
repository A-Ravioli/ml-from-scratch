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
    import torch.nn.functional as F
    from torch import nn

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    SIGMA_B = 0.3


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # parameterization & signal propagation · lab

    every experiment uses **your** `variance_map`, `q_fixed_point`, `chi1`, `propagate`,
    `MuMLP`, `param_groups` and `activation_deltas` from `exercise.py`. the trained models use
    8,192 MNIST digits, average-pooled to 14 × 14 and standardized per pixel. the lab takes
    about two minutes on a laptop CPU.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()
    _n = min(8192, len(_d["X_train"]))
    _X = torch.tensor(_d["X_train"][:_n], dtype=torch.float32)[:, None] / 255.0
    _X = F.avg_pool2d(_X, 2).reshape(_n, -1)
    X = (_X - _X.mean(0)) / (_X.std(0) + 0.1)
    y = torch.tensor(_d["y_train"][:_n])
    mo.md(f"{'MNIST' if not datasets.is_offline() else 'offline synthetic stand-in'}: {len(X)} images, {X.shape[1]} features")
    return X, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the edge of chaos

    a deep tanh network with weights $\mathcal N(0, \sigma_w^2/\text{fan-in})$ and biases
    $\mathcal N(0, 0.3^2)$. for a range of $\sigma_w$: your $\chi_1$ at the fixed point $q^*$
    (theory), the correlation between two inputs (initially 0.6) after 30 layers of a real
    width-500 network (your `propagate`), and the training loss of a **20-layer**, width-128
    tanh network after 200 steps of SGD on the digits (two seeds). the critical $\sigma_w$,
    where $\chi_1 = 1$, is found by bisection on your `chi1`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which initialization scale trains a 20-layer tanh network best in 200 steps: small weights (the signal "
        "shrinks), large weights (it saturates), or one particular value in between? how sharp is the optimum?"
    )
    p1
    return (p1,)


@app.cell
def _(X, p1, y):
    ui.gate(p1)
    _t0 = time.perf_counter()

    def _chi_at_fixed_point(sw):
        return ex.chi1(ex.q_fixed_point(sw, SIGMA_B, "tanh"), sw, "tanh")

    _lo, _hi = 0.5, 3.0
    for _ in range(40):
        _mid = (_lo + _hi) / 2
        _lo, _hi = (_mid, _hi) if _chi_at_fixed_point(_mid) < 1 else (_lo, _mid)
    sigma_crit = (_lo + _hi) / 2
    _rng = np.random.default_rng(0)
    _a = _rng.normal(size=200)
    _pair = np.stack([_a, 0.6 * _a + 0.8 * _rng.normal(size=200)])
    _sws = np.linspace(0.5, 3.0, 11)
    _chis = [_chi_at_fixed_point(s) for s in _sws]
    _corr = [float(ex.propagate(_pair, 30, 500, sigma_w=s, sigma_b=SIGMA_B, activation="tanh",
                                generator=torch.Generator().manual_seed(0))["corr"][-1]) for s in _sws]

    def _deep_tanh(sw, seed, depth=20, width=128):
        torch.manual_seed(seed)
        layers, d = [], X.shape[1]
        for _ in range(depth):
            lin = nn.Linear(d, width)
            nn.init.normal_(lin.weight, std=sw / math.sqrt(d))
            nn.init.normal_(lin.bias, std=SIGMA_B)
            layers += [lin, nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 10))
        return nn.Sequential(*layers)

    train_sws = [0.6, 1.0, 1.15, 1.25, round(sigma_crit, 2), 1.55, 1.7, 2.0, 3.0]
    trainability = {}
    for _sw in train_sws:
        _losses = []
        for _seed in (0, 1):
            _m = _deep_tanh(_sw, _seed)
            _opt = torch.optim.SGD(_m.parameters(), lr=0.01)
            _g = torch.Generator().manual_seed(_seed)
            for _ in range(200):
                _i = torch.randint(0, len(X), (128,), generator=_g)
                _loss = F.cross_entropy(_m(X[_i]), y[_i])
                _opt.zero_grad()
                _loss.backward()
                _opt.step()
            with torch.no_grad():
                _losses.append(float(F.cross_entropy(_m(X[:2048]), y[:2048])))
        trainability[_sw] = _losses
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.3))
    _axes[0].plot(_sws, _chis, "o-", label=r"$\chi_1$ at $q^*$ (theory)")
    _axes[0].plot(_sws, _corr, "s--", label="correlation after 30 layers (simulation)")
    _axes[0].axhline(1, color="gray", lw=0.7)
    _axes[0].axvline(sigma_crit, color="C3", ls=":", label=fr"edge of chaos, $\sigma_w$ = {sigma_crit:.2f}")
    _axes[0].set(xlabel=r"$\sigma_w$", ylim=(0, 2.5))
    _axes[0].legend(fontsize=7)
    for _sw, _l in trainability.items():
        _axes[1].plot([_sw] * 2, _l, "ko", ms=4)
    _axes[1].axvline(sigma_crit, color="C3", ls=":")
    _axes[1].set(xlabel=r"$\sigma_w$", ylabel="training loss after 200 steps (20 layers)")
    mo.vstack([_fig, mo.md(
        f"critical σ_w for σ_b = {SIGMA_B}: **{sigma_crit:.3f}**\n\n| σ_w | χ₁ | correlation after 30 layers |\n|---|---|---|\n"
        + "\n".join(f"| {s:.2f} | {c:.3f} | {r:.3f} |" for s, c, r in zip(_sws, _chis, _corr))
        + "\n\n| σ_w (20-layer net) | χ₁ | training loss after 200 steps (2 seeds) |\n|---|---|---|\n"
        + "\n".join(f"| {s} | {_chi_at_fixed_point(s):.2f} | {', '.join(f'{v:.3f}' for v in l)} |"
                    for s, l in trainability.items())
        + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return sigma_crit, trainability


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** below the edge ($\chi_1 < 1$) every input is driven to the same point, so
    after 30 layers the network can't tell two different images apart; above it ($\chi_1 > 1$)
    nearby inputs are pushed apart until their correlation forgets where it started. at the edge
    the correlation changes slowly, and information about the input survives many layers. why
    does the same $\chi_1$ also control whether *gradients* vanish or explode on the way back?
    (Schoenholz et al. 2017 show the two are the same quantity.) the losses form a window around
    $\chi_1 \approx 1$, not a spike: read off its width in $\chi_1$. why can a 20-layer network
    tolerate some distance from the edge, and why would a 200-layer one tolerate less?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the coordinate check

    two-hidden-layer ReLU MLPs of width 64 to 2,048, in standard (SP) and maximal-update (μP)
    parameterization, all with the same Adam learning rate $10^{-2}$ (base width 64, so at width
    64 the two differ only in μP's zero-initialized readout). three full-batch Adam steps on 512 digits, then your
    `activation_deltas`: the RMS change of every entry of each layer's output. two seeds.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with the same learning rate, a 2,048-wide network takes three Adam steps. compared with the 64-wide one, how "
        "much more do its hidden activations and its outputs change, in SP and in μP?"
    )
    p2
    return (p2,)


@app.cell
def _(X, p2, y):
    ui.gate(p2)
    _widths = [64, 128, 256, 512, 1024, 2048]
    coord = {}
    for _param in ("sp", "mup"):
        for _w in _widths:
            _runs = []
            for _seed in (0, 1):
                torch.manual_seed(_seed)
                _m = ex.MuMLP(X.shape[1], _w, 10, depth=2, parameterization=_param)
                _runs.append(ex.activation_deltas(_m, X[:512], y[:512], lr=1e-2, steps=3))
            coord[(_param, _w)] = np.mean(_runs, axis=0)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for _ax, _param in zip(_axes, ("sp", "mup")):
        for _j, _lab in enumerate(("hidden 1", "hidden 2", "output")):
            _ax.plot(_widths, [coord[(_param, w)][_j] for w in _widths], "o-", ms=3, label=_lab)
        _ax.set(xscale="log", yscale="log", xlabel="width", title=_param.upper(), xticks=_widths,
                xticklabels=[str(w) for w in _widths])
    _axes[0].set_ylabel("RMS change of activations after 3 steps")
    _axes[0].legend(fontsize=8)
    mo.vstack([_fig, mo.md("| parameterization | width | hidden 1 | hidden 2 | output |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {p} | {w} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} |" for (p, w), v in coord.items()))])
    return (coord,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** under SP, the activations of the second hidden layer and the outputs move
    more and more per step as the network gets wider, at a fixed learning rate; under μP every
    layer moves by about the same amount at every width. Adam moves every weight by about the
    learning rate per step: a hidden unit sums `width` such changes, coherently, because they
    were all computed from the same gradient signal. so how must the hidden learning rate scale
    with width to keep the change of a unit's input of order one? and why does the first layer,
    whose fan-in is fixed at 196, not need that correction?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: μTransfer

    a learning-rate sweep for widths 64, 256 and 1,024 in both parameterizations: Adam at
    $2^{-12}, 2^{-10}, \dots, 2^{-4}$, 200 steps of batches of 128, two seeds, training loss on
    all 8,192 images at the end. under SP, the best learning rate for one width is tuned on that
    width; under μP the claim is that the best learning rate of the *smallest* model is also
    (about) the best for the largest.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "under SP, does the best learning rate change as the width grows 16x? by how much? under μP?"
    )
    p3
    return (p3,)


@app.cell
def _(X, p3, y):
    ui.gate(p3)
    _t0 = time.perf_counter()
    lrs = [2.0 ** k for k in range(-12, -3, 2)]
    _widths = [64, 256, 1024]
    sweep = {}
    for _param in ("sp", "mup"):
        for _w in _widths:
            for _lr in lrs:
                _vals = []
                for _seed in (0, 1):
                    torch.manual_seed(_seed)
                    _m = ex.MuMLP(X.shape[1], _w, 10, depth=2, parameterization=_param)
                    _opt = torch.optim.Adam(ex.param_groups(_m, _lr))
                    _g = torch.Generator().manual_seed(_seed)
                    for _ in range(200):
                        _i = torch.randint(0, len(X), (128,), generator=_g)
                        _loss = F.cross_entropy(_m(X[_i]), y[_i])
                        _opt.zero_grad()
                        _loss.backward()
                        _opt.step()
                    with torch.no_grad():
                        _v = float(F.cross_entropy(_m(X), y))
                    _vals.append(_v if math.isfinite(_v) else 10.0)
                sweep[(_param, _w, _lr)] = _vals
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    _best = {}
    for _ax, _param in zip(_axes, ("sp", "mup")):
        for _w in _widths:
            _m = [np.mean(sweep[(_param, _w, lr)]) for lr in lrs]
            _best[(_param, _w)] = lrs[int(np.argmin(_m))]
            _ax.plot(lrs, _m, "o-", ms=3, label=f"width {_w}")
        _ax.set(xscale="log", yscale="log", xlabel="Adam learning rate", title=_param.upper(), ylim=(0.02, 5))
    _axes[0].set_ylabel("training loss after 200 steps")
    _axes[0].legend(fontsize=8)
    _rows = "\n".join(f"| {p} | {w} | 2^{int(math.log2(_best[(p, w)]))} | "
                      f"{min(np.mean(sweep[(p, w, lr)]) for lr in lrs):.3f} | "
                      f"{np.mean(sweep[(p, w, _best[(p, 64)])]):.3f} |" for p in ("sp", "mup") for w in _widths)
    mo.vstack([_fig, mo.md("| parameterization | width | best learning rate | its loss | loss at width 64's best rate |\n"
                           f"|---|---|---|---|---|\n{_rows}\n\n{time.perf_counter() - _t0:.0f} s")])
    return lrs, sweep


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** under SP the curves shift left as the network widens: the learning rate
    tuned at width 64 is too large for width 1,024, and the widest network's best result needs
    its own sweep. under μP the curves (nearly) line up, wider is better at every learning
    rate, and the small model's best rate is good for the big one. how much is the transfer
    worth, if a sweep at width 1,024 costs about 16 times one at width 64? and look closely:
    does μP's optimum stay exactly put, or only within the resolution of this sweep?
    """)
    return


if __name__ == "__main__":
    app.run()
