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

    from mlfs import datasets, notebooks, ui
    from mlfs.diffusion import TimeMLP  # yours, from topic 03
    from mlfs.generative import mode_coverage  # yours, from ch 21 · topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    STD = 0.05
    ANG = 2 * np.pi * np.arange(8) / 8
    CENTERS = 2.0 * np.c_[np.cos(ANG), np.sin(ANG)]
    X_DATA = datasets.eight_gaussians(10000, std=STD, rng=0)
    Z_TEST = torch.randn(4000, 2, generator=torch.Generator().manual_seed(5))
    ITERS = 3000


@app.function
def new_net(seed=0):
    torch.manual_seed(seed)
    return TimeMLP(2, hidden_dim=256, n_layers=3, time_dim=64)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # flow matching & rectified flow · lab

    every experiment uses **your** `flow_matching_loss`, `sample_flow`,
    `fit_flow_matching`, `make_reflow_pairs` and `straightness`, with topic 03's
    `TimeMLP` (3 layers of 256). the data are the 8 gaussians (std {STD}), 10,000 points,
    synthetic, so offline runs are the same experiments. compare with
    `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 1: straighter paths, fewer steps

    a flow trained with independent noise ({ITERS} steps of batch 512), then one round of
    reflow: 10,000 noise vectors pushed through it (100 Euler steps) become the pairs for a
    second, fresh network trained the same way. both are sampled from the same 4,000 noise
    vectors with 1 to 64 Euler steps.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the original flow with 1 Euler step: what do its samples look like? after one round of reflow, how many Euler "
        "steps match the original flow's quality at 16 steps?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    flow1 = new_net(0)
    _t0 = time.perf_counter()
    losses1 = ex.fit_flow_matching(flow1, X_DATA, n_iters=ITERS, batch_size=512, lr=1e-3, seed=0)
    _t1 = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    _x0, _x1 = ex.make_reflow_pairs(flow1, 10000, (2,), n_steps=100, generator=torch.Generator().manual_seed(1))
    _tp = time.perf_counter() - _t0
    flow2 = new_net(0)
    _t0 = time.perf_counter()
    losses2 = ex.fit_flow_matching(flow2, _x1, X0=_x0, n_iters=ITERS, batch_size=512, lr=1e-3, seed=0)
    _t2 = time.perf_counter() - _t0
    _steps = [1, 2, 4, 8, 16, 64]
    _rows, _show = [], {}
    for _name, _m in (("independent (1-rectified)", flow1), ("reflow (2-rectified)", flow2)):
        _cells = []
        for _n in _steps:
            _xs = ex.sample_flow(_m, (4000, 2), n_steps=_n, x0=Z_TEST).numpy()
            _k, _q, _ = mode_coverage(_xs, CENTERS, STD)
            _cells.append(f"{_q:.3f} ({_k})")
            if _n in (1, 4):
                _show[(_name, _n)] = _xs
        _rows.append(f"| {_name} | {ex.straightness(_m, Z_TEST, n_steps=64):.4f} | " + " | ".join(_cells) + " |")
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for _ax, ((_name, _n), _xs) in zip(_axes, _show.items()):
        _ax.scatter(*_xs.T, s=1, alpha=0.3)
        _ax.set(xlim=(-2.8, 2.8), ylim=(-2.8, 2.8), aspect="equal", title=f"{_name.split(' ')[0]}, {_n} Euler step(s)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"training {_t1:.0f} s + reflow pairs {_tp:.0f} s + retraining {_t2:.0f} s · final losses {np.mean(losses1[-200:]):.3f} "
        f"and {np.mean(losses2[-200:]):.4f} · cells: fraction within 3 std of a mode (modes covered)\n\n"
        "| flow | straightness | " + " | ".join(f"{n} step{'s' if n > 1 else ''}" for n in _steps) + " |\n|"
        + "---|" * (len(_steps) + 2) + "\n" + "\n".join(_rows))])
    return flow1, flow2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where does every 1-step sample of the original flow land, and why (on
    paper 4)? the reflowed flow is nearly straight (its straightness is hundreds of times
    smaller): why does that make one Euler step nearly exact? and why is the reflowed flow
    slightly *worse* than the original at 64 steps (what data was it trained on)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what the paths look like

    40 trajectories of each flow (64 Euler steps from the same noise), drawn over the data,
    together with the straight lines from start to end.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the original flow was trained only on straight lines between noise and data. will its trajectories be "
        "straight? what will they do near the middle of the plot?"
    )
    p2
    return (p2,)


@app.cell
def _(flow1, flow2, p2):
    ui.gate(p2)
    _z = Z_TEST[:40]
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for _ax, (_name, _m) in zip(_axes, (("independent coupling", flow1), ("after reflow", flow2))):
        _m.eval()
        _path, _x = [_z.clone()], _z.clone()
        with torch.no_grad():
            for _k in range(64):
                _x = _x + _m(_x, torch.full((40,), _k / 64)) / 64
                _path.append(_x.clone())
        _p = torch.stack(_path).numpy()
        _ax.scatter(*X_DATA[:2000].T, s=1, c="lightgray")
        for _i in range(40):
            _ax.plot(_p[:, _i, 0], _p[:, _i, 1], lw=0.8)
            _ax.plot([_p[0, _i, 0], _p[-1, _i, 0]], [_p[0, _i, 1], _p[-1, _i, 1]], "k:", lw=0.4)
        _ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=_name)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at $t = 0$, which way does the original flow's velocity point for every
    starting point (on paper 4), and how do the trajectories have to bend later to reach
    their modes? why can't two trajectories of the same flow cross, while the straight
    training lines between independent pairs cross all the time? how does reflow exploit
    that difference?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: noise to noise

    the most trivial "dataset" possible: the data are themselves standard normal (10,000
    points). the identity map would be a perfect flow. we train the same network on it for
    2,000 steps, and look at the loss and at samples with 1 and 64 Euler steps.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "data and noise have the same distribution. what loss will flow matching reach (per coordinate)? what will "
        "1-step samples look like?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _Xn = np.random.default_rng(3).normal(size=(10000, 2))
    _m = new_net(0)
    _losses = ex.fit_flow_matching(_m, _Xn, n_iters=2000, batch_size=512, lr=1e-3, seed=0)
    _one = ex.sample_flow(_m, (4000, 2), n_steps=1, x0=Z_TEST)
    _many = ex.sample_flow(_m, (4000, 2), n_steps=64, x0=Z_TEST)
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.4))
    _axes[0].plot(_losses, lw=0.5)
    _axes[0].axhline(math.pi / 2, c="r", ls="--", label="floor pi/2")
    _axes[0].set(xlabel="step", ylabel="loss", ylim=(1.3, 2.5), title="training loss")
    _axes[0].legend()
    for _ax, (_t, _s) in zip(_axes[1:], (("1 Euler step", _one), ("64 Euler steps", _many))):
        _ax.scatter(*_s.numpy().T, s=1, alpha=0.3)
        _ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=_t)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"final loss **{np.mean(_losses[-200:]):.3f}** (floor $\\pi/2$ = {math.pi / 2:.3f}) · 1-step samples: std "
        f"{_one.std(0).numpy().round(3).tolist()} · 64-step samples: std {_many.std(0).numpy().round(3).tolist()}, "
        f"mean distance from their own starting noise **{(_many - Z_TEST).norm(dim=1).mean():.3f}** · straightness "
        f"{ex.straightness(_m, Z_TEST, n_steps=64):.3f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the loss sits at $\pi/2$, not 0 (on paper 3): what is the network unable
    to know from $(x_t, t)$? the 64-step flow ends almost where it started, yet one step
    collapses everything to the origin: describe a path that starts and ends at the same
    point but isn't the identity (the closed form $x_t = x_0\sqrt{(1 - t)^2 + t^2}$ of idea
    2 is the answer). what would one round of reflow do here?
    """)
    return


if __name__ == "__main__":
    app.run()
