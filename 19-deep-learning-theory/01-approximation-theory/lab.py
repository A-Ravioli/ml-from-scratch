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
    from torch import nn

    from mlfs import notebooks, ui
    from mlfs.torchnn import MLP, count_parameters  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))


@app.function
def tent_k(x, k):
    """the tent map t(x) = 1 - |2x - 1| applied k times (torch or numpy)."""
    for _ in range(k):
        x = 1 - abs(2 * x - 1)
    return x


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # approximation theory · lab

    every experiment uses **your** `relu_interpolant`, `sawtooth_net`, `count_linear_pieces`,
    `width_for_budget` and `fit_regressor` from `exercise.py`, and your `MLP` from chapter 15.
    the data is synthetic (no downloads): the question is what networks of a given size can
    represent, and what gradient descent actually finds. the lab takes about a minute on a
    laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a function depth makes cheap

    the tent map applied $k$ times is a sawtooth with $2^k$ linear pieces. your `sawtooth_net(k)`
    computes it exactly with $6k + 1$ weights; a one-hidden-layer network needs a unit per kink,
    so your `relu_interpolant` through its $2^k + 1$ corners needs $2^k$ hidden units. first the
    count, then the question that matters for practice: if you *train* networks on the sawtooth,
    does depth help gradient descent find it?
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "a deep network of width 2 represents the 64-piece sawtooth exactly with 37 weights. trained from a random "
        "start on 1,024 samples of that sawtooth, how close will a depth-6 network get, compared with a shallow one "
        "of equal size?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    _rows = []
    for _k in range(1, 9):
        _deep = ex.sawtooth_net(_k)
        _corners = np.linspace(0, 1, 2 ** _k + 1)
        _shallow = ex.relu_interpolant(_corners, tent_k(_corners, _k))
        _rows.append(f"| {_k} | {2 ** _k} | {count_parameters(_deep)} | {ex.count_linear_pieces(_deep)} | "
                     f"{count_parameters(_shallow)} | {ex.count_linear_pieces(_shallow)} |")
    _x = torch.linspace(0, 1, 1024).reshape(-1, 1)
    saw_runs = {}
    for _k in (4, 6):
        _y = tent_k(_x, _k).reshape(-1)
        _arch = {"deep, width 2 (the sawtooth's own shape)": (2, _k), "deep, width 8": (8, _k)}
        _budget = count_parameters(MLP(1, [8] * _k, 1))
        _arch["shallow, same parameters as width 8"] = (ex.width_for_budget(_budget, 1), 1)
        for _name, (_w, _depth) in _arch.items():
            _errs = []
            for _seed in range(3):
                torch.manual_seed(_seed)
                _m = MLP(1, [_w] * _depth, 1)
                ex.fit_regressor(_m, _x, _y, steps=2000, lr=3e-3, batch_size=256, seed=_seed)
                with torch.no_grad():
                    _errs.append(float(((_m(_x).reshape(-1) - _y) ** 2).mean() / _y.var()))
            saw_runs[(_k, _name)] = dict(params=count_parameters(_m), err=_errs)
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _xs = np.linspace(0, 1, 2001)
    _ax.plot(_xs, ex.sawtooth_net(6)(torch.tensor(_xs, dtype=torch.float32).reshape(-1, 1)).detach().numpy(), lw=0.8)
    _ax.set(title="sawtooth_net(6): 64 pieces from 37 weights", xlabel="x")
    _train = "\n".join(f"| {k} | {name} | {r['params']} | {', '.join(f'{e:.2f}' for e in r['err'])} |"
                       for (k, name), r in saw_runs.items())
    mo.vstack([_fig, mo.md(
        "| k | pieces | deep: weights | deep: pieces counted | shallow: weights | shallow: pieces counted |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(_rows)
        + "\n\n**trained from random starts** (2,000 Adam steps; error = test MSE / variance of the target, so 1.0 "
        "means \"predicts the mean\"; 3 seeds)\n\n| k | network | weights | relative error (3 seeds) |\n|---|---|---|---|\n"
        + _train + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return (saw_runs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the hand-built deep network needs 49 weights for 256 pieces where the
    shallow one needs 769: exponentially fewer, Telgarsky's depth separation. yet trained from
    random starts, *no* network finds the sawtooth at $k = 6$, and the deep ones don't even beat
    the shallow one; most end up predicting roughly the mean. what does that say about the gap
    between "a network of this shape can represent $f$" and "gradient descent will find $f$"?
    (think about what the loss surface looks like for a function whose output depends on six
    successive folds of the input.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a function depth makes learnable

    a radial target in $d = 4$ dimensions: $x$ uniform in direction with radius $r$ uniform on
    $[0, 2]$, and $y = \sin(\omega\pi r)$, a ripple that depends only on the distance from the
    origin. four MLPs with the same parameter budget (3,000 weights, widths from your
    `width_for_budget`) and 1 to 4 hidden layers, trained with your `fit_regressor` on 20,000
    samples for 3,000 steps, three seeds each, measured on 4,000 new samples.
    """)
    return


@app.cell
def _():
    omega = mo.ui.dropdown(options={"2 (one ripple)": 2, "4 (two ripples)": 4}, value="4 (two ripples)",
                           label="frequency ω")
    p2 = ui.predict(
        "with the same 3,000 parameters, how much better (if at all) does a 3- or 4-layer network fit the radial "
        "ripple than a one-hidden-layer network?"
    )
    mo.vstack([p2, omega])
    return omega, p2


@app.cell
def _(omega, p2):
    ui.gate(p2)
    _t0 = time.perf_counter()

    def _radial(n, rng):
        u = rng.normal(size=(n, 4))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        r = rng.uniform(0, 2, size=(n, 1))
        return (torch.tensor(u * r, dtype=torch.float32),
                torch.tensor(np.sin(omega.value * np.pi * r[:, 0]), dtype=torch.float32))

    _rng = np.random.default_rng(0)
    _x, _y = _radial(20000, _rng)
    _xt, _yt = _radial(4000, _rng)
    radial = {}
    for _depth in (1, 2, 3, 4):
        _w = ex.width_for_budget(3000, _depth, in_dim=4)
        _errs = []
        for _seed in range(3):
            torch.manual_seed(_seed)
            _m = MLP(4, [_w] * _depth, 1)
            ex.fit_regressor(_m, _x, _y, steps=3000, lr=3e-3, batch_size=256, seed=_seed)
            with torch.no_grad():
                _errs.append(float(((_m(_xt).reshape(-1) - _yt) ** 2).mean() / _yt.var()))
        radial[_depth] = dict(width=_w, params=count_parameters(_m), err=_errs)
    _fig, _ax = plt.subplots(figsize=(4.5, 3))
    for _d, _r in radial.items():
        _ax.plot([_d] * 3, _r["err"], "ko", ms=4)
    _ax.plot(list(radial), [np.median(r["err"]) for r in radial.values()], "C0-")
    _ax.set(xlabel="hidden layers (equal parameter budget)", ylabel="test MSE / variance", xticks=[1, 2, 3, 4],
            yscale="log")
    mo.vstack([_fig, mo.md(
        f"ω = {omega.value}\n\n| hidden layers | width | weights | relative test error (3 seeds) |\n|---|---|---|---|\n"
        + "\n".join(f"| {d} | {r['width']} | {r['params']} | {', '.join(f'{e:.3f}' for e in r['err'])} |"
                    for d, r in radial.items()) + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return (radial,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a deeper network can first compute something like $\lVert x\rVert$ (a sum of
    per-coordinate bumps) and then a function of that single number; a one-hidden-layer network
    has to build the ripple out of ridge functions $\text{relu}(w \cdot x + b)$, each constant
    along a hyperplane, and a radial ripple is far from any small sum of those. why does that
    get worse as the dimension grows (Eldan & Shamir, 2016)? and why might this target be
    learnable by depth when the sawtooth wasn't? compare how each target's output depends on the
    first layer's weights.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how many pieces does a random network have?

    the sawtooth shows that depth *can* multiply the number of linear pieces. does it usually?
    take 1-D networks with 48 ReLUs in total, arranged as 1 × 48, 2 × 24, 4 × 12 or 8 × 6,
    initialized by PyTorch's default, and count their pieces on $[-3, 3]$ with your
    `count_linear_pieces` (five random networks each).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "a depth-8 network can have up to 2^8-ish times more pieces than a shallow one with the same number of units. "
        "how many pieces does a randomly initialized one actually have, compared with the shallow network's at most 49?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    regions = {}
    for _depth, _w in ((1, 48), (2, 24), (4, 12), (8, 6)):
        _counts = []
        for _seed in range(5):
            torch.manual_seed(_seed)
            _counts.append(ex.count_linear_pieces(MLP(1, [_w] * _depth, 1), -3.0, 3.0))
        regions[(_depth, _w)] = _counts
    _fig, _ax = plt.subplots(figsize=(4.5, 3))
    for (_d, _w), _c in regions.items():
        _ax.plot([_d] * len(_c), _c, "ko", ms=4)
    _ax.axhline(49, color="C3", ls=":", label="most a 48-unit shallow net can have")
    _ax.set(xscale="log", xticks=[1, 2, 4, 8], xticklabels=["1", "2", "4", "8"], xlabel="hidden layers (48 units in total)",
            ylabel="linear pieces on [-3, 3]")
    _ax.minorticks_off()
    _ax.legend(fontsize=7)
    mo.vstack([_fig, mo.md("| layers × width | pieces (5 random networks) |\n|---|---|\n"
                           + "\n".join(f"| {d} × {w} | {', '.join(map(str, c))} |" for (d, w), c in regions.items())
                           + f"\n\nfor comparison, `sawtooth_net(6)` has 12 units and {ex.count_linear_pieces(ex.sawtooth_net(6))} pieces")])
    return (regions,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** random networks have fewer pieces than they have units, nowhere near the
    exponential maximum, and the deepest have the fewest. Hanin & Rolnick (2019) proved
    that the *expected* number of pieces along a line grows with the total number of neurons,
    not exponentially with depth. what has to be true of the weights for depth to multiply
    pieces the way `sawtooth_net` does? (look at its weights: every layer folds $[0, 1]$ onto
    itself exactly.) why would random weights almost never do that?
    """)
    return


if __name__ == "__main__":
    app.run()
