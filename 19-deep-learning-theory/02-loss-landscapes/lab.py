import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
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
    N_TRAIN, STEPS, LR = 2000, 3000, 0.2


@app.function
def sq_loss(logits, targets):
    """half the squared error against one-hot labels, summed over classes, averaged over examples."""
    return 0.5 * ((logits - F.one_hot(targets, 10).float()) ** 2).sum(1).mean()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # loss landscapes · lab

    every experiment uses **your** `filter_normalized_direction`, `loss_surface_2d`,
    `linear_interpolation`, `loss_barrier` and `top_hessian_eigenvalue` from `exercise.py`, on
    one small problem: 2,000 MNIST digits, average-pooled to 14 × 14, and a one-hidden-layer
    ReLU network with 128 units (26,506 parameters), trained by plain SGD at learning rate 0.2
    on the squared error against one-hot labels (the loss Cohen et al. used to study sharpness;
    cross-entropy's curvature fades as the loss goes to zero, which muddies comparisons).
    for each experiment: write down a prediction, lock it in, run, explain. the lab takes about
    a minute on a laptop CPU.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()

    def _prep(x):
        x = torch.tensor(x, dtype=torch.float32)[:, None] / 255.0
        return F.avg_pool2d(x, 2).reshape(len(x), -1)

    _X = _prep(_d["X_train"][:N_TRAIN])
    _mu, _sd = _X.mean(), _X.std()
    X = (_X - _mu) / _sd
    y = torch.tensor(_d["y_train"][:N_TRAIN])
    X_test = (_prep(_d["X_test"][:2000]) - _mu) / _sd
    y_test = torch.tensor(_d["y_test"][:2000])
    mo.md(f"{'MNIST' if not datasets.is_offline() else 'offline synthetic stand-in'}: {len(X)} training and "
          f"{len(X_test)} test images, {X.shape[1]} pixels each")
    return X, X_test, y, y_test


@app.function
def make_net(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(196, 128), nn.ReLU(), nn.Linear(128, 10))


@app.cell
def _(X, X_test, y, y_test):
    def train(model, batch_size, steps, seed, lr=LR):
        """plain SGD on random minibatches drawn with a seeded generator."""
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            i = torch.randint(0, len(X), (min(batch_size, len(X)),), generator=g)
            loss = sq_loss(model(X[i]), y[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    def train_loss(model):
        with torch.no_grad():
            return sq_loss(model(X), y).item()

    def test_error(model):
        with torch.no_grad():
            return (model(X_test).argmax(1) != y_test).float().mean().item()

    def sharpness(model, seed=0):
        return ex.top_hessian_eigenvalue(model, lambda: sq_loss(model(X), y), n_iters=20,
                                         generator=torch.Generator().manual_seed(seed))

    return sharpness, test_error, train, train_loss


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: sharpness against batch size

    the same network, trained four ways that differ only in the minibatch size: 16, 64, 256,
    or all 2,000 images (full-batch gradient descent). every run takes 3,000 SGD steps at the
    same learning rate $\eta = 0.2$ and ends with a training loss between 0.015 and 0.03 and
    almost every training digit right. along the way and at the end we measure the
    **sharpness**: the largest eigenvalue of the Hessian of the full training loss, by your
    power iteration. three seeds per batch size; the trajectories are seed 0.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which batch size ends at the sharpest minimum? the learning rate is 0.2: does the number 2 / 0.2 = 10 "
        "mean anything here?"
    )
    p1
    return (p1,)


@app.cell
def _(p1, sharpness, test_error, train, train_loss):
    ui.gate(p1)
    _t0 = time.perf_counter()
    batch_runs = {}
    _bss = (16, 64, 256, N_TRAIN)
    for _bs in _bss:
        for _seed in (0, 1, 2):
            _m = make_net(_seed)
            _traj = [sharpness(_m)] if _seed == 0 else []  # step 0: the sharpness at initialization
            for _chunk in range(10):  # 10 x 300 steps, measuring the sharpness after each chunk
                train(_m, _bs, STEPS // 10, seed=1000 * _seed + _chunk)
                if _seed == 0 or _chunk == 9:
                    _traj.append(sharpness(_m))
            batch_runs[(_bs, _seed)] = dict(model=_m, loss=train_loss(_m), err=test_error(_m), sharp=_traj[-1],
                                            traj=_traj)
    _rows = []
    for _bs in _bss:
        _r = [batch_runs[(_bs, s)] for s in (0, 1, 2)]
        _rows.append(f"| {_bs} | {', '.join(f'{r['loss']:.4f}' for r in _r)} | "
                     f"{', '.join(f'{r['sharp']:.2f}' for r in _r)} | {', '.join(f'{r['err']:.3f}' for r in _r)} |")
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for _bs in _bss:
        _axes[0].plot(np.arange(0, 11) * STEPS // 10, batch_runs[(_bs, 0)]["traj"], "o-", ms=3, label=f"batch {_bs}")
        _axes[1].plot([_bs] * 3, [batch_runs[(_bs, s)]["sharp"] for s in (0, 1, 2)], "ko", ms=4)
    for _ax in _axes:
        _ax.axhline(2 / LR, color="gray", ls=":", label=r"$2/\eta$")
    _axes[0].set(xlabel="step", ylabel="top Hessian eigenvalue")
    _axes[0].legend(fontsize=7)
    _axes[1].set(xscale="log", xlabel="batch size", ylabel="final top eigenvalue (3 seeds)")
    mo.vstack([_fig, mo.md("| batch size | train loss (3 seeds) | sharpness (3 seeds) | test error (3 seeds) |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows) + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return (batch_runs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at initialization the sharpness is far above $2/\eta$, and within the first
    few hundred steps every run drops below it; full-batch (and batch-256) gradient descent then
    settles right at $2/\eta$ and stays there. on a quadratic with curvature $\lambda$, a gradient
    step multiplies the distance to the minimum by $1 - \eta\lambda$: what happens when $\lambda >
    2/\eta$, and why does that push the iterate out of the sharp region it started in? the
    small-batch runs stop far below the line. what is it about the *noise* in a
    small-batch gradient that makes a sharp region hard to stay in? and the test errors differ
    much less than the sharpness: what does that say about "flat minima generalize better" as an
    explanation?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what a picture of a minimum can and can't tell you

    first, two filter-normalized slices: the loss on a plane through the batch-16 solution and
    through the full-batch solution, each with its own pair of random directions from your
    `filter_normalized_direction`, drawn with the same contour levels.

    then a trick. a ReLU network doesn't change its function if you multiply the first layer
    (weights and bias) by $c$ and divide the second layer's weights by $c$: $\text{relu}(cz) =
    c\,\text{relu}(z)$. we do that to the batch-16 solution with $c = 5$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the rescaled network computes exactly the same function, so it has the same training loss and test error. "
        "does it have the same sharpness? does its filter-normalized picture look the same?"
    )
    p2
    return (p2,)


@app.cell
def _(X, batch_runs, p2, sharpness, test_error, train_loss, y):
    ui.gate(p2)

    def _eval(model):
        with torch.no_grad():
            return sq_loss(model(X), y).item()

    def _surface(model, seed):
        g = torch.Generator().manual_seed(seed)
        d1 = ex.filter_normalized_direction(model, generator=g)
        d2 = ex.filter_normalized_direction(model, generator=g)
        return ex.loss_surface_2d(model, _eval, d1, d2, span=1.0, resolution=21)

    _small = batch_runs[(16, 0)]["model"]
    _full = batch_runs[(N_TRAIN, 0)]["model"]
    _scaled = copy.deepcopy(_small)
    with torch.no_grad():
        _scaled[0].weight.mul_(5.0)
        _scaled[0].bias.mul_(5.0)
        _scaled[2].weight.div_(5.0)
    _models = {"batch 16": _small, "full batch": _full, "batch 16, rescaled by c = 5": _scaled}
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
    _levels = np.linspace(0, 4, 17)
    surfaces = {}
    for _ax, (_name, _m) in zip(_axes, _models.items()):
        _a, _b, _Z = _surface(_m, seed=1)
        surfaces[_name] = _Z
        _cs = _ax.contourf(_a, _b, _Z.T, levels=_levels, cmap="viridis", extend="max")
        _ax.set(title=_name, xlabel=r"$\alpha$ (direction 1)", ylabel=r"$\beta$ (direction 2)", aspect="equal")
    _fig.colorbar(_cs, ax=_axes, shrink=0.8, label="training loss")
    _rows = [f"| {k} | {train_loss(m):.4f} | {test_error(m):.3f} | {sharpness(m):.2f} | "
             f"{surfaces[k][[0, -1, 10, 10], [10, 10, 0, -1]].mean():.2f} |" for k, m in _models.items()]
    mo.vstack([_fig, mo.md("| solution | train loss | test error | sharpness | mean loss at the four edge midpoints |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (surfaces,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the rescaled network is the same function with a top Hessian eigenvalue
    many times larger. so is it a "sharp minimum"? which part of the computation made the
    eigenvalue change, and why does scaling every filter's direction to that filter's own norm
    make the picture blind to the rescaling? what
    does this mean for any claim of the form "flatness predicts generalization"?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: is there a hill between two solutions?

    train two networks and walk the straight line between their weights with your
    `linear_interpolation`, measuring test error along the way. in the first pair, both start
    from the **same** initialization, are trained together for $k$ steps, then continue for the
    remaining steps with different minibatch orders (Frankle et al.'s "spawning"). in the second
    pair, they start from **different** initializations. all runs use batch 64, 2,000 steps
    in total. the slider sets $k$.
    """)
    return


@app.cell
def _():
    spawn = mo.ui.slider(0, 500, step=100, value=0, label="shared steps before the two runs split, k")
    p3 = ui.predict(
        "two networks reach the same test error from the same initialization but see their minibatches in a "
        "different order. is the midpoint of their weights a good network? and for two different initializations?"
    )
    mo.vstack([p3, spawn])
    return p3, spawn


@app.cell
def _(p3, spawn, test_error, train, train_loss):
    ui.gate(p3)
    _t0 = time.perf_counter()
    _k = spawn.value
    lmc = {}
    for _trial in (0, 1, 2):
        _base = train(make_net(_trial), 64, _k, seed=100 + _trial)
        _a = train(copy.deepcopy(_base), 64, 2000 - _k, seed=1 + 10 * _trial)
        _b = train(copy.deepcopy(_base), 64, 2000 - _k, seed=2 + 10 * _trial)
        lmc[("same init", _trial)] = ex.linear_interpolation(_a, _a.state_dict(), _b.state_dict(), test_error,
                                                             n_points=11)
        lmc[("same init, train loss", _trial)] = ex.linear_interpolation(_a, _a.state_dict(), _b.state_dict(),
                                                                         train_loss, n_points=11)
        _c = train(make_net(10 + _trial), 64, 2000, seed=1)
        _e = train(make_net(20 + _trial), 64, 2000, seed=2)
        lmc[("different init", _trial)] = ex.linear_interpolation(_c, _c.state_dict(), _e.state_dict(), test_error,
                                                                  n_points=11)
        lmc[("different init, train loss", _trial)] = ex.linear_interpolation(_c, _c.state_dict(), _e.state_dict(),
                                                                              train_loss, n_points=11)
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for (_kind, _trial), (_al, _L) in lmc.items():
        _ax = _axes[1] if "train loss" in _kind else _axes[0]
        _ax.plot(_al, _L, ("o-" if _kind.startswith("same") else "s--"), ms=3, color="C0" if _kind.startswith("same") else "C3",
                 label=_kind.replace(", train loss", "") if _trial == 0 else None)
    _axes[0].set(xlabel=r"$\alpha$ (0 = network A, 1 = network B)", ylabel="test error")
    _axes[1].set(xlabel=r"$\alpha$", ylabel="training loss", yscale="log")
    _axes[0].legend(fontsize=8)
    _rows = [f"| {kind} | {trial} | {L[0]:.3f} / {L[-1]:.3f} | {L[5]:.3f} | {L.max():.3f} | {ex.loss_barrier(L):.3f} |"
             for (kind, trial), (_, L) in lmc.items() if "train loss" not in kind]
    mo.vstack([_fig, mo.md(f"k = {_k}\n\n| pair | trial | end test errors | midpoint test error | worst on the path | barrier (test error) |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows) + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return (lmc,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** from the same initialization the two networks are *linearly connected*:
    every point on the line between them is as good as the ends, even with $k = 0$. from
    different initializations the midpoint is worse than either end. hidden units have no fixed
    order: what happens to the function if you average unit 3 of one network with unit 3 of
    another network that learned a different feature in that slot? why doesn't that happen when
    both start from the same initialization? (Ainsworth et al. 2023 permute one network's units
    to match the other's, and the barrier mostly disappears.)
    """)
    return


if __name__ == "__main__":
    app.run()
