import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # approximation theory · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours. everything here is a plain `nn.Sequential` of `Linear` and `ReLU` layers whose
    weights you set by hand or train.

    | stage | you build |
    |---|---|
    | 1 | `relu_interpolant`: a one-hidden-layer ReLU network through any set of points |
    | 2 | `sawtooth_net`: a deep, width-2 network with $2^k$ linear pieces from $6k + 1$ weights |
    | 3 | `count_linear_pieces`: how many linear pieces a 1-D network really has |
    | 4 | `width_for_budget`: the width that gives an MLP of a given depth a given parameter count |
    | 5 | `fit_regressor`: the training loop for the depth-separation experiments |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def relu_interpolant(knots_x, knots_y):
    """stage 1: nn.Sequential(Linear(1, n-1), ReLU(), Linear(n-1, 1)) equal to the piecewise-linear interpolant of n >= 2 knots (x strictly increasing)."""
    x = np.asarray(knots_x, dtype=float)
    y = np.asarray(knots_y, dtype=float)
    if x.ndim != 1 or len(x) < 2 or len(x) != len(y) or np.any(np.diff(x) <= 0):
        raise ValueError("need n >= 2 knots with strictly increasing x and matching y")
    slopes = np.diff(y) / np.diff(x)
    coef = np.concatenate([[slopes[0]], np.diff(slopes)])  # change of slope at each knot
    m = len(x) - 1
    net = nn.Sequential(nn.Linear(1, m), nn.ReLU(), nn.Linear(m, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.ones(m, 1))
        net[0].bias.copy_(torch.as_tensor(-x[:-1]))
        net[2].weight.copy_(torch.as_tensor(coef).reshape(1, m))
        net[2].bias.fill_(float(y[0]))
    return net


@app.cell(hide_code=True)
def _():
    checks.run(1, relu_interpolant=relu_interpolant)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sawtooth_net(k):
    """stage 2: nn.Sequential with k hidden ReLU layers of width 2 computing the tent map applied k times on [0, 1]; 6k + 1 parameters."""
    if k < 1:
        raise ValueError("k must be at least 1")
    layers = [nn.Linear(1, 2), nn.ReLU()]
    for _ in range(k - 1):
        layers += [nn.Linear(2, 2), nn.ReLU()]
    layers.append(nn.Linear(2, 1))
    net = nn.Sequential(*layers)
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0], [1.0]]))
        net[0].bias.copy_(torch.tensor([0.0, -0.5]))
        for i in range(1, k):  # pre-activations of layer i+1 from tent(previous) = 2 h1 - 4 h2
            net[2 * i].weight.copy_(torch.tensor([[2.0, -4.0], [2.0, -4.0]]))
            net[2 * i].bias.copy_(torch.tensor([0.0, -0.5]))
        net[-1].weight.copy_(torch.tensor([[2.0, -4.0]]))
        net[-1].bias.zero_()
    return net


@app.cell(hide_code=True)
def _():
    checks.run(2, sawtooth_net=sawtooth_net)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def count_linear_pieces(f, lo=0.0, hi=1.0, *, n_grid=16385, tol=1e-4):
    """stage 3: 1 + number of maximal runs of grid points where |second difference| > tol * max |first difference|, for f on a float64 linspace(lo, hi, n_grid) (modules run on a float64 copy)."""
    with torch.no_grad():
        x = torch.linspace(lo, hi, n_grid, dtype=torch.float64).reshape(-1, 1)
        if isinstance(f, nn.Module):
            f = copy.deepcopy(f).double()
        v = torch.as_tensor(f(x)).reshape(-1).double().numpy()
    d1 = np.diff(v)
    d2 = np.diff(d1)
    scale = np.max(np.abs(d1))
    if scale == 0:
        return 1
    bent = np.abs(d2) > tol * scale
    runs = int(np.sum(bent[1:] & ~bent[:-1]) + (1 if bent[0] else 0))
    return runs + 1


@app.cell(hide_code=True)
def _():
    checks.run(3, count_linear_pieces=count_linear_pieces, sawtooth_net=sawtooth_net, relu_interpolant=relu_interpolant)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def mlp_param_count(width, depth, *, in_dim=1, out_dim=1):
    """stage 4: parameters of `depth` hidden layers of `width` (Linear + ReLU) followed by a Linear output."""
    return (in_dim + 1) * width + (depth - 1) * (width + 1) * width + (width + 1) * out_dim


@app.function
def width_for_budget(budget, depth, *, in_dim=1, out_dim=1):
    """stage 4: the width w >= 1 whose mlp_param_count is closest to budget (ties -> the smaller w)."""
    best_w, best_gap = 1, abs(mlp_param_count(1, depth, in_dim=in_dim, out_dim=out_dim) - budget)
    w = 1
    while True:
        w += 1
        p = mlp_param_count(w, depth, in_dim=in_dim, out_dim=out_dim)
        if abs(p - budget) < best_gap:
            best_w, best_gap = w, abs(p - budget)
        if p > budget:
            return best_w


@app.cell(hide_code=True)
def _():
    checks.run(4, width_for_budget=width_for_budget, mlp_param_count=mlp_param_count)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def fit_regressor(model, x, y, *, steps=2000, lr=3e-3, batch_size=256, seed=0):
    """stage 5: Adam on minibatch MSE, lr cosine-decayed lr * (1 + cos(pi t / steps)) / 2 at step t = 0..steps-1, batches drawn with replacement from a generator seeded with seed; train mode, then eval; returns the steps losses."""
    x = torch.as_tensor(x)
    y = torch.as_tensor(y).reshape(len(x), -1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    bs = min(batch_size, len(x))
    losses = []
    model.train()
    for t in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * t / steps))
        idx = torch.randint(0, len(x), (bs,), generator=g)
        loss = ((model(x[idx]).reshape(bs, -1) - y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


@app.cell(hide_code=True)
def _():
    checks.run(5, fit_regressor=fit_regressor)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it pits deep networks
    against shallow ones at equal parameter counts, on a function depth makes easy to *represent*
    and on one it makes easy to *learn*.
    """)
    return


if __name__ == "__main__":
    app.run()
