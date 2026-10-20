import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
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

    torch.set_num_threads(min(4, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)


@app.class_definition
class Field(nn.Module):
    """an MLP vector field f(t, y) = MLP([y, t]) on R^dim."""

    def __init__(self, dim=2, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(dim + 1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, dim))

    def forward(self, t, y):
        return self.net(torch.cat([y, t.to(y.dtype).expand(y.shape[0], 1)], dim=1))


@app.class_definition
class LinearField(nn.Module):
    """f(t, y) = A y, with A a learnable matrix."""

    def __init__(self, A):
        super().__init__()
        self.A = nn.Parameter(A.clone())

    def forward(self, t, y):
        return y @ self.A.T


@app.function
def saved_bytes(fn):
    """bytes of tensors autograd saves for the backward pass while fn() runs."""
    total = [0]

    def pack(x):
        total[0] += x.numel() * x.element_size()
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        out = fn()
    return total[0], out


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # neural ODEs · lab

    every experiment here uses **your** `odeint`, `odeint_adjoint` and `NeuralODE`. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare with `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what does the adjoint save, and what does it cost?

    a neural ODE with an MLP vector field (2 → 64 → 64 → 2, tanh) on a batch of 256 points,
    integrated from $t = 0$ to $1$ with RK4 in 10 to 500 steps. for backprop through the solver
    and for your adjoint we measure the bytes autograd stores for the backward pass (with
    saved-tensor hooks) and the time of one forward + backward pass (best of 3).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with 500 steps, how many times more memory does backprop through the solver need than the adjoint? and how "
        "many times slower is the adjoint?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _field = Field()
    _y0 = torch.randn(256, 2, generator=torch.Generator().manual_seed(0))
    _t = torch.tensor([0.0, 1.0])
    _steps = (10, 20, 50, 100, 200, 500)
    _mem, _time = {"backprop": [], "adjoint": []}, {"backprop": [], "adjoint": []}
    for _n in _steps:
        for _name, _solve in (("backprop", ex.odeint), ("adjoint", ex.odeint_adjoint)):
            _b, _ = saved_bytes(lambda: _solve(_field, _y0, _t, step_size=1.0 / _n))
            _best = float("inf")
            for _ in range(3):
                _t0 = time.perf_counter()
                (_solve(_field, _y0, _t, step_size=1.0 / _n)[-1] ** 2).sum().backward()
                _best = min(_best, time.perf_counter() - _t0)
            _mem[_name].append(_b / 1e6)
            _time[_name].append(_best * 1e3)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for _name in _mem:
        _a1.loglog(_steps, _mem[_name], "o-", label=_name)
        _a2.loglog(_steps, _time[_name], "o-", label=_name)
    _a1.set(xlabel="RK4 steps", ylabel="MB saved for backward", title="memory")
    _a2.set(xlabel="RK4 steps", ylabel="ms per forward + backward", title="time")
    _a1.legend()
    _a2.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| steps | " + " | ".join(map(str, _steps)) + " |\n|---|" + "---|" * len(_steps) + "\n"
        + "\n".join(f"| {k} MB | " + " | ".join(f"{v:.3g}" for v in vs) + " |" for k, vs in _mem.items()) + "\n"
        + "\n".join(f"| {k} ms | " + " | ".join(f"{v:.3g}" for v in vs) + " |" for k, vs in _time.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does backprop through the solver store per RK4 step, and why does the
    adjoint store nothing per step? count the evaluations of $f$ in each method's backward pass:
    does the time ratio match? when is the trade worth it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: is the adjoint gradient the right gradient?

    a linear system $dy/dt = Ay$ whose matrix $A$ has two decay rates: a slow one (0.1) and a
    fast one ($k$), in rotated coordinates so the two modes mix. we integrate 64 random starting
    points from $t = 0$ to $2$ with RK4 (step 0.05) and compute $\partial L/\partial A$ for
    $L = \sum \|y(2)\|^2$ four ways: backprop through the solver and the adjoint, each in float32
    and float64. the reference is float64 backprop with a ten times smaller step. we plot the
    relative error of each against $k$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the adjoint is exact in the limit of small steps. as the fast decay rate k grows from 0.5 to 20, how does the "
        "float32 adjoint's gradient error compare with float32 backprop's?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _ks = (0.5, 2.0, 5.0, 10.0, 15.0, 20.0)
    _c, _s = np.cos(0.6), np.sin(0.6)
    _Q = torch.tensor([[_c, -_s], [_s, _c]], dtype=torch.float64)
    _y0 = torch.randn(64, 2, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    _t = torch.tensor([0.0, 2.0], dtype=torch.float64)

    def _grad(A, dtype, adjoint, step):
        _f = LinearField(A.to(dtype))
        _solve = ex.odeint_adjoint if adjoint else ex.odeint
        ((_solve(_f, _y0.to(dtype), _t.to(dtype), step_size=step)[-1] ** 2).sum()).backward()
        return _f.A.grad.double()

    _res = {"backprop float32": [], "adjoint float32": [], "backprop float64": [], "adjoint float64": []}
    for _k in _ks:
        _A = _Q @ torch.diag(torch.tensor([-_k, -0.1], dtype=torch.float64)) @ _Q.T
        _ref = _grad(_A, torch.float64, False, 0.005)
        for _name in _res:
            _g = _grad(_A, torch.float32 if "32" in _name else torch.float64, "adjoint" in _name, 0.05)
            _res[_name].append(float((_g - _ref).norm() / _ref.norm()))
    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    for (_name, _e), _st in zip(_res.items(), ("o-", "s-", "o--", "s--")):
        _ax.semilogy(_ks, np.maximum(_e, 1e-17), _st, label=_name)
    _ax.set(xlabel="fast decay rate k", ylabel="relative error of dL/dA", title="gradient accuracy")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| k | " + " | ".join(map(str, _ks)) + " |\n|---|" + "---|" * len(_ks) + "\n"
                           + "\n".join(f"| {k} | " + " | ".join(f"{v:.1e}" for v in vs) + " |" for k, vs in _res.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the adjoint's backward pass recomputes $y(t)$ by integrating the ODE
    *backwards* from $y(2)$. running a decaying mode backwards in time makes it grow like
    $e^{kt}$: what happens to a rounding error of $10^{-7}$ in the fast direction after
    integrating back over 2 time units at $k = 10$? why does backprop not have this problem, and
    what does float64 buy? which of $\partial L/\partial y_0$ and $\partial L/\partial A$ is
    affected, and why (look at what the vector-Jacobian products need)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what a flow can't do

    classify two concentric circles (`mlfs.datasets.make_circles`): a `NeuralODE` maps each
    point through a learned flow from $t = 0$ to $1$ (RK4, 20 steps), then a linear classifier
    reads off the class. the plain model flows in the 2-D plane. the **augmented** one appends
    one zero coordinate and flows in 3-D (Dupont et al., 2019). both trained for 400 Adam steps.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "a neural ODE is a very flexible map of the plane. what test accuracy does the 2-D model reach on concentric "
        "circles? the 3-D one? where will the 2-D model's errors be?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X, _y = datasets.make_circles(600, noise=0.05, factor=0.4, rng=0)
    _X, _y = torch.tensor(_X, dtype=torch.float32), torch.tensor(_y)
    _Xt, _yt = datasets.make_circles(600, noise=0.05, factor=0.4, rng=1)
    _Xt, _yt = torch.tensor(_Xt, dtype=torch.float32), torch.tensor(_yt)

    class _Classifier(nn.Module):
        def __init__(self, extra):
            super().__init__()
            self.extra = extra
            self.node = ex.NeuralODE(Field(dim=2 + extra, hidden=32), n_steps=20)
            self.head = nn.Linear(2 + extra, 2)

        def lift(self, x):
            return torch.cat([x, x.new_zeros(len(x), self.extra)], dim=1)

        def forward(self, x):
            return self.head(self.node(self.lift(x)))

    _rows, _flows = [], {}
    for _extra in (0, 1):
        _accs = []
        for _seed in range(3):
            torch.manual_seed(_seed)
            _m = _Classifier(_extra)
            _opt = torch.optim.Adam(_m.parameters(), lr=1e-2)
            for _ in range(400):
                _loss = F.cross_entropy(_m(_X), _y)
                _opt.zero_grad()
                _loss.backward()
                _opt.step()
            with torch.no_grad():
                _accs.append(float((_m(_Xt).argmax(1) == _yt).float().mean()))
                if _seed == 0:
                    _flows[_extra] = (_m.node(_m.lift(_Xt)).numpy(), (_m(_Xt).argmax(1) != _yt).numpy())
        _rows.append(f"| {'2-D (plain)' if _extra == 0 else '3-D (augmented)'} | {np.mean(_accs):.3f} | "
                     + ", ".join(f"{a:.3f}" for a in _accs) + " |")
    _fig = plt.figure(figsize=(10, 3.4))
    _a0 = _fig.add_subplot(1, 3, 1)
    _a0.scatter(_Xt[:, 0], _Xt[:, 1], c=_yt, s=5, cmap="coolwarm")
    _a0.set(title="test points", aspect="equal")
    _a1 = _fig.add_subplot(1, 3, 2)
    _z2, _wrong = _flows[0]
    _a1.scatter(_z2[:, 0], _z2[:, 1], c=_yt, s=5, cmap="coolwarm")
    _a1.scatter(_z2[_wrong, 0], _z2[_wrong, 1], s=40, facecolors="none", edgecolors="k", label="misclassified")
    _a1.set(title="after the 2-D flow", aspect="equal")
    _a1.legend(fontsize=7, loc="center left")
    _z3 = _flows[1][0]
    _a2 = _fig.add_subplot(1, 3, 3, projection="3d")
    _a2.scatter(_z3[:, 0], _z3[:, 1], _z3[:, 2], c=_yt, s=4, cmap="coolwarm")
    _a2.set(title="after the 3-D flow")
    _fig.tight_layout()
    mo.vstack([mo.md("| model | mean test accuracy | seeds 0, 1, 2 |\n|---|---|---|\n" + "\n".join(_rows)), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the exact flow of an ODE is a continuous, invertible map whose trajectories
    never cross. what does that imply for a point inside the inner circle that must end up on
    the far side of the outer ring? look at where the 2-D flow sends the outer ring, and at
    which points it gets wrong: how did it get so close to 100% without breaking the rule?
    what does the extra dimension allow instead?
    """)
    return


if __name__ == "__main__":
    app.run()
