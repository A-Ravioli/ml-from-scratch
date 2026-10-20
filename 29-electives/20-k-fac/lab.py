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
    from mlfs.torchnn import MLP  # yours, from chapter 15

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # steadier timings on a shared machine


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # K-FAC · lab

    every experiment here runs **your** K-FAC from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. experiment 2 is the information-geometry block's
    **project**: K-FAC against a tuned Adam on chapter 15's MNIST network. the whole lab takes
    one to two minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how good is the Kronecker approximation?

    a small MLP (6 inputs → 12 → 12 tanh → 3 classes, 279 parameters) on three noisy blobs,
    trained briefly with Adam. on 150 examples we compute the exact Fisher matrix (by brute
    force, per example and per class), its block-diagonal part (one block per layer), and
    your K-FAC approximation $G_\ell\otimes A_\ell$ of each block. then we ask what matters for
    optimization: for a damping $\lambda$, the natural-gradient step $-(F + \lambda I)^{-1}g$
    achieves the largest decrease of the quadratic model
    $g^\top d + \tfrac12d^\top(F + \lambda I)d$; what fraction of that best decrease does each
    cheaper direction achieve (each with its own best step length)?
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the Kronecker blocks differ from the exact Fisher blocks by about half their size (relative Frobenius "
        "error). what fraction of the natural gradient's decrease do you expect K-FAC's direction to achieve, "
        "compared with the plain gradient's?"
    )
    p1
    return (p1,)


@app.cell
def _():
    state1 = mo.ui.dropdown({"early (20 Adam steps)": 20, "trained (150 Adam steps)": 150},
                            value="early (20 Adam steps)", label="network")
    state1
    return (state1,)


@app.cell
def _(p1, state1):
    ui.gate(p1)
    torch.set_default_dtype(torch.float64)
    _X, _y = datasets.make_blobs(300, centers=3, std=2.0, rng=0)
    _X = torch.tensor((_X - _X.mean(0)) / _X.std(0))
    _X = torch.cat([_X, torch.randn(300, 4, generator=torch.Generator().manual_seed(0))], 1)
    _y = torch.tensor(_y)
    torch.manual_seed(0)
    _net = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 12), nn.Tanh(), nn.Linear(12, 3))
    _opt = torch.optim.Adam(_net.parameters(), lr=1e-2)
    for _ in range(int(state1.value)):
        _opt.zero_grad()
        nn.functional.cross_entropy(_net(_X), _y).backward()
        _opt.step()
    _Xs, _ys = _X[:150], _y[:150]
    _layers = [m for m in _net.modules() if isinstance(m, nn.Linear)]

    def _layer_vec(grads):  # per-layer row-major [W | b], concatenated
        return torch.cat([torch.cat([gw, gb.unsqueeze(1)], 1).reshape(-1) for gw, gb in zip(grads[::2], grads[1::2])])

    _params = [p for m in _layers for p in (m.weight, m.bias)]
    _P = sum(p.numel() for p in _params)
    _F = torch.zeros(_P, _P)
    for _n in range(150):  # exact Fisher: per example, per class
        _lp = torch.log_softmax(_net(_Xs[_n:_n + 1]), -1)[0]
        for _c in range(3):
            _v = _layer_vec(torch.autograd.grad(_lp[_c], _params, retain_graph=True))
            _F += _lp[_c].exp().item() * torch.outer(_v, _v) / 150
    _factors = ex.kfac_factors(_net, _Xs)
    _bounds, _o = [], 0
    for _m in _layers:
        _k = _m.weight.shape[0] * (_m.weight.shape[1] + 1)
        _bounds.append((_o, _o + _k))
        _o += _k
    _K = torch.zeros_like(_F)
    _B = torch.zeros_like(_F)
    _errs = []
    for (_a, _b), (_A, _G) in zip(_bounds, _factors):
        _K[_a:_b, _a:_b] = torch.kron(_G, _A)
        _B[_a:_b, _a:_b] = _F[_a:_b, _a:_b]
        _errs.append(float((_K[_a:_b, _a:_b] - _F[_a:_b, _a:_b]).norm() / _F[_a:_b, _a:_b].norm()))
    _net.zero_grad()
    nn.functional.cross_entropy(_net(_Xs), _ys).backward()
    _g = _layer_vec([p.grad for p in _params])
    _rows = []
    _lams = [1e-4, 1e-3, 1e-2, 1e-1]
    _fracs = {k: [] for k in ["gradient", "K-FAC (yours, factored damping)", "block-diagonal exact Fisher"]}
    for _lam in _lams:
        _M = _F + _lam * torch.eye(_P)
        _ng = torch.linalg.solve(_M, _g)

        def _dec(d):
            return float(0.5 * (_g @ d) ** 2 / (d @ _M @ d))

        _kf = []
        for (_a, _b), (_A, _G), _m in zip(_bounds, _factors, _layers):
            _kf.append(ex.kron_precondition(_g[_a:_b].reshape(_m.weight.shape[0], -1),
                                            *ex.damped_inverses(_A, _G, _lam)).reshape(-1))
        _dirs = {"gradient": _g, "K-FAC (yours, factored damping)": torch.cat(_kf),
                 "block-diagonal exact Fisher": torch.linalg.solve(_B + _lam * torch.eye(_P), _g)}
        for _name, _d in _dirs.items():
            _fracs[_name].append(_dec(_d) / _dec(_ng))
    for _name, _v in _fracs.items():
        _rows.append(f"| {_name} | " + " | ".join(f"{x:.2f}" for x in _v) + " |")
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.2))
    _vmax = float(_F.abs().max())
    for _ax, _Mx, _t in [(_axes[0], _F, "|exact Fisher|"), (_axes[1], _K, "|K-FAC (block-diagonal G ⊗ A)|")]:
        _im = _ax.imshow(np.log10(_Mx.abs().numpy() + 1e-8 * _vmax), cmap="magma",
                         vmin=np.log10(_vmax) - 6, vmax=np.log10(_vmax))
        _ax.set_title(_t + ", log₁₀")
        for _a, _b in _bounds[1:]:
            _ax.axhline(_a - 0.5, color="c", lw=0.6)
            _ax.axvline(_a - 0.5, color="c", lw=0.6)
    _fig.colorbar(_im, ax=_axes[1], fraction=0.046, pad=0.03)
    for _i, (_name, _v) in enumerate(_fracs.items()):
        _axes[2].plot(_lams, _v, "o-", color=f"C{_i}", label=_name)
    _axes[2].axhline(1.0, color="k", ls=":", lw=1)
    _axes[2].set(xscale="log", xlabel="damping λ", ylabel="fraction of the natural-gradient decrease", ylim=(0, 1.05))
    _axes[2].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md("relative error ‖G⊗A − F_block‖ / ‖F_block‖ per layer: " + ", ".join(f"{e:.2f}" for e in _errs)
              + f"; off-diagonal (between-layer) blocks hold {float((_F - _B).norm() / _F.norm()):.0%} of ‖F‖."),
        mo.md("| direction | " + " | ".join(f"λ = {lam:g}" for lam in _lams) + " |\n|---|---|---|---|---|\n"
              + "\n".join(_rows)),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the Kronecker blocks are far from the exact blocks in Frobenius norm, and
    K-FAC also throws away the between-layer blocks, yet its direction gets most of the
    natural gradient's decrease and clearly beats the gradient at small damping. what does
    the Frobenius error weigh that the step doesn't care about (which eigen-directions
    dominate each)? why do all three directions converge towards the same value as the
    damping grows? switch to the trained network: does the picture change, and why might
    the gradient catch up with K-FAC there at the largest damping?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2 (project): K-FAC against a tuned Adam on chapter 15's network

    chapter 15's MLP (784 → 256 → 128 → 10, ReLU, 235,146 parameters; your `mlfs.torchnn.MLP`)
    on 20,000 MNIST training images, 3 epochs at batch size 256 (234 steps), float32, one CPU
    thread. 5,000 further training images are held out for validation, and the 10,000 test
    images are used only for the final report.

    **protocol.** tune each optimizer on seed 0 over a small grid, Adam's step size in
    $\{3\cdot10^{-4}, 10^{-3}, 3\cdot10^{-3}, 10^{-2}\}$ and K-FAC's damping in
    $\{10^{-2}, 10^{-1}, 1\}$ times its KL clip in $\{10^{-4}, 10^{-3}, 10^{-2}\}$ (step size 0.3, momentum
    0.9, factors every step with sampled labels, inverses every 20 steps). two selection
    rules, because there are two questions: the config with the **lowest final training loss**
    (which optimizer optimizes faster?) and the config with the **best validation accuracy**
    (which gives the better model?). each selected config then runs with three seeds. we also
    run two variants at the training-loss winner's settings: K-FAC with the **empirical**
    Fisher (`fisher="empirical"`), and a **cheap-curvature** K-FAC that updates its factors
    only every 10 steps and its inverses every 50. the training loss is measured every 6 steps on a fixed set of 2,000
    training images (outside the timed region). offline, a small synthetic stand-in replaces
    MNIST and the numbers mean little.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "to reach the training loss Adam ends with: how many times fewer steps does K-FAC need, and is it faster in "
        "wall-clock time too? which optimizer gives the better test accuracy?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    torch.set_default_dtype(torch.float32)
    _d = datasets.mnist()
    _n_tr = min(20_000, len(_d["X_train"]) - 400)
    _Xall = _d["X_train"].reshape(len(_d["X_train"]), -1).astype(np.float32) / 255
    _mu, _sd = _Xall[:_n_tr].mean(), _Xall[:_n_tr].std()
    _Xall = torch.tensor((_Xall - _mu) / _sd)
    _yall = torch.tensor(_d["y_train"])
    _va = slice(50_000, 55_000) if len(_Xall) >= 55_000 else slice(_n_tr, len(_Xall))
    _Xtr, _ytr, _Xva, _yva = _Xall[:_n_tr], _yall[:_n_tr], _Xall[_va], _yall[_va]
    _Xte = torch.tensor((_d["X_test"].reshape(len(_d["X_test"]), -1).astype(np.float32) / 255 - _mu) / _sd)
    _yte = torch.tensor(_d["y_test"])
    _Xmon, _ymon = _Xtr[:2000], _ytr[:2000]
    _bs, _epochs = 256, 3

    def _run(kind, seed, **kw):
        torch.manual_seed(seed)
        model = MLP(784, (256, 128), 10)
        opt = torch.optim.Adam(model.parameters(), **kw) if kind == "adam" else ex.KFAC(model, seed=seed, **kw)
        g = torch.Generator().manual_seed(seed)
        steps, secs, losses, clock, k = [], [], [], 0.0, 0
        for _ep in range(_epochs):
            perm = torch.randperm(_n_tr, generator=g)
            for i in range(0, _n_tr - _bs + 1, _bs):
                idx = perm[i:i + _bs]
                t0 = time.perf_counter()
                if kind == "adam":
                    opt.zero_grad()
                    nn.functional.cross_entropy(model(_Xtr[idx]), _ytr[idx]).backward()
                    opt.step()
                else:
                    opt.step(_Xtr[idx], _ytr[idx])
                clock += time.perf_counter() - t0
                k += 1
                if k % 6 == 0 or k == 1:
                    with torch.no_grad():
                        losses.append(nn.functional.cross_entropy(model(_Xmon), _ymon).item())
                    steps.append(k)
                    secs.append(clock)
        with torch.no_grad():
            va = (model(_Xva).argmax(1) == _yva).float().mean().item()
            te = (model(_Xte).argmax(1) == _yte).float().mean().item()
        return dict(steps=np.array(steps), secs=np.array(secs), loss=np.array(losses), val=va, test=te,
                    ms=1e3 * clock / k)

    _run("adam", 0, lr=1e-3)  # warm-up (lazy initialisation), not reported
    _adam_grid = {f"Adam lr={lr:g}": ("adam", dict(lr=lr)) for lr in [3e-4, 1e-3, 3e-3, 1e-2]}
    _kfac_grid = {f"K-FAC λ={dm:g}, κ={kl:g}": ("kfac", dict(lr=0.3, damping=dm, kl_clip=kl, inv_every=20))
                  for dm in [1e-2, 1e-1, 1.0] for kl in [1e-4, 1e-3, 1e-2]}
    _tune = {name: _run(k, 0, **kw) for name, (k, kw) in {**_adam_grid, **_kfac_grid}.items()}
    tune2 = "| config (seed 0) | final training loss | validation accuracy | ms per step |\n|---|---|---|---|\n" + "\n".join(
        f"| {n} | {r['loss'][-1]:.4f} | {r['val']:.4f} | {r['ms']:.1f} |" for n, r in _tune.items())
    _picks = {}
    for _family, _grid in [("Adam", _adam_grid), ("K-FAC", _kfac_grid)]:
        _by_loss = min(_grid, key=lambda n: _tune[n]["loss"][-1])
        _by_val = max(_grid, key=lambda n: _tune[n]["val"])
        if _by_loss == _by_val:
            _picks[f"{_family}, best training loss and validation ({_by_loss.split(' ', 1)[1]})"] = _grid[_by_loss]
        else:
            _picks[f"{_family}, best training loss ({_by_loss.split(' ', 1)[1]})"] = _grid[_by_loss]
            _picks[f"{_family}, best validation ({_by_val.split(' ', 1)[1]})"] = _grid[_by_val]
    _kbest = _kfac_grid[min(_kfac_grid, key=lambda n: _tune[n]["loss"][-1])][1]
    _picks["K-FAC, empirical Fisher (training-loss winner's settings)"] = ("kfac", dict(_kbest, fisher="empirical"))
    _picks["K-FAC, cheap curvature (same, factors every 10 steps, inverses every 50)"] = (
        "kfac", dict(_kbest, stat_every=10, inv_every=50))
    _final = {name: [_run(k, s, **kw) for s in range(3)] for name, (k, kw) in _picks.items()}
    runs2 = _final
    return runs2, tune2


@app.cell
def _(runs2, tune2):
    _adam_names = [n for n in runs2 if n.startswith("Adam")]
    _target = float(np.median([np.median([r["loss"][-1] for r in runs2[n]]) for n in _adam_names[:1]]))
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    _rows = []
    for _i, (_name, _rs) in enumerate(runs2.items()):
        for _j, _r in enumerate(_rs):
            _kw = dict(color=f"C{_i}", alpha=0.85, lw=1.2, label=_name if _j == 0 else None)
            _axes[0].plot(_r["steps"], _r["loss"], **_kw)
            _axes[1].plot(_r["secs"], _r["loss"], **_kw)

        def _first(r, key):
            hit = np.flatnonzero(r["loss"] <= _target)
            return r[key][hit[0]] if len(hit) else None

        _st = [_first(r, "steps") for r in _rs]
        _se = [_first(r, "secs") for r in _rs]
        _test = np.array([r["test"] for r in _rs])
        _rows.append(
            f"| {_name} | {np.median([r['loss'][-1] for r in _rs]):.4f} | "
            + ", ".join("—" if v is None else str(int(v)) for v in _st) + " | "
            + ", ".join("—" if v is None else f"{v:.2f}" for v in _se) + " | "
            + f"{np.median([r['ms'] for r in _rs]):.1f} | {_test.mean():.4f} ± {_test.std(ddof=1) / np.sqrt(3):.4f} |")
    _axes[0].axhline(_target, color="k", ls=":", lw=1)
    _axes[1].axhline(_target, color="k", ls=":", lw=1)
    _axes[0].set(yscale="log", xlabel="step", ylabel="training loss (2,000 fixed training images)")
    _axes[1].set(yscale="log", xlabel="wall-clock seconds in optimizer steps (one CPU thread)")
    _axes[0].legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"target: the final training loss of the Adam run tuned for training loss (median of 3 seeds), "
              f"**{_target:.4f}**\n\n| run (3 seeds) | final training loss (median) | steps to target | "
              "seconds to target | ms per step | test accuracy (mean ± s.e.) |\n|---|---|---|---|---|---|\n"
              + "\n".join(_rows)),
        mo.accordion({"tuning runs (seed 0)": mo.md(tune2)}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how many times fewer steps does K-FAC need to reach Adam's final training
    loss, and how much more does each step cost (count the passes: a forward and backward for
    the gradient, a forward and backward with sampled labels for the factors, the outer
    products, and an inverse every 20 steps)? does the step advantage survive in wall-clock
    time? the test accuracies tell a different story from the training losses: which
    optimizer would you ship, and what would you try to close the gap (think about what a
    lower training loss on 20,000 images means without any regularization)? finally, compare
    K-FAC with the empirical Fisher: is it better or worse than the true-Fisher K-FAC, and
    does that match topic 19?
    """)
    return


if __name__ == "__main__":
    app.run()
