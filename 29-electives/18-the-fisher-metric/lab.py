import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn
    from torch.nn.utils import vector_to_parameters

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)  # tiny models: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the fisher metric · lab

    every experiment here runs **your** Fisher code from `exercise.py`. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the whole lab takes under a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how big is a step?

    a small classifier (2 inputs → 16 tanh units → 3 classes, $P = 99$ parameters) trained
    for a while on three overlapping blobs. from its trained parameters $\theta$ we take 200
    steps $\theta + t u$, each with a random direction $u$ of Euclidean length 1 and the
    **same** size $t = 0.05$, and measure how far each one moves the model's predictions: the
    mean KL divergence $\overline{\mathrm{KL}} = \frac1N\sum_n \mathrm{KL}\big(p(\cdot\mid x_n,\theta)\,\|\,p(\cdot\mid x_n,\theta + tu)\big)$
    over 100 inputs. then, along three chosen directions, we grow $t$ from $10^{-3}$ to 3 and
    compare both KL directions with the Fisher quadratic form $\tfrac12 t^2 u^\top F u$,
    where $F$ is your `true_fisher`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "200 parameter steps, all of Euclidean length 0.05. how much does the KL they cause vary from the smallest "
        "to the largest: less than 2x, about 10x, or more than 100x? and is there a direction whose KL is exactly 0?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _X, _y = datasets.make_blobs(300, centers=3, std=2.5, rng=0)
    _X = torch.tensor((_X - _X.mean(0)) / _X.std(0))
    _y = torch.tensor(_y)
    torch.manual_seed(0)
    _net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))
    _opt = torch.optim.Adam(_net.parameters(), lr=1e-2)
    for _ in range(150):
        _opt.zero_grad()
        nn.functional.cross_entropy(_net(_X), _y).backward()
        _opt.step()
    _Xs = _X[:100]
    F1 = ex.true_fisher(_net, _Xs)
    _theta = ex.flat_params(_net)
    _probe = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))
    with torch.no_grad():
        _lp0 = torch.log_softmax(_net(_Xs), -1)

    def _kl(step, reverse=False):
        vector_to_parameters(_theta + step, _probe.parameters())
        with torch.no_grad():
            a, b = _lp0, torch.log_softmax(_probe(_Xs), -1)
            if reverse:
                a, b = b, a
            return float((a.exp() * (a - b)).sum(-1).mean())

    _g = torch.Generator().manual_seed(0)
    _U = torch.randn(200, _theta.numel(), generator=_g)
    _U = _U / _U.norm(dim=1, keepdim=True)
    _t = 0.05
    _kls = np.array([_kl(_t * u) for u in _U])
    _quad = np.array([0.5 * _t**2 * float(u @ F1 @ u) for u in _U])
    _evals, _evecs = torch.linalg.eigh(F1)
    # the null direction: shift every logit by the same amount (last layer rows and bias all move together)
    _null = torch.zeros_like(_theta)
    _null[-3 - 48:-3] = torch.tile(torch.randn(16, generator=_g), (3,))
    _null[-3:] = 1.0
    _null = _null / _null.norm()
    _dirs = {"top eigenvector of F": _evecs[:, -1], "random direction": _U[0], "softmax shift (null)": _null}
    _ts = np.logspace(-3, math.log10(3), 13)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _axes[0].scatter(_quad, _kls, s=10)
    _lim = [min(_quad.min(), _kls.min()) * 0.7, max(_quad.max(), _kls.max()) * 1.4]
    _axes[0].plot(_lim, _lim, "k:", lw=1, label="KL = ½ t² uᵀFu")
    _axes[0].set(xscale="log", yscale="log", xlabel="½ t² uᵀ F u  (your Fisher)", ylabel="measured mean KL",
                 title="200 random steps, all of Euclidean length 0.05")
    _axes[0].legend(fontsize=8)
    _rows = []
    for _i, (_name, _u) in enumerate(_dirs.items()):
        _fw = np.array([_kl(t * _u) for t in _ts])
        _bw = np.array([_kl(t * _u, True) for t in _ts])
        _q = np.array([0.5 * t**2 * float(_u @ F1 @ _u) for t in _ts])
        if _q.max() > 1e-14:
            _axes[1].plot(_ts, _fw, "o-", color=f"C{_i}", ms=4, label=f"{_name}: KL(old‖new)")
            _axes[1].plot(_ts, _bw, "s--", color=f"C{_i}", ms=3, alpha=0.6, label=f"{_name}: KL(new‖old)")
            _axes[1].plot(_ts, _q, ":", color=f"C{_i}", lw=2, alpha=0.8)
        for _tt in (1e-2, 0.3, 3.0):
            _j = int(np.argmin(np.abs(_ts - _tt)))
            _cells = " | ".join("0 (< 1e-15)" if abs(v) < 1e-15 else f"{v:.3g}" for v in (_q[_j], _fw[_j], _bw[_j]))
            _rows.append(f"| {_name} | {_ts[_j]:.2g} | {_cells} |")
    _axes[1].set(xscale="log", yscale="log", xlabel="step size t", ylabel="KL (dotted: ½ t² uᵀFu)",
                 title="growing the step along fixed directions")
    _axes[1].legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"random steps: KL from **{_kls.min():.2g}** to **{_kls.max():.2g}** "
              f"(a factor of **{_kls.max() / _kls.min():.0f}**); KL / quadratic form between "
              f"{(_kls / _quad).min():.3f} and {(_kls / _quad).max():.3f}. "
              f"eigenvalues of F: largest {_evals[-1]:.3g}; {int((_evals < 1e-15).sum())} of {_evals.numel()} are 0 "
              f"to machine precision; the smallest of the rest is {_evals[_evals >= 1e-15].min():.1g}."),
        mo.md("| direction | t | ½t²uᵀFu | KL(old‖new) | KL(new‖old) |\n|---|---|---|---|---|\n" + "\n".join(_rows)),
    ])
    return (F1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the steps were the same size in parameter space. why do their effects on
    the predictions differ by orders of magnitude, and what in $F$ predicts it? why do both
    KL directions agree with each other (and with the quadratic form) for small $t$ and split
    apart for large $t$ (what order is the first term where they differ)? the "softmax shift"
    direction changes the parameters but its KL is zero at every step size, not just small
    ones: what does that say about the model, and about $F$'s rank?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: is the empirical Fisher the Fisher?

    softmax regression on 2-D inputs, in two worlds. **well-specified:** the labels are
    sampled from a softmax-regression teacher, so the model can match the truth.
    **misspecified:** the labels come from a teacher that is confident at moderate $|x|$ and
    a coin toss far from the origin, which no linear model can mimic. for $N$ = 100, 300 and
    1,000 training points (three seeds each) we compare your `empirical_fisher` with your
    `true_fisher`, at a random initialization and at the maximum-likelihood fit (L-BFGS to
    convergence), by the relative difference $\lVert\hat F_{\text{emp}} - F\rVert_F/\lVert F\rVert_F$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "at the maximum-likelihood fit with 1,000 points, how close is the empirical Fisher to the true Fisher "
        "(relative difference) in each world? does more data close the gap in both?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _W = torch.tensor([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])

    def _labels(world, X, g):
        logits = X @ _W.T
        if world == "misspecified":
            logits = 3 * logits / (0.3 + (X**2).sum(1, keepdim=True))
        return torch.multinomial(torch.softmax(logits, -1), 1, generator=g).squeeze(1)

    def _fit(m, X, y):
        opt = torch.optim.LBFGS(m.parameters(), max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-10)

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(m(X), y)
            loss.backward()
            return loss

        opt.step(closure)

    def _rel(m, X, y):
        F = ex.true_fisher(m, X)
        return float((ex.empirical_fisher(m, X, y) - F).norm() / F.norm())

    _Ns = [100, 300, 1000]
    _res = {}
    for _world in ["well-specified", "misspecified"]:
        for _N in _Ns:
            _init, _mle = [], []
            for _seed in range(3):
                _g = torch.Generator().manual_seed(_seed)
                _X = torch.randn(_N, 2, generator=_g)
                _yy = _labels(_world, _X, _g)
                torch.manual_seed(_seed)
                _m = nn.Linear(2, 3)
                _init.append(_rel(_m, _X, _yy))
                _fit(_m, _X, _yy)
                _mle.append(_rel(_m, _X, _yy))
            _res[(_world, _N)] = (np.array(_init), np.array(_mle))
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _i, _world in enumerate(["well-specified", "misspecified"]):
        _med = [np.median(_res[(_world, n)][1]) for n in _Ns]
        for n in _Ns:
            _ax.scatter([n] * 3, _res[(_world, n)][1], color=f"C{_i}", s=14, alpha=0.6)
        _ax.plot(_Ns, _med, "o-", color=f"C{_i}", label=f"{_world}, at the MLE (median of 3)")
    _ax.plot(_Ns, [np.median(_res[("well-specified", n)][0]) for n in _Ns], "k:", label="well-specified, at init")
    _ax.set(xscale="log", xlabel="training points N", ylabel="‖EF − F‖ / ‖F‖", ylim=(0, None))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {w} | {n} | {np.median(_res[(w, n)][0]):.2f} | {np.median(_res[(w, n)][1]):.2f} | "
             f"{', '.join(f'{v:.2f}' for v in _res[(w, n)][1])} |" for w in ["well-specified", "misspecified"]
             for n in _Ns]
    mo.vstack([_fig, mo.md("| world | N | at init (median) | at the MLE (median) | at the MLE, per seed |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the empirical Fisher plugs the observed label into the score where the
    true Fisher averages over the model's own predictions. at initialization the two are far
    apart in both worlds: why? at the fit, the gap shrinks with $N$ in one world and stays
    put in the other. which two conditions must hold for $\mathbb E[\hat F_{\text{emp}}] = F$,
    and which one fails in the misspecified world? Adam's second-moment estimate is often
    described as a diagonal empirical Fisher, though it squares the minibatch-mean gradient
    rather than averaging per-example squares: what does this experiment suggest about
    calling Adam "approximately natural gradient"?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the length of a coin

    the Bernoulli family is a one-dimensional statistical manifold: a curve. we measure the
    path from Bernoulli($p_0$) to Bernoulli($p_1$) with your `fisher_rao_length`, in three
    coordinate systems: the probability $p$ (Fisher $1/(p(1-p))$), the logit
    $t = \log\frac{p}{1-p}$ (Fisher $p(1-p)$), and $s = \sqrt p$ (Fisher $4/(1 - s^2)$, from
    your `pullback_fisher`), next to the plain Euclidean length of each description. move the
    sliders.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the logit runs from -infinity to +infinity. what is the Fisher-Rao length of the WHOLE Bernoulli family, "
        "from p = 0 to p = 1: infinite, or a finite number (which)?"
    )
    p3
    return (p3,)


@app.cell
def _():
    s_p0 = mo.ui.slider(0.001, 0.5, value=0.01, step=0.001, label="p₀")
    s_p1 = mo.ui.slider(0.5, 0.999, value=0.99, step=0.001, label="p₁")
    mo.hstack([s_p0, s_p1])
    return s_p0, s_p1


@app.cell
def _(p3, s_p0, s_p1):
    ui.gate(p3)
    _p0, _p1 = float(s_p0.value), float(s_p1.value)
    # one curve, sampled evenly in the logit, described in three coordinate systems
    _tt = torch.linspace(math.log(_p0 / (1 - _p0)), math.log(_p1 / (1 - _p1)), 2001).reshape(-1, 1)
    _pp = torch.sigmoid(_tt)
    _paths = {
        "p": (_pp, lambda q: (1 / (q[0] * (1 - q[0]))).reshape(1, 1)),
        "logit t": (_tt, lambda t: (torch.sigmoid(t[0]) * (1 - torch.sigmoid(t[0]))).reshape(1, 1)),
        "√p": (torch.sqrt(_pp), lambda s: ex.pullback_fisher((1 / (s[0] ** 2 * (1 - s[0] ** 2))).reshape(1, 1),
                                                              (2 * s[0]).reshape(1, 1))),
    }
    _exact = 2 * abs(math.asin(math.sqrt(_p1)) - math.asin(math.sqrt(_p0)))
    _rows = []
    for _name, (_path, _fn) in _paths.items():
        _L = ex.fisher_rao_length(_fn, _path)
        _E = float((_path[-1] - _path[0]).abs())
        _rows.append(f"| {_name} | {_E:.4f} | {_L:.4f} |")
    _full = ex.fisher_rao_length(_paths["logit t"][1], torch.linspace(-21.0, 21.0, 4001).reshape(-1, 1))
    mo.md(f"from p₀ = {_p0} to p₁ = {_p1}; closed form 2|asin√p₁ − asin√p₀| = **{_exact:.4f}**\n\n"
          "| coordinates | Euclidean length | your Fisher-Rao length |\n|---|---|---|\n" + "\n".join(_rows)
          + f"\n\nthe whole family, logit from −21 to 21 (p from 10⁻⁹ to 1 − 10⁻⁹): **{_full:.4f}** (π = {math.pi:.4f})")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do the three Euclidean lengths disagree while the three Fisher-Rao
    lengths agree? the logit coordinate stretches to infinity near $p = 0$ and $p = 1$; why
    is the Fisher-Rao length of the whole family still finite? (hint: in $s = \sqrt p$
    coordinates, with $s_2 = \sqrt{1 - p}$, the family is a quarter of a circle of radius 2.)
    which pair of coins is "further apart": $(0.5, 0.6)$ or $(0.9, 0.99)$?
    """)
    return


if __name__ == "__main__":
    app.run()
