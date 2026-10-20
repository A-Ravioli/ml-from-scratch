import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    def _maml_smoke(m):  # raises NotImplementedError if a topic-09 stage used here isn't done yet
        p = m.init_mlp([2, 1], generator=torch.Generator().manual_seed(0))
        b = {"xs": torch.randn(2, 3, 2), "ys": torch.randn(2, 3, 1), "xq": torch.randn(2, 1, 2), "yq": torch.randn(2, 1, 1)}
        m.adapt(p, b["xs"], b["ys"], n_steps=1, first_order=True)
        m.maml_loss(p, b)
        m.train_maml(p, lambda g: b, n_iters=1)

    # your MAML from topic 09 (electives own no mlfs module, so it's loaded directly)
    prev = notebooks.load_topic("09-maml", key="maml", smoke=_maml_smoke)
    maml = prev.module
    torch.set_num_threads(2)

    D = 5  # input dimension
    N_POINTS = 15  # sequence length: predictions with 0..14 in-context examples
    TF = dict(d_model=32, n_layers=3, n_heads=4)  # the transformer
    TF_STEPS, TF_BATCH, TF_LR = 2500, 64, 2e-3
    N_TEST = 2000  # test tasks per scenario


@app.function
def test_tasks(seed, noise_std=0.0, x_scale=1.0):
    """N_TEST fresh tasks of the full sequence length."""
    return ex.sample_linear_tasks(N_TEST, N_POINTS, D, noise_std=noise_std, x_scale=x_scale,
                                  generator=torch.Generator().manual_seed(seed))


@app.function
def sq_err(preds, ys):
    """per-task, per-position squared errors as a NumPy array (B, n)."""
    return ((preds.double() - ys.double()) ** 2).numpy()


@app.function
def mean_se(a):
    """(mean over tasks, standard error) of a (B, n) array, per position."""
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))


@app.function
def err_curve(preds, ys):
    """(mean squared error per position, its standard error) as NumPy arrays."""
    return mean_se(sq_err(preds, ys))


@app.function
def fmt(m, se):
    """'mean ± se', or 'diverges' for non-finite or huge errors."""
    return "diverges" if not np.isfinite(m) or m > 1e4 else f"{m:.3f} ± {se:.3f}"


@app.function
def plot_curves(ax, rows, title):
    """error-vs-context curves on a log axis; values below 1e-3 (exact predictions) are drawn at 1e-3."""
    colors = {"predict 0": "0.6", "least squares": "C0", "transformer": "C1", "MAML (tuned per length)": "C2",
              "one GD step (best η per length)": "C4", "ridge, λ = σ² (oracle)": "k"}
    for name, (m, se) in rows.items():
        m = np.clip(np.nan_to_num(m, nan=1e4, posinf=1e4), 1e-3, 1e4)
        ax.errorbar(range(len(m)), m, yerr=np.minimum(se, 0.99 * m), fmt="o-", ms=3, capsize=2, label=name,
                    color=colors.get(name))
    ax.axvline(D, color="gray", ls=":", lw=1)
    ax.set(xlabel="in-context examples", ylabel="test MSE (floor 1e-3 = exact)", yscale="log", title=title)
    ax.legend(fontsize=7)


@app.function
def maml_predictions(params, xs, ys, *, inner_lr, n_steps):
    """(B, n): prediction i adapts the linear model on the first i pairs (n_steps of maml.adapt), then predicts x_i."""
    B, n, _ = xs.shape
    out = torch.zeros(B, n)
    p = [q.detach() for q in params]
    for i in range(n):
        fast = (maml.adapt(p, xs[:, :i], ys[:, :i, None], inner_lr=inner_lr, n_steps=n_steps, first_order=True)
                if i > 0 else [q.expand(B, *q.shape) for q in p])
        with torch.no_grad():
            out[:, i] = maml.mlp_forward(fast, xs[:, i:i + 1])[:, 0, 0]
    return out


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # in-context learning as meta-learning · lab

    every experiment here runs **your** code: this topic's tasks, baselines, linear
    self-attention and transformer, and in the project your `train_maml` and `adapt` from
    topic 09. the task family is Garg et al.'s: $y = w^\top x$ with $x, w \sim \mathcal N(0, I_{D})$,
    $D = {D}$, and sequences of {N_POINTS} pairs, so each model predicts with 0 to {N_POINTS - 1}
    in-context examples. errors are mean squared errors on {N_TEST:,} fresh test tasks; predicting 0
    always scores $D = {D}$. the lab takes about two and a half minutes on a laptop CPU, most of it
    training the transformer once.
    """)
    return


@app.cell
def _():
    prev.callout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what does one layer of attention learn?

    a single layer of **linear** self-attention, `lsa_predict` with two free $(D+1) \times (D+1)$
    matrices, trained with Adam on 10-example contexts to predict the query's $y$. idea 2 showed
    one choice of matrices that makes it exactly one step of gradient descent. nobody tells the
    training about that choice. what can be compared is the matrix through which the context's
    $y_j x_j$ reach the query, $cA + ub^\top$, where $c$ is $W_{PV}$'s $(y, y)$ entry, $u$ its
    $(y, x)$ row, $A$ the $(x, x)$ block of $W_{KQ}$ and $b$ its $(y, x)$ row (on paper 2).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "after training, will the attention weights look like the gradient-descent construction (an identity block "
        "times a step size), or like something else? which step size?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _n = 10
    _g = torch.Generator().manual_seed(0)
    W_kq = (0.1 * torch.randn(D + 1, D + 1, generator=_g)).requires_grad_()
    W_pv = (0.1 * torch.randn(D + 1, D + 1, generator=_g)).requires_grad_()
    _opt = torch.optim.Adam([W_kq, W_pv], lr=3e-3)
    for _step in range(3000):
        _b = ex.sample_linear_tasks(256, _n + 1, D, generator=_g)
        _pred = ex.lsa_predict(_b["xs"][:, :_n], _b["ys"][:, :_n], _b["xs"][:, _n], W_kq, W_pv)
        _loss = ((_pred - _b["ys"][:, _n]) ** 2).mean()
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    eta_star = 1 / (_n + D + 1)  # the best single GD step from w = 0 (on paper 3)
    # the prediction is sum_j (u.x_j + c y_j)(x_j^T A x_q + y_j b.x_q), with c = W_pv[y, y], u = W_pv[y, x],
    # A = W_kq[x, x], b = W_kq[y, x]: the terms linear in y act as one GD step preconditioned by c A + u b^T
    _u, _b = W_pv[D, :D].detach(), W_kq[D, :D].detach()
    _M = (W_pv[D, D] * W_kq[:D, :D]).detach() + torch.outer(_u, _b)
    _t = ex.sample_linear_tasks(N_TEST, _n + 1, D, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        _lsa = ex.lsa_predict(_t["xs"][:, :_n], _t["ys"][:, :_n], _t["xs"][:, _n], W_kq, W_pv)
    _gd = ex.gd_predictions(_t["xs"], _t["ys"], lr=eta_star)[:, _n]
    _y = _t["ys"][:, _n]
    _fig, _axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    _vmax = float(_M.abs().max())
    for _ax, _mat, _title in ((_axes[0], _M, "learned: c·A + u bᵀ"),
                              (_axes[1], eta_star * torch.eye(D), f"one GD step: η* I, η* = 1/{_n + D + 1}")):
        _im = _ax.imshow(_mat.numpy(), cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        _ax.set_title(_title, fontsize=9)
        _ax.set(xticks=[], yticks=[])
    _fig.colorbar(_im, ax=_axes, shrink=0.8)
    _off = float((_M - torch.diag(torch.diag(_M))).abs().max())
    mo.vstack([_fig, mo.md(
        f"diagonal of the learned matrix: {', '.join(f'{v:.4f}' for v in torch.diag(_M))} (η* = {eta_star:.4f}); "
        f"largest off-diagonal entry {_off:.4f}; |u| = {float(_u.norm()):.4f}, |b| = {float(_b.norm()):.4f}.\n\n"
        "| predictor (10 in-context examples) | test MSE |\n|---|---|\n"
        f"| predict 0 | {float((_y ** 2).mean()):.3f} |\n"
        f"| one GD step with η* | {float(((_gd - _y) ** 2).mean()):.3f} |\n"
        f"| trained linear self-attention | {float(((_lsa - _y) ** 2).mean()):.3f} |\n"
        f"| least squares | {float(((ex.least_squares_predictions(_t['xs'], _t['ys'])[:, _n] - _y) ** 2).mean()):.3f} |\n\n"
        f"mean squared difference between the trained attention's predictions and one GD step's: "
        f"{float(((_lsa - _gd) ** 2).mean()):.4f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the learned attention matrix is (close to) a multiple of the identity, and
    the multiple is (close to) the best single step size. why would gradient descent on the
    attention weights find *this* solution? (think about which other terms in the prediction
    can correlate with $y_q$, by symmetry.) and why is one step of GD so much worse than least
    squares with 10 examples in 5 dimensions?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2: a transformer learns to regress in context

    your `ICLTransformer` ({TF['n_layers']} blocks, $d_\text{{model}} = {TF['d_model']}$,
    {TF['n_heads']} heads) is trained with `train_icl` for {TF_STEPS:,} steps of {TF_BATCH} fresh
    tasks each ({TF_STEPS * TF_BATCH:,} tasks, none repeated), Adam at {TF_LR}, noiseless data, one
    training seed (the lab notes report five).
    then its error at every context length, next to least squares (which is the Bayes-optimal
    answer here: with no noise and a Gaussian prior, the posterior mean is the minimum-norm
    solution), one GD step with the best step size for each length, and predicting 0.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        f"plot the transformer's error against the number of in-context examples (0 to {N_POINTS - 1}). what happens "
        f"around {D} examples, the dimension? will it get to least squares' error of 0?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    torch.manual_seed(0)
    model = ex.ICLTransformer(D, max_points=N_POINTS, **TF)
    _t0 = time.perf_counter()
    tf_hist = ex.train_icl(model, n_points=N_POINTS, n_steps=TF_STEPS, batch_size=TF_BATCH, lr=TF_LR, seed=0)
    tf_secs = time.perf_counter() - _t0
    n_params = sum(p.numel() for p in model.parameters())
    clean = test_tasks(100)
    with torch.no_grad():
        _tf = model(clean["xs"], clean["ys"])
    curves = {
        "predict 0": err_curve(torch.zeros_like(clean["ys"]), clean["ys"]),
        "one GD step (best η per length)": err_curve(torch.stack(
            [ex.gd_predictions(clean["xs"], clean["ys"], lr=1 / (i + D + 1))[:, i] for i in range(N_POINTS)], 1),
            clean["ys"]),
        "least squares": err_curve(ex.least_squares_predictions(clean["xs"], clean["ys"]), clean["ys"]),
        "transformer": err_curve(_tf, clean["ys"]),
    }
    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [2, 1]})
    plot_curves(_ax, curves, f"noiseless tasks, D = {D}")
    _ax2.plot(np.convolve(tf_hist, np.ones(50) / 50, mode="valid"))
    _ax2.set(xlabel="training step", ylabel="training MSE (50-step mean)", yscale="log",
             title="training curve")
    _fig.tight_layout()
    _cols = (0, 1, 2, 3, 4, 5, 6, 10, 14)
    mo.vstack([_fig, mo.md(
        "| predictor | " + " | ".join(f"{c} ex." for c in _cols) + " |\n|---|" + "---|" * len(_cols) + "\n"
        + "\n".join(f"| {n} | " + " | ".join(f"{m[c]:.3f}" for c in _cols) + " |" for n, (m, _) in curves.items())
        + f"\n\ntransformer: {n_params:,} parameters, trained in {tf_secs:.0f} s; standard errors ≤ "
          f"{max(se.max() for _, se in curves.values()):.3f}.")])
    return clean, curves, model, n_params, tf_secs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with 1 or 2 examples the transformer is close to least squares, and with
    every context it is far better than one GD step. what is the error of *any* method with
    $i < D$ noiseless examples, at best, and why does it fall by about one unit per example?
    from 3 examples on the transformer falls behind, and the gap widens as $i$ approaches $D$:
    what does least squares exploit there that is hard to learn (think of how ill-conditioned
    $X^\top X$ is when $i \approx D$)? past $D$, least squares is exact and the transformer isn't:
    what limits it? (look at the training curve.) the model saw only random tasks, never the
    words "least squares": in what sense is it doing regression?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## project: in-context learning vs least squares vs MAML

    the question: given $i$ examples of a new linear task, how good is each meta-learned
    predictor, and how does it compare with the classical answer? the competitors:

    - **least squares** (minimum norm), no meta-learning at all;
    - **MAML** (your topic-09 code) on a linear model $x^\top W + b$: meta-trained with
      `train_maml` (10-shot tasks, 300 meta-iterations, meta-batch 25) for each inner learning
      rate in a grid and 1 or 5 training inner steps; at test time it adapts with $k$ inner steps.
      to tune it **fairly**, the inner learning rate, the training steps and $k$ are chosen *per
      context length* on 2,000 separate validation tasks, a per-length tuning the transformer
      doesn't get;
    - **the transformer** from experiment 2: one model for every context length, trained once.

    one caveat before you look: **the transformer's training seed matters a lot.** in five runs
    measured for the lab notes (seeds 0–4, same configuration), its error at 14 examples ranged
    from 0.22 to 0.72, about 3×, and seed 0, the one trained above, is at the bad end. tick the box
    below to train seeds 1–4 yourself and add a mean ± sd-over-seeds row (4–6 extra minutes). all
    predictors are scored on the *same* test tasks, so the second table compares them with paired
    per-task differences, whose standard errors are much smaller than those of the separate means.

    all three are trained or tuned on noiseless tasks only. then everyone is scored on three test
    sets: the training distribution; the same tasks with label noise $\sigma = 0.5$; and inputs
    scaled by 2 (so $x \sim \mathcal N(0, 4I)$). under noise, ridge regression with $\lambda = \sigma^2$
    is the Bayes-optimal predictor, shown as an oracle (it knows $\sigma$).
    """)
    return


@app.cell
def _():
    p4 = ui.predict(
        "on noiseless tasks with 10 examples, rank least squares, tuned MAML and the transformer. then: which of them "
        "suffers most from label noise, and which from inputs twice as large?"
    )
    p4
    return (p4,)


@app.cell
def _(p4):
    ui.gate(p4)

    def _sampler(g):
        b = ex.sample_linear_tasks(25, 20, D, generator=g)
        return {"xs": b["xs"][:, :10], "ys": b["ys"][:, :10, None], "xq": b["xs"][:, 10:], "yq": b["ys"][:, 10:, None]}

    _t0 = time.perf_counter()
    maml_inits = {}
    for _lr in (0.03, 0.1, 0.2, 0.3):
        for _steps in (1, 5):
            _init = maml.init_mlp([D, 1], generator=torch.Generator().manual_seed(7))
            maml_inits[(_lr, _steps)], _ = maml.train_maml(_init, _sampler, n_iters=300, meta_lr=1e-2, inner_lr=_lr,
                                                           n_steps=_steps, seed=0)
    _val = test_tasks(200)
    val_err = {}  # (lr, train steps, test steps) -> per-position validation MSE
    for (_lr, _steps), _params in maml_inits.items():
        for _k in (1, 5, 20, 50):
            _p = maml_predictions(_params, _val["xs"], _val["ys"], inner_lr=_lr, n_steps=_k)
            val_err[(_lr, _steps, _k)] = np.nan_to_num(err_curve(_p, _val["ys"])[0], nan=np.inf)
    best_cfg = [min(val_err, key=lambda c: val_err[c][i]) for i in range(N_POINTS)]
    maml_secs = time.perf_counter() - _t0
    mo.md("MAML's chosen (inner lr, training steps, test steps) by context length: "
          + ", ".join(f"{i}: {c}" for i, c in enumerate(best_cfg) if i in (1, 2, 5, 10, 14))
          + f". meta-training and tuning took {maml_secs:.0f} s.")
    return best_cfg, maml_inits, maml_secs


@app.cell
def _():
    more_seeds = mo.ui.checkbox(label="train 4 more transformer seeds (~4–6 min)")
    more_seeds
    return (more_seeds,)


@app.cell
def _(model, more_seeds):
    extra_models = [model]
    if more_seeds.value:
        for _s in (1, 2, 3, 4):
            torch.manual_seed(_s)
            _m = ex.ICLTransformer(D, max_points=N_POINTS, **TF)
            ex.train_icl(_m, n_points=N_POINTS, n_steps=TF_STEPS, batch_size=TF_BATCH, lr=TF_LR, seed=_s)
            extra_models.append(_m)
    return (extra_models,)


@app.cell
def _():
    scenario = mo.ui.dropdown(["noiseless", "label noise σ = 0.5", "inputs × 2"], value="noiseless",
                              label="test set for the plot")
    scenario
    return (scenario,)


@app.cell
def _(best_cfg, extra_models, maml_inits, model):
    project, project_sq, seed_curves = {}, {}, {}
    for _name, _kw, _seed in (("noiseless", {}, 300), ("label noise σ = 0.5", {"noise_std": 0.5}, 301),
                              ("inputs × 2", {"x_scale": 2.0}, 302)):
        _t = test_tasks(_seed, **_kw)
        _sq = {"predict 0": sq_err(torch.zeros_like(_t["ys"]), _t["ys"]),
               "least squares": sq_err(ex.least_squares_predictions(_t["xs"], _t["ys"]), _t["ys"])}
        if "noise_std" in _kw:
            _sq["ridge, λ = σ² (oracle)"] = sq_err(ex.least_squares_predictions(_t["xs"], _t["ys"], ridge=0.25),
                                                   _t["ys"])
        _cache, _mp = {}, torch.zeros_like(_t["ys"])
        for _i, (_lr, _s, _k) in enumerate(best_cfg):
            if (_lr, _s, _k) not in _cache:
                _cache[(_lr, _s, _k)] = maml_predictions(maml_inits[(_lr, _s)], _t["xs"], _t["ys"], inner_lr=_lr,
                                                         n_steps=_k)
            _mp[:, _i] = _cache[(_lr, _s, _k)][:, _i]
        _sq["MAML (tuned per length)"] = sq_err(_mp, _t["ys"])
        with torch.no_grad():
            _sq["transformer"] = sq_err(model(_t["xs"], _t["ys"]), _t["ys"])
            seed_curves[_name] = np.array([sq_err(_m(_t["xs"], _t["ys"]), _t["ys"]).mean(0) for _m in extra_models])
        project_sq[_name] = _sq
        project[_name] = {_n: mean_se(_a) for _n, _a in _sq.items()}
    return project, project_sq, seed_curves


@app.cell
def _(project, scenario):
    _fig, _ax = plt.subplots(figsize=(6.4, 3.4))
    plot_curves(_ax, project[scenario.value], scenario.value)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(maml_secs, n_params, project, project_sq, seed_curves, tf_secs):
    _cols = (2, 5, 10, 14)
    _head = " | ".join(f"{c} examples" for c in _cols) + " |\n|---|" + "---|" * len(_cols) + "\n"
    _md = []
    for _scen, _rows in project.items():
        _lines = [f"| {n} | " + " | ".join(fmt(m[c], se[c]) for c in _cols) + " |" for n, (m, se) in _rows.items()]
        _sc = seed_curves[_scen]
        if len(_sc) > 1:
            _lines.append(f"| transformer, seeds 0–{len(_sc) - 1} (mean ± sd over seeds) | "
                          + " | ".join(f"{_sc[:, c].mean():.3f} ± {_sc[:, c].std(ddof=1):.3f}" for c in _cols) + " |")
        _tf = project_sq[_scen]["transformer"]
        _diff = [f"| {n} − transformer | " + " | ".join(fmt(*[v[c] for v in mean_se(a - _tf)]) if np.isfinite(a).all()
                                                       else "diverges" for c in _cols) + " |"
                 for n, a in project_sq[_scen].items() if n not in ("transformer", "predict 0")]
        _md.append(f"**{_scen}**: test MSE ± standard error\n\n| predictor | " + _head + "\n".join(_lines)
                   + f"\n\n**{_scen}**: paired per-task difference from the transformer (seed 0), ± standard "
                     "error (negative: better than the transformer)\n\n| difference | " + _head + "\n".join(_diff))
    mo.md("\n\n".join(_md) + f"""

    {N_TEST:,} tasks per test set, shared by all predictors. **compute:** the transformer has
    {n_params:,} parameters and saw {TF_STEPS * TF_BATCH:,} training tasks ({tf_secs:.0f} s, one seed);
    MAML's 8 meta-training runs and its tuning took {maml_secs:.0f} s; least squares needs no training.
    at test time least squares solves a $D \\times D$ system, MAML takes up to 50 gradient steps on a
    $D + 1$-parameter model, and the transformer runs one forward pass of
    {2 * N_POINTS} tokens through {TF['n_layers']} blocks.""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain (the project write-up):** on the training distribution, who wins at 2, 5, 10
    and 14 examples, and by more than the standard errors? why can a linear MAML with enough
    inner steps match least squares exactly (topic 09, on paper 2), and what did tuning choose
    for it? under label noise, which predictor comes closest to the oracle near $D$ examples,
    and why does least squares blow up there? with inputs twice as large, which method breaks,
    and what does that say about *what* the transformer learned: an algorithm, or an algorithm
    that only works on the inputs it was trained on? what would you need to claim "the
    transformer implements least squares", and what does this lab support?
    """)
    return


if __name__ == "__main__":
    app.run()
