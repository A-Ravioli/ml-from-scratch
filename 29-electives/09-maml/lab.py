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
    torch.set_num_threads(1)  # tiny networks: one thread is fastest

    SIZES = [1, 40, 40, 1]  # Finn et al.'s sine network
    K = 10  # support points per task (10-shot regression)
    META_BATCH = 25
    N_ITERS = 3000  # meta-iterations per run (Finn et al. use 70,000)
    INNER_LR = 0.01
    SEEDS = (0, 1, 2)
    STEPS = list(range(11))  # test-time inner steps


@app.function
def sampler(generator):
    """a meta-batch of 10-shot sine tasks (10 query points each)."""
    return ex.sample_sine_tasks(META_BATCH, K, K, generator=generator)


@app.function
def train(seed, *, n_steps, first_order=False, n_iters=N_ITERS):
    """meta-train SIZES from a seeded init; n_steps=0 is plain multi-task training. returns (params, seconds)."""
    init = ex.init_mlp(SIZES, generator=torch.Generator().manual_seed(1000 + seed))
    t0 = time.perf_counter()
    params, _ = ex.train_maml(init, sampler, n_iters=n_iters, meta_lr=1e-3, inner_lr=INNER_LR, n_steps=n_steps,
                              first_order=first_order, seed=seed)
    return params, time.perf_counter() - t0


@app.function
def test_curve(params, batch, steps=STEPS, lr=INNER_LR):
    """mean query MSE over the batch's tasks after k test-time inner steps, for each k in steps."""
    out = []
    for k in steps:
        fast = ex.adapt([p.detach() for p in params], batch["xs"], batch["ys"], inner_lr=lr, n_steps=k,
                        first_order=True)
        with torch.no_grad():
            out.append(ex.task_mse(ex.mlp_forward(fast, batch["xq"]), batch["yq"]).mean().item())
    return np.array(out)


@app.function
def summary(values):
    """'mean ± sd' over seeds."""
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # MAML · lab

    every experiment here runs **your** `train_maml`, `adapt` and `mlp_forward`. the setting
    is Finn et al.'s: sine waves $A\sin(x - \varphi)$ with $A \in [0.1, 5]$, $\varphi \in [0, \pi]$,
    $x \in [-5, 5]$; a 1–40–40–1 ReLU network; {K} support points per task; meta-batches of
    {META_BATCH} tasks; Adam at $10^{{-3}}$ for {N_ITERS:,} meta-iterations (Finn et al. use
    70,000); inner learning rate {INNER_LR}. the baseline, **pretraining**, is the same loop with
    `n_steps=0`: one network fitted to all tasks at once, then fine-tuned at test time with the
    same inner steps. test scores are on 1,000 fresh tasks with 10 support and 100 query points
    each. the lab takes a little over a minute on one idle CPU core.
    """)
    return


@app.cell
def _():
    test_batch = ex.sample_sine_tasks(1000, K, 100, generator=torch.Generator().manual_seed(12345))
    return (test_batch,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what does meta-training buy?

    MAML (one inner step during training) and pretraining, three seeds each, then $k = 0, 1,
    \dots, 10$ gradient steps on each test task's 10 points.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "before any test-time step (k = 0), which init has the lower query MSE, MAML's or pretraining's? after "
        "one step? after ten?"
    )
    p1
    return (p1,)


@app.cell
def _(p1, test_batch):
    ui.gate(p1)
    maml1, pre = {}, {}
    curves = {"MAML (1 step)": [], "pretraining": []}
    secs = {"MAML (1 step)": [], "pretraining": []}
    for _s in SEEDS:
        maml1[_s], _t1 = train(_s, n_steps=1)
        pre[_s], _t0 = train(_s, n_steps=0)
        curves["MAML (1 step)"].append(test_curve(maml1[_s], test_batch))
        curves["pretraining"].append(test_curve(pre[_s], test_batch))
        secs["MAML (1 step)"].append(_t1)
        secs["pretraining"].append(_t0)
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    for _name, _c in curves.items():
        _c = np.array(_c)
        _ax.errorbar(STEPS, _c.mean(0), yerr=_c.std(0, ddof=1), fmt="o-", capsize=3, label=_name)
    _ax.set(xlabel="test-time inner steps k", ylabel="query MSE", yscale="log",
            title="10-shot sine regression, mean ± sd over 3 seeds")
    _ax.legend()
    _fig.tight_layout()
    _rows = "\n".join(
        f"| {n} | {summary([c[0] for c in v])} | {summary([c[1] for c in v])} | {summary([c[10] for c in v])} | "
        f"{np.mean(secs[n]):.1f} s |" for n, v in curves.items())
    mo.vstack([_fig, mo.md("| init | k = 0 | k = 1 | k = 10 | meta-training time |\n|---|---|---|---|---|\n" + _rows)])
    return curves, maml1, pre, secs


@app.cell
def _():
    task_pick = mo.ui.slider(0, 19, value=3, label="test task")
    k_pick = mo.ui.slider(0, 10, value=1, label="test-time steps k")
    mo.hstack([task_pick, k_pick])
    return k_pick, task_pick


@app.cell
def _(k_pick, maml1, pre, task_pick, test_batch):
    _t = task_pick.value
    _grid = torch.linspace(-5, 5, 200).reshape(1, -1, 1)
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _amp, _ph = test_batch["amp"][_t], test_batch["phase"][_t]
    _ax.plot(_grid[0, :, 0], (_amp * torch.sin(_grid - _ph))[0, :, 0], "k", lw=1, label="true")
    for _name, _params, _col in (("MAML", maml1[0], "C0"), ("pretraining", pre[0], "C1")):
        _fast = ex.adapt([p.detach() for p in _params], test_batch["xs"][_t:_t + 1], test_batch["ys"][_t:_t + 1],
                         inner_lr=INNER_LR, n_steps=k_pick.value, first_order=True)
        with torch.no_grad():
            _ax.plot(_grid[0, :, 0], ex.mlp_forward(_fast, _grid)[0, :, 0], _col, label=f"{_name}, k = {k_pick.value}")
    _ax.scatter(test_batch["xs"][_t, :, 0], test_batch["ys"][_t, :, 0], c="k", s=18, zorder=3, label="10 support points")
    _ax.set(xlabel="x", ylabel="y", ylim=(-5.5, 5.5))
    _ax.legend(fontsize=7, loc="lower left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at $k = 0$ both inits predict roughly the same thing. what is it, and why is it
    the best single function for *every* task at once? after one step MAML's error is a quarter of
    pretraining's, with the same learning rate and the same 10 points. what did MAML put into its
    weights that pretraining didn't? (drag the sliders: what does MAML's $k = 0$ curve look like?)
    and why does MAML keep improving beyond the single step it was trained for?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how many inner steps should training use?

    the same budget of meta-iterations, with 3 inner steps during training instead of 1, and
    first-order MAML with 1 step. three seeds each, scored like experiment 1.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "MAML trained with 3 inner steps vs with 1: which is better at test-time k = 1? at k = 10? how much "
        "slower is it to train, and how much worse is first-order MAML?"
    )
    p2
    return (p2,)


@app.cell
def _(curves, p2, secs, test_batch):
    ui.gate(p2)
    variants = {"MAML, 1 step (exp. 1)": (np.array(curves["MAML (1 step)"]), secs["MAML (1 step)"])}
    for _name, _kw in (("MAML, 3 steps", dict(n_steps=3)),
                       ("first-order MAML, 1 step", dict(n_steps=1, first_order=True))):
        _c, _t = [], []
        for _s in SEEDS:
            _m, _sec = train(_s, **_kw)
            _c.append(test_curve(_m, test_batch))
            _t.append(_sec)
        variants[_name] = (np.array(_c), _t)
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    for _name, (_c, _) in variants.items():
        _ax.errorbar(STEPS, _c.mean(0), yerr=_c.std(0, ddof=1), fmt="o-", capsize=3, label=_name)
    _ax.set(xlabel="test-time inner steps k", ylabel="query MSE", yscale="log", title="mean ± sd over 3 seeds")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| training | k = 1 | k = 5 | k = 10 | meta-training time |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {n} | {summary(c[:, 1])} | {summary(c[:, 5])} | {summary(c[:, 10])} | {np.mean(t):.1f} s |"
                    for n, (c, t) in variants.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at $k = 1$ the two inits are level; from $k = 5$ on the 3-step init is ahead.
    is its advantage larger than the seed-to-seed spread? why should an init trained for three
    steps be better after five or ten, and why might it have been *worse* after one (the lab notes
    report what 5 training steps did)? compare the time column with the number of inner
    steps: what does the second-order term cost, and where does the cost come from? first-order
    MAML drops the Hessian term entirely: how much does that cost in accuracy here, and why is it
    still far better than pretraining?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: adapting from half a wave

    now every test task's 5 support points lie in $[-5, 0]$, and we ask for predictions on the
    other half, $[0, 5]$, after 10 test-time steps. a network that only "fits points" has no reason
    to know what happens where it saw no data.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with 5 points on the left half only, will MAML's prediction on the right half be close to the true sine, "
        "close to zero, or wild?"
    )
    p3
    return (p3,)


@app.cell
def _(maml1, p3, pre):
    ui.gate(p3)
    _g = torch.Generator().manual_seed(54321)
    half = ex.sample_sine_tasks(1000, 5, 100, generator=_g)
    half["xs"] = half["xs"] / 2 - 2.5  # [-5, 5] -> [-5, 0]
    half["ys"] = half["amp"][:, None, None] * torch.sin(half["xs"] - half["phase"][:, None, None])
    half["xq"] = half["xq"] / 2 + 2.5  # [-5, 5] -> [0, 5]
    half["yq"] = half["amp"][:, None, None] * torch.sin(half["xq"] - half["phase"][:, None, None])
    _zero = (half["yq"] ** 2).mean().item()
    _res = {n: [test_curve(m[s], half, steps=[10])[0] for s in SEEDS] for n, m in (("MAML", maml1), ("pretraining", pre))}
    _grid = torch.linspace(-5, 5, 200).reshape(1, -1, 1)
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
    for _ax, _t in zip(_axes, (0, 1, 2)):
        _ax.axvspan(0, 5, color="0.93")
        _ax.plot(_grid[0, :, 0], (half["amp"][_t] * torch.sin(_grid - half["phase"][_t]))[0, :, 0], "k", lw=1)
        for _name, _m, _col in (("MAML", maml1[0], "C0"), ("pretraining", pre[0], "C1")):
            _fast = ex.adapt([p.detach() for p in _m], half["xs"][_t:_t + 1], half["ys"][_t:_t + 1],
                             inner_lr=INNER_LR, n_steps=10, first_order=True)
            with torch.no_grad():
                _ax.plot(_grid[0, :, 0], ex.mlp_forward(_fast, _grid)[0, :, 0], _col, label=_name)
        _ax.scatter(half["xs"][_t, :, 0], half["ys"][_t, :, 0], c="k", s=15, zorder=3)
        _ax.set(xlabel="x", ylim=(-5.5, 5.5))
    _axes[0].legend(fontsize=7)
    _fig.suptitle("support on [-5, 0] (white), predictions scored on [0, 5] (grey), k = 10", fontsize=9)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| init | query MSE on [0, 5] after 10 steps |\n|---|---|\n"
        f"| predict 0 everywhere | {_zero:.3f} |\n"
        + "\n".join(f"| {n} | {summary(v)} |" for n, v in _res.items()) + "\n\nmean ± sd over seeds 0–2.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** neither network saw a single point on $[0, 5]$ for these tasks. where does
    MAML's knowledge of the right half come from? the support set pins down amplitude and phase,
    two numbers: what does that say about what MAML's init "encodes"? why can't gradient steps
    on the pretrained init do the same, however many you take?
    """)
    return


if __name__ == "__main__":
    app.run()
