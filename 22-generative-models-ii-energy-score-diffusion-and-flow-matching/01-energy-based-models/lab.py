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
    from mlfs.generative import mode_coverage  # yours, from ch 21 · topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    STD = 0.15
    ANG = 2 * np.pi * np.arange(8) / 8
    CENTERS = 2.0 * np.c_[np.cos(ANG), np.sin(ANG)]
    X_TRAIN = datasets.eight_gaussians(5000, std=STD, rng=0)
    X_TEST = datasets.eight_gaussians(5000, std=STD, rng=1)
    ITERS = 2400
    TRAIN_STEPS = 20  # Langevin steps per training iteration
    STEP = 0.01


@app.function
def true_log_prob(x):
    """the exact log-density of the 8-gaussian mixture."""
    d2 = ((x[:, None, :] - CENTERS[None]) ** 2).sum(-1)
    comp = -d2 / (2 * STD ** 2) - np.log(2 * np.pi * STD ** 2)
    m = comp.max(1, keepdims=True)
    return (m + np.log(np.exp(comp - m).mean(1, keepdims=True)))[:, 0]


@app.function
def box_noise(n, seed):
    """uniform points on the training data's bounding box, where the training chains start."""
    lo, hi = X_TRAIN.min(0), X_TRAIN.max(0)
    g = torch.Generator().manual_seed(seed)
    return torch.tensor(lo, dtype=torch.float32) + torch.tensor(hi - lo, dtype=torch.float32) * torch.rand((n, 2), generator=g)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # energy-based models · lab

    every experiment uses **your** `langevin_dynamics`, `mala`, `EnergyMLP` and `train_ebm`.
    the data are the 8 gaussians (std {STD}, radius 2), 5,000 training and 5,000 test points,
    synthetic, so offline runs are the same experiments. for each experiment: write down a
    prediction, lock it in, run it, then explain what happened. compare with
    `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the price of a finite step

    20,000 Langevin chains on $\mathcal{N}(0, 1)$ (score $-x$) for 300 steps of size $h$,
    started at 0. on paper 2 says unadjusted Langevin settles at variance
    $1/(1 - h/4)$; MALA should give exactly 1 at any step.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at step size h = 2, what variance will unadjusted Langevin's samples have (target: 1)? at h = 3.9? and what "
        "acceptance rate will MALA need to stay exact at h = 3?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rows, _hs, _ula, _mal = [], [0.1, 0.5, 1.0, 2.0, 3.0, 3.9], [], []
    for _h in _hs:
        _x = ex.langevin_dynamics(lambda v: -v, torch.zeros(20000, 1), step_size=_h, n_steps=300,
                                  generator=torch.Generator().manual_seed(0))
        _xm, _rate = ex.mala(lambda v: 0.5 * (v ** 2).sum(1), torch.zeros(20000, 1), step_size=_h, n_steps=300,
                             generator=torch.Generator().manual_seed(0))
        _ula.append(_x.var().item())
        _mal.append(_xm.var().item())
        _rows.append(f"| {_h} | {1 / (1 - _h / 4):.3f} | {_ula[-1]:.3f} | {_mal[-1]:.3f} | {_rate:.3f} |")
    _fig, _ax = plt.subplots(figsize=(5.5, 3.5))
    _hh = np.linspace(0.05, 3.9, 200)
    _ax.plot(_hh, 1 / (1 - _hh / 4), "k--", label="theory: 1 / (1 - h/4)")
    _ax.plot(_hs, _ula, "o", label="your Langevin")
    _ax.plot(_hs, _mal, "s", label="your MALA")
    _ax.set(xlabel="step size h", ylabel="sample variance", yscale="log", title="target variance 1")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| h | theory (ULA) | Langevin variance | MALA variance | MALA acceptance |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the bias doesn't shrink with more steps: why not (what does a longer
    chain converge *to*)? what happens at $h \ge 4$, and what is the analogue for gradient
    descent on a quadratic (chapter 6)? MALA fixes the variance at every step size; what does
    it pay instead, and why would that payment be much larger in 3,072 dimensions?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2: short-run chains and what they teach the energy

    two ways to train the same `EnergyMLP` (3 layers of 64) for {ITERS} iterations with
    {TRAIN_STEPS} Langevin steps of size {STEP} per iteration: **short-run** (every chain
    restarted from box noise, `reinit_prob=1`) and **persistent** (a replay buffer,
    `reinit_prob=0.05`), two seeds each. then we sample each model the way it was trained,
    from box noise, but for {TRAIN_STEPS}, 200 and 2,000 steps (up to 100 times the training
    length), and count samples per mode
    (topic 05 of chapter 21's `mode_coverage`).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "a short-run EBM's samples after the 20 steps it was trained with look reasonable. run the same chains for 2,000 "
        "steps: do the samples get better, stay the same, or get worse? how evenly will they spread over the 8 modes?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    models, _rows, _t0 = {}, [], time.perf_counter()
    for _name, _reinit in (("short-run", 1.0), ("persistent", 0.05)):
        for _seed in (0, 1):
            torch.manual_seed(_seed)
            _e = ex.EnergyMLP(2, hidden_dim=64, n_layers=3)
            ex.train_ebm(_e, X_TRAIN, n_iters=ITERS, batch_size=128, lr=2e-3, langevin_steps=TRAIN_STEPS,
                         step_size=STEP, buffer_size=5000, reinit_prob=_reinit, alpha=0.1, seed=_seed)
            models[(_name, _seed)] = _e
    _train_s = time.perf_counter() - _t0
    snapshots = {}
    for (_name, _seed), _e in models.items():
        _g = torch.Generator().manual_seed(10 + _seed)
        _x, _done = box_noise(2000, 20 + _seed), 0
        for _k in (TRAIN_STEPS, 200, 2000):
            _x = ex.langevin_dynamics(ex.score_from_energy(_e), _x, step_size=STEP, n_steps=_k - _done, generator=_g)
            _done = _k
            _n, _q, _counts = mode_coverage(_x.numpy(), CENTERS, STD)
            snapshots[(_name, _seed, _k)] = _x.numpy()
            _rows.append(f"| {_name} | {_seed} | {_k} | {_n} | {_q:.3f} | {_counts.min()} – {_counts.max()} | "
                         f"{_counts.max() / max(_counts.min(), 1):.1f} |")
    _fig, _axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for _r, _name in enumerate(("short-run", "persistent")):
        for _c, _k in enumerate((TRAIN_STEPS, 200, 2000)):
            _s = snapshots[(_name, 0, _k)]
            _axes[_r, _c].scatter(*_s.T, s=1, alpha=0.4)
            _axes[_r, _c].scatter(*CENTERS.T, c="red", marker="x", s=20)
            _axes[_r, _c].set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=f"{_name}, {_k} steps (seed 0)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"training: {_train_s:.0f} s for the 4 models · 2,000 chains from box noise · 'within 3 std' is the fraction of "
        "samples within 3 std of a mode (the true mixture: 0.99)\n\n"
        "| training | seed | sampling steps | modes covered | within 3 std | samples per mode | max / min |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (models,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** contrastive divergence stops when the *negative samples* match the data
    (on paper 4). for a short-run model, which distribution are the negative samples drawn
    from, and why is it not $e^{-E}/Z$? during training, did any chain (short-run or
    persistent) ever move from one mode to another? what does that leave unconstrained about
    the energy (think about the depths of the 8 wells relative to each other), and how does a
    2,000-step chain expose it? persistent chains keep the mode proportions their buffer
    happened to start with: how could contrastive divergence then distort the well depths?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: is the energy a good density?

    in two dimensions we can do what is impossible for images: compute $Z_\theta$ on a grid
    ($351 \times 351$ points on $[-3.5, 3.5]^2$) and evaluate the normalized model density
    $e^{-E_\theta}/Z_\theta$. we report the test NLL (the true mixture's is shown for
    comparison) and the fraction of the model's probability mass within 3 std of a mode.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the true test NLL is about 1.11 nats. how close will the normalized EBM densities get? which training scheme "
        "will put more of its mass near the modes?"
    )
    p3
    return (p3,)


@app.cell
def _(models, p3):
    ui.gate(p3)
    _g1 = torch.linspace(-3.5, 3.5, 351)
    _h = (_g1[1] - _g1[0]).item()
    _grid = torch.cartesian_prod(_g1, _g1)
    _lab = np.argmin(((_grid.numpy()[:, None] - CENTERS[None]) ** 2).sum(-1), 1)
    _near = np.sqrt(((_grid.numpy() - CENTERS[_lab]) ** 2).sum(1)) < 3 * STD
    _rows = []
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.6))
    _true = np.exp(true_log_prob(_grid.numpy())).reshape(351, 351).T
    _axes[0].imshow(np.log(_true + 1e-8), extent=(-3.5, 3.5, -3.5, 3.5), origin="lower", cmap="magma", vmin=-12)
    _axes[0].set_title("true log density")
    for (_name, _seed), _e in models.items():
        with torch.no_grad():
            _E = _e(_grid)
            _logZ = torch.logsumexp(-_E, 0).item() + math.log(_h * _h)
            _nll = _e(torch.tensor(X_TEST, dtype=torch.float32)).mean().item() + _logZ
            _p = torch.exp(-_E - _logZ).numpy()
        _rows.append(f"| {_name} | {_seed} | {_nll:.3f} | {(_p * _h * _h)[_near].sum():.3f} |")
        if _seed == 0:
            _ax = _axes[1 if _name == "short-run" else 2]
            _ax.imshow(np.log(_p.reshape(351, 351).T + 1e-8), extent=(-3.5, 3.5, -3.5, 3.5), origin="lower",
                       cmap="magma", vmin=-12)
            _ax.set_title(f"{_name} EBM: log density")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"true test NLL **{-true_log_prob(X_TEST).mean():.3f}** nats; true mass within 3 std of a mode: 0.989\n\n"
        "| training | seed | test NLL (grid-normalized) | mass within 3 std |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** none of these numbers was available during training. what made them
    computable here, and why is that impossible for a 784-dimensional MNIST EBM? compare the
    density plots with the 2,000-step samples of experiment 2: does the density explain
    which modes the long chains crowd into? the models are within about 0.2 nats of the truth:
    what would you change in training (iterations, step size, number of Langevin steps) to
    close that gap, and what would each change cost?
    """)
    return


if __name__ == "__main__":
    app.run()
