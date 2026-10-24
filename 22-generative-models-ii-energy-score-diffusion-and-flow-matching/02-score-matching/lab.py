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

    from mlfs import notebooks, ui
    from mlfs.diffusion import langevin_dynamics  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    WEIGHTS = np.array([0.2, 0.8])
    MEANS = np.array([[-5.0, -5.0], [5.0, 5.0]])
    LADDER = [float(s) for s in ex.geometric_sigmas(20.0, 0.1, 10)]
    ITERS = 2000


@app.function
def sample_mixture(n, seed):
    """n points from 0.2 N((-5,-5), I) + 0.8 N((5,5), I) and their component labels."""
    rng = np.random.default_rng(seed)
    k = (rng.random(n) < WEIGHTS[1]).astype(int)
    return MEANS[k] + rng.normal(size=(n, 2)), k


@app.function
def mixture_score(x, sigma):
    """the exact score of the mixture blurred by N(0, sigma^2 I): each component becomes N(mu_k, (1 + sigma^2) I)."""
    sig = torch.as_tensor(sigma, dtype=x.dtype)
    sig = sig.expand(x.shape[0]) if sig.dim() == 0 else sig
    v = (1 + sig ** 2)[:, None, None]
    d = x[:, None, :] - torch.tensor(MEANS, dtype=x.dtype)[None]
    logw = torch.log(torch.tensor(WEIGHTS, dtype=x.dtype))[None] - 0.5 * (d ** 2).sum(-1) / v[:, :, 0]
    r = torch.softmax(logw, dim=1)
    return -(r[:, :, None] * d / v).sum(1)


@app.function
def frac_small_mode(x):
    """fraction of samples on the side of the small (weight 0.2) component."""
    return float((x[:, 0] + x[:, 1] < 0).float().mean())


@app.function
def uniform_start(n, seed):
    return 16 * torch.rand((n, 2), generator=torch.Generator().manual_seed(seed)) - 8


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # score matching · lab

    every experiment uses **your** `dsm_loss`, `sliced_score_matching_loss`,
    `NoiseConditionalScore`, `fit_dsm`, `langevin_dynamics` (topic 01) and
    `annealed_langevin`. the data are a lopsided mixture,
    $0.2\,\mathcal{N}((-5, -5), I) + 0.8\,\mathcal{N}((5, 5), I)$ (Song & Ermon's example),
    whose blurred score is known exactly at every noise level, so we can compare learned
    scores with the truth. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the exact score is not enough

    5,000 chains start uniformly in $[-8, 8]^2$. first, 2,000 steps of plain Langevin
    ($h = 0.1$) with the *exact* score of the mixture. second, your annealed Langevin with
    the exact blurred scores on 10 levels from $\sigma = 20$ down to $0.1$ (100 steps each,
    $\epsilon = 5 \times 10^{-4}$). the true fraction in the small component is 0.2.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with the exact score, what fraction of Langevin chains will end in the small (20%) component? and with "
        "annealed Langevin?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _x0 = uniform_start(5000, 0)
    _plain = langevin_dynamics(lambda v: mixture_score(v, 0.0), _x0, step_size=0.1, n_steps=2000,
                                  generator=torch.Generator().manual_seed(1))
    _ann = ex.annealed_langevin(mixture_score, _x0, LADDER, steps_per_sigma=100, eps=5e-4,
                                generator=torch.Generator().manual_seed(2))
    _xd, _ = sample_mixture(5000, 3)
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for _ax, (_t, _s) in zip(_axes, (("data", torch.tensor(_xd)), ("Langevin, exact score", _plain),
                                     ("annealed Langevin, exact scores", _ann))):
        _ax.scatter(*_s.numpy().T, s=1, alpha=0.3)
        _ax.set(xlim=(-9, 9), ylim=(-9, 9), aspect="equal", title=f"{_t}\nsmall component: {frac_small_mode(_s):.3f}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"| sampler | fraction in the small component | chains starting on its side |\n|---|---|---|\n"
        f"| Langevin, exact score | **{frac_small_mode(_plain):.3f}** | {frac_small_mode(_x0):.3f} |\n"
        f"| annealed Langevin, exact scores | **{frac_small_mode(_ann):.3f}** | {frac_small_mode(_x0):.3f} |\n"
        f"| true mixture | 0.200 | |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the score near $(-5, -5)$ is almost exactly the score of the small
    component alone: where do the weights 0.2 and 0.8 enter the score at all? why does a
    chain essentially never cross from one side to the other (estimate the density halfway
    between the modes)? what does the largest noise level do to the picture, and why do the
    chains keep the right proportions as $\sigma$ shrinks?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2: where a learned score can be trusted

    we train two of your `NoiseConditionalScore` networks (3 layers of 128) with `fit_dsm`
    for {ITERS} steps on 20,000 samples: one on the single level $\sigma = 0.1$, one on the
    10-level ladder. the map shows the single-level network's relative error
    $\lVert s_\theta - s\rVert / \lVert s\rVert$ at $\sigma = 0.1$ over the plane; then
    both networks generate samples from uniform noise (plain Langevin for the first,
    annealed for the second).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "where in the plane will the single-level network's score be badly wrong? when it generates samples with "
        "Langevin from uniform noise, will they look like the data?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    X, _ = sample_mixture(20000, 4)
    ex.fit_dsm(ex.NoiseConditionalScore(2, hidden_dim=8, n_layers=1), X[:64], [0.1], n_iters=2, batch_size=8)  # warm-up
    nets, secs = {}, {}
    for _name, _sig in (("one level (0.1)", [0.1]), ("ladder (20 to 0.1)", LADDER)):
        torch.manual_seed(0)
        _m = ex.NoiseConditionalScore(2, hidden_dim=128, n_layers=3)
        _t0 = time.perf_counter()
        ex.fit_dsm(_m, X, _sig, n_iters=ITERS, batch_size=256, lr=1e-3, seed=0)
        secs[_name] = time.perf_counter() - _t0
        nets[_name] = _m.eval()
    _g = torch.linspace(-8, 8, 161)
    _grid = torch.cartesian_prod(_g, _g)
    with torch.no_grad():
        _err = ((nets["one level (0.1)"](_grid, 0.1) - mixture_score(_grid, 0.1)).norm(dim=1)
                / mixture_score(_grid, 0.1).norm(dim=1).clamp_min(1e-3)).numpy()
    _x0 = uniform_start(5000, 5)
    with torch.no_grad():
        _one = langevin_dynamics(lambda v: nets["one level (0.1)"](v, 0.1), _x0, step_size=0.01, n_steps=2000,
                                    generator=torch.Generator().manual_seed(6))
        _lad = ex.annealed_langevin(nets["ladder (20 to 0.1)"], _x0, LADDER, steps_per_sigma=100, eps=5e-4,
                                    generator=torch.Generator().manual_seed(7))
    _near = lambda s: float(((s[:, None, :] - torch.tensor(MEANS, dtype=s.dtype)[None]).norm(dim=-1).min(1).values < 3).float().mean())
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _im = _axes[0].imshow(np.log10(_err.reshape(161, 161).T + 1e-4), extent=(-8, 8, -8, 8), origin="lower",
                          cmap="viridis", vmin=-2, vmax=1)
    _axes[0].scatter(*X[:500].T, s=1, c="white", alpha=0.5)
    _axes[0].set(title="one level: log10 relative error", aspect="equal")
    _fig.colorbar(_im, ax=_axes[0], fraction=0.046)
    for _ax, (_t, _s) in zip(_axes[1:], (("one level, Langevin", _one), ("ladder, annealed", _lad))):
        _ax.scatter(*_s.numpy().T, s=1, alpha=0.3)
        _ax.set(xlim=(-9, 9), ylim=(-9, 9), aspect="equal", title=_t)
    _fig.tight_layout()
    _data_err = _err.reshape(161, 161)
    _dist = np.sqrt(((_grid.numpy()[:, None] - MEANS[None]) ** 2).sum(-1)).min(1)
    mo.vstack([_fig, mo.md(
        f"median relative error of the one-level score within 2 of a mode: **{np.median(_err[_dist < 2]):.3f}**; "
        f"more than 5 from both modes: **{np.median(_err[_dist > 5]):.3f}** · training {secs['one level (0.1)']:.0f} s "
        f"and {secs['ladder (20 to 0.1)']:.0f} s\n\n"
        "| samples from uniform noise | small component | within 3 of a mode |\n|---|---|---|\n"
        f"| one level, Langevin | {frac_small_mode(_one):.3f} | {_near(_one):.3f} |\n"
        f"| ladder, annealed | {frac_small_mode(_lad):.3f} | {_near(_lad):.3f} |\n| true mixture | 0.200 | 0.989 |")])
    return X, nets, secs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the DSM loss is an average over data points, so nothing forces the score
    to be right in the empty parts of the plane: how wrong is it there, compared with near
    the data, and why is it 30–40% off even *near* the data at $\sigma = 0.1$ (think about the
    size of the target $-z/\sigma$ against the size of the score)? the one-level samples look
    fine in the scatter plot: what is wrong with them, and is it a problem of score accuracy
    or of mixing? how does training at large $\sigma$ put noisy data, and therefore training
    signal, everywhere?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: sliced score matching, and what DSM really learns

    two more networks on the same data: one trained with your
    `sliced_score_matching_loss` (one projection, no noise, the network called at a fixed
    dummy $\sigma = 1$), and one with DSM at a large single level, $\sigma = 1$. we measure
    each network's error against the *clean* score of the mixture and against the score
    blurred at $\sigma = 1$, averaged over data points, and the training time per step.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "which will be closer to the clean score: sliced score matching or DSM at sigma = 0.1? and how far will DSM at "
        "sigma = 1 be from the clean score?"
    )
    p3
    return (p3,)


@app.cell
def _(X, nets, p3, secs):
    ui.gate(p3)
    _Xt = torch.tensor(X, dtype=torch.float32)
    torch.manual_seed(0)
    _ssm = ex.NoiseConditionalScore(2, hidden_dim=128, n_layers=3)
    _opt = torch.optim.Adam(_ssm.parameters(), lr=1e-3)
    _g = torch.Generator().manual_seed(0)
    _t0 = time.perf_counter()
    for _ in range(ITERS):
        _xb = _Xt[torch.randint(len(_Xt), (256,), generator=_g)]
        _loss = ex.sliced_score_matching_loss(lambda v: _ssm(v, 1.0), _xb, n_projections=1, generator=_g)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    _t_ssm = time.perf_counter() - _t0
    torch.manual_seed(0)
    _big = ex.NoiseConditionalScore(2, hidden_dim=128, n_layers=3)
    _t0 = time.perf_counter()
    ex.fit_dsm(_big, X, [1.0], n_iters=ITERS, batch_size=256, lr=1e-3, seed=0)
    _t_big = time.perf_counter() - _t0
    _xe = torch.tensor(sample_mixture(5000, 8)[0], dtype=torch.float32)
    _rel = lambda s, ref: float((s - ref).norm(dim=1).pow(2).mean() / ref.norm(dim=1).pow(2).mean())
    _rows = []
    with torch.no_grad():
        for _name, _fn, _t in (("sliced SM (no noise)", lambda v: _ssm(v, 1.0), _t_ssm),
                               ("DSM, sigma = 0.1", lambda v: nets["one level (0.1)"](v, 0.1), secs["one level (0.1)"]),
                               ("DSM, sigma = 1", lambda v: _big(v, 1.0), _t_big)):
            _s = _fn(_xe)
            _rows.append(f"| {_name} | {_rel(_s, mixture_score(_xe, 0.0)):.4f} | {_rel(_s, mixture_score(_xe, 1.0)):.4f} | "
                         f"{1e3 * _t / ITERS:.2f} |")
    mo.md("relative squared error $\\mathbb{E}\\lVert s_\\theta - s\\rVert^2 / \\mathbb{E}\\lVert s\\rVert^2$ on 5,000 fresh data "
          "points\n\n| training | vs the clean score | vs the score blurred at sigma = 1 | ms per step |\n|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for unit-variance components, what is the ratio between the clean score
    and the score blurred at $\sigma = 1$ near a mode (on paper 3), and does the DSM row
    match it? why is sliced score matching's estimator noisier than DSM's, and where does
    its extra cost per step come from? when would you still choose it?
    """)
    return


if __name__ == "__main__":
    app.run()
