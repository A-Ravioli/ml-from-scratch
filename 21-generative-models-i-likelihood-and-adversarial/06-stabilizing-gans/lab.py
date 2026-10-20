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
    from mlfs.generative import mode_coverage  # yours, from topic 05
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    torch.set_num_threads(4)
    STD = 0.02
    ANG = 2 * np.pi * np.arange(8) / 8
    CENTERS = 2.0 * np.c_[np.cos(ANG), np.sin(ANG)]
    LATENT = 16
    CRITIC_STEPS = 300 if OFFLINE else 1000
    WGAN_STEPS = 200 if OFFLINE else 1500  # generator steps; 5 critic steps each
    SEEDS = (0, 1)


@app.function
def ring(n, scale, rng):
    """the 8 gaussians, with the circle's radius multiplied by `scale`."""
    return torch.tensor(scale * datasets.eight_gaussians(n, std=STD, rng=rng), dtype=torch.float32)


@app.function
def sn_mlp():
    """a 2 -> 256 -> 256 -> 1 MLP built from your SNLinear layers."""
    return nn.Sequential(ex.SNLinear(2, 256), nn.ReLU(), ex.SNLinear(256, 256), nn.ReLU(), ex.SNLinear(256, 1))


@app.function
def coverage(G, n_std=3.0):
    """(modes covered, high-quality fraction) of 2,000 samples from G at radius n_std * STD."""
    with torch.no_grad():
        x = G(torch.randn(2000, LATENT, generator=torch.Generator().manual_seed(999))).numpy()
    n, q, _ = mode_coverage(x, CENTERS, STD, n_std=n_std)
    return n, q


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # stabilizing GANs · lab

    every experiment uses **your** `critic_loss`, `gradient_penalty`, `clip_weights_`,
    `SNLinear` and `train_wgan` (and topic 05's `mode_coverage`) on the 8
    gaussians of topic 05's lab (std {STD}, radius 2). {'**offline: shortened runs, so the numbers are only a smoke test.**' if OFFLINE else ''}
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 1: what does the critic measure?

    two fixed distributions whose Wasserstein-1 distance is known: the 8 gaussians, and
    the same 8 gaussians on a circle of radius 1. moving every point radially by 1 is the
    optimal transport plan, so $W_1 \approx 1$. three critics (MLPs 2 → 256 → 256 → 1)
    are trained for {CRITIC_STEPS} steps to maximize $\mathbb{{E}}[f(\text{{real}})] -
    \mathbb{{E}}[f(\text{{fake}})]$, each with its own way of keeping $f$ 1-Lipschitz:
    **weight clipping** ($c = 0.01$, RMSprop, as in the original WGAN), the **gradient
    penalty** ($\lambda = 10$, Adam), and **spectral normalization** (your `SNLinear`
    layers, Adam). we report each critic's estimate of $W_1$ and the norm of its gradient
    at random interpolates.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "all three critics are (supposed to be) 1-Lipschitz. will their estimates of W1 (true value about 1) agree? "
        "which one, if any, will be far off?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _results = {}
    _t0 = time.perf_counter()
    for _name in ("weight clipping", "gradient penalty", "spectral norm"):
        torch.manual_seed(0)
        _f = sn_mlp() if _name == "spectral norm" else MLP(2, [256, 256], 1)
        _opt = (torch.optim.RMSprop(_f.parameters(), lr=5e-5) if _name == "weight clipping"
                else torch.optim.Adam(_f.parameters(), lr=1e-4, betas=(0.5, 0.9)))
        _est = []
        for _ in range(CRITIC_STEPS):
            _real, _fake = ring(256, 1.0, _rng), ring(256, 0.5, _rng)
            _loss = ex.critic_loss(_f(_real).view(-1), _f(_fake).view(-1))
            if _name == "gradient penalty":
                _loss = _loss + 10.0 * ex.gradient_penalty(_f, _real, _fake)
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
            if _name == "weight clipping":
                ex.clip_weights_(_f, 0.01)
            _est.append(-ex.critic_loss(_f(_real).view(-1), _f(_fake).view(-1)).item())
        _real, _fake = ring(2000, 1.0, _rng), ring(2000, 0.5, _rng)
        _eps = torch.rand(2000, 1)
        _x = (_eps * _real + (1 - _eps) * _fake).requires_grad_(True)
        (_g,) = torch.autograd.grad(_f(_x).sum(), _x)
        _results[_name] = (np.array(_est), _g.norm(dim=1).numpy())
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for _name, (_est, _gn) in _results.items():
        _a1.plot(np.convolve(_est, np.ones(25) / 25, mode="valid"), label=_name)
        _a2.hist(np.maximum(_gn, 1e-5), bins=np.logspace(-5, 0.5, 60), alpha=0.6, label=_name)
    _a1.axhline(1.0, color="k", ls="--", lw=0.8, label="true W1 ≈ 1")
    _a1.set(xlabel="critic step", ylabel="estimate of W1 (moving average)", ylim=(-0.05, 1.4))
    _a2.set(xlabel="‖∇f‖ at random interpolates", ylabel="count", xscale="log")
    _a1.legend(fontsize=8)
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {k} | {np.mean(e[-100:]):.3f} | {np.median(g):.3f} | {np.percentile(g, 10):.3f} – "
                      f"{np.percentile(g, 90):.3f} |" for k, (e, g) in _results.items())
    mo.vstack([_fig, mo.md(f"{time.perf_counter() - _t0:.0f} s\n\n| critic | W1 estimate (last 100 steps) | median ‖∇f‖ "
                           "| 10th – 90th percentile |\n|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the dual form $W_1 = \sup_{\|f\|_L \le 1} \mathbb{E}[f(\text{real})] -
    \mathbb{E}[f(\text{fake})]$ is attained by a critic whose gradient norm is 1 along the
    transport paths. clipping every weight to $\pm 0.01$ bounds the Lipschitz constant, but
    by how much? (multiply the bounds of three layers of width 256.) why does spectral
    normalization, which makes every *layer* 1-Lipschitz, still give a critic whose
    gradient is well below 1 almost everywhere? which of the three estimates could you
    use as a distance, and which only up to an unknown scale?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the gradient far from the data

    the generator learns only through the gradient of its loss with respect to its
    samples. here the "generator" is frozen: its samples are the 8 gaussians on a circle
    of radius $2s$, for $s$ from 1.25 to 4, so they lie further and further outside the
    data. for each $s$, a GAN discriminator (cross-entropy, Adam) and a WGAN-GP critic
    are trained for 300 steps against those samples, and we measure the average norm of
    the gradient each generator loss sends to a sample: $\nabla_x$ of the saturating loss
    $\log(1 - D(x))$, of the non-saturating loss $-\log D(x)$, and of the Wasserstein loss
    $-f(x)$ (per sample, i.e. multiplied by the batch size).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "as the generated samples move further from the data, what happens to the gradient each generator loss "
        "provides: saturating, non-saturating, Wasserstein?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _scales = (1.25, 1.5, 2.0, 3.0, 4.0)
    _rows, _res = [], {"saturating": [], "non-saturating": [], "Wasserstein (GP critic)": []}
    _t0 = time.perf_counter()
    for _s in _scales:
        _rng = np.random.default_rng(1)
        torch.manual_seed(0)
        _D, _C = MLP(2, [256, 256], 1), MLP(2, [256, 256], 1)
        _od = torch.optim.Adam(_D.parameters(), lr=1e-3, betas=(0.5, 0.999))
        _oc = torch.optim.Adam(_C.parameters(), lr=1e-3, betas=(0.5, 0.9))
        for _ in range(CRITIC_STEPS // 5):
            _real, _fake = ring(256, 1.0, _rng), ring(256, _s, _rng)
            _od.zero_grad()
            torch.nn.functional.softplus(-_D(_real)).mean().add(torch.nn.functional.softplus(_D(_fake)).mean()).backward()
            _od.step()
            _oc.zero_grad()
            (ex.critic_loss(_C(_real).view(-1), _C(_fake).view(-1)) + 10.0 * ex.gradient_penalty(_C, _real, _fake)).backward()
            _oc.step()
        _x = ring(1000, _s, _rng).requires_grad_(True)
        _l = _D(_x).view(-1)
        _gs = torch.autograd.grad(-torch.nn.functional.softplus(_l).sum(), _x, retain_graph=True)[0]  # log(1 - D)
        _gn = torch.autograd.grad(torch.nn.functional.softplus(-_l).sum(), _x)[0]  # -log D
        _gw = torch.autograd.grad((-_C(_x)).sum(), _x)[0]
        for _k, _g in zip(_res, (_gs, _gn, _gw)):
            _res[_k].append(_g.norm(dim=1).mean().item())
        _rows.append(f"| {_s:g} | {2 * (_s - 1):.1f} | {torch.sigmoid(_l).mean().item():.2e} | "
                     + " | ".join(f"{_res[_k][-1]:.3g}" for _k in _res) + " |")
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    for _k, _v in _res.items():
        _ax.semilogy([2 * (s - 1) for s in _scales], np.maximum(_v, 1e-12), "o-", label=_k)
    _ax.set(xlabel="distance from the data (radial)", ylabel="mean ‖∇ loss‖ per sample")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"{time.perf_counter() - _t0:.0f} s\n\n| s | distance | mean D(x) on the samples | saturating | "
                           "non-saturating | Wasserstein |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a discriminator that separates the two sets perfectly drives $D(x) \to
    0$ on the samples. which of the two GAN losses then multiplies its gradient by a
    vanishing factor (topic 05, on paper 2)? the non-saturating gradient is $(1 - D)$
    times the slope of the logit: what decides that slope far from the data, and is
    anything keeping it under control? the Lipschitz critic's slope is pinned near 1 by
    the penalty: why is that exactly the gradient you'd want a generator to receive?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 3: training with the stabilizers

    a WGAN-GP on topic 05's data and networks, trained by your `train_wgan` ({WGAN_STEPS}
    generator steps × 5 critic steps, lr $10^{{-3}}$, seeds {', '.join(map(str, SEEDS))}),
    with a `callback` that records the critic's Wasserstein estimate and the fraction of
    samples within 10 std of a mode. the final generators are scored at two radii: within 3
    std of a mode (sharp samples, topic 05's criterion) and within 10 std (roughly in the
    right place). for comparison, topic 05's non-saturating GAN on the same data (5,000
    steps, seeds 0–2) covered 5, 2 and 6 modes at 3 std, with 41%, 89% and 72% of its
    samples there.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "compared with topic 05's GAN, will the WGAN-GP cover more modes at the sharp radius (3 std)? at the loose "
        "one (10 std)? and does the critic's estimate fall as the samples improve?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _t0 = time.perf_counter()
    _final, _curves = [], {}
    for _seed in SEEDS:
        torch.manual_seed(_seed)
        _G, _C = MLP(LATENT, [256, 256], 2), MLP(2, [256, 256], 1)
        _rng, _gen = np.random.default_rng(_seed), torch.Generator().manual_seed(_seed)
        _curve = []

        def _record(step, hist, _G=_G, _curve=_curve):
            if step % max(1, WGAN_STEPS // 10) == 0:
                _curve.append((step, float(np.mean(hist["wasserstein"][-20:])), coverage(_G, 10.0)[1]))

        ex.train_wgan(_G, _C, lambda: ring(256, 1.0, _rng), n_steps=WGAN_STEPS, latent_dim=LATENT, lr=1e-3,
                      generator=_gen, callback=_record)
        _curves[_seed] = np.array(_curve)
        _final.append(("WGAN-GP", _seed, coverage(_G, 3.0), coverage(_G, 10.0)))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax2 = _ax.twinx()
    for _s, _c in _curves.items():
        _ax.plot(_c[:, 0], _c[:, 1], "o-", color=f"C{_s}", label=f"seed {_s}: W estimate")
        _ax2.plot(_c[:, 0], _c[:, 2], "s--", color=f"C{_s}", alpha=0.5)
    _ax.set(xlabel="generator step", ylabel="Wasserstein estimate (solid)", title="WGAN-GP")
    _ax2.set(ylim=(0, 1), ylabel="fraction within 10 std (dashed)")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _rows = "\n".join(f"| {m} | {s} | {a[0]} | {a[1]:.2f} | {b[0]} | {b[1]:.2f} |" for m, s, a, b in _final)
    mo.vstack([_fig, mo.md(f"{time.perf_counter() - _t0:.0f} s\n\n| method | seed | modes (3 std) | quality (3 std) | "
                           "modes (10 std) | quality (10 std) |\n|---|---|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compare the two methods' failure modes: what does a collapsed GAN get
    right, and what does a WGAN-GP that hasn't converged get right? when the critic's
    estimate falls, does sample quality rise (look at each seed separately)? the
    Wasserstein critic needs to stay near its optimum for its gradient to point along
    transport paths: what in this budget (5 critic steps, a learning rate of $10^{-3}$, a
    few thousand updates) works against that? what would you change first?
    """)
    return


if __name__ == "__main__":
    app.run()
