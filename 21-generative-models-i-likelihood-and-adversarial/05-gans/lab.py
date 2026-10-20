import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.generative import cached_feature_net, fid_from_features  # yours, from topic 01
    from mlfs.torchnn import MLP, get_device  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    torch.set_num_threads(min(4, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)
    STD = 0.02  # each of the 8 gaussians
    ANG = 2 * np.pi * np.arange(8) / 8
    CENTERS = 2.0 * np.c_[np.cos(ANG), np.sin(ANG)]
    STEPS = 1500 if OFFLINE else 3000
    LATENT = 16
    SEEDS = (0, 1, 2)


@app.function
def train_toy_gan(*, non_saturating=True, steps=STEPS, seed=0, log_every=500):
    """an MLP GAN (noise 16 -> 256 -> 256 -> 2 vs 2 -> 256 -> 256 -> 1) on 8 gaussians, with your gan_step and Adam.
    returns (G, history rows (step, modes covered, high-quality fraction), snapshots {step: samples})."""
    torch.manual_seed(seed)
    G = MLP(LATENT, [256, 256], 2)
    D = MLP(2, [256, 256], 1)
    og = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    hist, snaps = [], {}
    for step in range(1, steps + 1):
        real = torch.tensor(datasets.eight_gaussians(256, std=STD, rng=rng), dtype=torch.float32)
        ex.gan_step(G, D, og, od, real, latent_dim=LATENT, non_saturating=non_saturating, generator=gen)
        if step % log_every == 0:
            with torch.no_grad():
                x = G(torch.randn(2000, LATENT, generator=torch.Generator().manual_seed(999))).numpy()
            n, q, _ = ex.mode_coverage(x, CENTERS, STD)
            hist.append((step, n, q))
            snaps[step] = x
    return G, np.array(hist), snaps


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # GANs · lab

    experiments 1–2 train small MLP GANs with **your** `gan_step` and losses on the classic
    toy: 8 gaussians (std {STD}) on a circle of radius 2, scored with your `mode_coverage`
    (a sample is high quality within 3 std of a mode; a mode is covered if it gets 1% of
    the samples). experiment 3 trains your `DCGANGenerator` and `DCGANDiscriminator` on
    MNIST. {'**offline: shorter runs and synthetic images, so the numbers are only a smoke test.**' if OFFLINE else ''}
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 1: chasing modes

    the non-saturating GAN, {STEPS} steps of batch 256, three seeds. every 500 steps: the
    number of modes covered by 2,000 samples from the same fixed noise. the plots show
    seed 0's samples at four moments.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how many of the 8 modes will the GAN cover at the end of training? will the number go up steadily, or "
        "jump around?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    ns_runs = {s: train_toy_gan(seed=s) for s in SEEDS}
    _secs = (time.perf_counter() - _t0) / len(SEEDS)
    _snaps = ns_runs[0][2]
    _show = [k for k in sorted(_snaps)][:: max(1, len(_snaps) // 4)][:4]
    _fig, _axes = plt.subplots(1, len(_show) + 1, figsize=(3 * (len(_show) + 1), 3))
    for _ax, _k in zip(_axes, _show):
        _ax.scatter(*CENTERS.T, c="red", s=30, marker="x")
        _ax.scatter(*_snaps[_k].T, s=2, alpha=0.4)
        _ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=f"step {_k}")
    for _s, (_, _h, _) in ns_runs.items():
        _axes[-1].plot(_h[:, 0], _h[:, 1], "o-", ms=3, label=f"seed {_s}")
    _axes[-1].set(xlabel="step", ylabel="modes covered", ylim=(0, 8.5), title="coverage")
    _axes[-1].legend(fontsize=7)
    _fig.tight_layout()
    _rows = "\n".join(f"| {_s} | {' '.join(str(int(v)) for v in _h[:, 1])} | {_h[-1, 2]:.2f} |"
                      for _s, (_, _h, _) in ns_runs.items())
    mo.vstack([_fig, mo.md(f"about {_secs:.0f} s per run\n\n| seed | modes covered every 500 steps | final "
                           f"high-quality fraction |\n|---|---|---|\n" + _rows)])
    return (ns_runs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the generator is rewarded for samples the *current* discriminator
    believes are real. if it concentrates on one or two modes, what does the discriminator
    learn next, and where does the generator move then? why is "cover all 8 modes" not
    what one step of gradient descent on the generator loss asks for? what does the
    coverage curve say about judging a GAN by its last checkpoint?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the saturating loss

    the same three seeds with the original minimax generator loss, $\log(1 - D(G(z)))$,
    instead of $-\log D(G(z))$. before training, we also pretrain the discriminator for 50
    steps against the untrained generator and measure the generator's gradient norm under
    each loss.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the two losses have the same fixed point. after a 50-step head start for the discriminator, how much smaller "
        "is the saturating loss's gradient? and how many modes will the saturating GAN cover at the end?"
    )
    p2
    return (p2,)


@app.cell
def _(ns_runs, p2):
    ui.gate(p2)
    torch.manual_seed(0)
    _G, _D = MLP(LATENT, [256, 256], 2), MLP(2, [256, 256], 1)
    _od = torch.optim.Adam(_D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    _rng = np.random.default_rng(0)
    for _ in range(50):
        _real = torch.tensor(datasets.eight_gaussians(256, std=STD, rng=_rng), dtype=torch.float32)
        _od.zero_grad()
        ex.discriminator_loss(_D(_real).view(-1), _D(_G(torch.randn(256, LATENT)).detach()).view(-1)).backward()
        _od.step()
    _norms = {}
    for _flag in (True, False):
        _G.zero_grad()
        ex.generator_loss(_D(_G(torch.randn(1024, LATENT, generator=torch.Generator().manual_seed(1)))).view(-1),
                          non_saturating=_flag).backward()
        _norms[_flag] = torch.sqrt(sum((p.grad ** 2).sum() for p in _G.parameters())).item()
    with torch.no_grad():
        _dg = torch.sigmoid(_D(_G(torch.randn(1024, LATENT)))).mean().item()
    sat_runs = {s: train_toy_gan(non_saturating=False, seed=s) for s in SEEDS}
    _rows = []
    for _name, _runs in (("non-saturating", ns_runs), ("saturating", sat_runs)):
        _final = np.array([_runs[s][1][-1, 1:] for s in SEEDS])
        _rows.append(f"| {_name} | {', '.join(str(int(v)) for v in _final[:, 0])} | "
                     f"{', '.join(f'{v:.2f}' for v in _final[:, 1])} |")
    _fig, _axes = plt.subplots(1, len(SEEDS), figsize=(3 * len(SEEDS), 3))
    for _ax, _s in zip(_axes, SEEDS):
        _last = max(sat_runs[_s][2])
        _ax.scatter(*CENTERS.T, c="red", s=30, marker="x")
        _ax.scatter(*sat_runs[_s][2][_last].T, s=2, alpha=0.4)
        _ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=f"saturating, seed {_s}")
    _fig.tight_layout()
    mo.vstack([mo.md(f"after 50 discriminator steps, mean D(G(z)) = {_dg:.3f}; generator gradient norm: "
                     f"non-saturating **{_norms[True]:.3g}**, saturating **{_norms[False]:.3g}** "
                     f"(ratio {_norms[True] / max(_norms[False], 1e-12):.0f})"),
               _fig,
               mo.md("| generator loss | final modes covered (3 seeds) | final high-quality fraction |\n|---|---|---|\n"
                     + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** write $D = \sigma(\ell)$ and differentiate both losses with respect to
    the logit $\ell$: which one vanishes when $D(G(z)) \to 0$? early in training the
    discriminator wins easily: what does that do to the saturating generator's learning
    signal? compare the final coverage with experiment 1's: did the saturating GAN fail to
    learn, learn slowly, or end up somewhere else?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: a DCGAN on MNIST

    your DCGAN (latent 64, base width 32, $28 \times 28$) trained with `gan_step` and Adam
    (lr $2 \times 10^{-4}$, $\beta_1 = 0.5$) for 2 epochs on 20,000 digits scaled to
    $[-1, 1]$. we show 32 samples, the FID of 2,000 samples, and which digits the samples
    are, according to topic 01's `FeatureNet`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "will the DCGAN draw all ten digits about equally often? which digit, if any, do you expect to be "
        "over-represented?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _d = datasets.mnist()
    _gray = torch.tensor(_d["X_train"][:, None] / 255.0, dtype=torch.float32)
    _dev = torch.device("cpu") if OFFLINE else get_device()
    _featnet = cached_feature_net(_gray, _d["y_train"], name="mnist", epochs=2, seed=0, device=_dev).to("cpu")
    _n = min(20_000, len(_gray))
    _x = _gray[:_n] * 2 - 1
    _x = _x.to(_dev)
    torch.manual_seed(0)
    _G = ex.DCGANGenerator(64, 1, base_channels=32, img_size=28).to(_dev)
    _D = ex.DCGANDiscriminator(1, base_channels=32, img_size=28).to(_dev)
    _og = torch.optim.Adam(_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    _odd = torch.optim.Adam(_D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    _gen = torch.Generator().manual_seed(0)
    _t0 = time.perf_counter()
    for _ in range(1 if OFFLINE else 2):
        _perm = torch.randperm(_n, generator=_gen)
        for _i in range(0, _n - 63, 64):
            ex.gan_step(_G, _D, _og, _odd, _x[_perm[_i:_i + 64]], latent_dim=64, generator=_gen)
    _secs = time.perf_counter() - _t0
    _G.eval()
    with torch.no_grad():
        _s = ((_G(torch.randn(2000, 64, generator=torch.Generator().manual_seed(1)).to(_dev)) + 1) / 2).cpu()
    _test = torch.tensor(_d["X_test"][:2000, None] / 255.0, dtype=torch.float32)
    _fid = fid_from_features(_featnet.embed(_test), _featnet.embed(_s))
    with torch.no_grad():
        _cls = _featnet(_s).argmax(1).numpy()
    _hist = np.bincount(_cls, minlength=10) / len(_cls)
    _ent = float(-(_hist[_hist > 0] * np.log(_hist[_hist > 0])).sum())
    _fig, _axes = plt.subplots(2, 1, figsize=(8, 4.6), gridspec_kw=dict(height_ratios=[2, 1]))
    _grid = _s[:32, 0].view(4, 8, 28, 28).permute(0, 2, 1, 3).reshape(4 * 28, 8 * 28)
    _axes[0].imshow(_grid, cmap="gray_r", vmin=0, vmax=1)
    _axes[0].axis("off")
    _axes[1].bar(np.arange(10), _hist)
    _axes[1].axhline(0.1, color="gray", ls="--")
    _axes[1].set(xticks=np.arange(10), xlabel="digit (FeatureNet's label for each sample)", ylabel="fraction")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"trained in {_secs:.0f} s · FID of 2,000 samples vs 2,000 test digits: **{_fid:.1f}** · "
                           f"label entropy {_ent:.2f} nats (uniform: {np.log(10):.2f})")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a classifier's labels are a cheap mode detector for MNIST: a collapsed
    generator would put most samples in a few classes. what does the histogram say about
    this generator's coverage? why is the classifier's verdict on a *bad* sample (a blob
    that is no digit) unreliable, and how does that limit this check? compare the FID with
    topic 01's numbers for flawed "models" (dropped classes, noise): which flaw does this
    generator resemble more?
    """)
    return


if __name__ == "__main__":
    app.run()
