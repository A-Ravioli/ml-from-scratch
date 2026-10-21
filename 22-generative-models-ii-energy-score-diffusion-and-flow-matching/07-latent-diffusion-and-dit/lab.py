import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.diffusion import TimeUNet, fit_flow_matching, sample_flow  # yours, from topics 03 and 05
    from mlfs.generative import ConvVAE, cached_feature_net, fid_from_features  # yours, from ch 21
    from mlfs.torchnn import count_parameters, get_device, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    DEVICE = torch.device("cpu") if OFFLINE else get_device()
    # offline (tests, CI): a smoke test of every code path on the synthetic stand-in; online: the real experiment
    VAE_STEPS, GEN_STEPS, N_SAMPLES, BATCH = (60, 60, 64, 32) if OFFLINE else (1500, 800, 1000, 128)
    RUN_ALL = os.environ.get("MLFS_LAB_RUN_ALL", "") == "1"  # press every run button (used to record the notes)
    _d = datasets.mnist()
    X = torch.tensor(_d["X_train"][:, None] / 255.0, dtype=torch.float32)
    Y = _d["y_train"]
    featnet = cached_feature_net(X, Y, name="mnist", epochs=2, seed=0, device=DEVICE).to("cpu")
    REAL = featnet.embed(torch.tensor(_d["X_test"][:2000, None] / 255.0, dtype=torch.float32))
    REF_UNET = ("reference run (`mps`, 800 steps each, 1,000 samples):\n\n"
                "| | parameters | training | ms per step | sampling 1,000 | FID |\n|---|---|---|---|---|---|\n"
                "| latent DiT | 1,254,532 | 54 s | 67 | 4.2 s | 191.5 |\n"
                "| pixel U-Net | 390,273 | 156 s | 195 | 4.4 s | **59.9** |")
    REF_BETA = ("reference run (`mps`, 800 DiT steps):\n\n"
                "| VAE beta | reconstruction MSE | reconstruction FID | latent std | latent DiT FID |\n"
                "|---|---|---|---|---|\n| 0.001 | 0.0016 | 11.15 | 0.900 | 191.5 |\n"
                "| 0.1 | 0.0052 | 22.18 | 0.497 | 228.6 |")


@app.function
def fid(images):
    """course FID of images in [0, 1] (N, 1, 28, 28) against 2,000 test digits."""
    return float(fid_from_features(REAL, featnet.embed(images.clamp(0, 1))))


@app.function
def train_vae(beta, steps=VAE_STEPS, seed=0):
    """a ConvVAE (4 latent channels, width 32, unit-variance Gaussian decoder) trained with Adam on random batches."""
    torch.manual_seed(seed)
    vae = ConvVAE(1, latent_channels=4, base_channels=32).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for _ in range(steps):
        loss, rec, kl = vae.loss(X[torch.randint(len(X), (BATCH,), generator=g)].to(DEVICE), beta=beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return vae.eval(), time.perf_counter() - t0, kl.item()


@app.function
def latent_flow(vae, seed=0):
    """encode the training digits, normalize, train your DiT with flow matching; returns a dict of results."""
    z, _ = ex.encode_latents(vae, make_loader(X.numpy(), Y, batch_size=500, shuffle=False), device=DEVICE)
    std = float(z.std())
    torch.manual_seed(seed)
    dit = ex.DiT(7, 1, 4, d_model=128, n_layers=4, n_heads=4).to(DEVICE)
    t0 = time.perf_counter()
    losses = fit_flow_matching(dit, (z / std).to(DEVICE), n_iters=GEN_STEPS, batch_size=BATCH, lr=1e-3, seed=seed)
    secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    zs = sample_flow(dit, (N_SAMPLES, 4, 7, 7), n_steps=20, generator=torch.Generator().manual_seed(1), device=DEVICE)
    imgs = ex.decode_latents(vae, zs.cpu(), scale=std, device=DEVICE).clamp(0, 1)
    t_sample = time.perf_counter() - t0
    recon = ex.decode_latents(vae, z[:N_SAMPLES], device=DEVICE).clamp(0, 1)
    return dict(model=dit, std=std, losses=losses, secs=secs, t_sample=t_sample, imgs=imgs, recon=recon,
                fid=fid(imgs), fid_recon=fid(recon), mse=float(((recon - X[:N_SAMPLES]) ** 2).mean()))


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # latent diffusion & DiT · lab

    every experiment uses **your** `encode_latents`, `decode_latents` and `DiT`, with
    chapter 21's `ConvVAE` and this chapter's `fit_flow_matching` and `sample_flow`.
    {'**offline**: the synthetic stand-in and tiny budgets, a smoke test of every code path; the numbers mean nothing.'
     if OFFLINE else f'MNIST on **{DEVICE}** (about 2 minutes on a laptop GPU by default; two buttons add about 3 and 1.5 minutes).'} FIDs are course-internal (chapter 21's
    `FeatureNet`, not Inception). compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 1: what an autoencoder's latents look like

    a `ConvVAE` with 4 latent channels ($28 \times 28 \to 4 \times 7 \times 7$, 4× fewer numbers)
    trained for {VAE_STEPS} steps with a small KL weight, $\beta = 10^{{-3}}$. we look at the
    four latent channels of a few digits, at the reconstructions, and at the scale of the
    latents.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "will the 4 latent channels of a digit look like noise, or like small versions of the digit? what standard "
        "deviation will the latents have, roughly (with beta = 1e-3)?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    vae_small, _t_vae, _kl = train_vae(1e-3)
    with torch.no_grad():
        _mu, _lv = vae_small.encode(X[:6].to(DEVICE))
        _rec = vae_small.decode(_mu).cpu().clamp(0, 1)
    _mu = _mu.cpu()
    _fig, _axes = plt.subplots(6, 6, figsize=(6.5, 6.5))
    for _i in range(6):
        _axes[_i, 0].imshow(X[_i, 0], cmap="gray")
        for _c in range(4):
            _axes[_i, 1 + _c].imshow(_mu[_i, _c], cmap="coolwarm")
        _axes[_i, 5].imshow(_rec[_i, 0], cmap="gray", vmin=0, vmax=1)
    for _a, _t in zip(_axes[0], ["digit", "z1", "z2", "z3", "z4", "reconstruction"]):
        _a.set_title(_t, fontsize=8)
    for _a in _axes.flat:
        _a.set_axis_off()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"VAE training {_t_vae:.0f} s · final KL {_kl:.0f} nats per digit · latent std overall "
        f"**{float(_mu.std()):.3f}**, per channel {_mu.std((0, 2, 3)).numpy().round(3).tolist()} (6 digits) · posterior std "
        f"{float((0.5 * _lv).exp().mean()):.3f}")])
    return (vae_small,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with a tiny KL weight, what does the VAE's latent space look like: a
    Gaussian blob, or a compressed image? why would a large $\beta$ change that, and what
    would it cost? the posterior standard deviation is much smaller than the latents'
    spread: why does that justify encoding the *means* only?
    """)
    return


@app.function
def pixel_flow(seed=0):
    """the baseline: your flow matching on the pixels in [-1, 1] with a TimeUNet (base 16), GEN_STEPS steps."""
    torch.manual_seed(seed)
    unet = TimeUNet(1, base_channels=16, channel_mults=(1, 2, 2)).to(DEVICE)
    t0 = time.perf_counter()
    fit_flow_matching(unet, (X * 2 - 1).to(DEVICE), n_iters=GEN_STEPS, batch_size=BATCH, lr=1e-3, seed=seed)
    secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    xs = sample_flow(unet, (N_SAMPLES, 1, 28, 28), n_steps=20, generator=torch.Generator().manual_seed(1), device=DEVICE)
    imgs = ((xs.cpu() + 1) / 2).clamp(0, 1)
    return dict(model=unet, secs=secs, t_sample=time.perf_counter() - t0, imgs=imgs, fid=fid(imgs))


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2: latent DiT against pixel U-Net

    your `DiT` (4 blocks of width 128, patch size 1, 49 tokens) trained with flow matching
    for {GEN_STEPS} steps of batch {BATCH} on the normalized latents of experiment 1's VAE,
    sampled with 20 Euler steps and decoded. {N_SAMPLES:,} samples, scored against test digits;
    the VAE's own reconstructions give the floor a latent model can't beat. the button below
    adds the baseline: a `TimeUNet` (base width 16, two downsamplings) trained the same way on
    the pixels, scaled to $[-1, 1]$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which generator will be faster per training step, the latent DiT or the pixel U-Net, and which will have the "
        "better FID? how close will the latent model get to the FID of the VAE's reconstructions?"
    )
    p2
    return (p2,)


@app.cell
def _(p2, vae_small):
    ui.gate(p2)
    lat = latent_flow(vae_small)
    _fig, _axes = plt.subplots(2, 1, figsize=(8, 2.3))
    for _ax, (_t, _im) in zip(_axes, (("latent DiT samples", lat["imgs"]), ("VAE reconstructions (the floor)", lat["recon"]))):
        _ax.imshow(_im[:16, 0].permute(1, 0, 2).reshape(28, -1), cmap="gray")
        _ax.set_axis_off()
        _ax.set_title(_t, fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"| | parameters | training ({GEN_STEPS} steps) | ms per step | sampling {N_SAMPLES:,} | FID |\n|---|---|---|---|---|---|\n"
        f"| latent DiT (+ VAE decoder) | {count_parameters(lat['model']):,} | {lat['secs']:.0f} s | "
        f"{1e3 * lat['secs'] / GEN_STEPS:.1f} | {lat['t_sample']:.1f} s | **{lat['fid']:.2f}** |\n"
        f"| VAE reconstructions | | | | | {lat['fid_recon']:.2f} |\n| real training digits | | | | | {fid(X[:N_SAMPLES]):.2f} |"
        f"\n\nlatent std before normalization {lat['std']:.3f}")])
    return (lat,)


@app.cell
def _():
    unet_button = mo.ui.run_button(label=f"add the pixel U-Net baseline ({GEN_STEPS} steps, a few minutes on a laptop GPU)")
    unet_button
    return (unet_button,)


@app.cell
def _(lat, unet_button):
    if unet_button.value or RUN_ALL:
        _pix = pixel_flow()
        _fig, _ax = plt.subplots(figsize=(8, 1.1))
        _ax.imshow(_pix["imgs"][:16, 0].permute(1, 0, 2).reshape(28, -1), cmap="gray")
        _ax.set_axis_off()
        _ax.set_title("pixel U-Net samples", fontsize=8)
        _out = mo.vstack([_fig, mo.md(
            f"| | parameters | training | ms per step | sampling {N_SAMPLES:,} | FID |\n|---|---|---|---|---|---|\n"
            f"| latent DiT | {count_parameters(lat['model']):,} | {lat['secs']:.0f} s | {1e3 * lat['secs'] / GEN_STEPS:.1f} | "
            f"{lat['t_sample']:.1f} s | {lat['fid']:.2f} |\n"
            f"| pixel U-Net | {count_parameters(_pix['model']):,} | {_pix['secs']:.0f} s | "
            f"{1e3 * _pix['secs'] / GEN_STEPS:.1f} | {_pix['t_sample']:.1f} s | **{_pix['fid']:.2f}** |")])
    else:
        _out = mo.md(REF_UNET + "\n\npress the button to run it yourself.")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which part of the latent model's FID is the autoencoder's fault and which
    the DiT's? why is a step of the latent model cheaper even though the DiT is a
    transformer (count the numbers each model processes)? at the scale of a 512 × 512
    photograph, which way would the comparison tip, and why?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the autoencoder's KL weight

    the same pipeline with a VAE trained at $\beta = 0.1$ (a hundred times experiment 1's):
    smoother, more Gaussian latents that carry less information. we compare reconstruction
    error, reconstruction FID, and the FID of the latent DiT's samples. it trains a second VAE
    and DiT, so it runs when you press the button.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with beta = 0.1 instead of 1e-3, will the latent DiT's samples get better (easier latents to model) or worse "
        "(blurrier decoder)? by how much will the reconstruction FID change?"
    )
    p3
    return (p3,)


@app.cell
def _():
    beta_button = mo.ui.run_button(label="run experiment 3 (another VAE and DiT, a few minutes on a laptop GPU)")
    beta_button
    return (beta_button,)


@app.cell
def _(beta_button, lat, p3):
    ui.gate(p3)
    if beta_button.value or RUN_ALL:
        _vae_big, _, _kl = train_vae(0.1)
        _hi = latent_flow(_vae_big)
        _out = mo.md(
            "| VAE beta | reconstruction MSE | reconstruction FID | latent std | latent DiT FID |\n|---|---|---|---|---|\n"
            f"| 0.001 | {lat['mse']:.4f} | {lat['fid_recon']:.2f} | {lat['std']:.3f} | {lat['fid']:.2f} |\n"
            f"| 0.1 | {_hi['mse']:.4f} | {_hi['fid_recon']:.2f} | {_hi['std']:.3f} | {_hi['fid']:.2f} |")
    else:
        _out = mo.md(REF_BETA + "\n\npress the button to run it yourself.")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the KL weight trades the two terms of the latent model's error: which
    one dominates at each $\beta$ here? why did Rombach et al. choose a tiny KL weight (and
    add perceptual and adversarial losses to the autoencoder) rather than a VAE with
    $\beta = 1$?
    """)
    return


if __name__ == "__main__":
    app.run()
