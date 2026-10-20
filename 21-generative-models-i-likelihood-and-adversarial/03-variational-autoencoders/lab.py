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
    from mlfs.generative import bits_per_dim, cached_feature_net, fid_from_features  # yours, from topic 01
    from mlfs.torchnn import get_device  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    torch.set_num_threads(min(4, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)
    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _gray_train = torch.tensor(_d["X_train"][:, None] / 255.0, dtype=torch.float32)
    _dev = torch.device("cpu") if OFFLINE else get_device()  # the same cached network as topic 01's lab
    featnet = cached_feature_net(_gray_train, _d["y_train"], name="mnist", epochs=2, seed=0, device=_dev).to("cpu")
    N_TRAIN = min(10_000, len(_d["X_train"]))
    _g = torch.Generator().manual_seed(0)
    # binarized digits (each pixel ~ Bernoulli(intensity), fixed seed): a proper discrete likelihood for bits/dim
    X_train = torch.bernoulli(_gray_train[:N_TRAIN].flatten(1), generator=_g)
    gray_test = torch.tensor(_d["X_test"][:1000] / 255.0, dtype=torch.float32).flatten(1)
    X_test = torch.bernoulli(gray_test, generator=_g)
    y_test = _d["y_test"][:1000]
    feats_test = featnet.embed(gray_test.view(-1, 1, 28, 28))


@app.function
def train_vae(latent_dim, beta, *, epochs=10, seed=0):
    """your VAE (784 -> 512 -> 256 -> latent) trained with Adam on the binarized training digits."""
    torch.manual_seed(seed)
    vae = ex.VAE(784, latent_dim)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X_train), generator=g)
        for i in range(0, len(X_train), 128):
            opt.zero_grad()
            vae.loss(X_train[perm[i:i + 128]], beta=beta)[0].backward()
            opt.step()
    return vae.eval()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # variational autoencoders · lab

    every experiment trains **your** `VAE` (784 → 512 → 256 → latent, Bernoulli decoder) on
    {N_TRAIN:,} binarized MNIST digits and evaluates it on 1,000 test digits, with your
    `iwae_nll`, `active_units` and `kl_per_dim`, and topic 01's `FeatureNet` for FID.
    {'**offline: synthetic stand-in data, so the numbers say nothing about MNIST.**' if OFFLINE else ''}
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: walking through latent space

    a VAE with a 16-dimensional latent space ($\beta = 1$). take two test digits of
    different classes, encode them (the encoder means $\mu$), and decode 8 evenly spaced
    points on the straight line between the two codes. compare with the same walk in
    pixel space: $(1 - t)\,x_a + t\,x_b$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "halfway between the codes of a 1 and a 0: what does the decoder draw? and halfway in pixel space?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    vae16 = train_vae(16, 1.0)
    _secs = time.perf_counter() - _t0
    _pairs = []
    for _a, _b in ((1, 0), (4, 9), (3, 8)):
        _ia, _ib = int(np.flatnonzero(y_test == _a)[0]), int(np.flatnonzero(y_test == _b)[0])
        _pairs.append((_ia, _ib))
    _ts = torch.linspace(0, 1, 8)
    _fig, _axes = plt.subplots(2 * len(_pairs), 8, figsize=(8, 2.1 * len(_pairs)))
    with torch.no_grad():
        for _r, (_ia, _ib) in enumerate(_pairs):
            _za, _zb = vae16.encode(X_test[[_ia, _ib]])[0]
            _lat = torch.sigmoid(vae16.decode((1 - _ts[:, None]) * _za + _ts[:, None] * _zb))
            _pix = (1 - _ts[:, None]) * gray_test[_ia] + _ts[:, None] * gray_test[_ib]
            for _c in range(8):
                _axes[2 * _r, _c].imshow(_lat[_c].view(28, 28), cmap="gray_r", vmin=0, vmax=1)
                _axes[2 * _r + 1, _c].imshow(_pix[_c].view(28, 28), cmap="gray_r", vmin=0, vmax=1)
    for _ax in _axes.flat:
        _ax.set_xticks([])
        _ax.set_yticks([])
    for _r in range(len(_pairs)):
        _axes[2 * _r, 0].set_ylabel("latent", fontsize=7)
        _axes[2 * _r + 1, 0].set_ylabel("pixels", fontsize=7)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"VAE trained in {_secs:.0f} s")])
    return (vae16,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the pixel walk is a cross-fade: two ghosts, one fading in. why do the
    latent walk's middle frames look like single (if blurry) digits instead? what part of
    the ELBO forces nearby codes to decode to similar images, and what part packs the
    codes of all training digits into one region around the origin? where do the
    interpolations look least like any digit?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: sweeping β

    VAEs with a 32-dimensional latent space, trained with $\beta \in \{0.25, 1, 4, 16\}$
    (two seeds each). for each: the reconstruction term and the KL on the test digits,
    the number of **active units** (latent coordinates whose encoder mean varies across
    inputs by more than 0.01), the test NLL estimated with your `iwae_nll` ($k = 100$),
    and the FID against the test digits of 1,000 samples (decoded probabilities) and of the
    1,000 reconstructions of the test digits themselves.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with beta = 1, how many of the 32 latent dimensions will the VAE actually use? and with beta = 16?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rows = []
    _t0 = time.perf_counter()
    BETAS = (0.25, 1.0, 4.0, 16.0)
    kl_profiles = {}
    for _beta in BETAS:
        _vals = []
        for _seed in (0, 1):
            _m = train_vae(32, _beta, seed=_seed)
            with torch.no_grad():
                _x_out, _mu, _lv = _m(X_test)
                _recon = float(ex.reconstruction_nll(_x_out, X_test, "bernoulli").mean())
                _kl = float(ex.kl_standard_normal(_mu, _lv).mean())
                _nll = float(ex.iwae_nll(_m, X_test, k=100, generator=torch.Generator().manual_seed(1)).mean())
                _s = _m.sample(1000, generator=torch.Generator().manual_seed(2))
            _au = ex.active_units(_m, X_test)[0]
            _fid = fid_from_features(feats_test, featnet.embed(_s.view(-1, 1, 28, 28)))
            _fid_rec = fid_from_features(feats_test, featnet.embed(torch.sigmoid(_x_out).view(-1, 1, 28, 28)))
            _vals.append((_recon, _kl, _au, _nll, _fid, _fid_rec))
            if _seed == 0:
                kl_profiles[_beta] = np.sort(ex.kl_per_dim(_m, X_test))[::-1]
        _v = np.array(_vals)
        _mean, _half = _v.mean(0), np.abs(_v[0] - _v[1]) / 2
        _rows.append(f"| {_beta:g} | {_mean[0]:.1f} | {_mean[1]:.1f} | {_mean[0] + _mean[1]:.1f} | "
                     f"{_mean[2]:.1f} ± {_half[2]:.1f} | {_mean[3]:.1f} ± {_half[3]:.1f} "
                     f"({bits_per_dim(_mean[3], 784):.3f}) | {_mean[4]:.1f} ± {_half[4]:.1f} | {_mean[5]:.1f} ± {_half[5]:.1f} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    for _beta, _prof in kl_profiles.items():
        _ax.semilogy(np.arange(1, 33), np.maximum(_prof, 1e-4), "o-", ms=3, label=f"β = {_beta:g}")
    _ax.set(xlabel="latent coordinate (sorted by KL)", ylabel="mean KL (nats)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"mean of 2 seeds (± half their difference), {time.perf_counter() - _t0:.0f} s in total\n\n"
        "| β | recon (nats) | KL (nats) | −ELBO at β = 1 | active units (of 32) | IWAE NLL, k = 100 (bits/dim) | "
        "FID of samples | FID of reconstructions |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a latent coordinate whose KL is near 0 has $q(z_i \mid x) \approx
    \mathcal{N}(0, 1)$ for every $x$: it carries no information about the input. why
    does the optimizer switch coordinates off entirely instead of using all 32 a little
    (compare the cost of one more nat of KL with what it buys in reconstruction)? which β
    has the best likelihood, and why isn't it the one with the best reconstructions? do
    the samples' FID and the likelihood rank the β values the same way?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the shape of the aggregate posterior

    a VAE with a **2-dimensional** latent space. left: the encoder means of the 1,000 test
    digits, coloured by class, with the prior's 1σ, 2σ and 3σ circles. right: the decoder's
    output on a grid of $z$ values over $[-3, 3]^2$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the KL term pulls every code towards N(0, I). will the 1,000 codes fill the prior's disk evenly, or leave "
        "gaps? what will the decoder draw in a gap?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _m = train_vae(2, 1.0)
    _n = 12
    _gx = torch.linspace(-3, 3, _n)
    _gy = torch.linspace(3, -3, _n)  # top row of the picture = largest z_2
    _zz = torch.stack([_gx.repeat(_n), _gy.repeat_interleave(_n)], 1)  # row-major: (row r, col c) -> (gx[c], gy[r])
    with torch.no_grad():
        _mu, _lv = _m.encode(X_test)
        _imgs = torch.sigmoid(_m.decode(_zz)).view(_n, _n, 28, 28)
    _mu_np = _mu.numpy()
    _canvas = _imgs.permute(0, 2, 1, 3).reshape(_n * 28, _n * 28).numpy()
    _r = np.linalg.norm(_mu_np, axis=1)
    _agg_cov = np.cov(_mu_np, rowvar=False) + np.diag(np.exp(_lv.numpy()).mean(0))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 4.8))
    _sc = _a1.scatter(_mu_np[:, 0], _mu_np[:, 1], c=y_test, cmap="tab10", s=6)
    for _rad in (1, 2, 3):
        _a1.add_patch(plt.Circle((0, 0), _rad, fill=False, ls="--", color="gray"))
    _a1.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect="equal", title="encoder means of test digits",
            xlabel="$z_1$", ylabel="$z_2$")
    _fig.colorbar(_sc, ax=_a1, fraction=0.046)
    _a2.imshow(1 - _canvas, cmap="gray", extent=(-3, 3, -3, 3))
    _a2.set(title="decoder output on a grid of z", aspect="equal", xlabel="$z_1$", ylabel="$z_2$")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"| | test codes | the prior N(0, I) |\n|---|---|---|\n"
        f"| fraction with ‖μ‖ < 1 | {np.mean(_r < 1):.2f} | 0.39 |\n"
        f"| fraction with ‖μ‖ > 2 | {np.mean(_r > 2):.2f} | 0.14 |\n"
        f"| covariance of the aggregate posterior (Cov μ + E σ²) | "
        f"[[{_agg_cov[0, 0]:.2f}, {_agg_cov[0, 1]:.2f}], [{_agg_cov[1, 0]:.2f}, {_agg_cov[1, 1]:.2f}]] | "
        f"[[1, 0], [0, 1]] |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the KL term asks each $q(z \mid x)$ to be close to $\mathcal{N}(0, I)$,
    but nothing asks the *mixture* of all of them (the aggregate posterior) to equal the
    prior. how does that explain the gaps and the uneven density? when you sample $z$ from
    the prior and land in a gap, what does the decoder produce, and how would that show up
    in experiment 2's two FID columns?
    """)
    return


if __name__ == "__main__":
    app.run()
