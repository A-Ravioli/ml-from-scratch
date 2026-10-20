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
    import torch.nn.functional as F

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    torch.set_num_threads(min(4, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)
    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    N_TRAIN = min(10_000, len(_d["X_train"]))
    X_train = torch.tensor(_d["X_train"][:N_TRAIN, None] / 255.0, dtype=torch.float32)
    X_test = torch.tensor(_d["X_test"][:1000, None] / 255.0, dtype=torch.float32)
    EPOCHS = 3
    LOG_EVERY = 25


@app.function
def train_vqvae(n_codes, *, restarts=False, code_dim=16, epochs=EPOCHS, seed=0):
    """your VQVAE on 28 x 28 digits (7 x 7 codes); MSE + vq_loss, Adam; optionally restart_dead_codes every 50 steps.
    returns (model, history) with history rows (step, perplexity, fraction used, batch mse); the model's attribute
    `init_winners` is the number of codes that win any argmin on 1,000 training digits before training."""
    torch.manual_seed(seed)
    m = ex.VQVAE(1, n_codes=n_codes, code_dim=code_dim, base_channels=32)
    with torch.no_grad():
        m.init_winners = int(m.encode(X_train[:1000]).unique().numel())
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    counts = torch.zeros(n_codes, dtype=torch.long)
    hist, step = [], 0
    for _ in range(epochs):
        perm = torch.randperm(len(X_train), generator=g)
        for i in range(0, len(X_train), 128):
            xb = X_train[perm[i:i + 128]]
            recon, vq_loss, idx = m(xb)
            mse = F.mse_loss(recon, xb)
            opt.zero_grad()
            (mse + vq_loss).backward()
            opt.step()
            counts += torch.bincount(idx.flatten(), minlength=n_codes)
            step += 1
            if step % LOG_EVERY == 0:
                hist.append((step, *ex.codebook_usage(idx, n_codes), mse.item()))
            if restarts and step % 50 == 0:
                with torch.no_grad():
                    z_e = m.encoder(xb)
                ex.restart_dead_codes(m.vq, z_e, counts, generator=g)
                counts.zero_()
    return m.eval(), np.array(hist)


@app.function
def evaluate(m, n_codes):
    """test mse, perplexity and fraction of codes used on the 1,000 test digits."""
    with torch.no_grad():
        recon, _, idx = m(X_test)
    return F.mse_loss(recon, X_test).item(), *ex.codebook_usage(idx, n_codes)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # discrete latents · lab

    every experiment trains **your** `VQVAE` (two stride-2 convolutions down to a $7 \times 7$
    grid of codes, 16-dimensional codes, base width 32) on {N_TRAIN:,} MNIST digits for
    {EPOCHS} epochs with MSE + `vq_loss`, and evaluates on 1,000 test digits with your
    `codebook_usage`. {'**offline: synthetic stand-in data, so the numbers say nothing about MNIST.**' if OFFLINE else ''}
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how much of the codebook gets used?

    a codebook of $K = 256$ codes, trained without any special care (two seeds). every
    image is described by 49 codes; we track the **perplexity** (the effective number of
    codes in use, $e^{H}$) and the fraction of the 256 codes used in each batch.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "after training, how many of the 256 codes does the VQ-VAE actually use on the test set (roughly)? "
        "what will the perplexity be?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    plain = {s: train_vqvae(256, seed=s) for s in (0, 1)}
    _secs = (time.perf_counter() - _t0) / 2
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for _s, (_m, _h) in plain.items():
        _axes[0].plot(_h[:, 0], _h[:, 1], label=f"seed {_s}")
        _axes[1].plot(_h[:, 0], _h[:, 3], label=f"seed {_s}")
    _axes[0].set(xlabel="step", ylabel="perplexity (batch)", title="codes in use")
    _axes[1].set(xlabel="step", ylabel="batch MSE", yscale="log", title="reconstruction")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {_s} | {_m.init_winners} | {_e[0]:.4f} | {_e[1]:.1f} | {_e[2] * 256:.0f} of 256 |"
             for _s, _m, _e in ((_s, _m, evaluate(_m, 256)) for _s, (_m, _h) in plain.items())]
    mo.vstack([_fig, mo.md(f"about {_secs:.0f} s per model\n\n| seed | codes winning before training | test MSE | "
                           "test perplexity | codes used on the test set |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (plain,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the codebook starts in a tiny ball around 0 (uniform in $\pm 1/K$),
    while the encoder's outputs start wherever its initialization puts them. which codes
    win the first argmin, and which codes ever receive gradient from the codebook loss?
    why does a code that never wins stay dead forever? what does that do to the effective
    size of the codebook (and to the number of bits per image the model can use)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: restarting dead codes

    the same models, but every 50 steps your `restart_dead_codes` moves every code that
    wasn't used in the last 50 steps onto a random encoder output from the current batch.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with restarts, how many codes will be in use? will the test MSE go down, and by how much?"
    )
    p2
    return (p2,)


@app.cell
def _(p2, plain):
    ui.gate(p2)
    restarted = {s: train_vqvae(256, restarts=True, seed=s) for s in (0, 1)}
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    for _s in (0, 1):
        _ax.plot(plain[_s][1][:, 0], plain[_s][1][:, 1], "--", color=f"C{_s}", label=f"plain, seed {_s}")
        _ax.plot(restarted[_s][1][:, 0], restarted[_s][1][:, 1], "-", color=f"C{_s}", label=f"restarts, seed {_s}")
    _ax.set(xlabel="step", ylabel="perplexity (batch)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = []
    for _name, _runs in (("plain", plain), ("restarts", restarted)):
        _e = np.array([evaluate(_runs[_s][0], 256) for _s in (0, 1)])
        _rows.append(f"| {_name} | {_e[:, 0].mean():.4f} ± {abs(_e[0, 0] - _e[1, 0]) / 2:.4f} | "
                     f"{_e[:, 1].mean():.1f} ± {abs(_e[0, 1] - _e[1, 1]) / 2:.1f} | {_e[:, 2].mean() * 256:.0f} |")
    mo.vstack([_fig, mo.md("test set, mean of 2 seeds (± half their difference)\n\n| codebook | test MSE | perplexity | "
                           "codes used |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return (restarted,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does putting a dead code on top of a real encoder output bring it
    back to life? the restarted model uses almost every code: did its reconstructions
    improve? a code is only useful if the decoder has learned what to draw for it: how many
    training vectors does each code (and the decoder's response to it) see when 200 codes
    share the data instead of 15? look ahead to experiment 3: how many codes does this
    model actually need? when would you expect restarts to pay off?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how many bits is a digit?

    each image is stored as 49 code indices, so a codebook of $K$ codes gives
    $49 \log_2 K$ bits per image (the raw 8-bit image is $784 \times 8 = 6{,}272$ bits).
    VQ-VAEs with $K = 2$, $16$ and $256$ (with restarts), one seed each: test MSE and
    reconstructions of the same eight test digits.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with K = 2, each digit is described by just 49 bits: a 7 x 7 binary image. will the reconstructions still "
        "be recognizable as the right digits?"
    )
    p3
    return (p3,)


@app.cell
def _(p3, restarted):
    ui.gate(p3)
    _ks = (2, 16, 256)
    _models = {k: train_vqvae(k, restarts=True)[0] for k in _ks[:-1]}
    _models[256] = restarted[0][0]  # experiment 2's model, seed 0
    _fig, _axes = plt.subplots(len(_ks) + 1, 8, figsize=(8, 1.1 * (len(_ks) + 1)))
    _rows = []
    with torch.no_grad():
        for _c in range(8):
            _axes[0, _c].imshow(X_test[_c, 0], cmap="gray_r", vmin=0, vmax=1)
        for _r, _k in enumerate(_ks, start=1):
            _rec = _models[_k](X_test[:8])[0].clamp(0, 1)
            for _c in range(8):
                _axes[_r, _c].imshow(_rec[_c, 0], cmap="gray_r", vmin=0, vmax=1)
            _mse, _perp, _frac = evaluate(_models[_k], _k)
            _rows.append(f"| {_k} | {49 * math.log2(_k):.0f} | {_mse:.4f} | {_perp:.1f} |")
    for _ax in _axes.flat:
        _ax.set_xticks([])
        _ax.set_yticks([])
    _axes[0, 0].set_ylabel("input", fontsize=7)
    for _r, _k in enumerate(_ks, start=1):
        _axes[_r, 0].set_ylabel(f"K={_k}", fontsize=7)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| K | bits per image | test MSE | perplexity |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with $K = 2$ every code position is one bit, yet the decoder draws
    smooth strokes rather than blocky $4 \times 4$ squares: where does that detail come
    from (what does the decoder know that the code doesn't say)? how does the MSE change
    per extra bit as $K$ grows, and why does the return diminish? why does a VQ-VAE pair
    naturally with an autoregressive prior (topic 02) over its code indices?
    """)
    return


if __name__ == "__main__":
    app.run()
