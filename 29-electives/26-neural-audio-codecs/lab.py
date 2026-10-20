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
    from mlfs.generative import codebook_usage, restart_dead_codes  # yours, from ch 21

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    def _audio_smoke(m):  # raises NotImplementedError if a topic 24 stage this lab needs isn't done yet
        m.stft(m.harmonic_vowel(120.0, (730, 1090, 2440), 0.05, 8000), 64, 16)

    # topic 24's synthesis helpers and STFT: the reference under pytest/verify (or MLFS_REFERENCE=audio24), and
    # whenever a topic 24 stage it needs is unfinished
    prev = notebooks.load_topic("24-audio-representations", key="audio24", smoke=_audio_smoke)
    audio = prev.module
    torch.set_num_threads(1)  # small models: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # neural audio codecs · lab

    every experiment here runs **your** code: your RVQ and k-means codebooks, rates and
    metrics, `ResidualVQ` and `ConvCodec` from this topic. for each one: write down a
    prediction, lock it in, run it, then explain what happened. experiment 2 is the block's
    **project**. the lab takes about two minutes on one CPU core.

    **the audio is synthetic**: 3,000 training and 200 test clips of 1,024 samples (128 ms)
    at 8 kHz, a quarter each of topic 24's source–filter vowels (random pitch 90–250 Hz, random
    vowel), linear chirps between 100 Hz and 3.5 kHz, white noise through one random resonance,
    and mixtures of two vowels. every clip has an RMS level around 0.1, 10 ms fades and a faint
    background noise 60 dB down. that's far more regular than speech or music, which matters
    for everything below.
    """)
    return


@app.cell
def _():
    prev.callout()
    return


@app.cell
def _():
    SR, L = 8000, 1024
    VOWELS = list(audio.vowel_formants().values())

    def clip(rng):
        kind, dur = rng.integers(4), L / SR
        if kind == 0:
            x = audio.harmonic_vowel(rng.uniform(90, 250), VOWELS[rng.integers(5)], dur, SR,
                                     rng=int(rng.integers(1 << 30)), rms=1.0)
        elif kind == 1:
            x = audio.chirp(rng.uniform(100, 3500), rng.uniform(100, 3500), dur, SR) * np.sqrt(2)
        elif kind == 2:
            spec = np.fft.rfft(rng.normal(size=L))
            spec *= audio.formant_gain(np.fft.rfftfreq(L, 1 / SR), [rng.uniform(800, 3500)], [rng.uniform(200, 1500)])
            x = np.fft.irfft(spec, L)
            x = x / np.sqrt(np.mean(x ** 2))
        else:
            x = sum(audio.harmonic_vowel(rng.uniform(90, 250), VOWELS[rng.integers(5)], dur, SR,
                                         rng=int(rng.integers(1 << 30)), rms=1.0) for _ in range(2)) / np.sqrt(2)
        fade = np.minimum(1, np.minimum(np.arange(L), np.arange(L)[::-1]) / (0.01 * SR))
        return (x[:L] * fade * 0.1 * 10 ** rng.uniform(-0.15, 0.15) + 1e-4 * rng.normal(size=L)).astype(np.float32)

    _rng = np.random.default_rng(0)
    Xtr = np.stack([clip(_rng) for _ in range(3000)])
    _rng = np.random.default_rng(1)
    Xte = np.stack([clip(_rng) for _ in range(200)])
    return SR, Xte, Xtr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: residual vs single codebooks

    the vectors are raw waveform frames of 8 samples (the shape a hop-8 codec would quantize
    with no encoder at all): 8,000 training frames from the training clips, and every frame
    of the test clips. we fit codebooks with your `fit_rvq` and measure quality as SNR in dB
    from your `rvq_distortion`, on the training frames and on held-out test frames:

    - **one codebook** of $2^b$ codewords (`fit_rvq` with one stage) for $b$ = 4, 6, 8, 10 bits;
    - **RVQ** with 16 codes per stage (4 bits each), 1–6 stages, and with 4 codes per stage
      (2 bits each), 1–8 stages; and 256 codes per stage, 1–2 stages.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at 8 bits per frame, how many dB worse is RVQ with 2 stages of 16 codes than one codebook of 256? and does "
        "each extra 4-bit RVQ stage add the same number of dB?"
    )
    p1
    return (p1,)


@app.cell
def _(Xte, Xtr, p1):
    ui.gate(p1)
    _t = time.time()
    _fr = Xtr.reshape(-1, 8)
    _tr = _fr[np.random.default_rng(0).permutation(len(_fr))[:8000]]
    _te = Xte.reshape(-1, 8)

    def _db(X, books):
        return 10 * np.log10(np.mean(X ** 2) / np.asarray(ex.rvq_distortion(X, books)))

    curves = {}
    for _K, _S in [(16, 6), (4, 8), (256, 2)]:
        _books = ex.fit_rvq(_tr, _K, _S, n_iters=20, rng=0)
        curves[f"RVQ, {_K} codes per stage"] = (np.log2(_K) * np.arange(1, _S + 1), _db(_tr, _books), _db(_te, _books))
    _vq = {b: ex.fit_rvq(_tr, 2 ** b, 1, n_iters=20, rng=0) for b in (4, 6, 8, 10)}
    curves["one codebook"] = (np.array([4, 6, 8, 10]), np.array([_db(_tr, _vq[b])[0] for b in _vq]),
                              np.array([_db(_te, _vq[b])[0] for b in _vq]))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for _name, (_bits, _a, _b) in curves.items():
        _axes[0].plot(_bits, _a, "o-", label=_name)
        _axes[1].plot(_bits, _b, "o-", label=_name)
    _axes[0].set(title="training frames", xlabel="bits per 8-sample frame", ylabel="SNR (dB)")
    _axes[1].set(title="held-out test frames", xlabel="bits per 8-sample frame")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {n} | " + ", ".join(f"{int(b)}: {a:.1f} / {c:.1f}" for b, a, c in zip(*v)) + " |"
             for n, v in curves.items()]
    mo.vstack([_fig, mo.md("| quantizer | bits: train / test SNR (dB) |\n|---|---|\n" + "\n".join(_rows)
                           + f"\n\nfitting took {time.time() - _t:.1f} s.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does each extra RVQ stage buy fewer dB than the one before? at 8 bits,
    the single codebook can put its 256 codewords anywhere, while 2 × 16 RVQ can only reach
    sums: why is the loss on test frames so modest? and why does the gap between training and
    test grow so fast for the single codebook (how many training frames does each of its 1,024
    codewords get at 10 bits)? what would 80 bits per frame cost each design?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2 (project): bitrate against quality

    four codecs on the 200 test clips, every bitrate computed with your `bitrate`:

    - **PCM** and **μ-law** (your `pcm_quantize`, μ = 255), 1–4 bits per sample: 8–32 kbps;
    - **k-means RVQ on raw frames**: the waveform cut into 32-sample frames, quantized by your
      `fit_rvq` codebooks (8 stages of 256 codes, fit on 20,000 training frames) with no
      network at all: 250 frames/s × 8 bits = 2 kbps per stage, 2–16 kbps;
    - **your `ConvCodec`**: strides (2, 4, 4) (hop 32), 16 base channels, `code_dim` 16, 8
      stages of 256 codes, so the same 2–16 kbps. trained in two phases: 800 Adam steps as a
      plain autoencoder (no quantizer), then `init_from_data` (your k-means on encoder outputs
      of 300 clips), then 600 steps with the quantizer and **quantizer dropout** ($n_q$ drawn
      from 1–8 per batch), restarting dead codes every 100 steps (your ch 21
      `restart_dead_codes`). loss: $10\times$ waveform L1 + your `multiscale_spectral_loss`
      (`eps=1e-3`) + the vq loss; batch 16, lr 3e-3. two training seeds.

    metrics: your `snr_db` and `log_spectral_distance` with `eps=1e-4` (a floor about 55 dB
    below the loudest bins, so the faint background noise doesn't dominate), each averaged
    over clips (the dropdown under the explain prompt recomputes the LSD with other floors). the "autoencoder" row is the trained codec with the quantizer bypassed: the
    quality the network could reach with unlimited bits.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "at 2 kbps, which wins on SNR: your neural codec or k-means RVQ on raw frames? which wins on log-spectral "
        "distance? how does either compare with 1-bit mu-law at 8 kbps?"
    )
    p2
    return (p2,)


@app.cell
def _(Xtr):
    def train_codec(seed, ae_steps=800, rvq_steps=600):
        torch.manual_seed(seed)
        codec = ex.ConvCodec(channels=16, code_dim=16, strides=(2, 4, 4), n_stages=8, n_codes=256)
        opt = torch.optim.Adam(codec.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(seed)
        data = torch.tensor(Xtr)[:, None]

        def loss_fn(x, xh):
            return 10 * (x - xh).abs().mean() + ex.multiscale_spectral_loss(x, xh, eps=1e-3)

        for _ in range(ae_steps):
            x = data[torch.randint(len(data), (16,), generator=g)]
            loss = loss_fn(x, codec.decoder(codec.encoder(x)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            codec.quantizer.init_from_data(codec.encoder(data[:300]), rng=seed)
        counts = torch.zeros(8, 256)
        for step in range(rvq_steps):
            x = data[torch.randint(len(data), (16,), generator=g)]
            nq = int(torch.randint(1, 9, (1,), generator=g))
            xh, vq, codes = codec(x, nq)
            loss = loss_fn(x, xh) + vq
            opt.zero_grad()
            loss.backward()
            opt.step()
            for s in range(nq):
                counts[s] += torch.bincount(codes[:, s].flatten(), minlength=256)
            if (step + 1) % 100 == 0:
                with torch.no_grad():
                    r = codec.encoder(x)
                    for s, vq_s in enumerate(codec.quantizer.stages):
                        restart_dead_codes(vq_s, r, counts[s], threshold=1, generator=g)
                        r = r - vq_s(r)[0]
                counts.zero_()
        return codec.eval()

    return (train_codec,)


@app.cell
def _(SR, Xte, Xtr, p2, train_codec):
    ui.gate(p2)
    _t0 = time.time()

    recons = []  # (codec, kbps, seed, reconstructions), for the eps explorer below
    results = {}  # name -> list of (kbps, snr per clip, lsd per clip, seed)

    def add(name, kbps, rec, seed=0):
        recons.append((name, kbps, seed, rec))
        snr = np.array([ex.snr_db(a, b) for a, b in zip(Xte, rec)])
        lsd = np.array([ex.log_spectral_distance(a, b, eps=1e-4) for a, b in zip(Xte, rec)])
        results.setdefault(name, []).append((kbps, snr, lsd, seed))

    for _b in (1, 2, 3, 4):
        add("PCM", 8 * _b, ex.pcm_quantize(Xte, _b))
        add("μ-law", 8 * _b, ex.pcm_quantize(Xte, _b, mu=255))
    _fr = Xtr.reshape(-1, 32)
    books = ex.fit_rvq(_fr[np.random.default_rng(0).permutation(len(_fr))[:20000]], 256, 8, n_iters=15, rng=0)
    _codes_km = np.asarray(ex.rvq_encode(Xte.reshape(-1, 32), books))
    for _n in (1, 2, 4, 8):
        add("k-means RVQ, raw frames", ex.bitrate(SR, 32, _n, 256) / 1000,
            np.asarray(ex.rvq_decode(_codes_km[:, :_n], books)).reshape(Xte.shape))
    _t_km = time.time() - _t0
    codecs = {}
    _xt = torch.tensor(Xte)[:, None]
    for _seed in (0, 1):
        codecs[_seed] = train_codec(_seed)
        with torch.no_grad():
            for _n in (1, 2, 4, 8):
                add("your ConvCodec", ex.bitrate(SR, 32, _n, 256) / 1000, codecs[_seed](_xt, _n)[0][:, 0].numpy(), _seed)
            add("ConvCodec, quantizer bypassed", np.nan,
                codecs[_seed].decoder(codecs[_seed].encoder(_xt))[:, 0].numpy(), _seed)
    _rows = []
    for _name, _items in results.items():
        for _kb in sorted({it[0] for it in _items}, key=lambda v: (np.isnan(v), v)):
            _its = [it for it in _items if (np.isnan(_kb) and np.isnan(it[0])) or it[0] == _kb]
            _snr = [it[1].mean() for it in _its]
            _lsd = [it[2].mean() for it in _its]
            _se = np.mean([it[1].std(ddof=1) / np.sqrt(len(it[1])) for it in _its])
            _seeds = f" (seeds: {', '.join(f'{v:.1f}' for v in _snr)} / {', '.join(f'{v:.1f}' for v in _lsd)})" if len(_its) > 1 else ""
            _rows.append(f"| {_name} | {'∞' if np.isnan(_kb) else f'{_kb:g}'} | {np.mean(_snr):.1f} ± {_se:.1f} | "
                         f"{np.mean(_lsd):.1f}{_seeds} |")
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _name, _items in results.items():
        if _name.endswith("bypassed"):
            continue
        _kbs = sorted({it[0] for it in _items})
        _snr = [np.mean([it[1].mean() for it in _items if it[0] == k]) for k in _kbs]
        _lsd = [np.mean([it[2].mean() for it in _items if it[0] == k]) for k in _kbs]
        _axes[0].plot(_kbs, _snr, "o-", label=_name)
        _axes[1].plot(_kbs, _lsd, "o-", label=_name)
    _axes[0].set(xscale="log", xlabel="bitrate (kbps, log scale)", ylabel="SNR (dB), higher is better")
    _axes[1].set(xscale="log", xlabel="bitrate (kbps, log scale)", ylabel="log-spectral distance (dB), lower is better")
    _ae = [it for it in results["ConvCodec, quantizer bypassed"]]
    _axes[0].axhline(np.mean([it[1].mean() for it in _ae]), color="C3", ls=":", lw=1,
                     label="your ConvCodec, quantizer bypassed")
    _axes[1].axhline(np.mean([it[2].mean() for it in _ae]), color="C3", ls=":", lw=1)
    for _ax in _axes:
        _ax.minorticks_off()
        _ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    _fig.legend(*_axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=8)
    _fig.tight_layout(rect=(0, 0, 1, 0.9))
    mo.vstack([_fig, mo.md("| codec | kbps | SNR (dB) ± s.e. over clips | LSD (dB) |\n|---|---|---|---|\n"
                           + "\n".join(_rows)
                           + f"\n\nConvCodec rows average two training seeds (per-seed SNR / LSD in brackets). "
                           f"baselines took {_t_km:.0f} s, the two codecs {time.time() - _t0 - _t_km:.0f} s.")])
    return books, codecs, recons


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does plain k-means on raw frames win on SNR at every bitrate here, while
    your codec wins on log-spectral distance at the lowest ones? (look at the "quantizer
    bypassed" row: what limits the codec even with unlimited bits, and how long did it train
    compared with SoundStream's millions of steps?) what does each metric reward, and which
    of these codecs would a listener prefer at 2 kbps? (you can't answer that from these
    numbers: say what you'd need.) why is μ-law so poor below 4 bits per sample?
    """)
    return


@app.cell
def _():
    eps_dd = mo.ui.dropdown(["1e-2", "1e-4", "1e-6", "1e-10"], value="1e-4",
                            label="LSD floor eps (added to |STFT|²)")
    eps_dd
    return (eps_dd,)


@app.cell
def _(Xte, eps_dd, recons):
    _eps = float(eps_dd.value)
    _cell = {}
    for _name, _kb, _seed, _rec in recons:
        _lsd = np.mean([ex.log_spectral_distance(a, b, eps=_eps) for a, b in zip(Xte, _rec)])
        _cell.setdefault((_name, _kb), []).append(_lsd)
    _rows = [f"| {n} | {'∞' if np.isnan(k) else f'{k:g}'} | {np.mean(v):.1f} |" for (n, k), v in _cell.items()]
    mo.md(f"**log-spectral distance with eps = {eps_dd.value}** (ConvCodec: mean of the two seeds)\n\n"
          "| codec | kbps | LSD (dB) |\n|---|---|---|\n" + "\n".join(_rows)
          + "\n\nthe floor decides how much the quiet bins count. which codec wins at 2 kbps depends on it.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how many bits do the tokens really need?

    every code costs $\log_2 256 = 8$ bits. for each RVQ stage of your trained codec (seed 0)
    and of the k-means codec, on the test clips: the empirical entropy (your `entropy_bits`),
    the codebook perplexity and fraction of codes used (your ch 21 `codebook_usage`), and, for
    the first stage, the cross-entropy of a **bigram model** that predicts each code from the
    previous frame's code (counts from the training clips, add-0.1 smoothing): the simplest
    possible codec language model.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how many of the 8 bits per code does an entropy coder need for the first stage? and once it also knows the "
        "previous frame's code?"
    )
    p3
    return (p3,)


@app.cell
def _(Xte, Xtr, books, codecs, p3):
    ui.gate(p3)
    with torch.no_grad():
        _ctr = codecs[0].encode(torch.tensor(Xtr)[:, None]).numpy()
        _cte = codecs[0].encode(torch.tensor(Xte)[:, None]).numpy()
    _ktr = np.asarray(ex.rvq_encode(Xtr.reshape(-1, 32), books)).reshape(len(Xtr), 32, 8).transpose(0, 2, 1)
    _kte = np.asarray(ex.rvq_encode(Xte.reshape(-1, 32), books)).reshape(len(Xte), 32, 8).transpose(0, 2, 1)

    def bigram_bits(train, test, alpha=0.1):
        counts = np.full((256, 256), alpha)
        np.add.at(counts, (train[:, :-1].ravel(), train[:, 1:].ravel()), 1)
        p = counts / counts.sum(1, keepdims=True)
        return float(-np.mean(np.log2(p[test[:, :-1].ravel(), test[:, 1:].ravel()])))

    _rows = []
    for _name, _tr, _te in [("your ConvCodec", _ctr, _cte), ("k-means RVQ, raw frames", _ktr, _kte)]:
        for _s in range(8):
            _perp, _used = codebook_usage(torch.tensor(_te[:, _s]), 256)
            _bg = f"{bigram_bits(_tr[:, 0], _te[:, 0]):.2f}" if _s == 0 else "–"
            _rows.append(f"| {_name} | {_s + 1} | {ex.entropy_bits(_te[:, _s], 256):.2f} | {_perp:.0f} | {_used:.2f} | {_bg} |")
    mo.md("| codec | stage | entropy (bits) | perplexity | fraction used | bigram cross-entropy (bits) |\n"
          "|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the entropy is what a coder that treats codes as independent draws needs;
    the bigram cross-entropy is what one that knows the previous code needs. why is the first
    stage so much more predictable from its past than from its marginal frequencies (think
    about what a 128 ms synthetic vowel looks like frame after frame)? which stages look like
    noise, and why does that make sense for residuals? what would a transformer over all
    stages (EnCodec's entropy model, or a codec language model) be able to exploit that the
    bigram can't, and what would real speech change?
    """)
    return


if __name__ == "__main__":
    app.run()
