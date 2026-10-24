import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # audio representations · lab

    every experiment here runs **your** STFT, mel filterbank and MFCCs. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the lab takes a few seconds.

    **all signals are synthetic**: tones, clicks and a chirp, plus the notebook's
    source–filter vowels (harmonics of a pitch $f_0$ shaped by Peterson & Barney formants,
    with a little white noise 40 dB down so silence between harmonics isn't digital zero).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the time–frequency tradeoff

    one signal at $f_s = 8$ kHz: two tones at 1,000 and 1,040 Hz for the first 0.6 s, then
    two clicks 6 ms apart at 0.8 s. for each Hann frame length $N$ we ask two questions of
    your `stft` (hop $N/4$, centered):

    - **tones resolved?** in the time-averaged magnitude spectrum of the tone part, are there
      two peaks between 950 and 1,090 Hz with a dip of at least 3 dB between them?
    - **clicks resolved?** in the frame energy $\sum_k |X[k, m]|^2$ around 0.8 s (here with
      a hop of 8 samples, so the time axis isn't the bottleneck), are there two peaks?

    the last row keeps a 256-sample Hann window but zero-pads every frame to a 2,048-point DFT.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what is the smallest frame length N that separates the two tones (40 Hz apart, at 8 kHz)? the largest that "
        "separates the clicks (6 ms apart)? does zero-padding a 256-sample window to 2,048 points separate the tones?"
    )
    p1
    return (p1,)


@app.cell
def _():
    SR = 8000
    tones = ex.tone(1000, 0.6, SR) + ex.tone(1040, 0.6, SR)
    clicks = np.zeros(int(0.4 * SR))
    clicks[int(0.2 * SR)] = 1.0
    clicks[int(0.2 * SR) + int(0.006 * SR)] = 1.0
    signal = np.concatenate([tones, clicks])

    def two_peaks(values, min_dip_db=3.0):
        """two local maxima with a dip of >= min_dip_db between them?"""
        pk = [i for i in range(1, len(values) - 1) if values[i] >= values[i - 1] and values[i] >= values[i + 1]
              and values[i] > 0.1 * values.max()]
        if len(pk) < 2:
            return False
        a, b = sorted(sorted(pk, key=lambda i: -values[i])[:2])
        dip = values[a:b + 1].min()
        return bool(10 * np.log10(min(values[a], values[b]) / max(dip, 1e-300)) >= min_dip_db)

    def tones_resolved(n_win, n_fft=None):
        n_fft = n_fft or n_win
        w = np.r_[ex.hann(n_win), np.zeros(n_fft - n_win)]
        S = np.abs(ex.stft(tones, n_fft, max(n_fft // 4, 1), window=w)).mean(axis=1)
        f = np.arange(len(S)) * SR / n_fft
        band = (f > 950) & (f < 1090)
        return two_peaks(S[band] ** 2)

    def clicks_resolved(n_win):
        E = (np.abs(ex.stft(clicks, n_win, 8)) ** 2).sum(axis=0)
        t = np.arange(len(E)) * 8
        band = (t > 0.2 * SR - 200) & (t < 0.2 * SR + 300)
        return two_peaks(E[band])

    return SR, clicks_resolved, signal, tones_resolved


@app.cell
def _(SR, clicks_resolved, p1, tones_resolved):
    ui.gate(p1)
    _rows = []
    for _n in [64, 128, 256, 384, 512, 1024]:
        _rows.append(f"| {_n} | {1000 * _n / SR:.0f} ms | {SR / _n:.1f} Hz | "
                     f"{'yes' if tones_resolved(_n) else 'no'} | {'yes' if clicks_resolved(_n) else 'no'} |")
    _rows.append(f"| 256 zero-padded to 2048 | 32 ms | 3.9 Hz | {'yes' if tones_resolved(256, 2048) else 'no'} | – |")
    mo.md("| frame length N | duration | bin spacing | tones 40 Hz apart resolved? | clicks 6 ms apart resolved? |\n"
          "|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _():
    n_slider = mo.ui.slider(steps=[64, 128, 256, 512, 1024], value=256, label="frame length N (Hann, hop N/4)")
    n_slider
    return (n_slider,)


@app.cell
def _(SR, n_slider, p1, signal):
    ui.gate(p1)
    _n = n_slider.value
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    _S = ex.power_to_db(np.abs(ex.stft(signal, _n, _n // 4)) ** 2, top_db=60.0)
    _axes[0].imshow(_S, origin="lower", aspect="auto", cmap="magma", extent=[0, len(signal) / SR, 0, SR / 2])
    _axes[0].set(title=f"spectrogram, N = {_n} ({1000 * _n / SR:.0f} ms), top 60 dB", xlabel="time (s)",
                 ylabel="frequency (Hz)", ylim=(0, 2000))
    _zoom = signal[int(0.76 * SR):int(0.84 * SR)]
    for _m in sorted({64, _n}):
        _E = (np.abs(ex.stft(_zoom, _m, 8)) ** 2).sum(axis=0)
        _axes[1].plot(0.76 + np.arange(len(_E)) * 8 / SR, _E / _E.max(), label=f"N = {_m}")
    for _c in [0.8, 0.806]:
        _axes[1].axvline(_c, color="k", ls=":", lw=0.8)
    _axes[1].set(title="frame energy near the clicks (hop 8)", xlabel="time (s)", ylabel="energy / max",
                 xlim=(0.78, 0.83))
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does it take about *two* bins of spacing (not one) to separate two tones
    with a Hann window? (what's the width of its main lobe?) why does a long window smear two
    clicks into one blob, and what is the product of the tone condition and the click condition
    in seconds × Hz? zero-padding gave 8× more bins: why didn't the tones separate?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a chirp past Nyquist

    a linear chirp from 0 to 8 kHz over 2 s, sampled at 8 kHz, so its second half is above
    the 4 kHz Nyquist frequency. the dashed line is your `alias_frequency` of the chirp's
    true instantaneous frequency.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("what does the spectrogram show during the second second, when the true frequency climbs from 4 to 8 kHz?")
    p2
    return (p2,)


@app.cell
def _(SR, p2):
    ui.gate(p2)
    _x = ex.chirp(0.0, 8000.0, 2.0, SR)
    _S = ex.power_to_db(np.abs(ex.stft(_x, 256, 64)) ** 2, top_db=60.0)
    _t = np.arange(_S.shape[1]) * 64 / SR
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.imshow(_S, origin="lower", aspect="auto", cmap="magma", extent=[0, 2, 0, SR / 2])
    _ax.plot(_t, ex.alias_frequency(4000.0 * _t, SR), "c--", lw=1.5, label="your alias_frequency(true frequency)")
    _ax.plot(_t, 4000.0 * _t, "k:", lw=1.2, label="true frequency")
    _ax.axhline(SR / 2, color="grey", lw=0.8)
    _ax.text(0.05, SR / 2 + 150, "Nyquist (4 kHz): nothing above here is representable", fontsize=8)
    _ax.set(xlabel="time (s)", ylabel="frequency (Hz)", xlim=(0, 2), ylim=(0, SR))
    _ax.legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
    # downsampling by dropping samples: a 3 kHz tone at 8 kHz, every other sample kept (4 kHz)
    _y = ex.tone(3000.0, 1.0, SR)[::2]
    _peak = np.argmax(np.abs(np.fft.rfft(_y))) * (SR / 2) / len(_y)
    mo.vstack([_fig, mo.md(f"a 3 kHz tone at 8 kHz, downsampled to 4 kHz by keeping every other sample, peaks at "
                           f"**{_peak:.0f} Hz** (your `alias_frequency(3000, 4000)` = "
                           f"{ex.alias_frequency(3000.0, 4000):.0f} Hz).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the chirp turn around at 4 kHz and come back down, rather than
    disappearing? at the moment its true frequency is 6 kHz, which real tone would produce
    exactly the same samples? what should a resampler do before dropping samples?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: pitch or vowel?

    five vowels (/i/, /e/, /a/, /o/, /u/) at six pitches between 100 and 210 Hz, 0.4 s each
    at 16 kHz, random harmonic phases. each clip becomes one vector, the average over its
    frames ($N = 1024$, hop 256) of:

    - **log spectrum**: `power_to_db(|stft|²)`, 513 numbers (mean removed);
    - **log mel**: `power_to_db(mel_spectrogram)`, 40 mel bands (mean removed);
    - **MFCC c1–c39**: all of your MFCCs except $c_0$ (removing $c_0$ is removing the mean);
    - **MFCC c1–c12**: the usual truncation.

    we compare two average distances: the same vowel at two pitches (100 vs 140, 120 vs 170,
    150 vs 210 Hz) and two different vowels at the same pitch. the ratio is unit-free: below
    1, the representation cares more about the vowel than about the pitch. three seeds for
    the random phases and noise.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for each of the four representations, is the ratio (pitch distance / vowel distance) above or below 1? "
        "does the DCT (log mel -> MFCC c1-c39) change it?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _sr = 16000
    _V = ex.vowel_formants()
    _vs = list(_V)
    _feats = {
        "log spectrum": lambda x: ex.power_to_db(np.abs(ex.stft(x, 1024, 256)) ** 2, top_db=None).mean(1),
        "log mel": lambda x: ex.power_to_db(ex.mel_spectrogram(x, _sr, n_fft=1024, hop_length=256, n_mels=40),
                                            top_db=None).mean(1),
        "MFCC c1–c39": lambda x: ex.mfcc(x, _sr, n_mfcc=40, n_fft=1024, hop_length=256, n_mels=40)[1:].mean(1),
        "MFCC c1–c12": lambda x: ex.mfcc(x, _sr, n_mfcc=13, n_fft=1024, hop_length=256, n_mels=40)[1:].mean(1),
    }
    _pitch_pairs = [(100, 140), (120, 170), (150, 210)]
    _res = {k: [] for k in _feats}
    for _seed in range(3):
        _rng = np.random.default_rng(_seed)
        _clips = {(v, f0): ex.harmonic_vowel(f0, _V[v], 0.4, _sr, rng=int(_rng.integers(1 << 30)))
                  + 0.001 * _rng.normal(size=int(0.4 * _sr))
                  for v in _vs for f0 in [100, 120, 140, 150, 170, 210]}
        for _name, _f in _feats.items():
            _F = {k: _f(x) for k, x in _clips.items()}
            if _name.startswith("log"):
                _F = {k: v - v.mean() for k, v in _F.items()}
            _dp = np.mean([np.linalg.norm(_F[(v, a)] - _F[(v, b)]) for v in _vs for a, b in _pitch_pairs])
            _dv = np.mean([np.linalg.norm(_F[(_vs[i], f0)] - _F[(_vs[j], f0)])
                           for i in range(5) for j in range(i + 1, 5) for f0 in [100, 140, 210]])
            _res[_name].append((_dp, _dv))
    _rows = []
    for _name, _r in _res.items():
        _r = np.array(_r)
        _ratio = _r[:, 0] / _r[:, 1]
        _rows.append(f"| {_name} | {_r[:, 0].mean():.1f} | {_r[:, 1].mean():.1f} | {_ratio.mean():.2f} "
                     f"({', '.join(f'{v:.2f}' for v in _ratio)}) |")
    # one picture: /a/ at two pitches and /i/, log spectrum vs log mel
    _noise = np.random.default_rng(9).normal(size=(3, int(0.4 * _sr))) * 0.001
    _a1 = ex.harmonic_vowel(100, _V["a"], 0.4, _sr, rng=1) + _noise[0]
    _a2 = ex.harmonic_vowel(140, _V["a"], 0.4, _sr, rng=2) + _noise[1]
    _i1 = ex.harmonic_vowel(100, _V["i"], 0.4, _sr, rng=3) + _noise[2]
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 3.8))
    _f = np.arange(513) * _sr / 1024
    for _x, _lab in [(_a1, "/a/ at 100 Hz"), (_a2, "/a/ at 140 Hz"), (_i1, "/i/ at 100 Hz")]:
        _axes[0].plot(_f, _feats["log spectrum"](_x), lw=0.8, label=_lab)
        _axes[1].plot(_feats["log mel"](_x), "o-", ms=3, label=_lab)
    _axes[0].set(xlim=(0, 4000), xlabel="frequency (Hz)", ylabel="dB", title="log spectrum")
    _axes[1].set(xlabel="mel band", ylabel="dB", title="log mel spectrum (40 bands)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md("| representation | same vowel, two pitches | two vowels, same pitch | ratio (per seed) |\n"
                     "|---|---|---|---|\n" + "\n".join(_rows)), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why, in the log spectrum, is moving the pitch from 100 to 140 Hz a bigger
    change than switching from /a/ to /i/? (look at the left plot: where are the peaks and
    valleys of each curve?) which step removes most of that, and why does the DCT on its own
    leave the ratio *exactly* unchanged? what do you lose by keeping only c1–c12, and why is
    that loss welcome for recognizing vowels but not for, say, recognizing a speaker?
    """)
    return


if __name__ == "__main__":
    app.run()
