import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # audio representations · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. all audio in this block is **synthetic**, made by the
    helpers at the bottom of this notebook (`tone`, `chirp`, `harmonic_vowel`, `vowel_formants`).

    | stage | you build |
    |---|---|
    | 1 | `alias_frequency`, `dft`: what sampling does to a frequency, and the transform itself |
    | 2 | `hann`, `frame`, `stft`: windowed frames and the short-time Fourier transform |
    | 3 | `is_cola`, `istft`: overlap-add and an exact inverse |
    | 4 | `hz_to_mel`, `mel_to_hz`, `mel_filterbank`, `mel_spectrogram`, `power_to_db`: the mel scale |
    | 5 | `dct2`, `mfcc`: cepstral coefficients |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def alias_frequency(f, sr):
    """stage 1: the frequency in [0, sr/2] that a real sinusoid at f Hz appears at after sampling at sr Hz."""
    f = np.asarray(f, dtype=float)
    out = np.abs(f - sr * np.round(f / sr))
    return float(out) if out.ndim == 0 else out


@app.function
def dft(x):
    """stage 1: the DFT of x along its last axis, X[k] = sum_n x[n] exp(-2 pi i k n / N), from a DFT matrix."""
    x = np.asarray(x)
    n = x.shape[-1]
    k = np.arange(n)
    W = np.exp(-2j * np.pi * np.outer(k, k) / n)
    return x @ W.T


@app.cell(hide_code=True)
def _():
    checks.run(1, alias_frequency=alias_frequency, dft=dft)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def hann(n, *, periodic=True):
    """stage 2: Hann window of length n; periodic divides by n (for spectral analysis), symmetric by n - 1."""
    if n == 1:
        return np.ones(1)
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / (n if periodic else n - 1))


@app.function
def frame(x, frame_length, hop_length):
    """stage 2: (n_frames, frame_length) array of frames x[m*hop : m*hop + frame_length], n_frames = 1 + (len - frame_length) // hop."""
    x = np.asarray(x)
    if len(x) < frame_length:
        raise ValueError(f"signal of length {len(x)} is shorter than one frame ({frame_length})")
    n_frames = 1 + (len(x) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return x[idx]


@app.function
def stft(x, n_fft, hop_length, *, window=None, center=True):
    """stage 2: complex (n_fft // 2 + 1, n_frames) STFT, unscaled; window defaults to the periodic Hann; center zero-pads n_fft // 2 each side."""
    x = np.asarray(x, dtype=float)
    w = hann(n_fft) if window is None else np.asarray(window, dtype=float)
    if center:
        x = np.pad(x, (n_fft // 2, n_fft // 2))
    frames = frame(x, n_fft, hop_length)
    return np.fft.rfft(frames * w, axis=1).T


@app.cell(hide_code=True)
def _():
    checks.run(2, hann=hann, frame=frame, stft=stft)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def is_cola(window, hop_length, *, tol=1e-10):
    """stage 3: True if copies of window shifted by multiples of hop_length sum to a constant."""
    w = np.asarray(window, dtype=float)
    n = len(w)
    sums = np.zeros(hop_length)
    for start in range(0, n, hop_length):
        seg = w[start:start + hop_length]
        sums[:len(seg)] += seg
    return bool(np.max(sums) - np.min(sums) <= tol * max(1.0, np.max(np.abs(sums))))


@app.function
def istft(X, hop_length, *, n_fft=None, window=None, center=True, length=None):
    """stage 3: the least-squares inverse of stft (weighted overlap-add divided by the summed squared window); exact for an unmodified STFT."""
    X = np.asarray(X)
    if n_fft is None:
        n_fft = len(window) if window is not None else 2 * (X.shape[0] - 1)
    w = hann(n_fft) if window is None else np.asarray(window, dtype=float)
    frames = np.fft.irfft(X.T, n=n_fft, axis=1) * w
    n_frames = frames.shape[0]
    total = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(total)
    wsum = np.zeros(total)
    for m in range(n_frames):
        y[m * hop_length:m * hop_length + n_fft] += frames[m]
        wsum[m * hop_length:m * hop_length + n_fft] += w ** 2
    ok = wsum > 1e-10 * wsum.max()
    y[ok] /= wsum[ok]
    y[~ok] = 0.0
    if center:
        y = y[n_fft // 2:]
        if length is None:
            y = y[:len(y) - n_fft // 2]
    if length is not None:
        y = y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))
    return y


@app.cell(hide_code=True)
def _():
    checks.run(3, is_cola=is_cola, istft=istft, stft=stft, hann=hann)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def hz_to_mel(f):
    """stage 4: HTK mel scale, 2595 log10(1 + f / 700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


@app.function
def mel_to_hz(m):
    """stage 4: inverse of hz_to_mel, 700 (10^(m / 2595) - 1)."""
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


@app.function
def mel_filterbank(sr, n_fft, n_mels, *, fmin=0.0, fmax=None):
    """stage 4: (n_mels, n_fft // 2 + 1) triangular filters, peak 1, edges at n_mels + 2 mel-spaced points from fmin to fmax (default sr / 2)."""
    fmax = sr / 2 if fmax is None else fmax
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rise = (freqs[None, :] - lo) / (mid - lo)
    fall = (hi - freqs[None, :]) / (hi - mid)
    return np.maximum(0.0, np.minimum(rise, fall))


@app.function
def mel_spectrogram(x, sr, *, n_fft=512, hop_length=128, n_mels=40, fmin=0.0, fmax=None, power=2.0):
    """stage 4: (n_mels, n_frames) = mel_filterbank @ |stft(x)| ** power (centered STFT, periodic Hann)."""
    S = np.abs(stft(x, n_fft, hop_length)) ** power
    return mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax) @ S


@app.function
def power_to_db(S, *, ref=1.0, amin=1e-10, top_db=80.0):
    """stage 4: 10 log10(max(S, amin) / ref), then clipped from below at (max - top_db) unless top_db is None."""
    db = 10.0 * np.log10(np.maximum(np.asarray(S, dtype=float), amin) / ref)
    if top_db is not None:
        db = np.maximum(db, db.max() - top_db)
    return db


@app.cell(hide_code=True)
def _():
    checks.run(4, hz_to_mel=hz_to_mel, mel_to_hz=mel_to_hz, mel_filterbank=mel_filterbank,
               mel_spectrogram=mel_spectrogram, power_to_db=power_to_db, stft=stft)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def dct2(x, *, axis=0):
    """stage 5: orthonormal DCT-II of x along axis (scipy.fft.dct(x, type=2, norm='ortho', axis=axis))."""
    x = np.asarray(x, dtype=float)
    xm = np.moveaxis(x, axis, -1)
    n = xm.shape[-1]
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    C = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    C[0] /= np.sqrt(2.0)
    return np.moveaxis(xm @ C.T, -1, axis)


@app.function
def mfcc(x, sr, *, n_mfcc=13, n_fft=512, hop_length=128, n_mels=40, fmin=0.0, fmax=None):
    """stage 5: (n_mfcc, n_frames): the first n_mfcc rows of dct2(power_to_db(mel_spectrogram(...), top_db=None), axis=0)."""
    M = mel_spectrogram(x, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return dct2(power_to_db(M, top_db=None), axis=0)[:n_mfcc]


@app.cell(hide_code=True)
def _():
    checks.run(5, dct2=dct2, mfcc=mfcc, mel_spectrogram=mel_spectrogram, power_to_db=power_to_db)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## synthetic audio (provided)

    nothing in this block downloads audio, so every sound is made here. `tone` and `chirp`
    are pure test signals; `harmonic_vowel` is a source–filter model of a sung vowel:
    harmonics of a pitch $f_0$ with a $1/k$ roll-off (the source), each scaled by a cascade
    of second-order resonances at the vowel's formant frequencies (the filter). the formant
    table `vowel_formants()` holds rounded average adult male values from Peterson & Barney (1952).
    topics 25 and 26 build their datasets from these helpers.
    """)
    return


@app.function
def tone(freq, duration, sr, *, amp=1.0, phase=0.0):
    """a sinusoid amp * sin(2 pi freq t + phase) sampled at sr for duration seconds."""
    t = np.arange(int(round(duration * sr))) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


@app.function
def chirp(f0, f1, duration, sr, *, amp=1.0):
    """a linear chirp whose instantaneous frequency sweeps f0 -> f1 Hz over duration seconds."""
    t = np.arange(int(round(duration * sr))) / sr
    return amp * np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t ** 2))


@app.function
def formant_gain(f, formants, bandwidths=None):
    """magnitude response at f (Hz) of a cascade of 2nd-order resonators at `formants` (Hz) with `bandwidths` (Hz)."""
    f = np.asarray(f, dtype=float)
    bandwidths = [60.0 + 0.04 * F for F in formants] if bandwidths is None else bandwidths
    g = np.ones_like(f)
    for F, B in zip(formants, bandwidths):
        g = g / np.sqrt((1 - (f / F) ** 2) ** 2 + (f * B / F ** 2) ** 2)
    return g


@app.function
def harmonic_vowel(f0, formants, duration, sr, *, bandwidths=None, rng=None, rms=0.1):
    """source-filter vowel: harmonics k f0 below sr / 2, amplitude formant_gain(k f0) / k, random phases if rng; scaled to `rms`."""
    t = np.arange(int(round(duration * sr))) / sr
    ks = np.arange(1, int((sr / 2 - 1e-9) // f0) + 1)
    amps = formant_gain(ks * f0, formants, bandwidths) / ks
    phases = np.zeros(len(ks)) if rng is None else np.random.default_rng(rng).uniform(0, 2 * np.pi, len(ks))
    x = (amps[:, None] * np.sin(2 * np.pi * f0 * ks[:, None] * t[None, :] + phases[:, None])).sum(0)
    return x * (rms / np.sqrt(np.mean(x ** 2)))


@app.function
def vowel_formants():
    """{vowel: (F1, F2, F3)} in Hz: rounded adult male averages from Peterson & Barney (1952); "e" and "o" are their /ɛ/ and /ɔ/."""
    return {
        "i": (270, 2290, 3010),
        "e": (530, 1840, 2480),
        "a": (730, 1090, 2440),
        "o": (570, 840, 2410),
        "u": (300, 870, 2240),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it uses your STFT to
    measure the time–frequency resolution tradeoff, watches a chirp alias, and asks whether
    MFCCs care more about the pitch of a vowel or about which vowel it is.
    """)
    return


if __name__ == "__main__":
    app.run()
