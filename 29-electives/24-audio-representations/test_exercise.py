"""Checks for exercise.py (audio representations).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The DFT and STFT are
checked against NumPy and `scipy.signal.stft`; the inverse STFT against perfect reconstruction
and an explicit least-squares solve; the mel filterbank against a hand construction with the HTK
formula (the lesson's worked example); MFCCs against `scipy.fft.dct`. All signals are synthetic.
"""
import numpy as np
import scipy.fft
import scipy.signal

from mlfs.testing import bounded


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- stage 1: sampling and the DFT


def test_stage1_alias_frequency(nb):
    cases = {(7000, 8000): 1000, (1000, 8000): 1000, (4000, 8000): 4000, (9000, 8000): 1000,
             (12500, 8000): 3500, (0, 8000): 0, (16000, 8000): 0, (5000, 8000): 3000, (440, 16000): 440}
    for (f, sr), want in cases.items():
        got = nb.alias_frequency(f, sr)
        assert np.isclose(got, want), (
            f"a {f} Hz sinusoid sampled at {sr} Hz shows up at {want} Hz, got {got}. fold into [0, sr/2]: "
            "subtract the nearest multiple of sr, then take the absolute value (7 kHz at 8 kHz -> 1 kHz, not 3 kHz)")
    got = nb.alias_frequency(7000, 8000)
    assert type(got) is float, f"a scalar input gives a Python float, got {type(got).__name__}"
    arr = nb.alias_frequency(np.array([7000.0, 9000.0, 3000.0]), 8000)
    assert np.allclose(arr, [1000, 1000, 3000]), f"alias_frequency must work elementwise on arrays, got {arr}"


def test_stage1_alias_matches_sampled_sinusoid(nb):
    sr, n = 1000, 1000
    t = np.arange(n) / sr
    for f in [130.0, 610.0, 880.0, 1370.0, 2250.0]:
        x = np.cos(2 * np.pi * f * t)
        peak = np.argmax(np.abs(np.fft.rfft(x))) * sr / n
        got = nb.alias_frequency(f, sr)
        assert np.isclose(got, peak), f"{f} Hz sampled at {sr} Hz has its spectral peak at {peak} Hz; you said {got}"


def test_stage1_dft_hand_example(nb):
    got = np.asarray(nb.dft(np.array([1.0, 0.0, -1.0, 0.0])))
    assert got.shape == (4,), f"dft of a length-4 signal has 4 coefficients, got shape {got.shape}"
    assert np.allclose(got, [0, 2, 0, 2]), (
        f"DFT of (1, 0, -1, 0) is (0, 2, 0, 2) (lesson worked example), got {np.round(got, 3)}. "
        "if you got (0, 0.5, 0, 0.5), drop the 1/N: numpy's forward DFT is unnormalized")


def test_stage1_dft_matches_numpy_without_fft(nb):
    x = _rng(1).normal(size=(3, 24)) + 1j * _rng(2).normal(size=(3, 24))
    saved = {k: getattr(np.fft, k) for k in ("fft", "rfft", "ifft", "fftn")}

    def _no(*a, **k):
        raise AssertionError("stage 1 builds the DFT from its definition (a DFT matrix); don't call np.fft here")

    try:
        for k in saved:
            setattr(np.fft, k, _no)
        got = nb.dft(x)
        got_real = nb.dft(x.real[0])
    finally:
        for k, v in saved.items():
            setattr(np.fft, k, v)
    assert np.allclose(got, np.fft.fft(x, axis=-1)), (
        "dft must equal np.fft.fft along the last axis (batched over leading axes, complex input allowed). "
        "a sign error in the exponent gives the complex conjugate for real inputs: use exp(-2 pi i k n / N)")
    assert np.allclose(got_real, np.fft.fft(x.real[0])), "a real 1-D input must match np.fft.fft too"


# ---------------------------------------------------------------- stage 2: windows, frames, STFT


def test_stage2_hann_periodic_and_symmetric(nb):
    for n in [4, 8, 15, 64]:
        p = nb.hann(n)
        s = nb.hann(n, periodic=False)
        assert np.allclose(p, scipy.signal.get_window("hann", n, fftbins=True)), (
            f"periodic Hann (the default) of length {n}: 0.5 - 0.5 cos(2 pi k / n), k = 0..n-1")
        assert np.allclose(s, scipy.signal.get_window("hann", n, fftbins=False)), (
            f"symmetric Hann of length {n}: 0.5 - 0.5 cos(2 pi k / (n - 1)), so both ends are 0")
    assert np.allclose(nb.hann(4), [0, 0.5, 1, 0.5]), "hann(4) = (0, 0.5, 1, 0.5)"
    assert np.allclose(nb.hann(1), [1.0]) and np.allclose(nb.hann(1, periodic=False), [1.0]), "a length-1 window is [1]"


def test_stage2_frame(nb):
    x = np.arange(10.0)
    F = nb.frame(x, 4, 3)
    assert np.asarray(F).shape == (3, 4), (
        f"10 samples, frames of 4 every 3: n_frames = 1 + (10 - 4) // 3 = 3 (an incomplete last frame is dropped), "
        f"got shape {np.asarray(F).shape}")
    assert np.array_equal(F, [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]), f"frame m is x[3m : 3m + 4]; got {F}"
    assert np.asarray(nb.frame(np.arange(11.0), 4, 3)).shape == (3, 4), "11 samples still give 3 complete frames"
    assert np.asarray(nb.frame(np.arange(4.0), 4, 3)).shape == (1, 4), "exactly one frame's worth gives one frame"
    try:
        nb.frame(np.arange(3.0), 4, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("a signal shorter than one frame should raise ValueError")


def _scipy_stft(x, n_fft, hop, window="hann"):
    _, _, Z = scipy.signal.stft(x, window=window, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros",
                                padded=False)
    return Z


def test_stage2_stft_matches_scipy(nb):
    x = _rng(3).normal(size=1000)
    for n_fft, hop in [(64, 16), (64, 32), (256, 64), (100, 25), (63, 21)]:
        X = np.asarray(nb.stft(x, n_fft, hop))
        Z = _scipy_stft(x, n_fft, hop)
        assert X.shape == Z.shape, (
            f"n_fft={n_fft}, hop={hop}: expected shape (n_fft // 2 + 1, n_frames) = {Z.shape} (frequency first), "
            f"got {X.shape}. center=True pads n_fft // 2 zeros on both sides before framing")
        assert np.iscomplexobj(X), "stft returns complex coefficients"
        win = scipy.signal.get_window("hann", n_fft)
        assert np.allclose(X / win.sum(), Z, atol=1e-12), (
            f"n_fft={n_fft}, hop={hop}: stft / sum(window) should equal scipy.signal.stft (scipy divides by the "
            "window sum; yours is unscaled). check: periodic Hann, zero padding (not reflect), frame m starts at m*hop")


def test_stage2_stft_window_and_center(nb):
    x = _rng(4).normal(size=300)
    X = np.asarray(nb.stft(x, 32, 32, window=np.ones(32)))
    assert np.allclose(X / 32, _scipy_stft(x, 32, 32, window="boxcar")), "the window argument must be used when given"
    Xn = np.asarray(nb.stft(x, 32, 8, center=False))
    n_frames = 1 + (300 - 32) // 8
    assert Xn.shape == (17, n_frames), f"center=False: no padding, so {n_frames} frames; got shape {Xn.shape}"
    w = scipy.signal.get_window("hann", 32)
    want = np.fft.rfft(np.stack([x[m * 8:m * 8 + 32] * w for m in range(n_frames)]), axis=1).T
    assert np.allclose(Xn, want), "center=False: frame m is x[m*hop : m*hop + n_fft] times the window, then rfft"


# ---------------------------------------------------------------- stage 3: overlap-add and the inverse


def test_stage3_is_cola(nb):
    hp, hs = scipy.signal.get_window("hann", 64), scipy.signal.get_window("hann", 64, fftbins=False)
    cases = [
        (hp, 32, True, "periodic Hann, hop N/2"), (hp, 16, True, "periodic Hann, hop N/4"),
        (hs, 32, False, "symmetric Hann, hop N/2 (the two zeros at the ends overlap badly)"),
        (np.ones(64), 64, True, "rectangular, hop N"), (np.ones(64), 32, True, "rectangular, hop N/2"),
        (hp, 48, False, "periodic Hann, hop 3N/4"), (np.ones(64), 48, False, "rectangular, hop 3N/4"),
        (scipy.signal.get_window("blackman", 64), 32, False, "periodic Blackman, hop N/2"),
        (scipy.signal.get_window("blackman", 60), 20, True, "periodic Blackman, hop N/3"),
        (scipy.signal.get_window("hamming", 64), 32, True, "periodic Hamming, hop N/2 (constant 1.08, not 1)"),
        (scipy.signal.get_window("bartlett", 64), 32, True, "periodic Bartlett, hop N/2"),
        (scipy.signal.get_window("hann", 60), 20, True, "periodic Hann N=60, hop N/3"),
        (scipy.signal.get_window("hann", 60), 25, False, "hop that doesn't divide N"),
    ]
    for w, hop, want, label in cases:
        got = bounded(nb.is_cola, w, hop, what="is_cola")
        assert isinstance(got, (bool, np.bool_)), f"is_cola returns a bool, got {type(got)}"
        assert bool(got) == want, (
            f"{label}: is_cola should be {want}. sum the window's copies shifted by every multiple of hop and "
            "check that the result is constant at every sample (equivalently: fold the window into one hop's "
            "worth of positions and compare)")


def test_stage3_istft_perfect_reconstruction(nb):
    x = _rng(5).normal(size=1024)
    configs = [dict(n_fft=64, hop=16), dict(n_fft=64, hop=32), dict(n_fft=256, hop=64), dict(n_fft=100, hop=40)]
    for c in configs:
        X = nb.stft(x, c["n_fft"], c["hop"])
        y = np.asarray(nb.istft(X, c["hop"], length=len(x)))
        assert y.shape == x.shape, f"length={len(x)} must give exactly {len(x)} samples, got {y.shape}"
        assert np.allclose(y, x, atol=1e-10), (
            f"n_fft={c['n_fft']}, hop={c['hop']}: istft(stft(x)) must give back x (max error "
            f"{np.max(np.abs(y - x)):.2e}). irfft each frame, multiply by the window, overlap-add, and divide by "
            "the overlap-added SQUARED window; then drop the n_fft // 2 padding")
    X = nb.stft(x, 63, 21)
    y = np.asarray(nb.istft(X, 21, n_fft=63, length=len(x)))
    assert np.allclose(y, x, atol=1e-10), "odd n_fft can't be inferred from the bin count: use the n_fft argument"


def test_stage3_istft_does_not_need_cola(nb):
    x = _rng(6).normal(size=800)
    w = scipy.signal.get_window("hann", 64)
    assert not nb.is_cola(w, 24)
    X = nb.stft(x, 64, 24)
    y = np.asarray(nb.istft(X, 24, length=len(x)))
    assert np.allclose(y, x, atol=1e-10), (
        "periodic Hann with hop 24 is not COLA, yet dividing by the actual overlap-added squared window still "
        "inverts an unmodified STFT exactly. if this fails but COLA hops pass, you divided by a constant")
    ws = np.sqrt(w) + 0.1
    X = nb.stft(x, 64, 16, window=ws)
    y = np.asarray(nb.istft(X, 16, window=ws, length=len(x)))
    assert np.allclose(y, x, atol=1e-10), "a custom window must be used in istft too (and its length sets n_fft)"
    wo = scipy.signal.get_window("hann", 63) + 0.05
    X = nb.stft(x, 63, 16, window=wo)
    y = np.asarray(nb.istft(X, 16, window=wo, length=len(x)))
    assert np.allclose(y, x, atol=1e-10), (
        "an odd-length (63) custom window without n_fft: n_fft defaults to len(window), not 2 * (bins - 1) = 62")


def test_stage3_istft_is_least_squares(nb):
    """for a MODIFIED STFT, the result must minimize ||stft(y) - X||^2 (Griffin & Lim, 1984)."""
    n_fft, hop, L = 16, 4, 48
    x = _rng(7).normal(size=L)
    X = np.asarray(nb.stft(x, n_fft, hop))
    X = X * _rng(8).uniform(0.0, 2.0, size=X.shape)  # real gains keep DC and Nyquist real
    y = np.asarray(nb.istft(X, hop, length=L))
    # explicit least squares: windowed frames of the zero-padded signal are linear in y
    w = scipy.signal.get_window("hann", n_fft)
    n_frames = X.shape[1]
    target = np.fft.irfft(X.T, n=n_fft, axis=1)
    A = np.zeros((n_frames * n_fft, L))
    for m in range(n_frames):
        for i in range(n_fft):
            j = m * hop + i - n_fft // 2
            if 0 <= j < L:
                A[m * n_fft + i, j] = w[i]
    want = np.linalg.lstsq(A, target.reshape(-1), rcond=None)[0]
    assert np.allclose(y, want, atol=1e-10), (
        "after modifying the STFT there is no signal with exactly that STFT; istft must return the least-squares "
        "one, which is the window-weighted overlap-add divided by the summed squared window. plain overlap-add "
        "divided by the summed window only works for unmodified STFTs")


def test_stage3_istft_lengths(nb):
    x = _rng(9).normal(size=512)
    X = nb.stft(x, 64, 16)
    y = np.asarray(nb.istft(X, 16))
    assert len(y) == 512 and np.allclose(y, x), (
        f"with center=True and no length, drop n_fft // 2 samples at both ends: 512 samples back, got {len(y)}")
    y2 = np.asarray(nb.istft(X, 16, length=600))
    assert len(y2) == 600 and np.allclose(y2[:512], x) and np.allclose(y2[512:], 0), "a longer length zero-pads"
    Xn = nb.stft(x, 64, 16, center=False)
    yn = np.asarray(nb.istft(Xn, 16, center=False))
    assert len(yn) == 512, f"center=False: n_fft + hop * (n_frames - 1) = 512 samples, got {len(yn)}"
    assert np.allclose(yn[1:], x[1:]), "center=False reconstructs everywhere the window is nonzero (all but sample 0)"
    assert yn[0] == 0.0, (
        f"sample 0 is covered only by w[0] = 0, so the summed squared window is 0 there: the output must be exactly 0 "
        f"(divide only where it exceeds 1e-10 times its maximum), got {yn[0]}")
    wt = scipy.signal.get_window("hann", 64)
    wt[0] = 1e-7  # summed squared window 1e-14 at sample 0: below 1e-10 of its maximum
    yt = np.asarray(nb.istft(nb.stft(x, 64, 16, window=wt, center=False), 16, window=wt, center=False))
    assert yt[0] == 0.0 and np.allclose(yt[1:], x[1:]), (
        f"where the summed squared window is below 1e-10 times its maximum the output is set to 0 (dividing there "
        f"amplifies round-off); with w[0] = 1e-7 sample 0 must be 0, got {yt[0]}")


# ---------------------------------------------------------------- stage 4: the mel scale


def _htk_fb(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = sr / 2 if fmax is None else fmax
    m = np.linspace(2595 * np.log10(1 + fmin / 700), 2595 * np.log10(1 + fmax / 700), n_mels + 2)
    f = 700 * (10 ** (m / 2595) - 1)
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        for k, fk in enumerate(freqs):
            if f[i] <= fk <= f[i + 1]:
                fb[i, k] = (fk - f[i]) / (f[i + 1] - f[i])
            elif f[i + 1] < fk <= f[i + 2]:
                fb[i, k] = (f[i + 2] - fk) / (f[i + 2] - f[i + 1])
    return fb


def test_stage4_mel_scale(nb):
    assert np.isclose(nb.hz_to_mel(700.0), 2595 * np.log10(2)), "HTK: 2595 log10(1 + f / 700), so 700 Hz -> 781.2 mel"
    assert np.isclose(nb.hz_to_mel(0.0), 0.0) and np.isclose(nb.hz_to_mel(1000.0), 999.99, atol=0.05), (
        "0 Hz -> 0 mel and 1000 Hz -> ~1000 mel (the scale is anchored there)")
    f = np.array([0.0, 100.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(nb.mel_to_hz(nb.hz_to_mel(f)), f), "mel_to_hz must invert hz_to_mel elementwise"
    assert np.asarray(nb.hz_to_mel(f)).shape == (5,), "hz_to_mel works on arrays"


def test_stage4_filterbank_worked_example(nb):
    fb = np.asarray(nb.mel_filterbank(8000, 8, 1))
    assert fb.shape == (1, 5), f"(n_mels, n_fft // 2 + 1) = (1, 5), got {fb.shape}"
    assert np.allclose(fb, [[0, 0.8978, 0.6930, 0.3465, 0]], atol=1e-4), (
        f"one filter with edges 0, 1113.9, 4000 Hz evaluated at bins 0, 1000, 2000, 3000, 4000 Hz should be "
        f"(0, 0.898, 0.693, 0.346, 0) (lesson worked example), got {np.round(fb, 4)}. the centre is mel-spaced, "
        "not halfway in Hz, and the peak height is 1 (no area normalization)")


def test_stage4_filterbank_matches_hand_construction(nb):
    for sr, n_fft, n_mels, fmin, fmax in [(16000, 512, 40, 0.0, None), (8000, 256, 20, 100.0, 3800.0),
                                          (22050, 1024, 64, 50.0, None)]:
        got = np.asarray(nb.mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax))
        want = _htk_fb(sr, n_fft, n_mels, fmin, fmax)
        assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape}"
        assert np.allclose(got, want, atol=1e-10), (
            f"sr={sr}, n_fft={n_fft}, n_mels={n_mels}, fmin={fmin}, fmax={fmax}: filters differ from the HTK "
            "construction (n_mels + 2 points equally spaced in mel from fmin to fmax, triangles in Hz between "
            "neighbours, peak 1). common causes: n_mels points instead of n_mels + 2, ignoring fmin/fmax, "
            "spacing in Hz")


def test_stage4_filterbank_partition_of_unity(nb):
    sr, n_fft = 16000, 1024
    fb = np.asarray(nb.mel_filterbank(sr, n_fft, 30))
    edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 32) / 2595) - 1)
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    inside = (freqs >= edges[1]) & (freqs <= edges[-2])
    assert np.allclose(fb.sum(0)[inside], 1.0), (
        "between the first and last centre frequencies, neighbouring triangles sum to exactly 1: each falling "
        "edge is the next filter's rising edge")


def test_stage4_mel_spectrogram(nb):
    sr = 8000
    x = _rng(10).normal(size=2000)
    M = np.asarray(nb.mel_spectrogram(x, sr))
    S = np.abs(_scipy_stft(x, 512, 128) * scipy.signal.get_window("hann", 512).sum())
    fb = _htk_fb(sr, 512, 40)
    assert M.shape == (40, S.shape[1]), f"defaults n_fft=512, hop=128, n_mels=40 give shape {(40, S.shape[1])}, got {M.shape}"
    assert np.allclose(M, fb @ S ** 2), "mel_spectrogram = mel_filterbank @ |stft| ** 2 (power, unscaled STFT)"
    M1 = np.asarray(nb.mel_spectrogram(x, sr, n_fft=256, hop_length=64, n_mels=20, fmin=200.0, fmax=3000.0,
                                       power=1.0))
    S1 = np.abs(_scipy_stft(x, 256, 64) * scipy.signal.get_window("hann", 256).sum())
    assert np.allclose(M1, _htk_fb(sr, 256, 20, 200.0, 3000.0) @ S1), (
        "every keyword (n_fft, hop_length, n_mels, fmin, fmax, power) must be passed through")


def test_stage4_power_to_db(nb):
    S = np.array([1.0, 10.0, 100.0, 1e-3, 0.0])
    got = np.asarray(nb.power_to_db(S, top_db=None))
    assert np.allclose(got, [0, 10, 20, -30, -100]), (
        f"10 log10(max(S, 1e-10)): expected (0, 10, 20, -30, -100), got {got}. zeros are floored at amin")
    assert np.allclose(nb.power_to_db(S, ref=10.0, top_db=None), [-10, 0, 10, -40, -110]), "divide by ref"
    assert np.allclose(nb.power_to_db(S), [0, 10, 20, -30, -60]), "default top_db=80: nothing below max - 80 dB"
    assert np.allclose(nb.power_to_db(S, top_db=25.0), [0, 10, 20, -5, -5]), "top_db clips at max - top_db"
    assert np.allclose(nb.power_to_db(S, amin=1e-2, top_db=None), [0, 10, 20, -20, -20]), "amin is the floor"


# ---------------------------------------------------------------- stage 5: MFCCs


def test_stage5_dct2_matches_scipy(nb):
    x = _rng(11).normal(size=(12, 7))
    assert np.allclose(nb.dct2(x), scipy.fft.dct(x, type=2, norm="ortho", axis=0)), (
        "dct2 along axis 0 must equal scipy.fft.dct(x, type=2, norm='ortho', axis=0): row 0 scaled by sqrt(1/N), "
        "the rest by sqrt(2/N)")
    assert np.allclose(nb.dct2(x, axis=1), scipy.fft.dct(x, type=2, norm="ortho", axis=1)), "the axis argument"
    v = _rng(12).normal(size=9)
    assert np.allclose(nb.dct2(v), scipy.fft.dct(v, norm="ortho")), "1-D input"
    assert np.isclose(np.sum(np.asarray(nb.dct2(v)) ** 2), np.sum(v ** 2)), "an orthonormal DCT preserves energy"


def test_stage5_mfcc_matches_reference(nb):
    sr = 8000
    rng = _rng(13)
    x = np.concatenate([np.zeros(1000), rng.normal(size=1500), 1e-4 * rng.normal(size=1000)])
    for kw in [{}, dict(n_mfcc=20, n_fft=256, hop_length=80, n_mels=30, fmin=100.0, fmax=3500.0)]:
        got = np.asarray(nb.mfcc(x, sr, **kw))
        n_fft, hop = kw.get("n_fft", 512), kw.get("hop_length", 128)
        S = np.abs(_scipy_stft(x, n_fft, hop) * scipy.signal.get_window("hann", n_fft).sum()) ** 2
        M = _htk_fb(sr, n_fft, kw.get("n_mels", 40), kw.get("fmin", 0.0), kw.get("fmax")) @ S
        want = scipy.fft.dct(10 * np.log10(np.maximum(M, 1e-10)), type=2, norm="ortho", axis=0)[:kw.get("n_mfcc", 13)]
        assert got.shape == want.shape, f"expected shape (n_mfcc, n_frames) = {want.shape}, got {got.shape}"
        assert np.allclose(got, want, atol=1e-8), (
            f"mfcc with {kw or 'defaults'} differs from dct2(power_to_db(mel, top_db=None))[:n_mfcc]. check the "
            "log (dB, 10 log10, floor 1e-10, NO top_db clipping: this signal has near-silent frames), the DCT axis "
            "(mel bands, axis 0) and that every keyword is passed through")


def test_stage5_mfcc_gain_moves_only_c0(nb):
    sr = 8000
    x = _rng(14).normal(size=3000)
    a, b = np.asarray(nb.mfcc(x, sr)), np.asarray(nb.mfcc(10 * x, sr))
    assert np.allclose(b[0] - a[0], 20 * np.sqrt(40)), (
        "10x the amplitude adds 20 dB to every mel band; through an orthonormal DCT that adds 20 sqrt(n_mels) to "
        f"c0, got {np.mean(b[0] - a[0]):.3f} instead of {20 * np.sqrt(40):.3f}")
    assert np.allclose(b[1:], a[1:]), "a gain change must leave c1, c2, ... unchanged: they describe spectral shape"
