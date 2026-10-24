import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson, notebooks
    from mlfs.generative import VectorQuantizer  # yours, from ch 21
    from mlfs.unsupervised import kmeans_plus_plus  # yours, from ch 11

    def _audio_smoke(m):
        """touch the topic 24 function this notebook uses."""
        m.stft(np.zeros(64), 16, 4)

    # your topic 24 STFT: the reference under pytest and verify.py (or with MLFS_REFERENCE=audio24), and also
    # whenever the topic 24 stage it needs is still unfinished
    audio_load = notebooks.load_topic("24-audio-representations", key="audio24", smoke=_audio_smoke)
    audio = audio_load.module


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # neural audio codecs · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. your topic 24 STFT is available as `audio.stft`.

    | stage | you build |
    |---|---|
    | 1 | `nearest_codes`, `rvq_encode`, `rvq_decode`: residual vector quantization with given codebooks |
    | 2 | `fit_rvq`, `rvq_distortion`: codebooks by k-means on residuals, stage by stage |
    | 3 | `bitrate`, `entropy_bits`, `pcm_quantize`, `snr_db`, `log_spectral_distance`: rates, a baseline and metrics |
    | 4 | `ResidualVQ`: a trainable residual quantizer built from your ch 21 `VectorQuantizer` |
    | 5 | `ConvCodec`, `multiscale_spectral_loss`: an encoder–RVQ–decoder codec and its training loss |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    audio_load.callout()
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def nearest_codes(X, codebook):
    """stage 1: index (N,) int64 of the nearest row of codebook (K, D) for every row of X (N, D), squared Euclidean."""
    X, C = np.asarray(X, dtype=float), np.asarray(codebook, dtype=float)
    d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    return d.argmin(axis=1).astype(np.int64)


@app.function
def rvq_encode(X, codebooks):
    """stage 1: (N, S) int64 codes: stage s quantizes what the first s stages left over."""
    r = np.asarray(X, dtype=float).copy()
    codes = []
    for C in codebooks:
        C = np.asarray(C, dtype=float)
        idx = nearest_codes(r, C)
        r -= C[idx]
        codes.append(idx)
    return np.stack(codes, axis=1) if codes else np.zeros((len(r), 0), dtype=np.int64)


@app.function
def rvq_decode(codes, codebooks):
    """stage 1: (N, D) sum over the stages present in codes (N, S') of codebooks[s][codes[:, s]]; S' <= S."""
    codes = np.asarray(codes)
    D = np.asarray(codebooks[0]).shape[1]
    out = np.zeros((codes.shape[0], D))
    for s in range(codes.shape[1]):
        out += np.asarray(codebooks[s], dtype=float)[codes[:, s]]
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, nearest_codes=nearest_codes, rvq_encode=rvq_encode, rvq_decode=rvq_decode)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def fit_rvq(X, n_codes, n_stages, *, n_iters=20, rng=None):
    """stage 2: list of n_stages (n_codes, D) codebooks, each k-means (k-means++ init, n_iters Lloyd steps) on the residuals."""
    rng = np.random.default_rng(rng)
    r = np.asarray(X, dtype=float).copy()
    books = []
    for _ in range(n_stages):
        C = np.asarray(kmeans_plus_plus(r, n_codes, rng=rng), dtype=float).copy()
        for _ in range(n_iters):
            idx = nearest_codes(r, C)
            sums = np.zeros_like(C)
            np.add.at(sums, idx, r)
            counts = np.bincount(idx, minlength=n_codes)
            nz = counts > 0
            C[nz] = sums[nz] / counts[nz, None]
        r -= C[nearest_codes(r, C)]
        books.append(C)
    return books


@app.function
def rvq_distortion(X, codebooks):
    """stage 2: (S,) array: mean squared error per coordinate after 1, 2, ..., S stages."""
    X = np.asarray(X, dtype=float)
    codes = rvq_encode(X, codebooks)
    return np.array([np.mean((X - rvq_decode(codes[:, :s], codebooks)) ** 2) for s in range(1, len(codebooks) + 1)])


@app.cell(hide_code=True)
def _():
    checks.run(2, fit_rvq=fit_rvq, rvq_distortion=rvq_distortion, rvq_encode=rvq_encode, rvq_decode=rvq_decode)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def bitrate(sample_rate, hop_length, n_stages, n_codes):
    """stage 3: bits per second = (sample_rate / hop_length) frames/s * n_stages * log2(n_codes) bits per frame."""
    return sample_rate / hop_length * n_stages * math.log2(n_codes)


@app.function
def entropy_bits(codes, n_codes):
    """stage 3: Shannon entropy (bits) of the empirical distribution of the integer codes, a float in [0, log2 n_codes]."""
    counts = np.bincount(np.asarray(codes).ravel(), minlength=n_codes).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


@app.function
def pcm_quantize(x, bits, *, mu=None):
    """stage 3: x in [-1, 1] -> 2**bits midrise uniform levels (mu-law companded first if mu is given); same shape."""
    x = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    if mu is not None:
        x = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    L = 2 ** bits
    q = (np.floor((x + 1) / 2 * L).clip(0, L - 1) + 0.5) / L * 2 - 1
    if mu is not None:
        q = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return q


@app.function
def snr_db(ref, est):
    """stage 3: 10 log10(sum ref^2 / sum (ref - est)^2) in dB."""
    ref, est = np.asarray(ref, dtype=float), np.asarray(est, dtype=float)
    return float(10 * np.log10(np.sum(ref ** 2) / np.sum((ref - est) ** 2)))


@app.function
def log_spectral_distance(ref, est, *, n_fft=256, hop_length=64, eps=1e-10):
    """stage 3: mean over frames of the RMS over bins of 10 log10(|R|^2 + eps) - 10 log10(|E|^2 + eps), from audio.stft."""
    R = 10 * np.log10(np.abs(audio.stft(np.asarray(ref, dtype=float), n_fft, hop_length)) ** 2 + eps)
    E = 10 * np.log10(np.abs(audio.stft(np.asarray(est, dtype=float), n_fft, hop_length)) ** 2 + eps)
    return float(np.mean(np.sqrt(np.mean((R - E) ** 2, axis=0))))


@app.cell(hide_code=True)
def _():
    checks.run(3, bitrate=bitrate, entropy_bits=entropy_bits, pcm_quantize=pcm_quantize, snr_db=snr_db,
               log_spectral_distance=log_spectral_distance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class ResidualVQ(nn.Module):
    """n_stages VectorQuantizers in residual cascade over z_e (B, D, T)."""

    def __init__(self, n_stages, n_codes, code_dim, *, beta=0.25):
        """stage 4: self.stages = nn.ModuleList of n_stages VectorQuantizer(n_codes, code_dim, beta=beta) (ch 21)."""
        super().__init__()
        self.n_stages, self.n_codes, self.code_dim = n_stages, n_codes, code_dim
        self.stages = nn.ModuleList(VectorQuantizer(n_codes, code_dim, beta=beta) for _ in range(n_stages))

    def forward(self, z_e, n_stages=None):
        """stage 4: (z_e + sg(q - z_e), sum of the used stages' vq losses, codes (B, n_used, T) int64); q = sum of chosen codes."""
        n = self.n_stages if n_stages is None else n_stages
        r = z_e
        q = torch.zeros_like(z_e)
        loss = z_e.new_zeros(())
        codes = []
        for vq in self.stages[:n]:
            z_st, l, idx = vq(r)
            q_s = z_st.detach()
            q = q + q_s
            r = r - q_s
            loss = loss + l
            codes.append(idx)
        z_q = z_e + (q - z_e).detach()
        return z_q, loss, torch.stack(codes, dim=1)

    def decode_indices(self, codes):
        """stage 4: codes (B, S', T) -> (B, D, T): the sum of the first S' stages' code vectors."""
        out = 0
        for s in range(codes.shape[1]):
            out = out + self.stages[s].codebook(codes[:, s]).movedim(-1, 1)
        return out

    def init_from_data(self, z_e, *, n_iters=10, rng=None):
        """stage 4: set every stage's codebook to fit_rvq(z_e as (N, D) vectors, n_codes, n_stages) (no grad); returns self."""
        flat = z_e.detach().movedim(1, -1).reshape(-1, z_e.shape[1]).cpu().double().numpy()
        books = fit_rvq(flat, self.n_codes, self.n_stages, n_iters=n_iters, rng=rng)
        with torch.no_grad():
            for vq, C in zip(self.stages, books):
                vq.codebook.weight.copy_(torch.as_tensor(C, dtype=vq.codebook.weight.dtype))
        return self


@app.cell(hide_code=True)
def _():
    checks.run(4, ResidualVQ=ResidualVQ, rvq_encode=rvq_encode, fit_rvq=fit_rvq)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class ConvCodec(nn.Module):
    """waveform codec: x (B, 1, L) -> encoder (B, code_dim, L / hop) -> ResidualVQ -> decoder (B, 1, L)."""

    def __init__(self, *, channels=32, code_dim=16, strides=(2, 4, 4), n_stages=8, n_codes=256):
        """stage 5: attributes encoder, quantizer (ResidualVQ), decoder, hop_length = prod(strides)."""
        super().__init__()
        self.hop_length = int(np.prod(strides))
        enc, c_in, c_out = [], 1, channels
        for s in strides:  # each block divides the length by its (even) stride exactly
            enc += [nn.Conv1d(c_in, c_out, 2 * s, stride=s, padding=s // 2), nn.ELU()]
            c_in, c_out = c_out, 2 * c_out
        enc += [nn.Conv1d(c_in, code_dim, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)
        dec = [nn.Conv1d(code_dim, c_in, 3, padding=1), nn.ELU()]
        for k, s in enumerate(reversed(strides)):
            last = k == len(strides) - 1
            dec += [nn.ConvTranspose1d(c_in, 1 if last else c_in // 2, 2 * s, stride=s, padding=s // 2)]
            if not last:
                dec += [nn.ELU()]
                c_in //= 2
        self.decoder = nn.Sequential(*dec)
        self.quantizer = ResidualVQ(n_stages, n_codes, code_dim)

    def _check(self, x):
        if x.dim() != 3 or x.shape[1] != 1 or x.shape[-1] % self.hop_length:
            raise ValueError(f"expected (B, 1, L) with L divisible by {self.hop_length}, got {tuple(x.shape)}")

    def forward(self, x, n_stages=None):
        """stage 5: (x_hat (B, 1, L), vq_loss, codes (B, n_used, L / hop_length)); ValueError unless L % hop_length == 0."""
        self._check(x)
        z_q, loss, codes = self.quantizer(self.encoder(x), n_stages)
        return self.decoder(z_q), loss, codes

    def encode(self, x, n_stages=None):
        """stage 5: codes (B, n_used, L / hop_length) int64."""
        self._check(x)
        return self.quantizer(self.encoder(x), n_stages)[2]

    def decode(self, codes):
        """stage 5: waveform (B, 1, T * hop_length) from codes (B, S', T)."""
        return self.decoder(self.quantizer.decode_indices(codes))


@app.function
def multiscale_spectral_loss(x, x_hat, *, n_ffts=(64, 128, 256, 512), eps=1e-5):
    """stage 5: mean over n_ffts of mean|S - S_hat| + mean|log(S + eps) - log(S_hat + eps)|, S = |torch.stft| (hop n/4)."""
    x, x_hat = x.reshape(-1, x.shape[-1]), x_hat.reshape(-1, x_hat.shape[-1])
    total = 0.0
    for n in n_ffts:
        w = torch.hann_window(n, dtype=x.dtype, device=x.device)
        S = torch.stft(x, n, n // 4, window=w, center=True, pad_mode="constant", return_complex=True).abs()
        Sh = torch.stft(x_hat, n, n // 4, window=w, center=True, pad_mode="constant", return_complex=True).abs()
        total = total + (S - Sh).abs().mean() + (torch.log(S + eps) - torch.log(Sh + eps)).abs().mean()
    return total / len(n_ffts)


@app.cell(hide_code=True)
def _():
    checks.run(5, ConvCodec=ConvCodec, multiscale_spectral_loss=multiscale_spectral_loss, ResidualVQ=ResidualVQ)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it measures how residual
    quantization trades bits for error, then runs the block's **project**: train your codec
    on synthetic audio and map its bitrate against quality, next to classic PCM baselines.
    """)
    return


if __name__ == "__main__":
    app.run()
