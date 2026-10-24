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
    # neural audio codecs · build

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
    raise NotImplementedError("stage1: implement nearest_codes")


@app.function
def rvq_encode(X, codebooks):
    """stage 1: (N, S) int64 codes: stage s quantizes what the first s stages left over."""
    raise NotImplementedError("stage1: implement rvq_encode")


@app.function
def rvq_decode(codes, codebooks):
    """stage 1: (N, D) sum over the stages present in codes (N, S') of codebooks[s][codes[:, s]]; S' <= S."""
    raise NotImplementedError("stage1: implement rvq_decode")


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
    raise NotImplementedError("stage2: implement fit_rvq")


@app.function
def rvq_distortion(X, codebooks):
    """stage 2: (S,) array: mean squared error per coordinate after 1, 2, ..., S stages."""
    raise NotImplementedError("stage2: implement rvq_distortion")


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
    raise NotImplementedError("stage3: implement bitrate")


@app.function
def entropy_bits(codes, n_codes):
    """stage 3: Shannon entropy (bits) of the empirical distribution of the integer codes, a float in [0, log2 n_codes]."""
    raise NotImplementedError("stage3: implement entropy_bits")


@app.function
def pcm_quantize(x, bits, *, mu=None):
    """stage 3: x in [-1, 1] -> 2**bits midrise uniform levels (mu-law companded first if mu is given); same shape."""
    raise NotImplementedError("stage3: implement pcm_quantize")


@app.function
def snr_db(ref, est):
    """stage 3: 10 log10(sum ref^2 / sum (ref - est)^2) in dB."""
    raise NotImplementedError("stage3: implement snr_db")


@app.function
def log_spectral_distance(ref, est, *, n_fft=256, hop_length=64, eps=1e-10):
    """stage 3: mean over frames of the RMS over bins of 10 log10(|R|^2 + eps) - 10 log10(|E|^2 + eps), from audio.stft."""
    raise NotImplementedError("stage3: implement log_spectral_distance")


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
        raise NotImplementedError("stage4: implement ResidualVQ.__init__")

    def forward(self, z_e, n_stages=None):
        """stage 4: (z_e + sg(q - z_e), sum of the used stages' vq losses, codes (B, n_used, T) int64); q = sum of chosen codes."""
        raise NotImplementedError("stage4: implement ResidualVQ.forward")

    def decode_indices(self, codes):
        """stage 4: codes (B, S', T) -> (B, D, T): the sum of the first S' stages' code vectors."""
        raise NotImplementedError("stage4: implement ResidualVQ.decode_indices")

    def init_from_data(self, z_e, *, n_iters=10, rng=None):
        """stage 4: set every stage's codebook to fit_rvq(z_e as (N, D) vectors, n_codes, n_stages) (no grad); returns self."""
        raise NotImplementedError("stage4: implement ResidualVQ.init_from_data")


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
        raise NotImplementedError("stage5: implement ConvCodec.__init__")

    def _check(self, x):
        if x.dim() != 3 or x.shape[1] != 1 or x.shape[-1] % self.hop_length:
            raise ValueError(f"expected (B, 1, L) with L divisible by {self.hop_length}, got {tuple(x.shape)}")

    def forward(self, x, n_stages=None):
        """stage 5: (x_hat (B, 1, L), vq_loss, codes (B, n_used, L / hop_length)); ValueError unless L % hop_length == 0."""
        raise NotImplementedError("stage5: implement ConvCodec.forward")

    def encode(self, x, n_stages=None):
        """stage 5: codes (B, n_used, L / hop_length) int64."""
        raise NotImplementedError("stage5: implement ConvCodec.encode")

    def decode(self, codes):
        """stage 5: waveform (B, 1, T * hop_length) from codes (B, S', T)."""
        raise NotImplementedError("stage5: implement ConvCodec.decode")


@app.function
def multiscale_spectral_loss(x, x_hat, *, n_ffts=(64, 128, 256, 512), eps=1e-5):
    """stage 5: mean over n_ffts of mean|S - S_hat| + mean|log(S + eps) - log(S_hat + eps)|, S = |torch.stft| (hop n/4)."""
    raise NotImplementedError("stage5: implement multiscale_spectral_loss")


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
