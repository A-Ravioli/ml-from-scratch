"""Checks for exercise.py (neural audio codecs).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Residual quantization is
checked by hand (the lesson's worked example) and against an independent greedy encoder; k-means
codebooks against Lloyd's fixed point and your ch 11 k-means++; rates and metrics against hand
computations and SciPy; the torch quantizer against the NumPy one; the codec for shapes, contract
and the straight-through gradient. All signals are synthetic.
"""
import numpy as np
import scipy.signal
import torch

from mlfs.generative import VectorQuantizer
from mlfs.unsupervised import kmeans_plus_plus

torch.set_num_threads(1)  # small tensors: one thread is fastest, and avoids oversubscribing a busy machine


def _rng(seed=0):
    return np.random.default_rng(seed)


def _greedy_rvq(X, books):
    r = X.copy()
    codes = []
    for C in books:
        d = ((r[:, None, :] - C[None]) ** 2).sum(-1)
        i = d.argmin(1)
        codes.append(i)
        r = r - C[i]
    return np.stack(codes, 1), r


# ---------------------------------------------------------------- stage 1: residual vector quantization


def test_stage1_nearest_codes(nb):
    X, C = _rng(0).normal(size=(50, 3)), _rng(1).normal(size=(7, 3))
    got = np.asarray(nb.nearest_codes(X, C))
    want = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    assert got.shape == (50,) and np.issubdtype(got.dtype, np.integer), f"(N,) integer indices, got {got.shape} {got.dtype}"
    assert np.array_equal(got, want), "index of the nearest codeword by squared Euclidean distance, for every row"


def test_stage1_worked_example(nb):
    books = [np.array([[0.0, 0.0], [4.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])]
    X = np.array([[5.0, 0.9]])
    codes = np.asarray(nb.rvq_encode(X, books))
    assert codes.shape == (1, 2) and codes.tolist() == [[1, 0]], (
        f"(5, 0.9): stage 1 picks (4, 0) (code 1), leaving (1, 0.9); stage 2 picks (1, 0) (code 0, distance 0.81 "
        f"< 1.01 for (0, 1)). expected [[1, 0]], got {codes.tolist()}. stage 2 must quantize the RESIDUAL")
    assert np.allclose(nb.rvq_decode(codes, books), [[5.0, 0.0]]), "decode sums the chosen codewords: (5, 0)"
    assert np.allclose(nb.rvq_decode(codes[:, :1], books), [[4.0, 0.0]]), "decoding only the first stage: (4, 0)"


def test_stage1_matches_greedy_residual(nb):
    g = _rng(2)
    X = g.normal(size=(200, 4)) * 3
    books = [g.normal(size=(8, 4)) * s for s in (3.0, 1.0, 0.3)]
    codes = np.asarray(nb.rvq_encode(X, books))
    want, r = _greedy_rvq(X, books)
    assert codes.shape == (200, 3) and np.issubdtype(codes.dtype, np.integer), f"(N, S) integer codes, got {codes.shape}"
    assert np.array_equal(codes, want), "stage s picks the codeword nearest to what stages 1..s-1 left over"
    assert np.allclose(nb.rvq_decode(codes, books), X - r), "decode(encode(X)) = X minus the final residual"
    assert np.allclose(nb.rvq_decode(codes[:, :2], books), books[0][codes[:, 0]] + books[1][codes[:, 1]]), (
        "decoding with fewer stages than codebooks uses the first S' codebooks")


def test_stage1_error_never_grows_with_zero_codeword(nb):
    g = _rng(3)
    X = g.normal(size=(300, 3))
    books = [np.vstack([np.zeros(3), g.normal(size=(5, 3)) * s]) for s in (1.0, 0.5, 0.25, 0.1)]
    codes = np.asarray(nb.rvq_encode(X, books))
    errs = np.stack([((X - nb.rvq_decode(codes[:, :s], books)) ** 2).sum(1) for s in range(1, 5)])
    assert np.all(np.diff(errs, axis=0) <= 1e-12), (
        "when every codebook contains the zero vector, adding a stage can never make any vector's error larger "
        "(it can always pick 0). yours did: is each stage really choosing the nearest codeword to the residual?")


# ---------------------------------------------------------------- stage 2: k-means codebooks, stage by stage


def _clustered(n=1500, seed=4):
    g = _rng(seed)
    centres = g.normal(size=(6, 5)) * 4
    return centres[g.integers(6, size=n)] + g.normal(size=(n, 5))


def test_stage2_fit_shapes_and_determinism(nb):
    X = _clustered()
    books = nb.fit_rvq(X, 8, 3, n_iters=10, rng=0)
    assert len(books) == 3 and all(np.asarray(C).shape == (8, 5) for C in books), (
        f"a list of n_stages = 3 codebooks of shape (n_codes, D) = (8, 5), got {[np.asarray(C).shape for C in books]}")
    again = nb.fit_rvq(X, 8, 3, n_iters=10, rng=0)
    assert all(np.allclose(a, b) for a, b in zip(books, again)), "the same rng seed gives the same codebooks"
    init = nb.fit_rvq(X, 8, 1, n_iters=0, rng=5)[0]
    assert np.allclose(init, kmeans_plus_plus(X, 8, rng=np.random.default_rng(5))), (
        "n_iters=0 returns the initialization: kmeans_plus_plus(residuals, n_codes, rng=rng) from your ch 11 "
        "k-means, with rng = np.random.default_rng(rng)")


def _ref_fit(X, K, S, n_iters, seed):
    rng = np.random.default_rng(seed)
    r = X.copy()
    books = []
    for _ in range(S):
        C = np.asarray(kmeans_plus_plus(r, K, rng=rng), dtype=float).copy()
        for _ in range(n_iters):
            i = ((r[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            for k in range(K):
                if np.any(i == k):
                    C[k] = r[i == k].mean(0)
        r = r - C[((r[:, None] - C[None]) ** 2).sum(-1).argmin(1)]
        books.append(C)
    return books


def test_stage2_each_stage_is_kmeans_on_residuals(nb):
    X = _clustered(seed=6)
    got = [np.asarray(C, dtype=float) for C in nb.fit_rvq(X, 6, 3, n_iters=8, rng=1)]
    want = _ref_fit(X, 6, 3, 8, 1)
    assert np.allclose(got[0], want[0], atol=1e-8), (
        "stage 1: k-means++ init (one rng = np.random.default_rng(rng) shared by all stages), then n_iters Lloyd "
        "steps (assign every vector to its nearest codeword, move each used codeword to the mean of its vectors)")
    assert np.allclose(got[1], want[1], atol=1e-8) and np.allclose(got[2], want[2], atol=1e-8), (
        "stages 2 and 3 must run k-means on the RESIDUALS X - (sum of the previous stages' chosen codewords), "
        "not on X itself, continuing with the same rng")


def test_stage2_empty_codewords_stay(nb):
    pts = _rng(20).normal(size=(5, 3)) * 3
    X = np.repeat(pts, 20, axis=0)  # only 5 distinct vectors, but 8 codewords: some codewords get no vectors
    books = nb.fit_rvq(X, 8, 2, n_iters=5, rng=0)
    assert all(np.all(np.isfinite(np.asarray(C))) for C in books), (
        "a codeword that no vector chose must keep its position; dividing an empty sum by a zero count gives NaN")
    assert np.allclose(nb.rvq_distortion(X, books), 0.0), "5 distinct vectors and 8 codewords: zero error"
    d = ((np.asarray(books[0])[:, None] - pts[None]) ** 2).sum(-1).min(1)
    assert np.allclose(d, 0.0), (
        "every stage-1 codeword should sit on one of the 5 data vectors: k-means++ starts them there, and an "
        "unused codeword keeps its position (yours moved somewhere else, e.g. to 0)")


def test_stage2_distortion(nb):
    X = _clustered(seed=7)
    books = nb.fit_rvq(X, 8, 4, n_iters=15, rng=2)
    d = np.asarray(nb.rvq_distortion(X, books))
    codes, _ = _greedy_rvq(X, [np.asarray(C) for C in books])
    want = [np.mean((X - sum(np.asarray(books[j])[codes[:, j]] for j in range(s))) ** 2) for s in range(1, 5)]
    assert d.shape == (4,) and np.allclose(d, want), (
        f"rvq_distortion: the mean squared error PER COORDINATE after 1..S stages, expected {np.round(want, 4)}, "
        f"got {np.round(d, 4)}")
    assert np.all(np.diff(d) < 0), f"on the training data every k-means stage must lower the error, got {np.round(d, 4)}"


# ---------------------------------------------------------------- stage 3: rates, a baseline, metrics


def test_stage3_bitrate_and_entropy(nb):
    assert np.isclose(nb.bitrate(8000, 32, 8, 256), 16000.0), "8000/32 = 250 frames/s x 8 stages x 8 bits = 16 kbps"
    assert np.isclose(nb.bitrate(24000, 320, 4, 1024), 3000.0), "EnCodec-like: 75 frames/s x 4 x 10 bits = 3 kbps"
    assert np.isclose(nb.bitrate(16000, 160, 3, 1000), 100 * 3 * np.log2(1000)), "log2(n_codes) need not be an integer"
    assert np.isclose(nb.entropy_bits(np.array([0, 1, 2, 3]), 4), 2.0), "uniform over 4 codes: 2 bits"
    assert np.isclose(nb.entropy_bits(np.array([5, 5, 5]), 8), 0.0), "always the same code: 0 bits"
    p = np.array([1 / 6, 5 / 6])
    got = nb.entropy_bits(np.array([[0, 1], [1, 1], [1, 1]]), 256)
    assert np.isclose(got, -(p * np.log2(p)).sum()), (
        f"codes 0 (x1) and 1 (x5) in a 2-D array (count every entry; unused codes contribute nothing): H = 0.650 "
        f"bits, got {got}")


def test_stage3_pcm_quantize(nb):
    x = np.array([-1.0, -0.6, -0.2, 0.0, 0.3, 0.7, 1.0, 1.7, -3.0])
    got = np.asarray(nb.pcm_quantize(x, 2))
    assert np.allclose(got, [-0.75, -0.75, -0.25, 0.25, 0.25, 0.75, 0.75, 0.75, -0.75]), (
        f"2 bits = 4 midrise levels at -0.75, -0.25, 0.25, 0.75 (cells of width 0.5 over [-1, 1]); inputs outside "
        f"[-1, 1] are clipped. got {got}")
    g = _rng(8)
    x = g.uniform(-1, 1, 5000)
    for b in (1, 3, 6):
        q = np.asarray(nb.pcm_quantize(x, b))
        assert q.shape == x.shape and len(np.unique(q)) <= 2 ** b, f"{b} bits: at most {2 ** b} distinct levels"
        assert np.max(np.abs(q - x)) <= 1 / 2 ** b + 1e-12, f"{b} bits: error at most half a step, 1/2^{b}"
    mu = 255
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    yq = (np.floor((y + 1) / 2 * 16).clip(0, 15) + 0.5) / 16 * 2 - 1
    want = np.sign(yq) * ((1 + mu) ** np.abs(yq) - 1) / mu
    assert np.allclose(nb.pcm_quantize(x, 4, mu=mu), want), (
        "with mu: compress y = sign(x) ln(1 + mu|x|) / ln(1 + mu), quantize y uniformly, expand with "
        "sign(q) ((1 + mu)^|q| - 1) / mu")
    quiet = 0.01 * g.normal(size=2000)
    e_lin = np.mean((nb.pcm_quantize(quiet, 6) - quiet) ** 2)
    e_mu = np.mean((nb.pcm_quantize(quiet, 6, mu=255) - quiet) ** 2)
    assert e_mu < e_lin / 4, "for quiet signals, mu-law spends its levels where the signal is: far smaller error"


def test_stage3_snr_and_lsd(nb):
    assert np.isclose(nb.snr_db(np.array([1.0, 1.0]), np.array([1.0, 0.0])), 10 * np.log10(2)), "2 / 1 -> 3.01 dB"
    x = _rng(9).normal(size=1000)
    assert np.isclose(nb.snr_db(x, 0.9 * x), 20.0), "an error of 10% of the signal everywhere is 20 dB"
    assert np.isclose(nb.log_spectral_distance(x, x), 0.0), "identical signals: 0 dB"
    assert np.isclose(nb.log_spectral_distance(x, 2 * x), 20 * np.log10(2), atol=1e-6), (
        "doubling the amplitude raises every bin by 6.02 dB, so the distance is 6.02 dB")
    y = x + 0.3 * _rng(10).normal(size=1000)
    for n_fft, hop in [(256, 64), (128, 32)]:
        w = scipy.signal.get_window("hann", n_fft)
        R = np.abs(scipy.signal.stft(x, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros",
                                     padded=False)[2] * w.sum()) ** 2
        E = np.abs(scipy.signal.stft(y, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros",
                                     padded=False)[2] * w.sum()) ** 2
        want = np.mean(np.sqrt(np.mean((10 * np.log10(R + 1e-10) - 10 * np.log10(E + 1e-10)) ** 2, axis=0)))
        got = nb.log_spectral_distance(x, y, n_fft=n_fft, hop_length=hop)
        assert np.isclose(got, want), (
            f"n_fft={n_fft}, hop={hop}: expected {want:.4f} dB, got {got:.4f}. per frame, the RMS over bins of the "
            "difference of 10 log10(|STFT|^2 + eps) (your topic 24 stft); then the mean over frames")


def _scipy_lsd(x, y, n_fft, hop, eps):
    w = scipy.signal.get_window("hann", n_fft)

    def pw(v):
        return np.abs(scipy.signal.stft(v, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros",
                                        padded=False)[2] * w.sum()) ** 2

    return np.mean(np.sqrt(np.mean((10 * np.log10(pw(x) + eps) - 10 * np.log10(pw(y) + eps)) ** 2, axis=0)))


def test_stage3_lsd_eps_floors_quiet_bins(nb):
    g = _rng(21)
    t = np.arange(2048) / 8000
    x = 0.1 * np.sin(2 * np.pi * 440 * t) + 1e-5 * g.normal(size=2048)  # one loud line, everything else very quiet
    y = 0.1 * np.sin(2 * np.pi * 440 * t + 0.1) + 1e-3 * g.normal(size=2048)
    for eps in (1e-2, 1e-6):
        got = nb.log_spectral_distance(x, y, eps=eps)
        want = _scipy_lsd(x, y, 256, 64, eps)
        assert np.isclose(got, want, rtol=1e-6), (
            f"eps={eps}: expected {want:.3f} dB, got {got:.3f}. eps is added to the POWER |STFT|^2 before the log "
            "(10 log10(|X|^2 + eps)); it floors the quiet bins, so it changes the result a lot here")


# ---------------------------------------------------------------- stage 4: a trainable residual quantizer


def _z(seed=11, B=3, D=4, T=10):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, D, T, generator=g, dtype=torch.float64)


def _rvq(nb, S=3, K=6, D=4, seed=0, beta=0.25):
    torch.manual_seed(seed)
    q = nb.ResidualVQ(S, K, D, beta=beta).double()
    with torch.no_grad():
        for s, vq in enumerate(q.stages):
            vq.codebook.weight.normal_(0, 1.0 / (s + 1))
    return q


def test_stage4_structure_and_shapes(nb):
    q = _rvq(nb)
    assert len(q.stages) == 3 and all(isinstance(v, VectorQuantizer) for v in q.stages), (
        "self.stages: an nn.ModuleList of n_stages VectorQuantizer modules from mlfs.generative (your ch 21 code)")
    assert all(v.codebook.weight.shape == (6, 4) for v in q.stages), "each stage: n_codes x code_dim codebook"
    assert all(v.beta == 0.5 for v in nb.ResidualVQ(2, 4, 3, beta=0.5).stages), "beta is passed to every stage"
    z = _z()
    zq, loss, codes = q(z)
    assert zq.shape == z.shape and codes.shape == (3, 3, 10) and codes.dtype == torch.int64, (
        f"z_q (B, D, T) and codes (B, n_stages, T) int64; got {tuple(zq.shape)}, {tuple(codes.shape)} {codes.dtype}")
    assert loss.dim() == 0, "the vq loss is a scalar"
    assert q(z, n_stages=2)[2].shape == (3, 2, 10), "n_stages=2 uses (and returns codes for) the first 2 stages only"


def test_stage4_matches_numpy_rvq(nb):
    q = _rvq(nb)
    z = _z()
    zq, _, codes = q(z)
    books = [v.codebook.weight.detach().numpy() for v in q.stages]
    flat = z.movedim(1, -1).reshape(-1, 4).numpy()
    want = np.asarray(nb.rvq_encode(flat, books)).reshape(3, 10, 3).transpose(0, 2, 1)
    assert np.array_equal(codes.numpy(), want), (
        "the torch quantizer must pick the same codes as your stage 1 rvq_encode on the (B*T, D) vectors: stage s "
        "quantizes z_e minus the sum of the previous stages' code vectors")
    assert torch.allclose(zq, q.decode_indices(codes)), "z_q's value is the sum of the chosen code vectors"
    assert torch.allclose(q.decode_indices(codes[:, :1]), q.stages[0].codebook(codes[:, 0]).movedim(-1, 1)), (
        "decode_indices with S' < S codes sums only the first S' stages")
    zq2, _, _ = q(z, n_stages=2)
    assert torch.allclose(zq2, q.decode_indices(codes[:, :2])), "n_stages=2: z_q is the 2-stage reconstruction"


def test_stage4_straight_through_and_loss(nb):
    q = _rvq(nb)
    z = _z().requires_grad_()
    zq, loss, codes = q(z)
    assert zq.requires_grad, (
        "z_q carries no gradient back to z_e: return z_e + sg(q - z_e) (straight-through), not the detached code "
        "vectors")
    assert loss.requires_grad, (
        "the vq loss carries no gradient: don't detach the stages' losses (codebook and commitment terms must train)")
    g = torch.randn_like(zq)
    (dz,) = torch.autograd.grad((zq * g).sum(), z, retain_graph=True)
    assert torch.allclose(dz, g), (
        "the decoder's gradient must reach z_e unchanged (straight-through: z_e + sg(q - z_e)). if you got a "
        "multiple of it, you summed several straight-through outputs; if zero, you detached the output")
    zr = _z().requires_grad_()  # reference: residuals stay connected to z_e, code vectors are detached
    r = zr
    want = 0.0
    for vq in q.stages:
        zst, l, _ = vq(r)
        want = want + l
        r = r - zst.detach()
    assert torch.isclose(loss, want), (
        f"vq loss should be the sum of each stage's VectorQuantizer loss on ITS residual input ({want.item():.5f}), "
        f"got {loss.item():.5f}")
    (dl,) = torch.autograd.grad(loss, z, retain_graph=True)
    (dw,) = torch.autograd.grad(want, zr)
    assert torch.allclose(dl, dw), (
        "the vq loss's gradient w.r.t. z_e is wrong: every stage's commitment term must reach the encoder. stage "
        "s's input is r_{s-1} = z_e - (detached code vectors of the earlier stages): subtract the DETACHED code "
        "vectors (not the straight-through output, which also subtracts z_e's gradient), and don't detach r itself")
    loss.backward()
    assert all(v.codebook.weight.grad is not None and v.codebook.weight.grad.abs().sum() > 0 for v in q.stages), (
        "every stage's codebook must receive gradient from the vq loss (don't detach the stage losses)")
    q2 = _rvq(nb)
    _, l2, _ = q2(z.detach(), n_stages=1)
    assert torch.isclose(l2, q2.stages[0](z.detach())[1]), "with n_stages=1 the loss is the first stage's only"


def test_stage4_init_from_data(nb):
    g = _rng(12)
    centres = g.normal(size=(5, 4)) * 3
    pts = centres[g.integers(5, size=(4, 50))] + 0.3 * g.normal(size=(4, 50, 4))
    z = torch.tensor(pts.transpose(0, 2, 1))  # (B=4, D=4, T=50)
    q = nb.ResidualVQ(3, 8, 4).double()
    before = [((z - q(z, n)[0]) ** 2).mean().item() for n in (1, 2, 3)]
    out = q.init_from_data(z, n_iters=10, rng=3)
    assert out is q, "init_from_data returns self"
    books = nb.fit_rvq(z.movedim(1, -1).reshape(-1, 4).numpy(), 8, 3, n_iters=10, rng=3)
    assert all(np.allclose(v.codebook.weight.detach().numpy(), b) for v, b in zip(q.stages, books)), (
        "every stage's codebook becomes your fit_rvq codebook for the (B*T, D) vectors (same n_iters and rng)")
    after = [((z - q(z, n)[0]) ** 2).mean().item() for n in (1, 2, 3)]
    assert after[0] < before[0] / 5 and after[2] < after[1] < after[0], (
        f"k-means initialization should cut the error sharply, and each stage should lower it: before {before}, "
        f"after {after}")


# ---------------------------------------------------------------- stage 5: the codec


def _sig(B=2, L=256, seed=13):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L) / 8000.0
    f = 200 + 600 * torch.rand(B, 1, 1, generator=g)
    return 0.1 * torch.sin(2 * torch.pi * f * t) + 0.01 * torch.randn(B, 1, L, generator=g)


def test_stage5_codec_contract(nb):
    torch.manual_seed(0)
    c = nb.ConvCodec(channels=4, code_dim=6, strides=(2, 4), n_stages=3, n_codes=16)
    assert c.hop_length == 8, f"hop_length = prod(strides) = 8, got {c.hop_length}"
    assert isinstance(c.quantizer, nb.ResidualVQ) and len(c.quantizer.stages) == 3, "quantizer: a ResidualVQ(3, ...)"
    assert c.quantizer.stages[0].codebook.weight.shape == (16, 6), "n_codes=16 codes of code_dim=6"
    x = _sig()
    z = c.encoder(x)
    assert z.shape == (2, 6, 32), f"encoder: (B, 1, L) -> (B, code_dim, L / hop) = (2, 6, 32), got {tuple(z.shape)}"
    xh, loss, codes = c(x)
    assert xh.shape == x.shape and codes.shape == (2, 3, 32) and loss.dim() == 0, (
        f"forward -> (x_hat (B, 1, L), scalar vq loss, codes (B, n_stages, L / hop)); got {tuple(xh.shape)}, "
        f"{tuple(codes.shape)}")
    assert c(x, n_stages=1)[2].shape == (2, 1, 32), "forward(x, n_stages=1) uses one stage"
    c2 = nb.ConvCodec(channels=4, code_dim=6, strides=(2, 2, 2, 2), n_stages=2, n_codes=8)
    assert c2.hop_length == 16 and c2(x)[2].shape == (2, 2, 16), "strides (2, 2, 2, 2): hop 16, 16 frames"
    for bad in [torch.zeros(2, 1, 250), torch.zeros(2, 256)]:
        try:
            c(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"input of shape {tuple(bad.shape)} should raise ValueError ((B, 1, L), L % hop == 0)")


def test_stage5_encoder_gets_reconstruction_gradient(nb):
    torch.manual_seed(1)
    c = nb.ConvCodec(channels=4, code_dim=6, strides=(2, 4), n_stages=3, n_codes=16)
    x = _sig()
    xh, _, _ = c(x)
    ((x - xh) ** 2).mean().backward()  # the reconstruction loss alone: it reaches the encoder only via straight-through
    for name, p in c.encoder.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
            f"encoder.{name} got no gradient from the reconstruction loss. the encoder must learn through z_q's "
            "straight-through path: don't detach the encoder output or call the encoder under torch.no_grad() in forward")


def test_stage5_encode_decode_roundtrip(nb):
    torch.manual_seed(1)
    c = nb.ConvCodec(channels=4, code_dim=5, strides=(2, 2), n_stages=4, n_codes=8).eval()
    x = _sig(seed=14)
    with torch.no_grad():
        for n in (None, 2):
            codes = c.encode(x, n)
            assert codes.dtype == torch.int64 and codes.shape == (2, 4 if n is None else 2, 64), (
                f"encode -> int64 codes (B, n_used, L / hop), got {codes.dtype} {tuple(codes.shape)}")
            assert torch.allclose(c.decode(codes), c(x, n)[0], atol=1e-6), (
                "decode(encode(x, n)) must equal forward(x, n)'s reconstruction: the codes are all a receiver gets")


def _msl_ref(x, y, n_ffts, eps):
    total = 0.0
    for n in n_ffts:
        def mag(s):
            return np.abs(scipy.signal.stft(s, window="hann", nperseg=n, noverlap=n - n // 4, boundary="zeros",
                                            padded=False)[2] * scipy.signal.get_window("hann", n).sum())
        S, Sh = mag(x), mag(y)
        total += np.mean(np.abs(S - Sh)) + np.mean(np.abs(np.log(S + eps) - np.log(Sh + eps)))
    return total / len(n_ffts)


def test_stage5_multiscale_spectral_loss(nb):
    x = _sig(B=1, L=1024, seed=15).double()
    y = (x + 0.05 * torch.randn(1, 1, 1024, generator=torch.Generator().manual_seed(16), dtype=torch.float64))
    got = nb.multiscale_spectral_loss(x, y)
    want = _msl_ref(x[0, 0].numpy(), y[0, 0].numpy(), (64, 128, 256, 512), 1e-5)
    assert torch.is_tensor(got) and np.isclose(got.item(), want, rtol=1e-6), (
        f"expected {want:.6f}, got {float(got):.6f}. for each n_fft: magnitudes of a centered, zero-padded, "
        "periodic-Hann STFT with hop n_fft // 4 (torch.stft(..., pad_mode='constant')); mean |S - S_hat| plus mean "
        "|log(S + eps) - log(S_hat + eps)|; then the mean over scales")
    got2 = nb.multiscale_spectral_loss(x[:, 0], y[:, 0], n_ffts=(32, 128), eps=1e-3)
    assert np.isclose(got2.item(), _msl_ref(x[0, 0].numpy(), y[0, 0].numpy(), (32, 128), 1e-3), rtol=1e-6), (
        "(B, L) inputs work too, and n_ffts and eps are honoured")
    assert nb.multiscale_spectral_loss(x, x).item() == 0.0, "identical signals: 0"
    yg = y.clone().requires_grad_()
    nb.multiscale_spectral_loss(x, yg).backward()
    assert torch.isfinite(yg.grad).all() and yg.grad.abs().sum() > 0, "the loss must be differentiable in x_hat"


def _train_once(nb, seed, steps=500):
    torch.manual_seed(seed)
    c = nb.ConvCodec(channels=8, code_dim=8, strides=(2, 4), n_stages=2, n_codes=32)
    x = _sig(B=4, L=512, seed=17 + seed)
    opt = torch.optim.Adam(c.parameters(), lr=3e-3)

    def snr():
        with torch.no_grad():
            xh = c(x)[0]
        return np.mean([10 * torch.log10((x[i] ** 2).sum() / ((x[i] - xh[i]) ** 2).sum()).item() for i in range(4)])

    with torch.no_grad():
        c.quantizer.init_from_data(c.encoder(x), rng=0)
    snr0 = snr()
    for _ in range(steps):
        xh, vq, _ = c(x)
        loss = ((x - xh) ** 2).mean() / x.var() + vq
        opt.zero_grad()
        loss.backward()
        opt.step()
    return snr0, snr()


def test_stage5_codec_learns(nb):
    runs = []
    for seed in (0, 1, 2):
        runs.append(_train_once(nb, seed))
        if runs[-1][1] > 2.0:
            break
    assert max(r[1] for r in runs) > 2.0, (
        f"500 Adam steps on the reconstruction error of 4 noisy sinusoids should bring the reconstruction SNR above "
        f"2 dB in at least one of three runs (start, end in dB: {[(round(a, 1), round(b, 1)) for a, b in runs]}). a "
        "decoder that has learned only the average signal sits near 0 dB. first check that forward really decodes "
        "the quantizer's z_q (stage 4's checks cover its straight-through gradient), then that your layers can "
        "learn at all at this size: very narrow ReLU stacks can die at initialization")
