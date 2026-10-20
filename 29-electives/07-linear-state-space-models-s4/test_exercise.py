"""Checks for exercise.py (linear state-space models, S4).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Discretizations
are checked against matrix exponentials (`scipy.linalg.expm` of the augmented system) and the
bilinear formula; the recurrent and convolutional views against NumPy loops and against each
other; the FFT convolution against direct convolution, and its speed against a same-machine
FFT baseline; the HiPPO matrix by its defining property, reconstructing a signal's history
online; the layer against its own recurrent mode.
"""
import math
import time

import numpy as np
import torch
from numpy.polynomial import legendre
from scipy import linalg, stats

torch.set_num_threads(1)


def _rand_ssm(H, N, seed, scale=1.0):
    g = np.random.default_rng(seed)
    lam = -g.uniform(0.1, 1.0, (H, N)) + 1j * g.uniform(-3, 3, (H, N))
    dt = g.uniform(0.05, 0.5, (H, 1))
    Lb = np.exp(lam * dt)
    Bb = (Lb - 1) / lam * (g.normal(size=(H, N)) + 1j * g.normal(size=(H, N)))
    C = scale * (g.normal(size=(H, N)) + 1j * g.normal(size=(H, N)))
    t = lambda a: torch.tensor(a, dtype=torch.complex128)  # noqa: E731
    return t(Lb), t(Bb), t(C)


# ---------------------------------------------------------------- stage 1: discretization


def test_stage1_zoh_matches_matrix_exponential(nb):
    g = np.random.default_rng(0)
    lam = -g.uniform(0.1, 2, 4) + 1j * g.uniform(-5, 5, 4)
    B = g.normal(size=4) + 1j * g.normal(size=4)
    dt = 0.3
    Lb, Bb = nb.discretize(torch.tensor(lam), torch.tensor(B), dt, "zoh")
    M = np.zeros((5, 5), dtype=complex)  # augmented system [[A, B], [0, 0]]: expm gives exp(A dt) and int_0^dt exp(As) ds B
    M[:4, :4], M[:4, 4] = np.diag(lam), B
    E = linalg.expm(M * dt)
    assert np.allclose(Lb.numpy(), np.diag(E[:4, :4])), "ZOH: Lambda_bar = exp(Lambda dt), the exact state transition"
    assert np.allclose(Bb.numpy(), E[:4, 4]), (
        "ZOH: B_bar = (exp(Lambda dt) - 1) / Lambda * B, the exact effect of an input held constant over the step "
        "(the top-right block of expm([[A, B], [0, 0]] dt)). if you got dt * B, that's Euler, not ZOH")


def test_stage1_bilinear_formula(nb):
    g = np.random.default_rng(1)
    lam = -g.uniform(0.1, 2, 3) + 1j * g.uniform(-5, 5, 3)
    B = g.normal(size=3) + 0j
    dt = 0.7
    Lb, Bb = nb.discretize(torch.tensor(lam), torch.tensor(B), dt, "bilinear")
    A = np.diag(lam)
    inv = np.linalg.inv(np.eye(3) - A * dt / 2)
    assert np.allclose(Lb.numpy(), np.diag(inv @ (np.eye(3) + A * dt / 2))), (
        "bilinear: Lambda_bar = (1 + Lambda dt / 2) / (1 - Lambda dt / 2)")
    assert np.allclose(Bb.numpy(), inv @ (dt * B)), "bilinear: B_bar = dt / (1 - Lambda dt / 2) * B"


def test_stage1_methods_agree_for_small_steps_and_stay_stable(nb):
    lam = torch.tensor([-0.5 + 3j, -2.0 + 0.5j, -0.01 + 100j], dtype=torch.complex128)
    one = torch.ones(3, dtype=torch.complex128)
    z, zb = nb.discretize(lam, one, 1e-4, "zoh")
    b, bb = nb.discretize(lam, one, 1e-4, "bilinear")
    assert torch.allclose(z, b, atol=1e-7) and torch.allclose(zb, bb, rtol=1e-6), (
        "for tiny dt both discretizations approach Euler (1 + Lambda dt, dt B) and agree")
    for method in ("zoh", "bilinear"):
        big, _ = nb.discretize(lam, one, 10.0, method)
        assert torch.all(big.abs() < 1), f"{method}: a stable continuous mode (Re < 0) stays stable, |Lambda_bar| < 1"
    try:
        nb.discretize(lam, one, 0.1, "euler")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown method should raise ValueError")


def test_stage1_broadcasting_and_gradients(nb):
    lam = torch.complex(-torch.rand(4, 3) - 0.1, torch.randn(4, 3))
    log_dt = torch.zeros(4, requires_grad=True)
    Lb, Bb = nb.discretize(lam, torch.ones(4, 3, dtype=torch.complex64), torch.exp(log_dt)[:, None], "zoh")
    assert Lb.shape == (4, 3) and Bb.shape == (4, 3), "dt of shape (H, 1) broadcasts over the N modes of each channel"
    (Lb.real.sum() + Bb.imag.sum()).backward()
    assert log_dt.grad is not None and torch.isfinite(log_dt.grad).all(), "dt must stay differentiable (use torch ops)"
    Lb2, _ = nb.discretize(lam[1], torch.ones(3, dtype=torch.complex64), 1.0, "zoh")
    assert torch.allclose(Lb2, Lb[1]), "channel 1 has dt = exp(0) = 1"


# ---------------------------------------------------------------- stage 2: recurrence and kernel


def test_stage2_hand_example(nb):
    Lb = torch.tensor([[0.5 + 0j]])
    Bb = torch.tensor([[1.0 + 0j]])
    C = torch.tensor([[1.0 + 0j]])
    K = nb.ssm_kernel(Lb, Bb, C, 4)
    assert K.shape == (1, 4) and not torch.is_complex(K), f"the kernel is real with shape (H, L); got {K.shape}, {K.dtype}"
    assert torch.allclose(K, torch.tensor([[2.0, 1.0, 0.5, 0.25]])), (
        f"K_l = 2 Re(C B_bar Lambda_bar^l) = 2 * 0.5^l = (2, 1, 0.5, 0.25); got {K.tolist()}. the factor 2 is there "
        "because every complex mode stands for a conjugate pair")
    u = torch.tensor([[[1.0], [0.0], [2.0]]])
    y = nb.ssm_recurrent(Lb, Bb, C, u)
    assert y.shape == (1, 3, 1), f"output (batch, L, H); got {y.shape}"
    assert torch.allclose(y[0, :, 0], torch.tensor([2.0, 1.0, 4.5])), (
        f"x = (1, 0.5, 2.25), y = 2 Re x = (2, 1, 4.5); got {y[0, :, 0].tolist()}")


def test_stage2_recurrence_matches_numpy_loop(nb):
    Lb, Bb, C = _rand_ssm(3, 4, seed=2)
    u = torch.randn(2, 6, 3, dtype=torch.float64)
    y = nb.ssm_recurrent(Lb, Bb, C, u)
    Ln, Bn, Cn, un = Lb.numpy(), Bb.numpy(), C.numpy(), u.numpy()
    want = np.zeros((2, 6, 3))
    for b in range(2):
        x = np.zeros((3, 4), dtype=complex)
        for k in range(6):
            x = Ln * x + Bn * un[b, k][:, None]
            want[b, k] = 2 * (Cn * x).sum(-1).real
    assert torch.is_floating_point(y) and np.allclose(y.numpy(), want), (
        "x_k = Lambda_bar x_{k-1} + B_bar u_k (elementwise per channel and mode, x_{-1} = 0), y_k = 2 Re sum_n C x_k")


def test_stage2_kernel_matches_numpy_and_impulse_response(nb):
    Lb, Bb, C = _rand_ssm(3, 5, seed=3)
    K = nb.ssm_kernel(Lb, Bb, C, 9)
    l = np.arange(9)
    want = 2 * np.einsum("hn,hnl->hl", C.numpy() * Bb.numpy(), Lb.numpy()[..., None] ** l).real
    assert K.shape == (3, 9) and np.allclose(K.numpy(), want), "K[h, l] = 2 Re sum_n C B_bar Lambda_bar^l"
    impulse = torch.zeros(1, 9, 3, dtype=torch.float64)
    impulse[0, 0] = 1.0
    assert np.allclose(nb.ssm_recurrent(Lb, Bb, C, impulse)[0].numpy().T, K.numpy()), (
        "the kernel is the impulse response of the recurrence: feed u = (1, 0, 0, ...) and read y")


# ---------------------------------------------------------------- stage 3: FFT convolution


def test_stage3_matches_direct_convolution(nb):
    g = np.random.default_rng(4)
    u = g.normal(size=(2, 13, 3))
    K = g.normal(size=(3, 13))
    y = nb.causal_conv(torch.tensor(u), torch.tensor(K))
    want = np.zeros_like(u)
    for b in range(2):
        for h in range(3):
            want[b, :, h] = np.convolve(u[b, :, h], K[h])[:13]
    assert y.shape == u.shape and np.allclose(y.numpy(), want), (
        "y_k = sum_{j=0}^{k} K_j u_{k-j} per channel. if the start of the output is wrong, the FFT wrapped around: "
        "zero-pad both signals to length >= 2L - 1 before multiplying in frequency space, then keep the first L")
    u2 = u.copy()
    u2[:, 7:] += 5.0
    y2 = nb.causal_conv(torch.tensor(u2), torch.tensor(K))
    assert np.allclose(y2.numpy()[:, :7], y.numpy()[:, :7]), "causal: changing u from position 7 on can't change y before 7"


def test_stage3_convolutional_view_equals_recurrent_view(nb):
    Lb, Bb, C = _rand_ssm(4, 8, seed=5)
    u = torch.randn(3, 50, 4, dtype=torch.float64)
    y_rec = nb.ssm_recurrent(Lb, Bb, C, u)
    y_conv = nb.causal_conv(u, nb.ssm_kernel(Lb, Bb, C, 50))
    assert torch.allclose(y_rec, y_conv, atol=1e-9), (
        "the convolution with the SSM kernel must reproduce the recurrence exactly (up to rounding): "
        f"max difference {(y_rec - y_conv).abs().max().item():.2e}")


def test_stage3_is_fast(nb):
    u = torch.randn(1, 16384, 4)
    K = torch.randn(4, 16384)

    def ref():
        n = 2 * 16384
        return torch.fft.irfft(torch.fft.rfft(u, n=n, dim=1) * torch.fft.rfft(K.T, n=n, dim=0), n=n, dim=1)

    def best(f):
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t)
        return min(ts)

    nb.causal_conv(u, K)
    t_ref, t_you = best(ref), best(lambda: nb.causal_conv(u, K))
    assert t_you <= 20 * t_ref + 0.03, (
        f"causal_conv took {t_you * 1e3:.1f} ms on L = 16384, against {t_ref * 1e3:.1f} ms for one FFT round trip on "
        "this machine. use torch.fft (O(L log L)), not a direct O(L^2) convolution or a Python loop over positions")


def test_stage3_gradients(nb):
    u = torch.randn(2, 10, 3, requires_grad=True)
    K = torch.randn(3, 10, requires_grad=True)
    nb.causal_conv(u, K).sum().backward()
    assert u.grad is not None and K.grad is not None, "gradients must flow to both the input and the kernel"
    assert torch.allclose(K.grad, u.detach().sum(0).cumsum(0).flip(0).T, atol=1e-4), (
        "d(sum y)/dK_j = sum over batch and positions k >= j of u_{k-j}")


# ---------------------------------------------------------------- stage 4: HiPPO and S4D initialization


def test_stage4_hippo_hand_example(nb):
    A, B = nb.hippo_legs(3)
    r3, r5, r15 = math.sqrt(3), math.sqrt(5), math.sqrt(15)
    want = np.array([[-1, 0, 0], [-r3, -2, 0], [-r5, -r15, -3]])
    assert np.asarray(A).shape == (3, 3) and np.allclose(A, want), (
        f"A_nk = -sqrt(2n+1) sqrt(2k+1) below the diagonal, -(n+1) on it, 0 above; expected {np.round(want, 3).tolist()}, "
        f"got {np.round(np.asarray(A), 3).tolist()}")
    assert np.allclose(B, [1, r3, r5]), "B_n = sqrt(2n+1)"


def test_stage4_hippo_remembers_the_history(nb):
    N, L = 16, 1000
    A, B = (np.asarray(a, dtype=float) for a in nb.hippo_legs(N))
    t = np.arange(L) / L
    f = np.sin(6 * t) + 0.5 * np.cos(17 * t ** 2)
    c, I = np.zeros(N), np.eye(N)
    for k, fk in enumerate(f, start=1):  # HiPPO-LegS online update, bilinear discretization with step 1/k
        c = np.linalg.solve(I - A / (2 * k), (I + A / (2 * k)) @ c + B * fk / k)
    x = 2 * (np.arange(L) + 0.5) / L - 1
    recon = legendre.legval(x, c * np.sqrt(2 * np.arange(N) + 1))
    err = np.sqrt(np.mean((recon - f) ** 2))
    assert err < 0.02, (
        f"running HiPPO-LegS online over 1,000 samples and reading the 16 coefficients as a Legendre expansion "
        f"should reconstruct the whole history (RMS error ~0.006); got {err:.3f}. check the signs and the sqrt factors "
        "of A and B")


def test_stage4_s4d_inits(nb):
    n = np.arange(8)
    lin = nb.s4d_init(8, "lin")
    assert lin.shape == (8,) and lin.dtype == torch.complex64, f"(N,) complex64; got {lin.shape}, {lin.dtype}"
    assert np.allclose(lin.numpy(), -0.5 + 1j * np.pi * n), "S4D-Lin: -1/2 + i pi n"
    assert np.allclose(nb.s4d_init(8).numpy(), lin.numpy()), "the default init is 'lin'"
    inv = nb.s4d_init(8, "inv").numpy()
    assert np.allclose(inv, -0.5 + 1j * 8 / np.pi * (8 / (2 * n + 1) - 1), rtol=1e-5), (
        "S4D-Inv: -1/2 + i (N / pi) (N / (2n + 1) - 1)")
    legs = nb.s4d_init(8, "legs").numpy()
    s = np.sqrt(2 * np.arange(16) + 1.0)
    A16 = -np.tril(np.outer(s, s), -1) - np.diag(np.arange(16) + 1.0)
    P = np.sqrt(np.arange(16) + 0.5)
    ev = np.linalg.eigvals(A16 + np.outer(P, P))
    want = np.sort(ev.imag[ev.imag > 0])
    assert np.allclose(legs.real, -0.5, atol=1e-4) and np.allclose(legs.imag, want, rtol=1e-4), (
        "S4D-LegS: the eigenvalues of the normal part A + P P' of HiPPO-LegS of size 2N (P_n = sqrt(n + 1/2)), "
        "keeping the N with positive imaginary part, sorted by imaginary part. their real parts are all -1/2")
    try:
        nb.s4d_init(4, "random")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown init should raise ValueError")


# ---------------------------------------------------------------- stage 5: the layer


def test_stage5_parameters(nb):
    torch.manual_seed(0)
    layer = nb.S4DLayer(6, 4, dt_min=0.01, dt_max=0.2, init="inv")
    shapes = {k: tuple(v.shape) for k, v in layer.named_parameters()}
    assert shapes == {"log_dt": (6,), "log_neg_re": (6, 4), "im": (6, 4), "C": (6, 4, 2), "D": (6,)}, (
        f"parameters log_dt (H,), log_neg_re (H, N), im (H, N), C (H, N, 2), D (H,); got {shapes}")
    dt = layer.log_dt.exp()
    assert dt.min() >= 0.01 - 1e-6 and dt.max() <= 0.2 + 1e-6, "dt = exp(log_dt) starts inside [dt_min, dt_max]"
    wide_layer = nb.S4DLayer(4000, 4, dt_min=1e-3, dt_max=1e-1)
    Cw, Dw = wide_layer.C.detach().numpy(), wide_layer.D.detach().numpy()
    assert abs(Cw.mean()) < 0.03 and abs(Cw.var() - 0.5) < 0.03, (
        f"C's real and imaginary parts start N(0, 1/2) each (unit variance per complex entry); got mean {Cw.mean():.3f}, "
        f"variance {Cw.var():.3f}")
    assert abs(Dw.mean()) < 0.06 and abs(Dw.var() - 1.0) < 0.08, (
        f"D starts N(0, 1); got mean {Dw.mean():.3f}, variance {Dw.var():.3f}")
    wide = wide_layer.log_dt.detach().numpy()
    z = (wide - math.log(1e-3)) / (math.log(1e-1) - math.log(1e-3))  # should be Uniform(0, 1)
    ks = stats.kstest(z, "uniform").statistic
    assert ks < 0.04, (
        f"log_dt starts UNIFORM in [log dt_min, log dt_max] (log-uniform dt: as many channels with dt in [0.001, 0.01] "
        f"as in [0.01, 0.1]), independently per channel; the Kolmogorov-Smirnov distance from uniform is {ks:.3f}")
    assert torch.allclose(-layer.log_neg_re.exp(), torch.full((6, 4), -0.5)), "Re Lambda = -exp(log_neg_re) = -1/2 at init"
    assert torch.allclose(layer.im[3], nb.s4d_init(4, "inv").imag), "im starts from s4d_init(N, init) in every channel"
    with torch.no_grad():
        layer.log_neg_re.fill_(3.0)
    Lb, Bb, C = layer.discrete()
    assert torch.all(Lb.abs() < 1), "whatever log_neg_re is, Re Lambda = -exp(.) < 0, so every mode stays stable"
    lam = torch.complex(-layer.log_neg_re.exp(), layer.im)
    dtc = layer.log_dt.exp()[:, None]
    assert torch.allclose(Lb, torch.exp(lam * dtc), atol=1e-6) and torch.allclose(Bb, (torch.exp(lam * dtc) - 1) / lam, atol=1e-6), (
        "discrete() is ZOH with B = 1 and dt = exp(log_dt)")
    assert torch.allclose(C, torch.complex(layer.C[..., 0], layer.C[..., 1])), "C[..., 0] is the real part, C[..., 1] the imaginary"


def test_stage5_convolutional_mode_equals_recurrent_mode(nb):
    torch.manual_seed(1)
    layer = nb.S4DLayer(5, 8, dt_min=0.01, dt_max=0.5)
    u = torch.randn(2, 40, 5)
    y = layer(u)
    assert y.shape == u.shape and y.dtype == torch.float32, f"forward maps (batch, L, H) to (batch, L, H) float; got {y.shape}"
    state, ys = None, []
    for k in range(40):
        yk, state = layer.step(u[:, k], state)
        ys.append(yk)
    ys = torch.stack(ys, 1)
    assert torch.allclose(y, ys, atol=1e-4), (
        f"forward (convolution) and step (recurrence) must agree: max difference {(y - ys).abs().max().item():.2e}. "
        "both include the skip D * u")
    with torch.no_grad():
        layer.C.zero_()
    assert torch.allclose(layer(u), layer.D * u), "with C = 0 the output is just the skip connection D * u"


def test_stage5_gradients_reach_every_parameter(nb):
    torch.manual_seed(2)
    layer = nb.S4DLayer(3, 4)
    layer(torch.randn(2, 16, 3)).pow(2).sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
            f"no gradient reached {name}: build the kernel from the parameters with differentiable torch ops")


def test_stage5_learns_an_exponential_filter(nb):
    torch.manual_seed(3)
    layer = nb.S4DLayer(1, 4, dt_min=0.01, dt_max=0.1)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    g = torch.Generator().manual_seed(3)
    target_k = 0.1 * 0.9 ** torch.arange(64.0)
    losses = []
    for _ in range(80):
        u = torch.randn(8, 64, 1, generator=g)
        y = torch.nn.functional.conv1d(torch.nn.functional.pad(u[:, None, :, 0], (63, 0)), target_k.flip(0)[None, None])[:, 0]
        loss = ((layer(u)[..., 0] - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.1 * np.mean(losses[:10]), (
        f"trained for 80 Adam steps to reproduce an exponential moving average, the loss should fall at least 10x "
        f"(from {np.mean(losses[:10]):.4f} to {np.mean(losses[-10:]):.4f})")
