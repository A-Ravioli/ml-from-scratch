import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # linear state-space models (S4) · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `discretize`: zero-order hold and bilinear discretization of a diagonal SSM |
    | 2 | `ssm_recurrent`, `ssm_kernel`: the recurrent view and the convolution kernel |
    | 3 | `causal_conv`: causal convolution by FFT |
    | 4 | `hippo_legs`, `s4d_init`: the HiPPO matrix and the S4D initializations |
    | 5 | `S4DLayer`: a trainable diagonal SSM layer with a convolutional and a recurrent mode |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def discretize(Lambda, B, dt, method="zoh"):
    """stage 1: (Lambda_bar, B_bar) for x' = Lambda x + B u with step dt; method 'zoh' or 'bilinear'."""
    Lambda = torch.as_tensor(Lambda)
    B, dt = torch.as_tensor(B), torch.as_tensor(dt)
    if method == "zoh":
        Lb = torch.exp(Lambda * dt)
        return Lb, (Lb - 1) / Lambda * B
    if method == "bilinear":
        den = 1 - Lambda * dt / 2
        return (1 + Lambda * dt / 2) / den, dt / den * B
    raise ValueError(f"unknown method {method!r}: use 'zoh' or 'bilinear'")


@app.cell(hide_code=True)
def _():
    checks.run(1, discretize=discretize)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def ssm_recurrent(Lambda_bar, B_bar, C, u):
    """stage 2: run x_k = Lambda_bar x_{k-1} + B_bar u_k, y_k = 2 Re(C . x_k) over u (batch, L, H); returns (batch, L, H)."""
    batch, L, H = u.shape
    x = torch.zeros(batch, *Lambda_bar.shape, dtype=Lambda_bar.dtype, device=u.device)
    ys = []
    for k in range(L):
        x = Lambda_bar * x + B_bar * u[:, k, :, None]
        ys.append(2 * (C * x).sum(-1).real)
    return torch.stack(ys, dim=1).to(u.dtype)


@app.function
def ssm_kernel(Lambda_bar, B_bar, C, L):
    """stage 2: K[h, l] = 2 Re sum_n C[h, n] B_bar[h, n] Lambda_bar[h, n]^l for l = 0..L-1; shape (H, L), real."""
    powers = Lambda_bar[..., None] ** torch.arange(L, device=Lambda_bar.device)
    return 2 * torch.einsum("hn,hnl->hl", C * B_bar, powers).real


@app.cell(hide_code=True)
def _():
    checks.run(2, ssm_recurrent=ssm_recurrent, ssm_kernel=ssm_kernel, discretize=discretize)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def causal_conv(u, K):
    """stage 3: y[:, k, h] = sum_{j<=k} K[h, j] u[:, k-j, h] for u (batch, L, H), K (H, L), by FFT with zero padding."""
    L = u.shape[1]
    n = 2 * L
    Uf = torch.fft.rfft(u, n=n, dim=1)
    Kf = torch.fft.rfft(K.T, n=n, dim=0)
    return torch.fft.irfft(Uf * Kf, n=n, dim=1)[:, :L]


@app.cell(hide_code=True)
def _():
    checks.run(3, causal_conv=causal_conv, ssm_recurrent=ssm_recurrent, ssm_kernel=ssm_kernel, discretize=discretize)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def hippo_legs(N):
    """stage 4: the HiPPO-LegS matrices (A (N, N), B (N,)) as float64 NumPy arrays, with A's negative sign."""
    s = np.sqrt(2 * np.arange(N) + 1.0)
    A = -np.tril(np.outer(s, s), -1) - np.diag(np.arange(N) + 1.0)
    return A, s.copy()


@app.function
def s4d_init(N, init="lin"):
    """stage 4: N diagonal eigenvalues (complex64 tensor) for init in {'lin', 'inv', 'legs'}."""
    n = np.arange(N)
    if init == "lin":
        im = np.pi * n
    elif init == "inv":
        im = N / np.pi * (N / (2 * n + 1) - 1)
    elif init == "legs":
        A, _ = hippo_legs(2 * N)
        P = np.sqrt(np.arange(2 * N) + 0.5)
        ev = np.linalg.eigvals(A + np.outer(P, P))
        im = np.sort(ev.imag[ev.imag > 0])
    else:
        raise ValueError(f"unknown init {init!r}")
    return torch.tensor(-0.5 + 1j * im, dtype=torch.complex64)


@app.cell(hide_code=True)
def _():
    checks.run(4, hippo_legs=hippo_legs, s4d_init=s4d_init)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class S4DLayer(nn.Module):
    """stage 5: a diagonal SSM layer (S4D): H independent channels, N complex modes each, plus a skip D."""

    def __init__(self, d_model, d_state=64, *, dt_min=1e-3, dt_max=1e-1, init="lin"):
        """stage 5: parameters log_dt (H,), log_neg_re (H, N), im (H, N), C (H, N, 2), D (H,), initialized per the spec."""
        super().__init__()
        H, N = d_model, d_state
        lam = s4d_init(N, init)
        self.log_dt = nn.Parameter(math.log(dt_min) + torch.rand(H) * (math.log(dt_max) - math.log(dt_min)))
        self.log_neg_re = nn.Parameter(torch.log(-lam.real).repeat(H, 1))
        self.im = nn.Parameter(lam.imag.repeat(H, 1))
        self.C = nn.Parameter(torch.randn(H, N, 2) * math.sqrt(0.5))
        self.D = nn.Parameter(torch.randn(H))

    def discrete(self):
        """stage 5: (Lambda_bar, B_bar, C) as complex (H, N) tensors: ZOH with B = 1 and dt = exp(log_dt)."""
        lam = torch.complex(-torch.exp(self.log_neg_re), self.im)
        Lb, Bb = discretize(lam, torch.ones_like(lam), torch.exp(self.log_dt)[:, None], "zoh")
        return Lb, Bb, torch.view_as_complex(self.C.contiguous())

    def kernel(self, L):
        """stage 5: the (H, L) convolution kernel of the current parameters."""
        return ssm_kernel(*self.discrete(), L)

    def forward(self, u):
        """stage 5: u (batch, L, H) -> causal_conv(u, kernel(L)) + D * u."""
        return causal_conv(u, self.kernel(u.shape[1])) + self.D * u

    def step(self, u_t, state=None):
        """stage 5: one recurrent step; u_t (batch, H), state (batch, H, N) complex or None; returns (y_t, state)."""
        Lb, Bb, C = self.discrete()
        if state is None:
            state = torch.zeros(u_t.shape[0], *Lb.shape, dtype=Lb.dtype, device=u_t.device)
        state = Lb * state + Bb * u_t[..., None]
        return 2 * (C * state).sum(-1).real + self.D * u_t, state


@app.cell(hide_code=True)
def _():
    checks.run(5, S4DLayer=S4DLayer, causal_conv=causal_conv, ssm_kernel=ssm_kernel, discretize=discretize,
               s4d_init=s4d_init)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it watches your HiPPO
    matrix remember a signal online, looks at the kernels your initializations produce, times
    the convolutional view against the recurrent one, and trains your S4D layer against an
    LSTM on a task that needs memory across hundreds of steps.
    """)
    return


if __name__ == "__main__":
    app.run()
