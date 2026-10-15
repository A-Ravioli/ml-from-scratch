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
    # linear state-space models (S4) · build

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
    raise NotImplementedError("stage1: implement discretize")


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
    raise NotImplementedError("stage2: implement ssm_recurrent")


@app.function
def ssm_kernel(Lambda_bar, B_bar, C, L):
    """stage 2: K[h, l] = 2 Re sum_n C[h, n] B_bar[h, n] Lambda_bar[h, n]^l for l = 0..L-1; shape (H, L), real."""
    raise NotImplementedError("stage2: implement ssm_kernel")


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
    raise NotImplementedError("stage3: implement causal_conv")


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
    raise NotImplementedError("stage4: implement hippo_legs")


@app.function
def s4d_init(N, init="lin"):
    """stage 4: N diagonal eigenvalues (complex64 tensor) for init in {'lin', 'inv', 'legs'}."""
    raise NotImplementedError("stage4: implement s4d_init")


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
        raise NotImplementedError("stage5: implement S4DLayer.__init__")

    def discrete(self):
        """stage 5: (Lambda_bar, B_bar, C) as complex (H, N) tensors: ZOH with B = 1 and dt = exp(log_dt)."""
        raise NotImplementedError("stage5: implement S4DLayer.discrete")

    def kernel(self, L):
        """stage 5: the (H, L) convolution kernel of the current parameters."""
        raise NotImplementedError("stage5: implement S4DLayer.kernel")

    def forward(self, u):
        """stage 5: u (batch, L, H) -> causal_conv(u, kernel(L)) + D * u."""
        raise NotImplementedError("stage5: implement S4DLayer.forward")

    def step(self, u_t, state=None):
        """stage 5: one recurrent step; u_t (batch, H), state (batch, H, N) complex or None; returns (y_t, state)."""
        raise NotImplementedError("stage5: implement S4DLayer.step")


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
