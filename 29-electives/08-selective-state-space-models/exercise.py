import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.torchnn import RMSNorm  # yours, from chapter 15


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # selective state-space models · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `sequential_scan`: the linear recurrence h_t = a_t h_{t-1} + b_t, one step at a time |
    | 2 | `combine`, `parallel_scan`: the same recurrence in O(log L) vectorized sweeps |
    | 3 | `selective_scan`: Mamba's S6 recurrence with input-dependent step, B and C |
    | 4 | `SelectiveSSM`: the trainable selective layer, with a recurrent mode |
    | 5 | `MambaBlock`: projection, causal convolution, gating, residual |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def sequential_scan(a, b, h0=None):
    """stage 1: h_t = a_t * h_{t-1} + b_t along dim 1 (h_{-1} = h0, or 0); a, b (batch, L, ...); returns h like b."""
    raise NotImplementedError("stage1: implement sequential_scan")


@app.cell(hide_code=True)
def _():
    checks.run(1, sequential_scan=sequential_scan)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def combine(first, second):
    """stage 2: compose the affine maps h -> a1 h + b1 (first) then h -> a2 h + b2 (second); returns (a, b)."""
    raise NotImplementedError("stage2: implement combine")


@app.function
def parallel_scan(a, b, h0=None):
    """stage 2: the same result as sequential_scan, in O(log L) vectorized sweeps over the sequence."""
    raise NotImplementedError("stage2: implement parallel_scan")


@app.cell(hide_code=True)
def _():
    checks.run(2, combine=combine, parallel_scan=parallel_scan, sequential_scan=sequential_scan)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def selective_scan(u, delta, A, B, C, D=None, *, mode="parallel"):
    """stage 3: S6. u, delta (batch, L, Dm); A (Dm, N); B, C (batch, L, N); returns y (batch, L, Dm)."""
    raise NotImplementedError("stage3: implement selective_scan")


@app.cell(hide_code=True)
def _():
    checks.run(3, selective_scan=selective_scan)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class SelectiveSSM(nn.Module):
    """stage 4: the S6 layer: Delta, B, C computed from the input, A = -exp(A_log), skip D."""

    def __init__(self, d_inner, d_state=16, *, dt_rank=None, dt_min=1e-3, dt_max=1e-1):
        """stage 4: A_log (Dm, N), D (Dm,), x_proj Linear(Dm, R + 2N, no bias), dt_proj Linear(R, Dm), per the spec."""
        raise NotImplementedError("stage4: implement SelectiveSSM.__init__")

    def params(self, x):
        """stage 4: (delta (batch, L, Dm), B (batch, L, N), C (batch, L, N)) for input x (batch, L, Dm)."""
        raise NotImplementedError("stage4: implement SelectiveSSM.params")

    def forward(self, x):
        """stage 4: x (batch, L, Dm) -> selective_scan(x, delta, -exp(A_log), B, C, D)."""
        raise NotImplementedError("stage4: implement SelectiveSSM.forward")

    def step(self, x_t, h=None):
        """stage 4: one recurrent step; x_t (batch, Dm), h (batch, Dm, N) or None; returns (y_t, h)."""
        raise NotImplementedError("stage4: implement SelectiveSSM.step")


@app.cell(hide_code=True)
def _():
    checks.run(4, SelectiveSSM=SelectiveSSM, selective_scan=selective_scan)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class MambaBlock(nn.Module):
    """stage 5: x + out_proj(SSM(silu(conv(x_in))) * silu(z)), (x_in, z) = in_proj(RMSNorm(x))."""

    def __init__(self, d_model, *, d_state=16, expand=2, d_conv=4, dt_rank=None):
        """stage 5: norm, in_proj, conv1d (depthwise, causal), ssm, out_proj, per the spec."""
        raise NotImplementedError("stage5: implement MambaBlock.__init__")

    def forward(self, x):
        """stage 5: x (batch, L, d_model) -> same shape."""
        raise NotImplementedError("stage5: implement MambaBlock.forward")


@app.cell(hide_code=True)
def _():
    checks.run(5, MambaBlock=MambaBlock, SelectiveSSM=SelectiveSSM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it times your two scans,
    pits your selective model against your topic 07 S4D layer on selective copying, and ends
    with the block's project: your Mamba against a transformer on a long-sequence task,
    including lengths neither saw in training.
    """)
    return


if __name__ == "__main__":
    app.run()
