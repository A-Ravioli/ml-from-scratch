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
    # selective state-space models · build (reference solution)

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
    h = torch.zeros_like(b[:, 0]) if h0 is None else h0
    out = []
    for t in range(b.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


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
    a1, b1 = first
    a2, b2 = second
    return a1 * a2, a2 * b1 + b2


@app.function
def parallel_scan(a, b, h0=None):
    """stage 2: the same result as sequential_scan, in O(log L) vectorized sweeps over the sequence."""
    if h0 is not None:
        b = torch.cat([(a[:, 0] * h0 + b[:, 0]).unsqueeze(1), b[:, 1:]], dim=1)
    L = b.shape[1]
    step = 1
    while step < L:  # Hillis-Steele: after the sweep with offset `step`, position t holds the composition of t-2*step+1..t
        a_prev = torch.cat([torch.ones_like(a[:, :step]), a[:, :-step]], dim=1)
        b_prev = torch.cat([torch.zeros_like(b[:, :step]), b[:, :-step]], dim=1)
        a, b = combine((a_prev, b_prev), (a, b))
        step *= 2
    return b


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
    if mode not in ("parallel", "sequential"):
        raise ValueError(f"mode must be 'parallel' or 'sequential', got {mode!r}")
    a = torch.exp(delta.unsqueeze(-1) * A)  # (batch, L, Dm, N)
    bu = (delta * u).unsqueeze(-1) * B.unsqueeze(2)  # (batch, L, Dm, N): Euler for B, as in Mamba
    h = (parallel_scan if mode == "parallel" else sequential_scan)(a, bu)
    y = torch.einsum("bldn,bln->bld", h, C)
    return y if D is None else y + D * u


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
        super().__init__()
        self.d_state = d_state
        self.dt_rank = math.ceil(d_inner / 16) if dt_rank is None else dt_rank
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        with torch.no_grad():
            nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
            dt = torch.exp(math.log(dt_min) + torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def params(self, x):
        """stage 4: (delta (batch, L, Dm), B (batch, L, N), C (batch, L, N)) for input x (batch, L, Dm)."""
        dt_raw, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        return F.softplus(self.dt_proj(dt_raw)), B, C

    def forward(self, x):
        """stage 4: x (batch, L, Dm) -> selective_scan(x, delta, -exp(A_log), B, C, D)."""
        delta, B, C = self.params(x)
        return selective_scan(x, delta, -torch.exp(self.A_log), B, C, self.D)

    def step(self, x_t, h=None):
        """stage 4: one recurrent step; x_t (batch, Dm), h (batch, Dm, N) or None; returns (y_t, h)."""
        delta, B, C = (p[:, 0] for p in self.params(x_t[:, None]))
        if h is None:
            h = x_t.new_zeros(x_t.shape[0], x_t.shape[1], self.d_state)
        h = torch.exp(delta[..., None] * -torch.exp(self.A_log)) * h + (delta * x_t)[..., None] * B[:, None]
        return torch.einsum("bdn,bn->bd", h, C) + self.D * x_t, h


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
        super().__init__()
        d_inner = expand * d_model
        self.d_conv = d_conv
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.ssm = SelectiveSSM(d_inner, d_state, dt_rank=dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """stage 5: x (batch, L, d_model) -> same shape."""
        xi, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        L, k = x.shape[1], self.d_conv
        # causal depthwise convolution with conv1d's weight and bias: out_t = bias + sum_j w_j x_{t-k+1+j}.
        # (the same numbers as self.conv1d(...)[..., :L], written as k shifted products because PyTorch's
        # depthwise conv backward is slow on CPU)
        xp = F.pad(xi, (0, 0, k - 1, 0))
        w = self.conv1d.weight[:, 0]  # (channels, k)
        xi = self.conv1d.bias + sum(xp[:, j:j + L] * w[:, j] for j in range(k))
        return x + self.out_proj(self.ssm(F.silu(xi)) * F.silu(z))


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
