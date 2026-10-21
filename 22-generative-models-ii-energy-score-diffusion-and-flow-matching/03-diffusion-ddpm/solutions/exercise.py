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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # diffusion (DDPM) · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. a diffusion model destroys data
    with Gaussian noise in $T$ small steps, and learns to undo one step at a time. you'll
    build the noise schedule and its closed-form shortcut, the time-conditioned networks
    (an MLP for 2-D toys and a U-Net for images), the noise-prediction loss, and ancestral
    sampling.

    | stage | you build |
    |---|---|
    | 1 | `DiffusionSchedule`: linear and cosine schedules, `q_sample` → `mlfs.diffusion` |
    | 2 | `sinusoidal_features`, `TimeEmbedding`, `TimeMLP`: networks that know what time it is → `mlfs.diffusion` |
    | 3 | `ddpm_loss`, `fit_ddpm`: predict the noise → `mlfs.diffusion` |
    | 4 | `ddpm_sample`: ancestral sampling, $T$ network calls → `mlfs.diffusion` |
    | 5 | `TimeUNet`: a residual U-Net with time and class embeddings → `mlfs.diffusion` |

    the time convention used in the whole chapter: networks are called `model(x, t, y)` with
    `t` a float tensor `(B,)` in $[0, 1]$; for diffusion `t = step / n_steps`.

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class DiffusionSchedule:
    """the forward process q(x_t | x_0) = N(sqrt(ab_t) x_0, (1 - ab_t) I) for steps t = 0..n_steps-1."""

    def __init__(self, n_steps=1000, *, beta_start=0.0001, beta_end=0.02, schedule="linear"):
        """stage 1: float32 tensors betas, alphas = 1 - betas, alpha_bars = cumprod(alphas), all (n_steps,).
        'linear': betas = linspace(beta_start, beta_end, n_steps); 'cosine' (Nichol & Dhariwal, 2021): f(u) =
        cos^2(((u / n_steps) + 0.008) / 1.008 * pi / 2), alpha_bar_t = f(t + 1) / f(0), betas clipped at 0.999.
        ValueError for anything else."""
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)
        elif schedule == "cosine":
            u = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
            f = torch.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
            ab = f / f[0]
            betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        else:
            raise ValueError(f"schedule must be 'linear' or 'cosine', got {schedule!r}")
        self.n_steps = n_steps
        self.schedule = schedule
        self.betas = betas.float()
        self.alphas = (1 - betas).float()
        self.alpha_bars = torch.cumprod(1 - betas, 0).float()

    def q_sample(self, x0, t, noise=None):
        """stage 1: x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) noise for integer steps t (B,); noise ~ N(0, I) if None."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars.to(x0.device)[t.to(x0.device)].view((-1,) + (1,) * (x0.dim() - 1))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def to(self, device):
        """stage 1: move betas, alphas and alpha_bars to `device`; returns self."""
        self.betas, self.alphas, self.alpha_bars = (v.to(device) for v in (self.betas, self.alphas, self.alpha_bars))
        return self


@app.cell(hide_code=True)
def _():
    checks.run(1, DiffusionSchedule=DiffusionSchedule)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sinusoidal_features(t, dim, *, max_period=10000.0):
    """stage 2: t (B,) in [0, 1] -> (B, dim): [cos(1000 t f_k), sin(1000 t f_k)] for k < dim // 2 with
    f_k = max_period^(-k / (dim // 2)); a zero column is appended when dim is odd."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = 1000.0 * t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


@app.class_definition
class TimeEmbedding(nn.Module):
    """sinusoidal features of 1000 t followed by Linear -> SiLU -> Linear, (B,) -> (B, dim)."""

    def __init__(self, dim, *, max_period=10000.0):
        """stage 2: store dim and max_period; an MLP dim -> dim -> dim with a SiLU in between."""
        super().__init__()
        self.dim, self.max_period = dim, max_period
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        """stage 2: t (B,) -> (B, dim)."""
        return self.mlp(sinusoidal_features(t, self.dim, max_period=self.max_period))


@app.class_definition
class TimeMLP(nn.Module):
    """a 2-D toy denoiser / velocity network model(x (B, data_dim), t (B,), y=None) -> (B, data_dim)."""

    def __init__(self, data_dim=2, *, hidden_dim=256, n_layers=3, time_dim=64, n_classes=None):
        """stage 2: a TimeEmbedding(time_dim), a class table nn.Embedding(n_classes + 1, time_dim) if n_classes (last
        row = null label), and an MLP with n_layers hidden layers conditioned on their sum."""
        super().__init__()
        self.data_dim, self.n_classes = data_dim, n_classes
        self.time_emb = TimeEmbedding(time_dim)
        self.label_emb = nn.Embedding(n_classes + 1, time_dim) if n_classes is not None else None
        self.inp = nn.Linear(data_dim, hidden_dim)
        self.cond = nn.ModuleList(nn.Linear(time_dim, hidden_dim) for _ in range(n_layers))
        self.lins = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers))
        self.out = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t, y=None):
        """stage 2: y None -> the null label (n_classes) for a conditional model; ValueError if y is given to an
        unconditional one."""
        c = self.time_emb(t)
        if self.label_emb is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.n_classes, dtype=torch.long, device=x.device)
            c = c + self.label_emb(y)
        elif y is not None:
            raise ValueError("this model has no class embedding (n_classes=None) but was given labels")
        h = self.inp(x)
        for lin, cp in zip(self.lins, self.cond):
            h = h + lin(F.silu(h + cp(c)))
        return self.out(F.silu(h))


@app.cell(hide_code=True)
def _():
    checks.run(2, sinusoidal_features=sinusoidal_features, TimeEmbedding=TimeEmbedding, TimeMLP=TimeMLP)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def ddpm_loss(model, schedule, x0, *, y=None, generator=None):
    """stage 3: t = torch.randint(n_steps, (B,), generator), then noise = torch.randn(x0.shape, generator) (both on the
    CPU, moved to x0's device); MSE(model(q_sample(x0, t, noise), t.float() / n_steps, y), noise), a mean."""
    B = x0.shape[0]
    t = torch.randint(schedule.n_steps, (B,), generator=generator).to(x0.device)
    noise = torch.randn(x0.shape, generator=generator).to(x0.device)
    x_t = schedule.q_sample(x0, t, noise)
    return F.mse_loss(model(x_t, t.float() / schedule.n_steps, y), noise)


@app.function
def fit_ddpm(model, schedule, X, *, n_iters=2000, batch_size=256, lr=1e-3, seed=0):
    """stage 3: Adam(lr) on ddpm_loss over n_iters minibatches X[randint(N, (batch_size,), g)] with
    g = torch.Generator().manual_seed(seed) (also passed to ddpm_loss); model in train mode; returns the losses."""
    X = torch.as_tensor(X, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(n_iters):
        xb = X[torch.randint(len(X), (batch_size,), generator=g)]
        loss = ddpm_loss(model, schedule, xb, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@app.cell(hide_code=True)
def _():
    checks.run(3, ddpm_loss=ddpm_loss, fit_ddpm=fit_ddpm, DiffusionSchedule=DiffusionSchedule, TimeMLP=TimeMLP)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ddpm_sample(model, schedule, shape, *, y=None, generator=None, device="cpu"):
    """stage 4: x_T = torch.randn(shape, generator); for i = n_steps-1 .. 0: eps = model(x, full(i / n_steps), y),
    mean = (x - beta_i / sqrt(1 - ab_i) eps) / sqrt(alpha_i), x = mean + sqrt(beta_i) z for i > 0 (z = torch.randn(shape,
    generator), drawn on the CPU) and x = mean at i = 0. no gradients; an nn.Module is put in eval mode and its mode
    restored afterwards; returns x."""
    was = getattr(model, "training", None)
    if was is not None:
        model.eval()
    betas, alphas, abs_ = (v.detach().cpu().double() for v in (schedule.betas, schedule.alphas, schedule.alpha_bars))
    n = schedule.n_steps
    try:
        with torch.no_grad():
            x = torch.randn(shape, generator=generator).to(device)
            for i in reversed(range(n)):
                t = torch.full((shape[0],), i / n, device=device)
                eps = model(x, t, y)
                b, a, ab = float(betas[i]), float(alphas[i]), float(abs_[i])
                x = (x - b / math.sqrt(1 - ab) * eps) / math.sqrt(a)
                if i > 0:
                    x = x + math.sqrt(b) * torch.randn(shape, generator=generator).to(device)
    finally:
        if was is not None:
            model.train(was)
    return x


@app.cell(hide_code=True)
def _():
    checks.run(4, ddpm_sample=ddpm_sample, DiffusionSchedule=DiffusionSchedule)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class ResBlock(nn.Module):
    """given (use it or design your own): GroupNorm -> SiLU -> conv3x3, + projected embedding, GroupNorm -> SiLU ->
    dropout -> conv3x3, + a (1x1 when widths differ) skip."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(math.gcd(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(math.gcd(8, out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


@app.class_definition
class SelfAttention2d(nn.Module):
    """given (use it or design your own): single-head self-attention over the H*W positions, plus a residual."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(8, ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).flatten(2).transpose(1, 2).chunk(3, dim=2)  # (B, HW, C) each
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(h.transpose(1, 2).reshape(B, C, H, W))


@app.class_definition
class TimeUNet(nn.Module):
    """model(x (B, C, H, W), t (B,), y=None) -> same shape as x; residual U-Net with time (+ class) embeddings."""

    def __init__(self, in_channels=3, *, base_channels=64, channel_mults=(1, 2, 2), n_classes=None, dropout=0.0):
        """stage 5: per level, two ResBlocks at width base * mult (a stride-2 conv between levels); a middle with
        global context (self-attention here); a mirrored up path with skip concatenations; a time embedding (+ an
        (n_classes + 1)-row class table, last row = null) added inside every ResBlock; dropout in the ResBlocks."""
        super().__init__()
        self.in_channels, self.n_classes = in_channels, n_classes
        self.n_down = len(channel_mults) - 1
        emb = 4 * base_channels
        self.time_emb = TimeEmbedding(emb)
        self.label_emb = nn.Embedding(n_classes + 1, emb) if n_classes is not None else None
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        widths = [base_channels * m for m in channel_mults]
        self.down = nn.ModuleList()
        skips, ch = [base_channels], base_channels
        for i, w in enumerate(widths):
            blocks = nn.ModuleList([ResBlock(ch, w, emb, dropout), ResBlock(w, w, emb, dropout)])
            ch = w
            skips += [w, w]
            pool = nn.Conv2d(w, w, 3, stride=2, padding=1) if i < self.n_down else None
            if pool is not None:
                skips.append(w)
            self.down.append(nn.ModuleDict({"blocks": blocks, **({"pool": pool} if pool is not None else {})}))
        self.mid1 = ResBlock(ch, ch, emb, dropout)
        self.mid_attn = SelfAttention2d(ch)
        self.mid2 = ResBlock(ch, ch, emb, dropout)
        self.up = nn.ModuleList()
        for i, w in reversed(list(enumerate(widths))):
            blocks = nn.ModuleList()
            for _ in range(3):
                blocks.append(ResBlock(ch + skips.pop(), w, emb, dropout))
                ch = w
            up = nn.Conv2d(w, w, 3, padding=1) if i > 0 else None
            self.up.append(nn.ModuleDict({"blocks": blocks, **({"upconv": up} if up is not None else {})}))
        self.norm_out = nn.GroupNorm(math.gcd(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, y=None):
        """stage 5: H and W must be divisible by 2**(len(channel_mults) - 1) (ValueError otherwise)."""
        f = 2 ** self.n_down
        if x.shape[-1] % f or x.shape[-2] % f:
            raise ValueError(f"H and W must be divisible by {f}, got {tuple(x.shape[-2:])}")
        e = self.time_emb(t)
        if self.label_emb is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.n_classes, dtype=torch.long, device=x.device)
            e = e + self.label_emb(y)
        elif y is not None:
            raise ValueError("this model has no class embedding (n_classes=None) but was given labels")
        h = self.conv_in(x)
        hs = [h]
        for level in self.down:
            for block in level["blocks"]:
                h = block(h, e)
                hs.append(h)
            if "pool" in level:
                h = level["pool"](h)
                hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, e)), e)
        for level in self.up:
            for block in level["blocks"]:
                h = block(torch.cat([h, hs.pop()], dim=1), e)
            if "upconv" in level:
                h = level["upconv"](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out_conv(F.silu(self.norm_out(h)))


@app.cell(hide_code=True)
def _():
    checks.run(5, TimeUNet=TimeUNet)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.diffusion import DiffusionSchedule, TimeUNet, ddpm_loss, ddpm_sample` gives
    you a complete DDPM. topic 04 samples the same trained networks in 20 steps instead of
    1,000.
    """)
    return


if __name__ == "__main__":
    app.run()
