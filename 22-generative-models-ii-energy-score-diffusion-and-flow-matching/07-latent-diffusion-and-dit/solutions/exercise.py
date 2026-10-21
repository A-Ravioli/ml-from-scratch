import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.diffusion import TimeEmbedding  # yours, from topic 03
    from mlfs.transformer import FeedForward, MultiHeadAttention, PatchEmbedding, sinusoidal_positions  # ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # latent diffusion & DiT · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. two ideas behind most modern
    image generators: run the diffusion in the small latent space of a pretrained
    autoencoder instead of pixel space, and replace the U-Net with a transformer over
    latent patches whose every layer is steered by the time and class through adaptive
    layer norm.

    | stage | you build |
    |---|---|
    | 1 | `patchify`, `unpatchify`, `pos_embed_2d`: images as sequences of patches, and where each patch is |
    | 2 | `DiTBlock`: a transformer block modulated by adaLN-Zero |
    | 3 | `DiT`: the diffusion transformer → `mlfs.diffusion` |
    | 4 | `encode_latents`, `decode_latents`: from images to diffusable latents and back → `mlfs.diffusion` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def patchify(x, patch_size):
    """stage 1: (B, C, H, W) -> (B, N, p*p*C) with N = (H/p)(W/p) patches in raster order and each patch vector ordered
    (row in patch, column in patch, channel); ValueError if p doesn't divide H and W."""
    B, C, H, W = x.shape
    p = patch_size
    if H % p or W % p:
        raise ValueError(f"patch_size {p} must divide H={H} and W={W}")
    x = x.reshape(B, C, H // p, p, W // p, p)  # (B, C, gh, i, gw, j)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C)


@app.function
def unpatchify(tokens, patch_size, channels):
    """stage 1: the inverse of patchify for square images: (B, N, p*p*C) -> (B, C, sqrt(N) p, sqrt(N) p)."""
    B, N, _ = tokens.shape
    g = math.isqrt(N)
    p, C = patch_size, channels
    x = tokens.reshape(B, g, g, p, p, C)  # (B, gh, gw, i, j, C)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)


@app.function
def pos_embed_2d(grid_size, d_model):
    """stage 1: fixed 2-D sin-cos positions (grid_size^2, d_model), raster order: row (r, c) is
    [sinusoidal_positions(grid_size, d_model // 2)[r], sinusoidal_positions(grid_size, d_model // 2)[c]]."""
    pe = sinusoidal_positions(grid_size, d_model // 2)
    r = torch.arange(grid_size).repeat_interleave(grid_size)
    c = torch.arange(grid_size).repeat(grid_size)
    return torch.cat([pe[r], pe[c]], dim=1)


@app.cell(hide_code=True)
def _():
    checks.run(1, patchify=patchify, unpatchify=unpatchify, pos_embed_2d=pos_embed_2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class DiTBlock(nn.Module):
    """x + g1 * attn(modulate(LN(x), b1, s1)), then x + g2 * mlp(modulate(LN(x), b2, s2)); (b, s, g) from the condition."""

    def __init__(self, d_model, n_heads):
        """stage 2: two LayerNorms without affine parameters, mlfs.transformer MultiHeadAttention and FeedForward (4d),
        and adaLN: SiLU -> Linear(d_model, 6 d_model), zero-initialized (so the block starts as the identity)."""
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(d_model)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        """stage 2: x (B, N, d), c (B, d) -> (B, N, d); modulate(h, shift, scale) = h (1 + scale) + shift."""
        b1, s1, g1, b2, s2, g2 = self.ada(c)[:, None, :].chunk(6, dim=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + s1) + b1)
        return x + g2 * self.mlp(self.norm2(x) * (1 + s2) + b2)


@app.cell(hide_code=True)
def _():
    checks.run(2, DiTBlock=DiTBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class DiT(nn.Module):
    """Peebles & Xie (2023): model(x (B, in_channels, input_size, input_size), t (B,), y=None) -> same shape."""

    def __init__(self, input_size=8, patch_size=2, in_channels=4, *, d_model=256, n_layers=6, n_heads=4, n_classes=None):
        """stage 3: mlfs.transformer PatchEmbedding + fixed pos_embed_2d (a registered buffer, not a parameter); condition
        c = TimeEmbedding(d_model)(t) (+ an (n_classes + 1)-row class table, last row = null); self.blocks, an
        nn.ModuleList of n_layers DiTBlocks; self.ada_out (SiLU -> Linear(d_model, 2 d_model): the final (shift, scale))
        and self.out (Linear(d_model, p*p*in_channels)), both zero-initialized; unpatchify."""
        super().__init__()
        self.input_size, self.patch_size, self.in_channels, self.n_classes = input_size, patch_size, in_channels, n_classes
        self.patch_embed = PatchEmbedding(input_size, patch_size, in_channels, d_model)
        self.register_buffer("pos", pos_embed_2d(input_size // patch_size, d_model))
        self.time_emb = TimeEmbedding(d_model)
        self.label_emb = nn.Embedding(n_classes + 1, d_model) if n_classes is not None else None
        self.blocks = nn.ModuleList(DiTBlock(d_model, n_heads) for _ in range(n_layers))
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.out = nn.Linear(d_model, patch_size * patch_size * in_channels)
        for lin in (self.ada_out[1], self.out):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, t, y=None):
        """stage 3: y None -> null label for a conditional model; ValueError for labels without n_classes."""
        c = self.time_emb(t)
        if self.label_emb is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.n_classes, dtype=torch.long, device=x.device)
            c = c + self.label_emb(y)
        elif y is not None:
            raise ValueError("this model has no class embedding (n_classes=None) but was given labels")
        h = self.patch_embed(x) + self.pos
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.ada_out(c)[:, None, :].chunk(2, dim=-1)
        h = self.out(self.norm_out(h) * (1 + scale) + shift)
        return unpatchify(h, self.patch_size, self.in_channels)


@app.cell(hide_code=True)
def _():
    checks.run(3, DiT=DiT, DiTBlock=DiTBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def encode_latents(vae, loader, *, device="cpu"):
    """stage 4: run vae.encode(x) -> (mu, logvar) over a loader of (x, y) batches (eval mode, restored; no gradients);
    returns (latents = the means mu concatenated on the CPU, (N, C, h, w), labels (N,) int64)."""
    was = vae.training
    vae.eval()
    zs, ys = [], []
    try:
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                mu, _ = vae.encode(x.to(device))
                zs.append(mu.cpu())
                ys.append(torch.as_tensor(y).long().cpu())
    finally:
        vae.train(was)
    return torch.cat(zs), torch.cat(ys)


@app.function
def decode_latents(vae, z, *, scale=1.0, batch_size=256, device="cpu"):
    """stage 4: vae.decode(z * scale) in batches (eval mode, restored; no gradients), on the CPU; `scale` undoes the
    normalisation of the latents (pass the latents' std)."""
    was = vae.training
    vae.eval()
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(z), batch_size):
                out.append(vae.decode(z[i:i + batch_size].to(device) * scale).cpu())
    finally:
        vae.train(was)
    return torch.cat(out)


@app.cell(hide_code=True)
def _():
    checks.run(4, encode_latents=encode_latents, decode_latents=decode_latents)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.diffusion import DiT, encode_latents` gives you latent diffusion: train
    `flow_matching_loss` or `ddpm_loss` on `latents / latents.std()` with a `DiT`, sample,
    multiply by the std and decode. that's the recipe behind Stable Diffusion 3 and its
    relatives, at toy scale.
    """)
    return


if __name__ == "__main__":
    app.run()
