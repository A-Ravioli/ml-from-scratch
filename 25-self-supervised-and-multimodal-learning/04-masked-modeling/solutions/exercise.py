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
    from mlfs.transformer import TransformerBlock, sinusoidal_positions  # yours, from ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # masked modeling · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `patchify`, `unpatchify`: images ↔ sequences of flattened patches, in MAE order → `mlfs.ssl` |
    | 2 | `random_masking`: per-sample shuffle, keep a subset, remember how to undo it → `mlfs.ssl` |
    | 3 | `MAE.__init__`, `MAE.encode`: the encoder, and features for evaluation → `mlfs.ssl` |
    | 4 | `MAE.forward`: encode the visible patches, decode all, loss on the masked ones → `mlfs.ssl` |
    | 5 | `train_mae`: the pretraining loop (AdamW) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def patchify(imgs, patch_size):
    """stage 1: (B, C, H, W) -> (B, N, p*p*C), patches row-major, each flattened in (i, j, c) order; ValueError if H, W not divisible."""
    B, C, H, W = imgs.shape
    p = patch_size
    if H % p or W % p:
        raise ValueError(f"image size {(H, W)} is not divisible by patch_size={p}")
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(B, h * w, p * p * C)


@app.function
def unpatchify(patches, patch_size, channels=3):
    """stage 1: the inverse of patchify for square images: (B, N, p*p*C) -> (B, C, H, W)."""
    B, N, D = patches.shape
    p = patch_size
    h = math.isqrt(N)
    if h * h != N or D != p * p * channels:
        raise ValueError(f"can't unpatchify {tuple(patches.shape)} with patch_size={p}, channels={channels}")
    x = patches.reshape(B, h, h, p, p, channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(B, channels, h * p, h * p)


@app.cell(hide_code=True)
def _():
    checks.run(1, patchify=patchify, unpatchify=unpatchify)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def random_masking(x, mask_ratio, *, generator=None):
    """stage 2: (x_visible (B, N_keep, D), mask (B, N) float 1=masked, ids_restore (B, N) int64); noise = torch.rand(B, N, generator) on the CPU."""
    B, N, D = x.shape
    n_keep = math.floor(N * (1 - mask_ratio) + 0.5)
    if n_keep < 1:
        raise ValueError(f"mask_ratio={mask_ratio} leaves no visible patch out of {N}")
    noise = torch.rand(B, N, generator=generator).to(x.device)
    ids_shuffle = torch.argsort(noise, dim=1)  # ascending: the smallest noise values are kept
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N, device=x.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_visible, mask, ids_restore


@app.cell(hide_code=True)
def _():
    checks.run(2, random_masking=random_masking)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class MAE(nn.Module):
    """masked autoencoder: a ViT encoder on the visible patches, a small transformer decoder for all of them."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, *, encoder_dim=192, encoder_depth=6, encoder_heads=3,
                 decoder_dim=128, decoder_depth=2, decoder_heads=4, mask_ratio=0.75, norm_pix_loss=False):
        """stage 3: patch_embed (Linear), fixed sin-cos positions, blocks + norm (encoder); decoder_embed, mask_token, decoder_blocks, decoder_norm, decoder_pred."""
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"img_size={img_size} is not divisible by patch_size={patch_size}")
        self.patch_size, self.in_channels = patch_size, in_channels
        self.mask_ratio, self.norm_pix_loss = mask_ratio, norm_pix_loss
        self.n_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        # encoder
        self.patch_embed = nn.Linear(patch_dim, encoder_dim)
        self.register_buffer("pos_embed", sinusoidal_positions(self.n_patches, encoder_dim).unsqueeze(0))
        self.blocks = nn.ModuleList(TransformerBlock(encoder_dim, encoder_heads) for _ in range(encoder_depth))
        self.norm = nn.LayerNorm(encoder_dim)
        # decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("decoder_pos_embed", sinusoidal_positions(self.n_patches, decoder_dim).unsqueeze(0))
        self.decoder_blocks = nn.ModuleList(TransformerBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth))
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_dim)

    def encode(self, x):
        """stage 3: (B, C, H, W) -> (B, encoder_dim): every patch through the encoder (no masking), final norm, mean over tokens."""
        t = self.patch_embed(patchify(x, self.patch_size)) + self.pos_embed
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t).mean(dim=1)

    def forward(self, x):
        """stage 4: (loss, pred (B, N, p*p*C), mask (B, N)): mask, encode the visible patches, decode all, MSE on the masked ones."""
        patches = patchify(x, self.patch_size)
        t = self.patch_embed(patches) + self.pos_embed
        t, mask, ids_restore = random_masking(t, self.mask_ratio)
        for blk in self.blocks:
            t = blk(t)
        y = self.decoder_embed(self.norm(t))
        B, n_keep, D = y.shape
        y = torch.cat([y, self.mask_token.expand(B, self.n_patches - n_keep, D)], dim=1)
        y = torch.gather(y, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))  # back to the original patch order
        y = y + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            y = blk(y)
        pred = self.decoder_pred(self.decoder_norm(y))
        target = patches
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss = (per_patch * mask).sum() / mask.sum()
        return loss, pred, mask


@app.cell(hide_code=True)
def _():
    checks.run(3, MAE=MAE, patchify=patchify)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, MAE=MAE, patchify=patchify, random_masking=random_masking)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_mae(model, images, *, epochs=1, batch_size=256, lr=1e-3, weight_decay=0.05, seed=0, device="cpu"):
    """stage 5: AdamW(lr, betas=(0.9, 0.95), weight_decay) on model(x)[0]; returns {'loss': per-step floats}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    n = len(X)
    bs = min(batch_size, n)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    history = {"loss": []}
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n - bs + 1, bs):
            loss = model(X[perm[start:start + bs]].to(device))[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            history["loss"].append(loss.item())
    return history


@app.cell(hide_code=True)
def _():
    checks.run(5, train_mae=train_mae, MAE=MAE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `patchify`, `unpatchify`, `random_masking` and `MAE` are part
    of `mlfs.ssl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
