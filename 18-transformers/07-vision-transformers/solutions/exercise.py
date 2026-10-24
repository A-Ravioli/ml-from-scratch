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
    from mlfs.transformer import TransformerBlock  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # vision transformers · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `patchify`, `PatchEmbedding`: an image as a sequence of patch tokens → `mlfs.transformer` |
    | 2 | `ViT`: [CLS] token, learned positions, transformer blocks, a classification head → `mlfs.transformer` |
    | 3 | `attention_distance`: how far, in pixels, each head of a ViT looks |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def patchify(x, patch_size):
    """stage 1: (B, C, H, W) -> (B, N, C*p*p), patches in row-major order, each flattened in (c, i, j) order."""
    B, C, H, W = x.shape
    p = patch_size
    if H % p or W % p:
        raise ValueError(f"image size {(H, W)} is not divisible by patch_size={p}")
    x = x.reshape(B, C, H // p, p, W // p, p)  # (B, C, gh, i, gw, j)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)


@app.class_definition
class PatchEmbedding(nn.Module):
    """split an image into non-overlapping p x p patches and project each to d_model: Conv2d(kernel=stride=p)."""

    def __init__(self, img_size, patch_size, in_channels, d_model):
        """stage 1: proj = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size); n_patches; ValueError if not divisible."""
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"img_size={img_size} is not divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """stage 1: (B, C, H, W) -> (B, n_patches, d_model)."""
        return self.proj(x).flatten(2).transpose(1, 2)


@app.cell(hide_code=True)
def _():
    checks.run(1, patchify=patchify, PatchEmbedding=PatchEmbedding)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ViT(nn.Module):
    """vision transformer: patch tokens (+ [CLS]) + learned positions -> pre-LN blocks -> final norm -> head."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, *, d_model=192, n_layers=6, n_heads=3,
                 d_ff=None, dropout=0.0, pool="cls"):
        """stage 2: patch_embed, cls_token (if pool='cls'), pos_emb, blocks, norm, head (Identity if num_classes=0)."""
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
        self.pool, self.d_model = pool, d_model
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        n_tokens = self.patch_embed.n_patches + (1 if pool == "cls" else 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pool == "cls" else None
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout, norm_first=True)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes) if num_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x):
        """stage 2: (B, C, H, W) -> (B, N (+1), d_model) tokens after the final norm, [CLS] first."""
        t = self.patch_embed(x)
        if self.cls_token is not None:
            t = torch.cat([self.cls_token.expand(t.shape[0], -1, -1), t], dim=1)
        t = self.drop(t + self.pos_emb)
        for block in self.blocks:
            t = block(t)
        return self.norm(t)

    def forward(self, x):
        """stage 2: (B, C, H, W) -> logits (B, num_classes), or pooled features (B, d_model) if num_classes=0."""
        t = self.forward_features(x)
        pooled = t[:, 0] if self.pool == "cls" else t.mean(dim=1)
        return self.head(pooled)


@app.cell(hide_code=True)
def _():
    checks.run(2, ViT=ViT, PatchEmbedding=PatchEmbedding)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
@torch.no_grad()
def attention_distance(model, x):
    """stage 3: (n_layers, n_heads) mean distance in pixels from each patch query to the patches it attends to."""
    was_training = model.training
    model.eval()
    try:
        for block in model.blocks:
            block.attn.save_weights = True
        model(x)
        p = model.patch_embed.patch_size
        side = int(math.isqrt(model.patch_embed.n_patches))
        rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        coords = torch.stack([rows.flatten(), cols.flatten()], dim=1).float() * p  # patch positions in pixels
        dist = torch.cdist(coords, coords).to(x.device)  # (N, N)
        skip = 1 if model.cls_token is not None else 0
        out = []
        for block in model.blocks:
            w = block.attn.last_weights[:, :, skip:, skip:]  # patch queries -> patch keys (B, H, N, N)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-12)  # renormalize without the [CLS] key
            out.append((w * dist).sum(-1).mean(dim=(0, 2)))  # (H,)
    finally:
        for block in model.blocks:
            block.attn.save_weights = False
            block.attn.last_weights = None
        model.train(was_training)
    return torch.stack(out).cpu()


@app.cell(hide_code=True)
def _():
    checks.run(3, attention_distance=attention_distance, ViT=ViT)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import ViT, PatchEmbedding` gives you a vision transformer; chapter 22's
    diffusion transformer patchifies latents the same way, and chapter 25's MAE and CLIP use your
    ViT as their image encoder.
    """)
    return


if __name__ == "__main__":
    app.run()
