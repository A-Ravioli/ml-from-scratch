import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.transformer import TransformerBlock, sinusoidal_positions  # yours, from ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # multimodal learning · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `clip_loss`: the symmetric contrastive loss between images and captions → `mlfs.ssl` |
    | 2 | `TextTransformer`: a small text encoder, mean-pooled over real tokens |
    | 3 | `MiniCLIP`: two encoders, two projections, one learnable temperature → `mlfs.ssl` |
    | 4 | `zero_shot_predict`, `class_embeddings`: classifiers written in words → `mlfs.ssl` |
    | 5 | `train_clip`: the contrastive training loop |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def clip_loss(image_emb, text_emb, logit_scale):
    """stage 1: mean of cross-entropy over rows and over columns of logit_scale * cos-similarity matrix; pairs on the diagonal."""
    img = F.normalize(image_emb, dim=1)
    txt = F.normalize(text_emb, dim=1)
    logits = logit_scale * img @ txt.T
    targets = torch.arange(len(img), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


@app.cell(hide_code=True)
def _():
    checks.run(1, clip_loss=clip_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class TextTransformer(nn.Module):
    """token ids -> one vector per caption: embeddings + positions, transformer blocks, mean over real tokens."""

    def __init__(self, vocab_size, *, max_len=16, d_model=64, n_layers=2, n_heads=4, pad_id=0):
        """stage 2: tok_emb, fixed sinusoidal positions (buffer pos_emb), n_layers TransformerBlocks, final LayerNorm."""
        super().__init__()
        self.pad_id, self.max_len = pad_id, max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_emb", sinusoidal_positions(max_len, d_model))
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, tokens):
        """stage 2: tokens (B, T) int64 (T <= max_len, pad_id = padding) -> (B, d_model); pads are neither attended to nor pooled."""
        B, T = tokens.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len={self.max_len}")
        real = tokens != self.pad_id  # (B, T)
        x = self.tok_emb(tokens) + self.pos_emb[:T]
        key_mask = real[:, None, None, :]  # (B, 1, 1, T): True = may attend to this key
        for blk in self.blocks:
            x = blk(x, mask=key_mask)
        x = self.ln_f(x)
        w = real.unsqueeze(-1).to(x.dtype)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


@app.cell(hide_code=True)
def _():
    checks.run(2, TextTransformer=TextTransformer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class MiniCLIP(nn.Module):
    """an image encoder and a text encoder, projected into one embedding space, with a learnable temperature."""

    def __init__(self, image_encoder, text_encoder, image_dim, text_dim, *, embed_dim=128, init_temperature=0.07):
        """stage 3: image_proj, text_proj (Linear, no bias) to embed_dim; log_scale = Parameter(log(1 / init_temperature))."""
        super().__init__()
        self.image_encoder, self.text_encoder = image_encoder, text_encoder
        self.image_proj = nn.Linear(image_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, embed_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / init_temperature)))

    def encode_image(self, images):
        """stage 3: (B, embed_dim) unit vectors."""
        return F.normalize(self.image_proj(self.image_encoder(images)), dim=1)

    def encode_text(self, texts):
        """stage 3: (B, embed_dim) unit vectors."""
        return F.normalize(self.text_proj(self.text_encoder(texts)), dim=1)

    def forward(self, images, texts):
        """stage 3: (image_emb, text_emb, logit_scale): normalised embeddings and exp(log_scale) clamped to at most 100."""
        return self.encode_image(images), self.encode_text(texts), self.log_scale.exp().clamp(max=100.0)


@app.cell(hide_code=True)
def _():
    checks.run(3, MiniCLIP=MiniCLIP)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def zero_shot_predict(image_emb, class_emb):
    """stage 4: (B, d) and (n_classes, d) -> (B,) int64: the class of highest cosine similarity."""
    return (F.normalize(image_emb, dim=1) @ F.normalize(class_emb, dim=1).T).argmax(dim=1)


@app.function
def class_embeddings(model, class_names, templates, tokenize):
    """stage 4: (n_classes, d) unit vectors: per class, the mean of the normalised text embeddings of every template, renormalised."""
    was_training = model.training
    model.eval()
    try:
        out = []
        with torch.no_grad():
            for name in class_names:
                emb = model.encode_text(tokenize([t.format(name) for t in templates]))
                out.append(F.normalize(F.normalize(emb, dim=1).mean(dim=0), dim=0))
    finally:
        model.train(was_training)
    return torch.stack(out)


@app.cell(hide_code=True)
def _():
    checks.run(4, zero_shot_predict=zero_shot_predict, class_embeddings=class_embeddings, MiniCLIP=MiniCLIP)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_clip(model, images, texts, *, epochs=1, batch_size=256, lr=1e-3, seed=0, device="cpu"):
    """stage 5: Adam on clip_loss over shuffled (image, caption) batches; history {'loss', 'logit_scale'} per step."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    T = torch.as_tensor(np.asarray(texts), dtype=torch.int64)
    n = len(X)
    bs = min(batch_size, n)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "logit_scale": []}
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n - bs + 1, bs):
            idx = perm[start:start + bs]
            img, txt, scale = model(X[idx].to(device), T[idx].to(device))
            loss = clip_loss(img, txt, scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            history["loss"].append(loss.item())
            history["logit_scale"].append(scale.item())
    return history


@app.cell(hide_code=True)
def _():
    checks.run(5, train_clip=train_clip, MiniCLIP=MiniCLIP, zero_shot_predict=zero_shot_predict)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `clip_loss`, `MiniCLIP`, `zero_shot_predict` and
    `class_embeddings` are part of `mlfs.ssl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
