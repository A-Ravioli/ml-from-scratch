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
    from mlfs.transformer import TransformerBlock  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BERT-style encoders · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `mask_tokens`: BERT's 80/10/10 corruption for masked language modeling → `mlfs.transformer` |
    | 2 | `BERT`: a bidirectional encoder with a tied MLM head → `mlfs.transformer` |
    | 3 | `mlm_loss`, `pretrain_mlm`: the pretraining objective and loop |
    | 4 | `BERTClassifier`, `finetune`: a classification head on the pretrained encoder, and fine-tuning |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def mask_tokens(idx, *, mask_token_id, vocab_size, mask_prob=0.15, special_ids=(), generator=None):
    """stage 1: (inputs, labels): mask_prob of non-special positions selected; 80% [MASK], 10% random, 10% kept."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    special = torch.zeros_like(idx, dtype=torch.bool)
    for s in special_ids:
        special |= idx == s
    # draw on the generator's device (CPU by default) in a fixed order, then move to idx's device (MPS/CUDA)
    gdev = generator.device if generator is not None else torch.device("cpu")
    u = torch.rand(idx.shape, generator=generator, device=gdev).to(idx.device)
    selected = (u < mask_prob) & ~special
    action = torch.rand(idx.shape, generator=generator, device=gdev).to(idx.device)
    random_tokens = torch.randint(0, vocab_size, idx.shape, generator=generator, device=gdev).to(idx.device)
    inputs = idx.clone()
    inputs[selected & (action < 0.8)] = mask_token_id
    swap = selected & (action >= 0.8) & (action < 0.9)
    inputs[swap] = random_tokens[swap]
    labels = torch.where(selected, idx, torch.full_like(idx, -100))
    return inputs, labels


@app.cell(hide_code=True)
def _():
    checks.run(1, mask_tokens=mask_tokens)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class BERT(nn.Module):
    """a bidirectional transformer encoder: token + learned positions, pre-LN blocks, final norm, tied MLM head."""

    def __init__(self, vocab_size, *, max_len=128, n_layers=4, n_heads=4, d_model=128, d_ff=None, dropout=0.1):
        """stage 2: tok_emb, pos_emb, drop, blocks, ln_f; MLM head: mlm_dense, mlm_norm, mlm_bias (decoder = tok_emb)."""
        super().__init__()
        self.vocab_size, self.max_len, self.d_model = vocab_size, max_len, d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout, norm_first=True)
                                    for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_norm = nn.LayerNorm(d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, idx, attention_mask=None):
        """stage 2: idx (B, T), attention_mask (B, T) bool True = real token -> hidden (B, T, d_model)."""
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"sequence of {T} tokens is longer than max_len={self.max_len}")
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        mask = None if attention_mask is None else attention_mask[:, None, None, :].bool()
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.ln_f(x)

    def mlm_logits(self, hidden):
        """stage 2: hidden (B, T, d) -> (B, T, V): dense -> GELU -> LayerNorm -> tied decoder + bias."""
        h = self.mlm_norm(F.gelu(self.mlm_dense(hidden)))
        return h @ self.tok_emb.weight.T + self.mlm_bias


@app.cell(hide_code=True)
def _():
    checks.run(2, BERT=BERT)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def mlm_loss(model, inputs, labels, attention_mask=None):
    """stage 3: cross-entropy of model.mlm_logits(model(inputs, attention_mask)) against labels, -100 ignored;
    a batch with no target at all gives a zero loss (still connected to the graph), not NaN."""
    logits = model.mlm_logits(model(inputs, attention_mask))
    if not (labels != -100).any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


@app.function
def pretrain_mlm(model, data, *, mask_token_id, special_ids=(), attention_mask=None, steps=500, batch_size=32,
                 lr=1e-3, mask_prob=0.15, seed=0):
    """stage 3: AdamW on mlm_loss over random rows of data (N, T), freshly masked every step; returns the list of losses."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    for _ in range(steps):
        rows = torch.randint(0, data.shape[0], (batch_size,), generator=g)
        am = None if attention_mask is None else attention_mask[rows]
        inputs, labels = mask_tokens(data[rows], mask_token_id=mask_token_id, vocab_size=model.vocab_size,
                                     mask_prob=mask_prob, special_ids=special_ids, generator=g)
        loss = mlm_loss(model, inputs, labels, am)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


@app.cell(hide_code=True)
def _():
    checks.run(3, mlm_loss=mlm_loss, pretrain_mlm=pretrain_mlm, BERT=BERT, mask_tokens=mask_tokens)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class BERTClassifier(nn.Module):
    """an encoder plus a classification head on the first ([CLS]) token or on the mean of the real tokens."""

    def __init__(self, encoder, num_classes, *, pool="cls", dropout=0.1):
        """stage 4: keeps `encoder`, a `head` nn.Linear(d_model, num_classes); pool 'cls' or 'mean' (ValueError otherwise)."""
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
        self.encoder, self.pool = encoder, pool
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.d_model, num_classes)

    def forward(self, idx, attention_mask=None):
        """stage 4: idx (B, T), attention_mask (B, T) or None -> logits (B, num_classes)."""
        h = self.encoder(idx, attention_mask)
        if self.pool == "cls":
            pooled = h[:, 0]
        else:
            m = torch.ones(idx.shape, device=h.device) if attention_mask is None else attention_mask.float()
            pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(self.drop(pooled))


@app.function
def finetune(model, X, y, *, attention_mask=None, epochs=5, batch_size=32, lr=3e-4, seed=0):
    """stage 4: AdamW on cross-entropy over shuffled minibatches of (X, y) for `epochs`; returns per-epoch mean losses."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        total = 0.0
        for start in range(0, len(X), batch_size):
            b = perm[start:start + batch_size]
            logits = model(X[b], None if attention_mask is None else attention_mask[b])
            loss = F.cross_entropy(logits, y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        history.append(total / len(X))
    model.eval()
    return history


@app.cell(hide_code=True)
def _():
    checks.run(4, BERTClassifier=BERTClassifier, finetune=finetune, BERT=BERT)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import BERT, mask_tokens` gives you a bidirectional encoder and its
    pretraining corruption. chapter 25 returns to masked modeling for images (MAE) and to
    contrastive pretraining, and chapter 27 to fine-tuning.
    """)
    return


if __name__ == "__main__":
    app.run()
