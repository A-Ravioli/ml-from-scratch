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
    from mlfs.train import CosineSchedule  # yours, from ch 15 · topic 04
    from mlfs.transformer import TransformerBlock, sinusoidal_positions  # yours, from topics 02-03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # building GPT · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. everything here joins **`mlfs.transformer`**, and it's the
    model of the chapter's capstone and of chapter 27.

    | stage | you build |
    |---|---|
    | 1 | `get_batch`: random windows of a token stream and their next-token targets → `mlfs.transformer` |
    | 2 | `GPT`: a decoder-only transformer language model (forward, loss, cache hook, hidden states) → `mlfs.transformer` |
    | 3 | `sample_next_token`: temperature, top-k and nucleus (top-p) sampling → `mlfs.transformer` |
    | 4 | `GPT.generate`: autoregressive generation |
    | 5 | `estimate_loss`, `train_lm`: evaluation and the training loop (AdamW, warmup + cosine, clipping) → `mlfs.transformer` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def get_batch(data, block_size, batch_size, *, generator=None, device="cpu"):
    """stage 1: (x, y) int64 (batch_size, block_size): x = data[s:s+T], y = data[s+1:s+T+1], s uniform in [0, len - T - 1]."""
    data = torch.as_tensor(np.asarray(data) if not torch.is_tensor(data) else data, dtype=torch.long)
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    offsets = torch.arange(block_size)
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    return x.to(device), y.to(device)


@app.cell(hide_code=True)
def _():
    checks.run(1, get_batch=get_batch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class GPT(nn.Module):
    """a decoder-only transformer LM: token + position embeddings, pre-LN causal blocks, final norm, (tied) head."""

    def __init__(self, vocab_size, *, block_size=256, n_layers=4, n_heads=4, d_model=128, d_ff=None, dropout=0.0,
                 pos_encoding="learned", tie_weights=True):
        """stage 2: tok_emb, pos_emb (learned only), drop, blocks, ln_f, lm_head; GPT-2 initialization."""
        super().__init__()
        if pos_encoding not in ("learned", "sinusoidal"):
            raise ValueError(f"pos_encoding must be 'learned' or 'sinusoidal', got {pos_encoding!r}")
        self.vocab_size, self.block_size, self.n_layers = vocab_size, block_size, n_layers
        self.pos_encoding = pos_encoding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if pos_encoding == "learned":
            self.pos_emb = nn.Embedding(block_size, d_model)
        else:
            self.pos_emb = None
            self.register_buffer("pe", sinusoidal_positions(block_size, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout, norm_first=True)
                                    for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # GPT-2 initialization: N(0, 0.02) everywhere, residual output layers scaled by 1/sqrt(2 n_layers)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(block.mlp.fc2.weight, std=0.02 / math.sqrt(2 * n_layers))
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def _stream(self, idx, cache=None):
        B, T = idx.shape
        offset = cache.seq_len if cache is not None else 0  # read before the blocks update the cache
        if offset + T > self.block_size:
            raise ValueError(f"sequence of {offset + T} tokens is longer than block_size={self.block_size}")
        pos = torch.arange(offset, offset + T, device=idx.device)
        if self.pos_emb is not None:
            x = self.tok_emb(idx) + self.pos_emb(pos)
        else:  # Vaswani et al.: scale the 0.02-std embeddings by sqrt(d) so unit-size sinusoids don't drown them
            x = self.tok_emb(idx) * math.sqrt(self.tok_emb.embedding_dim) + self.pe[pos]
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, is_causal=True, cache=None if cache is None else cache[i])
        return self.ln_f(x)

    def hidden_states(self, idx):
        """stage 2: idx (B, T) -> (B, T, d_model), the final-norm output that the head reads."""
        return self._stream(idx)

    def forward(self, idx, targets=None, *, cache=None):
        """stage 2: (logits (B, T, V), loss or None); targets pre-shifted, -100 ignored; cache: block i gets cache[i]."""
        logits = self.lm_head(self._stream(idx, cache))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, generator=None):
        """stage 4: append max_new_tokens sampled tokens to idx (B, T); the context is cropped to block_size; no cache."""
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.block_size:])
                nxt = sample_next_token(logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p,
                                        generator=generator)
                idx = torch.cat([idx, nxt[:, None].to(idx.device)], dim=1)
        finally:
            self.train(was_training)
        return idx


@app.cell(hide_code=True)
def _():
    checks.run(2, GPT=GPT)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def sample_next_token(logits, *, temperature=1.0, top_k=None, top_p=None, generator=None):
    """stage 3: logits (B, V) -> (B,) int64; temperature 0 is argmax; top_k, then top_p (nucleus) filtering."""
    if top_p is not None and top_p <= 0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits.detach().float() / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        before = probs.cumsum(dim=-1) - probs  # mass of the tokens ranked above each one
        sorted_logits = sorted_logits.masked_fill(before >= top_p, float("-inf"))  # the top token always survives
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    gen_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(gen_device), 1, generator=generator).squeeze(-1).to(logits.device)


@app.cell(hide_code=True)
def _():
    checks.run(3, sample_next_token=sample_next_token)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, GPT=GPT, sample_next_token=sample_next_token)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def estimate_loss(model, data, block_size, *, batch_size=32, n_batches=20, device="cpu"):
    """stage 5: mean loss over n_batches batches (fixed generator seed 0), eval mode, no_grad; restores the mode."""
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(0)
    losses = []
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = get_batch(data, block_size, batch_size, generator=g, device=device)
                losses.append(model(x, y)[1].item())
    finally:
        model.train(was_training)
    return float(np.mean(losses))


@app.function
def train_lm(model, train_data, block_size, *, val_data=None, batch_size=32, max_steps=1000, lr=3e-4,
             weight_decay=0.1, warmup_steps=100, grad_clip=1.0, eval_interval=100, eval_batches=20, device="cpu",
             seed=0):
    """stage 5: AdamW (betas 0.9, 0.95; decay on >= 2-D params), warmup from 0 + cosine to lr/10, clipping; history."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    # warmup longer than the run is clamped; the first step uses lr 0 (CosineSchedule's lr_at(0))
    sched = CosineSchedule(opt, max_steps, warmup_steps=min(warmup_steps, max_steps - 1), min_lr=lr / 10)
    g = torch.Generator().manual_seed(seed)
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}
    model.train()
    for step in range(1, max_steps + 1):
        x, y = get_batch(train_data, block_size, batch_size, generator=g, device=device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        lr_used = opt.param_groups[0]["lr"]
        sched.step()
        if step % eval_interval == 0 or step == max_steps:
            history["step"].append(step)
            history["lr"].append(lr_used)
            history["train_loss"].append(estimate_loss(model, train_data, block_size, batch_size=batch_size,
                                                       n_batches=eval_batches, device=device))
            history["val_loss"].append(float("nan") if val_data is None else
                                       estimate_loss(model, val_data, block_size, batch_size=batch_size,
                                                     n_batches=eval_batches, device=device))
    model.eval()
    return history


@app.cell(hide_code=True)
def _():
    checks.run(5, estimate_loss=estimate_loss, train_lm=train_lm, GPT=GPT, get_batch=get_batch)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import GPT, train_lm, estimate_loss, get_batch, sample_next_token`
    gives you a complete language-model toolkit. topic 06 makes generation fast with a KV cache,
    and the capstone trains three GPTs on Shakespeare.
    """)
    return


if __name__ == "__main__":
    app.run()
