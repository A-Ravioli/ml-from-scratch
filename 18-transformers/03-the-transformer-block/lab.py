import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.text import CharTokenizer  # yours, from ch 17 · topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    DEPTH, WIDTH, SEQ, STEPS = 24, 64, 32, 200


@app.class_definition
class DeepLM(nn.Module):
    """a character LM: embeddings + learned positions, `n_layers` of your TransformerBlock (causal), a linear head."""

    def __init__(self, vocab_size, n_layers, *, norm_first, d=WIDTH, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList(ex.TransformerBlock(d, n_heads, norm_first=norm_first) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d) if norm_first else nn.Identity()  # post-LN already ends in a norm
        self.head = nn.Linear(d, vocab_size)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        for block in self.blocks:
            h = block(h, is_causal=True)
        return self.head(self.ln_f(h))


@app.function
def lm_batch(ids, generator, batch_size=16):
    starts = torch.randint(0, len(ids) - SEQ - 1, (batch_size,), generator=generator).tolist()
    return torch.stack([ids[s:s + SEQ] for s in starts]), torch.stack([ids[s + 1:s + SEQ + 1] for s in starts])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the transformer block · lab

    every experiment here stacks **your** `TransformerBlock` from `exercise.py` (with your
    `MultiHeadAttention` inside) into a 24-layer character language model on tiny
    Shakespeare, once pre-LN and once post-LN, and measures it with your `block_grad_norms`.
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare your explanations with `solutions/lab_notes.md` afterwards. the lab takes
    two to three minutes on a laptop CPU.
    """)
    return


@app.cell
def _():
    text = datasets.tiny_shakespeare()
    tok = CharTokenizer().train(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids = ids[: int(0.8 * len(ids))]
    mo.md(f"{'tiny Shakespeare' if not datasets.is_offline() else 'offline stand-in'}: {len(text):,} characters, "
          f"vocabulary {tok.vocab_size}; models: {DEPTH} blocks of width {WIDTH}, windows of {SEQ} characters")
    return tok, train_ids


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: where the gradient goes at initialization

    one batch, one backward pass, no training: the gradient norm that reaches each of the 24
    blocks, for a freshly initialized pre-LN and post-LN model (same seed).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "in which model does the gradient reaching the blocks change with depth, and does it grow or shrink "
        "towards the output?"
    )
    p1
    return (p1,)


@app.cell
def _(p1, tok, train_ids):
    ui.gate(p1)
    _g = torch.Generator().manual_seed(0)
    _x, _y = lm_batch(train_ids, _g)
    grads = {}
    for _nf in (True, False):
        torch.manual_seed(0)
        _m = DeepLM(tok.vocab_size, DEPTH, norm_first=_nf)
        F.cross_entropy(_m(_x).reshape(-1, tok.vocab_size), _y.reshape(-1)).backward()
        grads["pre-LN" if _nf else "post-LN"] = ex.block_grad_norms(_m.blocks)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    for _k, _v in grads.items():
        _ax.plot(np.arange(1, DEPTH + 1), _v, "o-", ms=3, label=_k)
    _ax.set(xlabel="block (1 = nearest the input)", ylabel="gradient norm", ylim=(0, None))
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(" · ".join(f"{k}: first block {v[0]:.3f}, last block {v[-1]:.3f}, ratio {v[-1] / v[0]:.2f}"
                                      for k, v in grads.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in a pre-LN model the residual stream runs untouched from the input to
    the final norm. what does that mean for the gradient arriving at block 1 compared with
    block 24? in a post-LN model every block's output goes through a layer norm before the next
    block: what does the backward pass pick up at each of those norms? the differences at
    initialization look modest; keep the shape in mind for experiment 2.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: training without a safety net

    three runs of 200 Adam steps at learning rate $10^{-3}$ (batches of 16 windows): pre-LN
    with a constant learning rate, post-LN with a constant learning rate, and post-LN with a
    linear **warmup** from 0 over the first 100 steps. the y-axis is bits per character on the
    training batches (a 25-step moving average).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "will the 24-layer post-LN model train at all without warmup? and with warmup, will it catch up with pre-LN?"
    )
    p2
    return (p2,)


@app.function
def train_deep(vocab_size, train_ids, *, norm_first, warmup=0, lr=1e-3, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = DeepLM(vocab_size, DEPTH, norm_first=norm_first)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / warmup) if warmup else lr
        x, y = lm_batch(train_ids, g)
        loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item() / math.log(2))
    return np.array(losses)


@app.cell
def _(p2, tok, train_ids):
    ui.gate(p2)
    _t0 = time.perf_counter()
    curves = {
        "pre-LN": train_deep(tok.vocab_size, train_ids, norm_first=True),
        "post-LN": train_deep(tok.vocab_size, train_ids, norm_first=False),
        "post-LN + warmup": train_deep(tok.vocab_size, train_ids, norm_first=False, warmup=100),
    }
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _k, _c in curves.items():
        _ax.plot(np.arange(25, STEPS + 1), np.convolve(_c, np.ones(25) / 25, "valid"), label=_k)
    _ax.set(xlabel="step", ylabel="training bits per character")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(" · ".join(f"{k}: last 25 steps {c[-25:].mean():.2f}" for k, c in curves.items())
                           + f" bits per character · {time.perf_counter() - _t0:.0f} s")])
    return (curves,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the stuck post-LN model sits at a loss you can identify: compare it with the
    entropy of the character frequencies alone (a model that ignores context). what has it
    learned, and what hasn't it? why does starting with tiny steps let it escape, when a 10×
    smaller or 10× larger constant learning rate doesn't (lab notes)? why is pre-LN, with the
    same learning rate and no warmup, fine? which variant would you choose for a 100-layer model?
    """)
    return


if __name__ == "__main__":
    app.run()
