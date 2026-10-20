import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # embeddings · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll build **word2vec**: skip-gram with
    negative sampling, the model that showed that a shallow network trained to predict
    neighbouring words learns vectors whose geometry encodes meaning. it's the first piece of
    **`mlfs.seq`**, and the first model of the chapter written in PyTorch.

    | stage | you build |
    |---|---|
    | 1 | `skipgram_pairs`: every (center, context) pair within a window |
    | 2 | `sample_negatives`: noise words from the unigram distribution raised to 3/4 |
    | 3 | `SkipGram`: two embedding tables and the negative-sampling loss |
    | 4 | `most_similar`, `analogy`: nearest neighbours and vector arithmetic by cosine |
    | 5 | `train_skipgram`: the training loop |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def skipgram_pairs(ids, window=2):
    """stage 1: (P, 2) int64 array of (ids[i], ids[j]) for 0 < |i - j| <= window, ordered by i, then j."""
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    offsets = np.array([d for d in range(-window, window + 1) if d != 0], dtype=np.int64)
    j = np.arange(n)[:, None] + offsets[None, :]  # (n, 2 * window) context positions, sorted along each row
    valid = (j >= 0) & (j < n)
    centers = np.broadcast_to(ids[:, None], j.shape)[valid]
    contexts = ids[j[valid]]
    return np.stack([centers, contexts], axis=1)


@app.cell(hide_code=True)
def _():
    checks.run(1, skipgram_pairs=skipgram_pairs)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sample_negatives(counts, n, *, power=0.75, generator=None):
    """stage 2: int64 tensor (n,) of ids drawn with replacement with probability proportional to counts ** power."""
    w = torch.as_tensor(np.asarray(counts, dtype=np.float64)) ** power
    w = torch.where(torch.as_tensor(np.asarray(counts)) > 0, w, torch.zeros_like(w))  # 0 ** 0 is 1: keep unseen ids out
    return torch.multinomial(w / w.sum(), n, replacement=True, generator=generator)


@app.cell(hide_code=True)
def _():
    checks.run(2, sample_negatives=sample_negatives)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class SkipGram(nn.Module):
    """word2vec's two tables: in_embed (center words, the vectors you keep) and out_embed (context words)."""

    def __init__(self, vocab_size, embed_dim):
        """stage 3: in_embed, out_embed = nn.Embedding(vocab_size, embed_dim); in_embed ~ U(-0.5/D, 0.5/D), out_embed = 0."""
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.in_embed.weight.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
            self.out_embed.weight.zero_()

    @property
    def embeddings(self):
        """the word vectors: in_embed.weight, (V, D)."""
        return self.in_embed.weight

    def forward(self, center, context, negatives):
        """stage 3: mean over the batch of -log s(u_o . v_c) - sum_k log s(-u_k . v_c); s is the sigmoid."""
        v = self.in_embed(center)  # (B, D)
        u_pos = self.out_embed(context)  # (B, D)
        u_neg = self.out_embed(negatives)  # (B, K, D)
        pos = F.logsigmoid((u_pos * v).sum(-1))
        neg = F.logsigmoid(-torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + neg).mean()


@app.cell(hide_code=True)
def _():
    checks.run(3, SkipGram=SkipGram)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def most_similar(embeddings, idx, k=10):
    """stage 4: (indices (k,), cosine similarities (k,)) of the k rows most similar to row idx, excluding idx; ndarrays."""
    E = embeddings.detach().cpu().double().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings, float)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    sims = E @ E[idx]
    sims[idx] = -np.inf
    top = np.argsort(-sims, kind="stable")[:k]
    return top, sims[top]


@app.function
def analogy(embeddings, a, b, c, k=1):
    """stage 4: indices (k,) of the rows closest by cosine to b - a + c (rows normalised first), excluding a, b, c."""
    E = embeddings.detach().cpu().double().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings, float)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    target = E[b] - E[a] + E[c]
    sims = E @ target
    sims[[a, b, c]] = -np.inf
    return np.argsort(-sims, kind="stable")[:k]


@app.cell(hide_code=True)
def _():
    checks.run(4, most_similar=most_similar, analogy=analogy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_skipgram(ids, vocab_size, *, embed_dim=50, window=2, n_negatives=5, epochs=1, batch_size=512, lr=0.01,
                   seed=0, device="cpu"):
    """stage 5: train a SkipGram on every pair of ids with Adam, fresh negatives per batch; returns (model, batch losses)."""
    g = torch.Generator().manual_seed(seed)
    pairs = torch.as_tensor(skipgram_pairs(ids, window))
    counts = np.bincount(np.asarray(ids), minlength=vocab_size)
    with torch.random.fork_rng():  # a seeded init that leaves the global generator alone
        torch.manual_seed(seed)
        model = SkipGram(vocab_size, embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(pairs), generator=g)
        for start in range(0, len(pairs), batch_size):
            batch = pairs[order[start:start + batch_size]]
            neg = sample_negatives(counts, len(batch) * n_negatives, generator=g).view(len(batch), n_negatives)
            loss = model(batch[:, 0].to(device), batch[:, 1].to(device), neg.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return model, losses


@app.cell(hide_code=True)
def _():
    checks.run(5, train_skipgram=train_skipgram, most_similar=most_similar)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.seq import SkipGram, most_similar, analogy` gives you these tools, and
    `nn.Embedding` is a lookup table you understand: topics 04–06 and the capstone start every
    model with one.
    """)
    return


if __name__ == "__main__":
    app.run()
