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
    # embeddings · build

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
    raise NotImplementedError("stage1: implement skipgram_pairs")


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
    raise NotImplementedError("stage2: implement sample_negatives")


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
        raise NotImplementedError("stage3: implement SkipGram.__init__")

    @property
    def embeddings(self):
        """the word vectors: in_embed.weight, (V, D)."""
        return self.in_embed.weight

    def forward(self, center, context, negatives):
        """stage 3: mean over the batch of -log s(u_o . v_c) - sum_k log s(-u_k . v_c); s is the sigmoid."""
        raise NotImplementedError("stage3: implement SkipGram.forward")


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
    raise NotImplementedError("stage4: implement most_similar")


@app.function
def analogy(embeddings, a, b, c, k=1):
    """stage 4: indices (k,) of the rows closest by cosine to b - a + c (rows normalised first), excluding a, b, c."""
    raise NotImplementedError("stage4: implement analogy")


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
    raise NotImplementedError("stage5: implement train_skipgram")


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
