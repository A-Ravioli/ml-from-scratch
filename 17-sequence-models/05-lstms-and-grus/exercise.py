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
    from mlfs.seq import RNN  # yours, from topic 04


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # LSTMs & GRUs · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll build the two gated recurrent
    cells that made RNNs work in practice, the **LSTM** and the **GRU**, then multi-layer
    sequence modules on top of them, all in **`mlfs.seq`**, and a recall task that measures how
    long each architecture can remember. after this topic, `nn.LSTM` and `nn.GRU` are
    unlocked: you'll know exactly what they compute.

    | stage | you build |
    |---|---|
    | 1 | `LSTMCell`: input, forget and output gates and a cell state |
    | 2 | `GRUCell`: reset and update gates, no separate cell state |
    | 3 | `LSTM`, `GRU`: multi-layer, batch-first sequence modules |
    | 4 | `recall_batch`, `SeqTagger`, `copy_accuracy`: a memory test for any recurrent module |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class LSTMCell(nn.Module):
    """one LSTM step: x (B, D), state (h, c) each (B, H) or None -> (h, c); gates in the order i, f, g, o."""

    def __init__(self, input_size, hidden_size):
        """stage 1: `ih` = Linear(D, 4H), `hh` = Linear(H, 4H); then set the forget-gate slice of ih.bias, [H:2H], to 1."""
        raise NotImplementedError("stage1: implement LSTMCell.__init__")

    def forward(self, x, state=None):
        """stage 1: c' = f * c + i * g, h' = o * tanh(c') with i, f, o sigmoid and g tanh of the four chunks of ih(x) + hh(h)."""
        raise NotImplementedError("stage1: implement LSTMCell.forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, LSTMCell=LSTMCell)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class GRUCell(nn.Module):
    """one GRU step: x (B, D), h (B, H) or None -> h; gates in the order r, z, n (PyTorch's convention)."""

    def __init__(self, input_size, hidden_size):
        """stage 2: `ih` = Linear(D, 3H), `hh` = Linear(H, 3H)."""
        raise NotImplementedError("stage2: implement GRUCell.__init__")

    def forward(self, x, h=None):
        """stage 2: r, z = sigmoid(...), n = tanh(ih_n + r * hh_n), h' = (1 - z) * n + z * h."""
        raise NotImplementedError("stage2: implement GRUCell.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, GRUCell=GRUCell)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class LSTM(nn.Module):
    """batch-first multi-layer LSTM: x (B, T, D) -> (outputs (B, T, H), (h_n, c_n) each (num_layers, B, H))."""

    def __init__(self, input_size, hidden_size, *, num_layers=1):
        """stage 3: `cells` = ModuleList of LSTMCell (input_size for the first layer, hidden_size above)."""
        raise NotImplementedError("stage3: implement LSTM.__init__")

    def forward(self, x, state=None):
        """stage 3: each layer runs over the whole sequence from its slice of state (zeros if None)."""
        raise NotImplementedError("stage3: implement LSTM.forward")


@app.class_definition
class GRU(nn.Module):
    """batch-first multi-layer GRU: x (B, T, D) -> (outputs (B, T, H), h_n (num_layers, B, H))."""

    def __init__(self, input_size, hidden_size, *, num_layers=1):
        """stage 3: `cells` = ModuleList of GRUCell (input_size for the first layer, hidden_size above)."""
        raise NotImplementedError("stage3: implement GRU.__init__")

    def forward(self, x, h0=None):
        """stage 3: each layer runs over the whole sequence from its slice of h0 (zeros if None)."""
        raise NotImplementedError("stage3: implement GRU.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, LSTM=LSTM, GRU=GRU, LSTMCell=LSTMCell, GRUCell=GRUCell)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def recall_batch(batch_size, delay, n_symbols, *, generator=None):
    """stage 4: x (B, delay + 1): a symbol in [0, K) then delay distractors in [K, 2K); y: -100 except y[:, -1] = x[:, 0]."""
    raise NotImplementedError("stage4: implement recall_batch")


@app.class_definition
class SeqTagger(nn.Module):
    """embedding -> a recurrent module ('rnn', 'gru' or 'lstm', yours) -> linear: one prediction per time step."""

    def __init__(self, vocab_size, hidden_size, *, cell="lstm"):
        """stage 4: `embed` (nn.Embedding(V, H)), `rnn` (RNN, GRU or LSTM with H units), `head` (nn.Linear(H, V))."""
        raise NotImplementedError("stage4: implement SeqTagger.__init__")

    def forward(self, idx):
        """stage 4: idx (B, T) -> logits (B, T, V)."""
        raise NotImplementedError("stage4: implement SeqTagger.forward")


@app.function
def copy_accuracy(model, x, y):
    """stage 4: fraction of the positions with y != -100 where argmax of model(x) equals y (no gradients, eval mode)."""
    raise NotImplementedError("stage4: implement copy_accuracy")


@app.cell(hide_code=True)
def _():
    checks.run(4, recall_batch=recall_batch, SeqTagger=SeqTagger, copy_accuracy=copy_accuracy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.seq import LSTM, GRU` gives you these modules, and `nn.LSTM`, `nn.GRU`,
    `nn.LSTMCell` and `nn.GRUCell` are unlocked: they compute exactly what yours do, much faster.
    """)
    return


if __name__ == "__main__":
    app.run()
