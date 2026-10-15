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
    # RNNs & BPTT · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll build a recurrent network twice:
    once in NumPy with backpropagation through time written by hand, and once as a PyTorch
    module (**`mlfs.seq.RNN`**). then you'll measure how gradients fade or explode across
    time steps, and train a character-level language model with truncated BPTT.

    | stage | you build |
    |---|---|
    | 1 | `rnn_forward`: the recurrence $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$ in NumPy |
    | 2 | `rnn_backward`: backpropagation through time, by hand |
    | 3 | `RNN`: a multi-layer PyTorch RNN built from `nn.Linear` |
    | 4 | `hidden_grad_norms`: how much the last step's loss reaches back to each step |
    | 5 | `batchify`, `tbptt_batches`, `CharRNN`, `train_tbptt`: a language model with truncated BPTT |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def rnn_forward(x, h0, W_xh, W_hh, b):
    """stage 1: hidden states hs (B, T, H): h_t = tanh(x_t @ W_xh.T + h_{t-1} @ W_hh.T + b), starting from h0 (B, H)."""
    raise NotImplementedError("stage1: implement rnn_forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, rnn_forward=rnn_forward)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def rnn_backward(x, h0, hs, W_xh, W_hh, dhs):
    """stage 2: dict of gradients dW_xh, dW_hh, db, dh0, dx, given dhs (B, T, H) = dL/dh_t from each step's own loss."""
    raise NotImplementedError("stage2: implement rnn_backward")


@app.cell(hide_code=True)
def _():
    checks.run(2, rnn_forward=rnn_forward, rnn_backward=rnn_backward)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class RNN(nn.Module):
    """batch-first multi-layer Elman RNN: x (B, T, D) -> (outputs (B, T, H) of the top layer, h_n (num_layers, B, H))."""

    def __init__(self, input_size, hidden_size, *, num_layers=1, nonlinearity="tanh"):
        """stage 3: ModuleLists `ih` (Linear(in, H) per layer, in = D then H) and `hh` (Linear(H, H)); ValueError for an unknown nonlinearity."""
        raise NotImplementedError("stage3: implement RNN.__init__")

    def forward(self, x, h0=None):
        """stage 3: run every layer over the whole sequence; layer k's outputs are layer k+1's inputs."""
        raise NotImplementedError("stage3: implement RNN.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, RNN=RNN)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def hidden_grad_norms(model, x):
    """stage 4: (T,) ndarray: ||dL/dh_t|| (mean over the batch) for L = sum of the final hidden state, one-layer model."""
    raise NotImplementedError("stage4: implement hidden_grad_norms")


@app.cell(hide_code=True)
def _():
    checks.run(4, RNN=RNN, hidden_grad_norms=hidden_grad_norms)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def batchify(ids, batch_size):
    """stage 5: int64 tensor (batch_size, N // batch_size): the ids cut into batch_size contiguous streams (the tail dropped)."""
    raise NotImplementedError("stage5: implement batchify")


@app.function
def tbptt_batches(data, seq_len):
    """stage 5: list of (x, y) chunks of data (B, N), consecutive in time: x = data[:, i:i+L], y = data[:, i+1:i+1+L]."""
    raise NotImplementedError("stage5: implement tbptt_batches")


@app.class_definition
class CharRNN(nn.Module):
    """embedding -> your RNN -> linear: a character-level language model."""

    def __init__(self, vocab_size, hidden_size, *, num_layers=1):
        """stage 5: `embed` (nn.Embedding(V, H)), `rnn` (RNN(H, H, num_layers=...)), `head` (nn.Linear(H, V))."""
        raise NotImplementedError("stage5: implement CharRNN.__init__")

    def forward(self, idx, h=None):
        """stage 5: idx (B, T) int64 -> (logits (B, T, V), h_n)."""
        raise NotImplementedError("stage5: implement CharRNN.forward")


@app.function
def train_tbptt(model, ids, *, batch_size=16, seq_len=32, epochs=1, lr=3e-3, clip=1.0):
    """stage 5: Adam over tbptt_batches(batchify(ids)), carrying the hidden state across chunks, detached; returns chunk losses."""
    raise NotImplementedError("stage5: implement train_tbptt")


@app.cell(hide_code=True)
def _():
    checks.run(5, batchify=batchify, tbptt_batches=tbptt_batches, CharRNN=CharRNN, train_tbptt=train_tbptt, RNN=RNN)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.seq import RNN` gives you this module. topic 05 replaces its update with gated
    ones that keep gradients alive.
    """)
    return


if __name__ == "__main__":
    app.run()
