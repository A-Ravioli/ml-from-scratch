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
    # RNNs & BPTT · build (reference solution)

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
    B, T, _ = x.shape
    hs = np.empty((B, T, W_hh.shape[0]))
    h = h0
    for t in range(T):
        h = np.tanh(x[:, t] @ W_xh.T + h @ W_hh.T + b)
        hs[:, t] = h
    return hs


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
    B, T, _ = x.shape
    dW_xh, dW_hh, db = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros(W_hh.shape[0])
    dx = np.zeros_like(x)
    dh = np.zeros_like(h0)  # gradient arriving at h_t from the future
    for t in reversed(range(T)):
        dh = dh + dhs[:, t]
        da = dh * (1.0 - hs[:, t] ** 2)  # back through tanh
        h_prev = hs[:, t - 1] if t > 0 else h0
        dW_xh += da.T @ x[:, t]
        dW_hh += da.T @ h_prev
        db += da.sum(axis=0)
        dx[:, t] = da @ W_xh
        dh = da @ W_hh  # on to h_{t-1}
    return {"dW_xh": dW_xh, "dW_hh": dW_hh, "db": db, "dh0": dh, "dx": dx}


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
        super().__init__()
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.nonlinearity = nonlinearity
        self.ih = nn.ModuleList(nn.Linear(input_size if k == 0 else hidden_size, hidden_size) for k in range(num_layers))
        self.hh = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(num_layers))

    def forward(self, x, h0=None):
        """stage 3: run every layer over the whole sequence; layer k's outputs are layer k+1's inputs."""
        B, T, _ = x.shape
        act = torch.tanh if self.nonlinearity == "tanh" else torch.relu
        if h0 is None:
            h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        layer_in, finals = x, []
        for k in range(self.num_layers):
            h, outs = h0[k], []
            pre = self.ih[k](layer_in)  # input contributions for all steps at once
            for t in range(T):
                h = act(pre[:, t] + self.hh[k](h))
                outs.append(h)
            layer_in = torch.stack(outs, dim=1)
            finals.append(h)
        return layer_in, torch.stack(finals, dim=0)


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
    B, T, _ = x.shape
    act = torch.tanh if model.nonlinearity == "tanh" else torch.relu
    h = x.new_zeros(B, model.hidden_size)
    hs = []
    pre = model.ih[0](x)
    for t in range(T):
        h = act(pre[:, t] + model.hh[0](h))
        hs.append(h)
    grads = torch.autograd.grad(hs[-1].sum(), hs)
    return np.array([g.norm(dim=1).mean().item() for g in grads])


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
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.int64)
    n = len(ids) // batch_size
    return ids[: n * batch_size].view(batch_size, n)


@app.function
def tbptt_batches(data, seq_len):
    """stage 5: list of (x, y) chunks of data (B, N), consecutive in time: x = data[:, i:i+L], y = data[:, i+1:i+1+L]."""
    out = []
    for i in range(0, data.shape[1] - 1, seq_len):
        L = min(seq_len, data.shape[1] - 1 - i)
        out.append((data[:, i:i + L], data[:, i + 1:i + 1 + L]))
    return out


@app.class_definition
class CharRNN(nn.Module):
    """embedding -> your RNN -> linear: a character-level language model."""

    def __init__(self, vocab_size, hidden_size, *, num_layers=1):
        """stage 5: `embed` (nn.Embedding(V, H)), `rnn` (RNN(H, H, num_layers=...)), `head` (nn.Linear(H, V))."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = RNN(hidden_size, hidden_size, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, h=None):
        """stage 5: idx (B, T) int64 -> (logits (B, T, V), h_n)."""
        out, h = self.rnn(self.embed(idx), h)
        return self.head(out), h


@app.function
def train_tbptt(model, ids, *, batch_size=16, seq_len=32, epochs=1, lr=3e-3, clip=1.0):
    """stage 5: Adam over tbptt_batches(batchify(ids)), carrying the hidden state across chunks, detached; returns chunk losses."""
    data = batchify(ids, batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        h = None  # each epoch starts the streams from the beginning, with a fresh state
        for x, y in tbptt_batches(data, seq_len):
            logits, h = model(x, h)
            h = h.detach()  # keep the state's value, cut the graph: gradients stop at the chunk boundary
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())
    return losses


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
