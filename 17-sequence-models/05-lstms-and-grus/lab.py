import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    K, HIDDEN, STEPS = 4, 64, 400  # symbols to remember (chance 1/4), hidden units, training steps


@app.function
def train_recall(cell, delay, *, seed=0, forget_bias=None, steps=STEPS, lr=3e-3):
    """train your SeqTagger on recall_batch for `steps` Adam steps; returns accuracy on 512 fresh sequences."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = ex.SeqTagger(2 * K, HIDDEN, cell=cell)
    if forget_bias is not None:
        with torch.no_grad():
            m.rnn.cells[0].ih.bias[HIDDEN:2 * HIDDEN] = forget_bias
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        x, y = ex.recall_batch(32, delay, K, generator=g)
        loss = nn.functional.cross_entropy(m(x).reshape(-1, 2 * K), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    x, y = ex.recall_batch(512, delay, K, generator=g)
    return ex.copy_accuracy(m, x, y)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # LSTMs & GRUs · lab

    every experiment here uses **your** `LSTMCell`, `GRUCell`, `LSTM`, `GRU`, `SeqTagger`,
    `recall_batch` and `copy_accuracy` from `exercise.py`, plus your topic-04 `RNN`, on a
    synthetic memory task: read a symbol (one of 4), then `delay` distractor symbols, then say
    which symbol came first. the only training signal is at the last step. for each experiment:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the lab takes about two minutes on an
    idle laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how long can each cell remember?

    a tagger with 64 hidden units of each kind (your `RNN`, `GRU`, `LSTM` with the forget-gate
    bias at 1), trained for 400 Adam steps on batches of 32 (learning rate $3 \times 10^{-3}$,
    gradients clipped at 1), at delays from 5 to 80. accuracy on 512 fresh sequences; chance is
    25%.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for which delays will the plain RNN recall the symbol? the LSTM? the GRU? (each gets the same 400 steps)"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    _delays = [5, 10, 20, 40, 80]
    _res = {c: [train_recall(c, d) for d in _delays] for c in ["rnn", "gru", "lstm"]}
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _c, _a in _res.items():
        _ax.plot(_delays, _a, "o-", label=_c.upper())
    _ax.axhline(1 / K, color="gray", ls="--", lw=0.8, label="chance")
    _ax.set(xscale="log", xticks=_delays, xticklabels=[str(d) for d in _delays], xlabel="delay (distractors)",
            ylabel="recall accuracy", ylim=(0, 1.05))
    _ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(f"{time.perf_counter() - _t0:.0f} s\n\n| cell | " + " | ".join(f"delay {d}" for d in _delays)
                           + " |\n|---|" + "---|" * len(_delays) + "\n"
                           + "\n".join(f"| {c.upper()} | " + " | ".join(f"{a:.2f}" for a in v) + " |"
                                       for c, v in _res.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the RNN fails even at 5 steps, though topic 04 showed it *can* carry
    information that far. what does it need to learn, and where does the gradient that would
    teach it come from? which path through the LSTM and the GRU lets that gradient arrive
    almost undamaged? the LSTM stops somewhere between 20 and 40: what is the gradient along
    its cell state multiplied by at every step, at initialization?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the forget-gate bias

    the same LSTM, with the forget gate's bias initialized at 0 (PyTorch's default for
    `nn.LSTM`, up to its small random init), at 1 (yours), and at 3, at delays 20, 40 and 80.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "it's one number per unit, set once at initialization, and training can change it freely. how much "
        "difference will the forget bias (0, 1 or 3) make to the longest delay the LSTM learns?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _t0 = time.perf_counter()
    _rows = []
    for _fb in [0.0, 1.0, 3.0]:
        _cells = []
        for _d in [20, 40, 80]:
            _cells.append(f"{train_recall('lstm', _d, forget_bias=_fb):.2f}")
        _rows.append(f"| {_fb:g} | σ = {1 / (1 + np.exp(-_fb)):.2f} | " + " | ".join(_cells) + " |")
    mo.md(f"{time.perf_counter() - _t0:.0f} s\n\n| forget bias | initial forget gate | delay 20 | delay 40 | delay 80 |\n"
          "|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with forget gate $f$ the cell state's gradient is multiplied by about $f$
    per step. compute $f^{d}$ at initialization for each bias and delay. why does an
    initialization matter so much when training could, in principle, raise the bias itself?
    (what gradient would tell it to?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the gradient's path, measured

    at initialization, before any training: how strongly does the final hidden state depend on
    the *first* input? for sequences of length $T$, we measure
    $\|\partial(\sum h_T)/\partial e_1\|$, the gradient with respect to the first input's
    embedding vector, for each cell (forget bias 1 for the LSTM), averaged over 64 random
    sequences.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "at initialization, by what factor does the first input's influence on h_T fall between T = 1 and T = 50, "
        "for the RNN? for the LSTM and the GRU?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _Ts = [1, 2, 5, 10, 20, 50]
    _res = {}
    for _c in ["rnn", "gru", "lstm"]:
        torch.manual_seed(0)
        _m = ex.SeqTagger(2 * K, HIDDEN, cell=_c)
        _vals = []
        for _T in _Ts:
            _x, _ = ex.recall_batch(64, _T - 1, K, generator=torch.Generator().manual_seed(1))
            _e = _m.embed(_x).detach().requires_grad_(True)
            _out, _ = _m.rnn(_e)
            (_g,) = torch.autograd.grad(_out[:, -1].sum(), _e)
            _vals.append(_g[:, 0].norm(dim=1).mean().item())
        _res[_c] = np.array(_vals)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _c, _v in _res.items():
        _ax.semilogy(_Ts, _v, "o-", label=_c.upper())
    _ax.set(xlabel="sequence length T", ylabel="‖∂ sum(h_T) / ∂ e_1‖", xscale="log", xticks=_Ts,
            xticklabels=[str(t) for t in _Ts])
    _ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| cell | " + " | ".join(f"T = {t}" for t in _Ts) + " |\n|---|" + "---|" * len(_Ts) + "\n"
                           + "\n".join(f"| {c.upper()} | " + " | ".join(f"{x:.1e}" for x in v) + " |"
                                       for c, v in _res.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** relate each curve's slope to a per-step factor: for the RNN,
    $\operatorname{diag}(1 - h^2)W_{hh}$; for the LSTM, roughly the forget gate; for the GRU, the
    update gate $z$. which of these factors is a *product of a matrix and a derivative*, and
    which is a *gate value* that can sit close to 1? does the GRU's curve explain why it beat
    the LSTM in experiment 1?
    """)
    return


if __name__ == "__main__":
    app.run()
