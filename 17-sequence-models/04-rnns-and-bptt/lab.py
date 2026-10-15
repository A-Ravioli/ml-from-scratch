import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    import numpy as np
    import torch

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    RHOS = [0.5, 0.9, 1.0, 1.5, 3.0]


@app.function
def rnn_with_radius(rho, nonlinearity="tanh", *, input_size=16, hidden_size=64, seed=0):
    """a one-layer RNN whose recurrent matrix is rho times a random orthogonal matrix (every singular value = rho)."""
    g = torch.Generator().manual_seed(seed)
    m = ex.RNN(input_size, hidden_size, nonlinearity=nonlinearity)
    q, _ = torch.linalg.qr(torch.randn(hidden_size, hidden_size, generator=g))
    with torch.no_grad():
        m.hh[0].weight.copy_(rho * q)
        m.hh[0].bias.zero_()
        m.ih[0].bias.zero_()
    return m


@app.function
def copy_stream(n_periods, k, vocab, rng):
    """k random symbols, then the same k again, repeated: the second half of each period needs a memory of k steps."""
    out = []
    for _ in range(n_periods):
        r = list(rng.integers(0, vocab, size=k))
        out += r + r
    return np.array(out)


@app.function
def copy_accuracy(model, k, vocab, rng):
    """next-symbol accuracy on the copy halves of a fresh stream (chance: 1 / vocab)."""
    ids = torch.as_tensor(copy_stream(200, k, vocab, rng))[None]
    with torch.no_grad():
        logits, _ = model(ids[:, :-1])
    pos = np.arange(1, ids.shape[1])
    mask = torch.as_tensor(pos % (2 * k) >= k)
    return (logits.argmax(-1)[0][mask] == ids[0, 1:][mask]).float().mean().item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RNNs & BPTT · lab

    every experiment here uses **your** `RNN`, `hidden_grad_norms`, `CharRNN`, `batchify`,
    `tbptt_batches` and `train_tbptt` from `exercise.py`, on synthetic data. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the lab takes about 40 seconds on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how far back does the gradient reach?

    a one-layer RNN (16 inputs, 64 hidden units) whose recurrent matrix is $\rho$ times a random
    orthogonal matrix, so every singular value of $W_{hh}$ is exactly $\rho$. biases are zero, the
    inputs are standard normal, 50 steps, a batch of 32. `hidden_grad_norms` measures
    $\|\partial L/\partial h_t\|$ for $L = \sum h_{50}$: how strongly the last step's loss
    depends on each earlier hidden state.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with tanh units and rho = 1 (the recurrent matrix preserves lengths exactly), how large is the gradient at "
        "step 1 relative to step 50? and at rho = 3: does tanh's saturation keep the gradient from exploding?"
    )
    p1
    return (p1,)


@app.cell
def _():
    rho = mo.ui.slider(0.25, 4.0, step=0.25, value=1.0, label="ρ for the extra curve")
    act = mo.ui.dropdown(["tanh", "relu"], value="tanh", label="nonlinearity")
    mo.hstack([rho, act])
    return act, rho


@app.cell
def _(act, p1, rho):
    ui.gate(p1)
    torch.manual_seed(0)
    _x = torch.randn(32, 50, 16)
    _res = {}
    for _nl in ["tanh", "relu"]:
        for _r in RHOS:
            _res[(_nl, _r)] = np.asarray(ex.hidden_grad_norms(rnn_with_radius(_r, _nl), _x))
    _extra = np.asarray(ex.hidden_grad_norms(rnn_with_radius(rho.value, act.value), _x))
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for _ax, _nl in zip(_axes, ["tanh", "relu"]):
        for _i, _r in enumerate(RHOS):
            _ax.semilogy(range(1, 51), np.maximum(_res[(_nl, _r)], 1e-30), color=f"C{_i}", label=f"ρ = {_r}")
        if act.value == _nl:
            _ax.semilogy(range(1, 51), np.maximum(_extra, 1e-30), "k--", label=f"ρ = {rho.value} (slider)")
        _ax.set(title=_nl, xlabel="time step t", ylim=(1e-30, 1e20))
        _ax.legend(fontsize=7)
    _axes[0].set_ylabel("‖∂L/∂h_t‖ (L = sum of h_50)")
    _rows = [f"| {nl} | {r} | " + " | ".join(f"{_res[(nl, r)][t]:.2e}" for t in [0, 24, 39, 48]) + " |"
             for nl in ["tanh", "relu"] for r in RHOS]
    mo.vstack([_fig, mo.md("| nonlinearity | ρ | t = 1 | t = 25 | t = 40 | t = 49 |\n|---|---|---|---|---|---|\n"
                           + "\n".join(_rows) + "\n\nat t = 50 the norm is always √64 = 8.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the gradient at step $t$ is the product of the Jacobians
    $\partial h_{s}/\partial h_{s-1} = \operatorname{diag}(\phi'(a_s))W_{hh}$ for
    $s = t+1, \dots, 50$. with $\rho = 1$ the matrix part preserves lengths exactly: what does
    the $\operatorname{diag}(\phi')$ part do, for tanh and for ReLU? why does $\rho = 3$ explode
    even though many tanh units are saturated (what happens to the units that *aren't*)? what
    range of $\rho$ would you need for the gradient to survive 50 steps, and why is that window
    so hard to stay in during training?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: truncation length against memory

    a stream of blocks: 6 random symbols (from 8), then the same 6 again, then 6 new ones, and
    so on. the first half of each block is unpredictable; the second half is perfectly
    predictable *if* the model remembers what it saw 6 steps earlier. your `CharRNN` (64 hidden
    units) is trained with your `train_tbptt` (15 epochs, batches of 32 streams) at different
    truncation lengths, and we measure its accuracy on the copy halves of fresh data (chance is
    1/8 = 12.5%). the hidden state is carried across chunks every time, so the *forward* pass
    can always see far back; only the gradient is cut.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the dependency spans 6 steps. how accurate is the model when truncated BPTT backpropagates only 3 steps? "
        "5? 6? 24? will the curve rise smoothly, or jump?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _t0 = time.perf_counter()
    _k, _V = 6, 8
    _lens = [3, 5, 6, 12, 24]
    _acc = {L: [] for L in _lens}
    for _L in _lens:
        for _s in range(2):
            torch.manual_seed(_s)
            _m = ex.CharRNN(_V, 64)
            ex.train_tbptt(_m, copy_stream(1500, _k, _V, np.random.default_rng(_s)), batch_size=32, seq_len=_L,
                           epochs=15, lr=3e-3)
            _acc[_L].append(copy_accuracy(_m, _k, _V, np.random.default_rng(100 + _s)))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.plot(_lens, [np.mean(_acc[L]) for L in _lens], "o-", label="copy-half accuracy (mean of 2 seeds)")
    _ax.axvline(_k, color="gray", ls=":", label="the dependency length, 6")
    _ax.axhline(1 / _V, color="C3", ls="--", lw=0.8, label="chance")
    _ax.set(xlabel="truncation length (seq_len)", ylabel="accuracy on the copy halves", ylim=(0, 1), xscale="log",
            xticks=_lens, xticklabels=[str(L) for L in _lens])
    _ax.xaxis.set_minor_formatter(NullFormatter())
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(f"{time.perf_counter() - _t0:.0f} s\n\n| seq_len | accuracy (2 seeds) |\n|---|---|\n"
                           + "\n".join(f"| {L} | {', '.join(f'{a:.2f}' for a in _acc[L])} |" for L in _lens))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can't the model learn the copy when the truncation is shorter than 6,
    even though the carried state does contain information about the symbols 6 steps back?
    (where would the gradient that teaches it to *store* a symbol have to flow?) why is it still
    above chance at length 3? truncated BPTT trades what for what, and how would you choose the
    truncation length for a character-level model of Shakespeare?
    """)
    return


if __name__ == "__main__":
    app.run()
