import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from numpy.polynomial import legendre
    from torch import nn

    from mlfs import notebooks, ui
    from mlfs.torchnn import count_parameters  # yours, from chapter 15

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(2)

    class S4DBlock(nn.Module):
        """pre-norm residual block: x + GLU(Linear(GELU(S4D(LayerNorm(x)))))."""

        def __init__(self, d_model, d_state):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.ssm = ex.S4DLayer(d_model, d_state)
            self.out = nn.Linear(d_model, 2 * d_model)

        def forward(self, x):
            return x + F.glu(self.out(F.gelu(self.ssm(self.norm(x)))), dim=-1)

    class S4DClassifier(nn.Module):
        def __init__(self, vocab, n_classes, d_model=32, d_state=16, n_layers=2):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.blocks = nn.Sequential(*[S4DBlock(d_model, d_state) for _ in range(n_layers)])
            self.head = nn.Linear(d_model, n_classes)

        def forward(self, x):
            return self.head(self.blocks(self.emb(x))[:, -1])

    class LSTMClassifier(nn.Module):
        """nn.LSTM (unlocked by your chapter 17 LSTM), with the forget-gate bias initialized high (a standard fix)."""

        def __init__(self, vocab, n_classes, d_model=32):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
            with torch.no_grad():  # gate order i, f, g, o: forget-gate bias 1 in each of the two bias vectors
                self.lstm.bias_ih_l0[d_model:2 * d_model].fill_(1.0)
                self.lstm.bias_hh_l0[d_model:2 * d_model].fill_(1.0)
            self.head = nn.Linear(d_model, n_classes)

        def forward(self, x):
            return self.head(self.lstm(self.emb(x))[0][:, -1])

    def first_token_batch(L, batch, generator):
        """remember-the-first-token: a class token (0-7) at position 0, then L - 1 random distractors (8-15)."""
        y = torch.randint(0, 8, (batch,), generator=generator)
        x = torch.randint(8, 16, (batch, L), generator=generator)
        x[:, 0] = y
        return x, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # linear state-space models (S4) · lab

    every experiment runs **your** code: your HiPPO matrix, your discretization, kernels and
    FFT convolution, and your `S4DLayer`. for each one: write down a prediction, lock it in,
    run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes one to two minutes on a CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a fixed-size memory of the whole past

    a signal of 1,000 samples arrives one sample at a time. your `hippo_legs(N)` defines an
    online update of an $N$-number state $c$ (the bilinear LegS recurrence from the lesson).
    after the last sample, we read $c$ as Legendre coefficients and reconstruct the *entire*
    history from those $N$ numbers.
    """)
    return


@app.cell
def _():
    n_slider = mo.ui.slider(2, 64, value=16, step=2, label="state size N")
    p1 = ui.predict(
        "with N = 16 numbers summarizing 1,000 samples, how well is the beginning of the signal reconstructed compared "
        "with the end? how does the error change from N = 4 to N = 64?"
    )
    mo.vstack([p1, n_slider])
    return n_slider, p1


@app.cell
def _(n_slider, p1):
    ui.gate(p1)

    def hippo_memory(f, N):
        A, B = (np.asarray(a, dtype=float) for a in ex.hippo_legs(N))
        c, I = np.zeros(N), np.eye(N)
        for k, fk in enumerate(f, start=1):
            c = np.linalg.solve(I - A / (2 * k), (I + A / (2 * k)) @ c + B * fk / k)
        x = 2 * (np.arange(len(f)) + 0.5) / len(f) - 1
        return legendre.legval(x, c * np.sqrt(2 * np.arange(N) + 1))

    _t = np.arange(1000) / 1000
    _sig = np.sin(6 * _t) + 0.5 * np.cos(17 * _t ** 2) + 0.3 * (_t > 0.5)
    _rows = []
    for _N in [4, 8, 16, 32, 64]:
        _r = hippo_memory(_sig, _N)
        _rows.append(f"| {_N} | {np.sqrt(np.mean((_r - _sig) ** 2)):.3f} | "
                     f"{np.sqrt(np.mean((_r[:100] - _sig[:100]) ** 2)):.3f} | {np.sqrt(np.mean((_r[-100:] - _sig[-100:]) ** 2)):.3f} |")
    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.plot(_t * 1000, _sig, "k", lw=1.5, label="signal (all 1,000 samples)")
    _ax.plot(_t * 1000, hippo_memory(_sig, n_slider.value), "C3", lw=1, label=f"reconstructed from N = {n_slider.value} numbers")
    _ax.set(xlabel="time step", title="what the HiPPO-LegS state remembers at step 1,000")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| N | RMS error, all | first 100 steps | last 100 steps |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does LegS remember the start of the signal as well as the end (what
    measure over the past does it project onto, and how does that differ from an
    exponentially decaying memory)? what limits the reconstruction near the jump at step 500,
    whatever $N$ is? why does the error stop improving (or get slightly worse) at large $N$
    with this discretization?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what the kernel remembers, and what the two views cost

    left: kernels of a single-channel `S4DLayer` (64 modes, $C$ fixed to 1 so you see the
    modes, not a random mix) for three step sizes $\Delta$. right: one forward pass of a
    layer with $H = 64$ channels and $N = 64$ modes, batch 4, computed by your FFT
    convolution and by your step-by-step recurrence, for growing $L$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "sketch |K| against the lag for an S4D-Lin kernel with dt = 0.01 out to lag 3,000: a smooth exponential decay, "
        "or something else? at L = 2048, how many times faster is the convolution than the recurrence?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _fig = plt.figure(figsize=(12, 6))
    _gs = _fig.add_gridspec(3, 2)
    _kax = [_fig.add_subplot(_gs[_i, 0]) for _i in range(3)]
    _tax = _fig.add_subplot(_gs[:, 1])
    _L = 3000
    _rows = []
    for _ki, _dt in enumerate([1e-3, 1e-2, 1e-1]):
        _layer = ex.S4DLayer(1, 64, dt_min=_dt, dt_max=_dt * (1 + 1e-6))
        with torch.no_grad():
            _layer.C.zero_()
            _layer.C[..., 0] = 1.0
            _k = _layer.kernel(_L)[0].numpy()
        _a = np.abs(_k) / np.abs(_k).max()
        _kax[_ki].plot(_a, lw=0.6, color=f"C{_ki}")
        _kax[_ki].set(yscale="log", ylim=(1e-4, 1.5), ylabel="|K|")
        _kax[_ki].set_title(f"S4D-Lin kernel, dt = {_dt:g} (normalized to lag 0)", fontsize=9)
        _P = int(round(2 / _dt))  # S4D-Lin frequencies are pi * n * dt: every mode is back in phase after 2 / dt steps
        _lo, _hi = _P // 2, min(_L, 3 * _P // 2)
        _peak = _lo + int(np.argmax(_a[_lo:_hi]))
        _between = np.median(_a[_P // 4: 3 * _P // 4])
        _rows.append(f"| {_dt:g} | {_peak} | {_a[_peak]:.3f} | {np.exp(-0.5 * _dt * _peak):.3f} | {_between:.4f} |")
    _kax[-1].set_xlabel("lag")
    _layer = ex.S4DLayer(64, 64)
    _times = {"convolution (forward)": [], "recurrence (step loop)": []}
    _Ls = [256, 1024, 2048]
    with torch.no_grad():
        for _Lx in _Ls:
            _u = torch.randn(4, _Lx, 64)
            _best = float("inf")
            for _ in range(3):
                _t0 = time.perf_counter()
                _layer(_u)
                _best = min(_best, time.perf_counter() - _t0)
            _times["convolution (forward)"].append(_best)
            _best = float("inf")
            for _ in range(2):
                _t0 = time.perf_counter()
                _s = None
                for _k in range(_Lx):
                    _, _s = _layer.step(_u[:, _k], _s)
                _best = min(_best, time.perf_counter() - _t0)
            _times["recurrence (step loop)"].append(_best)
    for _name, _ts in _times.items():
        _tax.plot(_Ls, _ts, "o-", label=_name)
    _tax.set(xscale="log", yscale="log", xlabel="sequence length L", ylabel="seconds (best of 2-3)", title="one forward pass")
    _tax.set_xticks(_Ls, [str(L) for L in _Ls])
    _tax.minorticks_off()
    _tax.legend(fontsize=8)
    _fig.tight_layout()
    _speed = " · ".join(f"L = {L}: {r / c:.0f}×" for L, c, r in zip(_Ls, *_times.values()))
    mo.vstack([_fig, mo.md("| dt | lag of the first revival | its height (lag 0 = 1) | exp(-dt lag / 2) | median height between lag 0 and the revival |\n|---|---|---|---|---|\n"
                           + "\n".join(_rows) + f"\n\nconvolution speed-up over the step loop: {_speed}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** S4D-Lin's modes have frequencies $\pi n$, so after discretization mode $n$
    rotates by $\pi n\Delta$ per step. why does that make the kernel come back at lag
    $2/\Delta$, and why is the revival's height $e^{-1}$ whatever $\Delta$ is? what does
    that say about the window of lags a channel can shape freely (think of the modes as a
    Fourier basis on that window) and about the role of $N$ versus $\Delta$? why is the
    layer initialized with a *range* of $\Delta$ across channels? the recurrence does
    $O(LN)$ work per channel and so does the kernel computation, yet the convolution is
    faster on this machine: where does the difference come from? when would you use the
    recurrent mode anyway?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: long-range memory against an LSTM

    the task: a class token (one of 8) at position 0, then $L - 1$ random distractor
    tokens; predict the class from the output at the *last* position. nothing in between
    carries information, so the model must hold one token in memory across the whole
    sequence. two models with about the same number of parameters: two of your S4D blocks
    ($H = 32$, $N = 16$), and an LSTM with 32 hidden units (PyTorch's `nn.LSTM`, unlocked
    by your chapter 17 LSTM, with the forget-gate bias initialized high, which is standard
    practice and helps it a lot here). each trains for 300 Adam steps (lr 3e-3, batch 32) at
    each length, from 3 seeds, and is tested on 512 fresh sequences. chance is 0.125.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "test accuracy of each model at L = 16, 64 and 256 after 300 steps? at which length does each one break?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _train(make, L, seed, steps=300):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(1000 + seed)
        model = make()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        t0 = time.perf_counter()
        for _ in range(steps):
            x, y = first_token_batch(L, 32, g)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        secs = time.perf_counter() - t0
        x, y = first_token_batch(L, 512, torch.Generator().manual_seed(99))
        with torch.no_grad():
            return (model(x).argmax(1) == y).float().mean().item(), secs

    _makers = {"S4D (2 blocks)": lambda: S4DClassifier(16, 8), "LSTM": lambda: LSTMClassifier(16, 8)}
    _lengths, _seeds = [16, 64, 256], [0, 1, 2]
    results = {name: {L: [_train(m, L, s) for s in _seeds] for L in _lengths} for name, m in _makers.items()}
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _rows = []
    for _i, (_name, _res) in enumerate(results.items()):
        _acc = np.array([[a for a, _ in _res[L]] for L in _lengths])
        _ax.plot(_lengths, _acc.mean(1), "o-", color=f"C{_i}", label=_name)
        for _j, _L in enumerate(_lengths):
            _ax.scatter([_L] * len(_seeds), _acc[_j], color=f"C{_i}", s=12, alpha=0.5)
            _rows.append(f"| {_name} | {_L} | {', '.join(f'{a:.2f}' for a in _acc[_j])} | "
                         f"{np.mean([t for _, t in _res[_L]]):.1f} |")
    _ax.axhline(0.125, color="k", ls=":", lw=1, label="chance")
    _ax.set(xscale="log", xlabel="sequence length L", ylabel="test accuracy", ylim=(0, 1.05),
            title="remember the first token: 300 steps, 3 seeds (dots)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _params = {k: count_parameters(m()) for k, m in _makers.items()}
    mo.vstack([_fig, mo.md("| model | L | test accuracy per seed (0, 1, 2) | train seconds per run |\n|---|---|---|---|\n"
                           + "\n".join(_rows) + "\n\nparameters: "
                           + ", ".join(f"{k} {v:,}" for k, v in _params.items()))])
    return (results,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for the LSTM to answer, the gradient of the loss at the last step has to
    reach the embedding of the first token through $L - 1$ multiplications by forget gates
    and recurrent Jacobians: what happens to it as $L$ grows, and why does initializing the
    forget gate high help only up to a point? the S4D model has a direct path: its kernel
    $K_{L-1}$ multiplies the first token's embedding straight into the last output. what does
    $K_{L-1}$ have to look like, and which channels' $\Delta$ make that possible? the S4D
    result at the longest length varies a lot across seeds: what would you need to report
    before claiming "S4D solves $L = 256$"?
    """)
    return


if __name__ == "__main__":
    app.run()
