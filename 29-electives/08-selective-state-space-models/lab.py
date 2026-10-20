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

    from mlfs import notebooks, ui
    from mlfs.torchnn import RMSNorm, count_parameters  # yours, from chapter 15
    from mlfs.transformer import TransformerBlock, alibi_bias, sinusoidal_positions  # yours, from chapter 18

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    # your S4D layer from topic 07, the time-invariant baseline (electives own no mlfs module, so load it directly)
    def _s4d_smoke(m):  # raises NotImplementedError if a stage you need here isn't done yet
        m.S4DLayer(2, 2)(torch.zeros(1, 3, 2))

    prev = notebooks.load_topic("07-linear-state-space-models-s4", key="s4", smoke=_s4d_smoke)
    s4 = prev.module
    torch.set_num_threads(1)

    VOCAB = 16  # token 1 is the trigger; tokens 2..15 are data

    def induction_batch(L, batch, generator):
        """single-query induction: random data tokens, the trigger once at a random position p followed by a target,
        and the trigger again at the last position. the answer is the token that followed the first trigger."""
        x = torch.randint(2, VOCAB, (batch, L), generator=generator)
        p = torch.randint(0, L - 2, (batch,), generator=generator)
        rows = torch.arange(batch)
        x[rows, p] = 1
        y = x[rows, p + 1].clone()
        x[:, -1] = 1
        return x, y

    class MambaLM(nn.Module):
        def __init__(self, vocab, d_model=16, n_layers=2, d_state=8):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.blocks = nn.ModuleList([ex.MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab)

        def forward(self, idx):
            h = self.emb(idx)
            for b in self.blocks:
                h = b(h)
            return self.head(self.norm(h))

    class S4DLM(nn.Module):
        """topic 07's block: x + GLU(Linear(GELU(S4D(LayerNorm(x)))))."""

        def __init__(self, vocab, d_model=32, n_layers=2, d_state=16):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.blocks = nn.ModuleList([s4.S4DLayer(d_model, d_state) for _ in range(n_layers)])
            self.outs = nn.ModuleList([nn.Linear(d_model, 2 * d_model) for _ in range(n_layers)])
            self.head = nn.Linear(d_model, vocab)

        def forward(self, idx):
            h = self.emb(idx)
            for norm, ssm, out in zip(self.norms, self.blocks, self.outs):
                h = h + F.glu(out(F.gelu(ssm(norm(h)))), dim=-1)
            return self.head(h)

    class TinyTransformer(nn.Module):
        """pre-LN causal transformer from your chapter 18 blocks. pos='sinusoidal': Vaswani et al.'s recipe (token
        embeddings initialized with std d^-1/2 and scaled by sqrt(d), plus sinusoids); pos='alibi': no position
        embedding, ALiBi attention biases."""

        def __init__(self, vocab, d_model=32, n_layers=2, n_heads=4, pos="alibi"):
            super().__init__()
            self.pos, self.d, self.n_heads = pos, d_model, n_heads
            self.emb = nn.Embedding(vocab, d_model)
            if pos == "sinusoidal":  # Vaswani et al.: embeddings of std d^-1/2, times sqrt(d) = unit std, like the sinusoids
                nn.init.normal_(self.emb.weight, std=d_model ** -0.5)
            self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab)

        def forward(self, idx):
            T = idx.shape[1]
            h = self.emb(idx)
            if self.pos == "sinusoidal":
                h = h * math.sqrt(self.d) + sinusoidal_positions(T, self.d)
                mask, causal = None, True
            else:
                mask, causal = alibi_bias(self.n_heads, T), False  # the bias is -inf above the diagonal: causal
            for b in self.blocks:
                h = b(h, mask=mask, is_causal=causal)
            return self.head(self.norm(h))

    MODELS = {
        "Mamba (yours, topic 08)": lambda: MambaLM(VOCAB),
        "S4D (yours, topic 07)": lambda: S4DLM(VOCAB),
        "transformer, ALiBi": lambda: TinyTransformer(VOCAB, pos="alibi"),
        "transformer, sinusoidal": lambda: TinyTransformer(VOCAB, pos="sinusoidal"),
    }
    L_TRAIN, STEPS, BATCH, LR, SEEDS = 32, 1000, 32, 3e-3, (0, 1, 2)
    EVAL_LENGTHS = (32, 64, 128, 256, 512)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # selective state-space models · lab

    every experiment runs **your** code: your scans and Mamba block from this topic, your S4D
    layer from topic 07, and transformers built from your chapter 18 blocks. for each one:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about two to
    four minutes on one CPU core depending on the machine and its load, almost all of it in
    the project.
    """)
    return


@app.cell
def _():
    prev.callout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what the parallel scan buys on a CPU

    your `sequential_scan` and `parallel_scan` on the shape a selective scan produces,
    `(batch 4, L, D 32, N 8)`, forward only, for growing $L$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the parallel scan does about log2(L) times MORE arithmetic than the loop. at L = 4096, is it faster or "
        "slower on this CPU, and by how much?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rows, _ts = [], {"sequential": [], "parallel": []}
    _Ls = [64, 256, 1024, 2048]
    g = torch.Generator().manual_seed(0)
    for _L in _Ls:
        _a = torch.rand(4, _L, 32, 8, generator=g) * 0.2 + 0.8
        _b = torch.randn(4, _L, 32, 8, generator=g)
        _res = {}
        for _name, _scan in [("sequential", ex.sequential_scan), ("parallel", ex.parallel_scan)]:
            _best = float("inf")
            for _ in range(3):
                _t0 = time.perf_counter()
                _res[_name] = _scan(_a, _b)
                _best = min(_best, time.perf_counter() - _t0)
            _ts[_name].append(_best)
        _err = (_res["sequential"] - _res["parallel"]).abs().max().item()
        _rows.append(f"| {_L} | {_ts['sequential'][-1] * 1e3:.1f} | {_ts['parallel'][-1] * 1e3:.1f} | "
                     f"{_ts['sequential'][-1] / _ts['parallel'][-1]:.1f}× | {math.ceil(math.log2(_L))} | {_err:.1e} |")
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    for _name, _t in _ts.items():
        _ax.plot(_Ls, _t, "o-", label=_name)
    _ax.set(xscale="log", yscale="log", xlabel="sequence length L", ylabel="seconds (best of 3)",
            title="scan of a (4, L, 32, 8) tensor")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| L | sequential (ms) | parallel (ms) | speed-up | rounds | max difference |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on this CPU the parallel scan is *slower* than the loop at every length.
    count what each version moves through memory: one loop iteration touches one
    `(4, 32, 8)` slice, one Hillis–Steele round reads and writes the whole `(4, L, 32, 8)`
    tensor several times, and there are $\log_2L$ rounds. why does fewer sequential steps not
    help when there are only a few cores, and what kind of hardware (and implementation)
    would turn the $O(\log L)$ depth into a real speed-up? the two results are not exactly
    equal: why?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2 (the project): selective SSM vs transformer on long sequences

    the task is **single-query induction** (Gu & Dao, 2023, §4.1.2; Olsson et al., 2022): a
    sequence of random tokens contains a trigger token once, at a random position, followed by
    a target token; the trigger appears again at the very end, and the model must output the
    target. it needs content-based retrieval (find the earlier trigger, wherever it is) and
    nothing else, so it isolates the ability that separates attention and selection from a
    time-invariant filter. chance is 1/14 ≈ 0.071.

    four models, each about 6–26k parameters, trained the same way: {STEPS} AdamW steps (lr
    {LR}, batch {BATCH}, loss on the last position only) on sequences of length
    **{L_TRAIN}**, seeds {', '.join(map(str, SEEDS))}; then tested, without any further
    training, on 128 fresh sequences of each length {', '.join(map(str, EVAL_LENGTHS))} (up to
    16× the training length). the contestants:

    - **your Mamba** (2 blocks, $d = 16$, $N = 8$);
    - **your S4D** from topic 07 (2 blocks, $d = 32$, $N = 16$): time-invariant, the same family without selection;
    - a **transformer with ALiBi** (2 layers, 4 heads, $d = 32$; Press et al., 2022), the position scheme designed for train-short-test-long;
    - a **transformer with sinusoidal positions** (same size; Vaswani et al.'s recipe).

    everything is synthetic, so the numbers are about this toy task at this tiny scale,
    with this training budget; they say nothing direct about language modelling.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for each model: accuracy at the training length (32) and at 16x the training length (512)? which model is "
        "fastest to train per step?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _train(make, seed):
        torch.manual_seed(seed)
        model = make()
        g = torch.Generator().manual_seed(1000 + seed)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            x, y = induction_batch(L_TRAIN, BATCH, g)
            loss = F.cross_entropy(model(x)[:, -1], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        secs = time.perf_counter() - t0
        model.eval()
        accs = []
        for L in EVAL_LENGTHS:
            x, y = induction_batch(L, 128, torch.Generator().manual_seed(77 + L))
            with torch.no_grad():
                pred = torch.cat([model(x[i:i + 32])[:, -1].argmax(-1) for i in range(0, 128, 32)])
            accs.append((pred == y).float().mean().item())
        return model, np.array(accs), secs

    project = {name: [_train(make, s) for s in SEEDS] for name, make in MODELS.items()}
    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _rows = []
    for _i, (_name, _runs) in enumerate(project.items()):
        _acc = np.array([r[1] for r in _runs])
        _ax.plot(EVAL_LENGTHS, _acc.mean(0), "o-", color=f"C{_i}", label=_name)
        for _a in _acc:
            _ax.plot(EVAL_LENGTHS, _a, ":", color=f"C{_i}", lw=0.8, alpha=0.6)
        for _s, (_m, _a, _secs) in zip(SEEDS, _runs):
            _rows.append(f"| {_name} | {_s} | " + " | ".join(f"{v:.2f}" for v in _a) + f" | {_secs:.0f} |")
    _ax.axhline(1 / 14, color="k", ls=":", lw=1, label="chance")
    _ax.axvline(L_TRAIN, color="0.7", lw=1)
    _ax.set(xscale="log", xlabel="test sequence length (trained at 32)", ylabel="accuracy", ylim=(0, 1.05),
            title="single-query induction: mean over seeds (dotted: each seed)")
    _ax.set_xticks(EVAL_LENGTHS, [str(L) for L in EVAL_LENGTHS])
    _ax.minorticks_off()
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _params = " · ".join(f"{k}: {count_parameters(v[0][0]):,}" for k, v in project.items())
    mo.vstack([_fig, mo.md("| model | seed | " + " | ".join(f"L = {L}" for L in EVAL_LENGTHS)
                           + " | train seconds (machine-dependent) |\n|---|---|" + "---|" * (len(EVAL_LENGTHS) + 1) + "\n"
                           + "\n".join(_rows) + f"\n\nparameters: {_params}")])
    return (project,)


@app.cell
def _(p2, project):
    ui.gate(p2)
    _rows = []
    for _name, _runs in project.items():
        _m = _runs[0][0]
        _cells = []
        for _L in (256, 1024):
            _x, _ = induction_batch(_L, 1, torch.Generator().manual_seed(5))
            with torch.no_grad():
                _m(_x)
                _best = float("inf")
                for _ in range(3):
                    _t0 = time.perf_counter()
                    _m(_x)
                    _best = min(_best, time.perf_counter() - _t0)
                _cells.append(_best)
        _rows.append(f"| {_name} | {_cells[0] * 1e3:.0f} | {_cells[1] * 1e3:.0f} | {_cells[1] / _cells[0]:.1f}× |")
    mo.md("**inference cost** (one sequence, one forward pass of the trained seed-0 model, best of 3, ms; these "
          "timings depend on the machine and its load, so compare ratios, not absolute numbers):\n\n"
          "| model | L = 256 | L = 1024 | ratio (4× longer) |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the S4D model has the same recurrence as Mamba and more parameters, yet
    it never works beyond the training length (and on most seeds not even at it): why can't
    a time-invariant kernel find "the token after the earlier trigger" wherever it is, and
    what might one seed have found that works at exactly $L = 32$? the ALiBi transformer is
    as good as Mamba at the training length: what happens to its attention as the sequence
    grows, and why does the linear distance penalty matter for a trigger that is now hundreds
    of tokens away? the sinusoidal transformer solves the task at the training length but
    collapses beyond it: what does it see at positions it never trained on? finally, what
    *can't* you conclude from this experiment (think of model size, training budget, task,
    and the copying argument from the lesson's on paper 4)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what did selection learn?

    unroll the selective recurrence and the output at the last position $L$ is a weighted
    sum over all earlier positions $s$, per channel $d$:

    $$y_{L,d} = \sum_{s \le L}\alpha_{L,s,d}\,u_{s,d}, \qquad \alpha_{L,s,d} = \sum_nC_{L,n}\,
    \exp\Big(A_{d,n}\sum_{s < r \le L}\Delta_{r,d}\Big)\,\Delta_{s,d}B_{s,n}.$$

    for a time-invariant SSM, $\alpha_{L,s}$ would be the kernel $K_{L-s}$, the same for every
    input; here it depends on the content, like an attention row (Ali, Zimerman & Wolf, 2024,
    call it Mamba's hidden attention). we open up the trained Mamba (seed 0), run it on one
    length-256 test sequence, and plot $|\alpha_{L,s}|$ (averaged over channels) for both of
    its selective layers.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for the last position of a length-256 sequence, which earlier positions will each layer's row put its weight "
        "on?"
    )
    p3
    return (p3,)


@app.cell
def _(p3, project):
    ui.gate(p3)
    _model = project["Mamba (yours, topic 08)"][0][0]
    _x, _y = induction_batch(256, 1, torch.Generator().manual_seed(11))
    _inputs = []
    _handles = [b.ssm.register_forward_hook(lambda mod, args, out: _inputs.append((mod, args[0][0].detach())))
                for b in _model.blocks]
    with torch.no_grad():
        _pred = _model(_x)[0, -1].argmax().item()
    for _h in _handles:
        _h.remove()
    _p = int((_x[0] == 1).nonzero()[0])
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    _rows = []
    for _i, (_ax, (_mod, _u)) in enumerate(zip(_axes, _inputs)):
        with torch.no_grad():
            _d, _B, _C = (t[0] for t in _mod.params(_u[None]))
            _A = -torch.exp(_mod.A_log)
            _later = torch.flip(torch.cumsum(torch.flip(_d, [0]), 0), [0]) - _d  # sum of delta over r > s
            _alpha = torch.einsum("n,sdn,sd,sn->sd", _C[-1], torch.exp(_A[None] * _later[..., None]), _d, _B)
        _a = _alpha.abs().mean(1).numpy()
        _ax.plot(_a, lw=0.8)
        _ax.plot([_p], [1.0], "v", color="k", ms=9, transform=_ax.get_xaxis_transform(), clip_on=False,
                 label=f"first trigger ({_p})")
        _ax.plot([_p + 1], [0.0], "^", color="tab:green", ms=9, transform=_ax.get_xaxis_transform(), clip_on=False,
                 label=f"target ({_p + 1})")
        _ax.set(ylabel=f"layer {_i + 1}: |α_L,s|", yscale="log")
        _top = np.argsort(-_a)[:3]
        _rows.append(f"| {_i + 1} | {', '.join(str(t) for t in _top)} | {_a[_top[0]]:.3g} | {np.median(_a):.3g} |")
    _axes[0].legend(fontsize=8, loc="upper left")
    _axes[1].set_xlabel("position s (the query is the last position, 255; markers: black = first trigger, green = target)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"prediction {_pred}, answer {_y.item()}; the first trigger is at {_p}, the target at "
                           f"{_p + 1}.\n\n| layer | top 3 positions | largest weight | median weight |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** layer 1's row looks only a few positions back, layer 2's row puts almost
    all its weight in one place: where exactly, relative to the target, and why there rather
    than *at* the target (think about the width-4 convolution in front of each SSM, and what
    the SSM input at a position "knows" about the previous tokens)? how do $B_s$ (written at
    position $s$) and $C_L$ (read at the end) have to line up for the weight to be large,
    given that the trigger token is what both positions have in common? why does this
    mechanism keep working at 16× the training length, when each extra distractor adds only
    a tiny weight?
    """)
    return


if __name__ == "__main__":
    app.run()
