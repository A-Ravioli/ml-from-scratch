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
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.text import CharTokenizer  # yours, from ch 17 · topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    SCHEMES = ("none", "learned", "sinusoidal", "rope", "alibi")
    TRAIN_LEN, EVAL_LEN, STEPS = 64, 512, 600


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # positional information · lab

    every experiment here runs **your** `sinusoidal_positions`, `RotaryEmbedding`,
    `alibi_bias` and `PositionalLM` from `exercise.py`. for each one: write down a prediction,
    lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the lab takes two to three minutes on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what a position "looks like" to attention

    left: the dot products $\text{PE}(m) \cdot \text{PE}(n)$ of your sinusoidal positions
    ($d = 64$) for all pairs of positions up to 128. right: the same query vector and the same key
    vector (both all-ones, $d_h = 64$), rotated by your `RotaryEmbedding` to positions $m$ and $n$,
    and their score $q_m \cdot k_n$ as a function of the distance $m - n$, up to 2,000.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the query and key are identical vectors; only their positions differ. as the distance grows from 0 to "
        "2,000, what does their RoPE score do: stay constant, oscillate, decay, or something else?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _pe = ex.sinusoidal_positions(128, 64)
    _rope = ex.RotaryEmbedding(64)
    _ones = torch.ones(1, 2048, 64)
    _rot = _rope(_ones)
    _scores = (_rot[0, :1] @ _rot[0].T)[0]  # key at position 0, query at m: depends only on m
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.4))
    _im = _axes[0].imshow((_pe @ _pe.T).numpy(), cmap="viridis")
    _axes[0].set(title="sinusoidal: PE(m) · PE(n)", xlabel="n", ylabel="m")
    plt.colorbar(_im, ax=_axes[0], fraction=0.046)
    _axes[1].plot(_scores.numpy(), lw=0.8)
    _axes[1].set(title="RoPE: q_m · k_0, q = k = ones(64)", xlabel="distance m − n", ylabel="score")
    mo.vstack([_fig, mo.md(f"score at distance 0: {_scores[0]:.1f}; mean |score| over distances 1–100: "
                           f"{_scores[1:101].abs().mean():.1f}, 500–1000: {_scores[500:1001].abs().mean():.1f}, "
                           f"1000–2000: {_scores[1000:2001].abs().mean():.1f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the sinusoidal matrix constant along its diagonals (on paper 1)?
    for RoPE, write $q_m \cdot k_n$ as a sum over channel pairs of
    $\cos((m - n)\theta_i)$ times a constant. why does such a sum start at its maximum and then
    shrink (on average) as the distance grows? which channel pairs still "see" a distance of
    2,000, and which have already gone around their circle many times?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: length extrapolation

    five copies of your `PositionalLM` (64 wide, 4 heads, 2 layers), identical except for `pos`,
    each trained for 600 AdamW steps on windows of **64 characters** of tiny Shakespeare. then
    each reads windows of **512 characters** from the validation text, and we record its loss at
    every position. positions 0–63 are what it was trained on; everything after is new.
    `learned` has no vector for position 64, so it is evaluated only up to 63.
    """)
    return


@app.cell
def _():
    text = datasets.tiny_shakespeare()
    tok = CharTokenizer().train(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    _a, _b = int(0.8 * len(ids)), int(0.9 * len(ids))
    train_ids, val_ids = ids[:_a], ids[_a:_b]
    mo.md(f"{'tiny Shakespeare' if not datasets.is_offline() else 'offline stand-in'}: {len(text):,} characters, "
          f"vocabulary {tok.vocab_size}")
    return tok, train_ids, val_ids


@app.cell
def _():
    p2 = ui.predict(
        "rank the five schemes by their loss at positions 256–511, 4 to 8 times longer than anything seen in "
        "training. does any of them get *worse* than using no positional information at all?"
    )
    p2
    return (p2,)


@app.function
def train_scheme(pos, train_ids, vocab_size, *, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = ex.PositionalLM(vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=TRAIN_LEN, pos=pos)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for step in range(steps):
        for group in opt.param_groups:  # short warmup, then cosine decay
            group["lr"] = 3e-3 * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        starts = torch.randint(0, len(train_ids) - TRAIN_LEN - 1, (32,), generator=g)
        x = torch.stack([train_ids[s:s + TRAIN_LEN] for s in starts.tolist()])
        y = torch.stack([train_ids[s + 1:s + TRAIN_LEN + 1] for s in starts.tolist()])
        loss = nn.functional.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


@app.function
def position_losses(model, val_ids, length, *, n_windows=48):
    """mean loss (nats) at each position 0..length-1 over n_windows evenly spaced validation windows."""
    starts = np.linspace(0, len(val_ids) - length - 2, n_windows).astype(int)
    x = torch.stack([val_ids[s:s + length] for s in starts])
    y = torch.stack([val_ids[s + 1:s + length + 1] for s in starts])
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    return loss.reshape(n_windows, length).mean(0).numpy()


@app.cell
def _(p2, tok, train_ids, val_ids):
    ui.gate(p2)
    _t0 = time.perf_counter()
    curves = {}
    for _pos in SCHEMES:
        _m = train_scheme(_pos, train_ids, tok.vocab_size)
        curves[_pos] = position_losses(_m, val_ids, TRAIN_LEN if _pos == "learned" else EVAL_LEN)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    for _pos, _c in curves.items():
        _smooth = np.convolve(_c, np.ones(8) / 8, mode="valid")
        _ax.plot(np.arange(len(_smooth)) + 4, _smooth / math.log(2), label=_pos)
    _ax.axvline(TRAIN_LEN, color="gray", ls="--", lw=0.8)
    _ax.set(xlabel="position in the window", ylabel="bits per character (8-position average)", ylim=(None, None))
    _ax.legend(fontsize=8)
    _rows = []
    for _pos, _c in curves.items():
        _b = [_c[a:b].mean() / math.log(2) if len(_c) >= b else float("nan")
              for a, b in ((16, 64), (64, 128), (128, 256), (256, 512))]
        _rows.append(f"| {_pos} | " + " | ".join("—" if math.isnan(v) else f"{v:.2f}" for v in _b) + " |")
    mo.vstack([_fig, mo.md("| scheme | positions 16–63 (trained) | 64–127 | 128–255 | 256–511 |\n|---|---|---|---|---|\n"
                           + "\n".join(_rows) + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return (curves,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the sinusoidal model has a formula for every position, so why does it fall
    apart past position 64? what does a layer trained only on sums $x + \text{PE}(p)$ with
    $p < 64$ do with vectors it has never seen? RoPE never adds anything to the residual stream;
    which of its frequencies produce angles at position 300 that training never showed it? why
    is ALiBi immune, and what is the price of a bias that *always* prefers nearby tokens? finally,
    explain how the model with no positional information can do so well (lab 01, experiment 3).
    """)
    return


if __name__ == "__main__":
    app.run()
