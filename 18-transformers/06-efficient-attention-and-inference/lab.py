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

    from mlfs import notebooks, ui
    from mlfs.transformer import GPT, scaled_dot_product_attention  # yours, from topics 01 and 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))


@app.function
def best_time(fn, repeats=3):
    """the fastest of `repeats` wall-clock timings of fn(), in seconds (the least disturbed by other processes)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # efficient attention & inference · lab

    the experiments use **your** `generate_cached`, `kv_cache_bytes`, `online_softmax_attention`
    and `linear_attention` from `exercise.py`, with your `GPT` from topic 04. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the lab takes about a minute on a laptop CPU.
    timings depend on your machine and on what else it's doing; the *ratios* are what matter.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: generation with and without the cache

    a GPT with 4 layers of width 128 and room for 1,024 positions (random weights: speed
    doesn't depend on what the model knows) generates $n$ tokens after a 16-token prompt, once
    with `GPT.generate` (every step re-reads the whole sequence) and once with your
    `generate_cached`, greedily, so both produce the same tokens.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how much faster is cached generation for 32 new tokens? for 1,000? does the speedup grow linearly with n, or "
        "faster?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    torch.manual_seed(0)
    _m = GPT(65, block_size=1024, n_layers=4, n_heads=4, d_model=128).eval()
    _prompt = torch.randint(0, 65, (1, 16))
    rows = []
    for _n in (32, 128, 512, 1000):
        _reps = 3 if _n <= 128 else 1
        _t_plain = best_time(lambda: _m.generate(_prompt, _n, temperature=0.0), _reps)
        _t_cache = best_time(lambda: ex.generate_cached(_m, _prompt, _n, temperature=0.0), _reps)
        rows.append((_n, _t_plain, _t_cache))
    _same = torch.equal(_m.generate(_prompt, 64, temperature=0.0), ex.generate_cached(_m, _prompt, 64, temperature=0.0))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.plot([r[0] for r in rows], [r[0] / r[1] for r in rows], "o-", label="GPT.generate (no cache)")
    _ax.plot([r[0] for r in rows], [r[0] / r[2] for r in rows], "o-", label="generate_cached")
    _ax.set(xscale="log", yscale="log", xlabel="new tokens", ylabel="tokens per second")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| new tokens | no cache (s) | cached (s) | speedup |\n|---|---|---|---|\n"
                           + "\n".join(f"| {n} | {a:.2f} | {b:.2f} | {a / b:.1f}× |" for n, a, b in rows)
                           + f"\n\nsame 64 tokens with and without the cache: **{_same}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** count the token-positions each method pushes through the model to produce
    $n$ tokens from a prompt of $p$ (on paper 1). why is the speedup well below that ratio for short
    generations? the cached version's tokens per second also falls slowly as $n$ grows: what work
    still grows with the length of the sequence?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what the cache costs in memory

    your `kv_cache_bytes` for the shapes of real models, at their full context and 16-bit
    values, next to the memory for the model's own weights (parameters × 2 bytes).
    """)
    return


@app.cell
def _():
    batch = mo.ui.slider(1, 64, value=16, label="batch size (sequences served at once)", show_value=True)
    p2 = ui.predict(
        "serving 16 sequences of 4,096 tokens with a 7B-parameter model (32 layers, 32 heads of 128, no GQA): is the KV "
        "cache smaller or larger than the 13.5 GB of weights?"
    )
    mo.vstack([p2, batch])
    return batch, p2


@app.cell
def _(batch, p2):
    ui.gate(p2)
    _models = [  # name, layers, kv heads, head dim, context, parameters
        ("GPT-2 small", 12, 12, 64, 1024, 124e6),
        ("Llama-2 7B (MHA)", 32, 32, 128, 4096, 6.7e9),
        ("Llama-2 70B (GQA, 8 kv heads)", 80, 8, 128, 4096, 69e9),
        ("70B shape without GQA (64 kv heads)", 80, 64, 128, 4096, 69e9),
    ]
    _rows = []
    for _name, _L, _H, _dh, _T, _P in _models:
        _per_seq = ex.kv_cache_bytes(_L, _H, _dh, _T)
        _total = ex.kv_cache_bytes(_L, _H, _dh, _T, batch_size=batch.value)
        _rows.append(f"| {_name} | {_per_seq / 2**30:.2f} GiB | {_total / 2**30:.1f} GiB | {2 * _P / 2**30:.1f} GiB |")
    mo.md(f"| model | KV cache, 1 sequence | KV cache, batch of {batch.value} | weights (16-bit) |\n|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the cache grow with batch size and context while the weights don't?
    what does grouped-query attention change, and what does it *not* change (the queries still
    have 64 heads)? why does that make GQA a serving-cost decision more than a modeling one?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: three ways to attend over long sequences

    one attention over $T$ tokens (4 heads of width 32, non-causal), computed three ways: your
    topic-01 `scaled_dot_product_attention` (builds the $T \times T$ matrix), your
    `online_softmax_attention` (blocks of 256 keys, never stores it) and your `linear_attention`
    (a different, cheaper model of attention). time, and the size of the largest score matrix
    each one holds.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "doubling T from 2,048 to 4,096: how much slower does each method get? does online softmax save time, memory, "
        "or both?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _g = torch.Generator().manual_seed(0)
    _rows = []
    for _T in (512, 1024, 2048, 4096):
        _q, _k, _v = (torch.randn(1, 4, _T, 32, generator=_g) for _ in range(3))
        _t_full = best_time(lambda: scaled_dot_product_attention(_q, _k, _v))
        _t_online = best_time(lambda: ex.online_softmax_attention(_q, _k, _v, block_size=256))
        _t_lin = best_time(lambda: ex.linear_attention(_q, _k, _v))
        _err = (ex.online_softmax_attention(_q, _k, _v, block_size=256) - scaled_dot_product_attention(_q, _k, _v)[0]).abs().max()
        _rows.append(f"| {_T:,} | {_t_full * 1e3:.1f} ms | {4 * _T * _T * 4 / 2**20:.0f} MiB | {_t_online * 1e3:.1f} ms | "
                     f"{4 * _T * 256 * 4 / 2**20:.0f} MiB | {_err:.1e} | {_t_lin * 1e3:.2f} ms |")
    mo.md("| T | full: time | full: score matrix | online: time | online: largest block | online vs full, max diff | "
          "linear: time |\n|---|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** online softmax does exactly the same arithmetic as the full version, plus a
    little rescaling, so why bother? (think about where a GPU keeps a 256-key block versus a
    $4096 \times 4096$ matrix, and what "memory-bound" means.) linear attention is in a different
    complexity class; what did it give up to get there?
    """)
    return


if __name__ == "__main__":
    app.run()
