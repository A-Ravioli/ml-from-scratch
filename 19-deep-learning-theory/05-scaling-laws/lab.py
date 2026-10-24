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

    from mlfs import datasets, notebooks, ui
    from mlfs.text import CharTokenizer  # yours, from ch 17 · topic 02
    from mlfs.transformer import GPT, train_lm  # yours, from ch 18 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    BLOCK, BATCH, STEPS, EVAL_EVERY = 32, 32, 150, 15
    SIZES = {"XS": dict(d_model=16, n_layers=1, n_heads=2), "S": dict(d_model=32, n_layers=1, n_heads=2),
             "M": dict(d_model=48, n_layers=2, n_heads=2), "L": dict(d_model=64, n_layers=2, n_heads=4),
             "XL": dict(d_model=96, n_layers=3, n_heads=4), "XXL": dict(d_model=128, n_layers=4, n_heads=4)}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # scaling laws · lab

    every experiment uses **your** `fit_power_law`, `power_law`, `local_exponents`,
    `compute_optimal` and `non_embedding_params` from `exercise.py`, and your GPT and `train_lm`
    from chapter 18. the models are character-level GPTs on tiny Shakespeare (the 80/10/10 split
    of chapters 04, 17 and 18), from 3 thousand to 800 thousand non-embedding parameters, all
    trained with the same recipe for the same 150 steps of 32 windows of 32 characters. the lab
    takes about two minutes on a laptop CPU.
    """)
    return


@app.cell
def _():
    text = datasets.tiny_shakespeare()
    tok = CharTokenizer().train(text)
    _ids = np.array(tok.encode(text), dtype=np.int64)
    _a, _b = int(0.8 * len(_ids)), int(0.9 * len(_ids))
    train_ids, val_ids = _ids[:_a], _ids[_a:_b]
    mo.md(f"{'tiny Shakespeare' if not datasets.is_offline() else 'offline stand-in'}: {len(text):,} characters, "
          f"vocabulary {tok.vocab_size}")
    return tok, train_ids, val_ids


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: predict the biggest model before you train it

    six GPTs, two seeds each. you fit a power law in the non-embedding parameter count $N$ to the
    **five smallest** models' final validation losses, with and without an offset, and use each
    fit to predict the sixth model's loss. then you look at what the sixth model actually got.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the largest model has 2.4x the parameters of the next one. will the power law fitted to the five smaller "
        "models predict its loss to within 0.05 nats? which fit, with or without the offset c, will do better?"
    )
    p1
    return (p1,)


@app.cell
def _(p1, tok, train_ids, val_ids):
    ui.gate(p1)
    _t0 = time.perf_counter()
    runs = {}
    for _name, _cfg in SIZES.items():
        for _seed in (0, 1):
            torch.manual_seed(_seed)
            _m = GPT(tok.vocab_size, block_size=BLOCK, **_cfg)
            _h = train_lm(_m, train_ids, BLOCK, val_data=val_ids, batch_size=BATCH, max_steps=STEPS, lr=3e-3,
                          warmup_steps=15, eval_interval=EVAL_EVERY, eval_batches=8, seed=_seed)
            runs[(_name, _seed)] = dict(n=ex.non_embedding_params(_m), history=_h)
    names = list(SIZES)
    N = np.array([runs[(k, 0)]["n"] for k in names], dtype=float)
    final = np.array([[runs[(k, s)]["history"]["val_loss"][-1] for s in (0, 1)] for k in names])
    _y = final.mean(1)
    fits = {"with offset": ex.fit_power_law(N[:5], _y[:5]), "no offset": ex.fit_power_law(N[:5], _y[:5], offset=False)}
    _fig, _ax = plt.subplots(figsize=(5.5, 3.6))
    _ax.plot(np.repeat(N, 2), final.ravel(), "ko", ms=3, label="runs (2 seeds)")
    _grid = np.geomspace(N[0] / 1.5, N[-1] * 4, 100)
    for (_lab, (_a, _b, _c)), _col in zip(fits.items(), ("C0", "C3")):
        _ax.plot(_grid, ex.power_law(_grid, _a, _b, _c), _col, label=f"{_lab}: b = {_b:.3f}, c = {_c:.2f}")
    _ax.axvline(N[4] * 1.2, color="gray", ls=":", lw=0.8, label="end of the fitted range")
    _ax.set(xscale="log", yscale="log", xlabel="non-embedding parameters N", ylabel="validation loss (nats/char)")
    _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    _ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    _ax.legend(fontsize=7)
    _pred = {k: float(ex.power_law(N[-1], *v)) for k, v in fits.items()}
    _per_seed = {s: (float(ex.power_law(N[-1], *ex.fit_power_law(N[:5], final[:5, s]))),
                     float(ex.power_law(N[-1], *ex.fit_power_law(N[:5], final[:5, s], offset=False)))) for s in (0, 1)}
    _rows = "\n".join(f"| {k} | {runs[(k, 0)]['n']:,} | {final[i, 0]:.3f}, {final[i, 1]:.3f} |" for i, k in enumerate(names))
    mo.vstack([_fig, mo.md(
        f"| model | N | final validation loss (2 seeds) |\n|---|---|---|\n{_rows}\n\n"
        + "\n\n".join(f"**{k}** predicts {v:.3f} for {names[-1]}, which got {_y[-1]:.3f} (error {v - _y[-1]:+.3f})"
                      for k, v in _pred.items())
        + "\n\nfitting each seed's five runs separately: " + "; ".join(
            f"seed {s}: {v[0]:.3f} with offset, {v[1]:.3f} without (its {names[-1]} got {final[-1, s]:.3f})"
            for s, v in _per_seed.items())
        + f"\n\nlocal exponents between neighbours (no offset): {np.round(ex.local_exponents(N, _y), 3).tolist()}"
        + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return N, final, names, runs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the local exponents mostly shrink as the models grow: the curve bends upward on a
    log-log plot. a pure power law can't bend, so the no-offset fit commits to a constant slope;
    the offset $c$ lets it bend towards a floor. what in *this* experiment (fixed steps, fixed
    data) could produce a floor that has nothing to do with the entropy of English? why does a
    fit through five points still leave you unsure about the sixth?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: one compute budget, five ways to spend it

    a model with $N$ parameters that trains on $D$ tokens spends about $C = 6ND$ floating-point
    operations (the forward pass costs $2N$ per token, the backward pass $4N$). fix a budget $C$:
    the cost of training model L for 100 steps. give every size from XS to XL exactly that
    budget, so the small models train for many steps and the large ones for few
    ($\text{steps} = C / (6N \cdot 32 \cdot 32)$, warmup 10% of the steps), two seeds each.
    this is one "IsoFLOP profile" of Hoffmann et al. (2022).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "at a fixed compute budget, which size reaches the lowest validation loss: the smallest (most steps), the "
        "largest (most parameters), or one in between? how many training tokens per parameter does the winner get?"
    )
    p2
    return (p2,)


@app.cell
def _(N, names, p2, tok, train_ids, val_ids):
    ui.gate(p2)
    _t0 = time.perf_counter()
    _tokens_per_step = BATCH * BLOCK
    budget = 6 * N[names.index("L")] * 100 * _tokens_per_step
    iso = {}
    for _i, _k in enumerate(names):
        if _k == "XXL":
            continue
        _steps = max(2, int(round(budget / (6 * N[_i] * _tokens_per_step))))
        _losses = []
        for _seed in (0, 1):
            torch.manual_seed(_seed)
            _m = GPT(tok.vocab_size, block_size=BLOCK, **SIZES[_k])
            _h = train_lm(_m, train_ids, BLOCK, val_data=val_ids, batch_size=BATCH, max_steps=_steps, lr=3e-3,
                          warmup_steps=max(1, _steps // 10), eval_interval=_steps, eval_batches=8, seed=_seed)
            _losses.append(_h["val_loss"][-1])
        iso[_k] = dict(n=N[_i], steps=_steps, tokens=_steps * _tokens_per_step, losses=_losses)
    _ks = list(iso)
    _n = np.array([iso[k]["n"] for k in _ks])
    _l = np.array([np.mean(iso[k]["losses"]) for k in _ks])
    _best = _ks[int(np.argmin(_l))]
    _nopt, _dopt = ex.compute_optimal(budget)
    _fig, _ax = plt.subplots(figsize=(5, 3.4))
    for _k in _ks:
        _ax.plot([iso[_k]["n"]] * 2, iso[_k]["losses"], "ko", ms=3)
    _ax.plot(_n, _l, "C0-")
    _ax.axvline(float(_nopt), color="C3", ls=":", label="compute_optimal N (20 tokens/param)")
    _ax.set(xscale="log", xlabel="non-embedding parameters N", ylabel="validation loss after spending C",
            title=f"C = {budget:.2e} FLOPs")
    _ax.legend(fontsize=7)
    _rows = "\n".join(f"| {k} | {int(iso[k]['n']):,} | {iso[k]['steps']} | {iso[k]['tokens'] / iso[k]['n']:.1f} | "
                      f"{', '.join(f'{v:.3f}' for v in iso[k]['losses'])} |" for k in _ks)
    mo.vstack([_fig, mo.md(f"| model | N | steps | tokens per parameter | validation loss (2 seeds) |\n|---|---|---|---|---|\n"
                           f"{_rows}\n\nbest: **{_best}**; `compute_optimal` would choose N = {float(_nopt):,.0f} "
                           f"trained on {float(_dopt):,.0f} tokens\n\n{time.perf_counter() - _t0:.0f} s")])
    return (iso,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** Hoffmann et al. traced profiles like this at many budgets and found them
    U-shaped: too small a model can't represent much however long it trains, too large a model
    can't take enough steps to use its capacity, and the bottom sat near 20 tokens per
    parameter. do you see the U here, and where is its bottom in tokens per parameter? the
    20-token rule was measured on subword-token language models with billions of parameters:
    why might the best ratio be very different for single characters and models a million
    times smaller? what would you run next to find the bottom?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: chapter 18's capstone, revisited

    chapter 18's capstone trained three GPTs on the same data for the same 2,000 steps of 64
    windows of 128 characters and measured validation bits per character: S (354,240
    parameters) 2.567, M (1,816,896) 2.190, L (7,460,160) 2.062. with three points, a fit with an
    offset has three parameters for three numbers. use it, and the no-offset fit, to predict a
    30-million-parameter model trained the same way, and ask what Chinchilla's rule would have
    done with each model's compute.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "chapter 18's local slopes were 0.097 (S to M) and 0.043 (M to L). what loss would a 30M-parameter model reach "
        "with the same 2,000 steps? would it have been better to spend L's compute on a different size?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    ch18 = dict(names=["S", "M", "L"], n=np.array([354_240, 1_816_896, 7_460_160], dtype=float),
                bits=np.array([2.567, 2.190, 2.062]))
    _tokens = 2000 * 64 * 128
    _with = ex.fit_power_law(ch18["n"], ch18["bits"])
    _without = ex.fit_power_law(ch18["n"], ch18["bits"], offset=False)
    _fig, _ax = plt.subplots(figsize=(5, 3.4))
    _grid = np.geomspace(2e5, 6e7, 100)
    _ax.plot(ch18["n"], ch18["bits"], "ko", label="chapter 18 capstone")
    for (_lab, _p), _col in ((("with offset", _with), "C0"), (("no offset", _without), "C3")):
        _ax.plot(_grid, ex.power_law(_grid, *_p), _col, label=f"{_lab}: b = {_p[1]:.3f}, c = {_p[2]:.2f}")
    _ax.axvline(3e7, color="gray", ls=":", lw=0.8)
    _ax.set(xscale="log", xlabel="parameters N", ylabel="validation bits/char")
    _ax.legend(fontsize=7)
    _rows = []
    for _k, _n in zip(ch18["names"], ch18["n"]):
        _C = 6 * _n * _tokens
        _nopt, _dopt = ex.compute_optimal(_C)
        _rows.append(f"| {_k} | {int(_n):,} | {_C:.2e} | {float(_nopt):,.0f} | {float(_dopt) / _tokens:.1f}x |")
    mo.vstack([_fig, mo.md(
        f"local exponents: {np.round(ex.local_exponents(ch18['n'], ch18['bits']), 3).tolist()}\n\n"
        f"predicted bits/char at N = 30M: **{float(ex.power_law(3e7, *_with)):.3f}** with the offset fit, "
        f"**{float(ex.power_law(3e7, *_without)):.3f}** without (L got 2.062)\n\n"
        f"each model saw D = {_tokens:,} tokens\n\n| model | N | compute 6ND | `compute_optimal` N | "
        "`compute_optimal` D / D used |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (ch18,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the offset fit says a 30M model would barely improve on L, because a curve
    through three bending points flattens fast; the no-offset fit says it would gain more. both
    fit the data you have, so the data can't tell you which is right: that is the core risk of
    extrapolating a scaling law. what experiment would separate them? and by Chinchilla's rule,
    L was too *large* for its compute (it should have been a smaller model trained on more
    characters): how does that explain part of the bend?
    """)
    return


if __name__ == "__main__":
    app.run()
