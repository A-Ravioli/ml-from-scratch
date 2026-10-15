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

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    BLOCK, STEPS = 64, 300
    SIZES = {"tiny": dict(d_model=32, n_layers=1, n_heads=2), "small": dict(d_model=64, n_layers=2, n_heads=4),
             "medium": dict(d_model=128, n_layers=3, n_heads=4), "large": dict(d_model=160, n_layers=4, n_heads=4)}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # building GPT · lab

    every experiment here uses **your** `GPT`, `train_lm`, `estimate_loss` and
    `sample_next_token` from `exercise.py`, on tiny Shakespeare at the character level (the
    80/10/10 split of chapters 04 and 17). for each one: write down a prediction, lock it in, run
    it, then explain what happened. compare your explanations with `solutions/lab_notes.md`
    afterwards. the lab takes one to two minutes on a laptop CPU.
    """)
    return


@app.cell
def _():
    text = datasets.tiny_shakespeare()
    tok = CharTokenizer().train(text)
    ids = np.array(tok.encode(text), dtype=np.int64)
    _a, _b = int(0.8 * len(ids)), int(0.9 * len(ids))
    train_ids, val_ids = ids[:_a], ids[_a:_b]
    mo.md(f"{'tiny Shakespeare' if not datasets.is_offline() else 'offline stand-in'}: {len(text):,} characters, "
          f"vocabulary {tok.vocab_size}")
    return text, tok, train_ids, val_ids


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: loss against model size, at a fixed number of steps

    four GPTs from 17 thousand to 1.3 million parameters, each trained with your `train_lm` for
    the **same** 300 steps on the same batches (32 windows of 64 characters, learning rate
    $3 \times 10^{-3}$, 40 warmup steps). every model sees exactly the same 614,400 characters,
    less than one pass over the training text.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "after the same 300 steps on the same data, which size has the lowest validation loss? bigger models have more "
        "to learn: do they need more steps to catch up?"
    )
    p1
    return (p1,)


@app.cell
def _(p1, tok, train_ids, val_ids):
    ui.gate(p1)
    _t0 = time.perf_counter()
    results = {}
    for _name, _cfg in SIZES.items():
        torch.manual_seed(0)
        _m = ex.GPT(tok.vocab_size, block_size=BLOCK, **_cfg)
        _n = sum(p.numel() for p in _m.parameters())
        _h = ex.train_lm(_m, train_ids, BLOCK, val_data=val_ids, batch_size=32, max_steps=STEPS, lr=3e-3,
                         warmup_steps=40, eval_interval=50, eval_batches=8, seed=0)
        results[_name] = dict(model=_m, n_params=_n, history=_h)
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for _name, _r in results.items():
        _axes[0].plot(_r["history"]["step"], np.array(_r["history"]["val_loss"]) / math.log(2), "o-", ms=3,
                      label=f"{_name} ({_r['n_params'] / 1e6:.2f}M)")
    _axes[0].set(xlabel="step", ylabel="validation bits per character")
    _axes[0].legend(fontsize=7)
    _n = np.array([r["n_params"] for r in results.values()])
    _v = np.array([r["history"]["val_loss"][-1] for r in results.values()]) / math.log(2)
    _axes[1].plot(_n, _v, "o-")
    _axes[1].set(xscale="log", xlabel="parameters", ylabel="final validation bits per character")
    mo.vstack([_fig, mo.md("| model | parameters | final train bits/char | final validation bits/char |\n|---|---|---|---|\n"
                           + "\n".join(f"| {k} | {r['n_params']:,} | {r['history']['train_loss'][-1] / math.log(2):.3f}"
                                       f" | {r['history']['val_loss'][-1] / math.log(2):.3f} |" for k, r in results.items())
                           + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return (results,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the larger models are better at *every* step, not only at the end: they
    extract more from each character they see. why might more parameters mean faster
    learning per example, rather than slower? what would eventually stop the trend if you kept
    making the model bigger at this fixed amount of data? (the capstone fits this curve properly,
    and chapter 19 is about why it's a power law.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how you sample matters as much as what you trained

    the largest model from experiment 1 writes 600 characters from the same prompt under six
    decoding rules. for each sample: the average log-likelihood the *model itself* assigns to
    what it wrote (bits per character; lower = the model finds it more probable), the fraction of
    distinct 8-character chunks (1.0 = no repetition), and how much of it appears verbatim in the
    training text.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which decoding rule produces the text the model itself considers most probable? is that also the best text?"
    )
    p2
    return (p2,)


@app.function
def sample_stats(model, tok, prompt, train_text, *, n=600, seed=0, **kw):
    idx = torch.tensor([tok.encode(prompt)])
    out = model.generate(idx, n, generator=torch.Generator().manual_seed(seed), **kw)
    with torch.no_grad():  # the model's own log-likelihood of every new token, given up to BLOCK tokens of context
        nll = [torch.nn.functional.cross_entropy(model(out[:, max(0, t - BLOCK):t])[0][0, -1], out[0, t]).item()
               for t in range(idx.shape[1], out.shape[1])]
    s = tok.decode(out[0, idx.shape[1]:].tolist())
    chunks = [s[i:i + 8] for i in range(len(s) - 8)]
    copied = np.mean([s[i:i + 20] in train_text for i in range(0, len(s) - 20, 10)])
    return s, float(np.mean(nll) / math.log(2)), len(set(chunks)) / max(len(chunks), 1), float(copied)


@app.cell
def _(p2, results, text, tok):
    ui.gate(p2)
    _m = results["large"]["model"]
    _prompt = "ROMEO:\n" if set("ROMEO:\n") <= set(tok.chars) else text[:20]
    _train_text = text[: int(0.8 * len(text))]
    rules = {"greedy (T = 0)": dict(temperature=0.0), "T = 0.8": dict(temperature=0.8), "T = 1.0": dict(temperature=1.0),
             "T = 1.5": dict(temperature=1.5), "top-k = 5": dict(top_k=5), "top-p = 0.9": dict(top_p=0.9)}
    samples = {k: sample_stats(_m, tok, _prompt, _train_text, **v) for k, v in rules.items()}
    mo.md("| rule | model's bits/char on its own text | distinct 8-char chunks | copied 20-char chunks |\n|---|---|---|---|\n"
          + "\n".join(f"| {k} | {v[1]:.2f} | {v[2]:.2f} | {v[3]:.2f} |" for k, v in samples.items()))
    return (samples,)


@app.cell
def _(samples):
    pick = mo.ui.dropdown(options=list(samples), value="greedy (T = 0)", label="show the sample for")
    pick
    return (pick,)


@app.cell
def _(pick, samples):
    mo.md(f"```\n{samples[pick.value][0][:600]}\n```")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** greedy decoding writes the text the model finds most probable, and it's the
    worst text in the table. why does always taking the most likely next character lead into a
    loop, and why can't a model that predicts one character at a time escape it? why does
    $T = 1.5$ produce text the model itself considers improbable? top-k and top-p both cut off
    the tail of the distribution: what failure of plain $T = 1$ sampling are they fixing?
    """)
    return


if __name__ == "__main__":
    app.run()
