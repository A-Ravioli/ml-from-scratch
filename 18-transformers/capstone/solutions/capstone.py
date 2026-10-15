import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, datasets
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.text import CharTokenizer  # yours, from ch 17 · topic 02
    from mlfs.torchnn import count_parameters, get_device  # yours, from ch 15 · topic 07
    from mlfs.transformer import GPT, generate_cached, train_lm  # yours, from topics 04 and 06

    plt.rcParams["figure.autolayout"] = True

    # small CPU config (offline, tests, verify) vs full config (the real tiny Shakespeare on a laptop GPU).
    # set MLFS_CAPSTONE=small to force the small config on the real data.
    OFFLINE = datasets.is_offline()
    FULL = not OFFLINE and os.environ.get("MLFS_CAPSTONE", "full") != "small"
    if FULL:
        CONFIG = dict(block_size=128, batch_size=64, steps=2000, lr=1e-3, warmup=100, dropout=0.1, eval_interval=250,
                      device=str(get_device()),
                      sizes={"S": dict(d_model=96, n_layers=3, n_heads=4), "M": dict(d_model=192, n_layers=4, n_heads=6),
                             "L": dict(d_model=320, n_layers=6, n_heads=8)})
    else:
        CONFIG = dict(block_size=64, batch_size=32, steps=200, lr=2e-3, warmup=20, dropout=0.0, eval_interval=100,
                      device="cpu",
                      sizes={"S": dict(d_model=32, n_layers=1, n_heads=2), "M": dict(d_model=48, n_layers=2, n_heads=2),
                             "L": dict(d_model=64, n_layers=2, n_heads=4)})


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 18 · capstone: nanoGPT (reference solution)

    read [`brief.md`](brief.md) first. you'll train GPTs of three sizes on tiny Shakespeare
    with your `mlfs.transformer`, measure the best one in bits per character on the same test
    set as chapter 17's LSTM, fit a scaling curve, and measure what your KV cache buys at
    generation time. the model and the trainer are yours from topic 04; this notebook is the
    experiment around them.

    | stage | you build |
    |---|---|
    | 1 | `split_ids`: the 80/10/10 split by position, as int64 arrays |
    | 2 | `token_nll`: the loss of every token of a long text, with sliding windows so each has real context |
    | 3 | `bpc_ci`: bits per character with a block-bootstrap interval |
    | 4 | `fit_scaling`: a power law $L = a N^{-b}$ through (parameters, loss) points |
    | 5 | `generation_speed`: tokens per second, with and without the KV cache |

    when the checks are green, scroll down to train, evaluate and write up.
    """)
    return


@app.function
def split_ids(ids, *, train_frac=0.8, val_frac=0.1):
    """stage 1: (train, val, test) int64 arrays: the first train_frac, the next val_frac, the rest, by position."""
    ids = np.asarray(ids, dtype=np.int64)
    a = int(train_frac * len(ids))
    b = int((train_frac + val_frac) * len(ids))
    return ids[:a], ids[a:b], ids[b:]


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", split_ids=split_ids)
    return


@app.function
@torch.no_grad()
def token_nll(model, ids, *, block_size, stride=None, batch_size=16, device="cpu"):
    """stage 2: (len(ids) - 1,) float64: -ln P(ids[t] | context) for t >= 1, windows of block_size moved by stride."""
    stride = block_size // 2 if stride is None else stride
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.long)
    n = len(ids) - 1  # number of predictions
    T = min(block_size, n)
    starts = list(range(0, n - T + 1, stride))
    if starts[-1] + T < n:
        starts.append(n - T)
    out = np.empty(n)
    done = 0  # predictions scored so far
    was_training = model.training
    model.eval()
    try:
        for i in range(0, len(starts), batch_size):
            group = starts[i:i + batch_size]
            x = torch.stack([ids[s:s + T] for s in group]).to(device)
            y = torch.stack([ids[s + 1:s + T + 1] for s in group]).to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
            for row, s in zip(nll.reshape(len(group), T).cpu().double().numpy(), group):
                first = done - s  # only positions not scored yet: each has >= T - stride tokens of context
                out[s + first:s + T] = row[first:]
                done = s + T
    finally:
        model.train(was_training)
    return out


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", token_nll=token_nll)
    return


@app.function
def bpc_ci(nll, *, chars_per_token=1.0, block_size=2000, n_boot=2000, alpha=0.05, rng=None):
    """stage 3: (bits per char, lo, hi): mean nll / ln 2 / chars_per_token, bootstrap over contiguous blocks."""
    nll = np.asarray(nll, dtype=np.float64)
    blocks = np.array_split(nll, max(len(nll) // block_size, 2))
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([len(b) for b in blocks], dtype=np.float64)
    scale = 1.0 / (math.log(2) * chars_per_token)
    lo, hi = bootstrap_ci((sums, counts), lambda s, c: s.sum() / c.sum() * scale, n_boot=n_boot, alpha=alpha, rng=rng)
    return float(nll.mean() * scale), float(lo), float(hi)


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", bpc_ci=bpc_ci)
    return


@app.function
def fit_scaling(n_params, losses):
    """stage 4: (a, b) of loss = a * N^(-b), least squares on log loss vs log N."""
    x, y = np.log(np.asarray(n_params, dtype=np.float64)), np.log(np.asarray(losses, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.exp(intercept)), float(-slope)


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", fit_scaling=fit_scaling)
    return


@app.function
def generation_speed(model, prompt, n_new, *, cached=True, device="cpu"):
    """stage 5: new tokens per second for greedy generation of n_new tokens after prompt (B, T), cached or not."""
    model.to(device)
    prompt = prompt.to(device)
    sync = torch.mps.synchronize if str(device).startswith("mps") else (
        torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None))
    sync()
    t0 = time.perf_counter()
    if cached:
        out = generate_cached(model, prompt, n_new, temperature=0.0)
    else:
        out = model.generate(prompt, n_new, temperature=0.0)
    out.cpu()
    sync()
    return n_new * prompt.shape[0] / (time.perf_counter() - t0)


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", generation_speed=generation_speed)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    tiny Shakespeare, split by position exactly like chapters 04 and 17: the first 80% trains,
    the next 10% chooses between models, the last 10% is the test set, used once. characters are
    ids through your `CharTokenizer`, so bits per character are directly comparable with chapter
    17's LSTM (2.353 on test) and order-7 n-gram (2.631).
    """)
    return


@app.cell
def _():
    text = datasets.tiny_shakespeare()
    tok = CharTokenizer().train(text)
    train_ids, val_ids, test_ids = split_ids(tok.encode(text))
    mo.md(f"{'**tiny Shakespeare**' if not OFFLINE else '**synthetic stand-in (offline)**'}: {len(text):,} characters, "
          f"vocabulary {tok.vocab_size} · train {len(train_ids):,} · validation {len(val_ids):,} · test {len(test_ids):,}"
          f" · config **{'full' if FULL else 'small'}**: {CONFIG}")
    return test_ids, text, tok, train_ids, val_ids


@app.cell
def _(tok, train_ids, val_ids):
    # the scaling run (yours, via train_lm): three sizes, the same data, steps and recipe
    runs = {}
    for _name, _cfg in CONFIG["sizes"].items():
        torch.manual_seed(0)
        _m = GPT(tok.vocab_size, block_size=CONFIG["block_size"], dropout=CONFIG["dropout"], **_cfg)
        _t0 = time.perf_counter()
        _h = train_lm(_m, train_ids, CONFIG["block_size"], val_data=val_ids, batch_size=CONFIG["batch_size"],
                      max_steps=CONFIG["steps"], lr=CONFIG["lr"], warmup_steps=CONFIG["warmup"],
                      eval_interval=CONFIG["eval_interval"], device=CONFIG["device"], seed=0)
        runs[_name] = dict(model=_m, history=_h, n_params=count_parameters(_m), seconds=time.perf_counter() - _t0)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _name, _r in runs.items():
        _ax.plot(_r["history"]["step"], np.array(_r["history"]["val_loss"]) / math.log(2), "o-", ms=3,
                 label=f"{_name}: {_r['n_params'] / 1e6:.2f}M")
    _ax.set(xlabel="step", ylabel="validation bits per character")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| model | parameters | train bits/char | val bits/char | minutes |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {k} | {r['n_params']:,} | {r['history']['train_loss'][-1] / math.log(2):.3f} | "
                                       f"{r['history']['val_loss'][-1] / math.log(2):.3f} | {r['seconds'] / 60:.1f} |"
                                       for k, r in runs.items()))])
    return (runs,)


@app.cell
def _(runs):
    # the scaling curve (yours, stage 4)
    _n = [r["n_params"] for r in runs.values()]
    _v = [r["history"]["val_loss"][-1] for r in runs.values()]
    scaling = fit_scaling(_n, _v)
    _fig, _ax = plt.subplots(figsize=(5.5, 3.4))
    _ax.loglog(_n, _v, "o", label="measured (final validation loss)")
    _grid = np.geomspace(min(_n) / 2, max(_n) * 4, 50)
    _ax.loglog(_grid, scaling[0] * _grid ** -scaling[1], "--", label=f"fit: {scaling[0]:.2f} N^-{scaling[1]:.3f}")
    _ax.set(xlabel="parameters N", ylabel="validation loss (nats per character)")
    _ax.legend(fontsize=8)
    _local = [math.log(_v[i] / _v[i + 1]) / math.log(_n[i + 1] / _n[i]) for i in range(len(_n) - 1)]
    mo.vstack([_fig, mo.md(f"fitted exponent b = **{scaling[1]:.3f}**; local slopes between neighbouring sizes: "
                           + ", ".join(f"{x:.3f}" for x in _local)
                           + f" (a straight line on log-log axes would have equal slopes); extrapolated loss at 4× "
                             f"the largest model: {scaling[0] * (4 * max(_n)) ** -scaling[1]:.3f} nats")])
    return (scaling,)


@app.cell
def _(runs, test_ids, train_ids, val_ids):
    # evaluation of the best model on every character of validation and test (yours, stages 2-3)
    best_name = min(runs, key=lambda k: runs[k]["history"]["val_loss"][-1])
    best = runs[best_name]["model"]
    _B = CONFIG["block_size"]

    def block_nll(block, before):
        return token_nll(best, np.concatenate([before[-_B:], block]), block_size=_B, device=CONFIG["device"])[_B - 1:]

    gpt_val, gpt_test = block_nll(val_ids, train_ids), block_nll(test_ids, val_ids)
    _bs = 2000 if FULL else 500
    _v, _t = bpc_ci(gpt_val, block_size=_bs, rng=0), bpc_ci(gpt_test, block_size=_bs, rng=0)
    mo.md(f"best model **{best_name}** ({runs[best_name]['n_params']:,} parameters):\n\n"
          "| | validation bits/char [95% CI] | test bits/char [95% CI] |\n|---|---|---|\n"
          f"| GPT {best_name} | {_v[0]:.3f} [{_v[1]:.3f}, {_v[2]:.3f}] | **{_t[0]:.3f}** [{_t[1]:.3f}, {_t[2]:.3f}] |\n"
          "| ch 17 char-LSTM, 2 × 512 | 2.026 [1.968, 2.086] | 2.353 [2.310, 2.397] |\n"
          "| ch 17 interpolated n-gram, order 7 | 2.266 [2.209, 2.324] | 2.631 [2.585, 2.679] |")
    return best, best_name, gpt_test, gpt_val


@app.cell
def _(best, tok):
    # cached inference (yours, stage 5): greedy generation speed, with and without the KV cache
    _prompt = torch.tensor([tok.encode("ROMEO:\n" if set("ROMEO:\n") <= set(tok.chars) else "the ")])
    _n = CONFIG["block_size"] - _prompt.shape[1]
    speed = {c: generation_speed(best, _prompt, _n, cached=c, device=CONFIG["device"]) for c in (False, True)}
    _speed_cpu = {c: generation_speed(best, _prompt, _n, cached=c, device="cpu") for c in (False, True)}
    best.to(CONFIG["device"])
    mo.md(f"{_n} new tokens, greedy: **{CONFIG['device']}** {speed[False]:.0f} tokens/s without the cache, "
          f"{speed[True]:.0f} with it ({speed[True] / speed[False]:.1f}×) · **cpu** {_speed_cpu[False]:.0f} vs "
          f"{_speed_cpu[True]:.0f} tokens/s ({_speed_cpu[True] / _speed_cpu[False]:.1f}×)")
    return


@app.cell
def _():
    temperature = mo.ui.slider(steps=[0.0, 0.5, 0.8, 1.0, 1.2], value=0.8, label="temperature", show_value=True)
    temperature
    return (temperature,)


@app.cell
def _(best, temperature, text, tok):
    # samples (given), with your cached generation, and how much is copied verbatim from the training text
    _start = "ROMEO:\n" if set("ROMEO:\n") <= set(tok.chars) else text[:20]
    _prompt = torch.tensor([tok.encode(_start)], device=CONFIG["device"])
    _out = generate_cached(best, _prompt, CONFIG["block_size"] - _prompt.shape[1], temperature=temperature.value,
                           top_k=None, generator=torch.Generator(device="cpu").manual_seed(0))
    _s = tok.decode(_out[0, _prompt.shape[1]:].tolist())
    _train_text = text[: int(0.8 * len(text))]
    _chunks = [_s[_i:_i + 20] for _i in range(0, len(_s) - 20, 10)]
    _copied = np.mean([_c in _train_text for _c in _chunks]) if _chunks else 0.0
    mo.md(f"temperature {temperature.value}:\n\n```\n{_start}{_s}\n```\n\nfraction of 20-character chunks found "
          f"verbatim in the training text: **{_copied:.2f}**")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **results.** validation and test bits per character of your best GPT with block-bootstrap
       intervals, next to chapter 17's LSTM and n-gram on the same split. compute the paired
       difference against the LSTM if you saved its per-character losses (otherwise say why the
       unpaired comparison is weaker). what does the interval cover, and what doesn't it?
    2. **scaling.** the three sizes' validation losses, the fitted exponent, and what the fit
       predicts for a 4× larger model. is three points enough to call it a power law? what would
       change if you trained each size for longer (compute-optimal vs fixed steps)?
    3. **overfitting and the recipe.** the learning curves: does any size overfit within the
       budget? one ablation from: no dropout, sinusoidal positions, untied embeddings, or no
       warmup, on the middle size, with its validation bits per character.
    4. **cached inference.** the speedup of `generate_cached` on your device and on the CPU. why
       do they differ? estimate the KV-cache memory of your largest model at its full context with
       topic 06's `kv_cache_bytes`.
    5. **samples.** samples at temperatures 0, 0.8 and 1.2; greedy repetition; the copying check;
       how they compare with chapter 17's LSTM samples.
    6. **connections.** where did each topic enter: attention and masks (01), positions (02), the
       block (03), the model, sampler and trainer (04), the encoder alternative (05: why not BERT
       here?), the cache (06), and what would change for images (07)?
    """)
    return


if __name__ == "__main__":
    app.run()
