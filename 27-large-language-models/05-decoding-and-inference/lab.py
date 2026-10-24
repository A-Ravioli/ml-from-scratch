import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re
    import time
    from collections import Counter

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.llm import Llama  # yours, from topic 02
    from mlfs.text import CharTokenizer  # yours, from ch 17 · 02
    from mlfs.transformer import generate_cached, sample_next_token, train_lm  # yours, from ch 18

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OFFLINE = datasets.is_offline()
    TEXT = datasets.tiny_shakespeare()  # a synthetic stand-in offline
    NAMES = ["ROMEO", "JULIET", "MERCUTIO", "BENVOLIO", "TYBALT", "NURSE", "CAPULET", "FRIAR LAURENCE"]
    TOK = CharTokenizer().train(TEXT + "".join(NAMES) + ":\n")  # the extra characters matter only offline
    IDS = np.asarray(TOK.encode(TEXT), dtype=np.int64)
    CUT = int(0.9 * len(IDS))
    V = TOK.vocab_size
    STEPS = dict(target=150, small=60, medium=100) if OFFLINE else dict(target=500, small=400, medium=500)


@app.class_definition
class BigramLM(nn.Module):
    """the cheapest possible draft: next-character frequencies from the training text (add-one smoothed)."""

    def __init__(self, data, vocab_size):
        super().__init__()
        counts = np.ones((vocab_size, vocab_size))
        np.add.at(counts, (data[:-1], data[1:]), 1)
        self.register_buffer("logits", torch.tensor(np.log(counts / counts.sum(1, keepdims=True)), dtype=torch.float32))
        self.block_size = None

    def forward(self, idx, targets=None, *, cache=None):
        return self.logits[idx], None


@app.function
@torch.no_grad()
def sample_with(model, prompt, n, *, temperature=1.0, min_p=None, penalty=None, seed=0):
    """generate n characters one at a time with your filters: repetition penalty (on the last 64 tokens), then min-p."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([TOK.encode(prompt)])
    model.eval()
    for _ in range(n):
        logits = model(idx[:, -model.block_size:])[0][:, -1]
        if penalty is not None:
            logits = ex.apply_repetition_penalty(logits, idx[:, -64:], penalty)
        if min_p is not None:
            logits = ex.min_p_filter(logits, min_p)
        idx = torch.cat([idx, sample_next_token(logits, temperature=temperature, generator=g)[:, None]], 1)
    return TOK.decode(idx[0, len(prompt):].tolist())


@app.function
@torch.no_grad()
def acceptance_alpha(target, draft, data, n_windows=20, ctx=128, seed=0):
    """alpha = E[sum_v min(p(v), q(v))] over positions of held-out text: the chance a draft token is accepted."""
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(data) - ctx - 1, (n_windows,), generator=g)
    x = torch.stack([torch.as_tensor(data[s:s + ctx]) for s in starts.tolist()])
    p = F.softmax(target(x)[0], -1)
    q = F.softmax(draft(x)[0], -1)
    return torch.minimum(p, q).sum(-1).mean().item()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # decoding & inference · lab

    the experiments decode from a character-level `Llama` (your topic 02 model: 4 layers,
    width 128, {STEPS["target"]} steps on tiny Shakespeare; a synthetic stand-in offline)
    with **your** `apply_repetition_penalty`, `min_p_filter`, `speculative_decode` and
    `constrained_decode`. timings are wall-clock on this machine's CPU, best of two, one run:
    treat differences under ~10% as noise. for each experiment: write down a prediction,
    lock it in, run it, then explain what happened. about three minutes.
    """)
    return


@app.cell
def _():
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    target = Llama(V, block_size=256, n_layers=4, n_heads=4, n_kv_heads=2, d_model=128)
    train_lm(target, IDS[:CUT], 128, batch_size=32, max_steps=STEPS["target"], lr=3e-3, warmup_steps=50,
             eval_interval=10**9, eval_batches=1)
    drafts = {"bigram counts": BigramLM(IDS[:CUT], V)}
    for _name, _layers, _d, _steps in (("tiny (1 layer, width 32)", 1, 32, STEPS["small"]),
                                       ("small (2 layers, width 64)", 2, 64, STEPS["medium"])):
        torch.manual_seed(1)
        _m = Llama(V, block_size=256, n_layers=_layers, n_heads=2, n_kv_heads=1, d_model=_d)
        train_lm(_m, IDS[:CUT], 128, batch_size=32, max_steps=_steps, lr=3e-3, warmup_steps=20,
                 eval_interval=10**9, eval_batches=1)
        drafts[_name] = _m.eval()
    target.eval()
    mo.md(f"trained the target ({sum(p.numel() for p in target.parameters()):,} parameters) and two draft models in "
          f"{time.perf_counter() - _t0:.0f} s: " + ", ".join(
              f"{k} ({sum(p.numel() for p in d.parameters()):,} parameters)" for k, d in drafts.items()) + ".")
    return drafts, target


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how to sample

    400 characters from the same prompt and seed with six recipes. two numbers per sample:
    the fraction of distinct word pairs (low means it repeats itself) and the target's own
    average log-probability per character of the first 250 characters it wrote (high means "plausible to the
    model", which is not the same as good).
    """)
    return


@app.cell
def _():
    p1 = ui.predict("which recipe repeats itself the most? does min-p at temperature 1.5 read better or worse "
                    "than plain temperature 1.5?")
    p1
    return (p1,)


@app.cell
def _(p1, target):
    ui.gate(p1)
    _recipes = {"greedy (T = 0)": dict(temperature=0.0), "greedy + repetition penalty 1.3": dict(temperature=0.0, penalty=1.3),
                "T = 0.7": dict(temperature=0.7), "T = 1.0": dict(temperature=1.0), "T = 1.5": dict(temperature=1.5),
                "T = 1.5 + min-p 0.1": dict(temperature=1.5, min_p=0.1)}
    _prompt = "ROMEO:\n"
    samples = {}
    for _name, _kw in _recipes.items():
        _txt = sample_with(target, _prompt, 400, **_kw)
        _w = _txt.split()
        _pairs = list(zip(_w, _w[1:]))
        _ids = torch.tensor([TOK.encode(_prompt + _txt)])[:, :target.block_size + 1]  # score the first ~250 characters
        with torch.no_grad():
            _lp = -F.cross_entropy(target(_ids[:, :-1])[0][0], _ids[0, 1:], reduction="none")[len(_prompt) - 1:].mean()
        samples[_name] = (len(set(_pairs)) / max(len(_pairs), 1), _lp.item(), _txt)
    pick = mo.ui.dropdown(list(samples), value="greedy (T = 0)", label="show sample")
    mo.vstack([mo.md("| recipe | distinct word pairs | log-prob per char (target) |\n|---|---|---|\n"
                     + "\n".join(f"| {k} | {d:.2f} | {lp:.2f} |" for k, (d, lp, _) in samples.items())), pick])
    return pick, samples


@app.cell
def _(pick, samples):
    mo.md(f"```\n{samples[pick.value][2][:400]}\n```")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the most likely continuation, chosen greedily, loop? why is the
    target's log-probability highest for the sample a human would rate worst? what does
    min-p remove that plain temperature keeps, and why is it scale-aware where top-k isn't?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: speculative decoding against draft quality

    for each draft, $\alpha = \mathbb{E}\big[\sum_v \min(p(v), q(v))\big]$ over held-out text
    is the probability that a drafted token is accepted. with $k$ drafts per round, if each
    is accepted independently with probability $\alpha$, a target call yields on average
    $(1 - \alpha^{k+1})/(1 - \alpha)$ tokens (Leviathan et al., 2023). your `speculative_decode`
    generates 150 characters at temperature 1 with $k = 4$, counting target calls and timing
    it, against the target alone without a cache (`generate`) and with your chapter 18 KV
    cache (`generate_cached`).
    """)
    return


@app.cell
def _():
    p2 = ui.predict("which draft gives the largest wall-clock speedup: the best one, or the cheapest one? will "
                    "speculative decoding beat the KV cache?")
    p2
    return (p2,)


@app.cell
def _(drafts, p2, target):
    ui.gate(p2)
    _prompt = torch.tensor([TOK.encode("ROMEO:\n")])
    N = 150

    def _best(fn, reps=2):
        _ts = []
        for _ in range(reps):
            _t0 = time.perf_counter()
            _out = fn()
            _ts.append(time.perf_counter() - _t0)
        return _out, min(_ts)

    _, t_plain = _best(lambda: target.generate(_prompt, N, generator=torch.Generator().manual_seed(0)))
    _, t_cache = _best(lambda: generate_cached(target, _prompt, N, generator=torch.Generator().manual_seed(0)))
    spec_rows = []
    for _name, _d in drafts.items():
        _alpha = acceptance_alpha(target, _d, IDS[CUT:])
        _calls = [0]
        _h = target.register_forward_hook(lambda *a: _calls.__setitem__(0, _calls[0] + 1))
        (_out, _rate), _t = _best(lambda: ex.speculative_decode(target, _d, _prompt, N, k=4,
                                                                generator=torch.Generator().manual_seed(0)), reps=1)
        _h.remove()
        _, _t = _best(lambda: ex.speculative_decode(target, _d, _prompt, N, k=4, generator=torch.Generator().manual_seed(0)))
        spec_rows.append((_name, _alpha, _rate, N / _calls[0], (1 - _alpha ** 5) / (1 - _alpha), N / _t))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    _al = np.linspace(0, 0.95, 50)
    _a1.plot(_al, (1 - _al ** 5) / (1 - _al), "k--", lw=1, label="theory, k = 4")
    for _r in spec_rows:
        _a1.plot(_r[1], _r[3], "o", ms=7, label=_r[0])
        _a2.bar(_r[0].split(" (")[0], _r[5], color="C0")
    _a1.set(xlabel="alpha (measured on held-out text)", ylabel="tokens per target call")
    _a1.legend(fontsize=7)
    _a2.axhline(N / t_plain, color="C3", ls="--", label="target alone, no cache")
    _a2.axhline(N / t_cache, color="C2", ls="--", label="target alone, KV cache")
    _a2.set(ylabel="tokens per second", title="speculative decoding, k = 4")
    _a2.legend(fontsize=7)
    mo.vstack([_fig, mo.md(
        "| draft | alpha | acceptance rate (accepted / proposed) | tokens per target call | theory | tokens/s |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(f"| {n} | {a:.2f} | {r:.2f} | {c:.2f} | {th:.2f} | {s:.0f} |" for n, a, r, c, th, s in spec_rows)
        + f"\n\ntarget alone: **{N / t_plain:.0f}** tokens/s without a cache, **{N / t_cache:.0f}** with it.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the outputs are exact samples from the target whatever the draft, so all
    the draft changes is speed. why can the cheapest draft win on wall-clock time even with a
    lower $\alpha$? your `speculative_decode` reruns both models on the whole context every
    step: which of its costs would a KV cache for the target and the draft remove, and why
    do production systems (and Leviathan et al.'s 2–3× speedups) need both? why is the
    acceptance rate (accepted / proposed) lower than $\alpha$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: constrained decoding is not conditioning

    a speech must start with one of eight *Romeo and Juliet* speaker names, then `:` and a
    newline. two ways to get a valid name from the model after a blank line: **constrained
    decoding** (your `constrained_decode`, masking every character that can't continue a
    valid name), and **rejection sampling** (sample freely, keep only the valid ones, which
    samples exactly from the model conditioned on validity). are the two name distributions
    the same?
    """)
    return


@app.cell
def _():
    p3 = ui.predict("constrained decoding guarantees a valid name every time. will it pick the eight names with the "
                    "same frequencies as the model conditioned on validity?")
    p3
    return (p3,)


@app.cell
def _(p3, target):
    ui.gate(p3)
    names = NAMES
    _codes = [TOK.encode(n + ":\n") for n in names]
    _ctx = TEXT[CUT:CUT + 2000]
    _start = _ctx.index("\n\n") + 2 if "\n\n" in _ctx else None  # the synthetic offline text has no blank lines
    _prompt = torch.tensor([TOK.encode(_ctx[max(0, _start - 200):_start] if _start else _ctx[:200] + "\n\n")])

    def _allowed(gen):
        _mask = torch.zeros(V, dtype=torch.bool)
        for _c in _codes:
            if _c[:len(gen)] == gen and len(_c) > len(gen):
                _mask[_c[len(gen)]] = True
        return _mask

    _n = 150 if OFFLINE else 300
    _g = torch.Generator().manual_seed(0)
    _cons = Counter()
    for _ in range(_n):
        _out = ex.constrained_decode(target, _prompt, 20, _allowed, generator=_g)[0, _prompt.shape[1]:].tolist()
        _cons[TOK.decode(_out).rstrip(":\n")] += 1
    _n_free = 4 * _n if OFFLINE else 3000
    _free = generate_cached(target, _prompt.repeat(_n_free, 1), 16, generator=torch.Generator().manual_seed(1))
    _rej = Counter()
    _n_valid = 0
    for _row in _free[:, _prompt.shape[1]:].tolist():
        _s = TOK.decode(_row)
        _m = re.match(r"([A-Z][A-Z ]*):\n", _s)
        if _m and _m.group(1) in names:
            _rej[_m.group(1)] += 1
            _n_valid += 1
    _fc = sum(_rej.values()) or 1
    _cc = sum(_cons.values())
    _tv = 0.5 * sum(abs(_cons[n] / _cc - _rej[n] / _fc) for n in names)
    # how different two samples of these sizes from the SAME distribution would look (the noise floor)
    _pr = np.array([_rej[n] / _fc for n in names])
    _rs = np.random.default_rng(0)
    _noise = np.mean([0.5 * np.abs(_rs.multinomial(_cc, _pr) / _cc - _rs.multinomial(_fc, _pr) / _fc).sum()
                      for _ in range(500)])
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _x = np.arange(len(names))
    _ax.bar(_x - 0.2, [_cons[n] / _cc for n in names], 0.4, label=f"constrained decoding ({_cc} samples)")
    _ax.bar(_x + 0.2, [_rej[n] / _fc for n in names], 0.4, label=f"rejection sampling ({_fc} valid of {_n_free})")
    _ax.set_xticks(_x, [n.title() for n in names], rotation=30, fontsize=8)
    _ax.set(ylabel="share of samples")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(f"free sampling produced one of the eight names **{100 * _n_valid / _n_free:.0f}%** of the "
                           f"time. total variation distance between the two distributions: **{_tv:.2f}**; two samples "
                           f"of these sizes from one distribution would differ by about {_noise:.2f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at each step the mask renormalizes the model's next-character
    distribution over the allowed characters. why does that over-weight names whose first
    letters the model likes, even when it wouldn't have finished them that way? (compare the
    first letters of the most over-represented names with what the model writes freely.)
    when is constrained decoding exactly right, and what would you need to sample from the
    conditional distribution efficiently?
    """)
    return


if __name__ == "__main__":
    app.run()
