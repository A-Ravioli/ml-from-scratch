import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.text import NGramLM, tune_lambdas  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    OFFLINE = datasets.is_offline()
    TEXT = datasets.tiny_shakespeare()  # real tiny Shakespeare, or a synthetic stand-in offline
    _a, _b = int(0.8 * len(TEXT)), int(0.9 * len(TEXT))
    TRAIN, VAL = TEXT[:_a], TEXT[_a:_b]
    SIZES = [256, 320, 512, 1024, 2048, 4096]


@app.function
def truncated(full, vocab_size):
    """the tokenizer that training to vocab_size would have produced: greedy merges are a prefix of a longer run."""
    tok = ex.BPETokenizer(vocab_size)
    tok.merges = list(full.merges[: vocab_size - 256])
    tok._build_vocab()
    return tok


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # tokenization · lab

    every experiment here runs **your** `BPETokenizer` (and, in experiment 3, your `NGramLM` from
    topic 01) on {'a synthetic stand-in for tiny Shakespeare (offline)' if OFFLINE else 'tiny Shakespeare'}:
    the first 80% of the text trains the tokenizer, the next 10% is held out. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. with the reference
    solution the lab takes about 10 seconds; a trainer that recounts every pair after every
    merge takes a minute or two longer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: vocabulary size against sequence length

    one BPE tokenizer is trained to 4,096 tokens; the smaller vocabularies are prefixes of its
    merge list (greedy training makes the first $k$ merges of a long run identical to a run of
    $k$ merges, so `truncated` gives exactly what training to that size would). we count how many
    tokens the held-out text needs at each size.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "raw bytes: about 1 token per character. at 512 tokens, how many characters does one token cover on "
        "held-out text? at 4096? does doubling the vocabulary from 2048 to 4096 shorten the sequences by a lot?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    full = ex.BPETokenizer(SIZES[-1]).train(TRAIN)
    _secs = time.perf_counter() - _t0
    _n_chars = len(VAL)
    _rows, _cpt = [], []
    for _v in SIZES:
        _tok = truncated(full, _v)
        _n = len(_tok.encode(VAL))
        _cpt.append(_n_chars / _n)
        _last = [_tok.vocab[i].decode("utf-8", "replace") for i in range(max(256, _tok.vocab_size - 4), _tok.vocab_size)]
        _rows.append(f"| {_tok.vocab_size} | {_n:,} | {_n_chars / _n:.2f} | " + " · ".join(f"`{s!r}`" for s in _last) + " |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.plot(SIZES, _cpt, "o-")
    _ax.set_xscale("log", base=2)
    _ax.xaxis.set_minor_formatter(NullFormatter())
    _ax.set(xlabel="vocabulary size", ylabel="held-out characters per token", xticks=SIZES,
            xticklabels=[str(s) for s in SIZES])
    mo.vstack([_fig, mo.md(
        f"training to {SIZES[-1]} tokens took {_secs:.1f} s ({len(full.merges)} merges: training stops early if "
        f"no pair is left) · held-out text: {_n_chars:,} characters\n\n"
        "| vocabulary | held-out tokens | characters per token | the last tokens learned |\n|---|---|---|---|\n"
        + "\n".join(_rows))])
    return (full,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each merge removes one token per occurrence of its pair. why do the first
    merges shrink the text so much and the last ones so little (look at the token counts the
    pair frequencies must have)? what do the last tokens learned look like, and why do the
    whole-word tokens usually start with a space? a transformer's cost grows with the sequence length, and its
    embedding table with the vocabulary: what does that trade-off suggest?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: text the tokenizer has never seen

    the same tokenizer (2,048 tokens, trained on Shakespeare) encodes short passages of other
    kinds of text. "bytes per token" is how much each token compresses; "tokens per character"
    is how long the sequence gets.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the Shakespeare tokenizer encodes modern English, French, Python code, digits, and Chinese. which one needs "
        "the most tokens per character? can any of them need MORE than one token per character?"
    )
    p2
    return (p2,)


@app.cell
def _(full, p2):
    ui.gate(p2)
    _tok = truncated(full, 2048)
    _samples = {
        "Shakespeare (held-out)": VAL[1000:1400],
        "modern English": "The new phone update ships next week, and the battery should last noticeably longer.",
        "French": "Le petit chat dort sur le canapé pendant que la pluie tombe doucement dehors.",
        "Python code": "def mean(xs):\n    return sum(xs) / len(xs) if xs else 0.0\n",
        "digits": "Order 48213 shipped on 2024-03-17 for $1,299.95 (ref 7730041).",
        "Chinese": "今天天气很好，我们一起去公园散步吧。",
        "emoji": "🙂🙂🎉👍🏽🚀",
    }
    _rows = []
    for _name, _s in _samples.items():
        _ids = _tok.encode(_s)
        assert _tok.decode(_ids) == _s
        _rows.append(f"| {_name} | {len(_s)} | {len(_s.encode('utf-8'))} | {len(_ids)} | "
                     f"{len(_s.encode('utf-8')) / len(_ids):.2f} | {len(_ids) / len(_s):.2f} |")
    _pieces = [_tok.decode([i]) for i in _tok.encode(_samples["digits"])]
    mo.vstack([mo.md("| text | characters | UTF-8 bytes | tokens | bytes per token | tokens per character |\n"
                     "|---|---|---|---|---|---|\n" + "\n".join(_rows)),
               mo.md("how the digits line was cut: " + " · ".join(f"`{p!r}`" for p in _pieces))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does every text still round-trip, even characters the tokenizer has
    never seen? why does Chinese end up with *more* tokens than characters? how are the numbers
    cut into tokens, and why would that make arithmetic hard for a language model trained on
    these tokens? what does this say about training a tokenizer on the same mix of text the
    model will see?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the same n-gram model on different tokens

    your interpolated `NGramLM` from topic 01, order 3, trained on the training lines encoded
    with characters, and with BPE at several vocabulary sizes (weights tuned by EM on part of
    the held-out lines, scored on the rest). perplexities per token can't be compared across
    tokenizations, so everything is converted to **bits per character**: total bits divided by
    the number of characters (a line's newline counts as one, and is predicted as `</s>`).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "a trigram over characters sees 2 characters of context; a trigram over 1024-token BPE sees 2 tokens, maybe "
        "8 characters. which has fewer bits per character? does the biggest vocabulary win?"
    )
    p3
    return (p3,)


@app.cell
def _(full, p3):
    ui.gate(p3)
    _t0 = time.perf_counter()
    _tr_lines = TRAIN.split("\n")
    _val = VAL.split("\n")
    _tune, _ev = _val[: len(_val) // 2], _val[len(_val) // 2:]
    _n_chars = sum(len(l) + 1 for l in _ev)
    _char = ex.CharTokenizer().train(TEXT)
    _setups = [("characters", _char.encode)] + [(f"BPE {v}", truncated(full, v).encode) for v in [320, 512, 1024, 2048]]
    _rows, _bpc, _names = [], [], []
    for _name, _enc in _setups:
        _lm = NGramLM(3).fit([_enc(l) for l in _tr_lines])
        _ev_ids = [_enc(l) for l in _ev]
        _lam, _ = tune_lambdas(_lm, [_enc(l) for l in _tune])
        _n_tok = sum(len(s) + 1 for s in _ev_ids)
        _ppl = _lm.perplexity(_ev_ids)
        _b = _n_tok * math.log2(_ppl) / _n_chars
        _bpc.append(_b)
        _names.append(_name)
        _rows.append(f"| {_name} | {_n_tok:,} | {_ppl:.2f} | **{_b:.3f}** | {' '.join(f'{x:.2f}' for x in _lam)} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.bar(_names, _bpc, color="C0")
    _ax.set(ylabel="bits per character (held-out)", ylim=(0, max(_bpc) * 1.15))
    for _i, _v in enumerate(_bpc):
        _ax.text(_i, _v, f"{_v:.2f}", ha="center", va="bottom", fontsize=8)
    mo.vstack([_fig, mo.md(
        f"held-out characters scored: {_n_chars:,} · {time.perf_counter() - _t0:.0f} s\n\n"
        "| tokens | held-out tokens | perplexity per token | bits per character | EM weights (orders 1, 2, 3) |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the same order-3 model get so much better when its tokens are
    BPE pieces? what happens to the number of distinct contexts, and to how often each is seen,
    as the vocabulary grows, and where does that start to hurt? why would comparing perplexity
    *per token* have told you the opposite story?
    """)
    return


if __name__ == "__main__":
    app.run()
