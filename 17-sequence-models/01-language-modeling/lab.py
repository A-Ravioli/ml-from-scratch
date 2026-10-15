import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    OFFLINE = datasets.is_offline()
    TEXT = datasets.tiny_shakespeare()  # real tiny Shakespeare, or a synthetic stand-in offline
    _a, _b = int(0.8 * len(TEXT)), int(0.9 * len(TEXT))
    # one sequence per line: '</s>' plays the role of the newline character
    TRAIN_LINES = TEXT[:_a].split("\n")
    VAL_LINES = TEXT[_a:_b].split("\n")
    TUNE_LINES, EVAL_LINES = VAL_LINES[: len(VAL_LINES) // 2], VAL_LINES[len(VAL_LINES) // 2:]
    PLAIN = FuncFormatter(lambda v, _: f"{v:g}")  # plain numbers on log axes


@app.function
def words(line):
    """lowercase word and punctuation tokens of one line."""
    return re.findall(r"[a-z']+|[^\sa-z']", line.lower())


@app.function
def sub_model(big, n, **kwargs):
    """an order-n model that shares the low-order counts of an already fitted order-N model (N >= n).

    valid because the counts of order j (contexts of length j) don't depend on the model's n, as long as
    j <= n - 1: padding with n-1 '<s>' only adds '<s>' that a length-j context never reaches past.
    """
    m = ex.NGramLM(n, **kwargs)
    m.counts_, m.totals_, m.vocab_ = big.counts_[:n], big.totals_[:n], big.vocab_
    return m


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # language modeling · lab

    every experiment here runs **your** `NGramLM` and `tune_lambdas` from `exercise.py` on
    {'a synthetic stand-in for tiny Shakespeare (offline)' if OFFLINE else 'tiny Shakespeare'}:
    the first 80% of the text trains, the next 10% is validation (its first half tunes the
    interpolation weights, its second half is scored). every line is one sequence. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened. compare
    your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about half
    a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: perplexity against n, characters

    character models of order $n = 1, \dots, 8$, scored on held-out lines: add-k with
    $k = 0.1$, and interpolation with weights tuned by your EM. one order-8 model is fitted, and
    the lower orders reuse its counts (`sub_model`, above, explains why that's legitimate).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "a character model sees the previous n-1 characters. as n goes from 1 to 8, what happens to the held-out "
        "perplexity of the add-k model? of the interpolated one? what does the training perplexity do?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    _N = 8
    char_big = ex.NGramLM(_N).fit(TRAIN_LINES)
    _train_sample = TRAIN_LINES[::40]
    char_rows, _curves = [], {"add-k (k = 0.1), held-out": [], "interpolated (EM), held-out": [],
                              "interpolated (EM), training": []}
    char_lambdas = {}
    for _n in range(1, _N + 1):
        _addk = sub_model(char_big, _n, smoothing="add_k", add_k=0.1)
        _interp = sub_model(char_big, _n)
        _lam, _ = ex.tune_lambdas(_interp, TUNE_LINES)
        char_lambdas[_n] = _lam
        _r = (_addk.perplexity(EVAL_LINES), _interp.perplexity(EVAL_LINES), _interp.perplexity(_train_sample))
        for _k, _v in zip(_curves, _r):
            _curves[_k].append(_v)
        char_rows.append(f"| {_n} | {_r[0]:.2f} | {_r[1]:.2f} | {_r[2]:.2f} | "
                         + " ".join(f"{x:.2f}" for x in _lam) + " |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    for _k, _v in _curves.items():
        _ax.plot(range(1, _N + 1), _v, "o-" if "held" in _k else "s:", label=_k)
    _ax.set(xlabel="order n (context = n - 1 characters)", ylabel="perplexity per character", yscale="log")
    _ax.yaxis.set_major_formatter(PLAIN)
    _ax.yaxis.set_minor_formatter(PLAIN)
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"vocabulary: {len(char_big.vocab_)} tokens · {time.perf_counter() - _t0:.0f} s\n\n"
        "| n | add-k held-out | interpolated held-out | interpolated train | EM weights (order 1 → n) |\n"
        "|---|---|---|---|---|\n" + "\n".join(char_rows))])
    return char_big, char_lambdas


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does add-k's held-out perplexity turn around and climb, while the
    training perplexity keeps falling? at $n = 8$, what fraction of held-out 7-character
    contexts do you think were never seen in training, and what probability does add-k give
    every continuation of such a context? how does interpolation escape that, and what do the
    EM weights tell you about which orders the model trusts?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: words, and the price of add-one

    the same text as **words** (lowercased; punctuation marks are tokens). orders 1 to 4,
    with add-1, add-0.01 and interpolated (EM) smoothing. a held-out word never seen in
    training is scored as `<unk>`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "add-one smoothing on word bigrams vs the plain unigram model (add-one too): which has the lower held-out "
        "perplexity? by roughly how much does interpolation beat the best add-k model?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _t0 = time.perf_counter()
    _tr = [words(l) for l in TRAIN_LINES]
    _tune = [words(l) for l in TUNE_LINES]
    _ev = [words(l) for l in EVAL_LINES]
    _big = ex.NGramLM(4).fit(_tr)
    _n_tok = sum(len(s) for s in _ev)
    _oov = sum(1 for s in _ev for w in s if not _big._known(w))
    _res = {"add-1": [], "add-0.01": [], "interpolated (EM)": []}
    for _n in range(1, 5):
        _res["add-1"].append(sub_model(_big, _n, smoothing="add_k", add_k=1.0).perplexity(_ev))
        _res["add-0.01"].append(sub_model(_big, _n, smoothing="add_k", add_k=0.01).perplexity(_ev))
        _m = sub_model(_big, _n)
        ex.tune_lambdas(_m, _tune)
        _res["interpolated (EM)"].append(_m.perplexity(_ev))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _k, _v in _res.items():
        _ax.plot(range(1, 5), _v, "o-", label=_k)
    _ax.set(xlabel="order n", ylabel="perplexity per word", yscale="log", xticks=range(1, 5))
    _ax.yaxis.set_major_formatter(PLAIN)
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"vocabulary {len(_big.vocab_):,} · held-out tokens {_n_tok:,}, of which {_oov / _n_tok:.1%} are "
        f"out of vocabulary · {time.perf_counter() - _t0:.0f} s\n\n"
        "| smoothing | n = 1 | n = 2 | n = 3 | n = 4 |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {k} | " + " | ".join(f"{x:.1f}" for x in v) + " |" for k, v in _res.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with a vocabulary of $|V|$ words, add-one hands every unseen bigram a
    pseudo-count of 1. for a context seen $c$ times, what fraction of its probability mass goes
    to continuations it has never been followed by? (try "the", seen thousands of times, and a
    word seen 3 times.) why does that make the bigram model *worse* than ignoring the context?
    you can't compare these numbers with experiment 1's: why not, and what would you convert
    both to?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what the models write, and how much of it is copied

    samples of 300 characters from character models of several orders, twice: with all the
    weight on order $n$ (`lambdas` = $(0, \dots, 0, 1)$, the unsmoothed maximum-likelihood
    model), and with the EM weights from experiment 1. for each sample we measure the fraction
    of its 12-character windows that occur verbatim somewhere in the training text (the
    fraction in the table is the mean over 3 samples). move the slider to see other samples.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what fraction of an order-8 maximum-likelihood sample's 12-character windows appear word for word in the "
        "training text? an order-3 sample's? and the order-8 interpolated model's, which has the best perplexity?"
    )
    p3
    return (p3,)


@app.cell
def _():
    order = mo.ui.slider(2, 8, value=8, label="order n of the sample below")
    order
    return (order,)


@app.cell
def _(char_big, char_lambdas, order, p3):
    ui.gate(p3)
    _train_text = "\n".join(TRAIN_LINES)

    def _copied(sample, w=12):
        wins = [sample[i:i + w] for i in range(len(sample) - w + 1)]
        return np.mean([x in _train_text for x in wins]) if wins else float("nan")

    _rows, _shown = [], {}
    for _n in [2, 3, 4, 6, 8]:
        _row = []
        for _name, _lam in [("ML", np.eye(_n)[-1]), ("EM", char_lambdas[_n])]:
            _m = sub_model(char_big, _n, lambdas=_lam)
            _fr = []
            for _s in range(3):
                _txt = "".join("\n" if t == "</s>" else t for t in _m.sample(300, rng=_s))
                _fr.append(_copied(_txt))
                if _n == order.value and _s == 0:
                    _shown[_name] = _txt
            _row.append(np.mean(_fr))
        _rows.append(f"| {_n} | {_row[0]:.0%} | {_row[1]:.0%} |")
    mo.vstack([mo.md("| order | all weight on order n (maximum likelihood) | EM-tuned interpolation |\n|---|---|---|\n"
                     + "\n".join(_rows)),
               mo.md(f"**order {order.value}, maximum likelihood:**\n\n```\n{_shown['ML']}\n```\n\n"
                     f"**order {order.value}, EM interpolation:**\n\n```\n{_shown['EM']}\n```")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the copied fraction climb so steeply with $n$ for the
    maximum-likelihood model? an order-8 model conditions on 7 characters: how many different
    continuations does a typical 7-character context of the training text have? why does the
    interpolated model, which mixes in the lower orders, copy so much less, while its samples
    still read like nonsense? what would a model need to write long-range coherent text without
    copying (this is the motivation for topics 03–06)?
    """)
    return


if __name__ == "__main__":
    app.run()
