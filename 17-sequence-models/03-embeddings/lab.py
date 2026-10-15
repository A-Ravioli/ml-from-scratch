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

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    OFFLINE = datasets.is_offline()
    _words = re.findall(r"[a-z']+", datasets.tiny_shakespeare().lower())
    _counts = Counter(_words)
    VOCAB = sorted([w for w, c in _counts.items() if c >= 5], key=lambda w: (-_counts[w], w))
    STOI = {w: i for i, w in enumerate(VOCAB)}
    IDS = np.array([STOI[w] for w in _words if w in STOI])  # rare words are dropped before windowing
    CONFIG = dict(embed_dim=64, window=3, n_negatives=5, epochs=5, batch_size=4096, lr=0.01)

    # a small analogy set in Shakespeare's vocabulary: a : b :: c : d
    ANALOGIES = [q.split() for q in """he him she her|he his she her|i me we us|i my we our|me my us our|
        thou thy you your|king queen lord lady|king queen duke duchess|man woman lord lady|man woman king queen|
        father mother son daughter|brother sister son daughter|father son mother daughter|husband wife king queen|
        boy girl son daughter|good better bad worse|is was are were|do did have had|come came go went|
        give gave take took|know knew think thought|speak spoke break broke|he himself she herself|
        uncle aunt brother sister|man men woman women|lord lords king kings""".replace("\n", "").split("|")]
    ANALOGIES = [[w.strip() for w in q] for q in ANALOGIES]
    ANALOGIES = [q for q in ANALOGIES if all(w in STOI for w in q)]


@app.function
def analogy_scores(E):
    """(accuracy@1, accuracy@5) with a, b, c excluded, and how often the plain nearest vector is a, b or c."""
    if not ANALOGIES:
        return float("nan"), float("nan"), float("nan")
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    hit1 = hit5 = own = 0
    for a, b, c, d in ANALOGIES:
        ia, ib, ic, i_d = STOI[a], STOI[b], STOI[c], STOI[d]
        top = list(ex.analogy(E, ia, ib, ic, k=5))
        hit1 += top[0] == i_d
        hit5 += i_d in top
        own += int(np.argmax(En @ (En[ib] - En[ia] + En[ic]))) in (ia, ib, ic)
    n = len(ANALOGIES)
    return hit1 / n, hit5 / n, own / n


@app.function
def ppmi_svd(ids, vocab_size, window, dim):
    """count-based embeddings: SVD of the positive PMI matrix of window co-occurrences (contexts smoothed by ^0.75)."""
    pairs = ex.skipgram_pairs(ids, window)
    C = np.zeros((vocab_size, vocab_size))
    np.add.at(C, (pairs[:, 0], pairs[:, 1]), 1.0)
    p_w = C.sum(1) / C.sum()
    p_c = C.sum(0) ** 0.75
    p_c /= p_c.sum()
    with np.errstate(divide="ignore"):
        pmi = np.log(C / C.sum()) - np.log(p_w)[:, None] - np.log(p_c)[None, :]
    U, S, _ = np.linalg.svd(np.maximum(pmi, 0.0), full_matrices=False)
    return U[:, :dim] * np.sqrt(S[:dim])


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # embeddings · lab

    every experiment here runs **your** `skipgram_pairs`, `train_skipgram`, `most_similar` and
    `analogy` from `exercise.py`, on the words of
    {'a synthetic stand-in for tiny Shakespeare (offline)' if OFFLINE else 'tiny Shakespeare'}
    (lowercased; the {len(VOCAB):,} words that occur at least 5 times, {len(IDS):,} tokens).
    for each experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. the lab takes about a
    minute on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what ends up next to what

    skip-gram with negative sampling: 64 dimensions, window 3, 5 negatives, 5 epochs over every
    pair (Adam, batches of 4,096). then nearest neighbours by cosine, and the cosine between a
    few pairs of words.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which word will be closer to 'love' in the embedding: a near-synonym like 'dear', or its opposite 'hate'? "
        "is 'day' closer to 'night' or to 'morning'?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    w2v, w2v_losses = ex.train_skipgram(IDS, len(VOCAB), seed=0, **CONFIG)
    E_w2v = w2v.embeddings.detach().cpu().numpy()
    _secs = time.perf_counter() - _t0
    _show = [w for w in ["king", "love", "thou", "rome", "night", "sword"] if w in STOI] or VOCAB[:6]
    _rows = []
    for _w in _show:
        _idx, _s = ex.most_similar(E_w2v, STOI[_w], k=6)
        _rows.append(f"| **{_w}** | " + ", ".join(f"{VOCAB[i]} ({s:.2f})" for i, s in zip(_idx, _s)) + " |")
    _pairs = [("love", "hate"), ("love", "dear"), ("day", "night"), ("day", "morning"), ("good", "bad"),
              ("king", "queen"), ("thou", "you"), ("thou", "sword")]
    _cos = []
    for _a, _b in _pairs:
        if _a in STOI and _b in STOI:
            _u, _v = E_w2v[STOI[_a]], E_w2v[STOI[_b]]
            _cos.append(f"| {_a} – {_b} | {_u @ _v / np.linalg.norm(_u) / np.linalg.norm(_v):.2f} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 2.8))
    _k = 50
    _ax.plot(np.convolve(w2v_losses, np.ones(_k) / _k, mode="valid"))
    _ax.set(xlabel="batch", ylabel="loss (moving average of 50)")
    mo.vstack([_fig, mo.md(f"training: {_secs:.1f} s, {len(w2v_losses)} batches\n\n"
                           "| word | nearest neighbours (cosine) |\n|---|---|\n" + "\n".join(_rows)
                           + "\n\n| pair | cosine |\n|---|---|\n" + "\n".join(_cos))])
    return (E_w2v,)


@app.cell
def _(E_w2v):
    word = mo.ui.dropdown(VOCAB[:400], value=VOCAB[min(40, len(VOCAB) - 1)], label="neighbours of")
    word
    return (word,)


@app.cell
def _(E_w2v, word):
    _idx, _s = ex.most_similar(E_w2v, STOI[word.value], k=10)
    mo.md(", ".join(f"{VOCAB[i]} ({s:.2f})" for i, s in zip(_idx, _s)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** skip-gram places two words close together when they are predicted by the
    same contexts. what contexts do "love" and "hate" share? why does that make opposites
    neighbours, and what does that tell you about the kind of similarity these vectors measure?
    the neighbours of "thou" are mostly verbs: why those verbs, in Shakespeare's English?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: vector arithmetic

    a hand-made set of analogies in Shakespeare's vocabulary (he : him :: she : her, king :
    queen :: lord : lady, come : came :: go : went, …). for each, `analogy` returns the words
    nearest to $b - a + c$ with $a$, $b$, $c$ excluded. we also check where the plain nearest
    vector to $b - a + c$ lands *without* the exclusion. two models, trained from different
    seeds.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what fraction of the analogies will the model get exactly right (top 1)? and without excluding a, b and c, "
        "how often is the nearest vector to b - a + c one of the three query words?"
    )
    p2
    return (p2,)


@app.cell
def _(E_w2v, p2):
    ui.gate(p2)
    _E2 = ex.train_skipgram(IDS, len(VOCAB), seed=1, **CONFIG)[0].embeddings.detach().cpu().numpy()
    _rows = []
    for _name, _E in [("seed 0", E_w2v), ("seed 1", _E2)]:
        _a1, _a5, _own = analogy_scores(_E)
        _rows.append(f"| {_name} | {_a1:.0%} | {_a5:.0%} | {_own:.0%} |")
    _ex_rows = []
    for _a, _b, _c, _d in ANALOGIES:
        _top = [VOCAB[i] for i in ex.analogy(E_w2v, STOI[_a], STOI[_b], STOI[_c], k=3)]
        _ex_rows.append(f"| {_a} : {_b} :: {_c} : **{_d}** | {', '.join(_top)} | {'✓' if _top[0] == _d else ''} |")
    mo.vstack([mo.md(f"{len(ANALOGIES)} analogies with all four words in the vocabulary\n\n"
                     "| model | correct (top 1) | correct in top 5 | nearest without exclusion is a, b or c |\n"
                     "|---|---|---|---|\n" + "\n".join(_rows)),
               mo.md("| analogy | top 3 (seed 0) | |\n|---|---|---|\n" + "\n".join(_ex_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the nearest vector to $b - a + c$ almost always $b$ or $c$ itself?
    (how big is the "offset" $b - a$ compared with $c$, for normalised vectors?) so how much of
    the famous king − man + woman ≈ queen result is the arithmetic, and how much is the
    exclusion rule? with 26 questions, how large a difference between the two seeds could be
    pure chance?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: prediction or counting?

    a different route to word vectors, with no neural network and no training loop: count
    co-occurrences within the same window, turn the counts into **positive pointwise mutual
    information** $\max(0, \log\frac{P(w, c)}{P(w)P(c)})$ (context probabilities smoothed by the
    same power 3/4 as the negatives), and keep the top 64 singular directions of that matrix.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "will a single SVD of a count matrix do better or worse than word2vec on the analogies? how many of a word's "
        "10 nearest neighbours will the two methods agree on?"
    )
    p3
    return (p3,)


@app.cell
def _(E_w2v, p3):
    ui.gate(p3)
    _t0 = time.perf_counter()
    _E_svd = ppmi_svd(IDS, len(VOCAB), CONFIG["window"], CONFIG["embed_dim"])
    _secs = time.perf_counter() - _t0
    _rows = []
    for _name, _E in [("word2vec (seed 0)", E_w2v), ("PPMI + SVD", _E_svd)]:
        _a1, _a5, _ = analogy_scores(_E)
        _rows.append(f"| {_name} | {_a1:.0%} | {_a5:.0%} |")
    _probe = list(range(min(300, len(VOCAB))))
    _overlap = [len(set(ex.most_similar(E_w2v, i, k=10)[0]) & set(ex.most_similar(_E_svd, i, k=10)[0])) / 10
                for i in _probe]
    _shared = [w for w in ["king", "love", "night"] if w in STOI]
    _nb = "\n".join(f"| {w} | {', '.join(VOCAB[i] for i in ex.most_similar(E_w2v, STOI[w], k=6)[0])} | "
                    f"{', '.join(VOCAB[i] for i in ex.most_similar(_E_svd, STOI[w], k=6)[0])} |" for w in _shared)
    mo.vstack([mo.md(f"SVD of the {len(VOCAB):,} × {len(VOCAB):,} PPMI matrix: {_secs:.1f} s\n\n"
                     "| method | analogies (top 1) | analogies (top 5) |\n|---|---|---|\n" + "\n".join(_rows)
                     + f"\n\nshared top-10 neighbours, averaged over the 300 most frequent words: "
                       f"**{np.mean(_overlap):.0%}**"),
               mo.md("| word | word2vec | PPMI + SVD |\n|---|---|---|\n" + _nb)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** Levy & Goldberg (2014) showed that skip-gram with $k$ negatives, trained to
    convergence, factorizes the matrix $\text{PMI}(w, c) - \log k$. how does that explain what
    you see? what does word2vec offer that the SVD doesn't (think about a vocabulary of a
    million words, and about streaming data)?
    """)
    return


if __name__ == "__main__":
    app.run()
