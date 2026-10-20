import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.sampling import categorical  # yours, from topic 02

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # small CPU config (offline, tests, verify) vs full config (the real tiny Shakespeare)
    OFFLINE = datasets.is_offline()
    N_MAX = 5 if OFFLINE else 6
    K_GRID = [1.0, 0.1, 0.01]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 04 · capstone: an n-gram text generator (reference solution)

    read [`brief.md`](brief.md) first. you'll build a character-level language model from
    nothing but counting and conditioning: an **n-gram model** predicts the next character from
    the previous $n - 1$, which makes it a Markov chain on strings (topic 07). you'll measure
    it with **perplexity** and generate text from it with your own `categorical` sampler.
    data loading, the train/validation/test split and the evaluation loop are written for you.

    | stage | you build |
    |---|---|
    | 1 | `build_vocab`, `encode`, `decode`: characters ↔ integer ids |
    | 2 | `NGramLM.__init__`, `NGramLM.fit`: count every (context, next character) pair |
    | 3 | `NGramLM.next_token_probs`, `NGramLM.perplexity`: add-k smoothed conditionals, and how surprised the model is |
    | 4 | `NGramLM.generate`: sample text, with a temperature |

    when the checks are green, scroll down to run the experiments and write up what you find.
    """)
    return


@app.function
def build_vocab(text: str) -> list:
    """stage 1: the sorted list of distinct characters in text. its index is a character's id."""
    return sorted(set(text))


@app.function
def encode(text: str, vocab: list) -> np.ndarray:
    """stage 1: the ids (int64 array, shape (len(text),)) of the characters of text.

    raises ValueError if text contains a character that isn't in vocab.
    """
    index = {ch: i for i, ch in enumerate(vocab)}
    try:
        return np.array([index[ch] for ch in text], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"character {e.args[0]!r} is not in the vocabulary") from None


@app.function
def decode(ids, vocab: list) -> str:
    """stage 1: the string whose character ids are ids (the inverse of encode)."""
    return "".join(vocab[int(i)] for i in ids)


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", build_vocab=build_vocab, encode=encode, decode=decode)
    return


@app.class_definition
class NGramLM:
    """A character n-gram language model with add-k smoothing."""

    def __init__(self, n: int, vocab_size: int, k: float = 1.0):
        """stage 2: an untrained model of order n >= 1 over ids 0 .. vocab_size - 1.

        stores n, vocab_size and k (the add-k pseudo-count), plus two empty dicts:
          counts: context tuple (the n - 1 previous ids) -> dict {next id: count}
          totals: context tuple -> total count of that context
        raises ValueError if n < 1.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n, self.vocab_size, self.k = n, vocab_size, k
        self.counts, self.totals = {}, {}

    def fit(self, ids):
        """stage 2: count, for every position t >= n - 1, the context ids[t - n + 1 : t] (a tuple of
        n - 1 ints; the empty tuple when n = 1) followed by the token ids[t].

        replaces any previous counts. returns self.
        """
        ids = [int(i) for i in ids]
        self.counts, self.totals = {}, {}
        m = self.n - 1
        for t in range(m, len(ids)):
            ctx = tuple(ids[t - m:t])
            nxt = self.counts.setdefault(ctx, {})
            nxt[ids[t]] = nxt.get(ids[t], 0) + 1
            self.totals[ctx] = self.totals.get(ctx, 0) + 1
        return self

    def _context(self, context) -> tuple:
        """the last n - 1 ids of context, as a tuple of ints (raises ValueError if it's too short)."""
        m = self.n - 1
        if m == 0:
            return ()
        if len(context) < m:
            raise ValueError(f"an order-{self.n} model needs at least {m} context ids, got {len(context)}")
        return tuple(int(i) for i in context[len(context) - m:])

    def next_token_probs(self, context) -> np.ndarray:
        """stage 3: P(next id | context), a (vocab_size,) array summing to 1.

        only the last n - 1 ids of context matter (it must have at least n - 1; raise ValueError
        otherwise). add-k smoothing: (count(ctx, j) + k) / (total(ctx) + k * vocab_size).
        if that denominator is 0 (an unseen context with k = 0), return the uniform distribution.
        """
        ctx = self._context(context)
        V = self.vocab_size
        probs = np.full(V, float(self.k))
        for j, c in self.counts.get(ctx, {}).items():
            probs[j] += c
        denom = self.totals.get(ctx, 0) + self.k * V
        return probs / denom if denom > 0 else np.full(V, 1.0 / V)

    def perplexity(self, ids, start=None) -> float:
        """stage 3: exp(-mean log P(ids[t] | the n - 1 ids before t)) over t = start .. len(ids) - 1.

        start defaults to n - 1 (the first position with a full context); it must be >= n - 1.
        pass the same start to compare models of different orders on the same tokens.
        returns float('inf') if some token has probability 0. a Python float.
        """
        ids = [int(i) for i in ids]
        start = self.n - 1 if start is None else start
        if start < self.n - 1:
            raise ValueError(f"start must be at least n - 1 = {self.n - 1}")
        V, k, m = self.vocab_size, self.k, self.n - 1
        total_log = 0.0
        for t in range(start, len(ids)):
            ctx = tuple(ids[t - m:t])
            denom = self.totals.get(ctx, 0) + k * V
            if denom == 0:
                p = 1.0 / V
            else:
                p = (self.counts.get(ctx, {}).get(ids[t], 0) + k) / denom
            if p == 0:
                return float("inf")
            total_log += math.log(p)
        return math.exp(-total_log / (len(ids) - start))

    def generate(self, prompt, n_tokens: int, *, temperature: float = 1.0, rng=None) -> np.ndarray:
        """stage 4: extend prompt (>= n - 1 ids) by n_tokens sampled ids; returns all of them (int64).

        each new id is drawn from next_token_probs(everything so far) sharpened by the
        temperature, p_j proportional to p_j ** (1 / temperature) (compute it from log p, minus its
        max, to avoid underflow), using YOUR mlfs.sampling.categorical. temperature == 0 means
        greedy: always take the most likely id (the smallest id on ties).
        """
        rng = np.random.default_rng(rng)
        out = [int(i) for i in prompt]
        self._context(out)  # validates the prompt length
        for _ in range(n_tokens):
            p = self.next_token_probs(out)
            if temperature == 0:
                nxt = int(np.argmax(p))
            else:
                with np.errstate(divide="ignore"):
                    logits = np.log(p) / temperature
                w = np.exp(logits - logits.max())
                nxt = categorical(w / w.sum(), rng=rng)
            out.append(nxt)
        return np.array(out, dtype=np.int64)


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", NGramLM=NGramLM)
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", NGramLM=NGramLM)
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", NGramLM=NGramLM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    tiny Shakespeare: about 1.1 million characters of plays, as one string. offline,
    `mlfs.datasets` substitutes a synthetic text of random common English words with the same
    format, so everything below still runs (and says something sensible). the text is split by
    position: the first 80% for training, the next 10% for validation (choosing $n$ and $k$),
    and the last 10% for the final test number.
    """)
    return


@app.cell
def _():
    text = datasets.tiny_shakespeare()
    vocab = build_vocab(text)
    ids_all = encode(text, vocab)
    _a, _b = int(0.8 * len(ids_all)), int(0.9 * len(ids_all))
    train_ids, val_ids, test_ids = ids_all[:_a], ids_all[_a:_b], ids_all[_b:]
    mo.md(f"{'**synthetic stand-in (offline)**' if OFFLINE else '**tiny Shakespeare**'}: "
          f"{len(text):,} characters, vocabulary of {len(vocab)} · train {len(train_ids):,} · "
          f"validation {len(val_ids):,} · test {len(test_ids):,}\n\n```\n{text[:300]}\n```")
    return test_ids, train_ids, val_ids, vocab


@app.cell
def _(train_ids, val_ids, vocab):
    # evaluation (given): fit each order once, then score every k on the validation set. every model
    # is scored on the same tokens (from position N_MAX - 1 on), so the numbers are comparable.
    _V = len(vocab)
    models, results = {}, {}
    _t0 = time.perf_counter()
    for _n in range(1, N_MAX + 1):
        models[_n] = NGramLM(_n, _V).fit(train_ids)
        for _k in K_GRID:
            models[_n].k = _k
            results[(_n, _k)] = (models[_n].perplexity(train_ids[:200_000], start=N_MAX - 1),
                                 models[_n].perplexity(val_ids, start=N_MAX - 1))
    _secs = time.perf_counter() - _t0
    best = min(results, key=lambda nk: results[nk][1])
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _k in K_GRID:
        _ns = list(range(1, N_MAX + 1))
        _ax.semilogy(_ns, [results[(_n, _k)][1] for _n in _ns], "o-", label=f"validation, k = {_k}")
        _ax.semilogy(_ns, [results[(_n, _k)][0] for _n in _ns], ":", alpha=0.6, label=f"train, k = {_k}")
    _ax.axhline(_V, color="gray", ls="--", lw=0.8, label=f"uniform guess ({_V})")
    _ax.set(xlabel="order n", ylabel="perplexity")
    _ax.legend(fontsize=7)
    _rows = "\n".join(f"| {_n} | " + " | ".join(f"{results[(_n, _k)][1]:.3f}" for _k in K_GRID) + " |"
                      for _n in range(1, N_MAX + 1))
    mo.vstack([_fig, mo.md(
        f"fitted orders 1–{N_MAX} in {_secs:.1f} s. **best on validation: n = {best[0]}, k = {best[1]}** "
        f"(perplexity {results[best][1]:.3f}).\n\n| n | " + " | ".join(f"val, k = {_k}" for _k in K_GRID)
        + " |\n|---|" + "---|" * len(K_GRID) + "\n" + _rows)])
    return best, models


@app.cell
def _(best, models, test_ids, vocab):
    # the final number (given): the chosen model, on the test set, once, with baselines.
    _m = models[best[0]]
    _m.k = best[1]
    _test = _m.perplexity(test_ids, start=N_MAX - 1)
    _uni = models[1]
    _uni.k = 1.0
    _unigram = _uni.perplexity(test_ids, start=N_MAX - 1)
    _bi = models[2]
    _bi.k = 1.0
    _bigram = _bi.perplexity(test_ids, start=N_MAX - 1)
    mo.md(f"""
    **test perplexity** of the chosen model (n = {best[0]}, k = {best[1]}): **{_test:.3f}**
    ({math.log2(_test):.3f} bits per character).

    | baseline | test perplexity |
    |---|---|
    | uniform over {len(vocab)} characters | {len(vocab):.3f} |
    | unigram (n = 1, k = 1) | {_unigram:.3f} |
    | bigram (n = 2, k = 1) | {_bigram:.3f} |
    """)
    return


@app.cell
def _():
    temperature = mo.ui.slider(steps=[0.0, 0.3, 0.6, 1.0, 1.5], value=0.6, label="temperature", show_value=True)
    temperature
    return (temperature,)


@app.cell
def _(best, models, temperature, train_ids, vocab):
    # samples (given), plus a copying check: how many of the generated 12-character chunks appear
    # verbatim in the training text?
    _m = models[best[0]]
    _m.k = best[1]
    _prompt = train_ids[:max(1, best[0] - 1)]
    _out = _m.generate(_prompt, 400, temperature=temperature.value, rng=0)
    _sample = decode(_out, vocab)
    _train_text = decode(train_ids[:300_000], vocab)
    _chunks = [_sample[_i:_i + 12] for _i in range(0, len(_sample) - 12, 6)]
    _copied = np.mean([_c in _train_text for _c in _chunks]) if _chunks else 0.0
    mo.md(f"n = {best[0]}, k = {best[1]}, temperature {temperature.value}:\n\n```\n{_sample}\n```\n\n"
          f"fraction of 12-character chunks found verbatim in (the first 300k characters of) the "
          f"training text: **{_copied:.2f}**")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **results.** the chosen $n$ and $k$ (picked on validation), the test perplexity, and the
       baselines. convert perplexity to bits per character. how much of the gap between uniform
       and your model comes from the unigram, and how much from context?
    2. **the overfitting curve.** describe train vs validation perplexity as $n$ grows, for each
       $k$. where does validation perplexity bottom out, and why? relate it to the number of
       distinct contexts versus the amount of training text.
    3. **ablation.** pick one: $k = 0$ (what happens, and why?), a different train/validation
       split (e.g. random positions instead of contiguous blocks), or a word-level model.
    4. **samples.** show samples at temperatures 0, 0.6 and 1.5. explain what greedy decoding
       does to an n-gram model, and use the copying check to discuss memorization at large $n$.
    5. **validation vs test, and the limit theorems.** perplexity is the exponential of an
       average log loss over about 110,000 characters. if those characters were i.i.d. draws
       from one distribution, the LLN and CLT would put validation and test within about 1% of
       each other. on tiny Shakespeare they differ by about 25% at every order. why? (the blocks
       are contiguous: the test block is different plays, with different characters' names,
       topics and style.) what does an i.i.d. error bar miss, and what would a fairer estimate
       of "perplexity on new Shakespeare" look like?
    6. **probability.** where did the other topics of this chapter enter: conditioning and
       Bayes' rule, sampling, expectation (perplexity is the exponential of an expected log
       loss), and Markov chains?
    """)
    return


if __name__ == "__main__":
    app.run()
