import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    from collections import Counter, defaultdict

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00

    BOS, EOS, UNK = "<s>", "</s>", "<unk>"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # language modeling · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll build **`NGramLM`**, a word- or
    character-level n-gram language model with add-k and interpolated smoothing, the first
    piece of **`mlfs.text`**. chapter 04's capstone counted characters; this one handles any
    tokens, sentence boundaries, unseen words and mixtures of orders, and tunes its mixture
    weights with EM.

    | stage | you build |
    |---|---|
    | 1 | `NGramLM.__init__`, `NGramLM.fit`: pad every sequence and count n-grams of every order up to n |
    | 2 | `NGramLM.logprob` with add-k smoothing |
    | 3 | interpolated `logprob`, and `NGramLM.perplexity` |
    | 4 | `tune_lambdas`: fit the interpolation weights on held-out text with EM |
    | 5 | `NGramLM.sample`: generate tokens |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class NGramLM:
    """An n-gram language model over any hashable tokens, with add-k or interpolated smoothing.

    after fit: `counts_[j]` maps each context tuple of length j (j = 0 .. n-1) to a Counter of the
    tokens that followed it, `totals_[j]` maps the same contexts to their total count, and `vocab_`
    is the sorted list of training tokens plus '</s>' and '<unk>' (never '<s>', which is never
    predicted). `lambdas[j]` weighs the estimate that uses a context of length j.
    """

    def __init__(self, n=3, *, smoothing="interpolated", add_k=1.0, lambdas=None):
        """stage 1: store the arguments; ValueError if n < 1, smoothing is unknown, or add_k <= 0."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if smoothing not in ("add_k", "interpolated"):
            raise ValueError(f"smoothing must be 'add_k' or 'interpolated', got {smoothing!r}")
        if add_k <= 0:
            raise ValueError(f"add_k must be > 0 (it keeps every probability positive), got {add_k}")
        self.n = n
        self.smoothing = smoothing
        self.add_k = add_k
        self.lambdas = lambdas

    def fit(self, sequences):
        """stage 1: pad each sequence with n-1 '<s>' and one '</s>'; fill counts_, totals_, vocab_; return self."""
        n = self.n
        counts = [defaultdict(Counter) for _ in range(n)]
        vocab = {EOS, UNK}
        for seq in sequences:
            seq = list(seq)
            vocab.update(seq)
            padded = [BOS] * (n - 1) + seq + [EOS]
            for t in range(n - 1, len(padded)):
                for j in range(n):
                    counts[j][tuple(padded[t - j:t])][padded[t]] += 1
        self.counts_ = [dict(c) for c in counts]
        self.totals_ = [{ctx: sum(c.values()) for ctx, c in cj.items()} for cj in self.counts_]
        self.vocab_ = sorted(vocab, key=str)
        return self

    def _known(self, token):
        """is token in vocab_? (a set, rebuilt whenever vocab_ changes)"""
        cache = getattr(self, "_vocab_cache", None)
        if cache is None or cache[0] is not self.vocab_:
            cache = self._vocab_cache = (self.vocab_, set(self.vocab_))
        return token in cache[1]

    def _context(self, context):
        """the last n-1 tokens of context as a tuple, unknown tokens -> '<unk>', left-padded with '<s>'."""
        m = self.n - 1
        if m == 0:
            return ()
        ctx = tuple(context)[-m:] if len(context) else ()
        ctx = tuple(t if t == BOS or self._known(t) else UNK for t in ctx)
        return (BOS,) * (m - len(ctx)) + ctx

    def prob_add_k(self, ctx, token):
        """stage 2: (count(ctx, token) + k) / (total(ctx) + k |V|) for a context ctx of exactly n-1 tokens."""
        c = self.counts_[self.n - 1].get(ctx)
        num = (c.get(token, 0) if c else 0) + self.add_k
        return num / (self.totals_[self.n - 1].get(ctx, 0) + self.add_k * len(self.vocab_))

    def prob_interpolated(self, ctx, token):
        """stage 3: sum_j lambdas[j] * P_j(token | last j tokens of ctx): add-k unigram, unseen contexts back off."""
        lam = [1.0 / self.n] * self.n if self.lambdas is None else [float(x) for x in self.lambdas]
        if len(lam) != self.n or min(lam) < 0 or not math.isclose(sum(lam), 1.0, abs_tol=1e-6):
            raise ValueError(f"lambdas must be {self.n} non-negative weights summing to 1, got {lam}")
        V = len(self.vocab_)
        p = (self.counts_[0][()].get(token, 0) + self.add_k) / (self.totals_[0][()] + self.add_k * V)
        total = lam[0] * p
        for j in range(1, self.n):
            sub = ctx[len(ctx) - j:]
            n_ctx = self.totals_[j].get(sub, 0)
            if n_ctx:  # an unseen context keeps the estimate of the order below
                p = self.counts_[j][sub].get(token, 0) / n_ctx
            total += lam[j] * p
        return total

    def logprob(self, context, token):
        """natural log P(token | the last n-1 tokens of context) (-inf if 0); unseen tokens count as '<unk>'."""
        ctx = self._context(context)
        token = token if self._known(token) else UNK
        p = self.prob_add_k(ctx, token) if self.smoothing == "add_k" else self.prob_interpolated(ctx, token)
        return math.log(p) if p > 0 else -math.inf

    def perplexity(self, sequences):
        """stage 3: exp(-mean logprob) over every token of every padded sequence, '</s>' included."""
        total, count = 0.0, 0
        for seq in sequences:
            padded = [BOS] * (self.n - 1) + list(seq) + [EOS]
            for t in range(self.n - 1, len(padded)):
                total += self.logprob(padded[t - self.n + 1:t], padded[t])
                count += 1
        if count == 0:
            raise ValueError("perplexity needs at least one sequence")
        return math.exp(-total / count)

    def sample(self, n_tokens, *, context=(), rng=None):
        """stage 5: n_tokens tokens drawn one at a time from P(. | history), never '<unk>'; '</s>' restarts the history."""
        rng = as_rng(rng)
        cands = [t for t in self.vocab_ if t != UNK]
        history = list(context)
        out = []
        for _ in range(n_tokens):
            logp = np.array([self.logprob(history, t) for t in cands])
            p = np.exp(logp - logp.max())
            tok = cands[int(rng.choice(len(cands), p=p / p.sum()))]
            out.append(tok)
            history = [] if tok == EOS else history + [tok]
        return out


@app.cell(hide_code=True)
def _():
    checks.run(1, NGramLM=NGramLM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.cell(hide_code=True)
def _():
    checks.run(2, NGramLM=NGramLM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, NGramLM=NGramLM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def tune_lambdas(model, sequences, *, n_iters=100, tol=1e-6):
    """stage 4: EM for the interpolation weights on held-out sequences; sets model.lambdas, returns (lambdas, loglik history)."""
    if model.smoothing != "interpolated":
        raise ValueError(f"tune_lambdas needs smoothing='interpolated', got {model.smoothing!r}")
    n = model.n
    columns = []
    for j in range(n):  # P_j for every held-out token: score with all the weight on order j
        model.lambdas = np.eye(n)[j]
        columns.append([model.prob_interpolated(model._context(padded[t - n + 1:t]),
                                                padded[t] if model._known(padded[t]) else UNK)
                        for seq in sequences
                        for padded in [[BOS] * (n - 1) + list(seq) + [EOS]]
                        for t in range(n - 1, len(padded))])
    P = np.array(columns).T  # (N tokens, n orders)
    lam = np.full(n, 1.0 / n)
    history = []
    for _ in range(n_iters):
        mix = P @ lam
        history.append(float(np.log(mix).sum()))
        if len(history) > 1 and history[-1] - history[-2] < tol * len(P):
            break
        lam = (P * lam / mix[:, None]).mean(axis=0)  # E step (responsibilities) + M step (their mean)
    model.lambdas = lam
    return lam, history


@app.cell(hide_code=True)
def _():
    checks.run(4, NGramLM=NGramLM, tune_lambdas=tune_lambdas)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.cell(hide_code=True)
def _():
    checks.run(5, NGramLM=NGramLM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.text import NGramLM` gives you this model; the chapter's capstone uses it as
    the baseline your LSTM has to beat.
    """)
    return


if __name__ == "__main__":
    app.run()
