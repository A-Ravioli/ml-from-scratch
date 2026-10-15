"""Checks for exercise.py (language modeling).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Counting is checked
against a brute-force n-gram count, the smoothed probabilities against the lesson's worked example
and against normalisation (every conditional sums to 1 over the vocabulary, seen context or not),
perplexity against its definition, EM against a grid search and its monotone likelihood, and
sampling against a deterministic corpus and long-run frequencies.
"""
import math
from collections import Counter

import numpy as np
import pytest

from mlfs.testing import bounded

CORPUS = [["the", "cat", "sat"], ["the", "dog", "sat"]]


def _random_corpus(seed, n_seq=40, vocab="abcde", max_len=8):
    rng = np.random.default_rng(seed)
    return [list(rng.choice(list(vocab), size=int(rng.integers(1, max_len)))) for _ in range(n_seq)]


def _markov_corpus(seed, n_seq=200, length=12):
    """sequences from a sticky order-1 chain over 4 symbols: strong sequential structure."""
    rng = np.random.default_rng(seed)
    T = np.full((4, 4), 0.05) + np.eye(4) * 0.8
    T /= T.sum(1, keepdims=True)
    out = []
    for _ in range(n_seq):
        s = [int(rng.integers(4))]
        for _ in range(length - 1):
            s.append(int(rng.choice(4, p=T[s[-1]])))
        out.append(["wxyz"[i] for i in s])
    return out


def _fit(nb, n, corpus=CORPUS, **kw):
    return bounded(nb.NGramLM(n, **kw).fit, corpus, what="NGramLM.fit")


# ---------------------------------------------------------------- stage 1: counting


def test_stage1_init_stores_and_validates(nb):
    lm = nb.NGramLM(4, smoothing="add_k", add_k=0.5, lambdas=(0.1, 0.2, 0.3, 0.4))
    assert (lm.n, lm.smoothing, lm.add_k, tuple(lm.lambdas)) == (4, "add_k", 0.5, (0.1, 0.2, 0.3, 0.4)), (
        "__init__ stores n, smoothing, add_k and lambdas as same-named attributes")
    d = nb.NGramLM()
    assert (d.n, d.smoothing, d.add_k, d.lambdas) == (3, "interpolated", 1.0, None), "defaults: n=3, interpolated"
    for bad in [dict(n=0), dict(smoothing="kneser"), dict(add_k=0.0)]:
        with pytest.raises(ValueError):
            nb.NGramLM(**{"n": 2, **bad})


def test_stage1_counts_worked_example(nb):
    lm = _fit(nb, 2)
    assert lm.fit(CORPUS) is lm, "fit returns self, so you can write NGramLM(2).fit(corpus)"
    assert dict(lm.counts_[1][("the",)]) == {"cat": 1, "dog": 1}, (
        f"after 'the' come 'cat' once and 'dog' once; got {lm.counts_[1].get(('the',))}")
    assert dict(lm.counts_[1][("<s>",)]) == {"the": 2}, "every sequence starts with n-1 = 1 '<s>' of padding"
    assert dict(lm.counts_[1][("sat",)]) == {"</s>": 2}, "and ends with one '</s>', which is a token you predict"
    assert dict(lm.counts_[0][()]) == {"the": 2, "cat": 1, "dog": 1, "sat": 2, "</s>": 2}, (
        f"order 0 (empty context) counts every predicted token, '</s>' included and '<s>' never; "
        f"got {dict(lm.counts_[0][()])}")
    assert lm.totals_[0][()] == 8 and lm.totals_[1][("the",)] == 2, "totals_ holds each context's total count"
    assert list(lm.vocab_) == sorted(["</s>", "<unk>", "cat", "dog", "sat", "the"]), (
        f"vocab_ = sorted training tokens plus '</s>' and '<unk>' (no '<s>'); got {lm.vocab_}")


def test_stage1_counts_match_brute_force(nb):
    for n in (1, 3, 4):
        corpus = _random_corpus(n)
        lm = _fit(nb, n, corpus)
        for j in range(n):
            want = Counter()
            for seq in corpus:
                p = ["<s>"] * (n - 1) + list(seq) + ["</s>"]
                for t in range(n - 1, len(p)):
                    want[(tuple(p[t - j:t]), p[t])] += 1
            got = Counter({(ctx, tok): c for ctx, cnt in lm.counts_[j].items() for tok, c in cnt.items()})
            assert got == want, (
                f"n={n}: counts_[{j}] (contexts of length {j}) disagree with a brute-force count. every position "
                f"t >= n-1 of the padded sequence contributes one count at every order j = 0..n-1, with context "
                f"padded[t-j:t]")
            for ctx, cnt in lm.counts_[j].items():
                assert lm.totals_[j][ctx] == sum(cnt.values()), "totals_[j][ctx] == sum of counts_[j][ctx]"


def test_stage1_any_hashable_tokens_and_refit(nb):
    lm = _fit(nb, 2, ["abca", "bb"])  # strings are sequences of characters
    assert dict(lm.counts_[1][("b",)]) == {"c": 1, "b": 1, "</s>": 1}, "a string is a sequence of character tokens"
    lm2 = _fit(nb, 2, [[1, 2, 3], [2, 3]])
    assert dict(lm2.counts_[1][(2,)]) == {3: 2}, "tokens can be any hashable value, e.g. ints"
    lm2.fit([[7, 8]])
    assert (2,) not in lm2.counts_[1], "fit replaces previous counts, it doesn't add to them"


# ---------------------------------------------------------------- stage 2: add-k


def test_stage2_add_k_worked_example(nb):
    lm = _fit(nb, 2, smoothing="add_k", add_k=1.0)
    got = {t: math.exp(lm.logprob(["the"], t)) for t in ["cat", "sat", "<unk>"]}
    assert np.isclose(got["cat"], 0.25), (
        f"P(cat | the) = (1 + 1) / (2 + 1 * 6) = 0.25 with |V| = 6 ('</s>' and '<unk>' count); got {got['cat']:.4f}")
    assert np.isclose(got["sat"], 0.125), f"an unseen continuation gets (0 + 1) / 8; got {got['sat']:.4f}"
    assert np.isclose(math.exp(lm.logprob(["the"], "zebra")), 0.125), "a never-seen token is scored as '<unk>'"
    assert np.isclose(math.exp(lm.logprob(["zebra"], "cat")), 1 / 6), (
        "an unseen context has total 0, so every token gets k / (k |V|) = 1/|V|")


def test_stage2_normalised_and_uses_last_n_minus_1(nb):
    corpus = _random_corpus(3)
    lm = _fit(nb, 3, corpus, smoothing="add_k", add_k=0.3)
    rng = np.random.default_rng(4)
    for _ in range(10):
        ctx = list(rng.choice(list("abcdeq"), size=int(rng.integers(0, 4))))
        s = sum(math.exp(lm.logprob(ctx, t)) for t in lm.vocab_)
        assert np.isclose(s, 1.0), f"P(. | {ctx}) must sum to 1 over vocab_; got {s:.6f}"
    assert lm.logprob(["e", "e", "a", "b"], "c") == lm.logprob(["a", "b"], "c"), "only the last n-1 tokens matter"
    assert lm.logprob(["a"], "c") == lm.logprob(["<s>", "a"], "c"), "a short context is left-padded with '<s>'"


def test_stage2_small_k_approaches_counts(nb):
    lm = _fit(nb, 2, smoothing="add_k", add_k=1e-9)
    assert np.isclose(math.exp(lm.logprob(["the"], "cat")), 0.5, atol=1e-6), (
        "as k -> 0, add-k becomes the maximum-likelihood estimate count(the cat) / count(the) = 1/2")


# ---------------------------------------------------------------- stage 3: interpolation, perplexity


def test_stage3_interpolated_worked_example(nb):
    lm = _fit(nb, 2)
    p = math.exp(lm.logprob(["the"], "cat"))
    assert np.isclose(p, 0.5 / 7 + 0.25), (
        f"P(cat | the) = 0.5 * (1 + 1)/(8 + 6) + 0.5 * 1/2 = 0.3214 (uniform lambdas, add-1 unigram); got {p:.4f}")
    assert np.isclose(math.exp(lm.logprob(["the"], "<unk>")), 0.5 / 14), "'<unk>' gets mass only from the unigram"
    assert np.isclose(lm.perplexity([["the", "cat", "sat"]]), 1.930904, atol=1e-5), (
        f"perplexity of [the cat sat] is exp(-mean of 4 log-probs, '</s>' included) = 1.9309; "
        f"got {lm.perplexity([['the', 'cat', 'sat']]):.4f}")
    lk = _fit(nb, 2, smoothing="add_k")
    assert np.isclose(lk.perplexity([["the", "cat", "sat"]]), 3.158758, atol=1e-5), "add-1 perplexity is 3.1588"


def test_stage3_normalised_with_backoff(nb):
    corpus = _random_corpus(5)
    rng = np.random.default_rng(6)
    for n in (2, 3, 4):
        lam = rng.dirichlet(np.ones(n))
        lm = _fit(nb, n, corpus, lambdas=lam, add_k=0.5)
        for _ in range(8):
            ctx = list(rng.choice(list("abcdeq"), size=int(rng.integers(0, 5))))
            s = sum(math.exp(lm.logprob(ctx, t)) for t in lm.vocab_)
            assert np.isclose(s, 1.0), (
                f"n={n}: the interpolated P(. | {ctx}) sums to {s:.6f}, not 1. when a context was never seen, "
                f"its order must reuse the estimate of the order below (not contribute 0)")


def test_stage3_unseen_context_backs_off(nb):
    lm = _fit(nb, 3, lambdas=(0.2, 0.3, 0.5))
    p0 = (2 + 1) / (8 + 6)  # add-1 unigram of 'sat'
    p1 = 1.0  # P(sat | cat)
    got = math.exp(lm.logprob(["dog", "cat"], "sat"))
    assert np.isclose(got, 0.2 * p0 + 0.3 * p1 + 0.5 * p1), (
        f"('dog', 'cat') was never seen, so the trigram estimate falls back to P(sat | cat) = 1: expected "
        f"0.2 * 3/14 + 0.8 * 1 = {0.2 * p0 + 0.8:.4f}, got {got:.4f}")
    with pytest.raises(ValueError):
        _fit(nb, 3, lambdas=(0.5, 0.5, 0.5)).logprob(["the"], "cat")


def test_stage3_perplexity_definition(nb):
    corpus = _random_corpus(7)
    held = _random_corpus(8, n_seq=10)
    lm = _fit(nb, 3, corpus, lambdas=(0.3, 0.3, 0.4))
    lp, n_tok = 0.0, 0
    for seq in held:
        p = ["<s>", "<s>"] + list(seq) + ["</s>"]
        for t in range(2, len(p)):
            lp += lm.logprob(p[:t], p[t])
            n_tok += 1
    got = bounded(lm.perplexity, held, what="perplexity")
    assert np.isclose(got, math.exp(-lp / n_tok)), (
        f"perplexity = exp(-(sum of log-probs) / (number of tokens incl. one '</s>' per sequence)); expected "
        f"{math.exp(-lp / n_tok):.4f}, got {got:.4f}")
    uni = _fit(nb, 1, [["a", "b"]], smoothing="add_k", add_k=1e6)
    assert np.isclose(uni.perplexity([["a", "b", "a"]]), 4.0, rtol=1e-4), (
        "a (nearly) uniform model over |V| = 4 tokens has perplexity 4")


# ---------------------------------------------------------------- stage 4: EM for the lambdas


def test_stage4_em_monotone_and_valid(nb):
    train, held = _markov_corpus(0), _markov_corpus(1, n_seq=60)
    lm = _fit(nb, 3, train)
    lam, hist = bounded(nb.tune_lambdas, lm, held, what="tune_lambdas")
    lam = np.asarray(lam, dtype=float)
    assert lam.shape == (3,) and np.all(lam >= 0) and np.isclose(lam.sum(), 1.0), f"3 weights summing to 1: {lam}"
    assert np.allclose(np.asarray(lm.lambdas, dtype=float), lam), "tune_lambdas sets model.lambdas"
    d = np.diff(hist)
    assert len(hist) >= 2 and np.all(d >= -1e-8), (
        f"EM never decreases the held-out log-likelihood; got steps {d[:5]}. the E step needs the responsibilities "
        f"lambda_j P_j / sum_i lambda_i P_i, the M step their mean over tokens")
    ll = sum(lm.logprob(p[:t], p[t]) for s in held for p in [["<s>"] * 2 + s + ["</s>"]] for t in range(2, len(p)))
    assert np.isclose(ll, hist[-1], rtol=1e-3), "history[-1] is the held-out log-likelihood (natural log)"


def test_stage4_em_matches_grid_search(nb):
    train, held = _markov_corpus(2), _markov_corpus(3, n_seq=60)
    lm = _fit(nb, 2, train)
    lam, _ = bounded(nb.tune_lambdas, lm, held, what="tune_lambdas")
    best = -np.inf
    for a in np.linspace(0.001, 0.999, 300):
        lm.lambdas = (a, 1 - a)
        best = max(best, -math.log(lm.perplexity(held)))
    lm.lambdas = lam
    assert -math.log(lm.perplexity(held)) >= best - 1e-4, (
        f"EM's lambdas {np.round(lam, 3)} should be (at least) as good on held-out as the best of a fine grid")


def test_stage4_weights_follow_the_structure(nb):
    train = _markov_corpus(4)
    lm = _fit(nb, 3, train)
    lam, _ = bounded(nb.tune_lambdas, lm, _markov_corpus(5, n_seq=60), what="tune_lambdas")
    assert lam[0] < 0.2, f"held-out text from the same sticky chain: context helps, the unigram weight is small: {lam}"
    rng = np.random.default_rng(6)
    shuffled = [list(rng.permutation(sum(_markov_corpus(7, n_seq=60), [])))[i:i + 12] for i in range(0, 720, 12)]
    lam2, _ = bounded(nb.tune_lambdas, lm, shuffled, what="tune_lambdas")
    assert lam2[0] > lam[0] + 0.2, (
        f"on shuffled held-out tokens the context is useless, so EM should move weight to the unigram: {lam2}")
    with pytest.raises(ValueError):
        nb.tune_lambdas(_fit(nb, 2, train, smoothing="add_k"), train[:5])


# ---------------------------------------------------------------- stage 5: sampling


def test_stage5_deterministic_corpus(nb):
    lm = _fit(nb, 3, [["a", "b", "c"]] * 5, lambdas=(0.0, 0.0, 1.0))
    out = bounded(lm.sample, 12, rng=0, what="sample")
    assert list(out) == ["a", "b", "c", "</s>"] * 3, (
        f"with all the weight on the trigram, the only continuation of every context is certain; expected "
        f"a b c </s> repeated (after '</s>' the history restarts from '<s> <s>'), got {out}")
    assert list(lm.sample(2, context=["a"], rng=1)) == ["b", "c"], "context sets the starting history"


def test_stage5_reproducible_and_never_unk(nb):
    lm = _fit(nb, 2, _random_corpus(9))
    a, b = lm.sample(40, rng=3), lm.sample(40, rng=3)
    assert list(a) == list(b), "the same rng seed gives the same sample"
    assert list(lm.sample(40, rng=4)) != list(a), "different seeds give different samples"
    assert len(a) == 40 and "<unk>" not in a and "<s>" not in a, "n_tokens tokens, never '<unk>' or '<s>'"
    flat = _fit(nb, 1, [["a"]], smoothing="add_k", add_k=1e6)  # '</s>', '<unk>' and 'a' almost equally likely
    out = flat.sample(300, rng=5)
    assert "<unk>" not in out and 100 < out.count("a") < 200, (
        "'<unk>' is a placeholder for unseen tokens, not a token to generate: leave it out of the candidates and "
        "renormalize (here 'a' and '</s>' should then each come up about half the time)")


def test_stage5_frequencies(nb):
    lm = _fit(nb, 1, [["a"] * 6 + ["b"] * 3], smoothing="add_k", add_k=1e-9)
    out = Counter(lm.sample(3000, rng=5))
    want = {"a": 0.6, "b": 0.3, "</s>": 0.1}
    for t, p in want.items():
        assert abs(out[t] / 3000 - p) < 0.035, (
            f"a unigram model with P = {want} should emit {t!r} about {p:.0%} of the time; got {out[t] / 3000:.3f}")
