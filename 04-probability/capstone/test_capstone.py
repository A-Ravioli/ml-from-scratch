"""Checks for capstone.py (the n-gram text generator): functional tests plus a perplexity threshold.

All text here is synthetic and generated deterministically: strings drawn from a known
order-2 Markov chain over six letters, whose true per-character perplexity can be computed
exactly. Nothing is downloaded.
"""
import math
from unittest import mock

import numpy as np
import pytest

ALPHABET = "abcdef"


def _chain(seed=0, length=60_000, alpha=0.3):
    """text from a random order-2 Markov chain over ALPHABET, plus its true entropy rate (nats/char)."""
    rng = np.random.default_rng(seed)
    V = len(ALPHABET)
    T = rng.dirichlet(np.full(V, alpha), size=(V, V))  # T[a, b] = P(next | previous two = a, b)
    # stationary distribution over pairs (a, b) -> (b, c)
    P = np.zeros((V * V, V * V))
    for a in range(V):
        for b in range(V):
            for c in range(V):
                P[a * V + b, b * V + c] = T[a, b, c]
    w, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(w - 1))])
    pi /= pi.sum()
    H = -sum(pi[a * V + b] * T[a, b, c] * math.log(T[a, b, c])
             for a in range(V) for b in range(V) for c in range(V) if T[a, b, c] > 0)
    ids = [0, 1]
    u = rng.random(length)
    for t in range(length - 2):
        cdf = np.cumsum(T[ids[-2], ids[-1]])
        ids.append(min(int(np.searchsorted(cdf, u[t] * cdf[-1], side="right")), V - 1))
    return "".join(ALPHABET[i] for i in ids), H, T


TEXT, H_TRUE, T_TRUE = _chain()


def _true_perplexity(ids, start, T=T_TRUE):
    """perplexity of the TRUE order-2 chain on ids[start:], each token given the two before it."""
    ids = np.asarray(ids)
    logp = np.log(T[ids[start - 2:-2], ids[start - 1:-1], ids[start:]])
    return float(np.exp(-logp.mean()))


FORBIDDEN = ("choice", "integers", "multinomial", "uniform", "normal", "standard_normal", "exponential",
             "gamma", "beta", "poisson", "binomial", "permutation", "shuffle")


class _OnlyUniform(np.random.Generator):
    """records any Generator method other than .random(): your own categorical uses only rng.random()."""

    def __init__(self, seed):
        super().__init__(np.random.PCG64(seed))
        self.used = set()

    def __getattribute__(self, name):
        if name in FORBIDDEN:
            object.__getattribute__(self, "used").add(name)
        return object.__getattribute__(self, name)


# ---------------------------------------------------------------- stage 1: vocabulary


def test_stage1_vocab_and_roundtrip(nb):
    vocab = nb.build_vocab("hello, world")
    assert list(vocab) == sorted(set("hello, world")), (
        f"the vocabulary is the SORTED list of distinct characters: {sorted(set('hello, world'))}, got {list(vocab)}")
    ids = np.asarray(nb.encode("hello", vocab))
    assert ids.dtype.kind == "i" and ids.tolist() == [vocab.index(c) for c in "hello"], (
        f"encode('hello') should give each character's index in the vocabulary, got {ids.tolist()}")
    assert nb.decode(ids, vocab) == "hello", "decode(encode(s)) should give back s"
    assert nb.decode(list(ids), vocab) == "hello", "decode should accept a list of ids too"


def test_stage1_unknown_character(nb):
    with pytest.raises(ValueError):
        nb.encode("abz", list("abc"))
        pytest.fail("'z' isn't in the vocabulary: encode should raise ValueError")


# ---------------------------------------------------------------- stage 2: counting


def test_stage2_init(nb):
    m = nb.NGramLM(3, 7, k=0.5)
    assert (m.n, m.vocab_size, m.k) == (3, 7, 0.5), "store n, vocab_size and k as attributes"
    assert m.counts == {} and m.totals == {}, "a new model has empty counts and totals"
    with pytest.raises(ValueError):
        nb.NGramLM(0, 5)
        pytest.fail("n = 0 makes no sense: raise ValueError")


def test_stage2_counts_hand_example(nb):
    vocab = list("abcdr")
    ids = nb.encode("abracadabra", vocab)
    m = nb.NGramLM(2, 5).fit(ids)
    a, b, c, d, r = range(5)
    assert m.counts[(a,)] == {b: 2, c: 1, d: 1}, (
        f"in 'abracadabra', 'a' is followed by b twice, c once and d once: counts[(a,)] should be "
        f"{{{b}: 2, {c}: 1, {d}: 1}}, got {m.counts.get((a,))}. contexts are TUPLES of ints.")
    assert m.totals[(a,)] == 4 and m.totals[(r,)] == 2, "totals[(a,)] = 4 (the last 'a' has no successor), totals[(r,)] = 2"
    assert sum(m.totals.values()) == 10, "an 11-character text has 10 (context, next) pairs for n = 2"


def test_stage2_orders_one_and_three(nb):
    ids = nb.encode("abab", list("ab"))
    uni = nb.NGramLM(1, 2).fit(ids)
    assert uni.counts == {(): {0: 2, 1: 2}}, f"for n = 1 the context is the empty tuple: {{(): {{0: 2, 1: 2}}}}, got {uni.counts}"
    tri = nb.NGramLM(3, 2).fit(ids)
    assert tri.counts == {(0, 1): {0: 1}, (1, 0): {1: 1}}, (
        f"for n = 3 on 'abab' the pairs are (a, b) -> a and (b, a) -> b, got {tri.counts}")


def test_stage2_refit_replaces(nb):
    m = nb.NGramLM(2, 2)
    m.fit(nb.encode("aaaa", list("ab")))
    m.fit(nb.encode("bbbb", list("ab")))
    assert (0,) not in m.counts and m.totals == {(1,): 3}, "fit should replace earlier counts, not add to them"
    assert m.fit(nb.encode("ab", list("ab"))) is m, "fit should return self"


# ---------------------------------------------------------------- stage 3: probabilities and perplexity


def test_stage3_add_k_hand_example(nb):
    vocab = list("abcdr")
    m = nb.NGramLM(2, 5, k=0.0).fit(nb.encode("abracadabra", vocab))
    p = np.asarray(m.next_token_probs([0]))
    assert p.shape == (5,) and np.allclose(p, [0, 0.5, 0.25, 0.25, 0]), (
        f"with k = 0, P(next | 'a') = counts / total = (0, 2, 1, 1, 0) / 4, got {np.round(p, 4).tolist()}")
    m.k = 1.0
    p = np.asarray(m.next_token_probs([3, 2, 0]))  # only the last id matters for n = 2
    assert np.allclose(p, np.array([1, 3, 2, 2, 1]) / 9), (
        f"with k = 1: (count + 1) / (4 + 5) = (1, 3, 2, 2, 1) / 9, got {np.round(p, 4).tolist()}. "
        "only the LAST n - 1 ids of the context matter.")


def test_stage3_unseen_context_and_validation(nb):
    m = nb.NGramLM(3, 4, k=0.0).fit(np.array([0, 1, 2, 3, 0, 1, 2]))
    p = np.asarray(m.next_token_probs([3, 3]))
    assert np.allclose(p, 0.25), f"an unseen context with k = 0 has no counts at all: return the uniform distribution, got {p.tolist()}"
    m.k = 0.5
    assert np.allclose(m.next_token_probs([3, 3]), 0.25), "an unseen context with k > 0 is uniform too (all pseudo-counts)"
    with pytest.raises(ValueError):
        m.next_token_probs([1])
        pytest.fail("an order-3 model needs at least 2 context ids: raise ValueError")
    assert np.allclose(nb.NGramLM(1, 3, k=1.0).fit(np.array([0, 0, 1])).next_token_probs([]), [0.5, 1 / 3, 1 / 6]), (
        "n = 1 ignores the context entirely: (2 + 1, 1 + 1, 0 + 1) / (3 + 3)")


def test_stage3_perplexity_hand_examples(nb):
    ids = np.array([0, 1] * 50)
    m = nb.NGramLM(2, 2, k=0.0).fit(ids)
    assert abs(m.perplexity(ids) - 1.0) < 1e-12, "a perfectly predictable text has perplexity 1 under the right model"
    flat = nb.NGramLM(1, 4, k=1e9).fit(np.array([0, 1, 2]))
    assert abs(flat.perplexity(np.array([0, 3, 2, 1, 1])) - 4.0) < 1e-6, (
        "a (practically) uniform model over 4 symbols has perplexity 4 on anything")
    m2 = nb.NGramLM(1, 2, k=0.0).fit(np.array([0, 0, 0, 1]))
    got = m2.perplexity(np.array([0, 1]))
    want = math.exp(-(math.log(0.75) + math.log(0.25)) / 2)
    assert abs(got - want) < 1e-12, f"unigram (3/4, 1/4) on [0, 1]: exp(-(log .75 + log .25) / 2) = {want:.4f}, got {got:.4f}"


def test_stage3_perplexity_start_and_zero_probability(nb):
    m = nb.NGramLM(2, 3, k=0.0).fit(np.array([0, 1, 0, 1]))
    assert m.perplexity(np.array([0, 2])) == float("inf"), (
        "with k = 0, the unseen pair (0 -> 2) has probability 0, so the perplexity is infinite")
    ids = np.array([2, 2, 0, 1, 0, 1])
    m.k = 1.0
    full, late = m.perplexity(ids), m.perplexity(ids, start=3)
    p = [m.next_token_probs(ids[:t])[ids[t]] for t in range(3, 6)]
    assert abs(late - math.exp(-np.mean(np.log(p)))) < 1e-12, "start=3 scores only positions 3, 4, 5"
    assert full != late, "the default start is n - 1 = 1, which scores more positions"
    with pytest.raises(ValueError):
        m.perplexity(ids, start=0)
        pytest.fail("start = 0 has no context for an order-2 model: raise ValueError")


def test_stage3_threshold_recovers_true_entropy(nb):
    vocab = nb.build_vocab(TEXT)
    ids = np.asarray(nb.encode(TEXT, vocab))
    train, val = ids[:50_000], ids[50_000:]
    ppl = {n: nb.NGramLM(n, len(vocab), k=0.1).fit(train).perplexity(val, start=3) for n in (1, 2, 3, 4)}
    # compare with the TRUE chain scored on the very same held-out characters, so the luck of which text
    # landed in the validation block cancels out and only the model's estimation error remains
    true_ppl = _true_perplexity(val, 3)
    assert ppl[3] <= 1.02 * true_ppl, (  # the reference stays below 1.005 across 300 random chains
        f"the text comes from an order-2 Markov chain (each character depends on the previous TWO). on the "
        f"validation characters the TRUE chain has perplexity {true_ppl:.3f}; an order-3 n-gram (context of 2) "
        f"fitted to 50,000 characters should get within 2% of it, but yours got {ppl[3]:.3f}. "
        f"all orders: { {k: round(v, 3) for k, v in ppl.items()} }")
    assert ppl[3] < ppl[2] < ppl[1], f"more context should help up to the true order: {ppl}"
    assert ppl[4] > ppl[3], f"order 4 adds no information but splits the counts, so it should be slightly worse: {ppl}"


# ---------------------------------------------------------------- stage 4: generation


def test_stage4_shape_and_prompt(nb):
    m = nb.NGramLM(3, 6, k=0.1).fit(np.asarray(nb.encode(TEXT, list(ALPHABET))))
    out = np.asarray(m.generate([0, 1], 50, rng=0))
    assert out.shape == (52,) and out.dtype.kind == "i", f"generate(prompt of 2, 50) returns 52 ints, got {out.shape} {out.dtype}"
    assert out[:2].tolist() == [0, 1], "the output starts with the prompt"
    assert np.array_equal(out, np.asarray(m.generate([0, 1], 50, rng=0))), "the same seed must give the same text"
    with pytest.raises(ValueError):
        m.generate([0], 5)
        pytest.fail("an order-3 model needs a prompt of at least 2 ids: raise ValueError")


def test_stage4_greedy(nb):
    m = nb.NGramLM(2, 3, k=0.0).fit(np.array([0, 1, 2, 0, 1, 2, 0, 1, 0]))
    out = np.asarray(m.generate([0], 6, temperature=0.0, rng=0))
    assert out.tolist() == [0, 1, 2, 0, 1, 2, 0], (
        f"greedy decoding (temperature 0) always takes the most likely next id: 0 -> 1 -> 2 -> 0 ..., got {out.tolist()}")


def test_stage4_sampling_matches_the_model(nb):
    m = nb.NGramLM(1, 3, k=0.0).fit(np.array([0] * 6 + [1] * 3 + [2]))  # P = (.6, .3, .1)
    rng = _OnlyUniform(1)
    with mock.patch.object(np.random, "choice", side_effect=AssertionError(
            "you called np.random.choice: sample with your own mlfs.sampling.categorical")):
        out = np.asarray(m.generate([], 20_000, rng=rng))
    assert not rng.used, (
        f"generate called rng.{sorted(rng.used)[0]}(...): sample each token with YOUR mlfs.sampling.categorical "
        "(imported in the setup cell), which draws with rng.random() only")
    freq = np.bincount(out, minlength=3) / 20_000
    assert np.allclose(freq, [0.6, 0.3, 0.1], atol=0.015), (
        f"at temperature 1, the frequencies of sampled ids should match the model's probabilities (.6, .3, .1); "
        f"got {np.round(freq, 3).tolist()}")


def test_stage4_temperature_sharpens(nb):
    m = nb.NGramLM(1, 3, k=0.0).fit(np.array([0] * 6 + [1] * 3 + [2]))
    cold = np.bincount(np.asarray(m.generate([], 20_000, temperature=0.5, rng=2)), minlength=3) / 20_000
    want = np.array([0.36, 0.09, 0.01]) / 0.46  # p^2, renormalized
    assert np.allclose(cold, want, atol=0.015), (
        f"temperature 0.5 raises probabilities to the power 1 / 0.5 = 2 and renormalizes: {np.round(want, 3).tolist()}; "
        f"got {np.round(cold, 3).tolist()}")
    hot = np.bincount(np.asarray(m.generate([], 20_000, temperature=1e6, rng=3)), minlength=3) / 20_000
    assert np.allclose(hot, 1 / 3, atol=0.015), f"a huge temperature flattens everything to uniform, got {np.round(hot, 3).tolist()}"


def test_stage4_generated_text_has_the_right_statistics(nb):
    vocab = list(ALPHABET)
    ids = np.asarray(nb.encode(TEXT, vocab))
    m = nb.NGramLM(3, 6, k=0.01).fit(ids)
    gen = np.asarray(m.generate(ids[:2], 30_000, rng=4))
    ppl_gen = nb.NGramLM(3, 6, k=0.1).fit(ids).perplexity(gen)
    assert abs(math.log(ppl_gen) - H_TRUE) < 0.05, (
        f"text sampled from a good order-3 model should look like the source to another order-3 model: its "
        f"perplexity should be near the true {math.exp(H_TRUE):.3f}, got {ppl_gen:.3f}")
