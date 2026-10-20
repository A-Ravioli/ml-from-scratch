"""Checks for exercise.py (embeddings).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Pairs are checked
against a hand example and a counting formula, negatives against their target distribution, the
loss against a direct computation (and its exact value (K + 1) log 2 at initialisation), the
similarity tools against hand-built vectors, and training on a synthetic two-topic corpus where
the right neighbours are known. Training runs under `mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlfs.testing import bounded  # noqa: E402


def _warm_up():
    """torch imports parts of itself on first use (millions of lines); do that here, outside any line budget."""
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([p])
    torch.nn.Embedding(3, 2)(torch.tensor([1])).sum().backward()
    (p ** 2).sum().backward()
    opt.step()


_warm_up()


def _topic_corpus(seed, n_sent=300, length=8):
    """words 0-9 only co-occur with each other, as do words 10-19: two topics."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sent):
        topic = int(rng.integers(2))
        out.append(rng.integers(0, 10, size=length) + 10 * topic)
    return np.concatenate(out)


# ---------------------------------------------------------------- stage 1: pairs


def test_stage1_hand_example(nb):
    got = nb.skipgram_pairs(np.array([5, 6, 7, 8]), window=1)
    want = np.array([[5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7]])
    assert isinstance(got, np.ndarray) and got.dtype == np.int64 and got.shape == (6, 2), (
        f"a (P, 2) int64 ndarray; got {type(got).__name__} {getattr(got, 'dtype', '')} {getattr(got, 'shape', '')}")
    assert np.array_equal(got, want), (
        f"window 1 over [5, 6, 7, 8]: each word with its left and right neighbour, ordered by center position, then "
        f"context position: {want.tolist()}; got {got.tolist()}")


def test_stage1_counts_and_order(nb):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 50, size=37)
    for w in (1, 2, 5):
        got = nb.skipgram_pairs(ids, window=w)
        n = len(ids)
        expected = sum(min(i, w) + min(n - 1 - i, w) for i in range(n))
        assert len(got) == expected, f"window {w}: {expected} pairs expected (fewer near the ends), got {len(got)}"
        want = [(ids[i], ids[j]) for i in range(n) for j in range(max(0, i - w), min(n, i + w + 1)) if j != i]
        assert np.array_equal(got, np.array(want)), f"window {w}: pairs or their order differ from the definition"
    assert nb.skipgram_pairs(np.array([3]), window=2).shape == (0, 2), "one word has no pairs: shape (0, 2)"
    assert nb.skipgram_pairs(np.array([1, 2]), window=1).tolist() == [[1, 2], [2, 1]], "and a list input works"


# ---------------------------------------------------------------- stage 2: negatives


def test_stage2_distribution(nb):
    counts = np.array([100, 0, 10, 1, 1000])
    g = torch.Generator().manual_seed(0)
    s = nb.sample_negatives(counts, 200_000, generator=g)
    assert torch.is_tensor(s) and s.dtype == torch.int64 and s.shape == (200_000,), (
        f"an int64 tensor of shape (n,); got {type(s).__name__} {getattr(s, 'dtype', '')} {getattr(s, 'shape', '')}")
    freq = np.bincount(s.numpy(), minlength=5) / len(s)
    p = counts ** 0.75 / (counts ** 0.75).sum()
    assert np.allclose(freq, p, atol=0.005), (
        f"negatives are drawn with probability proportional to counts ** 0.75 = {np.round(p, 4)}; got frequencies "
        f"{np.round(freq, 4)}. (with counts itself you'd get {np.round(counts / counts.sum(), 4)})")
    assert freq[1] == 0, "a word with count 0 is never drawn"


def test_stage2_power_and_reproducibility(nb):
    counts = [1, 4, 9, 16]
    s = nb.sample_negatives(counts, 100_000, power=0.0, generator=torch.Generator().manual_seed(1))
    assert np.allclose(np.bincount(s.numpy(), minlength=4) / 1e5, 0.25, atol=0.01), "power 0 is uniform over seen words"
    s = nb.sample_negatives(counts, 100_000, power=1.0, generator=torch.Generator().manual_seed(2))
    assert np.allclose(np.bincount(s.numpy(), minlength=4) / 1e5, np.array(counts) / 30, atol=0.01), "power 1: unigram"
    a = nb.sample_negatives(counts, 50, generator=torch.Generator().manual_seed(3))
    b = nb.sample_negatives(counts, 50, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b), "the same generator seed gives the same negatives (use torch.multinomial with generator=)"


# ---------------------------------------------------------------- stage 3: the model


def test_stage3_structure_and_init(nb):
    m = nb.SkipGram(30, 8)
    assert isinstance(m.in_embed, torch.nn.Embedding) and isinstance(m.out_embed, torch.nn.Embedding)
    assert tuple(m.in_embed.weight.shape) == (30, 8) and tuple(m.out_embed.weight.shape) == (30, 8)
    assert m.embeddings is m.in_embed.weight, "the embeddings property is in_embed.weight"
    assert torch.all(m.out_embed.weight == 0), "word2vec starts the output vectors at zero"
    assert m.in_embed.weight.abs().max() <= 0.5 / 8 and m.in_embed.weight.std() > 0.01, (
        "and the input vectors uniform in (-0.5/D, 0.5/D)")
    c, o, neg = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.randint(0, 30, (3, 5))
    loss = m(c, o, neg)
    assert loss.ndim == 0 and math.isclose(loss.item(), 6 * math.log(2), rel_tol=1e-6), (
        f"with out_embed = 0 every score is 0 and every sigmoid 1/2, so the loss is (K + 1) log 2 = {6 * math.log(2):.4f} "
        f"for K = 5 negatives; got {loss.item():.4f}")


def test_stage3_matches_formula(nb):
    g = torch.Generator().manual_seed(4)
    m = nb.SkipGram(20, 6)
    with torch.no_grad():
        m.in_embed.weight.copy_(torch.randn(20, 6, generator=g))
        m.out_embed.weight.copy_(torch.randn(20, 6, generator=g))
    c, o = torch.randint(0, 20, (7,), generator=g), torch.randint(0, 20, (7,), generator=g)
    neg = torch.randint(0, 20, (7, 4), generator=g)
    V, U = m.in_embed.weight, m.out_embed.weight
    want = 0.0
    for b in range(7):
        want += -math.log(torch.sigmoid(U[o[b]] @ V[c[b]]).item())
        want += -sum(math.log(torch.sigmoid(-(U[neg[b, k]] @ V[c[b]])).item()) for k in range(4))
    got = m(c, o, neg)
    assert math.isclose(got.item(), want / 7, rel_tol=1e-5), (
        f"loss = mean over the batch of -log s(u_o . v_c) - sum_k log s(-u_k . v_c); expected {want / 7:.5f}, got "
        f"{got.item():.5f}. (u from out_embed, v from in_embed; sum over the negatives, mean over the batch)")
    got.backward()
    assert m.in_embed.weight.grad is not None and m.out_embed.weight.grad is not None, "both tables get gradients"
    big = torch.full((7,), 0)
    with torch.no_grad():
        m.in_embed.weight[0] = 100.0 * m.out_embed.weight[0]
    assert torch.isfinite(m(big, big, big[:, None].expand(7, 4))), "use logsigmoid: no inf/nan for large scores"


# ---------------------------------------------------------------- stage 4: similarity


def test_stage4_most_similar(nb):
    E = np.array([[1.0, 0.0], [10.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]])
    idx, sims = nb.most_similar(E, 0, k=3)
    idx, sims = np.asarray(idx), np.asarray(sims, dtype=float)
    assert idx.tolist() == [1, 4, 2], (
        f"by cosine (direction only, not length): row 1 (cos 0.995), row 4 (0.707), row 2 (0); idx itself excluded; "
        f"got {idx.tolist()}")
    assert np.allclose(sims, [10 / math.sqrt(101), 1 / math.sqrt(2), 0.0]), f"the cosines, best first; got {sims}"
    idx_t, sims_t = nb.most_similar(torch.tensor(E, dtype=torch.float32), 0, k=2)
    assert np.asarray(idx_t).tolist() == [1, 4], "a tensor works too"


def test_stage4_analogy(nb):
    # gender direction g = (0, 1); royalty direction r = (1, 0); an extra unrelated direction
    man, woman, king, queen = [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]
    E = np.array([man, woman, king, queen, [1.0, 0.05, 1.05], [0.0, 0.0, -1.0]])
    got = np.asarray(nb.analogy(E, 0, 1, 2, k=1))
    assert got.tolist() == [3], (
        f"man : woman :: king : ? -> queen (row 3). normalise the rows, take woman - man + king, rank by cosine, and "
        f"exclude the three query words; got {got.tolist()}")
    got = np.asarray(nb.analogy(E, 0, 1, 2, k=2))
    assert len(got) == 2 and 2 not in got and 0 not in got and 1 not in got, "a, b, c are never returned"
    E3 = np.array([[0.0, 0.0, 5.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.5, 0.5, -1.5]])
    assert np.asarray(nb.analogy(E3, 0, 1, 2)).tolist() == [3], (
        "a long vector for 'man' (row 0) must not dominate b - a + c: normalise every row BEFORE forming the offset "
        "(without it, b - a + c = (1, 1, -3) and row 4 wins)")
    E2 = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [-0.3, 1.0]])
    assert np.asarray(nb.analogy(E2, 0, 1, 2)).tolist() == [3], (
        "b - a + c lands almost on c itself here: without excluding a, b and c the answer would be c")


# ---------------------------------------------------------------- stage 5: training


def test_stage5_learns_topics(nb):
    ids = _topic_corpus(0)
    model, losses = bounded(nb.train_skipgram, ids, 20, embed_dim=16, window=2, n_negatives=5, epochs=4,
                            batch_size=128, lr=0.02, seed=0, budget=3_000_000, what="train_skipgram")
    assert tuple(model.embeddings.shape) == (20, 16)
    losses = np.asarray(losses)
    k = max(1, len(losses) // 10)
    assert losses[-k:].mean() < losses[:k].mean() - 0.3, (
        f"the loss should fall clearly during training: first batches {losses[:k].mean():.3f}, last {losses[-k:].mean():.3f}")
    scores = []
    for s in range(5):
        model, _ = nb.train_skipgram(_topic_corpus(s), 20, embed_dim=16, window=2, epochs=4, batch_size=128, lr=0.02,
                                     seed=s)
        hits = [np.mean(np.asarray(nb.most_similar(model.embeddings, w, k=3)[0]) // 10 == w // 10) for w in range(20)]
        scores.append(np.mean(hits))
    assert np.median(scores) >= 0.9, (
        f"words 0-9 only appear with each other, as do 10-19: a word's nearest neighbours should share its topic. "
        f"fraction of top-3 neighbours in the same topic over 5 seeds: {np.round(scores, 2)}")


def test_stage5_reproducible(nb):
    ids = _topic_corpus(7, n_sent=40)
    a, la = nb.train_skipgram(ids, 20, embed_dim=8, epochs=1, batch_size=64, seed=3)
    b, lb = nb.train_skipgram(ids, 20, embed_dim=8, epochs=1, batch_size=64, seed=3)
    assert np.allclose(la, lb) and torch.allclose(a.embeddings, b.embeddings), (
        "the same seed must give the same model: seed a torch.Generator for the shuffling and the negatives, and the "
        "initialisation too")
