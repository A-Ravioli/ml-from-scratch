"""Checks for 27 · 07 retrieval, tools & reasoning.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. chunking on hand examples
(windows, overlap, the tail, errors); the hashing embedder against TF-IDF computed here from crc32-hashed
character n-grams, for idf, sublinear term frequency, normalization, n-gram range and process-independent
hashing; top-k cosine search against brute force; the vector index for exact search, ids, training, IVF
probing (n_probe = n_lists is exact; more probes, more recall) and padding; the RAG prompt, the agent loop
with scripted models and tools (results, errors, the step cap), and majority voting.
"""
import builtins
import math
import zlib

import numpy as np
import pytest

from mlfs.testing import bounded


# ---------------------------------------------------------------- stage 1: chunking


def _chunk(nb, *args, **kwargs):
    """chunk_text under a line budget: a window loop that never advances fails instead of hanging."""
    return bounded(nb.chunk_text, *args, what="chunk_text", budget=200_000,
                   hint="with overlap >= chunk_size the step is <= 0: raise ValueError before looping.", **kwargs)


def test_stage1_chunks(nb):
    text = " ".join(f"w{i}" for i in range(10))
    got = _chunk(nb, text, chunk_size=4, overlap=1)
    want = ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert got == want, (
        f"windows of 4 words starting every 4 - 1 = 3 words, stopping once a window reaches the end: {want}; got {got}")
    got = _chunk(nb, " ".join(f"w{i}" for i in range(11)), chunk_size=4, overlap=1)
    assert got[-1] == "w9 w10" and len(got) == 4, (
        f"the last window may be short: it starts at word 9 and holds the remaining 2 words; got {got}")
    assert _chunk(nb, "a  b\n\nc", chunk_size=5, overlap=2) == ["a b c"], "a short text is one chunk, words joined by single spaces"
    assert _chunk(nb, "   ") == [], "no words, no chunks"
    long = " ".join(str(i) for i in range(450))
    ch = _chunk(nb, long)
    assert [len(c.split()) for c in ch] == [200, 200, 150], "defaults: 200 words, overlap 50 (starts 0, 150, 300)"
    assert ch[1].split()[0] == "150"
    for bad in ((4, 4), (4, 5), (0, 0), (4, -1)):
        with pytest.raises(ValueError):
            _chunk(nb, text, chunk_size=bad[0], overlap=bad[1])


# ---------------------------------------------------------------- stage 2: embedder


def _want_vectors(texts, fit_texts, dim, ngram):
    def counts(t):
        t = " " + " ".join(t.lower().split()) + " "
        c = np.zeros(dim)
        for n in range(ngram[0], ngram[1] + 1):
            for i in range(len(t) - n + 1):
                c[zlib.crc32(t[i:i + n].encode()) % dim] += 1
        return c
    df = sum((counts(t) > 0).astype(float) for t in fit_texts)
    idf = np.log((1 + len(fit_texts)) / (1 + df)) + 1
    out = []
    for t in texts:
        c = counts(t)
        v = np.where(c > 0, 1 + np.log(np.maximum(c, 1)), 0) * idf
        out.append(v / max(np.linalg.norm(v), 1e-300))
    return np.array(out), idf


def test_stage2_tfidf(nb):
    docs = ["the cat sat on the mat", "the dog sat on the log", "a stray cat"]
    emb = nb.HashingEmbedder(dim=4096)
    assert emb.fit(docs) is emb, "fit returns self"
    want, idf = _want_vectors(docs, docs, 4096, (3, 5))
    assert np.allclose(emb.idf_, idf), (
        "idf_[j] = log((1 + N) / (1 + df_j)) + 1, where df_j counts the texts with at least one n-gram in bucket j; "
        "buckets are crc32(ngram) % dim over character 3- to 5-grams of ' ' + lowercased, space-normalized text + ' '")
    got = emb.transform(docs)
    assert got.shape == (3, 4096) and np.allclose(got, want), (
        "transform: (1 + log count) * idf in every bucket that occurs (0 elsewhere), each row scaled to unit length")
    assert np.allclose(np.linalg.norm(emb.transform(["anything at all", "x"]), axis=1), 1.0)
    assert np.allclose(*emb.transform(["The  CAT\nsat", "the cat sat"])), "lowercase and normalize whitespace first"
    s = got @ got.T
    assert s[0, 1] > s[0, 2] > 0, "sentences sharing words are closer than unrelated ones"
    uni = nb.HashingEmbedder(dim=4096, ngram=(1, 1)).fit(docs)
    a, b = uni.transform(["abc", "cab"])
    assert np.allclose(a, b), "ngram=(1, 1): character unigrams, so 'abc' and 'cab' are the same bag"
    assert not np.allclose(*emb.transform(["abc", "cab"])), "with 3- to 5-grams they differ: ngram is used"
    assert nb.HashingEmbedder(dim=64).fit(docs).transform(docs).shape == (3, 64), "dim sets the number of buckets"


def test_stage2_stable_hash(nb):
    docs = ["a stable hash", "survives a restart"]
    before = nb.HashingEmbedder(dim=512).fit(docs).transform(docs)
    real = builtins.hash
    builtins.hash = lambda o: real(o) ^ 0x2545F491
    try:
        after = nb.HashingEmbedder(dim=512).fit(docs).transform(docs)
    finally:
        builtins.hash = real
    assert np.array_equal(before, after), (
        "your vectors changed when Python's hash() changed: hash n-grams with zlib.crc32, so an index built today still "
        "matches queries embedded tomorrow")


# ---------------------------------------------------------------- stage 3: top-k


def test_stage3_cosine_top_k(nb):
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(50, 8))
    q = rng.normal(size=(5, 8))
    idx, sc = nb.cosine_top_k(q, keys * rng.uniform(0.1, 10, size=(50, 1)), k=4)
    sims = (q / np.linalg.norm(q, axis=1, keepdims=True)) @ (keys / np.linalg.norm(keys, axis=1, keepdims=True)).T
    want = np.argsort(-sims, axis=1)[:, :4]
    assert idx.shape == (5, 4) and np.array_equal(idx, want), (
        "the k keys with the highest COSINE similarity, best first (key lengths don't matter)")
    assert np.allclose(sc, np.take_along_axis(sims, want, 1)), "scores are the cosine similarities"
    i1, s1 = nb.cosine_top_k(q[0], keys, k=3)
    assert i1.shape == (3,) and np.array_equal(i1, want[0, :3]), "a 1-D query gives 1-D results"
    i2, _ = nb.cosine_top_k(q, keys[:2], k=5)
    assert i2.shape == (5, 2), "k larger than the number of keys is capped"
    z = np.vstack([np.zeros(8), keys[:3]])
    iz, sz = nb.cosine_top_k(q[0], z, k=4)
    assert np.all(np.isfinite(sz)) and iz[-1] == 0 or sz[list(iz).index(0)] == 0, "a zero vector has similarity 0, not nan"
    assert nb.cosine_top_k(q[0], keys)[0].shape == (5,), "default k is 5"


# ---------------------------------------------------------------- stage 4: vector index


def _clusters(seed=0, n_per=60, k=8, d=16):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3
    x = np.vstack([c + rng.normal(size=(n_per, d)) for c in centers])
    return x, centers


def test_stage4_exact(nb):
    x, _ = _clusters()
    idx = nb.VectorIndex(16)
    idx.add(x[:200])
    idx.add(x[200:])
    assert idx.ntotal == len(x), "ntotal counts the stored vectors"
    rng = np.random.default_rng(1)
    q = rng.normal(size=(6, 16)) * 3
    sc, ids = idx.search(q, k=5)
    want_i, want_s = nb.cosine_top_k(q, x, k=5)
    assert ids.shape == (6, 5) and np.array_equal(ids, want_i), (
        "n_lists=0 is exact search: the same neighbours as cosine_top_k; ids continue across add() calls (0..479)")
    assert np.allclose(sc, want_s), "search returns (scores, ids): cosine similarities first"
    own = nb.VectorIndex(16)
    own.add(x[:10], ids=np.arange(100, 110))
    assert set(own.search(x[:1], k=3)[1][0]) <= set(range(100, 110)) and own.search(x[3:4], k=1)[1][0, 0] == 103, (
        "custom ids are returned instead of positions")
    few = nb.VectorIndex(16)
    few.add(x[:2])
    s, i = few.search(x[:1], k=4)
    assert list(i[0, 2:]) == [-1, -1] and np.all(np.isneginf(s[0, 2:])), "missing results: id -1, score -inf"
    with pytest.raises(ValueError):
        nb.VectorIndex(8).add(x[:3])


def test_stage4_ivf(nb):
    x, centers = _clusters()
    rng = np.random.default_rng(2)
    q = rng.normal(size=(40, 16)) * 3  # queries anywhere, so their neighbours straddle cells
    exact = nb.cosine_top_k(q, x, k=10)[0]
    ivf = nb.VectorIndex(16, n_lists=8, n_probe=1, seed=0)
    with pytest.raises(RuntimeError):
        ivf.add(x)
    ivf.train(x)
    ivf.add(x)
    recalls = []
    for p in (1, 2, 8):
        ivf.n_probe = p
        ids = ivf.search(q, k=10)[1]
        recalls.append(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact)]))
    assert recalls[2] == 1.0, f"probing all n_lists lists is exact search; recall {recalls[2]:.2f}"
    ivf.n_probe = 1
    self_hit = ivf.search(x[:50], k=1)[1][:, 0]
    assert np.array_equal(self_hit, np.arange(50)), (
        "with n_probe=1, a stored vector queried as itself must come back first: its own list is the one whose "
        "centroid is MOST similar to it. probe the n_probe lists with the highest centroid similarity (sort descending)")
    assert recalls[0] > 0.15, f"one probe of 8 cells should still find some true neighbours; recall {recalls[0]:.2f}"
    assert recalls[0] < 1.0 and recalls[0] <= recalls[1] <= recalls[2], (
        f"one probed list misses neighbours that fell into other cells; more probes find more; recalls {recalls}")
    again = nb.VectorIndex(16, n_lists=8, n_probe=1, seed=0).train(x)
    again.add(x)
    again.n_probe = 1
    assert np.array_equal(again.search(q, k=10)[1], (ivf.__setattr__("n_probe", 1), ivf.search(q, k=10)[1])[1]), (
        "the seed fixes the k-means quantizer: same index, same results")
    scaled = x * np.random.default_rng(5).uniform(0.01, 100, size=(len(x), 1))
    a1 = nb.VectorIndex(16, n_lists=8, n_probe=1, seed=0).train(x)
    a2 = nb.VectorIndex(16, n_lists=8, n_probe=1, seed=0).train(scaled)
    assert np.allclose(np.linalg.norm(a1.centroids_, axis=1), 1.0), "centroids_ are unit vectors"
    assert np.allclose(a1.centroids_, a2.centroids_), (
        "train() normalizes the vectors before k-means: rescaling them must not change the cells (cosine ignores length)")
    blob = np.random.default_rng(6).normal(size=(300, 16))  # no cluster structure: k-means depends on its start
    c0 = nb.VectorIndex(16, n_lists=8, seed=0).train(blob).centroids_
    c1 = nb.VectorIndex(16, n_lists=8, seed=1).train(blob).centroids_
    assert not np.allclose(np.sort(c0, axis=0), np.sort(c1, axis=0)), "seed is passed to KMeans (rng=seed)"
    one = nb.VectorIndex(16, n_lists=8, n_probe=1, seed=0).train(x)
    one.add(x)
    s, ids = one.search(centers[:1], k=200)
    got = ids[0][ids[0] >= 0]
    assert 0 < len(got) < len(x) and np.all(np.isneginf(s[0][ids[0] < 0])), (
        "with n_probe=1 only the vectors in the query's nearest cell are candidates (the rest are padded with -1)")


# ---------------------------------------------------------------- stage 5: RAG, agents, voting


def test_stage5_rag_prompt(nb):
    got = nb.build_rag_prompt("Who rules Verona?", ["Escalus is prince.", "Verona is a city."])
    want = "Context:\n[1] Escalus is prince.\n[2] Verona is a city.\n\nQuestion: Who rules Verona?\nAnswer:"
    assert got == want, f"want {want!r}, got {got!r}"
    assert nb.build_rag_prompt("Q?", []) == "Question: Q?\nAnswer:", "no passages: just the question"


def test_stage5_agent_loop(nb):
    def script(replies):
        seen = []

        def llm(transcript):
            seen.append(transcript)
            return replies[len(seen) - 1]
        return llm, seen

    tools = {"add": lambda a: sum(int(x) for x in a.split(",")), "boom": lambda a: 1 / 0}
    llm, seen = script(["thinking... CALL add(2, 3)", "ANSWER: 5"])
    ans, tr = bounded(nb.run_agent, llm, tools, "add 2 and 3", what="run_agent")
    assert ans == "5", f"the agent returns the text after 'ANSWER:'; got {ans!r}"
    assert seen[0] == "TASK: add 2 and 3\n", f"the first transcript is 'TASK: ...\\n'; got {seen[0]!r}"
    assert "RESULT: 5" in seen[1] and seen[1].startswith(seen[0] + "thinking... CALL add(2, 3)\n"), (
        "after a CALL the transcript gets the model's reply and then 'RESULT: <tool output>'")
    assert tr.endswith("ANSWER: 5\n")
    llm, seen = script(["CALL nope(1)", "CALL boom(x)", "no idea", "ANSWER: 0"])
    ans, tr = nb.run_agent(llm, tools, "t")
    assert "RESULT: error: unknown tool nope" in seen[1], "an unknown tool is reported back to the model as an error"
    assert "RESULT: error: ZeroDivisionError" in seen[2], "a tool that raises is reported, not propagated"
    assert seen[3].count("RESULT: error") == 3 and ans == "0", (
        "a reply with neither CALL nor ANSWER also gets an error result, telling the model the expected format")
    stuck = lambda transcript: "CALL add(1, 1)"
    ans, tr = bounded(nb.run_agent, stuck, tools, "loop", max_steps=5, what="run_agent")
    assert ans is None and tr.count("CALL add") == 5, (
        "max_steps caps the loop: after 5 model calls without an answer, return (None, transcript)")
    n_calls = []

    def counting(transcript):
        n_calls.append(1)
        return "CALL add(1, 1)"

    ans, _ = bounded(nb.run_agent, counting, tools, "loop", what="run_agent")
    assert ans is None and len(n_calls) == 8, f"default max_steps is 8 model calls; got {len(n_calls)}"
    used = []
    both = {"add": lambda a: used.append(a) or 0}
    ans, tr = nb.run_agent(lambda t: "CALL add(1, 2) ... ANSWER: 3", both, "t")
    assert ans == "3" and not used, "a reply containing ANSWER ends the loop at once, even if it also contains a CALL"


def test_stage5_majority(nb):
    assert nb.majority_vote(["a", "b", "a", "c"]) == "a"
    assert nb.majority_vote(["b", "a", "b", "a"]) == "b", "ties go to the answer seen first"
    assert nb.majority_vote([None, "x", None, None]) == "x", "None (failed parses) doesn't vote"
    assert nb.majority_vote([]) is None and nb.majority_vote([None]) is None
