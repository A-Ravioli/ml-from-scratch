"""Checks for 27 · 01 pretraining data & compute.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. shingles and Jaccard on hand
examples (normalisation, short texts, empty sets, signatures); MinHash for its dtype, determinism, seed,
independence from Python's per-process string hashing, and, statistically, for estimating Jaccard without
bias; the LSH dedup on hand-built near-duplicates, for every knob (threshold, n_bands, n_perm, k, seed), for
clusters being connected components, and for not comparing all pairs; each quality rule on a page that
fails only that rule; mixture weights and effective tokens against their formulas.
"""
import builtins
import math

import numpy as np
import pytest

from mlfs.testing import bounded

WORDS = ("rose thorn crown sword river castle winter falcon mirror lantern harbor meadow candle "
         "thunder garden silver bridge orchard tempest willow marble shadow anchor ember").split()


def _doc(seed, n=60):
    rng = np.random.default_rng(seed)
    return " ".join(rng.choice(WORDS, size=n)) + f" {seed}"


def _edit(doc, n_changes, seed):
    """replace n_changes words with a word that isn't in the vocabulary."""
    rng = np.random.default_rng(seed)
    words = doc.split()
    for i in rng.choice(len(words), size=n_changes, replace=False):
        words[i] = f"zz{rng.integers(10**6)}q"
    return " ".join(words)


# ---------------------------------------------------------------- stage 1: shingles and jaccard


def test_stage1_shingles(nb):
    got = nb.shingles("The  Cat\nsat", k=3)
    want = {"the", "he ", "e c", " ca", "cat", "at ", "t s", " sa", "sat"}
    assert got == want, (
        f"lowercase, collapse every run of whitespace to one space, then all character 3-grams: want {sorted(want)}, "
        f"got {sorted(got) if isinstance(got, set) else got}")
    assert isinstance(nb.shingles("abcdefgh"), set), "shingles returns a Python set"
    assert nb.shingles("abcdefgh") == {"abcde", "bcdef", "cdefg", "defgh"}, "default k is 5"
    assert nb.shingles("  Hi  ", k=5) == {"hi"}, "a text shorter than k is one shingle: the whole normalised text"
    assert nb.shingles("   ", k=5) == set(), "an empty (or all-whitespace) text has no shingles"
    assert nb.shingles("ab ab", k=2) == {"ab", "b ", " a"}, "a set: repeated k-grams count once"


def test_stage1_jaccard(nb):
    assert nb.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5, "|{2,3}| / |{1,2,3,4}| = 0.5"
    assert nb.jaccard({"a"}, {"b"}) == 0.0
    j = nb.jaccard(nb.shingles("the cat sat", k=3), nb.shingles("the cat sang", k=3))
    assert np.isclose(j, 8 / 11), f"the lesson's worked example: 8 shared 3-grams out of 11 distinct, 0.727; got {j}"
    assert nb.jaccard(set(), set()) == 1.0, "two empty sets are identical: define their Jaccard as 1.0"
    assert nb.jaccard(set(), {"x"}) == 0.0
    s1 = np.array([1, 2, 3, 4], dtype=np.uint64)
    s2 = np.array([1, 9, 3, 7], dtype=np.uint64)
    assert np.isclose(nb.jaccard(s1, s2), 0.5), (
        "for two MinHash signatures (ndarrays), jaccard is the fraction of positions that agree: 2 of 4 here")
    with pytest.raises(ValueError):
        nb.jaccard(s1, s2[:3])


# ---------------------------------------------------------------- stage 2: minhash


def test_stage2_signature_basics(nb):
    s = nb.shingles("the quick brown fox jumps over the lazy dog")
    sig = nb.minhash_signature(s)
    assert isinstance(sig, np.ndarray) and sig.dtype == np.uint64 and sig.shape == (128,), (
        f"a uint64 array of n_perm=128 minimum hash values; got {type(sig).__name__} "
        f"{getattr(sig, 'dtype', '')} {getattr(sig, 'shape', '')}")
    assert np.array_equal(sig, nb.minhash_signature(set(s))), "same set, same signature (deterministic)"
    assert np.array_equal(sig, nb.minhash_signature(set(sorted(s, reverse=True)))), "order of the set doesn't matter"
    assert nb.minhash_signature(s, n_perm=32).shape == (32,), "n_perm sets the signature length"
    other = nb.minhash_signature(s, seed=1)
    assert np.mean(other == sig) < 0.5, "a different seed draws different hash functions"
    assert np.array_equal(nb.minhash_signature(s, seed=1), other), "the seed fixes the hash functions"
    sub = nb.minhash_signature(set(list(s)[:5]))
    assert np.all(sub >= sig), "the minimum over a subset can't be smaller than over the whole set"
    e1, e2 = nb.minhash_signature(set()), nb.minhash_signature(set())
    assert e1.shape == (128,) and np.array_equal(e1, e2), "the empty set has a fixed signature (e.g. all max uint64)"


def test_stage2_no_python_hash(nb):
    s = nb.shingles("a stable hash survives a restart of the python process")
    before = nb.minhash_signature(s)
    real_hash = builtins.hash
    builtins.hash = lambda obj: real_hash(obj) ^ 0x5DEECE66D
    try:
        after = nb.minhash_signature(s)
    finally:
        builtins.hash = real_hash
    assert np.array_equal(before, after), (
        "your signature changed when Python's hash() changed. str hashes are randomised per process "
        "(PYTHONHASHSEED), so a crawl deduplicated today wouldn't match tomorrow's: hash shingles with a fixed "
        "function such as hashlib.blake2b or a hand-written FNV")


def test_stage2_estimates_jaccard(nb):
    # sets with known Jaccard 1/3 and 0.8; average the estimate over 25 seeds (sd of the mean ~ 0.008)
    base = {f"s{i}" for i in range(300)}
    a, b = base | {f"a{i}" for i in range(150)}, base | {f"b{i}" for i in range(150)}
    c = base | {f"c{i}" for i in range(37)}
    for x, y, j in ((a, b, nb.jaccard(a, b)), (base, c, nb.jaccard(base, c))):
        est = [nb.jaccard(nb.minhash_signature(x, seed=s), nb.minhash_signature(y, seed=s)) for s in range(25)]
        assert abs(np.mean(est) - j) < 0.035, (
            f"signature agreement should be an unbiased estimate of Jaccard {j:.3f}; mean over 25 seeds {np.mean(est):.3f}. "
            "use one random hash function per position and take each one's minimum over the whole set")
        assert 0.01 < np.std(est) < 0.09, (
            f"each estimate should scatter by about sqrt(J(1-J)/128) = {math.sqrt(j * (1 - j) / 128):.3f}; "
            f"got sd {np.std(est):.3f}. are the 128 hash functions really different from each other?")


# ---------------------------------------------------------------- stage 3: LSH dedup


def _dedup(nb, docs, **kw):
    return bounded(nb.minhash_dedup, docs, what="minhash_dedup", budget=5_000_000, **kw)


def test_stage3_hand_example(nb):
    a, b, c = _doc(1), _doc(2), _doc(3)
    docs = [a, a.upper().replace(" ", "   "), b, _edit(a, 1, 0), c, b]
    keep = _dedup(nb, docs)
    assert list(keep) == [0, 2, 4], (
        f"docs 1 (doc 0 in capitals with extra spaces), 3 (doc 0 with one word changed) and 5 (a copy of doc 2) are "
        f"duplicates: keep the first of each cluster, [0, 2, 4], in sorted order; got {list(keep)}")
    assert all(isinstance(i, (int, np.integer)) for i in keep)
    assert list(_dedup(nb, [])) == [], "no documents, nothing to keep"
    assert list(_dedup(nb, [a])) == [0]
    with pytest.raises(ValueError):
        nb.minhash_dedup(docs, n_perm=100, n_bands=32)


def test_stage3_threshold(nb):
    a = _doc(5)
    near = _edit(a, 8, 1)  # Jaccard about 0.55 at k=5
    j = nb.jaccard(nb.shingles(a), nb.shingles(near))
    assert 0.45 < j < 0.7, f"(test setup) Jaccard {j}"
    assert list(_dedup(nb, [a, near])) == [0, 1], f"Jaccard {j:.2f} is below the default threshold 0.8: keep both"
    assert list(_dedup(nb, [a, near], threshold=0.3, n_bands=64)) == [0], (
        f"with threshold=0.3, Jaccard {j:.2f} is a duplicate: only doc 0 is kept. is `threshold` used?")


def test_stage3_bands_are_used(nb):
    a = _doc(7)
    near = _edit(a, 1, 2)  # Jaccard about 0.9: almost surely found with 32 bands of 4 rows
    assert list(_dedup(nb, [a, near])) == [0]
    kept = list(_dedup(nb, [a, near], n_bands=1))
    assert kept == [0, 1], (
        "with n_bands=1 a pair is a candidate only if all 128 signature entries agree, which a 0.9-similar pair "
        "almost never does, so both docs are kept. if you dropped the near-duplicate anyway, you compared pairs "
        "that LSH never proposed (all pairs?), or ignored n_bands")


def test_stage3_connected_components(nb):
    chain = [_doc(11)]
    for i in range(14):  # each doc is one word away from the previous one, until the ends are far apart
        chain.append(_edit(chain[-1], 1, 30 + i))
        if i >= 6 and nb.jaccard(nb.shingles(chain[0]), nb.shingles(chain[-1])) < 0.72:
            break
    sh = [nb.shingles(d) for d in chain]
    links = [nb.jaccard(sh[i], sh[i + 1]) for i in range(len(chain) - 1)]
    ends = nb.jaccard(sh[0], sh[-1])
    assert min(links) > 0.9 and ends < 0.75, f"(test setup) {links} {ends}"
    keep = list(_dedup(nb, chain, threshold=0.8))
    assert keep == [0], (
        f"each doc is a near-duplicate (Jaccard > 0.9) of the next, though the first and last share only {ends:.2f}. "
        f"clusters are connected components (union-find), so the whole chain is one cluster and only doc 0 is kept; got "
        f"{keep}. comparing each doc only with the docs you've kept so far keeps the far end of the chain")
    keep = list(_dedup(nb, chain[::-1], threshold=0.8))
    assert keep == [0], "the first index of each cluster is kept, whatever order the docs come in"


def test_stage3_bridge(nb):
    rng = np.random.default_rng(7)
    a = " ".join(rng.choice(WORDS[:12], size=60))
    b = " ".join(rng.choice(WORDS[12:], size=60))
    c = " ".join(a.split()[:30] + b.split()[30:])  # half of a, half of b: similar to both, which aren't similar
    sh = [nb.shingles(d) for d in (a, b, c)]
    assert nb.jaccard(sh[0], sh[1]) < 0.15 and min(nb.jaccard(sh[0], sh[2]), nb.jaccard(sh[1], sh[2])) > 0.3, (
        "(test setup)")
    keep = list(_dedup(nb, [a, b, c], threshold=0.25, n_bands=64))
    assert keep == [0], (
        f"doc 2 bridges docs 0 and 1, so all three are one cluster: keep [0]; got {keep}. when you merge two clusters, "
        "link their ROOTS (find(i), find(j)), not the documents i and j themselves: linking a non-root re-parents "
        "only that document and leaves its old cluster behind")


def test_stage3_k_n_perm_and_seed(nb):
    x, y = "listen silent enlist", "silent tinsel listen"
    assert list(_dedup(nb, [x, y], k=1)) == [0], "with k=1 both docs are the same set of characters: duplicates"
    assert list(_dedup(nb, [x, y])) == [0, 1], "with the default k=5 they share few 5-grams: both kept"
    # a borderline pair: the decision depends on the hash functions, so it must agree with your own signatures
    a = _doc(21)
    near = _edit(a, 2, 5)
    for n_perm, n_bands in ((128, 32), (64, 16)):
        sigs = [(nb.minhash_signature(nb.shingles(a), n_perm=n_perm, seed=s),
                 nb.minhash_signature(nb.shingles(near), n_perm=n_perm, seed=s)) for s in range(10)]
        agree = [float(np.mean(s1 == s2)) for s1, s2 in sigs]
        levels = sorted(set(agree))
        thr = levels[len(levels) // 2]  # some seeds fall below it, some reach it
        r = n_perm // n_bands
        for seed, (s1, s2) in enumerate(sigs):
            if not any(np.array_equal(s1[i * r:(i + 1) * r], s2[i * r:(i + 1) * r]) for i in range(n_bands)):
                continue  # never a candidate: kept either way
            want = [0] if agree[seed] >= thr else [0, 1]
            got = list(_dedup(nb, [a, near], threshold=thr, n_perm=n_perm, n_bands=n_bands, seed=seed))
            assert got == want, (
                f"n_perm={n_perm}, seed={seed}: your signatures agree on {agree[seed]:.3f} of positions, so with "
                f"threshold {thr:.3f} the result should be {want}; got {got}. pass n_perm, seed and k through to "
                "minhash_signature/shingles, and verify candidates by signature agreement")


def test_stage3_scales(nb):
    rng = np.random.default_rng(0)
    base = [_doc(100 + i) for i in range(150)]
    docs, truth = [], []
    for i, d in enumerate(base):
        docs.append(d)
        truth.append(len(docs) - 1)
        for c in range(int(rng.integers(0, 3))):
            docs.append(_edit(d, 1, 1000 * i + c))
    keep = _dedup(nb, docs)
    assert list(keep) == truth, (
        f"{len(docs)} docs: 150 originals, each followed by 0-2 one-word edits. keep exactly the originals; "
        f"{len(set(keep) ^ set(truth))} indices differ")


# ---------------------------------------------------------------- stage 4: quality filter

GOOD = ("the harbor was quiet when the boats came back from the river and the old falcon watched from the castle "
        "wall while the candles burned low in every window of the town above the meadow")


def test_stage4_good_page(nb):
    keep, reasons = nb.quality_filter(GOOD)
    assert keep is True and reasons == [], f"a normal paragraph passes every rule; got {keep}, {reasons}"


@pytest.mark.parametrize("text, rule, kwargs", [
    ("too few words here", "too_short", {}),
    (" ".join(["a b c d"] * 8), "word_length", {}),
    (" ".join(["extraordinarily incomprehensible"] * 12), "word_length", {}),
    (GOOD + " #tag #tag #tag #tag #tag", "symbols", {}),
    (GOOD + " 12 34 56 78 90 11 22 33 44", "non_alpha", {}),
    ("\n".join(["buy cheap castle candles now"] * 6 + [GOOD]), "repeated_lines", {}),
])
def test_stage4_each_rule(nb, text, rule, kwargs):
    keep, reasons = nb.quality_filter(text, **kwargs)
    assert keep is False and reasons == [rule], (
        f"this page breaks exactly one rule, {rule!r}: want (False, [{rule!r}]), got ({keep}, {reasons})")


def test_stage4_knobs(nb):
    short = "a perfectly normal sentence of words"
    assert nb.quality_filter(short)[1] == ["too_short"]
    assert nb.quality_filter(short, min_words=5) == (True, []), "min_words sets the word-count floor"
    assert nb.quality_filter(GOOD, word_len=(5.0, 9.0))[1] == ["word_length"], "word_len bounds the mean word length"
    hashy = GOOD + " #a #b"
    assert nb.quality_filter(hashy)[0], "2 symbols per 36 words is under the default ratio 0.1"
    assert nb.quality_filter(hashy, max_symbol_ratio=0.05)[1] == ["symbols"], "max_symbol_ratio is the limit"
    ellipses = GOOD + " ... ... ... …"
    assert nb.quality_filter(ellipses)[1] == ["symbols"], "count '#', '...' and '…' as symbols"
    nums = GOOD + " 1 2 3"
    assert nb.quality_filter(nums)[0] and nb.quality_filter(nums, min_alpha_frac=0.95)[1] == ["non_alpha"], (
        "min_alpha_frac: the fraction of words containing at least one letter")
    rep = "\n".join(["same line"] * 2 + [GOOD, "another line"])
    assert nb.quality_filter(rep)[0], "1 repeated line out of 4 (0.25) is under the default limit 0.3"
    assert nb.quality_filter("\n".join(["same line"] * 3 + [GOOD]), max_dup_line_frac=0.6)[0], (
        "max_dup_line_frac: 2 duplicate lines out of 4 (0.5) passes a limit of 0.6")
    assert nb.quality_filter("\n".join(["same line"] * 3 + [GOOD]))[1] == ["repeated_lines"], (
        "a line counts as a duplicate when an identical (stripped, non-empty) line came before it: 2 of 4 lines")
    exact = " ".join(["abc"] * 25)
    assert "word_length" not in nb.quality_filter(exact)[1], "word_len bounds are inclusive: a mean of exactly 3.0 passes"
    assert nb.quality_filter("") == (False, ["too_short"]), "an empty text is too short and nothing else"
    padded = "\n".join(["same line  ", "  same line"] + [GOOD, "another line"])
    assert nb.quality_filter(padded, max_dup_line_frac=0.2)[1] == ["repeated_lines"], (
        "lines are compared after stripping surrounding whitespace")
    order = nb.quality_filter("# 1 # 2 # 3")[1]
    assert order == ["too_short", "word_length", "symbols", "non_alpha"], (
        f"reasons in the spec's order (too_short, word_length, symbols, non_alpha, repeated_lines); got {order}")
    many = nb.quality_filter("# # #")
    assert many[0] is False and set(many[1]) >= {"too_short", "symbols", "non_alpha"}, (
        "report every rule a page fails, not just the first")


# ---------------------------------------------------------------- stage 5: mixtures and repetition


def test_stage5_mixture_weights(nb):
    sizes = [800, 150, 50]
    w = nb.mixture_weights(sizes)
    assert np.allclose(w, [0.8, 0.15, 0.05]), "temperature 1: proportional to size"
    w2 = nb.mixture_weights(sizes, temperature=2.0)
    want = np.sqrt(sizes) / np.sqrt(sizes).sum()
    assert np.allclose(w2, want), f"temperature 2: proportional to sqrt(size), {want.round(4)}; got {np.round(w2, 4)}"
    assert np.allclose(nb.mixture_weights(sizes, temperature=1e6), 1 / 3, atol=1e-4), "high temperature: uniform"
    assert np.allclose(nb.mixture_weights([1e12, 3e12], temperature=0.5), [0.1, 0.9]), "temperature 0.5: sizes squared"
    cold = nb.mixture_weights([1e12, 3e12], temperature=0.02)
    small = 3.0 ** -50 / (1 + 3.0 ** -50)
    assert np.all(np.isfinite(cold)) and math.isclose(cold[0], small, rel_tol=1e-6) and math.isclose(cold[1], 1.0), (
        f"temperature 0.02: (1e12)^50 = 1e600 overflows float64, so work in log space; want [{small:.3g}, 1], got {cold}")
    with pytest.raises(ValueError):
        nb.mixture_weights([5, -1])
    assert np.allclose(nb.mixture_weights([0, 10, 30]), [0, 0.25, 0.75]), "an empty source gets weight 0"
    with pytest.raises(ValueError):
        nb.mixture_weights(sizes, temperature=0.0)


def test_stage5_effective_tokens(nb):
    U = 1e9
    assert nb.effective_tokens(U, 0.5e9) == 0.5e9, "no repetition: every token counts"
    assert nb.effective_tokens(U, U) == U
    got = nb.effective_tokens(U, 4e9)
    want = U + U * 15.4 * (1 - math.exp(-3 / 15.4))
    assert np.isclose(got, want), (
        f"4 epochs = 3 repetitions: U + U r* (1 - exp(-3 / r*)) = {want:.4g}; got {got:.4g}")
    assert 3.6e9 < got < 4e9, "4 epochs are worth almost 4 epochs of fresh data"
    assert np.isclose(nb.effective_tokens(U, 1e12), U * (1 + 15.4 * (1 - math.exp(-999 / 15.4)))), (
        "1000 epochs are worth at most U (1 + r*): repeated data saturates")
    assert np.isclose(nb.effective_tokens(U, 3e9, r_star=5.0), U + U * 5 * (1 - math.exp(-2 / 5))), "r_star is used"
