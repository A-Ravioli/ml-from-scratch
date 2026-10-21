import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import hashlib
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # pretraining data & compute · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `shingles`, `jaccard`: documents as sets, and how similar two sets are → `mlfs.llm` |
    | 2 | `minhash_signature`: a 128-number sketch whose agreement estimates Jaccard → `mlfs.llm` |
    | 3 | `minhash_dedup`: locality-sensitive hashing to find near-duplicates without comparing all pairs → `mlfs.llm` |
    | 4 | `quality_filter`: heuristic rules that throw out junk pages |
    | 5 | `mixture_weights`, `effective_tokens`: how much of each source to train on, and what repeating data is worth |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def shingles(text, k=5):
    """stage 1: set of character k-grams of the lowercased, whitespace-normalised text (shorter text: itself)."""
    norm = " ".join(text.lower().split())
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i:i + k] for i in range(len(norm) - k + 1)}


@app.function
def jaccard(a, b):
    """stage 1: |a & b| / |a | b| for sets (1.0 for two empty sets); for two signatures, the fraction of equal entries."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            raise ValueError(f"signatures must have the same shape, got {a.shape} and {b.shape}")
        return float(np.mean(a == b))
    a, b = set(a), set(b)
    union = len(a | b)
    return 1.0 if union == 0 else len(a & b) / union


@app.cell(hide_code=True)
def _():
    checks.run(1, shingles=shingles, jaccard=jaccard)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def minhash_signature(shingle_set, *, n_perm=128, seed=0):
    """stage 2: uint64 (n_perm,): for each of n_perm seeded universal hash functions, its minimum over the set."""
    p = (1 << 31) - 1  # a Mersenne prime: a * x + b < 2^62 fits in uint64
    rng = np.random.default_rng(seed)
    a = rng.integers(1, p, size=(n_perm, 1), dtype=np.uint64)
    b = rng.integers(0, p, size=(n_perm, 1), dtype=np.uint64)
    if not shingle_set:
        return np.full(n_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    # a stable 64-bit hash of each shingle (Python's hash() of a str changes from process to process)
    x = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")
                     for s in shingle_set), dtype=np.uint64, count=len(shingle_set))
    x %= np.uint64(p)
    return ((a * x[None, :] + b) % np.uint64(p)).min(axis=1)


@app.cell(hide_code=True)
def _():
    checks.run(2, minhash_signature=minhash_signature, jaccard=jaccard, shingles=shingles)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def minhash_dedup(docs, *, threshold=0.8, n_perm=128, n_bands=32, k=5, seed=0):
    """stage 3: LSH banding -> candidate pairs -> keep pairs whose signatures agree >= threshold -> clusters (connected components); sorted indices to keep (first of each cluster)."""
    if n_perm % n_bands:
        raise ValueError(f"n_perm={n_perm} must be divisible by n_bands={n_bands}")
    rows = n_perm // n_bands
    sigs = np.stack([minhash_signature(shingles(d, k), n_perm=n_perm, seed=seed) for d in docs]) if docs else None
    parent = list(range(len(docs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for band in range(n_bands):
        buckets = {}
        for i in range(len(docs)):
            buckets.setdefault(sigs[i, band * rows:(band + 1) * rows].tobytes(), []).append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    ri, rj = find(i), find(j)
                    if ri != rj and np.mean(sigs[i] == sigs[j]) >= threshold:
                        parent[max(ri, rj)] = min(ri, rj)  # the smaller index stays the root
    return [i for i in range(len(docs)) if find(i) == i]


@app.cell(hide_code=True)
def _():
    checks.run(3, minhash_dedup=minhash_dedup, minhash_signature=minhash_signature, shingles=shingles, jaccard=jaccard)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def quality_filter(text, *, min_words=20, word_len=(3.0, 10.0), max_symbol_ratio=0.1, min_alpha_frac=0.8,
                   max_dup_line_frac=0.3):
    """stage 4: Gopher-style heuristics; returns (keep, reasons) with reasons a list of the rules the text fails."""
    words = text.split()
    reasons = []
    if len(words) < min_words:
        reasons.append("too_short")
    if words:
        mean_len = sum(len(w) for w in words) / len(words)
        if not word_len[0] <= mean_len <= word_len[1]:
            reasons.append("word_length")
        symbols = text.count("#") + text.count("...") + text.count("…")
        if symbols / len(words) > max_symbol_ratio:
            reasons.append("symbols")
        if sum(any(c.isalpha() for c in w) for w in words) / len(words) < min_alpha_frac:
            reasons.append("non_alpha")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        seen, dups = set(), 0
        for ln in lines:
            dups += ln in seen
            seen.add(ln)
        if dups / len(lines) > max_dup_line_frac:
            reasons.append("repeated_lines")
    return not reasons, reasons


@app.cell(hide_code=True)
def _():
    checks.run(4, quality_filter=quality_filter)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def mixture_weights(sizes, *, temperature=1.0):
    """stage 5: sampling probabilities p_i proportional to sizes_i^(1/temperature); (n_sources,) summing to 1."""
    sizes = np.asarray(sizes, dtype=float)
    if temperature <= 0 or np.any(sizes < 0) or sizes.sum() <= 0:
        raise ValueError("temperature must be > 0 and sizes non-negative with a positive total")
    logw = np.where(sizes > 0, np.log(np.where(sizes > 0, sizes, 1.0)) / temperature, -np.inf)
    w = np.exp(logw - logw.max())
    return w / w.sum()


@app.function
def effective_tokens(unique_tokens, total_tokens, *, r_star=15.4):
    """stage 5: Muennighoff et al. (2023): U + U r* (1 - exp(-R / r*)), R = total / U - 1 repetitions (no repeats: total)."""
    if total_tokens <= unique_tokens:
        return float(total_tokens)
    repeats = total_tokens / unique_tokens - 1
    return float(unique_tokens + unique_tokens * r_star * (1 - math.exp(-repeats / r_star)))


@app.cell(hide_code=True)
def _():
    checks.run(5, mixture_weights=mixture_weights, effective_tokens=effective_tokens)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import minhash_dedup` deduplicates any list of documents; the capstone
    runs it over its pretraining corpus, and topic 06 reuses `shingles` to look for test
    items leaking into training data.
    """)
    return


if __name__ == "__main__":
    app.run()
