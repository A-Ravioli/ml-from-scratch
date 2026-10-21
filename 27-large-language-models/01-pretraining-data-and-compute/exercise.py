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
    # pretraining data & compute · build

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
    raise NotImplementedError("stage1: implement shingles")


@app.function
def jaccard(a, b):
    """stage 1: |a & b| / |a | b| for sets (1.0 for two empty sets); for two signatures, the fraction of equal entries."""
    raise NotImplementedError("stage1: implement jaccard")


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
    raise NotImplementedError("stage2: implement minhash_signature")


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
    raise NotImplementedError("stage3: implement minhash_dedup")


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
    raise NotImplementedError("stage4: implement quality_filter")


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
    raise NotImplementedError("stage5: implement mixture_weights")


@app.function
def effective_tokens(unique_tokens, total_tokens, *, r_star=15.4):
    """stage 5: Muennighoff et al. (2023): U + U r* (1 - exp(-R / r*)), R = total / U - 1 repetitions (no repeats: total)."""
    raise NotImplementedError("stage5: implement effective_tokens")


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
