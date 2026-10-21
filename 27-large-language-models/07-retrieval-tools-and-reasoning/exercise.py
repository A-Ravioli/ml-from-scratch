import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re
    import zlib
    from collections import Counter

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.unsupervised import KMeans  # yours, from ch 11 · 01


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # retrieval, tools & reasoning · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `chunk_text`: documents cut into overlapping word windows → `mlfs.llm` |
    | 2 | `HashingEmbedder`: TF-IDF vectors of hashed character n-grams, a retriever with no training |
    | 3 | `cosine_top_k`: exact nearest neighbours by cosine similarity → `mlfs.llm` |
    | 4 | `VectorIndex`: exact search, or an inverted-file index that probes a few k-means cells → `mlfs.llm` |
    | 5 | `build_rag_prompt`, `run_agent`, `majority_vote`: retrieval-augmented prompts, a tool-use loop, self-consistency |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def chunk_text(text, chunk_size=200, overlap=50):
    """stage 1: word windows of chunk_size words starting every chunk_size - overlap words, until one reaches the end."""
    raise NotImplementedError("stage1: implement chunk_text")


@app.cell(hide_code=True)
def _():
    checks.run(1, chunk_text=chunk_text)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class HashingEmbedder:
    """TF-IDF over character n-grams hashed into dim buckets (crc32), rows L2-normalized; fit learns the IDF."""

    def __init__(self, dim=1024, *, ngram=(3, 5)):
        """stage 2: store dim and the (min, max) n-gram lengths."""
        raise NotImplementedError("stage2: implement HashingEmbedder.__init__")

    def _counts(self, text):
        t = " " + " ".join(text.lower().split()) + " "
        c = np.zeros(self.dim)
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(len(t) - n + 1):
                c[zlib.crc32(t[i:i + n].encode("utf-8")) % self.dim] += 1
        return c

    def fit(self, texts):
        """stage 2: idf_[j] = log((1 + N) / (1 + df_j)) + 1 over the texts' buckets; returns self."""
        raise NotImplementedError("stage2: implement HashingEmbedder.fit")

    def transform(self, texts):
        """stage 2: (len(texts), dim) float array: (1 + log count) * idf per bucket (0 where absent), unit rows."""
        raise NotImplementedError("stage2: implement HashingEmbedder.transform")


@app.cell(hide_code=True)
def _():
    checks.run(2, HashingEmbedder=HashingEmbedder)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def cosine_top_k(query, keys, k=5):
    """stage 3: query (d,) or (q, d), keys (n, d) -> (indices, scores), each (k,) or (q, k), best first; k capped at n."""
    raise NotImplementedError("stage3: implement cosine_top_k")


@app.cell(hide_code=True)
def _():
    checks.run(3, cosine_top_k=cosine_top_k)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class VectorIndex:
    """cosine-similarity vector index: exact (n_lists=0) or IVF with a k-means coarse quantizer probing n_probe lists."""

    def __init__(self, dim, *, n_lists=0, n_probe=1, seed=0):
        """stage 4: an empty index for dim-dimensional vectors."""
        raise NotImplementedError("stage4: implement VectorIndex.__init__")

    @staticmethod
    def _unit(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    @property
    def ntotal(self):
        """stage 4: number of vectors stored."""
        raise NotImplementedError("stage4: implement VectorIndex.ntotal")

    def train(self, vectors):
        """stage 4: fit n_lists k-means centroids (mlfs.unsupervised.KMeans, rng=seed) on the unit-normalized vectors."""
        raise NotImplementedError("stage4: implement VectorIndex.train")

    def add(self, vectors, ids=None):
        """stage 4: store unit-normalized vectors with ids (default: ntotal, ntotal + 1, ...); IVF assigns each to its nearest centroid."""
        raise NotImplementedError("stage4: implement VectorIndex.add")

    def search(self, queries, k=5):
        """stage 4: (scores (q, k), ids (q, k)), best first; IVF searches the n_probe nearest lists; missing slots score -inf with id -1."""
        raise NotImplementedError("stage4: implement VectorIndex.search")


@app.cell(hide_code=True)
def _():
    checks.run(4, VectorIndex=VectorIndex, cosine_top_k=cosine_top_k)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def build_rag_prompt(question, passages):
    """stage 5: 'Context:\\n[1] p1\\n[2] p2\\n...\\n\\nQuestion: {question}\\nAnswer:' (just the question line if no passages)."""
    raise NotImplementedError("stage5: implement build_rag_prompt")


@app.function
def run_agent(llm, tools, task, *, max_steps=8):
    """stage 5: loop llm(transcript) -> 'CALL name(arg)' (run tools[name](arg), append 'RESULT: ...') or 'ANSWER: x'; (answer or None, transcript)."""
    raise NotImplementedError("stage5: implement run_agent")


@app.function
def majority_vote(answers):
    """stage 5: the most common answer (None ignored); ties go to the answer seen first; None if there are none."""
    raise NotImplementedError("stage5: implement majority_vote")


@app.cell(hide_code=True)
def _():
    checks.run(5, build_rag_prompt=build_rag_prompt, run_agent=run_agent, majority_vote=majority_vote)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import chunk_text, cosine_top_k, VectorIndex` gives you the retrieval half
    of a RAG system; the capstone and chapter 30 can plug any embedder into it.
    """)
    return


if __name__ == "__main__":
    app.run()
