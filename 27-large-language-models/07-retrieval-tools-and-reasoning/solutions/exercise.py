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
    # retrieval, tools & reasoning · build (reference solution)

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
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"need chunk_size >= 1 and 0 <= overlap < chunk_size, got {chunk_size}, {overlap}")
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


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
        self.dim, self.ngram = dim, ngram

    def _counts(self, text):
        t = " " + " ".join(text.lower().split()) + " "
        c = np.zeros(self.dim)
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(len(t) - n + 1):
                c[zlib.crc32(t[i:i + n].encode("utf-8")) % self.dim] += 1
        return c

    def fit(self, texts):
        """stage 2: idf_[j] = log((1 + N) / (1 + df_j)) + 1 over the texts' buckets; returns self."""
        df = np.zeros(self.dim)
        for t in texts:
            df += self._counts(t) > 0
        self.idf_ = np.log((1 + len(texts)) / (1 + df)) + 1
        return self

    def transform(self, texts):
        """stage 2: (len(texts), dim) float array: (1 + log count) * idf per bucket (0 where absent), unit rows."""
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            c = self._counts(t)
            out[i] = np.where(c > 0, 1 + np.log(np.maximum(c, 1)), 0.0) * self.idf_
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms > 0, norms, 1.0)


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
    q = np.atleast_2d(np.asarray(query, dtype=float))
    K = np.asarray(keys, dtype=float)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    kn = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)
    sims = qn @ kn.T
    k = min(k, K.shape[0])
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1, kind="stable")
    idx = np.take_along_axis(part, order, axis=1)
    scores = np.take_along_axis(sims, idx, axis=1)
    if np.ndim(query) == 1:
        return idx[0], scores[0]
    return idx, scores


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
        self.dim, self.n_lists, self.n_probe, self.seed = dim, n_lists, n_probe, seed
        self.centroids_ = None
        self._vecs = np.zeros((0, dim))
        self._ids = np.zeros(0, dtype=np.int64)
        self._lists = np.zeros(0, dtype=np.int64)

    @staticmethod
    def _unit(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    @property
    def ntotal(self):
        """stage 4: number of vectors stored."""
        return len(self._ids)

    def train(self, vectors):
        """stage 4: fit n_lists k-means centroids (mlfs.unsupervised.KMeans, rng=seed) on the unit-normalized vectors."""
        if self.n_lists > 0:
            km = KMeans(self.n_lists, n_init=1, rng=self.seed).fit(self._unit(vectors))
            self.centroids_ = self._unit(km.cluster_centers_)
        return self

    def add(self, vectors, ids=None):
        """stage 4: store unit-normalized vectors with ids (default: ntotal, ntotal + 1, ...); IVF assigns each to its nearest centroid."""
        v = self._unit(vectors)
        if v.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dimension {self.dim}, got {v.shape[1]}")
        if self.n_lists > 0 and self.centroids_ is None:
            raise RuntimeError("an IVF index (n_lists > 0) must be trained before vectors are added")
        new_ids = np.arange(self.ntotal, self.ntotal + len(v)) if ids is None else np.asarray(ids, dtype=np.int64)
        if len(new_ids) != len(v):
            raise ValueError("one id per vector")
        lists = (v @ self.centroids_.T).argmax(1) if self.n_lists > 0 else np.zeros(len(v), dtype=np.int64)
        self._vecs = np.vstack([self._vecs, v])
        self._ids = np.concatenate([self._ids, new_ids])
        self._lists = np.concatenate([self._lists, lists])
        return self

    def search(self, queries, k=5):
        """stage 4: (scores (q, k), ids (q, k)), best first; IVF searches the n_probe nearest lists; missing slots score -inf with id -1."""
        q = self._unit(queries)
        if self.n_lists > 0 and self.centroids_ is None:
            raise RuntimeError("train the index first")
        scores = np.full((len(q), k), -np.inf)
        ids = np.full((len(q), k), -1, dtype=np.int64)
        if self.n_lists == 0:
            probe = None
        else:
            probe = np.argsort(-(q @ self.centroids_.T), axis=1, kind="stable")[:, :self.n_probe]
        for i in range(len(q)):
            cand = np.arange(self.ntotal) if probe is None else np.flatnonzero(np.isin(self._lists, probe[i]))
            if len(cand) == 0:
                continue
            sims = self._vecs[cand] @ q[i]
            top = np.argsort(-sims, kind="stable")[:k]
            scores[i, :len(top)] = sims[top]
            ids[i, :len(top)] = self._ids[cand[top]]
        return scores, ids


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
    ctx = "".join(f"[{i}] {p}\n" for i, p in enumerate(passages, 1))
    return (f"Context:\n{ctx}\n" if passages else "") + f"Question: {question}\nAnswer:"


@app.function
def run_agent(llm, tools, task, *, max_steps=8):
    """stage 5: loop llm(transcript) -> 'CALL name(arg)' (run tools[name](arg), append 'RESULT: ...') or 'ANSWER: x'; (answer or None, transcript)."""
    transcript = f"TASK: {task}\n"
    for _ in range(max_steps):
        reply = llm(transcript).strip()
        transcript += reply + "\n"
        done = re.search(r"ANSWER:\s*(.*)", reply)
        if done:
            return done.group(1).strip(), transcript
        call = re.search(r"CALL\s+(\w+)\((.*)\)", reply)
        if call is None:
            transcript += "RESULT: error: reply with CALL tool(argument) or ANSWER: ...\n"
            continue
        name, arg = call.group(1), call.group(2).strip()
        if name not in tools:
            result = f"error: unknown tool {name}"
        else:
            try:
                result = str(tools[name](arg))
            except Exception as e:  # a failing tool is reported to the model, not raised
                result = f"error: {type(e).__name__}: {e}"
        transcript += f"RESULT: {result}\n"
    return None, transcript


@app.function
def majority_vote(answers):
    """stage 5: the most common answer (None ignored); ties go to the answer seen first; None if there are none."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    return next(a for a in valid if counts[a] == best)


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
