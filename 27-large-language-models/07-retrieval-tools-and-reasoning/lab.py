import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.llm import minhash_dedup  # yours, from topic 01 (only to show the knowledge base has no duplicates)

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    OFFLINE = datasets.is_offline()
    N_COUNTRIES = 300 if OFFLINE else 1500
    FILLER = [ln.strip() for ln in datasets.tiny_shakespeare().splitlines() if len(ln.split()) >= 6][:4000]


@app.function
def make_kb(n_countries, seed=0):
    """a synthetic knowledge base: (documents, facts). facts[country] = dict(capital, ruler, river, founded, cousin);
    facts[city] = dict(population). every country has a 'cousin' with a similar name (a distractor for retrievers and
    readers). documents mix each entity's facts, in random order, with lines of Shakespeare as filler."""
    rng = np.random.default_rng(seed)
    syl = ["ka", "lo", "mir", "ven", "dra", "sel", "tor", "una", "bel", "quo", "rin", "zha", "phe", "ost", "lia",
           "gor", "nem", "ari", "thu", "cel"]
    names = set()

    def name(k, unique=True):
        for _ in range(1000):  # 20^3 = 8,000 three-syllable names: plenty for 1,500 countries and their capitals
            w = "".join(rng.choice(syl, size=k)).capitalize()
            if not unique or w not in names:
                names.add(w)
                return w
        raise RuntimeError("ran out of names")
    facts, docs = {}, []
    countries = []
    for _ in range(n_countries // 2):
        base = name(3)
        for country in (base, base + "ia"):  # a country and its similarly named cousin
            if country in facts:
                continue
            city = name(3)
            facts[country] = dict(capital=city, ruler=name(2, unique=False) + " the " +
                                  rng.choice(["Bold", "Wise", "Pale", "Just"]), river=name(2, unique=False),
                                  founded=int(rng.integers(800, 1600)))
            facts[city] = dict(population=int(rng.integers(2, 900)) * 1000)
            countries.append(country)
    for c in countries:
        cousin = c + "ia" if c + "ia" in facts else c[:-2]
        facts[c]["cousin"] = cousin
    for c in countries:
        f = facts[c]
        sents = [f"The capital of {c} is {f['capital']}.", f"{c} is ruled by {f['ruler']}.",
                 f"The river {f['river']} flows through {c}.", f"{c} was founded in {f['founded']}."]
        rng.shuffle(sents)
        filler = [FILLER[int(i)] for i in rng.integers(0, len(FILLER), 6)]
        docs.append(" ".join(s for pair in zip(sents, filler) for s in pair) + " " + " ".join(filler[4:]))
        city = f["capital"]
        docs.append(f"{city} is a city of {c}. " + " ".join(FILLER[int(i)] for i in rng.integers(0, len(FILLER), 3))
                    + f" {city} has a population of {facts[city]['population']}. "
                    + " ".join(FILLER[int(i)] for i in rng.integers(0, len(FILLER), 3)))
    return docs, facts, countries


@app.function
def read_capital(passages, country):
    """a toy extractive reader (no model): the first passage, in rank order, with 'The capital of X is Y.' where X looks
    like the asked country (same first five letters), answers Y. it can be fooled by a cousin with a similar name."""
    for p in passages:
        for x, y in re.findall(r"The capital of (\w+) is (\w+)\.", p):
            if x[:5] == country[:5]:
                return y
    return None


@app.function
def read_population(passages, city):
    for p in passages:
        m = re.search(rf"{city} has a population of (\d+)\.", p)
        if m:
            return int(m.group(1))
    return None


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # retrieval, tools & reasoning · lab

    the experiments run **your** `chunk_text`, `HashingEmbedder`, `cosine_top_k`,
    `VectorIndex`, `build_rag_prompt` and `run_agent` over a **synthetic** knowledge base
    (`make_kb`, {N_COUNTRIES} invented countries and their capitals, with lines of Shakespeare
    as filler): there is no document collection or question-answering model on this machine.
    the "reader" that turns retrieved passages into an answer is a few lines of pattern matching
    (`read_capital`), and in experiment 3 the "LLM" that drives the agent is a script: both are
    stand-ins, written to make one thing visible, what retrieval does and doesn't allow. every
    number is one run. for each experiment: write down a prediction, lock it in, run it, then
    explain what happened. about a minute.
    """)
    return


@app.cell
def _():
    docs, facts, countries = make_kb(N_COUNTRIES)
    chunks = [c for d in docs for c in ex.chunk_text(d, chunk_size=40, overlap=10)]
    _dups = 400 - len(minhash_dedup(chunks[:400]))
    mo.md(f"{len(docs):,} documents, cut by your `chunk_text` (40 words, overlap 10) into **{len(chunks):,} chunks** "
          f"(near-duplicates among the first 400: {_dups}). a chunk:\n\n> {chunks[0]}")
    return chunks, countries, facts


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: retrieval caps answer quality

    300 questions "What is the capital of X?". for several retrievers (your `HashingEmbedder`
    with 32 to 4,096 buckets, top-1 to top-10 chunks), two numbers: **recall@k**, the fraction
    of questions for which a retrieved chunk contains the true sentence, and **answer
    accuracy**, the reader's exact-match score on the same retrieved chunks.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("can answer accuracy ever exceed recall@k? with 4,096 buckets and top-10 retrieval, how close "
                    "to 100% will accuracy get, and what will stop it?")
    p1
    return (p1,)


@app.cell
def _(chunks, countries, facts, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(1)
    _qs = [countries[int(i)] for i in _rng.choice(len(countries), min(300, len(countries)), replace=False)]
    _questions = [f"What is the capital of {c}?" for c in _qs]
    rag_rows = []
    for _dim in (32, 256, 4096):
        _emb = ex.HashingEmbedder(dim=_dim).fit(chunks)
        _K = _emb.transform(chunks)
        _Q = _emb.transform(_questions)
        _top, _ = ex.cosine_top_k(_Q, _K, k=10)
        for _k in (1, 3, 10):
            _rec, _acc, _fooled = [], [], 0
            for _c, _row in zip(_qs, _top):
                _ps = [chunks[i] for i in _row[:_k]]
                _rec.append(any(f"The capital of {_c} is {facts[_c]['capital']}." in p for p in _ps))
                _ans = read_capital(_ps, _c)
                _acc.append(_ans == facts[_c]["capital"])
                _fooled += _ans is not None and _ans != facts[_c]["capital"]
            rag_rows.append((_dim, _k, np.mean(_rec), np.mean(_acc), _fooled / len(_qs)))
    _fig, _ax = plt.subplots(figsize=(5, 4))
    for _dim, _m in zip((32, 256, 4096), "o^s"):
        _r = [r for r in rag_rows if r[0] == _dim]
        _ax.plot([r[2] for r in _r], [r[3] for r in _r], _m + "-", label=f"{_dim} buckets (k = 1, 3, 10)")
    _ax.plot([0, 1], [0, 1], "k:", lw=1, label="accuracy = recall")
    _ax.annotate("32 buckets: all three k\nat recall 0, accuracy 0", (0.0, 0.0), xytext=(0.08, 0.15), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    _ax.set(xlabel="recall@k", ylabel="answer accuracy", xlim=(0, 1.02), ylim=(0, 1.02), aspect="equal")
    _ax.legend(fontsize=7)
    _prompt = ex.build_rag_prompt(_questions[0], [chunks[i] for i in _top[0][:2]])
    mo.vstack([_fig, mo.md("| buckets | k | recall@k | answer accuracy | wrong answer from a cousin |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {d} | {k} | {r:.2f} | {a:.2f} | {f:.2f} |" for d, k, r, a, f in rag_rows)
                           + f"\n\nwhat a model would be given (your `build_rag_prompt`, top 2):\n\n```\n{_prompt}\n```")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can't any reader beat recall@k? why does a 32-bucket embedder retrieve
    so badly (think about collisions among character n-grams)? the points fall below the
    diagonal: what fools the reader when the right chunk *was* retrieved, and why does that
    happen more often with large $k$? what would a real LLM reader change, and what wouldn't it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: approximate nearest neighbours

    your `VectorIndex` over all chunks (4,096 buckets), exact and as an inverted-file index
    with 64 k-means lists probing 1 to 64 of them. for 300 queries: recall@5 against exact
    search, the fraction of the chunks each query actually compares against, and wall-clock time.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("probing 4 of 64 lists scans about 6% of the chunks. what fraction of the true top-5 does it "
                    "find? and is it faster than exact search here?")
    p2
    return (p2,)


@app.cell
def _(chunks, countries, p2):
    ui.gate(p2)
    _emb = ex.HashingEmbedder(dim=4096).fit(chunks)
    _K = _emb.transform(chunks)
    _rng = np.random.default_rng(2)
    _Q = _emb.transform([f"What is the capital of {countries[int(i)]}?"
                         for i in _rng.choice(len(countries), min(300, len(countries)), replace=False)])
    _t0 = time.perf_counter()
    _exact = ex.VectorIndex(4096)
    _exact.add(_K)
    _, _truth = _exact.search(_Q, k=5)
    _t_exact = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    _, _np_ids = ex.cosine_top_k(_Q, _K, k=5)
    _t_np = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    _n_lists = min(64, len(chunks) // 20)
    _ivf = ex.VectorIndex(4096, n_lists=_n_lists, seed=0).train(_K[::2])
    _ivf.add(_K)
    _t_build = time.perf_counter() - _t0
    _sizes = np.bincount(_ivf._lists, minlength=_n_lists) if hasattr(_ivf, "_lists") else None
    ann_rows = []
    for _p in (1, 2, 4, 8, 16, _n_lists):
        _ivf.n_probe = _p
        _t0 = time.perf_counter()
        _, _ids = _ivf.search(_Q, k=5)
        _t = time.perf_counter() - _t0
        _rec = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(_ids, _truth)])
        ann_rows.append((_p, _rec, _p / _n_lists, _t))
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _ax.semilogx([r[2] for r in ann_rows], [r[1] for r in ann_rows], "o-")
    _ax.set(xlabel="fraction of lists probed", ylabel="recall@5 vs exact search", ylim=(0, 1.05))
    mo.vstack([_fig, mo.md(
        f"exact search with your `VectorIndex`: {1e3 * _t_exact:.0f} ms for {len(_Q)} queries; with `cosine_top_k` (one "
        f"matrix product): {1e3 * _t_np:.0f} ms. building the IVF index ({_n_lists} lists, k-means on half the chunks): "
        f"{1e3 * _t_build:.0f} ms.\n\n| lists probed | recall@5 | fraction probed | time for {len(_Q)} queries |\n"
        "|---|---|---|---|\n" + "\n".join(f"| {p} | {r:.2f} | {f:.3f} | {1e3 * t:.0f} ms |" for p, r, f, t in ann_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does probing a few lists find most true neighbours (where do a
    query's neighbours sit relative to the k-means cells)? why is this index *slower* than an
    exact matrix product on a few thousand chunks, and at what scale (and with what
    implementation) does IVF win? what does the build cost buy you?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: one retrieval, or an agent loop?

    two-hop questions: "What is the population of the capital of X?". the population is in the
    capital's own document, which never mentions X's name next to it. **single-shot RAG**
    retrieves the top 3 chunks for the question and reads them. the **agent** runs your
    `run_agent` with a search tool (top 3 chunks) and a calculator, driven by a scripted
    policy standing in for an LLM: search the capital, read it, search its population, read
    it, answer. the same 150 questions, 4,096 buckets.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("how often will single-shot retrieval answer correctly? and the agent? what limits the agent?")
    p3
    return (p3,)


@app.cell
def _(chunks, countries, facts, p3):
    ui.gate(p3)
    _emb = ex.HashingEmbedder(dim=4096).fit(chunks)
    _index = ex.VectorIndex(4096)
    _index.add(_emb.transform(chunks))

    def search(query):
        _, _ids = _index.search(_emb.transform([query]), k=3)
        return " || ".join(chunks[i] for i in _ids[0] if i >= 0)

    def calc(expr):
        if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
            raise ValueError("only arithmetic")
        return eval(expr)  # noqa: S307 (digits and operators only)

    def policy(transcript):
        """a scripted stand-in for an LLM: decides the next action from the transcript so far."""
        country = re.search(r"capital of (\w+)\?", transcript).group(1)
        results = re.findall(r"RESULT: (.*)", transcript)
        if not results:
            return f"I need the capital first. CALL search(What is the capital of {country}?)"
        city = read_capital(results[0].split(" || "), country)
        if city is None:
            return "ANSWER: unknown"
        if len(results) == 1:
            return f"The capital is {city}. CALL search(What is the population of {city}?)"
        pop = read_population(results[1].split(" || "), city)
        if pop is None:
            return "ANSWER: unknown"
        if len(results) == 2 and "twice" in transcript:
            return f"CALL calc({pop} * 2)"
        return f"ANSWER: {results[-1] if len(results) == 3 else pop}"

    _rng = np.random.default_rng(3)
    _qs = [countries[int(i)] for i in _rng.choice(len(countries), min(150, len(countries)), replace=False)]
    _single, _agent, _calls, _hop1 = [], [], [], []
    for _c in _qs:
        _truth = facts[facts[_c]["capital"]]["population"]
        _q = f"What is the population of the capital of {_c}?"
        _ps = search(_q).split(" || ")
        _city = read_capital(_ps, _c)
        _single.append(_city is not None and read_population(_ps, _city) == _truth)
        _ans, _tr = ex.run_agent(policy, {"search": search, "calc": calc}, _q, max_steps=6)
        _agent.append(_ans == str(_truth))
        _calls.append(_tr.count("CALL "))
        _hop1.append(read_capital(search(f"What is the capital of {_c}?").split(" || "), _c) == facts[_c]["capital"])
    _ans, _tr = ex.run_agent(policy, {"search": search, "calc": calc},
                             f"What is twice the population of the capital of {_qs[0]}?", max_steps=6)
    mo.vstack([mo.md(f"| method | accuracy | tool calls per question |\n|---|---|---|\n"
                     f"| single-shot RAG (top 3) | {np.mean(_single):.2f} | 1 retrieval |\n"
                     f"| agent loop (search, search, answer) | **{np.mean(_agent):.2f}** | {np.mean(_calls):.1f} |\n\n"
                     f"the agent's first hop alone (finding the capital) succeeds for {np.mean(_hop1):.2f} of the "
                     f"questions. a transcript, with the calculator:"),
               mo.md(f"```\n{_tr[:900]}\n```")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does single-shot retrieval almost never find the population? what
    does the agent know after its first step that the original question didn't contain? its
    accuracy is roughly (first-hop success) × (second-hop success): why does each extra hop
    multiply the chance of failure, and what does that mean for long tool-using chains? which
    part of this agent would a real LLM replace, and what new ways of failing would it bring?
    """)
    return


if __name__ == "__main__":
    app.run()
