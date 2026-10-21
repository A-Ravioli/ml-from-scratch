import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import re
    from collections import Counter

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    ROOT = notebooks.repo_root()
    HERE = mo.notebook_dir()
    ARXIV = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")


@app.function
def course_reading_set():
    """the course as a reading set: one note per lesson citing the arXiv papers it links to, one per paper.

    returns (notes, titles): titles maps an arXiv id to the first quoted title found on a line that links it.
    """
    notes, titles = [], {}
    for path in sorted(ROOT.glob("[0-9][0-9]-*/[0-9][0-9]-*/lesson.md")):
        ids = set()
        for line in path.read_text().splitlines():
            for m in ARXIV.finditer(line):
                ids.add(m.group(1))
                t = re.search(r'"([^"]{8,})"', line)
                if t and m.group(1) not in titles:
                    titles[m.group(1)] = t.group(1)
        notes.append({"key": f"{path.parent.parent.name[:2]}·{path.parent.name[:2]}", "cites": sorted(ids)})
    papers = sorted({c for n in notes for c in n["cites"]})
    return notes + [{"key": p} for p in papers], titles


@app.function
def simulate_field(seed, *, n=400, m=8, popularity=1.0):
    """a growing field: paper t cites m earlier papers with probability ∝ quality × (citations + 1)^popularity.

    returns (quality (n,), graph): graph maps 'p000'… to the keys each paper cites; papers are numbered by time.
    """
    rng = np.random.default_rng(seed)
    quality = rng.lognormal(0.0, 0.7, n)
    cites_so_far = np.zeros(n)
    graph = {"p000": []}
    for t in range(1, n):
        w = quality[:t] * (cites_so_far[:t] + 1) ** popularity
        chosen = rng.choice(t, size=min(m, t), replace=False, p=w / w.sum())
        cites_so_far[chosen] += 1
        graph[f"p{t:03d}"] = [f"p{c:03d}" for c in chosen]
    return quality, graph


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # reading papers · lab

    every experiment here runs **your** toolkit from `exercise.py`: `citation_graph` builds the
    graphs, `pagerank` and `next_to_read` rank them. the first experiment uses real data (this
    course's own lessons); the second a simulated field where we know which papers are good.
    for each one: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. the lab runs in a few
    seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what does this course read?

    treat the course as a reading set: every lesson is a "paper" that cites the arXiv papers
    its *going deeper* section (and the rest of the text) links to. which arXiv paper do the
    most lessons link to? (only arXiv links count, so papers cited by DOI or a publisher's page
    are missing; that's a real limitation of the census.)
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which arXiv paper is linked from the most lessons of this course, and by how many? will PageRank put "
        "the same paper first?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    course_notes, course_titles = course_reading_set()
    course_graph = ex.citation_graph(course_notes)
    course_lessons = [n["key"] for n in course_notes if "cites" in n]
    course_papers = [n["key"] for n in course_notes if "cites" not in n]
    _pr = ex.pagerank(course_graph)
    _count = Counter(c for k in course_lessons for c in course_graph[k])
    _by_count = sorted(course_papers, key=lambda k: (-_count[k], k))[:10]
    _by_pr = sorted(course_papers, key=lambda k: (-_pr[k], k))[:10]

    def _row(i, k):
        return (f"| {i + 1} | arXiv:{k} | {course_titles.get(k, '?')[:60]} | {_count[k]} | "
                f"{1000 * _pr[k]:.2f} | {_by_pr.index(k) + 1 if k in _by_pr else '>10'} |")

    _stats = (f"{len(course_lessons)} lessons link to {len(course_papers)} distinct arXiv papers; "
              f"{sum(c == 1 for c in _count.values())} of them from a single lesson.")
    mo.vstack([
        mo.md(_stats),
        mo.md("| rank | paper | title | lessons | PageRank ×1000 | PageRank rank |\n|---|---|---|---|---|---|\n"
              + "\n".join(_row(i, k) for i, k in enumerate(_by_count))),
        mo.md("PageRank's top 10 that aren't in the count's top 10: " + ", ".join(
            f"arXiv:{k} ({course_titles.get(k, '?')[:40]}, {_count[k]} lessons, linked from "
            f"{', '.join(c for c in course_lessons if k in course_graph[c])})"
            for k in _by_pr if k not in _by_count) + "."),
    ])
    return course_graph, course_lessons, course_papers, course_titles


@app.cell
def _(course_graph, course_lessons, course_papers, course_titles):
    # a reading queue for someone who has finished R1 (chapters 00–18)
    _done = [k for k in course_lessons if int(k[:2]) <= 18]
    _read = set(course_lessons) | {p for k in _done for p in course_graph[k]}
    _queue = ex.next_to_read(course_graph, _read, k=5)
    mo.md(f"**a reading queue.** after chapters 00–18 you've met {len(_read) - len(course_lessons)} of the "
          f"{len(course_papers)} papers. `next_to_read` suggests, from the rest: "
          + "; ".join(f"arXiv:{k} ({course_titles.get(k, '?')[:50]})" for k in _queue) + ".")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do PageRank and the raw count disagree about some papers? look at
    *which* lessons link to the papers that PageRank promotes: how many papers do those lessons
    link to in total? is that a sensible way to weight a citation for a reading queue?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: do citation counts find the best papers?

    a simulated field of 400 papers, published one at a time. each paper has a hidden
    **quality** (log-normal), and cites 8 earlier papers chosen with probability proportional
    to quality × (citations so far + 1)$^{\gamma}$: good papers and already-cited papers are
    both more likely to be cited (the "fitness" model of Bianconi & Barabási, 2001). we
    compare four ways to rank papers against the hidden quality: raw citation count, citations
    per unit of time since publication, your PageRank, and an age-adjusted score that compares
    each paper's count with the 40 papers published closest to it in time (a z-score). numbers
    are averaged over 5 simulated fields.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "of the 10 most-cited papers, how many come from the oldest 20% of the field? does PageRank or "
        "citations-per-year fix that?"
    )
    p2
    return (p2,)


@app.cell
def _():
    gamma = mo.ui.slider(0.0, 1.5, value=1.0, step=0.25, label="γ: preference for already-cited papers")
    gamma
    return (gamma,)


@app.cell
def _(gamma, p2):
    ui.gate(p2)
    _scores = {"citations": [], "citations per year": [], "PageRank": [], "age-adjusted z": []}
    _rows = {k: {"rho_q": [], "rho_age": [], "old": [], "overlap": []} for k in _scores}
    for _seed in range(5):
        _q, _g = simulate_field(_seed, popularity=gamma.value)
        _n = len(_q)
        _keys = list(_g)
        _count = np.zeros(_n)
        for _k in _keys:
            for _c in _g[_k]:
                _count[int(_c[1:])] += 1
        _age = _n - np.arange(_n)
        _pr = ex.pagerank(ex.citation_graph([{"key": k, "cites": v} for k, v in _g.items()]))
        _z = np.array([(_count[t] - _count[max(0, t - 20):t + 21].mean()) / (_count[max(0, t - 20):t + 21].std() + 1e-9)
                       for t in range(_n)])
        _s = {"citations": _count, "citations per year": _count / _age,
              "PageRank": np.array([_pr[k] for k in _keys]), "age-adjusted z": _z}
        _best = set(np.argsort(-_q)[:10])
        for _name, _v in _s.items():
            _top = np.argsort(-_v, kind="stable")[:10]
            _rows[_name]["rho_q"].append(spearmanr(_v, _q)[0])
            _rows[_name]["rho_age"].append(spearmanr(_v, _age)[0])
            _rows[_name]["old"].append(int(np.sum(_top < _n // 5)))
            _rows[_name]["overlap"].append(len(_best & set(_top)))
        if _seed == 0:
            _field0 = (_q, _count, _z)
    _table = "\n".join(
        f"| {name} | {np.mean(r['rho_q']):.2f} | {np.mean(r['rho_age']):+.2f} | {np.mean(r['old']):.1f} | "
        f"{np.mean(r['overlap']):.1f} |" for name, r in _rows.items())
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    _q0, _c0, _z0 = _field0
    for _ax, _y, _lab in ((_axes[0], _c0 + 1, "citations + 1 (log scale)"), (_axes[1], _z0, "age-adjusted z")):
        _sc = _ax.scatter(np.arange(len(_q0)), _y, c=np.log(_q0), s=10, cmap="viridis")
        _ax.set(xlabel="publication order (0 = oldest)", ylabel=_lab)
    _axes[0].set_yscale("log")
    _fig.colorbar(_sc, ax=_axes, label="log quality")
    _axes[0].set_title("raw counts: the oldest papers win")
    _axes[1].set_title("compared with contemporaries")
    mo.vstack([
        mo.md("| ranking | Spearman ρ with quality | ρ with age | top 10 from the oldest 20% | top 10 that are "
              "truly top-10 quality |\n|---|---|---|---|---|\n" + _table),
        mo.md(f"*(γ = {gamma.value}; by chance, about 2 of the 10 best papers fall in the oldest 20%.)*"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does PageRank inherit the age bias of raw counts, and why doesn't
    dividing by the time since publication remove it? set γ = 0 (no preference for cited
    papers): how much of the bias is left, and where does it come from? what would you do
    differently when building a reading list from citation counts?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## your reading set

    this topic's deliverable is ten annotated papers in your thesis area. put your
    bibliography in `my_reading.bib` and your notes (a JSON list of note dicts, in the
    template of idea 2) in `my_notes.json`, next to this notebook. this cell parses, lints and
    orders them with your toolkit; it re-runs whenever you re-run it.
    """)
    return


@app.cell
def _():
    _bib, _notes = HERE / "my_reading.bib", HERE / "my_notes.json"
    if not _bib.exists() or not _notes.exists():
        _out = mo.md(f"*(no `my_reading.bib` and `my_notes.json` in `{HERE.name}/` yet.)*")
    else:
        _entries = ex.parse_bibtex(_bib.read_text())
        _keys = [e["key"] for e in _entries]
        _mine = json.loads(_notes.read_text())
        _lint = {n.get("key", f"note {i}"): ex.check_note(n, known_keys=_keys) for i, n in enumerate(_mine)}
        _g = ex.citation_graph(_mine)
        try:
            _order = ", ".join(ex.reading_order(_g))
        except ValueError as e:
            _order = f"no order: {e}"
        _out = mo.md(
            f"{len(_entries)} bibliography entries, {len(_mine)} notes, "
            f"{sum(not p for p in _lint.values())} passing the linter.\n\n"
            + "\n".join(f"- **{k}**: " + ("ok" if not p else "; ".join(p)) for k, p in _lint.items())
            + f"\n\nreading order (foundations first): {_order}\n\n"
            f"read next: {', '.join(ex.next_to_read(_g, [], k=5))}")
    _out
    return


if __name__ == "__main__":
    app.run()
