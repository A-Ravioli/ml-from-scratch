import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import heapq
    import re

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson

    # where in a paper a claim is supported: "figure 3", "Table 2", "sec. 4.1", "eq. (5)", "appendix B", "§3"
    LOCATION = re.compile(
        r"(?:\b(?i:figure|fig|table|tab|section|sec|equation|eq|appendix|app|algorithm|alg|theorem|thm|lemma)\b\.?"
        r"|§)\s*\(?[0-9A-Z]"
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # reading papers · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. together these make a small reading toolkit: parse a bibliography,
    lint your paper notes, order a reading list so foundations come first, and rank what to
    read next.

    | stage | you build |
    |---|---|
    | 1 | `parse_bibtex`: a BibTeX file → a list of entries |
    | 2 | `check_note`: a linter for structured paper notes (three-pass template) |
    | 3 | `citation_graph`, `reading_order`: who cites whom in your reading set, foundations first |
    | 4 | `pagerank`, `next_to_read`: which unread paper your reading set leans on most |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def parse_bibtex(text):
    """stage 1: list of entries {'type', 'key', field: value, ...} in file order; ValueError on malformed input."""
    raise NotImplementedError("stage1: implement parse_bibtex")


@app.cell(hide_code=True)
def _():
    checks.run(1, parse_bibtex=parse_bibtex)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def check_note(note, *, known_keys=None, max_summary_words=120):
    """stage 2: list of problems ('field: message') with a structured paper note; [] means it passes."""
    raise NotImplementedError("stage2: implement check_note")


@app.cell(hide_code=True)
def _():
    checks.run(2, check_note=check_note)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def citation_graph(notes):
    """stage 3: dict key -> sorted list of the keys it cites within the reading set (no self-citations, no duplicates)."""
    raise NotImplementedError("stage3: implement citation_graph")


@app.function
def reading_order(graph):
    """stage 3: every key after all the keys it cites, ties broken alphabetically; ValueError naming a cycle's keys."""
    raise NotImplementedError("stage3: implement reading_order")


@app.cell(hide_code=True)
def _():
    checks.run(3, citation_graph=citation_graph, reading_order=reading_order)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def pagerank(graph, *, damping=0.85, tol=1e-10, max_iters=1000):
    """stage 4: dict key -> PageRank score (sums to 1); citing passes credit to the cited; dangling papers spread evenly."""
    raise NotImplementedError("stage4: implement pagerank")


@app.function
def next_to_read(graph, read, k=5, *, damping=0.85):
    """stage 4: up to k unread keys by PageRank (highest first, ties alphabetical); ValueError for unknown read keys."""
    raise NotImplementedError("stage4: implement next_to_read")


@app.cell(hide_code=True)
def _():
    checks.run(4, pagerank=pagerank, next_to_read=next_to_read)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. then use the toolkit for
    real: export your reference manager's library as BibTeX, write a note per paper as you
    read it, and let `next_to_read` suggest what your reading set leans on that you haven't
    read yet.
    """)
    return


if __name__ == "__main__":
    app.run()
