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
    # reading papers · build (reference solution)

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
    entries, seen, i, n = [], set(), 0, len(text)
    while True:
        at = text.find("@", i)
        if at < 0:
            return entries
        brace = text.find("{", at)
        if brace < 0:
            raise ValueError(f"'@' at position {at} isn't followed by '{{'")
        etype = text[at + 1:brace].strip().lower()
        if not re.fullmatch(r"[a-z]+", etype):
            raise ValueError(f"bad entry type {etype!r} at position {at}")
        depth, j = 1, brace + 1
        while j < n and depth:  # find the matching closing brace
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth:
            raise ValueError(f"unbalanced braces in the @{etype} entry at position {at}")
        body, i = text[brace + 1:j - 1], j
        if etype in ("comment", "preamble", "string"):
            continue
        key, _, rest = body.partition(",")
        key = key.strip()
        if not key or re.search(r"[\s={}\"]", key):
            raise ValueError(f"@{etype} entry at position {at} has no valid citation key")
        if key in seen:
            raise ValueError(f"duplicate citation key {key!r}")
        seen.add(key)
        entry = {"type": etype, "key": key}
        k, m = 0, len(rest)
        while True:
            while k < m and rest[k] in " \t\r\n,":
                k += 1
            if k >= m:
                break
            eq = rest.find("=", k)
            if eq < 0:
                raise ValueError(f"field without '=' in {key!r}: {rest[k:k + 30]!r}")
            name = rest[k:eq].strip().lower()
            if not re.fullmatch(r"[a-z][a-z0-9_\-]*", name):
                raise ValueError(f"bad field name {name!r} in {key!r}")
            k = eq + 1
            while k < m and rest[k] in " \t\r\n":
                k += 1
            if k < m and rest[k] == "{":
                depth, start = 1, k + 1
                k += 1
                while k < m and depth:
                    depth += {"{": 1, "}": -1}.get(rest[k], 0)
                    k += 1
                value = rest[start:k - 1]
            elif k < m and rest[k] == '"':
                end = rest.find('"', k + 1)
                if end < 0:
                    raise ValueError(f"unterminated quoted value in {key!r}")
                value, k = rest[k + 1:end], end + 1
            else:
                start = k
                while k < m and rest[k] not in ",\n":
                    k += 1
                value = rest[start:k].strip()
                if not re.fullmatch(r"[A-Za-z0-9_\-.:]+", value):
                    raise ValueError(f"bad bare value {value!r} for {name!r} in {key!r}")
            if name in entry:
                raise ValueError(f"field {name!r} appears twice in {key!r}")
            entry[name] = " ".join(value.replace("{", "").replace("}", "").split())
        entries.append(entry)


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
    problems = []

    def text_ok(field):
        v = note.get(field)
        if not isinstance(v, str) or not v.strip():
            problems.append(f"{field}: missing or empty")
            return False
        return True

    def list_ok(field):
        v = note.get(field)
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
            problems.append(f"{field}: needs a non-empty list of non-empty strings")
            return False
        return True

    for field in ("key", "title", "category", "summary"):
        text_ok(field)
    year = note.get("year")
    if not isinstance(year, int) or isinstance(year, bool) or not 1900 <= year <= 2100:
        problems.append(f"year: must be an integer between 1900 and 2100, got {year!r}")
    passes = note.get("pass")
    if passes not in (1, 2, 3) or isinstance(passes, bool):
        problems.append(f"pass: must be 1, 2 or 3, got {passes!r}")
        passes = 1
    if isinstance(note.get("summary"), str) and len(note["summary"].split()) > max_summary_words:
        problems.append(f"summary: {len(note['summary'].split())} words, more than {max_summary_words}")
    claims = note.get("claims")
    if not isinstance(claims, list) or not claims:
        problems.append("claims: needs at least one claim")
        claims = []
    for i, c in enumerate(claims):
        if not isinstance(c, dict) or not isinstance(c.get("claim"), str) or not c["claim"].strip():
            problems.append(f"claims[{i}].claim: missing or empty")
            continue
        if passes >= 2:
            ev = c.get("evidence")
            if not isinstance(ev, str) or not LOCATION.search(ev):
                problems.append(f"claims[{i}].evidence: must point to where the paper supports it "
                                "(figure 3, table 2, section 4.1, eq. 5, appendix B, ...)")
    if passes >= 2:
        text_ok("method")
    if passes >= 3:
        for field in ("assumptions", "limitations", "questions"):
            list_ok(field)
    cites = note.get("cites", [])
    if not isinstance(cites, list) or not all(isinstance(x, str) for x in cites):
        problems.append("cites: must be a list of citation keys")
    elif known_keys is not None:
        known = set(known_keys)
        for k in cites:
            if k not in known:
                problems.append(f"cites: unknown key {k!r}")
    return problems


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
    keys = [n["key"] for n in notes]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"duplicate keys in the reading set: {dup}")
    known = set(keys)
    return {n["key"]: sorted({c for c in n.get("cites", []) if c in known and c != n["key"]}) for n in notes}


@app.function
def reading_order(graph):
    """stage 3: every key after all the keys it cites, ties broken alphabetically; ValueError naming a cycle's keys."""
    remaining = {k: len(set(v)) for k, v in graph.items()}  # how many of its references are still unread
    cited_by = {k: [] for k in graph}
    for k, refs in graph.items():
        for r in set(refs):
            if r not in graph:
                raise ValueError(f"{k!r} cites {r!r}, which isn't in the graph")
            cited_by[r].append(k)
    ready = [k for k, c in remaining.items() if c == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        k = heapq.heappop(ready)
        order.append(k)
        for citer in cited_by[k]:
            remaining[citer] -= 1
            if remaining[citer] == 0:
                heapq.heappush(ready, citer)
    if len(order) < len(graph):
        stuck = sorted(k for k in graph if k not in set(order))
        raise ValueError(f"{stuck} couldn't be placed (they include a citation cycle)")
    return order


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
    keys = list(graph)
    n = len(keys)
    if n == 0:
        return {}
    if not 0 <= damping < 1:
        raise ValueError(f"damping must be in [0, 1), got {damping}")
    idx = {k: i for i, k in enumerate(keys)}
    out_deg = np.array([len(set(graph[k])) for k in keys], dtype=float)
    src, dst = [], []
    for k in keys:
        for c in set(graph[k]):
            if c not in idx:
                raise ValueError(f"{k!r} cites {c!r}, which isn't in the graph")
            src.append(idx[k])
            dst.append(idx[c])
    src, dst = np.array(src, dtype=int), np.array(dst, dtype=int)
    dangling = out_deg == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iters):
        new = np.full(n, (1 - damping) / n + damping * r[dangling].sum() / n)
        if len(src):
            np.add.at(new, dst, damping * r[src] / out_deg[src])
        done = np.abs(new - r).sum() < tol
        r = new
        if done:
            break
    r = r / r.sum()
    return {k: float(r[i]) for i, k in enumerate(keys)}


@app.function
def next_to_read(graph, read, k=5, *, damping=0.85):
    """stage 4: up to k unread keys by PageRank (highest first, ties alphabetical); ValueError for unknown read keys."""
    read = set(read)
    unknown = sorted(read - set(graph))
    if unknown:
        raise ValueError(f"keys marked as read aren't in the graph: {unknown}")
    pr = pagerank(graph, damping=damping)
    unread = [key for key in graph if key not in read]
    return sorted(unread, key=lambda key: (-pr[key], key))[:k]


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
