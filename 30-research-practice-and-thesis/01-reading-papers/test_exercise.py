"""Checks for 30 · 01 reading papers.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. parse_bibtex on a hand-written
bibliography (nested braces, quotes, bare values, comments) and on every malformed input it must refuse, without
hanging; check_note rule by rule (one failing note per rule); citation_graph and reading_order on a hand example,
random DAGs (the order must be the alphabetically smallest valid one) and cycles; pagerank against the Google
matrix's eigenvector, and next_to_read for exclusions, ties and its parameters.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

BIB = r"""
% my reading list, exported from a reference manager
@Article{He2016,
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  Title  = {Deep Residual Learning for {Image}
            Recognition},
  year   = 2016,
  booktitle = "Proceedings of {CVPR}",
}

@comment{this {whole} entry is ignored, even @article{fake, title={x}}}

@inproceedings{vaswani2017attention, title={Attention is All you Need}, year={2017}, month=dec}
"""


def _pr(nb, *args, **kwargs):
    return bounded(nb.pagerank, *args, what="pagerank", hint="cap the iterations at max_iters.", **kwargs)


def _ntr(nb, *args, **kwargs):
    return bounded(nb.next_to_read, *args, what="next_to_read", **kwargs)


def _parse(nb, text):
    return bounded(nb.parse_bibtex, text, what="parse_bibtex",
                   hint="every scanning loop must stop at the end of the text, even if a brace is never closed.")


# ---------------------------------------------------------------- stage 1: parse_bibtex


def test_stage1_worked_example(nb):
    got = _parse(nb, BIB)
    expected = [
        {"type": "article", "key": "He2016",
         "author": "He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian",
         "title": "Deep Residual Learning for Image Recognition", "year": "2016",
         "booktitle": "Proceedings of CVPR"},
        {"type": "inproceedings", "key": "vaswani2017attention", "title": "Attention is All you Need",
         "year": "2017", "month": "dec"},
    ]
    assert isinstance(got, list) and len(got) == 2, (
        f"the bibliography has two real entries (the @comment and the text outside entries are ignored), got "
        f"{len(got) if isinstance(got, list) else got!r}"
    )
    for g, e in zip(got, expected):
        assert g == e, (
            f"entry {e['key']} parsed wrong.\nexpected {e}\ngot      {g}\n(types and field names are lowercased, "
            "the key keeps its case, braces inside values are dropped, whitespace runs become one space, and "
            "bare values like 2016 stay strings)"
        )


def test_stage1_order_and_empty(nb):
    text = "@misc{b, note={2}}\n@misc{a, note={1}}\n@misc{c}"
    assert [e["key"] for e in _parse(nb, text)] == ["b", "a", "c"], "entries come back in file order"
    assert _parse(nb, text)[2] == {"type": "misc", "key": "c"}, "an entry may have no fields"
    assert _parse(nb, "no entries here, just prose.") == [], "text without '@' has no entries"
    assert _parse(nb, '@string{acl = "ACL"} @preamble{"x"} @misc{k, title="T"}') == [
        {"type": "misc", "key": "k", "title": "T"}], "@string and @preamble entries are skipped, like @comment"


def test_stage1_values_with_commas_quotes_and_nesting(nb):
    text = '@book{k1, title = "A {B, C} D", editor={X and {Y, Z}},\n note = {line one\n\n   line two} }'
    (e,) = _parse(nb, text)
    assert e["title"] == "A B, C D", f"a quoted value can contain braces and commas, got {e.get('title')!r}"
    assert e["editor"] == "X and Y, Z", f"nested braces protect commas, got {e.get('editor')!r}"
    assert e["note"] == "line one line two", f"newlines and runs of spaces collapse to one space, got {e.get('note')!r}"


def test_stage1_refuses_malformed_input(nb):
    bad = {
        "unbalanced braces": "@article{a, title = {Deep residual",
        "duplicate key": "@misc{a, title={x}}\n@misc{a, title={y}}",
        "missing key": "@article{, title={x}}",
        "field without '='": "@article{a, title {x}}",
        "no opening brace": "@article a, title={x}",
        "unterminated quote": '@article{a, title = "Deep}',
        "duplicate field": "@article{a, title={x}, year=2016, Title={y}}",
        "a stray '@' in prose": "email me at someone@example.org\n@misc{k, title={T}}",
    }
    for what, text in bad.items():
        with pytest.raises(ValueError):
            _parse(nb, text)
            pytest.fail(f"parse_bibtex should raise ValueError for {what}: {text!r}")


# ---------------------------------------------------------------- stage 2: check_note


def _good_note():
    return {
        "key": "He2016", "title": "Deep residual learning for image recognition", "year": 2016,
        "category": "method", "pass": 3,
        "summary": "deeper plain networks train worse; identity shortcuts fix it and let 152 layers train.",
        "claims": [
            {"claim": "plain nets degrade with depth", "evidence": "Figure 4 left: 34-layer above 18-layer"},
            {"claim": "residual nets don't", "evidence": "table 2 and sec. 4.1"},
        ],
        "method": "residual blocks y = F(x) + x with projection shortcuts when shapes change",
        "assumptions": ["the degradation is an optimization problem, not overfitting"],
        "limitations": ["no theory for why shortcuts help optimization"],
        "questions": ["does the effect survive without batch norm?"],
        "cites": ["Ioffe2015", "Simonyan2015"],
    }


def _fields(problems):
    assert isinstance(problems, list) and all(isinstance(p, str) for p in problems), (
        f"check_note returns a list of strings, got {problems!r}"
    )
    return {p.split(":")[0] for p in problems}


def test_stage2_good_note_passes(nb):
    note = _good_note()
    snapshot = repr(note)
    got = nb.check_note(note, known_keys=["Ioffe2015", "Simonyan2015", "He2016"])
    assert got == [], f"a complete pass-3 note has no problems, got {got}"
    assert repr(note) == snapshot, "check_note must not modify the note"


def test_stage2_one_problem_per_broken_field(nb):
    cases = {
        "title": lambda n: n.update(title="  "),
        "key": lambda n: n.pop("key"),
        "category": lambda n: n.update(category=None),
        "summary": lambda n: n.pop("summary"),
        "year": lambda n: n.update(year="2016"),
        "pass": lambda n: n.update({"pass": 4}),
        "claims": lambda n: n.update(claims=[]),
        "claims[1].claim": lambda n: n["claims"][1].update(claim=""),
        "claims[0].evidence": lambda n: n["claims"][0].update(evidence="the results clearly show it"),
        "method": lambda n: n.update(method=""),
        "assumptions": lambda n: n.update(assumptions=[]),
        "limitations": lambda n: n.update(limitations=[""]),
        "questions": lambda n: n.pop("questions"),
    }
    for field, breaker in cases.items():
        note = _good_note()
        breaker(note)
        got = _fields(nb.check_note(note))
        assert got == {field}, (
            f"breaking only {field!r} should give exactly one problem, prefixed '{field}: ...', got {sorted(got)}"
        )


def test_stage2_year_and_pass_types(nb):
    for year in (1850, 2200, True, 2016.0):
        note = _good_note()
        note["year"] = year
        assert _fields(nb.check_note(note)) == {"year"}, f"year={year!r} must be flagged (an int in 1900–2100)"
    note = _good_note()
    note["pass"] = True
    assert "pass" in _fields(nb.check_note(note)), "pass=True is not a pass number"
    note = _good_note()
    note["pass"] = 7
    for f in ("method", "assumptions", "limitations", "questions"):
        note.pop(f)
    note["claims"][0]["evidence"] = "somewhere"
    assert _fields(nb.check_note(note)) == {"pass"}, (
        "with an invalid pass, report it and check the rest as a pass-1 note (no evidence, method or pass-3 fields)"
    )


def test_stage2_requirements_grow_with_the_pass(nb):
    note = _good_note()
    for f in ("assumptions", "limitations", "questions"):
        note.pop(f)
    note["pass"] = 2
    assert nb.check_note(note) == [], "a pass-2 note doesn't need assumptions, limitations or questions"
    note.pop("method")
    note["claims"][0]["evidence"] = "trust me"
    assert _fields(nb.check_note(note)) == {"method", "claims[0].evidence"}, (
        "a pass-2 note needs a method and a location for every claim's evidence"
    )
    note["pass"] = 1
    note["claims"][1].pop("evidence")
    assert nb.check_note(note) == [], (
        "a pass-1 note (five minutes, abstract and figures) needs claims but no evidence locations or method"
    )


def test_stage2_evidence_locations(nb):
    for ev in ("fig. 3", "Figure 2(b)", "Table A1", "sec 4.1", "eq. (5)", "appendix B", "§3.2", "Thm 2",
               "see algorithm 1"):
        note = _good_note()
        note["claims"][0]["evidence"] = ev
        assert nb.check_note(note) == [], f"{ev!r} points to a location in the paper; it should pass"
    for ev in ("the results", "tables everywhere", "in the appendix", "Figure", None):
        note = _good_note()
        note["claims"][0]["evidence"] = ev
        assert _fields(nb.check_note(note)) == {"claims[0].evidence"}, f"{ev!r} doesn't say where; flag it"


def test_stage2_summary_length_and_cites(nb):
    note = _good_note()
    note["summary"] = " ".join(["word"] * 121)
    assert _fields(nb.check_note(note)) == {"summary"}, "a 121-word summary is over the default 120"
    assert nb.check_note(note, max_summary_words=150) == [], "max_summary_words sets the limit"
    note["summary"] = " ".join(["word"] * 120)
    assert nb.check_note(note) == [], "exactly 120 words is allowed (at most max_summary_words)"
    note.pop("cites")
    snapshot = repr(note)
    assert nb.check_note(note) == [] and repr(note) == snapshot, (
        "cites is optional, and check_note must not add it (or anything else) to the note"
    )
    note = _good_note()
    got = nb.check_note(note, known_keys=["Ioffe2015"])
    assert _fields(got) == {"cites"} and any("Simonyan2015" in p for p in got), (
        f"a cited key that isn't in known_keys should be flagged by name, got {got}"
    )
    assert nb.check_note(note) == [], "without known_keys, cites aren't checked"
    note["cites"] = "Ioffe2015"
    assert _fields(nb.check_note(note)) == {"cites"}, "cites must be a list"


# ---------------------------------------------------------------- stage 3: citation_graph, reading_order


def _order(nb, graph):
    return bounded(nb.reading_order, graph, what="reading_order",
                   hint="Kahn's algorithm stops when no paper is ready; a cycle means some never become ready.")


def test_stage3_citation_graph(nb):
    notes = [
        {"key": "resnet", "cites": ["vgg", "batchnorm", "vgg", "resnet", "imagenet-dataset"]},
        {"key": "vgg", "cites": ["alexnet"]},
        {"key": "batchnorm"},
        {"key": "alexnet", "cites": []},
    ]
    got = nb.citation_graph(notes)
    assert got == {"resnet": ["batchnorm", "vgg"], "vgg": ["alexnet"], "batchnorm": [], "alexnet": []}, (
        f"keep only citations inside the reading set, without duplicates or self-citations, sorted; got {got}"
    )
    assert list(got) == ["resnet", "vgg", "batchnorm", "alexnet"], "keys follow the notes' order"
    with pytest.raises(ValueError):
        nb.citation_graph([{"key": "a"}, {"key": "a"}])


def test_stage3_worked_example(nb):
    graph = {"resnet": ["batchnorm", "vgg"], "batchnorm": [], "vgg": ["alexnet"], "alexnet": [],
             "densenet": ["resnet", "batchnorm"]}
    got = _order(nb, graph)
    assert got == ["alexnet", "batchnorm", "vgg", "resnet", "densenet"], (
        f"expected alexnet, batchnorm, vgg, resnet, densenet (every paper after what it cites, ties "
        f"alphabetical), got {got}"
    )


def test_stage3_random_dags_give_the_smallest_valid_order(nb):
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(5, 40))
        names = [f"p{i:02d}" for i in rng.permutation(n)]  # names unrelated to the topological order
        graph = {names[i]: sorted({names[j] for j in rng.choice(i, size=min(i, int(rng.integers(0, 4))),
                                                                 replace=False)}) if i else []
                 for i in range(n)}
        got = _order(nb, graph)
        assert sorted(got) == sorted(graph), f"the order must contain every key exactly once (trial {trial})"
        pos = {k: i for i, k in enumerate(got)}
        placed = set()
        for k in got:
            ready = sorted(c for c in graph if c not in placed and all(r in placed for r in graph[c]))
            assert ready and k == ready[0], (
                f"at each step, take the alphabetically first paper whose references are all read: expected "
                f"{ready[:1]}, got {k!r} (trial {trial})"
            )
            placed.add(k)
        assert all(pos[r] < pos[k] for k in graph for r in graph[k])


def test_stage3_duplicates_cycles_and_unknown_keys(nb):
    assert _order(nb, {"a": ["b", "b"], "b": []}) == ["b", "a"], (
        "a duplicated reference counts once: 'a' is ready as soon as 'b' is read"
    )
    with pytest.raises(ValueError) as err:
        _order(nb, {"x": ["y"], "y": ["z"], "z": ["x"], "w": []})
    assert all(k in str(err.value) for k in "xyz"), (
        f"the cycle's error message should name the papers caught in it, got {err.value}"
    )
    with pytest.raises(ValueError) as err:
        _order(nb, {"a": ["ghost"]})
    assert "ghost" in str(err.value), f"name the unknown key in the error, got {err.value}"


# ---------------------------------------------------------------- stage 4: pagerank, next_to_read


def _google(graph, d):
    keys = list(graph)
    n = len(keys)
    idx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((n, n))
    for k, refs in graph.items():
        refs = set(refs)
        if refs:
            for r in refs:
                M[idx[r], idx[k]] = 1 / len(refs)
        else:
            M[:, idx[k]] = 1 / n
    return d * M + (1 - d) / n, keys


def _stationary(graph, d):
    G, keys = _google(graph, d)
    w, v = np.linalg.eig(G)
    r = np.real(v[:, np.argmax(np.real(w))])
    return dict(zip(keys, r / r.sum()))


def _random_graph(seed, n=25):
    rng = np.random.default_rng(seed)
    keys = [f"k{i}" for i in range(n)]
    return {k: sorted({keys[j] for j in rng.choice(n, size=int(rng.integers(0, 5))) if j != i})
            for i, k in enumerate(keys)}


def test_stage4_matches_the_google_matrix(nb):
    for seed, d in [(0, 0.85), (1, 0.85), (2, 0.5), (3, 0.95)]:
        g = _random_graph(seed)
        got = _pr(nb, g, damping=d)
        want = _stationary(g, d)
        assert list(got) == list(g), "scores come back in the graph's key order"
        assert np.isclose(sum(got.values()), 1.0), f"scores must sum to 1, got {sum(got.values())}"
        worst = max(abs(got[k] - want[k]) for k in g)
        assert worst < 1e-8, (
            f"pagerank (damping={d}) differs from the stationary vector of the Google matrix by {worst:.2e}. "
            "check that a paper's score is split evenly over the papers it cites, and that papers citing "
            "nothing in the set spread their score over every paper."
        )


def test_stage4_hand_example_and_damping(nb):
    got = _pr(nb, {"a": ["b"], "b": []}, damping=0.5)
    # r_a = 0.25 + 0.25 r_b and r_a + r_b = 1  ->  r_a = 0.4, r_b = 0.6
    assert np.isclose(got["a"], 0.4) and np.isclose(got["b"], 0.6), (
        f"a cites b, b cites nothing, damping 0.5: r_a = 0.4, r_b = 0.6. got {got}. if they don't sum to 1, the "
        "dangling paper's score leaked out and you never renormalized; if they sum to 1 but differ, check how "
        "the jump term and the citations are weighted."
    )
    flat = _pr(nb, {"a": ["b"], "b": ["c"], "c": []}, damping=0.0)
    assert np.allclose(list(flat.values()), 1 / 3), "damping=0 ignores citations: every paper gets 1/n"
    assert np.allclose(list(_pr(nb, {"a": [], "b": [], "c": [], "d": []}).values()), 0.25)
    dup = _pr(nb, {"a": ["b", "b", "c"], "b": ["c"], "c": [], "d": ["a", "a"]})
    clean = _pr(nb, {"a": ["b", "c"], "b": ["c"], "c": [], "d": ["a"]})
    assert all(np.isclose(dup[k], clean[k]) for k in clean), (
        f"a duplicated citation counts once (in the out-degree too): got {dup}, expected {clean}"
    )


def test_stage4_per_step_renormalization_changes_the_ranking(nb):
    g = {"a": ["e"], "b": ["a", "c"], "c": ["d"], "d": ["c", "e"], "e": []}
    got = _pr(nb, g)
    want = _stationary(g, 0.85)
    assert got["e"] > got["d"] and np.isclose(got["e"], want["e"]) and np.isclose(got["d"], want["d"]), (
        f"here PageRank ranks e ({want['e']:.4f}) above d ({want['d']:.4f}); got e {got['e']:.4f}, d {got['d']:.4f}. "
        "dropping the dangling term and renormalizing at the END is fine (it gives PageRank exactly), but "
        "renormalizing after EVERY step converges to a different vector, which ranks d above e."
    )


def test_stage4_refusals(nb):
    for d in (1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            _pr(nb, {"a": ["b"], "b": []}, damping=d)
    with pytest.raises(ValueError):
        _pr(nb, {"a": ["ghost"], "b": []})  # a cited key that isn't in the graph


def test_stage4_iteration_controls(nb):
    g = _random_graph(4)
    G, keys = _google(g, 0.85)
    one = G @ np.full(len(keys), 1 / len(keys))
    got = _pr(nb, g, max_iters=1)
    assert np.allclose([got[k] for k in keys], one / one.sum()), (
        "start from the uniform vector; with max_iters=1 you return the result of exactly one update"
    )
    loose = _pr(nb, g, tol=10.0)
    assert np.allclose([loose[k] for k in keys], one / one.sum()), (
        "with a huge tol the first update already counts as converged: stop after it"
    )
    assert _pr(nb, {}) == {}


def test_stage4_next_to_read(nb):
    g = {"survey": ["resnet", "vgg", "alexnet"], "resnet": ["vgg", "alexnet"], "vgg": ["alexnet"], "alexnet": [],
         "densenet": ["resnet"], "zoo": []}
    pr = _pr(nb, g)
    got = _ntr(nb, g, read={"alexnet"}, k=3)
    unread = sorted((k for k in g if k != "alexnet"), key=lambda k: (-pr[k], k))
    assert got == unread[:3], f"expected the 3 highest-PageRank unread papers {unread[:3]}, got {got}"
    assert "alexnet" not in _ntr(nb, g, ["alexnet"], k=10) and len(_ntr(nb, g, ["alexnet"], k=10)) == 5
    assert _ntr(nb, g, [], k=2, damping=0.0) == ["alexnet", "densenet"], (
        "with damping=0 every score ties, so the order is alphabetical; pass damping through to pagerank"
    )
    with pytest.raises(ValueError):
        _ntr(nb, g, ["AlexNet"])
