"""Checks for exercise.py (exact inference).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Factors are built
with the notebook's own `Factor` (topic 02). The oracle is the full joint table built with
np.einsum on small random models: marginals and conditionals are then a sum and a division.
Costs are checked on graphs whose answers you can count by hand (stars, chains, grids).
"""
import string
import tracemalloc

import numpy as np

from mlfs.testing import bounded


def _bnd(fn, what):
    """fn under a line budget: a loop that never ends fails with a teaching message instead of hanging."""
    return lambda *a, **k: bounded(fn, *a, what=what, budget=1_000_000,
                                   hint="every loop must remove what it processed (a variable, a factor) or stop.", **k)


def _model(nb, scopes, cards, rng, low=0.1, with_joint=True):
    """random factors over the given scopes, plus the einsum joint over the sorted variable list (if asked)."""
    names = sorted({v for s in scopes for v in s}, key=str)
    letter = {v: string.ascii_letters[i % 52] for i, v in enumerate(names)}  # only used for the joint (<= 52 vars)
    factors, terms, arrays = [], [], []
    for s in scopes:
        vals = rng.uniform(low, 1.0, size=[cards[v] for v in s])
        factors.append(nb.Factor(s, [cards[v] for v in s], vals))
        terms.append("".join(letter[v] for v in s))
        arrays.append(vals)
    if not with_joint:
        return factors, names, None
    joint = np.einsum(",".join(terms) + "->" + "".join(letter[v] for v in names), *arrays)
    return factors, names, joint


def _oracle(names, joint, query, evidence):
    idx = tuple(evidence.get(v, slice(None)) for v in names)
    rest = [v for v in names if v not in evidence]
    letters = string.ascii_letters[:len(rest)]
    t = np.einsum(f"{letters}->" + "".join(letters[rest.index(q)] for q in query), joint[idx])
    return t / t.sum()


def _chain(nb, n, rng, card=2):
    scopes = [(f"x{i}",) for i in range(1)] + [(f"x{i}", f"x{i + 1}") for i in range(n - 1)]
    return _model(nb, scopes, {f"x{i}": card for i in range(n)}, rng, with_joint=False)


def _star(nb, k):
    rng = np.random.default_rng(0)
    scopes = [("h", f"l{i}") for i in range(k)]
    return _model(nb, scopes, {"h": 2, **{f"l{i}": 2 for i in range(k)}}, rng)[0]


# ---------------------------------------------------------------- stage 1: eliminate one variable


def test_stage1_eliminate_hand(nb):
    A = nb.Factor(("a", "b"), (2, 2), [[1.0, 2.0], [3.0, 4.0]])
    B = nb.Factor(("b", "c"), (2, 2), [[1.0, 0.0], [1.0, 1.0]])
    C = nb.Factor(("c",), (2,), [5.0, 7.0])
    out = _bnd(nb.eliminate, "eliminate")([A, B, C], "b")
    assert len(out) == 2, f"eliminating b combines the two factors that mention it into one: 3 -> 2 factors; got {len(out)}"
    assert out[0].variables == ("c",), "the factors that don't mention b come first, unchanged"
    new = out[1]
    assert set(new.variables) == {"a", "c"}, f"the new factor is over a and c (b summed out); got {new.variables}"
    vals = np.transpose(new.values, [new.variables.index("a"), new.variables.index("c")])
    want = np.array([[1 + 2, 2], [3 + 4, 4]], dtype=float)  # sum_b A(a, b) B(b, c)
    assert np.allclose(vals, want), f"sum_b A(a,b) B(b,c) should be {want.tolist()}; got {vals.tolist()}"
    same = _bnd(nb.eliminate, "eliminate")([A, C], "z")
    assert len(same) == 2, "eliminating a variable no factor mentions changes nothing"


def test_stage1_eliminate_preserves_the_sum(nb):
    rng = np.random.default_rng(1)
    factors, names, joint = _model(nb, [("p", "q"), ("q", "r", "s"), ("s",), ("p", "s")],
                                   {"p": 2, "q": 3, "r": 2, "s": 3}, rng)
    out = factors
    for v in ["q", "r"]:
        out = _bnd(nb.eliminate, "eliminate")(out, v)
        assert all(v not in f.variables for f in out), f"after eliminating {v}, no factor may mention it"
    prod = out[0]
    for f in out[1:]:
        prod = prod * f
    vals = np.transpose(prod.values, [prod.variables.index("p"), prod.variables.index("s")])
    assert np.allclose(vals, joint.sum(axis=(1, 2))), (
        "eliminating q and r must leave factors whose product is the joint with q and r summed out")


# ---------------------------------------------------------------- stage 2: orders and costs


def test_stage2_star_costs(nb):
    fs = _star(nb, 5)
    leaves = [f"l{i}" for i in range(5)]
    hub_first = _bnd(nb.elimination_cost, "elimination_cost")(fs, ["h"] + leaves)
    leaves_first = _bnd(nb.elimination_cost, "elimination_cost")(fs, leaves + ["h"])
    assert hub_first == 2**6, (
        f"star with hub h and 5 binary leaves: eliminating the hub first multiplies all five factors, a table over "
        f"h and every leaf: 2^6 = 64 entries. got {hub_first}")
    assert leaves_first == 4, (
        f"eliminating the leaves first only ever multiplies a 2 x 2 factor (and then tables over h alone): cost 4. "
        f"got {leaves_first}")


def test_stage2_min_degree_order(nb):
    fs = _star(nb, 5)
    order = _bnd(nb.min_degree_order, "min_degree_order")(fs, ["h"] + [f"l{i}" for i in range(5)])
    assert sorted(order, key=str) == sorted(["h"] + [f"l{i}" for i in range(5)], key=str), "every variable exactly once"
    assert order[0] == "l0" and order.index("h") >= 4, (
        f"in a star the leaves have degree 1 and the hub degree 5: min-degree eliminates leaves first (ties go to "
        f"the earliest variable in the list, l0). once 4 leaves are gone, h and the last leaf tie at degree 1. got {order}")
    chain = _chain(nb, 4, np.random.default_rng(2))[0]
    got = _bnd(nb.min_degree_order, "min_degree_order")(chain, ["x1", "x2", "x0", "x3"])
    assert got == ["x0", "x1", "x2", "x3"], (
        f"chain x0-x1-x2-x3 with the variables listed as [x1, x2, x0, x3]: x0 and x3 have degree 1, and x0 comes "
        f"first in the list, so it goes first; then x1 has degree 1 and is earliest; and so on. got {got}")
    rng = np.random.default_rng(11)
    pairs = [("a", "e"), ("a", "f"), ("c", "d"), ("c", "f"), ("d", "e"), ("d", "f")]
    fs = [nb.Factor(("b",), (2,), [0.5, 0.5])] + [nb.Factor(p, (2, 2), rng.uniform(0.1, 1, (2, 2))) for p in pairs]
    got = _bnd(nb.min_degree_order, "min_degree_order")(fs, list("abcdef"))
    assert got[:3] == ["b", "a", "c"], (
        f"edges a-e, a-f, c-d, c-f, d-e, d-f (and b alone): b goes first (degree 0), then a (degree 2, earliest). "
        f"eliminating a connects its neighbours e and f (fill-in), so e now has degree 2 and c (also 2, earlier) goes "
        f"next. without the fill-in edge e would look like degree 1. got {got}")


def test_stage2_grid_fill_in(nb):
    rng = np.random.default_rng(3)
    n = 4
    name = lambda r, c: f"g{r}{c}"
    scopes = [(name(r, c), name(r, c + 1)) for r in range(n) for c in range(n - 1)]
    scopes += [(name(r, c), name(r + 1, c)) for r in range(n - 1) for c in range(n)]
    fs, names, _ = _model(nb, scopes, {name(r, c): 2 for r in range(n) for c in range(n)}, rng)
    order = _bnd(nb.min_degree_order, "min_degree_order")(fs, names)
    cost = _bnd(nb.elimination_cost, "elimination_cost")(fs, order)
    row_major = [name(r, c) for r in range(n) for c in range(n)]
    assert _bnd(nb.elimination_cost, "elimination_cost")(fs, row_major) == 2**5, (
        "4x4 grid eliminated row by row: the frontier is one row (4 variables) plus the variable being eliminated: "
        f"2^5 = 32 entries. got {_bnd(nb.elimination_cost, "elimination_cost")(fs, row_major)}. remember neighbours become connected (fill-in)")
    assert cost <= 2**5, f"min-degree should do at least as well as row-by-row on a 4x4 grid; got cost {cost}"


# ---------------------------------------------------------------- stage 3: variable elimination


def _abc(nb):
    """the lesson's worked example: A -> B -> C, all binary."""
    return [nb.Factor(("A",), (2,), [0.6, 0.4]),
            nb.Factor(("A", "B"), (2, 2), [[0.7, 0.3], [0.2, 0.8]]),
            nb.Factor(("B", "C"), (2, 2), [[0.9, 0.1], [0.4, 0.6]])]


def test_stage3_worked_example(nb):
    pc = _bnd(nb.variable_elimination, "variable_elimination")(_abc(nb), "C")
    assert np.allclose(pc.values, [0.65, 0.35]), (
        f"lesson worked example: eliminating A gives tau(B) = (0.5, 0.5), then P(C) = (0.65, 0.35); got "
        f"{np.round(pc.values, 4).tolist()}")
    pa = _bnd(nb.variable_elimination, "variable_elimination")(_abc(nb), "A", {"C": 1})
    assert np.allclose(pa.values, [3 / 7, 4 / 7]), (
        f"P(A | C = 1) is proportional to (0.6 * 0.25, 0.4 * 0.5) = (0.15, 0.2) -> (0.4286, 0.5714); got "
        f"{np.round(pa.values, 4).tolist()}. did you reduce every factor by the evidence first, and normalize last?")


def test_stage3_matches_joint(nb):
    rng = np.random.default_rng(4)
    scopes = [("a", "b"), ("b", "c", "d"), ("d", "e"), ("a", "e"), ("c",), ("e", "f")]
    cards = {"a": 2, "b": 3, "c": 2, "d": 2, "e": 3, "f": 2}
    fs, names, joint = _model(nb, scopes, cards, rng)
    for query, ev in [(("a",), {}), (("f",), {"b": 2}), (("c", "a"), {"e": 0}), (("d",), {"a": 1, "f": 0})]:
        got = _bnd(nb.variable_elimination, "variable_elimination")(fs, query if len(query) > 1 else query[0], ev)
        assert tuple(got.variables) == query, f"the result is a Factor over the query {query}; got {got.variables}"
        want = _oracle(names, joint, list(query), ev)
        assert np.allclose(got.values, want, atol=1e-10), (
            f"P({query} | {ev}) disagrees with the brute-force joint: got {np.round(got.values, 4).tolist()}, "
            f"want {np.round(want, 4).tolist()}")


def test_stage3_any_order_same_answer(nb):
    rng = np.random.default_rng(5)
    fs, names, joint = _model(nb, [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("b", "d")],
                              {v: 2 for v in "abcd"}, rng)
    ref = _bnd(nb.variable_elimination, "variable_elimination")(fs, "a", {"c": 1})
    for order in [["b", "d"], ["d", "b"]]:
        got = _bnd(nb.variable_elimination, "variable_elimination")(fs, "a", {"c": 1}, order=order)
        assert np.allclose(got.values, ref.values), f"order {order} must give the same answer as min-degree"
    try:
        _bnd(nb.variable_elimination, "variable_elimination")(fs, "a", order=["b"])
    except ValueError:
        pass
    else:
        raise AssertionError("an order that doesn't cover every hidden variable (here: c, d) should raise ValueError")


def test_stage3_never_builds_the_joint(nb):
    rng = np.random.default_rng(6)
    fs, _, _ = _model(nb, [("x0",)] + [(f"x{i}", f"x{i + 1}") for i in range(23)], {f"x{i}": 2 for i in range(24)}, rng,
                      with_joint=False)
    tracemalloc.start()
    try:
        got = bounded(nb.variable_elimination, fs, "x23", what="variable_elimination")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 20e6, (
        f"a chain of 24 binary variables: the joint has 2^24 entries (134 MB). variable elimination with a good "
        f"order only ever builds 2 x 2 tables, but yours peaked at {peak / 1e6:.0f} MB")
    msgs = np.array([1.0])
    for f in fs:  # forward pass by hand: the marginal of the last variable
        if len(f.variables) == 1:
            msgs = f.values
        else:
            msgs = msgs @ f.values
    assert np.allclose(got.values, msgs / msgs.sum()), "and it must still be right"


# ---------------------------------------------------------------- stage 4: belief propagation on trees


def _random_tree(nb, n, rng, card=3):
    scopes = [(f"t{i}",) for i in range(n) if rng.random() < 0.5]
    for i in range(1, n):
        scopes.append((f"t{int(rng.integers(i))}", f"t{i}"))
    if rng.random() < 0.7 and n >= 4:  # one triple factor joining a node to two fresh-ish leaves
        scopes.append((f"t{n - 1}", f"t{n}", f"t{n + 1}"))
        n += 2
    cards = {f"t{i}": card for i in range(n)}
    return _model(nb, scopes, cards, rng)


def test_stage4_worked_example(nb):
    got = bounded(nb.sum_product, _abc(nb), {"C": 1}, what="sum_product")
    assert set(got) == {"A", "B"}, f"with C observed, the result has marginals for A and B only; got {sorted(got)}"
    assert np.allclose(got["A"], [3 / 7, 4 / 7]), (
        f"lesson worked example: the message into A from below is (0.25, 0.5), so P(A | C = 1) = (0.4286, 0.5714); "
        f"got {np.round(got['A'], 4).tolist()}")
    assert np.allclose(got["B"], np.array([0.5 * 0.1, 0.5 * 0.6]) / 0.35), (
        "P(B | C = 1) is proportional to tau(B) * P(C = 1 | B) = (0.5 * 0.1, 0.5 * 0.6)")


def test_stage4_tree_marginals(nb):
    rng = np.random.default_rng(7)
    for trial in range(4):
        fs, names, joint = _random_tree(nb, 7, rng)
        got = bounded(nb.sum_product, fs, what="sum_product")
        assert set(got) == set(names), f"one marginal per variable: expected {names}, got {sorted(got, key=str)}"
        for v in names:
            want = _oracle(names, joint, [v], {})
            assert np.allclose(got[v], want, atol=1e-10), (
                f"tree {trial}: marginal of {v} is {np.round(got[v], 4).tolist()}, the joint says "
                f"{np.round(want, 4).tolist()}")


def test_stage4_evidence(nb):
    rng = np.random.default_rng(8)
    fs, names, joint = _random_tree(nb, 8, rng)
    ev = {names[1]: 2, names[5]: 0}
    got = bounded(nb.sum_product, fs, ev, what="sum_product")
    assert set(got) == set(names) - set(ev), "observed variables are left out of the result"
    for v in got:
        want = _oracle(names, joint, [v], ev)
        assert np.allclose(got[v], want, atol=1e-10), f"P({v} | evidence) disagrees with the joint"


def test_stage4_long_chain(nb):
    rng = np.random.default_rng(9)
    fs, names, _ = _chain(nb, 40, rng, card=2)
    got = bounded(nb.sum_product, fs, {"x20": 1}, what="sum_product")
    n = 40
    e = [np.ones(2) for _ in range(n)]
    e[20] = np.array([0.0, 1.0])
    pair = [f.values for f in fs[1:]]
    a = [fs[0].values * e[0]]
    for i in range(n - 1):  # forward messages, evidence folded in
        a.append((a[-1] @ pair[i]) * e[i + 1])
    b = [np.ones(2) for _ in range(n)]
    for i in range(n - 2, -1, -1):
        b[i] = pair[i] @ (e[i + 1] * b[i + 1])
    for v in ["x0", "x19", "x21", "x39"]:
        i = int(v[1:])
        want = a[i] * b[i] / np.sum(a[i] * b[i])
        assert np.allclose(got[v], want, atol=1e-10), (
            f"chain of 40 with x20 observed: the marginal of {v} should be {np.round(want, 4).tolist()} (forward times "
            f"backward messages); got {np.round(got[v], 4).tolist()}")


def test_stage4_very_long_chain(nb):
    n = 600
    fs, _, _ = _chain(nb, n, np.random.default_rng(12), card=2)
    try:
        got = bounded(nb.sum_product, fs, {"x599": 0}, what="sum_product", budget=3_000_000)  # evidence at the end keeps it one chain
    except RecursionError:
        raise AssertionError(
            "a chain of 600 variables hit Python's recursion limit: messages computed by recursion go about as deep as "
            "the tree is long. schedule them instead: a breadth-first order from a root, messages from the leaves to "
            "the root in reverse order, then back out") from None
    e = [np.ones(2) for _ in range(n)]
    e[599] = np.array([1.0, 0.0])
    pair = [f.values for f in fs[1:]]
    a = [fs[0].values * e[0] / np.sum(fs[0].values * e[0])]
    for i in range(n - 1):
        m = (a[-1] @ pair[i]) * e[i + 1]
        a.append(m / m.sum())
    b = [np.ones(2) for _ in range(n)]
    for i in range(n - 2, -1, -1):
        m = pair[i] @ (e[i + 1] * b[i + 1])
        b[i] = m / m.sum()
    for i in (0, 300, 598):
        want = a[i] * b[i] / np.sum(a[i] * b[i])
        assert np.allclose(got[f"x{i}"], want, atol=1e-10), f"chain of 600: marginal of x{i} is wrong"


def test_stage4_rejects_cycles(nb):
    rng = np.random.default_rng(10)
    fs, _, _ = _model(nb, [("a", "b"), ("b", "c"), ("c", "a")], {v: 2 for v in "abc"}, rng)
    try:
        bounded(nb.sum_product, fs, what="sum_product", budget=200_000)
    except ValueError:
        return
    raise AssertionError(
        "a triangle of pairwise factors has a cycle; sum_product is only exact on trees, so it must raise ValueError "
        "(check the factor graph first: #edges == #nodes - #components)")
