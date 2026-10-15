"""Checks for exercise.py (graphical models).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Factor operations
are checked against np.einsum on random tables; brute-force queries against the lesson's
sprinkler example; d-separation against hand cases and against NUMERICAL conditional
independence in random Bayes nets with random tables (for generic tables, independence holds
exactly when the graph says so).
"""
import itertools
import string

import numpy as np

from mlfs.testing import bounded


def _sep(nb, *args):
    return bounded(nb.separated, *args, what="separated", budget=200_000,
                   hint="a breadth-first search needs a visited set, or it walks back and forth forever.")


def _dsep(nb, *args):
    return bounded(nb.d_separated, *args, what="d_separated", budget=200_000,
                   hint="every search (ancestors, separation) needs a visited set.")


def _rand_factor(nb, variables, cards, rng):
    return nb.Factor(variables, cards, rng.uniform(0.1, 1.0, size=cards))


def _sprinkler(nb):
    """cloudy -> sprinkler, cloudy -> rain, (sprinkler, rain) -> wet; every variable binary (0 = no)."""
    C = nb.Factor(("C",), (2,), [0.5, 0.5])
    S = nb.Factor(("C", "S"), (2, 2), [[0.5, 0.5], [0.9, 0.1]])
    R = nb.Factor(("C", "R"), (2, 2), [[0.8, 0.2], [0.2, 0.8]])
    W = nb.Factor(("S", "R", "W"), (2, 2, 2), [[[1.0, 0.0], [0.1, 0.9]], [[0.1, 0.9], [0.01, 0.99]]])
    return [C, S, R, W]


# ---------------------------------------------------------------- stage 1: factors and products


def test_stage1_construct(nb):
    f = nb.Factor(["a", "b"], [2, 3], np.arange(6.0))
    assert f.variables == ("a", "b") and f.cardinalities == (2, 3), (
        f"variables and cardinalities are stored as tuples; got {f.variables!r}, {f.cardinalities!r}")
    assert f.values.shape == (2, 3) and f.values[1, 2] == 5.0, (
        "values is a float array with one axis per variable (a flat array of the right size is reshaped)")
    for bad in [(("a", "a"), (2, 2), np.ones(4)), (("a", "b"), (2, 2), np.ones(5))]:
        try:
            nb.Factor(*bad)
        except ValueError:
            continue
        raise AssertionError(f"Factor{bad[:2]} with {bad[2].size} values should raise ValueError")


def test_stage1_product_hand(nb):
    A = nb.Factor(("a",), (2,), [1.0, 2.0])
    B = nb.Factor(("b", "a"), (3, 2), [[1, 10], [2, 20], [3, 30]])
    P = A * B
    assert P.variables == ("a", "b") and P.cardinalities == (2, 3), (
        f"the product's variables are self's, then other's new ones: ('a', 'b'); got {P.variables}")
    want = np.array([[1, 2, 3], [20, 40, 60]], dtype=float)
    assert np.allclose(P.values, want), (
        f"(A * B)(a, b) = A(a) B(b, a): expected {want.tolist()}, got {np.asarray(P.values).tolist()}. "
        "did you align B's axes to the product's variable order before multiplying?")


def test_stage1_product_matches_einsum(nb):
    rng = np.random.default_rng(0)
    for trial in range(5):
        names = list("pqrst")
        cards = dict(zip(names, rng.integers(2, 4, size=5)))
        v1 = list(rng.choice(names, size=rng.integers(1, 4), replace=False))
        v2 = list(rng.choice(names, size=rng.integers(1, 4), replace=False))
        f1 = _rand_factor(nb, v1, [cards[v] for v in v1], rng)
        f2 = _rand_factor(nb, v2, [cards[v] for v in v2], rng)
        P = f1 * f2
        union = v1 + [v for v in v2 if v not in v1]
        assert list(P.variables) == union, f"variables of the product should be {union}, got {list(P.variables)}"
        sub = lambda vs: "".join(string.ascii_lowercase[union.index(v)] for v in vs)
        want = np.einsum(f"{sub(v1)},{sub(v2)}->{sub(union)}", f1.values, f2.values)
        assert np.allclose(P.values, want), f"trial {trial}: {v1} x {v2} disagrees with np.einsum"


# ---------------------------------------------------------------- stage 2: marginalize, reduce, normalize, query


def test_stage2_marginalize_reduce_normalize(nb):
    rng = np.random.default_rng(1)
    f = _rand_factor(nb, ("x", "y", "z"), (2, 3, 4), rng)
    before = f.values.copy()
    m = f.marginalize(["y"])
    assert m.variables == ("x", "z") and np.allclose(m.values, f.values.sum(1)), "marginalize(['y']) sums axis y out"
    m2 = f.marginalize("z")
    assert m2.variables == ("x", "y") and np.allclose(m2.values, f.values.sum(2)), "a single name works too"
    m3 = f.marginalize(["x", "y", "z"])
    assert m3.variables == () and np.isclose(float(m3.values), f.values.sum()), "summing out everything leaves a scalar factor"
    r = f.reduce({"y": 2, "w": 0})
    assert r.variables == ("x", "z") and np.allclose(r.values, f.values[:, 2, :]), (
        "reduce({'y': 2}) keeps the slice y = 2 and drops y; evidence on variables the factor lacks is ignored")
    n = f.normalize()
    assert np.isclose(n.values.sum(), 1) and n.variables == f.variables and np.allclose(n.values * f.values.sum(), f.values)
    assert np.array_equal(f.values, before) and f.variables == ("x", "y", "z"), (
        "marginalize, reduce and normalize return NEW factors; they must not modify self")


def test_stage2_sprinkler_queries(nb):
    fs = _sprinkler(nb)
    q = nb.joint_query(fs, "R", {"W": 1})
    assert q.variables == ("R",) and np.isclose(q.values[1], 0.7079, atol=1e-4), (
        f"P(rain | wet grass) = 0.7079 in the lesson's sprinkler network; got {np.round(q.values, 4).tolist()}")
    q2 = nb.joint_query(fs, "S", {"W": 1})
    assert np.isclose(q2.values[1], 0.4298, atol=1e-4), f"P(sprinkler | wet) = 0.4298; got {q2.values[1]:.4f}"
    q3 = nb.joint_query(fs, "S", {"W": 1, "R": 1})
    assert np.isclose(q3.values[1], 0.1945, atol=1e-4), (
        f"P(sprinkler | wet, rain) = 0.1945: learning it rained EXPLAINS AWAY the wet grass; got {q3.values[1]:.4f}")
    j = nb.joint_query(fs, ("W", "C"))
    assert j.variables == ("W", "C") and j.values.shape == (2, 2) and np.isclose(j.values.sum(), 1), (
        "a tuple query returns the joint over those variables, in that order")


# ---------------------------------------------------------------- stage 3: undirected graphs


def test_stage3_moralize(nb):
    edges = [("A", "C"), ("B", "C"), ("C", "D"), ("E", "D")]
    got = set(map(frozenset, nb.moralize(edges)))
    want = {frozenset(e) for e in [("A", "C"), ("B", "C"), ("C", "D"), ("E", "D"), ("A", "B"), ("C", "E")]}
    assert got == want, (
        f"moral graph: drop the arrows and connect every pair of parents of a common child (A-B for C, C-E for D). "
        f"missing {sorted(map(sorted, want - got))}, extra {sorted(map(sorted, got - want))}")


def test_stage3_markov_blanket(nb):
    edges = [("A", "C"), ("B", "C"), ("C", "D"), ("E", "D"), ("D", "F"), ("G", "A")]
    assert nb.markov_blanket(edges, "C") == {"A", "B", "D", "E"}, (
        f"the blanket of C is its parents (A, B), children (D) and the children's other parents (E); "
        f"got {nb.markov_blanket(edges, 'C')}")
    assert nb.markov_blanket(edges, "G") == {"A"} and nb.markov_blanket(edges, "F") == {"D"}


def test_stage3_separated(nb):
    und = [frozenset(e) for e in [(1, 2), (2, 3), (3, 4), (1, 5), (5, 4)]]  # a 5-cycle-ish square 1-2-3-4-5-1
    assert not _sep(nb, und, 1, 3), "1 and 3 are connected by a path, so they're not separated"
    assert not _sep(nb, und, 1, 3, [2]), "blocking 2 leaves the path 1-5-4-3"
    assert _sep(nb, und, 1, 3, [2, 4]), "blocking 2 and 4 cuts every path from 1 to 3"
    assert _sep(nb, und, {1}, {6}), "a node that appears in no edge is separated from everything"
    assert not _sep(nb, und, {1, 6}, {3}, [2]), "sets: some x in {1, 6} reaches 3"


# ---------------------------------------------------------------- stage 4: d-separation


def test_stage4_three_node_patterns(nb):
    chain, fork, collider = [("A", "B"), ("B", "C")], [("B", "A"), ("B", "C")], [("A", "B"), ("C", "B")]
    cases = [
        (chain, (), False, "chain A -> B -> C: A and C are dependent"),
        (chain, ("B",), True, "chain: observing B blocks it"),
        (fork, (), False, "fork A <- B -> C: dependent through the common cause"),
        (fork, ("B",), True, "fork: observing the common cause blocks it"),
        (collider, (), True, "collider A -> B <- C: independent causes are independent"),
        (collider, ("B",), False, "collider: observing the common EFFECT B makes them dependent (explaining away)"),
        (collider + [("B", "D")], ("D",), False, "collider: observing a DESCENDANT of B unblocks it too"),
    ]
    for edges, given, want, why in cases:
        got = _dsep(nb, edges, "A", "C", given)
        assert bool(got) == want, f"{why}; d_separated returned {got}"


def test_stage4_ancestors(nb):
    edges = [("A", "C"), ("B", "C"), ("C", "D"), ("E", "D"), ("D", "F")]
    assert bounded(nb.ancestors, edges, {"D"}, what="ancestors") == {"A", "B", "C", "D", "E"}, "ancestors include the nodes themselves"
    assert bounded(nb.ancestors, edges, {"A", "E"}, what="ancestors") == {"A", "E"}


def test_stage4_matches_numerical_independence(nb):
    rng = np.random.default_rng(3)
    n_checked = 0
    for trial in range(6):
        n = 6
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4]
        tables = []
        for j in range(n):
            pa = [i for i, k in edges if k == j]
            t = rng.dirichlet(np.ones(2) * 2.0, size=(2,) * len(pa))  # P(j | parents), shape (2,)*|pa| + (2,)
            tables.append((pa, t))
        letters = string.ascii_lowercase
        terms, ops = [], []
        for j, (pa, t) in enumerate(tables):
            terms.append("".join(letters[i] for i in pa) + letters[j])
            ops.append(t)
        joint = np.einsum(",".join(terms) + "->" + letters[:n], *ops)
        for _ in range(12):
            x, y = rng.choice(n, size=2, replace=False)
            rest = [k for k in range(n) if k not in (x, y)]
            z = [k for k in rest if rng.random() < 0.35]
            other = tuple(k for k in range(n) if k not in (x, y, *z))
            pxyz = joint.sum(axis=other)  # axes in order of the remaining variables
            keep = sorted([x, y, *z])
            pxyz = np.moveaxis(pxyz, [keep.index(x), keep.index(y)], [0, 1])
            pz = pxyz.sum(axis=(0, 1))
            pxz, pyz = pxyz.sum(axis=1), pxyz.sum(axis=0)
            indep = np.max(np.abs(pxyz * pz[None, None] - pxz[:, None] * pyz[None, :])) < 1e-13
            got = _dsep(nb, edges, int(x), int(y), [int(k) for k in z])
            assert bool(got) == indep, (
                f"random Bayes net with edges {edges}: x={x}, y={y}, given={z}. the numbers say "
                f"{'independent' if indep else 'DEPENDENT'}, d_separated says {got}. "
                "remember: colliders block unless they (or a descendant) are observed")
            n_checked += 1
    assert n_checked == 72


# ---------------------------------------------------------------- stage 5: factor graphs


def test_stage5_factor_graph(nb):
    fs = _sprinkler(nb)
    g = nb.factor_graph(fs)
    assert {k: sorted(v) for k, v in g.items()} == {"C": [0, 1, 2], "S": [1, 3], "R": [2, 3], "W": [3]}, (
        f"factor_graph maps each variable to the indices of the factors that mention it; got {g}")


def test_stage5_is_tree(nb):
    rng = np.random.default_rng(4)
    F = lambda *vs: _rand_factor(nb, vs, [2] * len(vs), rng)
    chain = [F("a"), F("a", "b"), F("b", "c"), F("c", "d")]
    assert nb.is_tree(chain), "a chain of pairwise factors is a tree"
    star = [F("h", "x"), F("h", "y"), F("h", "z"), F("h")]
    assert nb.is_tree(star), "a star (hub h) is a tree"
    assert not nb.is_tree(_sprinkler(nb)), (
        "the sprinkler network's factor graph has a cycle (C - P(S|C) - S - P(W|S,R) - R - P(R|C) - C)")
    loop = [F("a", "b"), F("b", "c"), F("c", "a")]
    assert not nb.is_tree(loop), "three pairwise factors around a triangle form a cycle"
    double = [F("a", "b"), F("a", "b")]
    assert not nb.is_tree(double), "two factors over the same pair a, b make a 4-cycle a - f0 - b - f1 - a"
    forest = [F("a", "b"), F("c", "d")]
    assert nb.is_tree(forest), "two disconnected trees (a forest) have no cycle: count it as a tree"
    single = [F("a", "b", "c")]
    assert nb.is_tree(single), "one factor over three variables is a star in the factor graph: a tree"
