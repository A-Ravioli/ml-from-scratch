"""Checks for exercise.py (causal discovery).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The graph
checks use independent oracles written here: d-separation by moralization (so the PC
algorithm can be run with a perfect independence test), and the CPDAG by brute force,
enumerating every DAG on the skeleton and keeping those with the same v-structures
(Verma & Pearl's characterization of Markov equivalence). Data checks use linear-Gaussian
models with strong edges and alpha = 0.001 or smaller.
"""
import itertools

import numpy as np
from scipy.special import erfc

from mlfs.testing import bounded

# ---------------------------------------------------------------- oracles


def _random_dag(p, prob, rng):
    order = rng.permutation(p)
    D = np.zeros((p, p), dtype=int)
    for a, b in itertools.combinations(range(p), 2):
        if rng.random() < prob:
            D[order[a], order[b]] = 1
    return D


def _ancestors(D, nodes):
    out, stack = set(nodes), list(nodes)
    while stack:
        v = stack.pop()
        for u in np.flatnonzero(D[:, v]):
            if u not in out:
                out.add(int(u))
                stack.append(int(u))
    return out


def _dsep(D, i, j, S):
    keep = _ancestors(D, {i, j} | set(S))
    nb = {v: set() for v in keep}
    for c in keep:
        pa = [int(u) for u in np.flatnonzero(D[:, c]) if u in keep]
        for u in pa:
            nb[u].add(c)
            nb[c].add(u)
        for a, b in itertools.combinations(pa, 2):
            nb[a].add(b)
            nb[b].add(a)
    seen, stack = {i}, [i]
    while stack:
        v = stack.pop()
        for w in nb[v]:
            if w == j:
                return False
            if w not in seen and w not in S:
                seen.add(w)
                stack.append(w)
    return True


def _oracle_test(D):
    """a perfect CI test: p-value 1 if d-separated, 0 otherwise."""
    return lambda i, j, S: 1.0 if _dsep(D, i, j, set(S)) else 0.0


def _vstructs(D):
    p = len(D)
    out = set()
    for k in range(p):
        for i, j in itertools.combinations(np.flatnonzero(D[:, k]), 2):
            if D[i, j] == 0 and D[j, i] == 0:
                out.add((int(min(i, j)), k, int(max(i, j))))
    return out


def _acyclic(D):
    D = D.copy()
    while len(D):
        roots = np.flatnonzero(D.sum(0) == 0)
        if not len(roots):
            return False
        D = np.delete(np.delete(D, roots[0], 0), roots[0], 1)
    return True


def _cpdag_oracle(D):
    p = len(D)
    edges = [(i, j) for i, j in itertools.combinations(range(p), 2) if D[i, j] or D[j, i]]
    target = _vstructs(D)
    members = []
    for bits in itertools.product([0, 1], repeat=len(edges)):
        E = np.zeros((p, p), dtype=int)
        for (i, j), b in zip(edges, bits):
            if b:
                E[i, j] = 1
            else:
                E[j, i] = 1
        if _acyclic(E) and _vstructs(E) == target:
            members.append(E)
    G = np.zeros((p, p), dtype=int)
    for i, j in edges:
        dirs = {int(E[i, j]) for E in members}
        if dirs == {1}:
            G[i, j] = 1
        elif dirs == {0}:
            G[j, i] = 1
        else:
            G[i, j] = G[j, i] = 1
    return G


def _min_sepsets(D):
    p = len(D)
    out = {}
    for i, j in itertools.combinations(range(p), 2):
        if D[i, j] or D[j, i]:
            continue
        rest = [k for k in range(p) if k not in (i, j)]
        for r in range(len(rest) + 1):
            S = next((S for S in itertools.combinations(rest, r) if _dsep(D, i, j, set(S))), None)
            if S is not None:
                out[(i, j)] = S
                break
    return out


def _linear_data(D, n, rng, lo=0.6, hi=1.2):
    p = len(D)
    W = D * rng.uniform(lo, hi, size=(p, p)) * rng.choice([-1, 1], size=(p, p))
    order = []
    remaining = list(range(p))
    while remaining:
        v = next(v for v in remaining if all(u in order for u in np.flatnonzero(D[:, v])))
        order.append(v)
        remaining.remove(v)
    X = np.zeros((n, p))
    for v in order:
        X[:, v] = X @ W[:, v] + rng.normal(size=n)
    return X, W


def _fmt(G):
    p = len(G)
    parts = []
    for i, j in itertools.combinations(range(p), 2):
        if G[i, j] and G[j, i]:
            parts.append(f"{i}-{j}")
        elif G[i, j]:
            parts.append(f"{i}->{j}")
        elif G[j, i]:
            parts.append(f"{j}->{i}")
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------- stage 1: CI tests


def test_stage1_partial_correlation_matches_residuals(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)) @ rng.normal(size=(5, 5))
    for cond in [(), (2,), (2, 3), (1, 3, 4)]:
        got = nb.partial_correlation(X, 0, 1 if 1 not in cond else 2, cond)
        j = 1 if 1 not in cond else 2
        A = np.column_stack([np.ones(300)] + [X[:, k] for k in cond])
        r0 = X[:, 0] - A @ np.linalg.lstsq(A, X[:, 0], rcond=None)[0]
        r1 = X[:, j] - A @ np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        want = np.corrcoef(r0, r1)[0, 1]
        assert abs(got - want) < 1e-8, (
            f"partial correlation of 0 and {j} given {cond}: the correlation of the residuals after regressing both "
            f"on the conditioning columns (with an intercept) is {want:.6f}; got {got:.6f}")


def test_stage1_chain_screening_off(nb):
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.normal(size=n)
    y = 0.9 * x + rng.normal(size=n)
    z = 0.9 * y + rng.normal(size=n)
    data = np.column_stack([x, y, z])
    assert abs(nb.partial_correlation(data, 0, 2)) > 0.3, "in the chain x -> y -> z, x and z are correlated"
    assert abs(nb.partial_correlation(data, 0, 2, (1,))) < 5 / np.sqrt(n), (
        "...but given y, their partial correlation should be about 0 (y screens x off from z)")


def test_stage1_fisher_z(nb):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 4))
    X[:, 1] += 0.25 * X[:, 0]
    for cond in [(), (2,), (2, 3)]:
        r = nb.partial_correlation(X, 0, 1, cond)
        want = erfc(np.sqrt(120 - len(cond) - 3) * abs(np.arctanh(r)) / np.sqrt(2))
        got = nb.fisher_z_test(X, 0, 1, cond)
        assert abs(got - want) < 1e-9, (
            f"fisher_z_test given {cond}: z = atanh(r), statistic sqrt(n - |S| - 3) |z|, two-sided normal p-value "
            f"= {want:.6f}; got {got:.6f}. remember the -|S| in the degrees of freedom.")
    perfect = np.column_stack([X[:, 0], 2 * X[:, 0], X[:, 2]])
    p = nb.fisher_z_test(perfect, 0, 1)
    assert np.isfinite(p) and p < 1e-10, f"a perfectly correlated pair should give p ~ 0 (not nan or an error); got {p}"


# ---------------------------------------------------------------- stage 2: skeleton


def test_stage2_oracle_skeleton_and_sepsets(nb):
    rng = np.random.default_rng(3)
    for _ in range(15):
        D = _random_dag(6, 0.4, rng)
        adj, sep = bounded(nb.pc_skeleton, 6, _oracle_test(D), what="pc_skeleton")
        adj = np.asarray(adj, dtype=bool)
        want = (D + D.T) > 0
        assert np.array_equal(adj, want), (
            f"with a perfect independence test PC must recover the true skeleton. DAG edges "
            f"{_fmt(D)}; missing {np.argwhere(want & ~adj).tolist()}, extra {np.argwhere(adj & ~want).tolist()}")
        nonadj = {(i, j) for i, j in itertools.combinations(range(6), 2) if not want[i, j]}
        assert set(sep) == nonadj, f"sepsets needs one key (i, j), i < j, for each removed edge; got keys {sorted(sep)}"
        for (i, j), S in sep.items():
            assert _dsep(D, i, j, set(S)), f"the recorded sepset {S} for ({i}, {j}) doesn't separate them"


def test_stage2_levels_and_calls(nb):
    calls = []

    def indep(i, j, S):
        calls.append(len(S))
        return 1.0

    adj, sep = bounded(nb.pc_skeleton, 6, indep, what="pc_skeleton")
    assert not np.asarray(adj).any() and len(calls) == 15, (
        f"if every test says independent, every edge goes at level 0 after one test per pair: 15 tests for 6 "
        f"variables, got {len(calls)}")
    assert all(S == () for S in sep.values()), "level-0 removals have the empty separating set ()"

    seen = []
    D = np.zeros((5, 5), dtype=int)
    D[0, 1] = D[1, 2] = D[2, 3] = D[3, 4] = 1
    orc = _oracle_test(D)
    adj, _ = nb.pc_skeleton(5, lambda i, j, S: (seen.append(len(S)), orc(i, j, S))[1], max_cond=0)
    assert max(seen) == 0, "max_cond=0 means only marginal (empty-set) tests"
    assert np.asarray(adj).sum() == 20, "a chain has no marginal independences, so max_cond=0 removes nothing"


def _fz(X):
    from math import atanh, erfc as _erfc, sqrt

    def test(i, j, S):
        idx = [i, j] + list(S)
        P = np.linalg.inv(np.corrcoef(X[:, idx], rowvar=False))
        r = float(np.clip(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]), -0.9999999, 0.9999999))
        return _erfc(sqrt(len(X) - len(S) - 3) * abs(atanh(r)) / sqrt(2))

    return test


def test_stage2_order_independent(nb):
    rng = np.random.default_rng(9)
    for _ in range(12):
        D = _random_dag(6, 0.5, rng)
        X, _ = _linear_data(D, 60, rng, lo=0.2, hi=0.6)  # few samples, weak edges: tests disagree a lot
        base = np.asarray(nb.pc_skeleton(6, _fz(X), alpha=0.05)[0], dtype=bool)
        for _ in range(3):
            perm = rng.permutation(6)
            adj_p = np.asarray(nb.pc_skeleton(6, _fz(X[:, perm]), alpha=0.05)[0], dtype=bool)
            back = np.zeros_like(adj_p)
            back[np.ix_(perm, perm)] = adj_p
            assert np.array_equal(back, base), (
                "relabelling the variables changed the skeleton. stable PC takes each level's conditioning sets "
                "from a SNAPSHOT of the neighbours made at the start of the level, not from the live adjacency.")


def test_stage2_recovers_skeleton_from_data(nb):
    rng = np.random.default_rng(4)
    D = np.zeros((4, 4), dtype=int)
    D[0, 1] = D[1, 2] = D[2, 3] = 1
    X, _ = _linear_data(D, 3000, rng)
    from math import atanh, erfc as _erfc, sqrt

    def fz(i, j, S):
        idx = [i, j] + list(S)
        P = np.linalg.inv(np.corrcoef(X[:, idx], rowvar=False))
        r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
        return _erfc(sqrt(3000 - len(S) - 3) * abs(atanh(r)) / sqrt(2))

    adj, _ = nb.pc_skeleton(4, fz, alpha=0.001)
    assert np.array_equal(np.asarray(adj, dtype=bool), (D + D.T) > 0), (
        f"on 3,000 samples from the chain 0 -> 1 -> 2 -> 3, PC should find exactly its 3 edges; got "
        f"{np.argwhere(np.triu(adj)).tolist()}")


# ---------------------------------------------------------------- stage 3: orientation


def test_stage3_meek_rules(nb):
    G = np.zeros((3, 3), dtype=int)
    G[0, 1] = 1
    G[1, 2] = G[2, 1] = 1
    out = nb.apply_meek_rules(G)
    assert out[1, 2] == 1 and out[2, 1] == 0, "R1: 0 -> 1 - 2 with 0, 2 non-adjacent orients 1 -> 2 (else a new v-structure)"
    assert G[2, 1] == 1, "apply_meek_rules must work on a copy, not modify its input"
    G = np.zeros((3, 3), dtype=int)
    G[0, 1] = G[1, 2] = 1
    G[0, 2] = G[2, 0] = 1
    out = nb.apply_meek_rules(G)
    assert out[0, 2] == 1 and out[2, 0] == 0, "R2: 0 -> 1 -> 2 with 0 - 2 orients 0 -> 2 (else a cycle)"
    G = np.zeros((4, 4), dtype=int)
    for k in (1, 2, 3):
        G[0, k] = G[k, 0] = 1
    G[2, 1] = G[3, 1] = 1
    out = nb.apply_meek_rules(G)
    assert out[0, 1] == 1 and out[1, 0] == 0, (
        "R3: 0 - 1, 0 - 2, 0 - 3, 2 -> 1 <- 3, with 2 and 3 non-adjacent, orients 0 -> 1")
    assert out[0, 2] == out[2, 0] == 1 and out[0, 3] == out[3, 0] == 1, "...and leaves 0 - 2 and 0 - 3 undirected"


def test_stage3_orient_matches_cpdag(nb):
    rng = np.random.default_rng(5)
    for _ in range(25):
        D = _random_dag(6, 0.45, rng)
        want = _cpdag_oracle(D)
        got = np.asarray(bounded(nb.orient, (D + D.T) > 0, _min_sepsets(D), what="orient"), dtype=int)
        assert np.array_equal(got, want), (
            f"orienting the true skeleton with true separating sets should give the CPDAG. DAG {_fmt(D)}: "
            f"expected {_fmt(want)}, got {_fmt(got)}")


def test_stage3_orient_first_decision_wins(nb):
    # chain skeleton 0 - 1 - 2 - 3 with sepsets that (wrongly) claim two v-structures: 0 -> 1 <- 2 and 1 -> 2 <- 3
    adj = np.zeros((4, 4), dtype=bool)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        adj[i, j] = adj[j, i] = True
    G = np.asarray(nb.orient(adj, {(0, 2): (), (1, 3): (), (0, 3): (1, 2)}), dtype=int)
    want = np.zeros((4, 4), dtype=int)
    want[0, 1] = want[2, 1] = want[3, 2] = 1
    assert np.array_equal(G, want), (
        f"conflicting v-structures: k = 1 is processed first and orients 0 -> 1 <- 2; at k = 2 the edge 1 - 2 is "
        f"already directed, so only 3 -> 2 is added. expected {_fmt(want)}, got {_fmt(G)}. orient an edge only "
        "while it is still undirected (never overwrite, never delete an edge).")


def test_stage3_pc_uses_alpha(nb):
    rng = np.random.default_rng(10)
    for _ in range(50):
        x = rng.normal(size=300)
        X = np.column_stack([x, 0.1 * x + rng.normal(size=300)])
        p = erfc(np.sqrt(297) * abs(np.arctanh(np.corrcoef(X.T)[0, 1])) / np.sqrt(2))
        if 1e-4 < p < 0.08:
            break
    hi = np.asarray(nb.pc(X, alpha=min(0.9, 10 * p)), dtype=int)
    lo = np.asarray(nb.pc(X, alpha=p / 10), dtype=int)
    assert hi[0, 1] == 1 and hi[1, 0] == 1, f"the marginal test has p = {p:.4f}; with alpha = {min(0.9, 10 * p):.3f} keep the edge"
    assert lo[0, 1] == 0 and lo[1, 0] == 0, f"...and with alpha = {p / 10:.5f} remove it: pass alpha through to pc_skeleton"


def test_stage3_pc_on_data(nb):
    rng = np.random.default_rng(6)
    D = np.zeros((4, 4), dtype=int)
    D[0, 2] = D[1, 2] = D[2, 3] = 1
    X, _ = _linear_data(D, 3000, rng)
    G = np.asarray(nb.pc(X, alpha=1e-4), dtype=int)
    assert np.array_equal(G, D), (
        f"0 -> 2 <- 1, 2 -> 3 is identifiable: the collider orients 0 -> 2 <- 1 and Meek's R1 then orients 2 -> 3. "
        f"from 3,000 samples PC gave {_fmt(G)}")


# ---------------------------------------------------------------- stage 4: CPDAGs and SHD


def test_stage4_dag_to_cpdag_hand(nb):
    chain = np.zeros((3, 3), dtype=int)
    chain[0, 1] = chain[1, 2] = 1
    G = np.asarray(nb.dag_to_cpdag(chain))
    assert G[0, 1] == G[1, 0] == G[1, 2] == G[2, 1] == 1, (
        f"0 -> 1 -> 2 is equivalent to 0 <- 1 <- 2 and 0 <- 1 -> 2: every edge is undirected; got {_fmt(G)}")
    coll = np.zeros((3, 3), dtype=int)
    coll[0, 1] = coll[2, 1] = 1
    assert np.array_equal(nb.dag_to_cpdag(coll), coll), "the collider 0 -> 1 <- 2 is alone in its class: fully directed"


def test_stage4_dag_to_cpdag_random(nb):
    rng = np.random.default_rng(7)
    for _ in range(25):
        D = _random_dag(6, 0.45, rng)
        want = _cpdag_oracle(D)
        got = np.asarray(nb.dag_to_cpdag(D), dtype=int)
        assert np.array_equal(got, want), f"DAG {_fmt(D)}: expected CPDAG {_fmt(want)}, got {_fmt(got)}"


def test_stage4_shd(nb):
    shd = nb.structural_hamming_distance
    A = np.zeros((4, 4), dtype=int)
    A[0, 1] = A[1, 2] = 1
    A[2, 3] = A[3, 2] = 1
    assert shd(A, A) == 0, "a graph is at distance 0 from itself"
    B = A.copy()
    B[0, 1], B[1, 0] = 0, 1
    assert shd(A, B) == 1, "one reversed edge costs 1"
    C = A.copy()
    C[3, 2] = 0
    assert shd(A, C) == 1, "an undirected edge that became directed costs 1"
    E = A.copy()
    E[1, 2] = 0
    E[0, 3] = 1
    assert shd(A, E) == 2, "a missing edge plus an extra edge cost 2"
    assert shd(np.zeros((4, 4), int), A) == 3, "the empty graph is 3 away from a graph with 3 edges"


# ---------------------------------------------------------------- stage 5: IDA


def test_stage5_possible_parent_sets_hand(nb):
    chain = np.zeros((3, 3), dtype=int)
    chain[0, 1] = chain[1, 0] = chain[1, 2] = chain[2, 1] = 1
    got = [tuple(s) for s in nb.possible_parent_sets(chain, 1)]
    assert got == [(), (0,), (2,)], (
        f"in 0 - 1 - 2, node 1 can have no parent, parent 0, or parent 2, but not both (0 -> 1 <- 2 would be a "
        f"new v-structure, since 0 and 2 aren't adjacent). expected [(), (0,), (2,)] sorted by size then "
        f"lexicographically; got {got}")
    tri = np.ones((3, 3), dtype=int) - np.eye(3, dtype=int)
    got = [tuple(s) for s in nb.possible_parent_sets(tri, 0)]
    assert got == [(), (1,), (2,), (1, 2)], f"in a fully undirected triangle every subset works; got {got}"
    G = np.zeros((4, 4), dtype=int)
    G[0, 2] = G[1, 2] = 1
    G[2, 3] = 1
    assert [tuple(s) for s in nb.possible_parent_sets(G, 2)] == [(0, 1)], (
        "with all edges at node 2 directed, its only parent set is its parents (0, 1)")


def test_stage5_ida_contains_truth(nb):
    rng = np.random.default_rng(8)
    D = np.zeros((4, 4), dtype=int)
    D[0, 1] = D[1, 2] = D[0, 2] = D[2, 3] = 1  # 0 -> 1 -> 2 -> 3 and 0 -> 2: no v-structure, all undirected
    X, W = _linear_data(D, 20000, rng)
    cp = _cpdag_oracle(D)
    eff = nb.ida_effects(X, 1, 3, cp)
    sets = nb.possible_parent_sets(cp, 1)
    assert len(eff) == len(sets), f"one effect per possible parent set ({len(sets)}); got {len(eff)}"
    truth = W[1, 2] * W[2, 3]
    assert min(abs(e - truth) for e in eff) < 0.05, (
        f"the true effect of 1 on 3 is {truth:.3f}; one of the IDA estimates (OLS of y on x and each possible "
        f"parent set) should be close to it. got {np.round(eff, 3).tolist()}")
    eff_back = nb.ida_effects(X, 2, 1, cp)  # node 1 is a possible parent of node 2
    assert any(e == 0.0 for e in eff_back), (
        "when y is in a possible parent set of x, x can't cause y in that DAG: report an effect of exactly 0.0 for it")
