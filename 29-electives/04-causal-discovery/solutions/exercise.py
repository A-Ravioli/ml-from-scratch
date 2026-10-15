import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # causal discovery · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. variables are the columns
    `0..p-1` of a data matrix. a graph is a `(p, p)` 0/1 integer matrix `G`: `G[i, j] = 1`
    and `G[j, i] = 0` means $i \to j$; `G[i, j] = G[j, i] = 1` means an undirected edge
    $i - j$. a DAG has no undirected edges; a CPDAG may have both kinds.

    | stage | you build |
    |---|---|
    | 1 | `partial_correlation`, `fisher_z_test`: a conditional independence test |
    | 2 | `pc_skeleton`: the PC algorithm's adjacency search, with separating sets |
    | 3 | `apply_meek_rules`, `orient`, `pc`: Meek's rules, v-structures, then the whole algorithm |
    | 4 | `dag_to_cpdag`, `structural_hamming_distance`: the target PC aims for, and how far off you are |
    | 5 | `possible_parent_sets`, `ida_effects`: estimating an effect when the graph is only partly oriented |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def partial_correlation(data, i, j, cond=()):
    """stage 1: correlation of columns i and j of data (n, p) after linearly regressing out the columns in cond."""
    data = np.asarray(data, dtype=float)
    idx = [i, j] + list(cond)
    P = np.linalg.pinv(np.corrcoef(data[:, idx], rowvar=False))
    return float(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))


@app.function
def fisher_z_test(data, i, j, cond=()):
    """stage 1: two-sided p-value of H0: partial correlation of i, j given cond = 0, by Fisher's z transform."""
    n = len(data)
    r = float(np.clip(partial_correlation(data, i, j, cond), -0.9999999, 0.9999999))
    stat = math.sqrt(n - len(cond) - 3) * abs(math.atanh(r))
    return math.erfc(stat / math.sqrt(2))


@app.cell(hide_code=True)
def _():
    checks.run(1, partial_correlation=partial_correlation, fisher_z_test=fisher_z_test)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def pc_skeleton(n_vars, ci_test, *, alpha=0.05, max_cond=None):
    """stage 2: stable-PC adjacency search; returns (adj bool (p, p) symmetric, sepsets {(i, j) i < j: tuple})."""
    adj = ~np.eye(n_vars, dtype=bool)
    sepsets = {}
    level = 0
    while max_cond is None or level <= max_cond:
        snapshot = [np.flatnonzero(adj[i]) for i in range(n_vars)]
        any_testable = False
        for i in range(n_vars):
            for j in snapshot[i]:
                if not adj[i, j]:
                    continue
                others = [k for k in snapshot[i] if k != j]
                if len(others) < level:
                    continue
                any_testable = True
                for S in itertools.combinations(others, level):
                    if ci_test(i, int(j), tuple(int(s) for s in S)) > alpha:
                        adj[i, j] = adj[j, i] = False
                        sepsets[(min(i, int(j)), max(i, int(j)))] = tuple(int(s) for s in S)
                        break
        if not any_testable:
            break
        level += 1
    return adj, sepsets


@app.cell(hide_code=True)
def _():
    checks.run(2, pc_skeleton=pc_skeleton)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def apply_meek_rules(G):
    """stage 3: a copy of the graph matrix G with Meek's rules R1-R3 applied until nothing changes."""
    G = np.array(G, dtype=int)
    p = len(G)

    def undirected(a, b):
        return G[a, b] == 1 and G[b, a] == 1

    def directed(a, b):
        return G[a, b] == 1 and G[b, a] == 0

    def adjacent(a, b):
        return G[a, b] == 1 or G[b, a] == 1

    changed = True
    while changed:
        changed = False
        for a, b in itertools.permutations(range(p), 2):
            if not undirected(a, b):
                continue
            r1 = any(directed(c, a) and not adjacent(c, b) for c in range(p) if c not in (a, b))
            r2 = any(directed(a, c) and directed(c, b) for c in range(p) if c not in (a, b))
            r3 = any(undirected(a, c) and undirected(a, d) and directed(c, b) and directed(d, b) and not adjacent(c, d)
                     for c, d in itertools.combinations([k for k in range(p) if k not in (a, b)], 2))
            if r1 or r2 or r3:
                G[b, a] = 0
                changed = True
    return G


@app.function
def orient(adj, sepsets):
    """stage 3: CPDAG from a skeleton and separating sets: v-structures, then Meek's rules R1-R3."""
    adj = np.asarray(adj, dtype=bool)
    p = len(adj)
    G = adj.astype(int)
    for k in range(p):
        for i, j in itertools.combinations(np.flatnonzero(adj[k]), 2):
            if adj[i, j]:
                continue
            if k not in sepsets.get((min(i, j), max(i, j)), ()):
                for a in (i, j):
                    if G[a, k] == 1 and G[k, a] == 1:
                        G[k, a] = 0
    return apply_meek_rules(G)


@app.function
def pc(data, *, alpha=0.05, max_cond=None):
    """stage 3: the PC algorithm on data (n, p) with Fisher-z tests; returns the estimated CPDAG (p, p)."""
    data = np.asarray(data, dtype=float)
    adj, sepsets = pc_skeleton(data.shape[1], lambda i, j, S: fisher_z_test(data, i, j, S),
                               alpha=alpha, max_cond=max_cond)
    return orient(adj, sepsets)


@app.cell(hide_code=True)
def _():
    checks.run(3, apply_meek_rules=apply_meek_rules, orient=orient, pc=pc, pc_skeleton=pc_skeleton)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def dag_to_cpdag(dag):
    """stage 4: the CPDAG of a DAG's Markov equivalence class: keep its v-structures, undirect the rest, apply Meek."""
    D = np.asarray(dag, dtype=int)
    p = len(D)
    skel = (D + D.T) > 0
    G = skel.astype(int)
    for k in range(p):
        pa = np.flatnonzero(D[:, k])
        for i, j in itertools.combinations(pa, 2):
            if not skel[i, j]:
                G[k, i] = G[k, j] = 0
    return apply_meek_rules(G)


@app.function
def structural_hamming_distance(G1, G2):
    """stage 4: number of node pairs whose edge (none, i->j, j->i, i-j) differs between graph matrices G1 and G2."""
    A, B = np.asarray(G1, dtype=int), np.asarray(G2, dtype=int)
    p = len(A)
    return int(sum((A[i, j], A[j, i]) != (B[i, j], B[j, i]) for i, j in itertools.combinations(range(p), 2)))


@app.cell(hide_code=True)
def _():
    checks.run(4, dag_to_cpdag=dag_to_cpdag, structural_hamming_distance=structural_hamming_distance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def possible_parent_sets(cpdag, x):
    """stage 5: every parent set of x consistent with the CPDAG (adds no new v-structure at x), as sorted tuples."""
    G = np.asarray(cpdag, dtype=int)
    p = len(G)
    parents = [k for k in range(p) if G[k, x] == 1 and G[x, k] == 0]
    sib = [k for k in range(p) if G[k, x] == 1 and G[x, k] == 1]
    adjacent = lambda a, b: G[a, b] == 1 or G[b, a] == 1  # noqa: E731
    out = []
    for r in range(len(sib) + 1):
        for S in itertools.combinations(sib, r):
            ok = all(adjacent(s, q) for s in S for q in list(S) + parents if q != s)
            if ok:
                out.append(tuple(sorted(parents + list(S))))
    return sorted(out, key=lambda t: (len(t), t))


@app.function
def ida_effects(data, x, y, cpdag):
    """stage 5: one OLS estimate of the effect of x on y per possible parent set (coefficient of x in y ~ x + parents)."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    effects = []
    for S in possible_parent_sets(cpdag, x):
        if y in S:
            effects.append(0.0)
            continue
        A = np.column_stack([np.ones(n), data[:, x]] + [data[:, s] for s in S])
        effects.append(float(np.linalg.lstsq(A, data[:, y], rcond=None)[0][1]))
    return effects


@app.cell(hide_code=True)
def _():
    checks.run(5, possible_parent_sets=possible_parent_sets, ida_effects=ida_effects)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`), which ends with the
    block's project: discover a graph on semi-synthetic data and estimate an effect from it.
    """)
    return


if __name__ == "__main__":
    app.run()
