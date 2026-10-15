import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # structural causal models & do-calculus · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. a causal graph is a plain dict
    `parents` that maps every node name to the list of its parents.

    | stage | you build |
    |---|---|
    | 1 | `topological_order`, `ancestors`, `descendants`: walking a DAG |
    | 2 | `SCM`, `intervene_graph`: a structural causal model you can sample from, with and without `do` |
    | 3 | `d_separated`, `is_backdoor_adjustment`: reading independences and valid adjustment sets off the graph |
    | 4 | `backdoor_adjust`, `frontdoor_adjust`: interventional means from observational data |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def topological_order(parents):
    """stage 1: the nodes of the DAG `parents` (dict node -> list of parents) in a topological order; ValueError on a cycle or unknown parent."""
    for node, pa in parents.items():
        for p in pa:
            if p not in parents:
                raise ValueError(f"{node!r} has parent {p!r}, which is not a node of the graph")
    placed, order = set(), []
    remaining = list(parents)
    while remaining:
        for node in remaining:
            if all(p in placed for p in parents[node]):
                order.append(node)
                placed.add(node)
                remaining.remove(node)
                break
        else:
            raise ValueError(f"the graph has a directed cycle among {remaining}")
    return order


@app.function
def ancestors(parents, node):
    """stage 1: the set of nodes with a directed path into `node` (not including `node`)."""
    seen, stack = set(), list(parents[node])
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(parents[p])
    return seen


@app.function
def descendants(parents, node):
    """stage 1: the set of nodes reachable from `node` along directed edges (not including `node`)."""
    children = {v: [c for c in parents if v in parents[c]] for v in parents}
    seen, stack = set(), list(children[node])
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.add(c)
            stack.extend(children[c])
    return seen


@app.cell(hide_code=True)
def _():
    checks.run(1, topological_order=topological_order, ancestors=ancestors, descendants=descendants)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class SCM:
    """a structural causal model: each node is f(its parents' values, its own N(0, 1) noise)."""

    def __init__(self, parents, mechanisms):
        """stage 2: store parents, nodes (dict order), mechanisms and order = topological_order(parents)."""
        if set(mechanisms) != set(parents):
            raise ValueError("mechanisms must have exactly one entry per node")
        self.parents = {v: list(pa) for v, pa in parents.items()}
        self.nodes = list(parents)
        self.mechanisms = dict(mechanisms)
        self.order = topological_order(self.parents)

    def sample(self, n, *, do=None, rng=None):
        """stage 2: n joint draws as a dict node -> float array (n,); `do` maps nodes to forced values."""
        do = {} if do is None else dict(do)
        for v in do:
            if v not in self.parents:
                raise ValueError(f"can't intervene on {v!r}: not a node")
        noise = as_rng(rng).normal(size=(n, len(self.nodes)))
        col = {v: k for k, v in enumerate(self.nodes)}
        values = {}
        for v in self.order:
            if v in do:
                values[v] = np.broadcast_to(np.asarray(do[v], dtype=float), (n,)).copy()
            else:
                pa = {p: values[p] for p in self.parents[v]}
                out = np.asarray(self.mechanisms[v](pa, noise[:, col[v]]), dtype=float)
                values[v] = np.broadcast_to(out, (n,)).copy()
        return {v: values[v] for v in self.nodes}


@app.function
def intervene_graph(parents, nodes):
    """stage 2: a new parents dict with every edge INTO the given nodes removed (graph surgery)."""
    nodes = {nodes} if isinstance(nodes, str) else set(nodes)
    return {v: ([] if v in nodes else list(pa)) for v, pa in parents.items()}


@app.cell(hide_code=True)
def _():
    checks.run(2, SCM=SCM, intervene_graph=intervene_graph, topological_order=topological_order)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def d_separated(parents, xs, ys, zs=()):
    """stage 3: True if every path between the node sets xs and ys is blocked by zs in the DAG `parents`."""
    as_set = lambda s: {s} if isinstance(s, str) else set(s)  # noqa: E731
    xs, ys, zs = as_set(xs), as_set(ys), as_set(zs)
    if xs & ys:
        return False
    keep = xs | ys | zs
    for v in list(keep):
        keep |= ancestors(parents, v)
    nbrs = {v: set() for v in keep}
    for child in keep:  # moralize the ancestral graph
        pa = [p for p in parents[child] if p in keep]
        for p in pa:
            nbrs[p].add(child)
            nbrs[child].add(p)
        for i, a in enumerate(pa):
            for b in pa[i + 1:]:
                nbrs[a].add(b)
                nbrs[b].add(a)
    seen, stack = set(xs), list(xs)
    while stack:  # search from xs, never entering zs
        v = stack.pop()
        for w in nbrs[v]:
            if w in zs or w in seen:
                continue
            if w in ys:
                return False
            seen.add(w)
            stack.append(w)
    return True


@app.function
def is_backdoor_adjustment(parents, x, y, zs):
    """stage 3: True if zs satisfies the backdoor criterion for the effect of node x on node y."""
    zs = {zs} if isinstance(zs, str) else set(zs)
    if x in zs or y in zs:
        return False
    if zs & descendants(parents, x):
        return False
    no_out = {v: [p for p in pa if p != x] for v, pa in parents.items()}
    return d_separated(no_out, {x}, {y}, zs)


@app.cell(hide_code=True)
def _():
    checks.run(3, d_separated=d_separated, is_backdoor_adjustment=is_backdoor_adjustment)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def backdoor_adjust(data, x, y, zs, x_value):
    """stage 4: E[y | do(x = x_value)] = sum_z P(z) E[y | x_value, z] for discrete zs; ValueError if a stratum lacks x_value."""
    zs = [zs] if isinstance(zs, str) else list(zs)
    xv = np.asarray(data[x])
    yv = np.asarray(data[y], dtype=float)
    treated = xv == x_value
    if not zs:
        if not treated.any():
            raise ValueError(f"no rows with {x} = {x_value}")
        return float(yv[treated].mean())
    Z = np.column_stack([np.asarray(data[z]) for z in zs])
    strata, label = np.unique(Z, axis=0, return_inverse=True)
    label = label.ravel()
    total = 0.0
    for s in range(len(strata)):
        in_s = label == s
        cell = in_s & treated
        if not cell.any():
            raise ValueError(f"positivity fails: no rows with {x} = {x_value} in stratum {dict(zip(zs, strata[s]))}")
        total += in_s.mean() * yv[cell].mean()
    return float(total)


@app.function
def frontdoor_adjust(data, x, y, m, x_value):
    """stage 4: E[y | do(x = x_value)] = sum_m P(m | x_value) sum_x' P(x') E[y | m, x'] for discrete x and m."""
    xv, mv = np.asarray(data[x]), np.asarray(data[m])
    yv = np.asarray(data[y], dtype=float)
    treated = xv == x_value
    if not treated.any():
        raise ValueError(f"no rows with {x} = {x_value}")
    total = 0.0
    for m_val in np.unique(mv[treated]):
        p_m = np.mean(mv[treated] == m_val)
        inner = 0.0
        for x_val in np.unique(xv):
            cell = (mv == m_val) & (xv == x_val)
            if not cell.any():
                raise ValueError(f"positivity fails: no rows with {m} = {m_val} and {x} = {x_val}")
            inner += np.mean(xv == x_val) * yv[cell].mean()
        total += p_m * inner
    return float(total)


@app.cell(hide_code=True)
def _():
    checks.run(4, backdoor_adjust=backdoor_adjust, frontdoor_adjust=frontdoor_adjust, SCM=SCM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
