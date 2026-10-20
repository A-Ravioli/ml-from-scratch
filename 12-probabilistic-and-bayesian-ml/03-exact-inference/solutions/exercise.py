import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.bayes import Factor  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # exact inference · build (reference solution)

    work through the stages in order. everything here operates on lists of your `Factor`s
    from topic 02: the model is their product, and inference means summing it without ever
    building the full joint table. `variable_elimination` and `sum_product` become part of
    `mlfs.bayes`.

    | stage | you build |
    |---|---|
    | 1 | `eliminate`: sum one variable out of a factor list |
    | 2 | `min_degree_order`, `elimination_cost`: choose an order, and predict what it costs |
    | 3 | `variable_elimination` → `mlfs.bayes`: any marginal or conditional, exactly |
    | 4 | `sum_product` → `mlfs.bayes`: every marginal of a tree at once, by message passing |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def eliminate(factors, var):
    """stage 1: multiply the factors that mention var, sum var out; returns the new list (the others first, in order)."""
    touching = [f for f in factors if var in f.variables]
    rest = [f for f in factors if var not in f.variables]
    if not touching:
        return list(factors)
    prod = touching[0]
    for f in touching[1:]:
        prod = prod * f
    return rest + [prod.marginalize([var])]


@app.cell(hide_code=True)
def _():
    checks.run(1, eliminate=eliminate, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def min_degree_order(factors, variables):
    """stage 2: a greedy elimination order for `variables`: always the one with fewest neighbours (ties: earliest in `variables`)."""
    adj = {}
    for f in factors:
        for v in f.variables:
            adj.setdefault(v, set()).update(u for u in f.variables if u != v)
    todo = list(variables)
    order = []
    while todo:
        v = min(todo, key=lambda u: len(adj.get(u, ())))  # min keeps the first of equal keys
        nbrs = adj.pop(v, set())
        for a in nbrs:
            adj[a].discard(v)
            adj[a].update(b for b in nbrs if b != a)
        todo.remove(v)
        order.append(v)
    return order


@app.function
def elimination_cost(factors, order):
    """stage 2: the number of entries of the largest product factor formed when eliminating `order` (no arithmetic)."""
    cards = {}
    scopes = []
    for f in factors:
        cards.update(zip(f.variables, f.cardinalities))
        scopes.append(set(f.variables))
    biggest = max((int(np.prod([cards[v] for v in s])) for s in scopes), default=1)
    for var in order:
        touching = [s for s in scopes if var in s]
        if not touching:
            continue
        union = set().union(*touching)
        biggest = max(biggest, int(np.prod([cards[v] for v in union])))
        scopes = [s for s in scopes if var not in s] + [union - {var}]
    return biggest


@app.cell(hide_code=True)
def _():
    checks.run(2, min_degree_order=min_degree_order, elimination_cost=elimination_cost, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def variable_elimination(factors, query, evidence=None, order=None):
    """stage 3: P(query | evidence) as a normalized Factor over query (a name or tuple), eliminating the rest in `order`."""
    evidence = evidence or {}
    query = (query,) if isinstance(query, (str, int)) else tuple(query)
    current = [f.reduce(evidence) for f in factors]
    all_vars = []
    for f in current:
        all_vars.extend(v for v in f.variables if v not in all_vars)
    hidden = [v for v in all_vars if v not in query]
    if order is None:
        order = min_degree_order(current, hidden)
    else:
        order = [v for v in order if v in hidden]
        missing = set(hidden) - set(order)
        if missing:
            raise ValueError(f"order doesn't include {sorted(map(str, missing))}")
    for v in order:
        current = eliminate(current, v)
    result = current[0]
    for f in current[1:]:
        result = result * f
    perm = [result.variables.index(q) for q in query]
    return Factor(query, [result.cardinalities[i] for i in perm], np.transpose(result.values, perm)).normalize()


@app.cell(hide_code=True)
def _():
    checks.run(3, variable_elimination=variable_elimination, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def sum_product(factors, evidence=None):
    """stage 4: belief propagation on a tree-shaped factor graph: dict var -> normalized marginal (card,) given evidence."""
    evidence = evidence or {}
    fs = [f.reduce(evidence) for f in factors]
    fs = [f for f in fs if f.variables]  # fully observed factors are constants
    nbrs_v = {}
    cards = {}
    for i, f in enumerate(fs):
        for v, c in zip(f.variables, f.cardinalities):
            nbrs_v.setdefault(v, []).append(i)
            cards[v] = c
    n_nodes = len(fs) + len(nbrs_v)
    n_edges = sum(len(f.variables) for f in fs)
    seen, components = set(), 0
    for start in nbrs_v:
        if start in seen:
            continue
        components += 1
        stack = [("v", start)]
        seen.add(start)
        while stack:
            kind, x = stack.pop()
            nexts = [("f", i) for i in nbrs_v[x]] if kind == "v" else [("v", u) for u in fs[x].variables]
            for node in nexts:
                key = node[1] if node[0] == "v" else ("factor", node[1])
                if key not in seen:
                    seen.add(key)
                    stack.append(node)
    if n_edges != n_nodes - components:
        raise ValueError("sum_product needs a tree-structured factor graph (no cycles)")

    # schedule: from each component's root, a breadth-first order of the bipartite tree. messages flow from the
    # leaves to the root (reverse order), then back out (forward order): 2 per edge, no recursion.
    adj = {("v", v): [("f", i) for i in nbrs_v[v]] for v in nbrs_v}
    adj.update({("f", i): [("v", u) for u in f.variables] for i, f in enumerate(fs)})
    msgs = {}

    def message(src, dst):
        if src[0] == "v":  # variable -> factor: product of the messages from the variable's other factors
            m = np.ones(cards[src[1]])
            for g in adj[src]:
                if g != dst:
                    m = m * msgs[(g, src)]
        else:  # factor -> variable: multiply in the other variables' messages, sum them out
            f = fs[src[1]]
            vals = f.values
            for axis, u in enumerate(f.variables):
                if u != dst[1]:
                    shape = [1] * vals.ndim
                    shape[axis] = cards[u]
                    vals = vals * msgs[(("v", u), src)].reshape(shape)
            m = vals.sum(axis=tuple(a for a, u in enumerate(f.variables) if u != dst[1]))
        return m / m.sum()

    done = set()
    for start in nbrs_v:
        root = ("v", start)
        if root in done:
            continue
        order, parent = [root], {root: None}
        k = 0
        while k < len(order):
            node = order[k]
            k += 1
            for nxt in adj[node]:
                if nxt not in parent:
                    parent[nxt] = node
                    order.append(nxt)
        done.update(order)
        for node in reversed(order):  # collect: leaves towards the root
            if parent[node] is not None:
                msgs[(node, parent[node])] = message(node, parent[node])
        for node in order:  # distribute: root towards the leaves
            for child in adj[node]:
                if child != parent[node]:
                    msgs[(node, child)] = message(node, child)

    out = {}
    for v in nbrs_v:
        b = np.ones(cards[v])
        for g in adj[("v", v)]:
            b = b * msgs[(g, ("v", v))]
        out[v] = b / b.sum()
    return out


@app.cell(hide_code=True)
def _():
    checks.run(4, sum_product=sum_product, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `variable_elimination` and `sum_product` are part of
    `mlfs.bayes`. then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
