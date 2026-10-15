import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from collections import deque

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # graphical models · build (reference solution)

    work through the stages in order. a **factor** is a table of non-negative numbers indexed
    by the values of some discrete variables; every distribution in this topic is a product of
    factors, and the graph says which variables share a factor. `Factor` and `d_separated`
    become part of `mlfs.bayes`.

    | stage | you build |
    |---|---|
    | 1 | `Factor` (`__init__`, `__mul__`) → `mlfs.bayes`: tables over named variables, and their product |
    | 2 | `Factor.marginalize`, `.reduce`, `.normalize`, and `joint_query`: brute-force inference |
    | 3 | `moralize`, `markov_blanket`, `separated`: undirected graphs (MRFs) and graph separation |
    | 4 | `ancestors`, `d_separated` → `mlfs.bayes`: reading independence off a Bayes net |
    | 5 | `factor_graph`, `is_tree`: the bipartite view that exact inference runs on |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Factor:
    """a non-negative table phi(variables): one axis per variable, in the order of `variables`."""

    def __init__(self, variables, cardinalities, values):
        """stage 1: store variables (tuple), cardinalities (tuple of ints) and values (float array of that shape)."""
        self.variables = tuple(variables)
        self.cardinalities = tuple(int(c) for c in cardinalities)
        if len(set(self.variables)) != len(self.variables):
            raise ValueError(f"repeated variable in {self.variables}")
        if len(self.variables) != len(self.cardinalities):
            raise ValueError("one cardinality per variable")
        values = np.asarray(values, dtype=float)
        if values.size != int(np.prod(self.cardinalities)):
            raise ValueError(f"values has {values.size} entries, cardinalities {self.cardinalities} need "
                             f"{int(np.prod(self.cardinalities))}")
        self.values = values.reshape(self.cardinalities).copy()

    def _aligned(self, variables, cards):
        """this factor's values with one axis per entry of `variables` (size 1 where absent)."""
        order = [self.variables.index(v) for v in variables if v in self.variables]
        vals = np.transpose(self.values, order)
        shape = [cards[i] if v in self.variables else 1 for i, v in enumerate(variables)]
        return vals.reshape(shape)

    def __mul__(self, other):
        """stage 1: the product over the union of variables (self's order, then other's new ones)."""
        variables = list(self.variables)
        cards = list(self.cardinalities)
        for v, c in zip(other.variables, other.cardinalities):
            if v in variables:
                if cards[variables.index(v)] != c:
                    raise ValueError(f"variable {v!r} has cardinality {cards[variables.index(v)]} and {c}")
            else:
                variables.append(v)
                cards.append(c)
        values = self._aligned(variables, cards) * other._aligned(variables, cards)
        return Factor(variables, cards, values)

    def marginalize(self, variables):
        """stage 2: sum out the given variable name(s); returns a new Factor over the rest."""
        if isinstance(variables, (str, int)) or not np.iterable(variables):
            variables = [variables]
        drop = set(variables)
        unknown = drop - set(self.variables)
        if unknown:
            raise ValueError(f"cannot marginalize {sorted(map(str, unknown))}: not in {self.variables}")
        axes = tuple(i for i, v in enumerate(self.variables) if v in drop)
        keep = [i for i, v in enumerate(self.variables) if v not in drop]
        return Factor([self.variables[i] for i in keep], [self.cardinalities[i] for i in keep],
                      self.values.sum(axis=axes))

    def reduce(self, evidence):
        """stage 2: fix the variables in `evidence` (dict name -> value index) and drop them; others are ignored."""
        index = []
        keep = []
        for i, v in enumerate(self.variables):
            if v in evidence:
                index.append(int(evidence[v]))
            else:
                index.append(slice(None))
                keep.append(i)
        return Factor([self.variables[i] for i in keep], [self.cardinalities[i] for i in keep],
                      self.values[tuple(index)])

    def normalize(self):
        """stage 2: a new Factor with the same variables whose values sum to 1."""
        return Factor(self.variables, self.cardinalities, self.values / self.values.sum())

    def __repr__(self):
        return f"Factor({self.variables}, {self.cardinalities})"


@app.cell(hide_code=True)
def _():
    checks.run(1, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def joint_query(factors, query, evidence=None):
    """stage 2: P(query | evidence) by brute force: multiply everything, reduce, sum out the rest, normalize."""
    evidence = evidence or {}
    query = (query,) if isinstance(query, (str, int)) else tuple(query)
    joint = factors[0]
    for f in factors[1:]:
        joint = joint * f
    joint = joint.reduce(evidence)
    rest = [v for v in joint.variables if v not in query]
    if rest:
        joint = joint.marginalize(rest)
    order = [joint.variables.index(q) for q in query]
    return Factor(query, [joint.cardinalities[i] for i in order], np.transpose(joint.values, order)).normalize()


@app.cell(hide_code=True)
def _():
    checks.run(2, Factor=Factor, joint_query=joint_query)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def moralize(edges):
    """stage 3: the moral graph of a DAG: a set of frozenset({a, b}) undirected edges (drop directions, marry parents)."""
    edges = list(edges)
    parents = {}
    for a, b in edges:
        parents.setdefault(b, set()).add(a)
    und = {frozenset((a, b)) for a, b in edges}
    for ps in parents.values():
        ps = sorted(ps, key=str)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                und.add(frozenset((ps[i], ps[j])))
    return und


@app.function
def markov_blanket(edges, node):
    """stage 3: the Markov blanket of `node` in a DAG: its parents, children and children's other parents (a set)."""
    edges = list(edges)
    children = {b for a, b in edges if a == node}
    blanket = {a for a, b in edges if b == node} | children
    blanket |= {a for a, b in edges if b in children}
    blanket.discard(node)
    return blanket


@app.function
def separated(undirected_edges, x, y, given=()):
    """stage 3: True if every path between x and y (nodes or sets) in an undirected graph passes through `given`."""
    xs = set(x) if isinstance(x, (set, frozenset, list, tuple)) else {x}
    ys = set(y) if isinstance(y, (set, frozenset, list, tuple)) else {y}
    zs = set(given)
    adj = {}
    for e in undirected_edges:
        a, b = tuple(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen = set(xs)
    queue = deque(xs)
    while queue:
        u = queue.popleft()
        if u in ys:
            return False
        for v in adj.get(u, ()):
            if v not in seen and v not in zs:
                seen.add(v)
                queue.append(v)
    return True


@app.cell(hide_code=True)
def _():
    checks.run(3, moralize=moralize, markov_blanket=markov_blanket, separated=separated)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ancestors(edges, nodes):
    """stage 4: the set of `nodes` together with all their ancestors in the DAG."""
    parents = {}
    for a, b in edges:
        parents.setdefault(b, set()).add(a)
    out = set(nodes)
    stack = list(out)
    while stack:
        u = stack.pop()
        for p in parents.get(u, ()):
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


@app.function
def d_separated(edges, x, y, given=()):
    """stage 4: True if x and y (nodes or sets) are d-separated given `given` in the DAG `edges` ((parent, child) pairs)."""
    edges = list(edges)
    xs = set(x) if isinstance(x, (set, frozenset, list, tuple)) else {x}
    ys = set(y) if isinstance(y, (set, frozenset, list, tuple)) else {y}
    zs = set(given)
    if xs & ys or xs & zs or ys & zs:
        raise ValueError("x, y and given must be disjoint")
    keep = ancestors(edges, xs | ys | zs)
    sub = [(a, b) for a, b in edges if a in keep and b in keep]
    return separated(moralize(sub), xs, ys, zs)


@app.cell(hide_code=True)
def _():
    checks.run(4, ancestors=ancestors, d_separated=d_separated)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def factor_graph(factors):
    """stage 5: the factor graph as a dict variable -> list of indices of the factors that contain it."""
    graph = {}
    for i, f in enumerate(factors):
        for v in f.variables:
            graph.setdefault(v, []).append(i)
    return graph


@app.function
def is_tree(factors):
    """stage 5: True if the bipartite factor graph (variables + factors) has no cycle (a tree or a forest)."""
    graph = factor_graph(factors)
    n_nodes = len(graph) + len(factors)
    n_edges = sum(len(fs) for fs in graph.values())
    parent = list(range(n_nodes))  # union-find over factor nodes 0..F-1, then variables
    index = {v: len(factors) + k for k, v in enumerate(graph)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    components = n_nodes
    for v, fs in graph.items():
        for i in fs:
            a, b = find(i), find(index[v])
            if a == b:
                return False
            parent[a] = b
            components -= 1
    return n_edges == n_nodes - components


@app.cell(hide_code=True)
def _():
    checks.run(5, Factor=Factor, factor_graph=factor_graph, is_tree=is_tree)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `Factor` and `d_separated` are part of `mlfs.bayes` (try
    `from mlfs.bayes import Factor` in a scratch cell). then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
