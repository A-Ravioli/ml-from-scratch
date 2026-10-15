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
    # graphical models · build

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
        raise NotImplementedError("stage1: implement Factor.__init__")

    def _aligned(self, variables, cards):
        """this factor's values with one axis per entry of `variables` (size 1 where absent)."""
        order = [self.variables.index(v) for v in variables if v in self.variables]
        vals = np.transpose(self.values, order)
        shape = [cards[i] if v in self.variables else 1 for i, v in enumerate(variables)]
        return vals.reshape(shape)

    def __mul__(self, other):
        """stage 1: the product over the union of variables (self's order, then other's new ones)."""
        raise NotImplementedError("stage1: implement Factor.__mul__")

    def marginalize(self, variables):
        """stage 2: sum out the given variable name(s); returns a new Factor over the rest."""
        raise NotImplementedError("stage2: implement Factor.marginalize")

    def reduce(self, evidence):
        """stage 2: fix the variables in `evidence` (dict name -> value index) and drop them; others are ignored."""
        raise NotImplementedError("stage2: implement Factor.reduce")

    def normalize(self):
        """stage 2: a new Factor with the same variables whose values sum to 1."""
        raise NotImplementedError("stage2: implement Factor.normalize")

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
    raise NotImplementedError("stage2: implement joint_query")


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
    raise NotImplementedError("stage3: implement moralize")


@app.function
def markov_blanket(edges, node):
    """stage 3: the Markov blanket of `node` in a DAG: its parents, children and children's other parents (a set)."""
    raise NotImplementedError("stage3: implement markov_blanket")


@app.function
def separated(undirected_edges, x, y, given=()):
    """stage 3: True if every path between x and y (nodes or sets) in an undirected graph passes through `given`."""
    raise NotImplementedError("stage3: implement separated")


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
    raise NotImplementedError("stage4: implement ancestors")


@app.function
def d_separated(edges, x, y, given=()):
    """stage 4: True if x and y (nodes or sets) are d-separated given `given` in the DAG `edges` ((parent, child) pairs)."""
    raise NotImplementedError("stage4: implement d_separated")


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
    raise NotImplementedError("stage5: implement factor_graph")


@app.function
def is_tree(factors):
    """stage 5: True if the bipartite factor graph (variables + factors) has no cycle (a tree or a forest)."""
    raise NotImplementedError("stage5: implement is_tree")


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
