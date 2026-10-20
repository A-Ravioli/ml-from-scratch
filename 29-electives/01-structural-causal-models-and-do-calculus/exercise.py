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
    raise NotImplementedError("stage1: implement topological_order")


@app.function
def ancestors(parents, node):
    """stage 1: the set of nodes with a directed path into `node` (not including `node`)."""
    raise NotImplementedError("stage1: implement ancestors")


@app.function
def descendants(parents, node):
    """stage 1: the set of nodes reachable from `node` along directed edges (not including `node`)."""
    raise NotImplementedError("stage1: implement descendants")


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
        raise NotImplementedError("stage2: implement SCM.__init__")

    def sample(self, n, *, do=None, rng=None):
        """stage 2: n joint draws as a dict node -> float array (n,); `do` maps nodes to forced values."""
        raise NotImplementedError("stage2: implement SCM.sample")


@app.function
def intervene_graph(parents, nodes):
    """stage 2: a new parents dict with every edge INTO the given nodes removed (graph surgery)."""
    raise NotImplementedError("stage2: implement intervene_graph")


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
    raise NotImplementedError("stage3: implement d_separated")


@app.function
def is_backdoor_adjustment(parents, x, y, zs):
    """stage 3: True if zs satisfies the backdoor criterion for the effect of node x on node y."""
    raise NotImplementedError("stage3: implement is_backdoor_adjustment")


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
    raise NotImplementedError("stage4: implement backdoor_adjust")


@app.function
def frontdoor_adjust(data, x, y, m, x_value):
    """stage 4: E[y | do(x = x_value)] = sum_m P(m | x_value) sum_x' P(x') E[y | m, x'] for discrete x and m."""
    raise NotImplementedError("stage4: implement frontdoor_adjust")


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
