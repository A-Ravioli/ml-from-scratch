import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # GNN expressivity · build

    spec-driven (L4): the spec boxes in the lesson define every function, the checks enforce
    them, and the internals are yours. a graph here is a pair `(edge_index, num_nodes)`, with
    `edge_index` a `(2, E)` int64 tensor or array (rows source, target; undirected graphs store
    both directions). you'll build the Weisfeiler–Leman test, the yardstick for what message
    passing can tell apart, and GIN, the message-passing network that reaches it.

    | stage | you build |
    |---|---|
    | 1 | `wl_colors`: 1-WL colour refinement on several graphs with one shared palette |
    | 2 | `wl_test`: are two graphs distinguishable by 1-WL? → `mlfs.gnn` |
    | 3 | `GINConv`: $\mathrm{MLP}((1+\epsilon)x_i + \sum_{j \to i} x_j)$ → `mlfs.gnn` |
    | 4 | `GIN`: a graph-level network with sum readouts from every layer |

    this topic is itself part of `mlfs.gnn`, so it can't import `mlfs.gnn` (that would load this
    notebook again): for sums over edges use `torch.Tensor.index_add`, the library version of
    the `scatter_sum` you built in topic 03.

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def wl_colors(graphs, n_iters=3):
    """stage 1: 1-WL on a list of (edge_index, num_nodes) with a shared palette -> per graph an int64 tensor (n_iters + 1, N)."""
    raise NotImplementedError("stage1: implement wl_colors")


@app.cell(hide_code=True)
def _():
    checks.run(1, wl_colors=wl_colors)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def wl_test(graph1, graph2, *, n_iters=3):
    """stage 2: False if 1-WL tells the graphs apart within n_iters rounds (certainly non-isomorphic), else True."""
    raise NotImplementedError("stage2: implement wl_test")


@app.cell(hide_code=True)
def _():
    checks.run(2, wl_test=wl_test, wl_colors=wl_colors)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class GINConv(nn.Module):
    """Xu et al. (2019): x_i' = MLP((1 + eps) x_i + sum_{j -> i} x_j), MLP = Linear -> ReLU -> Linear."""

    def __init__(self, in_dim, out_dim, *, eps=0.0, train_eps=False):
        """stage 3: the MLP in_dim -> out_dim -> out_dim; eps a learnable scalar Parameter if train_eps, else a fixed buffer."""
        raise NotImplementedError("stage3: implement GINConv.__init__")

    def forward(self, x, edge_index):
        """stage 3: x (N, in_dim) -> (N, out_dim); a sum (not a mean) over incoming edges."""
        # index_add is torch's scatter_sum: this topic is part of mlfs.gnn itself, so it can't import
        # mlfs.gnn (that would load this notebook again); topic 03's scatter is the same operation.
        raise NotImplementedError("stage3: implement GINConv.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, GINConv=GINConv)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class GIN(nn.Module):
    """n_layers GINConv layers (ReLU between); the graph embedding concatenates the sum readout of the input and of every layer."""

    def __init__(self, in_dim, hidden_dim, out_dim, *, n_layers=3):
        """stage 4: GINConv layers in_dim -> hidden_dim -> ... and a linear head from in_dim + n_layers * hidden_dim to out_dim."""
        raise NotImplementedError("stage4: implement GIN.__init__")

    def forward(self, x, edge_index, batch, num_graphs=None):
        """stage 4: (N, in_dim) node features of a batch of graphs -> (num_graphs, out_dim)."""
        raise NotImplementedError("stage4: implement GIN.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, GIN=GIN)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.gnn import wl_test, GINConv` gives you the test and the layer.
    """)
    return


if __name__ == "__main__":
    app.run()
