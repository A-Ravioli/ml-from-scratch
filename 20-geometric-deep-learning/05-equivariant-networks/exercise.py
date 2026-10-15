import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # equivariant networks · build

    spec-driven (L4): the spec boxes in the lesson define every function, the checks enforce
    them, and the internals are yours. a point cloud (a molecule) is node features `h`
    `(N, F)` plus positions `pos` `(N, 3)`; edges follow topic 03's convention (rows source,
    target; messages flow source → target), and `batch` `(N,)` holds graph ids. the checks
    apply random rotations, reflections, translations and relabellings, in float64, and
    demand exact equivariance.

    | stage | you build |
    |---|---|
    | 1 | `relative_geometry`, `radius_graph`: the invariant and equivariant raw material |
    | 2 | `EGNNLayer`: the E(n)-equivariant message-passing layer → `mlfs.gnn` |
    | 3 | `EGNN`: an invariant molecule-level model |

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
def relative_geometry(pos, edge_index):
    """stage 1: for each edge j -> i: diff = pos_i - pos_j (E, 3) and squared distance (E, 1)."""
    raise NotImplementedError("stage1: implement relative_geometry")


@app.function
def radius_graph(pos, cutoff, batch=None):
    """stage 1: every ordered pair (j, i), i != j, in the same graph with |pos_i - pos_j| < cutoff; (2, E) int64, sorted."""
    raise NotImplementedError("stage1: implement radius_graph")


@app.cell(hide_code=True)
def _():
    checks.run(1, relative_geometry=relative_geometry, radius_graph=radius_graph)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class EGNNLayer(nn.Module):
    """Satorras, Hoogeboom & Welling (2021): messages from invariants; positions move along relative vectors."""

    def __init__(self, hidden_dim, *, edge_dim=0):
        """stage 2: MLPs phi_e (2H + 1 + edge_dim -> H), phi_x (H -> 1), phi_h (2H -> H), SiLU activations."""
        raise NotImplementedError("stage2: implement EGNNLayer.__init__")

    def forward(self, h, pos, edge_index, edge_attr=None):
        """stage 2: (h (N, H), pos (N, 3)) -> (h', pos'); h' = h + phi_h(h, sum m_ij), pos' = pos + mean_j (x_i - x_j) phi_x(m_ij)."""
        # index_add is torch's scatter_sum: this topic is part of mlfs.gnn itself, so it can't import
        # mlfs.gnn (that would load this notebook again); topic 03's scatter is the same operation.
        raise NotImplementedError("stage2: implement EGNNLayer.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, EGNNLayer=EGNNLayer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class EGNN(nn.Module):
    """embedding -> n_layers EGNNLayers -> per-node MLP -> sum over each graph's nodes: an E(3)-invariant prediction."""

    def __init__(self, in_dim, hidden_dim, out_dim, *, n_layers=4, edge_dim=0):
        """stage 3: Linear(in_dim, hidden_dim), n_layers EGNNLayer(hidden_dim, edge_dim=edge_dim), an MLP head to out_dim."""
        raise NotImplementedError("stage3: implement EGNN.__init__")

    def forward(self, x, pos, edge_index, batch, num_graphs=None, edge_attr=None):
        """stage 3: -> (num_graphs, out_dim), invariant to rotations, reflections, translations and relabellings."""
        raise NotImplementedError("stage3: implement EGNN.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, EGNN=EGNN, radius_graph=radius_graph)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.gnn import EGNNLayer` gives you the layer; the capstone stacks it into a
    molecular energy model.
    """)
    return


if __name__ == "__main__":
    app.run()
