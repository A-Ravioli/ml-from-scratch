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
    # equivariant networks · build (reference solution)

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
    src, dst = edge_index
    diff = pos[dst] - pos[src]
    return diff, (diff * diff).sum(dim=-1, keepdim=True)


@app.function
def radius_graph(pos, cutoff, batch=None):
    """stage 1: every ordered pair (j, i), i != j, in the same graph with |pos_i - pos_j| < cutoff; (2, E) int64, sorted."""
    n = pos.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.int64, device=pos.device)
    d2 = torch.cdist(pos, pos).pow(2) if n else pos.new_zeros(0, 0)
    keep = (d2 < cutoff ** 2) & (batch[:, None] == batch[None, :])
    keep.fill_diagonal_(False)
    src, dst = keep.nonzero(as_tuple=True)  # row-major: sorted by source, then target
    return torch.stack([src, dst]).to(torch.int64)


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
        super().__init__()
        H = hidden_dim
        self.edge_dim = edge_dim
        self.phi_e = nn.Sequential(nn.Linear(2 * H + 1 + edge_dim, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        last = nn.Linear(H, 1, bias=False)
        nn.init.xavier_uniform_(last.weight, gain=0.001)  # start with tiny position updates (as in the paper)
        self.phi_x = nn.Sequential(nn.Linear(H, H), nn.SiLU(), last)
        self.phi_h = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, H))

    def forward(self, h, pos, edge_index, edge_attr=None):
        """stage 2: (h (N, H), pos (N, 3)) -> (h', pos'); h' = h + phi_h(h, sum m_ij), pos' = pos + mean_j (x_i - x_j) phi_x(m_ij)."""
        # index_add is torch's scatter_sum: this topic is part of mlfs.gnn itself, so it can't import
        # mlfs.gnn (that would load this notebook again); topic 03's scatter is the same operation.
        src, dst = edge_index
        n = h.shape[0]
        diff, d2 = relative_geometry(pos, edge_index)
        feats = [h[dst], h[src], d2]
        if self.edge_dim:
            feats.append(edge_attr)
        m = self.phi_e(torch.cat(feats, dim=-1))  # (E, H)
        count = torch.zeros(n, dtype=pos.dtype).index_add(0, dst, torch.ones_like(d2[:, 0])).clamp(min=1)
        shift = torch.zeros_like(pos).index_add(0, dst, diff * self.phi_x(m)) / count[:, None]
        agg = torch.zeros(n, m.shape[1], dtype=m.dtype).index_add(0, dst, m)
        return h + self.phi_h(torch.cat([h, agg], dim=-1)), pos + shift


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
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim, edge_dim=edge_dim) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x, pos, edge_index, batch, num_graphs=None, edge_attr=None):
        """stage 3: -> (num_graphs, out_dim), invariant to rotations, reflections, translations and relabellings."""
        if num_graphs is None:
            num_graphs = int(batch.max()) + 1
        h = self.embed(x)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)
        node = self.head(h)
        return torch.zeros(num_graphs, node.shape[1], dtype=node.dtype, device=node.device).index_add(0, batch, node)


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
