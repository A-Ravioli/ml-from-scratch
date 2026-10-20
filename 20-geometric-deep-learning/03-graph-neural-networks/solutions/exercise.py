import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # graph neural networks · build (reference solution)

    spec-driven (L4): the spec boxes in the lesson define every function, the checks enforce
    them, and the internals are yours. everything is plain PyTorch on the edge-list
    convention: `x` `(N, F)`, `edge_index` `(2, E)` int64 with rows (source, target), messages
    flowing source → target, undirected graphs storing both directions, and `batch` `(N,)`
    giving each node's graph id. no graph library: the scatter operations you write in
    stage 1 are the whole engine.

    | stage | you build |
    |---|---|
    | 1 | `scatter_sum`, `scatter_mean`, `scatter_max`: reduce edge messages into nodes → `mlfs.gnn` |
    | 2 | `add_self_loops`, `degree`, `GCNConv` → `mlfs.gnn` |
    | 3 | `SAGEConv`: a separate weight for yourself → `mlfs.gnn` |
    | 4 | `scatter_softmax`, `GATConv`: attention over incoming edges → `mlfs.gnn` |
    | 5 | `global_add_pool`, `global_mean_pool`, `batch_graphs`: whole-graph readouts → `mlfs.gnn` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def scatter_sum(src, index, dim_size):
    """stage 1: out[n] = sum of src[e] over edges e with index[e] == n; src (E, ...) -> (dim_size, ...); empty -> 0."""
    out = src.new_zeros((dim_size, *src.shape[1:]))
    return out.index_add(0, index, src)


@app.function
def scatter_mean(src, index, dim_size):
    """stage 1: the mean of the rows of src that go to each index; (dim_size, ...); empty groups -> 0."""
    total = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones_like(index, dtype=src.dtype), index, dim_size).clamp(min=1)
    return total / count.reshape(-1, *([1] * (src.dim() - 1)))


@app.function
def scatter_max(src, index, dim_size):
    """stage 1: the elementwise max of the rows of src that go to each index; (dim_size, ...); empty groups -> 0."""
    idx = index.reshape(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = src.new_full((dim_size, *src.shape[1:]), float("-inf"))
    out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    count = torch.bincount(index, minlength=dim_size).reshape(-1, *([1] * (src.dim() - 1)))
    return torch.where(count > 0, out, torch.zeros_like(out))


@app.cell(hide_code=True)
def _():
    checks.run(1, scatter_sum=scatter_sum, scatter_mean=scatter_mean, scatter_max=scatter_max)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def add_self_loops(edge_index, num_nodes):
    """stage 2: edge_index with an edge (i, i) appended for every node i: (2, E + num_nodes)."""
    loops = torch.arange(num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    return torch.cat([edge_index, torch.stack([loops, loops])], dim=1)


@app.function
def degree(index, num_nodes, dtype=torch.float32):
    """stage 2: how many times each node 0..num_nodes-1 occurs in index: (num_nodes,) of dtype."""
    return torch.bincount(index, minlength=num_nodes).to(dtype)


@app.class_definition
class GCNConv(nn.Module):
    """Kipf & Welling (2017): X' = D^-1/2 (A + I) D^-1/2 X W + b, as messages over edges."""

    def __init__(self, in_dim, out_dim):
        """stage 2: a weight (in_dim -> out_dim, no bias) and a bias added after aggregation."""
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, edge_index):
        """stage 2: x (N, in_dim) -> (N, out_dim): add self-loops, normalize each edge by 1/sqrt(deg_i deg_j), sum at targets."""
        n = x.shape[0]
        ei = add_self_loops(edge_index, n)
        src, dst = ei
        deg = degree(dst, n, x.dtype)
        dinv = deg.pow(-0.5)
        norm = dinv[src] * dinv[dst]
        h = self.lin(x)
        return scatter_sum(norm[:, None] * h[src], dst, n) + self.bias


@app.cell(hide_code=True)
def _():
    checks.run(2, add_self_loops=add_self_loops, degree=degree, GCNConv=GCNConv)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class SAGEConv(nn.Module):
    """Hamilton et al. (2017), mean aggregator: x_i' = W1 x_i + W2 mean_{j -> i} x_j + b."""

    def __init__(self, in_dim, out_dim):
        """stage 3: W1 (root, with the bias) and W2 (neighbours, no bias), both in_dim -> out_dim."""
        super().__init__()
        self.lin_root = nn.Linear(in_dim, out_dim)
        self.lin_nbr = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        """stage 3: x (N, in_dim) -> (N, out_dim); nodes without incoming edges get W1 x_i + b."""
        src, dst = edge_index
        agg = scatter_mean(x[src], dst, x.shape[0])
        return self.lin_root(x) + self.lin_nbr(agg)


@app.cell(hide_code=True)
def _():
    checks.run(3, SAGEConv=SAGEConv, GCNConv=GCNConv)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def scatter_softmax(src, index, dim_size):
    """stage 4: softmax of src (E, ...) within each group of equal index (numerically stable)."""
    gmax = scatter_max(src.detach(), index, dim_size)
    e = (src - gmax[index]).exp()
    return e / scatter_sum(e, index, dim_size)[index]


@app.class_definition
class GATConv(nn.Module):
    """Velickovic et al. (2018): attention-weighted mean over incoming edges, self-loops included, per head."""

    def __init__(self, in_dim, out_dim, *, heads=1, concat=True, dropout=0.0):
        """stage 4: lin (in_dim -> heads * out_dim, no bias), att_src and att_dst (heads, out_dim) parameters, bias."""
        super().__init__()
        self.heads, self.out_dim, self.concat, self.dropout = heads, out_dim, concat, dropout
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))

    def forward(self, x, edge_index):
        """stage 4: x (N, in_dim) -> (N, heads * out_dim) if concat else (N, out_dim) (mean over heads)."""
        n = x.shape[0]
        src, dst = add_self_loops(edge_index, n)
        h = self.lin(x).reshape(n, self.heads, self.out_dim)
        e = F.leaky_relu((h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst], 0.2)  # (E, heads)
        alpha = scatter_softmax(e, dst, n)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = scatter_sum(alpha[..., None] * h[src], dst, n)  # (N, heads, out_dim)
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias


@app.cell(hide_code=True)
def _():
    checks.run(4, scatter_softmax=scatter_softmax, GATConv=GATConv)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def global_add_pool(x, batch, num_graphs=None):
    """stage 5: sum the node features of each graph: x (N, F), batch (N,) -> (num_graphs, F); None -> batch.max() + 1."""
    if num_graphs is None:
        num_graphs = int(batch.max()) + 1 if batch.numel() else 0
    return scatter_sum(x, batch, num_graphs)


@app.function
def global_mean_pool(x, batch, num_graphs=None):
    """stage 5: the mean node feature of each graph: (num_graphs, F); empty graphs -> 0."""
    if num_graphs is None:
        num_graphs = int(batch.max()) + 1 if batch.numel() else 0
    return scatter_mean(x, batch, num_graphs)


@app.function
def batch_graphs(graphs):
    """stage 5: list of (x, edge_index) -> one disjoint big graph (x, edge_index, batch), node ids offset per graph."""
    xs, eis, bs, offset = [], [], [], 0
    for g, (x, ei) in enumerate(graphs):
        xs.append(x)
        eis.append(ei + offset)
        bs.append(torch.full((x.shape[0],), g, dtype=torch.int64))
        offset += x.shape[0]
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(bs)


@app.cell(hide_code=True)
def _():
    checks.run(5, global_add_pool=global_add_pool, global_mean_pool=global_mean_pool, batch_graphs=batch_graphs, GCNConv=GCNConv)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.gnn import GCNConv, SAGEConv, GATConv, scatter_sum, global_mean_pool` gives you
    a small, dependency-free graph library.
    """)
    return


if __name__ == "__main__":
    app.run()
