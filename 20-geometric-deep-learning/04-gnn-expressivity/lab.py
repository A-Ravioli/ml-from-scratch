import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import notebooks, ui
    from mlfs.gnn import GCNConv, global_add_pool, global_mean_pool  # yours, from topic 03

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    torch.set_num_threads(min(4, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)


@app.function
def undirected(pairs, n):
    e = torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2).T
    return torch.cat([e, e.flip(0)], dim=1), n


@app.function
def cycles(*sizes):
    """disjoint cycles of the given sizes as one graph."""
    pairs, off = [], 0
    for k in sizes:
        pairs += [(off + i, off + (i + 1) % k) for i in range(k)]
        off += k
    return undirected(pairs, off)


@app.function
def random_regular(n, d, rng):
    """a uniformly-ish random simple d-regular graph on n nodes (configuration model with restarts)."""
    for _ in range(1000):
        stubs = rng.permutation(np.repeat(np.arange(n), d)).reshape(-1, 2)
        if np.any(stubs[:, 0] == stubs[:, 1]):
            continue
        key = set(map(tuple, np.sort(stubs, axis=1)))
        if len(key) == len(stubs):
            return undirected(stubs.tolist(), n)
    raise RuntimeError("no simple regular graph found")


@app.function
def relabel(graph, rng):
    ei, n = graph
    p = torch.tensor(rng.permutation(n))
    inv = torch.empty_like(p)
    inv[p] = torch.arange(n)
    return inv[ei], n


@app.function
def count_triangles(graph):
    ei, n = graph
    a = torch.zeros(n, n)
    a[ei[0], ei[1]] = 1
    return int(torch.trace(a @ a @ a) / 6)


@app.function
def collate(graphs, feat_dim=1, random_features=0, generator=None):
    """one batched graph; node features = ones (feat_dim), plus `random_features` fresh N(0, 1) columns."""
    xs, eis, bs, off = [], [], [], 0
    for g, (ei, n) in enumerate(graphs):
        x = torch.ones(n, feat_dim)
        if random_features:
            x = torch.cat([x, torch.randn(n, random_features, generator=generator)], dim=1)
        xs.append(x)
        eis.append(ei + off)
        bs.append(torch.full((n,), g, dtype=torch.int64))
        off += n
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(bs)


@app.class_definition
class GCNGraph(torch.nn.Module):
    """3 GCN layers (ReLU) and a mean or sum readout, then a linear head."""

    def __init__(self, in_dim, hidden=32, out_dim=2, readout="mean", seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.convs = torch.nn.ModuleList([GCNConv(in_dim, hidden), GCNConv(hidden, hidden), GCNConv(hidden, hidden)])
        self.head = torch.nn.Linear(hidden, out_dim)
        self.pool = global_mean_pool if readout == "mean" else global_add_pool

    def forward(self, x, edge_index, batch):
        for c in self.convs:
            x = F.relu(c(x, edge_index))
        return self.head(self.pool(x, batch))


@app.function
def gin(in_dim, seed=0):
    torch.manual_seed(seed)
    return ex.GIN(in_dim, 32, 2, n_layers=3)


@app.function
def train_graph_classifier(model, train, test, *, epochs=150, random_features=0, seed=0):
    """full-batch Adam on (graphs, labels) lists; returns test accuracy (random features redrawn every pass)."""
    g = torch.Generator().manual_seed(seed)
    (gtr, ytr), (gte, yte) = train, test
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        x, ei, b = collate(gtr, random_features=random_features, generator=g)
        loss = F.cross_entropy(model(x, ei, b), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, ei, b = collate(gte, random_features=random_features, generator=g)
        return float((model(x, ei, b).argmax(1) == yte).float().mean())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # GNN expressivity · lab

    every experiment here uses **your** `wl_test` and `GIN` from this topic and your `GCNConv`
    and pools from topic 03. no graph here has node features: every node starts with the
    same feature vector, so whatever a network learns, it learns from the structure. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: pairs that look the same

    five pairs of non-isomorphic graphs. for each: does 1-WL tell them apart, how many triangles
    each has, and the largest difference between the two graph embeddings of a random GIN and
    of a random 3-layer GCN with mean readout (both untrained).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "a 6-cycle has no triangles; two separate triangles have two. can a 3-layer GIN, which is as powerful as "
        "1-WL, tell them apart? and a GCN?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _pairs = {
        "C6 vs 2 × C3": (cycles(6), cycles(3, 3)),
        "C12 vs 3 × C4": (cycles(12), cycles(4, 4, 4)),
        "decalin vs bicyclopentyl": (
            undirected([(i, (i + 1) % 6) for i in range(6)] + [(5, 6), (6, 7), (7, 8), (8, 9), (9, 0)], 10),
            undirected([(i, (i + 1) % 5) for i in range(5)] + [(5 + i, 5 + (i + 1) % 5) for i in range(5)]
                       + [(0, 5)], 10)),
        "3-regular vs 4-regular (12 nodes)": (random_regular(12, 3, np.random.default_rng(0)),
                                              random_regular(12, 4, np.random.default_rng(0))),
        "path P6 vs C4 + edge": (undirected([(i, i + 1) for i in range(5)], 6),
                                 undirected([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)], 6)),
    }
    _gin, _gcn = gin(1).double(), GCNGraph(1).double()
    _rows = []
    for _name, (_a, _b) in _pairs.items():
        with torch.no_grad():
            _x, _ei, _bt = collate([_a, _b])
            _dg = float((_gin(_x.double(), _ei, _bt)[0] - _gin(_x.double(), _ei, _bt)[1]).abs().max())
            _dc = float((_gcn(_x.double(), _ei, _bt)[0] - _gcn(_x.double(), _ei, _bt)[1]).abs().max())
        _same = ex.wl_test(_a, _b)
        _rows.append(f"| {_name} | {'indistinguishable' if _same else 'distinguished'} | {count_triangles(_a)} vs "
                     f"{count_triangles(_b)} | {_dg:.1e} | {_dc:.1e} |")
    mo.md("| pair | 1-WL | triangles | GIN embedding difference | GCN embedding difference |\n|---|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do GIN's embeddings agree *exactly* (to rounding) on exactly the pairs
    1-WL can't separate? why can't any message-passing network count triangles? one pair is
    separated by WL and by GIN but not by the GCN: which, and what does GCN's normalization do
    on a regular graph with identical node features?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: learning to tell regular graphs apart

    a classification task: random 3-regular versus random 4-regular graphs, all on 12 nodes
    (200 training graphs, 100 test graphs, balanced). we train a GCN with mean readout, a GCN
    with **sum** readout, and your GIN, all for 150 full-batch Adam epochs.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the two classes differ in every node's degree, the most basic graph statistic there is. what test accuracy "
        "does each model reach? does switching the GCN to a sum readout help?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _dataset(n_graphs, seed):
        _rng = np.random.default_rng(seed)
        _y = torch.tensor(_rng.integers(0, 2, n_graphs))
        return [random_regular(12, 3 + int(c), _rng) for c in _y], _y

    _train, _test = _dataset(200, 0), _dataset(100, 1)
    _res = {
        "GCN, mean readout": train_graph_classifier(GCNGraph(1, readout="mean"), _train, _test),
        "GCN, sum readout": train_graph_classifier(GCNGraph(1, readout="sum"), _train, _test),
        "GIN (sum aggregation, sum readouts)": train_graph_classifier(gin(1), _train, _test),
    }
    mo.md("| model | test accuracy |\n|---|---|\n" + "\n".join(f"| {k} | {v:.3f} |" for k, v in _res.items()))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compute one GCN layer by hand on a $d$-regular graph whose nodes all carry
    the same feature vector $x$. what does $\hat A x$ come out as, and why is the answer the
    same for $d = 3$ and $d = 4$? why does the sum readout not rescue it? what does GIN's sum
    aggregation compute instead?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: breaking the symmetry with random features

    a task beyond 1-WL: one long cycle $C_{2k}$ (label 0) versus two cycles $C_k + C_k$
    (label 1), for $k = 3, 4, 5$, randomly relabelled. every node looks the same to message
    passing. a trick from the literature: give every node a few **random** features, freshly
    drawn every forward pass, so nodes become distinguishable. we train GIN (3 layers, 600 epochs) with 0 and with
    8 random features, then feed the *same* test graph 20 times and measure how much the
    predicted probability varies.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with random node features, can GIN beat chance (50%) on one cycle vs two cycles? what's the price?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _dataset(n_graphs, seed):
        _rng = np.random.default_rng(seed)
        _gs, _ys = [], []
        for _ in range(n_graphs):
            _k, _lab = int(_rng.integers(3, 6)), int(_rng.integers(0, 2))
            _g = cycles(2 * _k) if _lab == 0 else cycles(_k, _k)
            _gs.append(relabel(_g, _rng))
            _ys.append(_lab)
        return _gs, torch.tensor(_ys)

    _train, _test = _dataset(300, 2), _dataset(200, 3)
    _rows = []
    for _r in (0, 8):
        _accs, _spread = [], []
        for _s in range(3):
            _m = gin(1 + _r, seed=_s)
            _accs.append(train_graph_classifier(_m, _train, _test, epochs=600, random_features=_r, seed=_s))
            _g = torch.Generator().manual_seed(100 + _s)
            with torch.no_grad():
                _ps = torch.stack([torch.softmax(_m(*collate(_test[0][:20], random_features=_r, generator=_g)), 1)[:, 1]
                                   for _ in range(20)])
            _spread.append(float(_ps.std(0).mean()))
        _rows.append(f"| {_r} | {np.mean(_accs):.3f} ({', '.join(f'{a:.2f}' for a in _accs)}) | {np.mean(_spread):.3f} |")
    mo.md("| random features | test accuracy (3 seeds) | std of p(label 1) for the same graph, 20 draws |\n|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the model with no random features sit exactly at chance? what do
    random features let a node do that it couldn't before (think: "is the node two hops away
    from me, across the cycle, the same node I see two hops the other way?")? the last column
    measures a broken promise: which property from topic 01 has the network lost, and why
    would you care in practice?
    """)
    return


if __name__ == "__main__":
    app.run()
