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

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    torch.set_num_threads(min(4, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)


@app.function
def sbm(n_per_class=100, k=3, p_in=0.06, p_out=0.01, signal=1.0, dim=16, seed=0):
    """a stochastic block model: k communities, edge probability p_in inside and p_out across; node features are a
    weak class signal (signal * one-hot in the first k dims) plus N(0, 1) noise. returns x, edge_index, y, and a
    split with 20 training nodes per class (the rest split evenly into validation and test)."""
    g = torch.Generator().manual_seed(seed)
    n = n_per_class * k
    y = torch.arange(k).repeat_interleave(n_per_class)
    same = y[:, None] == y[None, :]
    prob = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
    a = torch.rand(n, n, generator=g) < prob
    a = torch.triu(a, diagonal=1)
    a = a | a.T
    src, dst = a.nonzero(as_tuple=True)
    x = torch.randn(n, dim, generator=g)
    x[:, :k] += signal * F.one_hot(y, k)
    perm = torch.randperm(n, generator=g)
    train = torch.cat([perm[y[perm] == c][:20] for c in range(k)])
    rest = perm[~torch.isin(perm, train)]
    return x, torch.stack([src, dst]), y, train, rest[: len(rest) // 2], rest[len(rest) // 2:]


@app.class_definition
class Net(torch.nn.Module):
    """input linear -> depth x [conv -> ReLU] (optionally residual) -> output linear; conv in {gcn, sage, gat, mlp}."""

    def __init__(self, in_dim, n_classes, *, conv="gcn", depth=2, hidden=32, residual=False, seed=0, init="default"):
        super().__init__()
        torch.manual_seed(seed)
        self.inp = torch.nn.Linear(in_dim, hidden)
        make = {"gcn": lambda: ex.GCNConv(hidden, hidden), "sage": lambda: ex.SAGEConv(hidden, hidden),
                "gat": lambda: ex.GATConv(hidden, hidden // 4, heads=4), "mlp": lambda: torch.nn.Linear(hidden, hidden)}
        self.convs = torch.nn.ModuleList([make[conv]() for _ in range(depth)])
        self.out = torch.nn.Linear(hidden, n_classes)
        self.conv, self.residual = conv, residual
        if init == "kaiming":  # every weight matrix kaiming-normal for ReLU, every bias zero (ch 15 · topic 01)
            with torch.no_grad():
                for p in self.parameters():
                    if p.dim() == 2:
                        torch.nn.init.kaiming_normal_(p, nonlinearity="relu")
                    else:
                        p.zero_()

    def embed(self, x, edge_index):
        h = F.relu(self.inp(x))
        for c in self.convs:
            z = F.relu(c(h) if self.conv == "mlp" else c(h, edge_index))
            h = h + z if self.residual else z
        return h

    def forward(self, x, edge_index):
        return self.out(self.embed(x, edge_index))


@app.function
def fit_nodes(model, data, *, epochs=200, lr=0.01, with_train=False):
    """full-batch Adam on the training nodes; returns the test accuracy at the best validation accuracy."""
    x, ei, y, tr, va, te = data
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val, best_test = -1.0, 0.0
    for _ in range(epochs):
        model.train()
        loss = F.cross_entropy(model(x, ei)[tr], y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei).argmax(1)
        val = float((pred[va] == y[va]).float().mean())
        if val > best_val:
            best_val, best_test = val, float((pred[te] == y[te]).float().mean())
            best_train = float((pred[tr] == y[tr]).float().mean())
    return (best_test, best_train) if with_train else best_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # graph neural networks · lab

    every experiment here uses **your** scatter operations and **your** `GCNConv`, `SAGEConv` and
    `GATConv`. the graphs are stochastic block models: 300 nodes in 3 communities, edges
    mostly inside communities (6% chance) and rarely across (1%), and 16 node features that
    carry a weak hint of the community buried in noise. we train on 20 labelled nodes per
    community and test on the rest. for each experiment: write down a prediction, lock it in,
    run it, then explain what happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what does repeated averaging converge to?

    no learning yet: apply GCN's propagation $\hat A = D^{-1/2}(A + I)D^{-1/2}$ (built from
    your `add_self_loops`, `degree` and `scatter_sum`) to the node features $k$ times, for $k$
    up to 256. left: how similar the nodes' feature vectors become (mean cosine similarity
    over all pairs). right: after 256 steps, one feature of each node against $\sqrt{d_i}$,
    the square root of its degree (self-loop included).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "after many rounds of GCN averaging, do all nodes end up with the same feature vector? if not, what "
        "distinguishes them?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _x, _ei, _y, *_ = sbm(seed=0)
    _x = _x.double()
    _n = _x.shape[0]
    _src, _dst = ex.add_self_loops(_ei, _n)
    _deg = ex.degree(_dst, _n, torch.float64)
    _norm = _deg.pow(-0.5)[_src] * _deg.pow(-0.5)[_dst]

    def _propagate(h):
        return ex.scatter_sum(_norm[:, None] * h[_src], _dst, _n)

    _ks, _cos, _h = [], [], _x.clone()
    for _k in range(257):
        if _k in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256):
            _u = F.normalize(_h, dim=1)
            _ks.append(_k)
            _cos.append(float((_u @ _u.T).mean()))
        _h = _propagate(_h)
    _corr = float(np.corrcoef(_h[:, 0].numpy(), _deg.sqrt().numpy())[0, 1])
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    _a1.semilogx([max(k, 0.5) for k in _ks], _cos, "o-")
    _a1.set(xlabel="propagation steps k (0 plotted at 0.5)", ylabel="mean pairwise cosine", ylim=(-0.05, 1.05),
            title="nodes become indistinguishable")
    _a2.scatter(_deg.sqrt(), _h[:, 0], s=10, c=_y, cmap="viridis")
    _a2.set(xlabel=r"$\sqrt{\mathrm{degree}}$ (with self-loop)", ylabel="feature 0 after 256 steps",
            title=f"the limit (correlation {_corr:.4f})")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| k | " + " | ".join(map(str, _ks)) + " |\n|---|" + "---|" * len(_ks)
                           + "\n| mean cosine | " + " | ".join(f"{c:.3f}" for c in _cos) + " |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** $\hat A$ is symmetric with largest eigenvalue 1. what is the eigenvector?
    (try $D^{1/2}\mathbf 1$.) why does $\hat A^k X$ collapse onto it, and how fast? what's left
    of the community structure after 256 steps, and what does that mean for a 32-layer GCN?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how deep should a GCN be?

    two ways to go deeper. **left:** a linear classifier (logistic regression, no hidden
    layers) on the propagated features $\hat A^k X$, for $k = 0$ to 256: depth with no
    learnable weights in between, so no optimization trouble. **right:** trained networks of
    depth 1 to 32 (hidden width 32, Adam, 200 full-batch epochs, model selection on validation
    accuracy): GCNs and, as a control, MLPs of the same depth that ignore the graph, with
    PyTorch's default initialization, with kaiming initialization and zero biases (chapter 15's
    fix for deep ReLU networks), and with residual connections
    $h \leftarrow h + \operatorname{ReLU}(\operatorname{layer}(h))$. test accuracy (and training
    accuracy), mean of 3 seeds.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "left: at which k does the linear classifier on A^k X do best, and what does it reach at k = 256? right: at "
        "depth 32, which of the four networks still work? is the plain GCN's failure caused by oversmoothing?"
    )
    p2
    return (p2,)


@app.class_definition
class LinearOnFeatures(torch.nn.Module):
    """logistic regression on given node features; ignores edge_index."""

    def __init__(self, in_dim, n_classes, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(in_dim, n_classes)

    def forward(self, x, edge_index):
        return self.lin(x)


@app.cell
def _(p2):
    ui.gate(p2)
    _ks = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
    _sgc = np.zeros(len(_ks))
    for _s in range(3):
        _x, _ei, _y, _tr, _va, _te = sbm(seed=_s)
        _n = _x.shape[0]
        _src, _dst = ex.add_self_loops(_ei, _n)
        _deg = ex.degree(_dst, _n, torch.float32)
        _norm = _deg.pow(-0.5)[_src] * _deg.pow(-0.5)[_dst]
        _h, _k = _x.clone(), 0
        for _i, _target in enumerate(_ks):
            while _k < _target:
                _h = ex.scatter_sum(_norm[:, None] * _h[_src], _dst, _n)
                _k += 1
            _feat = (_h - _h.mean(0)) / (_h.std(0) + 1e-12)  # standardize each feature for the classifier
            _sgc[_i] += fit_nodes(LinearOnFeatures(16, 3, seed=_s), (_feat, _ei, _y, _tr, _va, _te)) / 3
    _depths = (1, 2, 4, 8, 16, 32)
    _res, _train = {}, {}
    for _name, _conv, _r, _init in (("GCN", "gcn", False, "default"), ("GCN, kaiming init", "gcn", False, "kaiming"),
                                    ("GCN + residual", "gcn", True, "default"), ("MLP", "mlp", False, "default"),
                                    ("MLP, kaiming init", "mlp", False, "kaiming")):
        _runs = [[fit_nodes(Net(16, 3, conv=_conv, depth=_d, residual=_r, seed=s, init=_init), sbm(seed=s), with_train=True)
                  for s in range(3)] for _d in _depths]
        _res[_name] = [np.mean([r[0] for r in rs]) for rs in _runs]
        _train[_name] = [np.mean([r[1] for r in rs]) for rs in _runs]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    _a1.semilogx([max(k, 0.5) for k in _ks], _sgc, "o-", color="C2")
    _a1.axhline(1 / 3, color="k", ls=":", lw=1, label="chance")
    _a1.set(xlabel="propagation steps k (0 plotted at 0.5)", ylabel="test accuracy", ylim=(0, 1),
            title=r"logistic regression on $\hat A^k X$")
    _a1.legend(fontsize=8)
    for (_name, _acc), _st in zip(_res.items(), ("o-", "o:", "o-", "s--", "s:")):
        _a2.semilogx(_depths, _acc, _st, base=2, label=_name)
    _a2.axhline(1 / 3, color="k", ls=":", lw=1, label="chance")
    _a2.set(xlabel="number of layers", ylabel="test accuracy", ylim=(0, 1), title="trained deep networks")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig,
               mo.md("| k | " + " | ".join(map(str, _ks)) + " |\n|---|" + "---|" * len(_ks) + "\n| linear on Â^k X | "
                     + " | ".join(f"{a:.3f}" for a in _sgc) + " |"),
               mo.md("| depth | " + " | ".join(map(str, _depths)) + " |\n|---|" + "---|" * len(_depths) + "\n"
                     + "\n".join(f"| {k} | " + " | ".join(f"{a:.3f} ({b:.2f})" for a, b in zip(v, _train[k])) + " |"
                                 for k, v in _res.items())
                     + "\n\ntest accuracy (training accuracy in brackets), mean of 3 seeds")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on the left, why does accuracy first rise and then fall, and what is the
    classifier left with at $k = 256$ (experiment 1)? on the right, compare the plain GCN with the
    plain MLP of the same depth, then the kaiming-initialized pair: at depth 8, how much of the
    collapse is something you met in chapter 15? at depth 16-32, is the MLP control still
    failing, and what does that say about the GCN? what does the residual connection fix, and is
    the residual GCN at depth 32 really averaging over 32 hops?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: when neighbours are not like you

    the same 2-layer models on two graphs with the same number of edges: the homophilous
    one above (6% inside, 1% across communities) and a **heterophilous** one (0.6% inside, 3.7%
    across, so about 9 in 10 of a node's neighbours belong to *other* communities). test
    accuracy, mean of 3 seeds.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "on the heterophilous graph, rank the MLP, GCN, GraphSAGE and GAT. does using the graph still help?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _graphs = {"homophilous": dict(p_in=0.06, p_out=0.01), "heterophilous": dict(p_in=0.006, p_out=0.037)}
    _rows = []
    for _gname, _kw in _graphs.items():
        _accs = []
        for _conv in ("mlp", "gcn", "sage", "gat"):
            _accs.append(np.mean([fit_nodes(Net(16, 3, conv=_conv, depth=2, seed=s), sbm(seed=s, **_kw))
                                  for s in range(3)]))
        _x, _ei, _y, *_ = sbm(seed=0, **_kw)
        _h = float((_y[_ei[0]] == _y[_ei[1]]).float().mean())
        _rows.append(f"| {_gname} | {_h:.2f} | " + " | ".join(f"{a:.3f}" for a in _accs) + " |")
    mo.md("| graph | edge homophily | MLP | GCN | GraphSAGE | GAT |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** GCN mixes a node's own features into the same average as its neighbours'.
    what does that average look like when most neighbours are from other communities? why
    can GraphSAGE still use the graph, and what does its separate weight $W_1$ for the node's
    own features buy? where does GAT land, and why doesn't attention rescue it?
    """)
    return


if __name__ == "__main__":
    app.run()
