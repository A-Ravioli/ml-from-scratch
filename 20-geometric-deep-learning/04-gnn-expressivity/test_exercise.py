"""Checks for exercise.py (GNN expressivity).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. colour refinement is
checked against its definition (a partition oracle, refinement, a shared palette across graphs)
and on the classic pairs (C6 vs two triangles, decalin vs bicyclopentyl, path vs star, a pair that
needs two rounds); WL runs under `mlfs.testing.bounded`, so a refinement loop that never stops
fails instead of hanging. GINConv is checked through properties that pin down the formula without
looking inside (isolated-node identities, sums versus means, eps), and GIN for exact invariance and
for giving identical outputs exactly when 1-WL can't tell graphs apart.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlfs.testing import bounded  # noqa: E402

D = torch.float64


def _undirected(pairs, n):
    e = torch.tensor(pairs, dtype=torch.int64).T
    return (torch.cat([e, e.flip(0)], dim=1), n)


def _cycle(n, offset=0):
    return [(offset + i, offset + (i + 1) % n) for i in range(n)]


C6 = _undirected(_cycle(6), 6)
TWO_C3 = _undirected(_cycle(3) + _cycle(3, 3), 6)
P4 = _undirected([(0, 1), (1, 2), (2, 3)], 4)
STAR = _undirected([(0, 1), (0, 2), (0, 3)], 4)
P6 = _undirected([(i, i + 1) for i in range(5)], 6)
C4_P2 = _undirected(_cycle(4) + [(4, 5)], 6)  # same degree sequence as P6
DECALIN = _undirected(_cycle(6) + [(5, 6), (6, 7), (7, 8), (8, 9), (9, 0)], 10)
BICYCLOPENTYL = _undirected(_cycle(5) + _cycle(5, 5) + [(0, 5)], 10)


def _random_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.triu(torch.rand(n, n, generator=g) < p, diagonal=1)
    a = a | a.T
    return (torch.stack(a.nonzero(as_tuple=True)).to(torch.int64), n)


def _relabel(graph, seed):
    ei, n = graph
    p = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 11))
    inv = torch.empty_like(p)
    inv[p] = torch.arange(n)
    return (inv[ei], n), p


def _colors(nb, graphs, n_iters=3):
    return bounded(nb.wl_colors, graphs, n_iters, what="wl_colors",
                   hint="refine exactly n_iters times; don't loop 'until stable' without a cap.")


def _partition(c):
    """the partition of node indices induced by a colour vector, as a set of frozensets."""
    groups = {}
    for i, v in enumerate(c.tolist()):
        groups.setdefault(v, set()).add(i)
    return {frozenset(s) for s in groups.values()}


# ---------------------------------------------------------------- stage 1: colour refinement


def test_stage1_shapes_and_path(nb):
    (c,) = _colors(nb, [P4], 2)
    assert isinstance(c, torch.Tensor) and c.dtype == torch.int64 and tuple(c.shape) == (3, 4), (
        "wl_colors returns, per graph, an int64 tensor of shape (n_iters + 1, num_nodes): row t is the colouring "
        f"after t rounds; got {type(c).__name__} {getattr(c, 'dtype', None)} {tuple(getattr(c, 'shape', ()))}")
    assert len(set(c[0].tolist())) == 1, "round 0: every node has the same colour (no node labels)"
    want = {frozenset({0, 3}), frozenset({1, 2})}
    assert _partition(c[1]) == want, f"after one round the path 0-1-2-3 splits into ends and middles; got {c[1].tolist()}"
    assert _partition(c[2]) == want, "a second round doesn't split it further (both middles see an end and a middle)"


def test_stage1_matches_the_definition(nb):
    graph = _random_graph(12, 0.25, seed=3)
    (c,) = _colors(nb, [graph], 3)
    ei = graph[0]
    for t in range(3):
        prev = c[t].tolist()
        sig = [(prev[i], tuple(sorted(prev[s] for s, d in ei.T.tolist() if d == i))) for i in range(12)]
        oracle = {}
        for i, s in enumerate(sig):
            oracle.setdefault(s, set()).add(i)
        assert _partition(c[t + 1]) == {frozenset(v) for v in oracle.values()}, (
            f"round {t + 1}: two nodes must get the same new colour exactly when they had the same colour AND the same "
            "multiset of neighbour colours in round t (the multiset: sorted, with repeats)")


def test_stage1_shared_palette_and_inputs(nb):
    graph = _random_graph(10, 0.3, seed=5)
    other, p = _relabel(graph, seed=1)
    ca, cb = _colors(nb, [graph, other], 3)
    assert torch.equal(cb, ca[:, p]), (
        "colours come from one palette shared by all graphs: node i of the relabelled graph is node p[i] of the "
        "original, so it must get exactly the same colour id in every round")
    path, star = _colors(nb, [P4, STAR], 1)
    assert int(path[1, 0]) == int(star[1, 1]), (
        "a path's end and a star's leaf have the same round-1 signature (degree 1), so they must get the same colour id: "
        "number the signatures of ALL graphs together, not each graph separately")
    assert int(path[1, 1]) != int(star[1, 0]), (
        "a path's middle (degree 2) and a star's centre (degree 3) have different signatures, so different colours, "
        "even though they're in different graphs")
    arr = (graph[0].numpy(), np.int64(10))
    (cn,) = _colors(nb, [arr], 3)
    assert torch.equal(cn, ca), "edge_index may be a numpy array, num_nodes a numpy integer"


# ---------------------------------------------------------------- stage 2: the WL test


def test_stage2_isomorphic_graphs_pass(nb):
    for seed in range(3):
        graph = _random_graph(11, 0.3, seed=seed)
        other, _ = _relabel(graph, seed)
        assert bounded(nb.wl_test, graph, other, what="wl_test") is True, (
            "a relabelled copy of a graph is isomorphic to it, so 1-WL can never tell them apart: True")


def test_stage2_classic_pairs(nb):
    def wl(a, b, k=3):
        return bounded(nb.wl_test, a, b, n_iters=k, what="wl_test")

    assert wl(C6, TWO_C3) is True, (
        "a 6-cycle and two triangles are both 2-regular on 6 nodes: every node sees two neighbours of its own colour in "
        "every round, so 1-WL can't tell them apart (True), although they're not isomorphic")
    assert wl(DECALIN, BICYCLOPENTYL) is True, "decalin and bicyclopentyl are the textbook 1-WL failure: True"
    assert wl(P4, STAR) is False, "a path and a star on 4 nodes differ in their degrees: False after one round"
    assert wl(P6, (P6[0], 7)) is False, "different numbers of nodes: False"
    assert wl(P6, C4_P2, 1) is True, "same degree sequence: one round can't tell P6 from C4 + an edge"
    assert wl(P6, C4_P2, 2) is False, (
        "after two rounds it can: in P6 the degree-1 nodes have degree-2 neighbours, in C4 + edge they're adjacent "
        "to each other. compare the colour histograms of every round, up to n_iters")


# ---------------------------------------------------------------- stage 3: GINConv


def _layer(nb, seed=0, **kw):
    torch.manual_seed(seed)
    return nb.GINConv(3, 32, **kw).double()  # wide enough that a random init never has every ReLU dead


def test_stage3_formula_via_isolated_nodes(nb):
    layer = _layer(nb)
    g = torch.Generator().manual_seed(0)
    s, y = torch.randn(1, 3, generator=g, dtype=D), torch.randn(1, 3, generator=g, dtype=D)
    # node 0 isolated with features s; node 1 with features y receives from nodes 2 and 3 whose features sum to s - y
    x = torch.cat([s, y, 0.3 * (s - y), 0.7 * (s - y)])
    ei = torch.tensor([[2, 3], [1, 1]])
    out = layer(x, ei)
    assert tuple(out.shape) == (4, 32), f"GINConv(3, 32) maps (N, 3) to (N, 32); got {tuple(out.shape)}"
    assert torch.allclose(out[0], out[1], atol=1e-12), (
        "with eps = 0, an isolated node with features s and a node with features y whose in-neighbours sum to s - y "
        "both feed s into the MLP, so their outputs must be equal: x_i + SUM over incoming edges (not a mean, no "
        "normalization, messages from edge_index[0] to edge_index[1])")
    assert not torch.allclose(out[2], out[3], atol=1e-6), "different inputs, different outputs"


def test_stage3_sum_sees_multiplicity(nb):
    layer = _layer(nb, 1)
    x = torch.tensor([[0.0, 0, 0], [0, 0, 0], [1, -2, 0.5], [1, -2, 0.5], [1, -2, 0.5]], dtype=D)
    ei = torch.tensor([[2, 3, 4], [0, 1, 1]])  # node 0 hears one copy, node 1 two copies of the same message
    out = layer(x, ei)
    assert not torch.allclose(out[0], out[1], atol=1e-6), (
        "node 1 receives two identical messages and node 0 one; a sum aggregation must tell them apart "
        "(a mean or max can't, which is the point of GIN)")


def test_stage3_eps(nb):
    a = _layer(nb, 2, eps=1.0)
    b = _layer(nb, 2, eps=0.0)
    x = torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(2))
    empty = torch.zeros(2, 0, dtype=torch.int64)
    assert torch.allclose(a(x, empty), b(2 * x, empty), atol=1e-12), (
        "with the same initial weights, eps = 1 on x must equal eps = 0 on 2x for isolated nodes: (1 + eps) x_i")
    fixed = sum(p.numel() for p in _layer(nb, 3).parameters())
    learn = _layer(nb, 3, train_eps=True)
    n_learn = sum(p.numel() for p in learn.parameters())
    assert n_learn == fixed + 1, (
        f"train_eps=True adds exactly one learnable scalar (eps); parameter counts {fixed} vs {n_learn}. with "
        "train_eps=False eps must not be a Parameter (register it as a buffer or keep a float)")
    ei = torch.tensor([[0, 1], [1, 2]])
    learn(x, ei).sum().backward()
    grads = [p.grad for p in learn.parameters() if p.numel() == 1]
    assert grads and grads[0] is not None and float(grads[0].abs()) > 0, "eps receives a gradient when learnable"


def test_stage3_mlp_has_a_hidden_layer(nb):
    layer = _layer(nb, 5)
    x = torch.randn(30, 3, dtype=D, generator=torch.Generator().manual_seed(5)) * 3
    out = layer(x, torch.zeros(2, 0, dtype=torch.int64))
    assert float(out.min()) < 0 < float(out.max()), (
        "GINConv's MLP is Linear -> ReLU -> Linear: its outputs can be negative. all outputs >= 0 means it ends in a ReLU "
        "(a Linear -> ReLU 'MLP' has no hidden layer and can't be injective on multisets)")
    n = sum(p.numel() for p in layer.parameters())
    assert n == 3 * 32 + 32 + 32 * 32 + 32, f"MLP Linear(3, 32) -> ReLU -> Linear(32, 32): 1,184 parameters; got {n}"


def test_stage3_equivariance(nb):
    layer = _layer(nb, 4)
    graph = _random_graph(9, 0.35, seed=4)
    x = torch.randn(9, 3, dtype=D, generator=torch.Generator().manual_seed(4))
    out = layer(x, graph[0])
    (ei2, _), p = _relabel(graph, 4)
    err = float((layer(x[p], ei2) - out[p]).detach().abs().max())
    assert err < 1e-12, f"GINConv must be permutation-equivariant; error {err:.2e}"


# ---------------------------------------------------------------- stage 4: GIN


def _embed(nb, model, graphs, dim=2):
    xs, eis, bs, off = [], [], [], 0
    for g, (ei, n) in enumerate(graphs):
        xs.append(torch.ones(n, dim, dtype=D))
        eis.append(ei + off)
        bs.append(torch.full((n,), g, dtype=torch.int64))
        off += n
    with torch.no_grad():
        return model(torch.cat(xs), torch.cat(eis, dim=1), torch.cat(bs))


def test_stage4_shapes_and_invariance(nb):
    torch.manual_seed(0)
    model = nb.GIN(2, 16, 4, n_layers=3).double()
    graph = _random_graph(10, 0.3, seed=6)
    other, p = _relabel(graph, 6)
    g = torch.Generator().manual_seed(6)
    x = torch.randn(10, 2, generator=g, dtype=D)
    batch = torch.zeros(10, dtype=torch.int64)
    out = model(x, graph[0], batch)
    assert tuple(out.shape) == (1, 4), f"GIN(2, 16, 4) gives (num_graphs, 4); got {tuple(out.shape)}"
    err = float((model(x[p], other[0], batch) - out).detach().abs().max())
    assert err < 1e-10, f"a graph-level GIN is permutation-invariant; error {err:.2e}"
    both = _embed(nb, model, [P4, STAR])
    alone = _embed(nb, model, [STAR])
    assert tuple(both.shape) == (2, 4) and torch.allclose(both[1], alone[0], atol=1e-12), (
        "graphs in a batch are pooled separately (scatter the readout by `batch`)")


class _Const(torch.nn.Module):
    """a stub layer: every node gets the constant vector `value`."""

    def __init__(self, value, dim):
        super().__init__()
        self.value, self.dim = value, dim

    def forward(self, x, edge_index):
        return torch.full((x.shape[0], self.dim), self.value, dtype=x.dtype)


def test_stage4_readout_layout(nb):
    torch.manual_seed(2)
    model = nb.GIN(2, 4, 3, n_layers=3).double()
    assert isinstance(getattr(model, "convs", None), torch.nn.ModuleList) and len(model.convs) == 3, (
        "GIN keeps its layers in self.convs, an nn.ModuleList of n_layers GINConv layers (see the spec)")
    assert model.head.in_features == 2 + 3 * 4, (
        f"the head reads the input's sum readout and every layer's: in_dim + n_layers * hidden_dim = 14 inputs; got "
        f"{model.head.in_features}")
    model.convs = torch.nn.ModuleList([_Const(-1.0, 4), _Const(-2.0, 4), _Const(-3.0, 4)])
    seen = {}
    model.head.register_forward_pre_hook(lambda mod, inp: seen.setdefault("z", inp[0]))
    x = torch.full((5, 2), 0.5, dtype=D)
    model(x, torch.zeros(2, 0, dtype=torch.int64), torch.zeros(5, dtype=torch.int64))
    want = torch.tensor([[2.5, 2.5] + [0.0] * 4 + [0.0] * 4 + [-15.0] * 4], dtype=D)
    assert torch.allclose(seen["z"], want), (
        "with stub layers outputting -1, -2, -3 on a 5-node graph whose features are 0.5, the head's input should be "
        "the sum of x (2.5, 2.5), then the sums of ReLU(layer 1) and ReLU(layer 2) (zeros), then the sum of layer 3 "
        "(-15 each): a ReLU between layers but none after the last, and a sum readout of the input and of every layer, "
        f"in order. got {seen['z'].tolist()}")


def test_stage4_exactly_as_powerful_as_wl(nb):
    torch.manual_seed(1)
    model = nb.GIN(2, 16, 4, n_layers=3).double()
    e = _embed(nb, model, [C6, TWO_C3, DECALIN, BICYCLOPENTYL, P6, C4_P2])
    assert torch.allclose(e[0], e[1], atol=1e-10) and torch.allclose(e[2], e[3], atol=1e-10), (
        "pairs that 1-WL can't tell apart must get identical embeddings from ANY message-passing network with "
        "identical node features; yours differ, so something in it isn't message passing (node indices? random "
        "features?)")
    assert not torch.allclose(e[4], e[5], atol=1e-6), (
        "P6 and C4 + edge are told apart by 2 rounds of WL; a 3-layer GIN with sum aggregation and sum readouts "
        "should tell them apart too")
    c3, c6 = _embed(nb, model, [_undirected(_cycle(3), 3), C6])
    assert not torch.allclose(c3, c6, atol=1e-6), (
        "a triangle and a 6-cycle look the same locally, but a sum readout sees that one has twice as many nodes; a mean "
        "readout can't tell a graph from two copies of it")
    r3 = _undirected(_cycle(8) + [(i, i + 4) for i in range(4)], 8)  # 3-regular on 8 nodes
    r2 = _undirected(_cycle(8), 8)
    e2 = _embed(nb, model, [r2, r3])
    assert not torch.allclose(e2[0], e2[1], atol=1e-6), (
        "a 2-regular and a 3-regular graph on 8 nodes differ in every node's degree; sum aggregation sees it")
