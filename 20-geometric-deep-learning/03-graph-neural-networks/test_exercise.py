"""Checks for exercise.py (graph neural networks).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scatter operations
are checked against loops, for gradients (gradcheck) and for vectorization (against index_add on
the same input). layers are checked without looking inside them: each is probed (zero features
give the bias, isolated nodes give the weights) and its output on real graphs is compared with
the dense matrix formula in float64. every layer is also checked for exact permutation
equivariance under random relabellings of the nodes.
"""
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

D = torch.float64


def _random_graph(n=7, p=0.35, seed=0, undirected=True):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, n, generator=g) < p
    a.fill_diagonal_(False)
    if undirected:
        a = a | a.T
    src, dst = a.nonzero(as_tuple=True)
    return torch.stack([src, dst]).to(torch.int64)


def _dense(edge_index, n):
    """A[target, source] = number of edges source -> target."""
    a = torch.zeros(n, n, dtype=D)
    a.index_put_((edge_index[1], edge_index[0]), torch.ones(edge_index.shape[1], dtype=D), accumulate=True)
    return a


def _relabel(x, edge_index, seed=0):
    """new node i is old node p[i]."""
    n = x.shape[0]
    p = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 7))
    inv = torch.empty_like(p)
    inv[p] = torch.arange(n)
    return x[p], inv[edge_index], p


def _probe_linear(layer, in_dim):
    """bias (out,) and weight (in, out) of a layer that acts as x -> x W + b on isolated nodes."""
    empty = torch.zeros(2, 0, dtype=torch.int64)
    with torch.no_grad():
        b = layer(torch.zeros(1, in_dim, dtype=D), empty)[0]
        W = layer(torch.eye(in_dim, dtype=D), empty) - b
    return b, W


# ---------------------------------------------------------------- stage 1: scatter


def test_stage1_hand_example(nb):
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=D)
    index = torch.tensor([0, 2, 0])
    want = {
        "scatter_sum": [[6, 8], [0, 0], [3, 4], [0, 0]],
        "scatter_mean": [[3, 4], [0, 0], [3, 4], [0, 0]],
        "scatter_max": [[5, 6], [0, 0], [3, 4], [0, 0]],
    }
    for name, w in want.items():
        got = getattr(nb, name)(src, index, 4)
        assert tuple(got.shape) == (4, 2), f"{name}(src (3, 2), index, dim_size=4) has shape (4, 2); got {tuple(got.shape)}"
        assert torch.allclose(got, torch.tensor(w, dtype=D)), (
            f"{name}: rows 0 and 2 of src go to node 0, row 1 to node 2; nodes 1 and 3 receive nothing (-> 0). "
            f"expected {w}, got {got.tolist()}")


def test_stage1_random_against_loops(nb):
    g = torch.Generator().manual_seed(1)
    src = torch.randn(40, 3, generator=g, dtype=D) - 1.0  # mostly negative: a max that starts from 0 is wrong
    index = torch.randint(0, 9, (40,), generator=g)
    index[index == 4] = 5  # node 4 receives nothing
    s, m, mx = nb.scatter_sum(src, index, 10), nb.scatter_mean(src, index, 10), nb.scatter_max(src, index, 10)
    for n in range(10):
        rows = src[index == n]
        if len(rows) == 0:
            for name, t in (("sum", s), ("mean", m), ("max", mx)):
                assert torch.equal(t[n], torch.zeros(3, dtype=D)), f"empty groups give 0 ({name}); got {t[n].tolist()}"
            continue
        assert torch.allclose(s[n], rows.sum(0)) and torch.allclose(m[n], rows.mean(0))
        assert torch.allclose(mx[n], rows.max(0).values), (
            "scatter_max must return the largest member of each group, even when every member is negative "
            "(start from -inf, not from 0)")
    v = nb.scatter_sum(src[:, 0], index, 10)
    assert tuple(v.shape) == (10,) and torch.allclose(v, s[:, 0]), "1-D src (E,) gives (dim_size,)"


def test_stage1_gradients(nb):
    g = torch.Generator().manual_seed(2)
    src = torch.randn(12, 2, generator=g, dtype=D, requires_grad=True)
    index = torch.tensor([0, 1, 1, 3, 0, 3, 3, 1, 0, 0, 1, 3])
    for name in ("scatter_sum", "scatter_mean", "scatter_max"):
        fn = getattr(nb, name)
        assert torch.autograd.gradcheck(lambda t: fn(t, index, 5), (src,)), f"{name} must be differentiable"
    src0 = src.detach().clone()
    nb.scatter_sum(src, index, 5)
    assert torch.equal(src.detach(), src0), "scatter_* mustn't modify src"


def test_stage1_vectorized(nb):
    g = torch.Generator().manual_seed(3)
    src = torch.randn(100_000, 8, generator=g)
    index = torch.randint(0, 5000, (100_000,), generator=g)

    def best(fn):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    t_ref = best(lambda: torch.zeros(5000, 8).index_add_(0, index, src))
    for name in ("scatter_sum", "scatter_mean", "scatter_max"):
        fn = getattr(nb, name)
        t = best(lambda: fn(src, index, 5000))
        assert t <= 30 * t_ref + 0.02, (
            f"{name} took {t * 1e3:.1f} ms on 100,000 edges, against {t_ref * 1e3:.2f} ms for index_add_: "
            "a Python loop over edges or nodes? use index_add / scatter_reduce / bincount")


# ---------------------------------------------------------------- stage 2: GCN


def test_stage2_self_loops_and_degree(nb):
    ei = torch.tensor([[0, 1, 1], [1, 0, 2]])
    out = nb.add_self_loops(ei, 4)
    assert out.dtype == torch.int64 and tuple(out.shape) == (2, 7), f"(2, E + N) int64; got {tuple(out.shape)} {out.dtype}"
    assert torch.equal(out[:, :3], ei) and torch.equal(out[:, 3:], torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])), (
        "keep the original edges, then append (i, i) for every node, including isolated node 3")
    d = nb.degree(torch.tensor([1, 0, 2, 2]), 4)
    assert torch.equal(d.to(D), torch.tensor([1.0, 1.0, 2.0, 0.0], dtype=D)), f"degree counts occurrences; got {d}"


def test_stage2_gcn_hand_example_path(nb):
    torch.manual_seed(0)
    layer = nb.GCNConv(1, 1).double()
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # the path 0 - 1 - 2
    b, W = _probe_linear(layer, 1)
    with torch.no_grad():
        cols = [(layer(torch.eye(3, dtype=D)[:, j:j + 1], ei)[:, 0] - b[0]) / W[0, 0] for j in range(3)]
    A_hat = torch.stack(cols, dim=1)
    s6 = 1 / np.sqrt(6)
    want = torch.tensor([[1 / 2, s6, 0], [s6, 1 / 3, s6], [0, s6, 1 / 2]], dtype=D)
    assert torch.allclose(A_hat, want, atol=1e-12), (
        "on the path 0 - 1 - 2 (degrees with self-loops 2, 3, 2) GCN's propagation matrix is "
        f"D^-1/2 (A + I) D^-1/2 =\n{want.numpy().round(4)}\nyours acts like\n{A_hat.numpy().round(4)}\n"
        "check: self-loops added, degree counted with them, both endpoints normalized by 1/sqrt(deg)")


def test_stage2_gcn_matches_dense_formula(nb):
    for seed, undirected in ((0, True), (1, False)):
        torch.manual_seed(seed)
        layer = nb.GCNConv(3, 4).double()
        b, W = _probe_linear(layer, 3)
        n = 8
        ei = _random_graph(n, seed=seed, undirected=undirected)
        x = torch.randn(n, 3, dtype=D, generator=torch.Generator().manual_seed(seed))
        A = _dense(ei, n) + torch.eye(n, dtype=D)
        dinv = A.sum(1).pow(-0.5)
        want = dinv[:, None] * A * dinv[None, :] @ x @ W + b
        got = layer(x, ei)
        assert tuple(got.shape) == (n, 4)
        assert torch.allclose(got, want, atol=1e-10), (
            f"GCNConv should equal D^-1/2 (A + I) D^-1/2 X W + b, with A[i, j] = 1 for an edge j -> i and degrees "
            f"counted at the target ({'undirected' if undirected else 'directed'} graph). max error "
            f"{float((got - want).abs().max()):.2e}. messages flow from edge_index[0] to edge_index[1]")


def test_stage2_gcn_has_a_bias(nb):
    layer = nb.GCNConv(3, 4)
    n = sum(p.numel() for p in layer.parameters())
    assert n == 3 * 4 + 4, f"GCNConv(3, 4) has a 3 x 4 weight and a bias of 4: 16 parameters; got {n}"
    layer = layer.double()
    with torch.no_grad():
        for p in layer.parameters():
            p.fill_(0.5)
    bias, _ = _probe_linear(layer, 3)
    assert torch.allclose(bias, torch.full((4,), 0.5, dtype=D)), (
        f"with every parameter set to 0.5, a node with zero features should output the bias, 0.5 in every channel; "
        f"got {bias.tolist()}. add the bias after aggregating (it starts at zero, so the dense check can't see it)")


def test_stage2_gcn_equivariance(nb):
    torch.manual_seed(1)
    layer = nb.GCNConv(3, 5).double()
    ei = _random_graph(9, seed=4)
    x = torch.randn(9, 3, dtype=D)
    out = layer(x, ei)
    xp, eip, p = _relabel(x, ei, seed=1)
    err = float((layer(xp, eip) - out[p]).detach().abs().max())
    assert err < 1e-12, f"relabelling the nodes must relabel the outputs the same way; error {err:.2e}"


# ---------------------------------------------------------------- stage 3: SAGE


def test_stage3_sage_matches_dense_formula(nb):
    torch.manual_seed(2)
    layer = nb.SAGEConv(3, 4).double()
    b, W1 = _probe_linear(layer, 3)
    n = 8
    ei = _random_graph(n, seed=5, undirected=False)
    x = torch.randn(n, 3, dtype=D, generator=torch.Generator().manual_seed(5))
    A = _dense(ei, n)
    mean_nbr = A @ x / A.sum(1, keepdim=True).clamp(min=1)
    # W2 from a star: node 0 receives from 1..3 whose features are e_1..e_3
    star = torch.tensor([[1, 2, 3], [0, 0, 0]])
    with torch.no_grad():
        W2 = torch.stack([layer(torch.cat([torch.zeros(1, 3, dtype=D), torch.eye(3, dtype=D)[[j, j, j]]]), star)[0] - b
                          for j in range(3)])
    want = x @ W1 + mean_nbr @ W2 + b
    got = layer(x, ei)
    assert tuple(got.shape) == (n, 4)
    assert torch.allclose(got, want, atol=1e-10), (
        "SAGEConv should be W1 x_i + W2 mean_{j -> i} x_j + b, with the mean over incoming edges only (no self-loop) "
        f"and 0 for nodes with no incoming edges; max error {float((got - want).abs().max()):.2e}")
    with torch.no_grad():
        both = layer(torch.cat([torch.zeros(1, 3, dtype=D), torch.eye(3, dtype=D)]), star)[0] - b
    assert torch.allclose(both, W2.mean(0), atol=1e-10), "the neighbour term is a mean, not a sum"


def test_stage3_sage_and_gcn_differ_where_they_should(nb):
    torch.manual_seed(3)
    layer = nb.SAGEConv(2, 3).double()
    ei = _random_graph(10, seed=6)
    x = torch.randn(10, 2, dtype=D)
    out = layer(x, ei)
    xp, eip, p = _relabel(x, ei, seed=2)
    err = float((layer(xp, eip) - out[p]).detach().abs().max())
    assert err < 1e-12, f"SAGEConv must be permutation-equivariant; error {err:.2e}"
    # a node's own features enter through their own weight: doubling the edges changes nothing (mean)
    doubled = torch.cat([ei, ei], dim=1)
    assert torch.allclose(layer(x, doubled), out, atol=1e-12), "duplicated edges don't change a mean"


# ---------------------------------------------------------------- stage 4: GAT


def test_stage4_scatter_softmax(nb):
    g = torch.Generator().manual_seed(4)
    src = torch.randn(15, 2, generator=g, dtype=D) * 3
    index = torch.randint(0, 4, (15,), generator=g)
    got = nb.scatter_softmax(src, index, 5)
    assert tuple(got.shape) == (15, 2)
    for n in range(5):
        rows = index == n
        if rows.any():
            assert torch.allclose(got[rows], torch.softmax(src[rows], dim=0)), "softmax within each group of equal index"
    big = nb.scatter_softmax(src + 1000.0, index, 5)
    assert torch.isfinite(big).all() and torch.allclose(big, got), (
        "adding 1000 to every score must not change the softmax (or overflow): subtract each group's max first")
    s = torch.randn(8, generator=g, dtype=D, requires_grad=True)
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 0, 2])
    assert torch.autograd.gradcheck(lambda t: nb.scatter_softmax(t, idx, 3), (s,))


def _gat_dense(layer, x, ei, heads, out_dim, concat):
    """the GAT formula, computed densely from the layer's lin, att_src, att_dst and bias."""
    n = x.shape[0]
    A = _dense(ei, n) + torch.eye(n, dtype=D) > 0  # A[i, j]: edge j -> i, self-loops included
    h = (x @ layer.lin.weight.T).reshape(n, heads, out_dim)
    s = (h * layer.att_src).sum(-1)  # (N, heads), the source term
    t = (h * layer.att_dst).sum(-1)  # the target term
    e = torch.nn.functional.leaky_relu(t[:, None, :] + s[None, :, :], 0.2)  # e[i, j, k]
    e = e.masked_fill(~A[:, :, None], float("-inf"))
    alpha = torch.softmax(e, dim=1)
    out = torch.einsum("ijk,jkd->ikd", alpha, h)
    out = out.reshape(n, -1) if concat else out.mean(1)
    return out + layer.bias


@pytest.mark.parametrize("heads, concat", [(1, True), (3, True), (3, False)])
def test_stage4_gat_matches_the_formula(nb, heads, concat):
    torch.manual_seed(8)
    layer = nb.GATConv(4, 5, heads=heads, concat=concat).double().eval()
    for name in ("lin", "att_src", "att_dst", "bias"):
        assert hasattr(layer, name), f"GATConv needs the parameters lin, att_src, att_dst and bias (the spec); no {name!r}"
    with torch.no_grad():
        layer.bias.normal_(generator=torch.Generator().manual_seed(1))  # a nonzero bias, so a missing one shows
        layer.att_src.mul_(4.0)
        layer.att_dst.mul_(4.0)  # large scores: plenty of negative ones, where the 0.2 slope matters
    ei = _random_graph(9, p=0.4, seed=9, undirected=False)
    x = torch.randn(9, 4, dtype=D, generator=torch.Generator().manual_seed(9))
    want = _gat_dense(layer, x, ei, heads, 5, concat)
    got = layer(x, ei)
    assert torch.allclose(got, want, atol=1e-10), (
        f"GATConv(heads={heads}, concat={concat}) should compute e_ij = LeakyReLU_0.2(att_dst . h_i + att_src . h_j) "
        "(target i, source j, h = lin(x) split into heads), a softmax over i's incoming edges plus its self-loop, the "
        "weighted sum of h_j, then concatenate the heads (concat=True) or AVERAGE them (concat=False), plus bias. "
        f"max difference from that formula {float((got - want).abs().max()):.2e}")


def test_stage4_gat_shapes_and_equivariance(nb):
    torch.manual_seed(5)
    ei = _random_graph(9, seed=7)
    x = torch.randn(9, 4, dtype=D)
    for heads, concat, width in ((1, True, 6), (3, True, 18), (3, False, 6)):
        layer = nb.GATConv(4, 6, heads=heads, concat=concat).double().eval()
        out = layer(x, ei)
        assert tuple(out.shape) == (9, width), (
            f"GATConv(4, 6, heads={heads}, concat={concat}) gives (N, {width}); got {tuple(out.shape)}")
        xp, eip, p = _relabel(x, ei, seed=heads)
        err = float((layer(xp, eip) - out[p]).detach().abs().max())
        assert err < 1e-12, f"GATConv must be permutation-equivariant; error {err:.2e}"


def test_stage4_gat_is_an_attention_weighted_mean(nb):
    torch.manual_seed(6)
    layer = nb.GATConv(10, 12, heads=1).double().eval()  # in_dim >= group size: the weights are identifiable
    with torch.no_grad():
        for p in layer.parameters():  # larger attention parameters make the weights clearly non-uniform
            p.mul_(3.0)
    b, W = _probe_linear(layer, 10)
    n = 8
    ei = _random_graph(n, p=0.4, seed=8)
    x = torch.randn(n, 10, dtype=D, generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        out = layer(x, ei) - b
    h = x @ W
    nonuniform = False
    for i in range(n):
        nbrs = torch.cat([ei[0, ei[1] == i], torch.tensor([i])])
        alpha = torch.linalg.lstsq(h[nbrs].T, out[i]).solution
        resid = float((h[nbrs].T @ alpha - out[i]).abs().max())
        assert resid < 1e-8, (
            f"node {i}'s output should be a weighted combination of W x_j over its in-neighbours and itself "
            f"(the self-loop); residual {resid:.2e}")
        assert float(alpha.min()) > -1e-8 and abs(float(alpha.sum()) - 1) < 1e-8, (
            f"the weights must be a softmax over node {i}'s incoming edges (non-negative, summing to 1); "
            f"got {alpha.numpy().round(4)}")
        nonuniform |= float(alpha.max() - alpha.min()) > 1e-3
    assert nonuniform, "every node weights its neighbours equally: the attention scores don't depend on the features"


def test_stage4_gat_dropout_and_modes(nb):
    torch.manual_seed(7)
    ei = _random_graph(12, p=0.5, seed=9)
    x = torch.randn(12, 4, dtype=D)
    layer = nb.GATConv(4, 3, dropout=0.5).double()
    layer.train()
    assert not torch.allclose(layer(x, ei), layer(x, ei)), "with dropout in training mode, attention weights are dropped at random"
    layer.eval()
    assert torch.equal(layer(x, ei), layer(x, ei)), "in eval mode the layer is deterministic"
    plain = nb.GATConv(4, 3).double().train()
    assert torch.equal(plain(x, ei), plain(x, ei)), "dropout=0.0 is deterministic even in training mode"


# ---------------------------------------------------------------- stage 5: pooling and batching


def test_stage5_pools(nb):
    x = torch.arange(12, dtype=D).reshape(6, 2)
    batch = torch.tensor([0, 0, 1, 1, 1, 3])
    s = nb.global_add_pool(x, batch)
    assert tuple(s.shape) == (4, 2), "num_graphs=None means batch.max() + 1 graphs"
    assert torch.equal(s, torch.tensor([[2, 4], [18, 21], [0, 0], [10, 11]], dtype=D))
    m = nb.global_mean_pool(x, batch, num_graphs=5)
    assert tuple(m.shape) == (5, 2) and torch.equal(m[2], torch.zeros(2, dtype=D)) and torch.equal(m[4], m[2]), (
        "graphs with no nodes pool to 0, and num_graphs fixes the number of rows")
    assert torch.allclose(m[1], torch.tensor([6.0, 7.0], dtype=D))


def test_stage5_batching(nb):
    g = torch.Generator().manual_seed(10)
    graphs = [(torch.randn(3, 2, generator=g, dtype=D), torch.tensor([[0, 1], [1, 2]])),
              (torch.randn(2, 2, generator=g, dtype=D), torch.zeros(2, 0, dtype=torch.int64)),
              (torch.randn(4, 2, generator=g, dtype=D), torch.tensor([[0, 3, 2], [3, 0, 1]]))]
    x, ei, batch = nb.batch_graphs(graphs)
    assert tuple(x.shape) == (9, 2) and torch.equal(x[3:5], graphs[1][0])
    assert torch.equal(batch, torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])) and batch.dtype == torch.int64
    assert torch.equal(ei, torch.tensor([[0, 1, 5, 8, 7], [1, 2, 8, 5, 6]])), (
        f"each graph's node ids are offset by the number of nodes before it (0, 3, 5); got\n{ei}")
    torch.manual_seed(0)
    conv = nb.GCNConv(2, 3).double() if hasattr(nb, "GCNConv") else None
    if conv is not None:
        whole = nb.global_mean_pool(conv(x, ei), batch)
        alone = conv(graphs[2][0], graphs[2][1]).mean(0)
        assert torch.allclose(whole[2], alone, atol=1e-12), "graphs in a batch don't interact"
