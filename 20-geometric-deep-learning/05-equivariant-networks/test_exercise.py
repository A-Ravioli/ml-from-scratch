"""Checks for exercise.py (equivariant networks).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. everything is checked
in float64 under random rotations, reflections (all of O(3)), translations and node relabellings,
drawn as in topic 01 (Haar-random via QR with the sign fix): invariant outputs must not change, and
positions must move exactly with the transformation, to ~1e-10 relative error. the radius graph is
compared with a brute-force loop; the layer also passes geometric consequences of equivariance
(collinear points stay collinear, a two-atom update points along the bond); the model gets one small
learning check, aggregated over seeds.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

D = torch.float64


def _orthogonal(seed, proper=None):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if proper is not None and (np.linalg.det(q) > 0) != proper:
        q[:, 0] = -q[:, 0]
    return torch.tensor(q, dtype=D)


def _cloud(n=7, seed=0, scale=1.5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=g, dtype=D) * scale


def _full(n, offset=0):
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = i != j
    return torch.stack([j[keep], i[keep]]) + offset  # rows source, target


def _relabel_edges(edge_index, p):
    inv = torch.empty_like(p)
    inv[p] = torch.arange(len(p))
    return inv[edge_index]


def _rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-300))


# ---------------------------------------------------------------- stage 1: geometry


def test_stage1_relative_geometry(nb):
    pos = torch.tensor([[0.0, 0, 0], [1, 2, 2], [3, 0, 0]], dtype=D)
    ei = torch.tensor([[0, 1], [1, 2]])  # 0 -> 1, 1 -> 2
    diff, d2 = nb.relative_geometry(pos, ei)
    assert tuple(diff.shape) == (2, 3) and tuple(d2.shape) == (2, 1), (
        f"diff is (E, 3) and the squared distance (E, 1); got {tuple(diff.shape)} and {tuple(d2.shape)}")
    assert torch.allclose(diff, torch.tensor([[1.0, 2, 2], [2, -2, -2]], dtype=D)), (
        "for an edge j -> i, diff = pos_i - pos_j (target minus source): edge 0 -> 1 gives (1, 2, 2)")
    assert torch.allclose(d2[:, 0], torch.tensor([9.0, 12.0], dtype=D)), "squared distances 9 and 12 (not 3 and sqrt 12)"


def test_stage1_radius_graph_brute_force(nb):
    pos = _cloud(9, seed=1)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    ei = nb.radius_graph(pos, 2.5, batch)
    assert ei.dtype == torch.int64 and ei.shape[0] == 2
    want = [(j, i) for j in range(9) for i in range(9)
            if i != j and batch[i] == batch[j] and float((pos[i] - pos[j]).norm()) < 2.5]
    got = list(map(tuple, ei.T.tolist()))
    assert sorted(got) == sorted(want), (
        "radius_graph must contain every ordered pair (j, i), i != j, of nodes in the same graph closer than the "
        f"cutoff, in both directions, and nothing else. missing {sorted(set(want) - set(got))[:5]}, "
        f"extra {sorted(set(got) - set(want))[:5]}")
    assert got == sorted(got), "edges sorted by source, then target"
    full = nb.radius_graph(pos[:4], float("inf"))
    assert full.shape[1] == 12, "cutoff = inf with batch=None gives the complete graph (4 * 3 directed edges)"


def test_stage1_invariance_of_the_graph(nb):
    pos = _cloud(10, seed=2)
    ei = nb.radius_graph(pos, 2.0)
    for s in range(3):
        moved = pos @ _orthogonal(s).T + torch.tensor([3.0, -1.0, 2.0], dtype=D)
        assert torch.equal(nb.radius_graph(moved, 2.0), ei), "rotating/reflecting and translating the cloud keeps the graph"
    p = torch.randperm(10, generator=torch.Generator().manual_seed(3))
    eip = nb.radius_graph(pos[p], 2.0)
    assert sorted(map(tuple, _relabel_edges(ei, p).T.tolist())) == sorted(map(tuple, eip.T.tolist())), (
        "relabelling the nodes relabels the edges")
    diff, d2 = nb.relative_geometry(pos, ei)
    R = _orthogonal(4)
    diff2, d22 = nb.relative_geometry(pos @ R.T + 5.0, ei)
    assert torch.allclose(diff2, diff @ R.T, atol=1e-12) and torch.allclose(d22, d2, atol=1e-12), (
        "relative vectors rotate with the cloud and ignore translations; squared distances don't change")


# ---------------------------------------------------------------- stage 2: EGNNLayer


def _layer(nb, seed=0, H=8, edge_dim=0):
    torch.manual_seed(seed)
    return nb.EGNNLayer(H, edge_dim=edge_dim).double()


def test_stage2_shapes_and_no_edges(nb):
    layer = _layer(nb)
    h, pos = torch.randn(6, 8, dtype=D), _cloud(6, seed=3)
    out_h, out_pos = layer(h, pos, _full(6))
    assert tuple(out_h.shape) == (6, 8) and tuple(out_pos.shape) == (6, 3), (
        f"EGNNLayer(8) maps (h (N, 8), pos (N, 3)) to the same shapes; got {tuple(out_h.shape)}, {tuple(out_pos.shape)}")
    assert float((out_pos - pos).abs().max()) > 1e-9, "positions should move (by a little, at initialization)"
    _, pos_dup = layer(h, pos, torch.cat([_full(6), _full(6)], dim=1))
    assert torch.allclose(pos_dup - pos, out_pos - pos, rtol=1e-9, atol=1e-15), (
        "listing every edge twice must not change the position update: it's the MEAN over incoming edges of "
        "(x_i - x_j) phi_x(m_ij), not the sum")
    empty = torch.zeros(2, 0, dtype=torch.int64)
    h2, pos2 = layer(h, pos, empty)
    assert torch.equal(pos2, pos), "a node with no incoming edges keeps its position (no 0/0)"
    assert torch.isfinite(h2).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stage2_exact_e3_equivariance(nb, seed):
    layer = _layer(nb, seed)
    n = 7
    h, pos, ei = torch.randn(n, 8, dtype=D, generator=torch.Generator().manual_seed(seed)), _cloud(n, seed), _full(n)
    out_h, out_pos = layer(h, pos, ei)
    shift = out_pos - pos
    for proper in (True, False):
        R = _orthogonal(seed + 10, proper=proper)
        t = torch.tensor([1.0, -2.0, 0.5], dtype=D)
        h2, pos2 = layer(h, pos @ R.T + t, ei)
        kind = "rotation" if proper else "reflection"
        assert _rel(h2, out_h) < 1e-10, (
            f"node features must be invariant to a {kind} and a translation of the positions; relative error "
            f"{_rel(h2, out_h):.1e}. only invariants (squared distances) may enter the messages")
        assert _rel(pos2 - (pos @ R.T + t), shift @ R.T) < 1e-9, (
            f"the position update must rotate with the input ({kind}) and ignore the translation; relative error "
            f"{_rel(pos2 - (pos @ R.T + t), shift @ R.T):.1e}. move each node along relative vectors x_i - x_j, "
            "weighted by invariant scalars")
    p = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    h3, pos3 = layer(h[p], pos[p], _relabel_edges(ei, p))
    assert _rel(h3, out_h[p]) < 1e-10 and _rel(pos3, out_pos[p]) < 1e-10, "relabelling nodes relabels the outputs"


def test_stage2_geometric_consequences(nb):
    layer = _layer(nb, 3)
    h = torch.randn(5, 8, dtype=D, generator=torch.Generator().manual_seed(5))
    line = torch.linspace(-2, 2, 5, dtype=D)[:, None] * torch.tensor([[1.0, 2.0, -0.5]], dtype=D)
    _, out = layer(h, line, _full(5))
    shift = out - line
    direction = torch.tensor([1.0, 2.0, -0.5], dtype=D) / torch.tensor([1.0, 2.0, -0.5], dtype=D).norm()
    perp = shift - (shift @ direction)[:, None] * direction
    assert float(perp.abs().max()) < 1e-14 + 1e-9 * float(shift.abs().max()), (
        "points on a line must stay on that line: a rotation about the line leaves the input unchanged, so it must "
        f"leave the output unchanged too. off-line movement {float(perp.abs().max()):.1e}")
    two = torch.tensor([[0.0, 0, 0], [1.0, 1.0, 0]], dtype=D)
    _, out2 = layer(h[:2], two, _full(2))
    s = out2[0] - two[0]
    cross = torch.linalg.cross(s, two[1] - two[0])
    assert float(cross.abs().max()) < 1e-14 + 1e-9 * float(s.abs().max()), "with two atoms, each moves along the bond"


def test_stage2_residual_and_directed_mean(nb):
    layer = _layer(nb, 5)
    h, pos = torch.randn(5, 8, dtype=D, generator=torch.Generator().manual_seed(5)), _cloud(5, 5)
    zeroed = _layer(nb, 5)
    with torch.no_grad():
        for p in zeroed.parameters():
            p.zero_()
    h0, pos0 = zeroed(h, pos, _full(5))
    assert torch.allclose(h0, h, atol=1e-14) and torch.allclose(pos0, pos, atol=1e-14), (
        "with every parameter zeroed the MLPs output 0, so h' = h + phi_h(...) = h and the positions stay put. if h' is "
        "0, the residual connection h + ... is missing")
    star = torch.tensor([[1, 2, 3], [0, 0, 0]])  # node 0 hears from nodes 1, 2, 3
    more = torch.tensor([[1, 2, 3, 0, 0], [0, 0, 0, 4, 2]])  # ...and also SENDS to nodes 4 and 2
    _, a = layer(h, pos, star)
    _, b = layer(h, pos, more)
    assert torch.allclose(a[0], b[0], atol=1e-14), (
        "node 0's update depends only on its INCOMING edges: edges leaving node 0 must change nothing for node 0. count "
        "neighbours at the target (edge_index[1]), not the source")


def test_stage2_squared_distance_is_smooth(nb):
    layer = _layer(nb, 6)
    pos = torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0.5, 0]], dtype=D, requires_grad=True)  # two atoms coincide
    h = torch.randn(3, 8, dtype=D, generator=torch.Generator().manual_seed(6))
    out_h, out_pos = layer(h, pos, _full(3))
    (out_h.sum() + out_pos.sum()).backward()
    assert torch.isfinite(pos.grad).all(), (
        "two atoms at the same position give nan gradients: the messages must use the SQUARED distance (a polynomial), "
        "not the distance, whose derivative is infinite at 0")


def test_stage2_edge_attributes(nb):
    layer = _layer(nb, 4, edge_dim=2)
    n = 5
    h, pos, ei = torch.randn(n, 8, dtype=D), _cloud(n, 4), _full(n)
    a = torch.randn(ei.shape[1], 2, dtype=D, generator=torch.Generator().manual_seed(4))
    h1, _ = layer(h, pos, ei, a)
    h2, _ = layer(h, pos, ei, a * 2)
    assert not torch.allclose(h1, h2, atol=1e-8), "edge attributes (edge_dim > 0) enter the messages"
    R = _orthogonal(5)
    h3, _ = layer(h, pos @ R.T, ei, a)
    assert _rel(h3, h1) < 1e-10, "edge attributes are invariant inputs: rotating positions still doesn't change h"


# ---------------------------------------------------------------- stage 3: EGNN


def _molecules(sizes, seed):
    g = torch.Generator().manual_seed(seed)
    xs, ps, eis, bs, off = [], [], [], [], 0
    for k, n in enumerate(sizes):
        xs.append(F.one_hot(torch.randint(0, 3, (n,), generator=g), 3).to(D))
        ps.append(torch.randn(n, 3, generator=g, dtype=D))
        eis.append(_full(n, off))
        bs.append(torch.full((n,), k, dtype=torch.int64))
        off += n
    return torch.cat(xs), torch.cat(ps), torch.cat(eis, dim=1), torch.cat(bs)


def test_stage3_invariance(nb):
    torch.manual_seed(0)
    model = nb.EGNN(3, 16, 2, n_layers=3).double()
    x, pos, ei, batch = _molecules([5, 7, 4], seed=0)
    out = model(x, pos, ei, batch)
    assert tuple(out.shape) == (3, 2), f"EGNN(3, 16, 2) gives (num_graphs, 2); got {tuple(out.shape)}"
    for proper in (True, False):
        R = _orthogonal(7, proper=proper)
        out2 = model(x, pos @ R.T + torch.tensor([10.0, 0.0, -3.0], dtype=D), ei, batch)
        assert _rel(out2, out) < 1e-10, (
            f"the molecule-level output must be invariant to {'rotations' if proper else 'reflections'} and "
            f"translations; relative error {_rel(out2, out):.1e}")
    p = torch.randperm(len(x), generator=torch.Generator().manual_seed(1))
    out3 = model(x[p], pos[p], _relabel_edges(ei, p), batch[p])
    assert _rel(out3, out) < 1e-10, "relabelling atoms (even across the batch order) changes nothing"
    alone = model(x[5:12], pos[5:12], _full(7), torch.zeros(7, dtype=torch.int64))
    assert _rel(alone[0], out[1]) < 1e-10, "molecules in a batch don't interact"
    twice = model(torch.cat([x[5:12], x[5:12]]), torch.cat([pos[5:12], pos[5:12] + 100.0]),
                  torch.cat([_full(7), _full(7, 7)], dim=1), torch.zeros(14, dtype=torch.int64))
    assert _rel(twice[0], 2 * alone[0]) < 1e-10, (
        "one graph made of two far-apart, unconnected copies of a molecule must predict twice the molecule's value: the "
        "readout is a SUM over the graph's nodes, not a mean")


def _gyration_task(seed, n_mol=64):
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(4, 8, (n_mol,), generator=g).tolist()
    scales = torch.rand(n_mol, generator=g) + 0.5
    xs, ps, eis, bs, ys, off = [], [], [], [], [], 0
    for k, n in enumerate(sizes):
        p = torch.randn(n, 3, generator=g) * scales[k]
        xs.append(torch.ones(n, 1))
        ps.append(p)
        eis.append(_full(n, off))
        bs.append(torch.full((n,), k, dtype=torch.int64))
        ys.append(((p - p.mean(0)) ** 2).sum(1).mean())
        off += n
    return torch.cat(xs), torch.cat(ps), torch.cat(eis, dim=1), torch.cat(bs), torch.stack(ys)[:, None]


def test_stage3_learns_an_invariant_target(nb):
    def run(seed):
        torch.manual_seed(seed)
        model = nb.EGNN(1, 16, 1, n_layers=2)
        x, pos, ei, b, y = _gyration_task(seed)
        bounded(model, x, pos, ei, b, what="EGNN.forward")  # a runaway loop fails here, not in training
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(100):
            loss = F.mse_loss(model(x, pos, ei, b), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        xt, pt, et, bt, yt = _gyration_task(seed + 100)
        with torch.no_grad():
            return float(F.mse_loss(model(xt, pt, et, bt), yt) / yt.var())

    rel = sorted(run(s) for s in range(5))
    assert rel[2] < 0.25, (
        f"an EGNN should learn the radius of gyration of small point clouds (an invariant target) to a relative "
        f"test MSE below 0.25 in 100 Adam steps; median over 5 seeds {rel[2]:.3f}")
