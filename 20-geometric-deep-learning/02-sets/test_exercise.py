"""Checks for exercise.py (sets).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the pooling is
checked on hand examples and for exact independence from padding; the networks for exact
permutation invariance/equivariance in float64, for padding invariance, and for the structural
fingerprints of each aggregation (max ignores duplicates, the equivariant linear layer has
exactly two free numbers per channel pair, attention pooling can't see multiplicities); the
attention against torch's scaled_dot_product_attention. one small learning check (DeepSets
learns a set sum) runs under `mlfs.testing.bounded`, aggregated over seeds.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

D = torch.float64


def _batch(seed=0, B=3, N=5, Fdim=4):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, Fdim, generator=g, dtype=D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 3:] = False
    mask[2, 1:] = False
    return x, mask


def _junk(x, mask, value=1e3):
    """same real elements, garbage in the padded rows."""
    y = x.clone()
    y[~mask] = value
    return y


def _permute_sets(x, mask, seed=0):
    g = torch.Generator().manual_seed(seed + 100)
    perms = torch.stack([torch.randperm(x.shape[1], generator=g) for _ in range(x.shape[0])])
    idx = torch.arange(x.shape[0])[:, None]
    return x[idx, perms], mask[idx, perms], perms


# ---------------------------------------------------------------- stage 1: masked_pool


def test_stage1_hand_example(nb):
    h = torch.tensor([[[1.0, -2.0], [3.0, 4.0], [100.0, 100.0]]], dtype=D)
    mask = torch.tensor([[True, True, False]])
    got = {a: nb.masked_pool(h, mask, a) for a in ("sum", "mean", "max")}
    want = {"sum": [4.0, 2.0], "mean": [2.0, 1.0], "max": [3.0, 4.0]}
    for a in want:
        assert tuple(got[a].shape) == (1, 2), f"masked_pool returns (B, F); got {tuple(got[a].shape)} for {a!r}"
        assert torch.allclose(got[a], torch.tensor([want[a]], dtype=D)), (
            f"{a} over the two real rows (1, -2) and (3, 4) is {want[a]}; got {got[a].tolist()}. "
            "the third row is padding: it must not count, neither in the value nor in the mean's denominator")
    assert torch.allclose(nb.masked_pool(h, None, "sum"), h.sum(1)), "mask=None means every row is real"


def test_stage1_padding_is_ignored_exactly(nb):
    x, mask = _batch(1)
    for a in ("sum", "mean", "max"):
        for junk in (1e6, -1e6):
            assert torch.equal(nb.masked_pool(x, mask, a), nb.masked_pool(_junk(x, mask, junk), mask, a)), (
                f"changing the padded rows changed the {a}-pool: multiply by the mask (sum/mean) or fill padded "
                "rows with -inf (max) before reducing")
    empty = torch.zeros(2, 3, dtype=torch.bool)
    empty[0, 0] = True
    for a in ("sum", "mean", "max"):
        out = nb.masked_pool(x[:2, :3], empty, a)
        assert torch.isfinite(out).all() and torch.equal(out[1], torch.zeros(4, dtype=D)), (
            f"an empty set pools to 0 for every aggregation (no nan, no -inf); got {out[1].tolist()} for {a!r}")
    with pytest.raises(ValueError):
        nb.masked_pool(x, mask, "median")


def test_stage1_gradients(nb):
    x, mask = _batch(2)
    x.requires_grad_(True)
    nb.masked_pool(x, mask, "max").sum().backward()
    g = x.grad
    assert torch.equal(g[~mask], torch.zeros_like(g[~mask])), "padding gets no gradient"
    assert torch.allclose(g.sum(1), torch.ones(3, 4, dtype=D)), (
        "the max passes each output's gradient to exactly one real element (the argmax)")
    x.grad = None
    nb.masked_pool(x, mask, "mean").sum().backward()
    assert torch.allclose(x.grad[1, :3], torch.full((3, 4), 1 / 3, dtype=D)), "mean over 3 real rows: gradient 1/3"


# ---------------------------------------------------------------- stage 2: DeepSets


def _deepsets(nb, agg="sum", seed=0, i=4, h=16, o=3):
    torch.manual_seed(seed)
    return nb.DeepSets(i, h, o, aggregation=agg).double()


def test_stage2_shapes_and_validation(nb):
    x, mask = _batch(0)
    m = _deepsets(nb)
    out = m(x, mask)
    assert tuple(out.shape) == (3, 3), f"DeepSets(4, 16, 3) maps (B, N, 4) to (B, 3); got {tuple(out.shape)}"
    assert m.aggregation == "sum", "store the aggregation as self.aggregation"
    assert sum(p.numel() for p in m.parameters()) > 0
    assert torch.allclose(m(x), m(x, torch.ones(3, 5, dtype=torch.bool))), "mask=None means every element is real"
    with pytest.raises(ValueError):
        nb.DeepSets(4, 16, 3, aggregation="median")


@pytest.mark.parametrize("agg", ["sum", "mean", "max"])
def test_stage2_exact_invariance(nb, agg):
    x, mask = _batch(3)
    m = _deepsets(nb, agg, seed=1)
    base = m(x, mask)
    for s in range(3):
        xp, mp, _ = _permute_sets(x, mask, s)
        err = float((m(xp, mp) - base).detach().abs().max())
        assert err < 1e-12, f"DeepSets({agg!r}) must be exactly permutation-invariant (float64); error {err:.2e}"
    err = float((m(_junk(x, mask), mask) - base).abs().max())
    assert err < 1e-12, f"padded elements must not change the output; error {err:.2e}"
    one = m(x[2:3, :1])
    assert torch.allclose(one, base[2:3], atol=1e-12), "a padded set equals the same set without its padding"


def test_stage2_aggregation_fingerprints(nb):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 3, 4, generator=g, dtype=D)
    dup = torch.cat([x, x[:, :1]], dim=1)  # the first element twice
    doubled = torch.cat([x, x], dim=1)  # every element twice
    mx, mean, sm = _deepsets(nb, "max", 2), _deepsets(nb, "mean", 2), _deepsets(nb, "sum", 2)
    assert torch.allclose(mx(dup), mx(x), atol=1e-12), (
        "with max aggregation a duplicated element changes nothing: max sees which elements occur, not how often")
    assert torch.allclose(mean(doubled), mean(x), atol=1e-12), "mean can't tell a set from the same set twice"
    assert not torch.allclose(sm(doubled), sm(x), atol=1e-6), (
        "sum aggregation sees multiplicities: doubling every element must change the output (did you pool with "
        "the aggregation you were given?)")


def test_stage2_phi_is_nonlinear(nb):
    m = _deepsets(nb, "sum", 3, i=1)
    a = torch.tensor([[[1.0], [-1.0]]], dtype=D)  # same size, same sum (0), different elements
    b = torch.tensor([[[0.0], [0.0]]], dtype=D)
    assert not torch.allclose(m(a), m(b), atol=1e-8), (
        "{1, -1} and {0, 0} have the same size and sum; if DeepSets can't tell them apart, phi is linear (then the sum "
        "of phi(x) only depends on the sum of x). phi needs a hidden layer and a nonlinearity")


def _set_sum_task(seed, n=256):
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(1, 9, (n,), generator=g)
    x = torch.rand(n, 8, 1, generator=g) * 2 - 1
    mask = torch.arange(8)[None] < sizes[:, None]
    y = (x[..., 0] * mask).sum(1, keepdim=True)
    return x, mask, y


def test_stage2_learns_a_set_sum(nb):
    def run(seed):
        torch.manual_seed(seed)
        m = nb.DeepSets(1, 32, 1, aggregation="sum")
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        x, mask, y = _set_sum_task(seed)
        bounded(m, x, mask, what="DeepSets.forward")  # one traced pass: a runaway loop fails here, not in training
        for _ in range(100):
            loss = F.mse_loss(m(x, mask), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        xt, mt, yt = _set_sum_task(seed + 50)
        with torch.no_grad():
            return float(F.mse_loss(m(xt, mt), yt) / yt.var())

    rel = sorted(run(s) for s in range(5))
    assert rel[2] < 0.05, (
        f"DeepSets with sum aggregation should learn y = sum of the set to a relative test error < 0.05 in 100 Adam "
        f"steps; median over 5 seeds {rel[2]:.3f}. check that phi runs on every element and rho after pooling")


# ---------------------------------------------------------------- stage 3: PermEquivariant


def test_stage3_shapes_equivariance_padding(nb):
    x, mask = _batch(5)
    for agg in ("sum", "mean", "max"):
        torch.manual_seed(0)
        layer = nb.PermEquivariant(4, 6, aggregation=agg).double()
        y = layer(x, mask)
        assert tuple(y.shape) == (3, 5, 6), f"(B, N, in) -> (B, N, out); got {tuple(y.shape)}"
        assert torch.equal(y[~mask], torch.zeros_like(y[~mask])), "padded rows of the output are exactly 0"
        xp, mp, perms = _permute_sets(x, mask, 1)
        idx = torch.arange(3)[:, None]
        err = float((layer(xp, mp) - y[idx, perms]).abs().max())
        assert err < 1e-12, f"PermEquivariant({agg!r}) must be permutation-equivariant; error {err:.2e}"
        err = float((layer(_junk(x, mask), mask) - y).abs().max())
        assert err < 1e-12, f"the pool inside the layer must ignore padding; error {err:.2e} ({agg!r})"
    with pytest.raises(ValueError):
        nb.PermEquivariant(4, 6, aggregation="median")


def test_stage3_two_numbers_per_channel_pair(nb):
    torch.manual_seed(3)
    layer = nb.PermEquivariant(1, 1, aggregation="mean").double()
    n = 5
    f0 = layer(torch.zeros(1, n, 1, dtype=D))[0, :, 0]
    M = torch.stack([layer(torch.eye(n, dtype=D)[j].reshape(1, n, 1))[0, :, 0] - f0 for j in range(n)], dim=1)
    diag, off = M.diagonal(), M[~torch.eye(n, dtype=torch.bool)]
    assert torch.allclose(diag, diag[0].expand(n), atol=1e-12) and torch.allclose(off, off[0].expand(len(off)),
                                                                                 atol=1e-12), (
        "a permutation-equivariant linear map on one channel is a*I + b*11^T: all diagonal entries equal, all "
        f"off-diagonal entries equal. got the matrix\n{M}")
    assert abs(float(off[0])) > 1e-6, (
        "the off-diagonal entries are 0: each output only sees its own input. the pooled term "
        "pool(X) Gamma lets elements see each other")
    x1, x2 = torch.randn(1, n, 1, dtype=D), torch.randn(1, n, 1, dtype=D)
    lin = layer(x1 + x2) - f0 - (layer(x1) - f0) - (layer(x2) - f0)
    assert float(lin.abs().max()) < 1e-12, "with mean pooling the layer is affine: f(x1 + x2) - f(0) is additive"


# ---------------------------------------------------------------- stage 4: attention


def test_stage4_masked_attention_matches_torch(nb):
    g = torch.Generator().manual_seed(6)
    q = torch.randn(3, 2, 8, generator=g, dtype=D)
    k = torch.randn(3, 5, 8, generator=g, dtype=D)
    v = torch.randn(3, 5, 4, generator=g, dtype=D)
    _, mask = _batch(0)
    got = nb.masked_attention(q, k, v, mask)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, :])
    assert tuple(got.shape) == (3, 2, 4), f"(B, M, d), (B, N, d), (B, N, dv) -> (B, M, dv); got {tuple(got.shape)}"
    assert torch.allclose(got, want, atol=1e-12), (
        "masked_attention should match softmax(q k^T / sqrt(d)) v with padded keys excluded from the softmax. "
        "did you scale by sqrt(d), mask before the softmax (set scores to -inf), and softmax over the keys?")
    assert torch.allclose(nb.masked_attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-12)


def test_stage4_empty_sets_give_zero_and_finite_gradients(nb):
    g = torch.Generator().manual_seed(7)
    q = torch.randn(2, 1, 4, generator=g, dtype=D, requires_grad=True)
    k = torch.randn(2, 3, 4, generator=g, dtype=D, requires_grad=True)
    v = torch.randn(2, 3, 4, generator=g, dtype=D, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    out = nb.masked_attention(q, k, v, mask)
    assert torch.equal(out[1], torch.zeros(1, 4, dtype=D)), f"a set with no real elements attends to nothing: 0; got {out[1]}"
    out.sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in (q, k, v)), (
        "nan in the gradients: a softmax over a row that's all -inf produces nan, and multiplying by 0 afterwards "
        "doesn't remove it from the backward pass. avoid all -inf rows before the softmax")


def test_stage4_attention_pool(nb):
    x, mask = _batch(8, Fdim=8)
    torch.manual_seed(0)
    pool = nb.AttentionPool(8, heads=2, n_seeds=3).double()
    y = pool(x, mask)
    assert tuple(y.shape) == (3, 3, 8), f"AttentionPool(8, n_seeds=3) maps (B, N, 8) to (B, 3, 8); got {tuple(y.shape)}"
    for s in range(3):
        xp, mp, _ = _permute_sets(x, mask, s)
        err = float((pool(xp, mp) - y).abs().max())
        assert err < 1e-12, f"attention pooling is permutation-invariant; error {err:.2e}"
    assert float((pool(_junk(x, mask), mask) - y).abs().max()) < 1e-12, "padded elements must not be attended to"
    assert len([p for p in pool.parameters()]) > 0
    doubled = torch.cat([x[:1, :4], x[:1, :4]], dim=1)
    assert torch.allclose(pool(doubled), pool(x[:1, :4]), atol=1e-12), (
        "attention weights are a softmax, so every element of a set listed twice gets half the weight: the output "
        "can't change. (if it does, something outside the softmax depends on the number of elements)")
    with pytest.raises(ValueError):
        nb.AttentionPool(8, heads=3)
    one = nb.AttentionPool(8, heads=1, n_seeds=1).double()
    assert not torch.allclose(one(x[:1, :2]), one(x[:1, 1:3]), atol=1e-6), "different sets, different outputs"
