"""Checks for 28 · 03 superposition & dictionary learning.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the sparse data generator for its
sparsity (a binomial test), its value distribution and the generator; the toy model for its parameters, buffer
and forward pass against the formula; the loss against a hand computation (importance used); training for
what it learns at low and high sparsity (5 features in 2 dimensions); feature dimensionality on known
geometries; the sparse autoencoder against its formulas (unit-norm dictionary, l1_coeff in the loss); SAE
training for recovering a planted dictionary; MMCS and the SAE statistics against hand computations.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


# ---------------------------------------------------------------- stage 1: sparse features


def test_stage1_sparsity_and_values(nb):
    x = nb.sparse_feature_batch(4000, 50, 0.7, generator=torch.Generator().manual_seed(0))
    assert x.shape == (4000, 50) and x.dtype == torch.float32, "shape (n, n_features), float32"
    zero = (x == 0).float().mean().item()
    se = math.sqrt(0.7 * 0.3 / x.numel())
    assert abs(zero - 0.7) < 5 * se, (
        f"each feature is zero with probability sparsity = 0.7; got a zero fraction of {zero:.4f}. "
        "if it's near 0.3, the mask is inverted")
    nz = x[x > 0]
    assert nz.max() < 1 and abs(nz.mean().item() - 0.5) < 0.01 and abs(nz.var().item() - 1 / 12) < 0.005, (
        "the nonzero values are uniform on [0, 1): mean 1/2, variance 1/12")
    col = (x == 0).float().mean(0)
    assert (col - 0.7).abs().max() < 0.05, "every feature has the same sparsity (independent per entry)"
    assert torch.all(nb.sparse_feature_batch(10, 4, 1.0) == 0), "sparsity 1: every feature is zero"
    assert torch.all(nb.sparse_feature_batch(1000, 4, 0.0, generator=torch.Generator().manual_seed(1)) > 0), (
        "sparsity 0: dense features")
    with pytest.raises(ValueError):
        nb.sparse_feature_batch(10, 4, 1.5)


def test_stage1_generator(nb):
    a = nb.sparse_feature_batch(20, 6, 0.5, generator=torch.Generator().manual_seed(3))
    b = nb.sparse_feature_batch(20, 6, 0.5, generator=torch.Generator().manual_seed(3))
    c = nb.sparse_feature_batch(20, 6, 0.5, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, b) and not torch.equal(a, c), "all randomness comes from the generator"


# ---------------------------------------------------------------- stage 2: the toy model


def test_stage2_parameters(nb):
    torch.manual_seed(0)
    m = nb.ToySuperposition(6, 3)
    assert isinstance(m, nn.Module) and m.W.shape == (3, 6) and m.b.shape == (6,), "W (n_hidden, n_features), b (n_features,)"
    assert {n for n, _ in m.named_parameters()} == {"W", "b"}, "the parameters are W and b"
    assert torch.equal(m.b, torch.zeros(6)), "b starts at zero"
    big = nb.ToySuperposition(60, 30)
    std = big.W.detach().std().item()
    assert abs(std - math.sqrt(2 / 90)) < 0.2 * math.sqrt(2 / 90), (
        f"Xavier-normal init: std sqrt(2 / (n_hidden + n_features)) = {math.sqrt(2 / 90):.3f} for 60 features in 30 "
        f"dimensions; got {std:.3f}")
    assert "importance" in dict(m.named_buffers()) and torch.equal(m.importance, torch.ones(6)), (
        "importance is a buffer (not a parameter), ones by default")
    imp = 0.9 ** torch.arange(6.0)
    m2 = nb.ToySuperposition(6, 3, importance=imp)
    assert torch.allclose(m2.importance, imp) and "importance" not in dict(m2.named_parameters()), "store importance"
    with pytest.raises(ValueError):
        nb.ToySuperposition(6, 3, importance=torch.ones(5))


def test_stage2_forward(nb):
    torch.manual_seed(0)
    m = nb.ToySuperposition(5, 2)
    with torch.no_grad():
        m.b.copy_(torch.tensor([-0.5, 0.1, -0.2, 0.3, 0.0]))
    x = nb.sparse_feature_batch(7, 5, 0.3, generator=torch.Generator().manual_seed(1))
    got = m(x)
    want = torch.relu(x @ m.W.T @ m.W + m.b)
    assert got.shape == (7, 5) and torch.allclose(got, want, atol=1e-6), (
        "forward: h = W x (n_hidden numbers), then ReLU(W^T h + b); for a batch, x @ W.T @ W + b")
    assert (got >= 0).all(), "the ReLU is part of the model"


# ---------------------------------------------------------------- stage 3: training and geometry


def test_stage3_loss(nb):
    torch.manual_seed(0)
    m = nb.ToySuperposition(3, 2, importance=torch.tensor([1.0, 0.5, 0.0]))
    with torch.no_grad():
        m.W.zero_()
        m.b.copy_(torch.tensor([0.2, 0.0, 0.0]))
    x = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.5, 0.0]])
    # outputs are relu(b) = (0.2, 0, 0) for both rows
    want = ((1 * 0.8 ** 2 + 0.5 * 1.0 + 0) + (1 * 0.2 ** 2 + 0.5 * 0.25 + 0)) / 2
    got = nb.superposition_loss(m, x)
    assert torch.is_tensor(got) and np.isclose(got.item(), want), (
        f"mean over rows of sum_i importance_i (x_i - out_i)^2 = {want:.4f}; got {float(got):.4f}")


def _toy(nb, sparsity, seed):
    torch.manual_seed(seed)
    m = nb.ToySuperposition(5, 2, importance=0.8 ** torch.arange(5.0))
    losses = bounded(nb.train_toy_model, m, sparsity, budget=20_000_000, what="train_toy_model",
                     steps=1500, batch_size=256, lr=1e-2, generator=torch.Generator().manual_seed(seed))
    return m, losses


def _toy_train(nb, *args, **kw):
    return bounded(nb.train_toy_model, *args, budget=20_000_000, what="train_toy_model", **kw)


def _sae_train(nb, *args, **kw):
    return bounded(nb.train_sae, *args, budget=20_000_000, what="train_sae", **kw)


def test_stage3_training_learns_superposition(nb):
    dense, sparse = [], []
    for seed in range(5):
        m, losses = _toy(nb, 0.0, seed)
        assert len(losses) == 1500 and all(isinstance(v, float) for v in losses), "one float loss per step"
        assert np.mean(losses[-100:]) < 0.8 * np.mean(losses[:10]), "training lowers the loss"
        dense.append(int((m.W.norm(dim=0) > 0.5).sum()))
        m, _ = _toy(nb, 0.9, seed)
        sparse.append(int((m.W.norm(dim=0) > 0.5).sum()))
    assert sorted(dense)[2] == 2, (
        f"dense features (sparsity 0), 5 features in 2 dimensions: the model keeps the 2 most important and drops "
        f"the rest (like PCA); features with ||W_i|| > 0.5 per seed: {dense}")
    assert sorted(sparse)[2] >= 4, (
        f"sparse features (sparsity 0.9): the model packs at least 4 features into 2 dimensions (superposition); "
        f"got {sparse}. check that each step draws new data at the given sparsity")


def test_stage3_training_options(nb):
    kept = []
    for seed in range(3):
        torch.manual_seed(seed)
        m = nb.ToySuperposition(4, 2, importance=torch.tensor([1.0, 1.0, 0.0, 0.0]))
        _toy_train(nb, m, 0.0, steps=1000, batch_size=128, lr=1e-2, generator=torch.Generator().manual_seed(seed))
        kept.append(float(m.W.norm(dim=0)[:2].min()))
    assert np.median(kept) > 0.7, (
        f"with dense features and 2 dimensions, the model keeps the two features that count: features with "
        f"importance 0 don't appear in the loss. the smaller ||W_i|| of features 0 and 1 per seed: "
        f"{np.round(kept, 2).tolist()}. the training loss must use model.importance")
    models = []
    for _ in range(3):
        torch.manual_seed(0)
        models.append(nb.ToySuperposition(4, 2))
    runs = []
    for mm, lr in zip(models, (1e-2, 1e-2, 1e-4)):
        torch.randn(7)  # the global RNG moves on between runs: only the generator may matter
        runs.append(_toy_train(nb, mm, 0.5, steps=50, batch_size=64, lr=lr,
                                       generator=torch.Generator().manual_seed(1)))
    assert runs[0] == runs[1], "with the same initial weights and generator, training is reproducible (use generator)"
    assert runs[2][-1] > runs[0][-1], "lr is the optimizer's learning rate (1e-4 learns far less in 50 steps)"
    torch.manual_seed(0)
    frozen = _toy_train(nb, nb.ToySuperposition(4, 2), 0.5, steps=20, batch_size=64, lr=0.0,
                                generator=torch.Generator().manual_seed(2))
    assert len(set(frozen)) > 1, (
        "with lr = 0 the model never changes, so the losses can only differ if every step draws a NEW batch: "
        "draw fresh data inside the loop")


def test_stage3_uses_adam(nb):
    torch.manual_seed(0)
    m = nb.ToySuperposition(6, 3)
    before = [p.detach().clone() for p in m.parameters()]
    _toy_train(nb, m, 0.3, steps=1, batch_size=256, lr=0.01, generator=torch.Generator().manual_seed(0))
    moves = torch.cat([(p.detach() - q).abs().flatten() for p, q in zip(m.parameters(), before)])
    assert torch.allclose(moves, torch.full_like(moves, 0.01), rtol=1e-3), (
        f"train_toy_model uses Adam: its first step moves every parameter by exactly lr (= 0.01, whatever the gradient's "
        f"size); got moves between {moves.min():.5f} and {moves.max():.5f}. plain SGD would move by lr * gradient")


def test_stage3_feature_dimensionality(nb):
    D = nb.feature_dimensionality(torch.eye(3))
    assert torch.allclose(D, torch.ones(3)), f"orthogonal features each get a dimension: D = 1; got {D.tolist()}"
    anti = torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 2.0]])
    assert torch.allclose(nb.feature_dimensionality(anti), torch.tensor([0.5, 0.5, 1.0])), (
        "an antipodal pair shares a dimension (D = 1/2 each); a lone feature of any length has D = 1")
    ang = torch.arange(5) * 2 * math.pi / 5
    penta = torch.stack([torch.cos(ang), torch.sin(ang)])
    assert torch.allclose(nb.feature_dimensionality(penta), torch.full((5,), 0.4), atol=1e-5), (
        "a regular pentagon in 2 dimensions: D = 2/5 each")
    z = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    Dz = nb.feature_dimensionality(z)
    assert torch.allclose(Dz, torch.tensor([1.0, 0.0])) and not torch.isnan(Dz).any(), (
        "a feature with a zero column isn't represented: D = 0 (not NaN)")
    scaled = nb.feature_dimensionality(anti * torch.tensor([3.0, 0.5, 1.0]))
    assert torch.allclose(scaled, torch.tensor([9 / 9.25, 0.25 / 9.25, 1.0]), atol=1e-5), (
        "D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2: an antipodal pair of lengths 3 and 0.5 gets 9 / 9.25 and "
        f"0.25 / 9.25; got {scaled.tolist()}")


# ---------------------------------------------------------------- stage 4: the sparse autoencoder


def _sae(nb, d_in=4, d_hidden=6, l1=0.01, seed=0):
    torch.manual_seed(seed)
    return nb.SparseAutoencoder(d_in, d_hidden, l1_coeff=l1)


def test_stage4_parameters(nb):
    s = _sae(nb)
    shapes = {n: tuple(p.shape) for n, p in s.named_parameters()}
    assert shapes == {"W_enc": (4, 6), "b_enc": (6,), "W_dec": (4, 6), "b_dec": (4,)}, (
        f"parameters W_enc (d_in, d_hidden), b_enc, W_dec (d_in, d_hidden), b_dec; got {shapes}")
    assert torch.allclose(s.W_dec.norm(dim=0), torch.ones(6), atol=1e-5), "decoder columns start at unit norm"
    assert torch.allclose(s.W_enc, s.W_dec), "the encoder starts as a copy of the decoder"
    assert torch.equal(s.b_enc, torch.zeros(6)) and torch.equal(s.b_dec, torch.zeros(4)), "biases start at zero"
    assert s.l1_coeff == 0.01 and nb.SparseAutoencoder(4, 6).l1_coeff == 0.001, "store l1_coeff (default 0.001)"


def test_stage4_encode_decode(nb):
    s = _sae(nb)
    with torch.no_grad():
        s.W_dec.mul_(torch.linspace(0.5, 3.0, 6))  # columns no longer unit norm
        s.W_enc.copy_(torch.randn(4, 6, generator=torch.Generator().manual_seed(1)))
        s.b_enc.copy_(torch.linspace(-0.5, 0.5, 6))
        s.b_dec.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(2))
    f = s.encode(x)
    assert f.shape == (5, 6) and torch.allclose(f, torch.relu((x - s.b_dec) @ s.W_enc + s.b_enc), atol=1e-6), (
        "encode: ReLU((x - b_dec) W_enc + b_enc)")
    D = s.dictionary()
    assert D.shape == (4, 6) and torch.allclose(D.norm(dim=0), torch.ones(6), atol=1e-5), (
        "dictionary(): W_dec with every column scaled to unit norm, whatever W_dec's current norms are")
    assert torch.allclose(D, s.W_dec / s.W_dec.norm(dim=0), atol=1e-6)
    eye = torch.eye(6)
    atoms = s.decode(eye) - s.decode(torch.zeros(1, 6))
    assert torch.allclose(atoms.norm(dim=1), torch.ones(6), atol=1e-5), (
        "decode uses the unit-norm dictionary: feature j alone adds a unit vector to b_dec")
    assert torch.allclose(s.decode(f), f @ D.T + s.b_dec, atol=1e-6), "decode: f dictionary()^T + b_dec"


def test_stage4_loss(nb):
    x = torch.randn(8, 4, generator=torch.Generator().manual_seed(3))
    for l1 in (0.0, 0.05, 0.5):
        s = _sae(nb, l1=l1)
        x_hat, f, loss = s(x)
        assert torch.allclose(x_hat, s.decode(s.encode(x))) and torch.allclose(f, s.encode(x)), (
            "forward returns (decode(encode(x)), encode(x), loss)")
        want = ((x_hat - x) ** 2).sum(-1).mean() + l1 * f.abs().sum(-1).mean()
        assert np.isclose(loss.item(), want.item(), rtol=1e-5), (
            f"l1_coeff={l1}: loss = mean over rows of ||x_hat - x||^2 (summed over d_in) + l1_coeff * mean over rows "
            f"of ||f||_1 = {want.item():.5f}; got {loss.item():.5f}")
    loss.backward()
    assert all(p.grad is not None for p in s.parameters()), "every parameter gets a gradient"


# ---------------------------------------------------------------- stage 5: training and evaluation


def _planted(seed, n=4000):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(6, 12, generator=g)
    D = D / D.norm(dim=0)
    f = (torch.rand(n, 12, generator=g) * (torch.rand(n, 12, generator=g) >= 0.85))
    return D, f @ D.T


def test_stage5_recovers_planted_dictionary(nb):
    scores, l0 = [], {}
    for seed in range(3):
        D, acts = _planted(seed)
        torch.manual_seed(seed)
        s = nb.SparseAutoencoder(6, 24, l1_coeff=0.3)
        before = nb.mmcs(s.dictionary(), D)
        losses = bounded(nb.train_sae, s, acts, budget=20_000_000, what="train_sae", steps=1500, batch_size=256,
                         lr=3e-3, generator=torch.Generator().manual_seed(seed))
        assert len(losses) == 1500 and np.mean(losses[-50:]) < np.mean(losses[:10]), "train_sae lowers the SAE loss"
        scores.append((before, nb.mmcs(s.dictionary(), D)))
        if seed == 0:
            l0["0.3"] = nb.sae_stats(s, acts)["l0"]
    assert np.median([a for _, a in scores]) > 0.9 and all(a > b for b, a in scores), (
        f"12 sparse features in 6 dimensions: a trained 24-feature SAE should find directions within cos > 0.9 of the "
        f"true ones (MMCS before -> after per seed: {[(round(b, 2), round(a, 2)) for b, a in scores]})")
    D, acts = _planted(0)
    torch.manual_seed(0)
    s = nb.SparseAutoencoder(6, 24, l1_coeff=0.01)
    _sae_train(nb, s, acts, steps=1500, batch_size=256, lr=3e-3, generator=torch.Generator().manual_seed(0))
    weak = nb.sae_stats(s, acts)["l0"]
    assert weak > l0["0.3"] + 2, (
        f"l1_coeff controls sparsity: l1=0.01 should use many more active features per row than l1=0.3 "
        f"(got L0 {weak:.1f} vs {l0['0.3']:.1f}); does the loss use self.l1_coeff?")


def test_stage5_train_sae_options(nb):
    D, acts = _planted(1, n=500)
    runs, saes = [], []
    for lr in (3e-3, 3e-3, 1e-6):
        torch.manual_seed(0)
        s = nb.SparseAutoencoder(6, 12, l1_coeff=0.1)
        runs.append(_sae_train(nb, s, acts, steps=60, batch_size=32, lr=lr, generator=torch.Generator().manual_seed(5)))
        saes.append(s)
    assert runs[0] == runs[1], "minibatches come from the generator: same seed, same run"
    torch.manual_seed(0)
    init = nb.SparseAutoencoder(6, 12, l1_coeff=0.1)
    with torch.no_grad():
        full = [float(m(acts)[2]) for m in (init, saes[0], saes[2])]
    assert full[1] < 0.9 * full[0] and abs(full[2] - full[0]) < 0.02 * full[0], (
        f"lr is Adam's learning rate: on the whole dataset, lr=3e-3 for 60 steps lowers the loss from {full[0]:.3f} "
        f"(got {full[1]:.3f}) while lr=1e-6 leaves it where it started (got {full[2]:.3f})")
    torch.manual_seed(0)
    s = nb.SparseAutoencoder(6, 12, l1_coeff=0.1)
    other = _sae_train(nb, s, acts, steps=60, batch_size=32, lr=3e-3, generator=torch.Generator().manual_seed(6))
    assert other != runs[0], "a different generator seed draws different minibatches"
    two = torch.zeros(2, 6)
    two[1] = 3.0  # row 0 is all zeros (reconstructed perfectly at init), row 1 isn't
    torch.manual_seed(0)
    s = nb.SparseAutoencoder(6, 12, l1_coeff=0.1)
    ls = _sae_train(nb, s, two, steps=5, batch_size=16, lr=1e-6, generator=torch.Generator().manual_seed(0))
    assert max(ls) > 1.0, (
        "minibatch rows are drawn from ALL rows 0..n-1: with two rows, the last one must show up "
        "(torch.randint's upper bound is exclusive)")


def test_stage5_mmcs(nb):
    true = torch.eye(3)
    assert np.isclose(nb.mmcs(torch.eye(3), true), 1.0)
    assert np.isclose(nb.mmcs(-torch.eye(3), true), 0.0), (
        "signed cosine: a dictionary pointing the opposite way matches nothing (an SAE feature only fires positively)")
    learned = torch.tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0], [3.0, 0.0, 0.0, 0.0]])
    assert np.isclose(nb.mmcs(learned, true), 1.0), "cosine, not dot product, and the order of features doesn't matter"
    half = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    want = (2 / math.sqrt(2) + 1) / 3
    assert np.isclose(nb.mmcs(half, true), want), (
        f"for each TRUE feature take the best learned match, then average over true features: {want:.4f}")
    assert np.isclose(nb.mmcs(true, half), (1 / math.sqrt(2) + 1) / 2), (
        "the arguments are (learned, true), and it isn't symmetric: mmcs(eye, half) averages over half's 2 columns")
    assert isinstance(nb.mmcs(true, true), float)


def test_stage5_stats(nb):
    s = _sae(nb, d_in=2, d_hidden=3)
    with torch.no_grad():
        s.W_enc.copy_(torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]]))
        s.b_enc.zero_()
        s.b_dec.zero_()
        s.W_dec.copy_(torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    acts = torch.tensor([[1.0, 0.0], [2.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    st = nb.sae_stats(s, acts)
    # f = relu([x0, -x0, 0]): rows have 1, 1, 1, 0 active features; feature 2 is never active
    # x_hat = (relu(x0) + relu(-x0), 0) = (|x0|, 0): errors (0,0), (0,1), (2,0), (0,0) -> 5
    # variance: mean (0.5, 0.25): sum of squares 0.25+2.25+2.25+0.25 + 0.0625*3+0.5625 = 5.75
    assert set(st) >= {"l0", "dead", "fvu"}
    assert np.isclose(st["l0"], 0.75) and np.isclose(st["dead"], 1 / 3), (
        f"l0 = mean number of active (> 0) features per row = 0.75; dead = fraction of features never active = 1/3; "
        f"got {st}")
    assert np.isclose(st["fvu"], 5 / 5.75), (
        f"fvu = sum ||x - x_hat||^2 / sum ||x - mean(x)||^2 = 5 / 5.75 = {5 / 5.75:.4f}; got {st['fvu']:.4f}")
