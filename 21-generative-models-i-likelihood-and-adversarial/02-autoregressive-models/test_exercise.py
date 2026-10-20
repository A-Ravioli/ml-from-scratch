"""Checks for exercise.py (autoregressive models).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Masked layers are
checked against F.linear / F.conv2d with an explicitly masked weight, and for zero gradients on masked
entries. MADE's autoregressive property is checked through its full Jacobian (computed with autograd),
before and after an optimizer step, and its connectivity (each output does see its immediate
predecessor). Likelihoods are checked against hand formulas; samplers by counting forward passes and by
comparing empirical frequencies with the model's own conditionals. PixelCNN's causality is checked by
gradients with respect to every input pixel. Short training runs on structured toy data run under
`mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _rows_constant(n, size, seed):
    """binary images whose rows are constant: each row an independent fair coin (entropy: size bits)."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, 2, (n, 1, size, 1), generator=g).float()
    return rows.expand(n, 1, size, size).contiguous()


def _made_jacobian(model, x):
    """d out[0, o, i] / d x[0, j] as an array (n_out, D, D)."""
    x = x.clone().requires_grad_(True)
    out = model(x)[0]
    n_out, d = out.shape
    jac = np.zeros((n_out, d, d))
    for o in range(n_out):
        for i in range(d):
            (g,) = torch.autograd.grad(out[o, i], x, retain_graph=True, allow_unused=True)
            jac[o, i] = 0.0 if g is None else g[0].numpy()
    return jac


# ---------------------------------------------------------------- stage 1: MaskedLinear


def test_stage1_masked_linear(nb):
    torch.manual_seed(0)
    lin = nb.MaskedLinear(4, 3)
    assert isinstance(lin, torch.nn.Module) and lin.weight.shape == (3, 4) and lin.bias.shape == (3,), (
        "a linear layer with weight (out, in) and a bias")
    assert "mask" in dict(lin.named_buffers()), "register the mask as a buffer (it moves with .to() but isn't trained)"
    assert torch.equal(lin.mask, torch.ones(3, 4)), "the mask starts as all ones"
    m = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
    lin.set_mask(m)
    x = torch.randn(5, 4)
    want = F.linear(x, lin.weight * torch.tensor(m, dtype=torch.float32), lin.bias)
    assert torch.allclose(lin(x), want, atol=1e-6), "forward must use weight * mask"
    lin(x).sum().backward()
    assert torch.all(lin.weight.grad[torch.tensor(m) == 0] == 0), "masked weights must get exactly zero gradient"
    lin.set_mask(torch.ones(3, 4))
    assert torch.allclose(lin(x), F.linear(x, lin.weight, lin.bias), atol=1e-6), "set_mask also accepts a tensor"
    assert nb.MaskedLinear(4, 3, bias=False).bias is None


# ---------------------------------------------------------------- stage 2: MADE


def test_stage2_shapes_and_masks(nb):
    torch.manual_seed(1)
    m = nb.MADE(6, [32, 32], n_outputs_per_dim=3)
    out = m(torch.randn(5, 6))
    assert out.shape == (5, 3, 6), f"x (B, D) -> (B, n_outputs_per_dim, D); got {tuple(out.shape)}"
    lins = [mod for mod in m.modules() if isinstance(mod, nb.MaskedLinear)]
    assert len(lins) == 3, f"two hidden layers and an output layer, all MaskedLinear; found {len(lins)}"
    a, b = nb.MADE(6, [32], seed=7), nb.MADE(6, [32], seed=7)
    ma = [mod.mask for mod in a.modules() if isinstance(mod, nb.MaskedLinear)]
    mb = [mod.mask for mod in b.modules() if isinstance(mod, nb.MaskedLinear)]
    assert all(torch.equal(p, q) for p, q in zip(ma, mb)), "the same seed must give the same masks"


def test_stage2_autoregressive_jacobian(nb):
    torch.manual_seed(2)
    D = 7
    m = nb.MADE(D, [64, 64], n_outputs_per_dim=2, seed=3)
    allowed = np.tril(np.ones((D, D)), -1)  # [i, j] = 1 iff j < i
    for trial in range(2):
        jac = _made_jacobian(m, torch.randn(1, D))
        bad = np.abs(jac) * (1 - allowed)[None]
        assert bad.max() == 0, (
            f"output [:, o, i] may depend only on x[:, :i]; found d out[o, i] / d x[j] != 0 for j >= i at (o, i, j) = "
            f"{tuple(int(v) for v in np.argwhere(bad > 0)[0])}. masks: hidden degree >= input degree, output degree > "
            "hidden degree (strict), with output d having degree d + 1")
        # a training step must not break the property: the mask has to be applied in every forward pass
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        opt.zero_grad()
        (m(torch.randn(16, D)) ** 2).sum().backward()
        opt.step()


def test_stage2_connectivity(nb):
    torch.manual_seed(4)
    D = 6
    m = nb.MADE(D, [128, 128], seed=0)
    seen = np.zeros((D, D), bool)
    for _ in range(8):
        seen |= np.abs(_made_jacobian(m, torch.randn(1, D))[0]) > 0
    for i in range(1, D):
        assert seen[i, i - 1], (
            f"output {i} never depends on x[{i - 1}]: with 128 hidden units per layer every allowed connection should "
            "exist. hidden degrees range over 1..D-1 (not 1..D-2), and the output test is strict: degree(out) > degree(h)")
    assert not seen[0].any(), "output 0 depends on nothing: it is the marginal p(x_0)"


def test_stage2_tiny_dims(nb):
    m = nb.MADE(1, [4])
    assert m(torch.randn(3, 1)).shape == (3, 1, 1), "D = 1 must work (the single output is an unconditional logit)"
    m2 = nb.MADE(2, [8])
    jac = _made_jacobian(m2, torch.randn(1, 2))
    assert jac[0, 0].tolist() == [0.0, 0.0] and jac[0, 1, 1] == 0.0, "D = 2: out[0] sees nothing, out[1] sees x[0]"


# ---------------------------------------------------------------- stage 3: likelihood and sampling


def test_stage3_nll_matches_bernoulli(nb):
    torch.manual_seed(5)
    m = nb.MADE(5, [16])
    x = torch.randint(0, 2, (4, 5)).float()
    logits = m(x)[:, 0, :]
    p = torch.sigmoid(logits)
    want = -(x * torch.log(p) + (1 - x) * torch.log(1 - p)).sum(1)
    got = nb.made_nll(m, x)
    assert got.shape == (4,), f"one nll per example, shape (B,); got {tuple(got.shape)}"
    assert torch.allclose(got, want, atol=1e-5), "nll = -sum_i [x_i log sigma(l_i) + (1 - x_i) log(1 - sigma(l_i))]"


def test_stage3_sums_to_one(nb):
    torch.manual_seed(6)
    D = 4
    m = nb.MADE(D, [16, 16])
    allx = torch.tensor([[(k >> j) & 1 for j in range(D)] for k in range(2 ** D)], dtype=torch.float32)
    total = torch.exp(-nb.made_nll(m, allx)).sum().item()
    assert math.isclose(total, 1.0, rel_tol=1e-5), (
        f"the probabilities of all 2^{D} binary vectors must sum to 1 (the chain rule guarantees it); got {total:.6f}")


def test_stage3_sampler(nb):
    torch.manual_seed(7)
    m = nb.MADE(3, [32])
    calls = []
    h = m.register_forward_hook(lambda *a: calls.append(1))
    x = nb.made_sample(m, 4000, generator=torch.Generator().manual_seed(8))
    h.remove()
    assert x.shape == (4000, 3) and set(x.unique().tolist()) <= {0.0, 1.0}, "binary samples of shape (n, D)"
    assert len(calls) == 3, f"one forward pass per dimension: D = 3 passes; counted {len(calls)}"
    allx = torch.tensor([[(k >> j) & 1 for j in range(3)] for k in range(8)], dtype=torch.float32)
    probs = torch.exp(-nb.made_nll(m, allx)).detach()
    counts = torch.tensor([int(((x == v).all(1)).sum()) for v in allx], dtype=torch.float32)
    chi2 = (((counts - 4000 * probs) ** 2) / (4000 * probs)).sum().item()
    assert chi2 < 30, (
        f"the samples' frequencies must match the model's own probabilities of the 8 vectors (chi^2 = {chi2:.1f}, "
        "7 dof): sample x_i from sigmoid(logit_i) computed with x_<i already filled in")
    y = nb.made_sample(m, 50, generator=torch.Generator().manual_seed(9))
    z = nb.made_sample(m, 50, generator=torch.Generator().manual_seed(9))
    assert torch.equal(y, z), "draw from `generator` so sampling is reproducible"


def test_stage3_made_learns(nb):
    torch.manual_seed(10)
    x = _rows_constant(512, 4, 0).flatten(1)  # 16 dims, 4 bits of entropy
    m = nb.MADE(16, [64, 64])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    def run():
        for _ in range(150):
            opt.zero_grad()
            nb.made_nll(m, x).mean().backward()
            opt.step()
        return nb.made_nll(m, x).mean().item()

    nll = bounded(run, budget=20_000_000, what="training a MADE")
    assert nll < 6.0, (
        f"rows of 4 identical bits: the true entropy is 4 ln 2 = 2.77 nats, a uniform model costs 11.1; "
        f"after 150 Adam steps the nll should be under 6, got {nll:.2f}")


# ---------------------------------------------------------------- stage 4: MaskedConv2d


def test_stage4_masks(nb):
    a = nb.MaskedConv2d("A", 2, 3, 5)
    b = nb.MaskedConv2d("B", 2, 3, 5)
    assert a.padding == (2, 2) and a.weight.shape == (3, 2, 5, 5), "padding kernel_size // 2 keeps the image size"
    want_a = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    want_b = want_a.clone()
    want_b[2, 2] = 1
    assert torch.equal(a.mask[0, 0].long(), want_a), f"type A: rows above, and left of the centre:\n{a.mask[0, 0]}"
    assert torch.equal(b.mask[1, 1].long(), want_b), "type B: like A, plus the centre"
    assert torch.all(a.mask == a.mask[0, 0]), "every (out, in) channel pair gets the same spatial mask"
    with pytest.raises(ValueError):
        nb.MaskedConv2d("C", 1, 1, 3)


def test_stage4_forward_and_gradients(nb):
    torch.manual_seed(11)
    conv = nb.MaskedConv2d("A", 1, 2, 3)
    x = torch.randn(1, 1, 6, 6, requires_grad=True)
    y = conv(x)
    assert y.shape == (1, 2, 6, 6)
    assert torch.allclose(y, F.conv2d(x, conv.weight * conv.mask, conv.bias, padding=1), atol=1e-6)
    y[0, :, 3, 3].sum().backward()
    g = x.grad[0, 0]
    assert g[3, 3] == 0 and g[3, 4] == 0 and torch.all(g[4] == 0), (
        "a type-A output at (3, 3) must not see input pixel (3, 3) or anything after it in raster order")
    assert g[3, 2] != 0 and g[2, 4] != 0, "it does see the left neighbour and the row above (within the kernel)"
    conv.weight.grad = None
    conv(torch.randn(2, 1, 6, 6)).sum().backward()
    assert torch.all(conv.weight.grad[conv.mask == 0] == 0), "masked weights get zero gradient"


# ---------------------------------------------------------------- stage 5: PixelCNN


def test_stage5_shapes_and_nll(nb):
    torch.manual_seed(12)
    m = nb.PixelCNN(3, n_levels=5, hidden_channels=8, n_layers=3)
    x = torch.randint(0, 5, (2, 3, 6, 7)).float() / 4
    logits = m(x)
    assert logits.shape == (2, 5, 3, 6, 7), f"logits (B, n_levels, C, H, W); got {tuple(logits.shape)}"
    want = F.cross_entropy(logits, torch.round(x * 4).long(), reduction="none").sum(dim=(1, 2, 3))
    got = nb.pixelcnn_nll(m, x)
    assert got.shape == (2,) and torch.allclose(got, want, atol=1e-4), (
        "pixelcnn_nll: cross-entropy of the logits against the levels round(x * (n_levels - 1)), summed per image")
    assert any(isinstance(mod, nb.MaskedConv2d) and mod.mask_type == "A" for mod in m.modules()), (
        "the first layer must be a type-A masked convolution")


def test_stage5_causal(nb):
    torch.manual_seed(13)
    m = nb.PixelCNN(2, n_levels=4, hidden_channels=8, n_layers=4)
    H = W = 6
    for (i, j) in [(0, 0), (2, 3), (5, 5), (3, 0)]:
        x = torch.rand(1, 2, H, W, requires_grad=True)
        m(x)[0, :, :, i, j].sum().backward()
        g = x.grad[0].abs().sum(0)  # over channels
        future = torch.zeros(H, W, dtype=torch.bool)
        future.view(-1)[i * W + j:] = True
        assert torch.all(g[future] == 0), (
            f"the logits of pixel ({i}, {j}) depend on input pixels at or after it in raster order: "
            f"{[tuple(v) for v in torch.nonzero(g * future).tolist()][:4]}. only the first layer may be type A; "
            "every later layer is type B")
        if j > 0:
            assert g[i, j - 1] > 0, f"pixel ({i}, {j}) must see its left neighbour"


def test_stage5_sampler(nb):
    torch.manual_seed(14)
    m = nb.PixelCNN(1, n_levels=3, hidden_channels=8, n_layers=2)
    with torch.no_grad():  # make the conditionals far from uniform
        for p in m.parameters():
            p.mul_(4)
    calls = []
    h = m.register_forward_hook(lambda *a: calls.append(1))
    x = nb.PixelCNN.sample(m, 3000, (1, 2), generator=torch.Generator().manual_seed(15))
    h.remove()
    assert x.shape == (3000, 1, 1, 2), f"img_size (1, 2): samples (n, C, 1, 2); got {tuple(x.shape)}"
    assert len(calls) == 2, f"one forward pass per pixel: 2 passes for a 1 x 2 image; counted {len(calls)}"
    assert set(torch.round(x * 2).unique().tolist()) <= {0.0, 1.0, 2.0}, "values are levels / (n_levels - 1)"
    assert torch.allclose(x * 2, torch.round(x * 2), atol=1e-6)
    with torch.no_grad():
        p0 = torch.softmax(m(torch.zeros(1, 1, 1, 2))[0, :, 0, 0, 0], 0)
    f0 = torch.tensor([float((torch.round(x[:, 0, 0, 0] * 2) == v).float().mean()) for v in range(3)])
    assert torch.allclose(f0, p0, atol=0.04), (
        f"pixel (0, 0) must be drawn from the model's first conditional {p0.numpy().round(3)}; got frequencies "
        f"{f0.numpy().round(3)}")
    for v in range(3):  # pixel (0, 1) given pixel (0, 0) = v
        sel = torch.round(x[:, 0, 0, 0] * 2) == v
        if sel.sum() < 300:
            continue
        ctx = torch.zeros(1, 1, 1, 2)
        ctx[0, 0, 0, 0] = v / 2
        with torch.no_grad():
            p1 = torch.softmax(m(ctx)[0, :, 0, 0, 1], 0)
        f1 = torch.tensor([float((torch.round(x[sel, 0, 0, 1] * 2) == w).float().mean()) for w in range(3)])
        assert torch.allclose(f1, p1, atol=0.08), (
            f"pixel (0, 1) given pixel (0, 0) = level {v}: expected {p1.numpy().round(3)}, got {f1.numpy().round(3)}. "
            "fill in each sampled pixel before the next forward pass")
    a = m.sample(20, 3, generator=torch.Generator().manual_seed(1))
    b = m.sample(20, 3, generator=torch.Generator().manual_seed(1))
    assert a.shape == (20, 1, 3, 3) and torch.equal(a, b), "an int img_size means square; seeded samples reproduce"


def test_stage5_learns(nb):
    torch.manual_seed(16)
    x = _rows_constant(128, 6, 1)  # 6 bits of entropy per image: each pixel copies its left neighbour
    m = nb.PixelCNN(1, n_levels=2, hidden_channels=16, n_layers=3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    def run():
        for _ in range(60):
            opt.zero_grad()
            nb.pixelcnn_nll(m, x).mean().backward()
            opt.step()
        return nb.pixelcnn_nll(m, x).mean().item()

    nll = bounded(run, budget=20_000_000, what="training a PixelCNN")
    assert nll < 9.0, (
        f"constant rows of a 6 x 6 binary image: entropy 6 ln 2 = 4.2 nats, uniform 36 ln 2 = 25 nats; after 60 steps "
        f"the nll should be under 9, got {nll:.2f}")


# ---------------------------------------------------------------- added clauses


def test_stage2_seed_changes_masks(nb):
    a, b = nb.MADE(6, [32], seed=1), nb.MADE(6, [32], seed=2)
    ma = [m.mask for m in a.modules() if isinstance(m, nb.MaskedLinear)]
    mb = [m.mask for m in b.modules() if isinstance(m, nb.MaskedLinear)]
    assert any(not torch.equal(p, q) for p, q in zip(ma, mb)), (
        "the hidden degrees are drawn with np.random.default_rng(seed): different seeds give different masks")


def test_stage5_mask_types(nb):
    m = nb.PixelCNN(1, n_levels=2, hidden_channels=8, n_layers=4)
    types = [mod.mask_type for mod in m.modules() if isinstance(mod, nb.MaskedConv2d)]
    assert types.count("A") == 1 and len(types) >= 2 and all(t == "B" for t in types[1:]) and types[0] == "A", (
        f"exactly one type-A layer, the first, and type B everywhere after it; got {types}")


def test_stage5_levels_are_rounded(nb):
    torch.manual_seed(20)
    m = nb.PixelCNN(1, n_levels=256, hidden_channels=4, n_layers=2)
    k = torch.arange(256).view(1, 1, 16, 16)
    x = (k.float() - 0.3).clamp(min=0) / 255  # values a little below their level: round, don't truncate
    want = F.cross_entropy(m(x), k, reduction="none").sum(dim=(1, 2, 3))
    assert torch.allclose(nb.pixelcnn_nll(m, x), want, atol=1e-3), (
        "levels are round(x * (n_levels - 1)): truncating with .long() turns (k - 0.3)/255 into k - 1")


def test_stage5_sampler_channels_and_mode(nb):
    torch.manual_seed(21)
    m = nb.PixelCNN(2, n_levels=3, hidden_channels=8, n_layers=2)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(4)
    m.train()
    x = m.sample(3000, (1, 1), generator=torch.Generator().manual_seed(0))
    assert m.training, "sample must restore the training mode it found"
    with torch.no_grad():
        logits = m.eval()(torch.zeros(1, 2, 1, 1))[0, :, :, 0, 0]  # (L, C)
    for c in range(2):
        p = torch.softmax(logits[:, c], 0)
        f = torch.tensor([float((torch.round(x[:, c, 0, 0] * 2) == v).float().mean()) for v in range(3)])
        assert torch.allclose(f, p, atol=0.04), (
            f"channel {c}: expected frequencies {p.numpy().round(3)}, got {f.numpy().round(3)}. logits are "
            "(n, L, C) at a pixel: move the channel axis before flattening to (n * C, L)")
    m.eval()
    m.sample(2, 2, generator=torch.Generator().manual_seed(1))
    assert not m.training, "an eval-mode model stays in eval mode"
