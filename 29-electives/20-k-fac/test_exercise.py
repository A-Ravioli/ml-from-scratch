"""Checks for exercise.py (K-FAC).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Kronecker solves
are checked against explicit `torch.kron` matrices; factors against hand formulas and against
the exact Fisher block, which K-FAC reproduces exactly for a single example; the step and the
optimizer against explicit recomputation (exact-expectation mode is deterministic), plus a
small learning check over several seeds. Parameter vectors of a layer are laid out as the
(out, in + 1) matrix [W | b], flattened row by row.
"""
import copy
import math

import torch
from torch import nn

torch.set_num_threads(1)
D = torch.float64


def _mlp(seed, d=3, h=4, k=3):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, k)).to(D)


def _data(seed, n=8, d=3, k=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=D), torch.randint(0, k, (n,), generator=g)


def _wb(layer):
    return torch.cat([layer.weight.detach(), layer.bias.detach().unsqueeze(1)], 1)


def _kron_apply(grad, A, G):
    """(G kron A)^-1 vec_row(grad), as a matrix: the explicit, slow oracle."""
    v = torch.linalg.solve(torch.kron(G, A), grad.reshape(-1))
    return v.reshape(grad.shape)


def _exact_block(model, X, layer):
    """the exact true-Fisher block of one Linear layer, in row-major [W | b] layout."""
    F = 0
    for n in range(X.shape[0]):
        logp = torch.log_softmax(model(X[n:n + 1]), -1)[0]
        for c in range(logp.shape[0]):
            gw, gb = torch.autograd.grad(logp[c], [layer.weight, layer.bias], retain_graph=True)
            v = torch.cat([gw, gb.unsqueeze(1)], 1).reshape(-1)
            F = F + logp[c].exp().item() * torch.outer(v, v)
    return F / X.shape[0]


def _layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


# ---------------------------------------------------------------- stage 1: Kronecker inverses


def test_stage1_hand_example(nb):
    A = torch.diag(torch.tensor([1.0, 4.0], dtype=D))
    G = torch.diag(torch.tensor([2.0, 1.0], dtype=D))
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=D)
    got = nb.kron_precondition(grad, *nb.damped_inverses(A, G, 0.0))
    want = torch.tensor([[0.5, 0.25], [3.0, 1.0]], dtype=D)
    assert torch.allclose(got, want, atol=1e-12), (
        f"G^-1 grad A^-1 with G = diag(2, 1), A = diag(1, 4): rows are divided by (2, 1), columns by (1, 4), giving "
        f"[[0.5, 0.25], [3, 1]]; got\n{got}")


def test_stage1_undamped_is_the_kronecker_inverse(nb):
    g = torch.Generator().manual_seed(0)
    Ba, Bg = torch.randn(4, 4, generator=g, dtype=D), torch.randn(3, 3, generator=g, dtype=D)
    A, G = Ba @ Ba.T + 0.5 * torch.eye(4, dtype=D), Bg @ Bg.T + 0.5 * torch.eye(3, dtype=D)
    grad = torch.randn(3, 4, generator=g, dtype=D)
    A_inv, G_inv = nb.damped_inverses(A, G, 0.0)
    assert tuple(A_inv.shape) == (4, 4) and tuple(G_inv.shape) == (3, 3), "A_inv is (in+1, in+1), G_inv is (out, out)"
    got = nb.kron_precondition(grad, A_inv, G_inv)
    want = _kron_apply(grad, A, G)
    assert tuple(got.shape) == (3, 4), f"the preconditioned gradient has the gradient's shape (3, 4); got {tuple(got.shape)}"
    assert torch.allclose(got, want, atol=1e-10), (
        "with damping 0, G_inv @ grad @ A_inv must equal (G kron A)^-1 applied to the row-major flattened gradient. "
        f"max difference {(got - want).abs().max():.3g}. if it's large, check the order (G on the left, A on the "
        "right) and that you didn't transpose the gradient")


def test_stage1_factored_damping_hand_example(nb):
    A = 4.0 * torch.eye(3, dtype=D)  # tr(A)/3 = 4
    G = torch.eye(2, dtype=D)        # tr(G)/2 = 1, so pi = sqrt(4 / 1) = 2
    A_inv, G_inv = nb.damped_inverses(A, G, 0.01)  # sqrt(damping) = 0.1
    assert torch.allclose(A_inv, torch.eye(3, dtype=D) / 4.2, atol=1e-12), (
        f"pi = sqrt((tr A / 3) / (tr G / 2)) = 2, so A gets pi * sqrt(0.01) = 0.2 added: A_inv = I / 4.2. "
        f"got diagonal {torch.diagonal(A_inv).tolist()}. if A got 0.05 and G 0.2, pi is inverted (each factor's "
        "damping should grow with that factor's own scale)")
    assert torch.allclose(G_inv, torch.eye(2, dtype=D) / 1.05, atol=1e-12), (
        f"G gets sqrt(0.01) / pi = 0.05 added: G_inv = I / 1.05. got diagonal {torch.diagonal(G_inv).tolist()}")


def test_stage1_damped_matches_explicit(nb):
    g = torch.Generator().manual_seed(1)
    Ba, Bg = torch.randn(5, 5, generator=g, dtype=D), torch.randn(2, 2, generator=g, dtype=D)
    A, G = Ba @ Ba.T, 3.0 * Bg @ Bg.T
    A0, G0 = A.clone(), G.clone()
    grad = torch.randn(2, 5, generator=g, dtype=D)
    lam = 0.3
    pi = math.sqrt((torch.trace(A) / 5).item() / (torch.trace(G) / 2).item())
    Ad, Gd = A + pi * math.sqrt(lam) * torch.eye(5, dtype=D), G + math.sqrt(lam) / pi * torch.eye(2, dtype=D)
    got = nb.kron_precondition(grad, *nb.damped_inverses(A, G, lam))
    assert torch.allclose(got, _kron_apply(grad, Ad, Gd), atol=1e-10), (
        "the damped preconditioner must equal ((G + sqrt(l)/pi I) kron (A + pi sqrt(l) I))^-1 applied to the gradient")
    assert torch.equal(A, A0) and torch.equal(G, G0), "don't modify the factors in place when you add the damping"


def test_stage1_float32(nb):
    A, G = torch.eye(3) * 2, torch.eye(2)
    A_inv, G_inv = nb.damped_inverses(A, G, 1e-2)
    out = nb.kron_precondition(torch.ones(2, 3), A_inv, G_inv)
    assert A_inv.dtype == torch.float32 and out.dtype == torch.float32, (
        "work in the dtype of the inputs (create identity matrices with dtype=A.dtype): the lab trains in float32")


# ---------------------------------------------------------------- stage 2: the factors


def test_stage2_shapes_and_order(nb):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 2)).to(D)
    X, _ = _data(0, d=3, k=2)
    f = nb.kfac_factors(model, X)
    assert isinstance(f, list) and len(f) == 3, f"one (A, G) pair per nn.Linear, in order; got {type(f).__name__} of length {len(f)}"
    for (A, G), (din, dout) in zip(f, [(3, 5), (5, 4), (4, 2)]):
        assert tuple(A.shape) == (din + 1, din + 1) and tuple(G.shape) == (dout, dout), (
            f"a Linear({din}, {dout}) layer has A of shape ({din + 1}, {din + 1}) (inputs plus a 1 for the bias) and G "
            f"of shape ({dout}, {dout}); got {tuple(A.shape)} and {tuple(G.shape)}")
    for layer in _layers(model):
        assert len(layer._forward_hooks) == 0, "remove your forward hooks after use (keep the handles and call .remove())"
        assert layer.weight.grad is None, "use torch.autograd.grad, not .backward(): don't leave gradients in .grad"


def test_stage2_hand_formulas(nb):
    model = _mlp(1)
    X, _ = _data(1, n=10)
    (A1, G1), (A2, G2) = nb.kfac_factors(model, X)
    xa = torch.cat([X, torch.ones(10, 1, dtype=D)], 1)
    assert torch.allclose(A1, xa.T @ xa / 10, atol=1e-12), (
        "A of the first layer is the mean of [x, 1][x, 1]^T over the batch (the bias is a weight on a constant 1, "
        "appended LAST)")
    with torch.no_grad():
        h = torch.tanh(model[0](X))
        p = torch.softmax(model(X), -1)
    ha = torch.cat([h, torch.ones(10, 1, dtype=D)], 1)
    assert torch.allclose(A2, ha.T @ ha / 10, atol=1e-12), "A of the second layer uses its INPUTS: the tanh activations"
    want_G2 = (torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)).mean(0)
    assert torch.allclose(G2, want_G2, atol=1e-12), (
        "for the output layer, d log p(y) / d logits = e_y - p, so the exact expectation over y ~ p gives "
        "G = mean of diag(p) - p p^T. check the 1/N and that you weight each class by p(y | x_n)")


def test_stage2_single_example_is_exact(nb):
    model = _mlp(2)
    X, _ = _data(2, n=1)
    factors = nb.kfac_factors(model, X)
    for layer, (A, G) in zip(_layers(model), factors):
        exact = _exact_block(model, X, layer)
        approx = torch.kron(G, A)
        assert torch.allclose(approx, exact, atol=1e-10), (
            f"with one example, G kron A must equal the layer's exact Fisher block (row-major [W | b] layout); max "
            f"difference {(approx - exact).abs().max():.3g}. check that g is the gradient of log p(y | x) with respect "
            "to the layer's OUTPUT (pre-activation), and A uses the layer's INPUT")


def test_stage2_hidden_G_matches_loop(nb):
    model = _mlp(3)
    X, _ = _data(3, n=6)
    _, G1 = nb.kfac_factors(model, X)[0]
    want = torch.zeros(4, 4, dtype=D)
    for n in range(6):
        s = model[0](X[n:n + 1])
        logp = torch.log_softmax(model[2](torch.tanh(s)), -1)[0]
        for c in range(3):
            (g,) = torch.autograd.grad(logp[c], s, retain_graph=True)
            want += logp[c].exp().item() * torch.outer(g[0], g[0]) / 6
    assert torch.allclose(G1, want, atol=1e-12), (
        f"G of the hidden layer differs from (1/N) sum_n sum_y p(y|x_n) g g^T by {(G1 - want).abs().max():.3g}")


def test_stage2_given_labels(nb):
    model = _mlp(4)
    X, y = _data(4, n=9)
    (_, _), (_, G2) = nb.kfac_factors(model, X, y)
    with torch.no_grad():
        r = nn.functional.one_hot(y, 3).to(D) - torch.softmax(model(X), -1)
    assert torch.allclose(G2, r.T @ r / 9, atol=1e-12), (
        "with labels given, G uses only those labels: for the output layer, the mean of (e_y - p)(e_y - p)^T")
    assert all(p.grad is None for p in model.parameters()), (
        "with labels given too, use torch.autograd.grad, not .backward(): don't leave gradients in .grad")


# ---------------------------------------------------------------- stage 3: one step


def _grads(model, X, y):
    model.zero_grad()
    nn.functional.cross_entropy(model(X), y).backward()
    return [torch.cat([m.weight.grad, m.bias.grad.unsqueeze(1)], 1).clone() for m in _layers(model)]


def _damped(A, G, lam):
    pi = math.sqrt((torch.trace(A) / A.shape[0]).item() / (torch.trace(G) / G.shape[0]).item())
    return A + pi * math.sqrt(lam) * torch.eye(A.shape[0], dtype=A.dtype), G + math.sqrt(lam) / pi * torch.eye(G.shape[0], dtype=G.dtype)


def test_stage3_update_matches_explicit(nb):
    model = _mlp(5)
    X, y = _data(5, n=12)
    factors = [(A.clone(), G.clone()) for A, G in nb.kfac_factors(model, X)]
    grads = _grads(model, X, y)
    before = [_wb(m) for m in _layers(model)]
    nu = nb.kfac_step(model, factors, lr=0.3, damping=0.05)
    assert nu == 1.0, f"without kl_clip the scale is 1.0; got {nu}"
    for m, b, g, (A, G) in zip(_layers(model), before, grads, factors):
        want = b - 0.3 * _kron_apply(g, *_damped(A, G, 0.05))
        assert torch.allclose(_wb(m), want, atol=1e-10), (
            "each layer's [W | b] must move by -lr * (damped G kron A)^-1 [dW | db], using the gradients already in "
            f".grad; max difference {(_wb(m) - want).abs().max():.3g}. weights get the first `in` columns, the "
            "bias the last")


def test_stage3_kl_clip(nb):
    model = _mlp(6)
    X, y = _data(6, n=12)
    factors = nb.kfac_factors(model, X)
    grads = _grads(model, X, y)
    ups = [_kron_apply(g, *_damped(A, G, 0.05)) for g, (A, G) in zip(grads, factors)]
    vg = sum(float((u * g).sum()) for u, g in zip(ups, grads))
    m2 = copy.deepcopy(model)
    before = [_wb(m) for m in _layers(model)]
    kl = 1e-3 * 0.5**2 * vg  # forces nu = sqrt(1e-3)
    nu = nb.kfac_step(model, factors, lr=0.5, damping=0.05, kl_clip=kl)
    assert abs(nu - math.sqrt(1e-3)) < 1e-9, (
        f"nu = min(1, sqrt(kl_clip / (lr^2 sum_l <U_l, grad_l>))) = {math.sqrt(1e-3):.6f} here; got {nu}")
    for m, b, u in zip(_layers(model), before, ups):
        assert torch.allclose(_wb(m), b - 0.5 * nu * u, atol=1e-10), "the update must be scaled by nu"
    m2.zero_grad()
    nn.functional.cross_entropy(m2(X), y).backward()
    assert nb.kfac_step(m2, factors, lr=0.5, damping=0.05, kl_clip=1e9) == 1.0, "a loose kl_clip leaves nu = 1"


# ---------------------------------------------------------------- stage 4: the optimizer


def _manual_update(model, factors, grads, lam, lr, kl, velocity=None, mu=0.0):
    """one clipped (and optionally momentum) K-FAC update, applied in place; returns (nu, new velocity)."""
    ups = [_kron_apply(g, *_damped(A, G, lam)) for g, (A, G) in zip(grads, factors)]
    vg = sum(float((u * g).sum()) for u, g in zip(ups, grads))
    nu = min(1.0, math.sqrt(kl / (lr ** 2 * vg)))
    vel = [nu * u if velocity is None else mu * v + nu * u for u, v in zip(ups, velocity or ups)]
    with torch.no_grad():
        for m, v in zip(_layers(model), vel):
            m.weight -= lr * v[:, :-1]
            m.bias -= lr * v[:, -1]
    return nu, vel


def test_stage4_first_step_is_kfac_step(nb):
    model = _mlp(7)
    X, y = _data(7, n=16)
    ref = copy.deepcopy(model)
    opt = nb.KFAC(model, lr=0.2, damping=0.03, momentum=0.0, inv_every=1, kl_clip=None, fisher="exact")
    assert opt.steps == 0, "a fresh optimizer has steps = 0"
    with torch.no_grad():
        want_loss = nn.functional.cross_entropy(ref(X), y).item()
    loss = opt.step(X, y)
    assert isinstance(loss, float) and abs(loss - want_loss) < 1e-12, (
        f"step returns the minibatch loss BEFORE the update, as a float ({want_loss:.6f}); got {loss!r}")
    factors = [(A.clone(), G.clone()) for A, G in nb.kfac_factors(ref, X)]
    ref.zero_grad()
    nn.functional.cross_entropy(ref(X), y).backward()
    nb.kfac_step(ref, factors, lr=0.2, damping=0.03)
    for a, b in zip(_layers(model), _layers(ref)):
        assert torch.allclose(_wb(a), _wb(b), atol=1e-10), (
            "with fisher='exact', momentum 0 and no KL clip, the first step must equal kfac_step with the factors "
            "of the current minibatch (no running average yet: the first factors are used as they are)")
    assert opt.steps == 1


def test_stage4_running_average_and_momentum(nb):
    model = _mlp(8)
    X, y = _data(8, n=16)
    ref = copy.deepcopy(model)
    e, lr, mu, lam = 0.7, 0.2, 0.5, 0.03
    opt = nb.KFAC(model, lr=lr, damping=lam, momentum=mu, ema_decay=e, inv_every=1, kl_clip=None, fisher="exact")
    f0 = nb.kfac_factors(ref, X)
    g0 = _grads(ref, X, y)
    u0 = [_kron_apply(g, *_damped(A, G, lam)) for g, (A, G) in zip(g0, f0)]
    with torch.no_grad():
        for m, u in zip(_layers(ref), u0):
            m.weight -= lr * u[:, :-1]
            m.bias -= lr * u[:, -1]
    f1 = [(e * A0 + (1 - e) * A1, e * G0 + (1 - e) * G1) for (A0, G0), (A1, G1) in zip(f0, nb.kfac_factors(ref, X))]
    g1 = _grads(ref, X, y)
    v1 = [mu * u + _kron_apply(g, *_damped(A, G, lam)) for u, g, (A, G) in zip(u0, g1, f1)]
    with torch.no_grad():
        for m, v in zip(_layers(ref), v1):
            m.weight -= lr * v[:, :-1]
            m.bias -= lr * v[:, -1]
    opt.step(X, y)
    opt.step(X, y)
    for (A, G), (Aw, Gw) in zip(opt.factors, f1):
        assert torch.allclose(A, Aw, atol=1e-12) and torch.allclose(G, Gw, atol=1e-12), (
            "after the second step, self.factors must be ema_decay * old + (1 - ema_decay) * new for both A and G")
    for a, b in zip(_layers(model), _layers(ref)):
        assert torch.allclose(_wb(a), _wb(b), atol=1e-10), (
            "two steps with momentum: v <- momentum v + nu U, then [W | b] <- [W | b] - lr v, with U preconditioned by "
            "the running-average factors. your parameters differ from that")


def test_stage4_kl_clip_first_step(nb):
    model = _mlp(14)
    X, y = _data(14, n=16)
    ref = copy.deepcopy(model)
    lr, lam = 0.5, 0.03
    factors = nb.kfac_factors(ref, X)
    grads = _grads(ref, X, y)
    ups = [_kron_apply(g, *_damped(A, G, lam)) for g, (A, G) in zip(grads, factors)]
    vg = sum(float((u * g).sum()) for u, g in zip(ups, grads))
    kl = 0.01 * lr ** 2 * vg  # forces nu = 0.1
    nu, _ = _manual_update(ref, factors, grads, lam, lr, kl)
    assert abs(nu - 0.1) < 1e-9
    opt = nb.KFAC(model, lr=lr, damping=lam, momentum=0.0, inv_every=1, kl_clip=kl, fisher="exact")
    opt.step(X, y)
    for a, b in zip(_layers(model), _layers(ref)):
        assert torch.allclose(_wb(a), _wb(b), atol=1e-10), (
            "with a binding kl_clip the first step must be scaled by nu = min(1, sqrt(kl_clip / (lr^2 sum <U, grad>))) "
            "= 0.1 here, exactly as in kfac_step. did you apply the clip (with lr squared)?")


def test_stage4_kl_clip_with_momentum(nb):
    model = _mlp(15)
    X, y = _data(15, n=16)
    ref = copy.deepcopy(model)
    lr, lam, mu, kl = 0.5, 0.03, 0.6, 1e-4
    f0 = nb.kfac_factors(ref, X)
    nu0, v = _manual_update(ref, f0, _grads(ref, X, y), lam, lr, kl)
    f1 = nb.kfac_factors(ref, X)  # ema_decay = 0: the factors are replaced each step
    nu1, _ = _manual_update(ref, f1, _grads(ref, X, y), lam, lr, kl, velocity=v, mu=mu)
    assert nu0 < 1 and nu1 < 1, "test setup: the clip should bind on both steps"
    opt = nb.KFAC(model, lr=lr, damping=lam, momentum=mu, ema_decay=0.0, inv_every=1, kl_clip=kl, fisher="exact")
    opt.step(X, y)
    opt.step(X, y)
    for a, b in zip(_layers(model), _layers(ref)):
        assert torch.allclose(_wb(a), _wb(b), atol=1e-10), (
            "with momentum and an active KL clip: v <- momentum v + nu U (nu computed from this step's U and "
            "gradient, applied to U, not to the velocity), then [W | b] <- [W | b] - lr v")


def test_stage4_default_seed_is_zero(nb):
    torch.manual_seed(16)
    big = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 3)).to(D)
    Xb = torch.randn(64, 2, generator=torch.Generator().manual_seed(16), dtype=D)
    yb = torch.zeros(64, dtype=torch.long)
    o1, o2 = nb.KFAC(copy.deepcopy(big), fisher="sample"), nb.KFAC(copy.deepcopy(big), fisher="sample", seed=0)
    torch.manual_seed(123)
    o1.step(Xb, yb)
    torch.manual_seed(456)
    o2.step(Xb, yb)
    assert all(torch.equal(a[1], b[1]) for a, b in zip(o1.factors, o2.factors)), (
        "seed=None must behave like seed=0 (a torch.Generator seeded with 0), independent of the global torch seed")


def test_stage4_inverse_and_statistics_schedule(nb):
    model = _mlp(9)
    X, y = _data(9, n=16)
    opt = nb.KFAC(model, lr=0.1, damping=0.03, inv_every=3, stat_every=2, fisher="exact")
    snaps = []
    for _ in range(4):
        opt.step(X, y)
        snaps.append(([A.clone() for A, _ in opt.factors], [Ai.clone() for Ai, _ in opt.inverses]))
    same = lambda a, b: all(torch.equal(x, z) for x, z in zip(a, b))  # noqa: E731
    assert same(snaps[0][1], snaps[1][1]) and same(snaps[1][1], snaps[2][1]), (
        "with inv_every=3 the inverses computed at step 0 must be reused at steps 1 and 2 (self.inverses unchanged)")
    assert not same(snaps[2][1], snaps[3][1]), "with inv_every=3 the inverses must be recomputed at step 3"
    assert same(snaps[0][0], snaps[1][0]) and not same(snaps[1][0], snaps[2][0]), (
        "with stat_every=2 the factors are updated at steps 0, 2, 4, ... and left alone in between")


def test_stage4_label_modes(nb):
    try:
        nb.KFAC(_mlp(0), fisher="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("fisher must be 'sample', 'exact' or 'empirical': raise ValueError for anything else")
    # empirical: the observed labels
    model = _mlp(10)
    X, y = _data(10, n=12)
    with torch.no_grad():
        r = nn.functional.one_hot(y, 3).to(D) - torch.softmax(model(X), -1)
    opt = nb.KFAC(model, fisher="empirical", kl_clip=None)
    opt.step(X, y)
    assert torch.allclose(opt.factors[1][1], r.T @ r / 12, atol=1e-12), (
        "fisher='empirical' computes the factors with the minibatch's own labels")
    # sample: reproducible with a seed, and unbiased for the exact G
    torch.manual_seed(11)
    big = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 3)).to(D)
    Xb = torch.randn(4000, 2, generator=torch.Generator().manual_seed(11), dtype=D)
    yb = torch.zeros(4000, dtype=torch.long)
    o1, o2 = nb.KFAC(copy.deepcopy(big), fisher="sample", seed=3), nb.KFAC(copy.deepcopy(big), fisher="sample", seed=3)
    o1.step(Xb, yb)
    o2.step(Xb, yb)
    assert all(torch.equal(a[1], b[1]) for a, b in zip(o1.factors, o2.factors)), (
        "fisher='sample' with the same seed must give the same factors: draw the labels with a torch.Generator "
        "seeded from `seed`")
    exact = nb.kfac_factors(big, Xb)
    for (_, Gs), (_, Ge) in zip(o1.factors, exact):
        rel = ((Gs - Ge).norm() / Ge.norm()).item()
        assert rel < 0.15, (
            f"labels sampled from the model give an unbiased estimate of G; with 4,000 examples it should be within "
            f"15% of the exact G, but it's {rel:.0%} off. sample y ~ softmax(model(X)), not the data labels")


def test_stage4_trains(nb):
    finals = []
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        centers = 2.5 * torch.randn(4, 5, generator=g, dtype=D)
        yy = torch.randint(0, 4, (256,), generator=g)
        XX = centers[yy] + torch.randn(256, 5, generator=g, dtype=D)
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 4)).to(D)
        opt = nb.KFAC(model, lr=0.3, damping=1e-2, inv_every=5, seed=seed)
        for _ in range(25):
            opt.step(XX, yy)
        with torch.no_grad():
            finals.append(nn.functional.cross_entropy(model(XX), yy).item())
    finals.sort()
    assert finals[2] < 0.15, (
        f"25 full-batch K-FAC steps (lr 0.3, damping 1e-2) on four separated blobs should reach a median loss "
        f"below 0.15 over 5 seeds; got {[f'{v:.3f}' for v in finals]}")
