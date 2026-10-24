"""Checks for 28 · 04 adversarial robustness.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. tiny torch models written here
stand in for an image classifier. FGSM against the sign of the autograd gradient (the loss goes up, clamping,
custom losses, eval mode); PGD for its projection onto both balls, its step size (default 2.5 eps / n_steps,
traced on a 1-D problem by hand), its l2 direction, its random start and generator, and against FGSM;
adversarial training for what it trains on, train/eval modes (batch norm counts the batches) and zero_grad;
robust accuracy and the corruptions against hand computations.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _mlp(d=8, k=3, seed=0, dropout=0.0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Dropout(dropout), nn.Linear(16, k))


def _data(B=6, d=8, k=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return 0.2 + 0.6 * torch.rand(B, d, generator=g), torch.randint(0, k, (B,), generator=g)


def _grad(model, x, y, loss_fn=F.cross_entropy):
    was = model.training
    model.eval()
    xg = x.clone().requires_grad_(True)
    loss_fn(model(xg), y).backward()
    model.zero_grad(set_to_none=True)
    model.train(was)
    return xg.grad


# ---------------------------------------------------------------- stage 1: FGSM


def test_stage1_worked_example(nb):
    lin = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]]))
    got = nb.fgsm(lin, torch.full((1, 3), 0.5), torch.tensor([1]), 0.1)
    assert torch.allclose(got, torch.tensor([[0.4, 0.6, 0.4]])), (
        f"the lesson's worked example: the loss gradient is -(1 - p1) w, its sign (-1, 1, -1), so x' = (0.4, 0.6, 0.4); "
        f"got {got.tolist()}")


def test_stage1_fgsm_formula(nb):
    m = _mlp()
    x, y = _data()
    got = nb.fgsm(m, x, y, 0.1)
    assert all(p.grad is None for p in m.parameters()), (
        "don't leave gradients on the model's parameters (use torch.autograd.grad(loss, x), not loss.backward())")
    want = (x + 0.1 * _grad(m, x, y).sign()).clamp(0, 1)
    assert got.shape == x.shape and torch.allclose(got, want, atol=1e-6), (
        "x_adv = clamp(x + epsilon * sign(grad_x loss)), with the cross-entropy of the logits by default")
    assert not got.requires_grad and torch.equal(x, _data()[0]), "return a detached tensor; leave x untouched"
    assert all(p.grad is None for p in m.parameters()), "don't leave gradients on the model's parameters"
    with torch.no_grad():
        inside = nb.fgsm(m, x, y, 0.1)
    assert torch.allclose(inside, want, atol=1e-6), (
        "fgsm must also work when called inside torch.no_grad(), as evaluation code often does: compute the "
        "gradient under torch.enable_grad()")
    with torch.no_grad():
        before, after = F.cross_entropy(m(x), y), F.cross_entropy(m(nb.fgsm(m, x, y, 0.02)), y)
    assert after > before, (
        f"FGSM ASCENDS the loss: a small step along sign(grad) must raise it ({before.item():.4f} -> "
        f"{after.item():.4f}); check the sign")
    assert torch.allclose(nb.fgsm(m, x, y, 0.0), x), "epsilon = 0 changes nothing"


def test_stage1_clamp_and_loss_fn(nb):
    m = _mlp(seed=2)
    x, y = _data(seed=3)
    x = x.clone()
    x[:, 0] = 0.98
    g = _grad(m, x, y)
    free = nb.fgsm(m, x, y, 0.3, clamp=None)
    assert torch.allclose(free, x + 0.3 * g.sign(), atol=1e-6), "clamp=None: no clamping at all"
    boxed = nb.fgsm(m, x, y, 0.3)
    assert boxed.min() >= 0 and boxed.max() <= 1 and (boxed[:, 0][g[:, 0] > 0] == 1).all(), (
        "the default clamp keeps pixels in [0, 1]")
    wide = nb.fgsm(m, x, y, 0.3, clamp=(-0.1, 1.05))
    assert torch.allclose(wide, (x + 0.3 * g.sign()).clamp(-0.1, 1.05), atol=1e-6), "use the clamp range you're given"

    def target_logit(out, yy):  # an attacker pushing class-0 logits DOWN
        return -out[:, 0].sum()

    got = nb.fgsm(m, x, y, 0.05, loss_fn=target_logit)
    assert torch.allclose(got, (x + 0.05 * _grad(m, x, y, target_logit).sign()).clamp(0, 1), atol=1e-6), (
        "loss_fn(model(x), y) replaces the cross-entropy")


def test_stage1_eval_mode(nb):
    m = _mlp(dropout=0.5).train()
    x, y = _data()
    a, b = nb.fgsm(m, x, y, 0.1), nb.fgsm(m, x, y, 0.1)
    assert torch.equal(a, b) and torch.allclose(a, (x + 0.1 * _grad(m, x, y).sign()).clamp(0, 1)), (
        "attack the model in eval mode (dropout off)")
    assert m.training, "and put it back in train mode afterwards"


# ---------------------------------------------------------------- stage 2: PGD


def _pgd(nb, *args, **kw):
    return bounded(nb.pgd_attack, *args, budget=5_000_000, what="pgd_attack", **kw)


def _adv_step(nb, *args, **kw):
    return bounded(nb.adversarial_training_step, *args, budget=5_000_000, what="adversarial_training_step", **kw)


def _racc(nb, *args, **kw):
    return bounded(nb.robust_accuracy, *args, budget=20_000_000, what="robust_accuracy", **kw)


def test_stage2_projection(nb):
    m = _mlp(seed=4)
    x, y = _data(B=16, seed=5)
    for norm, eps in (("linf", 0.1), ("l2", 0.5)):
        adv = _pgd(nb, m, x, y, eps, n_steps=15, alpha=eps / 2, norm=norm, generator=torch.Generator().manual_seed(0))
        delta = (adv - x).flatten(1)
        size = delta.abs().max(1).values if norm == "linf" else delta.norm(dim=1)
        assert (size <= eps + 1e-5).all(), (
            f"{norm}: every step is projected back onto the epsilon ball around x: max distance {size.max():.4f} > "
            f"epsilon {eps}")
        assert adv.min() >= 0 and adv.max() <= 1, "and into the clamp range"
        assert (size > 0.8 * eps).float().mean() > 0.5, f"{norm}: a 15-step attack should reach the ball's edge"
    edge = x.clone()
    edge[:, :3] = 0.99
    adv = _pgd(nb, m, edge, y, 0.1, n_steps=5, generator=torch.Generator().manual_seed(0))
    assert adv.max() <= 1, "clamping applies after the projection too"


def test_stage2_step_size_by_hand(nb):
    # maximize -(x - 0.33)^2 from x0 = 0 with sign steps of size alpha, no random start, no clamp:
    # alpha = 0.1  -> 0.1, 0.2, 0.3, 0.4        alpha default 2.5 * 1 / 4 = 0.625 -> 0.625, 0, 0.625, 0
    ident = nn.Identity()
    pull = lambda out, yy: -((out - 0.33) ** 2).sum()  # noqa: E731
    x = torch.zeros(1, 1)
    small = _pgd(nb, ident, x, None, 1.0, alpha=0.1, n_steps=4, random_start=False, loss_fn=pull, clamp=None)
    assert np.isclose(small.item(), 0.4, atol=1e-6), (
        f"alpha=0.1, 4 sign steps of gradient ASCENT towards 0.33 from 0: 0.1, 0.2, 0.3, 0.4; got {small.item():.4f}")
    dflt = _pgd(nb, ident, x, None, 1.0, n_steps=4, random_start=False, loss_fn=pull, clamp=None)
    assert np.isclose(dflt.item(), 0.0, atol=1e-6), (
        f"the default alpha is 2.5 * epsilon / n_steps = 0.625: 0.625, 0.0, 0.625, 0.0; got {dflt.item():.4f}")
    few = _pgd(nb, ident, x, None, 1.0, alpha=0.1, n_steps=2, random_start=False, loss_fn=pull, clamp=None)
    assert np.isclose(few.item(), 0.2, atol=1e-6), "n_steps is the number of steps"
    tight = _pgd(nb, ident, x, None, 0.15, alpha=0.1, n_steps=4, random_start=False, loss_fn=pull, clamp=None)
    assert np.isclose(tight.item(), 0.15, atol=1e-6), "epsilon = 0.15 projects every step back to |x - x0| <= 0.15"


def test_stage2_l2_direction_and_fgsm(nb):
    torch.manual_seed(0)
    lin = nn.Linear(8, 3)
    x, y = _data(seed=6)
    margin = lambda out, yy: (out[:, 0] - out[:, 1]).sum()  # noqa: E731  (linear in x: a constant gradient)
    g = _grad(lin, x, y, margin)
    adv = _pgd(nb, lin, x, y, 0.3, n_steps=5, norm="l2", random_start=False, clamp=None, loss_fn=margin)
    want = x + 0.3 * g / g.norm(dim=1, keepdim=True)
    assert torch.allclose(adv, want, atol=1e-5), (
        "l2 PGD on a linear model ends on the ball's edge along the gradient direction g / ||g|| (per example), "
        "not along sign(g)")
    scale = torch.tensor([0.1, 1.0, 5.0, 0.5, 2.0, 20.0])
    scaled = lambda out, yy: ((out[:, 0] - out[:, 1]) * scale).sum()  # noqa: E731  (rows with different gradients)
    small = _pgd(nb, lin, x, y, 10.0, alpha=0.05, n_steps=1, norm="l2", random_start=False, clamp=None, loss_fn=scaled)
    moved = (small - x).norm(dim=1)
    assert torch.allclose(moved, torch.full((6,), 0.05), atol=1e-5), (
        f"an l2 step moves EVERY row by exactly alpha (0.05), whatever its gradient's size: normalise each row's "
        f"gradient by its own norm, not the whole batch's; rows moved {moved.tolist()}")
    m = _mlp(seed=7)
    one = _pgd(nb, m, x, y, 0.1, alpha=0.1, n_steps=1, random_start=False)
    assert torch.allclose(one, nb.fgsm(m, x, y, 0.1), atol=1e-6), (
        "one linf step of size epsilon from x with no random start is exactly FGSM")
    with torch.no_grad():
        lf = F.cross_entropy(m(nb.fgsm(m, x, y, 0.1)), y, reduction="none")
        lp = F.cross_entropy(m(_pgd(nb, m, x, y, 0.1, n_steps=20, random_start=False)), y, reduction="none")
    assert lp.mean() >= lf.mean() - 1e-6, f"20 PGD steps should find at least FGSM's loss: {lp.mean():.4f} < {lf.mean():.4f}"
    with pytest.raises(ValueError):
        _pgd(nb, m, x, y, 0.1, norm="l1")


def test_stage2_random_start(nb):
    m = _mlp(seed=8)
    x, y = _data(B=64, seed=9)
    start = _pgd(nb, m, x, y, 0.1, n_steps=0, clamp=None, generator=torch.Generator().manual_seed(0))
    d = (start - x)
    assert d.abs().max() <= 0.1 + 1e-6 and d.abs().max() > 0.09 and abs(d.mean().item()) < 0.01, (
        "random_start (linf): start from x plus uniform noise in [-epsilon, epsilon] (n_steps=0 returns the start)")
    assert abs(d.std().item() - 0.1 / np.sqrt(3)) < 0.01, "uniform on [-eps, eps] has std eps / sqrt(3)"
    again = _pgd(nb, m, x, y, 0.1, n_steps=0, clamp=None, generator=torch.Generator().manual_seed(0))
    other = _pgd(nb, m, x, y, 0.1, n_steps=0, clamp=None, generator=torch.Generator().manual_seed(1))
    assert torch.equal(start, again) and not torch.equal(start, other), "the random start comes from generator"
    l2 = _pgd(nb, m, x, y, 0.5, n_steps=0, norm="l2", clamp=None, generator=torch.Generator().manual_seed(0))
    r = (l2 - x).norm(dim=1)
    assert (r <= 0.5 + 1e-5).all() and r.std() > 1e-3 and r.min() > 0, (
        "l2 random start: a random point inside the l2 ball (not always on its surface)")
    big = torch.rand(2000, 8, generator=torch.Generator().manual_seed(3))
    rr = (_pgd(nb, m, big, torch.zeros(2000, dtype=torch.long), 0.5, n_steps=0, norm="l2", clamp=None,
               generator=torch.Generator().manual_seed(4)) - big).norm(dim=1) / 0.5
    assert abs(rr.mean().item() - 8 / 9) < 0.01, (
        f"uniform in the l2 ball of dimension d = 8: the radius is epsilon * u^(1/d), so its mean is d/(d+1) = 0.889 "
        f"epsilon; got {rr.mean().item():.3f} epsilon")
    none = _pgd(nb, m, x, y, 0.1, n_steps=0, random_start=False, clamp=None)
    assert torch.equal(none, x), "random_start=False starts at x itself"


def test_stage2_eval_mode(nb):
    m = _mlp(dropout=0.5).train()
    x, y = _data()
    a = _pgd(nb, m, x, y, 0.1, generator=torch.Generator().manual_seed(0))
    b = _pgd(nb, m, x, y, 0.1, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b), "attack in eval mode: with dropout on, two identically seeded attacks would differ"
    assert m.training and not a.requires_grad, "restore the mode; return a detached tensor"


# ---------------------------------------------------------------- stage 3: adversarial training


class _BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc1, self.bn, self.fc2 = nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Linear(16, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


def test_stage3_training_step(nb):
    m = _mlp(seed=10)
    ref = copy.deepcopy(m)
    x, y = _data(B=16, seed=11)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    ropt = torch.optim.SGD(ref.parameters(), lr=0.5)
    for step in range(2):
        loss = _adv_step(nb, m, opt, x, y, epsilon=0.1, n_steps=3,
                                            generator=torch.Generator().manual_seed(step))
        x_adv = _pgd(nb, ref, x, y, 0.1, n_steps=3, generator=torch.Generator().manual_seed(step))
        ref.train()
        rl = F.cross_entropy(ref(x_adv), y)
        ropt.zero_grad()
        rl.backward()
        ropt.step()
        assert isinstance(loss, float) and np.isclose(loss, rl.item(), atol=1e-5), (
            f"step {step}: the returned loss is the cross-entropy on PGD examples made against the CURRENT model "
            f"(n_steps, epsilon and generator passed through): {rl.item():.5f}, got {loss:.5f}")
    for (k, p), q in zip(m.named_parameters(), ref.parameters()):
        assert torch.allclose(p, q, atol=1e-5), (
            f"after two steps {k} differs from a reference update: zero the gradients before each backward")
    m0, ref0 = _mlp(seed=12), _mlp(seed=12)
    o0, r0 = torch.optim.SGD(m0.parameters(), lr=0.5), torch.optim.SGD(ref0.parameters(), lr=0.5)
    _adv_step(nb, m0, o0, x, y, epsilon=0.0)
    F.cross_entropy(ref0(x), y).backward()
    r0.step()
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(m0.parameters(), ref0.parameters())), (
        "epsilon = 0 is ordinary training on the clean batch")


def test_stage3_modes(nb):
    m = _BNNet()
    x, y = _data(B=32, seed=12)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m.eval()  # handed over in eval mode: the step itself must switch to train mode for the update
    _adv_step(nb, m, opt, x, y, epsilon=0.1, n_steps=4, generator=torch.Generator().manual_seed(0))
    assert m.bn.num_batches_tracked.item() == 1, (
        f"batch norm counted {m.bn.num_batches_tracked.item()} training-mode forward passes: craft the attack in eval "
        "mode (so the attack doesn't update running statistics) and take the update in train mode (exactly one)")
    assert m.training, "the model is left in train mode after a training step"


def test_stage3_robust_accuracy(nb):
    m = _mlp(seed=13, dropout=0.5).train()
    x, y = _data(B=50, seed=14)
    m.eval()
    with torch.no_grad():
        clean = (m(x).argmax(1) == y).float().mean().item()
        zeros = (m(torch.zeros_like(x)).argmax(1) == y).float().mean().item()
    m.train()
    got = _racc(nb, m, x, y, batch_size=7)
    assert isinstance(got, float) and np.isclose(got, clean), (
        f"attack=None: clean accuracy, in eval mode, over every batch (50 rows in batches of 7): {clean}, got {got}")
    seen = []

    def zero_attack(model, xb, yb):
        seen.append(len(xb))
        return torch.zeros_like(xb)

    assert np.isclose(_racc(nb, m, x, y, zero_attack, batch_size=7), zeros), (
        "the accuracy is measured on attack(model, xb, yb)")
    assert seen == [7] * 7 + [1], f"attack is called once per batch of batch_size rows: {seen}"
    assert m.training, "restore the model's mode"
    pgd = _racc(nb, m, x, y, lambda mm, xb, yb: nb.pgd_attack(mm, xb, yb, 0.3, n_steps=10,
                                                                       generator=torch.Generator().manual_seed(0)))
    assert pgd <= clean + 1e-6, "an attack can't raise accuracy above clean (on average)"


# ---------------------------------------------------------------- stage 4: corruptions


def _imgs(B=4, seed=15):
    return torch.rand(B, 1, 6, 6, generator=torch.Generator().manual_seed(seed)) * 0.8 + 0.1


def test_stage4_corruptions(nb):
    x = _imgs()
    for kind in ("gaussian_noise", "contrast", "brightness", "shift"):
        assert torch.allclose(nb.corrupt(x, kind, 0, generator=torch.Generator().manual_seed(0)), x), (
            f"{kind}: severity 0 is the identity")
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    assert torch.allclose(nb.corrupt(x, "contrast", 2), (mean + (x - mean) * 0.7).clamp(0, 1), atol=1e-6), (
        "contrast: each image's mean + (x - mean)(1 - 0.15 severity)")
    assert torch.allclose(nb.corrupt(x, "brightness", 3), (x + 0.3).clamp(0, 1), atol=1e-6), (
        "brightness: x + 0.1 severity, clamped to [0, 1]")
    sh = nb.corrupt(x, "shift", 2)
    assert torch.equal(sh[..., 2:], x[..., :-2]) and torch.all(sh[..., :2] == 0), (
        "shift: move the image right by `severity` pixels along the last axis, filling with zeros")
    big = torch.full((2000, 1, 4, 4), 0.5)
    noisy = nb.corrupt(big, "gaussian_noise", 2, generator=torch.Generator().manual_seed(1))
    assert abs((noisy - 0.5).std().item() - 0.16) < 0.005 and abs((noisy - 0.5).mean().item()) < 0.004, (
        "gaussian_noise: add N(0, (0.08 severity)^2) noise (std 0.16 at severity 2)")
    again = nb.corrupt(big, "gaussian_noise", 2, generator=torch.Generator().manual_seed(1))
    assert torch.equal(noisy, again), "noise comes from the generator"
    assert nb.corrupt(x, "gaussian_noise", 5, generator=torch.Generator().manual_seed(2)).max() <= 1, "clamp to [0, 1]"
    assert torch.equal(x, _imgs()), "return a corrupted copy; don't modify x"
    with pytest.raises(ValueError):
        nb.corrupt(x, "fog", 1)
    with pytest.raises(ValueError):
        nb.corrupt(x, "brightness", 6)


def test_stage4_corruption_accuracy(nb):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(36, 3)).train()
    x = _imgs(B=40, seed=16)
    y = torch.randint(0, 3, (40,), generator=torch.Generator().manual_seed(17))
    got = nb.corruption_accuracy(m, x, y, ["contrast", "shift"], [0, 1, 3], generator=torch.Generator().manual_seed(0))
    assert set(got) == {"contrast", "shift"} and got["shift"].shape == (3,), "one array per kind, one entry per severity"
    m.eval()
    with torch.no_grad():
        for kind in ("contrast", "shift"):
            want = [(m(nb.corrupt(x, kind, s)).argmax(1) == y).float().mean().item() for s in (0, 1, 3)]
            assert np.allclose(got[kind], want), f"{kind}: accuracy at each severity, in eval mode: {want}, got {got[kind]}"
    m.train()
    nb.corruption_accuracy(m, x, y, ["brightness"], [1])
    assert m.training, "restore the model's mode"
