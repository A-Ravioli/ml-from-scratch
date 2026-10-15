"""Checks for exercise.py (regularization approaches: EWC and SI).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the Fisher is
checked against a brute-force loop over examples and classes computed here with autograd (exact,
empirical, and restricted to a task's classes); EWC's penalty against its formula with that
brute-force Fisher, including its gradient, the online variant and the sample count; SI against the
lesson's hand-computed path integral and against the loss decrease of gradient descent on a
separable quadratic, where the path integral is known exactly.
"""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _mlp(seed=0, d=3, h=4, c=3):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, c))


def _brute_fisher(model, X, y=None, classes=None, mode="exact"):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    out = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    for i in range(len(X)):
        logits = model(X[i:i + 1])[0]
        if classes is not None:
            logits = logits[list(classes)]
        logp = F.log_softmax(logits, -1)
        if mode == "exact":
            pairs = [(c, logp[c].exp().item()) for c in range(len(logp))]
        else:
            lab = int(y[i]) if classes is None else list(classes).index(int(y[i]))
            pairs = [(lab, 1.0)]
        for c, w in pairs:
            grads = torch.autograd.grad(logp[c], [dict(model.named_parameters())[n] for n in names],
                                        retain_graph=True)
            for n, g in zip(names, grads):
                out[n] += w * g ** 2
    return {n: v / len(X) for n, v in out.items()}


def _data(n=7, d=3, seed=1, c=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randint(0, c, (n,), generator=g)


def _close(got, want, what, tol=1e-5):
    assert isinstance(got, dict), f"{what}: expected a dict name -> tensor, got {type(got).__name__}"
    assert set(got) == set(want), f"{what}: keys should be the trainable parameter names {sorted(want)}, got {sorted(got)}"
    for n in want:
        assert tuple(got[n].shape) == tuple(want[n].shape), f"{what}: {n} should have shape {tuple(want[n].shape)}"
        err = float((got[n] - want[n]).abs().max())
        scale = float(want[n].abs().max())
        assert err <= tol * max(1.0, scale) + 1e-9, (
            f"{what}: entry {n} is off by {err:.3g} (largest true value {scale:.3g}). square the gradient of "
            "*each example's* log-probability, then average; the square of the averaged gradient is much smaller.")


# ---------------------------------------------------------------- stage 1: the diagonal Fisher


def test_stage1_exact_fisher_matches_brute_force(nb):
    model = _mlp()
    X, y = _data()
    got = nb.fisher_diagonal(model, X)
    _close(got, _brute_fisher(model, X), "exact Fisher")
    for n, v in got.items():
        assert not v.requires_grad, f"{n}: return detached tensors (no autograd graph)"
        assert (v >= 0).all(), f"{n}: a mean of squares can't be negative"


def test_stage1_empirical_and_restricted(nb):
    model = _mlp(seed=2, c=5)
    X, y = _data(n=9, seed=3, c=5)
    _close(nb.fisher_diagonal(model, X, y=y, mode="empirical"), _brute_fisher(model, X, y=y, mode="empirical"),
           "empirical Fisher (squared gradient of log p(true label))")
    y2 = torch.tensor([3, 1, 1, 3, 3, 1, 3, 1, 1])
    _close(nb.fisher_diagonal(model, X, classes=(1, 3)), _brute_fisher(model, X, classes=(1, 3)),
           "exact Fisher restricted to classes (1, 3): softmax over those two logits only")
    _close(nb.fisher_diagonal(model, X, y=y2, classes=(1, 3), mode="empirical"),
           _brute_fisher(model, X, y=y2, classes=(1, 3), mode="empirical"),
           "empirical Fisher restricted to classes (1, 3)")
    with pytest.raises(ValueError):
        nb.fisher_diagonal(model, X, mode="empirical")
    with pytest.raises(ValueError):
        nb.fisher_diagonal(model, X, mode="true")


def test_stage1_frozen_parameters_and_untouched_model(nb):
    model = _mlp(seed=4)
    model[0].weight.requires_grad_(False)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    X, _ = _data(seed=5)
    got = nb.fisher_diagonal(model, X)
    assert "0.weight" not in got, "frozen parameters (requires_grad=False) get no Fisher entry"
    _close(got, _brute_fisher(model, X), "Fisher with a frozen layer")
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), f"computing the Fisher must not change the parameters ({n} changed)"


# ---------------------------------------------------------------- stage 2: EWC


def _task(n=12, seed=6, c=3, classes=None):
    X, y = _data(n=n, seed=seed, c=c)
    return {"X": X, "y": y, "classes": tuple(range(c)) if classes is None else classes}


def _shift(model, eps, seed=7):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(eps * torch.randn(p.shape, generator=g))


def test_stage2_penalty_formula_and_gradient(nb):
    model = _mlp(seed=8)
    task = _task()
    ewc = nb.EWC(50.0, n_samples=None)
    p0 = ewc.penalty(model)
    assert p0.detach().item() == 0.0, f"before any consolidate the penalty is 0, got {p0.detach().item()}"
    ewc.consolidate(model, task)
    assert abs(ewc.penalty(model).item()) < 1e-10, "right after consolidate, at the anchor, the penalty is 0"
    anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
    fisher = _brute_fisher(model, task["X"])
    _shift(model, 0.3)
    want = 25.0 * sum((fisher[n] * (p.detach() - anchor[n]) ** 2).sum() for n, p in model.named_parameters())
    got = ewc.penalty(model)
    assert torch.is_tensor(got) and got.requires_grad, "penalty(model) must be a differentiable scalar tensor"
    assert abs(got.detach().item() - want.detach().item()) < 1e-5 * max(1.0, want.detach().item()), (
        f"penalty should be (lam / 2) * sum_i F_i (theta_i - anchor_i)^2 = {want.detach().item():.6g}, got {got.detach().item():.6g}. "
        "(the anchor must be a *copy* of the weights at consolidate time, and don't forget the 1/2)")
    got.backward()
    for n, p in model.named_parameters():
        g_want = 50.0 * fisher[n] * (p.detach() - anchor[n])
        assert torch.allclose(p.grad, g_want, atol=1e-6), f"d penalty / d {n} should be lam * F * (theta - anchor)"


def test_stage2_samples_scenario_and_two_tasks(nb):
    model = _mlp(seed=9, c=4)
    t1 = _task(n=20, seed=10, c=4, classes=(0, 1))
    t2 = _task(n=20, seed=11, c=4, classes=(2, 3))
    ewc = nb.EWC(2.0, n_samples=8)
    ewc.consolidate(model, t1, scenario="task")
    a1 = {n: p.detach().clone() for n, p in model.named_parameters()}
    f1 = _brute_fisher(model, t1["X"][:8], classes=(0, 1))
    _shift(model, 0.2, seed=12)
    ewc.consolidate(model, t2)  # scenario 'class': all 4 classes
    a2 = {n: p.detach().clone() for n, p in model.named_parameters()}
    f2 = _brute_fisher(model, t2["X"][:8])
    _shift(model, 0.2, seed=13)
    params = dict(model.named_parameters())
    want = sum((f1[n] * (params[n] - a1[n]) ** 2).sum() + (f2[n] * (params[n] - a2[n]) ** 2).sum() for n in params)
    got = ewc.penalty(model).item()
    assert abs(got - want.detach().item()) < 1e-5 * max(1.0, want.detach().item()), (
        f"after two consolidations (task 1 with scenario='task', task 2 with 'class'), the penalty is the sum of one "
        f"quadratic per task, each with its own anchor and Fisher: want {want.detach().item():.6g}, got {got:.6g}. check that "
        "the Fisher uses the first n_samples examples, and the task's classes only when scenario='task'")


def test_stage2_online(nb):
    model = _mlp(seed=14)
    t1, t2 = _task(seed=15), _task(seed=16)
    ewc = nb.EWC(4.0, n_samples=None, online=True, gamma=0.5)
    ewc.consolidate(model, t1)
    f1 = _brute_fisher(model, t1["X"])
    _shift(model, 0.25, seed=17)
    ewc.consolidate(model, t2)
    f2 = _brute_fisher(model, t2["X"])
    anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
    _shift(model, 0.25, seed=18)
    params = dict(model.named_parameters())
    want = 2.0 * sum(((0.5 * f1[n] + f2[n]) * (params[n] - anchor[n]) ** 2).sum() for n in params)
    got = ewc.penalty(model).item()
    assert abs(got - want.detach().item()) < 1e-5 * max(1.0, want.detach().item()), (
        f"online EWC keeps one quadratic: F <- gamma * F + F_new, anchored at the latest weights: want "
        f"{want.detach().item():.6g}, got {got:.6g}")


# ---------------------------------------------------------------- stage 3: SI


class Vec(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(v, dtype=torch.float64))

    def forward(self, x):
        return x * self.p


def _fake_step(model, grad, new):
    with torch.no_grad():
        model.p.copy_(torch.tensor(new, dtype=torch.float64))
    model.p.grad = torch.tensor(grad, dtype=torch.float64)


def test_stage3_hand_example(nb):
    model = Vec([1.0, -2.0])
    si = nb.SI(2.0)
    assert abs(si.xi - 0.1) < 1e-12, f"the default damping xi should be 0.1, got {si.xi}"
    si.begin_task(model)
    assert si.penalty(model).item() == 0.0, "before the first consolidate the penalty is 0"
    _fake_step(model, [0.5, -1.0], [0.8, -1.5])
    si.after_step(model)
    _fake_step(model, [0.2, 0.4], [0.7, -1.7])
    si.after_step(model)
    si.consolidate(model)
    with torch.no_grad():
        model.p.copy_(torch.tensor([1.0, -1.0], dtype=torch.float64))
    got = si.penalty(model)
    assert torch.is_tensor(got) and got.requires_grad, "penalty(model) must be a differentiable scalar tensor"
    assert abs(got.detach().item() - 3.1052632) < 1e-6, (
        f"the lesson's worked example: w = (0.12, 0.58), Omega = w / (0.3^2 + 0.1) = (0.6316, 3.0526), penalty at "
        f"(1, -1) = 2 * (0.6316 * 0.3^2 + 3.0526 * 0.7^2) = 3.10526; got {got.detach().item():.6f}. w accumulates "
        "-grad * (theta - theta_prev) after every step, using the gradient that produced that step")
    model.p.grad = None
    got.backward()
    want_g = 2 * 2.0 * torch.tensor([0.12 / 0.19, 0.58 / 0.19], dtype=torch.float64) * torch.tensor([0.3, 0.7],
                                                                                                    dtype=torch.float64)
    assert torch.allclose(model.p.grad, want_g, atol=1e-6), f"d penalty / d theta should be 2c Omega (theta - anchor)"


def test_stage3_second_task_accumulates(nb):
    model = Vec([1.0, -2.0])
    si = nb.SI(1.0, xi=0.5)
    si.begin_task(model)
    _fake_step(model, [1.0, 0.0], [0.0, -2.0])     # w = (1, 0); delta = (-1, 0); Omega += (1 / 1.5, 0)
    si.after_step(model)
    si.consolidate(model)
    si.begin_task(model)
    _fake_step(model, [0.0, -2.0], [0.0, -1.0])    # w = (0, 2); delta = (0, 1); Omega += (0, 2 / 1.5)
    si.after_step(model)
    model.p.grad = None
    with torch.no_grad():
        model.p.copy_(torch.tensor([0.0, -1.0], dtype=torch.float64))
    si.after_step(model)                           # no gradient: nothing to add
    si.consolidate(model)
    with torch.no_grad():
        model.p.copy_(torch.tensor([1.0, 0.0], dtype=torch.float64))
    got = si.penalty(model).item()
    want = (1 / 1.5) * 1.0 + (2 / 1.5) * 1.0       # anchor (0, -1), theta - anchor = (1, 1)
    assert abs(got - want) < 1e-9, (
        f"importances add up across tasks, begin_task resets w and the start point, and the anchor moves to the end "
        f"of the latest task: want {want:.6f}, got {got:.6f}. (a parameter whose .grad is None adds nothing)")


def test_stage3_path_integral_on_a_quadratic(nb):
    a = torch.tensor([0.5, 2.0, 8.0], dtype=torch.float64)
    m = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
    model = Vec([0.0, 0.0, 0.0])
    si = nb.SI(1.0, xi=1e-12)
    si.begin_task(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.002)
    for _ in range(2000):
        loss = (a * (model.p - m) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        si.after_step(model)
    drop = a * m ** 2 - a * (model.p.detach() - m) ** 2   # loss decrease per coordinate
    si.consolidate(model)
    delta2 = model.p.detach() ** 2
    with torch.no_grad():
        model.p.add_(1.0)
    got = si.penalty(model).item()
    want = float((drop / delta2).sum())
    assert abs(got - want) < 0.03 * want, (
        f"on a separable quadratic, plain gradient descent's path integral w_i equals the loss decrease along "
        f"coordinate i (up to a relative O(a_i * lr), 1.6% here), so Omega_i = drop_i / delta_i^2 and the penalty at anchor + 1 is "
        f"sum(drop / delta^2) = {want:.4f}; got {got:.4f}")


def test_stage2_empirical_mode(nb):
    model = _mlp(seed=19, c=4)
    task = _task(n=15, seed=20, c=4)
    ewc = nb.EWC(6.0, mode="empirical", n_samples=10)
    ewc.consolidate(model, task)
    anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
    fisher = _brute_fisher(model, task["X"][:10], y=task["y"][:10], mode="empirical")
    _shift(model, 0.3, seed=21)
    params = dict(model.named_parameters())
    want = 3.0 * sum((fisher[n] * (params[n].detach() - anchor[n]) ** 2).sum() for n in params).item()
    got = ewc.penalty(model).detach().item()
    assert abs(got - want) < 1e-5 * max(1.0, want), (
        f"EWC(mode='empirical') must pass its mode to fisher_diagonal, with the task's labels: want {want:.6g}, "
        f"got {got:.6g}")


def test_stage3_signed_path_integral(nb):
    model = Vec([0.0, 0.0])
    si = nb.SI(1.0, xi=0.5)
    si.begin_task(model)
    _fake_step(model, [1.0, 1.0], [-0.5, 0.5])   # weight 2 moves *uphill*: -g * delta = -0.5
    si.after_step(model)
    si.consolidate(model)                          # Omega = (0.5, -0.5) / (0.25 + 0.5) = (2/3, -2/3)
    with torch.no_grad():
        model.p.copy_(torch.tensor([0.5, 2.5], dtype=torch.float64))   # anchor + (1, 2)
    got = si.penalty(model).detach().item()
    assert abs(got - (-2.0)) < 1e-9, (
        f"w keeps its sign: a weight that moved uphill (gradient and step in the same direction) gets negative "
        f"importance. Omega = (2/3, -2/3) and the penalty at anchor + (1, 2) is 2/3 - 4 * 2/3 = -2; got {got:.6f}. "
        "don't take absolute values or clip at zero")
