"""Checks for exercise.py (optimization in practice).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. get_lr/set_lr are
checked on mlfs.optim optimizers, a duck-typed param_groups object and (when installed) real torch
optimizers; schedulers on hand-computed values and against torch.optim.lr_scheduler; clipping on a
hand example and against torch.nn.utils.clip_grad_norm_; the range test on a small regression whose
divergence point is known, under `mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

from mlfs import nn
from mlfs.optim import SGD, Adam, Param
from mlfs.testing import bounded


class _FakeTorchOptimizer:
    """duck-types torch.optim: no .lr, a list of param_group dicts."""

    def __init__(self, lrs):
        self.param_groups = [{"lr": lr, "params": []} for lr in lrs]


def _mlfs_opt(lr=0.1):
    return SGD([Param(np.zeros(2))], lr=lr)


# ---------------------------------------------------------------- stage 1: get_lr / set_lr


def test_stage1_mlfs_optimizers(nb):
    opt = _mlfs_opt(0.05)
    assert nb.get_lr(opt) == 0.05, "an mlfs.optim optimizer keeps its learning rate in .lr"
    nb.set_lr(opt, 0.2)
    assert opt.lr == 0.2 and nb.get_lr(opt) == 0.2
    adam = Adam([Param(np.zeros(1))], lr=3e-4)
    assert nb.get_lr(adam) == 3e-4


def test_stage1_param_groups(nb):
    opt = _FakeTorchOptimizer([0.1, 0.01])
    assert nb.get_lr(opt) == 0.1, "torch optimizers have no .lr: read param_groups[0]['lr']"
    nb.set_lr(opt, 0.5)
    assert [g["lr"] for g in opt.param_groups] == [0.5, 0.5], "set_lr sets EVERY param_group's 'lr'"
    assert not hasattr(opt, "lr"), "don't create a .lr attribute on a torch optimizer (nothing reads it)"


def test_stage1_real_torch(nb):
    torch = pytest.importorskip("torch")
    w = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([w], lr=1e-3)
    assert math.isclose(nb.get_lr(opt), 1e-3)
    nb.set_lr(opt, 5e-4)
    assert opt.param_groups[0]["lr"] == 5e-4


# ---------------------------------------------------------------- stage 2: schedulers


def test_stage2_base_scheduler_protocol(nb):
    opt = _mlfs_opt(0.3)
    sched = nb.LRScheduler(opt)
    assert sched.base_lr == 0.3 and sched.lr_at(0) == 0.3 and sched.lr_at(17) == 0.3, (
        "the base LRScheduler records base_lr = get_lr(optimizer) and keeps it constant")
    sched.step()
    assert sched.last_step == 1 and opt.lr == 0.3


def test_stage2_step_lr(nb):
    opt = _mlfs_opt(1.0)
    sched = nb.StepLR(opt, step_size=3, gamma=0.5)
    seen = [opt.lr]
    for _ in range(7):
        sched.step()
        seen.append(opt.lr)
    assert np.allclose(seen, [1, 1, 1, 0.5, 0.5, 0.5, 0.25, 0.25]), (
        f"StepLR(step_size=3, gamma=0.5) from lr 1: steps 0-2 at 1, 3-5 at 0.5, 6-8 at 0.25; got {seen}. "
        "construction sets lr_at(0); each step() advances the counter, then sets lr_at(counter)")
    assert sched.lr_at(9) == 0.125 and sched.base_lr == 1.0, "lr_at(step) = base_lr * gamma ** (step // step_size)"


def test_stage2_cosine_with_warmup(nb):
    opt = _mlfs_opt(1.0)
    sched = nb.CosineSchedule(opt, total_steps=10, warmup_steps=2, min_lr=0.1)
    assert opt.lr == 0.0, (
        "with warmup, the schedule starts at lr 0: the constructor sets lr_at(0) = 0 on the optimizer "
        "(and base_lr remembers the optimizer's original lr)")
    want = {0: 0.0, 1: 0.5, 2: 1.0, 6: 0.1 + 0.9 * 0.5, 10: 0.1, 25: 0.1}
    got = {s: sched.lr_at(s) for s in want}
    assert all(math.isclose(got[s], want[s], abs_tol=1e-12) for s in want), (
        f"warmup: base_lr * step / warmup_steps; then min_lr + (base_lr - min_lr) * (1 + cos(pi t)) / 2 with "
        f"t = (step - warmup) / (total - warmup); min_lr from total_steps on. want {want}, got {got}")
    for _ in range(4):
        sched.step()
    assert math.isclose(opt.lr, sched.lr_at(4)), "after 4 step() calls the optimizer's lr is lr_at(4)"
    plain = nb.CosineSchedule(_mlfs_opt(2.0), total_steps=4)
    assert [round(plain.lr_at(s), 10) for s in range(5)] == [2.0, round(1 + 2 ** -0.5, 10), 1.0,
                                                             round(1 - 2 ** -0.5, 10), 0.0]
    with pytest.raises(ValueError):
        nb.CosineSchedule(_mlfs_opt(), total_steps=5, warmup_steps=5)


def test_stage2_monotone_cosine(nb):
    sched = nb.CosineSchedule(_mlfs_opt(1.0), total_steps=50, warmup_steps=10)
    lrs = [sched.lr_at(s) for s in range(60)]
    assert all(a < b for a, b in zip(lrs[:10], lrs[1:11])), "the lr rises during warmup"
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:])), "after warmup it never increases"
    assert math.isclose(max(lrs), 1.0) and lrs[10] == max(lrs), "it peaks at base_lr exactly when warmup ends"


def test_stage2_matches_torch_schedulers(nb):
    torch = pytest.importorskip("torch")
    w = torch.nn.Parameter(torch.zeros(1))
    opt_t, opt_m = torch.optim.SGD([w], lr=0.8), torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.8)
    ref = torch.optim.lr_scheduler.StepLR(opt_t, step_size=4, gamma=0.3)
    mine = nb.StepLR(opt_m, step_size=4, gamma=0.3)
    for _ in range(10):
        opt_t.step()
        ref.step()
        opt_m.step()
        mine.step()
        assert math.isclose(opt_m.param_groups[0]["lr"], opt_t.param_groups[0]["lr"], rel_tol=1e-9), (
            "your StepLR on a torch optimizer should follow torch's StepLR step for step")
    opt_c = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.5)
    ref_c = torch.optim.lr_scheduler.CosineAnnealingLR(torch.optim.SGD([w], lr=0.5), T_max=20, eta_min=0.01)
    mine_c = nb.CosineSchedule(opt_c, total_steps=20, min_lr=0.01)
    for _ in range(20):
        ref_c.optimizer.step()
        ref_c.step()
        mine_c.step()
        assert math.isclose(opt_c.param_groups[0]["lr"], ref_c.optimizer.param_groups[0]["lr"], rel_tol=1e-6,
                            abs_tol=1e-9), "without warmup, CosineSchedule matches torch's CosineAnnealingLR"


# ---------------------------------------------------------------- stage 3: clip_grad_norm


def test_stage3_hand_example(nb):
    a, b, c = Param(np.zeros(2)), Param(np.zeros(1)), Param(np.zeros(3))
    a.grad, b.grad, c.grad = np.array([3.0, 0.0]), np.array([4.0]), None
    before = a.grad
    norm = nb.clip_grad_norm([a, b, c], max_norm=1.0)
    assert math.isclose(norm, 5.0), f"the global norm of (3, 0) and (4,) is 5 (None grads skipped); got {norm}"
    assert a.grad is before, "clip in place: scale the existing arrays (g *= scale)"
    assert np.allclose(a.grad, [0.6, 0.0], atol=1e-6) and np.allclose(b.grad, [0.8], atol=1e-6), (
        "every gradient is scaled by the SAME factor max_norm / norm, so the direction is kept and the global norm "
        f"becomes max_norm; got {a.grad}, {b.grad}")
    assert c.grad is None


def test_stage3_no_op_below_threshold(nb):
    p = Param(np.zeros(3))
    p.grad = np.array([0.1, -0.2, 0.2])
    norm = nb.clip_grad_norm(iter([p]), max_norm=1.0)
    assert math.isclose(norm, 0.3) and np.allclose(p.grad, [0.1, -0.2, 0.2]), (
        "below max_norm nothing changes (and params may be any iterable, even a generator)")


def test_stage3_matches_torch(nb):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    grads = [rng.normal(size=s) * 10 for s in [(4, 3), (3,), (2, 2)]]
    ps = []
    for g in grads:
        p = Param(np.zeros_like(g))
        p.grad = g.copy()
        ps.append(p)
    tps = [torch.nn.Parameter(torch.zeros(g.shape, dtype=torch.float64)) for g in grads]
    for tp, g in zip(tps, grads):
        tp.grad = torch.tensor(g)
    want = torch.nn.utils.clip_grad_norm_(tps, max_norm=2.5).item()
    got = nb.clip_grad_norm(ps, 2.5)
    assert math.isclose(got, want, rel_tol=1e-9), "returns the norm BEFORE clipping, like torch"
    for p, tp in zip(ps, tps):
        assert np.allclose(p.grad, tp.grad.numpy(), atol=1e-12), "clipped gradients differ from torch's"


def test_stage3_tames_an_explosion(nb):
    model = nn.Linear(1, 1, rng=0)
    model.weight.data[...] = 0.0
    model.bias.data[...] = 0.0
    opt = SGD(model.parameters(), lr=0.1)
    x, y = np.array([[100.0]]), np.array([[1000.0]])
    opt.zero_grad()
    nn.mse_loss(model(x), y).backward()
    raw = nb.clip_grad_norm(model.parameters(), max_norm=1.0)
    opt.step()
    step = math.hypot(model.weight.data.item(), model.bias.data.item())
    assert raw > 1e4 and step <= 0.1 + 1e-9, (
        f"after clipping to norm 1, an SGD step with lr 0.1 moves the parameters by at most 0.1; it moved {step:.3g} "
        f"(raw gradient norm {raw:.3g})")


# ---------------------------------------------------------------- stage 4: range test


def _regression(seed=0, n=256, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([[1.0], [-2.0], [0.5], [3.0]]) + noise * rng.normal(size=(n, 1))
    return X, y


def _range_test(nb, *args, **kwargs):
    return bounded(nb.lr_range_test, *args, budget=3_000_000, what="lr_range_test",
                   hint="loop over range(n_steps) and break when the loss blows up.", **kwargs)


def test_stage4_sweep_and_restore(nb):
    X, y = _regression(noise=0.0)  # noise-free, so the minibatch losses fall smoothly
    model = nn.Linear(4, 1, rng=1)
    before = {k: v.copy() for k, v in model.state_dict().items()}
    opt = SGD(model.parameters(), lr=0.123)
    lrs, losses = _range_test(nb, model, nn.mse_loss, opt, X, y, min_lr=1e-4, max_lr=1e-1, n_steps=30, batch_size=64,
                              rng=0)
    assert len(lrs) == len(losses) == 30, f"no blow-up below lr 0.1 here: all 30 steps run; got {len(lrs)}"
    assert np.allclose(lrs, 1e-4 * 1000 ** (np.arange(30) / 29)), (
        "the learning rates sweep exponentially: lr_i = min_lr * (max_lr / min_lr) ** (i / (n_steps - 1)), "
        "from exactly min_lr to exactly max_lr")
    assert losses[-1] < losses[0], "on this easy regression the loss falls during the sweep"
    assert all(np.array_equal(before[k], v) for k, v in model.state_dict().items()), (
        "the range test is an experiment, not training: restore the weights you started with (state_dict)")
    assert opt.lr == 0.123, "and restore the optimizer's learning rate"


def test_stage4_restores_adam_state(nb):
    X, y = _regression(3, noise=0.0)
    model = nn.Linear(4, 1, rng=4)
    opt = Adam(model.parameters(), lr=0.01)
    for _ in range(3):  # a few real training steps, so Adam has moments and step counts
        opt.zero_grad()
        nn.mse_loss(model(X[:32]), y[:32]).backward()
        opt.step()
    state = {i: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()} for i, st in opt.state.items()}
    _range_test(nb, model, nn.mse_loss, opt, X, y, min_lr=1e-4, max_lr=1.0, n_steps=20, batch_size=32, rng=0)
    assert set(opt.state) == set(state), "the optimizer's per-parameter state is restored"
    for i, st in state.items():
        assert opt.state[i]["step"] == st["step"], (
            f"Adam's step count was {st['step']} before the range test and {opt.state[i]['step']} after: restore the "
            "optimizer's state (a deep copy of optimizer.state), or the next real step uses the test's moments")
        assert all(np.array_equal(opt.state[i][k], st[k]) for k in ("exp_avg", "exp_avg_sq")), (
            "Adam's moments must be exactly as they were before the range test")


def test_stage4_stops_on_blowup(nb):
    X, y = _regression(1)
    model = nn.Linear(4, 1, rng=2)
    lrs, losses = _range_test(nb, model, nn.mse_loss, SGD(model.parameters(), lr=0.01), X, y, min_lr=1e-3,
                              max_lr=100.0, n_steps=100, batch_size=64, rng=0)
    assert len(lrs) < 100, "past the divergence point the loss explodes: the test must stop early"
    assert losses[-1] > 4 * min(losses), "it stops at the first step whose loss exceeds 4 x the best so far"
    assert all(v <= 4 * min(losses[:i + 1]) for i, v in enumerate(losses[:-1])), "...and not before"
    assert 0.1 < lrs[-1] < 30, (
        f"for this regression (Hessian eigenvalues ~1-ish), SGD diverges somewhere around lr 1; it stopped at {lrs[-1]:.3g}")


def test_stage4_one_minibatch_per_step(nb):
    X, y = _regression(2, n=96)
    sizes = []

    def loss_fn(pred, target):
        sizes.append(len(target))
        return nn.mse_loss(pred, target)

    model = nn.Linear(4, 1, rng=0)
    _range_test(nb, model, loss_fn, SGD(model.parameters(), lr=0.1), X, y, min_lr=1e-5, max_lr=1e-3, n_steps=9,
                batch_size=32, rng=0)
    assert len(sizes) == 9 and all(s == 32 for s in sizes), (
        f"one minibatch (at most batch_size rows) per step, n_steps steps, cycling through epochs; got sizes {sizes}")


def test_stage4_suggest_lr(nb):
    lrs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    losses = [2.0, 2.0, 1.0, 0.5, 9.0]
    assert math.isclose(nb.suggest_lr(lrs, losses, beta=0.0), 1e-2), (
        "with beta = 0 there's no smoothing: the lowest loss (0.5) is at lr 0.1, and the suggestion is a tenth of it")
    smooth = []
    avg = 0.0
    for k, v in enumerate(losses, start=1):
        avg = 0.8 * avg + 0.2 * v
        smooth.append(avg / (1 - 0.8 ** k))
    assert math.isclose(nb.suggest_lr(lrs, losses), lrs[int(np.argmin(smooth))] / 10), (
        "the default smooths the losses with a bias-corrected EMA, beta = 0.8: avg = beta*avg + (1-beta)*loss, "
        "smoothed_k = avg / (1 - beta**k)")
    lrs2 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    noisy = [2.0, 1.8, 0.2, 1.2, 0.9, 0.7, 0.6, 5.0]  # one lucky minibatch at lr 1e-3
    avg, sm = 0.0, []
    for k, v in enumerate(noisy, start=1):
        avg = 0.8 * avg + 0.2 * v
        sm.append(avg / (1 - 0.8 ** k))
    assert int(np.argmin(sm)) != int(np.argmin(noisy))
    assert math.isclose(nb.suggest_lr(lrs2, noisy), lrs2[int(np.argmin(sm))] / 10), (
        "suggest_lr reads the minimum of the SMOOTHED curve, so one lucky minibatch can't move the suggestion")
    with pytest.raises(ValueError):
        nb.suggest_lr([], [])
