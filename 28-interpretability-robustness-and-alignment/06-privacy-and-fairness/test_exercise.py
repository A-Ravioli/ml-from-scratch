"""Checks for 28 · 06 privacy & fairness.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. tiny torch models written here.
per-sample gradients against a loop of single-example backward passes (custom losses, frozen parameters);
DP-SGD with zero noise against a hand-computed clipped update (per-example clipping, not batch clipping, the
clip norm used), its noise statistically (std = noise_multiplier * max_grad_norm / B), the generator and the
returned loss; the accountant against the closed form at q = 1, numerical integration of the RDP definition and
Abadi et al.'s published example; the attack and its low-FPR score, and the fairness metrics, by hand.
"""
import math

import numpy as np
import pytest
from scipy import integrate

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _mlp(seed=0, d=6, h=8, k=3):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, k))


def _data(B=10, d=6, k=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, d, generator=g), torch.randint(0, k, (B,), generator=g)


def _loop_grads(model, loss_fn, x, y):
    out = {n: [] for n, p in model.named_parameters() if p.requires_grad}
    for i in range(len(x)):
        model.zero_grad()
        loss_fn(model(x[i:i + 1]), y[i:i + 1]).backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                out[n].append(p.grad.clone())
    model.zero_grad(set_to_none=True)
    return {n: torch.stack(v) for n, v in out.items()}


# ---------------------------------------------------------------- stage 1: per-sample gradients


def test_stage1_matches_loop(nb):
    m = _mlp()
    x, y = _data()
    got = nb.per_sample_gradients(m, F.cross_entropy, x, y)
    want = _loop_grads(m, F.cross_entropy, x, y)
    assert set(got) == set(want), f"one entry per parameter name: {sorted(want)}"
    for n in want:
        assert got[n].shape == (10, *dict(m.named_parameters())[n].shape), f"{n}: shape (B, *param.shape)"
        assert torch.allclose(got[n], want[n], atol=1e-6), (
            f"{n}: row i is the gradient of the loss on example i alone (not the batch mean's gradient)")
    assert all(p.grad is None for p in m.parameters()), "don't write anything into .grad"


def test_stage1_loss_fn_and_frozen(nb):
    m = _mlp(seed=1)
    m[0].weight.requires_grad_(False)
    x, _ = _data(seed=2)
    target = torch.randn(10, 3, generator=torch.Generator().manual_seed(3))
    got = nb.per_sample_gradients(m, F.mse_loss, x, target)
    want = _loop_grads(m, F.mse_loss, x, target)
    assert "0.weight" not in got, "frozen parameters (requires_grad=False) get no entry"
    assert all(torch.allclose(got[n], want[n], atol=1e-6) for n in want), (
        "use the loss_fn you're given (here MSE against float targets), called on a batch of one")


# ---------------------------------------------------------------- stage 2: DP-SGD


def _clipped_mean(model, x, y, C):
    g = _loop_grads(model, F.cross_entropy, x, y)
    B = len(x)
    norms = torch.sqrt(sum(v.reshape(B, -1).pow(2).sum(1) for v in g.values()))
    f = torch.clamp(C / norms, max=1.0)
    return {n: (v * f.view(B, *([1] * (v.dim() - 1)))).sum(0) / B for n, v in g.items()}, norms


def test_stage2_clipping_without_noise(nb):
    m = _mlp(seed=4)
    x, y = _data(B=8, seed=5)
    x[0] *= 30.0  # one example with a much larger gradient
    for C in (0.5, 5.0):
        mm = _mlp(seed=4)
        want, norms = _clipped_mean(mm, x, y, C)
        opt = torch.optim.SGD(mm.parameters(), lr=0.1)
        before = {n: p.detach().clone() for n, p in mm.named_parameters()}
        loss = bounded(nb.dp_sgd_step, mm, F.cross_entropy, opt, x, y, budget=3_000_000, what="dp_sgd_step",
                       max_grad_norm=C, noise_multiplier=0.0)
        for n, p in mm.named_parameters():
            assert torch.allclose(p.grad, want[n], atol=1e-6), (
                f"max_grad_norm={C}: with no noise, .grad is the mean of the per-example gradients, each scaled by "
                f"min(1, C / ||g_i||) with ||g_i|| the norm over ALL parameters (example norms here: "
                f"{[round(v, 2) for v in norms.tolist()]}). clipping the batch gradient instead, or each parameter "
                "separately, gives something else")
            assert torch.allclose(p.detach(), before[n] - 0.1 * want[n], atol=1e-6), "then call optimizer.step()"
        with torch.no_grad():
            want_loss = F.cross_entropy(_mlp(seed=4)(x), y).item()
        assert isinstance(loss, float) and np.isclose(loss, want_loss, atol=1e-6), (
            "return the batch's mean loss (before the update) as a float")
    del m


def test_stage2_noise(nb):
    torch.manual_seed(0)
    m = nn.Linear(100, 80)
    x = torch.randn(16, 100, generator=torch.Generator().manual_seed(1))
    y = torch.randint(0, 80, (16,), generator=torch.Generator().manual_seed(2))
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    for sigma, C in ((0.5, 1.0), (2.0, 3.0)):
        clean, _ = _clipped_mean(m, x, y, C)
        nb.dp_sgd_step(m, F.cross_entropy, opt, x, y, max_grad_norm=C, noise_multiplier=sigma,
                       generator=torch.Generator().manual_seed(0))
        bias_resid = (m.bias.grad - clean["bias"]) * 16
        assert 0.6 * sigma * C < bias_resid.std().item() < 1.4 * sigma * C, (
            f"every trainable parameter gets noise, biases included: bias noise std {bias_resid.std().item():.3f}, "
            f"expected about {sigma * C}")
        resid = (m.weight.grad - clean["weight"]) * 16
        assert abs(resid.std().item() - sigma * C) < 0.05 * sigma * C and abs(resid.mean().item()) < 4 * sigma * C / resid.numel() ** 0.5, (
            f"noise_multiplier={sigma}, max_grad_norm={C}: the noise added to the SUM of clipped gradients has std "
            f"noise_multiplier * max_grad_norm = {sigma * C}, before dividing by B; measured {resid.std().item():.3f}")
    nb.dp_sgd_step(m, F.cross_entropy, opt, x, y, generator=torch.Generator().manual_seed(7))
    a = m.weight.grad.clone()
    nb.dp_sgd_step(m, F.cross_entropy, opt, x, y, generator=torch.Generator().manual_seed(7))
    b = m.weight.grad.clone()
    nb.dp_sgd_step(m, F.cross_entropy, opt, x, y, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and not torch.equal(b, m.weight.grad), "the noise comes from generator"


def test_stage2_frozen_and_learning(nb):
    m = _mlp(seed=6)
    m[0].bias.requires_grad_(False)
    x, _ = _data(B=32, seed=7)
    y = (x @ torch.randn(6, 3, generator=torch.Generator().manual_seed(8))).argmax(1)  # a learnable rule
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.5)
    losses = [nb.dp_sgd_step(m, F.cross_entropy, opt, x, y, max_grad_norm=1.0, noise_multiplier=0.1,
                             generator=torch.Generator().manual_seed(i)) for i in range(40)]
    assert m[0].bias.grad is None, "frozen parameters are left alone"
    assert losses[-1] < 0.8 * losses[0], f"DP-SGD with little noise still learns: {losses[0]:.3f} -> {losses[-1]:.3f}"


# ---------------------------------------------------------------- stage 3: the accountant


def _rdp_by_integration(q, sigma, a):
    f = lambda z: math.exp(-z * z / (2 * sigma ** 2)) / math.sqrt(2 * math.pi * sigma ** 2) * (  # noqa: E731
        (1 - q) + q * math.exp((2 * z - 1) / (2 * sigma ** 2))) ** a
    val, _ = integrate.quad(f, -12 * sigma, 12 * sigma + 2 * a, limit=400)
    return math.log(val) / (a - 1)


def test_stage3_rdp(nb):
    orders = [2, 3, 10, 64]
    assert np.allclose(nb.rdp_subsampled_gaussian(1.0, 2.0, orders), np.array(orders) / 8), (
        "q = 1 is the plain Gaussian mechanism: RDP(alpha) = alpha / (2 sigma^2)")
    assert np.allclose(nb.rdp_subsampled_gaussian(0.0, 1.0, orders), 0), "q = 0: nothing is released"
    for q, sigma in ((0.05, 1.0), (0.2, 2.0), (0.01, 0.7)):
        got = nb.rdp_subsampled_gaussian(q, sigma, [2, 4, 8])
        want = [_rdp_by_integration(q, sigma, a) for a in (2, 4, 8)]
        assert np.allclose(got, want, rtol=1e-4, atol=1e-8), (
            f"q={q}, sigma={sigma}: RDP = log(A_alpha) / (alpha - 1) with A_alpha = sum_k C(alpha, k) (1-q)^(alpha-k) "
            f"q^k exp((k^2 - k) / (2 sigma^2)); expected {np.round(want, 6)}, got {np.round(got, 6)}")
    big = nb.rdp_subsampled_gaussian(0.01, 0.5, [200])
    assert np.isfinite(big).all(), "high orders overflow unless you sum in log space"


def test_stage3_epsilon(nb):
    one_step = nb.dp_sgd_epsilon(1.0, 2.0, 1, 1e-5)
    assert np.isclose(one_step, 11 / 8 + math.log(1e5) / 10), (
        f"the lesson's worked example: q = 1, sigma = 2, one step: min over alpha of alpha / 8 + log(1e5) / (alpha - 1) "
        f"= 2.526 at alpha = 11; got {one_step:.4f}")
    eps = nb.dp_sgd_epsilon(0.01, 4.0, 10_000, 1e-5)
    assert abs(eps - 1.2586) < 2e-3, (
        f"Abadi et al. (2016): q = 0.01, sigma = 4, 10,000 steps, delta = 1e-5 gives eps about 1.26 with their "
        f"moments accountant; RDP at integer orders computes the same bound, 1.2586; got {eps:.4f}")
    e = [nb.dp_sgd_epsilon(0.02, 1.0, T, 1e-5) for T in (10, 100, 1000)]
    assert e[0] < e[1] < e[2], f"more steps cost more privacy: {e}"
    assert nb.dp_sgd_epsilon(0.02, 2.0, 100, 1e-5) < e[1] < nb.dp_sgd_epsilon(0.04, 1.0, 100, 1e-5), (
        "more noise lowers epsilon; a higher sampling rate raises it")
    assert nb.dp_sgd_epsilon(0.02, 1.0, 100, 1e-8) > e[1], "a smaller delta costs more epsilon"
    one = nb.dp_sgd_epsilon(0.02, 1.0, 100, 1e-5, orders=(2,))
    assert one > e[1] and np.isclose(one, 100 * nb.rdp_subsampled_gaussian(0.02, 1.0, [2])[0] + math.log(1e5)), (
        "orders are the candidate alphas: with only alpha = 2, eps = steps * RDP(2) + log(1/delta) / 1")


# ---------------------------------------------------------------- stage 4: membership inference


def test_stage4_attack(nb):
    assert np.isclose(nb.loss_threshold_attack([0.1, 0.2], [0.5, 0.6, 0.7]), 1.0), (
        "members have the lower losses: AUROC 1")
    assert np.isclose(nb.loss_threshold_attack([0.5, 0.6], [0.1, 0.2]), 0.0), "orientation: lower loss = member"
    assert np.isclose(nb.loss_threshold_attack([0.1, 0.4], [0.2, 0.3]), 0.5), "P(member loss < non-member loss) = 2/4"
    assert np.isclose(nb.loss_threshold_attack([0.3, 0.3], [0.3, 0.3]), 0.5), "ties count 1/2"
    rng = np.random.default_rng(0)
    mem, non = rng.exponential(0.5, 2000), rng.exponential(1.0, 2000)
    want = np.mean(mem[:, None] < non[None, :]) + 0.5 * np.mean(mem[:, None] == non[None, :])
    assert np.isclose(nb.loss_threshold_attack(mem, non), want, atol=1e-9)


def test_stage4_tpr_at_fpr(nb):
    non = np.arange(1, 101) / 100.0  # 0.01 .. 1.00
    mem = np.array([0.005, 0.015, 0.025, 0.5, 0.9])
    assert np.isclose(nb.tpr_at_fpr(mem, non, fpr=0.01), 0.4), (
        "fpr = 0.01 of 100 non-members: at most 1 may be flagged, so tau = the 2nd smallest non-member loss 0.02 "
        "(flag loss < 0.02); 2 of 5 members are below it")
    assert np.isclose(nb.tpr_at_fpr(mem, non, fpr=0.0), 0.2), "fpr = 0: tau = the smallest non-member loss 0.01"
    assert np.isclose(nb.tpr_at_fpr(mem, non, fpr=0.03), 0.6), "fpr = 0.03: tau = 0.04"
    assert np.isclose(nb.tpr_at_fpr(mem, non, fpr=1.0), 1.0), "fpr = 1: everyone may be flagged"
    assert np.isclose(nb.tpr_at_fpr(mem, non), 0.4), "the default fpr is 0.01"
    assert np.isclose(nb.tpr_at_fpr(mem, non, fpr=0.015), 0.4), (
        "fpr = 0.015 of 100 is 1.5 non-members: still at most 1 may be flagged (round down, never exceed the FPR)")
    assert nb.tpr_at_fpr([0.02], non, fpr=0.01) == 0.0, (
        "flag loss < tau strictly: a member with loss exactly tau = 0.02 isn't flagged (nor is the non-member at 0.02)")


# ---------------------------------------------------------------- stage 5: fairness


def test_stage5_metrics(nb):
    groups = np.array(["a"] * 4 + ["b"] * 6)
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    assert np.isclose(nb.demographic_parity_difference(y_pred, groups), 0.5 - 1 / 6), (
        "positive rates 2/4 and 1/6: difference 1/3")
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    # a: TPR 1/2, FPR 1/2; b: TPR 1/3, FPR 0
    got = nb.equalized_odds_difference(y_true, y_pred, groups)
    assert np.isclose(got, 0.5), f"TPR gap 1/6, FPR gap 1/2: the max is 0.5; got {got}"
    three = np.array([0, 0, 1, 1, 2, 2])
    assert np.isclose(nb.demographic_parity_difference([1, 1, 0, 1, 0, 0], three), 1.0), "max - min over 3 groups"
    with pytest.raises(ValueError):
        nb.equalized_odds_difference([1, 1, 0, 1], [1, 0, 0, 1], ["a", "a", "b", "b"])  # group a has no negatives


def test_stage5_thresholds(nb):
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.95, 0.5, 0.4, 0.3, 0.1])
    y_true = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0])
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    th = nb.equal_opportunity_thresholds(scores, y_true, groups, 0.75)
    assert set(th) == {0, 1}, "one threshold per group"
    assert np.isclose(th[0], 0.7) and np.isclose(th[1], 0.4), (
        f"group 0 positives 0.9, 0.8, 0.7, 0.6: TPR >= 0.75 needs 3 of 4 above the threshold, and the largest such "
        f"threshold is the 3rd largest positive score, 0.7; group 1 positives 0.95, 0.5, 0.4 need ceil(2.25) = 3 of 3: "
        f"0.4; got {th}")
    th2 = nb.equal_opportunity_thresholds(scores, y_true, groups, 0.5)
    assert np.isclose(th2[0], 0.8) and np.isclose(th2[1], 0.5), f"target 0.5: 2 of 4 -> 0.8, 2 of 3 -> 0.5; got {th2}"
    preds = scores >= np.vectorize(th.get)(groups)
    tprs = [preds[(groups == g) & (y_true == 1)].mean() for g in (0, 1)]
    assert min(tprs) >= 0.75, "predicting score >= threshold reaches the target TPR in every group"
