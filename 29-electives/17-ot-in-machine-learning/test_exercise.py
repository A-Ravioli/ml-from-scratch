"""Checks for exercise.py (OT in machine learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The exact coupling is checked
against brute force over all permutations and scipy's linear_sum_assignment; the entropic coupling
against its two limits, an exact replay of the specified sampling, and a chi-square test of its
conditional distributions; the Sinkhorn loss against an independent log-domain Sinkhorn and against
finite differences of its own value (the envelope theorem); domain adaptation against exact
assignments, translations and the large-eps limit. All data are synthetic.
"""
import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, linprog
from scipy.special import logsumexp
from scipy.stats import chi2

from mlfs.testing import bounded

torch.set_num_threads(1)  # small tensors: one thread is fastest, and avoids oversubscribing a busy machine


def _b(fn, *args, _budget=4_000_000, **kwargs):
    """run a call under a line budget, so a loop without an exit fails instead of hanging."""
    return bounded(fn, *args, budget=_budget, what=getattr(fn, "__name__", "the function"),
                   hint="every loop must have an exit: a Sinkhorn loop must stop after max_iters iterations",
                   **kwargs)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _sinkhorn(a, b, C, eps, max_iters=1000, tol=1e-9):
    """an independent log-domain Sinkhorn with the topic 16 conventions (g = 0 start, stop on the row error)."""
    g = np.zeros(len(b))
    for _ in range(max_iters):
        f = -eps * logsumexp(np.log(b)[None] + (g[None] - C) / eps, axis=1)
        g = -eps * logsumexp(np.log(a)[:, None] + (f[:, None] - C) / eps, axis=0)
        P = np.exp(np.log(a)[:, None] + np.log(b)[None] + (f[:, None] + g[None] - C) / eps)
        if np.abs(P.sum(1) - a).sum() <= tol:
            break
    return P, f, g


def _sqdist(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


# ---------------------------------------------------------------- stage 1: exact minibatch couplings


def test_stage1_worked_example(nb):
    x0 = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    x1 = torch.tensor([[2.0, 2.0], [2.0, 0.0]])
    out = _b(nb.minibatch_ot_coupling, x0, x1)
    assert torch.equal(out, torch.tensor([[0.0, 2.0], [0.0, 0.0]])), (
        f"the lesson's worked example: the given pairing crosses at (1, 1); the OT pairing sends (0, 2) to (2, 2) "
        f"and (0, 0) to (2, 0), so the reordered noise is [(0, 2), (0, 0)]. got {out.tolist()}")
    assert np.isclose(nb.pairing_cost(x0, x1), 8.0) and np.isclose(nb.pairing_cost(out, x1), 4.0), (
        "pairing cost 8 for the crossing pairs, 4 for the OT pairs")


def test_stage1_matches_brute_force(nb):
    g = torch.Generator().manual_seed(0)
    for _ in range(3):
        x0, x1 = torch.randn(6, 2, generator=g), torch.randn(6, 2, generator=g) * 2 + 1
        out = _b(nb.minibatch_ot_coupling, x0, x1)
        best = min(itertools.permutations(range(6)),
                   key=lambda p: float(((x0[list(p)] - x1) ** 2).sum()))
        assert torch.equal(out, x0[list(best)]), (
            "for 6 points, brute force over all 720 permutations finds the pairing with the least total squared "
            "distance; minibatch_ot_coupling must return x0 reordered by that permutation, aligned with x1 "
            "(out[k] is the noise paired with x1[k])")


def test_stage1_matches_assignment_and_is_a_permutation(nb):
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(64, 1, 3, 3, generator=g)  # image-shaped batches: flatten the non-batch dims
    x1 = torch.randn(64, 1, 3, 3, generator=g) + 0.5
    out = _b(nb.minibatch_ot_coupling, x0, x1)
    assert out.shape == x0.shape and out.dtype == x0.dtype, f"same shape and dtype as x0, got {out.shape} {out.dtype}"
    a, b = x0.reshape(64, -1).double().numpy(), x1.reshape(64, -1).double().numpy()
    C = _sqdist(b, a)
    _, cols = linear_sum_assignment(C)
    got = float(((out - x1) ** 2).sum())
    want = float(C[np.arange(64), cols].sum())
    assert abs(got - want) < 1e-4 * want, (
        f"the total squared distance of your pairing ({got:.4f}) must equal the optimal assignment's ({want:.4f})")
    rows = {tuple(r.tolist()) for r in out.reshape(64, -1)}
    assert rows == {tuple(r.tolist()) for r in x0.reshape(64, -1)}, (
        "every noise sample must appear exactly once: a coupling of two uniform batches reorders x0, it doesn't "
        "resample it")
    xd = x0.double()
    assert _b(nb.minibatch_ot_coupling, xd, x1.double()).dtype == torch.float64, "keeps x0's dtype"


def test_stage1_recovers_a_shuffle(nb):
    g = torch.Generator().manual_seed(2)
    x0 = torch.randn(50, 2, generator=g)
    perm = torch.randperm(50, generator=g)
    x1 = x0[perm] + 0.01
    out = _b(nb.minibatch_ot_coupling, x0, x1)
    assert torch.equal(out, x0[perm]), (
        "x1 is a shuffled copy of x0 (shifted by 0.01): the optimal pairing undoes the shuffle, so out must equal "
        "x0[perm]. if you got x0[argsort(perm)], you applied the inverse permutation: linear_sum_assignment on "
        "C[k, i] = |x1[k] − x0[i]|² gives, for each data point k, the noise index to put in slot k")


def test_stage1_pairing_cost_and_errors(nb):
    x0 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    x1 = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    got = nb.pairing_cost(x0, x1)
    assert isinstance(got, float) and np.isclose(got, (25 + 1) / 2), (
        f"mean over pairs of the squared distance: (25 + 1) / 2 = 13, got {got}")
    img0, img1 = torch.zeros(3, 1, 2, 2), torch.ones(3, 1, 2, 2)
    assert np.isclose(nb.pairing_cost(img0, img1), 4.0), "sum over all non-batch dims (4 pixels), mean over the batch"
    g = torch.Generator().manual_seed(3)
    a, b = torch.randn(40, 2, generator=g), torch.randn(40, 2, generator=g) * 3
    assert nb.pairing_cost(_b(nb.minibatch_ot_coupling, a, b), b) <= nb.pairing_cost(a, b) + 1e-9, (
        "the OT pairing can't cost more than the original (independent) pairing")
    try:
        _b(nb.minibatch_ot_coupling, torch.zeros(4, 2), torch.zeros(5, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("batches of different shapes must raise ValueError")


# ---------------------------------------------------------------- stage 2: entropic couplings


def test_stage2_limits(nb):
    g = torch.Generator().manual_seed(4)
    x0, x1 = torch.randn(32, 2, generator=g), torch.randn(32, 2, generator=g) * 2
    exact = _b(nb.minibatch_ot_coupling, x0, x1)
    sharp = _b(nb.sinkhorn_coupling, x0, x1, 1e-3, generator=torch.Generator().manual_seed(0), max_iters=20000, _budget=40_000_000)
    agree = float((sharp == exact).all(1).float().mean())
    c_sharp, c_exact, c_ind = (nb.pairing_cost(z, x1) for z in (sharp, exact, x0))
    assert agree >= 0.5 and c_sharp <= c_exact + 0.03 * (c_ind - c_exact), (
        f"as eps → 0 the entropic plan becomes the exact one: at eps = 1e-3 most draws should match the exact "
        f"pairing ({agree:.0%} did) and the pairing cost should close 97% of the gap down to the exact {c_exact:.4f} "
        f"(got {c_sharp:.4f}; independent pairs cost {c_ind:.4f})")
    counts = torch.zeros(32)
    for s in range(40):
        out = _b(nb.sinkhorn_coupling, x0, x1, 1e6, generator=torch.Generator().manual_seed(s))
        idx = (out[:, None, :] == x0[None]).all(-1).float().argmax(1)
        counts += torch.bincount(idx, minlength=32)
    stat = float(((counts - 40) ** 2 / 40).sum())
    assert stat < chi2.ppf(0.9999, 31), (
        f"as eps → ∞ the plan becomes independent and every noise index is equally likely for every data point; "
        f"chi-square = {stat:.1f} over 1,280 draws (99.99% bound {chi2.ppf(0.9999, 31):.1f})")


def test_stage2_worked_example(nb):
    x0 = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    x1 = torch.tensor([[2.0, 2.0], [2.0, 0.0]])
    hits = 0
    for s in range(800):
        out = _b(nb.sinkhorn_coupling, x0, x1, 4.0, generator=torch.Generator().manual_seed(s))
        hits += int(torch.equal(out[0], x0[1]))
    p = 1 / (1 + np.exp(-1.0))
    assert abs(hits / 800 - p) < 4 * np.sqrt(p * (1 - p) / 800), (
        f"the lesson's worked example at eps = 4: (2, 2) gets its OT partner (0, 2) with probability σ(4/eps) = "
        f"{p:.3f}; it got it in {hits / 800:.3f} of 800 draws")


def _expected_draw(x0, x1, eps, seed, max_iters=1000, tol=1e-9):
    B = len(x0)
    C = _sqdist(x0.reshape(B, -1).double().numpy(), x1.reshape(B, -1).double().numpy())
    w = np.full(B, 1 / B)
    P, _, _ = _sinkhorn(w, w, C, eps, max_iters=max_iters, tol=tol)
    Q = torch.as_tensor((P / P.sum(0, keepdims=True)).T)
    return torch.multinomial(Q, 1, replacement=True, generator=torch.Generator().manual_seed(seed)).squeeze(1), P


def test_stage2_replays_the_specified_draw(nb):
    g = torch.Generator().manual_seed(5)
    x0, x1 = torch.randn(20, 3, generator=g), torch.randn(20, 3, generator=g) + 1
    for eps, seed in [(0.5, 0), (2.0, 1)]:
        idx, _ = _expected_draw(x0, x1, eps, seed)
        out = _b(nb.sinkhorn_coupling, x0, x1, eps, generator=torch.Generator().manual_seed(seed))
        assert out.shape == x0.shape and torch.equal(out, x0[idx]), (
            f"eps = {eps}: with C[i, k] = |x0[i] − x1[k]|² and P from your topic 16 sinkhorn_log (uniform weights, "
            "the given max_iters and tol), draw idx = torch.multinomial((P / P.sum(0)).T, 1, replacement=True, "
            "generator=generator)[:, 0] and return x0[idx]: column k of the plan is where x1[k]'s noise comes from")
    idx1, _ = _expected_draw(x0, x1, 0.5, 0, max_iters=1)
    out1 = _b(nb.sinkhorn_coupling, x0, x1, 0.5, generator=torch.Generator().manual_seed(0), max_iters=1)
    assert torch.equal(out1, x0[idx1]), "max_iters is passed on to sinkhorn_log"
    out_tol = _b(nb.sinkhorn_coupling, x0, x1, 0.5, generator=torch.Generator().manual_seed(0), tol=1e3)
    assert torch.equal(out_tol, x0[idx1]), (
        "tol is passed on to sinkhorn_log too: with tol=1e3 Sinkhorn stops after one iteration, so the draw must "
        "equal the max_iters=1 draw")
    a = _b(nb.sinkhorn_coupling, x0, x1, 0.5, generator=torch.Generator().manual_seed(7))
    b = _b(nb.sinkhorn_coupling, x0, x1, 0.5, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "the same generator seed gives the same draw"


def test_stage2_conditionals_by_chi_square(nb):
    g = torch.Generator().manual_seed(6)
    x0, x1 = torch.randn(6, 2, generator=g), torch.randn(6, 2, generator=g) * 1.5
    eps = 1.0
    _, P = _expected_draw(x0, x1, eps, 0)
    cond = P / P.sum(0, keepdims=True)  # column k: distribution of the noise index paired with x1[k]
    counts = np.zeros((6, 6))
    for s in range(600):
        out = _b(nb.sinkhorn_coupling, x0, x1, eps, generator=torch.Generator().manual_seed(1000 + s))
        idx = (out[:, None, :] == x0[None]).all(-1).float().argmax(1).numpy()
        counts[idx, np.arange(6)] += 1
    expected = cond * 600
    stat, dof = 0.0, 0
    for k in range(6):  # per column, pool the cells expecting fewer than 10 draws (smallest first) until >= 10
        order = np.argsort(expected[:, k])
        obs, exp, po, pe = [], [], 0.0, 0.0
        for i in order:
            if pe < 10:
                po, pe = po + counts[i, k], pe + expected[i, k]
            else:
                obs.append(counts[i, k])
                exp.append(expected[i, k])
        obs.append(po)
        exp.append(pe)
        stat += sum((o - e) ** 2 / e for o, e in zip(obs, exp))
        dof += len(obs) - 1
    assert stat < chi2.ppf(0.9999, dof), (
        f"each x1[k] must get noise index i with probability P[i, k] / sum_i P[i, k] (a column of the plan, not a "
        f"row); chi-square over 600 draws is {stat:.1f} (99.99% bound {chi2.ppf(0.9999, dof):.1f})")


# ---------------------------------------------------------------- stage 3: the Sinkhorn loss


def _ot_value(X, Y, eps):
    C = _sqdist(X, Y)
    a, b = np.full(len(X), 1 / len(X)), np.full(len(Y), 1 / len(Y))
    _, f, g = _sinkhorn(a, b, C, eps, max_iters=20000, tol=1e-13)
    return f @ a + g @ b


def test_stage3_values(nb):
    g = _rng(7)
    X, Y = g.normal(size=(7, 2)), g.normal(size=(5, 2)) + 1
    for eps in (0.5, 2.0):
        x, y = torch.tensor(X), torch.tensor(Y)
        biased = _b(nb.sinkhorn_loss, x, y, eps, debiased=False, max_iters=20000, tol=1e-13, _budget=40_000_000)
        assert isinstance(biased, torch.Tensor) and biased.dim() == 0, "a 0-dim tensor (so you can call .backward())"
        want = _ot_value(X, Y, eps)
        assert np.isclose(float(biased), want, atol=1e-8), (
            f"eps = {eps}: debiased=False returns OT_eps(x, y), the entropic dual value at the converged potentials, "
            f"{want:.8f}; got {float(biased):.8f}")
        deb = _b(nb.sinkhorn_loss, x, y, eps, max_iters=20000, tol=1e-13, _budget=40_000_000)
        want_d = want - _ot_value(X, X, eps) / 2 - _ot_value(Y, Y, eps) / 2
        assert np.isclose(float(deb), want_d, atol=1e-8), (
            f"eps = {eps}: the default is debiased: OT(x, y) − ½ OT(x, x) − ½ OT(y, y) = {want_d:.8f}, "
            f"got {float(deb):.8f}")
    one = float(_b(nb.sinkhorn_loss, torch.tensor(X), torch.tensor(Y), 1.0, debiased=False, max_iters=1))
    a, b = np.full(7, 1 / 7), np.full(5, 1 / 5)
    C = _sqdist(X, Y)
    _, f, gg = _sinkhorn(a, b, C, 1.0, max_iters=1)
    want1 = f @ a + gg @ b - np.sum(np.outer(a, b) * (np.exp((f[:, None] + gg[None] - C) / 1.0) - 1))
    assert np.isclose(one, want1, atol=1e-10), (
        "max_iters is passed on to sinkhorn_log; with max_iters=1 the loss is the full dual objective "
        "(including the −eps Σ ab (e^… − 1) term, which is not zero before convergence)")
    one_tol = float(_b(nb.sinkhorn_loss, torch.tensor(X), torch.tensor(Y), 1.0, debiased=False, tol=1e3))
    assert np.isclose(one_tol, want1, atol=1e-10), (
        "tol is passed on to sinkhorn_log too: with tol=1e3 it stops after one iteration, so the loss must equal "
        "the max_iters=1 value")


def test_stage3_gradients_match_finite_differences(nb):
    g = _rng(8)
    X, Y = g.normal(size=(5, 2)), g.normal(size=(6, 2)) * 0.8 + 0.5
    for debiased in (False, True):
        x = torch.tensor(X, requires_grad=True)
        y = torch.tensor(Y, requires_grad=True)
        loss = _b(nb.sinkhorn_loss, x, y, 1.0, debiased=debiased, max_iters=20000, tol=1e-13, _budget=40_000_000)
        loss.backward()
        assert x.grad is not None and y.grad is not None, "the loss must be differentiable in both x and y"

        def value(Xv, Yv):
            return float(_b(nb.sinkhorn_loss, torch.tensor(Xv), torch.tensor(Yv), 1.0, debiased=debiased,
                                          max_iters=20000, tol=1e-13, _budget=40_000_000))

        h = 1e-5
        for (i, j) in [(0, 0), (3, 1)]:
            E = np.zeros_like(X)
            E[i, j] = h
            fd = (value(X + E, Y) - value(X - E, Y)) / (2 * h)
            assert np.isclose(float(x.grad[i, j]), fd, atol=1e-6, rtol=1e-4), (
                f"debiased={debiased}: d loss / d x[{i}, {j}] by finite differences is {fd:.6f}, autograd gives "
                f"{float(x.grad[i, j]):.6f}. by the envelope theorem the gradient is Σ_j P_ij ∂C_ij/∂x: build the "
                "dual from DETACHED potentials and a differentiable cost (if you got twice the right value, you let "
                "the gradient through both potentials; if 0, through neither the cost nor anything else)")
            F = np.zeros_like(Y)
            F[j + 2, i % 2] = h
            fdy = (value(X, Y + F) - value(X, Y - F)) / (2 * h)
            assert np.isclose(float(y.grad[j + 2, i % 2]), fdy, atol=1e-6, rtol=1e-4), (
                f"debiased={debiased}: the gradient with respect to y is wrong ({float(y.grad[j + 2, i % 2]):.6f} vs "
                f"{fdy:.6f}); the self-term ½ OT(y, y) depends on y through both arguments of its cost")


def test_stage3_translation_and_dtype(nb):
    g = _rng(9)
    X = g.normal(size=(8, 2))
    t = np.array([0.6, -0.8])
    s = _b(nb.sinkhorn_loss, torch.tensor(X), torch.tensor(X + t), 2.0, max_iters=20000, tol=1e-12, _budget=40_000_000)
    assert np.isclose(float(s), t @ t, atol=1e-7), (
        f"the debiased loss of a translated cloud is |t|² = {t @ t:.3f} at every eps (topic 16, on paper 3), "
        f"got {float(s):.6f}")
    x32 = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    out = _b(nb.sinkhorn_loss, x32, torch.tensor(X + t, dtype=torch.float32), 2.0, max_iters=20000, tol=1e-11, _budget=40_000_000)
    out.backward()
    assert out.dtype == torch.float32 and x32.grad.dtype == torch.float32, (
        "float32 inputs give a float32 loss and gradients (run Sinkhorn in float64 on the detached cost, then build "
        "the loss in the inputs' dtype)")
    assert np.allclose(x32.grad.numpy(), -2 * t / 8, atol=1e-4), (
        "for a translated cloud the gradient on every point is −2t/n: the loss pulls x towards y rigidly")


# ---------------------------------------------------------------- stage 4: domain adaptation


def test_stage4_exact_maps(nb):
    g = _rng(10)
    Xs = g.normal(size=(30, 2))
    perm = g.permutation(30)
    t = np.array([3.0, -1.0])
    mapped = np.asarray(_b(nb.ot_domain_adaptation, Xs, Xs[perm] * 1.0 + t))
    assert mapped.shape == (30, 2) and np.allclose(mapped, Xs + t, atol=1e-6), (
        "a translated (and shuffled) target: the exact plan matches every source point to its own translate, so "
        "the barycentric images are Xs + t")
    Xt = g.normal(size=(30, 2)) * 2
    C = _sqdist(Xs, Xt)
    _, cols = linear_sum_assignment(C)
    got = np.asarray(_b(nb.ot_domain_adaptation, Xs, Xt))
    assert np.allclose(got, Xt[cols], atol=1e-6), (
        "with eps=None and equal sizes the exact plan is an assignment, and each source point maps onto its partner")


def test_stage4_unequal_sizes_and_entropic(nb):
    g = _rng(11)
    Xs, Xt = g.normal(size=(20, 2)), g.normal(size=(13, 2)) + np.array([2.0, 0.0])
    mapped = np.asarray(_b(nb.ot_domain_adaptation, Xs, Xt))
    assert mapped.shape == (20, 2), "one image per source point"
    assert np.allclose(mapped.mean(0), Xt.mean(0), atol=1e-8), (
        "with uniform weights the images average to the target's mean (every target point receives its full mass)")
    n, m = 20, 13
    A = np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])
    C = _sqdist(Xs, Xt)
    res = linprog(C.ravel(), A_eq=A, b_eq=np.r_[np.full(n, 1 / n), np.full(m, 1 / m)], bounds=(0, None),
                  method="highs")
    P = res.x.reshape(n, m)
    want = (P @ Xt) / P.sum(1, keepdims=True)
    cost_got = np.sum((Xs - mapped) ** 2)
    assert np.isclose(cost_got, np.sum((Xs - want) ** 2), rtol=1e-6), (
        "eps=None: barycentric map of the exact LP plan (your topic 15 kantorovich_lp) with uniform weights")
    far = np.asarray(_b(nb.ot_domain_adaptation, Xs, Xt, eps=1e6))
    assert np.allclose(far, Xt.mean(0), atol=1e-4), (
        "eps → ∞: the plan is independent and every source point maps to the target's mean")
    a, b = np.full(n, 1 / n), np.full(m, 1 / m)
    for eps, it in [(0.5, 1000), (0.5, 1)]:
        Pe, _, _ = _sinkhorn(a, b, C, eps, max_iters=it)
        want_e = (Pe @ Xt) / Pe.sum(1, keepdims=True)
        got_e = np.asarray(_b(nb.ot_domain_adaptation, Xs, Xt, eps=eps, max_iters=it))
        assert np.allclose(got_e, want_e, atol=1e-8), (
            f"eps = {eps}, max_iters = {it}: barycentric map of your topic 16 sinkhorn_log plan (uniform weights, "
            "squared cost, tol 1e-9, the given max_iters)")
