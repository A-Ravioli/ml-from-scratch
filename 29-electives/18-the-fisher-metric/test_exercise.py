"""Checks for exercise.py (the fisher metric).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Closed forms are
checked by hand and against the Hessian of an exact KL divergence (computed here with autograd);
network Fisher matrices against explicit per-example loops; lengths against known geodesic
distances. Everything is float64 and deterministic.
"""
import math

import numpy as np
import torch
from torch import nn

from mlfs.testing import bounded

torch.set_num_threads(1)
D = torch.float64


def _warm_up():
    """run torch's lazily-initialised autograd paths once, so a line budget measures your code, not torch's imports."""
    m = nn.Linear(2, 2).to(D)
    out = m(torch.ones(1, 2, dtype=D))
    w = torch.zeros_like(out, requires_grad=True)
    g = torch.autograd.grad(out, list(m.parameters()), grad_outputs=w, create_graph=True)
    torch.autograd.grad(sum(gi.sum() for gi in g), w)
    torch.autograd.functional.hessian(lambda t: (t ** 3).sum(), torch.ones(2, dtype=D))
    torch.linalg.solve(torch.eye(2, dtype=D), torch.ones(2, dtype=D))


_warm_up()


def _kl_cat(t0, t):
    p0 = torch.softmax(t0, -1)
    return (p0 * (torch.log_softmax(t0, -1) - torch.log_softmax(t, -1))).sum()


def _kl_gauss(t0, t):
    """KL(N(m0, s0^2) || N(m, s^2)) with t = (mean, log_std)."""
    m0, l0 = t0[0], t0[1]
    m, l = t[0], t[1]
    return l - l0 + (torch.exp(2 * l0) + (m0 - m) ** 2) / (2 * torch.exp(2 * l)) - 0.5


def _hess(f, x):
    return torch.autograd.functional.hessian(f, x)


def _mlp(seed, d=3, h=4, k=3):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, k)).to(D)


def _loop_fisher(model, X, y=None):
    """explicit per-example, per-class loop: true Fisher (y None) or empirical Fisher."""
    params = list(model.parameters())
    P = sum(p.numel() for p in params)
    F = torch.zeros(P, P, dtype=D)
    for n in range(X.shape[0]):
        logp = torch.log_softmax(model(X[n:n + 1]), -1)[0]
        classes = range(logp.shape[0]) if y is None else [int(y[n])]
        for c in classes:
            g = torch.autograd.grad(logp[c], params, retain_graph=True)
            g = torch.cat([gi.reshape(-1) for gi in g])
            w = logp[c].exp().item() if y is None else 1.0
            F += w * torch.outer(g, g)
    return F / X.shape[0]


# ---------------------------------------------------------------- stage 1: closed forms


def test_stage1_categorical_hand_example(nb):
    logits = torch.tensor([0.0, 0.0, math.log(2.0)], dtype=D)
    F = nb.categorical_fisher(logits)
    want = torch.tensor([[3, -1, -2], [-1, 3, -2], [-2, -2, 4]], dtype=D) / 16
    assert tuple(F.shape) == (3, 3), f"logits (3,) -> Fisher (3, 3); got shape {tuple(F.shape)}"
    assert torch.allclose(F, want, atol=1e-12), (
        f"p = (1/4, 1/4, 1/2), so F = diag(p) - p p^T = [[3,-1,-2],[-1,3,-2],[-2,-2,4]]/16.\n"
        f"got\n{F}\nif the off-diagonal entries are 0, you forgot the -p p^T term (diag(p) alone is not the Fisher)")


def test_stage1_categorical_is_kl_curvature(nb):
    g = torch.Generator().manual_seed(0)
    for _ in range(3):
        t0 = 2 * torch.randn(5, generator=g, dtype=D)
        want = _hess(lambda t: _kl_cat(t0, t), t0.clone())
        got = nb.categorical_fisher(t0)
        assert torch.allclose(got, want, atol=1e-10), (
            f"the Fisher matrix must equal the Hessian of KL(p_theta0 || p_theta) at theta0. for logits {t0.tolist()} "
            f"the max difference is {(got - want).abs().max():.3g}")


def test_stage1_categorical_batched_and_null_direction(nb):
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(4, 6, generator=g, dtype=D)
    F = nb.categorical_fisher(logits)
    assert tuple(F.shape) == (4, 6, 6), f"batched logits (4, 6) -> (4, 6, 6); got {tuple(F.shape)}"
    for i in range(4):
        assert torch.allclose(F[i], nb.categorical_fisher(logits[i]), atol=1e-12), (
            "the batched result must match the unbatched one row by row")
    ones = torch.ones(6, dtype=D)
    assert torch.allclose(F @ ones, torch.zeros(4, 6, dtype=D), atol=1e-12), (
        "F @ (1, ..., 1) should be 0: adding a constant to every logit leaves softmax unchanged, so the Fisher "
        "must give that direction zero length")


def test_stage1_gaussian(nb):
    F = nb.gaussian_fisher(torch.tensor(1.0, dtype=D), torch.tensor(math.log(2.0), dtype=D))
    want = torch.tensor([[0.25, 0.0], [0.0, 2.0]], dtype=D)
    assert torch.is_tensor(F) and tuple(F.shape) == (2, 2), "gaussian_fisher returns a (2, 2) tensor"
    assert torch.allclose(F.to(D), want, atol=1e-12), (
        f"N(1, 2^2) in (mean, log_std) coordinates: F = diag(1/sigma^2, 2) = diag(0.25, 2); got\n{F}\n"
        "if the second entry is 2/sigma^2 = 0.5, you computed the Fisher for sigma, not log sigma")
    for m, ls in [(0.0, 0.0), (-3.0, -1.2), (5.0, 0.7)]:
        t0 = torch.tensor([m, ls], dtype=D)
        want = _hess(lambda t: _kl_gauss(t0, t), t0.clone())
        got = nb.gaussian_fisher(t0[0], t0[1]).to(D)
        assert torch.allclose(got, want, atol=1e-10), (
            f"at (mean, log_std) = ({m}, {ls}) the Hessian of the Gaussian KL is\n{want}\nbut you returned\n{got}")


def test_stage1_gaussian_accepts_floats(nb):
    F = nb.gaussian_fisher(0.5, math.log(0.5))
    assert torch.is_tensor(F) and tuple(F.shape) == (2, 2), (
        "gaussian_fisher(mean, log_std) with plain Python floats must return a (2, 2) tensor too")
    assert torch.allclose(F.to(D), torch.diag(torch.tensor([4.0, 2.0], dtype=D)), atol=1e-12), (
        f"N(0.5, 0.5^2): F = diag(1/sigma^2, 2) = diag(4, 2); got\n{F}")


# ---------------------------------------------------------------- stage 2: any small model


def test_stage2_exact_categorical(nb):
    t = torch.tensor([0.3, -1.0, 2.0, 0.0], dtype=D)
    F = bounded(nb.fisher_exact, lambda th, x: torch.log_softmax(th, -1)[x], t, range(4), what="fisher_exact")
    p = torch.softmax(t, -1)
    want = torch.diag(p) - torch.outer(p, p)
    assert tuple(F.shape) == (4, 4), f"theta (4,) -> Fisher (4, 4); got {tuple(F.shape)}"
    assert torch.allclose(F, want, atol=1e-12), (
        f"summing p(x) s(x) s(x)^T over x in 0..3 should give diag(p) - p p^T\n{want}\ngot\n{F}\n"
        "if every entry is too big by a constant factor, you averaged s s^T without weighting by p_theta(x)")
    assert not t.requires_grad, "don't modify the theta you were given (work on a detached copy)"


def test_stage2_exact_binomial(nb):
    n = 6

    def log_prob(th, x):
        p = torch.sigmoid(th[0])
        return (math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
                + x * torch.log(p) + (n - x) * torch.log1p(-p))

    for th in [-1.5, 0.0, 0.8]:
        F = bounded(nb.fisher_exact, log_prob, torch.tensor([th], dtype=D), range(n + 1))
        p = 1 / (1 + math.exp(-th))
        assert abs(F.reshape(-1)[0].item() - n * p * (1 - p)) < 1e-10, (
            f"Binomial({n}, sigmoid({th})) in the logit has Fisher n p (1 - p) = {n * p * (1 - p):.6f}; "
            f"got {F.reshape(-1)[0].item():.6f}")


def test_stage2_exact_equals_expected_negative_hessian(nb):
    # a 2-parameter discrete model on {0, ..., 4}: log p(x) = a x + b x^2 - A(a, b)
    xs = torch.arange(5, dtype=D)

    def log_prob(th, x):
        e = th[0] * xs + th[1] * xs ** 2
        return e[x] - torch.logsumexp(e, 0)

    th = torch.tensor([0.4, -0.2], dtype=D)
    F = bounded(nb.fisher_exact, log_prob, th, range(5))
    p = torch.stack([log_prob(th, x) for x in range(5)]).exp()
    want = sum(p[x] * -_hess(lambda t: log_prob(t, x), th.clone()) for x in range(5))
    assert torch.allclose(F, want, atol=1e-10), (
        f"E[s s^T] must equal -E[Hessian of log p] (the two definitions of Fisher information):\n{want}\ngot\n{F}")


def test_stage2_kl_hessian(nb):
    t0 = torch.tensor([0.5, -0.3, 1.2], dtype=D)
    H = nb.kl_hessian(_kl_cat, t0)
    p = torch.softmax(t0, -1)
    assert torch.allclose(H, torch.diag(p) - torch.outer(p, p), atol=1e-10), (
        f"the Hessian of KL(p_theta0 || p_theta) at theta0 is the categorical Fisher; got\n{H}")
    t0 = torch.tensor([2.0, math.log(0.5)], dtype=D)
    H = nb.kl_hessian(_kl_gauss, t0)
    assert torch.allclose(H, torch.diag(torch.tensor([4.0, 2.0], dtype=D)), atol=1e-10), (
        f"for N(2, 0.5^2) in (mean, log_std) the KL Hessian is diag(1/0.25, 2) = diag(4, 2); got\n{H}\n"
        "check that you differentiate twice (a Hessian, not a gradient) and evaluate at theta = theta0. "
        "(differentiating in either argument gives the same Hessian at theta0, by on paper 3)")


def test_stage2_kl_hessian_at_theta0_not_elsewhere(nb):
    # a KL whose Hessian depends on where you evaluate it: catch evaluating at zeros or at a shifted point
    t0 = torch.tensor([1.5, -2.0, 0.0, 0.7], dtype=D)
    H = nb.kl_hessian(_kl_cat, t0)
    p = torch.softmax(t0, -1)
    assert torch.allclose(H, torch.diag(p) - torch.outer(p, p), atol=1e-10), (
        "kl_hessian must evaluate the Hessian at theta = theta0 (the KL's minimum), not at zero")


# ---------------------------------------------------------------- stage 3: network Fisher matrices


def test_stage3_true_fisher_matches_loop(nb):
    model = _mlp(0)
    X = torch.randn(5, 3, generator=torch.Generator().manual_seed(0), dtype=D)
    F = nb.true_fisher(model, X)
    P = sum(p.numel() for p in model.parameters())
    assert tuple(F.shape) == (P, P), f"P = {P} parameters -> ({P}, {P}); got {tuple(F.shape)}"
    want = _loop_fisher(model, X)
    err = (F - want).abs().max().item()
    assert err < 1e-10, (
        f"true Fisher differs from (1/N) sum_n sum_y p(y|x_n) g g^T by up to {err:.3g}. common causes: using the "
        "data labels (that's the EMPIRICAL Fisher), forgetting the 1/N, or flattening parameters in a different "
        "order than model.parameters() (weight then bias, layer by layer, each .reshape(-1))")


def test_stage3_true_fisher_is_nll_hessian_for_softmax_regression(nb):
    torch.manual_seed(3)
    model = nn.Linear(3, 4).to(D)
    X = torch.randn(6, 3, generator=torch.Generator().manual_seed(1), dtype=D)
    y = torch.tensor([0, 1, 2, 3, 1, 0])
    W, b = model.weight.detach(), model.bias.detach()

    def nll(theta):
        Wt, bt = theta[:12].reshape(4, 3), theta[12:]
        return nn.functional.cross_entropy(X @ Wt.T + bt, y)

    H = _hess(nll, torch.cat([W.reshape(-1), b]))
    F = nb.true_fisher(model, X)
    assert torch.allclose(F, H, atol=1e-10), (
        "for softmax regression (a linear layer + softmax) the true Fisher equals the Hessian of the mean "
        f"cross-entropy, whatever the labels. max difference {(F - H).abs().max():.3g}")


def test_stage3_empirical_fisher_matches_loop(nb):
    model = _mlp(2)
    g = torch.Generator().manual_seed(2)
    X = torch.randn(7, 3, generator=g, dtype=D)
    y = torch.randint(0, 3, (7,), generator=g)
    EF = nb.empirical_fisher(model, X, y)
    want = _loop_fisher(model, X, y)
    err = (EF - want).abs().max().item()
    assert err < 1e-10, (
        f"empirical Fisher differs from (1/N) sum_n g_n g_n^T (g_n the score at the DATA label y_n) by {err:.3g}")
    params_before = [p.detach().clone() for p in model.parameters()]
    nb.empirical_fisher(model, X, y)
    assert all(torch.equal(a, p) for a, p in zip(params_before, model.parameters())), (
        "computing a Fisher matrix must not change the model's parameters")


def test_stage3_empirical_is_not_true(nb):
    model = _mlp(4)
    g = torch.Generator().manual_seed(4)
    X = torch.randn(3, 3, generator=g, dtype=D)
    y = torch.randint(0, 3, (3,), generator=g)
    EF, F = nb.empirical_fisher(model, X, y), nb.true_fisher(model, X)
    rank_ef = torch.linalg.matrix_rank(EF, atol=1e-9).item()
    rank_f = torch.linalg.matrix_rank(F, atol=1e-9).item()
    assert rank_ef <= 3, (
        f"with N = 3 examples the empirical Fisher is a sum of 3 outer products, so its rank is at most 3; got {rank_ef}")
    assert rank_f > rank_ef, (
        f"the true Fisher averages over all K classes per example (rank up to N(K-1) = 6), the empirical one only over "
        f"the observed label: ranks {rank_f} vs {rank_ef}. are you using the labels in true_fisher?")
    for M, name in [(EF, "empirical"), (F, "true")]:
        assert torch.allclose(M, M.T, atol=1e-12), f"the {name} Fisher must be symmetric"
        assert torch.linalg.eigvalsh(M).min() > -1e-10, f"the {name} Fisher must be positive semidefinite"


# ---------------------------------------------------------------- stage 4: coordinates and lengths


def test_stage4_pullback_hand_example(nb):
    F = torch.diag(torch.tensor([2.0, 3.0], dtype=D))
    J = torch.tensor([[1.0, 1.0], [0.0, 2.0]], dtype=D)
    got = nb.pullback_fisher(F, J)
    want = torch.tensor([[2.0, 2.0], [2.0, 14.0]], dtype=D)
    assert torch.allclose(got, want), (
        f"J^T F J with F = diag(2, 3), J = [[1, 1], [0, 2]] is [[2, 2], [2, 14]]; got\n{got}\n"
        "if you got [[5, 6], [6, 12]], you computed J F J^T")
    J2 = torch.tensor([[1.0], [2.0]], dtype=D)
    got = nb.pullback_fisher(F, J2)
    assert tuple(got.shape) == (1, 1) and abs(got.item() - 14.0) < 1e-12, (
        "a 1-parameter curve inside a 2-parameter model: J is (2, 1) and the pulled-back Fisher is (1, 1) = 14")


def test_stage4_pullback_bernoulli(nb):
    # Bernoulli Fisher in p is 1/(p(1-p)); pulled back through p = sigmoid(t) it must become p(1-p)
    for t in [-2.0, 0.3, 1.7]:
        p = 1 / (1 + math.exp(-t))
        got = nb.pullback_fisher(torch.tensor([[1 / (p * (1 - p))]], dtype=D), torch.tensor([[p * (1 - p)]], dtype=D))
        assert abs(got.item() - p * (1 - p)) < 1e-12, f"at logit {t}: expected p(1-p) = {p * (1 - p):.6f}, got {got.item()}"


def _bern_logit(t):
    p = torch.sigmoid(t[0])
    return (p * (1 - p)).reshape(1, 1)


def _bern_p(q):
    return (1 / (q[0] * (1 - q[0]))).reshape(1, 1)


def test_stage4_length_bernoulli_is_coordinate_free(nb):
    p0, p1 = 0.1, 0.9
    exact = 2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p0)))
    tt = torch.linspace(math.log(p0 / (1 - p0)), math.log(p1 / (1 - p1)), 801, dtype=D).reshape(-1, 1)
    L_logit = bounded(nb.fisher_rao_length, _bern_logit, tt, what="fisher_rao_length", budget=2_000_000)
    L_p = bounded(nb.fisher_rao_length, _bern_p, torch.sigmoid(tt))
    assert isinstance(L_logit, float), f"return a Python float, got {type(L_logit).__name__}"
    assert abs(L_logit - exact) < 1e-4, (
        f"the Fisher-Rao distance from Bernoulli(0.1) to Bernoulli(0.9) is 2|asin sqrt(0.9) - asin sqrt(0.1)| = "
        f"{exact:.5f}; your length along the logit path is {L_logit:.5f}. did you take the square root of each "
        "segment's d^T F d before summing?")
    assert abs(L_p - exact) < 1e-4, (
        f"the same curve described by p instead of the logit must have the same length {exact:.5f}; got {L_p:.5f}")


def test_stage4_length_ignores_point_spacing(nb):
    # same curve (sigma from 1 to 8 at fixed mean), unevenly spaced points: length sqrt(2) log 8
    s = torch.tensor(np.geomspace(1.0, 8.0, 400), dtype=D)
    s = torch.cat([s[:50], s[50:-1:3], s[-1:]])  # dense, then sparse, ending exactly at 8
    path = torch.stack([torch.zeros_like(s), s], 1)

    def fisher_mean_std(t):
        return torch.diag(torch.stack([1 / t[1] ** 2, 2 / t[1] ** 2]))

    L = bounded(nb.fisher_rao_length, fisher_mean_std, path)
    want = math.sqrt(2) * math.log(8.0)
    assert abs(L - want) < 2e-3, (
        f"scaling a Gaussian's sigma from 1 to 8 has Fisher-Rao length sqrt(2) log 8 = {want:.4f}, however the "
        f"points along the path are spaced; got {L:.4f}. evaluate F at each segment's MIDPOINT (the spec), not its start")
    flat = bounded(nb.fisher_rao_length, lambda t: torch.eye(2, dtype=D), path)
    assert abs(flat - 7.0) < 1e-9, f"with F = I the length is the Euclidean length 7; got {flat}"
