"""Checks for exercise.py (natural gradient).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Directions are
checked by hand and by brute force over the constraint ellipse; the optimizer by exact affine
invariance and by its one-step behaviour on a Gaussian; the matrix-free Fisher against an
explicit per-example loop; the classifier step against Newton's method on softmax regression
(where Fisher = Hessian). Everything is float64; loops run under a line budget.
"""
import math

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


def _gauss_grad(x):
    """gradient of the mean NLL of N(mu, exp(l)^2) in (mu, l)."""
    def g(t):
        mu, l = t[0], t[1]
        s2 = torch.exp(2 * l)
        return torch.stack([-(x.mean() - mu) / s2, 1 - ((x - mu) ** 2).mean() / s2])
    return g


def _gauss_fisher(t):
    return torch.diag(torch.stack([torch.exp(-2 * t[1]), torch.tensor(2.0, dtype=D)]))


def _mlp(seed, d=3, h=4, k=3):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, k)).to(D)


def _loop_fisher(model, X):
    params = list(model.parameters())
    P = sum(p.numel() for p in params)
    F = torch.zeros(P, P, dtype=D)
    for n in range(X.shape[0]):
        logp = torch.log_softmax(model(X[n:n + 1]), -1)[0]
        for c in range(logp.shape[0]):
            g = torch.cat([gi.reshape(-1) for gi in torch.autograd.grad(logp[c], params, retain_graph=True)])
            F += logp[c].exp().item() * torch.outer(g, g)
    return F / X.shape[0]


# ---------------------------------------------------------------- stage 1: steepest descent under a metric


def test_stage1_steepest_hand_example(nb):
    g = torch.tensor([1.0, -1.0], dtype=D)
    M = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=D)
    d = nb.steepest_direction(g, M, 0.1)
    # M^-1 g = (1.5, -2.5) / 1.75 = (0.857, -1.429); g . M^-1 g = 4 / 1.75 = 2.286
    want = -0.1 * torch.tensor([1.5, -2.5], dtype=D) / 1.75 / math.sqrt(4 / 1.75)
    assert torch.allclose(d, want, atol=1e-10), (
        f"expected -r M^-1 g / sqrt(g^T M^-1 g) = {want.tolist()}, got {d.tolist()}. if you got a multiple of "
        "-g, you ignored the metric; if the length is wrong, check d^T M d = r^2 (not ||d|| = r)")


def test_stage1_steepest_beats_every_direction_on_the_ellipse(nb):
    g = torch.tensor([0.3, 2.0], dtype=D)
    M = torch.tensor([[5.0, 1.0], [1.0, 0.5]], dtype=D)
    r = 0.7
    d = nb.steepest_direction(g, M, r)
    assert abs(float(d @ M @ d) - r * r) < 1e-10, f"the step must lie on d^T M d = r^2 = {r * r}; got {float(d @ M @ d)}"
    L = torch.linalg.cholesky(M)  # M = L L^T, so d = r L^-T u covers the ellipse as u covers the circle
    ang = torch.linspace(0, 2 * math.pi, 3601, dtype=D)
    U = torch.stack([ang.cos(), ang.sin()], 1)
    cands = r * torch.linalg.solve_triangular(L.T, U.T, upper=True).T
    best = (cands @ g).min().item()
    assert float(g @ d) <= best + 1e-6, (
        f"some step on the ellipse decreases the linear model more: best of 3,601 candidates gives g.d = {best:.5f}, "
        f"yours gives {float(g @ d):.5f}")
    d_eu = nb.steepest_direction(g, torch.eye(2, dtype=D), r)
    assert torch.allclose(d_eu, -r * g / g.norm(), atol=1e-12), "with M = I it's plain steepest descent, -r g / ||g||"


def test_stage1_natural_gradient(nb):
    F = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=D)
    F_copy = F.clone()
    g = torch.tensor([1.0, -1.0], dtype=D)
    ng = nb.natural_gradient(g, F)
    assert torch.allclose(ng, torch.tensor([1.5, -2.5], dtype=D) / 1.75, atol=1e-12), (
        f"F^-1 g = (0.857, -1.429); got {ng.tolist()}")
    ng = nb.natural_gradient(g, F, damping=0.5)
    assert torch.allclose(ng, torch.linalg.solve(F + 0.5 * torch.eye(2, dtype=D), g), atol=1e-12), (
        "with damping, solve (F + damping I) x = g")
    assert torch.equal(F, F_copy), "don't modify the Fisher matrix in place when adding the damping"


def test_stage1_direction_is_coordinate_free(nb):
    # theta = A psi: gradient and metric pull back; the steepest step in psi, pushed through A, is the one in theta
    g = torch.tensor([0.4, -1.3, 0.2], dtype=D)
    B = torch.randn(3, 3, generator=torch.Generator().manual_seed(0), dtype=D)
    M = B @ B.T + 0.5 * torch.eye(3, dtype=D)
    A = torch.tensor([[2.0, 0.0, 1.0], [0.0, 0.1, 0.0], [1.0, 1.0, 3.0]], dtype=D)
    d_theta = nb.steepest_direction(g, M, 0.2)
    d_psi = nb.steepest_direction(A.T @ g, A.T @ M @ A, 0.2)
    assert torch.allclose(A @ d_psi, d_theta, atol=1e-10), (
        "the steepest step must not depend on the coordinates: with gradient A^T g and metric A^T M A in psi, "
        "A d_psi should equal d_theta")


# ---------------------------------------------------------------- stage 2: the optimizer


def test_stage2_gaussian_one_step(nb):
    x = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=D)
    t0 = torch.tensor([0.0, math.log(2.0)], dtype=D)
    path = bounded(nb.natural_gradient_descent, _gauss_grad(x), _gauss_fisher, t0, lr=1.0, n_steps=1,
                   what="natural_gradient_descent")
    assert tuple(path.shape) == (2, 2), f"1 step on 2 parameters -> path (2, 2) including theta0; got {tuple(path.shape)}"
    assert torch.equal(path[0], t0), "path[0] must be theta0"
    # natural step on mu: (x_bar - mu) exactly; on log sigma: -(1 - m2 / sigma^2) / 2 with m2 = 12.5, sigma^2 = 4
    want = torch.tensor([3.0, math.log(2.0) + 1.0625], dtype=D)
    assert torch.allclose(path[1], want, atol=1e-10), (
        f"from (mu, log sigma) = (0, log 2) on data (1, 2, 3, 6) one natural step with lr 1 lands on mu = 3 (the "
        f"sample mean) and log sigma = log 2 + 1.0625 = {want[1]:.4f}; got {path[1].tolist()}. "
        "if mu moved to 0.75, you took a plain gradient step")
    assert torch.equal(t0, torch.tensor([0.0, math.log(2.0)], dtype=D)), "don't modify theta0 in place"


def test_stage2_exactly_invariant_to_affine_reparameterization(nb):
    x = torch.tensor([0.5, 1.5, 2.0, 4.0, -1.0], dtype=D)
    grad, fisher = _gauss_grad(x), _gauss_fisher
    A = torch.tensor([[3.0, 1.0], [0.0, 0.2]], dtype=D)
    b = torch.tensor([1.0, -0.5], dtype=D)
    psi0 = torch.tensor([0.0, 1.0], dtype=D)
    path_psi = bounded(nb.natural_gradient_descent, lambda p: A.T @ grad(A @ p + b), lambda p: A.T @ fisher(A @ p + b) @ A,
                                           psi0, lr=0.3, n_steps=15)
    path_theta = bounded(nb.natural_gradient_descent, grad, fisher, A @ psi0 + b, lr=0.3, n_steps=15)
    mapped = path_psi @ A.T + b
    err = (mapped - path_theta).abs().max().item()
    assert err < 1e-9, (
        f"under theta = A psi + b, natural gradient descent in psi must trace exactly the same models as in theta "
        f"(any step size). the paths differ by up to {err:.3g}: are you preconditioning by the Fisher?")


def test_stage2_nearly_invariant_to_nonlinear_reparameterization(nb):
    x = torch.tensor([0.5, 1.5, 2.0, 4.0, -1.0], dtype=D)
    grad_l = _gauss_grad(x)

    def grad_s(t):  # coordinates (mu, sigma)
        g = grad_l(torch.stack([t[0], torch.log(t[1])]))
        return torch.stack([g[0], g[1] / t[1]])

    def fisher_s(t):
        return torch.diag(torch.stack([1 / t[1] ** 2, 2 / t[1] ** 2]))

    pl = bounded(nb.natural_gradient_descent, grad_l, _gauss_fisher, torch.tensor([0.0, 0.0], dtype=D), lr=0.01, n_steps=600)
    ps = bounded(nb.natural_gradient_descent, grad_s, fisher_s, torch.tensor([0.0, 1.0], dtype=D), lr=0.01, n_steps=600)
    gap = (pl[:, 1].exp() - ps[:, 1]).abs().max().item()
    assert gap < 0.01, (
        f"with a small step size, natural gradient descent in (mu, sigma) and in (mu, log sigma) should trace nearly "
        f"the same distributions (a plain gradient descent pair differs by about 0.2); their sigmas differ by up to "
        f"{gap:.3f}")
    assert abs(pl[-1, 0].item() - x.mean().item()) < 0.05 and abs(pl[-1, 1].exp().item() - x.std(unbiased=False).item()) < 0.05, (
        "after 600 steps of lr 0.01 the fit should be close to the MLE (sample mean and std)")


def test_stage2_damping(nb):
    x = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=D)
    t0 = torch.tensor([0.5, 0.3], dtype=D)
    path = bounded(nb.natural_gradient_descent, _gauss_grad(x), _gauss_fisher, t0, lr=1.0, n_steps=1, damping=1e6)
    want = t0 - _gauss_grad(x)(t0) / 1e6
    assert torch.allclose(path[1], want, rtol=1e-5, atol=1e-12), (
        "with huge damping, (F + damping I)^-1 g is about g / damping: did you pass damping through?")


# ---------------------------------------------------------------- stage 3: matrix-free


def test_stage3_cg_solves(nb):
    g = torch.Generator().manual_seed(1)
    B = torch.randn(8, 8, generator=g, dtype=D)
    A = B @ B.T + 2 * torch.eye(8, dtype=D)
    b = torch.randn(8, generator=g, dtype=D)
    x = bounded(nb.conjugate_gradient, lambda v: A @ v, b, max_iters=100, tol=1e-24, what="conjugate_gradient")
    want = torch.linalg.solve(A, b)
    assert torch.allclose(x, want, atol=1e-7), f"CG should solve A x = b; max error {(x - want).abs().max():.3g}"


def test_stage3_cg_first_iterate_and_budget(nb):
    A = torch.diag(torch.tensor([1.0, 4.0, 10.0], dtype=D))
    b = torch.tensor([1.0, 1.0, 1.0], dtype=D)
    x1 = bounded(nb.conjugate_gradient, lambda v: A @ v, b, max_iters=1)
    want = (b @ b) / (b @ A @ b) * b
    assert torch.allclose(x1, want, atol=1e-12), (
        f"one CG iteration from x0 = 0 is the exact line search along b: x1 = (b.b / b.Ab) b = {want.tolist()}; "
        f"got {x1.tolist()}")
    calls = []

    def counted(v):
        calls.append(1)
        return A @ v

    x = bounded(nb.conjugate_gradient, counted, b, max_iters=50)
    assert len(calls) <= 4, (
        f"a 3x3 SPD system needs at most 3 CG iterations, then the residual is ~0 and you should stop "
        f"(squared residual <= tol); you called mvp {len(calls)} times")
    assert torch.allclose(x, torch.tensor([1.0, 0.25, 0.1], dtype=D), atol=1e-9)
    calls.clear()
    bounded(nb.conjugate_gradient, counted, torch.zeros(3, dtype=D))
    assert len(calls) == 0, "b = 0 is already solved by x0 = 0: check the residual before the first mvp"


def test_stage3_cg_stops_on_squared_residual(nb):
    calls = []

    def counted(v):
        calls.append(1)
        return 2.0 * v

    b = torch.tensor([0.5, 0.0, 0.0], dtype=D)  # r.r = 0.25 <= tol = 0.3, but ||r|| = 0.5 > 0.3
    x = bounded(nb.conjugate_gradient, counted, b, max_iters=10, tol=0.3, what="conjugate_gradient")
    assert len(calls) == 0 and torch.equal(x, torch.zeros(3, dtype=D)), (
        "the stopping rule compares the SQUARED residual norm r.r with tol: here r.r = 0.25 <= 0.3, so CG must "
        f"return x0 = 0 without any mvp call; you made {len(calls)} calls")


def test_stage3_fvp_matches_explicit(nb):
    model = _mlp(0)
    X = torch.randn(6, 3, generator=torch.Generator().manual_seed(0), dtype=D)
    F = _loop_fisher(model, X)
    P = F.shape[0]
    V = torch.randn(3, P, generator=torch.Generator().manual_seed(1), dtype=D)
    for v in V:
        got = nb.fisher_vector_product(model, X, v)
        assert tuple(got.shape) == (P,), f"v is flat (P,) = ({P},); the product must be too, got {tuple(got.shape)}"
        assert torch.allclose(got, F @ v, atol=1e-10), (
            f"(F v) differs from the explicit true Fisher times v by {(got - F @ v).abs().max():.3g}. check the "
            "middle factor diag(p) - p p^T, the 1/N, and the parameters() flattening order")
    got = nb.fisher_vector_product(model, X, V[0], damping=0.3)
    assert torch.allclose(got, F @ V[0] + 0.3 * V[0], atol=1e-10), "with damping, return (F + damping I) v"
    assert all(p.grad is None for p in model.parameters()), (
        "use torch.autograd.grad, not .backward(): a Fisher-vector product must not leave gradients in .grad")


# ---------------------------------------------------------------- stage 4: one step on a classifier


def test_stage4_is_newton_for_softmax_regression(nb):
    torch.manual_seed(5)
    model = nn.Linear(3, 4).to(D)
    gen = torch.Generator().manual_seed(5)
    X = torch.randn(20, 3, generator=gen, dtype=D)
    y = torch.randint(0, 4, (20,), generator=gen)
    theta = torch.cat([model.weight.detach().reshape(-1), model.bias.detach()])

    def nll(t):
        return nn.functional.cross_entropy(X @ t[:12].reshape(4, 3).T + t[12:], y)

    H = torch.autograd.functional.hessian(nll, theta.clone())
    g = torch.autograd.functional.jacobian(nll, theta.clone())
    newton = theta - torch.linalg.pinv(H, hermitian=True) @ g
    loss0 = bounded(nb.ngd_step, model, X, y, lr=1.0, damping=0.0, cg_iters=16, what="ngd_step",
                    budget=2_000_000)
    assert isinstance(loss0, float) and abs(loss0 - nll(theta).item()) < 1e-12, (
        f"ngd_step returns the loss BEFORE the step as a float ({nll(theta).item():.6f}); got {loss0!r}")
    after = torch.cat([model.weight.detach().reshape(-1), model.bias.detach()])
    err = (after - newton).abs().max().item()
    assert err < 1e-3 and abs(nll(after).item() - nll(newton).item()) < 1e-5, (
        f"for softmax regression the Fisher is the Hessian, so a natural step with lr 1 and no damping is a "
        f"Newton step; your parameters differ from Newton's by {err:.3g}. did you update in place, in "
        f"parameters() order, with step = F^-1 g from conjugate_gradient?")


def test_stage4_lr_and_damping_scale(nb):
    model = _mlp(3)
    gen = torch.Generator().manual_seed(3)
    X = torch.randn(10, 3, generator=gen, dtype=D)
    y = torch.randint(0, 3, (10,), generator=gen)
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    params = list(model.parameters())
    g = torch.cat([gi.reshape(-1) for gi in torch.autograd.grad(nn.functional.cross_entropy(model(X), y), params)])
    F = _loop_fisher(model, X)
    want = 0.5 * torch.linalg.solve(F + 10.0 * torch.eye(F.shape[0], dtype=D), g)
    bounded(nb.ngd_step, model, X, y, lr=0.5, damping=10.0, cg_iters=60, what="ngd_step", budget=3_000_000)
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    err = ((theta - after) - want).norm().item() / want.norm().item()
    assert err < 1e-4, (
        f"with lr 0.5 and damping 10 the step must be 0.5 (F + 10 I)^-1 g; relative error {err:.2g}. check that "
        "you pass lr and damping through")


def test_stage4_uses_cg_iters(nb):
    # one CG iteration from 0 is the exact line search along g: step = (g.g / g.(F + l I) g) g
    model = _mlp(12)
    gen = torch.Generator().manual_seed(12)
    X = torch.randn(10, 3, generator=gen, dtype=D) * torch.tensor([20.0, 1.0, 0.05], dtype=D)
    y = torch.randint(0, 3, (10,), generator=gen)
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    params = list(model.parameters())
    g = torch.cat([gi.reshape(-1) for gi in torch.autograd.grad(nn.functional.cross_entropy(model(X), y), params)])
    M = _loop_fisher(model, X) + 1e-3 * torch.eye(g.numel(), dtype=D)
    want = (g @ g) / (g @ M @ g) * g
    bounded(nb.ngd_step, model, X, y, lr=1.0, damping=1e-3, cg_iters=1, what="ngd_step", budget=2_000_000)
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    err = ((theta - after) - want).norm().item() / want.norm().item()
    assert err < 1e-8, (
        f"with cg_iters=1 the step must be one CG iteration, (g.g / g.(F + damping I) g) g; relative error "
        f"{err:.2g}. pass cg_iters to conjugate_gradient as max_iters")


def test_stage3_fvp_never_forms_the_fisher(nb):
    model = _mlp(13)
    X = torch.randn(40, 3, generator=torch.Generator().manual_seed(13), dtype=D)
    calls, back = [], []

    def _fwd(m, i, o):
        calls.append(1)
        if o.requires_grad:  # count every backward pass that reaches the logits
            o.register_hook(lambda g: back.append(1))

    h = model.register_forward_hook(_fwd)
    try:
        P = sum(p.numel() for p in model.parameters())
        nb.fisher_vector_product(model, X, torch.ones(P, dtype=D))
    finally:
        h.remove()
    assert len(calls) <= 3, (
        f"a Fisher-vector product should run the model on the whole batch once (a few times at most), then use "
        f"backward passes; you ran it {len(calls)} times. per-example or per-parameter loops form F (or J) "
        "explicitly, which is what this function exists to avoid")
    assert len(back) <= 4, (
        f"a Fisher-vector product needs only a few backward passes through the logits (about 2-3 with the "
        f"double-backward trick); yours made {len(back)}. building the logit Jacobian row by row (one "
        "autograd.grad per logit) forms J, and hence F, explicitly, which is what this function exists to avoid")


def test_stage4_trains_fast(nb):
    # ill-conditioned softmax regression: one feature 30x larger than another
    finals = []
    for seed in range(5):
        gen = torch.Generator().manual_seed(seed)
        X = torch.randn(200, 3, generator=gen, dtype=D) * torch.tensor([30.0, 1.0, 0.1], dtype=D)
        W = torch.tensor([[0.05, 1.0, 5.0], [-0.05, -1.0, 0.0], [0.0, 0.5, -5.0]], dtype=D)
        y = torch.multinomial(torch.softmax(X @ W.T, -1), 1, generator=gen).squeeze(1)
        torch.manual_seed(seed)
        model = nn.Linear(3, 3).to(D)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        ref = nn.Linear(3, 3).to(D)
        opt = torch.optim.LBFGS(ref.parameters(), max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(ref(X), y)
            loss.backward()
            return loss

        opt.step(closure)
        best = nn.functional.cross_entropy(ref(X), y).item()
        for _ in range(8):
            bounded(nb.ngd_step, model, X, y, lr=1.0, damping=1e-6, cg_iters=30, what="ngd_step",
                    budget=3_000_000)
        finals.append(nn.functional.cross_entropy(model(X), y).item() - best)
    finals.sort()
    assert finals[2] < 1e-4, (
        f"8 natural-gradient steps (lr 1) on ill-conditioned softmax regression should reach the optimum "
        f"(median excess loss < 1e-4 over 5 seeds); got excess losses {[f'{v:.2g}' for v in finals]}")
