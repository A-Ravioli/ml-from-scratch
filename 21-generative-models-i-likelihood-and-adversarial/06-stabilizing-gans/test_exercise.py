"""Checks for exercise.py (stabilizing GANs).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The Wasserstein losses and
weight clipping are checked against their definitions. The gradient penalty is checked against closed
forms (a linear critic has the same gradient everywhere; a quadratic critic's gradient at the
interpolate is known once the interpolation weights are reproduced from the generator), for being
differentiable with respect to the critic's parameters, and for image-shaped inputs. Spectral
normalization is checked step by step against the power-iteration formulas, for convergence to the
largest singular value, for its gradient path, and in the SNLinear layer's train/eval behaviour. The
WGAN-GP loop is checked against an independent implementation with the same random draws, with
counters for n_critic, and for the use of gp_lambda.
"""
import copy
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional
nn = torch.nn


# ---------------------------------------------------------------- stage 1: wasserstein losses


def test_stage1_losses(nb):
    real, fake = torch.tensor([1.0, 2.0, 6.0]), torch.tensor([-1.0, 0.5])
    got = nb.critic_loss(real, fake)
    assert got.dim() == 0 and math.isclose(got.item(), -0.25 - 3.0), (
        f"critic_loss = fake.mean() - real.mean() = -0.25 - 3 = -3.25; got {got.item()}. the critic MINIMIZES it, "
        "pushing real scores up and fake scores down")
    assert math.isclose(nb.wgan_generator_loss(fake).item(), 0.25), "wgan_generator_loss = -fake.mean()"
    s = torch.tensor([3.0], requires_grad=True)
    nb.wgan_generator_loss(s).backward()
    assert s.grad.item() == -1.0, "no logs, no sigmoids: the generator's gradient on a score is a constant -1/B"


def test_stage1_clip_weights(nb):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.where(torch.rand(p.shape, generator=g) < 0.5, -1.0, 1.0))
    out = nb.clip_weights_(m, 0.05)
    assert out is m, "clip_weights_ works in place and returns the module"
    for p in m.parameters():
        assert p.abs().max() <= 0.05 + 1e-7 and (p.abs() == 0.05).any(), "every parameter clamped to [-c, c]"
    small = nn.Linear(2, 2)
    with torch.no_grad():
        small.weight.fill_(0.001)
    with torch.no_grad():
        small.bias.fill_(0.5)
    nb.clip_weights_(small)
    assert torch.all(small.weight == 0.001), "values already inside [-c, c] are unchanged"
    assert torch.allclose(small.bias, torch.full((2,), 0.01)), "c defaults to 0.01 (the WGAN paper's value)"


# ---------------------------------------------------------------- stage 2: gradient penalty


def test_stage2_linear_critic(nb):
    torch.manual_seed(1)
    critic = nn.Linear(3, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[2.0, -1.0, 2.0]]))  # ||w|| = 3 everywhere
    real, fake = torch.randn(16, 3), torch.randn(16, 3)
    gp = nb.gradient_penalty(critic, real, fake, generator=torch.Generator().manual_seed(0))
    assert gp.dim() == 0 and math.isclose(gp.item(), 4.0, rel_tol=1e-5), (
        f"a linear critic with ||w|| = 3 has gradient norm 3 at every point: penalty (3 - 1)^2 = 4; got {gp.item()}")
    gp.backward()
    want = 2 * (3 - 1) * critic.weight.detach() / 3
    assert critic.weight.grad is not None and torch.allclose(critic.weight.grad, want, atol=1e-5), (
        "the penalty must be differentiable with respect to the critic's parameters: use "
        "torch.autograd.grad(..., create_graph=True); d/dw (||w|| - 1)^2 = 2 (||w|| - 1) w / ||w||")


def test_stage2_interpolates(nb):
    real = torch.randn(64, 2, generator=torch.Generator().manual_seed(2)) + 3
    fake = torch.randn(64, 2, generator=torch.Generator().manual_seed(3))
    critic = lambda x: 0.5 * (x ** 2).sum(1)  # gradient at x_hat is x_hat itself
    gp = nb.gradient_penalty(critic, real, fake, generator=torch.Generator().manual_seed(4))
    eps = torch.rand((64, 1), generator=torch.Generator().manual_seed(4))
    x_hat = eps * real + (1 - eps) * fake
    want = ((x_hat.norm(dim=1) - 1) ** 2).mean()
    assert math.isclose(gp.item(), want.item(), rel_tol=1e-5), (
        f"expected {want.item():.4f}: interpolate x_hat = eps * real + (1 - eps) * fake with ONE eps per example, "
        "eps = torch.rand((B, 1), generator=generator), and penalize (||grad||_2 - 1)^2 averaged over the batch")


def test_stage2_image_shapes(nb):
    torch.manual_seed(5)
    critic = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 4 * 4, 1))
    real, fake = torch.randn(6, 1, 6, 6), torch.randn(6, 1, 6, 6)
    gp = nb.gradient_penalty(critic, real, fake, generator=torch.Generator().manual_seed(0))
    eps = torch.rand((6, 1, 1, 1), generator=torch.Generator().manual_seed(0))
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    (g,) = torch.autograd.grad(critic(x_hat).sum(), x_hat)
    want = ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()
    assert torch.allclose(gp, want, atol=1e-5), (
        "for images, eps has shape (B, 1, 1, 1) and the norm is taken over all of C, H, W per example")
    assert real.grad is None and fake.grad is None


# ---------------------------------------------------------------- stage 3: spectral normalization


def test_stage3_power_iteration_step(nb):
    g = torch.Generator().manual_seed(6)
    w = torch.randn(4, 3, 2, generator=g)  # a conv-like weight: reshaped to (4, 6)
    u0 = F.normalize(torch.randn(4, generator=g), dim=0)
    w_sn, u1, sigma = nb.spectral_normalize(w, u0.clone())
    W = w.reshape(4, -1)
    v = F.normalize(W.T @ u0, dim=0)
    u = F.normalize(W @ v, dim=0)
    s = u @ W @ v
    assert torch.allclose(u1, u, atol=1e-6), "one iteration: v = normalize(W^T u), then u = normalize(W v)"
    assert torch.allclose(sigma, s, atol=1e-5), "sigma = u . (W v) with the updated u and v"
    assert w_sn.shape == w.shape and torch.allclose(w_sn, w / s, atol=1e-6), "return weight / sigma, original shape"
    _, u2, s2 = nb.spectral_normalize(w, u0.clone(), n_power_iterations=2)
    v2 = F.normalize(W.T @ u, dim=0)
    assert torch.allclose(u2, F.normalize(W @ v2, dim=0), atol=1e-6), "n_power_iterations repeats the pair of updates"


def test_stage3_converges_to_top_singular_value(nb):
    g = torch.Generator().manual_seed(7)
    w = torch.randn(8, 5, generator=g)
    top = torch.linalg.svdvals(w)[0]
    u = F.normalize(torch.randn(8, generator=g), dim=0)
    for _ in range(50):
        _, u, sigma = nb.spectral_normalize(w, u)
    assert torch.allclose(sigma, top, rtol=1e-4), f"power iteration converges to the top singular value {top:.4f}"
    w_sn, _, _ = nb.spectral_normalize(w, u)
    assert math.isclose(torch.linalg.svdvals(w_sn)[0].item(), 1.0, rel_tol=1e-3), "W / sigma has spectral norm 1"


def test_stage3_gradient_path(nb):
    g = torch.Generator().manual_seed(8)
    w = torch.randn(3, 4, generator=g, requires_grad=True)
    u0 = F.normalize(torch.randn(3, generator=g), dim=0)
    _, u1, sigma = nb.spectral_normalize(w, u0)
    assert not u1.requires_grad, "the power-iteration vectors carry no gradient (computed under no_grad)"
    sigma.backward()
    W = w.detach()
    v = F.normalize(W.T @ u0, dim=0)
    u = F.normalize(W @ v, dim=0)
    assert torch.allclose(w.grad, torch.outer(u, v), atol=1e-5), (
        "d sigma / dW = u v^T: gradients flow through W in sigma = u . (W v), with u and v held fixed")


def test_stage3_snlinear(nb):
    torch.manual_seed(9)
    lin = nb.SNLinear(5, 7)
    assert "u" in dict(lin.named_buffers()) and lin.u.shape == (7,), "a buffer u of shape (out_features,)"
    assert math.isclose(lin.u.norm().item(), 1.0, rel_tol=1e-5), "u starts as a unit vector"
    x = torch.randn(3, 5)
    u_before = lin.u.clone()
    y = lin(x)
    w_sn, u_new, _ = nb.spectral_normalize(lin.weight, u_before)
    assert torch.allclose(y, F.linear(x, w_sn, lin.bias), atol=1e-5), "forward uses the normalized weight"
    assert torch.allclose(lin.u, u_new, atol=1e-6), "in training mode, forward stores the updated u"
    lin.eval()
    u_eval = lin.u.clone()
    lin(x)
    assert torch.equal(lin.u, u_eval), "in eval mode, u is not updated"
    lin.train()
    for _ in range(100):
        lin(x)
    assert math.isclose(torch.linalg.svdvals(nb.spectral_normalize(lin.weight, lin.u)[0].detach())[0].item(), 1.0,
                        rel_tol=1e-2), "after repeated forward passes, the effective weight has spectral norm 1"


# ---------------------------------------------------------------- stage 4: the WGAN-GP loop


def _nets(seed):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    C = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    return G, C


def _reference_wgan(G, C, batches, n_steps, latent_dim, n_critic, gp_lambda, lr, betas, gen):
    oc = torch.optim.Adam(C.parameters(), lr=lr, betas=betas)
    og = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    it = iter(batches)
    hist = {"critic_loss": [], "gp": []}
    for _ in range(n_steps):
        for _ in range(n_critic):
            real = next(it)
            b = real.shape[0]
            z = torch.randn((b, latent_dim), generator=gen)
            fake = G(z).detach()
            eps = torch.rand((b, 1), generator=gen)
            x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
            (gr,) = torch.autograd.grad(C(x_hat).sum(), x_hat, create_graph=True)
            gp = ((gr.norm(dim=1) - 1) ** 2).mean()
            c_loss = C(fake).mean() - C(real).mean()
            loss = c_loss + gp_lambda * gp
            oc.zero_grad()
            loss.backward()
            oc.step()
        z = torch.randn((b, latent_dim), generator=gen)
        og.zero_grad()
        (-C(G(z)).mean()).backward()
        og.step()
        hist["critic_loss"].append(c_loss.item())
        hist["gp"].append(gp.item())
    return hist


def test_stage4_matches_reference(nb):
    batches = [torch.randn(16, 2, generator=torch.Generator().manual_seed(100 + i)) + 2 for i in range(40)]
    for n_critic, gp_lambda, betas in ((3, 10.0, (0.5, 0.9)), (2, 0.5, (0.0, 0.99))):
        G, C = _nets(10)
        G2, C2 = copy.deepcopy(G), copy.deepcopy(C)
        it = iter(batches)
        hist = bounded(nb.train_wgan, G, C, lambda: next(it), n_steps=4, latent_dim=2, n_critic=n_critic,
                       gp_lambda=gp_lambda, lr=1e-3, betas=betas, generator=torch.Generator().manual_seed(0),
                       budget=20_000_000, what="train_wgan")
        ref = _reference_wgan(G2, C2, batches, 4, 2, n_critic, gp_lambda, 1e-3, betas, torch.Generator().manual_seed(0))
        for p, q in zip(list(C.parameters()) + list(G.parameters()), list(C2.parameters()) + list(G2.parameters())):
            assert torch.allclose(p, q, atol=1e-5), (
                f"n_critic={n_critic}, gp_lambda={gp_lambda}: parameters differ from the reference after 4 generator "
                "steps. per critic step: real = sample_real(), z = randn((B, latent_dim), generator), fake = G(z) "
                "detached, the penalty's eps from the same generator, loss = critic_loss + gp_lambda * gp; then the "
                "generator step on a fresh z. Adam(lr, betas) for both")
        assert set(hist) >= {"critic_loss", "gp", "wasserstein"} and len(hist["wasserstein"]) == 4, (
            "history: one entry per GENERATOR step in each list")
        assert np.allclose(hist["critic_loss"], ref["critic_loss"], atol=1e-5), (
            "history['critic_loss']: fake.mean() - real.mean() of the last critic step, WITHOUT the penalty")
        assert np.allclose(hist["gp"], ref["gp"], atol=1e-5), "history['gp']: the raw penalty, not multiplied by gp_lambda"
        assert np.allclose(hist["wasserstein"], -np.array(ref["critic_loss"]), atol=1e-5), (
            "the Wasserstein estimate is -critic_loss (real.mean() - fake.mean())")


def test_stage4_counts_and_lambda(nb):
    calls = {"n": 0}

    def sample_real():
        calls["n"] += 1
        return torch.randn(8, 2) + 1

    G, C = _nets(11)
    nb.train_wgan(G, C, sample_real, n_steps=3, latent_dim=2, n_critic=4, generator=torch.Generator().manual_seed(0))
    assert calls["n"] == 12, f"n_critic = 4 critic steps per generator step, each with a fresh real batch: 12; got {calls['n']}"
    seen = []
    G, C = _nets(12)
    nb.train_wgan(G, C, sample_real, n_steps=3, latent_dim=2, n_critic=1, generator=torch.Generator().manual_seed(0),
                  callback=lambda step, hist: seen.append((step, len(hist["wasserstein"]))))
    assert seen == [(1, 1), (2, 2), (3, 3)], (
        f"callback(step, history) runs after every generator step, with step counting from 1; got {seen}")


def test_stage3_snlinear_iterations(nb):
    torch.manual_seed(25)
    lin = nb.SNLinear(4, 6, n_power_iterations=3)
    u0 = lin.u.clone()
    lin(torch.randn(2, 4))
    _, want, _ = nb.spectral_normalize(lin.weight, u0, 3)
    assert torch.allclose(lin.u, want, atol=1e-6), "SNLinear runs n_power_iterations power iterations per forward pass"
