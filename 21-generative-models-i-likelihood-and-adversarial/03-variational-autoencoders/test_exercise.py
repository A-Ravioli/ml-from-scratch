"""Checks for exercise.py (variational autoencoders).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The reparameterization is
checked for its moments, its gradients and its use of the generator; the KL against
`torch.distributions`. The VAE's loss is recomputed from its own forward pass (same random draws), so
beta, the reconstruction term and the KL term are each checked. The importance-weighted bound is
checked against log p(x) computed exactly by quadrature over a 2-D latent space: ELBO >= IWAE_k >=
the truth, approaching it as k grows. ConvVAE shapes and loss are checked the same way, and the
collapse diagnostics against stub encoders with known outputs. Training runs under
`mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional
D = torch.distributions


def _exact_log_px(vae, x, lim=6.0, n=241):
    """log p(x) = log int p(x | z) N(z; 0, I) dz by the trapezoid-free midpoint rule on a 2-D grid."""
    g = torch.linspace(-lim, lim, n, dtype=torch.float64)
    h = (g[1] - g[0]).item()
    zz = torch.cartesian_prod(g, g)  # (n^2, 2)
    log_pz = D.Normal(0.0, 1.0).log_prob(zz).sum(1)
    with torch.no_grad():
        out = vae.decode(zz.float()).double()
    res = []
    for xi in x.double():
        if vae.likelihood == "gaussian":
            log_px_z = D.Normal(out, 1.0).log_prob(xi).sum(1)
        else:
            log_px_z = -F.binary_cross_entropy_with_logits(out, xi.expand_as(out), reduction="none").sum(1)
        res.append(torch.logsumexp(log_px_z + log_pz, 0) + 2 * math.log(h))
    return torch.stack(res)


# ---------------------------------------------------------------- stage 1: reparameterization and KL


def test_stage1_reparameterize(nb):
    mu = torch.tensor([[1.0, -2.0, 0.5]]).expand(20_000, 3)
    logvar = torch.log(torch.tensor([[0.25, 4.0, 1.0]])).expand(20_000, 3)
    z = nb.reparameterize(mu, logvar, generator=torch.Generator().manual_seed(0))
    assert z.shape == mu.shape
    assert torch.allclose(z.mean(0), mu[0], atol=0.05), f"E[z] = mu; got {z.mean(0)}"
    assert torch.allclose(z.var(0), torch.exp(logvar[0]), rtol=0.05), (
        f"Var[z] = exp(logvar) (logvar is the log VARIANCE: z = mu + exp(logvar / 2) eps); got {z.var(0)}")
    z2 = nb.reparameterize(mu, logvar, generator=torch.Generator().manual_seed(0))
    assert torch.equal(z, z2), "draw eps from `generator`: a seeded generator must reproduce z"


def test_stage1_reparameterize_gradients(nb):
    mu = torch.zeros(4, 2, requires_grad=True)
    logvar = torch.full((4, 2), math.log(4.0), requires_grad=True)
    z = nb.reparameterize(mu, logvar, generator=torch.Generator().manual_seed(1))
    eps = torch.randn((4, 2), generator=torch.Generator().manual_seed(1))
    assert torch.allclose(z, 2.0 * eps), "z = mu + exp(logvar / 2) * eps with eps = torch.randn(mu.shape, generator=...)"
    z.sum().backward()
    assert torch.allclose(mu.grad, torch.ones(4, 2)), "dz/dmu = 1: the noise is outside the computation graph"
    assert torch.allclose(logvar.grad, 0.5 * 2.0 * eps), "dz/dlogvar = 0.5 exp(logvar / 2) eps"


def test_stage1_kl(nb):
    g = torch.Generator().manual_seed(2)
    mu, logvar = torch.randn(5, 3, generator=g), torch.randn(5, 3, generator=g)
    want = D.kl_divergence(D.Normal(mu, torch.exp(0.5 * logvar)), D.Normal(0.0, 1.0)).sum(1)
    got = nb.kl_standard_normal(mu, logvar)
    assert got.shape == (5,), f"one KL per example, shape (B,); got {tuple(got.shape)}"
    assert torch.allclose(got, want, atol=1e-5), "KL = 0.5 sum(mu^2 + exp(logvar) - 1 - logvar)"
    assert torch.allclose(nb.kl_standard_normal(torch.zeros(2, 4), torch.zeros(2, 4)), torch.zeros(2)), (
        "the KL of N(0, I) to itself is 0")
    mu4, lv4 = torch.randn(2, 3, 4, 4, generator=g), torch.randn(2, 3, 4, 4, generator=g)
    got4 = nb.kl_standard_normal(mu4, lv4)
    want4 = D.kl_divergence(D.Normal(mu4, torch.exp(0.5 * lv4)), D.Normal(0.0, 1.0)).sum(dim=(1, 2, 3))
    assert got4.shape == (2,) and torch.allclose(got4, want4, atol=1e-4), (
        "spatial latents (B, C, h, w): sum over every non-batch dimension")


# ---------------------------------------------------------------- stage 2: the VAE


def test_stage2_reconstruction_nll(nb):
    g = torch.Generator().manual_seed(3)
    logits, x = torch.randn(4, 6, generator=g), torch.rand(4, 6, generator=g)
    want = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)
    assert torch.allclose(nb.reconstruction_nll(logits, x, "bernoulli"), want, atol=1e-5), (
        "bernoulli: binary cross-entropy with logits, summed over dimensions (targets may be in [0, 1])")
    mean = torch.randn(4, 2, 3, generator=g)
    xg = torch.randn(4, 2, 3, generator=g)
    want_g = -D.Normal(mean, 1.0).log_prob(xg).sum(dim=(1, 2))
    assert torch.allclose(nb.reconstruction_nll(mean, xg, "gaussian"), want_g, atol=1e-5), (
        "gaussian: -log N(x; mean, 1) summed over non-batch dims, INCLUDING the 0.5 log(2 pi) per dimension")
    with pytest.raises(ValueError):
        nb.reconstruction_nll(mean, xg, "laplace")


def test_stage2_shapes(nb):
    torch.manual_seed(4)
    vae = nb.VAE(12, 3, hidden_dims=(16,))
    x = torch.rand(5, 12)
    mu, logvar = vae.encode(x)
    assert mu.shape == (5, 3) and logvar.shape == (5, 3), "encode -> (mu, logvar), each (B, latent_dim)"
    assert vae.decode(mu).shape == (5, 12), "decode -> (B, in_dim)"
    x_out, mu2, lv2 = vae(x)
    assert x_out.shape == (5, 12) and torch.allclose(mu, mu2) and torch.allclose(logvar, lv2)
    torch.manual_seed(21)
    x_out = vae(x)[0]
    torch.manual_seed(21)
    z = nb.reparameterize(*vae.encode(x))
    assert torch.allclose(x_out, vae.decode(z), atol=1e-6) and not torch.allclose(x_out, vae.decode(mu), atol=1e-3), (
        "forward decodes a SAMPLE z = reparameterize(mu, logvar) (global RNG), not the mean mu")
    s = vae.sample(7, generator=torch.Generator().manual_seed(0))
    assert s.shape == (7, 12) and torch.all((s > 0) & (s < 1)), "bernoulli samples are decoded PROBABILITIES"
    s2 = vae.sample(7, generator=torch.Generator().manual_seed(0))
    assert torch.equal(s, s2), "sample draws z from `generator`"
    gz = nb.VAE(12, 3, hidden_dims=(16,), likelihood="gaussian")
    assert gz.sample(3).shape == (3, 12)
    with pytest.raises(ValueError):
        nb.VAE(12, 3, likelihood="poisson")


def test_stage2_loss_terms(nb):
    torch.manual_seed(5)
    for lik in ("bernoulli", "gaussian"):
        vae = nb.VAE(10, 2, hidden_dims=(8, 8), likelihood=lik)
        x = torch.rand(6, 10)
        torch.manual_seed(11)
        loss, recon, kl = vae.loss(x, beta=2.5)
        torch.manual_seed(11)
        x_out, mu, logvar = vae(x)
        want_recon = nb.reconstruction_nll(x_out, x, lik).mean()
        want_kl = nb.kl_standard_normal(mu, logvar).mean()
        assert torch.allclose(recon, want_recon, atol=1e-5), (
            f"{lik}: recon must be the batch mean of the per-example reconstruction nll from ONE forward pass")
        assert torch.allclose(kl, want_kl, atol=1e-5), "kl is the batch mean of the per-example KL"
        assert torch.allclose(loss, recon + 2.5 * kl, atol=1e-5), "loss = recon + beta * kl: beta multiplies the KL"
        loss1, recon1, kl1 = vae.loss(x)
        assert torch.allclose(loss1, recon1 + kl1, atol=1e-5), "beta defaults to 1"


def test_stage2_trains(nb):
    torch.manual_seed(6)
    g = torch.Generator().manual_seed(0)
    protos = (torch.rand(4, 20, generator=g) > 0.5).float()
    x = protos[torch.randint(0, 4, (256,), generator=g)]
    vae = nb.VAE(20, 2, hidden_dims=(32,))
    opt = torch.optim.Adam(vae.parameters(), lr=1e-2)

    def run():
        out = []
        for _ in range(150):
            opt.zero_grad()
            loss, _, _ = vae.loss(x)
            loss.backward()
            opt.step()
            out.append(loss.item())
        return out

    losses = bounded(run, budget=20_000_000, what="training a VAE")
    assert np.mean(losses[-10:]) < 0.4 * np.mean(losses[:10]), (
        f"4 binary prototypes: the negative ELBO should fall well below its start ({np.mean(losses[:10]):.1f} -> "
        f"{np.mean(losses[-10:]):.1f})")


# ---------------------------------------------------------------- stage 3: importance weighting


def _avg_iwae(nb, vae, x, k, reps, g):
    """mean of `reps` independent estimates per example, computed in one call on a repeated batch."""
    est = nb.iwae_nll(vae, x.repeat(reps, *([1] * (x.dim() - 1))), k=k, generator=g)
    return est.view(reps, len(x)).mean(0)


def test_stage3_iwae_against_exact(nb):
    torch.manual_seed(7)
    for lik in ("gaussian", "bernoulli"):
        vae = nb.VAE(5, 2, hidden_dims=(16,), likelihood=lik)
        x = torch.randn(4, 5) if lik == "gaussian" else (torch.rand(4, 5) > 0.5).float()
        exact = -_exact_log_px(vae, x).float()  # the true nll, by quadrature
        g = torch.Generator().manual_seed(1)
        assert nb.iwae_nll(vae, x, k=3, generator=g).shape == (4,), "one estimate per example, shape (B,)"
        elbo = _avg_iwae(nb, vae, x, 1, 4000, g)
        iw10 = _avg_iwae(nb, vae, x, 10, 1000, g)
        iw1000 = _avg_iwae(nb, vae, x, 1000, 20, g)
        assert torch.all(elbo >= exact - 0.03), f"{lik}: the ELBO (k = 1) bounds the true nll from above"
        assert torch.all(elbo + 0.03 >= iw10) and torch.all(iw10 + 0.03 >= iw1000), (
            f"{lik}: more importance samples give a tighter (smaller) bound: k=1 {elbo.numpy().round(3)}, "
            f"k=10 {iw10.numpy().round(3)}, k=1000 {iw1000.numpy().round(3)}")
        assert torch.allclose(iw1000, exact, atol=0.05), (
            f"{lik}: with k = 1000 the estimate should match the exact nll {exact.numpy().round(3)} (quadrature); got "
            f"{iw1000.numpy().round(3)}. log w = log p(x|z) + log p(z) - log q(z|x), all terms with constants, "
            "then -(logsumexp(log w) - log k)")
        assert (elbo - exact).max() > 0.02, "sanity: with random weights the ELBO is visibly loose"


def test_stage3_iwae_k1_is_elbo(nb):
    torch.manual_seed(8)
    vae = nb.VAE(8, 3, hidden_dims=(16,))
    x = (torch.rand(3, 8) > 0.5).float()
    g = torch.Generator().manual_seed(2)
    est = torch.stack([nb.iwae_nll(vae, x, k=1, generator=g) for _ in range(3000)]).mean(0)
    mu, logvar = vae.encode(x)
    eps = torch.randn((4000, *mu.shape), generator=torch.Generator().manual_seed(3))
    z = mu + torch.exp(0.5 * logvar) * eps
    recon = F.binary_cross_entropy_with_logits(vae.decode(z.reshape(-1, 3)),
                                               x.repeat(4000, 1), reduction="none").sum(1).view(4000, 3).mean(0)
    neg_elbo = (recon + nb.kl_standard_normal(mu, logvar)).detach()
    assert torch.allclose(est, neg_elbo, atol=0.15), (
        f"with k = 1, the importance-weighted estimate averages to the negative ELBO {neg_elbo.numpy().round(2)}; got "
        f"{est.numpy().round(2)}")
    a = nb.iwae_nll(vae, x, k=5, generator=torch.Generator().manual_seed(4))
    b = nb.iwae_nll(vae, x, k=5, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, b) and not a.requires_grad, "seeded and gradient-free"


# ---------------------------------------------------------------- stage 4: ConvVAE


def test_stage4_shapes(nb):
    torch.manual_seed(9)
    m = nb.ConvVAE(3, latent_channels=4, base_channels=8)
    x = torch.randn(2, 3, 16, 12)
    mu, logvar = m.encode(x)
    assert mu.shape == (2, 4, 4, 3) and logvar.shape == (2, 4, 4, 3), (
        f"spatial latents (B, latent_channels, H/4, W/4); got {tuple(mu.shape)}")
    assert m.decode(mu).shape == (2, 3, 16, 12), "decode upsamples by 4 back to the image size"
    assert m(x)[0].shape == x.shape
    assert m.sample(5, 8).shape == (5, 3, 8, 8) and m.sample(2, (12, 16)).shape == (2, 3, 12, 16)
    with pytest.raises(ValueError):
        m.encode(torch.randn(1, 3, 10, 12))
    b = nb.ConvVAE(1, latent_channels=2, base_channels=8, likelihood="bernoulli")
    s = b.sample(3, 8, generator=torch.Generator().manual_seed(0))
    assert s.shape == (3, 1, 8, 8) and torch.all((s > 0) & (s < 1)), "bernoulli samples are probabilities"
    assert torch.equal(s, b.sample(3, 8, generator=torch.Generator().manual_seed(0)))


def test_stage4_loss_terms(nb):
    torch.manual_seed(10)
    m = nb.ConvVAE(1, latent_channels=2, base_channels=8)
    x = torch.rand(3, 1, 8, 8)
    torch.manual_seed(12)
    loss, recon, kl = m.loss(x, beta=0.1)
    torch.manual_seed(12)
    x_out, mu, logvar = m(x)
    assert torch.allclose(recon, nb.reconstruction_nll(x_out, x, "gaussian").mean(), atol=1e-4), (
        "the default likelihood is gaussian (unit variance, decode returns means)")
    assert torch.allclose(kl, nb.kl_standard_normal(mu, logvar).mean(), atol=1e-4)
    assert torch.allclose(loss, recon + 0.1 * kl, atol=1e-4), "loss = recon + beta * kl"


# ---------------------------------------------------------------- stage 5: collapse diagnostics


class _StubEncoder:
    def __init__(self, mu, logvar):
        self.mu, self.logvar = mu, logvar

    def encode(self, x):
        return self.mu, self.logvar


def test_stage5_active_units(nb):
    g = torch.Generator().manual_seed(13)
    mu = torch.cat([torch.randn(200, 2, generator=g), 0.01 * torch.randn(200, 3, generator=g),
                    0.3 * torch.randn(200, 1, generator=g)], 1)
    stub = _StubEncoder(mu, torch.zeros(200, 6))
    n, var = nb.active_units(stub, torch.zeros(200, 1))
    want = mu.double().var(0, unbiased=False).numpy()
    assert isinstance(var, np.ndarray) and var.shape == (6,) and np.allclose(var, want, rtol=1e-6), (
        "variances: the population variance (ddof = 0) over the batch of each latent's encoder MEAN")
    assert n == 3, f"dims 0, 1 and 5 vary across inputs (> 0.01), dims 2-4 don't: 3 active units; got {n}"
    assert nb.active_units(stub, torch.zeros(200, 1), threshold=0.5)[0] == 2
    spatial = _StubEncoder(torch.randn(50, 2, 2, 2, generator=g), torch.zeros(50, 2, 2, 2))
    assert nb.active_units(spatial, None)[1].shape == (8,), "spatial latents are flattened"


def test_stage5_kl_per_dim(nb):
    g = torch.Generator().manual_seed(14)
    mu, logvar = torch.randn(100, 4, generator=g), torch.randn(100, 4, generator=g)
    got = nb.kl_per_dim(_StubEncoder(mu, logvar), None)
    want = (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)).mean(0).double().numpy()
    assert isinstance(got, np.ndarray) and np.allclose(got, want, atol=1e-6), (
        "the batch mean of each coordinate's KL to N(0, 1)")
    torch.manual_seed(15)
    vae = nb.VAE(6, 3, hidden_dims=(8,))
    x = torch.rand(10, 6)
    assert math.isclose(nb.kl_per_dim(vae, x).sum(), nb.kl_standard_normal(*vae.encode(x)).mean().item(), rel_tol=1e-5), (
        "the per-dimension KLs add up to the mean KL")


# ---------------------------------------------------------------- added clauses


def test_stage2_gaussian_sample_is_decoded_mean(nb):
    torch.manual_seed(16)
    vae = nb.VAE(6, 3, hidden_dims=(8,), likelihood="gaussian")
    s = vae.sample(5, generator=torch.Generator().manual_seed(3))
    z = torch.randn((5, 3), generator=torch.Generator().manual_seed(3))
    assert torch.allclose(s, vae.decode(z).detach(), atol=1e-6), (
        "gaussian VAE.sample returns the decoded MEANS decode(z), with z = torch.randn((n, latent_dim), generator) "
        "and no sigmoid")


def test_stage3_iwae_chunks(nb):
    torch.manual_seed(17)
    vae = nb.VAE(6, 2, hidden_dims=(8,))
    x = (torch.rand(4, 6) > 0.5).float()
    full = nb.iwae_nll(vae, x, k=10, generator=torch.Generator().manual_seed(0))
    chunked = nb.iwae_nll(vae, x, k=10, generator=torch.Generator().manual_seed(0), chunk_size=3)
    assert torch.allclose(full, chunked, atol=1e-5), (
        "chunk_size only changes how many samples are decoded at once: draw all eps first, then decode in chunks")


def test_stage3_iwae_convvae(nb):
    torch.manual_seed(18)
    m = nb.ConvVAE(1, latent_channels=2, base_channels=4, likelihood="bernoulli")
    x = (torch.rand(3, 1, 8, 8, generator=torch.Generator().manual_seed(1)) > 0.5).float()
    g = torch.Generator().manual_seed(2)
    k1 = nb.iwae_nll(m, x.repeat(400, 1, 1, 1), k=1, generator=g).view(400, 3).mean(0)
    k50 = nb.iwae_nll(m, x.repeat(100, 1, 1, 1), k=50, generator=g).view(100, 3).mean(0)
    with torch.no_grad():
        mu, logvar = m.encode(x)
        eps = torch.randn((400, *mu.shape), generator=torch.Generator().manual_seed(3))
        z = (mu + torch.exp(0.5 * logvar) * eps).reshape(-1, *mu.shape[1:])
        recon = nb.reconstruction_nll(m.decode(z), x.repeat(400, 1, 1, 1), "bernoulli").view(400, 3).mean(0)
        neg_elbo = recon + nb.kl_standard_normal(mu, logvar)
    assert k1.shape == (3,) and torch.allclose(k1, neg_elbo, atol=0.6), (
        f"ConvVAE (spatial latents): with k = 1 the estimate averages to the negative ELBO {neg_elbo.numpy().round(2)}; "
        f"got {k1.numpy().round(2)}. sum the log-densities over EVERY latent dimension (C, h, w), not just one")
    assert torch.all(k50 < k1 + 0.05), "more importance samples give a tighter (smaller) bound"


def test_stage4_bernoulli_loss_and_gaussian_sample(nb):
    torch.manual_seed(19)
    m = nb.ConvVAE(1, latent_channels=2, base_channels=4, likelihood="bernoulli")
    x = (torch.rand(3, 1, 8, 8) > 0.5).float()
    torch.manual_seed(5)
    loss, recon, kl = m.loss(x)
    torch.manual_seed(5)
    x_out, mu, logvar = m(x)
    assert torch.allclose(recon, nb.reconstruction_nll(x_out, x, "bernoulli").mean(), atol=1e-4), (
        "ConvVAE.loss must use the model's own likelihood (here 'bernoulli': decode returns logits)")
    gm = nb.ConvVAE(1, latent_channels=2, base_channels=4)
    s = gm.sample(2, 8, generator=torch.Generator().manual_seed(4))
    z = torch.randn((2, 2, 2, 2), generator=torch.Generator().manual_seed(4))
    assert torch.allclose(s, gm.decode(z).detach(), atol=1e-6), "gaussian ConvVAE.sample returns decode(z), no sigmoid"


def test_stage5_threshold_is_strict(nb):
    mu = torch.tensor([[0.5, 0.0], [-0.5, 0.0]]).repeat(50, 1)  # coordinate 0 has variance exactly 0.25
    n, var = nb.active_units(_StubEncoder(mu, torch.zeros_like(mu)), None, threshold=0.25)
    assert math.isclose(var[0], 0.25) and n == 0, "active means variance > threshold (strictly greater)"
