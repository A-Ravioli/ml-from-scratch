"""Checks for capstone.py (four families, one dataset).

Each stage's checks pin down one piece of the comparison on small synthetic binary images (8 x 8, rows
of identical pixels: 8 bits of information per 64-pixel image, so a good model gets far below the
1 bit/dim of a uniform one). binarization against its definition; each family's model type and
configuration; training runs that must lower each family's loss (under `mlfs.testing.bounded`),
honour the epochs, learning rate and seed; likelihoods against the chapter's own functions (exact
for the PixelCNN, the importance-weighted bound for the VAE, the dequantized bound for the flow);
samples for shape, values and reproducibility; and the evaluation and interval helpers against their
definitions. Data are generated here; nothing is downloaded.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")

TINY = {
    "autoregressive": dict(hidden_channels=16, n_layers=3, epochs=3, batch_size=32, lr=5e-3),
    "vae": dict(latent_dim=8, hidden_dims=(64,), epochs=10, batch_size=32, lr=1e-2, iwae_k=10),
    "gan": dict(latent_dim=8, base_channels=8, epochs=2, batch_size=32, lr=5e-4),
    "flow": dict(n_layers=4, hidden_dim=64, epochs=6, batch_size=32, lr=2e-3, alpha=0.05),
}
FAMILIES = ("autoregressive", "vae", "gan", "flow")


def _rows(n, seed, size=8):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, 1, size, 1), generator=g).float().expand(n, 1, size, size).contiguous()


def _train(nb, family, X, cfg=None, seed=0):
    m = nb.build_family(family, cfg or TINY[family], img_size=X.shape[-1], seed=seed)
    out = bounded(nb.train_family, family, m, X, cfg or TINY[family], seed=seed, budget=60_000_000,
                  what=f"train_family({family!r})", hint="loop over a fixed number of epochs and batches.")
    return m, out


# ---------------------------------------------------------------- stage 1: the data


def test_stage1_binarize(nb):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (400, 6, 6)).astype(np.uint8)
    B = nb.binarize(X, seed=3)
    assert isinstance(B, torch.Tensor) and B.dtype == torch.float32 and B.shape == (400, 1, 6, 6), (
        "uint8 (N, H, W) -> float32 tensor (N, 1, H, W)")
    assert set(B.unique().tolist()) <= {0.0, 1.0}, "binary values only"
    want = torch.bernoulli(torch.tensor(X, dtype=torch.float32)[:, None] / 255,
                           generator=torch.Generator().manual_seed(3))
    assert torch.equal(B, want), (
        "each pixel ~ Bernoulli(X / 255), drawn in one torch.bernoulli call with torch.Generator().manual_seed(seed)")
    assert not torch.equal(nb.binarize(X, seed=4), B), "a different seed gives a different binarization"
    assert torch.all(nb.binarize(np.zeros((2, 3, 3), np.uint8)) == 0) and torch.all(
        nb.binarize(np.full((2, 3, 3), 255, np.uint8)) == 1), "black stays 0 and white stays 1"


# ---------------------------------------------------------------- stage 2: models and training


def test_stage2_build(nb):
    from mlfs import generative as G

    m = nb.build_family("autoregressive", TINY["autoregressive"], img_size=8)
    assert isinstance(m, G.PixelCNN) and m.n_levels == 2, "the autoregressive family is a PixelCNN with 2 levels"
    v = nb.build_family("vae", TINY["vae"], img_size=8)
    assert isinstance(v, G.VAE) and v.likelihood == "bernoulli" and v.in_dim == 64 and v.latent_dim == 8, (
        "the latent-variable family is a Bernoulli VAE on flattened images (in_dim = img_size**2)")
    g = nb.build_family("gan", TINY["gan"], img_size=8)
    assert isinstance(g["G"], G.DCGANGenerator) and isinstance(g["D"], G.DCGANDiscriminator), (
        "the adversarial family is an nn.ModuleDict with a DCGANGenerator 'G' and a DCGANDiscriminator 'D'")
    assert g["G"](torch.randn(2, 8)).shape == (2, 1, 8, 8)
    f = nb.build_family("flow", TINY["flow"], img_size=8)
    assert isinstance(f, G.RealNVP) and f.dim == 64 and len(f.layers) == 4, "the flow is RealNVP(img_size**2, n_layers)"
    a, b = nb.build_family("vae", TINY["vae"], img_size=8, seed=1), nb.build_family("vae", TINY["vae"], img_size=8, seed=1)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())), "the seed fixes the initialization"
    with pytest.raises(ValueError):
        nb.build_family("diffusion", {}, img_size=8)


def test_stage2_likelihood_families_train(nb):
    X = _rows(192, 0)
    for family in ("autoregressive", "vae", "flow"):
        m, out = _train(nb, family, X)
        assert set(out) >= {"seconds", "losses"} and out["seconds"] > 0, "return dict(seconds, losses)"
        assert len(out["losses"]) == TINY[family]["epochs"], "one mean loss per epoch"
        assert out["losses"][-1] < 0.8 * out["losses"][0], (
            f"{family}: the per-epoch loss should fall clearly on rows-of-identical-pixels data; got {out['losses']}")
        assert not m.training, "return the model in eval mode"


def test_stage2_gan_trains_and_config(nb):
    X = _rows(128, 1)
    m = nb.build_family("gan", TINY["gan"], img_size=8)
    before = [p.detach().clone() for p in m["G"].parameters()]
    out = nb.train_family("gan", m, X, TINY["gan"], seed=0)
    assert len(out["losses"]) == 2 and any(not torch.equal(a, b) for a, b in zip(before, m["G"].parameters())), (
        "train the GAN with gan_step (images scaled to [-1, 1]); the generator must change")
    frozen = nb.build_family("vae", dict(TINY["vae"], lr=0.0, epochs=1), img_size=8)
    w = [p.detach().clone() for p in frozen.parameters()]
    nb.train_family("vae", frozen, X, dict(TINY["vae"], lr=0.0, epochs=1))
    assert all(torch.equal(a, b) for a, b in zip(w, frozen.parameters())), "use config['lr'] (lr = 0: no change)"
    l1 = nb.train_family("vae", nb.build_family("vae", TINY["vae"], img_size=8), X, dict(TINY["vae"], epochs=2), seed=5)
    l2 = nb.train_family("vae", nb.build_family("vae", TINY["vae"], img_size=8), X, dict(TINY["vae"], epochs=2), seed=5)
    assert l1["losses"] == l2["losses"], "shuffle with a torch.Generator seeded by `seed`: same seed, same run"
    l3 = nb.train_family("vae", nb.build_family("vae", TINY["vae"], img_size=8), X, dict(TINY["vae"], epochs=2), seed=6)
    assert l3["losses"] != l1["losses"], "a different `seed` must give a different shuffle (don't use the global RNG)"
    seen = []
    m2 = nb.build_family("gan", TINY["gan"], img_size=8)
    m2["D"].register_forward_pre_hook(lambda mod, inp: seen.append(float(inp[0].detach().min())))
    nb.train_family("gan", m2, X, dict(TINY["gan"], epochs=1))
    assert min(seen) < -0.99, (
        "the DCGAN generator ends in tanh, so the discriminator must see real images scaled to [-1, 1] (x * 2 - 1)")


# ---------------------------------------------------------------- stage 3: likelihoods


def test_stage3_matches_chapter_functions(nb):
    from mlfs import generative as G

    X = _rows(60, 2)
    d = 64
    pc = nb.build_family("autoregressive", TINY["autoregressive"], img_size=8)
    got = nb.nll_bits_per_dim("autoregressive", pc, X, TINY["autoregressive"])
    assert isinstance(got, np.ndarray) and got.shape == (60,) and got.dtype == np.float64, "float64 ndarray (N,)"
    with torch.no_grad():
        want = (G.pixelcnn_nll(pc, X) / (d * math.log(2))).double().numpy()
    assert np.allclose(got, want, atol=1e-5), "autoregressive: the exact nll pixelcnn_nll, converted to bits/dim"
    v = nb.build_family("vae", TINY["vae"], img_size=8)
    got = nb.nll_bits_per_dim("vae", v, X, TINY["vae"], generator=torch.Generator().manual_seed(0))
    want = (G.iwae_nll(v, X.flatten(1), k=10, generator=torch.Generator().manual_seed(0)) / (d * math.log(2))).numpy()
    assert np.allclose(got, want, atol=1e-5), "vae: iwae_nll with k = config['iwae_k'] and the given generator"
    f = nb.build_family("flow", TINY["flow"], img_size=8)
    got = nb.nll_bits_per_dim("flow", f, X, TINY["flow"], generator=torch.Generator().manual_seed(0))
    want = G.flow_bits_per_dim(f, X.long().flatten(1), n_levels=2, alpha=0.05, generator=torch.Generator().manual_seed(0)).numpy()
    assert np.allclose(got, want, atol=1e-4), "flow: flow_bits_per_dim with n_levels = 2 and config['alpha']"
    assert nb.nll_bits_per_dim("gan", nb.build_family("gan", TINY["gan"], img_size=8), X, TINY["gan"]) is None, (
        "a GAN has no likelihood: return None")


def test_stage3_trained_models_beat_uniform(nb):
    X, Xt = _rows(192, 3), _rows(100, 4)
    for family in ("autoregressive", "vae", "flow"):
        m, _ = _train(nb, family, X)
        bpd = nb.nll_bits_per_dim(family, m, Xt, TINY[family], generator=torch.Generator().manual_seed(0))
        assert bpd.mean() < 0.85, (
            f"{family}: a uniform model of binary pixels costs exactly 1 bit/dim; after training on rows of identical "
            f"pixels the test bits/dim should be clearly lower, got {bpd.mean():.3f}")


# ---------------------------------------------------------------- stage 4: samples


def test_stage4_samples(nb):
    for family in FAMILIES:
        m = nb.build_family(family, TINY[family], img_size=8)
        if family == "gan":  # make the generator's output depend strongly on z
            with torch.no_grad():
                for p in m["G"].parameters():
                    p.mul_(20)
        s = nb.sample_binary(family, m, 12, TINY[family], img_size=8, generator=torch.Generator().manual_seed(0))
        assert isinstance(s, torch.Tensor) and s.dtype == torch.float32 and s.shape == (12, 1, 8, 8), (
            f"{family}: a float32 tensor (n, 1, img_size, img_size); got {getattr(s, 'shape', None)}")
        assert s.device.type == "cpu" and set(s.unique().tolist()) <= {0.0, 1.0}, f"{family}: binary values, on the CPU"
        s2 = nb.sample_binary(family, m, 12, TINY[family], img_size=8, generator=torch.Generator().manual_seed(0))
        assert torch.equal(s, s2), f"{family}: draw all randomness from `generator`"


def test_stage4_samples_follow_the_model(nb):
    from mlfs import generative as G

    v = nb.build_family("vae", TINY["vae"], img_size=8)
    with torch.no_grad():
        for p in v.decoder.parameters() if hasattr(v, "decoder") else []:
            p.zero_()
    s = nb.sample_binary("vae", v, 4000, TINY["vae"], img_size=8, generator=torch.Generator().manual_seed(1))
    probs = v.sample(1, generator=torch.Generator().manual_seed(0)).mean().item()
    assert abs(s.mean().item() - probs) < 0.03, (
        f"vae samples are Bernoulli draws from the decoded probabilities (mean {probs:.3f}); got mean {s.mean():.3f}")
    f = nb.build_family("flow", TINY["flow"], img_size=8)  # an untrained flow is the identity map in logit space
    s = nb.sample_binary("flow", f, 2000, TINY["flow"], img_size=8, generator=torch.Generator().manual_seed(2))
    z = torch.randn((2000, 64), generator=torch.Generator().manual_seed(2))
    want = (((torch.sigmoid(f.inverse(z).detach()) - 0.05) / 0.9) >= 0.5).float().view(2000, 1, 8, 8)
    assert torch.equal(s, want), (
        "flow samples: inverse logit transform u = (sigmoid(y) - alpha) / (1 - 2 alpha), then pixel = 1 where u >= 1/2")


# ---------------------------------------------------------------- stage 5: evaluation


class _StubNet(torch.nn.Module):
    """a stand-in feature network: features are the first 16 pixels, labels are 3 bins of the ink density."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m = x.flatten(1).mean(1, keepdim=True)
        return -((m - torch.tensor([[0.2, 0.5, 0.8]])) ** 2) * 100 + 0 * self.dummy

    def embed(self, x, batch_size=500):
        return x.flatten(1)[:, :16].double().numpy()


def test_stage5_evaluate_samples(nb):
    from mlfs import generative as G

    net = _StubNet()
    real = (torch.rand(300, 1, 8, 8, generator=torch.Generator().manual_seed(1)) > 0.5).float()
    thr = torch.linspace(0.1, 0.9, 200)[:, None, None, None]
    fake = (torch.rand(200, 1, 8, 8, generator=torch.Generator().manual_seed(2)) > thr).float()
    real_feats = net.embed(real)
    out = nb.evaluate_samples(fake, real_feats, net, k=3)
    ff = net.embed(fake)
    assert math.isclose(out["fid"], G.fid_from_features(real_feats, ff), rel_tol=1e-6), (
        "fid: fid_from_features(real_feats, featnet.embed(samples))")
    p, r = G.precision_recall_knn(real_feats, ff, k=3)
    assert math.isclose(out["precision"], p) and math.isclose(out["recall"], r), (
        "precision_recall_knn(real_feats, fake_feats, k=k), in that order")
    with torch.no_grad():
        lab = net(fake).argmax(1).numpy()
    h = np.bincount(lab) / len(lab)
    h = h[h > 0]
    assert len(h) == 3, "sanity: the stub network predicts three classes"
    assert math.isclose(out["class_entropy"], float(-(h * np.log(h)).sum()), rel_tol=1e-6), (
        "class_entropy: the entropy in nats of the histogram of the feature network's predicted labels (argmax)")


def test_stage5_bpd_ci(nb):
    from mlfs.stats import bootstrap_ci

    x = np.random.default_rng(3).normal(0.2, 0.05, size=500)
    mean, lo, hi = nb.bpd_ci(x, n_boot=1000, seed=4)
    want = bootstrap_ci(x, np.mean, n_boot=1000, alpha=0.05, rng=4)
    assert math.isclose(mean, x.mean()) and math.isclose(lo, want[0]) and math.isclose(hi, want[1]), (
        "bpd_ci: (mean, lo, hi) with mlfs.stats.bootstrap_ci(bpd, np.mean, n_boot, alpha, rng=seed)")
    assert lo < mean < hi
    m2, lo2, hi2 = nb.bpd_ci(x, n_boot=1000, alpha=0.5, seed=4)
    assert hi2 - lo2 < hi - lo, "alpha sets the interval's level"
