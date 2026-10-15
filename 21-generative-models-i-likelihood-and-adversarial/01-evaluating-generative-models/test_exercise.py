"""Checks for exercise.py (evaluating generative models).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Likelihood conversions
are checked by hand examples (a uniform density is 8 bits/dim) and by the dequantization bound against
exact bin probabilities of a known density. The Frechet distance is checked against
`scipy.linalg.sqrtm` on random Gaussians, against closed forms (1-D, commuting covariances, rank
deficiency), and for symmetry. FID's small-sample bias and its 1/n extrapolation are checked on
Gaussian features whose true distance is known. Precision/recall is checked against a brute-force
definition and on a hand example. The feature network is checked for shapes, training, determinism
and the cache.
"""
import math
import os
import tempfile

import numpy as np
import pytest
import scipy.linalg

from mlfs.testing import bounded

torch = pytest.importorskip("torch")


def _spd(d, rng, scale=1.0):
    a = rng.normal(size=(d, d))
    return scale * (a @ a.T / d + 0.1 * np.eye(d))


# ---------------------------------------------------------------- stage 1: likelihood units


def test_stage1_bits_per_dim_units(nb):
    d = 784
    got = nb.bits_per_dim(d * math.log(256), d)
    assert math.isclose(float(got), 8.0, rel_tol=1e-9), (
        f"an nll of D ln 256 nats (a uniform distribution over 256 levels) is exactly 8 bits/dim; got {got}. "
        "divide by n_dims AND by ln 2")
    t = torch.tensor([d * math.log(2.0), 2 * d * math.log(2.0)])
    got_t = nb.bits_per_dim(t, d)
    assert isinstance(got_t, torch.Tensor) and torch.allclose(got_t, torch.tensor([1.0, 2.0])), (
        "bits_per_dim must work elementwise on a tensor of per-example nlls (and keep it a tensor)")
    a = np.array([3072 * math.log(2.0) * 3.5])
    assert np.allclose(nb.bits_per_dim(a, 3072), 3.5)


def test_stage1_dequantize(nb):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (2000, 3), generator=g)
    y = nb.dequantize(x, generator=torch.Generator().manual_seed(1))
    assert isinstance(y, torch.Tensor) and y.dtype == torch.float32, "return a float32 tensor"
    assert y.shape == x.shape
    assert torch.all(y >= x / 256.0) and torch.all(y <= (x + 1) / 256.0), (
        "each value must land inside its own bin [x/256, (x+1)/256): y = (x + u) / n_levels with u in [0, 1)")
    frac = (y * 256 - x).flatten()
    assert 0.47 < frac.mean().item() < 0.53 and frac.std().item() > 0.27, (
        f"the offsets u must be uniform on [0, 1) (mean 0.5, std 0.289); got mean {frac.mean():.3f}, "
        f"std {frac.std():.3f}. a constant offset (like 0.5) is not dequantization")
    y2 = nb.dequantize(x, generator=torch.Generator().manual_seed(1))
    assert torch.equal(y, y2), "draw u from `generator` so a seeded generator reproduces the result"
    z = nb.dequantize(torch.tensor([0, 1, 3]), n_levels=4, generator=g)
    assert torch.all((z >= torch.tensor([0, 0.25, 0.75])) & (z < torch.tensor([0.25, 0.5, 1.0]))), (
        "n_levels sets the bin width 1/n_levels")


def test_stage1_uniform_is_8_bits(nb):
    got = nb.discrete_nll(torch.zeros(5), 784)
    assert torch.allclose(got, torch.full((5,), 784 * math.log(256))), (
        "a uniform density on [0, 1)^D has log-density 0; as a model of 8-bit data it costs D ln 256 nats: "
        "discrete_nll = -log_density + D ln(n_levels)")
    assert math.isclose(float(nb.bits_per_dim(nb.discrete_nll(0.0, 784), 784)), 8.0, rel_tol=1e-9)
    assert math.isclose(float(nb.discrete_nll(-10.0, 4, n_levels=2)), 10 + 4 * math.log(2)), "n_levels=2: + D ln 2"


def test_stage1_dequantization_bound(nb):
    # density p(y) = 2y on [0, 1), data with L = 4 levels: P(level k) = ((k+1)^2 - k^2) / 16 exactly, and the
    # dequantized bound E_u[-log p((k+u)/L)] + log L = -log 2 + 2 log L - int_k^{k+1} log t dt
    L = 4
    k = torch.arange(L).repeat_interleave(50_000)
    y = nb.dequantize(k, n_levels=L, generator=torch.Generator().manual_seed(2)).double()
    nll_bound = nb.discrete_nll(torch.log(2 * y), 1, n_levels=L)

    def int_log(a, b):  # integral of log t from a to b (t log t -> 0 at 0)
        f = lambda t: t * math.log(t) - t if t > 0 else 0.0
        return f(b) - f(a)

    for lev in range(L):
        exact = -math.log(((lev + 1) ** 2 - lev ** 2) / 16)
        bound = -math.log(2) + 2 * math.log(L) - int_log(lev, lev + 1)
        est = nll_bound[k == lev].mean().item()
        assert abs(est - bound) < 0.02, (
            f"level {lev}: the average dequantized nll should be {bound:.4f}, got {est:.4f}. check the + D ln(n_levels) "
            "term and that dequantize returns (x + u) / n_levels")
        assert bound >= exact, "sanity: Jensen's inequality makes the dequantized nll an upper bound"


# ---------------------------------------------------------------- stage 2: frechet distance


def test_stage2_sqrtm_psd(nb):
    rng = np.random.default_rng(0)
    A = _spd(7, rng)
    S = nb.sqrtm_psd(A)
    assert np.allclose(S, S.T, atol=1e-10), "the PSD square root is symmetric"
    assert np.allclose(S @ S, A, atol=1e-8), "S @ S must equal A"
    assert np.allclose(S, scipy.linalg.sqrtm(A).real, atol=1e-8), "should match scipy.linalg.sqrtm on an SPD matrix"
    B = np.diag([4.0, 1.0, 0.0])
    B[2, 2] = -1e-10  # a round-off negative eigenvalue
    assert np.all(np.isfinite(nb.sqrtm_psd(B))) and np.allclose(nb.sqrtm_psd(B), np.diag([2.0, 1.0, 0.0]), atol=1e-7), (
        "clip tiny negative eigenvalues to 0 before the square root (no nan)")


def test_stage2_matches_scipy(nb):
    rng = np.random.default_rng(1)
    for d in (1, 3, 10):
        mu1, mu2 = rng.normal(size=d), rng.normal(size=d)
        c1, c2 = _spd(d, rng), _spd(d, rng, 2.0)
        want = float(np.sum((mu1 - mu2) ** 2) + np.trace(c1 + c2 - 2 * scipy.linalg.sqrtm(c1 @ c2).real))
        got = nb.frechet_distance(mu1, c1, mu2, c2)
        assert isinstance(got, float), f"return a python float, got {type(got)}"
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-8), (
            f"d={d}: expected {want:.6f} (scipy.linalg.sqrtm reference), got {got:.6f}. note sqrt(C1 C2) is NOT "
            "sqrt(C1) sqrt(C2) unless they commute: use Tr sqrt(C1^1/2 C2 C1^1/2)")
        assert math.isclose(nb.frechet_distance(mu2, c2, mu1, c1), got, rel_tol=1e-8, abs_tol=1e-10), "it is symmetric"


def test_stage2_closed_forms(nb):
    got = nb.frechet_distance(np.array([1.0]), np.array([[4.0]]), np.array([-1.0]), np.array([[1.0]]))
    assert math.isclose(got, 4 + (2 - 1) ** 2, rel_tol=1e-9), (
        f"1-D: (m1 - m2)^2 + (s1 - s2)^2 = 4 + 1 = 5; got {got}")
    a, b = np.array([1.0, 4.0, 0.0]), np.array([9.0, 0.0, 0.0])  # commuting and rank deficient
    got = nb.frechet_distance(np.zeros(3), np.diag(a), np.zeros(3), np.diag(b))
    want = float(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2))
    assert math.isclose(got, want, rel_tol=1e-8, abs_tol=1e-9), f"diagonal covariances: sum (sqrt a - sqrt b)^2 = {want}"
    rng = np.random.default_rng(2)
    for _ in range(20):
        c = _spd(20, rng, 1e4)
        same = nb.frechet_distance(np.ones(20), c, np.ones(20), c)
        assert 0.0 <= same < 1e-4, (
            f"the distance of a Gaussian to itself is 0, and never negative: round-off can make the formula slightly "
            f"negative, so clip the result at 0; got {same}")


# ---------------------------------------------------------------- stage 3: FID from samples


def test_stage3_fid_from_features(nb):
    rng = np.random.default_rng(3)
    f1 = rng.normal(size=(300, 5))
    f2 = rng.normal(size=(200, 5)) @ np.diag([1, 2, 1, 1, 0.5]) + 0.3
    want = nb.frechet_distance(f1.mean(0), np.cov(f1, rowvar=False), f2.mean(0), np.cov(f2, rowvar=False))
    got = nb.fid_from_features(f1, f2)
    assert math.isclose(got, want, rel_tol=1e-9), (
        f"fit each set's mean and covariance (np.cov(..., rowvar=False), ddof=1) and call frechet_distance; "
        f"expected {want:.6f}, got {got:.6f}")
    biased = nb.frechet_distance(f1.mean(0), np.cov(f1, rowvar=False, ddof=0), f2.mean(0),
                                 np.cov(f2, rowvar=False, ddof=0))
    assert not math.isclose(got, biased, rel_tol=1e-6), "use the unbiased covariance (ddof=1), as pytorch-fid does"


def test_stage3_small_sample_bias(nb):
    rng = np.random.default_rng(4)
    d = 16
    fids = {n: np.mean([nb.fid_from_features(rng.normal(size=(n, d)), rng.normal(size=(n, d))) for _ in range(5)])
            for n in (50, 200, 1000)}
    assert fids[50] > fids[200] > fids[1000] > 0, (
        f"two samples from the SAME Gaussian: FID should be positive and shrink as n grows; got {fids}")


def test_stage3_fid_infinity(nb):
    rng = np.random.default_rng(5)
    d = 16
    real = rng.normal(size=(20_000, d))
    mu = np.zeros(d)
    mu[0] = 1.0
    fake = rng.normal(size=(20_000, d)) + mu  # true Frechet distance: ||mu||^2 = 1
    ns = [150, 250, 400, 700, 1200, 2000]
    a, b, fids = bounded(nb.fid_infinity, real, fake, ns, rng=0, what="fid_infinity")
    assert np.shape(fids) == (len(ns),), "fids holds one FID per n"
    assert b > 0, f"the FID of a small sample is biased UP, so the 1/n slope is positive; got b = {b:.3f}"
    assert fids[0] > fids[-1], f"the FID at n = 150 should be above the FID at n = 2000 (bias ~ 1/n); got {fids}"
    assert abs(a - 1.0) < 0.3, (
        f"the intercept a (FID at n -> infinity) should estimate the true distance 1.0; got {a:.3f}. fit "
        "fids = a + b / n by least squares over the ns")
    A = np.c_[np.ones(len(ns)), 1.0 / np.asarray(ns, float)]
    (a_ls, b_ls), *_ = np.linalg.lstsq(A, np.asarray(fids), rcond=None)
    assert math.isclose(a, a_ls, rel_tol=1e-6) and math.isclose(b, b_ls, rel_tol=1e-6), (
        "(a, b) must be the least-squares fit of the returned fids against 1/n")
    fit = a + b / np.asarray(ns, float)
    assert np.max(np.abs(fit - fids)) < 0.5, "a + b/n should fit the measured fids"
    a2, b2, f2 = nb.fid_infinity(real, fake, ns, rng=0)
    assert np.allclose(f2, fids), "subsample with np.random.default_rng(rng): a fixed rng must reproduce the result"
    small = fake[:300]
    ref = real[:2000]  # keep it small: a transposed covariance would be (n x n)
    _, _, f3 = nb.fid_infinity(ref, small, [100, 300], rng=1)
    want = nb.fid_from_features(ref, small)
    assert math.isclose(f3[-1], want, rel_tol=1e-9), (
        "with n = len(fake) the subset is the whole fake set, so its FID must equal fid_from_features(real, fake): "
        "draw subsets WITHOUT replacement")


# ---------------------------------------------------------------- stage 4: precision and recall


def _pr_brute(real, fake, k):
    def radii(x):
        d = np.linalg.norm(x[:, None] - x[None], axis=-1)
        return np.sort(d, axis=1)[:, k]  # column 0 is the point itself

    rr, rf = radii(real), radii(fake)
    p = np.mean([np.any(np.linalg.norm(real - f, axis=1) <= rr) for f in fake])
    r = np.mean([np.any(np.linalg.norm(fake - x, axis=1) <= rf) for x in real])
    return p, r


def test_stage4_hand_example(nb):
    real = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]])
    fake = np.array([[0.5], [-1.0], [6.0], [20.0], [21.0]])
    p, r = nb.precision_recall_knn(real, fake, k=1)
    # real radii (k=1): 1, 1, 1, 1, 7 -> fake 0.5, -1 (exactly on 0's ball: inclusive) and 6 are inside: precision 3/5
    # fake radii (k=1): 1.5, 1.5, 5.5, 1, 1 -> real 0, 1, 2 (on the boundary) in 0.5's ball, 3 and 10 in 6's: recall 1
    assert math.isclose(p, 0.6) and math.isclose(r, 1.0), (
        f"expected (precision, recall) = (0.6, 1.0); got ({p}, {r}). each point's radius is the distance to its k-th "
        "nearest OTHER point of its own set; a point on a ball's boundary counts as inside; precision asks whether "
        "each FAKE point lies in some REAL ball")
    p3, r3 = nb.precision_recall_knn(real, fake, k=2)
    assert (p3, r3) == pytest.approx(_pr_brute(real, fake, 2))


def test_stage4_matches_definition(nb):
    rng = np.random.default_rng(6)
    real = rng.normal(size=(120, 4))
    fake = rng.normal(size=(90, 4)) * 1.5 + 0.5
    for k in (1, 3, 5):
        got = nb.precision_recall_knn(real, fake, k=k)
        want = _pr_brute(real, fake, k)
        assert got == pytest.approx(want), f"k={k}: expected {want}, got {got}"


def test_stage4_separates_fidelity_and_coverage(nb):
    rng = np.random.default_rng(7)
    real = rng.normal(size=(600, 2))
    collapsed = rng.normal(size=(600, 2)) * 0.2  # good samples, one small region: high precision, low recall
    blurry = rng.normal(size=(600, 2)) * 5.0     # covers everything plus junk: high recall, low precision
    pc, rc = nb.precision_recall_knn(real, collapsed)
    pb, rb = nb.precision_recall_knn(real, blurry)
    assert pc > 0.9 and rc < 0.4, f"a collapsed model: precision ~1, recall low; got ({pc:.2f}, {rc:.2f})"
    assert pb < 0.5 and rb > 0.85, f"an over-dispersed model: precision low, recall ~1; got ({pb:.2f}, {rb:.2f})"


# ---------------------------------------------------------------- stage 5: the feature network


def _toy_images(n, seed, size=16):
    t = np.random.default_rng(99).normal(size=(4, size, size))
    for ax in (1, 2):
        for _ in range(3):
            t = (t + np.roll(t, 1, ax) + np.roll(t, -1, ax)) / 3
    r = np.random.default_rng(seed)
    y = r.integers(0, 4, n)
    x = t[y] / t.std() + 0.6 * r.normal(size=(n, size, size))
    return torch.tensor(x[:, None], dtype=torch.float32), torch.tensor(y)


def test_stage5_featurenet_shapes(nb):
    torch.manual_seed(0)
    net = nb.FeatureNet(3, 7, feat_dim=32)
    x = torch.randn(4, 3, 20, 20)
    assert net.forward_features(x).shape == (4, 32), "forward_features -> (B, feat_dim)"
    assert net(x).shape == (4, 7), "forward -> logits (B, n_classes)"
    assert net(torch.randn(2, 3, 28, 28)).shape == (2, 7), "any image size: pool globally before the feature layer"
    net.train()
    f = net.embed(torch.randn(1100, 3, 8, 8), batch_size=300)
    assert isinstance(f, np.ndarray) and f.shape == (1100, 32) and f.dtype == np.float64, (
        "embed returns a float64 ndarray (n, feat_dim), computed in batches")
    assert net.training, "embed must restore the training mode it found (train stays train)"
    net.eval()
    xn = np.random.default_rng(0).normal(size=(5, 3, 8, 8))  # a float64 ndarray
    f2 = net.embed(xn)
    assert not net.training, "... and eval stays eval"
    assert np.allclose(f2, net.forward_features(torch.tensor(xn, dtype=torch.float32)).double().detach().numpy(),
                       atol=1e-6), "embed accepts ndarrays (and float64), casting to float32 inside"


def _reference_training(nb, X, y, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    net = nb.FeatureNet(X.shape[1], int(y.max()) + 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(X[idx]), y[idx]).backward()
            opt.step()
    return net.eval()


def test_stage5_training_procedure(nb):
    X, y = _toy_images(72, 0, size=8)
    got = nb.train_feature_net(X, y, epochs=2, batch_size=16, lr=2e-3, seed=3)
    want = _reference_training(nb, X, y, 2, 16, 2e-3, 3)
    assert not got.training, "return the network in eval mode"
    for (n, p), q in zip(got.named_parameters(), want.parameters()):
        assert torch.allclose(p, q, atol=1e-5), (
            f"parameter {n} differs from the specified procedure: torch.manual_seed(seed), build FeatureNet(C, y.max()+1), "
            "Adam(lr); every epoch a torch.randperm with a Generator seeded by `seed`, minibatches of batch_size in order")
    frozen = nb.train_feature_net(X, y, epochs=1, batch_size=16, lr=0.0, seed=3)
    torch.manual_seed(3)
    init = nb.FeatureNet(X.shape[1], int(y.max()) + 1)
    assert all(torch.equal(p, q) for p, q in zip(frozen.parameters(), init.parameters())), "use lr (lr = 0: no change)"
    other = nb.train_feature_net(X, y, epochs=2, batch_size=24, lr=2e-3, seed=3)
    assert any(not torch.allclose(p, q) for p, q in zip(other.parameters(), got.parameters())), "use batch_size"


def test_stage5_trains(nb):
    X, y = _toy_images(384, 0, size=12)
    Xt, yt = _toy_images(256, 1, size=12)
    net = bounded(nb.train_feature_net, X, y, epochs=4, batch_size=48, lr=3e-3, seed=0, budget=20_000_000,
                  what="train_feature_net")
    acc = (net(Xt).argmax(1) == yt).float().mean().item()
    assert acc > 0.6, f"four epochs on 4 easy classes (chance: 25%) should give > 60% test accuracy; got {acc:.2f}"


def test_stage5_cache(nb):
    X, y = _toy_images(64, 2, size=8)
    with tempfile.TemporaryDirectory() as d:
        a = nb.cached_feature_net(X, y, name="toy", epochs=1, seed=4, cache=d)
        files = os.listdir(d)
        assert len(files) == 1 and files[0].startswith("ch21-featnet-toy"), (
            f"the first call trains and saves one file named ch21-featnet-<name>-...; found {files}")
        want = nb.train_feature_net(X, y, epochs=1, seed=4)
        assert np.allclose(a.embed(X), want.embed(X), atol=1e-5), (
            "the cached net must be train_feature_net(X, y, epochs=epochs, seed=seed, device=device)")
        assert not a.training
        b = nb.cached_feature_net(X, y, name="toy", epochs=1, seed=4, cache=d)
        assert np.allclose(a.embed(X), b.embed(X)), "the second call must load the cached weights"
        assert not b.training, "the loaded net is returned in eval mode"
        assert len(os.listdir(d)) == 1, "loading must not write a second file"
        nb.cached_feature_net(X, y, name="toy", epochs=2, seed=4, cache=d)
        assert len(os.listdir(d)) == 2, "different settings (epochs) must not reuse the cached file"
        c = nb.cached_feature_net(X, y, name="toy", epochs=1, seed=5, cache=d)
        assert len(os.listdir(d)) == 3, "a different seed must not reuse the cached file"
        assert not np.allclose(c.embed(X), a.embed(X)), "... and must train with that seed"
        nb.cached_feature_net(X + 0.5, y, name="toy", epochs=1, seed=4, cache=d)
        assert len(os.listdir(d)) == 4, "different data must not reuse the cached file: hash X and y"
        nb.cached_feature_net(X, np.roll(np.asarray(y), 1), name="toy", epochs=1, seed=4, cache=d)
        assert len(os.listdir(d)) == 5, "different labels must not reuse the cached file: hash y as well as X"
        nb.cached_feature_net(X, y, name="toy", epochs=1, seed=4, device=torch.device("cpu"), cache=d)
        assert len(os.listdir(d)) == 5, "device='cpu' and torch.device('cpu') are the same device: reuse the file"
