import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import hashlib
    import inspect
    import math
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, datasets, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # evaluating generative models · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. before you train a single
    generative model, you build the instruments that grade them: exact likelihoods in bits
    per dimension, the Fréchet distance between feature distributions, precision and recall
    for samples, and the feature network that all of it depends on.

    | stage | you build |
    |---|---|
    | 1 | `bits_per_dim`, `dequantize`, `discrete_nll`: likelihoods on a common scale → `mlfs.generative` |
    | 2 | `sqrtm_psd`, `frechet_distance`: the distance between two Gaussians → `mlfs.generative` |
    | 3 | `fid_from_features`, `fid_infinity`: FID from samples, and its small-sample bias → `mlfs.generative` |
    | 4 | `precision_recall_knn`: fidelity and coverage as two separate numbers → `mlfs.generative` |
    | 5 | `FeatureNet`, `train_feature_net`, `cached_feature_net`: the course's own feature extractor |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def bits_per_dim(nll, n_dims):
    """stage 1: nll in nats per example (float, ndarray or tensor) -> nll / (n_dims ln 2), same type."""
    return nll / (n_dims * math.log(2.0))


@app.function
def dequantize(x, *, n_levels=256, generator=None):
    """stage 1: integer levels x (tensor, values in 0..n_levels-1) -> float32 (x + u) / n_levels, u ~ U[0, 1)."""
    x = torch.as_tensor(x)
    u = torch.rand(x.shape, generator=generator, dtype=torch.float32)
    return (x.to(torch.float32) + u.to(x.device)) / n_levels


@app.function
def discrete_nll(log_density, n_dims, *, n_levels=256):
    """stage 1: log-density (nats) of dequantized data scaled to [0, 1)^D -> upper bound on the discrete nll (nats);
    for data scaled to [-1, 1)^D the density is 2^-D smaller per unit volume, so add D ln 2 to it first."""
    return -log_density + n_dims * math.log(n_levels)


@app.cell(hide_code=True)
def _():
    checks.run(1, bits_per_dim=bits_per_dim, dequantize=dequantize, discrete_nll=discrete_nll)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sqrtm_psd(A):
    """stage 2: the symmetric PSD square root of a symmetric PSD matrix (eigh; negative eigenvalues clipped to 0)."""
    A = np.asarray(A, dtype=np.float64)
    w, V = np.linalg.eigh((A + A.T) / 2)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


@app.function
def frechet_distance(mu1, cov1, mu2, cov2):
    """stage 2: ||mu1 - mu2||^2 + Tr(C1 + C2 - 2 (C1 C2)^(1/2)) as a float >= 0, via symmetric square roots."""
    mu1, mu2 = np.atleast_1d(np.asarray(mu1, float)), np.atleast_1d(np.asarray(mu2, float))
    c1, c2 = np.atleast_2d(np.asarray(cov1, float)), np.atleast_2d(np.asarray(cov2, float))
    s1 = sqrtm_psd(c1)
    m = s1 @ c2 @ s1
    tr_sqrt = float(np.sum(np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0.0, None))))
    d = float(np.sum((mu1 - mu2) ** 2) + np.trace(c1) + np.trace(c2) - 2.0 * tr_sqrt)
    return max(d, 0.0)


@app.cell(hide_code=True)
def _():
    checks.run(2, sqrtm_psd=sqrtm_psd, frechet_distance=frechet_distance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def fid_from_features(feats1, feats2):
    """stage 3: Frechet distance between Gaussian fits (mean, np.cov with ddof=1) of feats (n1, d) and (n2, d)."""
    f1, f2 = np.asarray(feats1, np.float64), np.asarray(feats2, np.float64)
    return frechet_distance(f1.mean(0), np.cov(f1, rowvar=False), f2.mean(0), np.cov(f2, rowvar=False))


@app.function
def fid_infinity(real_feats, fake_feats, ns, *, rng=None):
    """stage 3: FID of the full real set vs n-subsets of fake (no replacement), for each n in ns; least-squares fit
    FID = a + b / n; returns (a, b, fids ndarray (len(ns),))."""
    rng = np.random.default_rng(rng)
    real = np.asarray(real_feats, np.float64)
    fake = np.asarray(fake_feats, np.float64)
    mu_r, cov_r = real.mean(0), np.cov(real, rowvar=False)
    fids = []
    for n in ns:
        sub = fake[rng.choice(len(fake), size=int(n), replace=False)]
        fids.append(frechet_distance(mu_r, cov_r, sub.mean(0), np.cov(sub, rowvar=False)))
    fids = np.array(fids)
    A = np.c_[np.ones(len(ns)), 1.0 / np.asarray(ns, float)]
    (a, b), *_ = np.linalg.lstsq(A, fids, rcond=None)
    return float(a), float(b), fids


@app.cell(hide_code=True)
def _():
    checks.run(3, fid_from_features=fid_from_features, fid_infinity=fid_infinity, frechet_distance=frechet_distance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def precision_recall_knn(real_feats, fake_feats, *, k=3):
    """stage 4: k-NN manifold precision (fake inside some real ball) and recall (real inside some fake ball); balls
    have radius = distance to the k-th nearest OTHER point of the same set, inclusive; returns two floats."""
    real = np.asarray(real_feats, np.float64)
    fake = np.asarray(fake_feats, np.float64)

    def sqdist(a, b):
        return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)

    chunk = 1024  # rows at a time: never materialise an n x n matrix (10k real images would need ~800 MB)

    def radii(x):
        out = np.empty(len(x))
        for s in range(0, len(x), chunk):
            d = sqdist(x[s:s + chunk], x)
            d[np.arange(len(d)), np.arange(s, s + len(d))] = np.inf  # a point is not its own neighbour
            out[s:s + chunk] = np.sqrt(np.partition(d, k - 1, axis=1)[:, k - 1])
        return out

    def inside(points, centres, r):
        return np.concatenate([np.any(np.sqrt(sqdist(points[s:s + chunk], centres)) <= r[None, :] * (1 + 1e-9), axis=1)
                               for s in range(0, len(points), chunk)]) if len(points) else np.zeros(0, bool)

    precision = float(np.mean(inside(fake, real, radii(real))))
    recall = float(np.mean(inside(real, fake, radii(fake))))
    return precision, recall


@app.cell(hide_code=True)
def _():
    checks.run(4, precision_recall_knn=precision_recall_knn)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class FeatureNet(nn.Module):
    """a small image classifier whose penultimate layer is the feature space for FID and precision/recall."""

    def __init__(self, in_channels=1, n_classes=10, *, feat_dim=128):
        """stage 5: conv blocks with downsampling, global average pooling, a feat_dim layer, a linear head `fc`."""
        super().__init__()
        self.feat_dim = feat_dim
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, feat_dim), nn.ReLU(),
        )
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward_features(self, x):
        """stage 5: x (B, C, H, W) float -> features (B, feat_dim)."""
        return self.body(x)

    def forward(self, x):
        """stage 5: logits (B, n_classes) = fc(forward_features(x))."""
        return self.fc(self.forward_features(x))

    @torch.no_grad()
    def embed(self, x, *, batch_size=500):
        """stage 5: features of a tensor or ndarray (cast to float32) as a float64 ndarray (n, feat_dim), eval mode,
        batched, the original train/eval mode restored."""
        was = self.training
        self.eval()
        dev = next(self.parameters()).device
        x = torch.as_tensor(x, dtype=torch.float32)
        out = [self.forward_features(x[i:i + batch_size].to(dev)).double().cpu() for i in range(0, len(x), batch_size)]
        self.train(was)
        return torch.cat(out).numpy()


@app.function
def train_feature_net(X, y, *, epochs=2, batch_size=128, lr=1e-3, seed=0, device="cpu"):
    """stage 5: train a FeatureNet (Adam, cross-entropy) on X float (N, C, H, W), y int64 (N,); seeded; returns it in eval mode."""
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.int64)
    net = FeatureNet(X.shape[1], int(y.max()) + 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            F.cross_entropy(net(X[idx].to(device)), y[idx].to(device)).backward()
            opt.step()
    return net.eval()


@app.function
def cached_feature_net(X, y, *, name="mnist", epochs=2, seed=0, device="cpu", cache=None):
    """stage 5: train_feature_net once and cache its state_dict under `cache` (default datasets.cache_dir()); the file
    name hashes the data, epochs, seed, device and the source of FeatureNet/train_feature_net, so a change retrains;
    returns the net in eval mode on `device`."""
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.int64)
    h = hashlib.sha1()
    h.update(X.numpy().tobytes())
    h.update(y.numpy().tobytes())
    h.update(f"{epochs}-{seed}-{torch.device(device)}".encode())
    for obj in (FeatureNet, train_feature_net):
        try:
            h.update(inspect.getsource(obj).encode())
        except (OSError, TypeError):  # no source file in some interactive sessions: hash the architecture
            h.update(repr(FeatureNet(X.shape[1], int(y.max()) + 1)).encode())
    path = Path(cache if cache is not None else datasets.cache_dir()) / f"ch21-featnet-{name}-{h.hexdigest()[:16]}.pt"
    if path.exists():
        net = FeatureNet(X.shape[1], int(y.max()) + 1)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return net.to(device).eval()
    net = train_feature_net(X, y, epochs=epochs, seed=seed, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, path)
    return net


@app.cell(hide_code=True)
def _():
    checks.run(5, FeatureNet=FeatureNet, train_feature_net=train_feature_net, cached_feature_net=cached_feature_net)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.generative import bits_per_dim, fid_from_features, precision_recall_knn`
    gives you these instruments: every model in this chapter (and the diffusion models of
    chapter 22) is graded with them.
    """)
    return


if __name__ == "__main__":
    app.run()
