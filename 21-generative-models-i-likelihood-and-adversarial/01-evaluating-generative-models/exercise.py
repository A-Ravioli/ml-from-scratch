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
    # evaluating generative models · build

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
    raise NotImplementedError("stage1: implement bits_per_dim")


@app.function
def dequantize(x, *, n_levels=256, generator=None):
    """stage 1: integer levels x (tensor, values in 0..n_levels-1) -> float32 (x + u) / n_levels, u ~ U[0, 1)."""
    raise NotImplementedError("stage1: implement dequantize")


@app.function
def discrete_nll(log_density, n_dims, *, n_levels=256):
    """stage 1: log-density (nats) of dequantized data scaled to [0, 1)^D -> upper bound on the discrete nll (nats);
    for data scaled to [-1, 1)^D the density is 2^-D smaller per unit volume, so add D ln 2 to it first."""
    raise NotImplementedError("stage1: implement discrete_nll")


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
    raise NotImplementedError("stage2: implement sqrtm_psd")


@app.function
def frechet_distance(mu1, cov1, mu2, cov2):
    """stage 2: ||mu1 - mu2||^2 + Tr(C1 + C2 - 2 (C1 C2)^(1/2)) as a float >= 0, via symmetric square roots."""
    raise NotImplementedError("stage2: implement frechet_distance")


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
    raise NotImplementedError("stage3: implement fid_from_features")


@app.function
def fid_infinity(real_feats, fake_feats, ns, *, rng=None):
    """stage 3: FID of the full real set vs n-subsets of fake (no replacement), for each n in ns; least-squares fit
    FID = a + b / n; returns (a, b, fids ndarray (len(ns),))."""
    raise NotImplementedError("stage3: implement fid_infinity")


@app.cell(hide_code=True)
def _():
    checks.run(3, fid_from_features=fid_from_features, fid_infinity=fid_infinity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def precision_recall_knn(real_feats, fake_feats, *, k=3):
    """stage 4: k-NN manifold precision (fake inside some real ball) and recall (real inside some fake ball); balls
    have radius = distance to the k-th nearest OTHER point of the same set, inclusive; returns two floats."""
    raise NotImplementedError("stage4: implement precision_recall_knn")


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
        raise NotImplementedError("stage5: implement FeatureNet.__init__")

    def forward_features(self, x):
        """stage 5: x (B, C, H, W) float -> features (B, feat_dim)."""
        raise NotImplementedError("stage5: implement FeatureNet.forward_features")

    def forward(self, x):
        """stage 5: logits (B, n_classes) = fc(forward_features(x))."""
        raise NotImplementedError("stage5: implement FeatureNet.forward")

    @torch.no_grad()
    def embed(self, x, *, batch_size=500):
        """stage 5: features of a tensor or ndarray (cast to float32) as a float64 ndarray (n, feat_dim), eval mode,
        batched, the original train/eval mode restored."""
        raise NotImplementedError("stage5: implement FeatureNet.embed")


@app.function
def train_feature_net(X, y, *, epochs=2, batch_size=128, lr=1e-3, seed=0, device="cpu"):
    """stage 5: train a FeatureNet (Adam, cross-entropy) on X float (N, C, H, W), y int64 (N,); seeded; returns it in eval mode."""
    raise NotImplementedError("stage5: implement train_feature_net")


@app.function
def cached_feature_net(X, y, *, name="mnist", epochs=2, seed=0, device="cpu", cache=None):
    """stage 5: train_feature_net once and cache its state_dict under `cache` (default datasets.cache_dir()); the file
    name hashes the data, epochs, seed, device and the source of FeatureNet/train_feature_net, so a change retrains;
    returns the net in eval mode on `device`."""
    raise NotImplementedError("stage5: implement cached_feature_net")


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
