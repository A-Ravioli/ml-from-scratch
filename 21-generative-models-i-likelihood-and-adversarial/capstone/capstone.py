import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, datasets
    from mlfs.generative import (  # yours, from this chapter
        PixelCNN,
        RealNVP,
        VAE,
        DCGANDiscriminator,
        DCGANGenerator,
        cached_feature_net,
        fid_from_features,
        flow_bits_per_dim,
        gan_step,
        iwae_nll,
        logit_transform,
        dequantize,
        pixelcnn_nll,
        precision_recall_knn,
    )
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.torchnn import count_parameters, get_device  # yours, from ch 15 · topic 07

    FAMILIES = ("autoregressive", "vae", "gan", "flow")
    # full config: about an hour on an Apple M2 GPU (see brief.md); small config: offline, tests, verify
    FULL = {
        "autoregressive": dict(hidden_channels=64, n_layers=7, epochs=8, batch_size=128, lr=1e-3),
        "vae": dict(latent_dim=32, hidden_dims=(512, 256), epochs=40, batch_size=128, lr=1e-3, iwae_k=500),
        "gan": dict(latent_dim=64, base_channels=64, epochs=15, batch_size=128, lr=2e-4),
        "flow": dict(n_layers=8, hidden_dim=1024, epochs=40, batch_size=256, lr=5e-4, alpha=0.05),
    }
    SMALL = {
        "autoregressive": dict(hidden_channels=16, n_layers=3, epochs=2, batch_size=64, lr=2e-3),
        "vae": dict(latent_dim=8, hidden_dims=(128,), epochs=4, batch_size=64, lr=2e-3, iwae_k=20),
        "gan": dict(latent_dim=16, base_channels=16, epochs=3, batch_size=64, lr=5e-4),
        "flow": dict(n_layers=4, hidden_dim=128, epochs=4, batch_size=64, lr=1e-3, alpha=0.05),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 21 · capstone: four families, one dataset

    read [`brief.md`](brief.md) first. you'll train one model from each family of the
    chapter (an autoregressive PixelCNN, a VAE, a DCGAN and a RealNVP flow) on the same
    binarized MNIST, and compare them on likelihood, sample quality, diversity and cost,
    with honest uncertainty. the checks in `test_capstone.py` pin down the interface; the
    design is yours.

    | stage | you build |
    |---|---|
    | 1 | `binarize`: the shared dataset |
    | 2 | `build_family`, `train_family`: one model per family, one training function |
    | 3 | `nll_bits_per_dim`: exact or bounded test likelihoods on a common scale |
    | 4 | `sample_binary`: samples in the data's own space, from every family |
    | 5 | `evaluate_samples`, `bpd_ci`: FID, precision/recall, label entropy, bootstrap intervals |
    """)
    return


@app.function
def binarize(X, *, seed=0):
    """stage 1: uint8 images (N, H, W) -> float32 tensor (N, 1, H, W) of 0/1, each pixel ~ Bernoulli(X / 255), drawn
    with torch.bernoulli(..., generator=torch.Generator().manual_seed(seed))."""
    raise NotImplementedError("stage1: implement binarize")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", binarize=binarize)
    return


@app.function
def build_family(family, config, *, img_size=28, seed=0):
    """stage 2: torch.manual_seed(seed), then 'autoregressive': PixelCNN(1, n_levels=2, ...); 'vae': VAE(img_size**2,
    latent_dim, hidden_dims, likelihood='bernoulli'); 'gan': nn.ModuleDict(G=DCGANGenerator, D=DCGANDiscriminator) for
    1 channel; 'flow': RealNVP(img_size**2, n_layers, hidden_dim); ValueError otherwise."""
    raise NotImplementedError("stage2: implement build_family")


@app.function
def flow_inputs(x, alpha, generator=None):
    """binary images -> (y flattened logit-space inputs, log_det of the logit transform), dequantized with 2 levels."""
    y, ld = logit_transform(dequantize(x.long(), n_levels=2, generator=generator), alpha=alpha)
    return y.flatten(1), ld


@app.function
def train_family(family, model, X, config, *, device="cpu", seed=0):
    """stage 2: train `model` on binary images X (N, 1, H, W) for config['epochs'] epochs of shuffled minibatches
    (torch.Generator seeded with `seed`), Adam(config['lr']); returns dict(seconds, losses = per-epoch mean loss)."""
    raise NotImplementedError("stage2: implement train_family")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", build_family=build_family, train_family=train_family, binarize=binarize)
    return


@app.function
def nll_bits_per_dim(family, model, X, config, *, device="cpu", generator=None, batch_size=500):
    """stage 3: per-image test bits/dim, float64 ndarray (N,): 'autoregressive' exact (pixelcnn_nll), 'vae' the
    importance-weighted bound (iwae_nll, k = config['iwae_k']), 'flow' the dequantized bound (flow_bits_per_dim,
    n_levels=2, alpha); None for 'gan' (no likelihood). eval mode, no gradients, in batches."""
    raise NotImplementedError("stage3: implement nll_bits_per_dim")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", nll_bits_per_dim=nll_bits_per_dim, build_family=build_family,
               train_family=train_family, binarize=binarize)
    return


@app.function
def sample_binary(family, model, n, config, *, img_size=28, device="cpu", generator=None):
    """stage 4: n binary samples, float32 CPU tensor (n, 1, img_size, img_size) of 0/1: PixelCNN samples directly; VAE
    and GAN outputs treated as pixel probabilities and drawn with torch.bernoulli(generator); flow samples mapped back
    through the inverse logit transform and quantized (pixel = 1 where the dequantized value >= 1/2)."""
    raise NotImplementedError("stage4: implement sample_binary")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", sample_binary=sample_binary, build_family=build_family,
               train_family=train_family, binarize=binarize)
    return


@app.function
def evaluate_samples(samples, real_feats, featnet, *, k=3):
    """stage 5: dict(fid, precision, recall, class_entropy): featnet.embed(samples) against real_feats with
    fid_from_features and precision_recall_knn(real_feats, fake, k=k); class_entropy = entropy (nats) of the
    histogram of featnet's predicted labels (argmax of featnet(samples))."""
    raise NotImplementedError("stage5: implement evaluate_samples")


@app.function
def bpd_ci(bpd, *, n_boot=2000, alpha=0.05, seed=0):
    """stage 5: (mean, lo, hi): the mean of per-image bits/dim and its percentile bootstrap interval (bootstrap_ci)."""
    raise NotImplementedError("stage5: implement bpd_ci")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", evaluate_samples=evaluate_samples, bpd_ci=bpd_ci)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the experiment

    your cells go here (see the brief's "deliverables"): binarize the 60,000 training
    digits with seed 0 and the 10,000 test digits with seed 1; train the feature network
    with `cached_feature_net(X_train, y_train, name="mnist-binary", epochs=2, seed=0)` and
    embed the test digits; then, for each family in `FAMILIES` with its `FULL` config:
    build, train (timed), score the test set with `nll_bits_per_dim` and `bpd_ci`, draw
    2,000 samples twice (timed) and evaluate them, and show 16 samples. add a reference
    row for 2,000 real training images. use `SMALL` (and a few thousand images) to check
    that your cells run before you start the full run (about an hour on an Apple M2
    GPU).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)):

    1. **the table**, with what each column can and cannot tell you: which likelihoods are
       exact and which are bounds, why the GAN has none, what the FID's two draws say about
       its noise, and why course FIDs aren't comparable with papers'.
    2. **ranking the families**: on likelihood, on sample quality, on diversity, on cost.
       do the rankings agree? where they disagree, explain the mechanism (mode-covering
       maximum likelihood vs adversarial training, bounds vs exact likelihoods, sequential
       vs one-pass sampling).
    3. **look at the samples**: one failure per family that the numbers miss, or that a
       number reveals before your eyes do.
    4. **what you'd change** with ten times the compute, for each family, and which
       conclusions you expect to survive.
    """)
    return


if __name__ == "__main__":
    app.run()
