import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import checks, datasets
    from mlfs.diffusion import (  # yours, from this chapter
        DiffusionSchedule,
        TimeUNet,
        alpha_bar_at,
        cfg_model,
        ddim_sample,
        ddpm_loss,
        drop_labels,
        flow_matching_loss,
        sample_flow,
    )
    from mlfs.generative import cached_feature_net, fid_from_features  # yours, from ch 21 · topic 01
    from mlfs.torchnn import count_parameters, ema_update, get_device  # yours, from ch 15 · topic 07

    OBJECTIVES = ("ddpm", "flow")
    N_CLASSES = 10
    # full config: about an hour on an Apple M2 GPU shared with other jobs (see brief.md); small: offline, tests, verify
    FULL = dict(base_channels=32, channel_mults=(1, 2, 2), dropout=0.1, n_iters=2500, batch_size=128, lr=5e-4,
                warmup=100, ema_decay=0.999, p_uncond=0.1, hflip=True, n_train_steps=1000, clip=True)
    SMALL = dict(base_channels=8, channel_mults=(1, 2), dropout=0.0, n_iters=40, batch_size=32, lr=1e-3,
                 warmup=5, ema_decay=0.9, p_uncond=0.1, hflip=True, n_train_steps=100, clip=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 22 · capstone: a class-conditional generator for CIFAR-10

    read [`brief.md`](brief.md) first. you'll train the same class-conditional U-Net twice
    on CIFAR-10, once as a DDPM noise predictor and once as a flow-matching velocity field,
    with the same data, budget and label dropout, and compare them under classifier-free
    guidance at equal numbers of network calls. the checks in `test_capstone.py` pin down the
    interface; the design is yours.

    | stage | you build |
    |---|---|
    | 1 | `prepare_cifar`, `to_unit`, `random_hflip`: the data in model space and back |
    | 2 | `build_model`, `make_schedule`: one network for both objectives |
    | 3 | `train_generator`: label dropout, warmup, EMA; the same budget for both objectives |
    | 4 | `ClipPrediction`, `generate`: classifier-free guidance at a given number of network calls |
    | 5 | `evaluate`: course FID and class accuracy of the samples |
    """)
    return


@app.function
def prepare_cifar(X, y):
    """stage 1: uint8 images (N, 32, 32, 3) NHWC -> (float32 tensor (N, 3, 32, 32) in [-1, 1] = X / 127.5 - 1, int64
    labels (N,))."""
    raise NotImplementedError("stage1: implement prepare_cifar")


@app.function
def to_unit(x):
    """stage 1: model-space images in [-1, 1] -> [0, 1], clamped, float32 on the CPU (for plots and FeatureNet)."""
    raise NotImplementedError("stage1: implement to_unit")


@app.function
def random_hflip(x, *, generator=None):
    """stage 1: flip each image (B, C, H, W) left-right where torch.rand(B, generator) < 0.5; returns a new tensor."""
    raise NotImplementedError("stage1: implement random_hflip")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", prepare_cifar=prepare_cifar, to_unit=to_unit, random_hflip=random_hflip)
    return


@app.function
def build_model(config, *, seed=0):
    """stage 2: torch.manual_seed(seed), then TimeUNet(3, base_channels, channel_mults, n_classes=10, dropout) from
    config."""
    raise NotImplementedError("stage2: implement build_model")


@app.function
def make_schedule(config):
    """stage 2: the DDPM schedule: DiffusionSchedule(config['n_train_steps']) with the linear betas rescaled to keep the
    default endpoints' total noise: beta_start = 1e-4 * 1000 / n, beta_end = 0.02 * 1000 / n."""
    raise NotImplementedError("stage2: implement make_schedule")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", build_model=build_model, make_schedule=make_schedule)
    return


@app.function
def train_generator(objective, model, X, y, config, *, device="cpu", seed=0):
    """stage 3: objective 'ddpm' (ddpm_loss on make_schedule(config)) or 'flow' (flow_matching_loss), ValueError
    otherwise. Adam(config['lr']) with a linear warmup from lr / warmup over config['warmup'] steps; all randomness
    from g = torch.Generator().manual_seed(seed): per step idx = randint(N, (batch_size,), g), random_hflip(..., g) if
    config['hflip'], labels drop_labels(y[idx], p_uncond, 10, g), then the loss (generator=g). an EMA copy of the model
    (copy.deepcopy before training) is updated after optimizer step k (k = 0, 1, ...) with ema_update(ema, model,
    min(ema_decay, (1 + k) / (10 + k))), a warmed-up decay so the EMA forgets the initialization. returns
    dict(ema = the EMA model in eval mode, losses, lrs (per step), seconds)."""
    raise NotImplementedError("stage3: implement train_generator")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", train_generator=train_generator, build_model=build_model,
               random_hflip=random_hflip)
    return


@app.class_definition
class ClipPrediction(torch.nn.Module):
    """a model wrapper whose implied clean-image prediction is clipped to [-1, 1] (Ho et al.'s sampler does this)."""

    def __init__(self, model, objective, schedule=None):
        """stage 4: keep the model as a submodule, the objective and (for 'ddpm') the schedule."""
        raise NotImplementedError("stage4: implement ClipPrediction.__init__")

    def forward(self, x, t, y=None):
        """stage 4: 'ddpm': x0 = clamp((x - sqrt(1 - ab) eps) / sqrt(ab), -1, 1) with ab = alpha_bar_at(schedule, t),
        returns (x - sqrt(ab) x0) / sqrt(1 - ab); 'flow': x1 = clamp(x + (1 - t) v, -1, 1), returns (x1 - x) / (1 - t)."""
        raise NotImplementedError("stage4: implement ClipPrediction.forward")


@app.function
def generate(objective, model, labels, config, *, guidance_scale=1.0, n_steps=50, device="cpu", generator=None,
             batch_size=500):
    """stage 4: images in [0, 1] (CPU float32, (N, 3, H, W)) for the given labels (N,), in chunks of batch_size, from
    cfg_model(model, guidance_scale, 10), wrapped in ClipPrediction if config['clip']: 'ddpm' with ddim_sample(eta=0,
    n_steps, schedule make_schedule(config)), 'flow' with sample_flow(method='euler', n_steps): n_steps network calls
    per chunk either way (each on 2 x chunk rows, for guidance). image size config.get('img_size', 32)."""
    raise NotImplementedError("stage4: implement generate")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", generate=generate, ClipPrediction=ClipPrediction,
               train_generator=train_generator, build_model=build_model, make_schedule=make_schedule)
    return


@app.function
def evaluate(images, labels, featnet, real_feats):
    """stage 5: dict(fid = fid_from_features(real_feats, featnet.embed(images)), accuracy = fraction of images whose
    featnet(images).argmax(1) equals their label (eval mode, no gradients))."""
    raise NotImplementedError("stage5: implement evaluate")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", evaluate=evaluate)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the experiment

    your cells go here (see the brief's "deliverables"). load CIFAR-10 with
    `datasets.cifar10()` and `prepare_cifar`; train the feature network once with
    `cached_feature_net(to_unit(X_train), y_train, name="cifar10", epochs=5, seed=0, device=...)`
    and embed the 10,000 test images; then, for each objective in `OBJECTIVES`, build the model
    with `build_model(FULL, seed=0)` and train it with `train_generator(..., FULL, ...)` (timed;
    keep the trained model itself as well as the returned EMA copy). sample 100 images per class
    from each EMA model with `generate` and score them with `evaluate`: at $w = 2$ with 5, 10 and
    25 network calls, and at 25 calls with $w = 1$, 2 and 4; draw the main setting ($w = 2$, 25
    calls) a second time with another generator seed, once with the raw (non-EMA) weights and
    once with `clip=False`; add the reference row for 1,000 real training images; show a grid of
    samples per objective at the smallest and largest $w$. use `SMALL` (and a few thousand
    images) to check that your cells run before the full run (about 45 minutes on a shared
    Apple M2 GPU; see the brief).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)):

    1. **the table**: FID and class accuracy for both objectives across guidance scales and
       numbers of network calls, with the sampling noise (the second draw), the real-data
       reference row and the hardware. say what the course FID can and cannot be compared
       with.
    2. **DDPM against flow matching at equal compute**: which wins at few calls, which at many,
       and why (the curvature of each ODE, topic 04's DDIM-as-Euler and topic 05's straight
       paths); is the difference larger than the sampling noise?
    3. **the guidance tradeoff**: how FID and accuracy move with $w$, where the best FID is,
       and what you see in the samples at large $w$.
    4. **what you'd change** with ten times the compute, and which conclusions you expect to
       survive a second training seed.
    """)
    return


if __name__ == "__main__":
    app.run()
