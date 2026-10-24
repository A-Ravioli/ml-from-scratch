import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # autoregressive models · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. an autoregressive model writes a
    joint distribution as a chain of conditionals, $p(x) = \prod_i p(x_i \mid x_{<i})$, and
    computes all of them in one forward pass by hiding the future with **masks**. you'll
    build the masked layers, MADE for vectors, and PixelCNN for images, with exact
    likelihoods and (slow) exact sampling.

    | stage | you build |
    |---|---|
    | 1 | `MaskedLinear`: a linear layer whose weight is multiplied by a fixed 0/1 mask → `mlfs.generative` |
    | 2 | `MADE`: masks from degrees, so output $i$ sees only inputs $< i$ → `mlfs.generative` |
    | 3 | `made_nll`, `made_sample`: exact likelihood of binary vectors, and sampling one dimension at a time |
    | 4 | `MaskedConv2d`: type-A and type-B masked convolutions → `mlfs.generative` |
    | 5 | `PixelCNN`, `pixelcnn_nll`: a stack of masked convolutions with a categorical output per pixel → `mlfs.generative` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class MaskedLinear(nn.Linear):
    """a linear layer computing x @ (weight * mask).T + bias, with mask a registered (out, in) buffer."""

    def __init__(self, in_features, out_features, bias=True):
        """stage 1: an nn.Linear plus a buffer `mask` of ones, shape (out_features, in_features)."""
        raise NotImplementedError("stage1: implement MaskedLinear.__init__")

    def set_mask(self, mask):
        """stage 1: copy a 0/1 mask (tensor or ndarray, shape (out, in)) into the buffer."""
        raise NotImplementedError("stage1: implement MaskedLinear.set_mask")

    def forward(self, x):
        """stage 1: F.linear with the masked weight, so masked entries get no output and no gradient."""
        raise NotImplementedError("stage1: implement MaskedLinear.forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, MaskedLinear=MaskedLinear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class MADE(nn.Module):
    """masked autoencoder for distribution estimation, natural ordering: output [:, :, i] depends only on x[:, :i]."""

    def __init__(self, in_dim, hidden_dims, *, n_outputs_per_dim=1, seed=0):
        """stage 2: input degrees 1..D; hidden degrees drawn with np.random.default_rng(seed) from [min of the previous
        layer's degrees, D - 1]; masks hidden >= previous, output d (degree d+1) > hidden; MaskedLinear + ReLU layers."""
        raise NotImplementedError("stage2: implement MADE.__init__")

    def forward(self, x):
        """stage 2: x (B, D) -> (B, n_outputs_per_dim, D); output unit o * D + d is entry [:, o, d]."""
        raise NotImplementedError("stage2: implement MADE.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, MADE=MADE, MaskedLinear=MaskedLinear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def made_nll(model, x):
    """stage 3: exact nll in nats (B,) of binary x (B, D) under a MADE with one output (a Bernoulli logit) per dim."""
    raise NotImplementedError("stage3: implement made_nll")


@app.function
def made_sample(model, n, *, generator=None):
    """stage 3: n binary samples (n, D) float, one dimension per forward pass (D passes), x_i ~ Bernoulli(sigmoid)."""
    raise NotImplementedError("stage3: implement made_sample")


@app.cell(hide_code=True)
def _():
    checks.run(3, made_nll=made_nll, made_sample=made_sample, MADE=MADE)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class MaskedConv2d(nn.Conv2d):
    """a raster-scan causal convolution: 'A' hides the centre pixel and everything after it, 'B' only what's after."""

    def __init__(self, mask_type, in_channels, out_channels, kernel_size):
        """stage 4: nn.Conv2d with padding kernel_size // 2 and a buffer `mask` (out, in, k, k); ValueError for other types."""
        raise NotImplementedError("stage4: implement MaskedConv2d.__init__")

    def forward(self, x):
        """stage 4: convolution with weight * mask."""
        raise NotImplementedError("stage4: implement MaskedConv2d.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, MaskedConv2d=MaskedConv2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class PixelCNN(nn.Module):
    """x (B, C, H, W) in [0, 1] -> logits (B, n_levels, C, H, W); channels of a pixel are independent given the past."""

    def __init__(self, in_channels=1, *, n_levels=256, hidden_channels=64, n_layers=7):
        """stage 5: a 7x7 type-A masked conv, n_layers - 1 residual 3x3 type-B masked convs, two 1x1 convs to n_levels * C."""
        raise NotImplementedError("stage5: implement PixelCNN.__init__")

    def forward(self, x):
        """stage 5: logits (B, n_levels, C, H, W); the input is rescaled to [-1, 1] first."""
        raise NotImplementedError("stage5: implement PixelCNN.forward")

    @torch.no_grad()
    def sample(self, n, img_size, *, generator=None, device="cpu"):
        """stage 5: n images (n, C, H, W) in [0, 1], one forward pass per pixel in raster order (H * W passes);
        img_size is an int or (H, W); levels drawn with torch.multinomial(..., generator=generator) on the CPU."""
        raise NotImplementedError("stage5: implement PixelCNN.sample")


@app.function
def pixelcnn_nll(model, x):
    """stage 5: exact nll in nats per image (B,) of x (B, C, H, W) in [0, 1] with levels round(x * (n_levels - 1))."""
    raise NotImplementedError("stage5: implement pixelcnn_nll")


@app.cell(hide_code=True)
def _():
    checks.run(5, PixelCNN=PixelCNN, pixelcnn_nll=pixelcnn_nll, MaskedConv2d=MaskedConv2d)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.generative import MADE, PixelCNN` gives you these models; the capstone
    trains your PixelCNN as the autoregressive family.
    """)
    return


if __name__ == "__main__":
    app.run()
