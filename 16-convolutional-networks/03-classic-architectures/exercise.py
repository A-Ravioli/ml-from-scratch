import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # classic architectures · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you've built convolution and pooling;
    now you'll assemble them the way the field did between 1998 and 2014: LeNet-5, the
    network that read cheques, and a small VGG, the "only $3 \times 3$ convolutions, just
    go deeper" design that most later networks start from. then a profiler that shows
    where these networks keep their parameters and where they spend their compute (it's
    not the same place). from here on you build with `nn.Conv2d` and `nn.MaxPool2d`.

    | stage | you build |
    |---|---|
    | 1 | `LeNet`: conv–pool–conv–pool–fc–fc–fc for $28 \times 28$ digits |
    | 2 | `MiniVGG`: blocks of two $3 \times 3$ convs (+ batch norm) and a max pool; global average pooling; a linear head |
    | 3 | `profile_model`: output shape, parameters and multiply–accumulates of every conv and linear layer |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class LeNet(nn.Module):
    """LeNet-5 (ReLU and max pooling): inputs smaller than 32x32 are zero-padded to 32x32; returns logits."""

    def __init__(self, num_classes=10, in_channels=1):
        """stage 1: conv1 (in -> 6, 5x5), conv2 (6 -> 16, 5x5), fc1 (400 -> 120), fc2 (120 -> 84), fc3 (84 -> num_classes)."""
        raise NotImplementedError("stage1: implement LeNet.__init__")

    def forward(self, x):
        """stage 1: pad to 32x32, conv1-relu-maxpool2, conv2-relu-maxpool2, flatten, fc1-relu, fc2-relu, fc3."""
        raise NotImplementedError("stage1: implement LeNet.forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, LeNet=LeNet)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class MiniVGG(nn.Module):
    """per width: 2x [conv3x3 (+BN) ReLU] then maxpool 2x2; global average pooling; a linear head `fc`."""

    def __init__(self, num_classes=10, in_channels=3, *, widths=(32, 64, 128), batch_norm=True):
        """stage 2: self.features (nn.Sequential of the blocks), self.pool (AdaptiveAvgPool2d(1)), self.fc; convs have a bias only without BN."""
        raise NotImplementedError("stage2: implement MiniVGG.__init__")

    def forward(self, x):
        """stage 2: fc(flatten(pool(features(x))))."""
        raise NotImplementedError("stage2: implement MiniVGG.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, MiniVGG=MiniVGG)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def profile_model(model, input_shape):
    """stage 3: for every Conv2d and Linear in forward order: dict(name, out_shape (no batch dim), params, macs) for ONE input of input_shape (C, H, W)."""
    raise NotImplementedError("stage3: implement profile_model")


@app.cell(hide_code=True)
def _():
    checks.run(3, profile_model=profile_model, LeNet=LeNet, MiniVGG=MiniVGG)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.cnn import LeNet, MiniVGG` gives you these networks, and you may use
    `torchvision.models.vgg*` and `alexnet` (and FLOP counters such as `fvcore` or
    `torch.utils.flop_counter`): you've built their pieces.
    """)
    return


if __name__ == "__main__":
    app.run()
