import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # residual networks · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. a deeper network can represent
    everything a shallower one can, yet plain deep networks train *worse*. the fix that
    made networks of 100+ layers routine is one line: add the input back. you'll build
    the residual block, the ResNet around it (the backbone chapters 18, 21, 25 and 28 reuse),
    and the bottleneck block that makes very deep ResNets affordable.

    | stage | you build |
    |---|---|
    | 1 | `BasicBlock`: two 3×3 convs with batch norm, plus a shortcut (identity or 1×1 projection) |
    | 2 | `ResNet`, `resnet18`: a stem, four stages of blocks, global average pooling, a linear head; `forward_features` |
    | 3 | `Bottleneck`: 1×1 reduce → 3×3 → 1×1 expand, the block of ResNet-50 and deeper |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class BasicBlock(nn.Module):
    """two 3x3 conv-BN layers; output relu(F(x) + shortcut(x)), or relu(F(x)) when residual=False."""

    def __init__(self, in_channels, out_channels, stride=1, *, residual=True):
        """stage 1: conv1 (3x3, stride, no bias), bn1, conv2 (3x3, no bias), bn2; shortcut = 1x1 conv + BN if the shape changes, else identity."""
        raise NotImplementedError("stage1: implement BasicBlock.__init__")

    def forward(self, x):
        """stage 1: relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x)); no addition when residual=False."""
        raise NotImplementedError("stage1: implement BasicBlock.forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, BasicBlock=BasicBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ResNet(nn.Module):
    """stem -> stages of BasicBlocks (width doubles, resolution halves from stage 2 on) -> global average pool -> fc."""

    def __init__(self, block_counts=(2, 2, 2, 2), num_classes=10, *, in_channels=3, base_width=64, residual=True,
                 small_input=True):
        """stage 2: stem (3x3/1 conv-BN-ReLU if small_input, else 7x7/2 conv-BN-ReLU + 3x3/2 maxpool), self.layers, self.fc; feature_dim."""
        raise NotImplementedError("stage2: implement ResNet.__init__")

    def forward_features(self, x):
        """stage 2: the pooled features (B, feature_dim), before the head."""
        raise NotImplementedError("stage2: implement ResNet.forward_features")

    def forward(self, x):
        """stage 2: logits (B, num_classes) = fc(forward_features(x))."""
        raise NotImplementedError("stage2: implement ResNet.forward")


@app.function
def resnet18(num_classes=10, *, in_channels=3, residual=True):
    """stage 2: ResNet with block counts (2, 2, 2, 2) and base width 64, CIFAR stem."""
    raise NotImplementedError("stage2: implement resnet18")


@app.cell(hide_code=True)
def _():
    checks.run(2, ResNet=ResNet, resnet18=resnet18, BasicBlock=BasicBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 (stride) -> 1x1 expand to mid_channels * expansion, each with BN; projection shortcut when the shape changes."""

    def __init__(self, in_channels, mid_channels, stride=1, *, expansion=4, residual=True):
        """stage 3: conv1 (1x1), bn1, conv2 (3x3, stride), bn2, conv3 (1x1 -> mid*expansion), bn3, shortcut; out_channels attribute."""
        raise NotImplementedError("stage3: implement Bottleneck.__init__")

    def forward(self, x):
        """stage 3: relu(bn3(conv3(relu(bn2(conv2(relu(bn1(conv1(x)))))))) + shortcut(x))."""
        raise NotImplementedError("stage3: implement Bottleneck.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, Bottleneck=Bottleneck, BasicBlock=BasicBlock)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.cnn import resnet18` gives you this network (the chapter's capstone trains
    it on CIFAR-10, and chapters 18, 21, 25 and 28 reuse it as a backbone), and you may use
    `torchvision.models.resnet18`/`resnet50` and their blocks: you've built them.
    """)
    return


if __name__ == "__main__":
    app.run()
