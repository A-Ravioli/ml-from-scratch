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
    # modern convnets & transfer · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. two ideas define convnets after ResNet:
    spend fewer multiplications per layer (depthwise separable convolutions, ConvNeXt's
    large depthwise kernels and inverted bottlenecks), and don't train from scratch when
    someone has already trained a network on more data (transfer learning). you'll build
    both blocks and the routine that turns a pretrained backbone into a classifier for a
    new task, by linear probing or fine-tuning.

    | stage | you build |
    |---|---|
    | 1 | `DepthwiseSeparableConv`: a per-channel $k \times k$ conv, then a $1 \times 1$ conv that mixes channels |
    | 2 | `ConvNeXtBlock`: $7 \times 7$ depthwise conv, layer norm, an inverted bottleneck MLP, layer scale, residual |
    | 3 | `replace_head`, `freeze_backbone`: a fresh classifier on a pretrained network, and a frozen body |
    | 4 | `fine_tune`: linear probing or full fine-tuning, with a smaller learning rate for the backbone |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class DepthwiseSeparableConv(nn.Module):
    """depthwise k x k conv (one filter per channel) -> BN -> ReLU -> pointwise 1x1 conv -> BN -> ReLU (MobileNet v1)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """stage 1: depthwise (groups=in_channels, padding k // 2, stride, no bias), bn1, pointwise (1x1, no bias), bn2."""
        raise NotImplementedError("stage1: implement DepthwiseSeparableConv.__init__")

    def forward(self, x):
        """stage 1: relu(bn2(pointwise(relu(bn1(depthwise(x))))))."""
        raise NotImplementedError("stage1: implement DepthwiseSeparableConv.forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, DepthwiseSeparableConv=DepthwiseSeparableConv)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ConvNeXtBlock(nn.Module):
    """x + layer_scale * pwconv2(gelu(pwconv1(layernorm(dwconv7x7(x))))), with the MLP applied channels-last."""

    def __init__(self, dim, *, expansion=4, layer_scale_init=1e-6):
        """stage 2: dwconv (7x7, padding 3, groups=dim), norm (LayerNorm(dim, eps=1e-6)), pwconv1/pwconv2 (Linear dim <-> expansion*dim), gamma = layer_scale_init * ones(dim)."""
        raise NotImplementedError("stage2: implement ConvNeXtBlock.__init__")

    def forward(self, x):
        """stage 2: (B, C, H, W) -> dwconv -> permute to (B, H, W, C) -> norm, pwconv1, GELU, pwconv2, * gamma -> permute back -> + x."""
        raise NotImplementedError("stage2: implement ConvNeXtBlock.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, ConvNeXtBlock=ConvNeXtBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def replace_head(model, num_classes):
    """stage 3: replace model.fc (or model.head) with a fresh nn.Linear(in_features, num_classes) on the same device/dtype; returns model."""
    raise NotImplementedError("stage3: implement replace_head")


@app.function
def freeze_backbone(model, *, head_name="fc"):
    """stage 3: requires_grad=False for every parameter outside the submodule `head_name` (True inside it); ValueError if there is no such submodule; returns model."""
    raise NotImplementedError("stage3: implement freeze_backbone")


@app.cell(hide_code=True)
def _():
    checks.run(3, replace_head=replace_head, freeze_backbone=freeze_backbone)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fine_tune(model, train_loader, num_classes, *, mode="full", epochs=5, lr=1e-3, backbone_lr_mult=0.1,
              weight_decay=0.0, head_name="fc", device="cpu"):
    """stage 4: replace the head, then train with AdamW: mode 'linear' = frozen backbone in eval mode, head only; 'full' = all, backbone at lr * backbone_lr_mult. returns {'train_loss': [per epoch]}."""
    raise NotImplementedError("stage4: implement fine_tune")


@app.cell(hide_code=True)
def _():
    checks.run(4, fine_tune=fine_tune, replace_head=replace_head, freeze_backbone=freeze_backbone)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.cnn import DepthwiseSeparableConv, replace_head, freeze_backbone` gives you
    these tools, and you may use `torchvision`'s MobileNet, EfficientNet and ConvNeXt models
    and pretrained weights for transfer learning: you've built their pieces.
    """)
    return


if __name__ == "__main__":
    app.run()
