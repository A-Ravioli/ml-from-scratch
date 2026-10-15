import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # detection & segmentation · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. a classifier says *what* is in an
    image. detection says what and *where* (a box per object), and segmentation says it
    for every pixel. you'll build the pieces every detector shares (overlap between boxes,
    non-maximum suppression, anchors and how they're matched to objects) and a complete
    small segmentation network, the U-Net, with the loss that makes it work on small
    objects.

    | stage | you build |
    |---|---|
    | 1 | `box_iou`: intersection over union between every pair of boxes |
    | 2 | `nms`: keep the best box of each cluster of overlapping detections |
    | 3 | `make_anchors`, `match_anchors`: a grid of reference boxes, and which object each one is responsible for |
    | 4 | `dice_loss`: a soft overlap loss for segmentation masks |
    | 5 | `DoubleConv`, `UNet`: an encoder–decoder with skip connections |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def box_iou(boxes1, boxes2):
    """stage 1: IoU of every pair of xyxy boxes: (N, 4), (M, 4) -> (N, M); 0 where the union is empty."""
    raise NotImplementedError("stage1: implement box_iou")


@app.cell(hide_code=True)
def _():
    checks.run(1, box_iou=box_iou)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def nms(boxes, scores, iou_threshold=0.5):
    """stage 2: greedy non-maximum suppression: int64 indices (on boxes.device) of the kept boxes, in decreasing score order; a box is dropped if IoU > threshold with a kept one."""
    raise NotImplementedError("stage2: implement nms")


@app.cell(hide_code=True)
def _():
    checks.run(2, nms=nms, box_iou=box_iou)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def make_anchors(feature_size, stride, sizes, ratios):
    """stage 3: xyxy anchors (H*W*len(sizes)*len(ratios), 4) centred at ((j + 0.5) stride, (i + 0.5) stride); order: row, column, size, ratio; ratio = height / width."""
    raise NotImplementedError("stage3: implement make_anchors")


@app.function
def match_anchors(anchors, gt_boxes, *, pos_threshold=0.5, neg_threshold=0.4):
    """stage 3: (labels, matched): labels 1 (IoU >= pos), 0 (< neg), -1 (in between); every object's best anchor is positive; matched = each anchor's best object, or for a forced anchor the object that forced it."""
    raise NotImplementedError("stage3: implement match_anchors")


@app.cell(hide_code=True)
def _():
    checks.run(3, make_anchors=make_anchors, match_anchors=match_anchors, box_iou=box_iou)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def dice_loss(logits, targets, *, eps=1.0):
    """stage 4: 1 - soft Dice per image, averaged over the batch: p = sigmoid(logits); dice = (2 sum(p t) + eps) / (sum(p) + sum(t) + eps)."""
    raise NotImplementedError("stage4: implement dice_loss")


@app.cell(hide_code=True)
def _():
    checks.run(4, dice_loss=dice_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class DoubleConv(nn.Module):
    """two (3x3 conv, padding 1, no bias -> BatchNorm2d -> ReLU) layers: U-Net's basic unit."""

    def __init__(self, in_channels, out_channels):
        """stage 5: self.net = nn.Sequential(conv, BN, ReLU, conv, BN, ReLU)."""
        raise NotImplementedError("stage5: implement DoubleConv.__init__")

    def forward(self, x):
        """stage 5: self.net(x)."""
        raise NotImplementedError("stage5: implement DoubleConv.forward")


@app.class_definition
class UNet(nn.Module):
    """encoder (DoubleConv + maxpool, channels doubling) -> bottleneck -> decoder (2x2 transposed conv, concat skip, DoubleConv) -> 1x1 conv."""

    def __init__(self, in_channels=1, out_channels=1, *, base_channels=32, depth=3):
        """stage 5: downs (depth DoubleConvs), bottleneck, ups (depth ConvTranspose2d k=2 s=2), up_convs (depth DoubleConvs), out_conv (1x1)."""
        raise NotImplementedError("stage5: implement UNet.__init__")

    def forward(self, x):
        """stage 5: logits (B, out_channels, H, W); H and W must be divisible by 2**depth (ValueError otherwise)."""
        raise NotImplementedError("stage5: implement UNet.forward")


@app.cell(hide_code=True)
def _():
    checks.run(5, UNet=UNet, DoubleConv=DoubleConv, dice_loss=dice_loss)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.cnn import box_iou, nms, UNet` gives you these tools, and you may use
    `torchvision.ops.box_iou`, `torchvision.ops.nms`, `nn.ConvTranspose2d` and
    `torchvision`'s detection and segmentation models: you've built their core.
    """)
    return


if __name__ == "__main__":
    app.run()
