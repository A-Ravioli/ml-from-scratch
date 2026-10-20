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
    # detection & segmentation · build (reference solution)

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
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=torch.finfo(inter.dtype).tiny), torch.zeros_like(inter))


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
    order = torch.argsort(scores, descending=True, stable=True)
    iou = box_iou(boxes, boxes)
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    keep = []
    for i in order.tolist():
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_threshold
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


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
    h, w = (feature_size, feature_size) if isinstance(feature_size, int) else feature_size
    shapes = []
    for s in sizes:
        for r in ratios:
            shapes.append((s / math.sqrt(r), s * math.sqrt(r)))  # (width, height), area s^2
    wh = torch.tensor(shapes, dtype=torch.float32)  # (A, 2)
    cy, cx = torch.meshgrid((torch.arange(h, dtype=torch.float32) + 0.5) * stride,
                            (torch.arange(w, dtype=torch.float32) + 0.5) * stride, indexing="ij")
    centres = torch.stack([cx, cy], dim=-1).reshape(-1, 1, 2)  # (H*W, 1, 2)
    return torch.cat([centres - wh / 2, centres + wh / 2], dim=-1).reshape(-1, 4)


@app.function
def match_anchors(anchors, gt_boxes, *, pos_threshold=0.5, neg_threshold=0.4):
    """stage 3: (labels, matched): labels 1 (IoU >= pos), 0 (< neg), -1 (in between); every object's best anchor is positive; matched = each anchor's best object, or for a forced anchor the object that forced it."""
    labels = torch.zeros(len(anchors), dtype=torch.int64)
    matched = torch.full((len(anchors),), -1, dtype=torch.int64)
    if len(gt_boxes) == 0:
        return labels, matched
    iou = box_iou(anchors, gt_boxes)  # (A, G)
    best_iou, matched = iou.max(dim=1)
    labels[(best_iou >= neg_threshold) & (best_iou < pos_threshold)] = -1
    labels[best_iou >= pos_threshold] = 1
    best_anchor = iou.argmax(dim=0)  # (G,): each object's best anchor
    labels[best_anchor] = 1
    matched[best_anchor] = torch.arange(len(gt_boxes))
    return labels, matched


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
    p = torch.sigmoid(logits).flatten(1)
    t = targets.to(p.dtype).flatten(1)
    dice = (2 * (p * t).sum(dim=1) + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)
    return 1 - dice.mean()


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
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(),
                                 nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        """stage 5: self.net(x)."""
        return self.net(x)


@app.class_definition
class UNet(nn.Module):
    """encoder (DoubleConv + maxpool, channels doubling) -> bottleneck -> decoder (2x2 transposed conv, concat skip, DoubleConv) -> 1x1 conv."""

    def __init__(self, in_channels=1, out_channels=1, *, base_channels=32, depth=3):
        """stage 5: downs (depth DoubleConvs), bottleneck, ups (depth ConvTranspose2d k=2 s=2), up_convs (depth DoubleConvs), out_conv (1x1)."""
        super().__init__()
        self.depth = depth
        widths = [base_channels * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c_in = in_channels
        for w in widths[:-1]:
            self.downs.append(DoubleConv(c_in, w))
            c_in = w
        self.bottleneck = DoubleConv(widths[-2], widths[-1])
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2))
            self.up_convs.append(DoubleConv(2 * widths[i], widths[i]))
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        """stage 5: logits (B, out_channels, H, W); H and W must be divisible by 2**depth (ValueError otherwise)."""
        h, w = x.shape[-2:]
        if h % 2 ** self.depth or w % 2 ** self.depth:
            raise ValueError(f"UNet with depth {self.depth} needs H and W divisible by {2 ** self.depth}, got {h}x{w}")
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = conv(torch.cat([skip, up(x)], dim=1))
        return self.out_conv(x)


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
