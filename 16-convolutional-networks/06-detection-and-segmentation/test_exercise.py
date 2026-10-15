"""Checks for exercise.py (detection & segmentation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. box_iou and nms are
checked on the lesson's hand examples and against torchvision.ops (the oracle) on random clustered
boxes; anchors and matching on hand-computed grids; dice on closed-form values, its per-image
averaging and its gradient; the U-Net on its structure, parameter count, a forward pass recomputed
from its own submodules, and a short training run. Loops run under `mlfs.testing.bounded`, and
"vectorized" is checked by counting executed Python lines.
"""
import math

import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
tv_ops = pytest.importorskip("torchvision.ops")
F = torch.nn.functional
nn = torch.nn


def _random_boxes(n, seed, clusters=6, size=100.0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.rand(clusters, 2, generator=g) * size
    c = centres[torch.randint(0, clusters, (n,), generator=g)] + 4 * torch.randn(n, 2, generator=g)
    wh = 8 + 12 * torch.rand(n, 2, generator=g)
    return torch.cat([c - wh / 2, c + wh / 2], dim=1).double(), torch.rand(n, generator=g).double()


# ---------------------------------------------------------------- stage 1: box_iou


def test_stage1_hand_examples(nb):
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 6.0, 6.0]])
    iou = nb.box_iou(a, b)
    assert iou.shape == (2, 3), f"one IoU per pair: (N, M) = (2, 3); got {tuple(iou.shape)}"
    expected = torch.tensor([[1 / 7, 1.0, 0.0], [0.0, 0.25, 0.0]])
    assert torch.allclose(iou, expected), (
        f"(0,0,2,2) vs (1,1,3,3): intersection 1, union 4 + 4 - 1 = 7, IoU 1/7; disjoint boxes 0; a box inside "
        f"another: 1/4. expected\n{expected}\ngot\n{iou}. if disjoint boxes give a positive value, clamp the "
        f"intersection's width and height at 0")


def test_stage1_matches_torchvision(nb):
    for seed in range(3):
        b1, _ = _random_boxes(40, seed)
        b2, _ = _random_boxes(25, seed + 10)
        got, want = nb.box_iou(b1, b2), tv_ops.box_iou(b1, b2)
        assert torch.allclose(got, want), f"differs from torchvision.ops.box_iou by {(got - want).abs().max():.2e}"


def test_stage1_degenerate_boxes(nb):
    z = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    got = nb.box_iou(z, z)
    assert torch.isfinite(got).all() and got.item() == 0.0, (
        f"two zero-area boxes have an empty union: return 0, not nan (0/0); got {got.item()}")


def test_stage1_vectorized(nb):
    b1, _ = _random_boxes(300, 3)
    try:
        bounded(nb.box_iou, b1, b1, budget=3000, what="box_iou")
    except AssertionError:
        raise AssertionError("box_iou ran more than 3,000 Python lines for 300 x 300 boxes: it loops over pairs. "
                             "broadcast instead: boxes1[:, None, :2] against boxes2[None, :, :2] gives (N, M, 2)") from None


# ---------------------------------------------------------------- stage 2: nms


def test_stage2_hand_example(nb):
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0],
                          [0.0, 0.0, 10.0, 9.0]])
    scores = torch.tensor([0.9, 0.95, 0.5, 0.3])
    keep = bounded(nb.nms, boxes, scores, 0.5, what="nms")
    assert keep.dtype == torch.int64, f"return int64 indices; got {keep.dtype}"
    assert keep.tolist() == [1, 2], (
        f"box 1 (score 0.95) suppresses boxes 0 (IoU 0.68) and 3 (IoU 0.61); box 2 overlaps nothing. expected [1, 2] "
        f"(decreasing score); got {keep.tolist()}")
    assert nb.nms(boxes, scores, 0.7).tolist() == [1, 0, 2], "with threshold 0.7, box 0 (IoU 0.68 < 0.7) survives"


def test_stage2_matches_torchvision(nb):
    for seed in range(4):
        boxes, scores = _random_boxes(60, seed)
        for thr in (0.3, 0.5, 0.7):
            got = bounded(nb.nms, boxes, scores, thr, what="nms")
            want = tv_ops.nms(boxes, scores, thr)
            assert torch.equal(got, want), (
                f"seed {seed}, threshold {thr}: kept {got.tolist()[:10]}... but torchvision keeps {want.tolist()[:10]}.... "
                f"visit boxes in decreasing score order; skip suppressed ones; a kept box suppresses every box with "
                f"IoU > threshold")


def test_stage2_strict_threshold_and_empty(nb):
    boxes = torch.tensor([[0.0, 0.0, 2.0, 1.0], [1.0, 0.0, 3.0, 1.0], [0.0, 0.0, 3.0, 1.0]], dtype=torch.float64)
    scores = torch.tensor([0.9, 0.8, 0.1], dtype=torch.float64)
    iou01 = nb.box_iou(boxes[:1], boxes[1:2]).item()
    assert math.isclose(iou01, 1 / 3)
    assert nb.nms(boxes[:2], scores[:2], 1 / 3).tolist() == [0, 1], (
        "suppress only when IoU is strictly GREATER than the threshold (IoU = 1/3 at threshold 1/3 keeps both)")
    empty = nb.nms(torch.zeros(0, 4), torch.zeros(0), 0.5)
    assert empty.dtype == torch.int64 and empty.numel() == 0, "no boxes: an empty int64 tensor"
    accel = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if accel is not None:  # only checkable on a machine with a GPU
        keep = nb.nms(boxes.float().to(accel), scores.float().to(accel), 0.5)
        assert keep.device.type == accel, f"return the indices on the boxes' device ({accel}); got {keep.device}"


# ---------------------------------------------------------------- stage 3: anchors


def test_stage3_make_anchors(nb):
    a = nb.make_anchors(2, 16, (32,), (0.5, 1.0, 2.0))
    assert a.shape == (12, 4), f"2 x 2 positions x 1 size x 3 ratios = 12 anchors; got {tuple(a.shape)}"
    w, h = 32 / math.sqrt(0.5), 32 * math.sqrt(0.5)
    assert torch.allclose(a[0], torch.tensor([8 - w / 2, 8 - h / 2, 8 + w / 2, 8 + h / 2])), (
        f"anchor 0: centre ((0 + 0.5) * 16, (0 + 0.5) * 16) = (8, 8), ratio = height / width = 0.5 with area 32^2: "
        f"width 32/sqrt(0.5) = {w:.2f}, height {h:.2f}; got {a[0].tolist()}")
    assert torch.allclose(a[1], torch.tensor([-8.0, -8.0, 24.0, 24.0])), "anchor 1: ratio 1, a 32 x 32 square at (8, 8)"
    assert torch.allclose(a[3, :2] + a[3, 2:], torch.tensor([48.0, 16.0])), (
        "anchors 3-5 sit at the next COLUMN, centre (24, 8): order is row, column, size, ratio")
    areas = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    assert torch.allclose(areas, torch.full((12,), 1024.0)), "every ratio keeps the area size^2"
    b = nb.make_anchors((1, 3), 8, (8, 16), (1.0,))
    assert b.shape == (6, 4) and torch.allclose(b[1], torch.tensor([-4.0, -4.0, 12.0, 12.0]))


def test_stage3_match_anchors(nb):
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [0.0, 0.0, 4.0, 4.0],
                            [40.0, 40.0, 50.0, 50.0], [0.0, 0.0, 10.0, 7.5]])
    gts = torch.tensor([[0.0, 0.0, 10.0, 10.0], [41.0, 41.0, 53.0, 53.0]])
    labels, matched = nb.match_anchors(anchors, gts, pos_threshold=0.5, neg_threshold=0.3)
    # IoUs with gt 0: 1, 1/3, 0.16, 0, 0.75; anchor 3 vs gt 1: 81/163 = 0.497 (below 0.5, but gt 1's best anchor)
    assert labels.tolist() == [1, -1, 0, 1, 1], (
        f"labels: IoU >= 0.5 -> 1, < 0.3 -> 0, in between -> -1, and every object's best anchor is forced to 1 "
        f"(anchor 3 has IoU 0.497 with object 1, its best); got {labels.tolist()}")
    assert matched[0].item() == 0 and matched[3].item() == 1 and matched[4].item() == 0, (
        f"matched gives the index of each anchor's best object; got {matched.tolist()}")
    # object 1's best anchor (a1, IoU 0.25) overlaps object 0 more (IoU 0.64): it's forced positive, matched to OBJECT 1
    anchors2 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 8.0, 8.0], [0.0, 5.0, 10.0, 10.0]])
    gts2 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 4.0, 4.0]])
    labels2, matched2 = nb.match_anchors(anchors2, gts2, pos_threshold=0.5, neg_threshold=0.3)
    assert matched2[1].item() == 1, (
        f"anchor 1 is object 1's best anchor (IoU 0.25), so it is forced positive and must be matched to object 1, the "
        f"object that forced it, even though it overlaps object 0 more (IoU 0.64); got matched {matched2.tolist()}")
    assert labels2.tolist() == [1, 1, 1] and matched2[2].item() == 0, (
        f"anchor 2 has IoU exactly 0.5 with object 0: IoU >= pos_threshold is positive; got labels {labels2.tolist()}")
    lab0, m0 = nb.match_anchors(anchors, torch.zeros(0, 4))
    assert lab0.tolist() == [0] * 5 and m0.tolist() == [-1] * 5, "no objects: every anchor is background, matched -1"


# ---------------------------------------------------------------- stage 4: dice loss


def test_stage4_closed_form(nb):
    t = torch.zeros(1, 1, 4, 4)
    t[0, 0, :2] = 1.0  # 8 of 16 pixels
    got = nb.dice_loss(torch.zeros(1, 1, 4, 4), t)
    want = 1 - (2 * 4 + 1) / (8 + 8 + 1)
    assert math.isclose(got.item(), want, rel_tol=1e-6), (
        f"logits 0 -> p = 0.5 everywhere: dice = (2 * 4 + 1) / (8 + 8 + 1), loss {want:.4f}; got {got.item():.4f}")
    perfect = nb.dice_loss(20 * (2 * t - 1), t)
    assert perfect.item() < 1e-3, "confident, correct logits give a loss near 0"
    empty = nb.dice_loss(torch.full((1, 1, 4, 4), -20.0), torch.zeros(1, 1, 4, 4))
    assert empty.item() < 1e-3, "an empty mask predicted empty: the eps terms make dice = eps / eps = 1, loss 0"
    assert math.isclose(nb.dice_loss(torch.zeros(1, 1, 4, 4), t, eps=0.0).item(), 1 - 8 / 16, rel_tol=1e-6)


def test_stage4_per_image_mean_and_gradient(nb):
    t = torch.zeros(2, 1, 8, 8)
    t[0, 0, :4, :4] = 1.0  # a big object
    t[1, 0, 0, 0] = 1.0  # a one-pixel object
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    loss = nb.dice_loss(logits, t)
    d0 = (2 * 8 + 1) / (32 + 16 + 1)
    d1 = (2 * 0.5 + 1) / (32 + 1 + 1)
    assert math.isclose(loss.item(), 1 - (d0 + d1) / 2, rel_tol=1e-6), (
        f"Dice is computed per image, then averaged over the batch: expected {1 - (d0 + d1) / 2:.4f}, got "
        f"{loss.item():.4f}. pooling the sums over the whole batch lets big objects drown small ones")
    loss.backward()
    assert logits.grad is not None and logits.grad[1, 0, 0, 0] < 0, (
        "the gradient raises the logit of the one-pixel object's pixel (keep everything in torch, differentiable)")


# ---------------------------------------------------------------- stage 5: U-Net


def test_stage5_structure_and_parameters(nb):
    dc = nb.DoubleConv(3, 4)
    kinds = [type(m).__name__ for m in dc.net]
    assert kinds == ["Conv2d", "BatchNorm2d", "ReLU"] * 2, f"DoubleConv: (conv3x3 -> BN -> ReLU) twice; got {kinds}"
    m = nb.UNet(1, 2, base_channels=4, depth=2)
    assert len(m.downs) == 2 and len(m.ups) == 2 and len(m.up_convs) == 2
    assert all(isinstance(u, nn.ConvTranspose2d) and u.kernel_size == (2, 2) and u.stride == (2, 2) for u in m.ups), (
        "upsampling with 2x2 transposed convolutions of stride 2")
    widths = [d.net[0].out_channels for d in m.downs] + [m.bottleneck.net[0].out_channels]
    assert widths == [4, 8, 16], f"encoder widths base * 2^i, bottleneck base * 2^depth; got {widths}"
    assert [c.net[0].in_channels for c in m.up_convs] == [16, 8], (
        "each decoder DoubleConv takes the upsampled map concatenated with the skip: 2 x the level's width")
    assert isinstance(m.out_conv, nn.Conv2d) and m.out_conv.kernel_size == (1, 1) and m.out_conv.out_channels == 2

    def dcp(i, o):
        return i * o * 9 + o * o * 9 + 4 * o

    expected = (dcp(1, 4) + dcp(4, 8) + dcp(8, 16) + (16 * 8 * 4 + 8) + dcp(16, 8) + (8 * 4 * 4 + 4) + dcp(8, 4)
                + (4 * 2 + 2))
    got = sum(p.numel() for p in m.parameters())
    assert got == expected, f"expected {expected:,} parameters for UNet(1, 2, base_channels=4, depth=2); got {got:,}"


def test_stage5_forward_and_sizes(nb):
    torch.manual_seed(0)
    m = nb.UNet(1, 1, base_channels=4, depth=2).double().eval()
    x = torch.randn(2, 1, 12, 16, dtype=torch.float64)
    s0 = m.downs[0](x)
    s1 = m.downs[1](F.max_pool2d(s0, 2))
    h = m.bottleneck(F.max_pool2d(s1, 2))
    h = m.up_convs[0](torch.cat([s1, m.ups[0](h)], dim=1))
    h = m.up_convs[1](torch.cat([s0, m.ups[1](h)], dim=1))
    want = m.out_conv(h)
    got = m(x)
    assert got.shape == (2, 1, 12, 16), f"logits at the input resolution (B, out_channels, H, W); got {tuple(got.shape)}"
    assert torch.allclose(got, want), (
        "forward: DoubleConv + 2x2 max pool per level (keep each DoubleConv output as a skip), bottleneck, then per "
        "level: transposed conv up, concatenate [skip, upsampled] on channels, DoubleConv; finally the 1x1 conv")
    with pytest.raises(ValueError):
        m(torch.zeros(1, 1, 14, 16, dtype=torch.float64))
    assert nb.UNet()(torch.zeros(1, 1, 32, 32)).shape == (1, 1, 32, 32), "defaults: 1 -> 1 channel, base 32, depth 3"


def test_stage5_learns_to_segment(nb):
    torch.manual_seed(1)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 1, 16, 16, generator=g) * 0.3
    t = torch.zeros(8, 1, 16, 16)
    for i in range(8):
        r, c = torch.randint(0, 10, (2,), generator=g).tolist()
        t[i, 0, r:r + 6, c:c + 6] = 1.0
    x = x + 2 * t
    m = nb.UNet(1, 1, base_channels=8, depth=2)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)

    def run():
        losses = []
        for _ in range(60):
            opt.zero_grad()
            out = m(x)
            loss = F.binary_cross_entropy_with_logits(out, t) + nb.dice_loss(out, t)
            loss.backward()
            opt.step()
            losses.append(nb.dice_loss(out.detach(), t).item())
        return losses

    losses = bounded(run, budget=20_000_000, what="training the U-Net")
    assert losses[-1] < 0.25, f"60 Adam steps on bright squares should reach a dice loss below 0.25; got {losses[-1]:.3f}"
