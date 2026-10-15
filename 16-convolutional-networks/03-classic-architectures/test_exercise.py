"""Checks for exercise.py (classic architectures).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The networks are
checked against their specs: layer names, parameter counts (the lesson's worked example), output
shapes for several input sizes, and a forward pass recomputed by hand from the model's own weights
with torch.nn.functional (so padding, pooling and activation choices are pinned down). Short
training runs on synthetic data, under `mlfs.testing.bounded`, check that gradients flow. The
profiler is checked on LeNet's hand-computed table, a grouped convolution and its side effects.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional
nn = torch.nn


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


def _train_steps(model, x, y, steps=30, lr=0.05):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def _blobs(n, c, h, seed=0, k=4):
    g = torch.Generator().manual_seed(1234)
    templates = torch.randn(k, c, h, h, generator=g)
    templates = F.avg_pool2d(templates, 5, 1, 2) * 3
    g2 = torch.Generator().manual_seed(seed)
    y = torch.randint(0, k, (n,), generator=g2)
    return templates[y] + 0.5 * torch.randn(n, c, h, h, generator=g2), y


# ---------------------------------------------------------------- stage 1: LeNet


def test_stage1_parameters_and_names(nb):
    torch.manual_seed(0)
    m = nb.LeNet()
    names = [n for n, _ in m.named_children()]
    assert names == ["conv1", "conv2", "fc1", "fc2", "fc3"], f"submodules conv1, conv2, fc1, fc2, fc3 in order; got {names}"
    assert _n_params(m) == 61706, (
        f"LeNet-5 has 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706 parameters; got {_n_params(m):,}. "
        f"fc1 takes 16 x 5 x 5 = 400 inputs")
    assert _n_params(nb.LeNet(num_classes=3, in_channels=3)) == 61706 + 2 * 150 - 7 * 84 - 7


def test_stage1_matches_the_spec(nb):
    torch.manual_seed(1)
    m = nb.LeNet().double()
    x = torch.randn(2, 1, 28, 28, dtype=torch.float64)
    h = F.pad(x, (2, 2, 2, 2))
    h = F.max_pool2d(F.relu(F.conv2d(h, m.conv1.weight, m.conv1.bias)), 2)
    h = F.max_pool2d(F.relu(F.conv2d(h, m.conv2.weight, m.conv2.bias)), 2)
    h = F.relu(F.linear(h.flatten(1), m.fc1.weight, m.fc1.bias))
    h = F.relu(F.linear(h, m.fc2.weight, m.fc2.bias))
    want = F.linear(h, m.fc3.weight, m.fc3.bias)
    got = m(x)
    assert got.shape == (2, 10), f"logits of shape (B, num_classes); got {tuple(got.shape)}"
    assert torch.allclose(got, want), (
        "LeNet's forward: zero-pad 28x28 to 32x32, conv1 (5x5, no padding) -> ReLU -> 2x2 max pool, conv2 -> ReLU -> "
        "max pool, flatten, fc1 -> ReLU, fc2 -> ReLU, fc3 (no activation on the logits)")


def test_stage1_takes_32x32_colour_images(nb):
    m = nb.LeNet(num_classes=10, in_channels=3)
    out = m(torch.zeros(4, 3, 32, 32))
    assert out.shape == (4, 10), (
        "a 32x32 input needs no padding (LeNet-5 was designed for 32x32): pad only inputs smaller than 32. if you used "
        "padding=2 inside conv1 instead, 32x32 inputs give 6x6 maps and fc1 breaks")


def test_stage1_learns(nb):
    ratios = []
    for seed in range(3):  # median over 3 runs, so one unlucky initialization can't fail a correct LeNet
        torch.manual_seed(2 + seed)
        x, y = _blobs(32, 1, 28, seed=seed)
        m = nb.LeNet(num_classes=4)
        losses = bounded(_train_steps, m, x, y, steps=30, budget=20_000_000, what="training LeNet")
        ratios.append(losses[-1] / losses[0])
    assert sorted(ratios)[1] < 0.5, (
        f"30 SGD steps on 4 easy synthetic classes should halve the loss (median of 3 runs); final/initial loss "
        f"ratios {[round(r, 3) for r in ratios]}")


# ---------------------------------------------------------------- stage 2: MiniVGG


def test_stage2_structure(nb):
    m = nb.MiniVGG(num_classes=5, in_channels=3, widths=(8, 16))
    kinds = [type(x).__name__ for x in m.features]
    assert kinds == ["Conv2d", "BatchNorm2d", "ReLU", "Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d"] * 2, (
        f"per width: conv -> BN -> ReLU, twice, then a max pool; got {kinds}")
    convs = [x for x in m.features if isinstance(x, nn.Conv2d)]
    assert all(c.kernel_size == (3, 3) and c.padding == (1, 1) for c in convs), "3x3 convolutions with padding 1"
    assert all(c.bias is None for c in convs), "with batch norm the conv bias is redundant (BN's beta replaces it): bias=False"
    assert isinstance(m.pool, nn.AdaptiveAvgPool2d) and isinstance(m.fc, nn.Linear) and m.fc.in_features == 16
    expected = (3 * 8 * 9 + 2 * 8) + (8 * 8 * 9 + 2 * 8) + (8 * 16 * 9 + 2 * 16) + (16 * 16 * 9 + 2 * 16) + (16 * 5 + 5)
    assert _n_params(m) == expected, f"expected {expected:,} parameters, got {_n_params(m):,}"


def test_stage2_without_batch_norm_and_default(nb):
    plain = nb.MiniVGG(widths=(8,), batch_norm=False)
    assert not any(isinstance(x, nn.BatchNorm2d) for x in plain.modules())
    assert all(x.bias is not None for x in plain.modules() if isinstance(x, nn.Conv2d)), "no BN: the convs keep their bias"
    default = nb.MiniVGG()
    widths = [x.out_channels for x in default.features if isinstance(x, nn.Conv2d)]
    assert widths == [32, 32, 64, 64, 128, 128], f"default widths (32, 64, 128), two convs each; got {widths}"
    assert sum(isinstance(x, nn.MaxPool2d) for x in default.features) == 3


def test_stage2_forward_and_any_input_size(nb):
    torch.manual_seed(3)
    m = nb.MiniVGG(num_classes=4, widths=(8, 16)).double().eval()
    x = torch.randn(3, 3, 32, 32, dtype=torch.float64)
    want = m.fc(m.features(x).mean(dim=(2, 3)))
    assert torch.allclose(m(x), want), "forward = fc(global average of the last feature map)"
    assert m(torch.randn(2, 3, 16, 16, dtype=torch.float64)).shape == (2, 4), (
        "global average pooling makes the head independent of the input size (16x16 works as well as 32x32)")


def test_stage2_learns(nb):
    torch.manual_seed(4)
    x, y = _blobs(64, 3, 16)
    m = nb.MiniVGG(num_classes=4, widths=(8, 16))
    losses = bounded(_train_steps, m, x, y, budget=20_000_000, what="training MiniVGG")
    assert losses[-1] < 0.5 * losses[0], f"30 steps should halve the loss; went {losses[0]:.3f} -> {losses[-1]:.3f}"


# ---------------------------------------------------------------- stage 3: profile_model


def test_stage3_lenet_table(nb):
    rows = nb.profile_model(nb.LeNet(), (1, 28, 28))
    assert [r["name"] for r in rows] == ["conv1", "conv2", "fc1", "fc2", "fc3"], (
        f"one row per Conv2d/Linear, named as in model.named_modules(), in forward order; got {[r['name'] for r in rows]}")
    shapes = [tuple(r["out_shape"]) for r in rows]
    assert shapes == [(6, 28, 28), (16, 10, 10), (120,), (84,), (10,)], (
        f"output shapes without the batch dimension; got {shapes}")
    assert [r["params"] for r in rows] == [156, 2416, 48120, 10164, 850]
    macs = [r["macs"] for r in rows]
    assert macs == [117600, 240000, 48000, 10080, 840], (
        f"MACs = output entries x inputs per output: conv1 6*28*28*25 = 117,600, conv2 16*10*10*150 = 240,000, "
        f"fc1 120*400 = 48,000 ...; got {macs}")


def test_stage3_grouped_conv_and_side_effects(nb):
    m = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.Conv2d(8, 16, 1), nn.BatchNorm2d(16), nn.Flatten(),
                      nn.Linear(16 * 6 * 6, 3))
    m.train()
    rows = nb.profile_model(m, (8, 6, 6))
    assert [r["name"] for r in rows] == ["0", "1", "4"]
    assert rows[0]["macs"] == 8 * 36 * 9, (
        f"a grouped conv: each output reads in_channels / groups channels: 8 * 36 * 1 * 9 = 2,592 MACs; got {rows[0]['macs']}")
    assert rows[1]["macs"] == 16 * 36 * 8 and rows[2]["macs"] == 3 * 576
    assert rows[0]["params"] == 8 * 9 + 8
    assert m.training, "restore the model's mode"
    assert all(len(x._forward_hooks) == 0 for x in m.modules()), "remove your forward hooks afterwards"
    assert m[2].num_batches_tracked.item() == 0, "profile in eval mode under no_grad: running statistics must not move"
