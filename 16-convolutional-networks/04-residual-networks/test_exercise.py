"""Checks for exercise.py (residual networks).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Blocks are checked
against a forward pass recomputed from their own submodules, shortcut rules (identity vs 1x1
projection, none for plain blocks), the identity-at-init property (zeroed last BN scale), and
parameter counts. ResNet is checked against the contract later chapters rely on (fc, feature_dim,
forward_features), its stage structure, the ImageNet stem (whose parameter count equals
torchvision's resnet18), and a short training run under `mlfs.testing.bounded`.
"""
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional
nn = torch.nn


def _n(m):
    return sum(p.numel() for p in m.parameters())


def _block_by_hand(b, x, residual=True):
    out = b.bn2(b.conv2(F.relu(b.bn1(b.conv1(x)))))
    if residual:
        out = out + b.shortcut(x)
    return F.relu(out)


# ---------------------------------------------------------------- stage 1: BasicBlock


def test_stage1_identity_block(nb):
    torch.manual_seed(0)
    b = nb.BasicBlock(8, 8).double().eval()
    assert isinstance(b.conv1, nn.Conv2d) and b.conv1.kernel_size == (3, 3) and b.conv1.bias is None, (
        "conv1 is a 3x3 conv with padding 1 and no bias (batch norm follows)")
    assert isinstance(b.bn1, nn.BatchNorm2d) and isinstance(b.bn2, nn.BatchNorm2d)
    assert sum(p.numel() for p in b.shortcut.parameters()) == 0, (
        "same shape in and out: the shortcut is the identity (an empty nn.Sequential), no parameters")
    assert _n(b) == 2 * 8 * 8 * 9 + 4 * 8, f"two 3x3 convs and two BNs: {2 * 8 * 8 * 9 + 4 * 8} parameters; got {_n(b)}"
    x = torch.randn(2, 8, 6, 6, dtype=torch.float64)
    assert torch.allclose(b(x), _block_by_hand(b, x)), (
        "forward: relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x)); the final ReLU comes AFTER the addition")


def test_stage1_projection_shortcut(nb):
    torch.manual_seed(1)
    b = nb.BasicBlock(8, 16, stride=2).double().eval()
    x = torch.randn(2, 8, 8, 8, dtype=torch.float64)
    out = b(x)
    assert out.shape == (2, 16, 4, 4), f"stride 2 halves the resolution and the channels go 8 -> 16; got {tuple(out.shape)}"
    convs = [m for m in b.shortcut.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 1 and convs[0].kernel_size == (1, 1) and convs[0].stride == (2, 2), (
        "when the shape changes, the shortcut is a 1x1 conv with the block's stride, followed by batch norm")
    assert any(isinstance(m, nn.BatchNorm2d) for m in b.shortcut.modules())
    assert torch.allclose(out, _block_by_hand(b, x))
    b2 = nb.BasicBlock(8, 16, stride=1)
    assert any(isinstance(m, nn.Conv2d) for m in b2.shortcut.modules()), "a channel change alone also needs a projection"


def test_stage1_plain_block(nb):
    torch.manual_seed(2)
    b = nb.BasicBlock(8, 16, stride=2, residual=False).double().eval()
    assert sum(p.numel() for p in b.shortcut.parameters()) == 0, "residual=False: no shortcut at all, not even a projection"
    x = torch.randn(2, 8, 8, 8, dtype=torch.float64)
    assert torch.allclose(b(x), _block_by_hand(b, x, residual=False)), "the plain block is relu(bn2(conv2(...))), no addition"


def test_stage1_identity_at_init(nb):
    torch.manual_seed(3)
    b = nb.BasicBlock(4, 4).eval()
    nn.init.zeros_(b.bn2.weight)
    x = torch.randn(2, 4, 5, 5)
    assert torch.allclose(b(x), F.relu(x)), (
        "with bn2's scale at zero the residual branch outputs 0, so the block must compute relu(x): the identity "
        "path is what lets a residual block start out doing (almost) nothing")


# ---------------------------------------------------------------- stage 2: ResNet


def test_stage2_resnet18_contract(nb):
    torch.manual_seed(4)
    m = nb.resnet18()
    assert _n(m) == 11_173_962, (
        f"resnet18 for CIFAR (3x3 stem, no max pool, 10 classes) has 11,173,962 parameters; got {_n(m):,}")
    assert m.feature_dim == 512 and isinstance(m.fc, nn.Linear) and m.fc.in_features == 512
    x = torch.randn(2, 3, 32, 32)
    m.eval()
    feats = m.forward_features(x)
    assert feats.shape == (2, 512), f"forward_features returns pooled features (B, feature_dim); got {tuple(feats.shape)}"
    assert torch.allclose(m(x), m.fc(feats), atol=1e-6), "forward(x) = fc(forward_features(x))"
    assert torch.allclose(feats, m.layers(m.stem(x)).mean(dim=(2, 3)), atol=1e-6), (
        "forward_features = global AVERAGE pool of the last stage's map (B, 512, 4, 4), flattened to (B, 512)")
    plain = nb.resnet18(residual=False)
    assert _n(plain) == 11_000_138, (
        f"the plain resnet18 has no projection shortcuts: 11,000,138 parameters; got {_n(plain):,}")
    assert not any(getattr(b, "residual", False) for b in plain.modules() if isinstance(b, nb.BasicBlock))


def test_stage2_kaiming_init(nb):
    torch.manual_seed(7)
    conv = nb.resnet18().layers[3][1].conv2  # 512 -> 512, 3x3: fan_out = 512 * 9
    std = conv.weight.std().item()
    want = (2 / (512 * 9)) ** 0.5
    assert abs(std - want) < 0.05 * want, (
        f"convolutions are initialized kaiming-normal with mode='fan_out' and the ReLU gain: std sqrt(2 / fan_out) = "
        f"{want:.4f}; got {std:.4f} (PyTorch's default init gives about {(1 / (3 * 512 * 9)) ** 0.5:.4f})")


def test_stage2_stages_and_strides(nb):
    m = nb.ResNet((1, 2, 3), num_classes=7, base_width=8)
    assert len(m.layers) == 3, f"self.layers holds one nn.Sequential per stage; got {len(m.layers)}"
    counts = [sum(isinstance(b, nb.BasicBlock) for b in stage) for stage in m.layers]
    assert counts == [1, 2, 3], f"block_counts (1, 2, 3); got {counts}"
    widths = [stage[0].conv1.out_channels for stage in m.layers]
    assert widths == [8, 16, 32], f"widths double per stage from base_width: {widths}"
    strides = [stage[0].conv1.stride[0] for stage in m.layers]
    assert strides == [1, 2, 2], f"the first block of every stage after the first has stride 2; got {strides}"
    assert all(b.conv1.stride == (1, 1) for stage in m.layers for b in list(stage)[1:])
    assert m.feature_dim == 32 and m.fc.out_features == 7
    x = torch.randn(2, 3, 32, 32)
    assert m.layers(m.stem(x)).shape == (2, 32, 8, 8), "a 32x32 input reaches the last stage at 8x8 (two halvings)"


def test_stage2_stems(nb):
    small = nb.ResNet((1, 1), base_width=8, in_channels=1)
    convs = [x for x in small.stem.modules() if isinstance(x, nn.Conv2d)]
    assert convs[0].kernel_size == (3, 3) and convs[0].stride == (1, 1) and convs[0].in_channels == 1, (
        "small_input=True: a 3x3 stride-1 stem (CIFAR images are already small)")
    assert not any(isinstance(x, nn.MaxPool2d) for x in small.stem.modules())
    big = nb.ResNet(num_classes=1000, small_input=False)
    assert _n(big) == 11_689_512, (
        f"with the ImageNet stem (7x7/2 conv, BN, ReLU, 3x3/2 max pool) and 1000 classes, ResNet() has exactly "
        f"torchvision's resnet18 parameter count, 11,689,512; got {_n(big):,}")
    big.eval()
    assert big.stem(torch.zeros(1, 3, 64, 64)).shape == (1, 64, 16, 16), (
        "the ImageNet stem divides the resolution by 4: a 7x7 stride-2 conv (padding 3), then a 3x3 stride-2 max pool "
        "(padding 1)")
    assert big.forward_features(torch.zeros(1, 3, 64, 64)).shape == (1, 512)


def test_stage2_trains(nb):
    torch.manual_seed(5)
    g = torch.Generator().manual_seed(0)
    templates = F.avg_pool2d(torch.randn(4, 3, 16, 16, generator=g), 5, 1, 2) * 3
    y = torch.randint(0, 4, (64,), generator=g)
    x = templates[y] + 0.5 * torch.randn(64, 3, 16, 16, generator=g)
    m = nb.ResNet((1, 1), num_classes=4, base_width=8)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)

    def run():
        losses = []
        for _ in range(25):
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    losses = bounded(run, budget=20_000_000, what="training a small ResNet")
    assert losses[-1] < 0.5 * losses[0], f"25 steps on 4 easy classes should halve the loss; {losses[0]:.3f} -> {losses[-1]:.3f}"


# ---------------------------------------------------------------- stage 3: Bottleneck


def test_stage3_shapes_and_parameters(nb):
    b = nb.Bottleneck(256, 64)
    assert b.out_channels == 256
    assert _n(b) == 70_400, (
        f"1x1 256->64, 3x3 64->64, 1x1 64->256 (69,632 weights) and three BNs (768): 70,400 parameters; got {_n(b):,}. "
        f"an identity shortcut adds nothing")
    ks = [b.conv1.kernel_size, b.conv2.kernel_size, b.conv3.kernel_size]
    assert ks == [(1, 1), (3, 3), (1, 1)], f"reduce (1x1), transform (3x3), expand (1x1); got {ks}"
    b2 = nb.Bottleneck(64, 32, stride=2)
    out = b2(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 128, 4, 4) and b2.conv2.stride == (2, 2), "the stride goes in the 3x3 conv; output mid * 4 channels"
    assert any(isinstance(m, nn.Conv2d) for m in b2.shortcut.modules()), "64 -> 128 channels: projection shortcut"
    assert nb.Bottleneck(64, 32, expansion=2).out_channels == 64


def test_stage3_forward(nb):
    torch.manual_seed(6)
    b = nb.Bottleneck(16, 4).double().eval()
    x = torch.randn(2, 16, 5, 5, dtype=torch.float64)
    h = F.relu(b.bn1(b.conv1(x)))
    h = F.relu(b.bn2(b.conv2(h)))
    want = F.relu(b.bn3(b.conv3(h)) + b.shortcut(x))
    assert torch.allclose(b(x), want), "relu(bn3(conv3(relu(bn2(conv2(relu(bn1(conv1(x)))))))) + shortcut(x))"
    plain = nb.Bottleneck(16, 4, residual=False).double().eval()
    h = F.relu(plain.bn2(plain.conv2(F.relu(plain.bn1(plain.conv1(x))))))
    assert torch.allclose(plain(x), F.relu(plain.bn3(plain.conv3(h)))), "residual=False: no addition"
