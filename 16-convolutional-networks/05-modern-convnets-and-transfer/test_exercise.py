"""Checks for exercise.py (modern convnets & transfer).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Blocks are checked
against their specs (grouping, kernel sizes, parameter counts from the lesson) and a forward pass
recomputed by hand with torch.nn.functional. The transfer helpers are checked on a small backbone
with batch norm: which parameters change, which stay bit-for-bit identical, whether running
statistics move, and that the backbone/head learning rates are separate. Training runs under
`mlfs.testing.bounded`.
"""
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional
nn = torch.nn


def _n(m):
    return sum(p.numel() for p in m.parameters())


class _Backbone(nn.Module):
    """a tiny pretrained-style network: conv-BN-ReLU features, global pool, `fc` head."""

    def __init__(self, n_out=5, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                      nn.Conv2d(8, 16, 3, padding=1, stride=2), nn.BatchNorm2d(16), nn.ReLU())
        self.fc_pre = nn.Linear(16, 16)  # a backbone layer whose name starts with "fc"
        self.fc = nn.Linear(16, n_out)

    def forward(self, x):
        return self.fc(F.relu(self.fc_pre(self.features(x).mean(dim=(2, 3)))))


def _task(n=96, seed=0, k=3):
    g = torch.Generator().manual_seed(seed)
    templates = F.avg_pool2d(torch.randn(k, 3, 12, 12, generator=torch.Generator().manual_seed(99)), 3, 1, 1) * 3
    y = torch.randint(0, k, (n,), generator=g)
    x = templates[y] + 0.5 * torch.randn(n, 3, 12, 12, generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=True,
                                       generator=torch.Generator().manual_seed(seed))


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------- stage 1: depthwise separable


def test_stage1_structure_and_parameters(nb):
    b = nb.DepthwiseSeparableConv(16, 32)
    assert b.depthwise.groups == 16 and b.depthwise.kernel_size == (3, 3) and b.depthwise.out_channels == 16, (
        "the depthwise conv has one 3x3 filter per input channel: groups = in_channels = out_channels of that conv")
    assert b.depthwise.padding == (1, 1) and b.depthwise.bias is None and b.pointwise.bias is None
    assert b.pointwise.kernel_size == (1, 1) and b.pointwise.out_channels == 32
    assert _n(b) == 16 * 9 + 16 * 32 + 2 * 16 + 2 * 32, (
        f"16*9 depthwise + 16*32 pointwise weights + two BNs = {16 * 9 + 16 * 32 + 96}; got {_n(b)}. a standard 3x3 conv "
        f"16 -> 32 would have 4,608 weights")
    b5 = nb.DepthwiseSeparableConv(4, 6, kernel_size=5, stride=2)
    assert b5.depthwise.padding == (2, 2) and b5.depthwise.stride == (2, 2)
    assert b5(torch.zeros(2, 4, 9, 9)).shape == (2, 6, 5, 5), "padding k // 2 and the stride: 9x9 -> 5x5"


def test_stage1_forward(nb):
    torch.manual_seed(0)
    b = nb.DepthwiseSeparableConv(4, 6).double().eval()
    x = torch.randn(2, 4, 7, 7, dtype=torch.float64)
    h = F.relu(b.bn1(F.conv2d(x, b.depthwise.weight, padding=1, groups=4)))
    want = F.relu(b.bn2(F.conv2d(h, b.pointwise.weight)))
    assert torch.allclose(b(x), want), "relu(bn2(pointwise(relu(bn1(depthwise(x))))))"
    x2 = x.clone()
    x2[:, 1] += 5.0
    d = (b.bn1(b.depthwise(x2)) - b.bn1(b.depthwise(x))).abs().sum(dim=(0, 2, 3))
    assert d[0] == 0 and d[2] == 0 and d[1] > 0, "the depthwise stage never mixes channels: changing channel 1 only changes channel 1"


# ---------------------------------------------------------------- stage 2: ConvNeXt block


def test_stage2_structure_and_parameters(nb):
    b = nb.ConvNeXtBlock(16)
    assert b.dwconv.kernel_size == (7, 7) and b.dwconv.groups == 16 and b.dwconv.padding == (3, 3)
    assert isinstance(b.norm, nn.LayerNorm) and b.norm.normalized_shape == (16,) and b.norm.eps == 1e-6, (
        "layer norm over the channels only (LayerNorm(dim, eps=1e-6), applied channels-last)")
    assert b.pwconv1.out_features == 64 and b.pwconv2.in_features == 64, "the inverted bottleneck expands 4x"
    assert isinstance(b.gamma, nn.Parameter) and torch.allclose(b.gamma, torch.full((16,), 1e-6))
    expected = (49 * 16 + 16) + 2 * 16 + (16 * 64 + 64) + (64 * 16 + 16) + 16
    assert _n(b) == expected, f"expected {expected:,} parameters; got {_n(b):,}"


def test_stage2_forward(nb):
    torch.manual_seed(1)
    b = nb.ConvNeXtBlock(8, layer_scale_init=0.5).double()
    x = torch.randn(2, 8, 6, 5, dtype=torch.float64)
    h = F.conv2d(x, b.dwconv.weight, b.dwconv.bias, padding=3, groups=8).permute(0, 2, 3, 1)
    h = F.layer_norm(h, (8,), b.norm.weight, b.norm.bias, 1e-6)
    h = F.linear(F.gelu(F.linear(h, b.pwconv1.weight, b.pwconv1.bias)), b.pwconv2.weight, b.pwconv2.bias)
    want = x + (b.gamma * h).permute(0, 3, 1, 2)
    got = b(x)
    assert got.shape == x.shape, "the block keeps the shape (B, C, H, W)"
    assert torch.allclose(got, want), (
        "x + permute_back(gamma * pwconv2(gelu(pwconv1(norm(permute(dwconv(x))))))): normalize each pixel's channel "
        "vector, then the MLP acts on channels at every pixel")


def test_stage2_starts_near_identity(nb):
    torch.manual_seed(2)
    b = nb.ConvNeXtBlock(8)
    x = torch.randn(2, 8, 6, 6)
    assert (b(x) - x).abs().max() < 1e-3, "with layer scale 1e-6 the block starts as (almost) the identity"


# ---------------------------------------------------------------- stage 3: replace_head, freeze_backbone


def test_stage3_replace_head(nb):
    m = _Backbone().double()
    old = m.fc
    out = nb.replace_head(m, 7)
    assert out is m, "replace_head returns the same model object (modified in place)"
    assert m.fc is not old and isinstance(m.fc, nn.Linear) and m.fc.in_features == 16 and m.fc.out_features == 7
    assert m.fc.weight.dtype == torch.float64, "the new head matches the old one's dtype (and device)"
    assert m(torch.zeros(2, 3, 12, 12, dtype=torch.float64)).shape == (2, 7)

    class WithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(4, 6)
            self.head = nn.Linear(6, 2)

    w = nb.replace_head(WithHead(), 3)
    assert w.head.out_features == 3 and w.head.in_features == 6, "models with a `head` instead of `fc` work too"
    with pytest.raises(AttributeError):
        nb.replace_head(nn.Sequential(nn.Linear(2, 2)), 3)


def test_stage3_freeze_backbone(nb):
    m = _Backbone()
    out = nb.freeze_backbone(m)
    assert out is m
    trainable = sorted(n for n, p in m.named_parameters() if p.requires_grad)
    assert trainable == ["fc.bias", "fc.weight"], (
        f"only the head `fc` stays trainable; got {trainable}. careful: `fc_pre` starts with 'fc' but isn't the head "
        f"(match the submodule name exactly: 'fc' or 'fc.<...>')")
    with pytest.raises(ValueError):
        nb.freeze_backbone(_Backbone(), head_name="classifier")  # no such submodule: don't silently freeze everything
    nb.freeze_backbone(m, head_name="fc_pre")
    assert sorted(n for n, p in m.named_parameters() if p.requires_grad) == ["fc_pre.bias", "fc_pre.weight"], (
        "freeze_backbone also UNfreezes the head's parameters (requires_grad=True inside head_name)")


# ---------------------------------------------------------------- stage 4: fine_tune


def test_stage4_linear_probe_freezes_everything_but_the_head(nb):
    m = _Backbone(n_out=10)
    m.train()
    before = _snapshot(m)
    hist = bounded(nb.fine_tune, m, _task(), 3, mode="linear", epochs=4, lr=0.05, budget=20_000_000,
                   what="fine_tune")
    after = m.state_dict()
    assert len(hist["train_loss"]) == 4, "history['train_loss'] has one entry per epoch"
    assert m.fc.out_features == 3, "fine_tune starts by replacing the head (3 classes here)"
    changed = [k for k in before if not k.startswith("fc.") and not torch.equal(before[k], after[k])]
    assert not changed, (
        f"in linear mode nothing outside the head may change, including batch norm's running statistics: run the "
        f"frozen backbone in eval mode. changed: {changed}")
    assert hist["train_loss"][-1] < hist["train_loss"][0], f"the head should learn: {hist['train_loss']}"


def test_stage4_full_fine_tuning(nb):
    m = _Backbone(n_out=10)
    before = _snapshot(m)
    hist = bounded(nb.fine_tune, m, _task(seed=1), 3, mode="full", epochs=4, lr=0.01, budget=20_000_000,
                   what="fine_tune")
    after = m.state_dict()
    assert not torch.equal(before["features.0.weight"], after["features.0.weight"]), "full mode trains the backbone too"
    assert not torch.equal(before["features.1.running_mean"], after["features.1.running_mean"]), (
        "full mode runs the network in training mode, so batch norm's statistics adapt to the new data")
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_stage4_backbone_learning_rate(nb):
    m = _Backbone(n_out=10)
    before = _snapshot(m)
    bounded(nb.fine_tune, m, _task(seed=2), 3, mode="full", epochs=2, lr=0.01, backbone_lr_mult=0.0,
            budget=20_000_000, what="fine_tune")
    after = m.state_dict()
    assert torch.equal(before["features.0.weight"], after["features.0.weight"]) and torch.equal(
        before["fc_pre.weight"], after["fc_pre.weight"]), (
        "with backbone_lr_mult=0 the backbone's learning rate is 0: put the backbone and the head in separate "
        "optimizer param groups, with lr * backbone_lr_mult and lr")
    with pytest.raises(ValueError):
        nb.fine_tune(_Backbone(), _task(), 3, mode="partial", epochs=1)
