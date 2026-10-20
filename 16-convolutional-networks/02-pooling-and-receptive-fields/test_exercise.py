"""Checks for exercise.py (pooling & receptive fields).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Pooling is checked on
the lesson's hand example and against torch.nn.functional (forward and autograd backward) for
overlapping and non-overlapping windows, rectangular kernels and sizes that don't divide evenly;
receptive fields on hand-computed stacks; the effective receptive field on linear networks whose
gradients are known in closed form (a box, a triangle) and on a model's side effects (no parameter
gradients, mode restored). Loops run under `mlfs.testing.bounded`.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional
nn = torch.nn

POOLS = [  # (H, W, kernel_size, stride)
    (8, 8, 2, None),
    (7, 9, 3, 2),
    (9, 7, 3, 1),
    (6, 8, (2, 3), (2, 1)),
    (5, 5, 2, None),  # 5 isn't divisible by 2: the last row and column are dropped
]


def _x(h, w, seed, n=2, c=3):
    return np.random.default_rng(seed).normal(size=(n, c, h, w))


def _hand():
    return np.array([[[[1.0, 3.0, 2.0, 0.0], [4.0, 2.0, 1.0, 5.0], [0.0, 1.0, 3.0, 3.0], [2.0, 6.0, 1.0, 1.0]]]])


# ---------------------------------------------------------------- stage 1: pooling forward


def test_stage1_hand_example(nb):
    x = _hand()
    mp, ap = nb.max_pool2d(x, 2), nb.avg_pool2d(x, 2)
    assert mp.shape == (1, 1, 2, 2), f"2x2 pooling with the default stride (= kernel size) halves 4x4 to 2x2; got {mp.shape}"
    assert np.array_equal(mp[0, 0], [[4.0, 5.0], [6.0, 3.0]]), f"the max of each 2x2 block is [[4, 5], [6, 3]]; got {mp[0, 0]}"
    assert np.allclose(ap[0, 0], [[2.5, 2.0], [2.25, 2.0]]), f"the mean of each block is [[2.5, 2], [2.25, 2]]; got {ap[0, 0]}"
    mp3 = nb.max_pool2d(x, 3, stride=1)
    assert np.array_equal(mp3[0, 0], [[4.0, 5.0], [6.0, 6.0]]), (
        f"3x3 windows with stride 1 overlap: [[4, 5], [6, 6]]; got {mp3[0, 0]}")


@pytest.mark.parametrize("h, w, k, s", POOLS)
def test_stage1_matches_torch(nb, h, w, k, s):
    x = _x(h, w, 1)
    for name, ours, theirs in [("max_pool2d", nb.max_pool2d, F.max_pool2d), ("avg_pool2d", nb.avg_pool2d, F.avg_pool2d)]:
        got = bounded(ours, x, k, s, what=name)
        want = theirs(torch.tensor(x), k, s).numpy()
        assert got.shape == want.shape, (
            f"{name}(kernel {k}, stride {s}) on {h}x{w}: expected {want.shape}, got {got.shape}. windows that don't fit "
            f"are dropped (no padding), and stride=None means stride = kernel_size")
        assert np.allclose(got, want), f"{name}(kernel {k}, stride {s}): differs from torch by {np.abs(got - want).max():.2e}"


def test_stage1_channels_are_independent(nb):
    x = _x(4, 4, 2, n=1, c=2)
    x[0, 1] = 100.0 + x[0, 1]
    got = nb.max_pool2d(x, 2)
    assert got[0, 0].max() < 50, "pooling never mixes channels: each (image, channel) plane is pooled on its own"


# ---------------------------------------------------------------- stage 2: pooling backward


@pytest.mark.parametrize("h, w, k, s", POOLS)
def test_stage2_matches_torch_autograd(nb, h, w, k, s):
    x = _x(h, w, 3)
    for name, fwd, bwd in [("max", F.max_pool2d, nb.max_pool2d_backward), ("avg", F.avg_pool2d, nb.avg_pool2d_backward)]:
        xt = torch.tensor(x, requires_grad=True)
        out = fwd(xt, k, s)
        dout = np.random.default_rng(4).normal(size=out.shape)
        out.backward(torch.tensor(dout))
        got = bwd(dout, x if name == "max" else x.shape, k, s)
        assert got.shape == x.shape, f"{name}_pool2d_backward returns dL/dx with x's shape {x.shape}; got {got.shape}"
        assert np.allclose(got, xt.grad.numpy()), (
            f"{name} pooling (kernel {k}, stride {s}): differs from torch autograd by "
            f"{np.abs(got - xt.grad.numpy()).max():.2e}. "
            + ("each window's gradient goes to its max only; overlapping windows ADD (col2im does this)" if name == "max"
               else "each window's gradient is divided equally among its kh*kw pixels; overlaps add"))


def test_stage2_hand_example(nb):
    x = _hand()
    dx = nb.max_pool2d_backward(np.ones((1, 1, 2, 2)), x, 2)
    expected = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=float)
    assert np.array_equal(dx[0, 0], expected), (
        f"with dout = 1 everywhere, the gradient is 1 at each block's max (4, 5, 6 and the first 3) and 0 elsewhere:\n"
        f"{expected}\ngot\n{dx[0, 0]}")
    dx3 = nb.max_pool2d_backward(np.ones((1, 1, 2, 2)), x, 3, stride=1)
    assert dx3[0, 0, 3, 1] == 2.0 and dx3.sum() == 4.0, (
        "with overlapping 3x3 windows the 6 wins two windows, so it collects gradient 2: overlaps add up")
    da = nb.avg_pool2d_backward(np.ones((1, 1, 2, 2)), x.shape, 2)
    assert np.allclose(da, 0.25), "average pooling shares each window's gradient equally: 1/4 per pixel"


def test_stage2_dropped_pixels_get_zero(nb):
    x = _x(5, 5, 5)
    for got in (nb.max_pool2d_backward(np.ones((2, 3, 2, 2)), x, 2), nb.avg_pool2d_backward(np.ones((2, 3, 2, 2)), x.shape, 2)):
        assert np.all(got[:, :, 4, :] == 0) and np.all(got[:, :, :, 4] == 0), (
            "on a 5x5 input, 2x2 pooling never reads the last row and column, so their gradient is 0")


# ---------------------------------------------------------------- stage 3: receptive fields


def test_stage3_hand_examples(nb):
    assert tuple(nb.receptive_field([])) == (1, 1), "no layers: each output is one input pixel, jump 1"
    assert tuple(nb.receptive_field([(3, 1, 1)])) == (3, 1)
    assert tuple(nb.receptive_field([(3, 1, 1)] * 2)) == (5, 1), "two 3x3 convs see 5x5: each adds k - 1 = 2"
    assert tuple(nb.receptive_field([(3, 1, 1)] * 5)) == (11, 1)
    vgg = [(3, 1, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1), (3, 1, 1), (2, 2, 1)]
    got = tuple(nb.receptive_field(vgg))
    assert got == (16, 4), (
        f"conv3, conv3, pool2, conv3, conv3, pool2 -> rf 16, jump 4 (the lesson's worked example); got {got}. "
        f"after a stride-2 layer each further tap is 2 input pixels apart: rf += d (k - 1) * jump")
    assert tuple(nb.receptive_field([(3, 1, 1), (3, 1, 2), (3, 1, 4)])) == (15, 1), (
        "dilations 1, 2, 4 add 2 + 4 + 8: rf 15")
    assert tuple(nb.receptive_field([(7, 2, 1), (3, 2, 1)])) == (11, 4), "ResNet's stem: 7x7/2 then maxpool 3x3/2"


def test_stage3_order_matters(nb):
    a, b = nb.receptive_field([(2, 2, 1), (3, 1, 1)]), nb.receptive_field([(3, 1, 1), (2, 2, 1)])
    assert tuple(a) == (6, 2) and tuple(b) == (4, 2), (
        f"downsampling first makes later layers see farther: pool-then-conv gives (6, 2), conv-then-pool (4, 2); "
        f"got {a} and {b}")


def test_stage3_rf_layers(nb):
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Sequential(nn.Conv2d(4, 4, 5, stride=2, dilation=2), nn.BatchNorm2d(4)),
                          nn.AvgPool2d(3, stride=1), nn.Flatten(), nn.Linear(4, 2))
    got = [tuple(t) for t in nb.rf_layers(model)]
    assert got == [(3, 1, 1), (2, 2, 1), (5, 2, 2), (3, 1, 1)], (
        f"conv 3, maxpool 2 (stride defaults to the kernel), nested conv 5/2 dilation 2, avgpool 3/1, in order; "
        f"other modules skipped; got {got}")


# ---------------------------------------------------------------- stage 4: effective receptive field


def _ones_stack(n_layers, k=3):
    layers = []
    for _ in range(n_layers):
        conv = nn.Conv2d(1, 1, k, padding=k // 2, bias=False)
        nn.init.ones_(conv.weight)
        layers.append(conv)
    return nn.Sequential(*layers)


def test_stage4_box_and_triangle(nb):
    erf = nb.effective_receptive_field(_ones_stack(1), 9)
    assert erf.shape == (9, 9), f"one value per input pixel: (9, 9); got {erf.shape}"
    assert np.isclose(erf.sum(), 1.0), "normalize the map to sum to 1"
    expected = np.zeros((9, 9))
    expected[3:6, 3:6] = 1 / 9
    assert np.allclose(erf, expected), (
        "an all-ones 3x3 conv: the centre output depends equally on the 3x3 block around the centre (1/9 each), and "
        "not at all on other pixels")
    erf2 = nb.effective_receptive_field(_ones_stack(2), 9)
    tri = np.outer([1, 2, 3, 2, 1], [1, 2, 3, 2, 1]) / 81
    assert np.allclose(erf2[2:7, 2:7], tri) and np.isclose(erf2[2:7, 2:7].sum(), 1.0), (
        "two all-ones convs: the centre pixel is reached by 9 paths, the corners of the 5x5 field by 1 (a triangle "
        "1 2 3 2 1 per axis): the effective field is concentrated in the middle")
    signed = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    w = torch.tensor([[1.0, -2.0, 0.0], [0.5, 3.0, -1.0], [0.0, 0.0, -0.5]])
    with torch.no_grad():
        signed.weight.copy_(w.reshape(1, 1, 3, 3))
    erf3 = nb.effective_receptive_field(signed, 7)
    expected = np.zeros((7, 7))
    expected[2:5, 2:5] = np.abs(w.numpy()) / np.abs(w.numpy()).sum()
    assert np.allclose(erf3, expected), (
        "for one conv the centre output's gradient is the kernel itself, placed around the centre (cross-correlation: "
        "no flip). the ERF uses its ABSOLUTE value: negative weights matter as much as positive ones")


def test_stage4_channels_and_side_effects(nb):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 5, 3, padding=1))
    model.train()
    erf = nb.effective_receptive_field(model, (11, 13), in_channels=3)
    assert erf.shape == (11, 13) and np.isclose(erf.sum(), 1.0)
    ci, cj = 5, 6
    assert erf[ci - 2:ci + 3, cj - 2:cj + 3].sum() > 1 - 1e-9, "two 3x3 convs: nothing outside the central 5x5"
    assert model.training, "restore the model's mode afterwards (compute in eval mode, so batch norm uses running stats)"
    assert all(p.grad is None for p in model.parameters()), (
        "don't leave gradients on the model's parameters: differentiate with respect to the input only "
        "(torch.autograd.grad(..., x))")
    again = nb.effective_receptive_field(model, (11, 13), in_channels=3)
    assert np.allclose(erf, again), "fix the inputs with a seeded torch.Generator: the same call gives the same map"


def test_stage4_matches_definition(nb):
    torch.manual_seed(3)
    model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 3, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(3, 3, 3, padding=1)).double()
    got = nb.effective_receptive_field(model, 9, in_channels=2, n_samples=5, seed=7)
    x = torch.randn(5, 2, 9, 9, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    per_sample = []
    for i in range(5):  # one sample at a time: |d(sum over channels of the centre output) / d input|
        xi = x[i:i + 1].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(model(xi)[:, :, 4, 4].sum(), xi)
        per_sample.append(g.abs())
    want = torch.cat(per_sample).sum(dim=(0, 1)).numpy()
    want = want / want.sum()
    assert np.allclose(got, want, atol=1e-12), (
        "the ERF is the mean over samples of |gradient| of the centre output SUMMED OVER ALL OUTPUT CHANNELS, with the "
        "absolute value taken per sample (before averaging over samples) and then summed over input channels. draw "
        "x with torch.randn(n_samples, in_channels, H, W, generator=torch.Generator().manual_seed(seed), dtype=...)")


def test_stage4_float64_model(nb):
    model = _ones_stack(2).double()
    erf = nb.effective_receptive_field(model, 7)
    assert np.isclose(erf[3, 3], 9 / 81), "a float64 model needs float64 inputs (use the parameters' dtype)"


def test_stage4_erf_radius(nb):
    box = np.zeros((9, 9))
    box[3:6, 3:6] = 1 / 9
    assert bounded(nb.erf_radius, box, 0.9, what="erf_radius") == 1, "a 3x3 box: radius 1 holds all of it"
    assert nb.erf_radius(box, 0.1) == 0, "the centre pixel alone holds 1/9 > 0.1"
    tri = np.zeros((9, 9))
    tri[2:7, 2:7] = np.outer([1, 2, 3, 2, 1], [1, 2, 3, 2, 1]) / 81
    got = nb.erf_radius(tri, 0.9)
    assert got == 2, f"the 3x3 centre of the triangle holds 49/81 = 0.60 < 0.9, so the radius is 2; got {got}"
    assert nb.erf_radius(tri, 0.5) == 1
