"""Checks for exercise.py (convolution).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Forward passes are
checked on the lesson's hand example and against torch.nn.functional.conv2d (the oracle) over a
grid of strides, paddings, dilations and rectangular kernels; im2col's row order and its adjoint
identity <im2col(x), C> = <x, col2im(C)>; the backward pass against torch autograd and finite
differences. "vectorized" is checked by counting executed Python lines with
`mlfs.testing.bounded` (a loop over output pixels runs thousands of lines; a loop over the kernel's
offsets runs a few dozen), which doesn't depend on the machine's speed.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional

CASES = [  # (N, C_in, H, W, C_out, kh, kw, stride, padding, dilation)
    (2, 3, 7, 7, 4, 3, 3, 1, 0, 1),
    (2, 3, 8, 9, 4, 3, 3, 2, 1, 1),
    (1, 2, 9, 8, 3, 3, 2, 1, 2, 1),
    (2, 1, 11, 10, 2, 3, 3, 1, 0, 2),
    (1, 2, 10, 12, 3, 5, 3, (2, 1), (1, 2), (1, 2)),
]


def _case(case, seed):
    n, c, h, w_, o, kh, kw, s, p, d = case
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, c, h, w_)), rng.normal(size=(o, c, kh, kw)), rng.normal(size=o), dict(
        stride=s, padding=p, dilation=d)


def _torch_conv(x, w, b, kw):
    return F.conv2d(torch.tensor(x), torch.tensor(w), None if b is None else torch.tensor(b), **kw).numpy()


def _hand_example():
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    return x, w


def _line_budget(fn, *args, budget, what, **kwargs):
    try:
        return bounded(fn, *args, budget=budget, what=what, **kwargs)
    except AssertionError:
        raise AssertionError(
            f"{what} ran more than {budget:,} Python lines on a 2 x 3 x 96 x 96 input: it loops over output pixels "
            f"(96 x 96 = 9,216 of them). loop over the kernel's kh x kw offsets instead, and move a whole "
            f"(N, C, H_out, W_out) slice per offset with strided slicing.") from None


# ---------------------------------------------------------------- stage 1: definition


def test_stage1_output_size(nb):
    assert nb.conv_output_size(32, 3) == 30, "no padding: a 3-wide kernel fits 30 times in 32 pixels"
    assert nb.conv_output_size(32, 3, padding=1) == 32, "'same' padding p = (k-1)/2 keeps the size"
    assert nb.conv_output_size(32, 3, stride=2, padding=1) == 16
    assert nb.conv_output_size(7, 3, stride=2) == 3, "floor((7 - 3)/2) + 1 = 3"
    assert nb.conv_output_size(8, 3, stride=2) == 3, (
        f"floor((8 - 3)/2) + 1 = 3 (the last pixel is never a window start); got {nb.conv_output_size(8, 3, stride=2)}")
    assert nb.conv_output_size(10, 3, dilation=2) == 6, (
        "a dilated 3-tap kernel with d = 2 spans d(k-1)+1 = 5 pixels, so 10 - 5 + 1 = 6 outputs")
    assert nb.conv_output_size(28, 5, padding=2) == 28


def test_stage1_hand_example(nb):
    x, w = _hand_example()
    out = nb.conv2d_naive(x, w)
    assert out.shape == (1, 1, 3, 3), f"a 2x2 kernel on a 4x4 image gives 3x3 outputs; got {out.shape}"
    assert np.allclose(out, -5.0), (
        f"every output is x[r, c] - x[r+1, c+1] = -5 for this image; got {out[0, 0]}. if you got +5, you flipped the "
        f"kernel (true convolution): deep learning's 'convolution' is cross-correlation, no flip")
    out_b = nb.conv2d_naive(x, w, np.array([2.0]), stride=2, padding=1)
    expected = np.array([[[[2.0, 0.0, 2.0], [-6.0, -3.0, 9.0], [2.0, 15.0, 17.0]]]])
    assert np.allclose(out_b, expected), (
        f"with padding 1, stride 2 and bias 2 the lesson's worked example gives\n{expected[0, 0]}\ngot\n{out_b[0, 0]}")


@pytest.mark.parametrize("case", CASES)
def test_stage1_matches_torch(nb, case):
    x, w, b, kw = _case(case, 0)
    got = bounded(nb.conv2d_naive, x, w, b, what="conv2d_naive", **kw)
    want = _torch_conv(x, w, b, kw)
    assert got.shape == want.shape, f"{kw}: expected shape {want.shape}, got {got.shape}"
    assert np.allclose(got, want, atol=1e-10), (
        f"{kw}: differs from F.conv2d by {np.abs(got - want).max():.2e}. check where each window starts "
        f"(i*stride), how far apart its taps are (dilation), and that the padding is zeros on both sides")


def test_stage1_no_bias_and_channel_check(nb):
    x, w, _, kw = _case(CASES[0], 1)
    assert np.allclose(nb.conv2d_naive(x, w), _torch_conv(x, w, None, {})), "b=None means no bias"
    with pytest.raises(ValueError):
        nb.conv2d_naive(x, w[:, :2])


# ---------------------------------------------------------------- stage 2: im2col / col2im


def test_stage2_im2col_layout(nb):
    x, _ = _hand_example()
    cols = nb.im2col(x, 2)
    assert cols.shape == (1, 4, 9), f"(N, C*kh*kw, H_out*W_out) = (1, 4, 9); got {cols.shape}"
    assert np.array_equal(cols[0, :, 0], [0.0, 1.0, 4.0, 5.0]), (
        f"column 0 is the top-left window, read row by row: (0, 1, 4, 5); got {cols[0, :, 0]}")
    assert np.array_equal(cols[0, 0], [0, 1, 2, 4, 5, 6, 8, 9, 10]), (
        "row 0 holds the top-left tap of every window, output pixels in row-major order")
    x2 = np.random.default_rng(2).normal(size=(2, 3, 5, 6))
    c2 = nb.im2col(x2, (2, 3), padding=1)
    assert c2.shape == (2, 18, 6 * 6)
    assert np.allclose(c2[1, 6 + 1 * 3 + 2, 0], x2[1, 1, 0, 1]), (
        "row c*kh*kw + i*kw + j of column 0 is tap (i, j) of channel c in the first window (padding included)")


@pytest.mark.parametrize("case", CASES)
def test_stage2_im2col_matches_torch_unfold(nb, case):
    x, w, _, kw = _case(case, 3)
    ks = w.shape[2:]
    got = nb.im2col(x, ks, **kw)
    want = F.unfold(torch.tensor(x), ks, **kw).numpy()
    assert got.shape == want.shape and np.allclose(got, want), (
        f"{kw}, kernel {ks}: im2col should match torch.nn.functional.unfold (same layout); "
        f"shapes {got.shape} vs {want.shape}")


@pytest.mark.parametrize("case", CASES)
def test_stage2_col2im_is_the_adjoint(nb, case):
    x, w, _, kw = _case(case, 4)
    ks = w.shape[2:]
    cols = nb.im2col(x, ks, **kw)
    c = np.random.default_rng(5).normal(size=cols.shape)
    back = nb.col2im(c, x.shape, ks, **kw)
    assert back.shape == x.shape, f"col2im returns the image shape {x.shape}; got {back.shape}"
    lhs, rhs = np.sum(cols * c), np.sum(x * back)
    assert np.isclose(lhs, rhs), (
        f"{kw}: <im2col(x), C> = {lhs:.6f} but <x, col2im(C)> = {rhs:.6f}. col2im must ADD each column entry back "
        f"to the pixel it came from (overlapping windows sum), and drop the padding at the end")


def test_stage2_col2im_counts_overlaps(nb):
    ones = np.ones((1, 1, 4, 4))
    counts = nb.col2im(nb.im2col(ones, 3, padding=1), ones.shape, 3, padding=1)
    expected = np.array([[4, 6, 6, 4], [6, 9, 9, 6], [6, 9, 9, 6], [4, 6, 6, 4]], dtype=float)
    assert np.array_equal(counts[0, 0], expected), (
        f"col2im(im2col(ones)) counts how many 3x3 windows cover each pixel (9 inside, 4 in the corners); got\n"
        f"{counts[0, 0]}\nif you assigned instead of adding, overlaps were overwritten")


def test_stage2_vectorized(nb):
    x = np.random.default_rng(6).normal(size=(2, 3, 96, 96))
    cols = _line_budget(nb.im2col, x, 3, padding=1, budget=3000, what="im2col")
    _line_budget(nb.col2im, cols, x.shape, 3, padding=1, budget=3000, what="col2im")


# ---------------------------------------------------------------- stage 3: conv2d


@pytest.mark.parametrize("case", CASES)
def test_stage3_matches_torch(nb, case):
    x, w, b, kw = _case(case, 7)
    got = nb.conv2d(x, w, b, **kw)
    want = _torch_conv(x, w, b, kw)
    assert got.shape == want.shape, f"{kw}: expected {want.shape}, got {got.shape}"
    assert np.allclose(got, want, atol=1e-10), (
        f"{kw}: differs from F.conv2d by {np.abs(got - want).max():.2e}. the weight matrix is w.reshape(C_out, -1), "
        f"whose column order (c, i, j) must match im2col's row order")


def test_stage3_matches_naive_and_hand_example(nb):
    x, w = _hand_example()
    assert np.allclose(nb.conv2d(x, w), -5.0)
    x, w, b, kw = _case(CASES[1], 8)
    assert np.allclose(nb.conv2d(x, w, b, **kw), nb.conv2d_naive(x, w, b, **kw), atol=1e-10)
    assert np.allclose(nb.conv2d(x, w), _torch_conv(x, w, None, {}), atol=1e-10), "b=None means no bias"


def test_stage3_vectorized(nb):
    rng = np.random.default_rng(9)
    x, w = rng.normal(size=(2, 3, 96, 96)), rng.normal(size=(4, 3, 3, 3))
    out = _line_budget(nb.conv2d, x, w, None, padding=1, budget=3000, what="conv2d")
    assert out.shape == (2, 4, 96, 96)


# ---------------------------------------------------------------- stage 4: backward


@pytest.mark.parametrize("case", CASES)
def test_stage4_matches_torch_autograd(nb, case):
    x, w, b, kw = _case(case, 10)
    xt, wt, bt = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
    out = F.conv2d(xt, wt, bt, **kw)
    dout = np.random.default_rng(11).normal(size=out.shape)
    out.backward(torch.tensor(dout))
    dx, dw, db = nb.conv2d_backward(dout, x, w, **kw)
    for name, got, want in [("dx", dx, xt.grad), ("dw", dw, wt.grad), ("db", db, bt.grad)]:
        want = want.numpy()
        assert got.shape == want.shape, f"{kw}: {name} should have shape {want.shape}, got {got.shape}"
        assert np.allclose(got, want, atol=1e-9), (
            f"{kw}: {name} differs from torch autograd by {np.abs(got - want).max():.2e}. "
            + {"dx": "dx = col2im(W^T dout): the columns' gradients summed back to their pixels",
               "dw": "dw sums dout x im2col(x)^T over the batch and output positions",
               "db": "db sums dout over the batch and both spatial axes"}[name])


def test_stage4_finite_differences(nb):
    rng = np.random.default_rng(12)
    x, w = rng.normal(size=(1, 2, 5, 5)), rng.normal(size=(2, 2, 3, 3))
    kw = dict(stride=2, padding=1, dilation=1)
    dout = rng.normal(size=nb.conv2d(x, w, **kw).shape)
    dx, dw, _ = nb.conv2d_backward(dout, x, w, **kw)
    eps = 1e-6
    for arr, grad, name in [(x, dx, "dx"), (w, dw, "dw")]:
        for idx in ([(0, 0, 0, 0), (1, 0, 2, 1), (1, 1, 2, 2)] if name == "dw" else [(0, 0, 0, 0), (0, 1, 2, 3), (0, 1, 4, 4)]):
            old = arr[idx]
            arr[idx] = old + eps
            up = np.sum(nb.conv2d(x, w, **kw) * dout)
            arr[idx] = old - eps
            down = np.sum(nb.conv2d(x, w, **kw) * dout)
            arr[idx] = old
            num = (up - down) / (2 * eps)
            assert np.isclose(grad[idx], num, rtol=1e-5, atol=1e-7), (
                f"{name}{idx}: analytic {grad[idx]:.6f} vs finite difference {num:.6f}")


def test_stage4_vectorized(nb):
    rng = np.random.default_rng(13)
    x, w = rng.normal(size=(2, 3, 96, 96)), rng.normal(size=(4, 3, 3, 3))
    dout = rng.normal(size=(2, 4, 96, 96))
    dx, dw, db = _line_budget(nb.conv2d_backward, dout, x, w, padding=1, budget=6000, what="conv2d_backward")
    assert dx.shape == x.shape and dw.shape == w.shape and db.shape == (4,)
