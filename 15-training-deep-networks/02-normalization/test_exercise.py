"""Checks for exercise.py (normalization).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The shared
standardize core is checked on a hand example and against finite differences; each layer against
its defining properties, mlfs.tensor.gradcheck, and (when installed) torch.nn in float64, including
BatchNorm's running statistics and evaluation mode. Each layer must be ONE Tensor.from_op node
whose parents are its input and its parameters, so its backward is the one you derived.
"""
import numpy as np
import pytest

from mlfs.numgrad import numerical_gradient
from mlfs.tensor import Tensor, gradcheck


def _one_node(out, x, params, layer):
    parents = getattr(out, "_parents", ())
    assert any(p is x for p in parents) and all(any(p is q for p in parents) for q in params), (
        f"{layer}'s output should be ONE Tensor.from_op node whose parents are the input and the parameters "
        "(Tensor.from_op(data, (x, weight, bias), vjp)), with your hand-derived vjp: don't build it from many "
        "small Tensor operations")


# ---------------------------------------------------------------- stage 1: the shared core


def test_stage1_standardize_hand_example(nb):
    x = np.array([[1.0, 2.0], [3.0, 6.0]])
    xhat, inv_std = nb.standardize(x, axis=0, eps=0.0)
    assert np.allclose(xhat, [[-1, -1], [1, 1]]), (
        f"columns (1, 3) and (2, 6) have means (2, 4) and biased stds (1, 2): xhat = [[-1, -1], [1, 1]]; got {xhat}")
    assert inv_std.shape == (1, 2) and np.allclose(inv_std, [[1.0, 0.5]]), (
        f"inv_std = 1/sqrt(var + eps), kept with keepdims=True so it broadcasts: want [[1, 0.5]] (1, 2), "
        f"got {np.asarray(inv_std).tolist()} {np.shape(inv_std)}")
    xhat_r, inv_r = nb.standardize(x, axis=-1, eps=0.0)
    assert inv_r.shape == (2, 1) and np.allclose(xhat_r, [[-1, 1], [-1, 1]])
    xe, ie = nb.standardize(np.ones((3, 2)), axis=0, eps=1e-5)
    assert np.allclose(xe, 0) and np.allclose(ie, 1 / np.sqrt(1e-5)), (
        "a constant column has variance 0: eps keeps the division finite (use the BIASED variance, ddof=0)")


def test_stage1_worked_example_backward(nb):
    x = np.array([[0.0], [1.0], [5.0]])
    g = np.array([[1.0], [0.0], [0.0]])
    xhat, inv_std = nb.standardize(x, axis=0, eps=0.0)
    dx = nb.standardize_backward(g, xhat, inv_std, axis=0)
    want = np.array([[0.17635], [-0.22043], [0.04409]])
    assert np.allclose(dx, want, atol=1e-5), (
        f"the lesson's worked example: x = (0, 1, 5), g = (1, 0, 0) gives dx = (0.17635, -0.22043, 0.04409); "
        f"got {np.ravel(dx).round(5).tolist()}")
    two = nb.standardize_backward(np.array([[0.3], [-1.2]]), *nb.standardize(np.array([[2.0], [7.0]]), axis=0, eps=0.0),
                                  axis=0)
    assert np.allclose(two, 0), (
        "with a batch of two, xhat is always (-1, +1) whatever x is, so the gradient with respect to x is exactly 0")


@pytest.mark.parametrize("axis", [0, -1])
def test_stage1_backward_matches_finite_differences(nb, axis):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 4)) * 2 + 1
    G = rng.normal(size=(5, 4))
    xhat, inv_std = nb.standardize(x, axis=axis)
    got = nb.standardize_backward(G, xhat, inv_std, axis=axis)
    want = numerical_gradient(lambda v: float((nb.standardize(v, axis=axis)[0] * G).sum()), x)
    assert np.allclose(got, want, atol=1e-6), (
        f"standardize_backward along axis {axis} disagrees with finite differences by "
        f"{np.abs(got - want).max():.2e}. dx = inv_std * (g - mean(g) - xhat * mean(g * xhat)): all three terms, "
        "means along the normalized axis")


def test_stage1_backward_properties(nb):
    rng = np.random.default_rng(2)
    x, G = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    xhat, inv_std = nb.standardize(x, axis=0, eps=0.0)  # eps = 0 makes both identities exact
    dx = nb.standardize_backward(G, xhat, inv_std, axis=0)
    assert np.allclose(dx.sum(axis=0), 0, atol=1e-10), (
        "shifting a column by a constant doesn't change xhat, so dx must sum to zero along the normalized axis "
        "(did you forget the - mean(g) term?)")
    assert np.allclose((dx * x).sum(axis=0), 0, atol=1e-8), (
        "scaling a column doesn't change xhat either, so dx is orthogonal to x along the axis "
        "(did you forget the - xhat * mean(g * xhat) term?)")


# ---------------------------------------------------------------- stage 2: BatchNorm1d


def test_stage2_parameters_and_buffers(nb):
    bn = nb.BatchNorm1d(3)
    assert [n for n, _ in bn.named_parameters()] == ["weight", "bias"], (
        "weight (gamma) then bias (beta) are the only Parameters; running_mean/running_var are plain ndarrays "
        "(buffers), so the optimizer never touches them")
    assert np.allclose(bn.weight.data, 1) and np.allclose(bn.bias.data, 0)
    assert isinstance(bn.running_mean, np.ndarray) and not isinstance(bn.running_mean, Tensor)
    assert np.allclose(bn.running_mean, 0) and np.allclose(bn.running_var, 1), "running_mean starts at 0, running_var at 1"
    assert bn.eps == 1e-5 and bn.momentum == 0.1


def test_stage2_train_forward_and_running_stats(nb):
    rng = np.random.default_rng(0)
    bn = nb.BatchNorm1d(3, momentum=0.1)
    bn.weight.data[...] = [1.0, 2.0, 0.5]
    bn.bias.data[...] = [0.0, -1.0, 3.0]
    x = rng.normal(size=(16, 3)) * [1, 5, 0.1] + [0, 10, -3]
    out = bn(Tensor(x, requires_grad=True))
    xhat = (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)
    assert np.allclose(out.data, xhat * [1, 2, 0.5] + [0, -1, 3]), (
        "training mode: normalize each column with the BATCH mean and BIASED variance, then * weight + bias")
    want_mean = 0.9 * 0 + 0.1 * x.mean(0)
    want_var = 0.9 * 1 + 0.1 * x.var(0, ddof=1)
    assert np.allclose(bn.running_mean, want_mean, rtol=1e-12, atol=1e-12), (
        "running_mean <- (1 - momentum) * running_mean + momentum * batch_mean (torch's convention: momentum is "
        "the weight of the NEW value)")
    assert np.allclose(bn.running_var, want_var, rtol=1e-12, atol=1e-12), (
        "running_var uses the UNBIASED batch variance (ddof=1), like torch, even though the normalization uses the "
        f"biased one; want {want_var}, got {bn.running_var}")
    with pytest.raises(ValueError):
        bn(np.ones((1, 3)))  # one example per batch: its variance is 0, nothing to normalize with


def test_stage2_eval_uses_running_stats(nb):
    rng = np.random.default_rng(3)
    bn = nb.BatchNorm1d(2)
    for _ in range(5):
        bn(rng.normal(size=(32, 2)) * 3 + 1)
    rm, rv = bn.running_mean.copy(), bn.running_var.copy()
    bn.eval()
    x = rng.normal(size=(4, 2))
    out = bn(x)
    assert np.allclose(out.data, (x - rm) / np.sqrt(rv + 1e-5)), (
        "eval mode normalizes with running_mean and running_var (and doesn't update them)")
    assert np.allclose(bn.running_mean, rm), "eval mode must not update the running statistics"
    one = bn(x[:1])
    assert np.allclose(one.data, out.data[:1]), (
        "in eval mode each example is normalized on its own: a batch of one works and gives the same answer")


def test_stage2_backward_gradcheck_and_one_node(nb):
    rng = np.random.default_rng(4)
    bn = nb.BatchNorm1d(3)
    bn.weight.data[...] = rng.normal(size=3)
    bn.bias.data[...] = rng.normal(size=3)
    x = Tensor(rng.normal(size=(6, 3)), requires_grad=True)
    _one_node(bn(x), x, [bn.weight, bn.bias], "BatchNorm1d")
    assert gradcheck(lambda v, w, b: bn(v), [x, bn.weight, bn.bias]), (
        "BatchNorm1d's training-mode gradients (input, weight, bias) disagree with finite differences: "
        "dx = standardize_backward(g * weight, ...), dweight = sum(g * xhat, axis=0), dbias = sum(g, axis=0)")
    bn.eval()
    assert gradcheck(lambda v, w, b: bn(v), [x, bn.weight, bn.bias]), (
        "eval-mode gradients: the statistics are constants now, so dx = g * weight * inv_std")


def test_stage2_matches_torch(nb):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    bn, tbn = nb.BatchNorm1d(4, momentum=0.2), torch.nn.BatchNorm1d(4, momentum=0.2).double()
    for _ in range(3):
        x = rng.normal(size=(10, 4)) * 2 + 0.5
        G = rng.normal(size=(10, 4))
        xt = torch.tensor(x, requires_grad=True)
        tout = tbn(xt)
        tout.backward(torch.tensor(G))
        xm = Tensor(x, requires_grad=True)
        out = bn(xm)
        out.backward(G)
        assert np.allclose(out.data, tout.detach().numpy(), atol=1e-10), "training forward differs from torch"
        assert np.allclose(xm.grad, xt.grad.numpy(), atol=1e-10), "input gradient differs from torch"
        assert np.allclose(bn.weight.grad, tbn.weight.grad.numpy(), atol=1e-10), "weight gradient differs from torch"
        assert np.allclose(bn.bias.grad, tbn.bias.grad.numpy(), atol=1e-10)
        bn.zero_grad()
        tbn.zero_grad()
    assert np.allclose(bn.running_mean, tbn.running_mean.numpy(), rtol=1e-12, atol=1e-12) and np.allclose(
            bn.running_var, tbn.running_var.numpy(), rtol=1e-12, atol=1e-12), (
        "running statistics after 3 batches differ from torch's (momentum weights the new value; unbiased variance)")
    bn.eval()
    tbn.eval()
    x = rng.normal(size=(3, 4))
    assert np.allclose(bn(x).data, tbn(torch.tensor(x)).detach().numpy(), atol=1e-10), "eval forward differs from torch"


# ---------------------------------------------------------------- stage 3: LayerNorm


def test_stage3_rows_normalized_and_batch_independent(nb):
    rng = np.random.default_rng(6)
    ln = nb.LayerNorm(5)
    assert [n for n, _ in ln.named_parameters()] == ["weight", "bias"] and ln.eps == 1e-5
    x = rng.normal(size=(4, 5)) * 3 + 2
    out = ln(x).data
    assert np.allclose(out.mean(axis=1), 0, atol=1e-10) and np.allclose(out.var(axis=1), 1, atol=1e-4), (
        "LayerNorm normalizes each ROW (the features of one example), not each column")
    assert np.allclose(ln(x[:1]).data, out[:1]), "an example's output doesn't depend on the rest of the batch"
    ln.eval()
    assert np.allclose(ln(x).data, out), "LayerNorm behaves the same in training and evaluation"
    x3 = rng.normal(size=(2, 3, 5))
    assert np.allclose(ln(x3).data.reshape(6, 5), ln(x3.reshape(6, 5)).data), (
        "inputs with extra leading dimensions (batch, time, features) normalize the last axis only")


def test_stage3_backward(nb):
    rng = np.random.default_rng(7)
    ln = nb.LayerNorm(4)
    ln.weight.data[...] = rng.normal(size=4)
    ln.bias.data[...] = rng.normal(size=4)
    x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    _one_node(ln(x), x, [ln.weight, ln.bias], "LayerNorm")
    assert gradcheck(lambda v, w, b: ln(v), [x, ln.weight, ln.bias]), (
        "LayerNorm gradients disagree with finite differences: same core as BatchNorm, along axis -1")
    x3 = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=True)
    assert gradcheck(lambda v, w, b: ln(v), [x3, ln.weight, ln.bias]), (
        "with a (2, 3, 4) input, the weight and bias gradients sum over BOTH leading axes")


def test_stage3_matches_torch(nb):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    ln, tln = nb.LayerNorm(6), torch.nn.LayerNorm(6).double()
    w, b = rng.normal(size=6), rng.normal(size=6)
    ln.weight.data[...], ln.bias.data[...] = w, b
    with torch.no_grad():
        tln.weight.copy_(torch.tensor(w))
        tln.bias.copy_(torch.tensor(b))
    x, G = rng.normal(size=(2, 5, 6)), rng.normal(size=(2, 5, 6))
    xt, xm = torch.tensor(x, requires_grad=True), Tensor(x, requires_grad=True)
    tln(xt).backward(torch.tensor(G))
    out = ln(xm)
    out.backward(G)
    assert np.allclose(out.data, tln(torch.tensor(x)).detach().numpy(), atol=1e-10)
    assert np.allclose(xm.grad, xt.grad.numpy(), atol=1e-10), "input gradient differs from torch.nn.LayerNorm"
    assert np.allclose(ln.weight.grad, tln.weight.grad.numpy(), atol=1e-10)
    assert np.allclose(ln.bias.grad, tln.bias.grad.numpy(), atol=1e-10)


# ---------------------------------------------------------------- stage 4: RMSNorm


def test_stage4_forward_and_properties(nb):
    rms = nb.RMSNorm(2)
    assert [n for n, _ in rms.named_parameters()] == ["weight"] and rms.eps == 1e-6, (
        "RMSNorm has a weight (ones) and no bias; eps defaults to 1e-6")
    out = rms(np.array([[3.0, 4.0]])).data
    r = 1 / np.sqrt((9 + 16) / 2 + 1e-6)
    assert np.allclose(out, [[3 * r, 4 * r]]), (
        f"(3, 4) has mean square 12.5: output (3, 4) / sqrt(12.5 + eps) = {[3 * r, 4 * r]}; got {out}. "
        "no mean is subtracted")
    rng = np.random.default_rng(9)
    x = rng.normal(size=(5, 8)) + 3
    rms = nb.RMSNorm(8)
    y = rms(x).data
    assert np.allclose((y ** 2).mean(axis=1), 1, atol=1e-5), "each row of the output has mean square 1"
    assert np.allclose(rms(7.0 * x).data, y, atol=1e-6), "RMSNorm is invariant to rescaling an example"
    assert not np.allclose(y.mean(axis=1), 0), "unlike LayerNorm, RMSNorm keeps the mean: (x + 3) stays shifted"


def test_stage4_backward(nb):
    rng = np.random.default_rng(10)
    rms = nb.RMSNorm(5)
    rms.weight.data[...] = rng.normal(size=5)
    x = Tensor(rng.normal(size=(4, 5)), requires_grad=True)
    _one_node(rms(x), x, [rms.weight], "RMSNorm")
    assert gradcheck(lambda v, w: rms(v), [x, rms.weight]), (
        "RMSNorm gradients disagree with finite differences. with u = g * weight: dx = r * u - x * r^3 * mean(u * x), "
        "dweight = sum(g * x * r) over the leading axes")
    x3 = Tensor(rng.normal(size=(2, 3, 5)), requires_grad=True)
    assert gradcheck(lambda v, w: rms(v), [x3, rms.weight])


def test_stage4_matches_torch(nb):
    torch = pytest.importorskip("torch")
    if not hasattr(torch.nn, "RMSNorm"):
        pytest.skip("torch.nn.RMSNorm needs torch >= 2.4")
    rng = np.random.default_rng(11)
    rms, trms = nb.RMSNorm(6), torch.nn.RMSNorm(6, eps=1e-6).double()
    w = rng.normal(size=6)
    rms.weight.data[...] = w
    with torch.no_grad():
        trms.weight.copy_(torch.tensor(w))
    x, G = rng.normal(size=(3, 6)), rng.normal(size=(3, 6))
    xt, xm = torch.tensor(x, requires_grad=True), Tensor(x, requires_grad=True)
    trms(xt).backward(torch.tensor(G))
    out = rms(xm)
    out.backward(G)
    assert np.allclose(out.data, trms(torch.tensor(x)).detach().numpy(), atol=1e-10)
    assert np.allclose(xm.grad, xt.grad.numpy(), atol=1e-10), "input gradient differs from torch.nn.RMSNorm"
    assert np.allclose(rms.weight.grad, trms.weight.grad.numpy(), atol=1e-10)
