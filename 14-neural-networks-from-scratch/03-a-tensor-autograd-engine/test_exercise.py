"""Checks for exercise.py (a tensor autograd engine).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Gradients are checked
three ways: against the lesson's hand-worked example, against central differences (your
`mlfs.numgrad.numerical_gradient` from ch 02, the reference under pytest), and, where PyTorch is
installed, against torch.autograd. Every backward pass runs under `mlfs.testing.bounded`, so a
topological sort that never finishes fails instead of hanging.
"""
import contextlib
import io
import warnings

import numpy as np
import pytest

from mlfs.numgrad import numerical_gradient
from mlfs.testing import bounded

_HINT = ("walk the graph once to build a topological order (iteratively, with an explicit stack), "
         "then visit it in reverse; mark nodes as seen so shared subgraphs aren't revisited.")


def _backward(t, grad=None):
    try:
        return bounded(t.backward, grad, what="Tensor.backward", hint=_HINT)
    except RecursionError:
        raise AssertionError(
            "Tensor.backward hit Python's recursion limit: a recursive topological sort can't handle deep "
            "graphs (an RNN unrolled over 1,000 steps is one). build the order with an explicit stack instead."
        ) from None


def _grads(nb, f, *arrays, seed=1):
    """(analytic, numeric) gradients of sum(f(*tensors) * W) for a fixed random W, one pair per input."""
    ts = [nb.Tensor(a, requires_grad=True) for a in arrays]
    out = f(*ts)
    W = np.random.default_rng(seed).normal(size=out.shape)
    _backward(out, W)
    pairs = []
    for i, a in enumerate(arrays):
        def scalar(x, i=i):
            args = [nb.Tensor(x if j == i else arrays[j]) for j in range(len(arrays))]
            return float((f(*args).data * W).sum())
        analytic = ts[i].grad if ts[i].grad is not None else np.zeros_like(np.asarray(a, dtype=float))
        pairs.append((np.asarray(analytic), numerical_gradient(scalar, np.asarray(a, dtype=float))))
    return out, pairs


def _assert_grads(nb, f, *arrays, what, seed=1, rtol=1e-5, atol=1e-7):
    out, pairs = _grads(nb, f, *arrays, seed=seed)
    for i, (g, g_fd) in enumerate(pairs):
        assert g.shape == np.shape(arrays[i]), (
            f"{what}: the gradient of input {i} has shape {g.shape}, but the input has shape {np.shape(arrays[i])}. "
            "a gradient always has its variable's shape: did you forget to unbroadcast?")
        worst = np.max(np.abs(g - g_fd)) if g.size else 0.0
        assert np.allclose(g, g_fd, rtol=rtol, atol=atol), (
            f"{what}: the gradient of input {i} disagrees with finite differences (largest gap {worst:.3g}).\n"
            f"backward gave\n{np.round(g, 5)}\nfinite differences give\n{np.round(g_fd, 5)}")
    return out


# ---------------------------------------------------------------- stage 1: unbroadcast


def test_stage1_hand_examples(nb):
    g = np.arange(6.0).reshape(2, 3)  # [[0, 1, 2], [3, 4, 5]]
    cases = [((3,), [3.0, 5.0, 7.0], "a (3,) operand was broadcast down the rows: sum axis 0"),
             ((2, 1), [[3.0], [12.0]], "a (2, 1) operand was broadcast across the columns: sum axis 1, keep it"),
             ((1, 3), [[3.0, 5.0, 7.0]], "a (1, 3) operand: sum axis 0 but keep it as size 1"),
             ((), 15.0, "a scalar operand was broadcast everywhere: sum everything"),
             ((2, 3), g, "same shape: nothing to do")]
    for shape, want, why in cases:
        got = np.asarray(nb.unbroadcast(g, shape))
        assert got.shape == shape, f"unbroadcast(grad (2, 3), {shape}) must have shape {shape}, got {got.shape}. {why}"
        assert np.allclose(got, want), f"unbroadcast(grad (2, 3), {shape}) should be {want}, got {got}. {why}"


def test_stage1_leading_and_inner_axes(nb):
    g = np.ones((4, 2, 3))
    got = np.asarray(nb.unbroadcast(g, (2, 1)))
    assert got.shape == (2, 1) and np.allclose(got, 12.0), (
        f"(2, 1) broadcast to (4, 2, 3): drop the new leading axis (sum it) AND sum the size-1 axis: every entry "
        f"collects 4 * 3 = 12 ones. got shape {got.shape}, values {got.ravel()[:3]}")
    got = np.asarray(nb.unbroadcast(np.ones((5, 1, 3)), (1, 3)))
    assert got.shape == (1, 3) and np.allclose(got, 5.0), (
        "a size-1 axis of the gradient that was size 1 in the operand too needs no sum; the leading 5 does")


def test_stage1_adjoint_of_broadcasting(nb):
    # unbroadcast is the adjoint (transpose) of broadcasting: <G, broadcast(x)> = <unbroadcast(G), x>
    rng = np.random.default_rng(0)
    for small, big in [((3,), (4, 3)), ((4, 1), (4, 5)), ((1, 5), (2, 3, 5)), ((), (2, 2)), ((2, 1, 3), (2, 4, 3)),
                       ((1, 1), (3, 4))]:
        x, G = rng.normal(size=small), rng.normal(size=big)
        lhs = np.sum(G * np.broadcast_to(x, big))
        rhs = np.sum(np.asarray(nb.unbroadcast(G, small)) * x)
        assert np.isclose(lhs, rhs), (
            f"for x of shape {small} broadcast to {big}: <G, broadcast(x)> = {lhs:.6f} but <unbroadcast(G), x> = "
            f"{rhs:.6f}. every copy of an entry of x must send its gradient back to that entry: sum, don't average "
            "or pick one")
    G0 = G.copy()
    nb.unbroadcast(G, (1, 1))
    assert np.array_equal(G, G0), "unbroadcast must not modify the gradient you pass in"


# ---------------------------------------------------------------- stage 2: the graph and arithmetic


def test_stage2_tensor_basics(nb):
    t = nb.Tensor([[1, 2, 3], [4, 5, 6]])
    assert isinstance(t.data, np.ndarray) and t.data.dtype == np.float64, "Tensor.data is a float64 ndarray"
    assert t.shape == (2, 3) and t.ndim == 2 and t.size == 6 and t.grad is None and t.requires_grad is False
    a = np.ones(3)
    p = nb.Tensor(a, requires_grad=True)
    a[0] = 5.0
    assert p.data[0] == 1.0, "Tensor(array) copies the array, so later changes to the array don't leak in"


def test_stage2_worked_example(nb):
    x = nb.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = nb.Tensor([1.0, 0.0, -1.0], requires_grad=True)
    c = nb.Tensor(2.0, requires_grad=True)
    L = ((x * b + c) ** 2).sum()
    assert np.isclose(L.item(), 70.0), f"L = sum((x * b + c)^2) should be 70 (the lesson's worked example), got {L.item()}"
    _backward(L)
    assert b.grad is not None and b.grad.shape == (3,), (
        f"b has shape (3,), so b.grad must too; got {None if b.grad is None else b.grad.shape}. b was broadcast "
        "over 2 rows: its gradient is the column sum (unbroadcast)")
    assert np.allclose(b.grad, [54.0, 28.0, -54.0]), f"dL/db should be (54, 28, -54), got {b.grad}"
    assert np.shape(c.grad) == () and np.isclose(c.grad, 16.0), f"dL/dc should be the scalar 16 (sum of 2u), got {c.grad}"
    assert np.allclose(x.grad, [[6.0, 0.0, 2.0], [12.0, 0.0, 8.0]]), f"dL/dx should be 2u * b, got {x.grad}"


def test_stage2_from_op_records_only_when_needed(nb):
    a = nb.Tensor([1.0, 2.0], requires_grad=True)
    y = nb.Tensor.from_op(2.0 * a.data, (a,), lambda g: (2.0 * g,))
    assert isinstance(y, nb.Tensor) and y.requires_grad, "from_op with a parent that requires grad gives a result that does"
    _backward(y, np.array([1.0, 10.0]))
    assert np.allclose(a.grad, [2.0, 20.0]), f"backward must call the node's vjp with the incoming gradient: got {a.grad}"
    k = nb.Tensor([1.0, 2.0])
    z = nb.Tensor.from_op(k.data + 1.0, (k,), lambda g: (g,))
    assert z.requires_grad is False, "an operation on constants only (no parent requires grad) records nothing"
    assert (k + k).requires_grad is False and (a + k).requires_grad is True


def test_stage2_backward_rules(nb):
    x = nb.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    with pytest.raises(ValueError):
        (x * 2.0).backward()  # non-scalar output without a gradient
    y = (x * x).sum()
    _backward(y)
    _backward((x * 3.0).sum())
    assert np.allclose(x.grad, 2 * x.data + 3.0), (
        f"x * x needs both of its inputs' contributions (2x), and a second backward ADDS to .grad (+3): want "
        f"{2 * x.data + 3.0}, got {x.grad}. use += on leaves, and sum gradients arriving from several children")
    h = x * 2.0
    _backward((h + h * h).sum())
    assert h.grad is None, "only leaves keep .grad; intermediate results don't"
    with pytest.raises(RuntimeError):
        nb.Tensor([1.0]).backward()


def test_stage2_leaf_grads_are_independent_copies(nb):
    w = nb.Tensor(np.ones((2, 3)), requires_grad=True)
    _backward(w.sum())
    _backward(w.sum())
    assert np.allclose(w.grad, 2.0), (
        f"two backward passes through w.sum() must give 2 everywhere, got {w.grad}. sum's VJP may hand you a read-only "
        "broadcast view: store a COPY the first time a leaf gets a gradient, then add new arrays to it")
    a = nb.Tensor(np.zeros(3), requires_grad=True)
    b = nb.Tensor(np.zeros(3), requires_grad=True)
    _backward((a + b).sum())
    assert not np.shares_memory(a.grad, b.grad), (
        "a.grad and b.grad share memory: the add's VJP gave both parents the same array, and storing it without a copy "
        "aliases them (scaling one gradient in place, as gradient clipping does, would scale the other too)")
    assert a.grad.flags.writeable and b.grad.flags.writeable, "a leaf's .grad must be an ordinary writeable array"


def test_stage2_diamond_and_reuse(nb):
    a = nb.Tensor(3.0, requires_grad=True)
    b = a * 2.0
    c = a + b
    d = b * c  # d = 2a * 3a = 6a^2, so dd/da = 12a = 36
    _backward(d)
    assert np.isclose(a.grad, 36.0), (
        f"d = (2a)(a + 2a) = 6a^2, dd/da = 36 at a = 3; got {a.grad}. b is used twice: its gradient must be complete "
        "(both uses summed) before it's passed back to a: process nodes in reverse topological order")


def test_stage2_arithmetic_gradients(nb):
    rng = np.random.default_rng(2)
    A, B, s = rng.normal(size=(3, 4)), rng.normal(size=(4,)), rng.normal(size=(3, 1))
    _assert_grads(nb, lambda a, b: a + b, A, B, what="a + b with b broadcast (4,) -> (3, 4)")
    _assert_grads(nb, lambda a, b: a - b, A, s, what="a - b with b (3, 1)")
    _assert_grads(nb, lambda a, b: a * b, A, B, what="a * b")
    _assert_grads(nb, lambda a, b: a / b, A, np.abs(s) + 1.0, what="a / b")
    _assert_grads(nb, lambda a: -a, A, what="-a")
    _assert_grads(nb, lambda a, b: (a * b - a / b) * a, A, np.abs(B) + 0.5, what="(a b - a / b) a")
    _assert_grads(nb, lambda a: a.sum(axis=0), A, what="a.sum(axis=0)")
    _assert_grads(nb, lambda a: a.sum(axis=(0, 1), keepdims=True), A, what="a.sum(axis=(0, 1), keepdims=True)")


def test_stage2_constants_and_reflected_ops(nb):
    x = nb.Tensor([1.0, 2.0, 4.0], requires_grad=True)
    exprs = {"2 - x": (2 - x, [-1, -1, -1]), "3 * x": (3 * x, [3, 3, 3]), "1 / x": (1 / x, [-1, -0.25, -0.0625]),
             "np.ones(3) + x": (np.ones(3) + x, [1, 1, 1]), "np.arange(3.0) * x": (np.arange(3.0) * x, [0, 1, 2]),
             "x - np.float64(1)": (x - np.float64(1.0), [1, 1, 1])}
    for name, (y, want) in exprs.items():
        assert isinstance(y, nb.Tensor), f"{name} must give a Tensor (non-Tensor operands are constants), got {type(y)}"
        x.grad = None
        _backward(y, np.ones(3))
        assert np.allclose(x.grad, want), f"d({name})/dx should be {want}, got {x.grad}"


def test_stage2_powers(nb):
    for p, want in [(2, [2.0, 4.0, 8.0]), (0.5, [0.5, 1 / (2 * np.sqrt(2)), 0.25]), (-1, [-1.0, -0.25, -0.0625]),
                    (0, [0.0, 0.0, 0.0]), (1, [1.0, 1.0, 1.0])]:
        x = nb.Tensor([1.0, 2.0, 4.0], requires_grad=True)
        _backward((x ** p).sum())
        assert np.allclose(x.grad, want), f"d/dx sum(x ** {p}) at x = (1, 2, 4) should be {np.round(want, 4)}, got {x.grad}"
    x = nb.Tensor([0.0, 3.0], requires_grad=True)
    with np.errstate(all="ignore"):
        _backward((x ** 0).sum())
    assert np.allclose(x.grad, [0.0, 0.0]), f"x ** 0 is the constant 1: its gradient is 0 everywhere, even at x = 0 (not nan); got {x.grad}"
    with pytest.raises(TypeError):
        nb.Tensor([1.0]) ** nb.Tensor([2.0])


def test_stage2_deep_graph(nb):
    x = nb.Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(3000):
        y = y * 1.0 + 0.001
    _backward(y)
    assert np.isclose(x.grad, 1.0), f"dy/dx through 3,000 steps of y = y * 1 + 0.001 is 1, got {x.grad}"
    z = nb.Tensor(1.0, requires_grad=True)
    y = z
    for _ in range(40):
        y = y + y  # each node is used twice: without a `seen` set, the walk visits 2^40 paths
    _backward(y)
    assert np.isclose(z.grad, 2.0 ** 40), (
        f"y = z doubled 40 times has dy/dz = 2^40, got {z.grad}. (if this timed out: mark nodes as seen, so a node shared "
        "by two children is expanded once)")


# ---------------------------------------------------------------- stage 3: elementwise functions and reductions


def test_stage3_elementwise_gradients(nb):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, 4))
    for name in ["exp", "tanh", "sigmoid"]:
        out = _assert_grads(nb, lambda a, n=name: getattr(a, n)(), X, what=f"{name}")
        want = {"exp": np.exp(X), "tanh": np.tanh(X), "sigmoid": 1 / (1 + np.exp(-X))}[name]
        assert np.allclose(out.data, want), f"{name}: wrong forward values"
    out = _assert_grads(nb, lambda a: a.log(), np.abs(X) + 0.1, what="log")
    X2 = X + np.sign(X) * 0.05  # keep away from the kink
    out = _assert_grads(nb, lambda a: a.relu(), X2, what="relu")
    assert np.allclose(out.data, np.maximum(X2, 0))
    z = nb.Tensor([0.0, -1.0, 2.0], requires_grad=True)
    _backward(z.relu().sum())
    assert np.allclose(z.grad, [0.0, 0.0, 1.0]), f"relu's gradient is 1 where x > 0 and 0 elsewhere (also at 0), got {z.grad}"


def test_stage3_sigmoid_is_stable(nb):
    x = nb.Tensor([-1000.0, -50.0, 0.0, 50.0, 1000.0], requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            s = x.sigmoid()
            _backward(s.sum())
        except RuntimeWarning as w:
            raise AssertionError(f"sigmoid overflowed ({w}): compute it so exp never sees a large positive "
                                 "argument (e.g. exp(-logaddexp(0, -x)))") from None
    assert np.allclose(s.data, [0, 1 / (1 + np.exp(50)), 0.5, 1 / (1 + np.exp(-50)), 1]), f"sigmoid values: {s.data}"
    assert np.all(np.isfinite(x.grad)) and np.isclose(x.grad[2], 0.25), f"sigmoid' = s(1 - s), finite: got {x.grad}"


def test_stage3_reductions(nb):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2, 3, 4))
    for axis in [None, 0, 1, -1, (0, 2)]:
        for keepdims in [False, True]:
            out = _assert_grads(nb, lambda a: a.mean(axis=axis, keepdims=keepdims), X,
                                what=f"mean(axis={axis}, keepdims={keepdims})")
            want = X.mean(axis=axis, keepdims=keepdims)
            assert np.shape(out.data) == np.shape(want) and np.allclose(out.data, want), (
                f"mean(axis={axis}, keepdims={keepdims}) should have shape {np.shape(want)} like ndarray.mean, "
                f"got {np.shape(out.data)}")
    for axis in [None, 1, -1]:
        out = _assert_grads(nb, lambda a: a.max(axis=axis), X, what=f"max(axis={axis})")
        assert np.allclose(out.data, X.max(axis=axis))
        out = _assert_grads(nb, lambda a: a.logsumexp(axis=axis), X, what=f"logsumexp(axis={axis})")
        assert np.allclose(out.data, np.log(np.exp(X).sum(axis=axis)))
    out = nb.Tensor(X).logsumexp()
    assert out.shape == (2, 3), "logsumexp's default axis is -1"
    assert nb.Tensor(X).logsumexp(axis=None).shape == (), "logsumexp(axis=None) reduces everything to a 0-d Tensor"
    assert nb.Tensor(X).max().shape == (), "max() reduces everything to a 0-d Tensor"
    out = nb.Tensor(X).logsumexp(axis=1, keepdims=True)
    assert out.shape == (2, 1, 4)


def test_stage3_max_ties_split(nb):
    x = nb.Tensor([1.0, 3.0, 3.0], requires_grad=True)
    _backward(x.max())
    assert np.allclose(x.grad, [0.0, 0.5, 0.5]), (
        f"two entries tie for the max: split the gradient evenly between them (0, 0.5, 0.5), got {x.grad}")
    y = nb.Tensor([[2.0, 2.0, 1.0], [0.0, 5.0, -1.0]], requires_grad=True)
    _backward(y.max(axis=1).sum())
    assert np.allclose(y.grad, [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]), f"row-wise ties are split per row, got {y.grad}"


def test_stage3_logsumexp_is_stable(nb):
    z = nb.Tensor([[1000.0, 1000.0, 0.0], [-1000.0, -1001.0, -1002.0]], requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            out = z.logsumexp(axis=1)
            _backward(out.sum())
        except RuntimeWarning as w:
            raise AssertionError(f"logsumexp overflowed or underflowed ({w}): subtract the max first") from None
    want = [1000 + np.log(2), -1000 + np.log(1 + np.exp(-1) + np.exp(-2))]
    assert np.allclose(out.data, want), f"logsumexp of huge logits should be {want}, got {out.data}"
    p = np.exp(z.data - np.array(want)[:, None])
    assert np.allclose(z.grad, p), f"the gradient of logsumexp is the softmax: want {np.round(p, 3)}, got {z.grad}"


# ---------------------------------------------------------------- stage 4: linear algebra, shapes, indexing


def test_stage4_matmul(nb):
    rng = np.random.default_rng(5)
    A, B, v, w = rng.normal(size=(3, 4)), rng.normal(size=(4, 2)), rng.normal(size=4), rng.normal(size=4)
    C = rng.normal(size=(5, 4, 2))
    _assert_grads(nb, lambda a, b: a @ b, A, B, what="(3, 4) @ (4, 2)")
    _assert_grads(nb, lambda a, b: a @ b, A, v, what="matrix @ vector")
    _assert_grads(nb, lambda a, b: a @ b, v, B, what="vector @ matrix")
    _assert_grads(nb, lambda a, b: a @ b, v, w, what="vector @ vector (a dot product)")
    _assert_grads(nb, lambda a, b: a @ b, A, C, what="(3, 4) @ (5, 4, 2), the (3, 4) broadcast over the batch")
    _assert_grads(nb, lambda a, b: a @ b, C.transpose(0, 2, 1), A.T, what="(5, 2, 4) @ (4, 3): sum the batch for b")
    _assert_grads(nb, lambda a, b: a @ b, rng.normal(size=(2, 1, 3, 4)), C, what="(2, 1, 3, 4) @ (5, 4, 2)")
    out = nb.Tensor(A) @ B
    assert isinstance(out, nb.Tensor) and np.allclose(out.data, A @ B)
    x = nb.Tensor(B, requires_grad=True)
    _backward((A @ x).sum())  # ndarray @ Tensor
    assert np.allclose(x.grad, A.T @ np.ones((3, 2))), "ndarray @ Tensor must work too (__rmatmul__)"


def test_stage4_reshape_transpose(nb):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(2, 3, 4))
    _assert_grads(nb, lambda a: a.reshape(6, 4) * np.arange(4.0), X, what="reshape(6, 4)")
    _assert_grads(nb, lambda a: a.reshape((4, -1)), X, what="reshape((4, -1))")
    out = _assert_grads(nb, lambda a: a.transpose(2, 0, 1), X, what="transpose(2, 0, 1)")
    assert out.shape == (4, 2, 3)
    out = _assert_grads(nb, lambda a: a.transpose((1, 2, 0)), X, what="transpose((1, 2, 0))")
    assert out.shape == (3, 4, 2), "transpose accepts a tuple of axes too"
    M = rng.normal(size=(3, 5))
    assert nb.Tensor(M).T.shape == (5, 3) and nb.Tensor(X).transpose().shape == (4, 3, 2)
    _assert_grads(nb, lambda a: a.T * np.arange(3.0), M, what=".T")


def test_stage4_indexing(nb):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(4, 5))
    _assert_grads(nb, lambda a: a[1:3, ::2], X, what="a[1:3, ::2]")
    _assert_grads(nb, lambda a: a[-1], X, what="a[-1]")
    out = _assert_grads(nb, lambda a: a[np.array([0, 2, 0, 0])], X, what="a[[0, 2, 0, 0]] (repeated rows)")
    x = nb.Tensor(X, requires_grad=True)
    _backward(x[np.array([0, 2, 0, 0])].sum())
    assert np.allclose(x.grad[0], 3.0) and np.allclose(x.grad[1], 0.0), (
        "row 0 was picked 3 times, so its gradient is 3: repeated indices must ADD (np.add.at), not overwrite "
        f"(grad[idx] += g keeps only one copy). got row 0 = {x.grad[0]}")
    labels = np.array([4, 0, 0, 2])
    _assert_grads(nb, lambda a: a[np.arange(4), labels], X, what="a[arange(n), labels] (picking one entry per row)")
    _assert_grads(nb, lambda a: a[X > 0], X, what="a[mask] (boolean mask)")
    rows = nb.Tensor(np.array([3, 0]))
    out = _assert_grads(nb, lambda a: a[rows, 1:3], X, what="a[Tensor_of_ints, 1:3] (a Tensor inside a tuple index)")
    assert np.allclose(out.data, X[[3, 0], 1:3]), "an index may contain Tensors of integers: use their .data as ints"


def test_stage4_concatenate_stack_where(nb):
    rng = np.random.default_rng(8)
    A, B = rng.normal(size=(2, 3)), rng.normal(size=(4, 3))
    out = _assert_grads(nb, lambda a, b: nb.concatenate([a, b], axis=0), A, B, what="concatenate axis 0")
    assert out.shape == (6, 3)
    _assert_grads(nb, lambda a, b: nb.concatenate([a, b.T, a], axis=1), A, rng.normal(size=(3, 2)),
                  what="concatenate axis 1 with a repeated input")
    out = nb.concatenate([nb.Tensor(A, requires_grad=True), np.zeros((1, 3))])
    assert out.shape == (3, 3) and out.requires_grad, "constants (arrays) can be concatenated with Tensors"
    out = _assert_grads(nb, lambda a, b: nb.stack([a, b * 2.0], axis=1), A, A + 1.0, what="stack axis 1")
    assert out.shape == (2, 2, 3)
    _assert_grads(nb, lambda a: nb.stack([a, a], axis=-1), A, what="stack axis -1")
    cond = A > np.median(A)  # half the entries True, half False
    x = nb.Tensor(A, requires_grad=True)
    y = nb.Tensor(np.arange(3.0), requires_grad=True)
    out = nb.where(cond, x * 2.0, y)
    assert np.allclose(out.data, np.where(cond, 2 * A, np.arange(3.0))), "where: wrong forward values"
    _backward(out.sum())
    assert np.allclose(x.grad, 2.0 * cond), f"where: x gets gradient only where the condition holds, got {x.grad}"
    assert np.allclose(y.grad, (~cond).sum(axis=0)), (
        f"where: y (3,) was broadcast, so its gradient sums over the rows where it was chosen: want "
        f"{(~cond).sum(axis=0)}, got {y.grad}")


def test_stage4_composite_mlp(nb):
    rng = np.random.default_rng(9)
    X, y = rng.normal(size=(6, 4)), np.array([0, 2, 1, 2, 0, 1])
    W1, b1, W2 = rng.normal(size=(5, 4)) * 0.5, rng.normal(size=5) * 0.1, rng.normal(size=(3, 5)) * 0.5

    def loss(w1, b1_, w2):
        h = (nb.Tensor(X) @ w1.T + b1_).tanh()
        z = h @ w2.T
        return (z.logsumexp(axis=1) - z[np.arange(6), y]).mean()

    _assert_grads(nb, loss, W1, b1, W2, what="a 2-layer tanh network with softmax cross-entropy")


def test_stage4_matches_pytorch(nb):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(10)
    X, W, b = rng.normal(size=(8, 5)), rng.normal(size=(3, 5)), rng.normal(size=3)
    y = rng.integers(0, 3, 8)
    tX, tW, tb = (nb.Tensor(a, requires_grad=True) for a in (X, W, b))
    z = (tX @ tW.T + tb).relu() * 1.5 - (tX[:, :3] ** 2).sigmoid()
    L = (z.logsumexp(axis=1) - z[np.arange(8), y]).mean() + (z.max(axis=0) / 4.0).sum()
    _backward(L)
    pX, pW, pb = (torch.tensor(a, requires_grad=True) for a in (X, W, b))
    pz = torch.relu(pX @ pW.T + pb) * 1.5 - torch.sigmoid(pX[:, :3] ** 2)
    pL = torch.nn.functional.cross_entropy(pz, torch.tensor(y)) + (pz.amax(dim=0) / 4.0).sum()
    pL.backward()
    assert np.isclose(L.item(), pL.item()), f"loss {L.item()} != PyTorch's {pL.item()}"
    for name, mine, ref in [("X", tX, pX), ("W", tW, pW), ("b", tb, pb)]:
        assert np.allclose(mine.grad, ref.grad.numpy(), atol=1e-10), f"d loss / d {name} differs from PyTorch's"


# ---------------------------------------------------------------- stage 5: no_grad and gradcheck


def test_stage5_no_grad(nb):
    w = nb.Tensor([1.0, 2.0], requires_grad=True)
    assert nb.is_grad_enabled() is True
    with nb.no_grad():
        assert nb.is_grad_enabled() is False, "is_grad_enabled() is False inside no_grad"
        y = w * 3.0 + 1.0
        with nb.no_grad():
            pass
        assert nb.is_grad_enabled() is False, "leaving an inner no_grad must not re-enable grad in the outer one"
    assert nb.is_grad_enabled() is True, "grad is enabled again after the block"
    assert y.requires_grad is False, "results computed inside no_grad don't require grad (no graph was recorded)"
    assert w.requires_grad is True, "no_grad doesn't change existing tensors' flags"
    try:
        with nb.no_grad():
            raise KeyError("boom")
    except KeyError:
        pass
    assert nb.is_grad_enabled() is True, "an exception inside no_grad must still restore the switch (use __exit__)"


def test_stage5_gradcheck_accepts_correct_ops(nb):
    rng = np.random.default_rng(11)
    a = nb.Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    b = nb.Tensor(rng.normal(size=(4,)), requires_grad=True)
    c = nb.Tensor(rng.normal(size=(4, 2)))  # a constant: skipped
    a0, c0 = a.data.copy(), c.data.copy()
    a.grad = np.full((3, 4), 7.0)
    ok = bounded(nb.gradcheck, lambda x, y, z: ((x * y).tanh() @ z).sigmoid(), [a, b, c], what="gradcheck")
    assert ok is True, "gradcheck should return True for correct gradients"
    assert np.array_equal(a.data, a0) and np.array_equal(c.data, c0), "gradcheck must leave the inputs' data unchanged"
    assert np.allclose(a.grad, 7.0), "gradcheck must leave the inputs' existing .grad as it found it"


def test_stage5_gradcheck_catches_bugs(nb):
    Tensor = nb.Tensor

    def no_unbroadcast_add(x, y):  # forgets to sum y's gradient over the broadcast rows: takes row 0 only
        return Tensor.from_op(x.data + y.data, (x, y), lambda g: (g, g[0]))

    def reversed_but_not_back(x):  # reverses x, but the vjp forgets to reverse g
        return Tensor.from_op(x.data[::-1], (x,), lambda g: (g,))

    rng = np.random.default_rng(12)
    x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    y = Tensor(rng.normal(size=(4,)), requires_grad=True)
    v = Tensor(rng.normal(size=5), requires_grad=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bad1 = bounded(nb.gradcheck, no_unbroadcast_add, [x, y], what="gradcheck")
        bad2 = bounded(nb.gradcheck, reversed_but_not_back, [v], what="gradcheck")
    assert bad1 is False, "gradcheck missed a broadcasting bug in the gradient of y"
    assert bad2 is False, (
        "gradcheck missed a permuted gradient. if you reduce f's output with a plain .sum(), every output entry "
        "gets the same weight and a permutation is invisible: weight the outputs with fixed random numbers")
    assert buf.getvalue().strip(), "on failure, gradcheck prints the worst entry (analytic vs numerical)"

    def one_percent_off(x):  # the vjp is 1% too large: a real bug that a loose tolerance would let through
        return Tensor.from_op(3.0 * x.data, (x,), lambda g: (3.03 * g,))

    with contextlib.redirect_stdout(io.StringIO()):
        bad3 = bounded(nb.gradcheck, one_percent_off, [Tensor(rng.normal(size=5), requires_grad=True)], what="gradcheck")
    assert bad3 is False, (
        "gradcheck accepted a gradient that is 1% wrong: compare with np.allclose(analytic, numeric, atol=atol, "
        "rtol=rtol) using the defaults atol=1e-5, rtol=1e-3")
