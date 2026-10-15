"""Checks for exercise.py (backprop in depth).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. VJPs and Jacobians are
checked against hand-derived formulas and finite differences; graph memory against exact byte
counts; checkpointing against the plain computation (identical values and gradients), against
counts of how often each function runs, and against tracemalloc measurements of the same
computation without checkpointing, taken in the same test (relative, never absolute, limits).
Every backward pass runs under `mlfs.testing.bounded`.
"""
import gc
import tracemalloc

import numpy as np

from mlfs.autodiff import numerical_jacobian
from mlfs.tensor import Tensor, no_grad
from mlfs.testing import bounded


def _backward(t, grad=None):
    return bounded(t.backward, grad, what="backward", budget=2_000_000,
                   hint="a checkpointed node's vjp must recompute its function once, not call itself again.")


def _layers(rng, d=6, L=4, scale=1.0):
    Ws = [Tensor(scale * rng.normal(size=(d, d)) / np.sqrt(d), requires_grad=True) for _ in range(L)]
    bs = [Tensor(0.1 * rng.normal(size=d), requires_grad=True) for _ in range(L)]
    calls = [0] * L

    def make(i):
        def layer(h):
            calls[i] += 1
            return (h @ Ws[i].T + bs[i]).tanh()
        return layer

    return [make(i) for i in range(L)], Ws + bs, calls


def _grads(params):
    return [None if p.grad is None else p.grad.copy() for p in params]


def _clear(params):
    for p in params:
        p.grad = None


# ---------------------------------------------------------------- stage 1: VJPs and reverse-mode Jacobians


def test_stage1_vjp_hand_example(nb):
    A = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]])
    x = np.array([0.5, -1.0, 2.0])
    v = np.array([1.0, -2.0])
    y, g = bounded(nb.vjp, lambda t: t @ A.T * 2.0, x, v, what="vjp")
    assert isinstance(y, np.ndarray) and isinstance(g, np.ndarray), "vjp returns (f(x), J^T v) as two ndarrays"
    assert np.allclose(y, 2 * A @ x), "the first value is f(x)"
    assert g.shape == x.shape and np.allclose(g, 2 * A.T @ v), (
        f"f(x) = 2 A x has Jacobian 2A, so J^T v = 2 A^T v = {2 * A.T @ v}; got {g}")


def test_stage1_vjp_matches_finite_differences(nb):
    rng = np.random.default_rng(0)
    A, x, v = rng.normal(size=(4, 5)), rng.normal(size=5), rng.normal(size=4)

    def f(t):
        return (t @ A.T).tanh() * t[:4]

    def f_np(a):
        return np.tanh(A @ a) * a[:4]

    x0 = x.copy()
    y, g = bounded(nb.vjp, f, x, v, what="vjp")
    J = numerical_jacobian(f_np, x)
    assert np.allclose(y, f_np(x)) and np.allclose(g, J.T @ v, atol=1e-7), "vjp disagrees with J^T v from finite differences"
    assert np.array_equal(x, x0), "vjp must not modify x"


def test_stage1_jacobian_reverse(nb):
    rng = np.random.default_rng(1)
    A, x = rng.normal(size=(3, 5)), rng.normal(size=5)

    def f(t):
        return (t @ A.T).sigmoid() + t[1:4] ** 2

    J = bounded(nb.jacobian_reverse, f, x, what="jacobian_reverse")
    want = numerical_jacobian(lambda a: 1 / (1 + np.exp(-(A @ a))) + a[1:4] ** 2, x)
    assert J.shape == (3, 5), f"the Jacobian of R^5 -> R^3 is (3, 5) (outputs x inputs), got {J.shape}"
    assert np.allclose(J, want, atol=1e-7), (
        "row i of the Jacobian is the VJP with the i-th basis vector e_i. if rows look summed together, "
        "reset x.grad to None between backward passes (gradients accumulate)")
    M = rng.normal(size=(4, 4))  # square and non-symmetric: a transposed Jacobian has the right shape but wrong entries
    calls = []

    def g(t):
        calls.append(1)
        return (t @ M.T).tanh()

    x4 = rng.normal(size=4)
    J4 = bounded(nb.jacobian_reverse, g, x4, what="jacobian_reverse")
    want4 = (1 - np.tanh(M @ x4) ** 2)[:, None] * M
    assert np.allclose(J4, want4), "J[i, j] = d f_i / d x_j (rows are outputs): don't return the transpose"
    assert len(calls) == 1, (
        f"jacobian_reverse called f {len(calls)} times for 4 outputs: build the graph ONCE and run one backward pass "
        "per row through it")
    J1 = bounded(nb.jacobian_reverse, lambda t: (t * t).sum(), x, what="jacobian_reverse")
    assert np.shape(J1) == (1, 5) and np.allclose(J1, 2 * x), "a scalar function has a (1, n) Jacobian: its gradient as a row"


# ---------------------------------------------------------------- stage 2: graph memory


def test_stage2_graph_nbytes_exact(nb):
    rng = np.random.default_rng(2)
    x = Tensor(rng.normal(size=(10, 4)), requires_grad=True)
    W = Tensor(rng.normal(size=(4, 4)), requires_grad=True)
    h = x @ W            # (10, 4): 320 bytes
    a = h.tanh()         # 320
    out = (a * a).sum()  # a * a: 320, sum: 8
    got = nb.graph_nbytes(out)
    assert got == 3 * 320 + 8, (
        f"the non-leaf nodes are x @ W, tanh, a * a and the sum: 320 + 320 + 320 + 8 = 968 bytes; got {got}. "
        "leaves (x, W) are your data and weights, not activations: don't count them. a, used twice, counts once")
    assert nb.graph_nbytes(x) == 0, "a leaf on its own holds no graph"
    with no_grad():
        c = (x @ W).tanh()
    assert nb.graph_nbytes(c) == 0, "under no_grad nothing is recorded, so the result holds no graph"


def test_stage2_graph_nbytes_grows_with_depth(nb):
    rng = np.random.default_rng(3)
    fs, params, _ = _layers(rng, d=8, L=6)
    x = Tensor(rng.normal(size=(50, 8)))
    sizes = []
    h = x
    for f in fs:
        h = f(h)
        sizes.append(nb.graph_nbytes(h))
    per_layer = np.diff(sizes)
    assert np.all(per_layer == per_layer[0]) and per_layer[0] > 0, (
        f"each identical layer adds the same number of bytes to the graph (linear growth with depth); got {sizes}")
    assert per_layer[0] >= 3 * 50 * 8 * 8, "each layer keeps at least its matmul, bias add and tanh outputs (3 x 50 x 8 floats)"


# ---------------------------------------------------------------- stage 3: checkpoint


def test_stage3_checkpoint_same_values_and_grads(nb):
    rng = np.random.default_rng(4)
    fs, params, calls = _layers(rng)
    X = rng.normal(size=(7, 6))

    def block(h):
        for f in fs:
            h = f(h)
        return h

    x1 = Tensor(X, requires_grad=True)
    y1 = block(x1)
    _backward((y1 * y1).sum())
    g_plain, gx_plain = _grads(params), x1.grad.copy()
    _clear(params)
    calls[:] = [0] * len(calls)
    x2 = Tensor(X, requires_grad=True)
    y2 = nb.checkpoint(block, x2)
    assert isinstance(y2, Tensor) and y2.requires_grad, "checkpoint returns a Tensor that's part of the graph"
    assert np.allclose(y2.data, y1.data, rtol=0, atol=1e-12), "checkpoint(fn, x) has the same value as fn(x)"
    assert calls == [1] * len(fs), f"the forward pass runs fn once; layer call counts were {calls}"
    _backward((y2 * y2).sum())
    assert calls == [2] * len(fs), (
        f"backward recomputes fn exactly once (each layer runs once more); layer call counts were {calls}")
    for i, (a, b) in enumerate(zip(g_plain, _grads(params))):
        assert b is not None and np.allclose(a, b, rtol=0, atol=1e-12), (
            f"parameter {i}: checkpointed gradients must be identical to the plain call's. parameters that fn uses "
            "(closes over) get their gradients when the recomputed segment is backpropagated")
    assert np.allclose(x2.grad, gx_plain, rtol=0, atol=1e-12), "the input's gradient must match too"


def test_stage3_checkpoint_stores_nothing_inside(nb):
    rng = np.random.default_rng(5)
    fs, params, _ = _layers(rng, d=8, L=5)
    x = Tensor(rng.normal(size=(20, 8)), requires_grad=True)

    def block(h):
        for f in fs:
            h = f(h)
        return h

    plain = nb.graph_nbytes(block(x))
    ck = nb.checkpoint(block, x)
    got = nb.graph_nbytes(ck)
    assert got == ck.data.nbytes, (
        f"a checkpointed block is ONE graph node: it keeps its output ({ck.data.nbytes} bytes) and nothing from "
        f"inside fn (the plain call keeps {plain}). run fn under no_grad in the forward pass; got {got}")


def test_stage3_constant_inputs_and_closed_over_params(nb):
    rng = np.random.default_rng(6)
    fs, params, _ = _layers(rng)
    X = rng.normal(size=(5, 6))

    def block(h):
        for f in fs:
            h = f(h)
        return h

    _backward(block(Tensor(X)).sum())
    want = _grads(params)
    _clear(params)
    out = nb.checkpoint(block, Tensor(X))  # the input is plain data: only the closed-over weights need grads
    assert out.requires_grad, (
        "fn's parameters need gradients even when no input does: the checkpointed result must still be a graph node "
        "(set requires_grad and record the node yourself, instead of relying on from_op's 'any parent requires grad')")
    _backward(out.sum())
    assert all(g is not None and np.allclose(g, w) for g, w in zip(_grads(params), want)), (
        "the weights used inside fn must get the same gradients as without checkpointing")
    with no_grad():
        c = nb.checkpoint(block, Tensor(X))
    assert not c.requires_grad, "under no_grad, checkpoint returns a plain constant"


def test_stage3_memory_with_tracemalloc(nb):
    rng = np.random.default_rng(7)
    fs, params, _ = _layers(rng, d=128, L=8)
    X = rng.normal(size=(256, 128))

    def block(h):
        for f in fs:
            h = f(h)
        return h

    def retained(build):
        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            out = build()
            after = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
        return after - before, out

    plain, _ = retained(lambda: block(Tensor(X, requires_grad=True)))
    ck, _ = retained(lambda: nb.checkpoint(block, Tensor(X, requires_grad=True)))
    assert ck < 0.25 * plain, (
        f"memory kept alive after the forward pass: {ck / 1e6:.2f} MB with checkpoint vs {plain / 1e6:.2f} MB for the "
        "same 8 layers without it. checkpoint should keep only the input and output, not the 24 intermediates")


def test_stage3_nested_checkpoints(nb):
    rng = np.random.default_rng(8)
    fs, params, _ = _layers(rng)
    X = rng.normal(size=(4, 6))
    _backward((fs[3](fs[2](fs[1](fs[0](Tensor(X)))))).sum())
    want = _grads(params)
    _clear(params)

    def inner(h):
        return nb.checkpoint(lambda z: fs[2](fs[1](z)), h)

    out = nb.checkpoint(lambda h: fs[3](inner(fs[0](h))), Tensor(X))
    _backward(out.sum())
    assert all(np.allclose(g, w) for g, w in zip(_grads(params), want)), "a checkpoint inside a checkpoint must still work"


# ---------------------------------------------------------------- stage 4: checkpoint_sequential


def test_stage4_same_gradients_any_segments(nb):
    rng = np.random.default_rng(9)
    fs, params, calls = _layers(rng, L=6)
    X = rng.normal(size=(5, 6))
    h = Tensor(X)
    for f in fs:
        h = f(h)
    _backward((h ** 2).sum())
    want, want_out = _grads(params), h.data.copy()
    for segments in [1, 2, 3, 4, 6, 10]:
        _clear(params)
        calls[:] = [0] * len(calls)
        out = nb.checkpoint_sequential(fs, Tensor(X), segments)
        assert np.allclose(out.data, want_out, rtol=0, atol=1e-12), f"segments={segments}: wrong output"
        assert calls == [1] * 6, f"segments={segments}: the forward pass runs every layer once; counts {calls}"
        _backward((out ** 2).sum())
        assert calls == [2] * 6, (
            f"segments={segments}: backward recomputes every layer exactly once; counts {calls}. "
            "(segments larger than the number of layers means one layer per segment)")
        assert all(np.allclose(g, w, rtol=0, atol=1e-12) for g, w in zip(_grads(params), want)), (
            f"segments={segments}: gradients differ from the plain computation")


def test_stage4_graph_holds_one_tensor_per_segment(nb):
    rng = np.random.default_rng(10)
    fs, params, _ = _layers(rng, d=8, L=9)
    X = rng.normal(size=(30, 8))
    act = 30 * 8 * 8
    for segments, sizes in [(1, [9]), (3, [3, 3, 3]), (4, [3, 2, 2, 2]), (9, [1] * 9), (20, [1] * 9)]:
        out = nb.checkpoint_sequential(fs, Tensor(X), segments)
        got = nb.graph_nbytes(out)
        assert got == len(sizes) * act, (
            f"segments={segments}: the graph should hold one (30, 8) output per segment = {len(sizes) * act} bytes, "
            f"got {got}. split 9 layers like np.array_split: {sizes} (never more segments than layers)")


def test_stage4_peak_memory_drops(nb):
    rng = np.random.default_rng(11)
    fs, params, _ = _layers(rng, d=128, L=8)
    X = rng.normal(size=(256, 128))

    def peak(build):
        _clear(params)
        gc.collect()
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            out = build()
            _backward(out.sum())
            top = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return top - base

    def plain():
        h = Tensor(X)
        for f in fs:
            h = f(h)
        return h

    p_plain = peak(plain)
    p_seq = peak(lambda: nb.checkpoint_sequential(fs, Tensor(X), 4))
    assert p_seq < 0.85 * p_plain, (
        f"peak memory over forward + backward: {p_seq / 1e6:.2f} MB with 4 checkpointed segments vs "
        f"{p_plain / 1e6:.2f} MB without, measured on the same 8 layers. only one segment's intermediates should be "
        "alive at a time during backward")
