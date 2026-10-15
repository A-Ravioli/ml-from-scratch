"""Checks for exercise.py (scalar reverse-mode autodiff).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Stages 1 and 2 check
each operation's value and local backward rule (driving the graph with a small sweep written
here, so they don't depend on your stage 3); stage 3 checks the topological sort and full
backward pass against the lesson's worked example and central differences; stage 4 checks the
convenience functions. Graph walks run under `mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

from mlfs.numgrad import numerical_gradient
from mlfs.testing import bounded

_HINT = "use an explicit stack and a set of visited nodes (by id), so shared nodes aren't expanded twice."


def _sweep(root):
    """a reference backward sweep (topological order + local rules), so stages 1-2 don't need your stage 3."""
    order, seen = [], set()
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
        elif id(node) not in seen:
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._prev)
    root.grad = 1.0
    for v in reversed(order):
        v._backward()


def _backward(v):
    try:
        return bounded(v.backward, what="Value.backward", hint=_HINT)
    except RecursionError:
        raise AssertionError("Value.backward hit Python's recursion limit: build the topological order with an "
                             "explicit stack, not a recursive function") from None


def _grad_of(nb, f, x):
    xs = [nb.Value(v) for v in x]
    out = f(xs)
    _backward(out)
    return out, np.array([v.grad for v in xs])


# ---------------------------------------------------------------- stage 1: Value, + and *


def test_stage1_value_basics(nb):
    v = nb.Value(3)
    assert isinstance(v.data, float) and v.data == 3.0 and v.grad == 0.0, "Value(3): data 3.0 (a float), grad 0.0"
    assert tuple(v._prev) == () and v._op == "", "a Value you create yourself has no parents and no op"
    v._backward()  # the default must exist and do nothing


def test_stage1_add_mul_record_the_graph(nb):
    a, b = nb.Value(2.0), nb.Value(-3.0)
    c = a * b
    d = c + a
    assert c.data == -6.0 and d.data == -4.0
    assert tuple(c._prev) == (a, b) and c._op == "*", "a * b records its parents (a, b) and op '*'"
    assert tuple(d._prev) == (c, a) and d._op == "+"
    e = a + 1.5
    assert isinstance(e, nb.Value) and e.data == 3.5 and len(e._prev) == 2, (
        "a number operand is wrapped as a constant Value (so it appears in _prev)")
    for expr, want in [(2 * a, 4.0), (1 + a, 3.0), (a * 0.5, 1.0)]:
        assert isinstance(expr, nb.Value) and expr.data == want, "reflected operators (2 * a, 1 + a) work too"


def test_stage1_local_rules(nb):
    a, b = nb.Value(2.0), nb.Value(-3.0)
    c = a * b
    c.grad = 10.0
    c._backward()
    assert a.grad == -30.0 and b.grad == 20.0, (
        f"c = a * b with dL/dc = 10: dL/da = 10 * b = -30, dL/db = 10 * a = 20; got {a.grad}, {b.grad}")
    s = a + b
    s.grad = 1.0
    s._backward()
    assert a.grad == -29.0 and b.grad == 21.0, (
        "_backward must ADD (+=) into the parents' grads: a value used in several places collects the sum of all "
        f"its contributions; got {a.grad}, {b.grad}")


def test_stage1_square_by_reuse(nb):
    x = nb.Value(3.0)
    y = x * x
    _sweep(y)
    assert x.grad == 6.0, (
        f"y = x * x: dy/dx = 2x = 6, got {x.grad}. both parents of the product are x, and each gets a contribution "
        "(3 + 3): use +=, not =")


# ---------------------------------------------------------------- stage 2: more operations


@pytest.mark.parametrize("name", ["neg", "sub", "rsub", "pow2", "pow_half", "pow_neg", "div", "rdiv", "exp", "log",
                                  "tanh", "relu_pos", "relu_neg"])
def test_stage2_operation(nb, name):
    x0 = 1.7
    cases = {
        "neg": (lambda x: -x, lambda v: -v, lambda v: -1.0),
        "sub": (lambda x: x - 0.4, lambda v: v - 0.4, lambda v: 1.0),
        "rsub": (lambda x: 2.0 - x, lambda v: 2.0 - v, lambda v: -1.0),
        "pow2": (lambda x: x ** 2, lambda v: v ** 2, lambda v: 2 * v),
        "pow_half": (lambda x: x ** 0.5, lambda v: v ** 0.5, lambda v: 0.5 / math.sqrt(v)),
        "pow_neg": (lambda x: x ** -2, lambda v: v ** -2, lambda v: -2 * v ** -3),
        "div": (lambda x: x / 4.0, lambda v: v / 4.0, lambda v: 0.25),
        "rdiv": (lambda x: 3.0 / x, lambda v: 3.0 / v, lambda v: -3.0 / v ** 2),
        "exp": (lambda x: x.exp(), math.exp, math.exp),
        "log": (lambda x: x.log(), math.log, lambda v: 1 / v),
        "tanh": (lambda x: x.tanh(), math.tanh, lambda v: 1 - math.tanh(v) ** 2),
        "relu_pos": (lambda x: x.relu(), lambda v: max(v, 0), lambda v: 1.0),
        "relu_neg": (lambda x: (-x).relu(), lambda v: max(-v, 0), lambda v: 0.0),
    }
    f, value, deriv = cases[name]
    x = nb.Value(x0)
    y = f(x)
    assert isinstance(y, nb.Value) and np.isclose(y.data, value(x0)), f"{name}: value should be {value(x0):.6f}, got {y.data}"
    _sweep(y)
    assert np.isclose(x.grad, deriv(x0)), f"{name}: derivative at x = {x0} should be {deriv(x0):.6f}, got {x.grad}"


def test_stage2_value_over_value(nb):
    a, b = nb.Value(3.0), nb.Value(-2.0)
    q = a / b
    _sweep(q)
    assert np.isclose(q.data, -1.5) and np.isclose(a.grad, -0.5) and np.isclose(b.grad, -3.0 / 4.0), (
        f"q = a / b: dq/da = 1/b = -0.5, dq/db = -a/b^2 = -0.75; got {a.grad}, {b.grad}")
    d = a - b
    _sweep(d)
    assert d.data == 5.0
    with pytest.raises(TypeError):
        a ** b
    z = nb.Value(0.0)
    r = z.relu()
    _sweep(r)
    assert r.data == 0.0 and z.grad == 0.0, f"relu at exactly 0: value 0 and derivative 0 (the convention), got {z.grad}"


# ---------------------------------------------------------------- stage 3: topological order and backward


def test_stage3_topological_order(nb):
    a, b = nb.Value(1.0), nb.Value(2.0)
    c = a * b
    d = c + a
    e = d * c
    order = bounded(nb.topological_order, e, what="topological_order", hint=_HINT)
    ids = [id(v) for v in order]
    assert len(ids) == len(set(ids)), "each Value appears once, even if it's used several times (a and c are)"
    assert set(ids) == {id(v) for v in (a, b, c, d, e)}, "the order contains every Value e depends on, and e itself"
    pos = {id(v): i for i, v in enumerate(order)}
    for child in order:
        for p in child._prev:
            assert pos[id(p)] < pos[id(child)], "every parent must come before its children"
    assert order[-1] is e, "the root comes last"


def test_stage3_worked_example(nb):
    a, b = nb.Value(2.0), nb.Value(-3.0)
    c = a * b
    d = c + a
    L = d.tanh()
    _backward(L)
    t = math.tanh(-4.0)
    assert np.isclose(L.data, t), f"L = tanh(a b + a) = tanh(-4) = {t:.6f}"
    assert np.isclose(L.grad, 1.0) and np.isclose(d.grad, 1 - t * t), "backward starts with dL/dL = 1, then dL/dd = 1 - tanh^2"
    assert np.isclose(a.grad, (1 - t * t) * (b.data + 1)) and np.isclose(b.grad, (1 - t * t) * a.data), (
        f"dL/da = (1 - tanh^2)(b + 1) = {(1 - t * t) * -2:.6f} (a is used twice), dL/db = (1 - tanh^2) a = "
        f"{(1 - t * t) * 2:.6f}; got {a.grad}, {b.grad}")


def test_stage3_matches_finite_differences(nb):
    fns = [
        (lambda v: (v[0] * v[1] + v[2].exp()).tanh() * v[0], lambda u: math.tanh(u[0] * u[1] + math.exp(u[2])) * u[0]),
        (lambda v: (v[0] ** 2 + v[1] ** 2).log() / (1 + v[2].relu()), lambda u: math.log(u[0] ** 2 + u[1] ** 2) / (1 + max(u[2], 0))),
        (lambda v: v[0] * v[0] * v[0] - 2 / v[1] + v[2] * v[0], lambda u: u[0] ** 3 - 2 / u[1] + u[2] * u[0]),
    ]
    rng = np.random.default_rng(0)
    for f, f_np in fns:
        x = rng.uniform(0.5, 1.5, size=3)
        _, g = _grad_of(nb, f, x)
        g_fd = numerical_gradient(lambda u: f_np(u), x)
        assert np.allclose(g, g_fd, atol=1e-7), f"gradient {g} disagrees with finite differences {g_fd}"


def test_stage3_backward_twice_accumulates_leaves(nb):
    a, b = nb.Value(2.0), nb.Value(5.0)
    c = a * b
    L = c * c  # (ab)^2: dL/da = 2 a b^2 = 100
    _backward(L)
    _backward(L)
    assert np.isclose(a.grad, 200.0), (
        f"two backward calls ADD into the leaves (100 + 100 = 200), like PyTorch; got {a.grad}. if you got more, the "
        "intermediate nodes kept their old grads: zero every non-leaf node's grad at the start of backward")


def test_stage3_deep_chain(nb):
    x = nb.Value(0.5)
    y = x
    for _ in range(5000):
        y = y * 1.0 + 0.0
    _backward(y)
    assert np.isclose(x.grad, 1.0), f"d/dx through a chain of 10,000 operations is 1, got {x.grad}"
    z = nb.Value(1.0)
    y = z
    for _ in range(40):
        y = y + y  # every node is used twice: without a seen set, the walk would visit 2^40 paths
    _backward(y)
    assert np.isclose(z.grad, 2.0 ** 40), f"y = z doubled 40 times: dy/dz = 2^40, got {z.grad}"


# ---------------------------------------------------------------- stage 4: convenience functions


def test_stage4_value_and_grad(nb):
    def f(v):
        return (v[0] * v[1]).tanh() + v[1] * v[2] - v[0] ** 2

    x = np.array([0.3, -1.2, 2.0])
    val, g = bounded(nb.value_and_grad, f, x, what="value_and_grad")
    want = math.tanh(0.3 * -1.2) + -1.2 * 2.0 - 0.09
    assert isinstance(val, float) and np.isclose(val, want), f"the value is f(x) as a float ({want:.6f}), got {val!r}"
    assert isinstance(g, np.ndarray) and g.shape == (3,), "the gradient is an ndarray shaped like x"
    s = 1 - math.tanh(-0.36) ** 2
    assert np.allclose(g, [s * -1.2 - 0.6, s * 0.3 + 2.0, -1.2]), f"gradient: got {g}"


def test_stage4_graph_size(nb):
    a, b = nb.Value(1.0), nb.Value(2.0)
    e = (a * b + a).tanh()
    assert bounded(nb.graph_size, e, what="graph_size") == 5, "a, b, a * b, + and tanh: 5 distinct Values"
    f = a * 3.0
    assert nb.graph_size(f) == 3, "the constant 3.0 becomes a Value too: a, 3.0, and the product"
