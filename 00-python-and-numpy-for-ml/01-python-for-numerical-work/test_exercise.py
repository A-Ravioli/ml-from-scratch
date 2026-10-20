"""Checks for exercise.py (python for numerical work).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import dataclasses
import math

import numpy as np

import pytest

# ---------------------------------------------------------------- stage 1: functions


def test_stage1_dot_hand_example(nb):
    got = nb.dot([1.0, 2.0, 3.0], [4.0, -5.0, 6.0])
    assert got == pytest.approx(12.0), (
        f"dot([1, 2, 3], [4, -5, 6]) should be 1*4 + 2*(-5) + 3*6 = 12, got {got}. "
        "multiply matching entries, then add up all the products.")


def test_stage1_dot_returns_a_python_float(nb):
    got = nb.dot([1, 2], [3, 4])
    assert got == 11, f"dot([1, 2], [3, 4]) should be 11, got {got}"
    assert type(got) is float, (
        f"dot should return a plain Python float, got {type(got).__name__}. with integer inputs "
        "the products are ints: start the running total at 0.0 (or convert with float(...) at the "
        "end). this topic is plain Python: no NumPy (np.dot returns a numpy.float64).")


def test_stage1_dot_works_on_tuples_and_empty(nb):
    assert nb.dot((1.0, 1.0), (2.0, 3.0)) == pytest.approx(5.0), "dot should accept tuples as well as lists"
    got = nb.dot([], [])
    assert got == 0.0 and type(got) is float, f"the dot product of two empty sequences is 0.0, got {got!r}"


def test_stage1_dot_rejects_length_mismatch(nb):
    with pytest.raises(ValueError):
        nb.dot([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        nb.dot([1.0], [1.0, 2.0])  # zip() would silently stop at the shorter one: check first


def test_stage1_norm(nb):
    got = nb.norm([3.0, 4.0])
    assert got == pytest.approx(5.0), (
        f"norm([3, 4]) should be sqrt(9 + 16) = 5, got {got}. "
        + ("you returned the SQUARED length: take the square root." if got == pytest.approx(25.0) else ""))
    assert nb.norm([1.0, 2.0, 2.0]) == pytest.approx(3.0), "norm([1, 2, 2]) should be 3"
    assert nb.norm([-6.0]) == pytest.approx(6.0), "norm([-6]) should be 6: a length is never negative"
    assert type(nb.norm([3, 4])) is float, "norm should return a Python float"


# ---------------------------------------------------------------- stage 2: a container class


def test_stage2_stores_a_tuple_of_floats(nb):
    v = nb.Vector([1, 2, 3])
    assert isinstance(v.values, tuple), (
        f"Vector([1, 2, 3]).values should be a tuple, got {type(v.values).__name__}. a tuple can't be "
        "changed after it's made, which keeps every Vector immutable: use tuple(...).")
    assert v.values == (1.0, 2.0, 3.0) and all(type(x) is float for x in v.values), (
        f"Vector([1, 2, 3]).values should be (1.0, 2.0, 3.0), floats, got {v.values!r}")
    assert nb.Vector(x * x for x in range(3)).values == (0.0, 1.0, 4.0), (
        "Vector should accept any iterable, including a generator expression")


def test_stage2_does_not_alias_the_input_list(nb):
    data = [1.0, 2.0]
    v = nb.Vector(data)
    data[0] = 99.0
    assert v.values == (1.0, 2.0), (
        "changing the list you built a Vector from changed the Vector. store your own tuple copy.")


def test_stage2_len_getitem_iter(nb):
    v = nb.Vector([5, 6, 7])
    assert len(v) == 3, f"len(Vector([5, 6, 7])) should be 3, got {len(v)}"
    assert v[0] == 5.0 and v[2] == 7.0, "v[i] should return component i"
    assert v[-1] == 7.0, "v[-1] should be the last component, as with lists"
    assert list(v) == [5.0, 6.0, 7.0], f"list(v) should be [5.0, 6.0, 7.0], got {list(v)}"
    assert [2 * x for x in v] == [10.0, 12.0, 14.0], "`for x in v` should loop over the components"
    with pytest.raises(IndexError):
        v[3]


def test_stage2_repr(nb):
    got = repr(nb.Vector([1, 2.5]))
    assert got == "Vector([1.0, 2.5])", (
        f"repr(Vector([1, 2.5])) should be exactly 'Vector([1.0, 2.5])', got {got!r}. "
        "use the class name and list(self.values).")
    assert repr(nb.Vector([])) == "Vector([])", "an empty Vector should print as Vector([])"


def test_stage2_equality(nb):
    V = nb.Vector
    assert V([1, 2]) == V([1.0, 2.0]), "Vectors with the same values should be equal"
    assert not (V([1, 2]) == V([1, 3])), "Vectors with different values should not be equal"
    assert not (V([1, 2]) == V([1, 2, 0])), "Vectors of different lengths are never equal"
    assert V([1, 2]) != V([2, 1]), "!= should be the opposite of =="
    result = V([1.0]).__eq__([1.0])
    assert result is NotImplemented, (
        f"comparing a Vector with a non-Vector should RETURN NotImplemented (got {result!r}), so "
        "Python can try the other object's __eq__ before deciding. don't return False yourself.")
    assert not (V([1.0]) == [1.0]), "a Vector should not equal a plain list"


def test_stage2_zeros(nb):
    z = nb.Vector.zeros(3)
    assert isinstance(z, nb.Vector), f"Vector.zeros(3) should return a Vector, got {type(z).__name__}"
    assert z.values == (0.0, 0.0, 0.0), f"Vector.zeros(3) should have values (0.0, 0.0, 0.0), got {z.values!r}"
    assert len(nb.Vector.zeros(0)) == 0, "Vector.zeros(0) should be empty"


# ---------------------------------------------------------------- stage 3: arithmetic


def test_stage3_add_sub_neg(nb):
    V = nb.Vector
    u, v = V([1, 2, 3]), V([10, 20, 30])
    assert u + v == V([11, 22, 33]), f"Vector([1, 2, 3]) + Vector([10, 20, 30]) should be [11, 22, 33], got {u + v!r}"
    assert v - u == V([9, 18, 27]), f"v - u should be [9, 18, 27], got {v - u!r}"
    assert -u == V([-1, -2, -3]), f"-Vector([1, 2, 3]) should be [-1, -2, -3], got {-u!r}"
    assert u == V([1, 2, 3]) and v == V([10, 20, 30]), "u + v must not change u or v"
    assert isinstance(u + v, V), "u + v should return a new Vector"


def test_stage3_add_rejects_bad_operands(nb):
    V = nb.Vector
    with pytest.raises(ValueError):
        V([1, 2]) + V([1, 2, 3])
    with pytest.raises(ValueError):
        V([1, 2]) - V([1])
    got = V([1, 2]).__add__(5)
    assert got is NotImplemented, (
        f"Vector.__add__ with a non-Vector should return NotImplemented, got {got!r}. Python then "
        "raises a TypeError for you, which is the right outcome for `v + 5`.")
    with pytest.raises(TypeError):
        V([1, 2]) + 5
    with pytest.raises(TypeError):
        V([1, 2]) - [1, 2]


def test_stage3_scalar_multiply_and_divide(nb):
    V = nb.Vector
    v = V([1, -2, 4])
    assert v * 3 == V([3, -6, 12]), f"v * 3 should be [3, -6, 12], got {v * 3!r}"
    assert 3 * v == V([3, -6, 12]), (
        "3 * v failed. Python first tries int.__mul__(3, v), which returns NotImplemented, then "
        "calls v.__rmul__(3): define __rmul__.")
    assert v * 0.5 == V([0.5, -1, 2]), "multiplying by a float should work too"
    assert v / 4 == V([0.25, -0.5, 1]), f"v / 4 should be [0.25, -0.5, 1], got {v / 4!r}"
    assert v * np.int64(2) == V([2, -4, 8]) and v / np.float64(2.0) == V([0.5, -1, 2]), (
        "NumPy numbers like np.int64(2) are real numbers too: check isinstance(scalar, numbers.Real) "
        "rather than (int, float), which np.int64 isn't")
    with pytest.raises(ZeroDivisionError):
        v / 0
    with pytest.raises(TypeError):
        v * v  # use @ for the dot product
    with pytest.raises(TypeError):
        v / v


def test_stage3_matmul_is_dot_and_abs_is_norm(nb):
    V = nb.Vector
    got = V([1, 2, 3]) @ V([4, -5, 6])
    assert got == pytest.approx(12.0) and isinstance(got, float), (
        f"Vector([1, 2, 3]) @ Vector([4, -5, 6]) should be the dot product 12.0 (a float), got {got!r}")
    with pytest.raises(ValueError):
        V([1, 2]) @ V([1, 2, 3])
    with pytest.raises(TypeError):
        V([1, 2]) @ 3
    assert abs(V([3, 4])) == pytest.approx(5.0), f"abs(Vector([3, 4])) should be 5.0, got {abs(V([3, 4]))}"


def test_stage3_sum_of_vectors(nb):
    V = nb.Vector
    vs = [V([1, 0]), V([0, 2]), V([3, 3])]
    got = sum(vs)
    assert got == V([4, 5]), (
        f"sum([Vector([1, 0]), Vector([0, 2]), Vector([3, 3])]) should be Vector([4, 5]), got {got!r}. "
        "sum() starts from the number 0 and computes 0 + vs[0] first: your __radd__ must treat 0 "
        "as the zero vector.")
    with pytest.raises(TypeError):
        1 + V([1, 2])  # only 0 is a sensible thing to add to a vector


def test_stage3_expression_mixing_operators(nb):
    V = nb.Vector
    u, v = V([1, 2]), V([3, -1])
    w = 2 * u - v / 2 + -u
    assert w == V([-0.5, 2.5]), f"2*u - v/2 + -u should be [-0.5, 2.5] for u=[1, 2], v=[3, -1], got {w!r}"
    assert abs(u - u) == 0.0, "abs(u - u) should be 0"


# ---------------------------------------------------------------- stage 4: a dataclass model


def test_stage4_is_a_frozen_dataclass(nb):
    V, M = nb.Vector, nb.LinearModel
    assert dataclasses.is_dataclass(M), "LinearModel should be decorated with @dataclass(frozen=True)"
    m = M(V([1, 2]), 0.5)
    assert m.weights == V([1, 2]) and m.bias == 0.5, "LinearModel(weights, bias) should store both fields"
    assert M(V([1, 2])).bias == 0.0, "the bias should default to 0.0"
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.bias = 1.0


def test_stage4_call_predicts(nb):
    V, M = nb.Vector, nb.LinearModel
    m = M(V([1, 2]), 0.5)
    got = m(V([3, 4]))
    assert got == pytest.approx(11.5), (
        f"LinearModel(weights=[1, 2], bias=0.5)([3, 4]) should be 1*3 + 2*4 + 0.5 = 11.5, got {got}. "
        "__call__ is what runs when you write model(x).")


def test_stage4_mse(nb):
    V, M = nb.Vector, nb.LinearModel
    m = M(V([1.0]), 0.0)
    xs, ys = [V([1.0]), V([2.0]), V([3.0])], [1.0, 1.0, 1.0]
    got = m.mse(xs, ys)
    assert got == pytest.approx(5 / 3), (
        f"predictions 1, 2, 3 against targets 1, 1, 1 have squared errors 0, 1, 4, so the MSE is "
        f"5/3 = 1.667; got {got}. "
        + ("you divided by n - 1 instead of n." if got == pytest.approx(2.5) else
           "you returned the SUM, not the mean." if got == pytest.approx(5.0) else ""))
    with pytest.raises(ValueError):
        m.mse(xs, ys[:2])
    with pytest.raises(ValueError):
        m.mse([], [])


def test_stage4_step_hand_example(nb):
    V, M = nb.Vector, nb.LinearModel
    m = M(V([0.0, 0.0]), 0.0)
    xs, ys = [V([1.0, 0.0]), V([0.0, 2.0])], [1.0, 2.0]
    new = m.step(xs, ys, lr=0.1)
    assert isinstance(new, M), f"step should return a new LinearModel, got {type(new).__name__}"
    w, b = list(new.weights), new.bias
    assert w == pytest.approx([0.1, 0.4]) and b == pytest.approx(0.3), (
        f"errors are e = (0 - 1, 0 - 2) = (-1, -2), so grad_w = (2/2)(-1*[1, 0] - 2*[0, 2]) = [-1, -4] "
        f"and grad_b = (2/2)(-3) = -3. with lr = 0.1 the new model is weights [0.1, 0.4], bias 0.3; "
        f"got weights {w}, bias {b}. "
        + ("your step is half as big: did you forget the factor 2?" if w == pytest.approx([0.05, 0.2]) else
           "you moved the wrong way: SUBTRACT lr * gradient." if w == pytest.approx([-0.1, -0.4]) else ""))
    assert m.weights == V([0.0, 0.0]) and m.bias == 0.0, "step must leave the original model unchanged"


def test_stage4_training_recovers_a_line(nb):
    V, M = nb.Vector, nb.LinearModel
    xs = [V([x / 4]) for x in range(-4, 5)]
    ys = [2.0 * x[0] + 1.0 for x in xs]  # y = 2x + 1, no noise
    m = M(V([0.0]), 0.0)
    losses = [m.mse(xs, ys)]
    for _ in range(300):
        m = m.step(xs, ys, lr=0.2)
        losses.append(m.mse(xs, ys))
    assert losses[1] < losses[0], "the loss should go DOWN after one step. check the sign of your update."
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])), "with a small learning rate the loss should never go up"
    assert m.weights[0] == pytest.approx(2.0, abs=1e-3) and m.bias == pytest.approx(1.0, abs=1e-3), (
        f"300 steps on points from y = 2x + 1 should give weight ~2 and bias ~1, got weight "
        f"{m.weights[0]:.4f}, bias {m.bias:.4f}. if the weight is right but the bias isn't, check grad_b.")
    assert math.isfinite(losses[-1]) and losses[-1] < 1e-6, f"final loss should be ~0, got {losses[-1]:.3g}"
