import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # scalar reverse-mode autodiff · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll build `Value`, a number that
    records the computation that produced it, and then, with one call to `backward()`,
    computes the derivative of the result with respect to *every* input at once. it's
    about a hundred lines, and it's the complete idea behind PyTorch's autograd; topic 03
    scales it from numbers to arrays.

    | stage | you build |
    |---|---|
    | 1 | `Value`: data, grad, the graph record, `+` and `*` with their local backward rules |
    | 2 | `Value`: `-`, `/`, `**`, `exp`, `log`, `tanh`, `relu` |
    | 3 | `topological_order` and `Value.backward` |
    | 4 | `value_and_grad` and `graph_size` |

    stages 1 and 2 are methods of the `Value` class, in the cell below; `Value.backward` is
    stage 3. fill in each stage's methods (their docstrings start with "stage N"), check
    them, then come back for the next.

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Value:
    """a scalar that remembers how it was computed: .data, .grad, the parents in ._prev and the operation in ._op."""

    def __init__(self, data, _prev=(), _op=""):
        """stage 1: store float(data), grad 0.0, the parents as a tuple ._prev, the op name ._op, and a do-nothing ._backward."""
        raise NotImplementedError("stage1: implement Value.__init__")

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    # ---------------------------------------------------------------- stage 1: + and *

    def __add__(self, other):
        """stage 1: self + other; wrap a number as Value(other) so it's a parent too; out._backward adds out.grad into both parents' .grad."""
        raise NotImplementedError("stage1: implement Value.__add__")

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        """stage 1: self * other; the local derivatives are other.data and self.data."""
        raise NotImplementedError("stage1: implement Value.__mul__")

    def __rmul__(self, other):
        return self * other

    # ---------------------------------------------------------------- stage 2: more operations

    def __neg__(self):
        """stage 2: -self."""
        raise NotImplementedError("stage2: implement Value.__neg__")

    def __sub__(self, other):
        """stage 2: self - other."""
        raise NotImplementedError("stage2: implement Value.__sub__")

    def __rsub__(self, other):
        return Value(other) - self

    def __pow__(self, exponent):
        """stage 2: self ** exponent for a Python int/float exponent; derivative exponent * data ** (exponent - 1)."""
        raise NotImplementedError("stage2: implement Value.__pow__")

    def __truediv__(self, other):
        """stage 2: self / other, as self * other ** -1."""
        raise NotImplementedError("stage2: implement Value.__truediv__")

    def __rtruediv__(self, other):
        return Value(other) / self

    def exp(self):
        """stage 2: e ** self."""
        raise NotImplementedError("stage2: implement Value.exp")

    def log(self):
        """stage 2: natural log, for data > 0."""
        raise NotImplementedError("stage2: implement Value.log")

    def tanh(self):
        """stage 2: tanh; derivative 1 - tanh^2."""
        raise NotImplementedError("stage2: implement Value.tanh")

    def relu(self):
        """stage 2: max(self, 0); derivative 1 where data > 0, else 0."""
        raise NotImplementedError("stage2: implement Value.relu")

    # ---------------------------------------------------------------- stage 3: backward

    def backward(self):
        """stage 3: zero the grads of the non-leaf nodes, set self.grad = 1, run every _backward in reverse topological order."""
        raise NotImplementedError("stage3: implement Value.backward")


@app.cell(hide_code=True)
def _():
    checks.run(1, Value=Value)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.cell(hide_code=True)
def _():
    checks.run(2, Value=Value)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def topological_order(root):
    """stage 3: every Value that root depends on (root included), each once, parents before children; no recursion."""
    raise NotImplementedError("stage3: implement topological_order")


@app.cell(hide_code=True)
def _():
    checks.run(3, Value=Value, topological_order=topological_order)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def value_and_grad(f, x):
    """stage 4: (f(x) as a float, its gradient as an ndarray (n,)) from one forward and one backward pass; f takes a list of n Values."""
    raise NotImplementedError("stage4: implement value_and_grad")


@app.function
def graph_size(root):
    """stage 4: the number of distinct Values in root's graph (inputs and constants included)."""
    raise NotImplementedError("stage4: implement graph_size")


@app.cell(hide_code=True)
def _():
    checks.run(4, Value=Value, value_and_grad=value_and_grad, graph_size=graph_size)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
