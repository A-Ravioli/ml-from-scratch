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
    # scalar reverse-mode autodiff · build (reference solution)

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
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    # ---------------------------------------------------------------- stage 1: + and *

    def __add__(self, other):
        """stage 1: self + other; wrap a number as Value(other) so it's a parent too; out._backward adds out.grad into both parents' .grad."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        """stage 1: self * other; the local derivatives are other.data and self.data."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # ---------------------------------------------------------------- stage 2: more operations

    def __neg__(self):
        """stage 2: -self."""
        return self * -1.0

    def __sub__(self, other):
        """stage 2: self - other."""
        return self + (-other if isinstance(other, Value) else -float(other))

    def __rsub__(self, other):
        return Value(other) - self

    def __pow__(self, exponent):
        """stage 2: self ** exponent for a Python int/float exponent; derivative exponent * data ** (exponent - 1)."""
        if isinstance(exponent, Value):
            raise TypeError("Value ** Value isn't supported: the exponent must be an int or float")
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            if exponent != 0:
                self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """stage 2: self / other, as self * other ** -1."""
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        return Value(other) / self

    def exp(self):
        """stage 2: e ** self."""
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        """stage 2: natural log, for data > 0."""
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self):
        """stage 2: tanh; derivative 1 - tanh^2."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """stage 2: max(self, 0); derivative 1 where data > 0, else 0."""
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    # ---------------------------------------------------------------- stage 3: backward

    def backward(self):
        """stage 3: zero the grads of the non-leaf nodes, set self.grad = 1, run every _backward in reverse topological order."""
        order = topological_order(self)
        for v in order:
            if v._prev:
                v.grad = 0.0
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


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
    order, seen, stack = [], set(), [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for p in node._prev:
            if id(p) not in seen:
                stack.append((p, False))
    return order


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
    xs = [Value(v) for v in np.asarray(x, dtype=float).ravel()]
    out = f(xs)
    out = out if isinstance(out, Value) else Value(out)
    out.backward()
    return out.data, np.array([v.grad for v in xs])


@app.function
def graph_size(root):
    """stage 4: the number of distinct Values in root's graph (inputs and constants included)."""
    return len(topological_order(root))


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
