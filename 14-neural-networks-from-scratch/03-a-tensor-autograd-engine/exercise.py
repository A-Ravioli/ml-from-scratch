import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.numgrad import numerical_gradient  # yours, from ch 02 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # a tensor autograd engine · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. topic 02's `Value` differentiated one
    number at a time; here every node holds a whole NumPy array, every operation knows its
    **vector-Jacobian product**, and one backward pass gives the gradient of a loss with
    respect to millions of numbers. everything here becomes **`mlfs.tensor`**: your
    `mlfs.nn` (topic 04), the rest of this chapter, the capstone's MNIST network and
    chapter 15 all run on it.

    | stage | you build |
    |---|---|
    | 1 | `unbroadcast`: sum a gradient back down to an operand's shape |
    | 2 | `Tensor`: `from_op`, `backward`, and arithmetic (`+ - * / **`, unary `-`) with broadcasting |
    | 3 | `Tensor`: `exp`, `log`, `tanh`, `sigmoid`, `relu` and the reductions `sum`, `mean`, `max`, `logsumexp` |
    | 4 | `Tensor`: `@`, `reshape`, `transpose`, indexing; `concatenate`, `stack`, `where` |
    | 5 | `no_grad` and `gradcheck` |

    stages 2–4 are all methods of the `Tensor` class, in one cell below. fill in each
    stage's methods (their docstrings start with "stage N"), check them, then come back
    for the next stage.

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def unbroadcast(grad, shape):
    """stage 1: sum grad (a broadcast result's shape) down to `shape`: drop leading axes, sum axes where shape had 1."""
    raise NotImplementedError("stage1: implement unbroadcast")


@app.cell(hide_code=True)
def _():
    checks.run(1, unbroadcast=unbroadcast)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def is_grad_enabled():
    """True unless we're inside a `with no_grad():` block (read by Tensor.from_op)."""
    return no_grad._enabled


@app.class_definition
class Tensor:
    """a float64 NumPy array that remembers how it was computed, so backward() can differentiate through it."""

    __array_ufunc__ = None  # ndarray (op) Tensor defers to Tensor's reflected operators

    def __init__(self, data, requires_grad=False):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents = ()   # the Tensors this one was computed from (empty for leaves)
        self._vjp = None     # g -> one gradient (or None) per parent

    # ---------------------------------------------------------------- plumbing (given)

    @staticmethod
    def _lift(x):
        """x as a Tensor: Tensors pass through, anything else becomes a constant."""
        return x if isinstance(x, Tensor) else Tensor(x)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.transpose()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        flag = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4, threshold=20)}{flag})"

    def detach(self):
        """a new constant Tensor sharing this one's data, cut off from the graph."""
        out = Tensor.__new__(Tensor)
        out.data, out.grad, out.requires_grad, out._parents, out._vjp = self.data, None, False, (), None
        return out

    def numpy(self):
        """the data as an ndarray (the array itself, not a copy)."""
        return self.data

    def item(self):
        """the value of a one-element Tensor as a Python float."""
        return float(self.data.item())

    # ---------------------------------------------------------------- stage 2: the graph

    @staticmethod
    def from_op(data, parents, vjp):
        """stage 2: a new Tensor holding `data`; it records `parents` and `vjp` only if grad is enabled and a parent requires grad."""
        raise NotImplementedError("stage2: implement Tensor.from_op")

    def backward(self, grad=None):
        """stage 2: reverse-mode sweep from self in reverse topological order; ACCUMULATE (+=) into the .grad of leaves."""
        raise NotImplementedError("stage2: implement Tensor.backward")

    # ---------------------------------------------------------------- stage 2: arithmetic

    def __add__(self, other):
        """stage 2: self + other, broadcasting like NumPy."""
        raise NotImplementedError("stage2: implement Tensor.__add__")

    def __radd__(self, other):
        return Tensor._lift(other) + self

    def __sub__(self, other):
        """stage 2: self - other."""
        raise NotImplementedError("stage2: implement Tensor.__sub__")

    def __rsub__(self, other):
        return Tensor._lift(other) - self

    def __mul__(self, other):
        """stage 2: self * other (elementwise)."""
        raise NotImplementedError("stage2: implement Tensor.__mul__")

    def __rmul__(self, other):
        return Tensor._lift(other) * self

    def __truediv__(self, other):
        """stage 2: self / other (elementwise)."""
        raise NotImplementedError("stage2: implement Tensor.__truediv__")

    def __rtruediv__(self, other):
        return Tensor._lift(other) / self

    def __neg__(self):
        """stage 2: -self."""
        raise NotImplementedError("stage2: implement Tensor.__neg__")

    def __pow__(self, exponent):
        """stage 2: self ** exponent for a Python/NumPy scalar exponent (a Tensor exponent is a TypeError)."""
        raise NotImplementedError("stage2: implement Tensor.__pow__")

    # ---------------------------------------------------------------- stage 3: elementwise functions

    def exp(self):
        """stage 3: e ** self."""
        raise NotImplementedError("stage3: implement Tensor.exp")

    def log(self):
        """stage 3: natural log."""
        raise NotImplementedError("stage3: implement Tensor.log")

    def tanh(self):
        """stage 3: hyperbolic tangent."""
        raise NotImplementedError("stage3: implement Tensor.tanh")

    def sigmoid(self):
        """stage 3: 1 / (1 + e^-x), without overflow for large |x|."""
        raise NotImplementedError("stage3: implement Tensor.sigmoid")

    def relu(self):
        """stage 3: max(x, 0); its gradient at exactly 0 is 0 (like PyTorch)."""
        raise NotImplementedError("stage3: implement Tensor.relu")

    # ---------------------------------------------------------------- stage 2 (sum) and stage 3: reductions

    @staticmethod
    def _expand(g, shape, axis, keepdims):
        """broadcast a reduction's output gradient g back to the input `shape` (given)."""
        g = np.asarray(g, dtype=np.float64)
        if axis is not None and not keepdims:
            axes = (axis,) if np.ndim(axis) == 0 else tuple(axis)
            g = np.expand_dims(g, tuple(a % len(shape) for a in axes))
        return np.broadcast_to(g, shape)

    def sum(self, axis=None, keepdims=False):
        """stage 2: sum over `axis` (None = all), like ndarray.sum (you need it to get a scalar loss)."""
        raise NotImplementedError("stage2: implement Tensor.sum")

    def mean(self, axis=None, keepdims=False):
        """stage 3: mean over `axis` (None = all)."""
        raise NotImplementedError("stage3: implement Tensor.mean")

    def max(self, axis=None, keepdims=False):
        """stage 3: max over `axis`; the gradient is split evenly among tied maxima."""
        raise NotImplementedError("stage3: implement Tensor.max")

    def logsumexp(self, axis=-1, keepdims=False):
        """stage 3: log(sum(exp(x))) over `axis`, shifted by the max so it never overflows."""
        raise NotImplementedError("stage3: implement Tensor.logsumexp")

    # ---------------------------------------------------------------- stage 4: linear algebra and shapes

    def __matmul__(self, other):
        """stage 4: self @ other with np.matmul semantics (1-D operands, batched N-D with broadcasting)."""
        raise NotImplementedError("stage4: implement Tensor.__matmul__")

    def __rmatmul__(self, other):
        return Tensor._lift(other) @ self

    def reshape(self, *shape):
        """stage 4: reshape (accepts t.reshape(2, 3) and t.reshape((2, 3)), -1 allowed)."""
        raise NotImplementedError("stage4: implement Tensor.reshape")

    def transpose(self, *axes):
        """stage 4: permute axes (no arguments reverses them, like ndarray.transpose)."""
        raise NotImplementedError("stage4: implement Tensor.transpose")

    def __getitem__(self, idx):
        """stage 4: basic and integer-array indexing; backward scatters with np.add.at (repeats add up)."""
        raise NotImplementedError("stage4: implement Tensor.__getitem__")


@app.cell(hide_code=True)
def _():
    checks.run(2, Tensor=Tensor, unbroadcast=unbroadcast, is_grad_enabled=is_grad_enabled)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, Tensor=Tensor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def concatenate(tensors, axis=0):
    """stage 4: join Tensors (or arrays, as constants) along an existing axis, like np.concatenate."""
    raise NotImplementedError("stage4: implement concatenate")


@app.function
def stack(tensors, axis=0):
    """stage 4: join same-shape Tensors along a new axis, like np.stack."""
    raise NotImplementedError("stage4: implement stack")


@app.function
def where(condition, x, y):
    """stage 4: x where condition (a bool array) is True, else y; each branch gets gradient only where chosen."""
    raise NotImplementedError("stage4: implement where")


@app.cell(hide_code=True)
def _():
    checks.run(4, Tensor=Tensor, concatenate=concatenate, stack=stack, where=where)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class no_grad:
    """context manager: inside `with no_grad():` new operations record no graph (and results don't require grad)."""

    _enabled = True  # the global switch is_grad_enabled() reads; a class attribute, shared by every no_grad

    def __enter__(self):
        """stage 5: remember the current switch, then turn recording off."""
        raise NotImplementedError("stage5: implement no_grad.__enter__")

    def __exit__(self, *exc):
        """stage 5: restore the switch you remembered (so nested blocks work), even after an exception."""
        raise NotImplementedError("stage5: implement no_grad.__exit__")


@app.function
def gradcheck(f, inputs, *, eps=1e-6, atol=1e-5, rtol=1e-3):
    """stage 5: compare backward's gradients of f(*inputs) with central differences; True if every input with requires_grad matches."""
    raise NotImplementedError("stage5: implement gradcheck")


@app.cell(hide_code=True)
def _():
    checks.run(5, Tensor=Tensor, no_grad=no_grad, is_grad_enabled=is_grad_enabled, gradcheck=gradcheck)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.tensor import Tensor` gives you this engine.
    """)
    return


if __name__ == "__main__":
    app.run()
