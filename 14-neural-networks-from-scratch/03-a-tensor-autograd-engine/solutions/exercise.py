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
    # a tensor autograd engine · build (reference solution)

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
    grad = np.asarray(grad, dtype=np.float64)
    shape = tuple(shape)
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


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
        out = Tensor(data)
        parents = tuple(parents)
        if is_grad_enabled() and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._vjp = vjp
        return out

    def backward(self, grad=None):
        """stage 2: reverse-mode sweep from self in reverse topological order; ACCUMULATE (+=) into the .grad of leaves."""
        if not self.requires_grad:
            raise RuntimeError("backward() on a Tensor that doesn't require grad: no input of this "
                               "computation has requires_grad=True (or it ran inside no_grad)")
        if grad is None:
            if self.data.size != 1:
                raise ValueError(f"backward() without a gradient needs a one-element Tensor, got shape "
                                 f"{self.shape}: reduce to a scalar first (e.g. .sum()), or pass grad")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.shape:
            raise ValueError(f"grad has shape {grad.shape}, but this Tensor has shape {self.shape}")

        # topological order, parents before children, without recursion (graphs can be deep)
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))

        grads = {id(self): grad}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if not node._parents:  # a leaf: accumulate
                node.grad = np.array(g, dtype=np.float64) if node.grad is None else node.grad + g
                continue
            for p, gp in zip(node._parents, node._vjp(g)):
                if gp is None or not p.requires_grad:
                    continue
                grads[id(p)] = grads[id(p)] + gp if id(p) in grads else gp

    # ---------------------------------------------------------------- stage 2: arithmetic

    def __add__(self, other):
        """stage 2: self + other, broadcasting like NumPy."""
        other = Tensor._lift(other)
        a, b = self.shape, other.shape
        return Tensor.from_op(self.data + other.data, (self, other),
                              lambda g: (unbroadcast(g, a), unbroadcast(g, b)))

    def __radd__(self, other):
        return Tensor._lift(other) + self

    def __sub__(self, other):
        """stage 2: self - other."""
        other = Tensor._lift(other)
        a, b = self.shape, other.shape
        return Tensor.from_op(self.data - other.data, (self, other),
                              lambda g: (unbroadcast(g, a), unbroadcast(-g, b)))

    def __rsub__(self, other):
        return Tensor._lift(other) - self

    def __mul__(self, other):
        """stage 2: self * other (elementwise)."""
        other = Tensor._lift(other)
        x, y = self.data, other.data
        return Tensor.from_op(x * y, (self, other),
                              lambda g: (unbroadcast(g * y, x.shape), unbroadcast(g * x, y.shape)))

    def __rmul__(self, other):
        return Tensor._lift(other) * self

    def __truediv__(self, other):
        """stage 2: self / other (elementwise)."""
        other = Tensor._lift(other)
        x, y = self.data, other.data
        return Tensor.from_op(x / y, (self, other),
                              lambda g: (unbroadcast(g / y, x.shape), unbroadcast(-g * x / (y * y), y.shape)))

    def __rtruediv__(self, other):
        return Tensor._lift(other) / self

    def __neg__(self):
        """stage 2: -self."""
        return Tensor.from_op(-self.data, (self,), lambda g: (-g,))

    def __pow__(self, exponent):
        """stage 2: self ** exponent for a Python/NumPy scalar exponent (a Tensor exponent is a TypeError)."""
        if isinstance(exponent, Tensor) or np.ndim(exponent) != 0:
            raise TypeError("Tensor ** exponent: the exponent must be a Python or NumPy scalar")
        p = float(exponent)
        x = self.data
        if p == 0:
            return Tensor.from_op(np.ones_like(x), (self,), lambda g: (np.zeros_like(x),))
        return Tensor.from_op(x ** p, (self,), lambda g: (g * p * x ** (p - 1),))

    # ---------------------------------------------------------------- stage 3: elementwise functions

    def exp(self):
        """stage 3: e ** self."""
        out = np.exp(self.data)
        return Tensor.from_op(out, (self,), lambda g: (g * out,))

    def log(self):
        """stage 3: natural log."""
        x = self.data
        return Tensor.from_op(np.log(x), (self,), lambda g: (g / x,))

    def tanh(self):
        """stage 3: hyperbolic tangent."""
        t = np.tanh(self.data)
        return Tensor.from_op(t, (self,), lambda g: (g * (1.0 - t * t),))

    def sigmoid(self):
        """stage 3: 1 / (1 + e^-x), without overflow for large |x|."""
        s = np.exp(-np.logaddexp(0.0, -self.data))
        return Tensor.from_op(s, (self,), lambda g: (g * s * (1.0 - s),))

    def relu(self):
        """stage 3: max(x, 0); its gradient at exactly 0 is 0 (like PyTorch)."""
        x = self.data
        return Tensor.from_op(np.maximum(x, 0.0), (self,), lambda g: (g * (x > 0),))

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
        shape = self.shape
        return Tensor.from_op(self.data.sum(axis=axis, keepdims=keepdims), (self,),
                              lambda g: (Tensor._expand(g, shape, axis, keepdims),))

    def mean(self, axis=None, keepdims=False):
        """stage 3: mean over `axis` (None = all)."""
        shape = self.shape
        out = self.data.mean(axis=axis, keepdims=keepdims)
        n = self.data.size / max(np.size(out), 1)  # how many entries each mean averaged
        return Tensor.from_op(out, (self,), lambda g: (Tensor._expand(g, shape, axis, keepdims) / n,))

    def max(self, axis=None, keepdims=False):
        """stage 3: max over `axis`; the gradient is split evenly among tied maxima."""
        x = self.data
        m = x.max(axis=axis, keepdims=True)
        mask = (x == m).astype(np.float64)
        mask /= mask.sum(axis=axis, keepdims=True)
        out = m if keepdims else (m.reshape(()) if axis is None else np.squeeze(m, axis=axis))
        return Tensor.from_op(out, (self,),
                              lambda g: (Tensor._expand(g, x.shape, axis, keepdims) * mask,))

    def logsumexp(self, axis=-1, keepdims=False):
        """stage 3: log(sum(exp(x))) over `axis`, shifted by the max so it never overflows."""
        x = self.data
        m = x.max(axis=axis, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        lse = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
        out = lse if keepdims else (lse.reshape(()) if axis is None else np.squeeze(lse, axis=axis))
        soft = np.exp(x - lse)
        return Tensor.from_op(out, (self,),
                              lambda g: (Tensor._expand(g, x.shape, axis, keepdims) * soft,))

    # ---------------------------------------------------------------- stage 4: linear algebra and shapes

    def __matmul__(self, other):
        """stage 4: self @ other with np.matmul semantics (1-D operands, batched N-D with broadcasting)."""
        other = Tensor._lift(other)
        A, B = self.data, other.data

        def vjp(g):
            a1, b1 = A.ndim == 1, B.ndim == 1
            A2 = A[None, :] if a1 else A
            B2 = B[:, None] if b1 else B
            G = g
            if a1 and b1:
                G = np.reshape(g, (1, 1))
            elif a1:
                G = np.expand_dims(g, -2)
            elif b1:
                G = np.expand_dims(g, -1)
            gA = G @ np.swapaxes(B2, -1, -2)
            gB = np.swapaxes(A2, -1, -2) @ G
            if a1:
                gA = gA[..., 0, :]
            if b1:
                gB = gB[..., :, 0]
            return unbroadcast(gA, A.shape), unbroadcast(gB, B.shape)

        return Tensor.from_op(A @ B, (self, other), vjp)

    def __rmatmul__(self, other):
        return Tensor._lift(other) @ self

    def reshape(self, *shape):
        """stage 4: reshape (accepts t.reshape(2, 3) and t.reshape((2, 3)), -1 allowed)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.shape
        return Tensor.from_op(self.data.reshape(shape), (self,), lambda g: (np.reshape(g, old),))

    def transpose(self, *axes):
        """stage 4: permute axes (no arguments reverses them, like ndarray.transpose)."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(axes))
        return Tensor.from_op(self.data.transpose(axes), (self,), lambda g: (np.transpose(g, inverse),))

    def __getitem__(self, idx):
        """stage 4: basic and integer-array indexing; backward scatters with np.add.at (repeats add up)."""
        if isinstance(idx, Tensor):
            idx = idx.data.astype(np.int64)
        elif isinstance(idx, tuple):
            idx = tuple(i.data.astype(np.int64) if isinstance(i, Tensor) else i for i in idx)
        x = self.data

        def vjp(g):
            full = np.zeros_like(x)
            np.add.at(full, idx, g)
            return (full,)

        return Tensor.from_op(x[idx], (self,), vjp)


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
    ts = [Tensor._lift(t) for t in tensors]
    sizes = [t.shape[axis] for t in ts]
    cuts = np.cumsum(sizes)[:-1]
    return Tensor.from_op(np.concatenate([t.data for t in ts], axis=axis), ts,
                          lambda g: tuple(np.split(g, cuts, axis=axis)))


@app.function
def stack(tensors, axis=0):
    """stage 4: join same-shape Tensors along a new axis, like np.stack."""
    ts = [Tensor._lift(t) for t in tensors]
    out = np.stack([t.data for t in ts], axis=axis)
    ax = axis % out.ndim
    return Tensor.from_op(out, ts, lambda g: tuple(np.take(g, i, axis=ax) for i in range(len(ts))))


@app.function
def where(condition, x, y):
    """stage 4: x where condition (a bool array) is True, else y; each branch gets gradient only where chosen."""
    cond = np.asarray(condition.data if isinstance(condition, Tensor) else condition, dtype=bool)
    x, y = Tensor._lift(x), Tensor._lift(y)
    return Tensor.from_op(np.where(cond, x.data, y.data), (x, y),
                          lambda g: (unbroadcast(np.where(cond, g, 0.0), x.shape),
                                     unbroadcast(np.where(cond, 0.0, g), y.shape)))


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
        self._prev = no_grad._enabled
        no_grad._enabled = False
        return self

    def __exit__(self, *exc):
        """stage 5: restore the switch you remembered (so nested blocks work), even after an exception."""
        no_grad._enabled = self._prev
        return False


@app.function
def gradcheck(f, inputs, *, eps=1e-6, atol=1e-5, rtol=1e-3):
    """stage 5: compare backward's gradients of f(*inputs) with central differences; True if every input with requires_grad matches."""
    inputs = list(inputs)
    out = f(*inputs)
    weights = np.random.default_rng(0).normal(size=out.shape)  # a fixed random projection to a scalar
    saved = [getattr(x, "grad", None) for x in inputs]
    for x in inputs:
        if isinstance(x, Tensor):
            x.grad = None
    loss = (out * weights).sum()
    if loss.requires_grad:
        loss.backward()
    ok = True
    try:
        for i, x in enumerate(inputs):
            if not (isinstance(x, Tensor) and x.requires_grad):
                continue
            analytic = x.grad if x.grad is not None else np.zeros_like(x.data)

            def scalar(arr, x=x):
                old = x.data
                x.data = arr
                try:
                    with no_grad():
                        return float((np.asarray(f(*inputs).data) * weights).sum())
                finally:
                    x.data = old

            numeric = numerical_gradient(scalar, x.data, h=eps)
            if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
                excess = np.abs(analytic - numeric) - (atol + rtol * np.abs(numeric))
                worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(excess)), x.shape)) if x.ndim else ()
                print(f"gradcheck: input {i} (shape {x.shape}) fails worst at index {worst}: "
                      f"backward gives {analytic[worst]:.6g}, finite differences {numeric[worst]:.6g}")
                ok = False
    finally:
        for x, g in zip(inputs, saved):
            if isinstance(x, Tensor):
                x.grad = g
    return ok


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
