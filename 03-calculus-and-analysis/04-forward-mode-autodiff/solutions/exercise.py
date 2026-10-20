import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # forward-mode autodiff · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. everything here becomes the second half
    of **`mlfs.autodiff`**: exact derivatives, gradients, Jacobians and Jacobian-vector
    products for any function written with ordinary arithmetic and NumPy's elementary
    functions. the capstone builds a Newton solver on top of it, and chapter 12's HMC
    sampler uses your `grad`.

    | stage | you build |
    |---|---|
    | 1 | `Dual`: arithmetic on dual numbers $a + b\varepsilon$ with $\varepsilon^2 = 0$ |
    | 2 | `Dual`: `exp`, `log`, `log1p`, `expm1`, `sin`, `cos`, `tanh`, `sqrt`, powers and `abs` |
    | 3 | `dual_parts`, `derivative`, `value_and_grad`, `grad` |
    | 4 | `jvp`, `jacobian` |

    stages 1 and 2 are both methods of the `Dual` class, in the cell below. fill in the
    stage 1 methods first, check them, then come back for stage 2.

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Dual:
    """a dual number real + dual·ε, with ε² = 0.

    f(a + bε) = f(a) + f'(a) b ε, so evaluating a function on Dual(x, 1) computes f(x) in
    .real and f'(x) in .dual. .real and .dual may themselves be Duals: nesting gives second
    derivatives (lesson, idea 2). so inside these methods use only arithmetic and
    np.exp / np.log / ... (never math.exp), and never call float() on the parts.
    when the other operand of +, -, *, /, ** is a NumPy array, return NotImplemented, so that
    NumPy applies the operation entry by entry (Dual * array -> object array of Duals).
    """

    def __init__(self, real, dual=0.0):
        """stage 1: store the two parts as the attributes .real and .dual (unchanged)."""
        self.real = real
        self.dual = dual

    def __repr__(self):
        return f"Dual({self.real!r}, {self.dual!r})"

    def __bool__(self):
        # truthiness follows the value, like comparisons: `if d:` is False exactly when d.real == 0
        return bool(self.real)

    # ------------------------------------------------------------ stage 1: arithmetic

    def __add__(self, other):
        """stage 1: self + other, where other is a Dual or a plain number. returns a new Dual."""
        if isinstance(other, np.ndarray):
            return NotImplemented  # let NumPy apply the operation entry by entry
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        return Dual(self.real + other, self.dual)

    def __radd__(self, other):
        """stage 1: other + self, for a plain number other (Python calls this for 2 + d)."""
        return Dual(other + self.real, self.dual)

    def __sub__(self, other):
        """stage 1: self - other."""
        if isinstance(other, np.ndarray):
            return NotImplemented  # let NumPy apply the operation entry by entry
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        return Dual(self.real - other, self.dual)

    def __rsub__(self, other):
        """stage 1: other - self, for a plain number other."""
        return Dual(other - self.real, -self.dual)

    def __mul__(self, other):
        """stage 1: self * other: (a + bε)(c + dε) = ac + (ad + bc)ε."""
        if isinstance(other, np.ndarray):
            return NotImplemented  # let NumPy apply the operation entry by entry
        if isinstance(other, Dual):
            return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)
        return Dual(self.real * other, self.dual * other)

    def __rmul__(self, other):
        """stage 1: other * self, for a plain number other."""
        return Dual(other * self.real, other * self.dual)

    def __truediv__(self, other):
        """stage 1: self / other: (a + bε)/(c + dε) = a/c + (bc - ad)/c² ε."""
        if isinstance(other, np.ndarray):
            return NotImplemented  # let NumPy apply the operation entry by entry
        if isinstance(other, Dual):
            return Dual(self.real / other.real,
                        (self.dual * other.real - self.real * other.dual) / (other.real * other.real))
        return Dual(self.real / other, self.dual / other)

    def __rtruediv__(self, other):
        """stage 1: other / self, for a plain number other: c/(a + bε) = c/a - c b/a² ε."""
        return Dual(other / self.real, -other * self.dual / (self.real * self.real))

    def __neg__(self):
        """stage 1: -self."""
        return Dual(-self.real, -self.dual)

    def __lt__(self, other):
        """stage 1: comparisons look only at the real parts: self.real < other's real part."""
        return self.real < (other.real if isinstance(other, Dual) else other)

    def __le__(self, other):
        """stage 1: self.real <= other's real part."""
        return self.real <= (other.real if isinstance(other, Dual) else other)

    def __gt__(self, other):
        """stage 1: self.real > other's real part."""
        return self.real > (other.real if isinstance(other, Dual) else other)

    def __ge__(self, other):
        """stage 1: self.real >= other's real part."""
        return self.real >= (other.real if isinstance(other, Dual) else other)

    def __eq__(self, other):
        """stage 1: self.real == other's real part (so Duals aren't hashable)."""
        return self.real == (other.real if isinstance(other, Dual) else other)

    # ------------------------------------------------------------ stage 2: elementary functions

    def exp(self):
        """stage 2: exp(a + bε) = exp(a) + exp(a) b ε. (np.exp(d) calls this method.)"""
        e = np.exp(self.real)
        return Dual(e, e * self.dual)

    def log(self):
        """stage 2: log(a + bε) = log(a) + (b / a) ε, for a > 0."""
        return Dual(np.log(self.real), self.dual / self.real)

    def sin(self):
        """stage 2: sin(a + bε) = sin(a) + cos(a) b ε."""
        return Dual(np.sin(self.real), np.cos(self.real) * self.dual)

    def cos(self):
        """stage 2: cos(a + bε) = cos(a) - sin(a) b ε."""
        return Dual(np.cos(self.real), -np.sin(self.real) * self.dual)

    def tanh(self):
        """stage 2: tanh(a + bε) = tanh(a) + (1 - tanh(a)²) b ε."""
        t = np.tanh(self.real)
        return Dual(t, (1 - t * t) * self.dual)

    def log1p(self):
        """stage 2: log(1 + a) + b / (1 + a) ε, for a > -1. (np.log1p(d) calls this method.)"""
        return Dual(np.log1p(self.real), self.dual / (1 + self.real))

    def expm1(self):
        """stage 2: (exp(a) - 1) + exp(a) b ε. (np.expm1(d) calls this method.)"""
        return Dual(np.expm1(self.real), np.exp(self.real) * self.dual)

    def sqrt(self):
        """stage 2: sqrt(a + bε) = sqrt(a) + b / (2 sqrt(a)) ε, for a > 0."""
        s = np.sqrt(self.real)
        return Dual(s, self.dual / (2 * s))

    def __pow__(self, p):
        """stage 2: self ** p. for a plain number p: a^p + p a^(p-1) b ε (and exactly
        Dual(1, 0) when p == 0). for a Dual p: exp(p log(self)), which needs self.real > 0.
        for a NumPy array p, return NotImplemented. (at a = 0 with p < 1 the derivative is
        infinite and Python raises ZeroDivisionError: that's fine, the derivative doesn't exist.)
        """
        if isinstance(p, np.ndarray):
            return NotImplemented
        if isinstance(p, Dual):
            return (p * self.log()).exp()
        if p == 0:
            return Dual(self.real ** 0, 0.0 * self.dual)
        return Dual(self.real ** p, p * self.real ** (p - 1) * self.dual)

    def __rpow__(self, c):
        """stage 2: c ** self for a plain number c > 0: c^a + c^a log(c) b ε."""
        v = c ** self.real
        return Dual(v, v * np.log(c) * self.dual)

    def __abs__(self):
        """stage 2: abs(self): -self if self.real < 0, else a copy of self (derivative +1 at 0)."""
        return -self if self.real < 0 else Dual(self.real, self.dual)


@app.cell(hide_code=True)
def _():
    checks.run(1, Dual=Dual)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *stage 2's methods (`exp`, `log`, `log1p`, `expm1`, `sin`, `cos`, `tanh`, `sqrt`,
    `__pow__`, `__rpow__`, `__abs__`) are in the `Dual` class above.*
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(2, Dual=Dual)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def dual_parts(y):
    """stage 3: split y into (real parts, dual parts).

    y is a Dual, a plain number (its dual part is 0), or an array/list of Duals and numbers
    (any shape, usually dtype=object). returns (float, float) for a scalar y, or two float
    arrays of y's shape.
    """
    if isinstance(y, Dual):
        return float(y.real), float(y.dual)
    if isinstance(y, (list, tuple)):
        y = np.array(y, dtype=object)
    if isinstance(y, np.ndarray):
        if y.dtype != object:
            return y.astype(float), np.zeros(y.shape)
        parts = [dual_parts(v) for v in y.ravel()]
        real = np.array([p[0] for p in parts], dtype=float).reshape(y.shape)
        dual = np.array([p[1] for p in parts], dtype=float).reshape(y.shape)
        return real, dual
    return float(y), 0.0


@app.function
def derivative(f, x: float) -> float:
    """stage 3: f'(x) for f: R -> R built from Dual-aware operations, in one evaluation of f.

    evaluate f at Dual(x, 1.0) and return the dual part as a float.
    """
    return dual_parts(f(Dual(float(x), 1.0)))[1]


@app.function
def value_and_grad(f, x: np.ndarray) -> tuple[float, np.ndarray]:
    """stage 3: (f(x), grad f(x)) for f: (n,) -> float, using exactly n evaluations of f.

    evaluation i passes f an object array of Duals with real parts x and dual parts e_i (one
    seed at a time); its dual part is d f / d x_i. the value comes from the real part of the
    first evaluation. x is a float array (n,) (any shape works: the gradient has x's shape).
    """
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    n = flat.size
    if n == 0:
        return dual_parts(f(x))[0], np.zeros(x.shape)
    g = np.empty(n)
    value = None
    for i in range(n):
        xd = np.empty(n, dtype=object)
        for j in range(n):
            xd[j] = Dual(float(flat[j]), 1.0 if j == i else 0.0)
        val, g[i] = dual_parts(f(xd.reshape(x.shape)))
        if value is None:
            value = val
    return float(value), g.reshape(x.shape)


@app.function
def grad(f, x: np.ndarray) -> np.ndarray:
    """stage 3: grad f(x) for f: (n,) -> float, shape (n,), using n forward passes."""
    return value_and_grad(f, x)[1]


@app.cell(hide_code=True)
def _():
    checks.run(3, Dual=Dual, dual_parts=dual_parts, derivative=derivative,
               value_and_grad=value_and_grad, grad=grad)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def jvp(f, x: np.ndarray, v: np.ndarray):
    """stage 4: (f(x), J(x) @ v) in ONE evaluation of f, by seeding the dual parts with v.

    f: (n,) -> (m,) or float; x, v float arrays (n,). returns two float arrays (m,), or two
    floats if f is scalar-valued (then the second is grad f(x) . v).
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    xd = np.empty(x.size, dtype=object)
    for j, (a, b) in enumerate(zip(x.ravel(), v.ravel())):
        xd[j] = Dual(float(a), float(b))
    return dual_parts(f(xd.reshape(x.shape)))


@app.function
def jacobian(f, x: np.ndarray) -> np.ndarray:
    """stage 4: the Jacobian of f: (n,) -> (m,) at x, shape (m, n), one jvp per column.

    column j is J(x) @ e_j. a scalar-valued f gives shape (1, n). uses n evaluations of f.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    cols = []
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        cols.append(np.atleast_1d(jvp(f, x, e.reshape(x.shape))[1]).ravel())
    return np.stack(cols, axis=1)


@app.cell(hide_code=True)
def _():
    checks.run(4, Dual=Dual, jvp=jvp, jacobian=jacobian)
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
