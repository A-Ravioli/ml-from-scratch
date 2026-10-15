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
    # forward-mode autodiff · build

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
        raise NotImplementedError("stage1: implement Dual.__init__")

    def __repr__(self):
        return f"Dual({self.real!r}, {self.dual!r})"

    def __bool__(self):
        # truthiness follows the value, like comparisons: `if d:` is False exactly when d.real == 0
        return bool(self.real)

    # ------------------------------------------------------------ stage 1: arithmetic

    def __add__(self, other):
        """stage 1: self + other, where other is a Dual or a plain number. returns a new Dual."""
        raise NotImplementedError("stage1: implement Dual.__add__")

    def __radd__(self, other):
        """stage 1: other + self, for a plain number other (Python calls this for 2 + d)."""
        raise NotImplementedError("stage1: implement Dual.__radd__")

    def __sub__(self, other):
        """stage 1: self - other."""
        raise NotImplementedError("stage1: implement Dual.__sub__")

    def __rsub__(self, other):
        """stage 1: other - self, for a plain number other."""
        raise NotImplementedError("stage1: implement Dual.__rsub__")

    def __mul__(self, other):
        """stage 1: self * other: (a + bε)(c + dε) = ac + (ad + bc)ε."""
        raise NotImplementedError("stage1: implement Dual.__mul__")

    def __rmul__(self, other):
        """stage 1: other * self, for a plain number other."""
        raise NotImplementedError("stage1: implement Dual.__rmul__")

    def __truediv__(self, other):
        """stage 1: self / other: (a + bε)/(c + dε) = a/c + (bc - ad)/c² ε."""
        raise NotImplementedError("stage1: implement Dual.__truediv__")

    def __rtruediv__(self, other):
        """stage 1: other / self, for a plain number other: c/(a + bε) = c/a - c b/a² ε."""
        raise NotImplementedError("stage1: implement Dual.__rtruediv__")

    def __neg__(self):
        """stage 1: -self."""
        raise NotImplementedError("stage1: implement Dual.__neg__")

    def __lt__(self, other):
        """stage 1: comparisons look only at the real parts: self.real < other's real part."""
        raise NotImplementedError("stage1: implement Dual.__lt__")

    def __le__(self, other):
        """stage 1: self.real <= other's real part."""
        raise NotImplementedError("stage1: implement Dual.__le__")

    def __gt__(self, other):
        """stage 1: self.real > other's real part."""
        raise NotImplementedError("stage1: implement Dual.__gt__")

    def __ge__(self, other):
        """stage 1: self.real >= other's real part."""
        raise NotImplementedError("stage1: implement Dual.__ge__")

    def __eq__(self, other):
        """stage 1: self.real == other's real part (so Duals aren't hashable)."""
        raise NotImplementedError("stage1: implement Dual.__eq__")

    # ------------------------------------------------------------ stage 2: elementary functions

    def exp(self):
        """stage 2: exp(a + bε) = exp(a) + exp(a) b ε. (np.exp(d) calls this method.)"""
        raise NotImplementedError("stage2: implement Dual.exp")

    def log(self):
        """stage 2: log(a + bε) = log(a) + (b / a) ε, for a > 0."""
        raise NotImplementedError("stage2: implement Dual.log")

    def sin(self):
        """stage 2: sin(a + bε) = sin(a) + cos(a) b ε."""
        raise NotImplementedError("stage2: implement Dual.sin")

    def cos(self):
        """stage 2: cos(a + bε) = cos(a) - sin(a) b ε."""
        raise NotImplementedError("stage2: implement Dual.cos")

    def tanh(self):
        """stage 2: tanh(a + bε) = tanh(a) + (1 - tanh(a)²) b ε."""
        raise NotImplementedError("stage2: implement Dual.tanh")

    def log1p(self):
        """stage 2: log(1 + a) + b / (1 + a) ε, for a > -1. (np.log1p(d) calls this method.)"""
        raise NotImplementedError("stage2: implement Dual.log1p")

    def expm1(self):
        """stage 2: (exp(a) - 1) + exp(a) b ε. (np.expm1(d) calls this method.)"""
        raise NotImplementedError("stage2: implement Dual.expm1")

    def sqrt(self):
        """stage 2: sqrt(a + bε) = sqrt(a) + b / (2 sqrt(a)) ε, for a > 0."""
        raise NotImplementedError("stage2: implement Dual.sqrt")

    def __pow__(self, p):
        """stage 2: self ** p. for a plain number p: a^p + p a^(p-1) b ε (and exactly
        Dual(1, 0) when p == 0). for a Dual p: exp(p log(self)), which needs self.real > 0.
        for a NumPy array p, return NotImplemented. (at a = 0 with p < 1 the derivative is
        infinite and Python raises ZeroDivisionError: that's fine, the derivative doesn't exist.)
        """
        raise NotImplementedError("stage2: implement Dual.__pow__")

    def __rpow__(self, c):
        """stage 2: c ** self for a plain number c > 0: c^a + c^a log(c) b ε."""
        raise NotImplementedError("stage2: implement Dual.__rpow__")

    def __abs__(self):
        """stage 2: abs(self): -self if self.real < 0, else a copy of self (derivative +1 at 0)."""
        raise NotImplementedError("stage2: implement Dual.__abs__")


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
    raise NotImplementedError("stage3: implement dual_parts")


@app.function
def derivative(f, x: float) -> float:
    """stage 3: f'(x) for f: R -> R built from Dual-aware operations, in one evaluation of f.

    evaluate f at Dual(x, 1.0) and return the dual part as a float.
    """
    raise NotImplementedError("stage3: implement derivative")


@app.function
def value_and_grad(f, x: np.ndarray) -> tuple[float, np.ndarray]:
    """stage 3: (f(x), grad f(x)) for f: (n,) -> float, using exactly n evaluations of f.

    evaluation i passes f an object array of Duals with real parts x and dual parts e_i (one
    seed at a time); its dual part is d f / d x_i. the value comes from the real part of the
    first evaluation. x is a float array (n,) (any shape works: the gradient has x's shape).
    """
    raise NotImplementedError("stage3: implement value_and_grad")


@app.function
def grad(f, x: np.ndarray) -> np.ndarray:
    """stage 3: grad f(x) for f: (n,) -> float, shape (n,), using n forward passes."""
    raise NotImplementedError("stage3: implement grad")


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
    raise NotImplementedError("stage4: implement jvp")


@app.function
def jacobian(f, x: np.ndarray) -> np.ndarray:
    """stage 4: the Jacobian of f: (n,) -> (m,) at x, shape (m, n), one jvp per column.

    column j is J(x) @ e_j. a scalar-valued f gives shape (1, n). uses n evaluations of f.
    """
    raise NotImplementedError("stage4: implement jacobian")


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
