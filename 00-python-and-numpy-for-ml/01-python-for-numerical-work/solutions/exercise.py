import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import numbers
    from collections.abc import Iterable, Sequence
    from dataclasses import dataclass, replace

    import marimo as mo

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # python for numerical work · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    something to write, and checks your code live as you edit. a stage is done when its
    check box turns green.

    this topic is plain Python, with no NumPy yet. you'll write a small `Vector` class that
    behaves like a mathematical vector (`u + v`, `2 * v`, `u @ v`), then use it to build and
    train a tiny linear model. it's a preview of the `Value` and `Tensor` classes you'll
    write in chapter 14, which work exactly the same way underneath.

    | stage | you build |
    |---|---|
    | 1 | `dot`, `norm`: functions on lists of numbers |
    | 2 | `Vector`: a class that stores numbers and prints, compares and indexes like a container |
    | 3 | `Vector` arithmetic: `+`, `-`, `*`, `/`, `@`, `abs`, `sum` through dunder methods |
    | 4 | `LinearModel`: a dataclass that predicts, measures its error and takes a training step |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def dot(u: Sequence[float], v: Sequence[float]) -> float:
    """stage 1: the dot product u[0]*v[0] + u[1]*v[1] + ... of two equal-length sequences.

    u, v: lists or tuples of numbers (plain Python, no NumPy).
    returns a Python float (0.0 for two empty sequences).
    raises ValueError if the lengths differ.
    """
    # step 1: if len(u) != len(v), raise ValueError with a message giving both lengths
    if len(u) != len(v):
        raise ValueError(f"dot needs equal lengths, got {len(u)} and {len(v)}")
    # step 2: start a running total at 0.0, then loop over the pairs with zip(u, v)
    total = 0.0
    for a, b in zip(u, v):
        # step 3: add a * b to the total
        total += a * b
    # step 4: return the total as a float
    return float(total)


@app.function
def norm(u: Sequence[float]) -> float:
    """stage 1: the Euclidean length sqrt(u[0]**2 + u[1]**2 + ...) of a sequence. a Python float."""
    # step 1: the squared length is the dot product of u with itself (reuse your dot)
    # step 2: return its square root (math.sqrt)
    return math.sqrt(dot(u, u))


@app.cell(hide_code=True)
def _():
    checks.run(1, dot=dot, norm=norm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class Vector:
    """An immutable vector of floats with arithmetic operators.

    stage 2 methods make it a container: build it, measure it, index it, loop over it,
    print it and compare it. stage 3 methods give it arithmetic.
    """

    def __init__(self, values: Iterable[float]):
        """stage 2: store the numbers as a tuple of floats in `self.values`.

        Vector([1, 2, 3]).values == (1.0, 2.0, 3.0). any iterable works (list, tuple, generator).
        """
        # step 1: convert every value to float and store them as a tuple in self.values
        self.values = tuple(float(x) for x in values)

    @classmethod
    def zeros(cls, n: int) -> "Vector":
        """stage 2: the zero vector of length n: Vector.zeros(3) == Vector([0, 0, 0])."""
        # step 1: build and return cls([0.0] * n)
        return cls([0.0] * n)

    def __len__(self) -> int:
        """stage 2: len(v) is the number of components."""
        # step 1: return the length of self.values
        return len(self.values)

    def __getitem__(self, i: int) -> float:
        """stage 2: v[i] is component i (negative indices count from the end, like lists)."""
        # step 1: return self.values[i] (a tuple already understands negative indices)
        return self.values[i]

    def __iter__(self):
        """stage 2: `for x in v` loops over the components; list(v) gives a list of floats."""
        # step 1: return an iterator over self.values (the built-in iter)
        return iter(self.values)

    def __repr__(self) -> str:
        """stage 2: repr(Vector([1, 2])) == 'Vector([1.0, 2.0])'."""
        # step 1: build the string from the class name and list(self.values)
        return f"Vector({list(self.values)})"

    def __eq__(self, other: object) -> bool:
        """stage 2: two Vectors are equal when their values are equal (same length too).

        comparing with anything that isn't a Vector returns NotImplemented, so Python
        falls back to its default (Vector([1]) == [1.0] is False).
        """
        # step 1: if other isn't a Vector, return NotImplemented
        if not isinstance(other, Vector):
            return NotImplemented
        # step 2: otherwise compare the two tuples of values
        return self.values == other.values

    def __add__(self, other: "Vector") -> "Vector":
        """stage 3: u + v, componentwise. raises ValueError if the lengths differ.

        returns NotImplemented if other isn't a Vector.
        """
        # step 1: if other isn't a Vector, return NotImplemented
        if not isinstance(other, Vector):
            return NotImplemented
        # step 2: if the lengths differ, raise ValueError
        if len(self) != len(other):
            raise ValueError(f"can't add vectors of lengths {len(self)} and {len(other)}")
        # step 3: return a NEW Vector of the pairwise sums (zip the two value tuples)
        return Vector(a + b for a, b in zip(self.values, other.values))

    def __radd__(self, other: object) -> "Vector":
        """stage 3: other + v when other isn't a Vector. only the number 0 is allowed, which
        makes the built-in sum(list_of_vectors) work (sum starts from 0). else NotImplemented.
        """
        # step 1: if other == 0, return self (0 + v is v)
        if isinstance(other, numbers.Real) and other == 0:
            return self
        # step 2: anything else: return NotImplemented
        return NotImplemented

    def __neg__(self) -> "Vector":
        """stage 3: -v, every component negated."""
        # step 1: return a new Vector with every component negated
        return Vector(-a for a in self.values)

    def __sub__(self, other: "Vector") -> "Vector":
        """stage 3: u - v, componentwise. same rules as +."""
        # step 1: if other isn't a Vector, return NotImplemented
        if not isinstance(other, Vector):
            return NotImplemented
        # step 2: reuse what you have: u - v is u + (-v)
        return self + (-other)

    def __mul__(self, scalar: float) -> "Vector":
        """stage 3: v * c for a real number c: any numbers.Real, which covers int, float and
        NumPy numbers like np.float64(2.5). Vector * Vector returns NotImplemented (so Python
        raises TypeError): use @ for the dot product.
        """
        # step 1: if scalar isn't a real number (isinstance(scalar, numbers.Real)), return NotImplemented
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        # step 2: return a new Vector with every component multiplied by scalar
        return Vector(a * scalar for a in self.values)

    def __rmul__(self, scalar: float) -> "Vector":
        """stage 3: c * v, the same as v * c."""
        # step 1: multiplication by a number commutes, so hand the work to __mul__
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector":
        """stage 3: v / c for a real number c (numbers.Real). dividing by 0 raises ZeroDivisionError."""
        # step 1: if scalar isn't a real number, return NotImplemented
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        # step 2: return a new Vector with every component divided by scalar
        return Vector(a / scalar for a in self.values)

    def __matmul__(self, other: "Vector") -> float:
        """stage 3: u @ v is the dot product, a float. NotImplemented if other isn't a Vector."""
        # step 1: if other isn't a Vector, return NotImplemented
        if not isinstance(other, Vector):
            return NotImplemented
        # step 2: return your stage 1 dot of the two value tuples
        return dot(self.values, other.values)

    def __abs__(self) -> float:
        """stage 3: abs(v) is the Euclidean length, a float."""
        # step 1: return your stage 1 norm of the values
        return norm(self.values)


@app.cell(hide_code=True)
def _():
    checks.run(2, Vector=Vector)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    the stage 3 methods (`__add__` through `__abs__`) are in the `Vector` cell above, below
    the stage 2 ones. scroll up, fill them in, and this check updates.
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, Vector=Vector)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
@dataclass(frozen=True)
class LinearModel:
    """A linear model y_hat = weights @ x + bias, as a frozen (immutable) dataclass.

    @dataclass writes __init__, __repr__ and __eq__ for you from the two fields below:
    LinearModel(Vector([1, 2]), 0.5) or LinearModel(weights=Vector([1, 2]), bias=0.5).
    frozen=True forbids changing a field after creation, so training returns NEW models.
    """

    weights: Vector
    bias: float = 0.0

    def __call__(self, x: Vector) -> float:
        """stage 4: the prediction for one input: model(x) == weights @ x + bias, a float."""
        # step 1: use your Vector's @ for the dot product, then add the bias
        return self.weights @ x + self.bias

    def mse(self, xs: Sequence[Vector], ys: Sequence[float]) -> float:
        """stage 4: mean squared error (1/n) * sum_i (model(xs[i]) - ys[i])**2 over n examples.

        raises ValueError if xs and ys have different lengths or are empty.
        """
        # step 1: raise ValueError if len(xs) != len(ys) or there are no examples
        if len(xs) != len(ys) or len(xs) == 0:
            raise ValueError(f"need the same, nonzero number of xs and ys, got {len(xs)} and {len(ys)}")
        # step 2: add up (self(x) - y) ** 2 over the pairs (zip), then divide by n
        return sum((self(x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

    def step(self, xs: Sequence[Vector], ys: Sequence[float], lr: float) -> "LinearModel":
        """stage 4: one gradient-descent step on the mean squared error; returns a NEW model.

        with errors e_i = model(xs[i]) - ys[i], the gradients of the MSE are
            grad_w = (2/n) * sum_i e_i * xs[i]     (a Vector)
            grad_b = (2/n) * sum_i e_i             (a float)
        and the new model has weights - lr * grad_w and bias - lr * grad_b.
        the model you call this on is left unchanged (it's frozen).
        """
        # step 1: n = len(xs); compute the list of errors e_i = self(x) - y
        n = len(xs)
        errors = [self(x) - y for x, y in zip(xs, ys)]
        # step 2: grad_w = (2 / n) * sum of e_i * x_i (sum() works on Vectors thanks to __radd__)
        grad_w = (2 / n) * sum(e * x for e, x in zip(errors, xs))
        # step 3: grad_b = (2 / n) * sum of the e_i
        grad_b = (2 / n) * sum(errors)
        # step 4: return replace(self, weights=..., bias=...) with the updated values
        return replace(self, weights=self.weights - lr * grad_w, bias=self.bias - lr * grad_b)


@app.cell(hide_code=True)
def _():
    checks.run(4, Vector=Vector, LinearModel=LinearModel)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, try it out in a scratch cell: build a few `Vector`s, add them,
    train a `LinearModel` for a few hundred steps on points from `y = 2x + 1` and watch the
    weights approach 2 and 1. then open the lab: `marimo edit lab.py`. it asks how fast all
    of this is, and the answer explains why the rest of the course uses NumPy.
    """)
    return


if __name__ == "__main__":
    app.run()
