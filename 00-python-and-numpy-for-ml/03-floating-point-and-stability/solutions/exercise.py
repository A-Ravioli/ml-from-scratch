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
    # floating point & stability · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks your code live as you edit.

    | stage | you build |
    |---|---|
    | 1 | `decompose`, `machine_epsilon`: what a floating point number is made of |
    | 2 | `quadratic_roots`: dodging catastrophic cancellation |
    | 3 | `sigmoid`, `softplus`, `log_sigmoid`: no overflow, no lost digits → `mlfs.utils` |
    | 4 | `logsumexp`: adding up probabilities in log space → `mlfs.utils` |
    | 5 | `softmax`, `log_softmax`: the stable versions every classifier uses → `mlfs.utils` |

    stages 3–5 become part of `mlfs.utils`. every classifier, language model and
    probabilistic model later in the course computes its probabilities and losses with them.
    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def decompose(x: float) -> tuple[int, float, int]:
    """stage 1: split a nonzero, finite float into (sign, mantissa, exponent) with

        x == (-1)**sign * mantissa * 2**exponent,   sign in {0, 1},   1 <= mantissa < 2,

    and exponent an int. decompose(6.0) == (0, 1.5, 2); decompose(-0.375) == (1, 1.5, -2).
    raises ValueError for 0, inf or nan. (don't use math.frexp: build it with a loop.)
    """
    # step 1: x = float(x); raise ValueError if x == 0 or it isn't finite (math.isfinite)
    x = float(x)
    if x == 0 or not math.isfinite(x):
        raise ValueError(f"decompose needs a nonzero finite number, got {x}")
    # step 2: sign = 1 if x < 0 else 0; continue with m = abs(x) and exponent = 0
    sign = 1 if x < 0 else 0
    m, exponent = abs(x), 0
    # step 3: while m >= 2: halve m and add 1 to the exponent
    while m >= 2.0:
        m /= 2.0
        exponent += 1
    # step 4: while m < 1: double m and subtract 1 from the exponent
    # step 4: (multiplying and dividing by 2 is EXACT in binary floating point)
    while m < 1.0:
        m *= 2.0
        exponent -= 1
    return sign, m, exponent


@app.function
def machine_epsilon(dtype: type = np.float64) -> float:
    """stage 1: the gap between 1 and the next larger number representable in `dtype`.

    find it by halving: start from eps = 1 and keep halving while 1 + eps/2 still differs
    from 1, doing ALL the arithmetic in `dtype` (np.float16, np.float32 or np.float64).
    returns a Python float: 2**-52 for float64, 2**-23 for float32, 2**-10 for float16.
    """
    # step 1: make one = dtype(1), two = dtype(2) and eps = dtype(1): numpy scalars of that type
    one, two, eps = dtype(1), dtype(2), dtype(1)
    # step 2: while one + eps / two != one: eps = eps / two
    while one + eps / two != one:
        eps = eps / two
    # step 3: return float(eps)
    return float(eps)


@app.cell(hide_code=True)
def _():
    checks.run(1, decompose=decompose, machine_epsilon=machine_epsilon)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    """stage 2: the two real roots of a x**2 + b x + c = 0, accurate to ~1e-15 relative error
    even when one root is tiny (e.g. a=1, b=1e8, c=1).

    returns (smaller root, larger root) as floats. avoids catastrophic cancellation:
    compute q = -(b + sign(b) * sqrt(b*b - 4*a*c)) / 2 (no cancellation inside), then the
    roots are q / a and c / q. raises ValueError if a == 0 or the roots aren't real.
    """
    # step 1: raise ValueError if a == 0; compute disc = b*b - 4*a*c and raise if it's negative
    if a == 0:
        raise ValueError("a must be nonzero for a quadratic")
    disc = b * b - 4.0 * a * c
    if disc < 0:
        raise ValueError(f"no real roots: discriminant {disc} < 0")
    # step 2: q = -(b + copysign(sqrt(disc), b)) / 2: b and the root get the SAME sign, so they add
    q = -(b + math.copysign(math.sqrt(disc), b)) / 2.0
    # step 3: if q == 0 (only when b == 0 and c == 0), both roots are 0
    if q == 0:
        return 0.0, 0.0
    # step 4: the roots are q / a and c / q (their product is c / a, as it must be); return them sorted
    r1, r2 = q / a, c / q
    return (min(r1, r2), max(r1, r2))


@app.cell(hide_code=True)
def _():
    checks.run(2, quadratic_roots=quadratic_roots)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def sigmoid(x: np.ndarray) -> np.ndarray:
    """stage 3: the logistic function 1 / (1 + exp(-x)), elementwise, for any array (or number).

    returns a float array of the same shape, with values in [0, 1], and must not overflow:
    no warnings even for |x| up to 1e4. never call exp on a large positive number.
    """
    # step 1: x = np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    # step 2: e = exp(-|x|) is always between 0 and 1, so it can't overflow
    e = np.exp(-np.abs(x))
    # step 3: for x >= 0, sigmoid = 1 / (1 + e); for x < 0, sigmoid = e / (1 + e). pick with np.where
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


@app.function
def softplus(x: np.ndarray) -> np.ndarray:
    """stage 3: log(1 + exp(x)), elementwise, computed stably.

    exact to full precision for tiny results (softplus(-40) ~ 4.25e-18, so use np.log1p) and
    no overflow for large x (softplus(1000) == 1000). same shape as x, float.
    """
    # step 1: x = np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    # step 2: log(1 + e^x) = max(x, 0) + log(1 + e^(-|x|)); the second term never overflows
    # step 3: use np.log1p(t) for log(1 + t): it keeps the digits when t is tiny
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


@app.function
def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """stage 3: log(sigmoid(x)), elementwise, computed stably as -softplus(-x).

    log_sigmoid(-1000) == -1000 (not -inf), log_sigmoid(40) ~ -4.25e-18 (not 0).
    """
    # step 1: log(1 / (1 + e^-x)) = -log(1 + e^-x) = -softplus(-x): reuse your softplus
    return -softplus(-np.asarray(x, dtype=float))


@app.cell(hide_code=True)
def _():
    checks.run(3, sigmoid=sigmoid, softplus=softplus, log_sigmoid=log_sigmoid)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def logsumexp(x: np.ndarray, axis: int | tuple[int, ...] | None = None,
              keepdims: bool = False) -> np.ndarray:
    """stage 4: log(sum(exp(x))) along `axis` (all entries when axis=None), computed stably.

    like scipy.special.logsumexp: subtract the max m first, log(sum(exp(x - m))) + m.
    finite for huge inputs (logsumexp([1000, 1000]) == 1000 + log 2), -inf when every entry
    is -inf (no nan, no warnings). keepdims=True keeps the reduced axes with length 1.
    returns a float (numpy scalar) for a full reduction, else an array.
    """
    # step 1: x = np.asarray(x, dtype=float); m = np.max(x, axis=axis, keepdims=True)
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    # step 2: where m isn't finite (a slice of all -inf), use 0 instead, so x - m isn't nan
    m = np.where(np.isfinite(m), m, 0.0)
    # step 3: s = sum of exp(x - m) along axis (keepdims=True); every term is at most 1
    s = np.sum(np.exp(x - m), axis=axis, keepdims=True)
    # step 4: out = log(s) + m. log(0) = -inf is correct here: silence the warning with np.errstate
    with np.errstate(divide="ignore"):
        out = np.log(s) + m
    # step 5: if not keepdims, drop the reduced axes with np.squeeze(out, axis=axis)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out[()]  # a 0-d array becomes a numpy scalar; bigger arrays are unchanged


@app.cell(hide_code=True)
def _():
    checks.run(4, logsumexp=logsumexp)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """stage 5: exp(x) / sum(exp(x)) along `axis` (the last axis by default), computed stably.

    subtract the max along the axis first (softmax doesn't change, nothing overflows).
    returns a float array of the same shape whose entries along `axis` sum to 1.
    -inf entries get probability exactly 0.
    """
    # step 1: x = np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    # step 2: z = x - max along axis (keepdims=True so it broadcasts back)
    z = x - np.max(x, axis=axis, keepdims=True)
    # step 3: e = exp(z), then divide by its sum along the same axis (keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


@app.function
def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """stage 5: log(softmax(x)) along `axis`, computed stably as x - logsumexp(x, axis, keepdims=True).

    log_softmax([0, -1000]) == [0, -1000] (log(softmax) would give -inf).
    """
    # step 1: x = np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    # step 2: subtract your logsumexp along axis, with keepdims=True so it broadcasts
    return x - logsumexp(x, axis=axis, keepdims=True)


@app.cell(hide_code=True)
def _():
    checks.run(5, softmax=softmax, log_softmax=log_softmax)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, these functions are `mlfs.utils.sigmoid`, `softplus`,
    `log_sigmoid`, `logsumexp`, `softmax` and `log_softmax`. then open the lab:
    `marimo edit lab.py`. it breaks the naive versions on purpose, and finds the exact logit
    where they fail.
    """)
    return


if __name__ == "__main__":
    app.run()
