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
    # numpy arrays & vectorization · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks your code live as you edit.

    the rule for this whole topic: **no Python loops** (no `for`, no `while`, and no list
    comprehensions either: they're loops too). each function is one to five lines of NumPy.
    the checks read your code and time it against NumPy, so they'll notice. if you catch
    yourself writing a loop, look for the array operation that does it (the hints point to
    them).

    | stage | you build |
    |---|---|
    | 1 | `batch_dot`, `row_norms`: elementwise operations and reductions along an axis |
    | 2 | `split_features_target`, `one_hot`, `moving_average`: slicing (views) and fancy indexing (copies) |
    | 3 | `standardize`, `pairwise_differences`: broadcasting |
    | 4 | `pairwise_distances`: every distance between two point sets, no loops → `mlfs.utils` |

    `pairwise_distances` becomes part of `mlfs.utils`, the course's utility library: chapter
    02 uses it for nearest neighbours, chapter 09 for kernels and k-NN, chapter 11 for
    clustering. you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def batch_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """stage 1: the dot product of each row of A with the same row of B.

    A, B: float arrays of the same shape (n, d). returns a float array of shape (n,) with
    out[i] = A[i, 0]*B[i, 0] + ... + A[i, d-1]*B[i, d-1].
    raises ValueError if the shapes differ. no Python loops.
    """
    # step 1: convert both inputs with np.asarray(..., dtype=float)
    # step 2: if A.shape != B.shape, raise ValueError (mention both shapes)
    # step 3: multiply elementwise (A * B), then add up along axis 1 (the columns)
    raise NotImplementedError("stage1: implement batch_dot")


@app.function
def row_norms(X: np.ndarray) -> np.ndarray:
    """stage 1: the Euclidean length of each row of X (n, d), as a float array of shape (n,)."""
    # step 1: the squared length of row i is the dot product of row i with itself (batch_dot)
    # step 2: take the elementwise square root (np.sqrt)
    raise NotImplementedError("stage1: implement row_norms")


@app.cell(hide_code=True)
def _():
    checks.run(1, batch_dot=batch_dot, row_norms=row_norms)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def split_features_target(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """stage 2: split a table (n, d + 1) into features X (n, d) (every column but the last)
    and target y (n,) (the last column).

    both outputs must be VIEWS of `data` (basic slicing), not copies: no data is duplicated.
    """
    # step 1: X is every row and all columns except the last: a slice with :-1
    # step 2: y is every row of the last column: index the column with -1
    # step 3: return (X, y). slicing never copies, so both share data's memory
    raise NotImplementedError("stage2: implement split_features_target")


@app.function
def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """stage 2: one-hot encode integer labels.

    labels: int array (n,) with values in 0 .. n_classes-1. returns a float array (n, n_classes)
    of zeros with a single 1.0 in each row i, in column labels[i].
    raises ValueError if any label is negative or >= n_classes. no Python loops.
    """
    # step 1: labels = np.asarray(labels); check every label is in range (np.any is handy)
    # step 2: start from a float array of zeros with shape (n, n_classes)
    # step 3: fancy indexing: set out[row i, column labels[i]] = 1 for every i at once,
    # step 3: using the row indices np.arange(n) and the column indices labels
    raise NotImplementedError("stage2: implement one_hot")


@app.function
def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """stage 2: averages of every run of `window` consecutive values of x (n,).

    returns a float array of shape (n - window + 1,) with out[i] = mean(x[i : i + window]).
    raises ValueError unless 1 <= window <= n. no Python loops: use a cumulative sum.
    """
    # step 1: x = np.asarray(x, dtype=float); check 1 <= window <= len(x)
    # step 2: c = cumulative sum of x with a 0 in front, so c[k] = x[0] + ... + x[k-1]
    # step 3: the sum of x[i : i + window] is c[i + window] - c[i]: subtract two slices of c
    # step 4: divide by window
    raise NotImplementedError("stage2: implement moving_average")


@app.cell(hide_code=True)
def _():
    checks.run(2, split_features_target=split_features_target, one_hot=one_hot,
               moving_average=moving_average)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def standardize(X: np.ndarray) -> np.ndarray:
    """stage 3: rescale each column of X (n, d) to mean 0 and standard deviation 1.

    returns a NEW float array (X - mean) / std, with mean and std computed per column
    (std with ddof=0, NumPy's default). a constant column (every value equal) becomes
    exactly 0 instead of nan or noise. careful: the computed std of a constant column such
    as [0.1, 0.1, 0.1] is about 1e-17, not 0, so detect constant columns exactly with
    X.max(axis=0) == X.min(axis=0). X itself must not change. no Python loops.
    """
    # step 1: X = np.asarray(X, dtype=float) (so the result is a float array whatever X was)
    # step 2: the column means and stds: reduce along axis 0, giving shape (d,)
    # step 3: which columns are constant? compare the column max and min exactly: shape (d,)
    # step 4: broadcasting: (n, d) minus (d,) subtracts the mean row from every row; divide by
    # step 4: std, using 1 instead for constant columns (np.where), so nothing divides by ~0
    # step 5: set the constant columns to exactly 0 (np.where broadcasts the (d,) mask over rows)
    raise NotImplementedError("stage3: implement standardize")


@app.function
def pairwise_differences(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """stage 3: every difference between a row of X (n, d) and a row of Y (m, d).

    returns a float array D of shape (n, m, d) with D[i, j] = X[i] - Y[j].
    raises ValueError if the numbers of columns differ. no Python loops: broadcast.
    """
    # step 1: convert both to float arrays; raise ValueError if X.shape[1] != Y.shape[1]
    # step 2: give X shape (n, 1, d) and Y shape (1, m, d) with None (np.newaxis),
    # step 2: then subtract: broadcasting stretches both to (n, m, d)
    raise NotImplementedError("stage3: implement pairwise_differences")


@app.cell(hide_code=True)
def _():
    checks.run(3, standardize=standardize, pairwise_differences=pairwise_differences)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def pairwise_distances(X: np.ndarray, Y: np.ndarray | None = None, *,
                       squared: bool = False) -> np.ndarray:
    """stage 4: the Euclidean distance between every row of X (n, d) and every row of Y (m, d).

    returns a float array (n, m) with out[i, j] = ||X[i] - Y[j]||, or its square when
    squared=True. Y defaults to X: then the result is exactly symmetric (out == out.T) and
    its diagonal is exactly 0.
    uses ||x - y||^2 = ||x||^2 - 2 x.y + ||y||^2 with one matrix product, so it never builds
    the (n, m, d) array of differences. rounding can make a squared distance slightly
    negative: clip at 0 before the square root. raises ValueError for inputs that aren't
    2-D or whose numbers of columns differ. no Python loops.
    """
    # step 1: convert X to a float array; if Y is None, use X; check both are 2-D with equal columns
    # step 2: the squared row norms of X as a column (n, 1) and of Y as a row (1, m)
    # step 3: all the dot products X[i] . Y[j] at once: the matrix product X @ Y.T, shape (n, m)
    # step 4: combine with broadcasting: xx - 2 X @ Y.T + yy, then clip negatives to 0 (np.maximum)
    # step 5: if Y was None: symmetrize, d2 = (d2 + d2.T) / 2, and np.fill_diagonal(d2, 0.0)
    # step 6: return d2 if squared, else its square root
    raise NotImplementedError("stage4: implement pairwise_distances")


@app.cell(hide_code=True)
def _():
    checks.run(4, pairwise_distances=pairwise_distances)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your `pairwise_distances` is `mlfs.utils.pairwise_distances`
    (try `from mlfs.utils import pairwise_distances` in a scratch cell). then open the lab:
    `marimo edit lab.py`. it measures how much all this vectorization buys you, and finds a
    case where the fast formula quietly gives wrong answers.
    """)
    return


if __name__ == "__main__":
    app.run()
