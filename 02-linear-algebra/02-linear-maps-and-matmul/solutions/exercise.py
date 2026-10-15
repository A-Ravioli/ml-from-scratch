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
    # linear maps & matmul · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you a
    function to write, and checks your code live as you edit. in stages 1–3, don't use `@`,
    `np.dot` or `np.matmul` on whole matrices: the point is to build the product yourself,
    once per view. (inside, a dot product of two vectors via `np.sum(u * v)` is fine.)

    | stage | you build |
    |---|---|
    | 1 | `matmul_entries`: every entry is a row-times-column dot product |
    | 2 | `matmul_columns`, `matmul_rows`: combinations of columns, and of rows |
    | 3 | `matmul_outer`: a sum of rank-one outer products |
    | 4 | `matrix_of`, `is_linear`: from a linear map to its matrix |
    | 5 | `matmul_blocked`: multiplying tile by tile, the way fast libraries do |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def matmul_entries(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """stage 1: C = A B for A (m, k) and B (k, n), entry by entry with three nested loops:
    C[i, j] = sum over p of A[i, p] * B[p, j]. returns a float (m, n) array."""
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += A[i, p] * B[p, j]
            C[i, j] = total
    return C


@app.cell(hide_code=True)
def _():
    checks.run(1, matmul_entries=matmul_entries)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def matmul_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """stage 2: C = A B built column by column: column j of C is the combination of A's
    columns with coefficients B[:, j], i.e. sum over p of B[p, j] * A[:, p]. returns (m, n)."""
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n))
    for j in range(n):
        for p in range(k):
            C[:, j] += B[p, j] * A[:, p]
    return C


@app.function
def matmul_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """stage 2: C = A B built row by row: row i of C is the combination of B's rows with
    coefficients A[i, :], i.e. sum over p of A[i, p] * B[p, :]. returns (m, n)."""
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n))
    for i in range(m):
        for p in range(k):
            C[i, :] += A[i, p] * B[p, :]
    return C


@app.cell(hide_code=True)
def _():
    checks.run(2, matmul_columns=matmul_columns, matmul_rows=matmul_rows)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def matmul_outer(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """stage 3: C = A B as a sum of k rank-one matrices: sum over p of the outer product of
    column p of A (m,) with row p of B (n,). returns (m, n)."""
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n))
    for p in range(k):
        C += np.outer(A[:, p], B[p, :])
    return C


@app.cell(hide_code=True)
def _():
    checks.run(3, matmul_outer=matmul_outer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def matrix_of(f, n: int) -> np.ndarray:
    """stage 4: the matrix (m, n) of a linear map f: R^n -> R^m.

    f takes a 1-D array of length n and returns a 1-D array of length m. column j of the
    matrix is f(e_j), the image of the j-th standard basis vector.
    """
    return np.column_stack([np.asarray(f(e), dtype=float) for e in np.eye(n)])


@app.function
def is_linear(f, n: int, *, trials: int = 20, tol: float = 1e-9, rng=None) -> bool:
    """stage 4: test whether f: R^n -> R^m looks linear on random inputs.

    for `trials` random vectors x, y (standard normal) and random scalars a, b (standard
    normal, so both signs), check f(a x + b y) == a f(x) + b f(y) to relative tolerance tol.
    draw everything from np.random.default_rng(rng). returns a bool.
    """
    rng = np.random.default_rng(rng)
    for _ in range(trials):
        x, y = rng.normal(size=n), rng.normal(size=n)
        a, b = rng.normal(size=2)
        lhs = np.asarray(f(a * x + b * y), dtype=float)
        rhs = a * np.asarray(f(x), dtype=float) + b * np.asarray(f(y), dtype=float)
        scale = max(1.0, np.abs(lhs).max(), np.abs(rhs).max())
        if np.abs(lhs - rhs).max() > tol * scale:
            return False
    return True


@app.cell(hide_code=True)
def _():
    checks.run(4, matrix_of=matrix_of, is_linear=is_linear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def matmul_blocked(A: np.ndarray, B: np.ndarray, *, block: int = 32) -> np.ndarray:
    """stage 5: C = A B computed tile by tile: C[I, J] = sum over K of A[I, K] @ B[K, J],
    where I, J, K run over blocks of `block` rows/columns (the last tile in each direction may
    be smaller). each tile product may use @. returns (m, n).
    """
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n))
    for i in range(0, m, block):
        for j in range(0, n, block):
            for p in range(0, k, block):
                C[i:i + block, j:j + block] += A[i:i + block, p:p + block] @ B[p:p + block, j:j + block]
    return C


@app.cell(hide_code=True)
def _():
    checks.run(5, matmul_blocked=matmul_blocked)
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
