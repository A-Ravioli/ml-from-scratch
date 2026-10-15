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
    # vector spaces · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you a
    function to write, and checks your code live as you edit.

    | stage | you build |
    |---|---|
    | 1 | `rref`: reduced row echelon form, and where the pivots are |
    | 2 | `rank`, `in_span`: dimension and membership, read off the pivots |
    | 3 | `is_basis`, `coordinates`: bases and the coordinates of a vector in one |
    | 4 | `null_space`: a basis for the solutions of `A x = 0` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def rref(A: np.ndarray, *, tol: float = 1e-10) -> tuple[np.ndarray, tuple[int, ...]]:
    """stage 1: the reduced row echelon form of A (m, n), and its pivot columns.

    returns (R, pivots): R is an (m, n) float array; pivots is a tuple of increasing column
    indices. work column by column; in each column pick the row (at or below the current one)
    with the largest |entry| as the pivot; entries with |value| <= tol count as zero.
    each pivot is scaled to 1 and cleared from every OTHER row (above and below).
    does not modify A.
    """
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivots = []
    row = 0
    for col in range(n):
        if row == m:
            break
        p = row + int(np.argmax(np.abs(R[row:, col])))
        if abs(R[p, col]) <= tol:  # no usable pivot: this column is a free column
            R[row:, col] = 0.0
            continue
        R[[row, p]] = R[[p, row]]
        R[row] /= R[row, col]
        others = np.arange(m) != row
        R[others] -= np.outer(R[others, col], R[row])
        R[others, col] = 0.0  # exact zeros
        pivots.append(col)
        row += 1
    return R, tuple(pivots)


@app.cell(hide_code=True)
def _():
    checks.run(1, rref=rref)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def rank(A: np.ndarray, *, tol: float = 1e-10) -> int:
    """stage 2: the rank of A (m, n): the number of pivots in its rref (same tol)."""
    return len(rref(A, tol=tol)[1])


@app.function
def in_span(V: np.ndarray, x: np.ndarray, *, tol: float = 1e-10) -> bool:
    """stage 2: is x (n,) a linear combination of the columns of V (n, k)? returns a bool.

    x is in the span exactly when appending it as a new column doesn't increase the rank.
    """
    V = np.asarray(V, dtype=float)
    return rank(np.column_stack([V, x]), tol=tol) == rank(V, tol=tol)


@app.cell(hide_code=True)
def _():
    checks.run(2, rank=rank, in_span=in_span)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def is_basis(V: np.ndarray, *, tol: float = 1e-10) -> bool:
    """stage 3: do the columns of V (n, k) form a basis of R^n? returns a bool.

    true exactly when there are n of them and they are independent (rank n).
    """
    V = np.asarray(V, dtype=float)
    n, k = V.shape
    return k == n and rank(V, tol=tol) == n


@app.function
def coordinates(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    """stage 3: the coordinates c (n,) of x (n,) in the basis given by the columns of V (n, n).

    c solves V @ c = x; compute it by row-reducing the augmented matrix [V | x].
    raises ValueError if the columns of V are not a basis.
    """
    if not is_basis(V):
        raise ValueError("the columns of V are not a basis of R^n")
    R, _ = rref(np.column_stack([np.asarray(V, dtype=float), x]))
    return R[:, -1]


@app.cell(hide_code=True)
def _():
    checks.run(3, is_basis=is_basis, coordinates=coordinates)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def null_space(A: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    """stage 4: a basis of the null space {x : A x = 0} of A (m, n), as the columns of an
    (n, n - rank) array. one column per free (non-pivot) column f of the rref: set x_f = 1,
    the other free variables to 0, and solve for the pivot variables.
    """
    R, pivots = rref(A, tol=tol)
    n = R.shape[1]
    free = [c for c in range(n) if c not in pivots]
    N = np.zeros((n, len(free)))
    for j, f in enumerate(free):
        N[f, j] = 1.0
        for i, p in enumerate(pivots):
            N[p, j] = -R[i, f]
    return N


@app.cell(hide_code=True)
def _():
    checks.run(4, null_space=null_space)
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
