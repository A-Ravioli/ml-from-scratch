import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # vectors & matrices, a first look · build (reference solution)

    a vector is a list of numbers and an arrow; a matrix is a table of numbers and a
    *transformation* that moves every arrow at once. in this notebook you build the basic
    operations yourself (dot products, lengths, angles, matrix times vector, matrix times
    matrix), and then a **visualizer** that shows what any 2×2 matrix does to the plane.

    the point is to build these, so inside your functions don't use `@`, `np.dot`,
    `np.linalg` or `np.matmul`: elementwise arithmetic (`*`, `+`, `np.sum`) and loops only.
    (after this topic you may use `@` freely.)

    every function has step-by-step comments inside it: replace its last line (the one that
    raises "not implemented") with code that follows the steps.

    | stage | you build |
    |---|---|
    | 1 | `dot`, `norm`: the dot product and the length of a vector |
    | 2 | `angle`, `project`: angles between vectors, and shadows |
    | 3 | `matvec`, `matmul`: a matrix acting on a vector, and composing matrices |
    | 4 | `rotation`, `scaling`, `shear`: a gallery of 2×2 transformations |
    | 5 | `transform_points`, `plot_transform`: the 2D transformation visualizer |
    | 6 | `det2`, `polygon_area`: how a matrix scales areas |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def dot(u, v):
    """stage 1: the dot product u · v = u[0] v[0] + u[1] v[1] + ... + u[n-1] v[n-1].

    u and v are 1-D arrays (or lists) of the same length n; raise ValueError if the lengths
    differ. returns a float.
    example: dot([1, 2, 3], [4, -5, 6]) = 4 - 10 + 18 = 12.0
    """
    # step 1: u = np.asarray(u, dtype=float), and the same for v.
    # step 2: if u.shape != v.shape, raise ValueError.
    # step 3: multiply entry by entry (u * v) and add everything up: return float(np.sum(u * v)).
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError(f"dot needs vectors of the same length, got {u.shape} and {v.shape}")
    return float(np.sum(u * v))


@app.function
def norm(v):
    """stage 1: the length of the vector v: sqrt(v[0]^2 + v[1]^2 + ...) (Pythagoras).

    v is a 1-D array (or list). returns a float. use your dot.
    example: norm([3, 4]) = 5.0
    """
    # step 1: the squared length is the dot product of v with itself.
    # step 2: return its square root, as a float.
    return float(np.sqrt(dot(v, v)))


@app.cell(hide_code=True)
def _():
    checks.run(1, dot=dot, norm=norm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def angle(u, v):
    """stage 2: the angle between two nonzero vectors u and v, in RADIANS (between 0 and pi).

    from u · v = |u| |v| cos(theta):  theta = arccos( u · v / (|u| |v|) ).
    rounding can push the ratio a hair outside [-1, 1] (e.g. 1.0000000000000002), where
    arccos is undefined, so clip it into [-1, 1] first. returns a float.
    example: angle([1, 0], [1, 1]) = pi / 4 (45 degrees)
    """
    # step 1: c = dot(u, v) / (norm(u) * norm(v)).
    # step 2: c = np.clip(c, -1.0, 1.0).
    # step 3: return float(np.arccos(c)).
    c = dot(u, v) / (norm(u) * norm(v))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


@app.function
def project(u, v):
    """stage 2: the projection ("shadow") of u onto the line through a nonzero vector v.

    it's the multiple of v closest to u:  (u · v / v · v) * v.  returns a 1-D array, same
    shape as v. what's left over, u - project(u, v), is perpendicular to v.
    example: project([2, 3], [1, 0]) = [2, 0]
    """
    # step 1: v = np.asarray(v, dtype=float).
    # step 2: the scale factor is dot(u, v) / dot(v, v).
    # step 3: return scale factor * v.
    v = np.asarray(v, dtype=float)
    return dot(u, v) / dot(v, v) * v


@app.cell(hide_code=True)
def _():
    checks.run(2, angle=angle, project=project, dot=dot)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def matvec(A, v):
    """stage 3: the matrix-vector product A v, as a combination of A's COLUMNS.

    A has shape (m, n) and v shape (n,); raise ValueError if A has a different number of
    columns than v has entries. A v = v[0] * (column 0) + v[1] * (column 1) + ... + v[n-1] * (column n-1),
    a vector of shape (m,). (no @ or np.dot: build it.)
    example: matvec([[1, 2], [3, 4]], [10, 1]) = 10 * [1, 3] + 1 * [2, 4] = [12, 34]
    """
    # step 1: A = np.asarray(A, dtype=float) and v = np.asarray(v, dtype=float).
    # step 2: if A.shape[1] != v.shape[0], raise ValueError.
    # step 3: start with result = np.zeros(A.shape[0]).
    # step 4: for each column index j: result = result + v[j] * A[:, j].
    # step 5: return result.
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    if A.shape[1] != v.shape[0]:
        raise ValueError(f"A has {A.shape[1]} columns but v has {v.shape[0]} entries")
    result = np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        result = result + v[j] * A[:, j]
    return result


@app.function
def matmul(A, B):
    """stage 3: the matrix product A B, built one column at a time with your matvec.

    A has shape (m, n), B shape (n, p); raise ValueError if A's columns don't match B's rows.
    column j of A B is A times column j of B, so the result has shape (m, p).
    A B means "first apply B, then A".
    example: matmul([[1, 2], [3, 4]], [[0, 1], [1, 0]]) = [[2, 1], [4, 3]]  (columns swapped)
    """
    # step 1: A and B as float arrays; if A.shape[1] != B.shape[0], raise ValueError.
    # step 2: C = np.zeros((A.shape[0], B.shape[1])).
    # step 3: for each column index j of B: C[:, j] = matvec(A, B[:, j]).
    # step 4: return C.
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"A has {A.shape[1]} columns but B has {B.shape[0]} rows")
    C = np.zeros((A.shape[0], B.shape[1]))
    for j in range(B.shape[1]):
        C[:, j] = matvec(A, B[:, j])
    return C


@app.cell(hide_code=True)
def _():
    checks.run(3, matvec=matvec, matmul=matmul)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def rotation(theta):
    """stage 4: the 2×2 matrix that rotates the plane COUNTERCLOCKWISE by theta radians.

    its columns are where e1 = (1, 0) and e2 = (0, 1) end up:
    e1 -> (cos theta, sin theta) and e2 -> (-sin theta, cos theta). returns shape (2, 2).
    example: rotation(np.pi / 2) sends (1, 0) to (0, 1)
    """
    # step 1: c = np.cos(theta), s = np.sin(theta).
    # step 2: return np.array([[c, -s], [s, c]])  (first column (c, s), second column (-s, c)).
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@app.function
def scaling(sx, sy):
    """stage 4: the 2×2 matrix that stretches x by sx and y by sy: e1 -> (sx, 0), e2 -> (0, sy).

    (a negative factor also flips that axis: scaling(-1, 1) is a mirror.) returns shape (2, 2).
    """
    # step 1: return a diagonal matrix with sx and sy on the diagonal and 0 elsewhere.
    return np.array([[sx, 0.0], [0.0, sy]], dtype=float)


@app.function
def shear(k):
    """stage 4: the horizontal shear by k: e1 -> (1, 0) stays put, e2 -> (k, 1) leans over.

    every point (x, y) moves sideways by k * y, like a deck of cards pushed from the side.
    returns shape (2, 2).
    """
    # step 1: the columns are (1, 0) and (k, 1): return np.array([[1, k], [0, 1]]) as floats.
    return np.array([[1.0, k], [0.0, 1.0]])


@app.cell(hide_code=True)
def _():
    checks.run(4, rotation=rotation, scaling=scaling, shear=shear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def transform_points(A, P):
    """stage 5: apply the 2×2 matrix A to many points at once.

    P has shape (n, 2): one point (x, y) per ROW. returns shape (n, 2), whose row i is
    A times row i of P. use your matmul: put the points into columns (P.T, shape (2, n)),
    multiply, and transpose back.
    example: transform_points(scaling(2, 3), [[1, 1], [1, 0]]) = [[2, 3], [2, 0]]
    """
    # step 1: P = np.asarray(P, dtype=float).
    # step 2: return matmul(A, P.T).T
    P = np.asarray(P, dtype=float)
    return matmul(A, P.T).T


@app.function
def plot_transform(A, *, ax=None, n=200):
    """stage 5: draw what the 2×2 matrix A does to the plane. returns the matplotlib Axes.

    if ax is None, make one with plt.subplots(). then draw, on ax:
      1. the unit circle: a line through the n points (cos t, sin t), t = np.linspace(0, 2 pi, n),
         dashed and gray, as the "before" picture;
      2. its image: a line through the same n points transformed by A (use transform_points);
      3. the images of e1 and e2 (the columns of A), as arrows from the origin
         (ax.annotate("", xy=column, xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
         or ax.arrow);
    and call ax.set_aspect("equal") so circles look like circles. return ax.
    """
    # step 1: if ax is None: _, ax = plt.subplots(figsize=(4, 4)).
    # step 2: t = np.linspace(0, 2 * np.pi, n); circle = np.column_stack([np.cos(t), np.sin(t)]).
    # step 3: ax.plot(circle[:, 0], circle[:, 1], "--", color="gray")  (before)
    # step 4: image = transform_points(A, circle); ax.plot(image[:, 0], image[:, 1])  (after)
    # step 5: for each column of A, draw an arrow from (0, 0) to it, labelled "A e1" / "A e2".
    # step 6: ax.set_aspect("equal"), and return ax.
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    A = np.asarray(A, dtype=float)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    ax.plot(circle[:, 0], circle[:, 1], "--", color="gray", lw=1, label="unit circle")
    image = transform_points(A, circle)
    ax.plot(image[:, 0], image[:, 1], color="C0", lw=2, label="its image")
    for j, color in ((0, "C1"), (1, "C2")):
        col = A[:, j]
        ax.annotate("", xy=col, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.text(col[0], col[1], f" A e{j + 1}", color=color)
    lim = max(1.2, 1.15 * np.abs(image).max())
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    return ax


@app.cell(hide_code=True)
def _():
    checks.run(5, transform_points=transform_points, plot_transform=plot_transform)
    return


@app.cell
def _():
    # try it: change the matrix and the picture updates (once stage 5 is green)
    try:
        _out = plot_transform(np.array([[2.0, 1.0], [1.0, 2.0]])).figure
    except Exception as _e:  # stage 5 isn't finished yet
        _out = mo.md(f"*the picture appears here when stage 5 works ({type(_e).__name__}: {_e})*")
    _out
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.function
def det2(A):
    """stage 6: the determinant of a 2×2 matrix [[a, b], [c, d]]: a d - b c. returns a float.

    it's the factor by which A scales areas, with a minus sign if A flips the plane over.
    example: det2([[2, 1], [1, 2]]) = 4 - 1 = 3.0
    """
    # step 1: A = np.asarray(A, dtype=float); read a, b, c, d off its entries.
    # step 2: return float(a * d - b * c).
    A = np.asarray(A, dtype=float)
    return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])


@app.function
def polygon_area(P):
    """stage 6: the SIGNED area of the polygon with corners P (shape (n, 2)), in order.

    shoelace formula: 0.5 * sum over i of (x_i y_(i+1) - x_(i+1) y_i), where the corner after
    the last one is the first one again. positive if the corners go counterclockwise,
    negative if clockwise. returns a float.
    example: the unit square [[0, 0], [1, 0], [1, 1], [0, 1]] has area 1.0
    """
    # step 1: P = np.asarray(P, dtype=float); x = P[:, 0], y = P[:, 1].
    # step 2: the "next" corners: x_next = np.roll(x, -1), y_next = np.roll(y, -1)
    #         (np.roll shifts everything one place, wrapping the first entry around to the end).
    # step 3: return float(0.5 * np.sum(x * y_next - x_next * y)).
    P = np.asarray(P, dtype=float)
    x, y = P[:, 0], P[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@app.cell(hide_code=True)
def _():
    checks.run(6, det2=det2, polygon_area=polygon_area, transform_points=transform_points)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. you'll turn the four
    knobs of a 2×2 matrix and watch the unit circle become an ellipse, see that the order of
    two transformations matters, and meet a matrix that looks like it stretches everything
    but doesn't change any area at all.
    """)
    return


if __name__ == "__main__":
    app.run()
