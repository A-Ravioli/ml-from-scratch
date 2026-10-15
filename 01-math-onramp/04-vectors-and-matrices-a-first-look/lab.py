import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # vectors & matrices, a first look · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell
def _():
    def circle_stats(A, n=2000):
        """longest and shortest radius of the circle's image, and the factor its area changed by."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        image = np.asarray(ex.transform_points(A, circle))
        radii = np.sqrt(np.sum(image**2, axis=1))
        area_factor = ex.polygon_area(image) / ex.polygon_area(circle)
        return radii.max(), radii.min(), area_factor

    return (circle_stats,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what a matrix does to the unit circle

    the four sliders are the entries of $A = \begin{pmatrix}a&b\\c&d\end{pmatrix}$. your
    `plot_transform` draws the unit circle and its image, and below it we measure the
    image's longest and shortest radius (with your `transform_points`) and how much its
    area changed (with your `polygon_area`), next to your `det2`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for A = [[2, 1], [1, 2]] (the starting position of the sliders): what shape is the "
        "image, how long are its longest and shortest radii, and in which directions?"
    )
    p1
    return (p1,)


@app.cell
def _():
    a = mo.ui.slider(-2, 2, value=2, step=0.1, label="a")
    b = mo.ui.slider(-2, 2, value=1, step=0.1, label="b")
    c = mo.ui.slider(-2, 2, value=1, step=0.1, label="c")
    d = mo.ui.slider(-2, 2, value=2, step=0.1, label="d")
    mo.hstack([mo.vstack([a, b]), mo.vstack([c, d])])
    return a, b, c, d


@app.cell
def _(a, b, c, circle_stats, d, p1):
    ui.gate(p1)
    _A = np.array([[a.value, b.value], [c.value, d.value]])
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    ex.plot_transform(_A, ax=_ax)
    _ax.set_title(f"A = [[{a.value:g}, {b.value:g}], [{c.value:g}, {d.value:g}]]")
    _rmax, _rmin, _area = circle_stats(_A)
    mo.hstack([_fig, mo.md(
        f"| quantity | value |\n|---|---|\n"
        f"| longest radius | {_rmax:.3f} |\n| shortest radius | {_rmin:.3f} |\n"
        f"| longest × shortest | {_rmax * _rmin:.3f} |\n"
        f"| area of image ÷ area of circle | {_area:.3f} |\n| your det2(A) | {ex.det2(_A):.3f} |")],
        justify="start")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for $\begin{pmatrix}2&1\\1&2\end{pmatrix}$, which two directions are
    stretched the most and the least, and by how much? (compute $A(1, 1)$ and $A(1, -1)$.)
    move the sliders: how do "longest × shortest", the area factor, and the determinant
    compare? what happens to the picture when the determinant is negative? when it's 0?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does the order matter?

    $R$ rotates by $90°$ counterclockwise and $S$ is the shear $\begin{pmatrix}1&1\\0&1\end{pmatrix}$.
    we draw $SR$ ("rotate, then shear") and $RS$ ("shear, then rotate") with your `matmul`
    and `plot_transform`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "will the two pictures be the same? where does e1 = (1, 0) end up in each case?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _R, _S = ex.rotation(np.pi / 2), ex.shear(1.0)
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _rows = []
    for _ax, (_name, _M) in zip(_axes, (("S R: rotate, then shear", ex.matmul(_S, _R)),
                                         ("R S: shear, then rotate", ex.matmul(_R, _S)))):
        ex.plot_transform(_M, ax=_ax)
        _ax.set_title(_name)
        _e1 = np.asarray(ex.matvec(_M, [1.0, 0.0]))
        _rows.append(f"| {_name} | {np.round(np.asarray(_M), 3).tolist()} | ({_e1[0]:.2f}, {_e1[1]:.2f}) |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| product | matrix | where e1 lands |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** follow $e_1$ through each sequence by hand: first the right-hand matrix,
    then the left. are the two ellipses the same *shape*, just turned? (compare their
    longest radii in experiment 1's way.) why does $AB$ mean "first $B$, then $A$"?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: stretching without growing

    three matrices: the shear $\begin{pmatrix}1&1\\0&1\end{pmatrix}$, the scaling
    $\begin{pmatrix}2&0\\0&0.5\end{pmatrix}$, and $\begin{pmatrix}1&2\\0.5&1\end{pmatrix}$,
    whose second column is twice its first.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "rank the three by how much they change the circle's area. for the shear, guess the "
        "longest and shortest radius of the image."
    )
    p3
    return (p3,)


@app.cell
def _(circle_stats, p3):
    ui.gate(p3)
    _mats = {
        "shear [[1, 1], [0, 1]]": ex.shear(1.0),
        "scaling [[2, 0], [0, 0.5]]": ex.scaling(2.0, 0.5),
        "parallel columns [[1, 2], [0.5, 1]]": np.array([[1.0, 2.0], [0.5, 1.0]]),
    }
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4.2))
    _rows = []
    for _ax, (_name, _M) in zip(_axes, _mats.items()):
        ex.plot_transform(_M, ax=_ax)
        _ax.set_title(_name, fontsize=9)
        _rmax, _rmin, _area = circle_stats(_M)
        _rows.append(f"| {_name} | {_rmax:.4f} | {_rmin:.4f} | {_area:.4f} | {ex.det2(_M):.4f} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| matrix | longest radius | shortest radius | area factor | det2 |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the shear visibly stretches the circle, so why doesn't its area change?
    (think of the deck of cards: does pushing it sideways change the area of its side?) the
    shear's two radii have a famous ratio; which number is it, and what is their product?
    what happened to the third matrix, and why can't you undo it?
    """)
    return


if __name__ == "__main__":
    app.run()
