import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.optim import gradient_descent, lbfgs  # yours, from topics 02 and 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def count_local_minima(Z):
    """grid points strictly lower than all 8 neighbours (the border is excluded)."""
    c = Z[1:-1, 1:-1]
    nbrs = [Z[1 + di:Z.shape[0] - 1 + di, 1 + dj:Z.shape[1] - 1 + dj]
            for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    return int(np.sum(np.all([c < n for n in nbrs], axis=0)))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # nonconvex optimization · lab

    every experiment here runs **your** code from `exercise.py` (and your optimizers from
    earlier topics). for each one: write down a prediction, lock it in, run it, then explain
    what happened. compare your explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: where gradient descent gets stuck on Rastrigin

    the 2-D Rastrigin function has its global minimum at the origin and a local minimum near
    every integer point. we start your `gradient_descent` (step 0.002, 400 steps) from 100
    random points in the square $[-R, R]^2$ and count how many end at the global minimum.
    then we do the same with your `perturbed_gradient_descent`, with a small kick (radius
    0.1) and a large one (radius 0.6) whenever the gradient is small.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "starting in [-2, 2]^2, what fraction of gradient-descent runs reach the global minimum? does adding "
        "small random kicks at flat points help? large ones?"
    )
    p1
    return (p1,)


@app.cell
def _():
    R = mo.ui.slider(0.5, 4.0, value=2.0, step=0.5, label="start box half-width R")
    R
    return (R,)


@app.cell
def _(R, p1):
    ui.gate(p1)
    _starts = np.random.default_rng(0).uniform(-R.value, R.value, size=(100, 2))
    _ends = {"gradient descent": [], "perturbed, radius 0.1": [], "perturbed, radius 0.6": []}
    _vals = {k: [] for k in _ends}
    for _i, _s in enumerate(_starts):
        _r = gradient_descent(ex.rastrigin, ex.rastrigin_grad, _s, lr=0.002, max_iters=400, tol=1e-8)
        _ends["gradient descent"].append(_r.x)
        _vals["gradient descent"].append(_r.fun)
        for _rad in (0.1, 0.6):
            _r = ex.perturbed_gradient_descent(ex.rastrigin, ex.rastrigin_grad, _s, lr=0.002, noise_radius=_rad,
                                               grad_threshold=0.05, max_iters=400, rng=_i)
            _ends[f"perturbed, radius {_rad}"].append(_r.x)
            _vals[f"perturbed, radius {_rad}"].append(_r.fun)
    _al, _be, _Z = ex.loss_landscape_2d(ex.rastrigin, np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                                        span=max(R.value, 1.0) + 0.5, resolution=161)
    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    _ax.contourf(_al, _be, _Z.T, levels=30, cmap="viridis")
    _E = np.array(_ends["gradient descent"])
    _ax.plot(_starts[:, 0], _starts[:, 1], "w.", ms=3, label="starts")
    _ax.plot(_E[:, 0], _E[:, 1], "rx", ms=6, label="gradient descent ends")
    _ax.set_aspect("equal")
    _ax.set_title("Rastrigin: where gradient descent stops")
    _ax.legend(fontsize=8, loc="upper right")
    _rows = []
    for _k in _ends:
        _E = np.array(_ends[_k])
        _hit = np.mean(np.linalg.norm(_E, axis=1) < 0.1)
        _rows.append(f"| {_k} | **{_hit:.0%}** | {np.median(_vals[_k]):.2f} | {len({tuple(np.round(e)) for e in _E})} |")
    mo.vstack([_fig, mo.md(
        "| method | reached the global minimum | median final f | distinct minima reached |\n|---|---|---|---|\n"
        + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which local minimum does gradient descent end in, as a function of where
    it starts (look at the plot)? why do small kicks change almost nothing here, when they
    rescued gradient descent from a saddle in the exercise? what are large kicks doing, and
    why is that closer to random search than to optimization? how many local minima would
    Rastrigin have in $[-2, 2]^{100}$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: Rosenbrock, a valley with a trap

    in 2-D, Rosenbrock has a single minimum at $(1, 1)$, at the end of a long curved valley.
    in 4 to 7 dimensions it famously gains a second, *local* minimum near
    $(-1, 1, 1, \dots, 1)$. we run your `lbfgs` (topic 05) from 60 random starts in
    $[-2, 2]^6$ and classify where it ends. then we take one gradient-descent run, find the
    plane its path lies in with your `trajectory_pca`, and draw the loss on that plane with
    your `loss_landscape_2d`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "in 6-D, what fraction of L-BFGS runs end in the local minimum instead of the global one? "
        "how much of a gradient-descent path's movement do the top two PCA directions capture?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _glob, _loc, _other = 0, 0, 0
    _local_x = None
    for _s in _rng.uniform(-2, 2, size=(60, 6)):
        _r = lbfgs(ex.rosenbrock, ex.rosenbrock_grad, _s, tol=1e-8, max_iters=2000)
        if np.allclose(_r.x, 1, atol=1e-3):
            _glob += 1
        elif _r.x[0] < -0.9 and np.linalg.norm(ex.rosenbrock_grad(_r.x)) < 1e-6:
            _loc += 1
            _local_x = _r.x
        else:
            _other += 1
    _path = [np.array([-1.2, 1.0, 1.0, 1.0, 1.0, 1.0])]
    gradient_descent(ex.rosenbrock, ex.rosenbrock_grad, _path[0], lr=1.0, line_search=True, max_iters=3000, tol=1e-8,
                     callback=lambda x, k: _path.append(x))
    _P = np.array(_path)
    _d1, _d2, _coords, _expl = ex.trajectory_pca(_P)
    _span = 1.2 * np.abs(_coords).max()
    _al, _be, _Z = ex.loss_landscape_2d(ex.rosenbrock, _P[-1], _d1, _d2, span=_span, resolution=81)
    _fig, _ax = plt.subplots(figsize=(5.8, 5))
    _ax.contour(_al, _be, np.log10(_Z.T + 1e-6), levels=25, cmap="viridis")
    _ax.plot(_coords[:, 0], _coords[:, 1], "r.-", ms=2, lw=0.8, label=f"gradient descent ({len(_P) - 1} steps)")
    _ax.plot(0, 0, "k*", ms=10)
    _ax.set_aspect("equal")
    _ax.set(xlabel="PCA direction 1", ylabel="PCA direction 2", title="6-D Rosenbrock on the path's plane (log10 f)")
    _ax.legend(fontsize=8)
    _fstr = "n/a" if _local_x is None else f"{ex.rosenbrock(_local_x):.3f} at {np.round(_local_x, 3).tolist()}"
    mo.vstack([_fig, mo.md(
        f"L-BFGS from 60 starts: global minimum **{_glob}**, local minimum **{_loc}**, other {_other}. "
        f"the local minimum has f = {_fstr}. the gradient-descent path from (−1.2, 1, …, 1) ends at "
        f"{np.round(_P[-1], 3).tolist()}; the top two PCA directions explain **{_expl.sum():.1%}** of its movement "
        f"({_expl[0]:.1%} + {_expl[1]:.1%}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does adding dimensions to Rosenbrock create a local minimum (look at
    the first two terms when $x_1 = -1$)? the PCA plane captures almost all of the path: why
    are optimization paths so low-dimensional, and what does the 2-D picture *not* show you
    (think about the other four directions)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: random slices of a high-dimensional landscape

    the $n$-dimensional Rastrigin function has about $11^n$ local minima in $[-5, 5]^n$. we
    draw it with your `loss_landscape_2d` on a plane through a random point, spanned by two
    random **unit** directions, with span 3, and count the local minima visible on the
    161 × 161 grid, for $n = 2$ and for larger $n$ (slider).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the 2-D slice of 2-D Rastrigin is an egg carton. what does a random 2-D slice of 1,000-D Rastrigin look "
        "like? how many local minima will it show?"
    )
    p3
    return (p3,)


@app.cell
def _():
    dim = mo.ui.slider(10, 2000, value=1000, step=10, label="dimension n of the second slice")
    dim
    return (dim,)


@app.cell
def _(dim, p3):
    ui.gate(p3)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.4))
    _rows = []
    for _ax, _n in zip(_axes, (2, dim.value)):
        _rng = np.random.default_rng(2)
        _x0 = _rng.uniform(-2, 2, size=_n)
        _d1 = _rng.normal(size=_n)
        _d1 /= np.linalg.norm(_d1)
        _d2 = _rng.normal(size=_n)
        _d2 -= (_d2 @ _d1) * _d1
        _d2 /= np.linalg.norm(_d2)
        _al, _be, _Z = ex.loss_landscape_2d(ex.rastrigin, _x0, _d1, _d2, span=3.0, resolution=161)
        _ax.contourf(_al, _be, _Z.T, levels=30, cmap="viridis")
        _ax.set_aspect("equal")
        _nmin = count_local_minima(_Z)
        _ax.set(title=f"n = {_n}: {_nmin} local minima on the slice", xlabel="α (direction 1)", ylabel="β (direction 2)")
        _rows.append(f"| {_n} | {_nmin} | {_Z.min():.1f} to {_Z.max():.1f} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| n | local minima on the slice | range of f on the slice |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** along a random unit direction $d$ in $n$ dimensions, how big is a typical
    coordinate $d_i$? how far does each coordinate $x_i$ move as $\alpha$ goes from $-3$ to
    $3$, compared with the period 1 of the cosines? what happens to $\sum_i\cos(2\pi(x_i +
    \alpha d_i))$ then? what does this say about pictures of neural-network loss landscapes
    drawn along random directions?
    """)
    return


if __name__ == "__main__":
    app.run()
