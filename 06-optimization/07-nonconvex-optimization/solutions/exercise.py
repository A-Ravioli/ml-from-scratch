import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.optim import OptimResult  # yours, from topic 02
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # nonconvex optimization · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you have the two
    classic nonconvex test functions, a gradient method that escapes saddle points, and the
    tools people use to look at loss landscapes they can't see: 2-D slices, and the plane an
    optimizer's trajectory actually moved in.

    | stage | you build |
    |---|---|
    | 1 | `rosenbrock`, `rosenbrock_grad`, `rastrigin`, `rastrigin_grad`: a curved valley and an egg carton → `mlfs.optim` |
    | 2 | `perturbed_gradient_descent`: gradient descent that kicks itself off flat spots → `mlfs.optim` |
    | 3 | `loss_landscape_2d`: the loss on a plane through a point → `mlfs.optim` |
    | 4 | `trajectory_pca`: the two directions an optimizer's path actually used |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def rosenbrock(x, a=1.0, b=100.0):
    """stage 1: the generalized Rosenbrock function of x (n,), n >= 2:

        sum_{i=0}^{n-2} [ b (x_{i+1} - x_i^2)^2 + (a - x_i)^2 ]

    minimum 0 at x = (a, a^2, ...) (all ones for a = 1). vectorized; returns a float.
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2))


@app.function
def rosenbrock_grad(x, a=1.0, b=100.0):
    """stage 1: the gradient of rosenbrock, shape (n,). each x_i appears in term i (as x_i)
    and in term i - 1 (as x_{i+1} there): add both contributions. vectorized."""
    x = np.asarray(x, dtype=float)
    r = x[1:] - x[:-1] ** 2
    g = np.zeros_like(x)
    g[:-1] = -4 * b * x[:-1] * r - 2 * (a - x[:-1])
    g[1:] += 2 * b * r
    return g


@app.function
def rastrigin(x, A=10.0):
    """stage 1: the Rastrigin function of x (n,):  A n + sum_i (x_i^2 - A cos(2 pi x_i)).
    global minimum 0 at x = 0, and a local minimum near every integer point. returns a float."""
    x = np.asarray(x, dtype=float)
    return float(A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


@app.function
def rastrigin_grad(x, A=10.0):
    """stage 1: the gradient of rastrigin: 2 x + 2 pi A sin(2 pi x), shape (n,)."""
    x = np.asarray(x, dtype=float)
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


@app.cell(hide_code=True)
def _():
    checks.run(1, rosenbrock=rosenbrock, rosenbrock_grad=rosenbrock_grad, rastrigin=rastrigin,
               rastrigin_grad=rastrigin_grad)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def perturbed_gradient_descent(f, grad, x0, *, lr=0.01, noise_radius=0.1, grad_threshold=0.001, max_iters=1000,
                               rng=None, callback=None):
    """stage 2: gradient descent that adds random noise where the gradient is small.

    rng = as_rng(rng). each of the max_iters steps (k = 1, 2, ...) computes g = grad(x) once:
      if ||g||_2 < grad_threshold:  x <- x + xi, with xi uniform in the ball of radius
                                     noise_radius (direction: a normalized standard normal
                                     vector; radius: noise_radius * U^(1/n), U uniform(0, 1))
      else:                          x <- x - lr * g
    then append f(x) to the history and call callback(x, k) (return value ignored). stop early only if f(x) is not
    finite. near a minimum the noise keeps kicking x around, so the result is the BEST
    iterate seen: OptimResult(x = the iterate with the lowest f (x0 included), fun = its f,
    n_iters = steps taken, converged = ||grad(best x)|| < grad_threshold,
    history = [f(x0), f(x1), ...]).
    """
    rng = as_rng(rng)
    x = np.array(x0, dtype=float)
    n = x.size
    fx = float(f(x))
    history = [fx]
    best_x, best_f = x, fx
    k = 0
    while k < max_iters:
        g = np.asarray(grad(x), dtype=float)
        if np.linalg.norm(g) < grad_threshold:
            u = rng.normal(size=x.shape)
            xi = u / np.linalg.norm(u) * noise_radius * rng.random() ** (1.0 / n)
            x = x + xi
        else:
            x = x - lr * g
        fx = float(f(x))
        history.append(fx)
        k += 1
        if callback is not None:
            callback(x, k)
        if not np.isfinite(fx):
            break
        if fx < best_f:
            best_x, best_f = x, fx
    converged = bool(np.linalg.norm(np.asarray(grad(best_x), dtype=float)) < grad_threshold)
    return OptimResult(best_x, best_f, k, converged, history)


@app.cell(hide_code=True)
def _():
    checks.run(2, perturbed_gradient_descent=perturbed_gradient_descent)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def loss_landscape_2d(f, x0, d1, d2, *, span=1.0, resolution=51):
    """stage 3: f on the plane through x0 spanned by d1 and d2.

    alphas = betas = np.linspace(-span, span, resolution); returns (alphas, betas, Z) with
    Z (resolution, resolution), Z[i, j] = f(x0 + alphas[i] d1 + betas[j] d2): rows follow d1,
    columns follow d2 (so plot it with Z.T if alphas go on the horizontal axis).
    """
    x0 = np.asarray(x0, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    alphas = np.linspace(-span, span, resolution)
    betas = np.linspace(-span, span, resolution)
    Z = np.array([[f(x0 + a * d1 + b * d2) for b in betas] for a in alphas], dtype=float)
    return alphas, betas, Z


@app.cell(hide_code=True)
def _():
    checks.run(3, loss_landscape_2d=loss_landscape_2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def trajectory_pca(xs):
    """stage 4: the plane an optimization path lives in (Li et al., 2018).

    xs (k, n): iterates, the LAST one being the final point x_f. center them at the final
    point, D = xs - x_f, and take the top two right singular vectors of D (np.linalg.svd) as
    the orthonormal directions d1, d2 (n,). coords (k, 2) = D @ [d1, d2] are the path's
    coordinates in that plane (so x_f + coords @ [d1, d2] approximates the path), and
    explained (2,) = the fraction of sum ||D_i||^2 captured by each direction (s_i^2 / sum s^2).
    returns (d1, d2, coords, explained). the signs of d1 and d2 are arbitrary.
    """
    xs = np.asarray(xs, dtype=float)
    D = xs - xs[-1]
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    coords = D @ np.stack([d1, d2], axis=1)
    total = float(np.sum(s**2))
    explained = s[:2] ** 2 / total if total > 0 else np.zeros(2)
    return d1, d2, coords, explained


@app.cell(hide_code=True)
def _():
    checks.run(4, trajectory_pca=trajectory_pca)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, the test functions, `perturbed_gradient_descent` and
    `loss_landscape_2d` are part of `mlfs.optim`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
