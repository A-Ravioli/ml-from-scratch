import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson

    torch.set_default_dtype(torch.float64)  # curvature work needs float64


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # natural gradient · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. everything is float64 torch.

    | stage | you build |
    |---|---|
    | 1 | `steepest_direction`, `natural_gradient`: steepest descent under a metric |
    | 2 | `natural_gradient_descent`: the optimizer, for small models with a Fisher you can write down |
    | 3 | `conjugate_gradient`, `fisher_vector_product`: the Fisher of a network without ever forming it |
    | 4 | `ngd_step`: one natural-gradient step on a classifier |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def steepest_direction(grad, metric, radius):
    """stage 1: the step d minimizing grad . d subject to d^T metric d = radius^2; shape like grad."""
    raise NotImplementedError("stage1: implement steepest_direction")


@app.function
def natural_gradient(grad, fisher, *, damping=0.0):
    """stage 1: (fisher + damping I)^-1 grad, by a linear solve; shape like grad."""
    raise NotImplementedError("stage1: implement natural_gradient")


@app.cell(hide_code=True)
def _():
    checks.run(1, steepest_direction=steepest_direction, natural_gradient=natural_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def natural_gradient_descent(grad_fn, fisher_fn, theta0, *, lr, n_steps, damping=0.0):
    """stage 2: n_steps of theta <- theta - lr (F(theta) + damping I)^-1 grad(theta); the path, (n_steps + 1, P)."""
    raise NotImplementedError("stage2: implement natural_gradient_descent")


@app.cell(hide_code=True)
def _():
    checks.run(2, natural_gradient_descent=natural_gradient_descent)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def conjugate_gradient(mvp, b, *, max_iters=100, tol=1e-10):
    """stage 3: solve A x = b for symmetric positive-definite A given only v -> A v, from x0 = 0; returns x."""
    raise NotImplementedError("stage3: implement conjugate_gradient")


@app.function
def fisher_vector_product(model, X, v, *, damping=0.0):
    """stage 3: (F + damping I) v for the true Fisher of a softmax classifier, without forming F; v flat (P,)."""
    raise NotImplementedError("stage3: implement fisher_vector_product")


@app.cell(hide_code=True)
def _():
    checks.run(3, conjugate_gradient=conjugate_gradient, fisher_vector_product=fisher_vector_product)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ngd_step(model, X, y, *, lr=1.0, damping=1e-3, cg_iters=50):
    """stage 4: one natural-gradient step on the mean cross-entropy, in place; returns the loss before the step."""
    raise NotImplementedError("stage4: implement ngd_step")


@app.cell(hide_code=True)
def _():
    checks.run(4, ngd_step=ngd_step)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. topic 20 replaces the
    conjugate-gradient solve with a Kronecker-factored inverse that costs about as much as a
    few Adam steps.
    """)
    return


if __name__ == "__main__":
    app.run()
