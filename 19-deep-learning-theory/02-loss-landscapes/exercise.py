import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # loss landscapes · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours. stages 1–3 join **`mlfs.dltheory`**, and the chapter's capstone and later chapters
    use them to look at the landscapes of their own models.

    | stage | you build |
    |---|---|
    | 1 | `filter_normalized_direction`: a random direction scaled filter by filter to the weights → `mlfs.dltheory` |
    | 2 | `loss_surface_2d`: the loss on a plane through the weights → `mlfs.dltheory` |
    | 3 | `linear_interpolation`, `loss_barrier`: the loss along the straight line between two solutions → `mlfs.dltheory` |
    | 4 | `hessian_vector_product`, `top_hessian_eigenvalue`: sharpness by power iteration |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def filter_normalized_direction(model, *, generator=None):
    """stage 1: dict name -> random tensor like each parameter, each filter (row along dim 0) rescaled to that filter's norm; 1-D params zero."""
    raise NotImplementedError("stage1: implement filter_normalized_direction")


@app.cell(hide_code=True)
def _():
    checks.run(1, filter_normalized_direction=filter_normalized_direction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def loss_surface_2d(model, eval_loss, d1, d2, *, span=1.0, resolution=21):
    """stage 2: (alphas, betas, Z) with Z[i, j] = eval_loss at theta + alphas[i] d1 + betas[j] d2 on linspace(-span, span, resolution); weights restored."""
    raise NotImplementedError("stage2: implement loss_surface_2d")


@app.cell(hide_code=True)
def _():
    checks.run(2, loss_surface_2d=loss_surface_2d, filter_normalized_direction=filter_normalized_direction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def linear_interpolation(model, state_a, state_b, eval_loss, *, n_points=21):
    """stage 3: (alphas, losses): eval_loss after loading (1 - a) state_a + a state_b (float entries; others from state_a); state restored."""
    raise NotImplementedError("stage3: implement linear_interpolation")


@app.function
def loss_barrier(losses):
    """stage 3: max over the path of loss minus the straight line between the two end losses (0 if the path never rises above it)."""
    raise NotImplementedError("stage3: implement loss_barrier")


@app.cell(hide_code=True)
def _():
    checks.run(3, linear_interpolation=linear_interpolation, loss_barrier=loss_barrier)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def hessian_vector_product(loss_fn, params, vector):
    """stage 4: list of tensors H v, H the Hessian of loss_fn() with respect to params (list of tensors), v a list shaped like params; parameters the loss doesn't use get zeros."""
    raise NotImplementedError("stage4: implement hessian_vector_product")


@app.function
def top_hessian_eigenvalue(model, loss_fn, *, n_iters=20, generator=None):
    """stage 4: the Hessian eigenvalue of largest magnitude (signed), by n_iters steps of power iteration from a random start; no side effects on params or .grad."""
    raise NotImplementedError("stage4: implement top_hessian_eigenvalue")


@app.cell(hide_code=True)
def _():
    checks.run(4, hessian_vector_product=hessian_vector_product, top_hessian_eigenvalue=top_hessian_eigenvalue)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.dltheory import filter_normalized_direction, loss_surface_2d, linear_interpolation`
    gives you the landscape tools for any PyTorch model.
    """)
    return


if __name__ == "__main__":
    app.run()
