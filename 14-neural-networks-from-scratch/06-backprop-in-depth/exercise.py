import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.tensor import Tensor, is_grad_enabled, no_grad  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # backprop in depth · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll look at your engine from the
    outside: what one backward pass computes (a vector-Jacobian product), what it costs in
    memory (every intermediate array the graph keeps alive), and how to trade that memory
    for compute (gradient checkpointing). `checkpoint` becomes part of **`mlfs.tensor`**.

    | stage | you build |
    |---|---|
    | 1 | `vjp` and `jacobian_reverse`: what one backward pass computes, and a Jacobian from $m$ of them |
    | 2 | `graph_nbytes`: how much memory a graph holds on to |
    | 3 | `checkpoint`: a node that forgets its intermediates and recomputes them in backward → `mlfs.tensor` |
    | 4 | `checkpoint_sequential`: $\sqrt{L}$ checkpointing for a stack of layers |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def vjp(f, x, v):
    """stage 1: (f(x), J_f(x)^T v) as ndarrays from ONE forward and ONE backward pass; f maps a Tensor to a Tensor."""
    raise NotImplementedError("stage1: implement vjp")


@app.function
def jacobian_reverse(f, x):
    """stage 1: the (m, n) Jacobian of f: R^n -> R^m at x (n,), one row per backward pass (one forward pass in total)."""
    raise NotImplementedError("stage1: implement jacobian_reverse")


@app.cell(hide_code=True)
def _():
    checks.run(1, vjp=vjp, jacobian_reverse=jacobian_reverse)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def graph_nbytes(out):
    """stage 2: total .data.nbytes of every non-leaf Tensor reachable from out through _parents (out included), each once."""
    raise NotImplementedError("stage2: implement graph_nbytes")


@app.cell(hide_code=True)
def _():
    checks.run(2, graph_nbytes=graph_nbytes)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def checkpoint(fn, *inputs):
    """stage 3: fn(*inputs) as a Tensor that stores no intermediates: forward under no_grad, backward recomputes fn."""
    raise NotImplementedError("stage3: implement checkpoint")


@app.cell(hide_code=True)
def _():
    checks.run(3, checkpoint=checkpoint, graph_nbytes=graph_nbytes)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def checkpoint_sequential(functions, x, segments):
    """stage 4: apply functions in order, split into `segments` contiguous chunks (np.array_split sizes), each run through checkpoint."""
    raise NotImplementedError("stage4: implement checkpoint_sequential")


@app.cell(hide_code=True)
def _():
    checks.run(4, checkpoint=checkpoint, checkpoint_sequential=checkpoint_sequential, graph_nbytes=graph_nbytes)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.tensor import checkpoint` gives you this function.
    """)
    return


if __name__ == "__main__":
    app.run()
