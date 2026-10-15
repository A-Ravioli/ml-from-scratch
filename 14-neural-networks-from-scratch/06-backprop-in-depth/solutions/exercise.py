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
    # backprop in depth · build (reference solution)

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
    t = Tensor(x, requires_grad=True)
    out = f(t)
    out.backward(np.asarray(v, dtype=np.float64).reshape(out.shape))
    g = t.grad if t.grad is not None else np.zeros_like(t.data)
    return out.data.copy(), g


@app.function
def jacobian_reverse(f, x):
    """stage 1: the (m, n) Jacobian of f: R^n -> R^m at x (n,), one row per backward pass (one forward pass in total)."""
    t = Tensor(x, requires_grad=True)
    out = f(t)
    m = out.size
    J = np.zeros((m, t.size))
    for i in range(m):
        t.grad = None
        e = np.zeros(m)
        e[i] = 1.0
        out.backward(e.reshape(out.shape))
        if t.grad is not None:
            J[i] = t.grad.ravel()
    return J


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
    total, seen, stack = 0, set(), [out]
    while stack:
        node = stack.pop()
        if id(node) in seen or not node._parents:
            continue
        seen.add(id(node))
        total += node.data.nbytes
        stack.extend(node._parents)
    return total


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
    inputs = [x if isinstance(x, Tensor) else Tensor(x) for x in inputs]
    with no_grad():
        out = fn(*inputs)
    result = Tensor(out.data)
    if not is_grad_enabled():
        return result

    def recompute_vjp(g):
        fresh = [Tensor(x.data, requires_grad=x.requires_grad) for x in inputs]
        y = fn(*fresh)  # rebuilds this segment's graph (params closed over by fn get their grads here)
        if y.requires_grad:
            y.backward(g)
        return tuple(d.grad if d.requires_grad else None for d in fresh)

    # always a graph node, even if no input requires grad: fn may close over Parameters
    result.requires_grad = True
    result._parents = tuple(inputs)
    result._vjp = recompute_vjp
    return result


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
    functions = list(functions)
    segments = max(1, min(int(segments), len(functions)))
    chunks = [list(c) for c in np.array_split(np.arange(len(functions)), segments)]

    def run(chunk):
        def segment(h):
            for i in chunk:
                h = functions[i](h)
            return h
        return segment

    for chunk in chunks:
        x = checkpoint(run(chunk), x)
    return x


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
