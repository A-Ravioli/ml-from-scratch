import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.transformer import TransformerBlock  # yours, from ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # in-context learning as meta-learning · build

    spec-driven (L4): the **spec** boxes in the lesson define every function and class, the
    checks enforce them, and the internals are yours. a batch of $B$ regression tasks is
    `xs (B, n, d)`, `ys (B, n)`; every predictor returns `(B, n)`, where entry $i$ predicts
    $y_i$ from the first $i$ pairs (the **context**) and $x_i$, so entry 0 has no context at all.

    | stage | you build |
    |---|---|
    | 1 | `sample_linear_tasks`, `least_squares_predictions`, `position_mse`: the task family and its classical answer |
    | 2 | `gd_predictions`, `lsa_predict`, `gd_lsa_weights`: one step of gradient descent, written as linear self-attention |
    | 3 | `interleave`, `ICLTransformer`: a causal transformer that reads $(x_1, y_1, \dots, x_i)$ and predicts $y_i$ |
    | 4 | `train_icl`: meta-training on a stream of fresh tasks |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def sample_linear_tasks(n_tasks, n_points, dim, *, noise_std=0.0, x_scale=1.0, generator=None):
    """stage 1: dict xs (B, n, d) ~ N(0, x_scale^2 I), w (B, d) ~ N(0, I), ys (B, n) = xs @ w + noise_std * N(0, 1)."""
    raise NotImplementedError("stage1: implement sample_linear_tasks")


@app.function
def least_squares_predictions(xs, ys, *, ridge=0.0):
    """stage 1: (B, n): entry i = x_i . w_hat from the first i pairs, minimum-norm least squares (ridge=0) or ridge; entry 0 is 0."""
    raise NotImplementedError("stage1: implement least_squares_predictions")


@app.function
def position_mse(preds, ys):
    """stage 1: (n,) mean over tasks of the squared error at each position."""
    raise NotImplementedError("stage1: implement position_mse")


@app.cell(hide_code=True)
def _():
    checks.run(1, sample_linear_tasks=sample_linear_tasks, least_squares_predictions=least_squares_predictions,
               position_mse=position_mse)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gd_predictions(xs, ys, *, lr, n_steps=1):
    """stage 2: (B, n): entry i = x_i . w after n_steps of GD from w = 0 on 1/2 sum_{j<i} (w . x_j - y_j)^2."""
    raise NotImplementedError("stage2: implement gd_predictions")


@app.function
def lsa_predict(xs, ys, x_query, W_kq, W_pv):
    """stage 2: (B,) last coordinate of e_q + W_pv sum_j e_j (e_j^T W_kq e_q), with e_j = (x_j, y_j) and e_q = (x_q, 0)."""
    raise NotImplementedError("stage2: implement lsa_predict")


@app.function
def gd_lsa_weights(dim, lr):
    """stage 2: (W_kq, W_pv), each (dim+1, dim+1), for which lsa_predict equals one GD step from w = 0 with step size lr."""
    raise NotImplementedError("stage2: implement gd_lsa_weights")


@app.cell(hide_code=True)
def _():
    checks.run(2, gd_predictions=gd_predictions, lsa_predict=lsa_predict, gd_lsa_weights=gd_lsa_weights)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def interleave(xs, ys):
    """stage 3: (B, 2n, d+1) tokens: position 2i is (x_i, 0), position 2i+1 is (0, ..., 0, y_i)."""
    raise NotImplementedError("stage3: implement interleave")


@app.class_definition
class ICLTransformer(nn.Module):
    """a causal transformer over interleave(xs, ys) that predicts y_i at the token of x_i."""

    def __init__(self, dim, *, max_points=40, d_model=64, n_layers=4, n_heads=4):
        """stage 3: read_in (d+1 -> d_model), learned positions pos (2*max_points, d_model), blocks, ln_f, read_out (d_model -> 1)."""
        raise NotImplementedError("stage3: implement ICLTransformer.__init__")

    def forward(self, xs, ys):
        """stage 3: (B, n, d), (B, n) -> (B, n) predictions; prediction i sees x_1, y_1, ..., x_i only."""
        raise NotImplementedError("stage3: implement ICLTransformer.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, interleave=interleave, ICLTransformer=ICLTransformer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_icl(model, *, n_points, n_steps, batch_size=64, lr=1e-3, noise_std=0.0, seed=0):
    """stage 4: Adam on the mean squared error over fresh tasks each step (train mode); returns the list of losses, model left in eval mode."""
    raise NotImplementedError("stage4: implement train_icl")


@app.cell(hide_code=True)
def _():
    checks.run(4, sample_linear_tasks=sample_linear_tasks, least_squares_predictions=least_squares_predictions,
               position_mse=position_mse, ICLTransformer=ICLTransformer, train_icl=train_icl)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it trains your transformer,
    looks inside a one-layer attention model, and ends with the block's project: in-context
    learning against least squares and your MAML from topic 09.
    """)
    return


if __name__ == "__main__":
    app.run()
