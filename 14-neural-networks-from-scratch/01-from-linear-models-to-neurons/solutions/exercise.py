import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # from linear models to neurons · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. everything here is plain NumPy:
    you'll build a multilayer perceptron from matrix products and nonlinearities, wire one
    by hand to compute XOR (which no linear model can), and train one with gradients you
    derive yourself. deriving those gradients by hand is the pain the rest of the chapter
    removes.

    | stage | you build |
    |---|---|
    | 1 | `linear_layer` (torch layout), `relu`, `sigmoid` |
    | 2 | `mlp_forward` and `collapse_linear` (stacked linear layers are one linear layer) |
    | 3 | `xor_params`: a 2-2-1 ReLU network that computes XOR exactly |
    | 4 | `two_layer_loss_and_grads`: backprop by hand for a one-hidden-layer network |
    | 5 | `train_two_layer`: gradient descent on it |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def linear_layer(X, W, b):
    """stage 1: X (n, in) @ W.T + b, with W (out, in) and b (out,); returns (n, out)."""
    return X @ W.T + b


@app.function
def relu(z):
    """stage 1: elementwise max(z, 0)."""
    return np.maximum(z, 0.0)


@app.function
def sigmoid(z):
    """stage 1: elementwise 1 / (1 + exp(-z)), with no overflow warnings for large |z|."""
    return np.exp(-np.logaddexp(0.0, -z))


@app.cell(hide_code=True)
def _():
    checks.run(1, linear_layer=linear_layer, relu=relu, sigmoid=sigmoid)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def mlp_forward(X, params, activation=relu):
    """stage 2: apply the layers [(W1, b1), ..., (WL, bL)] to X (n, d), with `activation` after every layer but the last."""
    h = X
    for i, (W, b) in enumerate(params):
        h = linear_layer(h, W, b)
        if i < len(params) - 1:
            h = activation(h)
    return h


@app.function
def collapse_linear(params):
    """stage 2: the single (W, b) that equals the layers [(W1, b1), ..., (WL, bL)] applied with NO activations."""
    W, b = params[0]
    W, b = np.array(W, dtype=float), np.array(b, dtype=float)
    for W2, b2 in params[1:]:
        W, b = W2 @ W, W2 @ b + b2
    return W, b


@app.cell(hide_code=True)
def _():
    checks.run(2, mlp_forward=mlp_forward, collapse_linear=collapse_linear, relu=relu, linear_layer=linear_layer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def xor_params():
    """stage 3: [(W1 (2, 2), b1 (2,)), (W2 (1, 2), b2 (1,))] so that mlp_forward(X, params) is XOR(x1, x2) on the four corners."""
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    W2 = np.array([[1.0, -2.0]])
    b2 = np.array([0.0])
    return [(W1, b1), (W2, b2)]


@app.cell(hide_code=True)
def _():
    checks.run(3, xor_params=xor_params, mlp_forward=mlp_forward, collapse_linear=collapse_linear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def two_layer_loss_and_grads(X, y, params, activation="tanh"):
    """stage 4: mean binary cross-entropy of sigmoid(W2 act(W1 x + b1) + b2) and its gradients [(dW1, db1), (dW2, db2)].

    activation is "tanh" or "identity"; X (n, d), y (n,) in {0, 1}; W2 has shape (1, hidden).
    """
    (W1, b1), (W2, b2) = params
    n = X.shape[0]
    z1 = linear_layer(X, W1, b1)
    h = np.tanh(z1) if activation == "tanh" else z1
    z2 = linear_layer(h, W2, b2)[:, 0]
    # mean of softplus(z2) - y z2: stable binary cross-entropy from the logit
    loss = float(np.mean(np.logaddexp(0.0, z2) - y * z2))
    dz2 = (sigmoid(z2) - y)[:, None] / n           # (n, 1)
    dW2 = dz2.T @ h                                # (1, hidden)
    db2 = dz2.sum(axis=0)                          # (1,)
    dh = dz2 @ W2                                  # (n, hidden)
    dz1 = dh * (1.0 - h ** 2) if activation == "tanh" else dh
    dW1 = dz1.T @ X                                # (hidden, d)
    db1 = dz1.sum(axis=0)                          # (hidden,)
    return loss, [(dW1, db1), (dW2, db2)]


@app.cell(hide_code=True)
def _():
    checks.run(4, two_layer_loss_and_grads=two_layer_loss_and_grads)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_two_layer(X, y, hidden, *, lr=1.0, n_steps=2000, activation="tanh", rng=None):
    """stage 5: full-batch gradient descent from a N(0, 1) init (W1, b1, W2, b2 in that order); returns (params, losses)."""
    rng = as_rng(rng)
    d = X.shape[1]
    params = [(rng.normal(size=(hidden, d)), rng.normal(size=hidden)),
              (rng.normal(size=(1, hidden)), rng.normal(size=1))]
    losses = []
    for _ in range(n_steps):
        loss, grads = two_layer_loss_and_grads(X, y, params, activation)
        losses.append(loss)
        params = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]
    return params, losses


@app.cell(hide_code=True)
def _():
    checks.run(5, train_two_layer=train_two_layer, two_layer_loss_and_grads=two_layer_loss_and_grads,
               mlp_forward=mlp_forward, sigmoid=sigmoid)
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
