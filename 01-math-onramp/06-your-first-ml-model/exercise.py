import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # your first ML model · build

    time to put everything together. you'll fit a straight line $\hat y = wx + b$ to data
    the way every model in this course is trained: measure how wrong it is with a **loss**,
    compute the **gradient** of the loss by hand with the derivative rules from topic 02,
    and walk downhill with **gradient descent**. then you'll check the answer against the
    exact solution you can derive with a little algebra, and learn why rescaling your data
    can make training a thousand times faster.

    every function has step-by-step comments inside it: replace its last line (the one that
    raises "not implemented") with code that follows the steps.

    | stage | you build |
    |---|---|
    | 1 | `predict`, `mse_loss`: the model and how wrong it is |
    | 2 | `mse_gradient`: the hand-derived gradient of the loss |
    | 3 | `gradient_descent`: training, step by step |
    | 4 | `closed_form_fit`: the exact best line |
    | 5 | `standardize`, `unstandardize_params`: rescaling inputs, and undoing it |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def predict(x, w, b):
    """stage 1: the model's predictions y_hat = w * x + b for every input in x.

    x: 1-D array of inputs (n,); w (slope) and b (intercept): floats. returns an array (n,).
    example: predict([0, 1, 2], 2.0, 1.0) = [1, 3, 5]
    """
    # step 1: x = np.asarray(x, dtype=float).
    # step 2: return w * x + b.
    raise NotImplementedError("stage1: implement predict")


@app.function
def mse_loss(x, y, w, b):
    """stage 1: the mean squared error of the line (w, b) on the data (x, y):

        L(w, b) = (1/n) * sum over i of (w x_i + b - y_i)^2

    x, y: 1-D arrays (n,). returns a float.
    example: mse_loss([1, 2, 3], [2, 4, 5], 1.0, 1.0) = (0^2 + 1^2 + 1^2) / 3 = 0.6667
    """
    # step 1: residuals r = predict(x, w, b) - y   (how far off each prediction is).
    # step 2: return the MEAN of r ** 2, as a float.
    raise NotImplementedError("stage1: implement mse_loss")


@app.cell(hide_code=True)
def _():
    checks.run(1, predict=predict, mse_loss=mse_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def mse_gradient(x, y, w, b):
    """stage 2: the gradient of mse_loss: the two partial derivatives (dL/dw, dL/db).

    with residuals r_i = w x_i + b - y_i (derive these on paper first!):
        dL/dw = (2/n) * sum of r_i * x_i
        dL/db = (2/n) * sum of r_i
    returns a tuple of two floats (dL_dw, dL_db).
    example: mse_gradient([1, 2, 3], [2, 4, 5], 1.0, 1.0) = (-10/3, -4/3)
    """
    # step 1: x, y as float arrays; n = len(x).
    # step 2: r = predict(x, w, b) - y.
    # step 3: dw = 2 * np.sum(r * x) / n  and  db = 2 * np.sum(r) / n.
    # step 4: return float(dw), float(db).
    raise NotImplementedError("stage2: implement mse_gradient")


@app.cell(hide_code=True)
def _():
    checks.run(2, mse_gradient=mse_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def gradient_descent(x, y, *, lr=0.1, n_steps=100, w0=0.0, b0=0.0):
    """stage 3: fit (w, b) by gradient descent on mse_loss. returns (w, b, losses).

    start at (w0, b0). each step computes the gradient at the CURRENT (w, b), then moves
    both parameters together, against the gradient:
        w <- w - lr * dL/dw,    b <- b - lr * dL/db
    losses: a 1-D array of length n_steps + 1, the loss BEFORE the first step and after each step
    (so losses[0] = mse_loss at (w0, b0) and losses[-1] = the loss at the returned (w, b)).
    w and b are returned as floats.
    """
    # step 1: w, b = float(w0), float(b0); losses = [mse_loss(x, y, w, b)].
    # step 2: repeat n_steps times:
    #           dw, db = mse_gradient(x, y, w, b)        (both from the same, current point)
    #           w, b = w - lr * dw, b - lr * db
    #           losses.append(mse_loss(x, y, w, b))
    # step 3: return w, b, np.array(losses).
    raise NotImplementedError("stage3: implement gradient_descent")


@app.cell(hide_code=True)
def _():
    checks.run(3, gradient_descent=gradient_descent, mse_loss=mse_loss, mse_gradient=mse_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def closed_form_fit(x, y):
    """stage 4: the exact least-squares line, from setting both partial derivatives to zero.

        w = sum((x_i - x_mean) * (y_i - y_mean)) / sum((x_i - x_mean)^2)
        b = y_mean - w * x_mean
    raise ValueError if all the x values are equal (then no single best slope exists).
    returns (w, b), two floats.
    example: closed_form_fit([1, 2, 3], [2, 4, 5]) = (1.5, 0.6667)
    """
    # step 1: x, y as float arrays; dx = x - x.mean() and dy = y - y.mean().
    # step 2: sxx = np.sum(dx * dx); if sxx == 0, raise ValueError.
    # step 3: w = np.sum(dx * dy) / sxx and b = y.mean() - w * x.mean().
    # step 4: return float(w), float(b).
    raise NotImplementedError("stage4: implement closed_form_fit")


@app.cell(hide_code=True)
def _():
    checks.run(4, closed_form_fit=closed_form_fit, gradient_descent=gradient_descent)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def standardize(x):
    """stage 5: rescale x to mean 0 and standard deviation 1. returns (z, mean, std).

    z = (x - mean) / std, with mean = x.mean() and std = x.std() (NumPy's default, dividing by n).
    raise ValueError if std == 0. mean and std are floats; z is an array shaped like x.
    example: standardize([1, 2, 3]) gives z = [-1.2247, 0, 1.2247], mean 2.0, std 0.8165
    """
    # step 1: x = np.asarray(x, dtype=float); mean = x.mean(); std = x.std().
    # step 2: if std == 0, raise ValueError.
    # step 3: return (x - mean) / std, float(mean), float(std).
    raise NotImplementedError("stage5: implement standardize")


@app.function
def unstandardize_params(w_z, b_z, mean, std):
    """stage 5: convert a line fitted to standardized inputs back to the original units.

    if y_hat = w_z * z + b_z with z = (x - mean) / std, then expanding gives
    y_hat = (w_z / std) * x + (b_z - w_z * mean / std). returns (w, b), two floats.
    """
    # step 1: w = w_z / std.
    # step 2: b = b_z - w_z * mean / std.
    # step 3: return float(w), float(b).
    raise NotImplementedError("stage5: implement unstandardize_params")


@app.cell(hide_code=True)
def _():
    checks.run(5, standardize=standardize, unstandardize_params=unstandardize_params,
               gradient_descent=gradient_descent, closed_form_fit=closed_form_fit)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. you'll watch learning
    rates that are too small, just right and too large, see a model in the wrong units refuse
    to train, and look at the shape of the loss surface that explains both. then the
    chapter capstone does the same with eight features and real house prices.
    """)
    return


if __name__ == "__main__":
    app.run()
