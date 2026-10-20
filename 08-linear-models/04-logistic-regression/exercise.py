import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.optim import gradient_descent  # yours, from ch 06 · topic 02
    from mlfs.utils import log_sigmoid, sigmoid  # yours, from ch 00 · topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # logistic regression · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. `LogisticRegression` joins
    `mlfs.linear`. it uses your `Classifier` (ch 07), `gradient_descent` (ch 06) and stable
    `sigmoid` / `log_sigmoid` (ch 00); if those aren't finished, launch with
    `MLFS_REFERENCE=base,optim,utils marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `logistic_loss`, `logistic_gradient`: the penalized cross-entropy and its gradient |
    | 2 | `logistic_hessian`: its curvature |
    | 3 | `LogisticRegression` with `solver="gd"`: `fit`, `predict_proba`, `decision_function` → `mlfs.linear` |
    | 4 | `irls`: Newton's method / iteratively reweighted least squares (`solver="irls"`, the default) |

    throughout, `A` is a design matrix (n, p) whose first column is ones when there's an
    intercept, `theta` (p,) holds [intercept, weights], `y` is 0/1 floats and `penalty` (p,)
    holds the ridge strength of each entry of theta (0 for the intercept). the two helpers
    below build `A` and `penalty`.

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.function
def with_intercept_column(X, fit_intercept):
    """given: the design matrix [1, X] (a column of ones first) if fit_intercept, else X."""
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([np.ones(len(X)), X]) if fit_intercept else X


@app.function
def penalty_weights(n_features, alpha, fit_intercept):
    """given: the per-coefficient ridge strengths for theta: [0, alpha, ..., alpha] with an
    intercept (it isn't penalized), [alpha, ..., alpha] without. shape (p,)."""
    pen = np.full(n_features, float(alpha))
    return np.concatenate([[0.0], pen]) if fit_intercept else pen


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def logistic_loss(theta, A, y, penalty):
    """stage 1: mean cross-entropy of sigmoid(A @ theta) against y in {0, 1}, plus
    0.5 * sum(penalty * theta**2). a float, finite even for scores of +-1e4 (use log_sigmoid)."""
    raise NotImplementedError("stage1: implement logistic_loss")


@app.function
def logistic_gradient(theta, A, y, penalty):
    """stage 1: the gradient of logistic_loss with respect to theta, shape (p,)."""
    raise NotImplementedError("stage1: implement logistic_gradient")


@app.cell(hide_code=True)
def _():
    checks.run(1, logistic_loss=logistic_loss, logistic_gradient=logistic_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def logistic_hessian(theta, A, penalty):
    """stage 2: the Hessian of logistic_loss, A^T diag(s (1 - s)) A / n + diag(penalty), with
    s = sigmoid(A @ theta). shape (p, p); no (n, n) diagonal matrix."""
    raise NotImplementedError("stage2: implement logistic_hessian")


@app.cell(hide_code=True)
def _():
    checks.run(2, logistic_hessian=logistic_hessian, logistic_gradient=logistic_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class LogisticRegression(Classifier):
    """binary logistic regression: P(classes_[1] | x) = sigmoid(x @ coef_ + intercept_), fitted
    by minimizing the mean cross-entropy + (alpha / 2) ||coef_||^2."""

    def __init__(self, *, alpha=0.0, solver="irls", lr=0.1, max_iters=100, tol=1e-08, fit_intercept=True):
        self.alpha = alpha
        self.solver = solver
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """stage 3: fit with solver "gd" (your gradient_descent) or "irls" (stage 4); returns self.

        sets classes_ (exactly 2, else ValueError), coef_ (d,), intercept_ (float, 0.0 without
        one), n_iter_ and converged_. an unknown solver or alpha < 0 raises ValueError.
        """
        raise NotImplementedError("stage3: implement LogisticRegression.fit")

    def decision_function(self, X):
        """stage 3: the logit X @ coef_ + intercept_ (n,) (ValueError on a wrong number of features)."""
        raise NotImplementedError("stage3: implement LogisticRegression.decision_function")

    def predict_proba(self, X):
        """stage 3: (n, 2) probabilities, columns [P(classes_[0]), P(classes_[1])]."""
        raise NotImplementedError("stage3: implement LogisticRegression.predict_proba")


@app.cell(hide_code=True)
def _():
    checks.run(3, LogisticRegression=LogisticRegression)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def irls(A, y, penalty, *, max_iters=100, tol=1e-08):
    """stage 4: minimize logistic_loss by Newton's method with step halving, from theta = 0.

    each step tries theta - t H^-1 g for t = 1, 1/2, ..., 2^-30 and takes the first whose loss
    is at most f + 1e-12 (1 + |f|) (no increase, up to rounding); if none is, stop. stop
    (converged) when ||gradient|| <= tol, or after max_iters Newton steps. returns
    (theta, n_steps, converged).
    """
    raise NotImplementedError("stage4: implement irls")


@app.cell(hide_code=True)
def _():
    checks.run(4, irls=irls, LogisticRegression=LogisticRegression, logistic_loss=logistic_loss)
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
