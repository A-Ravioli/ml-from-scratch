import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, check_X_y, clone  # yours, from ch 07 · topic 01
    from mlfs.optim import SGD, Param, gradient_descent, minibatch_indices  # yours, from ch 06
    from mlfs.utils import as_rng, log_softmax, one_hot, softmax  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # softmax regression · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. `SoftmaxRegression` joins
    `mlfs.linear`. it uses your `Classifier` and `clone` (ch 07), your `gradient_descent`,
    `SGD`, `Param` and `minibatch_indices` (ch 06) and your stable `softmax` / `log_softmax`
    / `one_hot` (ch 00); if those aren't finished, launch with
    `MLFS_REFERENCE=base,optim,utils marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `softmax_loss`, `softmax_gradient`: multiclass cross-entropy and its gradient |
    | 2 | `SoftmaxRegression`, full-batch gradient descent → `mlfs.linear` |
    | 3 | `SoftmaxRegression(batch_size=...)`: minibatch SGD |
    | 4 | `OneVsRest`: the other way to go multiclass, for comparison |

    as in topic 04, `A` is a design matrix (n, p) with a first column of ones when there's an
    intercept, and `penalty` (p,) holds each row's ridge strength (0 for the intercept). the
    parameters are a MATRIX `Theta` (p, k): one column of weights per class. `Y` is the
    one-hot label matrix (n, k).

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def softmax_loss(Theta, A, Y, penalty):
    """stage 1: mean cross-entropy -mean_i sum_c Y_ic log softmax(A Theta)_ic, plus
    0.5 * sum_j penalty_j ||Theta_j,:||^2. a float; stable for huge scores (log_softmax)."""
    Z = A @ Theta
    nll = -np.sum(Y * log_softmax(Z, axis=1)) / len(A)
    return float(nll + 0.5 * np.sum(penalty[:, None] * Theta**2))


@app.function
def softmax_gradient(Theta, A, Y, penalty):
    """stage 1: the gradient of softmax_loss with respect to Theta, shape (p, k)."""
    return A.T @ (softmax(A @ Theta, axis=1) - Y) / len(A) + penalty[:, None] * Theta


@app.cell(hide_code=True)
def _():
    checks.run(1, softmax_loss=softmax_loss, softmax_gradient=softmax_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class SoftmaxRegression(Classifier):
    """multinomial logistic regression: P(classes_[c] | x) = softmax(x @ coef_ + intercept_)_c,
    fitted by minimizing the mean cross-entropy + (alpha / 2) ||coef_||_F^2."""

    def __init__(self, *, alpha=0.0, lr=0.1, max_iters=1000, batch_size=None, tol=1e-06, fit_intercept=True,
                 rng=None):
        self.alpha = alpha
        self.lr = lr
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.rng = rng

    def fit(self, X, y):
        """stage 2: fit and return self.

        sets classes_ (>= 2, else ValueError), coef_ (d, k), intercept_ (k,) (zeros without an
        intercept), n_iter_ and converged_. batch_size=None: your gradient_descent on the full
        objective, from Theta = 0, with lr, max_iters and tol. batch_size an int: hand A, Y,
        penalty and Theta = 0 to self._sgd (stage 3).
        """
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        X, y = check_X_y(X, y)
        self.classes_, idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("need at least 2 classes")
        Y = one_hot(idx, k)
        A = np.column_stack([np.ones(len(X)), X]) if self.fit_intercept else X
        pen = np.full(A.shape[1], float(self.alpha))
        if self.fit_intercept:
            pen[0] = 0.0
        Theta0 = np.zeros((A.shape[1], k))
        if self.batch_size is None:
            res = gradient_descent(lambda T: softmax_loss(T, A, Y, pen), lambda T: softmax_gradient(T, A, Y, pen),
                                   Theta0, lr=self.lr, max_iters=self.max_iters, tol=self.tol)
            Theta, self.n_iter_, self.converged_ = res.x, res.n_iters, res.converged
        else:
            Theta, self.n_iter_, self.converged_ = self._sgd(A, Y, pen, Theta0)
        if self.fit_intercept:
            self.intercept_, self.coef_ = np.asarray(Theta[0], dtype=np.float64), np.asarray(Theta[1:])
        else:
            self.intercept_, self.coef_ = np.zeros(k), np.asarray(Theta)
        return self

    def _sgd(self, A, Y, pen, Theta0):
        """stage 3: max_iters EPOCHS of minibatch SGD from Theta0; returns (Theta, epochs, converged).

        rng_ = as_rng(self.rng); one Param holding Theta and your SGD(lr=self.lr); each epoch
        takes its batches from minibatch_indices(n, batch_size, shuffle=True, rng=rng_), sets
        the Param's grad to the softmax_gradient on that batch (penalty included) and steps.
        after each epoch, stop (converged) if the FULL gradient norm is <= tol.
        """
        self.rng_ = as_rng(self.rng)
        param = Param(Theta0)
        opt = SGD([param], lr=self.lr)
        epochs, converged = 0, False
        while epochs < self.max_iters:
            for rows in minibatch_indices(len(A), self.batch_size, shuffle=True, rng=self.rng_):
                param.grad = softmax_gradient(param.data, A[rows], Y[rows], pen)
                opt.step()
            epochs += 1
            if np.linalg.norm(softmax_gradient(param.data, A, Y, pen)) <= self.tol:
                converged = True
                break
        return param.data, epochs, converged

    def decision_function(self, X):
        """stage 2: the class scores X @ coef_ + intercept_, shape (n, k) (ValueError on a wrong
        number of features)."""
        X, _ = check_X_y(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"fitted with {self.coef_.shape[0]} features, got {X.shape[1]}")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """stage 2: softmax of the scores, (n, k), column c for classes_[c]."""
        return softmax(self.decision_function(X), axis=1)


@app.cell(hide_code=True)
def _():
    checks.run(2, SoftmaxRegression=SoftmaxRegression, softmax_loss=softmax_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, SoftmaxRegression=SoftmaxRegression, softmax_loss=softmax_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class OneVsRest(Classifier):
    """k binary classifiers, one per class ("this class" versus "all the others")."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        """stage 4: fit clone(estimator) on (X, y == c) for each class c in classes_ (sorted);
        store them in estimators_ (a list, in classes_ order) and return self."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least 2 classes")
        self.estimators_ = [clone(self.estimator).fit(X, (y == c).astype(int)) for c in self.classes_]
        return self

    def predict_proba(self, X):
        """stage 4: column c is estimators_[c]'s probability of its positive class, and each row
        is rescaled to sum to 1. shape (n, k)."""
        P = np.column_stack([np.asarray(est.predict_proba(X))[:, 1] for est in self.estimators_])
        return P / P.sum(axis=1, keepdims=True)


@app.cell(hide_code=True)
def _():
    checks.run(4, OneVsRest=OneVsRest, SoftmaxRegression=SoftmaxRegression)
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
