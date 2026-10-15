import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the perceptron · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. `Perceptron` joins `mlfs.linear`.
    it uses your `Classifier` and `check_X_y` (ch 07) and `as_rng` (ch 00); if those aren't
    finished, launch with `MLFS_REFERENCE=base,utils marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `perceptron_step`: one example, one possible mistake, one update |
    | 2 | `Perceptron`: epochs, shuffling, stopping, `decision_function`, `predict` → `mlfs.linear` |
    | 3 | the averaged perceptron (`averaged=True`), in O(d) per step |
    | 4 | `margin`, `mistake_bound`: the quantities in the convergence theorem |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def perceptron_step(w, b, x, y, *, fit_intercept=True):
    """stage 1: one perceptron step on the example (x, y), with y in {-1, +1}.

    a mistake is y * (w @ x + b) <= 0 (a point exactly on the boundary counts as a mistake).
    on a mistake: w + y x and, if fit_intercept, b + y (else b unchanged). otherwise nothing
    changes. returns (new w (d,), new b (float), mistake (bool)). never modify w in place:
    return a new array.
    """
    if y * (float(w @ x) + b) <= 0:
        return w + y * x, (b + y) if fit_intercept else b, True
    return w, b, False


@app.cell(hide_code=True)
def _():
    checks.run(1, perceptron_step=perceptron_step)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class Perceptron(Classifier):
    """Rosenblatt's perceptron for two classes (any two labels; classes_[1] is the +1 class).

    fitted attributes: classes_, coef_ (d,), intercept_ (float), n_iter_ (epochs run),
    n_updates_ (total mistakes), mistakes_ (list: mistakes per epoch) and rng_.
    """

    def __init__(self, *, max_epochs=100, averaged=False, fit_intercept=True, shuffle=True, rng=None):
        self.max_epochs = max_epochs
        self.averaged = averaged
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.rng = rng

    def fit(self, X, y):
        """stage 2 (averaged=False) and stage 3 (averaged=True): train and return self.

        X, y = check_X_y(X, y); classes_ = np.unique(y), exactly 2 (else ValueError); the
        targets are -1 for classes_[0] and +1 for classes_[1]. rng_ = as_rng(self.rng).
        start from w = 0, b = 0. each epoch visits every example once, in the order
        rng_.permutation(n) if shuffle else 0, 1, ..., n-1, calling perceptron_step. stop after
        the first epoch with no mistakes, or after max_epochs epochs.
        averaged=False: coef_, intercept_ are the final w, b.
        averaged=True: coef_, intercept_ are the AVERAGE of (w, b) over every visit of every
        epoch run (the value after each step, mistake or not). don't store every w: keep a
        running sum, or use the lesson's trick that only does work on mistakes.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Perceptron needs exactly 2 classes, got {len(self.classes_)}")
        t = np.where(y == self.classes_[1], 1.0, -1.0)
        self.rng_ = as_rng(self.rng)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        u, ub = np.zeros(d), 0.0  # sum over updates of (visits before the update) * update
        visits = 0
        self.mistakes_ = []
        for _ in range(self.max_epochs):
            order = self.rng_.permutation(n) if self.shuffle else np.arange(n)
            mistakes = 0
            for i in order:
                w_new, b_new, wrong = perceptron_step(w, b, X[i], t[i], fit_intercept=self.fit_intercept)
                if wrong:
                    mistakes += 1
                    u += visits * (w_new - w)
                    ub += visits * (b_new - b)
                    w, b = w_new, b_new
                visits += 1
            self.mistakes_.append(mistakes)
            if mistakes == 0:
                break
        self.n_iter_ = len(self.mistakes_)
        self.n_updates_ = int(sum(self.mistakes_))
        if self.averaged and visits:
            # mean over visits t = 1..T of w_t = w_T - (1/T) sum_updates (visits before it) * update
            w, b = w - u / visits, b - ub / visits
        self.coef_ = np.asarray(w, dtype=np.float64)
        self.intercept_ = float(b)
        return self

    def decision_function(self, X):
        """stage 2: the signed score X @ coef_ + intercept_ (n,); positive means classes_[1].
        X = check_X_y(X)[0]; a different number of features than in fit raises ValueError."""
        X, _ = check_X_y(X)
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"fitted with {len(self.coef_)} features, got {X.shape[1]}")
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """stage 2: classes_[1] where decision_function > 0, else classes_[0]. (the perceptron
        has no probabilities, so Classifier's argmax-of-predict_proba can't be used.)"""
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


@app.cell(hide_code=True)
def _():
    checks.run(2, Perceptron=Perceptron)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, Perceptron=Perceptron)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def margin(X, y, w, b):
    """stage 4: the geometric margin of the hyperplane w @ x + b = 0 on (X, y), y in {-1, +1}:
    min_i y_i (w @ x_i + b) / ||w||. negative if some point is misclassified. a float."""
    X = np.asarray(X, dtype=np.float64)
    return float(np.min(np.asarray(y) * (X @ w + b)) / np.linalg.norm(w))


@app.function
def mistake_bound(X, y, w, b):
    """stage 4: Novikoff's bound (R / gamma)^2 on the number of perceptron updates (with an
    intercept), for a separating hyperplane (w, b). work with augmented points [x_i, 1] and
    weights [w, b]: R = max_i ||[x_i, 1]||, gamma = min_i y_i ([w, b] @ [x_i, 1]) / ||[w, b]||.
    returns a float; np.inf if (w, b) doesn't separate the data (gamma <= 0)."""
    X = np.asarray(X, dtype=np.float64)
    Xa = np.column_stack([X, np.ones(len(X))])
    wa = np.append(np.asarray(w, dtype=np.float64), b)
    gamma = float(np.min(np.asarray(y) * (Xa @ wa)) / np.linalg.norm(wa))
    if gamma <= 0:
        return float(np.inf)
    R = float(np.max(np.linalg.norm(Xa, axis=1)))
    return (R / gamma) ** 2


@app.cell(hide_code=True)
def _():
    checks.run(4, margin=margin, mistake_bound=mistake_bound, Perceptron=Perceptron)
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
