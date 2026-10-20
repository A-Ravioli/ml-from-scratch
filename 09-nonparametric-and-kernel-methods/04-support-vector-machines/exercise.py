import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.kernels import get_kernel  # yours, from topic 03
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # support vector machines · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions and a class to write, and checks them live as you edit. `SVC` becomes part of
    `mlfs.kernels`; it builds its Gram matrices with your `get_kernel` from topic 03 (launch
    with `MLFS_REFERENCE=kernels marimo edit exercise.py` if that isn't finished).

    | stage | you build |
    |---|---|
    | 1 | `hinge_loss`, `primal_objective`: the soft-margin problem as a loss plus a penalty |
    | 2 | `dual_objective`, `kkt_violations`: the dual problem, and how to tell when you've solved it |
    | 3 | `smo_pair_update`: the analytic update of two dual variables, SMO's inner step |
    | 4 | `SVC`: a kernel SVM trained by SMO → `mlfs.kernels` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def hinge_loss(y, scores):
    """stage 1: the hinge loss max(0, 1 - y_i f_i) of every example, shape (n,), for labels y in {-1, +1} and scores f."""
    raise NotImplementedError("stage1: implement hinge_loss")


@app.function
def primal_objective(w, b, X, y, C):
    """stage 1: the soft-margin objective 0.5 ||w||^2 + C sum_i hinge(y_i, w . x_i + b), a float."""
    raise NotImplementedError("stage1: implement primal_objective")


@app.cell(hide_code=True)
def _():
    checks.run(1, hinge_loss=hinge_loss, primal_objective=primal_objective)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def dual_objective(alpha, K, y):
    """stage 2: the dual objective sum_i alpha_i - 0.5 sum_ij alpha_i alpha_j y_i y_j K_ij, a float."""
    raise NotImplementedError("stage2: implement dual_objective")


@app.function
def kkt_violations(alpha, y, f, C, tol=1e-3):
    """stage 2: boolean mask (n,) of examples breaking the KKT conditions by more than tol, given decision values f (with b)."""
    raise NotImplementedError("stage2: implement kkt_violations")


@app.cell(hide_code=True)
def _():
    checks.run(2, dual_objective=dual_objective, kkt_violations=kkt_violations)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def smo_pair_update(i, j, alpha, b, K, y, C, E):
    """stage 3: optimize alpha_i, alpha_j jointly (the rest fixed) -> (new alpha array, new b), or None if the pair can't make progress."""
    raise NotImplementedError("stage3: implement smo_pair_update")


@app.cell(hide_code=True)
def _():
    checks.run(3, smo_pair_update=smo_pair_update, dual_objective=dual_objective)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def final_bias(alpha, y, f0, C, b_fallback):
    """(given) the bias from the KKT conditions, given f0 = f - b at the training points: the mean of y_i - f0_i over
    free support vectors (0 < alpha_i < C); with none, the midpoint of the interval the bounded points allow
    (alpha = 0 needs y f >= 1, alpha = C needs y f <= 1), or b_fallback if that interval is empty or unbounded."""
    eps = 1e-8 * C
    r = y - f0
    free = (alpha > eps) & (alpha < C - eps)
    if free.any():
        return float(np.mean(r[free]))
    lower = ((alpha <= eps) & (y > 0)) | ((alpha >= C - eps) & (y < 0))  # these points need b >= r_i
    lo = r[lower].max() if lower.any() else -np.inf
    hi = r[~lower].min() if (~lower).any() else np.inf
    if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
        return float((lo + hi) / 2)
    return float(b_fallback)


@app.class_definition
class SVC(Classifier):
    """a binary kernel SVM: f(x) = sum_i alpha_i y_i k(x_i, x) + b, trained on the dual by SMO."""

    def __init__(self, C=1.0, *, kernel="rbf", gamma=1.0, degree=3, coef0=1.0, tol=0.001, max_passes=10,
                 max_iters=10000, rng=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.max_iters = max_iters
        self.rng = rng

    def fit(self, X, y):
        """stage 4: run SMO until max_passes sweeps change nothing (or max_iters sweeps); set classes_, alpha_, support_, support_vectors_, dual_coef_, intercept_, n_iter_; return self."""
        raise NotImplementedError("stage4: implement SVC.fit")

    def decision_function(self, X):
        """stage 4: f(x) = sum over support vectors of dual_coef_ k(sv, x) + intercept_, shape (m,)."""
        raise NotImplementedError("stage4: implement SVC.decision_function")

    def predict(self, X):
        """stage 4: classes_[1] where decision_function > 0, else classes_[0]."""
        raise NotImplementedError("stage4: implement SVC.predict")

    def predict_proba(self, X):
        raise NotImplementedError("an SVM's margin is not a probability: use decision_function (or calibrate it, ch 07)")


@app.cell(hide_code=True)
def _():
    checks.run(4, SVC=SVC, final_bias=final_bias, kkt_violations=kkt_violations)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `from mlfs.kernels import SVC` gives you your own SVM. open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
