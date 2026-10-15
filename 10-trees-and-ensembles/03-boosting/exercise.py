import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, check_X_y  # yours, from ch 07
    from mlfs.trees import DecisionTreeClassifier  # yours, from topic 01
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # boosting · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. a random forest averages strong,
    independent trees; AdaBoost does the opposite: it adds weak trees one at a time, each
    trained on a reweighted dataset that emphasizes the previous trees' mistakes.

    | stage | you build |
    |---|---|
    | 1 | `samme_update`, `exponential_loss`: one round of reweighting, and the loss AdaBoost minimizes |
    | 2 | `AdaBoostClassifier`: fit, `decision_function`, `predict_proba` → `mlfs.trees` |
    | 3 | `staged_decision_function` and `margins`: watching the ensemble grow |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def samme_update(y, h, sample_weight, learning_rate=1.0):
    """stage 1: one boosting round for labels and predictions in {-1, +1} -> (alpha, new_weight summing to 1, weighted error)."""
    raise NotImplementedError("stage1: implement samme_update")


@app.function
def exponential_loss(y, f):
    """stage 1: the mean exponential loss mean(exp(-y f)) for y in {-1, +1} and scores f (n,)."""
    raise NotImplementedError("stage1: implement exponential_loss")


@app.cell(hide_code=True)
def _():
    checks.run(1, samme_update=samme_update, exponential_loss=exponential_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class AdaBoostClassifier(Classifier):
    """discrete AdaBoost (SAMME) for two classes, on small CART trees fitted with sample weights."""

    def __init__(self, n_estimators=50, *, learning_rate=1.0, max_depth=1, rng=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.rng = rng

    def fit(self, X, y):
        """stage 2: up to n_estimators rounds of fit-a-weighted-tree + samme_update; estimators_, estimator_weights_, estimator_errors_; returns self."""
        raise NotImplementedError("stage2: implement AdaBoostClassifier.fit")

    def decision_function(self, X):
        """stage 2: f(x) = sum_m alpha_m h_m(x) with h_m in {-1, +1} (+1 = classes_[1]), shape (n,)."""
        raise NotImplementedError("stage2: implement AdaBoostClassifier.decision_function")

    def predict_proba(self, X):
        """stage 2: (n, 2) with P(classes_[1]) = sigmoid(f(x)), the exponential loss's population minimizer."""
        raise NotImplementedError("stage2: implement AdaBoostClassifier.predict_proba")

    def staged_decision_function(self, X):
        """stage 3: a generator yielding f after 1, 2, ..., M rounds, each an array (n,)."""
        raise NotImplementedError("stage3: implement AdaBoostClassifier.staged_decision_function")

    def margins(self, X, y):
        """stage 3: normalized margins y f(x) / sum(alpha) in [-1, 1], with y mapped to {-1, +1}, shape (n,)."""
        raise NotImplementedError("stage3: implement AdaBoostClassifier.margins")


@app.cell(hide_code=True)
def _():
    checks.run(2, AdaBoostClassifier=AdaBoostClassifier, samme_update=samme_update,
               exponential_loss=exponential_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, AdaBoostClassifier=AdaBoostClassifier)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `AdaBoostClassifier` is part of `mlfs.trees`. open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
