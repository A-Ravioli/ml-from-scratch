import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, Regressor, check_X_y  # yours, from ch 07
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gradient boosting · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. you'll build an XGBoost-style
    booster: features binned once into small integers, gradient/hessian histograms built with
    one `bincount`, second-order split gains, and shrinkage.

    | stage | you build |
    |---|---|
    | 1 | `quantile_bin_edges`, `bin_data`: every feature as at most `n_bins` integer codes |
    | 2 | `gradient_histograms`, `split_gain`, `leaf_weight`, `best_hist_split`: the second-order split search |
    | 3 | `HistGradientTree`: a depth-limited tree on (codes, gradients, hessians), with histogram subtraction |
    | 4 | `GradientBoostingRegressor`: squared loss, shrinkage, row subsampling → `mlfs.trees` |
    | 5 | `GradientBoostingClassifier`: binary log loss → `mlfs.trees` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def quantile_bin_edges(X, n_bins=256):
    """stage 1: per-feature sorted bin edges (a list of d arrays, each of length < n_bins): midpoints if few distinct values, else unique quantiles."""
    raise NotImplementedError("stage1: implement quantile_bin_edges")


@app.function
def bin_data(X, edges):
    """stage 1: integer codes (n, d), uint8: code = number of edges strictly below x, so code <= b exactly when x <= edges[b]."""
    raise NotImplementedError("stage1: implement bin_data")


@app.cell(hide_code=True)
def _():
    checks.run(1, quantile_bin_edges=quantile_bin_edges, bin_data=bin_data)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gradient_histograms(codes, g, h, n_bins):
    """stage 2: (G, H), each (d, n_bins): the sums of g and of h over the rows in each (feature, bin), from one bincount each."""
    raise NotImplementedError("stage2: implement gradient_histograms")


@app.function
def split_gain(G_L, H_L, G_R, H_R, reg_lambda=1.0, gamma=0.0):
    """stage 2: 1/2 [G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - (G_L+G_R)^2/(H_L+H_R+lam)] - gamma, elementwise."""
    raise NotImplementedError("stage2: implement split_gain")


@app.function
def leaf_weight(G, H, reg_lambda=1.0):
    """stage 2: the optimal leaf value -G / (H + lambda)."""
    raise NotImplementedError("stage2: implement leaf_weight")


@app.function
def best_hist_split(G_hist, H_hist, *, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
    """stage 2: best "code <= b" split from (d, n_bins) histograms -> (feature, bin, gain); (-1, -1, -inf) if no child-weight-valid split."""
    raise NotImplementedError("stage2: implement best_hist_split")


@app.cell(hide_code=True)
def _():
    checks.run(2, gradient_histograms=gradient_histograms, split_gain=split_gain, leaf_weight=leaf_weight,
               best_hist_split=best_hist_split)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class HistGradientTree:
    """one second-order regression tree on binned features: splits "code <= bin", leaves -G/(H + lambda)."""

    def __init__(self, max_depth=3, min_child_weight=1.0, reg_lambda=1.0, gamma=0.0, n_bins=256):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins

    def fit(self, codes, g, h):
        """stage 3: grow depth-first (explicit stack), splitting while depth < max_depth and the best gain > 0; returns self."""
        raise NotImplementedError("stage3: implement HistGradientTree.fit")

    def predict(self, codes):
        """stage 3: the leaf value of every row of codes (n, d), shape (n,)."""
        raise NotImplementedError("stage3: implement HistGradientTree.predict")


@app.cell(hide_code=True)
def _():
    checks.run(3, HistGradientTree=HistGradientTree, bin_data=bin_data, quantile_bin_edges=quantile_bin_edges)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def subsample_rows(n, fraction, rng=None):
    """stage 4: sorted row indices for the next tree: all n if fraction >= 1, else round(fraction * n) (at least 1) drawn WITHOUT replacement."""
    raise NotImplementedError("stage4: implement subsample_rows")


@app.class_definition
class GradientBoostingRegressor(Regressor):
    """gradient boosting for squared loss on binned features (XGBoost-style second-order trees)."""

    def __init__(self, n_estimators=100, *, learning_rate=0.1, max_depth=3, min_child_weight=1.0, reg_lambda=1.0,
                 gamma=0.0, n_bins=256, subsample=1.0, rng=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
        self.subsample = subsample
        self.rng = rng

    def fit(self, X, y):
        """stage 4: bin X, start at the mean, then n_estimators trees on g = F - y, h = 1, each added with learning_rate; returns self."""
        raise NotImplementedError("stage4: implement GradientBoostingRegressor.fit")

    def staged_predict(self, X):
        """stage 4: a generator yielding the prediction (n,) after 1, 2, ..., n_estimators trees."""
        raise NotImplementedError("stage4: implement GradientBoostingRegressor.staged_predict")

    def predict(self, X):
        """stage 4: init_ + learning_rate * the sum of the trees' predictions, (n,)."""
        raise NotImplementedError("stage4: implement GradientBoostingRegressor.predict")


@app.cell(hide_code=True)
def _():
    checks.run(4, GradientBoostingRegressor=GradientBoostingRegressor, subsample_rows=subsample_rows, HistGradientTree=HistGradientTree, bin_data=bin_data, quantile_bin_edges=quantile_bin_edges)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class GradientBoostingClassifier(Classifier):
    """binary gradient boosting with the log loss: the trees add up to the log-odds."""

    def __init__(self, n_estimators=100, *, learning_rate=0.1, max_depth=3, min_child_weight=1.0, reg_lambda=1.0,
                 gamma=0.0, n_bins=256, subsample=1.0, rng=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
        self.subsample = subsample
        self.rng = rng

    def fit(self, X, y):
        """stage 5: init_ = log-odds of the class-1 rate; trees on g = p - y, h = p (1 - p) with p = sigmoid(F); returns self."""
        raise NotImplementedError("stage5: implement GradientBoostingClassifier.fit")

    def staged_decision_function(self, X):
        """stage 5: a generator yielding the log-odds (n,) after 1, 2, ..., n_estimators trees."""
        raise NotImplementedError("stage5: implement GradientBoostingClassifier.staged_decision_function")

    def decision_function(self, X):
        """stage 5: the raw score F(x) = init_ + learning_rate * sum of trees, the log-odds of classes_[1], (n,)."""
        raise NotImplementedError("stage5: implement GradientBoostingClassifier.decision_function")

    def predict_proba(self, X):
        """stage 5: (n, 2) with P(classes_[1]) = sigmoid(decision_function)."""
        raise NotImplementedError("stage5: implement GradientBoostingClassifier.predict_proba")


@app.cell(hide_code=True)
def _():
    checks.run(5, GradientBoostingClassifier=GradientBoostingClassifier, bin_data=bin_data)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `GradientBoostingRegressor` and `GradientBoostingClassifier`
    are part of `mlfs.trees`, and the capstone races them against scikit-learn's. open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
