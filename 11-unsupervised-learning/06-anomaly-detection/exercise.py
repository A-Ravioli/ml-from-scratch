import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.utils import as_rng, pairwise_distances  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # anomaly detection · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. every score in this notebook follows one
    convention: **higher means more anomalous**.

    | stage | you build |
    |---|---|
    | 1 | `gaussian_scores`, `knn_distance_scores`: two density-style anomaly scores |
    | 2 | `average_path_length`: $c(n)$, the yardstick for isolation depth |
    | 3 | `isolation_tree`, `path_lengths`: grow one random isolation tree, measure how deep points land |
    | 4 | `IsolationForest`: average many trees into a score in $(0, 1]$ → `mlfs.unsupervised` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def gaussian_scores(X_train, X, *, reg=1e-6):
    """stage 1: squared Mahalanobis distance of each row of X under a Gaussian fitted to X_train (MLE cov + reg I), (n,)."""
    raise NotImplementedError("stage1: implement gaussian_scores")


@app.function
def knn_distance_scores(X, k=5):
    """stage 1: each point's distance to its k-th nearest OTHER point in X, (n,)."""
    raise NotImplementedError("stage1: implement knn_distance_scores")


@app.cell(hide_code=True)
def _():
    checks.run(1, gaussian_scores=gaussian_scores, knn_distance_scores=knn_distance_scores)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def average_path_length(n):
    """stage 2: c(n) = 2 (ln(n - 1) + Euler's gamma) - 2 (n - 1) / n, with c(2) = 1 and c(n) = 0 for n <= 1; vectorized."""
    raise NotImplementedError("stage2: implement average_path_length")


@app.cell(hide_code=True)
def _():
    checks.run(2, average_path_length=average_path_length)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def isolation_tree(X, max_depth, rng=None):
    """stage 3: a random isolation tree as a dict of node arrays: feature, threshold, left, right, depth, size."""
    raise NotImplementedError("stage3: implement isolation_tree")


@app.function
def path_lengths(tree, X):
    """stage 3: for each row of X, the depth of the leaf it reaches plus c(number of training points in that leaf), (n,)."""
    raise NotImplementedError("stage3: implement path_lengths")


@app.cell(hide_code=True)
def _():
    checks.run(3, isolation_tree=isolation_tree, path_lengths=path_lengths, average_path_length=average_path_length)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class IsolationForest(Estimator):
    """an ensemble of isolation trees; anomalies are the points that random splits isolate quickly."""

    def __init__(self, n_estimators=100, *, max_samples=256, contamination=0.1, rng=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.rng = rng

    def fit(self, X, y=None):
        """stage 4: each tree on min(max_samples, n) rows drawn without replacement, max_depth ceil(log2 of that); threshold_."""
        raise NotImplementedError("stage4: implement IsolationForest.fit")

    def score_samples(self, X):
        """stage 4: s(x) = 2^(-E[h(x)] / c(max_samples_)) in (0, 1]; higher = more anomalous."""
        raise NotImplementedError("stage4: implement IsolationForest.score_samples")

    def predict(self, X):
        """stage 4: 1 where score_samples(X) > threshold_ (an anomaly), else 0."""
        raise NotImplementedError("stage4: implement IsolationForest.predict")


@app.cell(hide_code=True)
def _():
    checks.run(4, IsolationForest=IsolationForest, average_path_length=average_path_length)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `IsolationForest` is part of `mlfs.unsupervised`. open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
