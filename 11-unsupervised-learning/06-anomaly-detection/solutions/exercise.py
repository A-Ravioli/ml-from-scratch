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
    # anomaly detection · build (reference solution)

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
    X_train, _ = check_X_y(X_train)
    X, _ = check_X_y(X)
    mu = X_train.mean(axis=0)
    C = np.cov(X_train.T, bias=True).reshape(X.shape[1], X.shape[1]) + reg * np.eye(X.shape[1])
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, (X - mu).T)
    return np.sum(z**2, axis=0)


@app.function
def knn_distance_scores(X, k=5):
    """stage 1: each point's distance to its k-th nearest OTHER point in X, (n,)."""
    X, _ = check_X_y(X)
    if not 1 <= k < len(X):
        raise ValueError(f"k must be in [1, {len(X) - 1}], got {k}")
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    return np.partition(D, k - 1, axis=1)[:, k - 1]


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
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[n == 2] = 1.0
    out[big] = 2.0 * (np.log(n[big] - 1.0) + np.euler_gamma) - 2.0 * (n[big] - 1.0) / n[big]
    return out if out.ndim else float(out)


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
    X, _ = check_X_y(X)
    rng = as_rng(rng)
    feature, threshold, left, right, depth, size = [], [], [], [], [], []

    def new_node(d, n):
        feature.append(-1)
        threshold.append(np.nan)
        left.append(-1)
        right.append(-1)
        depth.append(d)
        size.append(n)
        return len(feature) - 1

    stack = [(new_node(0, len(X)), np.arange(len(X)))]
    while stack:
        node, idx = stack.pop()
        if depth[node] >= max_depth or len(idx) <= 1:
            continue
        lo, hi = X[idx].min(axis=0), X[idx].max(axis=0)
        candidates = np.flatnonzero(hi > lo)
        if len(candidates) == 0:
            continue
        f = int(candidates[rng.integers(len(candidates))])
        t = float(rng.uniform(lo[f], hi[f]))
        goes_left = X[idx, f] < t
        feature[node], threshold[node] = f, t
        left[node] = new_node(depth[node] + 1, int(goes_left.sum()))
        right[node] = new_node(depth[node] + 1, int((~goes_left).sum()))
        stack.append((right[node], idx[~goes_left]))
        stack.append((left[node], idx[goes_left]))
    return {"feature": np.array(feature), "threshold": np.array(threshold), "left": np.array(left),
            "right": np.array(right), "depth": np.array(depth), "size": np.array(size)}


@app.function
def path_lengths(tree, X):
    """stage 3: for each row of X, the depth of the leaf it reaches plus c(number of training points in that leaf), (n,)."""
    X, _ = check_X_y(X)
    node = np.zeros(len(X), dtype=int)
    rows = np.arange(len(X))
    for _ in range(int(tree["depth"].max()) + 1):
        f = tree["feature"][node]
        inner = f >= 0
        if not inner.any():
            break
        go_left = X[rows[inner], f[inner]] < tree["threshold"][node[inner]]
        node[inner] = np.where(go_left, tree["left"][node[inner]], tree["right"][node[inner]])
    return tree["depth"][node] + average_path_length(tree["size"][node])


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
        X, _ = check_X_y(X)
        if len(X) < 2:
            raise ValueError("IsolationForest needs at least 2 samples")
        if self.max_samples < 2:
            raise ValueError(f"max_samples must be at least 2 (c(1) = 0 can't normalize a score), got {self.max_samples}")
        if not 0.0 < self.contamination < 0.5:
            raise ValueError(f"contamination must be in (0, 0.5), got {self.contamination}")
        rng = as_rng(self.rng)
        self.max_samples_ = min(self.max_samples, len(X))
        max_depth = int(np.ceil(np.log2(self.max_samples_)))
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(len(X), self.max_samples_, replace=False)
            self.estimators_.append(isolation_tree(X[idx], max_depth, rng))
        self.threshold_ = float(np.quantile(self.score_samples(X), 1.0 - self.contamination))
        return self

    def score_samples(self, X):
        """stage 4: s(x) = 2^(-E[h(x)] / c(max_samples_)) in (0, 1]; higher = more anomalous."""
        X, _ = check_X_y(X)
        mean_h = np.mean([path_lengths(t, X) for t in self.estimators_], axis=0)
        return 2.0 ** (-mean_h / average_path_length(self.max_samples_))

    def predict(self, X):
        """stage 4: 1 where score_samples(X) > threshold_ (an anomaly), else 0."""
        return (self.score_samples(X) > self.threshold_).astype(int)


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
