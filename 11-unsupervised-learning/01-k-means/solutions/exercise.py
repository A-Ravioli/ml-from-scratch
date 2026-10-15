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
    # k-means · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. together they become the first piece of
    `mlfs.unsupervised`.

    | stage | you build |
    |---|---|
    | 1 | `assign_to_nearest`, `update_centers`: the two halves of one Lloyd iteration |
    | 2 | `kmeans_plus_plus`: seeding by $D^2$ sampling → `mlfs.unsupervised` |
    | 3 | `KMeans`: Lloyd's algorithm with restarts, as an `Estimator` → `mlfs.unsupervised` |
    | 4 | `silhouette_score`: a label-free score for a clustering → `mlfs.unsupervised` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def assign_to_nearest(X, centers):
    """stage 1: (labels int (n,), squared distance to the nearest center (n,)); ties go to the lower index."""
    d2 = pairwise_distances(X, centers, squared=True)
    labels = np.argmin(d2, axis=1)
    return labels, d2[np.arange(len(X)), labels]


@app.function
def update_centers(X, labels, centers):
    """stage 1: the mean of each cluster's points (k, d); an empty cluster keeps its old center."""
    k = len(centers)
    counts = np.bincount(labels, minlength=k)
    X = np.asarray(X, dtype=float)
    new = np.array(centers, dtype=float, copy=True)
    filled = counts > 0
    # sort the points by cluster, then sum each contiguous block: O(n d) memory, no (k, n) one-hot matrix
    order = np.argsort(labels, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sums = np.add.reduceat(X[order], starts[filled], axis=0)
    new[filled] = sums / counts[filled, None]
    return new


@app.cell(hide_code=True)
def _():
    checks.run(1, assign_to_nearest=assign_to_nearest, update_centers=update_centers)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def kmeans_plus_plus(X, n_clusters, *, rng=None):
    """stage 2: k initial centers (k, d), rows of X: the first uniform, each next one with probability ∝ D(x)^2."""
    rng = as_rng(rng)
    X = np.asarray(X, dtype=float)
    n = len(X)
    if not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}")
    idx = [int(rng.integers(n))]
    d2 = pairwise_distances(X, X[idx], squared=True)[:, 0]
    for _ in range(1, n_clusters):
        total = d2.sum()
        p = d2 / total if total > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, pairwise_distances(X, X[[j]], squared=True)[:, 0])
    return X[idx].copy()


@app.cell(hide_code=True)
def _():
    checks.run(2, kmeans_plus_plus=kmeans_plus_plus)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class KMeans(Estimator):
    """k-means clustering by Lloyd's algorithm, best of n_init seeded runs."""

    def __init__(self, n_clusters=8, *, init="k-means++", n_init=10, max_iters=300, tol=0.0001, rng=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iters = max_iters
        self.tol = tol
        self.rng = rng

    def _initial_centers(self, X, rng):
        """the starting centers for one run: k-means++, k distinct random rows, or a given (k, d) array."""
        if isinstance(self.init, str) and self.init == "k-means++":
            return kmeans_plus_plus(X, self.n_clusters, rng=rng)
        if isinstance(self.init, str) and self.init == "random":
            return X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
        if isinstance(self.init, str):
            raise ValueError(f"init must be 'k-means++', 'random' or a (k, d) array, got {self.init!r}")
        centers = np.array(self.init, dtype=float)
        if centers.shape != (self.n_clusters, X.shape[1]):
            raise ValueError(f"init array must have shape {(self.n_clusters, X.shape[1])}, got {centers.shape}")
        return centers

    def fit(self, X, y=None):
        """stage 3: run Lloyd from n_init starts (one for an array init) and keep the lowest-inertia run."""
        X, _ = check_X_y(X)
        if not 1 <= self.n_clusters <= len(X):
            raise ValueError(f"n_clusters must be in [1, {len(X)}], got {self.n_clusters}")
        rng = as_rng(self.rng)
        tol = self.tol * float(np.mean(np.var(X, axis=0)))
        runs = 1 if not isinstance(self.init, str) else self.n_init
        best = None
        for _ in range(runs):
            centers = self._initial_centers(X, rng)
            n_iter = 0
            for n_iter in range(1, self.max_iters + 1):
                labels, _d2 = assign_to_nearest(X, centers)
                new = update_centers(X, labels, centers)
                shift = float(np.sum((new - centers) ** 2))
                centers = new
                if shift <= tol:
                    break
            labels, d2 = assign_to_nearest(X, centers)
            inertia = float(d2.sum())
            if best is None or inertia < best[0]:
                best = (inertia, centers, labels, n_iter)
        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        return self

    def predict(self, X):
        """stage 3: the index of the nearest fitted center, int (n,)."""
        X, _ = check_X_y(X)
        return assign_to_nearest(X, self.cluster_centers_)[0]

    def fit_predict(self, X, y=None):
        """stage 3: fit, then return labels_."""
        return self.fit(X).labels_

    def transform(self, X):
        """stage 3: Euclidean distances to every fitted center, (n, k)."""
        X, _ = check_X_y(X)
        return pairwise_distances(X, self.cluster_centers_)

    def score(self, X, y=None):
        """stage 3: minus the inertia of X under the fitted centers (higher is better)."""
        X, _ = check_X_y(X)
        return -float(assign_to_nearest(X, self.cluster_centers_)[1].sum())


@app.cell(hide_code=True)
def _():
    checks.run(3, KMeans=KMeans, kmeans_plus_plus=kmeans_plus_plus)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def silhouette_score(X, labels):
    """stage 4: the mean over points of (b - a) / max(a, b) (a singleton scores 0); needs 2 <= k <= n - 1."""
    X, _ = check_X_y(X)
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    n, k = len(X), int(inv.max()) + 1
    if not 2 <= k <= n - 1:
        raise ValueError(f"silhouette needs 2 <= number of clusters <= n - 1, got {k} clusters for {n} points")
    D = pairwise_distances(X)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), inv] = 1.0
    sums = D @ onehot  # (n, k): total distance from each point to each cluster
    counts = onehot.sum(axis=0)
    own = counts[inv]
    a = np.where(own > 1, sums[np.arange(n), inv] / np.maximum(own - 1, 1), 0.0)
    means = sums / counts
    means[np.arange(n), inv] = np.inf
    b = means.min(axis=1)
    s = np.where(own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-300), 0.0)
    return float(s.mean())


@app.cell(hide_code=True)
def _():
    checks.run(4, silhouette_score=silhouette_score)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `KMeans`, `kmeans_plus_plus` and `silhouette_score` are part of
    `mlfs.unsupervised`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
