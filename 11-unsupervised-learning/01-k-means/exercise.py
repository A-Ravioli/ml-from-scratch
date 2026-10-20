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
    # k-means · build

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
    raise NotImplementedError("stage1: implement assign_to_nearest")


@app.function
def update_centers(X, labels, centers):
    """stage 1: the mean of each cluster's points (k, d); an empty cluster keeps its old center."""
    raise NotImplementedError("stage1: implement update_centers")


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
    raise NotImplementedError("stage2: implement kmeans_plus_plus")


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
        raise NotImplementedError("stage3: implement KMeans.fit")

    def predict(self, X):
        """stage 3: the index of the nearest fitted center, int (n,)."""
        raise NotImplementedError("stage3: implement KMeans.predict")

    def fit_predict(self, X, y=None):
        """stage 3: fit, then return labels_."""
        raise NotImplementedError("stage3: implement KMeans.fit_predict")

    def transform(self, X):
        """stage 3: Euclidean distances to every fitted center, (n, k)."""
        raise NotImplementedError("stage3: implement KMeans.transform")

    def score(self, X, y=None):
        """stage 3: minus the inertia of X under the fitted centers (higher is better)."""
        raise NotImplementedError("stage3: implement KMeans.score")


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
    raise NotImplementedError("stage4: implement silhouette_score")


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
