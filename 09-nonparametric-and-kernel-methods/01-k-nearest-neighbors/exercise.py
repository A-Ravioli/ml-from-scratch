import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import heapq

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, Regressor, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.utils import pairwise_distances  # yours, from ch 00 · topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # k-nearest neighbors · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions and classes to write, and checks them live as you edit. this notebook starts
    `mlfs.kernels`: the neighbour models and the KD-tree are the first nonparametric models
    in the course, and they subclass your `Classifier` and `Regressor` from chapter 07.

    | stage | you build |
    |---|---|
    | 1 | `minkowski_distances`, `brute_kneighbors`: distances, and exact neighbour search by brute force |
    | 2 | `neighbor_weights`, `KNeighborsClassifier`, `KNeighborsRegressor`: prediction by local vote or average → `mlfs.kernels` |
    | 3 | `relative_contrast`, `edge_length`: two measurements of the curse of dimensionality |
    | 4 | `KDTree.build`: recursive median splits → `mlfs.kernels` |
    | 5 | `KDTree.search`, `KDTree.query`: exact search that skips most of the data (in low dimension) |

    this is an L3 notebook: docstrings say *what* in one line, and the lesson's ▶ build
    callouts give the conventions the checks expect. you can also run the checks from a
    terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def minkowski_distances(X, Y, p=2.0):
    """stage 1: (n, m) Minkowski-p distances between the rows of X (n, d) and Y (m, d); p = np.inf is the max norm."""
    raise NotImplementedError("stage1: implement minkowski_distances")


@app.function
def brute_kneighbors(X_train, X_query, k):
    """stage 1: the k nearest rows of X_train for each row of X_query (Euclidean) -> (dist (m, k), idx (m, k)), ascending, ties to the smaller index."""
    raise NotImplementedError("stage1: implement brute_kneighbors")


@app.cell(hide_code=True)
def _():
    checks.run(1, minkowski_distances=minkowski_distances, brute_kneighbors=brute_kneighbors)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def neighbor_weights(dist, weights):
    """stage 2: (m, k) vote weights: ones for 'uniform', 1/dist for 'distance' (rows with a zero distance: 1 on the zeros, 0 elsewhere)."""
    raise NotImplementedError("stage2: implement neighbor_weights")


@app.class_definition
class KNeighborsClassifier(Classifier):
    """predicts by a (weighted) vote among the k nearest training rows."""

    def __init__(self, n_neighbors=5, *, weights="uniform", algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm

    def fit(self, X, y):
        """stage 2: store X_, classes_ and y_idx_ (the label of each row as an index into classes_), build tree_ for algorithm='kd_tree'; return self."""
        raise NotImplementedError("stage2: implement KNeighborsClassifier.fit")

    def kneighbors(self, X, k=None):
        """stage 2: (dist (m, k), idx (m, k)) of the k nearest training rows (k defaults to n_neighbors), ascending."""
        raise NotImplementedError("stage2: implement KNeighborsClassifier.kneighbors")

    def predict_proba(self, X):
        """stage 2: (m, n_classes) weighted vote shares of the neighbours' classes, columns in classes_ order."""
        raise NotImplementedError("stage2: implement KNeighborsClassifier.predict_proba")


@app.class_definition
class KNeighborsRegressor(Regressor):
    """predicts the (weighted) mean target of the k nearest training rows."""

    def __init__(self, n_neighbors=5, *, weights="uniform", algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm

    def fit(self, X, y):
        """stage 2: store X_ and y_ (float), build tree_ for algorithm='kd_tree'; return self."""
        raise NotImplementedError("stage2: implement KNeighborsRegressor.fit")

    def kneighbors(self, X, k=None):
        """stage 2: (dist (m, k), idx (m, k)) of the k nearest training rows (k defaults to n_neighbors), ascending."""
        raise NotImplementedError("stage2: implement KNeighborsRegressor.kneighbors")

    def predict(self, X):
        """stage 2: (m,) weighted mean of the neighbours' targets."""
        raise NotImplementedError("stage2: implement KNeighborsRegressor.predict")


@app.cell(hide_code=True)
def _():
    checks.run(
        2,
        neighbor_weights=neighbor_weights,
        KNeighborsClassifier=KNeighborsClassifier,
        KNeighborsRegressor=KNeighborsRegressor,
    )
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def relative_contrast(X, q):
    """stage 3: (max_i ||x_i - q|| - min_i ||x_i - q||) / min_i ||x_i - q|| for the rows x_i of X (n, d) and a query q (d,)."""
    raise NotImplementedError("stage3: implement relative_contrast")


@app.function
def edge_length(fraction, d):
    """stage 3: the edge of the sub-cube that holds `fraction` of uniform data in the unit cube [0, 1]^d."""
    raise NotImplementedError("stage3: implement edge_length")


@app.cell(hide_code=True)
def _():
    checks.run(3, relative_contrast=relative_contrast, edge_length=edge_length)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class KDTree:
    """exact Euclidean nearest-neighbour search over the rows of X (n, d).

    the tree is a list of nodes, self.nodes, with the root at index 0. each node is a dict:
      "idx":   int array, the rows of X in this node's region
      "dim":   the split dimension (None for a leaf)
      "split": the split value (None for a leaf)
      "left", "right": child node ids (None for a leaf). every row in the left child has
               X[row, dim] <= split, every row in the right child has X[row, dim] >= split.
    """

    def __init__(self, X, *, leaf_size=16):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"X must be a non-empty 2-D array (n, d), got shape {X.shape}")
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {leaf_size}")
        self.X = X
        self.leaf_size = leaf_size
        self.nodes = []
        self.n_distances_ = 0
        self.build(np.arange(len(X)))

    def build(self, idx):
        """stage 4: append a node for rows idx (a leaf if len(idx) <= leaf_size, else split at the median of its widest dimension and build both halves); return its id."""
        raise NotImplementedError("stage4: implement KDTree.build")

    def search(self, node_id, q, k, heap):
        """stage 5: add the k nearest rows to q under node_id to heap (a max-heap of (-squared distance, row) with at most k entries), skipping a far child whenever it can't hold anything closer."""
        raise NotImplementedError("stage5: implement KDTree.search")

    def query(self, X, k=1):
        """stage 5: the k nearest rows of self.X for each row of X (m, d) -> (dist (m, k), idx (m, k)) ascending; counts the distances computed in n_distances_."""
        raise NotImplementedError("stage5: implement KDTree.query")


@app.cell(hide_code=True)
def _():
    checks.run(4, KDTree=KDTree)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.cell(hide_code=True)
def _():
    checks.run(5, KDTree=KDTree, KNeighborsClassifier=KNeighborsClassifier, KNeighborsRegressor=KNeighborsRegressor)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `from mlfs.kernels import KNeighborsClassifier, KDTree` gives
    you your own code. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
