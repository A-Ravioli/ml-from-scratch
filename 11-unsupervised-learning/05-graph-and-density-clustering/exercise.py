import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.unsupervised import KMeans  # yours, from topic 01
    from mlfs.utils import pairwise_distances  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # graph & density clustering · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit.

    | stage | you build |
    |---|---|
    | 1 | `contingency_matrix`, `adjusted_rand_score`, `normalized_mutual_info_score`: compare two clusterings → `mlfs.unsupervised` |
    | 2 | `affinity_matrix`, `SpectralClustering`: k-means on the eigenvectors of a graph → `mlfs.unsupervised` |
    | 3 | `DBSCAN`: clusters as dense regions, with noise → `mlfs.unsupervised` |
    | 4 | `AgglomerativeClustering`: merge the closest clusters, four ways → `mlfs.unsupervised` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def contingency_matrix(labels_true, labels_pred):
    """stage 1: C[i, j] = number of points with the i-th true label and the j-th predicted label (labels sorted)."""
    raise NotImplementedError("stage1: implement contingency_matrix")


@app.function
def adjusted_rand_score(labels_true, labels_pred):
    """stage 1: the Rand index over pairs, adjusted so a random labeling scores 0 and identical partitions 1."""
    raise NotImplementedError("stage1: implement adjusted_rand_score")


@app.function
def normalized_mutual_info_score(labels_true, labels_pred):
    """stage 1: I(U; V) / ((H(U) + H(V)) / 2) from the contingency table, in nats; 1.0 if both have one cluster."""
    raise NotImplementedError("stage1: implement normalized_mutual_info_score")


@app.cell(hide_code=True)
def _():
    checks.run(1, contingency_matrix=contingency_matrix, adjusted_rand_score=adjusted_rand_score,
               normalized_mutual_info_score=normalized_mutual_info_score)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def affinity_matrix(X, affinity="rbf", gamma=1.0, n_neighbors=10):
    """stage 2: symmetric (n, n) weights with a zero diagonal: exp(-gamma d^2), or (A + A^T) / 2 for the kNN graph A."""
    raise NotImplementedError("stage2: implement affinity_matrix")


@app.class_definition
class SpectralClustering(Estimator):
    """normalized spectral clustering (Ng, Jordan & Weiss): k-means on row-normalized Laplacian eigenvectors."""

    def __init__(self, n_clusters=2, *, affinity="rbf", gamma=1.0, n_neighbors=10, rng=None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.rng = rng

    def fit(self, X, y=None):
        """stage 2: affinity_matrix_, embedding_ (n, k) = unit rows of the bottom-k eigenvectors of L_sym, labels_ by KMeans."""
        raise NotImplementedError("stage2: implement SpectralClustering.fit")

    def fit_predict(self, X, y=None):
        """stage 2: fit, then return labels_."""
        raise NotImplementedError("stage2: implement SpectralClustering.fit_predict")


@app.cell(hide_code=True)
def _():
    checks.run(2, affinity_matrix=affinity_matrix, SpectralClustering=SpectralClustering,
               adjusted_rand_score=adjusted_rand_score)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class DBSCAN(Estimator):
    """density-based clustering: clusters grow from core points; points in no cluster are noise (label -1)."""

    def __init__(self, eps=0.5, *, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        """stage 3: core points have >= min_samples points (themselves included) within eps; grow clusters in index order."""
        raise NotImplementedError("stage3: implement DBSCAN.fit")

    def fit_predict(self, X, y=None):
        """stage 3: fit, then return labels_."""
        raise NotImplementedError("stage3: implement DBSCAN.fit_predict")


@app.cell(hide_code=True)
def _():
    checks.run(3, DBSCAN=DBSCAN)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class AgglomerativeClustering(Estimator):
    """bottom-up hierarchical clustering with single, complete, average or Ward linkage."""

    def __init__(self, n_clusters=2, *, linkage="single"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        """stage 4: the full merge tree by Lance-Williams updates (children_ pairs sorted, distances_), then labels_ for n_clusters."""
        raise NotImplementedError("stage4: implement AgglomerativeClustering.fit")

    def fit_predict(self, X, y=None):
        """stage 4: fit, then return labels_."""
        raise NotImplementedError("stage4: implement AgglomerativeClustering.fit_predict")


@app.cell(hide_code=True)
def _():
    checks.run(4, AgglomerativeClustering=AgglomerativeClustering)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `SpectralClustering`, `DBSCAN`, `AgglomerativeClustering`,
    `adjusted_rand_score` and `normalized_mutual_info_score` are part of
    `mlfs.unsupervised`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
