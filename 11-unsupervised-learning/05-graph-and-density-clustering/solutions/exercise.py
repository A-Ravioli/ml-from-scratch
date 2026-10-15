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
    # graph & density clustering · build (reference solution)

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
    _, a = np.unique(np.asarray(labels_true), return_inverse=True)
    _, b = np.unique(np.asarray(labels_pred), return_inverse=True)
    if len(a) != len(b):
        raise ValueError(f"labels_true and labels_pred differ in length: {len(a)} vs {len(b)}")
    C = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(C, (a, b), 1)
    return C


@app.function
def adjusted_rand_score(labels_true, labels_pred):
    """stage 1: the Rand index over pairs, adjusted so a random labeling scores 0 and identical partitions 1."""
    C = contingency_matrix(labels_true, labels_pred).astype(float)
    pairs = lambda m: float(np.sum(m * (m - 1) / 2))  # noqa: E731
    n = C.sum()
    index, rows, cols = pairs(C), pairs(C.sum(axis=1)), pairs(C.sum(axis=0))
    expected = rows * cols / (n * (n - 1) / 2) if n > 1 else 0.0
    top = 0.5 * (rows + cols) - expected
    if top == 0:
        return 1.0
    return float((index - expected) / top)


@app.function
def normalized_mutual_info_score(labels_true, labels_pred):
    """stage 1: I(U; V) / ((H(U) + H(V)) / 2) from the contingency table, in nats; 1.0 if both have one cluster."""
    C = contingency_matrix(labels_true, labels_pred).astype(float)
    P = C / C.sum()
    pu, pv = P.sum(axis=1), P.sum(axis=0)
    h = lambda p: float(-np.sum(p[p > 0] * np.log(p[p > 0])))  # noqa: E731
    hu, hv = h(pu), h(pv)
    if hu == 0 and hv == 0:
        return 1.0
    nz = P > 0
    mi = float(np.sum(P[nz] * np.log(P[nz] / np.outer(pu, pv)[nz])))
    return float(max(mi, 0.0) / (0.5 * (hu + hv)))


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
    X, _ = check_X_y(X)
    if affinity == "rbf":
        W = np.exp(-gamma * pairwise_distances(X, squared=True))
    elif affinity == "nearest_neighbors":
        n = len(X)
        D = pairwise_distances(X, squared=True)
        np.fill_diagonal(D, np.inf)
        nbrs = np.argsort(D, axis=1)[:, :n_neighbors]
        A = np.zeros((n, n))
        A[np.repeat(np.arange(n), nbrs.shape[1]), nbrs.ravel()] = 1.0
        W = 0.5 * (A + A.T)
    else:
        raise ValueError(f"affinity must be 'rbf' or 'nearest_neighbors', got {affinity!r}")
    np.fill_diagonal(W, 0.0)
    return W


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
        W = affinity_matrix(X, self.affinity, self.gamma, self.n_neighbors)
        d = np.maximum(W.sum(axis=1), 1e-12)
        s = 1.0 / np.sqrt(d)
        M = s[:, None] * W * s[None, :]  # I - L_sym: its TOP eigenvectors are L_sym's bottom ones
        w, V = np.linalg.eigh(M)
        U = V[:, np.argsort(w)[::-1][: self.n_clusters]]
        U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        self.affinity_matrix_ = W
        self.embedding_ = U
        self.labels_ = KMeans(self.n_clusters, n_init=10, rng=self.rng).fit(U).labels_
        return self

    def fit_predict(self, X, y=None):
        """stage 2: fit, then return labels_."""
        return self.fit(X).labels_


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
        X, _ = check_X_y(X)
        n = len(X)
        within = pairwise_distances(X) <= self.eps
        neighbours = [np.flatnonzero(row) for row in within]
        core = within.sum(axis=1) >= self.min_samples
        labels = np.full(n, -1)
        cluster = 0
        for i in range(n):
            if labels[i] != -1 or not core[i]:
                continue
            labels[i] = cluster
            stack = [i]
            while stack:
                p = stack.pop()
                for q in neighbours[p]:
                    if labels[q] == -1:
                        labels[q] = cluster
                        if core[q]:
                            stack.append(q)
            cluster += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core)
        return self

    def fit_predict(self, X, y=None):
        """stage 3: fit, then return labels_."""
        return self.fit(X).labels_


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
        X, _ = check_X_y(X)
        if self.linkage not in ("single", "complete", "average", "ward"):
            raise ValueError(f"linkage must be single, complete, average or ward, got {self.linkage!r}")
        n = len(X)
        if not 1 <= self.n_clusters <= n:
            raise ValueError(f"n_clusters must be in [1, {n}], got {self.n_clusters}")
        D = pairwise_distances(X, squared=self.linkage == "ward")
        np.fill_diagonal(D, np.inf)
        size = np.ones(n)
        node = np.arange(n)  # the tree node id currently stored in each row
        members = {i: [i] for i in range(n)}
        children, heights, snapshots = [], [], {}
        for step in range(n - 1):
            i, j = np.unravel_index(np.argmin(D), D.shape)
            i, j = min(i, j), max(i, j)
            dij = D[i, j]
            children.append((min(node[i], node[j]), max(node[i], node[j])))
            heights.append(np.sqrt(dij) if self.linkage == "ward" else dij)
            ni, nj = size[i], size[j]
            if self.linkage == "single":
                new = np.minimum(D[i], D[j])
            elif self.linkage == "complete":
                new = np.maximum(D[i], D[j])
            elif self.linkage == "average":
                new = (ni * D[i] + nj * D[j]) / (ni + nj)
            else:
                new = ((ni + size) * D[i] + (nj + size) * D[j] - size * dij) / (ni + nj + size)
            D[i, :] = new
            D[:, i] = new
            D[i, i] = np.inf
            D[j, :] = np.inf
            D[:, j] = np.inf
            size[i] = ni + nj
            members[i] = members[i] + members.pop(j)
            node[i] = n + step
            if n - 1 - step == self.n_clusters:
                snapshots = {k: list(v) for k, v in members.items()}
        if self.n_clusters == n:
            snapshots = {i: [i] for i in range(n)}
        labels = np.empty(n, dtype=int)
        for lab, group in enumerate(sorted(snapshots.values(), key=min)):
            labels[group] = lab
        self.children_ = np.array(children, dtype=int).reshape(-1, 2)
        self.distances_ = np.array(heights)
        self.labels_ = labels
        return self

    def fit_predict(self, X, y=None):
        """stage 4: fit, then return labels_."""
        return self.fit(X).labels_


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
