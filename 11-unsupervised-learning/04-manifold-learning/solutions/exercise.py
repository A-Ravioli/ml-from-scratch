import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.unsupervised import flip_signs  # yours, from topic 03
    from mlfs.utils import as_rng, pairwise_distances  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # manifold learning · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit.

    | stage | you build |
    |---|---|
    | 1 | `MDS`: coordinates from distances (classical, Torgerson) → `mlfs.unsupervised` |
    | 2 | `geodesic_distances`: shortest paths on a nearest-neighbour graph |
    | 3 | `Isomap`: classical MDS on geodesic distances → `mlfs.unsupervised` |
    | 4 | `conditional_probabilities`, `joint_probabilities`: t-SNE's input affinities, with a perplexity search |
    | 5 | `tsne_kl_gradient`: the Student-t output affinities, the KL divergence and its gradient |
    | 6 | `TSNE`: gradient descent with momentum, gains and early exaggeration → `mlfs.unsupervised` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class MDS(Estimator):
    """classical (Torgerson) multidimensional scaling: an embedding whose inner products match double-centred -D^2/2."""

    def __init__(self, n_components=2, *, dissimilarity="euclidean"):
        self.n_components = n_components
        self.dissimilarity = dissimilarity

    def fit(self, X, y=None):
        """stage 1: embedding_ (n, k) = top eigenvectors of B = -J D^2 J / 2 times sqrt(eigenvalue), columns sign-flipped."""
        if self.dissimilarity == "euclidean":
            X, _ = check_X_y(X)
            D2 = pairwise_distances(X, squared=True)
        elif self.dissimilarity == "precomputed":
            D = np.asarray(X, dtype=float)
            if D.ndim != 2 or D.shape[0] != D.shape[1] or not np.allclose(D, D.T):
                raise ValueError("a precomputed dissimilarity must be a symmetric (n, n) matrix")
            D2 = D**2
        else:
            raise ValueError(f"dissimilarity must be 'euclidean' or 'precomputed', got {self.dissimilarity!r}")
        B = -0.5 * (D2 - D2.mean(axis=0)[None, :] - D2.mean(axis=1)[:, None] + D2.mean())
        w, V = np.linalg.eigh(B)
        top = np.argsort(w)[::-1][: self.n_components]
        self.eigenvalues_ = w[top]
        self.embedding_ = flip_signs(V[:, top].T).T * np.sqrt(np.maximum(w[top], 0.0))
        return self

    def fit_transform(self, X, y=None):
        """stage 1: fit, then return embedding_."""
        return self.fit(X).embedding_


@app.cell(hide_code=True)
def _():
    checks.run(1, MDS=MDS)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def geodesic_distances(X, n_neighbors=10):
    """stage 2: shortest-path lengths (n, n) on the symmetrized k-nearest-neighbour graph (inf if unreachable)."""
    X, _ = check_X_y(X)
    n = len(X)
    D = pairwise_distances(X)
    nbrs = np.argsort(D, axis=1)[:, 1 : n_neighbors + 1]
    G = np.full((n, n), np.inf)
    rows = np.repeat(np.arange(n), nbrs.shape[1])
    G[rows, nbrs.ravel()] = D[rows, nbrs.ravel()]
    G = np.minimum(G, G.T)
    np.fill_diagonal(G, 0.0)
    for k in range(n):  # Floyd-Warshall, one intermediate vertex at a time, vectorized over all pairs
        G = np.minimum(G, G[:, k, None] + G[None, k, :])
    return G


@app.cell(hide_code=True)
def _():
    checks.run(2, geodesic_distances=geodesic_distances)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class Isomap(Estimator):
    """geodesic distances on a nearest-neighbour graph, then classical MDS."""

    def __init__(self, n_components=2, *, n_neighbors=10):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        """stage 3: dist_matrix_ (geodesics; ValueError if the graph is disconnected), then embedding_ by MDS."""
        self.dist_matrix_ = geodesic_distances(X, self.n_neighbors)
        if not np.all(np.isfinite(self.dist_matrix_)):
            raise ValueError("the neighbourhood graph is disconnected: increase n_neighbors")
        self.embedding_ = MDS(self.n_components, dissimilarity="precomputed").fit(self.dist_matrix_).embedding_
        return self

    def fit_transform(self, X, y=None):
        """stage 3: fit, then return embedding_."""
        return self.fit(X).embedding_


@app.cell(hide_code=True)
def _():
    checks.run(3, Isomap=Isomap, geodesic_distances=geodesic_distances)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def conditional_probabilities(D2, perplexity=30.0, *, tol=1e-5, max_iters=100):
    """stage 4: p_{j|i} ∝ exp(-beta_i D2[i, j]) (zero diagonal), with each beta_i bisected so the row's perplexity matches."""
    D2 = np.asarray(D2, dtype=float)
    n = len(D2)
    if not 1.0 < perplexity < n - 1:
        raise ValueError(f"perplexity must be in (1, n - 1) = (1, {n - 1}), got {perplexity}")
    target = np.log(perplexity)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    beta = np.ones(n)
    off = ~np.eye(n, dtype=bool)
    for _ in range(max_iters):
        logits = np.where(off, -beta[:, None] * D2, -np.inf)
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        H = -np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0), axis=1)
        if np.all(np.abs(H - target) < tol):
            break
        too_flat = H > target  # entropy too high: sharpen, i.e. increase beta
        lo = np.where(too_flat, beta, lo)
        hi = np.where(too_flat, hi, beta)
        beta = np.where(np.isinf(hi), beta * 2, np.where(np.isinf(lo), beta / 2, (lo + hi) / 2))
    return P


@app.function
def joint_probabilities(X, perplexity=30.0):
    """stage 4: the symmetric t-SNE input affinities P = (P_cond + P_cond^T) / (2n), (n, n), summing to 1."""
    X, _ = check_X_y(X)
    Pc = conditional_probabilities(pairwise_distances(X, squared=True), perplexity)
    return (Pc + Pc.T) / (2 * len(X))


@app.cell(hide_code=True)
def _():
    checks.run(4, conditional_probabilities=conditional_probabilities, joint_probabilities=joint_probabilities)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def tsne_kl_gradient(Y, P, *, kl=True):
    """stage 5: (KL(P || Q), or None if not kl; gradient (n, k)) with q_ij ∝ 1 / (1 + ||y_i - y_j||^2), zero diagonal."""
    Y = np.asarray(Y, dtype=float)
    num = 1.0 / (1.0 + pairwise_distances(Y, Y, squared=True))  # Y given twice: skips the symmetrizing pass
    np.fill_diagonal(num, 0.0)
    Q = np.maximum(num / num.sum(), 1e-300)
    W = (P - Q) * num
    grad = 4.0 * (W.sum(axis=1)[:, None] * Y - W @ Y)
    if not kl:
        return None, grad
    return float(np.sum(P * np.log(np.maximum(P, 1e-300) / Q))), grad


@app.cell(hide_code=True)
def _():
    checks.run(5, tsne_kl_gradient=tsne_kl_gradient, joint_probabilities=joint_probabilities)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.class_definition
class TSNE(Estimator):
    """exact t-SNE (van der Maaten & Hinton, 2008)."""

    def __init__(self, n_components=2, *, perplexity=30.0, lr=200.0, n_iters=1000, early_exaggeration=12.0, rng=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iters = n_iters
        self.early_exaggeration = early_exaggeration
        self.rng = rng

    def fit(self, X, y=None):
        """stage 6: Y0 ~ 1e-4 N(0, I); n_iters steps of momentum GD with gains; P exaggerated and momentum 0.5 for 250 iters."""
        X, _ = check_X_y(X)
        rng = as_rng(self.rng)
        P = joint_probabilities(X, self.perplexity)
        Y = 1e-4 * rng.standard_normal((len(X), self.n_components))
        update = np.zeros_like(Y)
        gains = np.ones_like(Y)
        for it in range(self.n_iters):
            early = it < 250
            _, grad = tsne_kl_gradient(Y, P * self.early_exaggeration if early else P, kl=False)
            momentum = 0.5 if early else 0.8
            gains = np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8)
            gains = np.maximum(gains, 0.01)
            update = momentum * update - self.lr * gains * grad
            Y = Y + update
            Y = Y - Y.mean(axis=0)
        self.embedding_ = Y
        self.kl_divergence_ = tsne_kl_gradient(Y, P)[0]
        return self

    def fit_transform(self, X, y=None):
        """stage 6: fit, then return embedding_."""
        return self.fit(X).embedding_


@app.cell(hide_code=True)
def _():
    checks.run(6, TSNE=TSNE, joint_probabilities=joint_probabilities, tsne_kl_gradient=tsne_kl_gradient)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `MDS`, `Isomap` and `TSNE` are part of `mlfs.unsupervised`.
    open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
