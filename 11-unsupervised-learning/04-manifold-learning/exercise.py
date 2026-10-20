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
    # manifold learning · build

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
        raise NotImplementedError("stage1: implement MDS.fit")

    def fit_transform(self, X, y=None):
        """stage 1: fit, then return embedding_."""
        raise NotImplementedError("stage1: implement MDS.fit_transform")


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
    raise NotImplementedError("stage2: implement geodesic_distances")


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
        raise NotImplementedError("stage3: implement Isomap.fit")

    def fit_transform(self, X, y=None):
        """stage 3: fit, then return embedding_."""
        raise NotImplementedError("stage3: implement Isomap.fit_transform")


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
    raise NotImplementedError("stage4: implement conditional_probabilities")


@app.function
def joint_probabilities(X, perplexity=30.0):
    """stage 4: the symmetric t-SNE input affinities P = (P_cond + P_cond^T) / (2n), (n, n), summing to 1."""
    raise NotImplementedError("stage4: implement joint_probabilities")


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
    raise NotImplementedError("stage5: implement tsne_kl_gradient")


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
        raise NotImplementedError("stage6: implement TSNE.fit")

    def fit_transform(self, X, y=None):
        """stage 6: fit, then return embedding_."""
        raise NotImplementedError("stage6: implement TSNE.fit_transform")


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
