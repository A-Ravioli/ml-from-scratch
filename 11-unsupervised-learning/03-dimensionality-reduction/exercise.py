import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Transformer, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.utils import as_rng, pairwise_distances  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # dimensionality reduction · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. every class is a `Transformer` (ch 07);
    `fit_transform` comes for free.

    | stage | you build |
    |---|---|
    | 1 | `PCA.fit`: principal directions and their variances from an SVD → `mlfs.unsupervised` |
    | 2 | `PCA.transform`, `PCA.inverse_transform`: project, whiten, reconstruct |
    | 3 | `ProbabilisticPCA`: PCA as a Gaussian latent-variable model, in closed form → `mlfs.unsupervised` |
    | 4 | `kernel_matrix`, `KernelPCA.fit`: PCA in a feature space you never compute → `mlfs.unsupervised` |
    | 5 | `KernelPCA.transform`: embed new points, centring them like the training data |

    PCA uses `np.linalg.svd`, NumPy's fast version of the SVD you built in chapter 02
    (AUTHORING §9: once you've built it, you may use the library for heavy lifting). you can
    also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def flip_signs(V):
    """flip each ROW of V so that its largest-magnitude entry is positive (the course's sign convention)."""
    V = np.array(V, dtype=float, copy=True)
    idx = np.argmax(np.abs(V), axis=1)
    signs = np.sign(V[np.arange(len(V)), idx])
    signs[signs == 0] = 1.0
    return V * signs[:, None]


@app.class_definition
class PCA(Transformer):
    """principal component analysis by an SVD of the centred data."""

    def __init__(self, n_components=None, *, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        """stage 1: mean_, components_ (k, d) (sign-flipped rows), explained_variance_ (ddof=1), explained_variance_ratio_."""
        raise NotImplementedError("stage1: implement PCA.fit")

    def transform(self, X):
        """stage 2: (X - mean_) projected on the components, (n, k); divided by sqrt(explained_variance_) if whiten."""
        raise NotImplementedError("stage2: implement PCA.transform")

    def inverse_transform(self, X):
        """stage 2: map scores (n, k) back to the input space, (n, d): the reconstruction from k components."""
        raise NotImplementedError("stage2: implement PCA.inverse_transform")


@app.cell(hide_code=True)
def _():
    checks.run(1, PCA=PCA, flip_signs=flip_signs)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.cell(hide_code=True)
def _():
    checks.run(2, PCA=PCA)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class ProbabilisticPCA(Transformer):
    """x = W z + mean + noise, z ~ N(0, I_k), noise ~ N(0, sigma^2 I_d), fitted by maximum likelihood."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        """stage 3: closed-form MLE: mean_, noise_variance_ (mean discarded eigenvalue of S = Xc^T Xc / n), W_ (d, k)."""
        raise NotImplementedError("stage3: implement ProbabilisticPCA.fit")

    def transform(self, X):
        """stage 3: the posterior mean E[z | x] = M^{-1} W^T (x - mean), M = W^T W + sigma^2 I, (n, k)."""
        raise NotImplementedError("stage3: implement ProbabilisticPCA.transform")

    def score_samples(self, X):
        """stage 3: log N(x | mean, W W^T + sigma^2 I) for each row, (n,)."""
        raise NotImplementedError("stage3: implement ProbabilisticPCA.score_samples")

    def sample(self, n=1, *, rng=None):
        """stage 3: n draws from the model, (n, d): z ~ N(0, I) then x = W z + mean + sigma * noise."""
        raise NotImplementedError("stage3: implement ProbabilisticPCA.sample")


@app.cell(hide_code=True)
def _():
    checks.run(3, ProbabilisticPCA=ProbabilisticPCA, PCA=PCA)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def kernel_matrix(X, Y, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
    """stage 4: K[i, j] = k(X[i], Y[j]) for kernel in {'linear', 'poly', 'rbf'}, (n, m)."""
    raise NotImplementedError("stage4: implement kernel_matrix")


@app.class_definition
class KernelPCA(Transformer):
    """PCA on the feature vectors phi(x) of a kernel, computed through the kernel matrix."""

    def __init__(self, n_components=2, *, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _kernel(self, X, Y):
        return kernel_matrix(X, Y, self.kernel, self.gamma, self.degree, self.coef0)

    def fit(self, X, y=None):
        """stage 4: centre the Gram matrix; lambdas_ (k,) its top eigenvalues, alphas_ (n, k) unit eigenvectors (sign-flipped)."""
        raise NotImplementedError("stage4: implement KernelPCA.fit")

    def transform(self, X):
        """stage 5: centred k(x, X_fit_) @ alphas_ / sqrt(lambdas_), (m, k); on the training data, alphas_ * sqrt(lambdas_)."""
        raise NotImplementedError("stage5: implement KernelPCA.transform")


@app.cell(hide_code=True)
def _():
    checks.run(4, kernel_matrix=kernel_matrix, KernelPCA=KernelPCA, PCA=PCA)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.cell(hide_code=True)
def _():
    checks.run(5, KernelPCA=KernelPCA, PCA=PCA)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `PCA`, `ProbabilisticPCA` and `KernelPCA` are part of
    `mlfs.unsupervised`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
