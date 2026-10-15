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
    # dimensionality reduction · build (reference solution)

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
        X, _ = check_X_y(X)
        n, d = X.shape
        if n < 2:
            raise ValueError(f"PCA needs at least 2 samples to estimate variances, got {n}")
        self.mean_ = X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        var = s**2 / (n - 1)
        ratio = var / var.sum()
        k = self.n_components
        if k is None:
            k = min(n, d)
        elif isinstance(k, float) and 0.0 < k < 1.0:
            k = int(np.searchsorted(np.cumsum(ratio), k - 1e-12) + 1)
        elif not (isinstance(k, (int, np.integer)) and 1 <= k <= min(n, d)):
            raise ValueError(f"n_components must be None, an int in [1, {min(n, d)}] or a float in (0, 1), got {k!r}")
        self.n_components_ = int(k)
        self.components_ = flip_signs(Vt[:k])
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.singular_values_ = s[:k]
        return self

    def transform(self, X):
        """stage 2: (X - mean_) projected on the components, (n, k); divided by sqrt(explained_variance_) if whiten."""
        X, _ = check_X_y(X)
        Z = (X - self.mean_) @ self.components_.T
        return Z / np.sqrt(self.explained_variance_) if self.whiten else Z

    def inverse_transform(self, X):
        """stage 2: map scores (n, k) back to the input space, (n, d): the reconstruction from k components."""
        Z = np.asarray(X, dtype=float)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_


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
        X, _ = check_X_y(X)
        n, d = X.shape
        k = self.n_components
        if not 1 <= k < d:
            raise ValueError(f"n_components must be in [1, {d - 1}], got {k}")
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        lam = s**2 / n  # the eigenvalues of S (the rest, if n < d, are 0)
        total = float(np.sum(Xc**2) / n)
        self.noise_variance_ = max((total - lam[:k].sum()) / (d - k), 1e-12)
        U = flip_signs(Vt[:k]).T
        self.W_ = U * np.sqrt(np.maximum(lam[:k] - self.noise_variance_, 0.0))
        return self

    def transform(self, X):
        """stage 3: the posterior mean E[z | x] = M^{-1} W^T (x - mean), M = W^T W + sigma^2 I, (n, k)."""
        X, _ = check_X_y(X)
        M = self.W_.T @ self.W_ + self.noise_variance_ * np.eye(self.W_.shape[1])
        return np.linalg.solve(M, self.W_.T @ (X - self.mean_).T).T

    def score_samples(self, X):
        """stage 3: log N(x | mean, W W^T + sigma^2 I) for each row, (n,)."""
        X, _ = check_X_y(X)
        d = X.shape[1]
        C = self.W_ @ self.W_.T + self.noise_variance_ * np.eye(d)
        L = np.linalg.cholesky(C)
        z = np.linalg.solve(L, (X - self.mean_).T)
        return -0.5 * (d * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + np.sum(z**2, axis=0))

    def sample(self, n=1, *, rng=None):
        """stage 3: n draws from the model, (n, d): z ~ N(0, I) then x = W z + mean + sigma * noise."""
        rng = as_rng(rng)
        d, k = self.W_.shape
        z = rng.standard_normal((n, k))
        noise = rng.standard_normal((n, d))
        return z @ self.W_.T + self.mean_ + np.sqrt(self.noise_variance_) * noise


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
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if kernel == "linear":
        return X @ Y.T
    if kernel == "poly":
        return (gamma * (X @ Y.T) + coef0) ** degree
    if kernel == "rbf":
        return np.exp(-gamma * pairwise_distances(X, Y, squared=True))
    raise ValueError(f"kernel must be 'linear', 'poly' or 'rbf', got {kernel!r}")


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
        X, _ = check_X_y(X)
        if not 1 <= self.n_components <= len(X):
            raise ValueError(f"n_components must be in [1, {len(X)}], got {self.n_components}")
        K = self._kernel(X, X)
        self.X_fit_ = X
        self.K_col_means_ = K.mean(axis=0)
        self.K_mean_ = float(K.mean())
        Kc = K - self.K_col_means_[None, :] - K.mean(axis=1)[:, None] + self.K_mean_
        w, V = np.linalg.eigh(Kc)
        top = np.argsort(w)[::-1][: self.n_components]
        self.lambdas_ = w[top]
        self.alphas_ = flip_signs(V[:, top].T).T
        return self

    def transform(self, X):
        """stage 5: centred k(x, X_fit_) @ alphas_ / sqrt(lambdas_), (m, k); on the training data, alphas_ * sqrt(lambdas_)."""
        X, _ = check_X_y(X)
        Kx = self._kernel(X, self.X_fit_)
        Kc = Kx - Kx.mean(axis=1)[:, None] - self.K_col_means_[None, :] + self.K_mean_
        lam = self.lambdas_
        scale = np.where(lam > 1e-12 * max(lam.max(), 1e-300), 1.0 / np.sqrt(np.abs(lam)), 0.0)
        return Kc @ self.alphas_ * scale


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
