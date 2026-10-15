"""Checks for exercise.py (dimensionality reduction).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scikit-learn's
PCA and KernelPCA and scipy's Gaussian density are oracles only. Signs of components are
fixed by the course convention (largest-magnitude entry positive), so where an oracle uses
another convention, columns are compared up to sign.
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA as SkPCA
from sklearn.decomposition import KernelPCA as SkKernelPCA
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

# a = (3, -3, 3, -3) along u = (0.6, 0.8) plus b = (1, 1, -1, -1) along v = (0.8, -0.6)
SQUARE = np.array([[2.6, 1.8], [-1.0, -3.0], [1.0, 3.0], [-2.6, -1.8]])


def _correlated(n=200, d=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, (d, d)) * np.linspace(3, 0.3, d)
    return rng.normal(0, 1, (n, d)) @ A.T + rng.normal(0, 5, d)


def _latent(n=300, d=12, k=3, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 2, (d, k))
    return rng.standard_normal((n, k)) @ W.T + 1.0 + noise * rng.standard_normal((n, d))


def _same_up_to_sign(A, B, atol=1e-6):
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    return all(np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol) for a, b in zip(A.T, B.T))


def _rings(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.where(np.arange(n) < n // 2, 1.0, 0.3)
    return np.c_[r * np.cos(t), r * np.sin(t)] + 0.05 * rng.standard_normal((n, 2)), (np.arange(n) < n // 2).astype(int)


# ---------------------------------------------------------------- stage 1: PCA.fit


def test_stage1_worked_example(nb):
    p = nb.PCA().fit(SQUARE)
    assert np.allclose(p.mean_, [0, 0]), "mean_ is the column mean"
    assert np.allclose(p.explained_variance_, [12, 4 / 3]), (
        f"the scores along (0.6, 0.8) and (0.8, -0.6) are (3, -3, 3, -3) and (1, 1, -1, -1), so the variances (ddof=1) are "
        f"36/3 = 12 and 4/3; got {p.explained_variance_}. explained_variance_ = s**2 / (n - 1)")
    assert np.allclose(p.explained_variance_ratio_, [0.9, 0.1]), f"ratios 0.9 and 0.1, got {p.explained_variance_ratio_}"
    assert np.allclose(p.components_, [[0.6, 0.8], [0.8, -0.6]]), (
        f"components_ rows are (0.6, 0.8) and (0.8, -0.6), each flipped so its largest-|.| entry is positive; "
        f"got {np.round(p.components_, 4).tolist()}")


def test_stage1_matches_sklearn(nb):
    X = _correlated(seed=1)
    p = nb.PCA(4).fit(X)
    sk = SkPCA(4).fit(X)
    assert p.components_.shape == (4, 6), f"components_ is (k, d) = (4, 6), got {p.components_.shape}"
    assert np.allclose(p.explained_variance_, sk.explained_variance_), "explained_variance_ should match sklearn (ddof=1)"
    assert np.allclose(p.explained_variance_ratio_, sk.explained_variance_ratio_), (
        "explained_variance_ratio_ is each variance over the TOTAL variance (all components, not just the kept ones)")
    assert _same_up_to_sign(p.components_.T, sk.components_.T), "components match sklearn's up to sign"
    big = np.abs(p.components_).argmax(1)
    assert np.all(p.components_[np.arange(4), big] > 0), "each component's largest-magnitude entry must be positive"
    assert np.allclose(p.components_ @ p.components_.T, np.eye(4)), "components are orthonormal"


def test_stage1_n_components_options(nb):
    X = _correlated(seed=2)
    assert nb.PCA().fit(X).components_.shape == (6, 6), "n_components=None keeps min(n, d)"
    assert np.isclose(nb.PCA().fit(X).explained_variance_ratio_.sum(), 1.0), "keeping everything explains 100%"
    p = nb.PCA(0.9).fit(X)
    cum = np.cumsum(nb.PCA().fit(X).explained_variance_ratio_)
    want = int(np.searchsorted(cum, 0.9) + 1)
    assert p.n_components_ == want and len(p.components_) == want, (
        f"n_components=0.9 keeps the fewest components explaining >= 90% of the variance: {want}, got {len(p.components_)}")
    wide = np.random.default_rng(3).normal(0, 1, (15, 300))
    assert nb.PCA().fit(wide).components_.shape == (15, 300), "with n < d (wide data) at most n components exist"
    try:
        nb.PCA().fit(X[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("one sample has no variance (n - 1 = 0): PCA.fit should raise ValueError, not return NaN")
    for bad in (0, 7, 1.5):
        try:
            nb.PCA(bad).fit(X)
        except ValueError:
            continue
        raise AssertionError(f"PCA(n_components={bad}) on 6 features should raise ValueError")


# ---------------------------------------------------------------- stage 2: transform / inverse


def test_stage2_worked_example(nb):
    p = nb.PCA(1).fit(SQUARE)
    Z = p.transform(SQUARE)
    assert np.allclose(Z.ravel(), [3, -3, 3, -3]), f"scores on PC1 are (3, -3, 3, -3), got {Z.ravel()}"
    R = p.inverse_transform(Z)
    assert np.allclose(R, [[1.8, 2.4], [-1.8, -2.4], [1.8, 2.4], [-1.8, -2.4]]), (
        f"reconstructions from 1 component are score * (0.6, 0.8) + mean, got {R.tolist()}")
    assert np.isclose(((SQUARE - R) ** 2).sum(), 4.0), "the squared reconstruction error is s_2^2 = 3 * 4/3 = 4"


def test_stage2_matches_sklearn_and_reconstructs(nb):
    X = _correlated(seed=4)
    for k in (2, 4):
        p = nb.PCA(k).fit(X)
        Z = p.transform(X)
        assert _same_up_to_sign(Z, SkPCA(k).fit(X).transform(X)), "transform = (X - mean_) @ components_.T"
        err = ((X - p.inverse_transform(Z)) ** 2).sum()
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        assert np.isclose(err, (s[k:] ** 2).sum()), (
            f"k={k}: the squared reconstruction error should equal the sum of the discarded s_j^2 = {(s[k:] ** 2).sum():.4f}, "
            f"got {err:.4f}")
    full = nb.PCA().fit(X)
    assert np.allclose(full.inverse_transform(full.transform(X)), X), "with every component, inverse_transform undoes transform"


def test_stage2_whiten(nb):
    X = _correlated(seed=5)
    p = nb.PCA(3, whiten=True).fit(X)
    Z = p.transform(X)
    assert np.allclose(np.cov(Z.T), np.eye(3)), "whitened scores have identity covariance (ddof=1)"
    assert _same_up_to_sign(Z, SkPCA(3, whiten=True).fit(X).transform(X)), "whitened scores match sklearn up to sign"
    full = nb.PCA(whiten=True).fit(X)
    assert np.allclose(full.inverse_transform(full.transform(X)), X), "inverse_transform undoes whitening too"


# ---------------------------------------------------------------- stage 3: probabilistic PCA


def test_stage3_closed_form(nb):
    X = _latent(seed=6)
    m = nb.ProbabilisticPCA(3).fit(X)
    n, d = X.shape
    lam = np.sort(np.linalg.eigvalsh(np.cov(X.T, bias=True)))[::-1]
    assert np.allclose(m.mean_, X.mean(0)), "mean_ is the sample mean"
    assert np.isclose(m.noise_variance_, lam[3:].mean()), (
        f"noise_variance_ = the mean of the d - k discarded eigenvalues of S = Xc^T Xc / n (divide by n: it's the MLE): "
        f"{lam[3:].mean():.5f}, got {m.noise_variance_:.5f}")
    assert m.W_.shape == (12, 3), f"W_ is (d, k) = (12, 3), got {m.W_.shape}"
    C = m.W_ @ m.W_.T + m.noise_variance_ * np.eye(d)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    assert np.allclose(ev[:3], lam[:3]) and np.allclose(ev[3:], m.noise_variance_), (
        "the model covariance W W^T + sigma^2 I must reproduce the top k eigenvalues of S exactly and put sigma^2 on the rest: "
        "W = U_k sqrt(Lambda_k - sigma^2 I)")


def test_stage3_score_samples_and_mle(nb):
    X = _latent(seed=7)
    m = nb.ProbabilisticPCA(3).fit(X)
    C = m.W_ @ m.W_.T + m.noise_variance_ * np.eye(12)
    assert np.allclose(m.score_samples(X), multivariate_normal(m.mean_, C).logpdf(X)), (
        "score_samples(X) = log N(x | mean_, W W^T + sigma^2 I)")
    sk = SkPCA(3).fit(X)  # same subspace, but variances estimated with ddof=1: not the MLE
    assert m.score_samples(X).mean() > sk.score(X), (
        "your closed-form parameters are the maximum-likelihood ones, so their mean training log-likelihood must beat "
        "scikit-learn's ddof=1 estimates. check that S divides by n")
    rng = np.random.default_rng(8)
    base = m.score_samples(X).mean()
    for _ in range(5):
        W0, s0 = m.W_.copy(), m.noise_variance_
        m.W_ = W0 + 0.02 * rng.standard_normal(W0.shape)
        m.noise_variance_ = s0 * rng.uniform(0.9, 1.1)
        worse = m.score_samples(X).mean()
        m.W_, m.noise_variance_ = W0, s0
        assert worse < base, "small perturbations of the fitted W_ and noise_variance_ must lower the likelihood (it's the MLE)"


def test_stage3_transform_and_sample(nb):
    X = _latent(seed=9)
    m = nb.ProbabilisticPCA(3).fit(X)
    M = m.W_.T @ m.W_ + m.noise_variance_ * np.eye(3)
    want = np.linalg.solve(M, m.W_.T @ (X - m.mean_).T).T
    assert np.allclose(m.transform(X), want), "transform is the posterior mean M^{-1} W^T (x - mean), M = W^T W + sigma^2 I"
    S = m.sample(40000, rng=0)
    C = m.W_ @ m.W_.T + m.noise_variance_ * np.eye(12)
    assert S.shape == (40000, 12), "sample(n) returns (n, d)"
    dg = np.diag(C)
    se_mean, se_cov = np.sqrt(dg / 40000), np.sqrt((np.outer(dg, dg) + C**2) / 40000)  # standard errors
    assert np.all(np.abs(S.mean(0) - m.mean_) < 6 * se_mean) and np.all(np.abs(np.cov(S.T) - C) < 6 * se_cov), (
        "samples should have mean mean_ and covariance W W^T + sigma^2 I")
    try:
        nb.ProbabilisticPCA(12).fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError("n_components must be < d (no dimensions left for the noise): raise ValueError")


# ---------------------------------------------------------------- stage 4: kernels and KernelPCA.fit


def test_stage4_kernels(nb):
    rng = np.random.default_rng(10)
    X, Y = rng.normal(0, 1, (20, 3)), rng.normal(0, 1, (7, 3))
    assert np.allclose(nb.kernel_matrix(X, Y, "linear"), linear_kernel(X, Y)), "linear: x . y"
    assert np.allclose(nb.kernel_matrix(X, Y, "poly", gamma=0.5, degree=3, coef0=1.0),
                       polynomial_kernel(X, Y, degree=3, gamma=0.5, coef0=1.0)), "poly: (gamma x . y + coef0)^degree"
    K = nb.kernel_matrix(X, Y, "rbf", gamma=0.7)
    assert K.shape == (20, 7) and np.allclose(K, rbf_kernel(X, Y, gamma=0.7)), "rbf: exp(-gamma ||x - y||^2), shape (n, m)"


def test_stage4_fit_eigen(nb):
    X, _ = _rings(seed=11)
    kp = nb.KernelPCA(3, kernel="rbf", gamma=2.0).fit(X)
    sk = SkKernelPCA(3, kernel="rbf", gamma=2.0, eigen_solver="dense").fit(X)
    assert np.allclose(kp.lambdas_, sk.eigenvalues_), (
        f"lambdas_ are the top eigenvalues of the CENTRED Gram matrix, descending: expected {np.round(sk.eigenvalues_, 4)}, "
        f"got {np.round(kp.lambdas_, 4)}. centre with K - column means - row means + overall mean")
    assert kp.alphas_.shape == (len(X), 3) and np.allclose(kp.alphas_.T @ kp.alphas_, np.eye(3)), (
        "alphas_ (n, k) are unit-norm, orthogonal eigenvectors")
    assert _same_up_to_sign(kp.alphas_, sk.eigenvectors_), "alphas_ match sklearn's eigenvectors_ up to sign"
    big = np.abs(kp.alphas_).argmax(0)
    assert np.all(kp.alphas_[big, np.arange(3)] > 0), "each alpha column's largest-magnitude entry is positive"


def test_stage4_linear_kernel_is_pca(nb):
    X = _correlated(seed=12)
    kp = nb.KernelPCA(3, kernel="linear").fit(X)
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    assert np.allclose(kp.lambdas_, s[:3] ** 2), (
        "with a linear kernel, the centred Gram matrix Xc Xc^T has the eigenvalues s_j^2 of the centred data")


# ---------------------------------------------------------------- stage 5: KernelPCA.transform


def test_stage5_training_embedding(nb):
    X, _ = _rings(seed=13)
    kp = nb.KernelPCA(2, kernel="rbf", gamma=3.0).fit(X)
    Z = kp.transform(X)
    assert np.allclose(Z, kp.alphas_ * np.sqrt(kp.lambdas_)), (
        "on the training points, transform gives alphas_ * sqrt(lambdas_): centre k(x, X_fit) with the TRAINING column "
        "means and overall mean, and the new rows' own means")
    assert np.allclose(kp.fit_transform(X), Z), "fit_transform = fit then transform (inherited from Transformer)"


def test_stage5_new_points_match_sklearn(nb):
    X, _ = _rings(seed=14)
    Xnew = np.random.default_rng(15).uniform(-1.2, 1.2, (40, 2))
    for kw in (dict(kernel="rbf", gamma=3.0), dict(kernel="poly", gamma=1.0, degree=2, coef0=1.0)):
        ours = nb.KernelPCA(3, **kw).fit(X).transform(Xnew)
        sk = SkKernelPCA(3, eigen_solver="dense", **kw).fit(X).transform(Xnew)
        assert _same_up_to_sign(ours, sk, atol=1e-6), f"{kw}: new points should match sklearn's KernelPCA up to sign"


def test_stage5_linear_kernel_new_points_are_pca(nb):
    X = _correlated(seed=16)
    Xnew = _correlated(n=30, seed=17)
    ours = nb.KernelPCA(3, kernel="linear").fit(X).transform(Xnew)
    pca = nb.PCA(3).fit(X).transform(Xnew)
    assert _same_up_to_sign(ours, pca), (
        "linear-kernel KernelPCA on NEW points must equal PCA.transform (centred with the TRAINING mean). if it doesn't, "
        "you probably centred the new kernel rows with their own column means instead of the training ones")
