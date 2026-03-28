import numpy as np

from exercise import (
    KernelPCA,
    KernelRidgeRegression,
    LaplacianKernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    SupportVectorMachine,
    compute_kernel_alignment,
    compute_kernel_matrix,
    kernel_centering,
    kernel_matrix_approximation,
)


def test_kernel_matrices_centering_and_alignment():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    linear_kernel = LinearKernel()
    polynomial_kernel = PolynomialKernel(degree=2, coef=1.0)
    rbf_kernel = RBFKernel(sigma=1.0)
    laplacian_kernel = LaplacianKernel(gamma=1.0)

    K_linear = compute_kernel_matrix(linear_kernel, X)
    assert np.allclose(K_linear, X @ X.T)
    assert np.allclose(np.diag(compute_kernel_matrix(rbf_kernel, X)), 1.0)
    assert np.allclose(
        compute_kernel_matrix(laplacian_kernel, X),
        compute_kernel_matrix(laplacian_kernel, X).T,
    )

    K_poly = compute_kernel_matrix(polynomial_kernel, X)
    centered = kernel_centering(K_linear)
    assert np.allclose(centered.sum(axis=0), 0.0)
    assert np.isclose(compute_kernel_alignment(K_poly, K_poly), 1.0)

    approximation, basis = kernel_matrix_approximation(K_linear, rank=2)
    assert approximation.shape == K_linear.shape
    assert basis.shape == (3, 2)


def test_kernel_ridge_regression_and_kernel_pca():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    ridge = KernelRidgeRegression(LinearKernel(), lambda_reg=1e-3)
    ridge.fit(X, y)
    predictions = ridge.predict(X)

    assert predictions.shape == y.shape
    assert np.mean((predictions - y) ** 2) < 1e-2

    kpca = KernelPCA(LinearKernel(), n_components=1)
    transformed = kpca.fit_transform(np.array([[0.0], [1.0], [2.0], [3.0]]))
    assert transformed.shape == (4, 1)


def test_kernel_svm_handles_tiny_separable_dataset():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0])

    svm = SupportVectorMachine(LinearKernel(), C=10.0, solver="quadratic_programming")
    svm.fit(X, y)
    predictions = svm.predict(X)

    assert predictions.shape == y.shape
    assert np.mean(predictions == y) >= 0.75
