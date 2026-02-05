"""
Vector Spaces and Linear Transformations - Reference Solutions
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class VectorSpace:
    """Abstract base class for vector spaces."""
    
    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
    
    def verify_axioms(self, vectors: List[np.ndarray], scalars: List[float], 
                     tolerance: float = 1e-10) -> dict:
        """Verify vector space axioms."""
        results = {}
        
        if len(vectors) < 2 or len(scalars) < 2:
            return {"error": "Need at least 2 vectors and 2 scalars"}
        
        u, v, w = vectors[0], vectors[1], vectors[0] if len(vectors) < 3 else vectors[2]
        a, b = scalars[0], scalars[1]
        
        # 1. Closure under addition (assumed by numpy arrays)
        results['addition_closure'] = True
        
        # 2. Closure under scalar multiplication (assumed by numpy arrays)
        results['scalar_multiplication_closure'] = True
        
        # 3. Associativity of addition
        results['addition_associativity'] = np.allclose((u + v) + w, u + (v + w), atol=tolerance)
        
        # 4. Commutativity of addition
        results['addition_commutativity'] = np.allclose(u + v, v + u, atol=tolerance)
        
        # 5. Existence of zero vector
        zero = np.zeros_like(u)
        results['zero_exists'] = np.allclose(u + zero, u, atol=tolerance)
        
        # 6. Existence of additive inverse
        results['inverse_exists'] = np.allclose(u + (-u), zero, atol=tolerance)
        
        # 7. Distributivity
        results['scalar_distributivity'] = np.allclose(a * (u + v), a * u + a * v, atol=tolerance)
        results['vector_distributivity'] = np.allclose((a + b) * u, a * u + b * u, atol=tolerance)
        
        # 8. Scalar multiplication associativity
        results['scalar_associativity'] = np.allclose(a * (b * u), (a * b) * u, atol=tolerance)
        
        # 9. Scalar multiplication identity
        results['scalar_identity'] = np.allclose(1 * u, u, atol=tolerance)
        
        results['all_axioms_satisfied'] = all(v for k, v in results.items() if k != 'all_axioms_satisfied')
        
        return results


def check_linear_independence(vectors: List[np.ndarray], tolerance: float = 1e-10) -> bool:
    """Check if vectors are linearly independent."""
    if not vectors:
        return True
    
    # Stack vectors as columns
    matrix = np.column_stack(vectors)
    
    # Check rank
    rank = np.linalg.matrix_rank(matrix, tol=tolerance)
    return rank == len(vectors)


def find_basis(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """Find a basis for the span of the given vectors."""
    if not vectors:
        return []
    
    # Stack vectors as rows for row reduction
    matrix = np.vstack(vectors)
    
    # Perform QR decomposition
    Q, R = np.linalg.qr(matrix.T)
    
    # Find non-zero columns (pivot columns)
    tol = 1e-10
    rank = np.sum(np.abs(np.diag(R)) > tol)
    
    # Return first 'rank' columns of Q as basis
    basis = []
    for i in range(rank):
        basis.append(Q[:, i])
    
    return basis


def gram_schmidt(vectors: List[np.ndarray], normalize: bool = True) -> List[np.ndarray]:
    """Modified Gram-Schmidt orthogonalization for numerical stability."""
    if not vectors:
        return []
    
    ortho_vectors = []
    
    for v in vectors:
        # Start with current vector
        u = v.copy().astype(float)
        
        # Subtract projections onto all previous orthogonal vectors
        for ortho_v in ortho_vectors:
            projection = np.dot(u, ortho_v) * ortho_v
            u = u - projection
        
        # Check if vector is non-zero (linearly independent)
        norm = np.linalg.norm(u)
        if norm > 1e-10:
            if normalize:
                u = u / norm
            ortho_vectors.append(u)
    
    return ortho_vectors


def projection_matrix(subspace_basis: List[np.ndarray]) -> np.ndarray:
    """Compute projection matrix onto subspace."""
    if not subspace_basis:
        return np.array([[]])
    
    # Orthonormalize basis first
    ortho_basis = gram_schmidt(subspace_basis, normalize=True)
    
    # Stack basis vectors as columns
    U = np.column_stack(ortho_basis)
    
    # Projection matrix P = UU^T
    P = U @ U.T
    
    return P


class LinearTransformation:
    """Linear transformation class."""
    
    def __init__(self, matrix: Optional[np.ndarray] = None, 
                 transform_func: Optional[Callable] = None):
        self.matrix = matrix
        self.transform_func = transform_func
    
    def apply(self, v: np.ndarray) -> np.ndarray:
        if self.matrix is not None:
            return self.matrix @ v
        elif self.transform_func is not None:
            return self.transform_func(v)
        else:
            raise ValueError("No transformation defined")
    
    def find_kernel(self, sample_vectors: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Find kernel (null space) of transformation."""
        if self.matrix is None:
            raise ValueError("Kernel finding only implemented for matrix transformations")
        
        # Find null space using SVD
        U, s, Vt = np.linalg.svd(self.matrix)
        
        # Tolerance for zero singular values
        tol = 1e-10
        rank = np.sum(s > tol)
        
        # Null space basis vectors are columns of V corresponding to zero singular values
        kernel_basis = []
        for i in range(rank, Vt.shape[0]):
            kernel_basis.append(Vt[i, :])
        
        return kernel_basis
    
    def find_image(self, sample_vectors: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Find image (column space) of transformation."""
        if self.matrix is None:
            raise ValueError("Image finding only implemented for matrix transformations")
        
        # Column space is found using QR decomposition
        Q, R = np.linalg.qr(self.matrix)
        
        # Find rank
        tol = 1e-10
        rank = np.sum(np.abs(np.diag(R)) > tol)
        
        # First 'rank' columns of Q form basis for column space
        image_basis = []
        for i in range(rank):
            image_basis.append(Q[:, i])
        
        return image_basis
    
    def verify_rank_nullity(self) -> dict:
        """Verify rank-nullity theorem."""
        if self.matrix is None:
            raise ValueError("Only implemented for matrix transformations")
        
        kernel_basis = self.find_kernel()
        image_basis = self.find_image()
        
        domain_dim = self.matrix.shape[1]
        kernel_dim = len(kernel_basis)
        image_dim = len(image_basis)
        
        return {
            'domain_dim': domain_dim,
            'kernel_dim': kernel_dim,
            'image_dim': image_dim,
            'sum': kernel_dim + image_dim,
            'verified': domain_dim == kernel_dim + image_dim
        }


def eigendecomposition(A: np.ndarray, max_iterations: int = 1000, 
                      tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """QR algorithm for eigendecomposition."""
    n = A.shape[0]
    Ak = A.copy()
    Q_total = np.eye(n)
    
    for _ in range(max_iterations):
        # QR decomposition
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
        Q_total = Q_total @ Q
        
        # Check for convergence (off-diagonal elements small)
        off_diagonal = Ak - np.diag(np.diag(Ak))
        if np.max(np.abs(off_diagonal)) < tolerance:
            break
    
    eigenvalues = np.diag(Ak)
    eigenvectors = Q_total
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def power_method(A: np.ndarray, max_iterations: int = 1000, 
                tolerance: float = 1e-10) -> Tuple[float, np.ndarray]:
    """Power method for dominant eigenvalue."""
    n = A.shape[0]
    
    # Random initial vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    eigenvalue = 0
    
    for _ in range(max_iterations):
        # Apply transformation
        Av = A @ v
        
        # Estimate eigenvalue (Rayleigh quotient)
        eigenvalue_new = np.dot(v, Av)
        
        # Normalize
        v_new = Av / np.linalg.norm(Av)
        
        # Check convergence
        if abs(eigenvalue_new - eigenvalue) < tolerance:
            break
        
        eigenvalue = eigenvalue_new
        v = v_new
    
    return eigenvalue, v


def svd_from_scratch(A: np.ndarray, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD using eigendecomposition."""
    m, n = A.shape
    
    # Compute A^T A
    AtA = A.T @ A
    
    # Eigendecomposition of A^T A
    eigenvals_V, V = np.linalg.eigh(AtA)
    
    # Sort in descending order
    idx = np.argsort(eigenvals_V)[::-1]
    eigenvals_V = eigenvals_V[idx]
    V = V[:, idx]
    
    # Singular values
    singular_vals = np.sqrt(np.maximum(eigenvals_V, 0))
    
    # Compute U
    r = np.sum(singular_vals > 1e-10)  # Rank
    U = np.zeros((m, m if full_matrices else r))
    
    for i in range(r):
        if singular_vals[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / singular_vals[i]
    
    # Complete U with orthogonal vectors if needed
    if full_matrices and r < m:
        # Use QR decomposition to complete basis
        Q, _ = np.linalg.qr(U[:, :r])
        U[:, :r] = Q
        
        # Find remaining orthogonal vectors
        null_basis = np.eye(m)
        for i in range(r, m):
            for j in range(m):
                v = null_basis[:, j]
                # Orthogonalize against existing columns
                for k in range(i):
                    v = v - np.dot(v, U[:, k]) * U[:, k]
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    U[:, i] = v / norm
                    break
    
    # Return compact or full matrices
    if full_matrices:
        return U, singular_vals, V.T
    else:
        return U[:, :r], singular_vals[:r], V[:, :r].T


def matrix_condition_number(A: np.ndarray, p: int = 2) -> float:
    """Compute condition number."""
    if p == 2:
        # Use SVD for 2-norm
        _, s, _ = np.linalg.svd(A)
        if s[-1] == 0:
            return np.inf
        return s[0] / s[-1]
    else:
        # Use numpy for other norms
        return np.linalg.cond(A, p)


def low_rank_approximation(A: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-k approximation using SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only top k components
    Uk = U[:, :rank]
    sk = s[:rank]
    Vtk = Vt[:rank, :]
    
    # Reconstruct
    return Uk @ np.diag(sk) @ Vtk


class PCA:
    """PCA implementation."""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, X: np.ndarray):
        """Fit PCA using eigendecomposition."""
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Store components and variance
        self.components = eigenvecs[:, :self.n_components].T
        self.explained_variance = eigenvals[:self.n_components]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components."""
        X_centered = X - self.mean
        return X_centered @ self.components.T
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform back to original space."""
        return X_transformed @ self.components + self.mean


def visualize_linear_transformation(T: LinearTransformation, 
                                   grid_range: Tuple[float, float] = (-2, 2),
                                   grid_points: int = 20):
    """Visualize linear transformation effects."""
    if T.matrix is None or T.matrix.shape != (2, 2):
        print("Visualization only supported for 2D matrix transformations")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create grid
    x = np.linspace(grid_range[0], grid_range[1], grid_points)
    y = np.linspace(grid_range[0], grid_range[1], grid_points)
    
    # Plot original grid
    for xi in x:
        points = np.array([[xi, yi] for yi in y])
        ax1.plot(points[:, 0], points[:, 1], 'b-', alpha=0.3)
    for yi in y:
        points = np.array([[xi, yi] for xi in x])
        ax1.plot(points[:, 0], points[:, 1], 'b-', alpha=0.3)
    
    # Plot transformed grid
    for xi in x:
        points = np.array([T.apply(np.array([xi, yi])) for yi in y])
        ax2.plot(points[:, 0], points[:, 1], 'r-', alpha=0.3)
    for yi in y:
        points = np.array([T.apply(np.array([xi, yi])) for xi in x])
        ax2.plot(points[:, 0], points[:, 1], 'r-', alpha=0.3)
    
    # Plot eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(T.matrix)
    for i in range(len(eigenvals)):
        if np.isreal(eigenvals[i]):
            v = eigenvecs[:, i].real
            ax1.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, 
                     fc='green', ec='green', linewidth=2)
            ax2.arrow(0, 0, eigenvals[i].real*v[0], eigenvals[i].real*v[1], 
                     head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)
    
    ax1.set_title('Original Space')
    ax2.set_title('Transformed Space')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_eigenspace_invariance(A: np.ndarray):
    """Show eigenvectors define invariant directions."""
    if A.shape != (2, 2):
        print("Demonstration only for 2x2 matrices")
        return
    
    eigenvals, eigenvecs = np.linalg.eig(A)
    
    fig, axes = plt.subplots(1, len(eigenvals), figsize=(6*len(eigenvals), 5))
    if len(eigenvals) == 1:
        axes = [axes]
    
    for idx, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
        if not np.isreal(eigenval):
            continue
            
        ax = axes[idx]
        
        # Show iterations of A on eigenvector
        v = eigenvec.real
        v = v / np.linalg.norm(v)  # Normalize
        
        points = [v]
        for i in range(5):
            v = A @ v
            v = v / np.linalg.norm(v)  # Normalize to prevent overflow
            points.append(v)
        
        points = np.array(points)
        
        # Plot trajectory
        ax.plot(points[:, 0], points[:, 1], 'bo-', markersize=8, linewidth=2)
        ax.arrow(0, 0, eigenvec[0], eigenvec[1], head_width=0.05, 
                head_length=0.05, fc='red', ec='red', linewidth=2)
        
        ax.set_title(f'Eigenvalue: {eigenval.real:.3f}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.show()


def analyze_gradient_descent_convergence(A: np.ndarray, b: np.ndarray, 
                                       learning_rates: List[float]):
    """Analyze GD convergence vs condition number."""
    # Optimal solution
    x_star = np.linalg.solve(A.T @ A, A.T @ b)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute condition number and eigenvalues
    cond = matrix_condition_number(A)
    eigenvals = np.linalg.eigvals(A.T @ A)
    
    for lr in learning_rates:
        x = np.zeros_like(b)
        errors = []
        
        for i in range(100):
            # Gradient of ||Ax - b||^2
            grad = 2 * A.T @ (A @ x - b)
            x = x - lr * grad
            
            error = np.linalg.norm(x - x_star)
            errors.append(error)
        
        ax1.semilogy(errors, label=f'lr={lr}')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error ||x - x*||')
    ax1.set_title(f'Convergence (κ={cond:.1f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Show eigenvalue distribution
    ax2.scatter(range(len(eigenvals)), sorted(eigenvals, reverse=True), s=50)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_title('Eigenvalue Distribution of A^T A')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Vector Spaces Solutions - Testing")
    
    # Test linear independence
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])
    print(f"Linear independence: {check_linear_independence([v1, v2, v3])}")
    
    # Test Gram-Schmidt
    vectors = [np.array([1.0, 1.0, 0.0]), 
               np.array([1.0, 0.0, 1.0])]
    ortho = gram_schmidt(vectors)
    print(f"Orthogonal vectors: {ortho}")
    
    # Test eigendecomposition
    A = np.array([[4, -2], [-1, 3]])
    eigenvals, eigenvecs = eigendecomposition(A)
    print(f"Eigenvalues: {eigenvals}")
    
    # Test SVD
    A = np.array([[1, 2], [3, 4], [5, 6]])
    U, s, Vt = svd_from_scratch(A)
    print(f"Singular values: {s}")
    
    # Visualizations
    T = LinearTransformation(matrix=np.array([[2, 1], [0, 1]]))
    visualize_linear_transformation(T)
