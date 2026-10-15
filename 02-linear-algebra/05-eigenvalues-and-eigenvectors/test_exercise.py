"""Checks for exercise.py (eigenvalues and eigenvectors).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.
"""
import numpy as np

A_WORKED = np.array([[2.0, 1.0], [1.0, 2.0]])  # eigenvalues 3 and 1


def _sym_with_eigs(eigs, seed):
    """A symmetric matrix with the given eigenvalues and random orthonormal eigenvectors."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(len(eigs), len(eigs))))
    return (Q * np.asarray(eigs, dtype=float)) @ Q.T


def _parallel(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))


# ---------------------------------------------------------------- stage 1: power iteration


def test_stage1_worked_example(nb):
    lam, v = nb.power_iteration(A_WORKED, rng=0)
    v = np.asarray(v)
    assert np.isclose(lam, 3.0, atol=1e-8), f"the dominant eigenvalue of [[2,1],[1,2]] is 3, got {lam}"
    assert v.shape == (2,) and np.isclose(np.linalg.norm(v), 1.0), "return a UNIT eigenvector of shape (n,)"
    assert _parallel(v, [1.0, 1.0]) > 1 - 1e-8, f"the eigenvector for 3 is ±(1, 1)/sqrt(2), got {np.round(v, 6).tolist()}"


def test_stage1_matches_numpy(nb):
    A = _sym_with_eigs([6.0, 3.0, -2.0, 1.0, 0.5, 0.1], seed=1)
    lam, v = nb.power_iteration(A, rng=2)
    assert np.isclose(lam, 6.0, atol=1e-8), f"expected the dominant eigenvalue 6, got {lam}"
    assert np.linalg.norm(A @ v - lam * v) < 1e-7, "A v should equal lambda v at convergence (small residual)"


def test_stage1_negative_dominant_eigenvalue(nb):
    A = _sym_with_eigs([-5.0, 2.0, 1.0, 0.5], seed=3)
    lam, v = nb.power_iteration(A, rng=4)
    assert np.isclose(lam, -5.0, atol=1e-8), (
        f"the eigenvalue of largest |λ| here is -5, got {lam}. if you got +5, you returned ||A v|| "
        "instead of the Rayleigh quotient v^T A v, which keeps the sign. (and note that v flips "
        "sign every step when λ < 0, so 'stop when v stops changing' never triggers.)")


def test_stage1_nonsymmetric_markov_matrix(nb):
    # a column-stochastic matrix: its dominant eigenvalue is exactly 1
    M = np.array([[0.5, 0.3, 0.2], [0.25, 0.4, 0.3], [0.25, 0.3, 0.5]])
    lam, v = nb.power_iteration(M, rng=5)
    assert np.isclose(lam, 1.0, atol=1e-8), f"a column-stochastic matrix has dominant eigenvalue 1, got {lam}"
    pi = np.asarray(v) / np.sum(v)
    assert np.allclose(M @ pi, pi, atol=1e-8), "the eigenvector should be the stationary distribution: M pi = pi"


def test_stage1_reproducible_rng(nb):
    A = _sym_with_eigs([4.0, 3.9, 1.0], seed=6)
    l1, v1 = nb.power_iteration(A, max_iters=5, rng=7)
    l2, v2 = nb.power_iteration(A, max_iters=5, rng=np.random.default_rng(7))
    assert np.isclose(l1, l2) and np.allclose(v1, v2), (
        "the same seed must give the same result, whether passed as an int or as a Generator. "
        "start from rng = np.random.default_rng(rng) and draw the initial vector from it.")
    _, v3 = nb.power_iteration(A, max_iters=2, rng=7)
    _, v4 = nb.power_iteration(A, max_iters=2, rng=8)
    assert not np.allclose(np.abs(v3), np.abs(v4)), (
        "different seeds should start from different random vectors (so after 2 steps the iterates "
        "differ), but yours agree: is the starting vector drawn from the rng argument?")


# ---------------------------------------------------------------- stage 2: inverse iteration


def test_stage2_worked_example(nb):
    lam, v = nb.inverse_iteration(A_WORKED, 0.9, rng=0)
    assert np.isclose(lam, 1.0, atol=1e-8), f"with shift 0.9 the nearest eigenvalue of [[2,1],[1,2]] is 1, got {lam}"
    assert _parallel(v, [1.0, -1.0]) > 1 - 1e-8, "the eigenvector for 1 is ±(1, -1)/sqrt(2)"


def test_stage2_finds_eigenvalue_nearest_the_shift(nb):
    eigs = np.array([7.0, 4.0, 2.5, 1.0, -3.0])
    A = _sym_with_eigs(eigs, seed=8)
    for mu in (3.9, 2.0, -2.0, 0.0):
        lam, v = nb.inverse_iteration(A, mu, rng=9)
        want = eigs[np.argmin(np.abs(eigs - mu))]
        assert np.isclose(lam, want, atol=1e-8), (
            f"shift {mu}: the nearest eigenvalue is {want}, got {lam}. inverse iteration multiplies by "
            "(A - mu I)^{-1}: factor A - mu I once with lu and call lu_solve each step.")
        assert np.isclose(np.linalg.norm(v), 1.0) and np.linalg.norm(A @ v - lam * v) < 1e-7


def test_stage2_shift_equal_to_an_eigenvalue(nb):
    A = np.diag([1.0, 2.0, 3.0])
    lam, v = nb.inverse_iteration(A, 2.0, rng=10)
    assert np.isclose(lam, 2.0, atol=1e-8), (
        f"shifting by exactly an eigenvalue makes A - mu I singular; nudge mu by a tiny amount and "
        f"carry on (it converges in one step). got {lam}")


def test_stage2_default_shift_gives_smallest_magnitude(nb):
    A = _sym_with_eigs([10.0, -6.0, 0.3, 2.0], seed=11)
    lam, _ = nb.inverse_iteration(A, rng=12)
    assert np.isclose(lam, 0.3, atol=1e-8), f"with the default mu = 0, inverse iteration finds the smallest |λ| (0.3), got {lam}"


# ---------------------------------------------------------------- stage 3: unshifted QR algorithm


def test_stage3_worked_example(nb):
    w = np.asarray(nb.qr_algorithm(A_WORKED))
    assert w.shape == (2,) and np.allclose(w, [3.0, 1.0], atol=1e-8), (
        f"qr_algorithm([[2,1],[1,2]]) should return [3, 1] (sorted descending), got {np.round(w, 8).tolist()}")


def test_stage3_symmetric_matches_numpy(nb):
    eigs = np.array([9.0, 5.0, 3.0, 1.5, 0.5, -1.0])
    w = np.asarray(nb.qr_algorithm(_sym_with_eigs(eigs, seed=13)))
    assert np.allclose(w, np.sort(eigs)[::-1], atol=1e-8), (
        f"expected {np.sort(eigs)[::-1].tolist()}, got {np.round(w, 6).tolist()}. each step is "
        "Q, R = qr(A_k); A_{k+1} = R @ Q, and the eigenvalues appear on the diagonal.")


def test_stage3_nonsymmetric_real_eigenvalues(nb):
    rng = np.random.default_rng(14)
    S = rng.normal(size=(4, 4)) + 3 * np.eye(4)
    A = S @ np.diag([4.0, 2.0, 1.0, -0.5]) @ np.linalg.inv(S)
    w = np.asarray(nb.qr_algorithm(A))
    assert np.allclose(w, [4.0, 2.0, 1.0, -0.5], atol=1e-7), (
        f"for a non-symmetric matrix with real eigenvalues, A_k tends to upper-triangular and its "
        f"diagonal holds the eigenvalues; expected [4, 2, 1, -0.5], got {np.round(w, 6).tolist()}. "
        "your stopping test should look at the part strictly BELOW the diagonal only.")


# ---------------------------------------------------------------- stage 4: Jacobi eigh


def _check_eigh(A, w, V, label, tol=1e-10):
    w, V = np.asarray(w, dtype=float), np.asarray(V, dtype=float)
    n = A.shape[0]
    assert w.shape == (n,) and V.shape == (n, n), f"{label}: expected w ({n},) and V ({n}, {n})"
    assert np.all(np.diff(w) >= -1e-12), f"{label}: eigenvalues must be sorted ASCENDING (like np.linalg.eigh)"
    assert np.allclose(w, np.linalg.eigvalsh(A), atol=tol * max(1, np.abs(A).max())), (
        f"{label}: eigenvalues {np.round(w, 6).tolist()} don't match {np.round(np.linalg.eigvalsh(A), 6).tolist()}")
    assert np.abs(V.T @ V - np.eye(n)).max() < 1e-10, f"{label}: the columns of V must be orthonormal"
    assert np.abs(A @ V - V * w).max() < tol * 10 * max(1, np.abs(A).max()), (
        f"{label}: A @ V should equal V @ diag(w): column j of V is the eigenvector for w[j]. if the "
        "eigenvalues are right but this fails, you sorted w without reordering V's columns.")


def test_stage4_worked_example(nb):
    w, V = nb.eigh(A_WORKED)
    _check_eigh(A_WORKED, w, V, "worked example")
    assert _parallel(np.asarray(V)[:, 0], [1.0, -1.0]) > 1 - 1e-10, "the first column (eigenvalue 1) should be ±(1, -1)/sqrt(2)"


def test_stage4_random_symmetric(nb):
    X = np.random.default_rng(15).normal(size=(12, 12))
    A = X + X.T
    w, V = nb.eigh(A)
    _check_eigh(A, w, V, "random symmetric 12x12")


def test_stage4_repeated_and_diagonal(nb):
    A = _sym_with_eigs([2.0, 2.0, 5.0, -1.0], seed=16)
    w, V = nb.eigh(A)
    _check_eigh(A, w, V, "repeated eigenvalue 2")
    D = np.diag([3.0, -1.0, 2.0])
    w, V = nb.eigh(D)
    _check_eigh(D, w, V, "an already-diagonal matrix")


def test_stage4_does_not_modify_input(nb):
    X = np.random.default_rng(17).normal(size=(5, 5))
    A = X + X.T
    A0 = A.copy()
    nb.eigh(A)
    assert np.array_equal(A, A0), "eigh changed its input; rotate a float copy"
