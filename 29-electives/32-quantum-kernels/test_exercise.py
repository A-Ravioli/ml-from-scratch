"""Checks for exercise.py (quantum kernels).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The ZZ feature map is
checked against its closed form, (D(z) H^n)^reps |0> with D(z) = exp(-i [sum_i z_i Z_i +
sum_{i<j} (pi - z_i)(pi - z_j) Z_i Z_j]), which is diagonal and needs no circuit simulation; the
product map against prod_q cos^2(scale (x_q - y_q) / 2); shot estimates against the binomial
distribution (statistically, then exactly for the specified draw order); alignment and the
geometric difference against direct matrix formulas (scipy's sqrtm and inverses); engineered labels
against the identity s_C / s_Q = g^2 that defines them.
"""
from functools import partial

import numpy as np
from scipy.linalg import sqrtm

S2 = 1 / np.sqrt(2)
H = S2 * np.array([[1, 1], [1, -1]], dtype=complex)


def _bits(n):
    """(2**n, n) array of the bits of every basis index, qubit 0 = most significant."""
    idx = np.arange(2 ** n)
    return (idx[:, None] >> (n - 1 - np.arange(n))) & 1


def _zz_state(x, reps=2, scale=1.0):
    z = scale * np.asarray(x, dtype=float)
    n = len(z)
    s = 1 - 2 * _bits(n)  # Z eigenvalues, +1 for bit 0
    phase = s @ z
    for i in range(n):
        for j in range(i + 1, n):
            phase = phase + (np.pi - z[i]) * (np.pi - z[j]) * s[:, i] * s[:, j]
    D = np.exp(-1j * phase)
    Hn = np.array([[1]], dtype=complex)
    for _ in range(n):
        Hn = np.kron(Hn, H)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    for _ in range(reps):
        psi = D * (Hn @ psi)
    return psi


def _unitary(nb, ops, n):
    cols = np.eye(2 ** n, dtype=complex)
    out = []
    for j in range(2 ** n):
        psi = cols[j]
        for U, t in ops:
            psi = nb.qsim.apply_gate(psi, U, t)
        out.append(psi)
    return np.stack(out, axis=1)


def _same_state(a, b):
    """equal up to a global phase."""
    return abs(abs(np.vdot(a, b)) - 1) < 1e-9


def _rbf(X, gamma):
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d)


def _centered_alignment(A, B):
    m = len(A)
    Hm = np.eye(m) - np.ones((m, m)) / m
    Ac, Bc = Hm @ A @ Hm, Hm @ B @ Hm
    return np.sum(Ac * Bc) / (np.linalg.norm(Ac) * np.linalg.norm(Bc))


def _g(Kc, Kq, reg=0.0):
    m = len(Kc)
    Kc = Kc * m / np.trace(Kc)
    Kq = Kq * m / np.trace(Kq)
    S = np.real(sqrtm(Kq))
    M = S @ np.linalg.inv(Kc + reg * np.eye(m)) @ S
    return np.sqrt(np.max(np.linalg.eigvalsh((M + M.T) / 2)))


# --- stage 1 -------------------------------------------------------------------------------------

def test_stage1_zz_map_matches_closed_form(nb):
    rng = np.random.default_rng(0)
    for n, reps, scale in [(1, 1, 1.0), (2, 1, 1.0), (3, 2, 1.0), (3, 2, 0.4)]:
        x = rng.uniform(0, 2 * np.pi, size=n)
        ops = nb.zz_feature_map(x, reps=reps, scale=scale)
        psi = nb.qsim.run_circuit(ops, n)
        want = _zz_state(x, reps, scale)
        assert _same_state(psi, want), (
            f"zz_feature_map(x, reps={reps}, scale={scale}) on {n} qubit(s) prepares the wrong state "
            f"(overlap with the closed form: {abs(np.vdot(psi, want)):.4f}). per rep: H on every qubit, RZ(2 z_q) on "
            "each qubit, then for every pair i < j: CNOT(i, j), RZ(2 (pi - z_i)(pi - z_j)) on j, CNOT(i, j); "
            "z = scale * x.")
    assert len(nb.zz_feature_map([0.1, 0.2], reps=1)) == 7, (
        "with 2 qubits and reps=1 the map has 2 H + 2 RZ + 3 gates for the pair = 7 gates")


def test_stage1_product_map(nb):
    x = np.array([0.4, -1.3, 2.0])
    ops = nb.product_feature_map(x, scale=0.5)
    assert len(ops) == 3 and all(tuple(t) == (q,) for q, (_, t) in enumerate(ops)), (
        "product_feature_map should be one RY per qubit, RY(scale * x_q) on qubit q, nothing else")
    psi = nb.qsim.run_circuit(ops, 3)
    want = np.array([1.0 + 0j])
    for v in 0.5 * x:
        want = np.kron(want, [np.cos(v / 2), np.sin(v / 2)])
    assert _same_state(psi, want), "product_feature_map: RY(scale * x_q)|0> on each qubit, a product state"


def test_stage1_inverse_ops(nb):
    rng = np.random.default_rng(1)
    ops = nb.zz_feature_map(rng.uniform(0, 2 * np.pi, size=3), reps=2)
    before = [(np.array(U), tuple(t)) for U, t in ops]
    inv = nb.inverse_ops(ops)
    U = _unitary(nb, ops, 3)
    Ui = _unitary(nb, inv, 3)
    assert np.allclose(Ui @ U, np.eye(8), atol=1e-10), (
        "running the circuit and then inverse_ops(circuit) should give the identity. reverse the "
        "order AND replace each matrix by its conjugate transpose (each alone is not enough).")
    assert all(np.array_equal(a, np.asarray(b[0])) and a_t == tuple(b[1]) for (a, a_t), b in zip(before, ops)), (
        "inverse_ops must not modify the ops passed in")


# --- stage 2 -------------------------------------------------------------------------------------

def test_stage2_feature_states(nb):
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 2 * np.pi, size=(4, 3))
    A = nb.feature_states(X)
    assert np.shape(A) == (4, 8), f"feature_states of 4 points on 3 qubits should be (4, 8), got {np.shape(A)}"
    for i in range(4):
        assert _same_state(A[i], _zz_state(X[i])), f"row {i} should be the ZZ-map state of X[{i}] (reps=2, scale=1)"
    B = nb.feature_states(X, lambda x: nb.product_feature_map(x, scale=0.3))
    assert np.allclose(np.abs(B[0]) ** 2, np.abs(nb.qsim.run_circuit(nb.product_feature_map(X[0], scale=0.3), 3)) ** 2), (
        "feature_states should use the feature_map it is given")


def test_stage2_product_kernel_closed_form(nb):
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    fm = lambda x: nb.product_feature_map(x, scale=0.7)  # noqa: E731
    K = nb.fidelity_kernel(X, Y, feature_map=fm)
    want = np.prod(np.cos(0.7 * (X[:, None, :] - Y[None, :, :]) / 2) ** 2, axis=-1)
    assert np.shape(K) == (5, 4), f"fidelity_kernel(X (5, 3), Y (4, 3)) should be (5, 4), got {np.shape(K)}"
    assert np.allclose(K, want, atol=1e-10), (
        f"for the product map the kernel is prod_q cos^2(scale (x_q - y_q) / 2); max error {np.max(np.abs(K - want)):.2e}. "
        "the fidelity is |<phi(x)|phi(y)>|^2: conjugate the first state, and square the modulus.")


def test_stage2_zz_kernel_properties(nb):
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 2 * np.pi, size=(6, 3))
    K = nb.fidelity_kernel(X)
    want = np.array([[abs(np.vdot(_zz_state(a), _zz_state(b))) ** 2 for b in X] for a in X])
    assert np.allclose(K, want, atol=1e-10), f"the ZZ fidelity kernel is off by up to {np.max(np.abs(K - want)):.2e}"
    assert np.array_equal(K, K.T), "with Y=None the kernel must be exactly symmetric"
    assert np.array_equal(np.diag(K), np.ones(6)), "with Y=None the diagonal must be exactly 1 (|<phi|phi>|^2)"
    assert np.linalg.eigvalsh(K).min() > -1e-10, "a fidelity kernel is positive semidefinite"
    K2 = nb.fidelity_kernel(X[:2], X[2:])
    assert np.allclose(K2, want[:2, 2:], atol=1e-10), "fidelity_kernel(X, Y) should be the (m, k) cross-kernel"


# --- stage 3 -------------------------------------------------------------------------------------

def test_stage3_overlap_circuit(nb):
    rng = np.random.default_rng(5)
    x, y = rng.uniform(0, 2 * np.pi, size=(2, 3))
    psi = nb.qsim.run_circuit(nb.overlap_ops(x, y), 3)
    want = abs(np.vdot(_zz_state(y), _zz_state(x))) ** 2
    assert np.isclose(abs(psi[0]) ** 2, want, atol=1e-10), (
        f"P(000) after U(y)^dagger U(x) should equal the fidelity {want:.6f}, got {abs(psi[0]) ** 2:.6f}")
    psi = nb.qsim.run_circuit(nb.overlap_ops(x, x), 3)
    assert np.isclose(abs(psi[0]) ** 2, 1), "U(x)^dagger U(x) = I, so x against itself gives P(000) = 1"
    fm = partial(nb.product_feature_map, scale=0.7)
    psi = nb.qsim.run_circuit(nb.overlap_ops(x, y, fm), 3)
    want = np.prod(np.cos(0.7 * (x - y) / 2) ** 2)
    assert np.isclose(abs(psi[0]) ** 2, want, atol=1e-10), (
        f"overlap_ops(x, y, product map): P(000) should be the product kernel {want:.6f}, got {abs(psi[0]) ** 2:.6f}. "
        "use the feature_map argument for both halves of the circuit.")


def test_stage3_estimate_kernel_statistics(nb):
    rng = np.random.default_rng(6)
    X = rng.uniform(0, 2 * np.pi, size=(5, 2))
    K = nb.fidelity_kernel(X)
    shots, reps = 50, 300
    ests = np.array([nb.estimate_kernel(X, shots, rng=100 + r) for r in range(reps)])
    E = ests[0]
    assert np.array_equal(E, E.T) and np.array_equal(np.diag(E), np.ones(5)), (
        "the estimated Gram matrix must be exactly symmetric with ones on the diagonal (estimate each pair once)")
    assert np.allclose(E * shots, np.round(E * shots)), "each estimate is a count of all-zero outcomes divided by shots"
    iu = np.triu_indices(5, 1)
    p = K[iu]
    mean, var = ests[:, iu[0], iu[1]].mean(0), ests[:, iu[0], iu[1]].var(0, ddof=1)
    se = np.sqrt(p * (1 - p) / shots / reps) + 1e-12
    assert np.all(np.abs(mean - p) < 5 * se + 1e-9), (
        f"averaged over {reps} runs the estimates should approach the exact fidelities {np.round(p, 3)}; got "
        f"{np.round(mean, 3)}")
    ratio = var[p > 0.05] / (p * (1 - p) / shots)[p > 0.05]
    assert np.all((ratio > 0.6) & (ratio < 1.6)), (
        f"each entry should have binomial variance p (1 - p) / shots; ratios observed/expected {np.round(ratio, 2)}")


def test_stage3_estimate_kernel_draw_order(nb):
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 2 * np.pi, size=(4, 2))
    K = nb.fidelity_kernel(X)
    E = nb.estimate_kernel(X, 200, rng=9)
    iu = np.triu_indices(4, 1)
    want = np.random.default_rng(9).binomial(200, np.clip(K[iu], 0, 1)) / 200
    assert np.allclose(E[iu], want), (
        "estimate_kernel should make ONE call rng.binomial(shots, p) with p the upper-triangle fidelities in "
        f"np.triu_indices order. got {E[iu]}, expected {want}")
    # and it must use the feature map it is given
    fm = partial(nb.product_feature_map, scale=0.7)
    Kp = np.prod(np.cos(0.7 * (X[:, None, :] - X[None, :, :]) / 2) ** 2, axis=-1)
    E = nb.estimate_kernel(X, 200, feature_map=fm, rng=10)
    want = np.random.default_rng(10).binomial(200, np.clip(Kp[iu], 0, 1)) / 200
    assert np.allclose(E[iu], want), (
        "estimate_kernel(X, shots, feature_map=product map) should estimate the PRODUCT kernel "
        f"prod_q cos^2(0.7 (x_q - y_q) / 2); got {E[iu]}, expected {want}. pass feature_map through.")


# --- stage 4 -------------------------------------------------------------------------------------

def test_stage4_alignment(nb):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(12, 3))
    K1, K2 = _rbf(X, 0.5), X @ X.T
    got = nb.kernel_alignment(K1, K2)
    assert isinstance(got, float), f"kernel_alignment should return a float, got {type(got).__name__}"
    assert np.isclose(got, _centered_alignment(K1, K2)), (
        f"centered alignment should be {_centered_alignment(K1, K2):.6f}, got {got:.6f}. center both kernels with "
        "H K H (center_gram) before taking the normalized Frobenius inner product.")
    assert np.isclose(nb.kernel_alignment(K1, 3 * K1), 1), "alignment ignores scale: A(K, 3K) = 1"
    assert np.isclose(nb.kernel_alignment(K1 + 5.0, K2), got), "alignment is centered: adding a constant to K changes nothing"
    y = np.where(X[:, 0] > 0, 1.0, -1.0)
    assert np.isclose(nb.kernel_alignment(np.outer(y, y), np.outer(y, y)), 1), "A(yy^T, yy^T) = 1"
    try:
        nb.kernel_alignment(np.ones((4, 4)), np.eye(4))
    except ValueError:
        pass
    else:
        raise AssertionError("a constant kernel centers to zero: kernel_alignment should raise ValueError")


def test_stage4_geometric_difference(nb):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(10, 2))
    Kc, Kq = _rbf(X, 0.3), _rbf(X, 2.0)
    got = nb.geometric_difference(Kc, Kq)
    assert np.isclose(got, _g(Kc, Kq), rtol=1e-6), (
        f"g(K_c || K_q) should be {_g(Kc, Kq):.6f}, got {got:.6f}: sqrt of the largest eigenvalue of "
        "sqrt(K_q) K_c^-1 sqrt(K_q), with both kernels scaled to trace m first.")
    assert np.isclose(nb.geometric_difference(Kq, Kq), 1, atol=1e-6), "g(K || K) = 1"
    assert np.isclose(nb.geometric_difference(4 * Kc, Kq), got, rtol=1e-6), "g is scale-free (both traces normalized to m)"
    for reg in (1e-3, 0.1):
        assert np.isclose(nb.geometric_difference(Kc, Kq, reg=reg), _g(Kc, Kq, reg), rtol=1e-6), (
            f"with reg={reg} invert K_c + reg I (after the trace normalization)")
    assert nb.geometric_difference(Kc, Kq, reg=0.1) < got, "regularizing K_c can only shrink g"


# --- stage 5 -------------------------------------------------------------------------------------

def test_stage5_engineered_targets_maximize_the_ratio(nb):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(14, 3))
    Kc, Kq = _rbf(X, 0.2), _rbf(X, 1.5)
    reg = 1e-3
    y, labels = nb.engineered_labels(Kc, Kq, reg=reg)
    m = len(X)
    Kcn, Kqn = Kc * m / np.trace(Kc), Kq * m / np.trace(Kq)
    s_c = y @ np.linalg.solve(Kcn + reg * np.eye(m), y)
    s_q = y @ np.linalg.solve(Kqn, y)
    g = _g(Kc, Kq, reg)
    assert np.shape(y) == (m,) and np.shape(labels) == (m,), "engineered_labels returns (y (m,), labels (m,))"
    assert np.isclose(s_c / s_q, g ** 2, rtol=1e-4), (
        f"for y = sqrt(K_q) v with v the top eigenvector, y^T (K_c + reg I)^-1 y / y^T K_q^-1 y should equal g^2 = "
        f"{g ** 2:.4f}; got {s_c / s_q:.4f}. use the trace-normalized kernels and the eigenvector of the LARGEST "
        "eigenvalue.")
    for seed in range(8):  # eigh's sign is arbitrary: check it's fixed on several problems
        Z = np.random.default_rng(100 + seed).normal(size=(9, 2))
        y2, _ = nb.engineered_labels(_rbf(Z, 0.3), _rbf(Z, 3.0), reg=1e-3)
        assert np.sum(y2) >= 0, "fix the eigenvector's sign so that sum(y) >= 0 (np.linalg.eigh's sign is arbitrary)"
    assert set(np.unique(labels)) <= {-1, 1} and np.sum(labels == 1) == m // 2, (
        f"labels = sign(y - median(y)) splits the points in half; got {np.sum(labels == 1)} positives out of {m}")
    assert np.array_equal(labels, np.where(y > np.median(y), 1, -1)), "labels should be +1 where y > median(y), else -1"
    y3, labels3 = nb.engineered_labels(3 * Kc, 0.5 * Kq, reg=reg)
    assert np.allclose(y3, y) and np.array_equal(labels3, labels), (
        "engineered_labels should scale both kernels to trace m first, so engineered_labels(3 K_c, 0.5 K_q) gives the "
        "same targets and labels as engineered_labels(K_c, K_q) (the regularizer reg I is added after normalizing)")


def test_stage2_worked_examples(nb):
    K = nb.fidelity_kernel(np.array([[0.3]]), np.array([[1.1]]), feature_map=lambda x: nb.zz_feature_map(x, reps=1))
    assert np.isclose(K[0, 0], np.cos(0.3 - 1.1) ** 2), (
        f"one qubit, one rep: the ZZ kernel is cos^2(z - w) = {np.cos(0.8) ** 2:.4f}, got {K[0, 0]:.4f}")
    K = nb.fidelity_kernel(np.array([[0.4, -1.3]]), np.array([[1.0, 0.7]]), feature_map=nb.product_feature_map)
    assert np.isclose(K[0, 0], 0.2664, atol=1e-4), f"the lesson's product-kernel example is 0.2664, got {K[0, 0]:.4f}"


def test_stage4_worked_example(nb):
    got = nb.geometric_difference(np.eye(2), np.array([[1.0, 0.44], [0.44, 1.0]]))
    assert np.isclose(got, 1.2), f"g(I || [[1, .44], [.44, 1]]) = sqrt(1.44) = 1.2, got {got:.4f}"


def test_stage5_unit_eigenvector(nb):
    rng = np.random.default_rng(11)
    K = _rbf(rng.normal(size=(8, 2)), 0.5)
    y, _ = nb.engineered_labels(K, K)
    m = len(K)
    Kn = K * m / np.trace(K)
    assert np.isclose(nb.geometric_difference(K, K), 1, atol=1e-6), (
        "when K_c = K_q no labels favour either kernel: g = 1")
    assert np.isclose(np.linalg.norm(np.real(sqrtm(np.linalg.pinv(Kn))) @ y), 1, atol=1e-5), (
        "y = sqrt(K_q) v with v a UNIT eigenvector, so ||sqrt(K_q)^-1 y|| = 1")
