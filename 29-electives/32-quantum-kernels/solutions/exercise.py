import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson, notebooks
    from mlfs.kernels import center_gram  # yours, from ch 09
    from mlfs.utils import as_rng  # yours, from ch 00

    def _qsim_smoke(m):
        """touch every simulator function this notebook uses, on a Bell pair."""
        bell = m.run_circuit([(m.gate("H"), [0]), (m.gate("CNOT"), [0, 1])], 2)
        m.apply_gate(bell, m.rot("y", 0.1), [1])
        m.probabilities(bell)
        m.sample_counts(bell, 4, 0)
        m.measure_qubit(bell, 0, 0)
        m.expval(bell, "ZZ")

    # your state-vector simulator from topic 30: the reference under pytest and verify.py (or with
    # MLFS_REFERENCE=qsim), and also whenever a topic 30 stage it needs is still unfinished
    qsim_load = notebooks.load_topic("30-qubits-and-circuits", key="qsim", smoke=_qsim_smoke)
    qsim = qsim_load.module


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # quantum kernels · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. circuits run on your topic 30 simulator, available here as
    `qsim`; the SVM is your ch 09 `mlfs.kernels.SVC`.

    | stage | you build |
    |---|---|
    | 1 | `zz_feature_map`, `product_feature_map`, `inverse_ops`: feature-map circuits and their inverses |
    | 2 | `feature_states`, `fidelity_kernel`: the quantum kernel $\lvert\langle\phi(x)\vert\phi(y)\rangle\rvert^2$ |
    | 3 | `overlap_ops`, `estimate_kernel`: the compute–uncompute circuit and a finite-shot Gram matrix |
    | 4 | `kernel_alignment`, `geometric_difference`: comparing kernels with each other and with labels |
    | 5 | `engineered_labels`: the labels on which one kernel beats another by the most |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    qsim_load.callout()
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def zz_feature_map(x, *, reps=2, scale=1.0):
    """stage 1: the ZZ feature map of z = scale * x: reps x [H on all; RZ(2 z_i) on i; for i < j: CNOT(i,j) RZ(2 (pi - z_i)(pi - z_j)) on j CNOT(i,j)]."""
    z = scale * np.asarray(x, dtype=float).ravel()
    n = len(z)
    H, CNOT = qsim.gate("H"), qsim.gate("CNOT")
    ops = []
    for _ in range(reps):
        ops += [(H, (q,)) for q in range(n)]
        ops += [(qsim.rot("z", 2 * z[q]), (q,)) for q in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                ops += [(CNOT, (i, j)), (qsim.rot("z", 2 * (np.pi - z[i]) * (np.pi - z[j])), (j,)), (CNOT, (i, j))]
    return ops


@app.function
def product_feature_map(x, *, scale=1.0):
    """stage 1: angle encoding of z = scale * x with no entangling gates: RY(z_q) on qubit q."""
    z = scale * np.asarray(x, dtype=float).ravel()
    return [(qsim.rot("y", v), (q,)) for q, v in enumerate(z)]


@app.function
def inverse_ops(ops):
    """stage 1: the inverse circuit: the ops in reverse order, each matrix replaced by its conjugate transpose."""
    return [(np.asarray(U).conj().T, tuple(t)) for U, t in reversed(list(ops))]


@app.cell(hide_code=True)
def _():
    checks.run(1, zz_feature_map=zz_feature_map, product_feature_map=product_feature_map,
               inverse_ops=inverse_ops, qsim=qsim)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def feature_states(X, feature_map=None):
    """stage 2: the encoded states |phi(x_i)> = U(x_i)|0...0> as rows of a complex (m, 2**n) array; feature_map: x -> ops (default zz_feature_map)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    fm = zz_feature_map if feature_map is None else feature_map
    n = X.shape[1]
    return np.array([qsim.run_circuit(fm(x), n) for x in X])


@app.function
def fidelity_kernel(X, Y=None, *, feature_map=None):
    """stage 2: K[i, j] = |<phi(x_i)|phi(y_j)>|**2, shape (m, k); with Y None, (m, m), symmetric with ones on the diagonal."""
    A = feature_states(X, feature_map)
    if Y is None:
        K = np.abs(A.conj() @ A.T) ** 2
        K = (K + K.T) / 2
        np.fill_diagonal(K, 1.0)
        return K
    B = feature_states(Y, feature_map)
    return np.abs(A.conj() @ B.T) ** 2


@app.cell(hide_code=True)
def _():
    checks.run(2, feature_states=feature_states, fidelity_kernel=fidelity_kernel,
               zz_feature_map=zz_feature_map, product_feature_map=product_feature_map, qsim=qsim)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def overlap_ops(x, y, feature_map=None):
    """stage 3: the compute-uncompute circuit U(y)^dagger U(x): P(all zeros) at its output is |<phi(y)|phi(x)>|**2."""
    fm = zz_feature_map if feature_map is None else feature_map
    return list(fm(x)) + inverse_ops(fm(y))


@app.function
def estimate_kernel(X, shots, *, feature_map=None, rng=None):
    """stage 3: finite-shot Gram matrix: one rng.binomial(shots, p) call over the upper-triangle fidelities (np.triu_indices order), divided by shots; mirrored, diagonal 1."""
    gen = as_rng(rng)
    K = fidelity_kernel(X, feature_map=feature_map)
    m = len(K)
    iu = np.triu_indices(m, 1)
    p = np.clip(K[iu], 0.0, 1.0)
    est = np.eye(m)
    est[iu] = gen.binomial(shots, p) / shots
    est.T[iu] = est[iu]
    return est


@app.cell(hide_code=True)
def _():
    checks.run(3, overlap_ops=overlap_ops, estimate_kernel=estimate_kernel, fidelity_kernel=fidelity_kernel,
               zz_feature_map=zz_feature_map, qsim=qsim, product_feature_map=product_feature_map)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def kernel_alignment(K1, K2):
    """stage 4: centered alignment <K1c, K2c>_F / (||K1c||_F ||K2c||_F), with Kc = H K H (center_gram)."""
    A, B = center_gram(np.asarray(K1, dtype=float)), center_gram(np.asarray(K2, dtype=float))
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    if denom == 0:
        raise ValueError("a centered kernel is zero: alignment is undefined")
    return float(np.sum(A * B) / denom)


@app.function
def psd_sqrt(K):
    """symmetric PSD square root via eigh, with negative eigenvalues clipped to 0."""
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


@app.function
def geometric_difference(K_c, K_q, *, reg=0.0):
    """stage 4: g(K_c || K_q) = sqrt(largest eigenvalue of sqrt(K_q) (K_c + reg I)^-1 sqrt(K_q)), both kernels first scaled to trace m."""
    K_c, K_q = np.asarray(K_c, dtype=float), np.asarray(K_q, dtype=float)
    m = len(K_c)
    K_c = K_c * (m / np.trace(K_c))
    K_q = K_q * (m / np.trace(K_q))
    S = psd_sqrt(K_q)
    M = S @ np.linalg.solve(K_c + reg * np.eye(m), S)
    return float(np.sqrt(max(np.linalg.eigvalsh((M + M.T) / 2)[-1], 0.0)))


@app.cell(hide_code=True)
def _():
    checks.run(4, kernel_alignment=kernel_alignment, geometric_difference=geometric_difference)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def engineered_labels(K_c, K_q, *, reg=0.0):
    """stage 5: targets y = sqrt(K_q) v (v the top unit eigenvector of sqrt(K_q)(K_c + reg I)^-1 sqrt(K_q), traces scaled to m) and labels sign(y - median(y)); returns (y, labels)."""
    K_c, K_q = np.asarray(K_c, dtype=float), np.asarray(K_q, dtype=float)
    m = len(K_c)
    K_c = K_c * (m / np.trace(K_c))
    K_q = K_q * (m / np.trace(K_q))
    S = psd_sqrt(K_q)
    M = S @ np.linalg.solve(K_c + reg * np.eye(m), S)
    _, V = np.linalg.eigh((M + M.T) / 2)
    y = S @ V[:, -1]
    if np.sum(y) < 0:  # fix the eigenvector's arbitrary sign
        y = -y
    labels = np.where(y > np.median(y), 1, -1)
    return y, labels


@app.cell(hide_code=True)
def _():
    checks.run(5, engineered_labels=engineered_labels, geometric_difference=geometric_difference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it ends with the block's
    project: when (if ever) does a quantum kernel beat a classical one?
    """)
    return


if __name__ == "__main__":
    app.run()
