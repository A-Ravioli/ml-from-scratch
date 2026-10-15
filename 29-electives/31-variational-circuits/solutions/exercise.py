import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson, notebooks
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
    # variational circuits · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. circuits run on your topic 30 simulator, available here as
    `qsim` (`qsim.run_circuit`, `qsim.rot`, `qsim.gate`, `qsim.expval`, …).

    | stage | you build |
    |---|---|
    | 1 | `ansatz_ops`, `encoding_ops`, `model_ops`: a hardware-efficient ansatz with angle encoding and optional re-uploading |
    | 2 | `expectation`, `estimate_pauli`: exact and shot-based expectation values |
    | 3 | `parameter_shift`: exact gradients from two shifted circuit runs per parameter |
    | 4 | `train_classifier`, `classify`: a variational classifier trained with parameter-shift gradients and Adam |
    | 5 | `gradient_samples`: the distribution of one gradient component over random circuits |

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
def ansatz_ops(theta):
    """stage 1: hardware-efficient ansatz for theta (L, n, 2): per layer RY(theta[l,q,0]) then RZ(theta[l,q,1]) on each qubit, then CNOT(q, q+1)."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 3 or theta.shape[2] != 2:
        raise ValueError(f"theta must have shape (L, n, 2), got {theta.shape}")
    L, n, _ = theta.shape
    ops = []
    for l in range(L):
        for q in range(n):
            ops.append((qsim.rot("y", theta[l, q, 0]), (q,)))
            ops.append((qsim.rot("z", theta[l, q, 1]), (q,)))
        for q in range(n - 1):
            ops.append((qsim.gate("CNOT"), (q, q + 1)))
    return ops


@app.function
def encoding_ops(x):
    """stage 1: angle encoding: RY(x[q]) on qubit q, for a feature vector x (n,)."""
    return [(qsim.rot("y", float(v)), (q,)) for q, v in enumerate(np.asarray(x, dtype=float).ravel())]


@app.function
def model_ops(theta, x, *, reupload=False):
    """stage 1: encoding then the whole ansatz; with reupload=True, the encoding before every layer instead."""
    theta = np.asarray(theta, dtype=float)
    if len(np.ravel(x)) != theta.shape[1]:
        raise ValueError(f"x has {len(np.ravel(x))} features but theta is for {theta.shape[1]} qubits")
    if not reupload:
        return encoding_ops(x) + ansatz_ops(theta)
    ops = []
    for l in range(theta.shape[0]):
        ops += encoding_ops(x) + ansatz_ops(theta[l:l + 1])
    return ops


@app.cell(hide_code=True)
def _():
    checks.run(1, ansatz_ops=ansatz_ops, encoding_ops=encoding_ops, model_ops=model_ops, qsim=qsim)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def expectation(theta, x, observable=None, *, reupload=False):
    """stage 2: <psi(theta, x)| O |psi(theta, x)> for a Pauli-string observable (default Z on qubit 0), a float."""
    n = np.shape(theta)[1]
    observable = "Z" + "I" * (n - 1) if observable is None else observable
    psi = qsim.run_circuit(model_ops(theta, x, reupload=reupload), n)
    return qsim.expval(psi, observable)


@app.function
def estimate_pauli(state, pauli, shots, rng=None):
    """stage 2: estimate <state| P |state> from `shots` measurements, rotating each X/Y qubit into the Z basis first."""
    psi = np.asarray(state, dtype=complex)
    if 2 ** len(pauli) != psi.size or set(pauli) - set("IXYZ"):
        raise ValueError(f"pauli must be a string of log2(len(state)) characters from 'IXYZ', got {pauli!r}")
    to_z = {"X": qsim.gate("H"), "Y": qsim.gate("H") @ qsim.gate("S").conj().T}
    for q, p in enumerate(pauli):
        if p in to_z:
            psi = qsim.apply_gate(psi, to_z[p], [q])
    support = [q for q, p in enumerate(pauli) if p != "I"]
    counts = qsim.sample_counts(psi, shots, as_rng(rng))
    total = 0
    for bits, c in counts.items():
        parity = sum(int(bits[q]) for q in support) % 2
        total += c * (1 - 2 * parity)
    return total / shots


@app.cell(hide_code=True)
def _():
    checks.run(2, expectation=expectation, estimate_pauli=estimate_pauli, qsim=qsim)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def parameter_shift(f, theta):
    """stage 3: the gradient of f at theta (any shape) by the parameter-shift rule, (f(t + pi/2 e_i) - f(t - pi/2 e_i)) / 2."""
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in np.ndindex(theta.shape):
        shifted = theta.copy()
        shifted[i] += np.pi / 2
        plus = f(shifted)
        shifted[i] -= np.pi
        minus = f(shifted)
        grad[i] = (plus - minus) / 2
    return grad


@app.cell(hide_code=True)
def _():
    checks.run(3, parameter_shift=parameter_shift, expectation=expectation)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_classifier(X, y, *, layers=2, steps=50, lr=0.1, reupload=False, rng=None):
    """stage 4: minimize mean((f(theta, x_i) - y_i)**2), y in {-1, +1}, with Adam on parameter-shift gradients; returns (theta, losses)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if not set(np.unique(y)) <= {-1.0, 1.0}:
        raise ValueError("labels must be -1 or +1")
    gen = as_rng(rng)
    theta = gen.uniform(0, 2 * np.pi, size=(layers, X.shape[1], 2))
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def outputs(t):
        return np.array([expectation(t, x, reupload=reupload) for x in X])

    losses = []
    for step in range(1, steps + 1):
        f = outputs(theta)
        losses.append(float(np.mean((f - y) ** 2)))
        resid = 2 * (f - y) / len(y)  # dL/df_i
        # one parameter-shift gradient for all samples at once: d f_i / d theta_j, contracted with resid
        grad = parameter_shift(lambda t: float(resid @ outputs(t)), theta)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        theta = theta - lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
    losses.append(float(np.mean((outputs(theta) - y) ** 2)))
    return theta, losses


@app.function
def classify(theta, X, *, reupload=False):
    """stage 4: predicted labels in {-1, +1}: the sign of f(theta, x) (+1 at exactly 0)."""
    f = np.array([expectation(theta, x, reupload=reupload) for x in np.asarray(X, dtype=float)])
    return np.where(f >= 0, 1, -1)


@app.cell(hide_code=True)
def _():
    checks.run(4, train_classifier=train_classifier, classify=classify, expectation=expectation)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def gradient_samples(n, layers, n_samples, *, observable, rng=None):
    """stage 5: d<O>/d theta[0, 0, 0] (no data, x = 0) at n_samples random theta ~ U(0, 2 pi)^(layers, n, 2); array (n_samples,)."""
    gen = as_rng(rng)
    x = np.zeros(n)
    out = np.empty(n_samples)
    for s in range(n_samples):
        theta = gen.uniform(0, 2 * np.pi, size=(layers, n, 2))
        plus, minus = theta.copy(), theta.copy()
        plus[0, 0, 0] += np.pi / 2
        minus[0, 0, 0] -= np.pi / 2
        out[s] = (expectation(plus, x, observable) - expectation(minus, x, observable)) / 2
    return out


@app.cell(hide_code=True)
def _():
    checks.run(5, gradient_samples=gradient_samples, expectation=expectation)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. topic 32 reuses your
    simulator, not this notebook, so nothing here is imported later.
    """)
    return


if __name__ == "__main__":
    app.run()
