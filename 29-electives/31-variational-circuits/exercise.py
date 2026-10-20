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
    # variational circuits · build

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
    raise NotImplementedError("stage1: implement ansatz_ops")


@app.function
def encoding_ops(x):
    """stage 1: angle encoding: RY(x[q]) on qubit q, for a feature vector x (n,)."""
    raise NotImplementedError("stage1: implement encoding_ops")


@app.function
def model_ops(theta, x, *, reupload=False):
    """stage 1: encoding then the whole ansatz; with reupload=True, the encoding before every layer instead."""
    raise NotImplementedError("stage1: implement model_ops")


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
    raise NotImplementedError("stage2: implement expectation")


@app.function
def estimate_pauli(state, pauli, shots, rng=None):
    """stage 2: estimate <state| P |state> from `shots` measurements, rotating each X/Y qubit into the Z basis first."""
    raise NotImplementedError("stage2: implement estimate_pauli")


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
    raise NotImplementedError("stage3: implement parameter_shift")


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
    raise NotImplementedError("stage4: implement train_classifier")


@app.function
def classify(theta, X, *, reupload=False):
    """stage 4: predicted labels in {-1, +1}: the sign of f(theta, x) (+1 at exactly 0)."""
    raise NotImplementedError("stage4: implement classify")


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
    raise NotImplementedError("stage5: implement gradient_samples")


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
