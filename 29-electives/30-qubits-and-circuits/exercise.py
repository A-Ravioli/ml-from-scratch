import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # qubits & circuits · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. the convention throughout is **big-endian**: in a ket
    $|q_0q_1\dots q_{n-1}\rangle$ qubit 0 is the leftmost bit, the *most* significant bit of
    the state-vector index.

    | stage | you build |
    |---|---|
    | 1 | `ket`, `gate`, `rot`: basis states, the named gates, rotations |
    | 2 | `apply_gate`: a $k$-qubit gate on chosen qubits of an $n$-qubit state, without a $2^n\times2^n$ matrix |
    | 3 | `run_circuit`, `circuit_unitary`: circuits as lists of gates |
    | 4 | `probabilities`, `sample_counts`, `measure_qubit`: the Born rule and collapse |
    | 5 | `expval`, `reduced_density_matrix`, `entanglement_entropy`: observables and entanglement |

    topics 31 and 32 run their circuits on this simulator. you can also run the checks from a
    terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def ket(bits):
    """stage 1: the computational basis state |bits> (a '0'/'1' string, qubit 0 first) as a complex128 vector (2**n,)."""
    raise NotImplementedError("stage1: implement ket")


@app.function
def gate(name):
    """stage 1: the matrix of a named gate: I, X, Y, Z, H, S, T (2x2) or CNOT, CZ, SWAP (4x4, first qubit is the control)."""
    raise NotImplementedError("stage1: implement gate")


@app.function
def rot(axis, theta):
    """stage 1: the rotation exp(-i theta P / 2) about axis 'x', 'y' or 'z' (P the matching Pauli), complex128 (2, 2)."""
    raise NotImplementedError("stage1: implement rot")


@app.cell(hide_code=True)
def _():
    checks.run(1, ket=ket, gate=gate, rot=rot)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def apply_gate(state, U, targets):
    """stage 2: apply the k-qubit unitary U (2**k, 2**k) to qubits `targets` (targets[0] = U's most significant bit); returns a new state."""
    raise NotImplementedError("stage2: implement apply_gate")


@app.cell(hide_code=True)
def _():
    checks.run(2, apply_gate=apply_gate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def run_circuit(ops, n, state=None):
    """stage 3: apply ops, a sequence of (U, targets), in order to `state` (default |0...0>) of n qubits; returns the final state."""
    raise NotImplementedError("stage3: implement run_circuit")


@app.function
def circuit_unitary(ops, n):
    """stage 3: the full (2**n, 2**n) unitary of a circuit; column j is the circuit applied to basis state j."""
    raise NotImplementedError("stage3: implement circuit_unitary")


@app.cell(hide_code=True)
def _():
    checks.run(3, run_circuit=run_circuit, circuit_unitary=circuit_unitary, ket=ket, gate=gate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def probabilities(state):
    """stage 4: the Born-rule probabilities |amplitude|**2 of every basis state, float (2**n,)."""
    raise NotImplementedError("stage4: implement probabilities")


@app.function
def sample_counts(state, shots, rng=None):
    """stage 4: measure every qubit `shots` times; dict bitstring (qubit 0 first) -> count, only outcomes seen."""
    raise NotImplementedError("stage4: implement sample_counts")


@app.function
def measure_qubit(state, q, rng=None):
    """stage 4: measure qubit q in the Z basis; returns (bit, post-measurement state renormalized)."""
    raise NotImplementedError("stage4: implement measure_qubit")


@app.cell(hide_code=True)
def _():
    checks.run(4, probabilities=probabilities, sample_counts=sample_counts, measure_qubit=measure_qubit, ket=ket)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def expval(state, pauli):
    """stage 5: <state| P |state> for a Pauli string like 'ZIX' (character k acts on qubit k), a float."""
    raise NotImplementedError("stage5: implement expval")


@app.function
def reduced_density_matrix(state, keep):
    """stage 5: trace out every qubit not in `keep`; the (2**k, 2**k) density matrix of the kept qubits, in the order given."""
    raise NotImplementedError("stage5: implement reduced_density_matrix")


@app.function
def entanglement_entropy(state, keep):
    """stage 5: the von Neumann entropy (in bits) of the reduced state of qubits `keep`."""
    raise NotImplementedError("stage5: implement entanglement_entropy")


@app.cell(hide_code=True)
def _():
    checks.run(
        5,
        expval=expval,
        reduced_density_matrix=reduced_density_matrix,
        entanglement_entropy=entanglement_entropy,
        ket=ket,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green you have a working state-vector simulator. topics 31 and 32
    load it from this notebook. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
