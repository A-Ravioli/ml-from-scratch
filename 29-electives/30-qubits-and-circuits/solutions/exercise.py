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
    # qubits & circuits · build (reference solution)

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
    if not isinstance(bits, str) or len(bits) == 0 or set(bits) - {"0", "1"}:
        raise ValueError(f"bits must be a non-empty string of '0' and '1', got {bits!r}")
    psi = np.zeros(2 ** len(bits), dtype=np.complex128)
    psi[int(bits, 2)] = 1.0
    return psi


@app.function
def gate(name):
    """stage 1: the matrix of a named gate: I, X, Y, Z, H, S, T (2x2) or CNOT, CZ, SWAP (4x4, first qubit is the control)."""
    s = 1 / np.sqrt(2)
    table = {
        "I": lambda: [[1, 0], [0, 1]],
        "X": lambda: [[0, 1], [1, 0]],
        "Y": lambda: [[0, -1j], [1j, 0]],
        "Z": lambda: [[1, 0], [0, -1]],
        "H": lambda: [[s, s], [s, -s]],
        "S": lambda: [[1, 0], [0, 1j]],
        "T": lambda: [[1, 0], [0, np.exp(1j * np.pi / 4)]],
        "CNOT": lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        "CZ": lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        "SWAP": lambda: [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    }
    if name not in table:
        raise ValueError(f"unknown gate {name!r}; expected one of {sorted(table)}")
    return np.array(table[name](), dtype=np.complex128)


@app.function
def rot(axis, theta):
    """stage 1: the rotation exp(-i theta P / 2) about axis 'x', 'y' or 'z' (P the matching Pauli), complex128 (2, 2)."""
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=np.complex128)


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
    state = np.asarray(state)
    n = int(round(np.log2(state.size)))
    if state.ndim != 1 or 2 ** n != state.size:
        raise ValueError(f"state must be a vector of length 2**n, got shape {state.shape}")
    targets = tuple(int(t) for t in targets)
    k = len(targets)
    if k == 0 or len(set(targets)) != k or min(targets) < 0 or max(targets) >= n:
        raise ValueError(f"targets must be distinct qubits in 0..{n - 1}, got {targets}")
    U = np.asarray(U)
    if U.shape != (2 ** k, 2 ** k):
        raise ValueError(f"a gate on {k} qubit(s) must be ({2 ** k}, {2 ** k}), got {U.shape}")
    if k == 1:  # fast path: view the state as (2**t, 2, rest) and multiply along the middle axis
        t = targets[0]
        out = np.matmul(U, state.reshape(2 ** t, 2, -1))
        return out.reshape(-1).astype(np.complex128, copy=False)
    psi = state.reshape((2,) * n)
    Ut = U.reshape((2,) * (2 * k))
    # contract U's input axes with the target axes; the k output axes come first in the result
    out = np.tensordot(Ut, psi, axes=(list(range(k, 2 * k)), list(targets)))
    out = np.moveaxis(out, list(range(k)), list(targets))
    return out.reshape(-1).astype(np.complex128, copy=False)


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
    psi = ket("0" * n) if state is None else np.array(state, dtype=np.complex128)
    if psi.shape != (2 ** n,):
        raise ValueError(f"state must have shape ({2 ** n},) for {n} qubits, got {psi.shape}")
    for U, targets in ops:
        psi = apply_gate(psi, U, targets)
    return psi


@app.function
def circuit_unitary(ops, n):
    """stage 3: the full (2**n, 2**n) unitary of a circuit; column j is the circuit applied to basis state j."""
    cols = np.eye(2 ** n, dtype=np.complex128)
    return np.stack([run_circuit(ops, n, cols[j]) for j in range(2 ** n)], axis=1)


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
    state = np.asarray(state)
    return (state.real ** 2 + state.imag ** 2).astype(float)


@app.function
def sample_counts(state, shots, rng=None):
    """stage 4: measure every qubit `shots` times; dict bitstring (qubit 0 first) -> count, only outcomes seen."""
    gen = as_rng(rng)
    p = probabilities(state)
    n = int(round(np.log2(p.size)))
    counts = gen.multinomial(shots, p / p.sum())
    return {format(i, f"0{n}b"): int(c) for i, c in enumerate(counts) if c > 0}


@app.function
def measure_qubit(state, q, rng=None):
    """stage 4: measure qubit q in the Z basis; returns (bit, post-measurement state renormalized)."""
    gen = as_rng(rng)
    state = np.asarray(state, dtype=np.complex128)
    n = int(round(np.log2(state.size)))
    psi = state.reshape((2,) * n)
    one = np.take(psi, 1, axis=q)
    p1 = float(np.sum(np.abs(one) ** 2) / np.sum(np.abs(psi) ** 2))
    bit = int(gen.random() < p1)
    post = psi.copy()
    idx = [slice(None)] * n
    idx[q] = 1 - bit
    post[tuple(idx)] = 0
    post = post.reshape(-1)
    return bit, post / np.linalg.norm(post)


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
    state = np.asarray(state, dtype=np.complex128)
    n = int(round(np.log2(state.size)))
    if len(pauli) != n or set(pauli) - set("IXYZ"):
        raise ValueError(f"pauli must be a string of {n} characters from 'IXYZ', got {pauli!r}")
    phi = state
    for q, p in enumerate(pauli):
        if p != "I":
            phi = apply_gate(phi, gate(p), [q])
    return float(np.real(np.vdot(state, phi)))


@app.function
def reduced_density_matrix(state, keep):
    """stage 5: trace out every qubit not in `keep`; the (2**k, 2**k) density matrix of the kept qubits, in the order given."""
    state = np.asarray(state, dtype=np.complex128)
    n = int(round(np.log2(state.size)))
    keep = [int(q) for q in keep]
    if len(set(keep)) != len(keep) or any(q < 0 or q >= n for q in keep):
        raise ValueError(f"keep must be distinct qubits in 0..{n - 1}, got {keep}")
    rest = [q for q in range(n) if q not in keep]
    psi = np.transpose(state.reshape((2,) * n), keep + rest).reshape(2 ** len(keep), -1)
    return psi @ psi.conj().T


@app.function
def entanglement_entropy(state, keep):
    """stage 5: the von Neumann entropy (in bits) of the reduced state of qubits `keep`."""
    rho = reduced_density_matrix(state, keep)
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(max(0.0, -np.sum(w * np.log2(w))))


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
