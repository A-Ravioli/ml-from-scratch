"""Checks for exercise.py (qubits & circuits).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Gates are checked
against their textbook matrices, scipy's matrix exponential and gate identities; `apply_gate` is
checked against a full matrix built index by index (an independent oracle that encodes the
big-endian convention); circuits against Bell and GHZ states; sampling with chi-square and
binomial tests whose false-alarm rate is far below 1 in 1,000; entanglement against closed forms.
"""
import tracemalloc

import numpy as np
from scipy.linalg import expm
from scipy.stats import binomtest, chisquare

S2 = 1 / np.sqrt(2)
PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
H = S2 * np.array([[1, 1], [1, -1]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def _basis(bits):
    v = np.zeros(2 ** len(bits), dtype=complex)
    v[int(bits, 2)] = 1
    return v


def _random_state(n, rng):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def _random_unitary(k, rng):
    a = rng.normal(size=(2 ** k, 2 ** k)) + 1j * rng.normal(size=(2 ** k, 2 ** k))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _full_matrix(U, targets, n):
    """the 2**n x 2**n matrix of U on `targets`, built entry by entry (big-endian: qubit 0 = top bit)."""
    k = len(targets)
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rest = [q for q in range(n) if q not in targets]

    def bit(i, q):
        return (i >> (n - 1 - q)) & 1

    for i in range(2 ** n):
        for j in range(2 ** n):
            if any(bit(i, q) != bit(j, q) for q in rest):
                continue
            a = sum(bit(i, t) << (k - 1 - m) for m, t in enumerate(targets))
            b = sum(bit(j, t) << (k - 1 - m) for m, t in enumerate(targets))
            M[i, j] = U[a, b]
    return M


def _close(a, b, atol=1e-10):
    return np.allclose(np.asarray(a), np.asarray(b), atol=atol)


def _ops(nb, spec):
    """('H', 0), ('CNOT', 0, 1) ... -> [(matrix, targets)] using the reference matrices."""
    mats = {"H": H, "X": PAULI["X"], "Z": PAULI["Z"], "CNOT": CNOT, "I": PAULI["I"]}
    return [(mats[name], tuple(qs)) for name, *qs in spec]


# --- stage 1 -------------------------------------------------------------------------------------

def test_stage1_ket_is_big_endian(nb):
    psi = nb.ket("01")
    assert psi.shape == (4,) and np.iscomplexobj(psi), (
        f"ket('01') should be a complex vector of shape (4,), got {getattr(psi, 'dtype', type(psi))} "
        f"{getattr(psi, 'shape', None)}")
    assert _close(psi, [0, 1, 0, 0]), (
        f"ket('01') should be (0, 1, 0, 0): index int('01', 2) = 1, qubit 0 is the leftmost, most "
        f"significant bit. got {np.round(psi, 3)}")
    assert _close(nb.ket("110"), _basis("110")), "ket('110') should have its 1 at index 6 (= 0b110)."
    assert _close(nb.ket("0"), [1, 0]), "ket('0') is (1, 0)."


def test_stage1_ket_rejects_bad_input(nb):
    for bad in ("", "012", "ab"):
        try:
            nb.ket(bad)
        except ValueError:
            continue
        raise AssertionError(f"ket({bad!r}) should raise ValueError: bits must be a non-empty 0/1 string")


def test_stage1_named_gates(nb):
    for name, M in {**PAULI, "H": H, "CNOT": CNOT}.items():
        got = nb.gate(name)
        assert _close(got, M), f"gate({name!r}) should be\n{M}\ngot\n{np.round(got, 3)}"
    assert _close(nb.gate("S"), np.diag([1, 1j])), "S = diag(1, i)."
    assert _close(nb.gate("T"), np.diag([1, np.exp(1j * np.pi / 4)])), "T = diag(1, e^{i pi/4})."
    assert _close(nb.gate("CZ"), np.diag([1, 1, 1, -1])), "CZ = diag(1, 1, 1, -1)."
    assert _close(nb.gate("SWAP"), np.eye(4)[[0, 2, 1, 3]]), "SWAP exchanges |01> and |10>."
    try:
        nb.gate("FOO")
    except ValueError:
        pass
    else:
        raise AssertionError("gate('FOO') should raise ValueError")


def test_stage1_gate_identities(nb):
    g = nb.gate
    for name in ("I", "X", "Y", "Z", "H", "S", "T", "CNOT", "CZ", "SWAP"):
        M = g(name)
        assert _close(M.conj().T @ M, np.eye(len(M))), f"gate({name!r}) is not unitary: U^dagger U != I"
    assert _close(g("H") @ g("X") @ g("H"), g("Z")), "H X H should equal Z (H swaps the X and Z bases)."
    assert _close(g("S") @ g("S"), g("Z")), "S^2 should equal Z."
    assert _close(g("T") @ g("T"), g("S")), "T^2 should equal S."
    assert _close(g("X") @ g("Y"), 1j * g("Z")), "XY should equal iZ: check the signs in Y = [[0, -i], [i, 0]]."


def test_stage1_rotations_match_matrix_exponential(nb):
    rng = np.random.default_rng(0)
    for axis, P in zip("xyz", ("X", "Y", "Z")):
        for theta in rng.uniform(-7, 7, size=5):
            want = expm(-0.5j * theta * PAULI[P])
            got = nb.rot(axis, theta)
            assert _close(got, want), (
                f"rot({axis!r}, {theta:.3f}) should be exp(-i theta {P} / 2) =\n{np.round(want, 4)}\ngot\n"
                f"{np.round(got, 4)}. note the 1/2: rot(axis, 2 pi) = -I, not I.")
    assert _close(nb.rot("x", np.pi), -1j * PAULI["X"]), "rot('x', pi) should be -iX."
    assert _close(nb.rot("y", np.pi / 2) @ _basis("0"), [S2, S2]), "rot('y', pi/2)|0> should be |+>."
    try:
        nb.rot("w", 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("rot('w', 0.1) should raise ValueError")


# --- stage 2 -------------------------------------------------------------------------------------

def test_stage2_qubit_ordering(nb):
    X = PAULI["X"]
    got = nb.apply_gate(_basis("00"), X, [0])
    assert _close(got, _basis("10")), (
        f"X on qubit 0 of |00> should give |10> = (0, 0, 1, 0) (qubit 0 is the MOST significant bit), got "
        f"{np.round(got, 3)}. if you got |01>, you used little-endian order.")
    got = nb.apply_gate(_basis("10"), CNOT, [0, 1])
    assert _close(got, _basis("11")), f"CNOT with control 0, target 1 maps |10> to |11>, got {np.round(got, 3)}"
    got = nb.apply_gate(_basis("10"), CNOT, [1, 0])
    assert _close(got, _basis("10")), (
        f"CNOT with control 1 (targets=[1, 0]) leaves |10> alone (qubit 1 is 0), got {np.round(got, 3)}. "
        "targets[0] must play the role of U's most significant bit.")
    got = nb.apply_gate(_basis("001"), CNOT, [2, 0])
    assert _close(got, _basis("101")), f"CNOT control 2 -> target 0 maps |001> to |101>, got {np.round(got, 3)}"


def test_stage2_matches_full_matrix(nb):
    rng = np.random.default_rng(1)
    for n, targets in [(1, [0]), (3, [1]), (3, [2, 0]), (4, [1, 3]), (4, [3, 0, 2])]:
        U = _random_unitary(len(targets), rng)
        psi = _random_state(n, rng)
        want = _full_matrix(U, targets, n) @ psi
        got = nb.apply_gate(psi, U, targets)
        assert np.shape(got) == (2 ** n,), f"apply_gate should return shape ({2 ** n},), got {np.shape(got)}"
        assert _close(got, want), (
            f"apply_gate on n={n}, targets={targets} disagrees with the full matrix by "
            f"{np.max(np.abs(got - want)):.2e}. check which axis each target maps to, and that the output axes "
            "go back to the target positions (np.moveaxis) before you flatten.")


def test_stage2_preserves_input_and_norm(nb):
    rng = np.random.default_rng(2)
    psi = _random_state(3, rng)
    before = psi.copy()
    out = nb.apply_gate(psi, _random_unitary(2, rng), [0, 2])
    assert np.array_equal(psi, before), "apply_gate must not modify the input state: return a new array"
    assert np.isclose(np.linalg.norm(out), 1), "a unitary gate keeps the state normalized"


def test_stage2_rejects_bad_targets(nb):
    psi = _basis("000")
    for U, t in [(CNOT, [1, 1]), (PAULI["X"], [3]), (CNOT, [0])]:
        try:
            nb.apply_gate(psi, U, t)
        except ValueError:
            continue
        raise AssertionError(f"apply_gate with a {U.shape} gate on targets {t} of 3 qubits should raise ValueError")


def test_stage2_no_full_matrix(nb):
    n = 10
    psi = _basis("0" * n)
    nb.apply_gate(psi, H, [3])  # warm-up
    tracemalloc.start()
    nb.apply_gate(psi, H, [3])
    nb.apply_gate(psi, CNOT, [7, 2])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    state_bytes = psi.nbytes
    assert peak < 20 * state_bytes + 100_000, (
        f"applying one gate to a 10-qubit state (a {state_bytes / 1e3:.0f} kB vector) peaked at "
        f"{peak / 1e6:.1f} MB. a 2**n x 2**n matrix (np.kron with identities) would need 16 MB at n = 10 and "
        "16 GB at n = 20. reshape the state to (2,)*n and contract only the target axes.")


# --- stage 3 -------------------------------------------------------------------------------------

def test_stage3_bell_and_ghz(nb):
    bell = nb.run_circuit(_ops(nb, [("H", 0), ("CNOT", 0, 1)]), 2)
    assert _close(bell, [S2, 0, 0, S2]), f"H on 0 then CNOT(0, 1) should give (|00> + |11>)/sqrt2, got {np.round(bell, 3)}"
    ghz = nb.run_circuit(_ops(nb, [("H", 0), ("CNOT", 0, 1), ("CNOT", 1, 2)]), 3)
    want = np.zeros(8, dtype=complex)
    want[[0, 7]] = S2
    assert _close(ghz, want), f"the GHZ circuit should give (|000> + |111>)/sqrt2, got {np.round(ghz, 3)}"
    start = _basis("01")
    out = nb.run_circuit(_ops(nb, [("X", 0)]), 2, start)
    assert _close(out, _basis("11")), "run_circuit should start from the given state when one is passed"
    assert _close(start, _basis("01")), "run_circuit must not modify the state passed in"
    assert _close(nb.run_circuit([], 3), _basis("000")), "an empty circuit on 3 qubits returns |000>"


def test_stage3_circuit_identities(nb):
    u = nb.circuit_unitary
    assert _close(u(_ops(nb, [("H", 0), ("X", 0), ("H", 0)]), 1), PAULI["Z"]), "the circuit H, X, H should equal Z"
    Hs = [("H", 1)]
    cz = [(np.diag([1, 1, 1, -1]).astype(complex), (0, 1))]
    got = u(_ops(nb, Hs) + cz + _ops(nb, Hs), 2)
    assert _close(got, CNOT), f"H on the target, CZ, H on the target should equal CNOT, got\n{np.round(got, 3)}"
    swap = u(_ops(nb, [("CNOT", 0, 1), ("CNOT", 1, 0), ("CNOT", 0, 1)]), 2)
    assert _close(swap, np.eye(4)[[0, 2, 1, 3]]), f"three alternating CNOTs should equal SWAP, got\n{np.round(swap, 3)}"
    rev = u(_ops(nb, [("H", 0), ("H", 1), ("CNOT", 0, 1), ("H", 0), ("H", 1)]), 2)
    assert _close(rev, _full_matrix(CNOT, [1, 0], 2)), "conjugating CNOT by H on both qubits reverses control and target"


def test_stage3_unitary_columns_and_unitarity(nb):
    rng = np.random.default_rng(3)
    ops = [(_random_unitary(1, rng), (q,)) for q in range(3)] + [(_random_unitary(2, rng), (2, 0))]
    U = nb.circuit_unitary(ops, 3)
    assert np.shape(U) == (8, 8), f"circuit_unitary on 3 qubits should be (8, 8), got {np.shape(U)}"
    assert _close(U.conj().T @ U, np.eye(8)), "a circuit of unitaries is unitary"
    for j in (0, 5):
        col = nb.run_circuit(ops, 3, _basis(format(j, "03b")))
        assert _close(U[:, j], col), f"column {j} of circuit_unitary should be the circuit applied to basis state {j}"
    want = _full_matrix(ops[3][0], [2, 0], 3)
    for M, t in ops[2::-1]:
        want = want @ _full_matrix(M, list(t), 3)
    assert _close(U, want), "circuit_unitary should equal U_last ... U_first (the first gate acts first)"


# --- stage 4 -------------------------------------------------------------------------------------

def test_stage4_probabilities(nb):
    rng = np.random.default_rng(4)
    psi = _random_state(3, rng)
    p = nb.probabilities(psi)
    assert np.shape(p) == (8,) and not np.iscomplexobj(p), "probabilities should be a real (2**n,) array"
    assert _close(p, np.abs(psi) ** 2), "probabilities are |amplitude|**2 (amplitude times its conjugate)"
    assert np.isclose(np.sum(p), 1), "probabilities of a normalized state sum to 1"


def test_stage4_sample_counts_format(nb):
    counts = nb.sample_counts(_basis("011"), 500, rng=0)
    assert counts == {"011": 500}, f"measuring |011> 500 times gives {{'011': 500}}, got {counts}"
    bell = np.array([S2, 0, 0, S2], dtype=complex)
    c = nb.sample_counts(bell, 1000, rng=1)
    assert set(c) <= {"00", "11"} and sum(c.values()) == 1000, (
        f"a Bell state only ever gives 00 or 11 and the counts sum to shots, got {c}")
    assert nb.sample_counts(bell, 50, rng=7) == nb.sample_counts(bell, 50, rng=7), "the same seed should give the same counts"


def test_stage4_sample_counts_distribution(nb):
    rng = np.random.default_rng(5)
    psi = _random_state(3, rng)
    p = np.abs(psi) ** 2
    shots = 4000
    counts = nb.sample_counts(psi, shots, rng=np.random.default_rng(6))
    obs = np.array([counts.get(format(i, "03b"), 0) for i in range(8)])
    pval = chisquare(obs, p * shots).pvalue
    assert pval > 1e-4, (
        f"4000 samples from a known 3-qubit state fail a chi-square test (p = {pval:.1e}). observed {obs}, "
        f"expected {np.round(p * shots).astype(int)}. are the bitstrings in big-endian order (index i -> "
        "format(i, '03b'))?")


def test_stage4_measure_qubit_collapses(nb):
    bell = np.array([S2, 0, 0, S2], dtype=complex)
    ones = 0
    for seed in range(400):
        bit, post = nb.measure_qubit(bell, 0, rng=seed)
        assert bit in (0, 1), f"measure_qubit should return bit 0 or 1, got {bit}"
        want = _basis(f"{bit}{bit}")
        assert _close(post, want), (
            f"after measuring qubit 0 of a Bell pair and seeing {bit}, the state is |{bit}{bit}>, got {np.round(post, 3)}."
            " zero the amplitudes that disagree with the outcome and renormalize.")
        ones += bit
    pval = binomtest(ones, 400, 0.5).pvalue
    assert pval > 1e-4, f"qubit 0 of a Bell pair should read 1 half the time; got {ones}/400 (p = {pval:.1e})"


def test_stage4_measure_qubit_probability_and_spectator(nb):
    # cos(a)|0> + sin(a)|1> on qubit 1, |+> on qubit 0: P(qubit 1 = 1) = sin^2 a
    a = 0.6
    psi = np.kron([S2, S2], [np.cos(a), np.sin(a)]).astype(complex)
    ones = sum(nb.measure_qubit(psi, 1, rng=s)[0] for s in range(1000))
    pval = binomtest(ones, 1000, np.sin(a) ** 2).pvalue
    assert pval > 1e-4, (
        f"P(qubit 1 = 1) should be sin^2(0.6) = {np.sin(a) ** 2:.3f}; got {ones}/1000 (p = {pval:.1e}). "
        "check that you sum |amplitude|^2 over the slice where qubit q is 1, not where qubit n-1-q is.")
    bit, post = nb.measure_qubit(psi, 1, rng=0)
    want = np.kron([S2, S2], [1 - bit, bit])
    assert _close(post, want), f"measuring qubit 1 should leave qubit 0 in |+>, got {np.round(post, 3)}"
    assert np.isclose(np.linalg.norm(post), 1), "the post-measurement state must be renormalized"


# --- stage 5 -------------------------------------------------------------------------------------

def test_stage5_bell_correlations(nb):
    bell = np.array([S2, 0, 0, S2], dtype=complex)
    for p, want in [("ZZ", 1), ("XX", 1), ("YY", -1), ("ZI", 0), ("IX", 0), ("II", 1)]:
        got = nb.expval(bell, p)
        assert isinstance(got, float), f"expval should return a Python float, got {type(got).__name__}"
        assert np.isclose(got, want, atol=1e-10), f"<Bell|{p}|Bell> should be {want}, got {got}"
    for bad in ("Z", "ZZZ", "ZA"):
        try:
            nb.expval(bell, bad)
        except ValueError:
            continue
        raise AssertionError(f"expval(bell, {bad!r}) should raise ValueError: a 2-qubit state needs a 2-character "
                             "string over 'IXYZ' (a short string would silently act as identity on the rest)")


def test_stage5_expval_matches_matrix(nb):
    rng = np.random.default_rng(7)
    psi = _random_state(3, rng)
    for p in ("ZIX", "YYZ", "IXI", "XYZ"):
        M = np.kron(np.kron(PAULI[p[0]], PAULI[p[1]]), PAULI[p[2]])
        want = np.real(psi.conj() @ M @ psi)
        got = nb.expval(psi, p)
        assert np.isclose(got, want, atol=1e-10), (
            f"expval(psi, {p!r}) should be {want:.6f} (= psi^dagger (P0 kron P1 kron P2) psi), got {got:.6f}. "
            "character k must act on qubit k.")


def test_stage5_reduced_density_matrix(nb):
    bell = np.array([S2, 0, 0, S2], dtype=complex)
    rho = nb.reduced_density_matrix(bell, [0])
    assert _close(rho, np.eye(2) / 2), f"either half of a Bell pair is maximally mixed, I/2; got\n{np.round(rho, 3)}"
    rng = np.random.default_rng(8)
    a, b, c = _random_state(1, rng), _random_state(1, rng), _random_state(1, rng)
    prod = np.kron(np.kron(a, b), c)
    assert _close(nb.reduced_density_matrix(prod, [1]), np.outer(b, b.conj())), (
        "for a product state a (x) b (x) c, keeping qubit 1 gives the pure state |b><b|")
    assert _close(nb.reduced_density_matrix(prod, [2, 0]), np.outer(np.kron(c, a), np.kron(c, a).conj())), (
        "keep=[2, 0] should order the kept qubits as given: |c a><c a|")
    psi = _random_state(4, rng)
    rho = nb.reduced_density_matrix(psi, [3, 1])
    assert np.shape(rho) == (4, 4), f"keeping 2 of 4 qubits gives a (4, 4) matrix, got {np.shape(rho)}"
    assert _close(rho, rho.conj().T) and np.isclose(np.trace(rho).real, 1), "a density matrix is Hermitian with trace 1"
    assert np.linalg.eigvalsh(rho).min() > -1e-10, "a density matrix is positive semidefinite"
    assert _close(nb.reduced_density_matrix(psi, [0, 1, 2, 3]), np.outer(psi, psi.conj())), (
        "keeping every qubit gives |psi><psi|")


def test_stage5_entanglement_entropy(nb):
    bell = np.array([S2, 0, 0, S2], dtype=complex)
    assert np.isclose(nb.entanglement_entropy(bell, [1]), 1.0), "a Bell pair carries exactly 1 bit of entanglement"
    assert np.isclose(nb.entanglement_entropy(_basis("010"), [0]), 0, atol=1e-9), "a basis state has zero entanglement"
    ghz = np.zeros(8, dtype=complex)
    ghz[[0, 7]] = S2
    assert np.isclose(nb.entanglement_entropy(ghz, [0, 2]), 1.0), "any part of a GHZ state has entropy 1 bit"
    for a in (0.3, 1.1):
        psi = np.array([np.cos(a), 0, 0, np.sin(a)], dtype=complex)
        p = np.cos(a) ** 2
        want = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        got = nb.entanglement_entropy(psi, [0])
        assert np.isclose(got, want, atol=1e-9), (
            f"cos(a)|00> + sin(a)|11> with a = {a} has entropy H(cos^2 a) = {want:.4f} bits, got {got:.4f}. "
            "use log base 2, and skip zero eigenvalues (0 log 0 = 0).")
