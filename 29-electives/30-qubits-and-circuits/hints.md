# qubits & circuits · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`ket`: `psi = np.zeros(2 ** len(bits), dtype=np.complex128); psi[int(bits, 2)] = 1`. validate first with `set(bits) <= {"0", "1"}`. `gate`: a dict from names to nested lists, converted with `np.array(..., dtype=np.complex128)`; `T`'s corner is `np.exp(1j * np.pi / 4)`. `rot`: use the closed form $\cos\frac\theta2 I - i\sin\frac\theta2 P$ with $P$ from `gate(axis.upper())`.

</details>

## stage 2

<details><summary>hint</summary>

reshape the state to `(2,) * n` so axis $q$ is qubit $q$, and $U$ to `(2,) * (2k)` (outputs first, then inputs). the gate is the contraction $\psi'[\dots b'_t \dots] = \sum_{b_t}U[b'_t, b_t]\,\psi[\dots b_t \dots]$ over the target axes: `np.tensordot` computes it, but puts the $k$ new axes at the front, so they have to be moved back to the target positions before you flatten.

</details>

## stage 3

<details><summary>hint</summary>

`run_circuit` is a loop: `psi = ket("0" * n)` (or a complex copy of `state`), then `psi = apply_gate(psi, U, targets)` for each op. `circuit_unitary`: `np.stack([run_circuit(ops, n, e) for e in np.eye(2 ** n, dtype=complex)], axis=1)`; `axis=1` makes each result a column.

</details>

## stage 4

<details><summary>hint</summary>

`sample_counts`: `rng.multinomial(shots, p)` gives a count per basis index; turn index `i` into a key with `format(i, f"0{n}b")` and skip zeros. `measure_qubit`: reshape to `(2,) * n`; `np.take(psi, 1, axis=q)` is the slice where qubit `q` is 1, so `p1` is its squared norm. draw `bit = int(rng.random() < p1)`, zero the other slice on a copy (index with a tuple that has `1 - bit` at position `q`), flatten and divide by the norm.

</details>

## stage 5

<details><summary>hint</summary>

`expval`: `phi = state`; for each qubit `q` whose character isn't `'I'`, `phi = apply_gate(phi, gate(p), [q])`; return `float(np.vdot(state, phi).real)`. `reduced_density_matrix`: `rest = [q for q in range(n) if q not in keep]`, `M = np.transpose(state.reshape((2,) * n), list(keep) + rest).reshape(2 ** len(keep), -1)`, `rho = M @ M.conj().T`. entropy: `w = np.linalg.eigvalsh(rho)`, keep `w > 1e-12`, return `-np.sum(w * np.log2(w))`.

</details>
