# quantum kernels · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`z = scale * np.asarray(x, float)`; per rep: `[(qsim.gate("H"), (q,)) for q in range(n)]`, then `[(qsim.rot("z", 2 * z[q]), (q,)) ...]`, then a double loop `for i in range(n): for j in range(i + 1, n):` appending CNOT$(i, j)$, `qsim.rot("z", 2 * (np.pi - z[i]) * (np.pi - z[j]))` on `(j,)`, and CNOT$(i, j)$ again. `inverse_ops`: `[(np.asarray(U).conj().T, tuple(t)) for U, t in reversed(list(ops))]`.

</details>

## stage 2

<details><summary>hint</summary>

`feature_states`: `np.array([qsim.run_circuit(fm(x), X.shape[1]) for x in X])`. `fidelity_kernel`: with `A` and `B` the state matrices, `np.abs(A.conj() @ B.T) ** 2`. for `Y=None`, use `B = A`, then symmetrize with `(K + K.T) / 2` and `np.fill_diagonal(K, 1.0)` so tiny rounding errors don't break exact symmetry.

</details>

## stage 3

<details><summary>hint</summary>

`overlap_ops` is one line: `list(fm(x)) + inverse_ops(fm(y))`. `estimate_kernel`: `K = fidelity_kernel(X, feature_map=feature_map)`, `iu = np.triu_indices(m, 1)`, `est = np.eye(m)`, `est[iu] = gen.binomial(shots, np.clip(K[iu], 0, 1)) / shots`, then `est.T[iu] = est[iu]` mirrors it (the transpose is a view).

</details>

## stage 4

<details><summary>hint</summary>

alignment: `A, B = center_gram(K1), center_gram(K2)`, return `np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B))`. geometric difference: normalize each kernel by `m / np.trace(K)`, `S = psd_sqrt(K_q)`, `M = S @ np.linalg.solve(K_c + reg * np.eye(m), S)`, return `np.sqrt(np.linalg.eigvalsh((M + M.T) / 2)[-1])` (symmetrize first: `solve` leaves rounding asymmetry).

</details>

## stage 5

<details><summary>hint</summary>

reuse the stage 4 construction of `S` and `M`, then `w, V = np.linalg.eigh((M + M.T) / 2)` and `v = V[:, -1]` (eigh sorts ascending, and its eigenvectors are already unit length). `y = S @ v`, flip it if `y.sum() < 0`, and `labels = np.where(y > np.median(y), 1, -1)`.

</details>
