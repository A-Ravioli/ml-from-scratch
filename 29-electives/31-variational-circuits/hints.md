# variational circuits · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

two nested loops per layer: `for q in range(n): ops += [(qsim.rot("y", theta[l, q, 0]), (q,)), (qsim.rot("z", theta[l, q, 1]), (q,))]`, then `for q in range(n - 1): ops.append((qsim.gate("CNOT"), (q, q + 1)))`. for re-uploading, `ansatz_ops(theta[l:l + 1])` is the ansatz for layer `l` alone, so `model_ops` is a loop that appends `encoding_ops(x) + ansatz_ops(theta[l:l + 1])`.

</details>

## stage 2

<details><summary>hint</summary>

`expectation`: `psi = qsim.run_circuit(model_ops(theta, x, reupload=reupload), n)` then `qsim.expval(psi, observable)`. `estimate_pauli`: apply `qsim.gate("H")` on each `'X'` qubit and `qsim.gate("H") @ qsim.gate("S").conj().T` on each `'Y'` qubit, call `counts = qsim.sample_counts(psi, shots, rng)` once, and sum `count * (-1) ** (number of 1s among the support qubits of bits)` over the dictionary.

</details>

## stage 3

<details><summary>hint</summary>

loop over every index of `theta` (`np.ndindex(theta.shape)`), evaluate `f` at a *copy* of `theta` shifted by $+\frac\pi2$ and by $-\frac\pi2$ in that one entry, and store $\big(f(\theta + \frac\pi2e_i) - f(\theta - \frac\pi2e_i)\big)/2$. never shift `theta` itself in place.

</details>

## stage 4

<details><summary>hint</summary>

inside the loop: `f = np.array([expectation(theta, x, reupload=reupload) for x in X])`, record the loss, set `r = 2 * (f - y) / len(y)`, and get the gradient in one call: `parameter_shift(lambda t: float(r @ outputs(t)), theta)` where `outputs(t)` recomputes `f` at `t`. then the Adam update with `t = step` starting from 1: `m = b1 m + (1 - b1) g`, `v = b2 v + (1 - b2) g²`, `theta -= lr * (m / (1 - b1**t)) / (sqrt(v / (1 - b2**t)) + eps)`. don't forget the final loss after the loop.

</details>

## stage 5

<details><summary>hint</summary>

per sample: `theta = gen.uniform(0, 2 * np.pi, size=(layers, n, 2))`, make two copies with `[0, 0, 0]` shifted by $\pm\frac\pi2$, and take half the difference of `expectation(copy, np.zeros(n), observable)`. you only need the one component, so don't call `parameter_shift` on the whole array (that's $2P$ circuits instead of 2).

</details>
