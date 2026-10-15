# qubits & circuits

## why this matters

quantum machine learning papers are full of claims: exponential speedups, feature spaces "too large to compute classically", models that train better on quantum hardware. you can't judge any of them without knowing what a quantum computer actually does, and the fastest way to know is to build one in software. a simulator for $n$ qubits is about a hundred lines of NumPy: a vector of $2^n$ complex numbers, some small unitary matrices, and a rule for turning amplitudes into random measurement outcomes.

building it teaches you two things at once. first, what quantum computation is: linear algebra on complex vectors, plus a measurement rule that is stranger than it looks (experiment 2 of the lab shows two qubits winning a game that no classical strategy can win more than 75% of the time). second, why it's hard to fake classically: the vector doubles with every qubit, so the same hundred lines that simulate 20 qubits in milliseconds would need more memory than exists on Earth for 60. both facts come back in topics 31 and 32, where you'll train circuits as models and use them as kernels, and ask whether any of it beats a classical method.

the bugs in this topic are almost all about **qubit ordering**. which bit of the index is qubit 0? which qubit of a two-qubit gate is the control? textbooks and software libraries disagree, and a simulator that mixes conventions produces states that look plausible and are wrong. we fix one convention and test it relentlessly.

## warm-up: predict first

1. a single qubit is in state $(|0\rangle + |1\rangle)/\sqrt2$. you measure it 1,000 times (preparing it fresh each time). what do you see? then you apply a Hadamard before measuring. what now?
2. how many real numbers does it take to describe a general state of 10 qubits? of 50?
3. two qubits are in state $(|00\rangle + |11\rangle)/\sqrt2$. you measure the first one and get 1. what's the state of the second?
4. a gate acts on qubit 3 of a 10-qubit register. do you need a $1024\times1024$ matrix to apply it?
5. can measuring your half of an entangled pair send a message to whoever holds the other half?

## prerequisites check

- [ ] you can multiply complex numbers, conjugate them, and take $|z|^2 = z\bar z$ (→ ch 01)
- [ ] you know unitary matrices, eigenvalues, and the Kronecker product (→ ch 02 · topics 01–02 and 05)
- [ ] you can reshape, transpose and `np.tensordot` multi-dimensional arrays (→ ch 00 · topic 02)
- [ ] you know categorical and multinomial distributions, and the entropy of a distribution (→ ch 04, ch 05 · topic 05)

## core ideas

### idea 1: qubits, kets and gates

a classical bit is 0 or 1. a **qubit** is a unit vector in $\mathbb C^2$,

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix}\alpha\\\beta\end{pmatrix}, \qquad |\alpha|^2 + |\beta|^2 = 1,$$

where $|0\rangle = (1, 0)^\top$ and $|1\rangle = (0, 1)^\top$ are the **computational basis**. the "ket" $|\cdot\rangle$ is just a column vector; the "bra" $\langle\psi|$ is its conjugate transpose, so $\langle\phi|\psi\rangle$ is the complex inner product. the amplitudes $\alpha, \beta$ aren't probabilities, but they determine them (idea 4): measuring gives 0 with probability $|\alpha|^2$. a global phase is unobservable: $|\psi\rangle$ and $e^{i\varphi}|\psi\rangle$ are the same physical state.

$n$ qubits live in the tensor product $\mathbb C^{2}\otimes\dots\otimes\mathbb C^2 = \mathbb C^{2^n}$, with basis states $|b_0b_1\dots b_{n-1}\rangle = |b_0\rangle\otimes|b_1\rangle\otimes\dots$ for every bit string. that's $2^n$ complex amplitudes (warm-up 2: $2\cdot2^{10} - 2$ real parameters for 10 qubits after normalization and global phase; about $2\cdot10^{15}$ for 50).

**the ordering convention.** we use the **big-endian** convention of Nielsen & Chuang: qubit 0 is the *leftmost* symbol in the ket and the *most significant* bit of the index. so $|01\rangle$ (qubit 0 is 0, qubit 1 is 1) is basis vector number $0\cdot2 + 1 = 1$, and $|10\rangle$ is number 2. equivalently, $|a\rangle\otimes|b\rangle$ = `np.kron(a, b)`, with qubit 0 on the left. Qiskit uses the opposite (little-endian: qubit 0 is the rightmost bit). neither is wrong; mixing them is.

the evolution of a closed quantum system is a **unitary** matrix $U$ ($U^\dagger U = I$): it preserves the norm, so it maps states to states. a **gate** is a unitary on a few qubits. the ones you'll use:

$$X = \begin{pmatrix}0&1\\1&0\end{pmatrix},\ Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix},\ Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix},\ H = \tfrac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\ S = \begin{pmatrix}1&0\\0&i\end{pmatrix},\ T = \begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}.$$

$X$ is the quantum NOT. $Z$ flips the sign of $|1\rangle$. the **Hadamard** $H$ maps $|0\rangle \mapsto |+\rangle = (|0\rangle + |1\rangle)/\sqrt2$ and $|1\rangle \mapsto |-\rangle = (|0\rangle - |1\rangle)/\sqrt2$: it swaps the $Z$ basis with the $X$ basis, which is why $HXH = Z$. the **rotations** are the continuous family

$$R_P(\theta) = e^{-i\theta P/2} = \cos\tfrac\theta2\,I - i\sin\tfrac\theta2\,P, \qquad P\in\{X, Y, Z\},$$

where the closed form holds because $P^2 = I$. note the $\theta/2$: $R_P(2\pi) = -I$, a global phase, and you need $4\pi$ to come back exactly. $R_Y(\theta)|0\rangle = \cos\frac\theta2|0\rangle + \sin\frac\theta2|1\rangle$ has real amplitudes, which makes $R_Y$ the workhorse of topic 31.

two-qubit gates are $4\times4$. the **CNOT** flips its second qubit (the target) when the first (the control) is 1; **CZ** flips the sign of $|11\rangle$; **SWAP** exchanges the two qubits. in our convention the matrix rows and columns are ordered $|00\rangle, |01\rangle, |10\rangle, |11\rangle$ with the *first* qubit of the gate as the high bit:

$$\text{CNOT} = \begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{pmatrix}, \qquad \text{CZ} = \operatorname{diag}(1, 1, 1, -1).$$

> **spec.** `ket(bits)`: `bits` a non-empty string of `'0'`/`'1'`, qubit 0 first; returns the complex128 vector of length $2^n$ with a 1 at index `int(bits, 2)`. anything else raises `ValueError`. `gate(name)`: the complex128 matrix of `'I'`, `'X'`, `'Y'`, `'Z'`, `'H'`, `'S'`, `'T'` ($2\times2$) or `'CNOT'`, `'CZ'`, `'SWAP'` ($4\times4$, first qubit = control = high bit); unknown names raise `ValueError`. `rot(axis, theta)`: $e^{-i\theta P/2}$ for `axis` in `'x'`, `'y'`, `'z'`, complex128 $(2, 2)$; another axis raises `ValueError`.

> **▶ build: stage 1.** implement `ket`, `gate` and `rot`.

### idea 2: applying a gate without the big matrix

the textbook way to apply a single-qubit gate $U$ to qubit $q$ of $n$ is the Kronecker product $I\otimes\dots\otimes U\otimes\dots\otimes I$, a $2^n\times2^n$ matrix. at $n = 20$ that's $2^{40}$ entries, 16 TB of complex128. but almost all of them are zero: $U$ only mixes amplitudes that differ in bit $q$.

the trick is to stop thinking of the state as a vector. reshape it into an $n$-dimensional array of shape $(2, 2, \dots, 2)$: `psi[b0, b1, ..., b_{n-1}]` is the amplitude of $|b_0b_1\dots b_{n-1}\rangle$. with C-order reshaping, axis $q$ is exactly qubit $q$ in the big-endian convention (the first axis varies slowest, like the most significant bit). a gate on qubit $q$ is then a matrix multiplication along axis $q$ alone:

$$\psi'[\dots, b_q', \dots] = \sum_{b_q}U[b_q', b_q]\,\psi[\dots, b_q, \dots].$$

for a $k$-qubit gate on qubits $t_0, \dots, t_{k-1}$, reshape $U$ from $(2^k, 2^k)$ to $(2, \dots, 2)$ with $2k$ axes (the first $k$ are output bits, the last $k$ input bits, $t_0$ most significant in each group) and contract its input axes with state axes $t_0, \dots, t_{k-1}$. `np.tensordot` does the contraction and puts the $k$ new axes *first*; `np.moveaxis` puts them back at positions $t_0, \dots, t_{k-1}$. cost: $O(2^n\cdot 2^k)$ time and $O(2^n)$ memory, against $O(4^n)$ for the big matrix.

the order of `targets` matters. `apply_gate(psi, CNOT, [1, 0])` uses qubit 1 as the control, because `targets[0]` plays the role of the gate's first (high) qubit.

> **spec.** `apply_gate(state, U, targets)`: `state` complex $(2^n,)$, `U` $(2^k, 2^k)$, `targets` a sequence of $k$ distinct qubits in $0..n-1$ with `targets[0]` as $U$'s most significant bit. returns a **new** complex vector $(2^n,)$, never building a $2^n\times2^n$ matrix, and leaves `state` untouched. repeated or out-of-range targets, or a $U$ of the wrong size, raise `ValueError`.

> **🔮 predict:** apply $X$ to qubit 0 of $|00\rangle$. which index of the 4-vector holds the 1 afterwards? and for CNOT with `targets=[1, 0]` acting on $|10\rangle$?

> **▶ build: stage 2.** implement `apply_gate`.

### idea 3: circuits

a **circuit** is a sequence of gates applied in order, starting (by convention) from $|0\dots0\rangle$. we represent it as a plain list of `(U, targets)` pairs. the first gate in the list acts first, so the circuit's overall unitary is $U_{\text{last}}\cdots U_2U_1$: matrix products read right to left, circuits read left to right.

two circuits are the same if they have the same unitary (up to a global phase), and identities between small circuits are both useful and the best tests a simulator can have:

- $HXH = Z$ and $HZH = X$;
- $\text{CNOT} = (I\otimes H)\,\text{CZ}\,(I\otimes H)$: a CZ is a CNOT seen in the $X$ basis of the target;
- three alternating CNOTs make a SWAP;
- $H$ on both qubits before and after a CNOT reverses its control and target.

the circuit $H$ on qubit 0 followed by CNOT(0, 1) maps $|00\rangle$ to the **Bell state** $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt2$. one more CNOT from qubit 1 to qubit 2 gives the **GHZ state** $(|000\rangle + |111\rangle)/\sqrt2$.

to get a circuit's full unitary for testing, you don't need the Kronecker product either: column $j$ of $U$ is $U|j\rangle$, so run the circuit on each basis state. that's $2^n$ runs, fine for the handful of qubits where you'd want the matrix anyway.

> **spec.** `run_circuit(ops, n, state=None)`: `ops` a sequence of `(U, targets)`, applied in order to `state` (default $|0\dots0\rangle$; a given state must have shape $(2^n,)$ and is not modified); returns the final state. `circuit_unitary(ops, n)`: the $(2^n, 2^n)$ matrix whose column $j$ is the circuit applied to basis state $j$.

> **▶ build: stage 3.** implement `run_circuit` and `circuit_unitary`.

### idea 4: measurement

you can't read amplitudes off a quantum computer. you can only **measure**, and measuring all $n$ qubits in the computational basis returns bit string $b$ with probability

$$p(b) = |\langle b|\psi\rangle|^2 = |\psi_b|^2 \qquad \text{(the Born rule)}.$$

the outcome is random, and the state afterwards is $|b\rangle$: the superposition is gone. so everything you learn from a quantum computer comes from **shots**: repeat the preparation and measurement many times, and estimate probabilities from counts. a probability estimated from $N$ shots has standard error $\sqrt{p(1-p)/N}$, which matters enormously in topics 31 and 32, where every number a quantum model produces is such an estimate. sampling $N$ shots at once is one multinomial draw with probabilities $p$.

measuring a single qubit $q$ is the partial version. the probability of reading 1 is the total weight of the amplitudes whose bit $q$ is 1,

$$p_q(1) = \sum_{b:\,b_q = 1}|\psi_b|^2,$$

and the state **collapses**: amplitudes inconsistent with the outcome are set to zero, and the rest are renormalized. the other qubits are left in whatever state is consistent with what you saw. measure qubit 0 of $|\Phi^+\rangle$: you get 0 or 1 with probability $\frac12$, and the pair collapses to $|00\rangle$ or $|11\rangle$, so qubit 1 now agrees with qubit 0 with certainty (warm-up 3).

measuring in another basis is a rotation followed by a $Z$-basis measurement: to measure in the $X$ basis, apply $H$ first. the lab measures along any direction in the $x$–$z$ plane by applying $R_Y(-\theta)$ first.

> **spec.** `probabilities(state)`: real $(2^n,)$ array $|\psi_b|^2$. `sample_counts(state, shots, rng=None)`: one multinomial draw of `shots` outcomes; returns a `dict` from bit strings (length $n$, qubit 0 first) to counts, containing only outcomes that occurred. `measure_qubit(state, q, rng=None)`: draws the outcome of measuring qubit $q$ with probability $p_q(1)$, returns `(bit, post_state)` with the collapsed state renormalized. `rng` is `None`, a seed, or a `Generator` (use `as_rng`).

> **🔮 predict:** you prepare $|\Phi^+\rangle$, apply $H$ to qubit 0 only, then measure both qubits. which of the four outcomes can you see, and with what probabilities?

> **▶ build: stage 4.** implement `probabilities`, `sample_counts` and `measure_qubit`.

### idea 5: observables and entanglement

an **observable** is a Hermitian matrix $O$; its expectation in state $|\psi\rangle$ is $\langle O\rangle = \langle\psi|O|\psi\rangle$, the average of the measured eigenvalues over many shots. the observables quantum ML uses most are **Pauli strings** such as $Z\otimes I\otimes X$, written `'ZIX'`, with eigenvalues $\pm1$. you can compute their expectations without a big matrix: apply each non-identity Pauli as a gate, then take the inner product with the original state. on hardware, $\langle Z_0\rangle$ is estimated from shots as $P(b_0 = 0) - P(b_0 = 1)$.

a state is a **product state** if it factors as $|a\rangle\otimes|b\rangle$, and **entangled** otherwise. $|\Phi^+\rangle$ is entangled: no pair of single-qubit states multiplies to it (on paper 2). to quantify entanglement you need the state of one part on its own. when a subsystem $A$ is entangled with $B$, $A$ alone isn't described by a vector but by a **density matrix**, the **reduced state**

$$\rho_A = \operatorname{Tr}_B\,|\psi\rangle\langle\psi|.$$

computationally: reshape $\psi$ into a matrix $M$ with the kept qubits' bits as rows and the rest as columns (transpose the axes so the kept qubits come first, in the order you want); then $\rho_A = MM^\dagger$. it's Hermitian, positive semidefinite, with trace 1. for a product state it's a pure projector $|a\rangle\langle a|$; for either half of a Bell pair it's $I/2$, the maximally mixed state: individually, each qubit is a fair coin.

the **entanglement entropy** of $A$ is the von Neumann entropy of its reduced state, the Shannon entropy of its eigenvalues,

$$S(\rho_A) = -\sum_i\lambda_i\log_2\lambda_i,$$

zero for product states and at most $\min(|A|, |B|)$ bits. a Bell pair carries exactly 1 bit. (for a pure global state, $S(\rho_A) = S(\rho_B)$: the eigenvalues are the squared singular values of $M$, on paper 3.)

> **spec.** `expval(state, pauli)`: `pauli` a string of length $n$ over `'IXYZ'` (character $k$ acts on qubit $k$; anything else raises `ValueError`); returns $\langle\psi|P|\psi\rangle$ as a Python `float`. `reduced_density_matrix(state, keep)`: the $(2^k, 2^k)$ reduced state of the qubits in `keep`, ordered as given (`keep[0]` is the high bit); invalid `keep` raises `ValueError`. `entanglement_entropy(state, keep)`: $S(\rho_{\text{keep}})$ in bits, ignoring eigenvalues below $10^{-12}$.

> **▶ build: stage 5.** implement `expval`, `reduced_density_matrix` and `entanglement_entropy`.

## worked example

**ordering.** $X$ on qubit 0 of $|00\rangle$ gives $|10\rangle$: the 4-vector $(0, 0, 1, 0)$, index 2. CNOT with `targets=[1, 0]` on $|10\rangle$: the control is qubit 1, which is 0, so nothing happens. CNOT with `targets=[2, 0]` on $|001\rangle$: control qubit 2 is 1, so qubit 0 flips: $|101\rangle$.

**the Bell circuit, step by step.** start with $|00\rangle = (1, 0, 0, 0)$. $H$ on qubit 0: $\frac1{\sqrt2}(|0\rangle + |1\rangle)\otimes|0\rangle = \frac1{\sqrt2}(1, 0, 1, 0)$. CNOT(0, 1) swaps the amplitudes of $|10\rangle$ and $|11\rangle$: $\frac1{\sqrt2}(1, 0, 0, 1) = |\Phi^+\rangle$.

**measuring it.** $p = (\frac12, 0, 0, \frac12)$: only 00 and 11. measuring qubit 0 gives 1 with probability $\frac12$, leaving $|11\rangle$.

**observables.** $Z\otimes Z$ is $+1$ on $|00\rangle$ and $|11\rangle$, so $\langle ZZ\rangle = 1$. $X\otimes X$ maps $|00\rangle\leftrightarrow|11\rangle$, so it fixes $|\Phi^+\rangle$: $\langle XX\rangle = 1$. $Y\otimes Y|00\rangle = (i)(i)|11\rangle = -|11\rangle$ and $Y\otimes Y|11\rangle = -|00\rangle$, so $\langle YY\rangle = -1$. $\langle ZI\rangle = \frac12(1) + \frac12(-1) = 0$.

**entanglement.** reshape $|\Phi^+\rangle$ to $M = \frac1{\sqrt2}\begin{pmatrix}1&0\\0&1\end{pmatrix}$ (rows: qubit 0, columns: qubit 1). $\rho_0 = MM^\dagger = I/2$, eigenvalues $\frac12, \frac12$, entropy $1$ bit. for $\cos a|00\rangle + \sin a|11\rangle$, $\rho_0 = \operatorname{diag}(\cos^2a, \sin^2a)$ and the entropy is the binary entropy $H_2(\cos^2a)$: $a = 0.3$ gives $0.427$ bits.

your code must reproduce all of these.

## on paper

1. **rotations.** show that $e^{-i\theta P/2} = \cos\frac\theta2 I - i\sin\frac\theta2 P$ for any $P$ with $P^2 = I$, and compute $R_Y(\theta)|0\rangle$ and $R_X(\pi)$.

<details><summary>solution</summary>

expand the exponential: $e^{-i\theta P/2} = \sum_k\frac{(-i\theta/2)^k}{k!}P^k$. since $P^2 = I$, even powers give $I$ and odd powers give $P$: $\sum_{k\text{ even}}\frac{(-i\theta/2)^k}{k!} = \cos\frac\theta2$ and $\sum_{k\text{ odd}}\frac{(-i\theta/2)^k}{k!} = -i\sin\frac\theta2$. so $R_Y(\theta) = \begin{pmatrix}\cos\frac\theta2 & -\sin\frac\theta2\\\sin\frac\theta2&\cos\frac\theta2\end{pmatrix}$ (the $-i\cdot Y$ entries are real), $R_Y(\theta)|0\rangle = (\cos\frac\theta2, \sin\frac\theta2)$, and $R_X(\pi) = -iX$: $X$ up to a global phase.

</details>

2. **the Bell state is entangled.** show that there are no $(a_0, a_1), (b_0, b_1)$ with $(a_0, a_1)\otimes(b_0, b_1) = \frac1{\sqrt2}(1, 0, 0, 1)$. then give a test for whether a two-qubit state is a product state in terms of its $2\times2$ matrix $M$.

<details><summary>solution</summary>

the product is $(a_0b_0, a_0b_1, a_1b_0, a_1b_1)$. we need $a_0b_1 = 0$, so $a_0 = 0$ or $b_1 = 0$; but $a_0 = 0$ kills $a_0b_0 = \frac1{\sqrt2}$ and $b_1 = 0$ kills $a_1b_1 = \frac1{\sqrt2}$. contradiction. in general $M = \begin{pmatrix}\psi_{00}&\psi_{01}\\\psi_{10}&\psi_{11}\end{pmatrix}$ equals $ab^\top$ for a product state, so the state is a product iff $M$ has rank 1, iff $\det M = 0$. here $\det M = \frac12 \ne 0$.

</details>

3. **Schmidt decomposition.** with $M$ the $2^{|A|}\times2^{|B|}$ reshaping of $|\psi\rangle$ and SVD $M = U\Sigma V^\dagger$, show that $\rho_A = U\Sigma^2U^\dagger$, so the eigenvalues of $\rho_A$ are the squared singular values $\sigma_i^2$, and $S(\rho_A) = S(\rho_B)$.

<details><summary>solution</summary>

$\rho_A = MM^\dagger = U\Sigma V^\dagger V\Sigma U^\dagger = U\Sigma^2U^\dagger$, and $\rho_B = (M^\top)(M^\top)^\dagger = \overline{V}\Sigma^2V^\top$ has the same nonzero eigenvalues. writing $|\psi\rangle = \sum_i\sigma_i|u_i\rangle|\bar v_i\rangle$ (the **Schmidt form**) shows the number of nonzero $\sigma_i$, the Schmidt rank, is at most $\min(2^{|A|}, 2^{|B|})$, hence $S \le \min(|A|, |B|)$ bits.

</details>

4. **CNOT from CZ.** verify $(I\otimes H)\,\text{CZ}\,(I\otimes H) = \text{CNOT}$ by acting on the four basis states.

<details><summary>solution</summary>

if qubit 0 is 0, CZ does nothing and $H\cdot H = I$ on qubit 1: identity, like CNOT. if qubit 0 is 1, CZ applies $Z$ to qubit 1, so qubit 1 sees $HZH = X$: a flip, like CNOT.

</details>

## experiment

`lab.py` uses your simulator to:

1. measure an entangled pair: what one side's measurement does to the other's statistics, in two bases.
2. play the CHSH game, and compare the Bell pair with every classical strategy.
3. watch random circuits build up entanglement between the two halves of 8 qubits.
4. time your `apply_gate` as the number of qubits grows.

## check your understanding

1. why is a global phase unobservable, but a *relative* phase (the sign in $|+\rangle$ vs $|-\rangle$) observable?

<details><summary>answer</summary>

every probability is $|\langle b|U|\psi\rangle|^2$; multiplying $|\psi\rangle$ by $e^{i\varphi}$ multiplies every amplitude by the same unit number, which drops out of $|\cdot|^2$. a relative phase changes how amplitudes interfere *after* another gate: $|+\rangle$ and $|-\rangle$ give identical $Z$ statistics, but $H$ maps them to $|0\rangle$ and $|1\rangle$.

</details>

2. (warm-up 1.) what do you see measuring $|+\rangle$, with and without a Hadamard first?

<details><summary>answer</summary>

without: about 500 zeros and 500 ones, a fair coin. with $H$ first: $H|+\rangle = |0\rangle$, so 0 every time. a classical fair coin can't be "un-randomized" by a fixed operation; the superposition is not ignorance about a hidden bit.

</details>

3. why does reshaping with C order make axis $q$ correspond to qubit $q$ in the big-endian convention? what changes in your code if you switch to little-endian?

<details><summary>answer</summary>

in C order the first axis varies slowest, so flat index $= \sum_qb_q2^{n-1-q}$: $b_0$ carries weight $2^{n-1}$, the most significant bit. for little-endian, qubit $q$ is axis $n-1-q$: map each target $t \mapsto n-1-t$ before contracting (and reverse bit strings when printing counts).

</details>

4. in the spec, `measure_qubit` must renormalize. what goes wrong in a sequence of measurements if it doesn't?

<details><summary>answer</summary>

the post-measurement vector then has norm $\sqrt{p(\text{outcome})}$. the next measurement's probabilities $\sum|\psi_b|^2$ no longer sum to 1, so a "probability of 1" computed as a raw sum is too small, and repeated measurements drift towards always reading 0.

</details>

5. (warm-up 5.) the collapse in idea 4 is instantaneous, whatever the distance between the qubits. why can't Alice use it to signal Bob?

<details><summary>answer</summary>

Bob's statistics depend only on his reduced state $\rho_B$, and no operation on Alice's qubit alone (unitary or measurement, averaged over her outcomes) changes $\rho_B = \operatorname{Tr}_A|\psi\rangle\langle\psi|$. Alice's outcomes are correlated with Bob's, but Bob only sees the correlation after Alice tells him her results through an ordinary channel. lab experiment 1 measures this.

</details>

6. why is `entanglement_entropy` at most $\min(|A|, |B|)$ bits, and why does the simulator's cost grow with entanglement only in principle, not in your implementation?

<details><summary>answer</summary>

the Schmidt rank is at most $2^{\min(|A|,|B|)}$ (on paper 3), and the entropy of a distribution over $r$ outcomes is at most $\log_2r$. your state-vector simulator stores all $2^n$ amplitudes regardless; simulators that exploit low entanglement (matrix product states) store something whose size grows with the Schmidt ranks across cuts, so weakly entangled circuits stay cheap at 100 qubits while highly entangled ones don't.

</details>

7. $\langle ZZ\rangle = 1$ for both $|\Phi^+\rangle$ and the classical mixture "50% $|00\rangle$, 50% $|11\rangle$". how could you tell them apart?

<details><summary>answer</summary>

measure in another basis: $\langle XX\rangle = 1$ for $|\Phi^+\rangle$ but $0$ for the mixture (both $|00\rangle$ and $|11\rangle$ give $\langle XX\rangle = 0$). correlations in *several* bases at once are the signature of entanglement, and CHSH (lab experiment 2) turns that into an inequality.

</details>

8. (the predict callout in idea 4.) $H$ on qubit 0 of $|\Phi^+\rangle$, then measure both: which outcomes?

<details><summary>answer</summary>

$(H\otimes I)\frac1{\sqrt2}(|00\rangle + |11\rangle) = \frac12(|00\rangle + |10\rangle + |01\rangle - |11\rangle)$: all four outcomes, each with probability $\frac14$. the $Z$-basis correlation is gone, because measuring qubit 0 after $H$ is measuring it in the $X$ basis, and $|\Phi^+\rangle$'s $X$ results are uncorrelated with qubit 1's $Z$ results (lab experiment 1).

</details>

## common bugs

- **little-endian by accident.** symptom: $X$ on qubit 0 of $|00\rangle$ gives $(0, 1, 0, 0)$, and Bell circuits still work (they're symmetric), so the bug hides until an asymmetric circuit. test with $|01\rangle$ vs $|10\rangle$.
- **forgetting `np.moveaxis` after `np.tensordot`.** symptom: correct for a gate on qubit 0, wrong on any other qubit, because the contracted axes end up at the front.
- **$e^{-i\theta P}$ instead of $e^{-i\theta P/2}$.** symptom: everything has twice the frequency; $R_Y(\pi)|0\rangle$ gives $-|0\rangle$ instead of $|1\rangle$.
- **conjugating twice, or not at all.** symptom: complex expectation values. $\langle\psi|O|\psi\rangle$ is `np.vdot(psi, O_psi)` (which conjugates its first argument); take the real part only after checking the imaginary part is ~0.
- **not renormalizing after collapse.** symptom: sequential measurements drift (check your understanding 4).
- **`np.log` instead of `np.log2`, or including zero eigenvalues.** symptom: a Bell pair has entropy $0.693$, or `nan`.

## going deeper

- Nielsen & Chuang (2010), *Quantum Computation and Quantum Information*, 10th anniversary ed., Cambridge University Press. chapters 1–2 (qubits, gates, measurement, the density operator) and §4.2–4.3 (single-qubit rotations, controlled gates). the convention used here. [doi:10.1017/CBO9780511976667](https://doi.org/10.1017/CBO9780511976667)
- Schuld & Petruccione (2021), *Machine Learning with Quantum Computers*, 2nd ed., Springer. chapter 3 is a quantum computing primer written for ML readers. [doi:10.1007/978-3-030-83098-4](https://doi.org/10.1007/978-3-030-83098-4)
- Clauser, Horne, Shimony & Holt (1969), "Proposed experiment to test local hidden-variable theories", *Physical Review Letters* 23, 880. the CHSH inequality of lab experiment 2. [doi:10.1103/PhysRevLett.23.880](https://doi.org/10.1103/PhysRevLett.23.880)
- Page (1993), "Average entropy of a subsystem", *Physical Review Letters* 71, 1291. the value random states approach in lab experiment 3. [arXiv:gr-qc/9305007](https://arxiv.org/abs/gr-qc/9305007)
- Preskill (2018), "Quantum computing in the NISQ era and beyond", *Quantum* 2, 79. what today's noisy devices can and can't do. [arXiv:1801.00862](https://arxiv.org/abs/1801.00862)

## connections

- ← builds on: complex vectors, unitary matrices, Kronecker products and the SVD (ch 02), multinomial sampling and entropy (ch 04, ch 05), reshaping and contracting arrays (ch 00).
- → used in: topic 31 builds parameterized circuits on this simulator and differentiates them with the parameter-shift rule; topic 32 turns circuits into kernels for your ch 09 SVM. the concentration of random states you see in lab experiment 3 is the root of both barren plateaus (topic 31) and kernel concentration (topic 32).
