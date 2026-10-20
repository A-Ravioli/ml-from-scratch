# quantum kernels

## why this matters

topic 31 trained circuits by gradient descent and ran into barren plateaus. there's a way to use a quantum computer for learning with no training loop at all. encode each data point as a quantum state $|\phi(x)\rangle$, and let the quantum computer do one thing only: estimate how much two encoded states overlap. those overlaps are a **kernel**, and a kernel is all an SVM needs (ch 09). the quantum computer computes inner products in a feature space of dimension $2^n$, or $4^n$ counting density matrices, and the classical SVM does the rest with a convex optimization that has no plateaus and no local minima.

this proposal (Havlíček et al., 2019; Schuld & Killoran, 2019) is the cleanest version of the quantum ML dream, and it comes with a precise question: is there data on which a quantum kernel beats every classical kernel we could reasonably use? the answer, worked out over the following years, is subtle and sobering. a big feature space isn't an advantage by itself: in fact it's usually a liability, because in a huge space every pair of points looks orthogonal and the kernel matrix turns into the identity (**kernel concentration**). an advantage exists only when the labels happen to line up with the structure of the specific quantum feature map, and there's a computable quantity, the **geometric difference** (Huang et al., 2021), that tells you when that is even possible.

this topic ends the quantum block with a project: you'll build the kernels, construct a dataset on which the quantum kernel wins by a landslide, and then test whether anything about that win survives on real data.

## warm-up: predict first

1. an $n$-qubit feature map sends data into a $2^n$-dimensional space. is a bigger feature space better for an SVM? what happens to the Gram matrix if every pair of distinct points maps to nearly orthogonal states?
2. a feature map applies $R_Y(x_q)$ to each qubit and nothing else. can a classical computer evaluate its kernel quickly?
3. you estimate a kernel entry of true value $0.002$ from 1,000 shots. how many all-zero outcomes do you expect, and what's the relative error?
4. a quantum kernel beats the RBF kernel by 40 points of accuracy on some dataset. does it follow that it will beat it on another dataset, or even with a slightly different bandwidth?
5. how would you construct, on purpose, a dataset on which kernel A beats kernel B?

## prerequisites check

- [ ] this notebook runs on your topic 30 simulator (→ topic 30). if a stage it needs is unfinished, it uses the reference simulator instead and says so in a box at the top; finishing topic 30 switches it back to yours (`MLFS_REFERENCE=qsim` forces the reference)
- [ ] you know what makes a function a valid (PSD) kernel, the kernel trick, and centering (→ ch 09 · topic 03)
- [ ] your `SVC` takes a callable kernel `K(X, Y=None)` (→ ch 09 · topic 04)
- [ ] you know binomial sampling and its variance (→ ch 04, ch 05)
- [ ] you know eigendecompositions and matrix square roots of PSD matrices (→ ch 02)

## core ideas

### idea 1: quantum feature maps

a **quantum feature map** is a circuit $U(x)$ whose gates depend on the data, applied to $|0\dots0\rangle$: $|\phi(x)\rangle = U(x)|0\dots0\rangle$. you met the simplest in topic 31: the **product** (angle) map, $R_Y(x_q)$ on each qubit. it never entangles anything, so $|\phi(x)\rangle$ is a product of single-qubit states and everything about it can be computed qubit by qubit (warm-up 2).

the map Havlíček et al. proposed, which we'll call the **ZZ feature map**, is built to entangle. one repetition is a layer of Hadamards followed by data-dependent phases on single qubits and pairs:

$$U_\Phi(z) = \exp\Big(-i\Big[\sum_iz_iZ_i + \sum_{i<j}(\pi - z_i)(\pi - z_j)Z_iZ_j\Big]\Big)H^{\otimes n}, \qquad |\phi(x)\rangle = U_\Phi(z)^{\text{reps}}|0\dots0\rangle,$$

with $z = s\cdot x$ for a **bandwidth** (scale) $s$ that we'll need in idea 3. the exponential is diagonal in the computational basis, and in gates it's $R_Z(2z_i)$ on each qubit (since $R_Z(2\varphi) = e^{-i\varphi Z}$) and, for each pair, CNOT$(i, j)$, $R_Z(2(\pi - z_i)(\pi - z_j))$ on $j$, CNOT$(i, j)$: the CNOTs compute the parity $b_i\oplus b_j$ into qubit $j$, the rotation applies a phase depending on it, and the second CNOT uncomputes. that sandwich is exactly $e^{-i\varphi Z_iZ_j}$ (on paper 1).

with one repetition the state is a uniform superposition with phases, and its overlaps can be computed classically. with two or more, the Hadamards between diagonal layers make it a circuit family conjectured to be hard to simulate classically at scale. that conjecture is the *only* reason to think the kernel could do something classical kernels can't, so keep it in mind when you read results: a quantum kernel that can be simulated efficiently is just a classical kernel computed expensively.

to estimate overlaps on hardware (idea 3) you also need the **inverse circuit**: $U^\dagger = (U_k\cdots U_1)^\dagger = U_1^\dagger\cdots U_k^\dagger$, the gates in reverse order, each conjugate-transposed.

> **spec.** `zz_feature_map(x, *, reps=2, scale=1.0)`: with $z$ = `scale * x` and $n$ = `len(x)`, returns the ops (for your simulator) of `reps` repetitions of: $H$ on qubits $0..n-1$; $R_Z(2z_q)$ on each qubit $q$; then for each pair $i < j$ in lexicographic order, CNOT$(i, j)$, $R_Z(2(\pi - z_i)(\pi - z_j))$ on $j$, CNOT$(i, j)$. `product_feature_map(x, *, scale=1.0)`: $R_Y(\text{scale}\cdot x_q)$ on each qubit $q$. `inverse_ops(ops)`: a new list, the ops reversed with each matrix replaced by its conjugate transpose; the input is not modified.

> **▶ build: stage 1.** implement `zz_feature_map`, `product_feature_map` and `inverse_ops`.

### idea 2: the fidelity kernel

the natural similarity between two encoded states is their **fidelity**,

$$k(x, y) = |\langle\phi(x)|\phi(y)\rangle|^2.$$

why the square of the modulus, rather than the inner product itself? because the global phase of a state is unphysical (topic 30), so only $|\langle\cdot|\cdot\rangle|$ is meaningful, and because $|\langle\phi(x)|\phi(y)\rangle|^2 = \operatorname{Tr}(\rho(x)\rho(y))$ with $\rho(x) = |\phi(x)\rangle\langle\phi(x)|$: an ordinary inner product between the density matrices, viewed as real vectors of dimension $4^n$. that makes $k$ a valid PSD kernel automatically (on paper 2), with values in $[0, 1]$ and $k(x, x) = 1$.

for the product map the fidelity factorizes over qubits: $|\langle0|R_Y(a)^\dagger R_Y(b)|0\rangle|^2 = \cos^2\frac{b - a}2$, so

$$k_{\text{product}}(x, y) = \prod_q\cos^2\Big(\frac{s(x_q - y_q)}2\Big),$$

a perfectly classical kernel: a product of periodic bumps, close to an RBF kernel for small $s$ (on paper 2). for the ZZ map there's no such shortcut beyond small $n$.

on your simulator, compute every state once (an $m\times2^n$ matrix $A$ of rows $\phi(x_i)$), and the whole Gram matrix is one matrix product: $K = |\bar AB^\top|^2$ elementwise. hand it to your ch 09 SVM as a callable `kernel=lambda X, Y=None: fidelity_kernel(X, Y)`.

> **spec.** `feature_states(X, feature_map=None)`: `X` $(m, n)$; returns the complex $(m, 2^n)$ array whose row $i$ is `run_circuit(feature_map(X[i]), n)`; `feature_map` is a callable $x \mapsto$ ops, default `zz_feature_map` (with its defaults). `fidelity_kernel(X, Y=None, *, feature_map=None)`: $K_{ij} = |\langle\phi(x_i)|\phi(y_j)\rangle|^2$, shape $(m, k)$; with `Y=None`, $(m, m)$, exactly symmetric with an exactly-1 diagonal.

> **🔮 predict:** 40 random points, ZZ map at scale 1. as you go from 2 to 10 qubits, what happens to the typical off-diagonal kernel value?

> **▶ build: stage 2.** implement `feature_states` and `fidelity_kernel`.

### idea 3: estimating kernels with shots, and kernel concentration

on hardware you can't read out states, but you can estimate each fidelity with the **compute–uncompute** circuit: run $U(x)$, then $U(y)^\dagger$, and measure. the probability of seeing all zeros is

$$P(0\dots0) = |\langle0\dots0|U(y)^\dagger U(x)|0\dots0\rangle|^2 = |\langle\phi(y)|\phi(x)\rangle|^2 = k(x, y).$$

with $N$ shots, the number of all-zero outcomes is binomial, so the estimate $\hat k = \text{count}/N$ is unbiased with variance $k(1 - k)/N$. a Gram matrix needs $m(m-1)/2$ such circuits (the diagonal is 1 by definition), which already makes quantum kernels expensive for large datasets.

now the catch. suppose the feature map spreads points around the $2^n$-dimensional space so well that the states look random. for Haar-random states, $\mathbb E|\langle\phi|\psi\rangle|^2 = 2^{-n}$ (on paper 3), with fluctuations of the same order. so off-diagonal kernel entries **concentrate** exponentially around a tiny value (Thanasilp et al., 2022). two consequences:

- even with exact values, a Gram matrix close to the identity makes the SVM memorize: every training point is its own isolated bump, and the prediction on a new point is a tiny number dominated by whichever training point happens to be least orthogonal. you get perfect training accuracy and chance test accuracy.
- with shots it's worse: to resolve differences of size $2^{-n}$ between entries you need the binomial noise $\sqrt{k/N} \approx \sqrt{2^{-n}/N}$ to be smaller than them, i.e. $N \gtrsim 2^n$ shots *per entry* (warm-up 3).

the fix is the **bandwidth**: scale the data down ($s \ll 1$) so the feature map moves states only a little, and kernel values stay in a useful range (Shaydulin & Wild, 2022; Canatar et al., 2022). but notice the tension: the smaller the bandwidth, the closer the map stays to its starting state, and the closer the kernel gets to something simple, low-order, and classically tractable. the same trade-off as topic 31's barren plateaus, wearing a different hat.

> **spec.** `overlap_ops(x, y, feature_map=None)`: the compute–uncompute circuit, `feature_map(x)` followed by `inverse_ops(feature_map(y))` (default map `zz_feature_map`). `estimate_kernel(X, shots, *, feature_map=None, rng=None)`: an $(m, m)$ shot estimate of the Gram matrix. take the exact fidelities $p$ of the upper triangle in `np.triu_indices(m, 1)` order (computing them from the states is fine: they equal the compute–uncompute probabilities), make **one** call `as_rng(rng).binomial(shots, p)`, divide by `shots`, mirror to the lower triangle, and put 1 on the diagonal.

> **▶ build: stage 3.** implement `overlap_ops` and `estimate_kernel`.

### idea 4: comparing kernels

two tools for asking whether a kernel suits a task, or differs from another kernel in a way that matters.

**alignment.** the centered **kernel alignment** (Cortes, Mohri & Rostamizadeh, 2012) is the cosine between two centered Gram matrices,

$$A(K_1, K_2) = \frac{\langle K_1^c, K_2^c\rangle_F}{\|K_1^c\|_F\,\|K_2^c\|_F}, \qquad K^c = HKH,\ H = I - \tfrac1m\mathbf 1\mathbf 1^\top.$$

with $K_2 = yy^\top$ it measures how well $K_1$'s geometry matches the labels. it's cheap and scale-free, and it's a heuristic: high alignment doesn't guarantee a good SVM (you'll see one counterexample in the lab).

**geometric difference.** Huang et al. (2021) asked a sharper question: is there *any* labelling on which kernel $K_Q$ could beat kernel $K_C$? for kernel ridge regression, the generalization error with kernel $K$ on labels $y$ is controlled by the **model complexity** $s_K(y) = y^\top K^{-1}y$ (with $K$ normalized to trace $m$): smaller is better. they showed $s_C(y) \le g^2\,s_Q(y)$ for every $y$, with

$$g(K_C\|K_Q) = \sqrt{\big\|\sqrt{K_Q}\,K_C^{-1}\sqrt{K_Q}\big\|_\infty},$$

the square root of the largest eigenvalue (on paper 4). so if $g$ is small (close to 1), the classical kernel does essentially as well as the quantum one on **every** possible labelling of these inputs, and no experiment on them can show an advantage. a large $g$ is necessary, not sufficient: it says some labelling favours $K_Q$, not that yours does. in practice $K_C$ is nearly singular, so you regularize: $(K_C + \lambda I)^{-1}$. (that's a simplification: Huang et al.'s regularized version is $\sqrt{\|\sqrt{K_Q}\sqrt{K_C}(K_C + \lambda I)^{-2}\sqrt{K_C}\sqrt{K_Q}\|_\infty}$, so the numbers you compute here aren't comparable with the paper's.)

> **spec.** `kernel_alignment(K1, K2)`: the centered alignment above as a `float` (use `center_gram`); raises `ValueError` if a centered kernel is all zeros. `geometric_difference(K_c, K_q, *, reg=0.0)`: scale both kernels to trace $m$, then return $\sqrt{\lambda_{\max}\big(\sqrt{K_q}(K_c + \text{reg}\,I)^{-1}\sqrt{K_q}\big)}$, with $\sqrt{K_q}$ the symmetric PSD square root (the notebook gives you `psd_sqrt`).

> **▶ build: stage 4.** implement `kernel_alignment` and `geometric_difference`.

### idea 5: a dataset made for the quantum kernel

the bound $s_C(y) \le g^2s_Q(y)$ is tight, and the labels that achieve it are easy to write down: let $v$ be the top unit eigenvector of $M = \sqrt{K_Q}K_C^{-1}\sqrt{K_Q}$ and set $y = \sqrt{K_Q}\,v$. then $s_Q(y) = v^\top v = 1$ and $s_C(y) = v^\top Mv = g^2$ (on paper 4). these are the targets on which the quantum kernel has the biggest possible edge over the classical one, Huang et al.'s **engineered dataset**. binarize at the median to get balanced classes for an SVM.

this is how you should read every claimed quantum-kernel advantage: the method is legitimate (any $K_Q$ with large $g$ against the classical kernels you care about has *some* winning labelling), and it's also the clearest demonstration that an advantage is a property of the pair (labels, kernel), not of the kernel. there are rigorous advantages, too, but they're for problems constructed from cryptographic hardness assumptions, like learning a function based on the discrete logarithm (Liu, Arunachalam & Temme, 2021), rather than for data that occurs in the wild.

> **spec.** `engineered_labels(K_c, K_q, *, reg=0.0)`: scale both kernels to trace $m$; $v$ = the unit eigenvector of the **largest** eigenvalue of $\sqrt{K_q}(K_c + \text{reg}\,I)^{-1}\sqrt{K_q}$; targets $y = \sqrt{K_q}\,v$, sign-flipped if needed so that $\sum_iy_i \ge 0$; labels $+1$ where $y > \operatorname{median}(y)$, else $-1$. returns `(y, labels)`.

> **🔮 predict:** on labels engineered for the ZZ kernel at one bandwidth, how will the *same* circuit do at twice or half that bandwidth?

> **▶ build: stage 5.** implement `engineered_labels`.

## worked example

**the ZZ map on one qubit, one rep.** $H|0\rangle = |+\rangle$, then $R_Z(2z)$: $\frac1{\sqrt2}(e^{-iz}|0\rangle + e^{iz}|1\rangle)$ (no pairs with one qubit). the fidelity of $z$ and $w$ is $|\frac12(e^{i(z-w)} + e^{-i(z-w)})|^2 = \cos^2(z - w)$.

**the product kernel.** $x = (0.4, -1.3)$, $y = (1.0, 0.7)$, scale 1: $\cos^2(0.3)\cos^2(1.0) = 0.9127\times0.2919 = 0.2664$.

**estimating an entry.** true fidelity $0.002$, 1,000 shots: expect 2 all-zero outcomes; the standard deviation is $\sqrt{0.002\cdot0.998\cdot1000} = 1.4$ counts, a relative error of 70% (warm-up 3).

**geometric difference with two points.** $K_C = I$, $K_Q = \begin{pmatrix}1&0.44\\0.44&1\end{pmatrix}$ (both already have trace 2). then $\sqrt{K_Q}I^{-1}\sqrt{K_Q} = K_Q$, whose largest eigenvalue is $1.44$, so $g = 1.2$. the identity kernel memorizes, so any labelling where the two points share a label ($y \propto (1, 1)$, the top eigenvector) favours $K_Q$, but only by a factor $g^2 = 1.44$.

your code must reproduce these.

## on paper

1. **the ZZ sandwich.** show that $\text{CNOT}(i,j)\,(I\otimes R_Z(2\varphi))_j\,\text{CNOT}(i,j) = e^{-i\varphi Z_iZ_j}$ by acting on the basis states $|b_ib_j\rangle$.

<details><summary>solution</summary>

the first CNOT maps $|b_ib_j\rangle \mapsto |b_i, b_i\oplus b_j\rangle$. $R_Z(2\varphi) = \operatorname{diag}(e^{-i\varphi}, e^{i\varphi})$ on qubit $j$ multiplies by $e^{-i\varphi(-1)^{b_i\oplus b_j}}$. the second CNOT restores $|b_ib_j\rangle$. and $Z_iZ_j|b_ib_j\rangle = (-1)^{b_i}(-1)^{b_j}|b_ib_j\rangle = (-1)^{b_i\oplus b_j}|b_ib_j\rangle$, so the product is $e^{-i\varphi Z_iZ_j}$ on every basis state.

</details>

2. **fidelity kernels are kernels; the product kernel.** show that $k(x, y) = \operatorname{Tr}(\rho(x)\rho(y))$, conclude that $k$ is PSD, and derive $k_{\text{product}}$. what does it look like for small $s$?

<details><summary>solution</summary>

$\operatorname{Tr}(|\phi_x\rangle\langle\phi_x|\phi_y\rangle\langle\phi_y|) = \langle\phi_y|\phi_x\rangle\langle\phi_x|\phi_y\rangle = |\langle\phi_x|\phi_y\rangle|^2$. $\operatorname{Tr}(AB) = \langle A, B\rangle_{HS}$ is an inner product on Hermitian matrices (a real vector space of dimension $4^n$), so $k$ is a Gram matrix of feature vectors $\rho(x)$: PSD. for one qubit, $\langle0|R_Y(-a)R_Y(b)|0\rangle = \langle0|R_Y(b - a)|0\rangle = \cos\frac{b-a}2$; tensor products multiply overlaps. for small $s$, $\cos^2\frac u2 = 1 - \frac{u^2}4 + O(u^4) \approx e^{-u^2/4}$, so $k_{\text{product}} \approx \exp(-\frac{s^2}4\|x - y\|^2)$: an RBF kernel with $\gamma = s^2/4$.

</details>

3. **random states are nearly orthogonal.** for a fixed unit vector $|\psi\rangle \in \mathbb C^d$ and $|\phi\rangle$ uniformly random, show $\mathbb E|\langle\phi|\psi\rangle|^2 = 1/d$. how many shots does it take to estimate such a fidelity to 10% relative error for $n = 20$ qubits?

<details><summary>solution</summary>

by symmetry $\mathbb E|\langle\phi|e_i\rangle|^2$ is the same for every basis vector $e_i$, and the $d$ values sum to $\|\phi\|^2 = 1$, so each is $1/d$; rotate $\psi$ to $e_1$. relative error of a binomial estimate is $\sqrt{(1 - p)/(pN)} \approx 1/\sqrt{pN}$; for 10% at $p = 2^{-20}$ that's $N = 100\cdot2^{20} \approx 10^8$ shots, for a single kernel entry.

</details>

4. **the geometric-difference bound.** with $K_C, K_Q$ positive definite, show $y^\top K_C^{-1}y \le g^2\,y^\top K_Q^{-1}y$ for all $y$, with equality for $y = \sqrt{K_Q}v$, $v$ the top eigenvector of $M = \sqrt{K_Q}K_C^{-1}\sqrt{K_Q}$.

<details><summary>solution</summary>

write $u = K_Q^{-1/2}y$. then $y^\top K_C^{-1}y = u^\top\sqrt{K_Q}K_C^{-1}\sqrt{K_Q}u = u^\top Mu \le \lambda_{\max}(M)\,u^\top u = g^2\,y^\top K_Q^{-1}y$, by the Rayleigh quotient. equality holds when $u$ is the top eigenvector, i.e. $y = \sqrt{K_Q}v$. a large ratio means the labels are "simple" for $K_Q$ (small $s_Q$) and "complex" for $K_C$, which by the generalization bounds of kernel ridge regression means $K_C$ needs far more data to learn them.

</details>

## experiment

`lab.py` investigates, with your kernels and your ch 09 `SVC`:

1. kernel concentration: the spread of kernel values as qubits are added, at four bandwidths, and how much of the kernel survives finite shots.
2. **project, part 1:** labels engineered for the ZZ kernel at one bandwidth, on real MNIST inputs: every kernel family, including the same circuit at other bandwidths, tries to learn them.
3. **project, part 2:** real labels (MNIST 3 vs 5 and California housing): the ZZ kernel, the product kernel, RBF and linear, each tuned by cross-validation.

## check your understanding

1. (warm-up 1.) why is a bigger feature space not automatically better for a kernel method?

<details><summary>answer</summary>

the kernel only sees inner products. in a huge space, a feature map that spreads points out makes all inner products small and similar: the Gram matrix approaches the identity, which carries no information about which points are alike. generalization needs a kernel whose notion of similarity matches the labels, and a bigger space gives more ways to be wrong.

</details>

2. why can't the bandwidth fix concentration for free?

<details><summary>answer</summary>

shrinking $s$ keeps $|\phi(x)\rangle$ near $U_\Phi(0)^{\text{reps}}|0\rangle$, so the kernel varies smoothly and only through low-order terms in $s\,x$. in that regime the quantum kernel is well approximated by a classical one (e.g. by expanding in $s$), so the regime where it's useful is also the regime where it's least likely to be quantum in any essential way.

</details>

3. the product kernel uses a quantum circuit. is it a quantum kernel?

<details><summary>answer</summary>

it's computed by a circuit, but the closed form $\prod_q\cos^2(s(x_q - y_q)/2)$ costs $O(n)$ classically. a kernel is only interesting as a *quantum* kernel if it can't be computed efficiently classically, which requires entanglement (and more: shallow or weakly entangling maps can often be simulated too).

</details>

4. what does $g(K_C\|K_Q) \approx 1$ tell you, and what does $g = 40$ tell you?

<details><summary>answer</summary>

$g \approx 1$: the classical kernel's model complexity is at most about the quantum one's for every labelling of these inputs, so no experiment on these points can show a meaningful quantum advantage. $g = 40$: there exist labellings where the classical kernel's complexity is up to $1600\times$ larger; whether your actual labels are among them is a separate question, which only the experiment answers.

</details>

5. why does `estimate_kernel` fix the diagonal at 1 and estimate each pair once?

<details><summary>answer</summary>

$k(x, x) = 1$ exactly, so spending shots on it adds noise for nothing. and estimating $(i, j)$ and $(j, i)$ separately would give a non-symmetric matrix, which an SVM's dual doesn't expect; mirroring one estimate keeps it symmetric (though shot noise can still make it slightly indefinite, which is why practical pipelines sometimes project the estimate back onto the PSD cone).

</details>

6. you report "our quantum kernel beats RBF by 5% on dataset D". what else must you report for that to mean anything?

<details><summary>answer</summary>

that the RBF's bandwidth (and $C$) were tuned as carefully as the quantum kernel's, on the training set only; uncertainty over several splits; whether the quantum kernel is classically simulable at the size used (if it is, the comparison is between two classical kernels); its bandwidth and whether it's concentrated; and ideally the geometric difference against the tuned classical kernels.

</details>

## common bugs

- **forgetting to conjugate** the first state in the overlap. symptom: the kernel isn't symmetric or exceeds 1; the product-kernel closed form fails. it's `A.conj() @ B.T`.
- **inner product instead of fidelity.** symptom: complex Gram matrices. take `np.abs(...) ** 2`.
- **reversing the inverse without daggering (or daggering without reversing).** symptom: `overlap_ops(x, x)` doesn't return to $|0\dots0\rangle$.
- **scaling $x$ inside $(\pi - x_i)$ but not in $R_Z(2x_i)$, or vice versa.** symptom: the closed-form check fails only when `scale != 1`. scale once, at the start.
- **forgetting the trace normalization** in `geometric_difference`. symptom: $g$ changes when you multiply a kernel by a constant.
- **taking the eigenvector of the smallest eigenvalue.** `np.linalg.eigh` sorts ascending: the top one is the last column.

## going deeper

- Havlíček, Córcoles, Temme, Harrow, Kandala, Chow & Gambetta (2019), "Supervised learning with quantum-enhanced feature spaces", *Nature* 567, 209–212. the ZZ feature map and quantum kernel estimation on hardware. [doi:10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
- Schuld & Killoran (2019), "Quantum machine learning in feature Hilbert spaces", *Physical Review Letters* 122, 040504. quantum states as feature maps, and the kernel view. [doi:10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504)
- Huang, Broughton, Mohseni, Babbush, Boixo, Neven & McClean (2021), "Power of data in quantum machine learning", *Nature Communications* 12, 2631. the geometric difference, engineered datasets, projected quantum kernels. [doi:10.1038/s41467-021-22539-9](https://doi.org/10.1038/s41467-021-22539-9)
- Thanasilp, Wang, Cerezo & Holmes (2022), "Exponential concentration and untrainability in quantum kernel methods". [arXiv:2208.11060](https://arxiv.org/abs/2208.11060)
- Shaydulin & Wild (2022), "Importance of kernel bandwidth in quantum machine learning", *Physical Review A* 106, 042407. [arXiv:2111.05451](https://arxiv.org/abs/2111.05451)
- Canatar, Peters, Pehlevan, Wild & Shaydulin (2022), "Bandwidth enables generalization in quantum kernel models". [arXiv:2206.06686](https://arxiv.org/abs/2206.06686)
- Liu, Arunachalam & Temme (2021), "A rigorous and robust quantum speed-up in supervised machine learning", *Nature Physics* 17, 1013–1017. a provable advantage, on a problem built from the discrete logarithm. [arXiv:2010.02174](https://arxiv.org/abs/2010.02174)
- Cortes, Mohri & Rostamizadeh (2012), "Algorithms for learning kernels based on centered alignment", *JMLR* 13, 795–828. [jmlr.org/papers/v13/cortes12a.html](https://www.jmlr.org/papers/v13/cortes12a.html)
- Schuld (2021), "Supervised quantum machine learning models are kernel methods". why topic 31's variational models are kernel machines too. [arXiv:2101.11020](https://arxiv.org/abs/2101.11020)
- Schuld & Petruccione (2021), *Machine Learning with Quantum Computers*, 2nd ed., Springer, chapter 6 (quantum models as kernel methods). [doi:10.1007/978-3-030-83098-4](https://doi.org/10.1007/978-3-030-83098-4)

## connections

- ← builds on: your simulator (topic 30), angle encoding and the expressivity of encodings (topic 31), PSD kernels, centering, the RBF kernel and your SMO `SVC` (ch 09 · topics 03–04), cross-validation (ch 07), PCA (ch 11).
- → used in: the chapter 29 capstone (*elective portfolio*) asks for a short report on this block's project against a classical baseline. the concentration phenomena here and in topic 31 are the quantum versions of the curse of dimensionality you met with kNN and RBF kernels in ch 09.
