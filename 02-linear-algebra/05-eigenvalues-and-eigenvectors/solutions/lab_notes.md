# eigenvalues & eigenvectors · lab notes

## experiment 1: PageRank on a tiny web

**what people usually predict:** `deals` first (4 incoming links), then `home` (3), then `shop` (2), with the one-link pages tied somewhere in the middle and the unlinked spam pages last.

**what happens** (α = 0.85): `deals` 0.264, **`spam1` 0.241**, `home` 0.197, `shop` 0.103, `blog` and `about` 0.072 each, and `spam2`–`spam4` 0.017 each. `spam1`, with a single incoming link, beats `home` with three. the eigenvalue your code finds is 1 to about ten digits.

**why:** PageRank is the solution of $\pi = G\pi$: your importance is the sum of the importance of pages linking to you, each divided by its number of outgoing links. `deals` links *only* to `spam1`, so `spam1` inherits nearly all of `deals`' large rank. meanwhile `home`'s three in-links come from pages that share their rank among several targets. the `deals` ↔ `spam1` pair is a tiny **link farm**: the random surfer, once inside, can only leave by teleporting (probability $1 - \alpha$ per step), so the pair soaks up rank. raise α and the trap gets stronger (at α = 0.99 the pair holds over half of all rank). lower α and everything flattens toward uniform. real search engines fight exactly this structure. note also that the matrix isn't symmetric. power iteration doesn't need symmetry, only a dominant eigenvalue, and the teleportation term guarantees one with $|\lambda_2| \le \alpha$.

## experiment 2: how fast does power iteration converge?

**what people usually predict:** for r = 0.9, a few dozen iterations, and the eigenvalue converging at the same rate as the vector (or slower, "since it's computed from the vector").

**what happens:** the vector error falls like $r^k$: for r = 0.9 it reaches $10^{-8}$ after about $\ln 10^{-8}/\ln 0.9 \approx 175$ iterations. for r = 0.98 it's still at a few percent after 175 iterations. for r = 0.5 it takes about 27. the **eigenvalue error falls twice as fast**, like $r^{2k}$: about 87 iterations to $10^{-8}$ for r = 0.9. on the log plot the dashed lines are exactly twice as steep.

**why:** the component of $v_k$ along the other eigenvectors shrinks by $|\lambda_2/\lambda_1| = r$ per step. the Rayleigh quotient of a symmetric matrix has zero first-order sensitivity to eigenvector errors (on paper 2: $r(v) - \lambda_1 = \sin^2\theta\,(\cdots)$), so its error is the *square* of the vector error. two practical consequences: an eigenvalue estimate is often good long before the eigenvector is, and a convergence test on the eigenvalue alone will stop too early if you need the vector. in PageRank, $r = |\lambda_2| \le \alpha$, so α = 0.85 means roughly 100 iterations for 8 digits, independent of the size of the web. that's why it was practical at web scale.

## experiment 3: watching the QR algorithm, and breaking it

**what people usually predict:** a symmetric matrix with distinct eigenvalues ±1, and only 2×2, will converge in a handful of steps.

**what happens:** for the 5×5 matrix, the off-diagonal entries fade step by step. every adjacent eigenvalue ratio here is 0.5 (4/8, 2/4, 1/2, 0.5/1), so all the subdiagonal entries shrink at the same rate, halving each step (entries further from the diagonal shrink faster, as products of those ratios). by about k = 50 the matrix is diagonal to machine precision, with 8, 4, 2, 1, 0.5 down the diagonal in order of magnitude. for $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ the subdiagonal stays at **exactly 1 forever**: the algorithm never moves. `qr_algorithm` returns about $[0, 0]$ after 1000 iterations. your Jacobi `eigh` returns $[-1, 1]$ immediately.

**why:** for this matrix, $A$ is itself orthogonal, so its QR factorization is $Q = \pm A$, $R = \pm I$, and $RQ = A$. a fixed point. more generally, the unshifted QR algorithm is power iteration on subspaces, and it separates eigenvalues by *magnitude*. $+1$ and $-1$ have the same magnitude, so nothing distinguishes them. the cure is a **shift**: run QR on $A - \mu I$, where the eigenvalues $1 - \mu$ and $-1 - \mu$ have different magnitudes. practical codes use the Wilkinson shift, which guarantees convergence for symmetric matrices. Jacobi doesn't care about magnitudes at all: it zeroes off-diagonal entries directly, using the rotation angle from the 2×2 block.
