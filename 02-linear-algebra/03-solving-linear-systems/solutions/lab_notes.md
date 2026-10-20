# solving linear systems · lab notes

## experiment 1: which matrices break elimination without pivoting?

**what people usually predict:** as δ → 0 the matrix approaches the harmless $\begin{pmatrix}0&1\\1&1\end{pmatrix}$, so both methods should be fine. or perhaps no-pivot elimination blows up only at δ = 0 exactly, where it divides by zero.

**what happens:** without pivoting, the error grows steadily as δ shrinks, roughly like $10^{-16}/\delta$: about $10^{-12}$ at δ = 1e-5, $10^{-5}$ at δ = 1e-12. at δ ≈ 1e-16 the error is 100%: the method returns $x_1 = 0$ instead of $1$. your pivoted `solve` stays at machine precision the whole way.

**why:** the multiplier is $\ell_{21} = 1/\delta$, so $u_{22} = 1 - 1/\delta$. in floating point, $1 - 10^{16}$ can't hold the "1" (only ~16 digits fit), so the factors describe $\begin{pmatrix}\delta&1\\1&0\end{pmatrix}$, a different matrix. the problem was never the matrix (it's well conditioned). it was the *algorithm's* choice of pivot. swapping rows makes the multiplier δ instead of 1/δ, and nothing gets rounded away. this is the clearest example of an **unstable algorithm on a well-conditioned problem**, a distinction topic 08 makes precise.

## experiment 2: can partial pivoting fail?

**what people usually predict:** with $|\ell_{ik}| \le 1$ everything stays bounded, so the growth factor stays small for every matrix and `solve` is always accurate.

**what happens:** for random matrices the growth factor stays small, roughly 1.5 to 5 at every size (about 1.6–4.5 with this seed), and the error stays near $10^{-15}$. for Wilkinson's matrix the growth factor is exactly $2^{n-1}$: about 500 at n = 10, $5 \times 10^{14}$ at n = 50, $6 \times 10^{17}$ at n = 60. the error rises in step with it, and at n = 60 the "solution" is pure garbage (relative error above 100%).

**why:** in Wilkinson's matrix every pivot is a tie at magnitude 1, so no swaps happen, and each elimination step adds the last column to itself: $u_{in} = 2u_{i-1,n}$. the last column doubles $n - 1$ times. rounding errors are relative to these huge numbers, and after division they swamp the $O(1)$ solution. libraries still use partial pivoting because such matrices are vanishingly rare in practice. Trefethen & Schreiber (1990) found that for random matrices the growth factor rises only slowly, like a small power of $n$, and you just saw ~3. *complete* pivoting (searching the whole trailing block) provably limits growth, but costs $O(n^3)$ comparisons, and almost nobody needs it.

## experiment 3: how many symmetric matrices are positive definite?

**what people usually predict:** about half (each matrix is "positive" or "negative"), or a slow decline.

**what happens:** about 49% for n = 1, 15% for n = 2, 2.5% for n = 3, 0.25% for n = 4, and essentially **zero** (none in 2000 trials) for n ≥ 6. meanwhile 200 out of 200 sample covariance matrices pass.

**why:** positive definite means *every* eigenvalue is positive. a random symmetric matrix has eigenvalues spread roughly symmetrically around zero (Wigner's semicircle for large n), so having all $n$ of them on the positive side is a large-deviation event. its probability decays like $e^{-cn^2}$ (Dean & Majumdar, 2006), far faster than $2^{-n}$. covariance matrices are different: $v^\top C v = \frac{1}{m-1}\|X_c v\|^2 \ge 0$ for every $v$. they're positive semidefinite by construction, and positive definite once there are more (generic) samples than dimensions. the lesson for ML: positive definiteness isn't a property you stumble into. it comes from structure ($X^\top X$, kernels, Hessians at nondegenerate minima). when you build such a matrix by hand (say, by averaging or perturbing), `cholesky` is the quick way to check you still have it.

reference: Dean & Majumdar (2006), "Large deviations of extreme eigenvalues of random matrices", *Physical Review Letters* 97, 160201. [arXiv:cond-mat/0609651](https://arxiv.org/abs/cond-mat/0609651)
