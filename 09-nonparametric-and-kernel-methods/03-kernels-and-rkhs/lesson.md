# kernels & RKHS

## why this matters

a linear model is fast, convex and well understood, and it can only draw straight lines. the classic fix is to engineer features: add $x_1^2$, $x_1 x_2$, $\sin x_3$, and fit a linear model on those. chapter 07's `PolynomialFeatures` did exactly that. but the number of features explodes: all monomials of degree up to 5 in 100 variables is about 96 million columns.

the **kernel trick** is the observation that many algorithms never need the features themselves, only the inner products between them, and that for the right feature maps those inner products are cheap to compute directly: $(x^\top y + 1)^5$ costs one dot product, not 96 million multiplications. better still, some kernels correspond to *infinitely* many features, which no explicit method could ever compute. the Gaussian (RBF) kernel is one of them, and it's behind SVMs (topic 04), kernel ridge regression (topic 05), Gaussian processes (topic 06), kernel PCA (chapter 11) and the neural tangent kernel (chapter 19).

this topic builds the kernel library that the rest of the chapter uses, and the theory that says which functions are allowed as kernels (Mercer's condition: positive semi-definiteness) and what kind of functions a kernel method can learn (the reproducing kernel Hilbert space, and the representer theorem, which says the answer is always a weighted sum of kernels centred on the training points).

## warm-up: predict first

1. how many features does $(x^\top y + 1)^2$ correspond to, for $x \in \mathbb R^d$? and $(x^\top y + 1)^p$?
2. the Gaussian kernel $e^{-\gamma\|x - y\|^2}$ is an inner product of feature vectors. how long are they?
3. is every symmetric "similarity" between 0 and 1 a valid kernel? what about $\tanh(x^\top y - 1)$?
4. you interpolate 20 points with a sum of Gaussian bumps $f = \sum_i \alpha_i e^{-\gamma(x - x_i)^2}$. what does $f$ look like between the points when $\gamma$ is huge? when it's tiny?
5. you subtract the mean of the feature vectors in feature space. can you do it without ever computing a feature vector?

## prerequisites check

- [ ] you know what a positive semi-definite matrix is, and that $A^\top A$ is one (→ ch 02 · topic 05)
- [ ] you can compute an eigendecomposition and read definiteness off the eigenvalues (→ ch 02 · topic 05)
- [ ] you can compute pairwise squared distances without loops (→ ch 00 · topic 02)
- [ ] you've solved least-squares and minimum-norm problems with the pseudoinverse (→ ch 02 · topics 04 and 06)
- [ ] you've used polynomial features to make a linear model nonlinear (→ ch 07 · topic 05)

## core ideas

### idea 1: kernels and Gram matrices

a **kernel** is a function $k(x, y)$ that measures similarity between two inputs. stack its values for two sets of points into a **Gram matrix** $K_{ij} = k(x_i, y_j)$, of shape $(n, m)$; with one set, $K$ is $(n, n)$ and symmetric. the four you'll build:

| name | $k(x, y)$ | parameters |
|---|---|---|
| linear | $x^\top y$ | none |
| polynomial | $(\gamma\, x^\top y + c_0)^p$ | degree $p$, scale $\gamma$, offset $c_0$ |
| RBF (Gaussian) | $\exp(-\gamma\|x - y\|^2)$ | width $\gamma > 0$ |
| Laplacian | $\exp(-\gamma\|x - y\|_1)$ | width $\gamma > 0$ |

the RBF kernel is 1 when $x = y$ and decays to 0 with distance; $\gamma$ sets how fast, and $1/\sqrt{2\gamma}$ is its length scale (later topics write $\gamma = 1/(2\ell^2)$). large $\gamma$: only very close points are similar. small $\gamma$: everything is similar to everything. the Laplacian kernel decays with the $\ell_1$ distance and has a sharp peak at $x = y$, which makes its functions rougher.

every one of them is a few matrix operations: $XY^\top$ for the inner products, and your `pairwise_distances` for the squared distances. the $\ell_1$ distance has no such shortcut, but you can still avoid an $(n, m, d)$ array by accumulating $|x_{ij} - y_{kj}|$ one feature $j$ at a time, so memory stays $O(nm)$.

> **▶ build: stage 1.** implement `linear_kernel`, `polynomial_kernel`, `rbf_kernel` and `laplacian_kernel`, all `(X, Y=None, ...)` with `Y` defaulting to `X` and an $(n, m)$ result. for the RBF kernel, pass `Y=None` straight through to `pairwise_distances` (it then guarantees an exactly zero diagonal, so $K_{ii} = 1$ exactly).

### idea 2: feature maps, and the trick

why call these inner products? expand the degree-2 polynomial kernel with $\gamma = c_0 = 1$:

$$(x^\top y + 1)^2 = 1 + 2\sum_i x_i y_i + \sum_i x_i^2 y_i^2 + 2\sum_{i<j} x_i x_j\, y_i y_j = \varphi(x)^\top \varphi(y),$$

with the **feature map**

$$\varphi(x) = \big(1,\ \sqrt2\,x_1, \dots, \sqrt2\,x_d,\ x_1^2, \dots, x_d^2,\ \sqrt2\,x_1x_2, \dots, \sqrt2\,x_{d-1}x_d\big),$$

which has $1 + d + d + \binom d2 = \frac{(d+1)(d+2)}{2}$ entries (warm-up 1). a linear model on $\varphi(x)$ is a quadratic model on $x$, and any algorithm that touches the data only through inner products $\varphi(x_i)^\top\varphi(x_j)$ can use $k(x_i, x_j)$ instead, at the cost of one $d$-dimensional dot product. that substitution is the **kernel trick**. in general, $(x^\top y + 1)^p$ corresponds to all $\binom{d+p}{p}$ monomials of degree up to $p$, weighted by multinomial coefficients.

the RBF kernel goes further. factor it as $e^{-\gamma\|x\|^2} e^{-\gamma\|y\|^2} e^{2\gamma x^\top y}$ and expand the last factor as a power series, $\sum_{p \ge 0} \frac{(2\gamma)^p}{p!}(x^\top y)^p$: a sum of polynomial kernels of *every* degree. its feature map has infinitely many coordinates (warm-up 2), so no explicit method could compute it, and yet each kernel value is one exponential.

the trick changes what's expensive. explicit features cost $O(nD)$ memory and $O(n^2 D)$ for all inner products, with $D$ the number of features. kernels cost $O(n^2 d)$ to fill a Gram matrix, independent of $D$, but the Gram matrix is $n \times n$: kernel methods are cheap in the feature dimension and expensive in the number of examples. topic 05 is about getting the best of both.

> **▶ build: stage 2.** implement `quadratic_features(X)`: the map above, in that order (constant, the $\sqrt2 x_i$, the squares, then the $\sqrt2 x_i x_j$ for $i < j$ in `np.triu_indices(d, k=1)` order). the check verifies $\varphi(X)\varphi(Y)^\top = (XY^\top + 1)^2$.

> **🔮 predict:** with $d = 10$ and degree 4, how many times more numbers does the explicit map $\varphi(x) = (x; 1)^{\otimes 4}$ (every product of four coordinates of $(x_1, \dots, x_d, 1)$, duplicates included) store per point than $x$ itself? the lab times both ways of computing the Gram matrix.

### idea 3: one interface for every kernel

every kernel method in this chapter (the SVM, kernel ridge, Nyström features) takes the same hyperparameters, `kernel`, `gamma`, `degree`, `coef0`, as scikit-learn does. `get_kernel(kernel, **params)` turns them into a function `K(X, Y=None)`: a name picks the kernel and binds only the parameters that kernel uses (so `get_kernel("rbf", gamma=0.5, degree=3)` ignores the degree), and a callable is returned unchanged, so you can pass your own kernel anywhere. one small function, but it means each estimator writes `K = get_kernel(self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)` and never needs to know which kernel it has.

> **▶ build: stage 3.** implement `get_kernel(kernel, **params)` for `"linear"`, `"poly"`, `"rbf"` and `"laplacian"`. parameters you don't pass keep the kernel functions' own defaults. an unknown name raises `ValueError`.

### idea 4: which functions are kernels? Mercer's condition

if $k(x, y) = \varphi(x)^\top\varphi(y)$ for some feature map, then every Gram matrix is $K = \Phi\Phi^\top$ with $\Phi$ the matrix of feature vectors, and for any vector $c$,

$$c^\top K c = \sum_{i,j} c_i c_j\, \varphi(x_i)^\top\varphi(x_j) = \Big\|\sum_i c_i\varphi(x_i)\Big\|^2 \ge 0.$$

so a kernel's Gram matrices are always **positive semi-definite** (PSD), for every finite set of points. the converse is the deep part: every symmetric function whose Gram matrices are all PSD *is* an inner product in some feature space (Mercer, 1909, for continuous kernels on compact sets; Moore–Aronszajn in general). so "valid kernel" and "symmetric with PSD Gram matrices" mean the same thing, and you never need to write the feature map down.

PSD is what the algorithms need. an SVM's dual problem (topic 04) is a convex quadratic program only if $K$ is PSD; kernel ridge (topic 05) solves $(K + \alpha I)\beta = y$, which is well-posed for every $\alpha > 0$ only if $K$ has no negative eigenvalues; a Gaussian process (topic 06) uses $K$ as a covariance matrix, which must be PSD. not every plausible similarity qualifies (warm-up 3). the "sigmoid kernel" $\tanh(\gamma x^\top y + c_0)$ is not PSD for most parameters, and $\exp(-\|x - y\|^4)$, which looks like a sharper Gaussian, isn't either.

checking a Gram matrix numerically means checking symmetry and that its smallest eigenvalue isn't negative, with a tolerance for round-off: eigenvalues of $-10^{-13}$ show up in perfectly good Gram matrices. a failure on one dataset proves that a function is not a kernel; passing on one dataset proves nothing about other points.

kernels also combine. if $k_1$ and $k_2$ are kernels, so are $k_1 + k_2$, $c\,k_1$ for $c > 0$, the product $k_1 k_2$ (on paper 2), $f(x)k_1(x, y)f(y)$ for any function $f$, and $\exp(k_1)$ (a limit of sums of products). that's how you build kernels for structured data from simple pieces.

finally, **centring**. many methods want features with zero mean (kernel PCA in chapter 11, for one). in feature space that means $\tilde\varphi(x_i) = \varphi(x_i) - \bar\varphi$ with $\bar\varphi = \frac1n\sum_j\varphi(x_j)$. you can't subtract vectors you never computed, but you don't need to (warm-up 5): with $H = I - \frac1n\mathbf 1\mathbf 1^\top$, the centred Gram matrix is $\tilde K = HKH$, which in entries is $K_{ij}$ minus its row mean, minus its column mean, plus the overall mean.

> **▶ build: stage 4.** implement `is_psd(K, *, tol=1e-8)`: `False` unless $K$ is square and symmetric (up to `tol` times its scale), then compare the smallest eigenvalue from `np.linalg.eigvalsh` with $-\text{tol}\cdot\max(1, |\lambda_{\max}|)$. implement `center_gram(K)` with row, column and overall means (no $n \times n$ matrix products needed).

> **🔮 predict:** the "triangle" function $\max(0, 1 - \|x - y\|/2)$ is a perfectly good kernel for points on a line. is it still a kernel for points in the plane?

### idea 5: the RKHS and the representer theorem

which functions can a kernel method represent? start from the kernel's "bumps", $k(x_i, \cdot)$, and take finite combinations $f = \sum_i \alpha_i k(x_i, \cdot)$. define an inner product between two of them by declaring $\langle k(x, \cdot), k(y, \cdot)\rangle = k(x, y)$ and extending linearly. completing this space gives the **reproducing kernel Hilbert space** $\mathcal H$ of $k$. its defining property is **reproduction**: evaluating a function is an inner product with a bump,

$$f(x) = \langle f, k(x, \cdot)\rangle_{\mathcal H},$$

and the norm of a finite combination is

$$\|f\|_{\mathcal H}^2 = \Big\langle \sum_i\alpha_i k(x_i, \cdot), \sum_j\alpha_j k(x_j, \cdot)\Big\rangle = \alpha^\top K\alpha.$$

that norm measures complexity. for the RBF kernel, functions with a small norm are smooth and slowly varying; wiggly functions need large coefficients that partly cancel, and a large norm. in feature-space terms, $f(x) = w^\top\varphi(x)$ and $\|f\|_{\mathcal H} = \|w\|$, so penalizing the RKHS norm is ridge regression in feature space.

the payoff is the **representer theorem** (Kimeldorf & Wahba, 1971; Schölkopf, Herbrich & Smola, 2001): for any loss $L$ on the training points and any increasing penalty of the norm, a minimizer of

$$L\big(f(x_1), \dots, f(x_n)\big) + \Omega\big(\|f\|_{\mathcal H}\big)$$

over the whole (possibly infinite-dimensional) $\mathcal H$ has the form $f^\star = \sum_{i=1}^n \alpha_i k(x_i, \cdot)$: only $n$ unknowns, one per training point, however large $\mathcal H$ is (on paper 3). every kernel method in this chapter is an instance: the SVM (hinge loss), kernel ridge (squared loss), and the Gaussian process mean.

the simplest instance: among all functions in $\mathcal H$ that **interpolate** the data, $f(x_i) = y_i$, find the one with the smallest norm. by the theorem it's $\sum_i\alpha_i k(x_i, \cdot)$, the interpolation conditions say $K\alpha = y$, and its norm is $\sqrt{\alpha^\top K\alpha} = \sqrt{y^\top K^{-1}y}$. when $K$ is singular (repeated points) any solution of $K\alpha = y$ gives the same function, and least squares finds one. as $\gamma$ grows, the RBF interpolant becomes a row of narrow spikes on the data with nothing in between (warm-up 4); as $\gamma$ shrinks, $K$ becomes nearly singular and the interpolant swings wildly to hit every point exactly. topic 05 tames both by not insisting on exact interpolation.

> **▶ build: stage 5.** implement `interpolate(X, y, kernel)` (the coefficients from `np.linalg.lstsq(K, y, rcond=None)`, with `kernel` a function like the ones `get_kernel` returns), `kernel_expansion(alpha, X_train, X_new, kernel)` ($f$ at new points: $K(X_\text{new}, X_\text{train})\,\alpha$) and `rkhs_norm(alpha, K)`. the checks confirm the representer theorem: for the quadratic kernel, your interpolant equals the minimum-norm solution computed explicitly in feature space.

## worked example

two points, $x = (1, 2)$ and $y = (3, -1)$:

- linear: $x^\top y = 3 - 2 = 1$.
- polynomial, $p = 2$, $\gamma = c_0 = 1$: $(1 + 1)^2 = 4$. with the explicit map, $\varphi(x) = (1, \sqrt2, 2\sqrt2, 1, 4, 2\sqrt2)$ and $\varphi(y) = (1, 3\sqrt2, -\sqrt2, 9, 1, -3\sqrt2)$, and $\varphi(x)^\top\varphi(y) = 1 + 6 - 4 + 9 + 4 - 12 = 4$. ✓
- RBF, $\gamma = 0.5$: $\|x - y\|^2 = 4 + 9 = 13$, so $e^{-6.5} \approx 0.0015$.
- Laplacian, $\gamma = 0.5$: $\|x - y\|_1 = 2 + 3 = 5$, so $e^{-2.5} \approx 0.0821$.

**a minimum-norm interpolant.** RBF kernel with $\gamma = 1$, data $x = (0, 1)$, $y = (1, 2)$. the Gram matrix is $K = \begin{pmatrix}1 & e^{-1}\\ e^{-1} & 1\end{pmatrix}$ with determinant $1 - e^{-2} = 0.864665$, so

$$\alpha = K^{-1}y = \frac{1}{0.864665}\begin{pmatrix}1 - 2e^{-1}\\ 2 - e^{-1}\end{pmatrix} = \begin{pmatrix}0.305597\\ 1.887587\end{pmatrix}.$$

halfway, $f(0.5) = (\alpha_1 + \alpha_2)e^{-0.25} = 2.193184 \times 0.778801 = 1.708054$, above the straight line's 1.5: the two bumps add up between the points. the norm is $\|f\|^2_{\mathcal H} = \alpha^\top K\alpha = \alpha^\top y = 0.305597 + 2 \times 1.887587 = 4.080771$. the tests check all of these.

## on paper

1. **the polynomial feature count.** show that $(x^\top y + 1)^p$ is an inner product of feature vectors indexed by the monomials of degree at most $p$ in $d$ variables, and that there are $\binom{d+p}{p}$ of them. evaluate for $d = 100$, $p = 5$.

<details><summary>solution</summary>

write $x^\top y + 1 = \sum_{j=0}^{d} x_j y_j$ with $x_0 = y_0 = 1$. by the multinomial theorem, $(\sum_j x_j y_j)^p = \sum_{|a| = p}\binom{p}{a}\prod_j (x_j y_j)^{a_j}$, summing over $a \in \mathbb N^{d+1}$ with $\sum a_j = p$, so $\varphi_a(x) = \sqrt{\binom pa}\prod_j x_j^{a_j}$. dropping $x_0$, the $a$'s correspond to monomials in $x_1..x_d$ of degree $\le p$; counting them is stars and bars, $\binom{d+p}{p}$. for $d = 100$, $p = 5$: $\binom{105}{5} = 96{,}560{,}646$.

</details>

2. **products of kernels are kernels.** show that if $K_1$ and $K_2$ are PSD, their elementwise (Schur) product $K_1 \odot K_2$ is PSD. (hint: write $K_2 = \sum_r \lambda_r u_r u_r^\top$.)

<details><summary>solution</summary>

$c^\top(K_1 \odot K_2)c = \sum_{ij}c_ic_j K_{1,ij}\sum_r\lambda_r u_{r,i}u_{r,j} = \sum_r\lambda_r (c \odot u_r)^\top K_1 (c \odot u_r) \ge 0$, since each $\lambda_r \ge 0$ and $K_1$ is PSD. in feature space: the product kernel's features are all products $\varphi_1(x)_a\varphi_2(x)_b$ (a tensor product).

</details>

3. **the representer theorem.** let $V = \mathrm{span}\{k(x_i, \cdot)\}$ and write any $f \in \mathcal H$ as $f = f_V + f_\perp$ with $f_\perp \perp V$. show that $f_\perp$ doesn't change $f(x_i)$ but does increase $\|f\|$, and conclude.

<details><summary>solution</summary>

by reproduction, $f(x_i) = \langle f, k(x_i, \cdot)\rangle = \langle f_V, k(x_i, \cdot)\rangle + \langle f_\perp, k(x_i, \cdot)\rangle = f_V(x_i) + 0$, since $f_\perp$ is orthogonal to every $k(x_i, \cdot)$. so the loss depends only on $f_V$. and $\|f\|^2 = \|f_V\|^2 + \|f_\perp\|^2 \ge \|f_V\|^2$, so an increasing penalty $\Omega$ can only prefer $f_\perp = 0$. a minimizer can therefore be taken in $V$, i.e. $f^\star = \sum_i\alpha_i k(x_i, \cdot)$ (and must be in $V$ if $\Omega$ is strictly increasing).

</details>

4. **centring in feature space.** show that $\big(\varphi(x_i) - \bar\varphi\big)^\top\big(\varphi(x_j) - \bar\varphi\big) = K_{ij} - \bar K_{i\cdot} - \bar K_{\cdot j} + \bar K_{\cdot\cdot}$, where the bars are row, column and overall means of $K$.

<details><summary>solution</summary>

expand: $\varphi_i^\top\varphi_j - \varphi_i^\top\bar\varphi - \bar\varphi^\top\varphi_j + \bar\varphi^\top\bar\varphi$. with $\bar\varphi = \frac1n\sum_k\varphi_k$: $\varphi_i^\top\bar\varphi = \frac1n\sum_k K_{ik} = \bar K_{i\cdot}$, similarly $\bar\varphi^\top\varphi_j = \bar K_{\cdot j}$, and $\bar\varphi^\top\bar\varphi = \frac{1}{n^2}\sum_{kl}K_{kl} = \bar K_{\cdot\cdot}$. in matrix form this is $HKH$.

</details>

## experiment

`lab.py` uses your kernel library:

1. explicit features against the kernel trick: memory and time to build a polynomial Gram matrix, as the degree grows.
2. a gallery of plausible similarity functions: which ones pass your `is_psd`, in 1-D and in 2-D?
3. minimum-norm RBF interpolation as $\gamma$ varies: the shape of the interpolant, its RKHS norm, and the conditioning of $K$.

## check your understanding

1. why does the kernel trick need the algorithm to depend on the data only through inner products?

<details><summary>answer</summary>

because the kernel replaces exactly $\varphi(x)^\top\varphi(y)$ and nothing else. if the algorithm needed, say, a single coordinate of $\varphi(x)$, or $\varphi(x)$ itself to store a weight vector, you'd have to compute the features. the representer theorem is what lets weight vectors be stored implicitly as $w = \sum_i\alpha_i\varphi(x_i)$.

</details>

2. the RBF Gram matrix of $n$ distinct points is strictly positive definite. what does that imply about fitting data with RBF bumps?

<details><summary>answer</summary>

$K$ is invertible, so $K\alpha = y$ has a solution for *any* targets $y$: an RBF expansion can interpolate any labels at distinct points, noise included. that's the sense in which the RBF kernel is infinitely flexible, and why kernel methods need regularization (topics 04–06).

</details>

3. a colleague's custom similarity passes `is_psd` on 500 random points. is it a kernel?

<details><summary>answer</summary>

not proven. PSD must hold for *every* finite set of points; one passing sample is evidence, not proof (the lab's $\exp(-r^3)$ can pass in some dimensions and fail in others). one failing sample, on the other hand, is a proof that it isn't a kernel.

</details>

4. what's the RKHS of the linear kernel?

<details><summary>answer</summary>

the linear functions $f(x) = w^\top x$, with $\|f\|_{\mathcal H} = \|w\|$. regularizing the RKHS norm with the linear kernel is ordinary ridge regression.

</details>

5. why might $\gamma$ matter more than the choice of kernel?

<details><summary>answer</summary>

$\gamma$ sets the length scale, i.e. which points count as similar. too large and every point is only similar to itself ($K \approx I$: memorization); too small and everything is similar ($K \approx \mathbf{1}\mathbf{1}^\top$: a nearly constant function, and a badly conditioned $K$). kernels of different shapes with matched length scales often give similar results.

</details>

6. you centre a Gram matrix with `center_gram`. what happens to its constant eigenvector $\mathbf 1$?

<details><summary>answer</summary>

$H\mathbf 1 = 0$, so $HKH\mathbf 1 = 0$: the centred Gram matrix always has eigenvalue 0 with eigenvector $\mathbf 1$. it has removed the "mean direction" from feature space.

</details>

## common bugs

- **an $(n, m, d)$ intermediate.** `np.abs(X[:, None] - Y[None]).sum(-1)` is fine for 100 points and needs gigabytes for 10,000. symptom: `MemoryError` or swapping. loop over features instead.
- **a slightly asymmetric RBF matrix with diagonal $1 - 10^{-16}$.** computing distances between `X` and a copy of `X` loses the exact-zero diagonal. symptom: `is_psd` with `tol=0` fails, and Cholesky (topic 06) occasionally fails. pass `Y=None` through.
- **forgetting the parameters `get_kernel` should ignore.** passing `degree=3` to `rbf_kernel` raises a `TypeError` deep inside an estimator. bind only the parameters each kernel takes.
- **testing PSD with `np.linalg.eig`.** for a symmetric matrix, use `eigvalsh`: it's faster and returns real, sorted eigenvalues; `eig` can return tiny imaginary parts.
- **a zero tolerance.** symptom: `is_psd` rejects perfectly good Gram matrices because of eigenvalues around $-10^{-14}$.
- **`np.linalg.solve` on a singular Gram matrix.** symptom: `LinAlgError` with repeated points. least squares (or a small ridge, topic 05) handles it.

## going deeper

- Schölkopf & Smola (2002), *Learning with Kernels*, MIT Press, chapters 2 and 4: kernels, feature spaces, the RKHS and the representer theorem. [doi:10.7551/mitpress/4175.001.0001](https://doi.org/10.7551/mitpress/4175.001.0001)
- Aronszajn (1950), "Theory of reproducing kernels", *Transactions of the American Mathematical Society* 68(3): 337–404. [doi:10.1090/S0002-9947-1950-0051437-7](https://doi.org/10.1090/S0002-9947-1950-0051437-7)
- Kimeldorf & Wahba (1971), "Some results on Tchebycheffian spline functions", *Journal of Mathematical Analysis and Applications* 33(1): 82–95. the original representer theorem. [doi:10.1016/0022-247X(71)90184-3](https://doi.org/10.1016/0022-247X(71)90184-3)
- Schölkopf, Herbrich & Smola (2001), "A generalized representer theorem", *COLT*. [doi:10.1007/3-540-44581-1_27](https://doi.org/10.1007/3-540-44581-1_27)
- Shawe-Taylor & Cristianini (2004), *Kernel Methods for Pattern Analysis*, Cambridge University Press, chapters 2–3 and 9 (kernel construction). [doi:10.1017/CBO9780511809682](https://doi.org/10.1017/CBO9780511809682)
- H.-T. Lin & C.-J. Lin (2003), "A study on sigmoid kernels for SVM and the training of non-PSD kernels by SMO-type methods", technical report, National Taiwan University. [csie.ntu.edu.tw/~cjlin/papers/tanh.pdf](https://www.csie.ntu.edu.tw/~cjlin/papers/tanh.pdf)

## connections

- ← builds on: inner products, PSD matrices and eigenvalues, the pseudoinverse (ch 02), pairwise distances (ch 00), polynomial features (ch 07), similarity and the curse of dimensionality (topic 01).
- → used in: SVMs (topic 04), kernel ridge regression and Nyström features (topic 05), Gaussian process covariances (topic 06), kernel PCA (ch 11), kernel two-sample tests and MMD (ch 21), and the neural tangent kernel, where a wide network trains like kernel regression (ch 19).
