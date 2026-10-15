# dimensionality reduction

## why this matters

a 64 × 64 photo of a face is a point in 4,096 dimensions, but faces don't fill that space. almost every random 4,096-vector is static; faces live near a much smaller set, parametrized by things like pose, lighting and identity. if you could find coordinates for that set, you could store faces compactly, denoise them, plot them, and feed downstream models 50 numbers instead of 4,096. the same holds for MNIST digits (784 pixels, maybe a few dozen degrees of freedom), gene expression, word embeddings and sensor arrays.

**principal component analysis** is the workhorse: find the directions along which the data vary most, and keep only those. you met its engine, the SVD, in chapter 02; here you turn it into a tool and look at it from three more angles. as a *reconstruction* method, which makes "explained variance" a statement about training error (and raises the question of test error). as a *probabilistic model* (probabilistic PCA), which gives PCA a likelihood, so you can choose the number of components on held-out data. and as a *kernel method* (kernel PCA), which finds curved structure by running PCA in a feature space you never compute. the capstone uses your PCA to compress MNIST before clustering it.

## warm-up: predict first

1. PCA on 4,096-pixel faces keeps the components that explain 95% of the variance. what fraction of a *new* face's variance (one PCA wasn't fitted on) will the same components capture?
2. you keep more and more components. can the reconstruction error of held-out data ever go *up*?
3. two features have correlation 0.99. how many principal components do you need to capture 99.5% of their variance? and if you multiply one feature by 1,000 first?
4. an eigenvector is only defined up to sign. why does that matter for code, and for tests?
5. two concentric rings in the plane. can any linear projection separate them? can PCA in a feature space that contains $x_1^2 + x_2^2$?

## prerequisites check

- [ ] you know the SVD $X = U\Sigma V^\top$ and the Eckart–Young theorem (→ ch 02 · topic 06)
- [ ] you know that the eigenvectors of a symmetric matrix are orthogonal, and how to get them with `eigh` (→ ch 02 · topic 05)
- [ ] you can write the log density of a multivariate Gaussian and know how a linear map transforms a Gaussian (→ ch 04 · topic 04)
- [ ] you can write a `Transformer` with `fit`, `transform` and `inverse_transform` (→ ch 07 · topics 01 and 05)
- [ ] you've seen the idea of a feature map $\phi(x)$, e.g. polynomial features (→ ch 07 · topic 05)

## core ideas

### idea 1: principal directions from the SVD

centre the data: $X_c = X - \mathbf 1\bar x^\top$, shape $(n, d)$. the first principal direction is the unit vector $v$ along which the projected data $X_cv$ have the largest variance:

$$v_1 = \arg\max_{\|v\| = 1} \frac{1}{n - 1}\|X_cv\|^2 = \arg\max_{\|v\| = 1} v^\top S v, \qquad S = \frac{1}{n - 1}X_c^\top X_c .$$

the maximizer is the top eigenvector of the covariance matrix $S$ (on paper 1), and the variance along it is the top eigenvalue. the next direction maximizes the variance among unit vectors orthogonal to $v_1$, and so on: the principal directions are the eigenvectors of $S$, in order of eigenvalue.

you don't need to form $S$. with the SVD $X_c = U\Sigma V^\top$, $S = V\frac{\Sigma^2}{n - 1}V^\top$: the right singular vectors are the principal directions, and the variances are $\lambda_j = s_j^2/(n - 1)$. this is better numerically (forming $X_c^\top X_c$ squares the condition number, chapter 02) and cheaper when $d \gg n$: for faces, the SVD of a $320 \times 4096$ matrix costs a fraction of a second, while $S$ would be $4096 \times 4096$. you built an SVD in chapter 02; here use `np.linalg.svd`, which is the same factorization done fast (AUTHORING §9).

the vocabulary: `components_` are the top $k$ rows of $V^\top$ (a $(k, d)$ array, one direction per row), `explained_variance_` the $\lambda_j$, and `explained_variance_ratio_` the fractions $\lambda_j/\sum_l\lambda_l$, where the sum runs over *all* $\min(n, d)$ components (the total variance), not just the kept ones. `n_components` can be an int, `None` (keep $\min(n, d)$), or a float in $(0, 1)$ meaning "the fewest components whose ratios add up to at least this".

**signs.** if $v$ is a principal direction, so is $-v$, and different SVD routines pick different signs (warm-up 4). to make results reproducible, the course flips each component so that its largest-magnitude entry is positive (the helper `flip_signs` is given). tests that compare with other libraries compare up to sign.

> **🔮 predict:** two features with correlation 0.99 and equal variances: what's the first component's explained variance ratio? now multiply the second feature by 1,000. what is it now, and what does the first component point along?

> **▶ build: stage 1.** implement `PCA(n_components=None, *, whiten=False).fit`: `mean_`, `components_` (sign-flipped), `explained_variance_` (ddof=1), `explained_variance_ratio_`, `singular_values_` and `n_components_`, with a `ValueError` for an impossible `n_components`.

### idea 2: projecting, reconstructing, whitening

the **scores** of a point are its coordinates along the components: $z = V_k^\top(x - \bar x)$, a $k$-vector (`transform`). mapping them back gives the **reconstruction** $\hat x = \bar x + V_kz$ (`inverse_transform`), the closest point to $x$ in the $k$-dimensional affine subspace through the mean spanned by the components.

this connects the two views of PCA. the total squared reconstruction error on the training data splits by Pythagoras into the discarded directions:

$$\sum_i\|x_i - \hat x_i\|^2 = \sum_{j > k}s_j^2 = (n - 1)\sum_{j > k}\lambda_j,$$

and by Eckart–Young no other $k$-dimensional subspace does better (on paper 2). so "the first 50 components explain 88% of the variance" means exactly "reconstructing the *training* data from 50 numbers per point leaves 12% of its variance as error". for new data there's no such identity: the components were chosen to fit the training points, and a new point has variance in directions they missed. projecting onto a bigger subspace can never move a point further away, so held-out reconstruction error still decreases with $k$ (warm-up 2), but it doesn't go to 0, and the gap between training and held-out error is the lab's first experiment.

**whitening** divides each score by $\sqrt{\lambda_j}$, so every kept direction has unit variance and the scores are uncorrelated: their covariance is the identity. it makes later distance computations treat every kept direction equally, which helps some algorithms (and amplifies noise in low-variance directions, which hurts others). `inverse_transform` must undo it first.

> **▶ build: stage 2.** implement `PCA.transform` (scores, divided by $\sqrt{\lambda_j}$ when `whiten`) and `PCA.inverse_transform` (multiply back if whitened, then $Z V_k + \bar x$).

### idea 3: probabilistic PCA

PCA is a projection, not a model: it can't say how likely a point is, and it can't compare $k = 5$ with $k = 6$ except by "more variance explained", which always favours bigger $k$. **probabilistic PCA** (Tipping & Bishop, 1999) turns it into a generative model with a latent variable, much like the mixture of topic 02:

$$z \sim \mathcal N(0, I_k), \qquad x = Wz + \mu + \varepsilon, \qquad \varepsilon \sim \mathcal N(0, \sigma^2I_d).$$

$W$ is $(d, k)$: $k$ directions of real variation plus isotropic noise everywhere. integrating out $z$, $x \sim \mathcal N(\mu, C)$ with $C = WW^\top + \sigma^2I$: a Gaussian whose covariance is "low rank plus a constant". the maximum likelihood estimates have a closed form. with $S = X_c^\top X_c/n$ (divide by $n$: it's the MLE) and its eigenvalues $\lambda_1 \ge \dots \ge \lambda_d$ and eigenvectors $U$,

$$\hat\mu = \bar x, \qquad \hat\sigma^2 = \frac{1}{d - k}\sum_{j > k}\lambda_j, \qquad \hat W = U_k(\Lambda_k - \hat\sigma^2I)^{1/2}$$

(up to a rotation of the latent space, which you fix by taking $R = I$). the noise variance is the average variance in the discarded directions, and $C$ reproduces the top $k$ eigenvalues of $S$ exactly while flattening the rest to their mean (on paper 3). the eigenvalues beyond the rank of $X_c$ (when $n < d$) are zero; compute the discarded sum as the total variance $\operatorname{tr}S$ minus the kept ones, so you never need all $d$ eigenvalues.

what you gain:

- **a likelihood.** `score_samples(x)` $= \log\mathcal N(x \mid \mu, C)$, computed with a Cholesky factor of $C$ as in topic 02. held-out log-likelihood penalizes both too few components (variance left in the noise term) and too many (fitting noise directions in the training data), so it can choose $k$.
- **a posterior.** the latent coordinates of $x$ have the Gaussian posterior with mean $M^{-1}W^\top(x - \mu)$, $M = W^\top W + \sigma^2I_k$ (`transform`). it's PCA's projection, shrunk towards 0 in directions where the signal is weak compared to the noise; as $\sigma^2 \to 0$ it becomes $\Lambda_k^{-1/2}U_k^\top(x - \mu)$, the whitened PCA scores.
- **samples.** draw $z$, then $x = Wz + \mu + \sigma\varepsilon$.

> **🔮 predict:** data with 5 true latent dimensions in 30, plus noise. as $k$ goes from 1 to 12, how do the *held-out* PCA reconstruction error and the *held-out* PPCA log-likelihood behave?

> **▶ build: stage 3.** implement `ProbabilisticPCA(n_components=2)`: `fit` (closed form; `W_`, `noise_variance_`, `mean_`; `ValueError` unless $1 \le k < d$), `transform` (posterior mean), `score_samples` and `sample(n=1, *, rng=None)`.

### idea 4: kernel PCA

PCA finds flat subspaces. two concentric rings have no flat structure that separates them (warm-up 5), but add the feature $x_1^2 + x_2^2$ and the radius becomes a direction, which PCA can find. in general: map each point to features $\phi(x)$ and run PCA there. the trouble is that interesting feature spaces are huge (all monomials up to degree 10) or infinite-dimensional (the RBF kernel's).

the **kernel trick** (Schölkopf, Smola & Müller, 1998) avoids ever computing $\phi$. every principal direction in feature space is a combination of the (centred) training features, $v = \sum_ia_i\tilde\phi(x_i)$, and everything PCA needs can be written with inner products $k(x, x') = \phi(x)^\top\phi(x')$. you choose $k$ directly:

$$k_{\text{linear}}(x, x') = x^\top x', \qquad k_{\text{poly}}(x, x') = (\gamma\,x^\top x' + c_0)^{p}, \qquad k_{\text{rbf}}(x, x') = e^{-\gamma\|x - x'\|^2}.$$

the recipe:

1. the Gram matrix $K_{ij} = k(x_i, x_j)$, $(n, n)$.
2. centre it, which is centring the features: $\tilde K = K - \mathbf 1_nK - K\mathbf 1_n + \mathbf 1_nK\mathbf 1_n$ with $\mathbf 1_n$ the $(n, n)$ matrix of $1/n$. in words: subtract each column's mean and each row's mean, add back the overall mean.
3. eigendecompose: $\tilde K\alpha_j = \lambda_j\alpha_j$ with unit eigenvectors $\alpha_j$, largest $\lambda_j$ first. store them as `alphas_` $(n, k)$ (sign-flipped) and `lambdas_` $(k,)$.

the direction $v_j = \sum_i a_{ij}\tilde\phi(x_i)$ has unit length when $a_j = \alpha_j/\sqrt{\lambda_j}$ (on paper 4), and the score of training point $i$ is then $\sqrt{\lambda_j}\,\alpha_{ij}$. with the linear kernel, $\tilde K = X_cX_c^\top$, whose eigenvalues are the $s_j^2$: kernel PCA is PCA, computed from the $n \times n$ side.

the RBF kernel's $\gamma$ sets the length scale of "similar". tiny $\gamma$: every pair looks alike, and the kernel behaves like a low-degree polynomial of the inputs. huge $\gamma$: every point is similar only to itself, $K \approx I$, and the components describe individual points rather than structure. the lab sweeps it.

> **▶ build: stage 4.** implement `kernel_matrix(X, Y, kernel="rbf", gamma=1.0, degree=3, coef0=1.0)` (vectorized, using `pairwise_distances` for the RBF) and `KernelPCA(n_components=2, *, kernel="rbf", gamma=1.0, degree=3, coef0=1.0).fit`: store `X_fit_`, the training Gram matrix's column means `K_col_means_` and overall mean `K_mean_` (you'll need them in stage 5), then `lambdas_` and `alphas_`.

### idea 5: embedding new points

to embed a new point $x$, compute its kernel row $k_x = (k(x, x_1), \dots, k(x, x_n))$, centre it, and project: $z_j = \tilde k_x^\top\alpha_j/\sqrt{\lambda_j}$. the centring must be the *training* one, because "centred" means "with the training mean of $\phi$ subtracted":

$$\tilde k_x = k_x - \bar k_x\mathbf 1 - \bar K_{\text{col}} + \bar K, $$

where $\bar k_x$ is the mean of the new row, $\bar K_{\text{col}}$ the vector of the training Gram matrix's column means, and $\bar K$ its overall mean. that's the same formula as step 2, with the new rows playing the role of the training rows. getting this wrong (centring the new rows with their own statistics, or not at all) gives embeddings that look plausible and are shifted. two checks catch it: on the training points, `transform` must return $\sqrt{\lambda_j}\alpha_{ij}$, and with the linear kernel it must agree with `PCA.transform` on *new* points.

> **▶ build: stage 5.** implement `KernelPCA.transform`: kernel rows against `X_fit_`, training-style centring, then `@ alphas_ / sqrt(lambdas_)` (guard against zero eigenvalues).

## worked example

four points, built from two orthogonal directions $u = (0.6, 0.8)$ and $v = (0.8, -0.6)$: $x = a\,u + b\,v$ with $a = (3, -3, 3, -3)$ and $b = (1, 1, -1, -1)$, i.e.

$$X = \begin{pmatrix}2.6 & 1.8\\ -1.0 & -3.0\\ 1.0 & 3.0\\ -2.6 & -1.8\end{pmatrix}.$$

the mean is 0. the coordinates along $u$ and $v$ are uncorrelated, with variances (ddof = 1) $36/3 = 12$ and $4/3$, so the principal directions are $u$ and $v$, `explained_variance_` $= (12, 1.333)$ and `explained_variance_ratio_` $= (0.9, 0.1)$. both directions already have a positive largest entry (0.8), so no sign flip: `components_` $= \begin{pmatrix}0.6 & 0.8\\ 0.8 & -0.6\end{pmatrix}$.

with one component, the scores are $a = (3, -3, 3, -3)$, and the reconstructions $3u = (1.8, 2.4)$ and $-3u$. each point is off by $b\,v$, a squared error of 1, so the total is $4 = s_2^2 = (n - 1)\lambda_2 = 3 \cdot \frac43$: the discarded variance, as idea 2 says. the tests check all of this.

## on paper

1. **maximum variance.** show that the unit vector maximizing $v^\top Sv$ is the top eigenvector of $S$, and that the maximum is the top eigenvalue.

<details><summary>solution</summary>

Lagrangian $v^\top Sv - \lambda(v^\top v - 1)$; setting the gradient to zero gives $2Sv - 2\lambda v = 0$, so $v$ is an eigenvector and $v^\top Sv = \lambda v^\top v = \lambda$. among the stationary points the largest value is the largest eigenvalue. (alternatively: write $v = \sum_jc_ju_j$ in the eigenbasis; $v^\top Sv = \sum_j\lambda_jc_j^2$ with $\sum c_j^2 = 1$ is a weighted average of eigenvalues, maximized by putting all weight on $\lambda_1$.)

</details>

2. **reconstruction error.** with $X_c = U\Sigma V^\top$ and $\hat X_c = X_cV_kV_k^\top$, show that $\|X_c - \hat X_c\|_F^2 = \sum_{j > k}s_j^2$.

<details><summary>solution</summary>

$X_cV_kV_k^\top = U\Sigma V^\top V_kV_k^\top = \sum_{j \le k}s_ju_jv_j^\top$ (since $V^\top V_k$ selects the first $k$ columns). so $X_c - \hat X_c = \sum_{j > k}s_ju_jv_j^\top$, and because the rank-one terms are orthonormal in the Frobenius inner product, $\|\cdot\|_F^2 = \sum_{j > k}s_j^2$. Eckart–Young (chapter 02) says no rank-$k$ matrix is closer, and projecting onto any $k$-dimensional subspace gives a rank-$k$ matrix, so PCA's subspace is optimal.

</details>

3. **PPCA's covariance.** with $\hat W = U_k(\Lambda_k - \sigma^2I)^{1/2}$, show that $C = \hat W\hat W^\top + \sigma^2I$ has eigenvalues $\lambda_1, \dots, \lambda_k$ (with eigenvectors $u_1, \dots, u_k$) and $\sigma^2$ (with multiplicity $d - k$).

<details><summary>solution</summary>

$\hat W\hat W^\top = U_k(\Lambda_k - \sigma^2I)U_k^\top$. for $j \le k$: $Cu_j = (\lambda_j - \sigma^2)u_j + \sigma^2u_j = \lambda_ju_j$. for any $w \perp u_1, \dots, u_k$: $\hat W^\top w = 0$, so $Cw = \sigma^2w$. the model matches the data's covariance exactly in the top $k$ directions and replaces the rest by their average, $\hat\sigma^2$. (that the average is the MLE comes from maximizing $-\frac n2[\log\det C + \operatorname{tr}(C^{-1}S)]$ over a common value for the discarded eigenvalues.)

</details>

4. **kernel PCA's normalization.** the direction $v = \sum_ia_i\tilde\phi(x_i)$ with $\tilde Ka = \lambda a$. show that $\|v\|^2 = \lambda\|a\|^2$, so $a = \alpha/\sqrt\lambda$ for a unit eigenvector $\alpha$ gives $\|v\| = 1$, and that the score of training point $j$ is $\sqrt\lambda\,\alpha_j$.

<details><summary>solution</summary>

$\|v\|^2 = \sum_{i,l}a_ia_l\tilde\phi(x_i)^\top\tilde\phi(x_l) = a^\top\tilde Ka = \lambda a^\top a$. with $a = \alpha/\sqrt\lambda$, $\|v\|^2 = \lambda\cdot\frac1\lambda = 1$. the score of $x_j$ is $v^\top\tilde\phi(x_j) = \sum_ia_i\tilde K_{ij} = (\tilde Ka)_j = \lambda a_j = \sqrt\lambda\,\alpha_j$.

</details>

## experiment

`lab.py` uses your PCA family:

1. PCA on the Olivetti faces: explained variance versus the reconstruction error of held-out faces.
2. choosing the dimension: held-out reconstruction error versus held-out PPCA log-likelihood, with 5 true latent dimensions.
3. kernel PCA on two concentric rings, sweeping the RBF width.

## check your understanding

1. why must PCA centre the data, and what does an SVD of uncentred data find instead?

<details><summary>answer</summary>

PCA maximizes variance, which is measured around the mean. without centring, the first singular vector points (roughly) at the mean itself, because $\|Xv\|^2$ includes $n(\bar x^\top v)^2$; the result describes the data's offset from the origin, not its spread. (for some data, like word counts, the uncentred SVD is used on purpose: that's latent semantic analysis / truncated SVD.)

</details>

2. you standardize features before PCA. when is that the right call and when the wrong one?

<details><summary>answer</summary>

right when features have arbitrary, incomparable units (dollars and years): otherwise the first component just points along the largest-unit feature (warm-up 3). wrong (or unnecessary) when features share a unit and their variances carry meaning, like pixels: standardizing would boost the near-constant border pixels to unit variance and make PCA fit noise there.

</details>

3. `explained_variance_ratio_` sums to 0.95 for 110 components. what has been "explained", exactly?

<details><summary>answer</summary>

95% of the total variance of the *training* data lies in the span of those 110 directions; equivalently, reconstructing the training points from 110 numbers each leaves 5% of their variance as squared error. it says nothing directly about new data, about which components matter for a downstream task, or about "explanation" in a causal sense.

</details>

4. why does PPCA's noise variance equal the *average* of the discarded eigenvalues, and what does that assume?

<details><summary>answer</summary>

the model has one isotropic noise level for all $d - k$ remaining directions, and the likelihood is maximized by matching it to their mean variance. it assumes the leftover variation really is the same in every direction (isotropic noise). if some discarded directions have much more variance than others, the model is misspecified; factor analysis (a diagonal noise covariance) relaxes this.

</details>

5. kernel PCA needs an $n \times n$ eigendecomposition. what does that cost for $n = 60{,}000$, and what do people do instead?

<details><summary>answer</summary>

the Gram matrix alone has $3.6 \times 10^9$ entries (29 GB in float64), and a dense eigendecomposition is $O(n^3)$. the standard approximations are Nyström (eigendecompose a landmark subset and extend) and random Fourier features (an explicit finite feature map approximating the RBF kernel, then ordinary PCA). chapter 09 builds both ideas.

</details>

6. why can't you `inverse_transform` kernel PCA scores the way you do for PCA?

<details><summary>answer</summary>

the reconstruction would be a point in feature space, $\sum_ja_j\tilde\phi(\cdot)$, which in general isn't $\phi(x)$ of any input $x$. finding an input whose image is close to it is the "pre-image problem", solved only approximately (e.g. by learning a regression from scores back to inputs).

</details>

## common bugs

- **forgetting to centre**, or centring test data with its own mean. symptom: components that point at the mean; test scores that are shifted.
- **`components_` as columns.** the convention is one component per *row*, $(k, d)$. symptom: shapes that only work when $k = d$, or transposed scores.
- **explained variance ratio over the kept components only.** symptom: ratios that always sum to 1, whatever $k$.
- **comparing eigenvectors without fixing signs.** symptom: a test or a downstream plot that flips between runs or library versions.
- **PPCA with $S$ divided by $n - 1$.** symptom: a model that's close but isn't the MLE; its likelihood loses to the closed form.
- **kernel PCA without centring the Gram matrix** (or centring new points with their own statistics). symptom: a first component that just measures each point's average similarity to everything; new-point embeddings shifted relative to the training ones.

## going deeper

- Pearson (1901), "On lines and planes of closest fit to systems of points in space", *Philosophical Magazine* 2(11): PCA as best-fitting subspace. [doi:10.1080/14786440109462720](https://doi.org/10.1080/14786440109462720)
- Hotelling (1933), "Analysis of a complex of statistical variables into principal components", *Journal of Educational Psychology* 24. [doi:10.1037/h0071325](https://doi.org/10.1037/h0071325)
- Tipping & Bishop (1999), "Probabilistic principal component analysis", *JRSS B* 61(3). [doi:10.1111/1467-9868.00196](https://doi.org/10.1111/1467-9868.00196)
- Schölkopf, Smola & Müller (1998), "Nonlinear component analysis as a kernel eigenvalue problem", *Neural Computation* 10(5). [doi:10.1162/089976698300017467](https://doi.org/10.1162/089976698300017467)
- Turk & Pentland (1991), "Eigenfaces for recognition", *Journal of Cognitive Neuroscience* 3(1): PCA on faces. [doi:10.1162/jocn.1991.3.1.71](https://doi.org/10.1162/jocn.1991.3.1.71)
- Minka (2000), "Automatic choice of dimensionality for PCA", *NeurIPS*: a Bayesian answer to "how many components". [papers.nips.cc/paper/1853](https://papers.nips.cc/paper/1853-automatic-choice-of-dimensionality-for-pca)
- Jolliffe & Cadima (2016), "Principal component analysis: a review and recent developments", *Phil. Trans. R. Soc. A* 374. [doi:10.1098/rsta.2015.0202](https://doi.org/10.1098/rsta.2015.0202)

## connections

- ← builds on: the SVD and Eckart–Young (ch 02 · topic 06), eigendecompositions (ch 02 · topic 05), the multivariate Gaussian (ch 04), latent-variable models (topic 02), transformers and feature maps (ch 07 · topic 05).
- → used in: classical MDS, which is PCA from distances, and Isomap (topic 04); PCA before t-SNE and clustering in the capstone; spectral methods (topic 05); kernels, the Nyström method and random features (ch 09); variational autoencoders, a nonlinear relative of probabilistic PCA (ch 21).
