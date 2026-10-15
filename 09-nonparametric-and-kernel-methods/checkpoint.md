# 09 · nonparametric & kernel methods · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. what makes a method "nonparametric"? give two examples from this chapter and say what grows with the data in each.

<details><summary>answer</summary>

its complexity isn't fixed in advance: the number of effective parameters grows with $n$. $k$-NN and KDE store the whole training set; kernel ridge regression and GPs have one coefficient per training point ($\beta = (K + \alpha I)^{-1}y$); an SVM keeps its support vectors. a fixed-degree polynomial is parametric however many points you fit. (topic 01, idea 2; topic 05, idea 1)

</details>

2. why does $k$-NN degrade in high dimension, and why does it still work on MNIST?

<details><summary>answer</summary>

with $d$ independent coordinates, squared distances are sums of $d$ terms that concentrate: the relative contrast between the nearest and farthest point shrinks like $1/\sqrt d$, and a neighbourhood holding a fixed fraction of the data spans most of every axis. MNIST digits have a low *intrinsic* dimension (they lie near a thin, curved set in pixel space), so their neighbours are genuinely close. (topic 01, idea 3 and lab)

</details>

3. a KD-tree is exact. why does it compute almost every distance in 20 dimensions?

<details><summary>answer</summary>

it skips a subtree only when the distance from the query to the splitting plane, a single coordinate gap, exceeds the current $k$-th best distance, which sums over all coordinates and is barely smaller than the distance to everything else. with $n$ points the tree is only $\log_2(n/\text{leaf size})$ levels deep, so most coordinates are split rarely. rule of thumb: useful while $n \gg 2^d$. (topic 01, idea 5)

</details>

4. why can't you choose a KDE bandwidth by maximizing the likelihood of the training data? what do you do instead?

<details><summary>answer</summary>

as $h \to 0$ each training point sits under its own spike and its density goes to infinity, so the training likelihood always prefers the smallest bandwidth. maximize the *held-out* log-likelihood by cross-validation, which estimates (minus) the KL divergence from the true density plus a constant. (topic 02, idea 5)

</details>

5. Silverman's rule picks a bandwidth three times larger than the MISE-optimal one on a bimodal density. why?

<details><summary>answer</summary>

it's the optimal bandwidth *for a Gaussian* with the sample's standard deviation. a bimodal density has a large standard deviation (from the gap between modes) but sharp curvature (from narrow modes), so the rule underestimates $\int p''^2$ and oversmooths. (topic 02, lab)

</details>

6. state Mercer's condition in the form you'd check numerically, and explain why every inner product of feature vectors satisfies it.

<details><summary>answer</summary>

$k$ is symmetric and every Gram matrix $K_{ij} = k(x_i, x_j)$ is positive semi-definite. if $k(x, y) = \varphi(x)^\top\varphi(y)$, then $c^\top Kc = \|\sum_i c_i\varphi(x_i)\|^2 \ge 0$. numerically: symmetric, and smallest eigenvalue $\ge -\text{tol}\cdot\max(1, |\lambda_{\max}|)$; a failure on any sample disproves it, a pass proves nothing. (topic 03, idea 4)

</details>

7. what does the representer theorem say, and why does it make kernel methods computable?

<details><summary>answer</summary>

a minimizer of (loss on the training points) + (increasing function of $\|f\|_{\mathcal H}$) over the whole RKHS can be written $f = \sum_{i=1}^n\alpha_i k(x_i, \cdot)$. so an optimization over a possibly infinite-dimensional function space reduces to $n$ coefficients, computable from the Gram matrix. (topic 03, idea 5)

</details>

8. what are the support vectors of an SVM, and what happens to the solution if you delete every other training point?

<details><summary>answer</summary>

the points with $\alpha_i > 0$: on the margin ($0 < \alpha_i < C$) or inside it or misclassified ($\alpha_i = C$). deleting points with $\alpha_i = 0$ leaves the solution unchanged, since the old $\alpha$ is still feasible and satisfies the KKT conditions. (topic 04, idea 2 and lab)

</details>

9. why does SMO optimize two dual variables at a time, and what does the box $0 \le \alpha_i \le C$ do to the update?

<details><summary>answer</summary>

the equality constraint $\sum_i\alpha_i y_i = 0$ means one variable can't move alone; with two, the constraint fixes a line along which the dual is a 1-D concave quadratic with a closed-form maximizer. the box clips that maximizer to a segment $[L, H]$. (topic 04, idea 3)

</details>

10. how do $C$ and $\gamma$ change an RBF SVM?

<details><summary>answer</summary>

$\gamma$ sets the reach of each support vector: large $\gamma$, wiggly boundaries that can wrap individual points; small $\gamma$, nearly linear. $C$ sets the price of margin violations: large $C$, narrow margin and few violations (low bias, high variance); small $C$, wide margin with many points inside it (and many support vectors). (topic 04, lab)

</details>

11. kernel ridge regression costs $O(n^3)$. name two ways around it, and say which one adapts to the data.

<details><summary>answer</summary>

random Fourier features (sample frequencies from the kernel's spectral density; error $\sim 1/\sqrt D$ regardless of the data) and the Nyström method (features from $m$ random landmark points, $z(x) = K(x, L)K_{LL}^{-1/2}$; error follows the eigenvalue decay of $K$). Nyström adapts to the data; both make ridge regression $O(nD^2)$. (topic 05, ideas 2–3)

</details>

12. what's the relation between the GP posterior mean and kernel ridge regression?

<details><summary>answer</summary>

with the same kernel, the GP posterior mean $K_*^\top(K + \sigma_n^2 I)^{-1}y$ is the KRR predictor with $\alpha = \sigma_n^2$ (or $\sigma_n^2/\sigma_f^2$ if the kernel is written with a separate variance). the GP adds the posterior covariance and the marginal likelihood. (topic 06, idea 2)

</details>

13. the log marginal likelihood has a data-fit term and a complexity term. which is which, and why does it not always prefer the shortest length scale?

<details><summary>answer</summary>

$-\frac12 y^\top K_y^{-1}y$ rewards fitting the data; $-\frac12\log\det K_y$ penalizes flexibility (a flexible model spreads probability over many datasets). short length scales improve the fit but increase $\log\det K_y$, so the evidence peaks at the simplest model that explains the data. (topic 06, idea 4)

</details>

14. why does expected improvement explore?

<details><summary>answer</summary>

$\mathrm{EI} = (\mu - f_\text{best} - \xi)\Phi(z) + \sigma\varphi(z)$ grows with $\sigma$ at a fixed mean: a point with a mediocre mean but large uncertainty has a real chance of beating the incumbent. (at $\sigma = 0$ the convention is EI = 0; the true limit is $\max(\mu - f_\text{best} - \xi, 0)$, which is also 0 at the evaluated points of a noise-free GP, where $\mu \le f_\text{best}$.) (topic 06, idea 6)

</details>

## 2. by hand

1. **a KDE and a 3-NN prediction.** data on a line: $x = (0, 1, 3, 4)$ with labels $(0, 0, 1, 1)$. (a) classify $q = 2.2$ with 3-NN, uniform and distance-weighted. (b) compute the top-hat KDE with bandwidth $h = 1.5$ at $x = 2$.

<details><summary>solution</summary>

(a) distances from 2.2: $2.2, 1.2, 0.8, 1.8$, so the 3 nearest are $x = 3$ (0.8, label 1), $x = 1$ (1.2, label 0), $x = 4$ (1.8, label 1). uniform: 2 votes for 1, $\hat p = (1/3, 2/3)$, predict 1. distance weights $1/0.8 = 1.25$, $1/1.2 = 0.833$, $1/1.8 = 0.556$: class 1 gets $1.806$, class 0 gets $0.833$, $\hat p_1 = 0.684$, predict 1. (b) the 1-D top-hat kernel is $\frac{1}{2h} = \frac13$ for $|x - x_i| < 1.5$: points 1 and 3 qualify (distance 1), 0 and 4 don't (distance 2). $\hat p(2) = \frac14(\frac13 + \frac13) = \frac16$.

</details>

2. **an SMO step.** linear kernel, points $x_1 = (1, 0)$ with $y_1 = +1$ and $x_2 = (-1, 0)$ with $y_2 = -1$, $C = 1$, starting from $\alpha = 0$, $b = 0$. perform the update of the pair $(1, 2)$ and give $w$, $b$ and the margin.

<details><summary>solution</summary>

$K = \begin{pmatrix}1 & -1\\ -1 & 1\end{pmatrix}$, $E = f - y = (-1, 1)$. labels differ: $L = 0$, $H = \min(1, 1) = 1$. $\eta = 2(-1) - 1 - 1 = -4$. $\alpha_2^\text{new} = 0 - (-1)(-1 - 1)/(-4) = 0.5$, inside $[0, 1]$; $\alpha_1^\text{new} = 0 + (1)(-1)(0 - 0.5) = 0.5$. $b_1 = 0 - (-1) - (1)(0.5)(1) - (-1)(0.5)(-1) = 1 - 0.5 - 0.5 = 0$; both free, so $b = 0$. $w = 0.5(1, 0) + 0.5 \cdot (-1)(-1, 0) = (1, 0)$: boundary $x_1 = 0$, margin $1/\|w\| = 1$, exactly the distance from each point to the boundary. ✓

</details>

3. **a GP posterior with one observation.** RBF kernel with $\ell = 1$, $\sigma_f^2 = 1$, noise $\sigma_n^2 = 0.25$, one observation $y = 2$ at $x = 0$, zero prior mean (no centring). compute the posterior mean and standard deviation at $x_* = 0$ and $x_* = 1$, and the expected improvement at $x_* = 1$ over $f_\text{best} = 1.5$ with $\xi = 0$.

<details><summary>solution</summary>

$K + \sigma_n^2 = 1.25$, $\alpha = 2/1.25 = 1.6$. at 0: $k_* = 1$, mean $1.6$, variance $1 - 1/1.25 = 0.2$, std $0.447$. at 1: $k_* = e^{-1/2} = 0.6065$, mean $0.970$, variance $1 - 0.3679/1.25 = 0.7057$, std $0.840$. EI at 1: $z = (0.970 - 1.5)/0.840 = -0.631$, $\Phi(z) = 0.264$, $\varphi(z) = 0.327$; $\mathrm{EI} = -0.530 \times 0.264 + 0.840 \times 0.327 = -0.140 + 0.275 = 0.135$.

</details>

## 3. blind rebuild (60 minutes)

with no notes and no looking at your old code, reimplement in a fresh copy of the notebooks:

- **the kernel library** (topic 03, stages 1 and 3): `linear_kernel`, `polynomial_kernel`, `rbf_kernel`, `laplacian_kernel` and `get_kernel`;
- **kernel ridge regression** (topic 05, stage 1);
- **the GP posterior and evidence** (topic 06, stages 2 and 4): `fit` without optimization, `predict` with `return_std` and `return_cov`, `log_marginal_likelihood` and `lml_gradient`.

then run the checks:

```bash
pytest 09-nonparametric-and-kernel-methods/03-kernels-and-rkhs -k "stage1 or stage3"
pytest 09-nonparametric-and-kernel-methods/05-kernel-ridge-and-random-features -k stage1
pytest 09-nonparametric-and-kernel-methods/06-gaussian-processes -k "stage2 or stage4"
```

(copy the notebooks somewhere first, or stash your solutions, so you really start from the stubs.) if you want a harder rebuild, add SMO (topic 04, stages 3 and 4) and give yourself another 45 minutes.

## 4. readiness

move on to chapter 10 if you got at least 80% of sections 1 and 2 together (say, 11 of the 14 concept questions and all three problems, or 12 and two of them) and every test in section 3 passes. the same checkpoint works as a placement test: if you pass it cold, skim the lessons for the conventions (`get_kernel` and its ignored parameters, `dual_coef_` as $\alpha_i y_i$ for the SVM but $(K + \alpha I)^{-1}y$ for kernel ridge, GP hyperparameters as $\log(\ell, \sigma_f^2, \sigma_n^2)$ with the noise as a variance), make sure your `mlfs.kernels` passes its tests, and go on.
