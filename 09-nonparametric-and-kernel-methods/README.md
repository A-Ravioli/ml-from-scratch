# 09 · nonparametric & kernel methods

> learn functions without fixing their form, and understand the kernel trick.

every model so far chose its shape before seeing the data: a line, a polynomial of some degree, a stump. this chapter's models let the data decide. a nearest-neighbour classifier has no training step at all; a kernel density estimate puts a bump on every point; an SVM, kernel ridge regression and a Gaussian process all fit functions from an infinite-dimensional space, computed through one idea, the **kernel trick**: algorithms that only touch inner products can swap them for a kernel, and a kernel can be an inner product of infinitely many features. you'll build all of it, then confront the price. nearest neighbours break down as distances concentrate in high dimension; kernel methods cost $O(n^3)$, which random Fourier features and Nyström approximations cut to $O(nD^2)$. the chapter ends with the most useful property of the Bayesian kernel machine: a Gaussian process knows where it's uncertain, which is exactly what you need to tune hyperparameters with a handful of expensive evaluations.

**you will build:** Minkowski distances and brute-force neighbour search; `KNeighborsClassifier` and `KNeighborsRegressor` with uniform and distance weights; a KD-tree with median splits and exact pruned search; histograms, Gaussian/Epanechnikov/top-hat kernels in $d$ dimensions, a log-space `KernelDensity` with sampling, Silverman's rule and cross-validated bandwidth selection; the linear, polynomial, RBF and Laplacian kernels, an explicit quadratic feature map, `get_kernel`, PSD checks, Gram-matrix centring and the minimum-norm interpolant of the representer theorem; the hinge loss, the SVM dual and its KKT conditions, the SMO pair update and a kernel `SVC`; `KernelRidge`, `RandomFourierFeatures`, `Nystroem` and ridge on approximate features; RBF and Matérn kernels, GP posterior mean, covariance and samples, the log marginal likelihood with its analytic gradient, hyperparameters by L-BFGS with restarts, and expected improvement. module: **`mlfs.kernels`** (all six topics).
**prerequisites:** chapter 00 (vectorized distances, log-sum-exp, `as_rng`), chapter 02 (PSD matrices, eigendecompositions, Cholesky, least squares and the pseudoinverse, matrix calculus), chapter 04 (multivariate Gaussians and conditioning, Monte Carlo), chapter 05 (maximum likelihood, KL divergence, the bootstrap), chapter 06 (Lagrangians and KKT conditions, L-BFGS) and chapter 07 (the estimator protocol, cross-validation, scalers). chapter 08's linear models help as a comparison, but nothing here uses them.
**time · scaffolding:** about 75 hours · L3 (independent): every function and method comes with its signature and a one-line docstring; the conventions the checks expect are in each lesson's ▶ build callouts. hints have two levels. the capstone gives you the data loading; you write the search.
**newly unlocked library components:** after this chapter you may use scikit-learn's `neighbors` (`KNeighborsClassifier`, `KNeighborsRegressor`, `KDTree`, `KernelDensity`), `metrics.pairwise` kernels, `svm.SVC`, `kernel_ridge.KernelRidge`, `kernel_approximation` (`RBFSampler`, `Nystroem`) and `gaussian_process.GaussianProcessRegressor` in labs and experiments, since you've built what they implement. inside `mlfs` modules, later chapters use *your* versions through `mlfs.kernels`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [k-nearest neighbors](01-k-nearest-neighbors/) | `minkowski_distances`, `brute_kneighbors`, `KNeighborsClassifier`, `KNeighborsRegressor`, `relative_contrast`, `edge_length`, `KDTree` → `mlfs.kernels` | in 784 dimensions the farthest of 1,000 uniform points is only 13% farther than the nearest, but for real 784-pixel digits it's 2.5× farther; the exact KD-tree computes 1% of the distances in 2-D and all of them from 16-D on; 1-NN on raw pixels gets 93% on MNIST with no training |
| 02 | [density estimation](02-density-estimation/) | `histogram_density`, `log_kernel`, `KernelDensity`, `silverman_bandwidth`, `select_bandwidth` → `mlfs.kernels` | the same 39 points and bin width give histograms with 2 to 6 peaks depending only on the origin; Silverman's rule oversmooths a bimodal density by 3× (MISE 8× the optimum) while CV hits the optimum; on a map of California, CV picks a bandwidth 5× narrower than the rule of thumb |
| 03 | [kernels & RKHS](03-kernels-and-rkhs/) | `linear_kernel`, `polynomial_kernel`, `rbf_kernel`, `laplacian_kernel`, `quadratic_features`, `get_kernel`, `is_psd`, `center_gram` → `mlfs.kernels`, `interpolate`, `kernel_expansion`, `rkhs_norm` | explicit degree-4 features need 14,641 numbers per point (and are roughly 10–30× slower) for what the kernel computes from 10; $\exp(-\|x-y\|^4)$ isn't a kernel, and the triangle and $\cos\|x-y\|$ are kernels on a line but not in the plane; the spikiest RBF interpolant has the *smallest* RKHS norm |
| 04 | [support vector machines](04-support-vector-machines/) | `hinge_loss`, `primal_objective`, `dual_objective`, `kkt_violations`, `smo_pair_update`, `SVC` → `mlfs.kernels` | larger $C$ means *fewer* support vectors; refitting on the 60 support vectors of 200 points changes nothing (up to solver precision), while deleting one support vector moves the boundary 19× more; 4 vs 9 on MNIST: 2.6% test error, with a support-vector bound 14× looser |
| 05 | [kernel ridge & random features](05-kernel-ridge-and-random-features/) | `KernelRidge`, `RandomFourierFeatures`, `Nystroem` → `mlfs.kernels`, `primal_ridge`, `gram_error` | Nyström approximates a real Gram matrix 8–20× better than random features with the same number of features, yet the two give the same test error at $D = 100$; 1,000 Nyström features match exact kernel ridge on 4,000 points to 0.4% |
| 06 | [gaussian processes](06-gaussian-processes/) | `stationary_kernel`, `GaussianProcessRegressor` (posterior, samples, evidence and gradient, L-BFGS with restarts), `expected_improvement` → `mlfs.kernels` | from a long length scale, L-BFGS decides the data are pure noise and never finds the sine wave that's 15,000 times more likely; hand-picked hyperparameters give "95%" intervals that cover 80% of real house prices, learned ones 95.2% |

## capstone

*bayesian optimization*: GP + expected improvement to tune a model's hyperparameters. you tune the regularization and kernel width of your `KernelRidge` on California housing, where every evaluation is a 5-fold cross-validation, with your GP and expected improvement, and race it against random search over 6 seeds with the same budget of 30 evaluations. measured: tuning cuts the test RMSE from 0.733 (defaults) to about 0.59 (\$100,000 units); every BO run ends within 0.005 of a 256-point grid's optimum against half of the random-search runs, but on this broad valley the two tuned models are indistinguishable on the test set. small CPU config: the synthetic stand-in (offline) in well under a minute. full config: about 3 minutes.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 14 concept questions, three by-hand problems (a 3-NN vote and a top-hat KDE, an SMO step, a one-point GP posterior with its expected improvement), and a 60-minute blind rebuild of the kernel library, kernel ridge regression and the GP posterior and evidence.

## notation

| symbol | meaning |
|---|---|
| $X$, $x_i$, $n$, $d$ | the data matrix $(n, d)$, a training input, number of examples, number of features |
| $q$, $x_*$, $X_*$ | a query point; test inputs |
| $N_k(x)$, $k$ | the $k$ nearest training points to $x$; the number of neighbours (context distinguishes it from a kernel) |
| $d_p(x, y)$ | Minkowski distance $(\sum_j \|x_j - y_j\|^p)^{1/p}$ |
| $h$, $K_h$ | KDE bandwidth; a kernel scaled to bandwidth $h$, integrating to 1 |
| $V_d$ | volume of the unit ball in $\mathbb R^d$ |
| $k(x, y)$, $K$, $K_*$, $K_{**}$ | a kernel (PSD similarity); Gram matrices train–train, train–test, test–test |
| $\varphi(x)$, $\mathcal H$, $\|f\|_{\mathcal H}$ | feature map; reproducing kernel Hilbert space; its norm |
| $\gamma$, $p$, $c_0$ | RBF/Laplacian width (also the polynomial kernel's scale); polynomial degree; offset |
| $H = I - \frac1n\mathbf 1\mathbf 1^\top$ | the centring matrix |
| $y_i \in \{-1, +1\}$, $f(x) = w^\top\varphi(x) + b$ | SVM labels and decision function |
| $\alpha_i$, $C$, $\xi_i$, $E_i$, $\eta$ | SVM dual variables, box bound, slacks, errors $f(x_i) - y_i$, SMO curvature $2K_{ij} - K_{ii} - K_{jj}$ |
| $\alpha$ (kernel ridge), $\beta$ | ridge strength; dual coefficients $(K + \alpha I)^{-1}y$ |
| $D$, $\omega$, $b$, $z(x)$ | number of approximate features; random frequencies and phases; an approximate feature map |
| $L$, $W = K_{LL}$, $m$ | Nyström landmarks, their Gram matrix, their number |
| $\ell$, $\sigma_f^2$, $\sigma_n^2$, $\theta$ | GP length scale, signal variance, noise variance; $\theta = \log(\ell, \sigma_f^2, \sigma_n^2)$ |
| $\mu$, $\sigma$, $f_{\text{best}}$, $\xi$ (EI) | GP posterior mean and std at a candidate; best value so far; the improvement margin |
| $\Phi$, $\varphi$ (normal) | standard normal CDF and density (context distinguishes $\varphi$ from a feature map) |

## how this connects

- builds on: vectorized distances and log-sum-exp (ch 00), PSD matrices, eigendecompositions, Cholesky and least squares (ch 02), Gaussians and conditioning (ch 04), maximum likelihood, KL divergence and the bootstrap (ch 05), duality, KKT conditions and L-BFGS (ch 06), the estimator protocol, cross-validation and hyperparameter search (ch 07).
- used later in: learning theory, which explains margins and kernel classes (ch 13), $k$-means, kernel PCA, Isomap, $t$-SNE and density-based clustering (ch 11), Bayesian linear regression and the evidence (ch 12), the neural tangent kernel and NNGP limits, which compare wide networks with your kernel ridge and GP (ch 19 · topic 03), kernel two-sample tests (ch 21), the $k$-NN probe for representations (ch 25), retrieval (ch 27), and calibration of uncertainty (ch 28).

## core references

- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, chapters 2, 6, 12 and 13 (local methods, kernel smoothing and density estimation, SVMs, nearest neighbours). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Schölkopf & Smola (2002), *Learning with Kernels*, MIT Press. kernels, the RKHS, SVMs and their implementation. [doi:10.7551/mitpress/4175.001.0001](https://doi.org/10.7551/mitpress/4175.001.0001)
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press. free online. [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml/)
- Wasserman (2006), *All of Nonparametric Statistics*, Springer. density estimation and nonparametric regression, with rates. [doi:10.1007/0-387-30623-4](https://doi.org/10.1007/0-387-30623-4)
- Rahimi & Recht (2007), "Random features for large-scale kernel machines", *NeurIPS*. [papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines)
- Snoek, Larochelle & Adams (2012), "Practical Bayesian optimization of machine learning algorithms", *NeurIPS*. [arXiv:1206.2944](https://arxiv.org/abs/1206.2944)
