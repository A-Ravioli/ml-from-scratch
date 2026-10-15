# gaussian mixtures & EM

## why this matters

k-means gives every point exactly one cluster, and assumes every cluster is a round ball of the same size. real groups overlap, stretch and differ in spread: a point halfway between two clusters really is uncertain, and an elongated cluster isn't a ball. a **Gaussian mixture model** fixes both. it says the data came from $k$ Gaussians, each with its own mean, covariance and share of the data, and it answers "which cluster?" with probabilities. it's also a *density*: it can tell you how likely a new point is (useful for anomaly detection, topic 06) and generate new data.

fitting it is where the chapter's most important algorithm appears. you don't know which Gaussian produced each point; if you did, fitting would be trivial, and if you knew the Gaussians, guessing the assignments would be trivial. **expectation-maximization** (EM) alternates between the two, and it comes with a guarantee: the likelihood never goes down. the same recipe fits hidden Markov models (chapter 12), and the bound it climbs, the ELBO, is the objective of variational inference and of variational autoencoders (chapters 12 and 21). you'll also see the guarantee's limits: EM climbs *a* hill, sometimes stalls for hundreds of iterations on a plateau, and a likelihood that can be made infinite needs a guard rail.

## warm-up: predict first

1. a point sits exactly halfway between two equally weighted Gaussians with the same covariance. what are its responsibilities? and if one Gaussian is much wider?
2. EM guarantees that the log-likelihood never decreases. does it guarantee that you end at the maximum likelihood estimate?
3. you start EM with every component's responsibilities nearly equal (all components almost identical). what happens in the first iterations?
4. one component sits exactly on a single data point and its variance shrinks toward 0. what happens to the likelihood?
5. k-means and a GMM with spherical covariances $\sigma^2 I$, equal weights, and $\sigma \to 0$: how are they related?

## prerequisites check

- [ ] you can evaluate a multivariate Gaussian log density with a Cholesky factor (→ ch 04 · topic 04, ch 02 · topic 03)
- [ ] you know maximum likelihood for a Gaussian: the sample mean and the biased sample covariance (→ ch 05 · topic 02)
- [ ] you can compute $\log\sum_j e^{a_j}$ without overflow (→ ch 00 · topic 03, `logsumexp`)
- [ ] you know KL divergence and that it's non-negative (→ ch 05 · topic 05)
- [ ] you've built `KMeans` (→ topic 01)

## core ideas

### idea 1: a mixture of Gaussians

a GMM generates a point in two steps. first pick a component $z \in \{1, \dots, k\}$ with probabilities $\pi_1, \dots, \pi_k$ (the **weights**, $\pi_j \ge 0$, $\sum_j\pi_j = 1$). then draw $x \sim \mathcal N(\mu_z, \Sigma_z)$. the component $z$ is a **latent variable**: it exists in the story but you never observe it. the density of $x$ sums over it:

$$p(x) = \sum_{j=1}^k \pi_j\,\mathcal N(x \mid \mu_j, \Sigma_j), \qquad \log\mathcal N(x \mid \mu, \Sigma) = -\tfrac12\left(d\log 2\pi + \log\det\Sigma + (x - \mu)^\top\Sigma^{-1}(x - \mu)\right).$$

the covariances decide the shapes. **full**: any ellipse, $d(d+1)/2$ numbers per component. **diag**: ellipses aligned with the axes, $d$ variances per component. **spherical**: balls, one variance per component. fewer parameters means less flexibility but more stability, and it matters a lot when $d$ is large: a full covariance in 784 dimensions has 307,720 parameters.

you'll need $\log\mathcal N$ for every point and component, and you must compute it in log space from the start: a point 40 standard deviations away has density $e^{-800}$, which is 0 in floating point, and $\log 0 = -\infty$ ruins everything downstream. for a full covariance, factor $\Sigma = LL^\top$ (Cholesky); then $\log\det\Sigma = 2\sum_i\log L_{ii}$ and the Mahalanobis term is $\|z\|^2$ with $Lz = x - \mu$, a triangular solve. never form $\Sigma^{-1}$. for diag and spherical, it's elementwise division.

> **▶ build: stage 1.** implement `gaussian_log_density(X, means, covariances, covariance_type="full")` → $(n, k)$ log densities, for covariances stored as $(k, d, d)$, $(k, d)$ or $(k,)$.

### idea 2: the E-step: who's responsible?

suppose you knew the parameters $\theta = (\pi, \mu, \Sigma)$. which component produced $x_i$? Bayes' rule gives the posterior over the latent variable, called the **responsibility** of component $j$ for point $i$:

$$r_{ij} = p(z_i = j \mid x_i) = \frac{\pi_j\,\mathcal N(x_i \mid \mu_j, \Sigma_j)}{\sum_l \pi_l\,\mathcal N(x_i \mid \mu_l, \Sigma_l)}.$$

each row of $r$ is a probability distribution over components: a soft assignment. a point halfway between two identical, equally weighted components gets $(\tfrac12, \tfrac12)$ (warm-up 1); if one is much wider, its tails are fatter, and the wide one takes most of the responsibility for points far from both centres.

compute it in log space: $\log r_{ij} = a_{ij} - \log\sum_l e^{a_{il}}$ with $a_{ij} = \log\pi_j + \log\mathcal N(x_i \mid \mu_j, \Sigma_j)$, and the `logsumexp` from chapter 00. the same normalizer is the log-likelihood of the point, $\log p(x_i) = \log\sum_l e^{a_{il}}$, so the E-step gets the data's log-likelihood for free. report it as a **mean** over points, so numbers are comparable across data sets of different sizes.

> **▶ build: stage 2.** implement `e_step(X, weights, means, covariances, covariance_type="full")` → `(resp, mean_loglik)`, stable for points far from every component.

### idea 3: the M-step: weighted maximum likelihood

now suppose you knew the assignments. with hard labels, fitting is just per-cluster maximum likelihood: weights are the cluster fractions, means are the cluster means, covariances the cluster covariances. with soft responsibilities, every point counts toward every component with weight $r_{ij}$. writing $N_j = \sum_i r_{ij}$ (the "effective number of points" in component $j$):

$$\pi_j = \frac{N_j}{n}, \qquad \mu_j = \frac{1}{N_j}\sum_i r_{ij}\,x_i, \qquad \Sigma_j = \frac{1}{N_j}\sum_i r_{ij}\,(x_i - \mu_j)(x_i - \mu_j)^\top .$$

these maximize the **expected complete-data log-likelihood** $Q(\theta) = \sum_i\sum_j r_{ij}\left[\log\pi_j + \log\mathcal N(x_i \mid \mu_j, \Sigma_j)\right]$, the log-likelihood you'd have if you knew the labels, averaged over the posterior (on paper 2). for diag, keep only the diagonal of $\Sigma_j$; for spherical, average the diagonal into one variance.

**the guard rail.** a component can collapse onto a single point: put $\mu_j = x_1$ and let its variance shrink, and $\mathcal N(x_1 \mid \mu_j, \sigma^2 I) \propto \sigma^{-d} \to \infty$ while the other components keep every other point's density positive. the likelihood is *unbounded* (warm-up 4, on paper 4): the maximum likelihood estimate of a GMM doesn't exist. the practical fix is to add a small `reg_covar` to every variance (to the diagonal of $\Sigma_j$), so no variance goes below it. also add a tiny $10\varepsilon$ to each $N_j$ so an empty component doesn't divide by zero.

> **🔮 predict:** you start EM from *random* responsibilities, each row a random distribution over 6 components. at the first M-step, how different are the 6 components' means?

> **▶ build: stage 3.** implement `m_step(X, resp, covariance_type="full", reg_covar=1e-6)` → `(weights, means, covariances)`, with `reg_covar` added to every variance.

### idea 4: EM, and why it climbs

**EM** alternates the two steps: compute responsibilities with the current parameters (E), refit the parameters with those responsibilities (M), repeat. why does this improve the actual objective, $\ell(\theta) = \sum_i\log p(x_i \mid \theta)$? for any distribution $q_i(z)$ over each point's latent variable,

$$\log p(x_i \mid \theta) = \underbrace{\sum_z q_i(z)\log\frac{p(x_i, z \mid \theta)}{q_i(z)}}_{\text{ELBO } \mathcal L(q_i, \theta)} + \underbrace{\mathrm{KL}\left(q_i(z)\,\|\,p(z \mid x_i, \theta)\right)}_{\ge 0} .$$

(on paper 3). so $\mathcal L$ is a lower bound on the log-likelihood, the **evidence lower bound**. the E-step sets $q_i$ to the posterior, which makes the KL zero: the bound touches $\ell$ at the current $\theta$. the M-step maximizes $\mathcal L$ over $\theta$ with $q$ fixed (that's $Q$ plus a constant), so the bound goes up, and $\ell$, which is above the bound, goes up at least as much. the likelihood never decreases (warm-up 2), and in practice it climbs quickly at first, then slowly. one caveat for your implementation: with `reg_covar` > 0 the M-step no longer maximizes $Q$ exactly (it's a regularized step that adds a floor to every variance), so the guarantee holds exactly only for `reg_covar = 0`. in practice the floor almost never binds; when a component collapses onto a few points and hits it, the log-likelihood can dip by a tiny amount (around $10^{-4}$).

what EM doesn't promise: the global maximum, or even a fast climb. the objective has many local maxima (relabelings of the same solution, plus genuinely worse ones), and saddle points. start all components nearly identical (random responsibilities make every mean the average of everything) and you sit near a symmetric saddle where every gradient is tiny; EM can creep along it for hundreds of iterations before the symmetry breaks (warm-up 3). a stopping rule like "the mean log-likelihood improved by less than `tol`" can't tell a plateau from a peak. the remedies are the same as for k-means: good starts (k-means labels as initial responsibilities, the default here) and restarts.

**k-means is a limit of EM.** take spherical covariances $\sigma^2 I$ shared by all components, equal weights, and let $\sigma \to 0$. the responsibilities become one-hot on the nearest mean (the ratio of two Gaussians is $e^{-(\|x-\mu_1\|^2 - \|x - \mu_2\|^2)/2\sigma^2}$, which goes to 0 or $\infty$), and the M-step means become cluster means: Lloyd's algorithm (warm-up 5).

your `fit` does: responsibilities from `_initial_resp` (given), an initial M-step, then up to `max_iters` iterations of E-step (record the mean log-likelihood in `lower_bound_history_`), M-step, and stop when the last two recorded values differ by less than `tol`. set `converged_` and `n_iter_`. the history is the log-likelihood of the parameters *before* each M-step, so it ends one step behind the final parameters, and `score(X)` is at least its last entry.

> **▶ build: stage 4.** implement `GaussianMixture.fit`, `predict_proba`, `predict`, `score_samples` and `score`. everything after `fit` uses only `weights_`, `means_` and `covariances_`.

### idea 5: sampling, and choosing k

a GMM is a **generative model**: sampling follows the story. draw a component for each sample from the weights, then $x = \mu_z + L_z\,\varepsilon$ with $\varepsilon \sim \mathcal N(0, I)$ and $L_zL_z^\top = \Sigma_z$ (Cholesky; for diag and spherical, $L$ is the square root of the variances). this is **ancestral sampling**: sample the parents, then the children given the parents.

the training log-likelihood can't choose $k$, for the same reason the inertia couldn't: more components never fit worse (with a good optimizer). two standard answers are held-out log-likelihood (fit on one part, `score` on another) and the **Bayesian information criterion**

$$\mathrm{BIC} = -2\sum_i\log p(x_i \mid \hat\theta) + p\log n,$$

where $p$ is the number of free parameters: $k - 1$ weights (they sum to 1), $kd$ means, and $kd(d+1)/2$, $kd$ or $k$ covariance parameters. lower is better. BIC approximates $-2\log$ of the model's marginal likelihood (the evidence, chapter 12) for large $n$; the $p\log n$ term penalizes parameters more heavily as the data grow, and it picks the true $k$ consistently when the model family is right. it has blind spots, and the lab finds one.

> **🔮 predict:** 400 points from four Gaussian blobs, plus six exact copies of one extra point. BIC picks $k$ from 1 to 8 with full covariances. does it still pick 4?

> **▶ build: stage 5.** implement `GaussianMixture.sample(n=1, *, rng=None)` → `(X, labels)` and `bic(X)` (the helper `n_parameters` is given).

## worked example

four points on a line, $x = 0, 1, 3, 4$, and two components with $\mu = (0, 4)$, variances 1, weights $(\tfrac12, \tfrac12)$.

**E-step.** the equal weights and variances cancel, so $r_{i1} = 1/(1 + e^{((x - 0)^2 - (x - 4)^2)/2}) = 1/(1 + e^{4x - 8})$:

| $x$ | 0 | 1 | 3 | 4 |
|---|---|---|---|---|
| $r_{i1}$ | 0.99966 | 0.98201 | 0.01799 | 0.00034 |

the mean log-likelihood is $\frac14\sum_i\log\left(\tfrac12\mathcal N(x_i \mid 0, 1) + \tfrac12\mathcal N(x_i \mid 4, 1)\right) = -1.85284$.

**M-step** (with `reg_covar = 0`). by symmetry $N_1 = N_2 = 2$, so the weights stay $(\tfrac12, \tfrac12)$. $\mu_1 = (0 \cdot 0.99966 + 1 \cdot 0.98201 + 3 \cdot 0.01799 + 4 \cdot 0.00034)/2 = 0.51866$ and $\mu_2 = 3.48134$. the variance $\sigma_1^2 = \sum_i r_{i1}(x_i - 0.51866)^2/2 = 0.30562$, the same for component 2. the next E-step's mean log-likelihood is $-1.42892$: up, as promised. EM goes on to $\mu = (0.5, 3.5)$, $\sigma^2 = 0.25$, the two pairs. the tests check these numbers.

## on paper

1. **responsibilities.** derive $r_{ij} = p(z_i = j \mid x_i)$ from the generative story with Bayes' rule, and show that for two components with equal weights and a shared covariance $\Sigma$, $\log(r_{i1}/r_{i2})$ is a linear function of $x_i$.

<details><summary>solution</summary>

$p(z = j \mid x) = p(z = j)p(x \mid z = j)/p(x) = \pi_j\mathcal N(x \mid \mu_j, \Sigma_j)/\sum_l\pi_l\mathcal N(x \mid \mu_l, \Sigma_l)$. with equal weights and $\Sigma_1 = \Sigma_2 = \Sigma$ the normalizing constants cancel and $\log\frac{r_1}{r_2} = -\frac12(x - \mu_1)^\top\Sigma^{-1}(x - \mu_1) + \frac12(x - \mu_2)^\top\Sigma^{-1}(x - \mu_2) = (\mu_1 - \mu_2)^\top\Sigma^{-1}x + \text{const}$: the quadratic terms cancel. the responsibility is a logistic function of a linear score, the link between GMMs and logistic regression (chapter 08). with different covariances the boundary is quadratic.

</details>

2. **the M-step.** maximize $Q = \sum_{i,j} r_{ij}[\log\pi_j + \log\mathcal N(x_i \mid \mu_j, \Sigma_j)]$ over $\mu_j$ and over $\pi$ (subject to $\sum_j\pi_j = 1$).

<details><summary>solution</summary>

$\partial Q/\partial\mu_j = \sum_i r_{ij}\Sigma_j^{-1}(x_i - \mu_j) = 0$ gives $\mu_j = \sum_i r_{ij}x_i/N_j$. for the weights, add a Lagrange multiplier: $\partial/\partial\pi_j\left[\sum_{i,l}r_{il}\log\pi_l + \lambda(1 - \sum_l\pi_l)\right] = N_j/\pi_j - \lambda = 0$, so $\pi_j = N_j/\lambda$, and $\sum_j\pi_j = 1$ forces $\lambda = \sum_jN_j = n$. (the covariance follows from the Gaussian MLE with weights: $\Sigma_j = \sum_ir_{ij}(x_i - \mu_j)(x_i - \mu_j)^\top/N_j$.)

</details>

3. **the ELBO identity.** show $\log p(x \mid \theta) = \sum_z q(z)\log\frac{p(x, z \mid \theta)}{q(z)} + \mathrm{KL}(q\,\|\,p(z \mid x, \theta))$ for any distribution $q$, and use it to prove that EM never decreases the log-likelihood.

<details><summary>solution</summary>

$p(x, z) = p(z \mid x)p(x)$, so $\log\frac{p(x,z)}{q(z)} = \log p(x) + \log\frac{p(z \mid x)}{q(z)}$. average over $q$: $\mathcal L(q, \theta) = \log p(x) - \mathrm{KL}(q\,\|\,p(z \mid x))$. let $\theta_t$ be the current parameters. the E-step sets $q = p(z \mid x, \theta_t)$, so $\mathcal L(q, \theta_t) = \ell(\theta_t)$. the M-step picks $\theta_{t+1}$ with $\mathcal L(q, \theta_{t+1}) \ge \mathcal L(q, \theta_t)$. then $\ell(\theta_{t+1}) = \mathcal L(q, \theta_{t+1}) + \mathrm{KL}(q\,\|\,p(z \mid x, \theta_{t+1})) \ge \mathcal L(q, \theta_{t+1}) \ge \ell(\theta_t)$.

</details>

4. **an unbounded likelihood.** a 1-D mixture of two components with $\mu_1 = x_1$, variance $\sigma_1^2$, and a second component with any fixed positive parameters. show that $\ell \to \infty$ as $\sigma_1 \to 0$. what does `reg_covar` do to this?

<details><summary>solution</summary>

the term for $x_1$ is $\log(\pi_1(2\pi\sigma_1^2)^{-1/2} + \pi_2\mathcal N(x_1 \mid \mu_2, \sigma_2^2)) \ge \log\pi_1 - \tfrac12\log(2\pi) - \log\sigma_1 \to \infty$. every other term is at least $\log(\pi_2\mathcal N(x_i \mid \mu_2, \sigma_2^2))$, a finite constant. so the sum goes to infinity. with `reg_covar` $= \epsilon$ the variance can't go below $\epsilon$, so that term is at most about $-\tfrac12\log(2\pi\epsilon)$: bounded, but large, which is why tiny `reg_covar` still lets a component lock onto a few duplicated points.

</details>

## experiment

`lab.py` runs your `GaussianMixture`:

1. log-likelihood curves from random and k-means starts: monotone, but where do they end, and when?
2. k-means versus GMMs with spherical, diagonal and full covariances on stretched, tilted clusters.
3. choosing $k$ with training log-likelihood, held-out log-likelihood and BIC, with and without a few duplicated points.

## check your understanding

1. why is the E-step's output a *distribution* per point rather than a label? what would you lose by rounding it?

<details><summary>answer</summary>

the posterior $p(z_i \mid x_i)$ carries uncertainty: a boundary point contributes half its weight to each neighbour's mean and covariance. rounding gives "hard EM" (classification EM), which is k-means-like: it optimizes a different objective, biases the covariances (each cluster's tails are cut off at the boundary), and loses the monotone-likelihood guarantee for the mixture likelihood.

</details>

2. your EM log-likelihood went *down* between two iterations. name two likely bugs.

<details><summary>answer</summary>

an M-step that uses stale responsibilities (from the previous E-step, or the initial ones), a covariance computed around the *old* means, biased or wrong normalization ($n$ instead of $N_j$, or $N_j - 1$), forgetting the weights in the E-step, or adding `reg_covar` inconsistently. a tiny decrease at the $10^{-12}$ level is round-off, not a bug; and with `reg_covar` > 0, a small dip (say $10^{-4}$) when a component's variance hits the floor is expected, since the regularized M-step isn't an exact maximization. rerun with `reg_covar = 0`: a correct EM is then monotone.

</details>

3. what does `reg_covar` cost you when the data really have a tiny variance in some direction?

<details><summary>answer</summary>

a floor on the variance biases it upward: with data varying by $10^{-4}$ in some direction and `reg_covar = 1e-6`, the floor is negligible, but with `reg_covar = 1e-2` the model would smear that direction out and underestimate densities. it's a prior in disguise (a MAP estimate with an inverse-Wishart-like prior), so scale it to your data or standardize first.

</details>

4. why is "converged" (log-likelihood change below `tol`) not proof that EM found a good solution?

<details><summary>answer</summary>

a small change happens at maxima, at saddle points, and on long plateaus. near a symmetric start the improvement per iteration can be $10^{-6}$ for hundreds of iterations before a big jump; `tol` can't tell the difference. compare restarts, look at the curve, and check the solution (are components distinct? are weights reasonable?).

</details>

5. a GMM with full covariances on 784-dimensional MNIST pixels, $k = 10$. what goes wrong, and what are the standard fixes?

<details><summary>answer</summary>

each component has 307,720 covariance parameters, more than the 6,000 or so points it gets, so the covariance estimates are singular (and the pixels near the border are constant, variance 0). fixes: reduce the dimension first (PCA, topic 03; the capstone does this), use diag or spherical covariances, or regularize heavily.

</details>

6. both BIC and held-out log-likelihood choose $k$. when would they disagree?

<details><summary>answer</summary>

BIC penalizes by $p\log n$ and is consistent when the model is right; held-out likelihood estimates predictive quality directly and tends to choose more components when the true clusters aren't Gaussian (it uses extra components to fit a non-Gaussian shape, which genuinely improves predictions). BIC can also be fooled by degenerate fits on training data (duplicate points), which held-out data would expose.

</details>

## common bugs

- **exponentiating before normalizing.** computing `pdf` values, then dividing. symptom: `NaN` responsibilities for any point far from every component.
- **dividing by $n$ instead of $N_j$** (or by $N_j - 1$). symptom: wrong covariances, likelihood that drops between iterations.
- **covariances around the old means.** computing $\Sigma_j$ with $\mu_j$ from the previous iteration. symptom: slow convergence, occasional likelihood decreases.
- **forgetting `reg_covar`, or adding it to every entry.** it belongs on the variances only (the diagonal). symptom: `LinAlgError` in Cholesky after a component collapses; or wrong off-diagonal covariances.
- **inverting $\Sigma$ explicitly.** symptom: inaccurate densities when $\Sigma$ is badly conditioned. use the Cholesky solve.
- **shared state across fits.** keeping `lower_bound_history_` from a previous `fit` call. symptom: histories that grow across refits.

## going deeper

- Dempster, Laird & Rubin (1977), "Maximum likelihood from incomplete data via the EM algorithm", *JRSS B* 39(1). [doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)
- Neal & Hinton (1998), "A view of the EM algorithm that justifies incremental, sparse, and other variants", in *Learning in Graphical Models*: EM as coordinate ascent on the free energy (the ELBO). [doi:10.1007/978-94-011-5014-9_12](https://doi.org/10.1007/978-94-011-5014-9_12)
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, chapter 9 (mixtures and EM, including the k-means limit and the singularity problem).
- Schwarz (1978), "Estimating the dimension of a model", *Annals of Statistics* 6(2): BIC. [doi:10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136)
- McLachlan & Krishnan (2008), *The EM Algorithm and Extensions*, 2nd ed., Wiley. [doi:10.1002/9780470191613](https://doi.org/10.1002/9780470191613)
- Jin et al. (2016), "Local maxima in the likelihood of Gaussian mixture models: structural results and algorithmic consequences", *NeurIPS*: bad local maxima exist even for well-separated mixtures. [arXiv:1609.00978](https://arxiv.org/abs/1609.00978)

## connections

- ← builds on: the multivariate Gaussian and Cholesky (ch 04, ch 02), maximum likelihood (ch 05), KL divergence (ch 05), logsumexp (ch 00), `KMeans` for initialization (topic 01).
- → used in: density-based anomaly scores (topic 06), the capstone's GMM clustering of MNIST, Baum–Welch for hidden Markov models and CAVI for a Bayesian GMM (ch 12), the ELBO in variational inference (ch 12) and variational autoencoders (ch 21).
