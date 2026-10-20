# density estimation

## why this matters

classifiers and regressors model one variable given others. sometimes you want the whole distribution: where do the data points live, how likely is this new one, and can you draw more like them? anomaly detection flags points with low density. a generative classifier (chapter 12) needs $p(x \mid y)$ for each class. the samplers of chapter 04 needed a density to sample from.

the parametric answer is to fit a Gaussian, or a mixture of a few. that works when the shape is right and fails silently when it isn't: one Gaussian fitted to data with two clusters puts its peak in the empty middle. the nonparametric answer puts a little bump of probability on every data point and adds them up. that's a **kernel density estimate** (KDE), the smooth cousin of the histogram, and it inherits $k$-NN's virtues (no assumed shape, consistency) and its main weakness (the curse of dimensionality).

everything about it hinges on one number, the **bandwidth**: the width of the bumps. too narrow, and the estimate is a forest of spikes that memorizes the sample. too wide, and it's a single blur that hides the structure. this topic is about building the estimator, and then about choosing that number from the data alone.

## warm-up: predict first

1. a histogram of the same 100 points, same bin width, with the bin edges shifted by half a bin. can the number of peaks change?
2. a KDE puts a Gaussian of width $h$ on each of $n$ points. does it integrate to 1? to $n$?
3. you draw new points from a KDE. is their variance larger, smaller, or the same as the data's?
4. if you choose the bandwidth that maximizes the likelihood of the *training* data, what do you get?
5. to halve the error of a density estimate, how much more data do you need: twice as much? four times? more?

## prerequisites check

- [ ] you know what a probability density is, and that it integrates to 1 (→ ch 04 · topic 02)
- [ ] you can compute $\log \sum_i e^{a_i}$ without overflow or underflow (→ ch 00 · topic 03)
- [ ] you've seen maximum likelihood, and why the training likelihood is optimistic (→ ch 05 · topic 02)
- [ ] you can cross-validate with `KFold` (→ ch 07 · topic 02)
- [ ] you know the bias–variance decomposition (→ ch 07 · topic 04)

## core ideas

### idea 1: the histogram

cut the line into bins of width $h$, starting from an origin $x_0$: bin $j$ is $[x_0 + jh,\ x_0 + (j+1)h)$. the histogram density at $x$ is

$$\hat p(x) = \frac{\#\{i : x_i \text{ in the bin of } x\}}{n h}.$$

dividing by $n$ turns counts into fractions, and dividing by $h$ turns fractions into a density, so that $\int \hat p = \sum_j \frac{\text{count}_j}{nh} \cdot h = 1$. it's the simplest nonparametric estimator, and it already shows the trade-off that runs through this topic: narrow bins follow the data closely but each contains few points, so the heights are noisy (variance); wide bins average many points but smear the shape (bias).

it also has two flaws. it's a step function, so its derivative is zero or undefined everywhere. and it depends on the origin: shift every bin edge by half a bin and the counts redistribute, so peaks can appear, merge or vanish (warm-up 1). in more than one dimension, the number of bins grows like $(1/h)^d$, and most of them are empty.

> **▶ build: stage 1.** implement `histogram_density(x_train, x_eval, bin_width, origin=0.0)` for 1-D data. compute every bin index $\lfloor (x - x_0)/h \rfloor$ at once and count the training points per bin without a Python loop (sort the training bins and use `np.searchsorted` twice, or `np.unique`). return `count / (n * h)` for each evaluation point.

### idea 2: kernels: smooth, normalized bumps

the origin problem goes away if every point gets its *own* bin, centred on it. that's a box of width $2h$ around each data point, the **top-hat** kernel. replace the box by a smooth bump and the estimate becomes smooth too. formally, a **kernel** here is a function $K \ge 0$ with $\int K = 1$, symmetric about 0, and its scaled version with bandwidth $h$ in $d$ dimensions is

$$K_h(u) = \frac{1}{h^d} K\!\left(\frac{u}{h}\right),$$

which still integrates to 1 (substitute $v = u/h$: $du = h^d dv$). the three you'll build depend on $u$ only through $r = \|u\|$:

| kernel | $K_h(u)$ for $r = \|u\|$ | support |
|---|---|---|
| gaussian | $(2\pi h^2)^{-d/2} \exp\!\left(-\frac{r^2}{2h^2}\right)$ | everywhere |
| epanechnikov | $\frac{d+2}{2 V_d h^d}\left(1 - \frac{r^2}{h^2}\right)$ | $r < h$ |
| top-hat | $\frac{1}{V_d h^d}$ | $r < h$ |

here $V_d = \pi^{d/2}/\Gamma(d/2 + 1)$ is the volume of the unit ball in $d$ dimensions ($V_1 = 2$, $V_2 = \pi$, $V_3 = 4\pi/3$). in 1-D the Epanechnikov kernel is $\frac{3}{4h}(1 - r^2/h^2)$, the familiar parabola. its constant comes from $\int_{\|v\| < 1}(1 - \|v\|^2)\,dv = V_d \cdot \frac{2}{d+2}$ (on paper 1).

you'll compute every kernel in **log space**: $\log K_h$, with $-\infty$ outside the support. the reason is idea 3: densities of far-away points underflow to exactly 0 in floating point, while their logs are ordinary numbers.

> **▶ build: stage 2.** implement `log_kernel(sq_dist, bandwidth, d, kernel)`: the log of the normalized kernel at the given *squared* distances (any array shape), for `"gaussian"`, `"epanechnikov"` and `"tophat"`; `-np.inf` where $r \ge h$ for the compact kernels; `ValueError` for anything else. `math.lgamma` gives $\log\Gamma$.

### idea 3: the kernel density estimate

put one kernel on every training point and average:

$$\hat p(x) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - x_i).$$

it's a density (an average of densities), it's as smooth as the kernel, and it doesn't depend on any origin. it's also a **mixture model** with $n$ equal-weight components, one per data point (warm-up 2: it integrates to 1, not $n$, because of the $1/n$).

evaluating it at $m$ query points is one pairwise squared-distance matrix $(m, n)$, one call to your log kernel, and one log-sum-exp per row:

$$\log \hat p(x) = \operatorname{logsumexp}_i\big(\log K_h(x - x_i)\big) - \log n.$$

never exponentiate first. a query 100 units from the data with $h = 0.1$ has kernel values around $e^{-500{,}000}$, which is 0 in float64, so $\log \hat p = -\infty$, where the true value is about $-500{,}000$: a finite number you can compare, add up or put in a likelihood ratio. with compact kernels some values really are $-\infty$, and a point outside every bump genuinely has density 0 (log-sum-exp of all $-\infty$ is $-\infty$).

like $k$-NN, the KDE stores the training data and does all its work at query time: $O(mnd)$. tree-based methods (topic 01's KD-tree, with bounds on kernel sums) make it faster.

> **▶ build: stage 3.** implement `KernelDensity.fit` (`check_X_y`, validate `bandwidth > 0` and the kernel name, store `X_`), `score_samples` (the log density at each row, as above) and `score` (the total log-likelihood, a float). it subclasses your `Estimator`.

> **🔮 predict:** a Gaussian KDE with a tiny bandwidth is evaluated at a held-out point halfway between two training points that are 1 unit apart. as $h \to 0$, what happens to its log density? and at a training point?

### idea 4: sampling from a KDE

a mixture is easy to sample: pick a component, then sample from it. for a KDE the components are kernels centred on the data points, so a draw is

$$x^\star = x_I + h\,\varepsilon, \qquad I \sim \text{uniform}\{1, \dots, n\},\ \varepsilon \sim \mathcal N(0, I_d)$$

for the Gaussian kernel. it's the bootstrap of chapter 05 with a little noise added (the **smoothed bootstrap**). the noise matters: the bootstrap only ever repeats observed values, while the KDE produces new ones in between.

by the law of total variance, the sample's covariance is the data's covariance plus $h^2 I$ (warm-up 3): a KDE is always a little *wider* than the data, and the wider the bandwidth, the more it inflates the spread. it's a simple generative model, and a useful baseline for the fancier ones in chapters 21–22.

> **▶ build: stage 4.** implement `KernelDensity.sample(n=1, *, rng=None)`: draw $n$ row indices uniformly with `as_rng(rng)`, then add `bandwidth * standard_normal((n, d))`. other kernels raise `NotImplementedError`.

### idea 5: choosing the bandwidth

the error of a KDE at a point splits into bias and variance, and the bandwidth trades one for the other. for small $h$ and large $n$ (on paper 2), in 1-D,

$$\mathbb E\,\hat p(x) - p(x) \approx \frac{h^2}{2}\,\sigma_K^2\, p''(x), \qquad \mathrm{Var}\,\hat p(x) \approx \frac{p(x)\, R(K)}{n h},$$

where $\sigma_K^2 = \int v^2 K(v)\,dv$ and $R(K) = \int K^2$. the bias grows with $h^2$ and is worst where the density curves sharply (peaks are flattened, valleys filled in); the variance falls like $1/(nh)$, since about $nh$ points fall under each bump. adding the squared bias and the variance and integrating over $x$ gives the asymptotic mean integrated squared error, minimized at

$$h^\star = \left(\frac{R(K)}{\sigma_K^4\, R(p'')\, n}\right)^{1/5} \propto n^{-1/5},$$

with an error that shrinks like $n^{-4/5}$: slower than the $n^{-1}$ of a parametric model, the price of not assuming a shape. in $d$ dimensions it becomes $n^{-4/(d+4)}$ (warm-up 5: in 1-D, halving the error takes $2^{5/4} \approx 2.4$ times the data; in 10-D, $2^{14/4} \approx 11$ times).

$h^\star$ depends on the unknown $p''$. two practical answers:

- **a rule of thumb.** assume $p$ is Gaussian with standard deviation $\sigma$ just to compute $R(p'')$, and plug in. for a Gaussian kernel in $d$ dimensions this gives Silverman's rule, $h = \left(\frac{4}{d+2}\right)^{1/(d+4)} n^{-1/(d+4)}\,\sigma$, which is $1.06\,\sigma n^{-1/5}$ in 1-D. it's fast and good for roughly Gaussian data, and it **oversmooths** anything multimodal, because a bimodal density has a large standard deviation but sharp features.
- **cross-validated likelihood.** the training likelihood can't choose $h$: as $h \to 0$, each training point sits under its own spike and $\hat p(x_i) \to \infty$ (warm-up 4). so score each $h$ by the log-likelihood of *held-out* points, which falls apart for tiny $h$ (a held-out point between spikes gets almost no density). split the data into $k$ folds, fit on $k - 1$, average $\log \hat p$ over the held-out fold, average over folds, and pick the $h$ with the highest score. maximizing held-out likelihood is minimizing an estimate of the KL divergence from the true density to the estimate (on paper 3).

> **▶ build: stage 5.** implement `silverman_bandwidth(X)` (with $\sigma$ the mean of the per-feature standard deviations, `ddof=1`) and `select_bandwidth(X, bandwidths, *, cv=5, rng=None)`: compute the folds **once** with `KFold(cv, shuffle=True, rng=as_rng(rng))`, score every bandwidth on the same folds (a Gaussian `KernelDensity`, the mean of `score_samples` on each test fold, then the mean over folds), and return `(best_bandwidth, mean_scores)`.

> **🔮 predict:** data from an equal mixture of $\mathcal N(-2, 0.5^2)$ and $\mathcal N(2, 0.5^2)$. will Silverman's rule pick a bandwidth that shows both peaks?

## worked example

three points in 1-D: $x = (0, 1, 3)$.

**histogram**, width 2, origin 0: bins $[0, 2)$ and $[2, 4)$ hold 2 and 1 points, so $\hat p = 2/(3 \cdot 2) = 1/3$ on $[0, 2)$, $1/(3 \cdot 2) = 1/6$ on $[2, 4)$, and 0 elsewhere.

**Gaussian KDE**, $h = 1$, at $x = 1$: the distances are $1, 0, 2$, and with $\varphi(r) = e^{-r^2/2}/\sqrt{2\pi}$,

$$\hat p(1) = \tfrac13\big(\varphi(1) + \varphi(0) + \varphi(2)\big) = \tfrac13(0.24197 + 0.39894 + 0.05399) = 0.23163,$$

so $\log \hat p(1) = -1.4626$.

**Epanechnikov KDE**, $h = 1.5$, at $x = 1$: $u = (x - x_i)/h = 2/3, 0, 4/3$. the third is outside the support. $K(u) = \frac34(1 - u^2)$ gives $\frac34 \cdot \frac59 = 0.41667$ and $0.75$, so

$$\hat p(1) = \frac{0.41667 + 0.75 + 0}{3 \cdot 1.5} = 0.25926.$$

the tests check all three with your code.

## on paper

1. **normalizing the Epanechnikov kernel.** show that $\int_{\|v\| < 1}(1 - \|v\|^2)\,dv = \frac{2V_d}{d+2}$. (use spherical shells: the surface area of the sphere of radius $r$ is $d V_d r^{d-1}$.)

<details><summary>solution</summary>

$\int_0^1 (1 - r^2)\, d V_d r^{d-1}\,dr = d V_d\left(\frac1d - \frac{1}{d+2}\right) = d V_d \cdot \frac{2}{d(d+2)} = \frac{2V_d}{d+2}$. so the normalizer is $\frac{d+2}{2V_d}$, and with bandwidth $h$ it gains the factor $h^{-d}$. in 1-D: $\frac{3}{2 \cdot 2} = \frac34$. ✓

</details>

2. **bias and variance of a 1-D KDE.** with $\hat p(x) = \frac1n\sum_i K_h(x - x_i)$ and $x_i$ i.i.d. from $p$, show $\mathbb E\hat p(x) = \int K(v)\,p(x - hv)\,dv$ and expand to second order in $h$. then show $\mathrm{Var}\,\hat p(x) \approx p(x)R(K)/(nh)$.

<details><summary>solution</summary>

$\mathbb E\hat p(x) = \mathbb E K_h(x - x_1) = \int \frac1h K\!\left(\frac{x - t}{h}\right)p(t)\,dt$, and with $t = x - hv$ this is $\int K(v)p(x - hv)\,dv$. Taylor: $p(x - hv) = p(x) - hv\,p'(x) + \frac{h^2v^2}{2}p''(x) + \dots$; since $\int K = 1$ and $\int vK = 0$ (symmetry), $\mathbb E\hat p(x) = p(x) + \frac{h^2}{2}\sigma_K^2 p''(x) + O(h^4)$. for the variance, $\mathrm{Var}\,\hat p(x) = \frac1n\mathrm{Var}\,K_h(x - x_1) \le \frac1n\mathbb E K_h(x - x_1)^2 = \frac1n\int \frac{1}{h^2}K\!\left(\frac{x-t}{h}\right)^2 p(t)\,dt = \frac{1}{nh}\int K(v)^2 p(x - hv)\,dv \approx \frac{p(x)R(K)}{nh}$ (the subtracted $(\mathbb E K_h)^2/n \approx p(x)^2/n$ is of lower order).

</details>

3. **why held-out likelihood.** show that $\mathbb E_{x \sim p}[\log \hat p(x)] = -\mathrm{KL}(p \,\|\, \hat p) - H(p)$, and conclude that maximizing the expected held-out log-likelihood minimizes the KL divergence from $p$ to $\hat p$.

<details><summary>solution</summary>

$\mathrm{KL}(p \,\|\, \hat p) = \int p \log\frac{p}{\hat p} = \int p\log p - \int p\log\hat p = -H(p) - \mathbb E_p\log\hat p$. rearranging gives the identity. $H(p)$ doesn't depend on $h$, so the $h$ with the largest expected log-likelihood on *fresh* points from $p$ has the smallest KL. the average over a held-out fold is an unbiased estimate of that expectation; the training average is not, because $\hat p$ was built from those very points.

</details>

4. **the optimal bandwidth's rate.** minimize $A h^4 + \frac{B}{nh}$ over $h$ and show that the minimum is proportional to $n^{-4/5}$.

<details><summary>solution</summary>

derivative: $4Ah^3 - \frac{B}{nh^2} = 0 \Rightarrow h^5 = \frac{B}{4An}$, so $h^\star = (B/4A)^{1/5} n^{-1/5}$. plugging in, both terms are proportional to $(h^\star)^4 \propto n^{-4/5}$. here $A = \frac14\sigma_K^4 R(p'')$ (integrated squared bias) and $B = R(K)$.

</details>

## experiment

`lab.py` uses your histogram, your `KernelDensity` and your bandwidth selectors:

1. one dataset, one bin width, two bin origins: how different can two histograms look?
2. the bias–variance trade-off in the bandwidth, measured over many samples from a known bimodal density, with Silverman's rule and cross-validation compared.
3. a KDE of where 20,000 California districts are, with the bandwidth chosen by cross-validated likelihood, and new "districts" sampled from it.

## check your understanding

1. why does a histogram depend on the origin while a KDE doesn't?

<details><summary>answer</summary>

a histogram assigns each point to a fixed bin, and which bin depends on where the edges are. a KDE centres a kernel on each data point, so there are no edges: shifting the coordinate system shifts the estimate with it.

</details>

2. a KDE with a compact kernel gives $\log \hat p = -\infty$ at a test point. what does that mean, and why is it a problem for cross-validation?

<details><summary>answer</summary>

no training point is within one bandwidth of the test point, so the estimate is exactly 0 there. one such point makes the average held-out log-likelihood $-\infty$, so cross-validation rejects every bandwidth smaller than the largest gap between a test point and the training data. the Gaussian kernel never has this problem, which is one reason it's the default.

</details>

3. which kernel should you use?

<details><summary>answer</summary>

it matters far less than the bandwidth. the Epanechnikov kernel is optimal for the asymptotic MISE, but the others are within a few percent of it. the Gaussian is smooth, has full support (no $-\infty$ log densities) and is easy to sample from; compact kernels are cheaper to evaluate with trees.

</details>

4. why does $h^\star$ shrink as $n$ grows, and why only like $n^{-1/5}$?

<details><summary>answer</summary>

more data means more points under each bump, so you can afford narrower bumps (less bias) at the same variance. the variance is $\propto 1/(nh)$ and the squared bias $\propto h^4$; balancing them gives $h \propto n^{-1/5}$, a slow rate because the bias falls fast in $h$ while the variance rises only like $1/h$.

</details>

5. a KDE in 20 dimensions with 10,000 points: what goes wrong?

<details><summary>answer</summary>

the curse of dimensionality: the error rate is $n^{-4/24} = n^{-1/6}$, so 10,000 points give roughly the accuracy of 4 or 5 points in 1-D. distances concentrate (topic 01), so a bandwidth that puts mass on a few neighbours of one point puts almost none on others. KDEs are for low dimensions; in high dimension you need structure (parametric models, mixtures, or the deep generative models of chapters 21–22).

</details>

6. your KDE's samples look too spread out compared with the data. is that a bug?

<details><summary>answer</summary>

not necessarily: the sample covariance is the data covariance plus $h^2 I$. if the extra spread is large, the bandwidth is large relative to the data's scale (oversmoothing). shrinking the samples back towards their centre is possible, but changes the estimate.

</details>

## common bugs

- **forgetting the $h^d$ (or the $V_d$).** symptom: the estimate integrates to $h^{d-1}$ or so, and sklearn comparisons are off by a constant factor that depends on $h$ and $d$.
- **exponentiating before summing.** `np.log(np.mean(np.exp(logk), axis=1))`. symptom: $-\infty$ log densities for points a few bandwidths away from the data, and cross-validation that crashes or prefers huge bandwidths.
- **scoring bandwidths on the training data.** symptom: the smallest bandwidth in the grid always wins.
- **reporting total instead of mean held-out log-likelihood.** with folds of unequal size the totals aren't comparable, and the numbers don't read as "nats per point". average per point.
- **different folds per bandwidth.** a new shuffle for each candidate adds noise to the comparison. compute the folds once.
- **noise of the wrong scale in `sample`.** adding `h**2 * noise` (a variance) instead of `h * noise` (a standard deviation). symptom: samples nearly identical to training points for $h < 1$.

## going deeper

- Silverman (1986), *Density Estimation for Statistics and Data Analysis*, Chapman & Hall. the standard reference; chapter 3 derives the bias, variance and the rule of thumb. [doi:10.1007/978-1-4899-3324-9](https://doi.org/10.1007/978-1-4899-3324-9)
- Rosenblatt (1956), "Remarks on some nonparametric estimates of a density function", *Annals of Mathematical Statistics* 27(3): 832–837. [doi:10.1214/aoms/1177728190](https://doi.org/10.1214/aoms/1177728190)
- Parzen (1962), "On estimation of a probability density function and mode", *Annals of Mathematical Statistics* 33(3): 1065–1076. [doi:10.1214/aoms/1177704472](https://doi.org/10.1214/aoms/1177704472)
- Wasserman (2006), *All of Nonparametric Statistics*, Springer, chapter 6 (density estimation, including the risk of histograms and KDEs). [doi:10.1007/0-387-30623-4](https://doi.org/10.1007/0-387-30623-4)
- Scott (2015), *Multivariate Density Estimation*, 2nd ed., Wiley. histograms, the curse of dimensionality for densities, and multivariate bandwidths. [doi:10.1002/9781118575574](https://doi.org/10.1002/9781118575574)

## connections

- ← builds on: densities and mixtures (ch 04), log-sum-exp (ch 00), maximum likelihood and KL divergence (ch 05), cross-validation (ch 07), neighbours and the curse of dimensionality (topic 01).
- → used in: kernels as similarity functions (topic 03), Gaussian mixtures and EM, the parametric counterpart (ch 11), generative classifiers (ch 12), mean shift and density-based clustering (ch 11), evaluating generative models by likelihood (ch 21), and score-based models, which learn the gradient of a smoothed log density (ch 22).
