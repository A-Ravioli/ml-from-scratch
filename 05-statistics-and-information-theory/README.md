# 05 · statistics & information theory

> estimate, quantify uncertainty, test hypotheses honestly, and measure information.

chapter 04 was about knowing the distribution and asking what data it produces. this chapter turns that around: you have the data, and you want to say something about the distribution, and to say *how sure* you are. that's every evaluation you'll ever run. "the new model is 1.4 points better" means nothing without an error bar, and the error bar means nothing if it was computed the wrong way, or after trying twenty analyses. you'll build the machinery from the ground up: a simulator that grades estimators by bias, variance and MSE; maximum likelihood and Bayesian updating; confidence intervals by normal theory and by the bootstrap; permutation tests and corrections for testing many things at once; the information-theoretic quantities (entropy, cross-entropy, KL, mutual information) behind every loss function you'll train with; and exponential families, Fisher information and the Cramér–Rao bound. along the way you'll catch an estimator that's biased *and* better, a "95%" interval that covers 83% of the time, a p-hacking routine that finds effects in pure noise 98% of the time, a compressor that beats "the entropy", and an estimator whose variance sits below a lower bound.

**you will build:** an estimator-evaluation harness; log-likelihoods, closed-form MLEs, beta–bernoulli and normal–normal posteriors, and grid posteriors with credible intervals; normal and bootstrap (percentile and BCa) confidence intervals with a coverage simulator; a Monte Carlo p-value, the permutation test (unpaired and paired), a size/power simulator, Bonferroni and Benjamini–Hochberg; entropy, cross-entropy, KL and mutual information for discrete distributions, entropy and KL for Gaussians, and a Huffman coder; an exponential-family class with autodiff-computed moments and Fisher information, a damped-Newton MLE fitter and the Cramér–Rao bound. module: **`mlfs.stats`** (topics 03, 04 and 05).
**prerequisites:** chapter 04 (distributions, sampling with a shared `rng`, expectation and variance, the CLT, concentration), chapter 03 (`mlfs.autodiff`: forward-mode `grad` and numerical Jacobians, used in topic 06; Newton's method), chapter 02 (`mlfs.linalg`: Cholesky and linear solves, used in topics 05–06), and chapter 00 (`as_rng`, `logsumexp`, vectorization).
**time · scaffolding:** about 70 hours · L2 (supported): every function comes with its signature and a docstring with shapes. hints have three levels. the capstone gives you the data, the models and the experiments; you build the toolkit.
**newly unlocked library components:** after this chapter you may use `scipy.stats.bootstrap`, `scipy.stats.permutation_test`, `scipy.stats.false_discovery_control`, `scipy.stats.entropy`, `scipy.stats.binomtest` (McNemar) and SciPy's distribution `fit` methods in labs and experiments, since you've built each of them. the standard parametric tests (`ttest_ind`, `ttest_rel`) are fair game as long as you can say what they assume that a permutation test doesn't. inside `mlfs` modules, later chapters use *your* versions through `mlfs.stats`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [estimation](01-estimation/) | `bias_variance_mse`, `simulate`, `evaluate`, `theta_mom`, `theta_max`, `theta_unbiased_max`, `scaled_variance`, `scaled_variance_mse`, `mse_curve`, `convergence_rate` | dividing by $n + 1$ beats the unbiased $n - 1$; James–Stein beats the MLE for *every* mean in 3+ dimensions; the uniform's max converges at rate $1/n^2$ |
| 02 | [maximum likelihood & MAP](02-maximum-likelihood-and-map/) | `bernoulli_loglik`, `normal_loglik`, `poisson_loglik`, the four MLEs, `beta_bernoulli_posterior`, `beta_mean`, `beta_mode`, `normal_posterior`, `grid_posterior`, `credible_interval` | a wrong prior beats the MLE for small $n$; the MLE's expected log loss is infinite at every $n$ |
| 03 | [confidence intervals & the bootstrap](03-confidence-intervals-and-the-bootstrap/) | `normal_ci`, `coverage`, `bootstrap_replicates`, `bootstrap_ci` (percentile + BCa), `jackknife`, `bca_interval` → `mlfs.stats` | "95%" intervals for skewed data cover 82–83% at $n = 10$; bootstrap intervals for a max cover 0% |
| 04 | [hypothesis testing](04-hypothesis-testing/) | `p_value`, `permutation_null`, `permutation_test`, `rejection_rate`, `bonferroni`, `benjamini_hochberg` → `mlfs.stats` | three innocent habits make 98% of null studies "significant"; significant small studies overestimate effects 2.7× |
| 05 | [information theory](05-information-theory/) | `entropy`, `cross_entropy`, `kl_divergence`, `mutual_information`, `gaussian_entropy`, `gaussian_kl`, `huffman_code`, `expected_length`, `encode`, `decode` → `mlfs.stats` | Huffman on single bits compresses nothing; zlib beats "the entropy"; independent variables show 0.47 nats of mutual information |
| 06 | [exponential families & fisher information](06-exponential-families-and-fisher-information/) | `ExponentialFamily` (`log_prob`, `mean_params`, `fisher_information`, `fit`, `score`, `cramer_rao_bound`), four families | biased estimators sit *below* the Cramér–Rao bound; the best unbiased estimator of $e^{-\lambda}$ can't reach it |

## capstone

*a model-comparison toolkit*: confidence intervals for any metric, paired intervals and paired permutation tests for the difference between two models, McNemar's exact test, and a many-models comparison with Benjamini–Hochberg, built on your `mlfs.stats` and used for every evaluation from here on. you apply it to six classifiers on the California housing data (is a district's median house value above the median?). small CPU config: the offline stand-in in about 4 seconds. full config: 6,130 test districts in about 8 seconds, where k-NN on location alone turns out to beat k-NN on every feature (88.4% against 85.1%), and nearest centroid's one-point gain over a one-feature rule isn't significant.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (an MSE comparison, a beta–bernoulli posterior with a Huffman code, and a permutation p-value with BH), and a 60-minute blind rebuild of `bootstrap_ci` and `permutation_test`.

## notation

| symbol | meaning |
|---|---|
| $\theta$, $\hat\theta$ | a parameter, an estimator (or estimate) of it |
| $\operatorname{bias}$, $\operatorname{Var}$, $\operatorname{MSE}$ | $\mathbb{E}\hat\theta - \theta$, $\mathbb{E}(\hat\theta - \mathbb{E}\hat\theta)^2$, $\mathbb{E}(\hat\theta - \theta)^2$ |
| $n$, $R$ | sample size, number of simulated repetitions |
| $s^2$, $\hat\sigma^2$, $S^2_c$ | sample variance (divide by $n - 1$), MLE (divide by $n$), divide by $c$ |
| $L(\theta)$, $\ell(\theta)$ | likelihood, log-likelihood |
| $p(\theta)$, $p(\theta \mid x)$ | prior, posterior |
| $\text{Beta}(a, b)$, $m = a + b$ | Beta prior with pseudo-counts $a$, $b$ and strength $m$ |
| $\tau^2$, $\tau_n^2$ | prior variance, posterior variance (normal–normal) |
| $1 - \alpha$, $z_{1-\alpha/2}$ | confidence level, standard normal quantile ($\Phi^{-1}$) |
| $\hat F$, $x^*$, $\hat\theta^*_b$, $B$ | empirical distribution, a bootstrap sample, the $b$-th bootstrap replicate, number of replicates |
| $\hat\theta_{(i)}$ | jackknife value: the statistic without observation $i$ |
| $z_0$, $a$ | BCa bias correction and acceleration |
| $H_0$, $T$, $t_{\text{obs}}$, $p$ | null hypothesis, test statistic, its observed value, p-value |
| $m$, $m_0$, $q$ | number of hypotheses, number of true nulls, target false discovery rate |
| FWER, FDR | family-wise error rate, false discovery rate |
| $H(p)$, $H(p, q)$, $\mathrm{KL}(p \,\Vert\, q)$ | entropy, cross-entropy, Kullback–Leibler divergence |
| $I(X; Y)$, $H(X \mid Y)$ | mutual information, conditional entropy |
| $h(f)$ | differential entropy of a density |
| $\ell_i$, $L$ | codeword length, expected code length |
| $T(x)$, $\eta$, $A(\eta)$, $h(x)$ | sufficient statistics, natural parameters, log-partition function, base measure |
| $\mu(\eta) = \nabla A(\eta)$ | mean parameters |
| $s(x; \eta)$, $I(\eta)$ | score, Fisher information of one observation |
| nats, bits | information with natural logs, with base-2 logs |
| `rng` | a `np.random.Generator`; functions take `rng=None` (None, an int seed, or a Generator), normalized with `as_rng` |

## how this connects

- builds on: sampling, expectation, the CLT and concentration (ch 04), forward-mode autodiff, numerical Jacobians and Newton's method (ch 03), Cholesky and linear solves (ch 02), `as_rng` and `logsumexp` (ch 00).
- used later in: optimization, where Newton's method and Fisher scoring reappear (ch 06); every evaluation with uncertainty, cross-validation and the danger of tuning on test data (ch 07, through `mlfs.stats`); logistic regression as an exponential-family GLM fitted by IRLS, ridge and lasso as MAP (ch 08); information gain in decision trees and permutation importance (ch 10); Bayesian ML, MCMC and variational inference, which minimizes a reverse KL (ch 12); generalization bounds (ch 13); cross-entropy losses and perplexity (ch 14–18, 27); the VAE's Gaussian KL (ch 21); InfoNCE and mutual information (ch 25); multi-seed comparisons in `mlfs.research` (ch 30).

## core references

- Wasserman (2004), *All of Statistics*, Springer, chapters 6–11: estimation, the bootstrap, MLE, hypothesis testing and Bayesian inference, concisely, for people heading to ML.
- Efron & Hastie (2016), *Computer Age Statistical Inference*, Cambridge University Press: the bootstrap, James–Stein, empirical Bayes and large-scale testing, with history. free at [hastie.su.domains/CASI](https://hastie.su.domains/CASI/).
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall/CRC: topic 03 in depth.
- Cover & Thomas (2006), *Elements of Information Theory*, 2nd ed., Wiley: topic 05 and much more.
- MacKay (2003), *Information Theory, Inference, and Learning Algorithms*, Cambridge University Press: information theory and Bayesian inference as one subject, written for ML. free at [inference.org.uk/itila](http://www.inference.org.uk/itila/).
- Lehmann & Casella (1998), *Theory of Point Estimation*, 2nd ed., Springer: unbiasedness, sufficiency, exponential families and the information inequality, rigorously (topics 01 and 06).
