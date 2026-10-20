# 04 · probability

> reason about uncertainty, sample from anything, and understand the limit theorems.

every model in this course is uncertain about something, and almost every number it reports is an average of random quantities. a classifier outputs probabilities. a language model is a machine for conditional distributions over the next token. training shuffles data, drops units and adds noise, all of it drawn from a uniform random number generator through transformations someone had to get right. and every evaluation is an average over a finite test set, so it comes with an error you need to bound. this chapter builds the probability those things rest on, by hand and in code: distributions you can condition and marginalize exactly, samplers built from nothing but uniform numbers, numerically stable moments, the multivariate Gaussian through one Cholesky factor, the law of large numbers and the central limit theorem (and where they break), guaranteed tail bounds, and Markov chains. you'll be surprised more than once: by the base-rate fallacy, by a variance formula that returns negative numbers, by a bell curve whose running average never settles, and by a random walk that's most likely to spend almost all its time on one side.

**you will build:** discrete distribution objects with conditioning, marginals and Bayes' rule; inverse-CDF, Box–Muller, rejection and Marsaglia–Tsang samplers that use only `rng.random()`; LOTUS and Monte Carlo estimates with standard errors; Welford/Chan streaming statistics and a two-pass covariance; a `MultivariateNormal` with sampling, log densities, marginals and conditionals; a CLT simulator with KS distances and Berry–Esseen bounds; Markov, Chebyshev, Hoeffding, Bernstein and Chernoff bound calculators; a Markov chain simulator, a stationary-distribution solver, mixing times and spectral gaps, and a Brownian motion sampler. module: **`mlfs.sampling`** (topics 02, 03, 04 and 07).
**prerequisites:** chapter 00 (NumPy, floating point), chapter 01 (sets, functions, derivatives, CDFs), chapter 02 (`cholesky`, triangular solves, eigenvalues: `mlfs.linalg` is used directly in topic 04) and chapter 03 (limits, change of variables). no prior probability course is assumed.
**time · scaffolding:** about 90 hours · L2 (supported): every function comes with its signature and a docstring with shapes. hints have three levels. the capstone gives you data loading and evaluation.
**newly unlocked library components:** after this chapter you may use NumPy's own samplers (`rng.normal`, `rng.exponential`, `rng.gamma`, `rng.beta`, `rng.poisson`, `rng.choice`, `rng.multivariate_normal`, …), `np.var`, `np.cov` and `np.corrcoef`, and `scipy.stats` distributions and tests (`norm`, `binom`, `kstest`, …) freely in labs and experiments, since you've built each of them (topic 05's lab already does). inside `mlfs` modules, later chapters use *your* versions through `mlfs.sampling`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [probability & random variables](01-probability-and-random-variables/) | `Discrete`, `pushforward`, `product`, `add`, `condition`, `marginal`, `is_independent`, `chain`, `bayes` | a 90%-sensitive test means only a 9% chance of disease at 1% prevalence; Monty Hall's answer changes if the host didn't know |
| 02 | [distributions & sampling](02-distributions-and-sampling/) | `uniform`, `inverse_cdf_sample`, `exponential`, `bernoulli`, `categorical`, `poisson`, `box_muller`, `normal`, `rejection_sample`, `gamma`, `beta` → `mlfs.sampling` | a broken Box–Muller still gives a perfectly straight QQ line; a proposal 10% too wide accepts 0.85% of draws in 50-D |
| 03 | [expectation, variance & covariance](03-expectation-variance-and-covariance/) | `expectation`, `mc_estimate`, `naive_variance`, `two_pass_variance`, `RunningStats`, `covariance`, `correlation` → `mlfs.sampling` | the textbook variance formula goes negative at offset $10^9$ (float64) and $10^4$ (float32); the sample covariance of pure noise has eigenvalues from 0.1 to 2.8 |
| 04 | [the multivariate gaussian](04-the-multivariate-gaussian/) | `MultivariateNormal` (`sample`, `logpdf`, `pdf`, `entropy`, `marginal`, `condition`) → `mlfs.sampling` | conditioning is slicing; a chain's precision matrix has zeros where its covariance has none; in 1000-D no sample comes within distance 25 of the mean |
| 05 | [limit theorems](05-limit-theorems/) | `running_mean`, `sample_means`, `normal_cdf`, `ks_distance`, `clt_distances`, `berry_esseen_bound`, `deviation_probability` | the Cauchy's running average never settles and its "CLT" gets worse with $n$; at $n = 30$, the normal tail is off by a factor of 3 |
| 06 | [concentration inequalities](06-concentration-inequalities/) | `markov_bound`, `chebyshev_bound`, `hoeffding_bound`, `hoeffding_sample_size`, `bernstein_bound`, `chernoff_bernoulli_bound`, `empirical_tail` | Hoeffding demands 18,445 test examples where 400 suffice; the best of 1000 equally good models looks 4 points better |
| 07 | [markov chains & stochastic processes](07-markov-chains-and-stochastic-processes/) | `evolve`, `simulate_markov_chain`, `stationary_distribution`, `total_variation`, `spectral_gap`, `mixing_time`, `brownian_motion` → `mlfs.sampling` | a barbell graph mixes 150 times slower than a complete graph; a Brownian path is *least* likely to split its time evenly |

**optional further reading (★ advanced track): measure-theoretic probability and martingales.** this chapter works with finite and countable sample spaces and densities, which is everything the course needs. if you want the rigorous foundations (σ-algebras, Lebesgue integration and expectation, conditional expectation as a projection, and martingales with the optional stopping and convergence theorems, which underlie the analysis of SGD and of many online learning algorithms), read Williams (1991), *Probability with Martingales*, Cambridge University Press, chapters 1–12, or Durrett (2019), *Probability: Theory and Examples*, 5th ed., chapters 1 and 4. a good companion exercise: prove that topic 07's random walk is a martingale, and use optional stopping to derive the gambler's-ruin probabilities, then check them with `simulate_markov_chain`. this track is proofs and simulations, not a built topic, and nothing later depends on it.

## capstone

*an n-gram text generator* with perplexity evaluation: an early look at language models (ch 17). you build a character-level n-gram model by counting, smooth it with add-$k$, choose its order on a validation set, measure it with perplexity against unigram and bigram baselines, and generate text with your own `categorical` at different temperatures. small CPU config: the offline synthetic stand-in in about a second. full config: tiny Shakespeare in about 6 seconds (the reference reaches a test perplexity of 7.11, 2.83 bits per character, against 12.2 for a bigram).

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (Bayes twice, a Gaussian conditioned, a Markov chain and a tail bound), and a 60-minute blind rebuild of `mlfs.sampling`'s samplers.

## notation

| symbol | meaning |
|---|---|
| $\Omega$, $\omega$ | sample space, an outcome |
| $A, B$ | events (subsets of $\Omega$); $A^c$ the complement |
| $P(A)$, $P(A \mid B)$ | probability, conditional probability |
| $X, Y, Z$ (uppercase) | random variables; lowercase $x, y, z$ are their values |
| $p(x)$, $f(x)$ | pmf (discrete) or density (continuous) |
| $F(x)$, $F^{-1}(u)$ | CDF $P(X \le x)$, inverse CDF |
| $\Phi$, $\varphi$ | standard normal CDF and density |
| $U$, $u$ | a Uniform(0, 1) random variable, a uniform draw |
| $\mathbb{E}[X]$, $\mu$ | expectation, mean |
| $\operatorname{Var}(X)$, $\sigma^2$, $\sigma$ | variance, standard deviation |
| $\operatorname{Cov}(X, Y)$, $\rho$ | covariance, correlation |
| $\bar X_n$, $\hat\mu$ | sample mean of $n$ draws, an estimate |
| $s$, $s/\sqrt n$ | sample standard deviation (`ddof=1`), standard error |
| $M_2$ | running sum of squared deviations (Welford/Chan) |
| $\mathcal{N}(\mu, \Sigma)$ | Gaussian with mean vector $\mu$ and covariance matrix $\Sigma$ |
| $L$, $\Sigma = LL^\top$ | lower Cholesky factor of $\Sigma$ |
| $\Delta^2 = (x - \mu)^\top\Sigma^{-1}(x - \mu)$ | squared Mahalanobis distance |
| $\Lambda = \Sigma^{-1}$ | precision matrix |
| $\Sigma_{aa}, \Sigma_{ab}, \dots$ | blocks of $\Sigma$ for index sets $a$ (observed) and $b$ (the rest) |
| $\xrightarrow{d}$, $\xrightarrow{p}$, $\xrightarrow{a.s.}$ | convergence in distribution, in probability, almost surely |
| $D_n$ | Kolmogorov–Smirnov distance $\sup_t\lvert F_n(t) - F(t)\rvert$ |
| $\rho = \mathbb{E}\lvert X - \mu\rvert^3$ | third absolute central moment (Berry–Esseen) |
| $\mathrm{KL}(q \,\Vert\, p)$ | Kullback–Leibler divergence between Bernoulli($q$) and Bernoulli($p$) |
| $P$, $P_{ij}$ | transition matrix of a Markov chain, $P(X_{t+1} = j \mid X_t = i)$ (row-stochastic) |
| $p_t$ | distribution of $X_t$, a row vector; $p_{t+1} = p_tP$ |
| $\pi$ | stationary distribution, $\pi P = \pi$ |
| $\lambda_2$, $\gamma = 1 - \lvert\lambda_2\rvert$ | second-largest eigenvalue (by modulus), spectral gap |
| $\mathrm{TV}(p, q)$, $t_{\text{mix}}$ | total variation distance, mixing time |
| $B_t$, $\Delta t$ | Brownian motion, time step |
| `rng` | a `np.random.Generator`; functions take `rng=None` (None, an int seed, or a Generator) |

## how this connects

- builds on: NumPy and floating point (ch 00), sets, functions and CDFs (ch 01), `mlfs.linalg`'s Cholesky, triangular solves and eigenvalues (ch 02), limits and change of variables (ch 03).
- used later in: estimation, confidence intervals, hypothesis tests and information theory (ch 05, which builds `mlfs.stats` on these samplers and moments), stochastic gradients (ch 06), honest evaluation with uncertainty (ch 07), naive Bayes and Gaussian discriminant analysis (ch 08), Gaussian processes (ch 09), Gaussian mixtures and PCA (ch 11), Bayesian ML, MCMC and variational inference with `MultivariateNormal` (ch 12), generalization bounds from Hoeffding plus a union bound (ch 13), weight initialization (ch 14), language models and sampling (ch 17–18, 27), generative models and the reparameterization trick (ch 21), diffusion models and SDEs (ch 22), and Markov decision processes (ch 23).

## core references

- Blitzstein & Hwang (2019), *Introduction to Probability*, 2nd ed., CRC Press. the main companion for topics 01, 03 and 05, full of puzzles. free at [probabilitybook.net](https://probabilitybook.net).
- Wasserman (2004), *All of Statistics*, Springer, chapters 1–5. probability for people who want to get to statistics and ML fast.
- Devroye (1986), *Non-Uniform Random Variate Generation*, Springer. the reference for topic 02, free from the author at [luc.devroye.org/rnbookindex.html](http://luc.devroye.org/rnbookindex.html).
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, chapter 2. distributions and the Gaussian, from an ML point of view (topic 04).
- Boucheron, Lugosi & Massart (2013), *Concentration Inequalities*, Oxford University Press. topic 06 and much more. [doi:10.1093/acprof:oso/9780199535255.001.0001](https://doi.org/10.1093/acprof:oso/9780199535255.001.0001)
- Levin & Peres (2017), *Markov Chains and Mixing Times*, 2nd ed., American Mathematical Society. topic 07, free from the authors.
