# 12 · probabilistic & bayesian ML

> model uncertainty explicitly and do inference exactly, by sampling, and by optimization.

every model so far has produced one answer: one weight vector, one prediction, one cluster assignment. this chapter's models produce *distributions*: over weights, over hidden states, over the parameters of a hierarchical model. that changes what you can say ("the effect is between 2 and 9 with 90% probability", "I've never seen inputs like this one") and what you have to compute, because a posterior is almost never something you can write down. so the chapter is organized around the three ways to get at one. **exactly**, when the model's structure allows it: conjugate Gaussians, graphical models with small treewidth, hidden Markov models. **by sampling**: Markov chain Monte Carlo, from the random walk to Hamiltonian dynamics driven by your own autodiff. **by optimization**: variational inference, which trades exactness for speed in a direction you'll learn to predict. on the way you'll be surprised by a regression that is *more* confident far from its data than between its data points, by an earthquake report that divides the probability of a burglary by 114, by decoding that gets worse than a memoryless rule, by four healthy-looking MCMC chains that are all wrong, by a sampler that stops dead when its trajectory is exactly $2\pi$ long, and by a variational posterior that reports a standard deviation of 0.44 where the truth is 1. the capstone puts it all into a tiny probabilistic programming language and uses it on the most famous small hierarchical model there is.

**you will build:** Bayesian linear regression with predictive uncertainty and evidence maximization; factors, factor products and brute-force inference, moral graphs, Markov blankets, graph separation and d-separation; variable elimination with elimination orders and their costs, and belief propagation on trees; hidden Markov models with forward–backward, Viterbi and Baum–Welch; Metropolis–Hastings and Gibbs samplers with autocorrelation, effective sample size and split $\hat R$; Hamiltonian Monte Carlo with the leapfrog integrator, a mass matrix, autodiff gradients and the U-turn criterion behind NUTS; and variational inference, both coordinate ascent for a Bayesian Gaussian mixture and reparameterized stochastic VI with diagonal and full-covariance Gaussians. modules: **`mlfs.bayes`** (topics 01–04) and **`mlfs.infer`** (topics 05–07).
**prerequisites:** chapter 04 (the multivariate Gaussian, Markov chains, Monte Carlo), chapter 05 (Bayes' rule, KL divergence, entropy), chapter 03 · topic 04 (forward-mode autodiff with `Dual`, used for HMC and VI gradients), chapter 06 (`Adam` and `Param`, used for VI), chapter 07 (the `Estimator`/`Regressor` protocol), chapter 02 (Cholesky factors and linear solves), and `as_rng`/`logsumexp` from chapter 00. no neural networks: everything here is NumPy (SciPy only in the checks).
**time · scaffolding:** about 85 hours · L3 (independent): every function comes with its signature and a one-line docstring; the exact conventions (orders of random draws, stopping rules, what to return) are in the lesson's spec boxes. hints have two levels. the capstone gives you the stubs, the eight-schools data and the evaluation of the two parameterizations; you write the funnel analysis.
**newly unlocked library components:** after this chapter you may use `scipy.stats` distributions and `scipy.special` (log-densities, `logsumexp`) for probabilistic computations in later labs, since you've written the densities, the conditioning and the log-space arithmetic yourself; `hmmlearn`-style HMM tools and probabilistic programming systems (Stan, PyMC, NumPyro) are fair game in later chapters' experiments, as the industrial versions of your `HMM`, `hmc` and `reparam_vi` (NUTS and ADVI are what those systems run by default). inside `mlfs`, later chapters use *your* code through the contracts in [`CONTRACTS.md`](../CONTRACTS.md): `mlfs.infer`'s samplers and diagnostics (ESS, $\hat R$) for any posterior or energy-based sampling, `reparam_vi`'s reparameterization gradient as the warm-up for variational autoencoders (ch 21), and `mlfs.bayes.BayesianLinearRegression` as the weight-space view of Gaussian processes.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [bayesian linear regression](01-bayesian-linear-regression/) | `polynomial_basis`, `gaussian_basis`, `posterior`, `sequential_posterior`, `predictive`, `log_evidence`, `BayesianLinearRegression` → `mlfs.bayes` | with Gaussian-bump features the model is *more* confident at $x = 3$, far outside the data, than in a gap inside it; the evidence picks the true polynomial degree while training error keeps falling; one data point turns the posterior into a ridge |
| 02 | [graphical models](02-graphical-models/) | `Factor` → `mlfs.bayes`, `joint_query`, `moralize`, `markov_blanket`, `separated`, `ancestors`, `d_separated` → `mlfs.bayes`, `factor_graph`, `is_tree` | an earthquake report divides the burglary probability by 114; selecting on fame makes independent traits correlate at $-0.68$; observing wet grass makes sprinkler and rain dependent *again*, twice as strongly as before |
| 03 | [exact inference](03-exact-inference/) | `eliminate`, `min_degree_order`, `elimination_cost`, `variable_elimination` → `mlfs.bayes`, `sum_product` → `mlfs.bayes` | on a $6\times6$ grid a typical random elimination order builds tables 64 times bigger than min-degree's, the worst one 8,000 times; one sum-product call gives every marginal of an 80-node tree a couple of hundred times faster than 80 eliminations |
| 04 | [hidden markov models](04-hidden-markov-models/) | `HMM` (`sample`, `scaled_messages`, `forward_backward`, `viterbi`, `fit`, `score`) → `mlfs.bayes` | Baum–Welch from random starts finds parameters that are wrong about everything yet explain 300 rolls *better* than the true casino; for short loaded stretches Viterbi decodes worse than a memoryless "six means loaded" rule |
| 05 | [MCMC](05-mcmc/) | `metropolis_hastings`, `gibbs_sampler`, `autocorrelation`, `effective_sample_size`, `r_hat` → `mlfs.infer` | the acceptance rate that mixes best is 0.49 in 1-D and 0.19 in 20-D, never the highest; four chains with healthy ESS each sit in one mode and are all wrong ($\hat R = 4.4$); Gibbs at $\rho = 0.99$ needs 100 sweeps per independent draw |
| 06 | [hamiltonian monte carlo](06-hamiltonian-monte-carlo/) | `hamiltonian`, `leapfrog` → `mlfs.infer`, `hmc` → `mlfs.infer` (with `mlfs.autodiff` gradients), `no_u_turn`, `u_turn_steps` | at equal cost HMC gets about 70 times more effective samples than random-walk Metropolis for the means of a 100-D Gaussian (27 times for the variances); acceptance falls off a cliff at step size exactly 2; a trajectory of length $2\pi$ goes nowhere |
| 07 | [variational inference](07-variational-inference/) | `elbo_diag`, `cavi_gmm` → `mlfs.infer`, `elbo_grad_diag`, `reparam_vi` → `mlfs.infer`, `elbo_grad_full` | mean-field VI reports a standard deviation of 0.44 where the truth is 1 (at $\rho = 0.9$); a single Gaussian fitted by VI usually sits on one mode while moment matching sits in the empty valley (but at moderate separations the best reverse-KL fit is broad, and a broad local optimum survives at large ones); 7 of 30 CAVI starts end in a worse clustering |

## capstone

*a tiny probabilistic programming language.* you write models as ordinary Python functions that `sample` parameters and `observe` data; your library finds the parameters by running the model once, maps constrained parameters to $\mathbb R^d$ with the right Jacobian, computes the whole gradient in a single forward pass by giving your dual numbers vector-valued tangents, runs your `hmc` (several chains) or `reparam_vi`, and summarizes the posterior with means, intervals, ESS and $\hat R$. then you apply it to the **eight schools** (Rubin, 1981): you compute the exact posterior by one-dimensional quadrature, and find that the textbook (centered) parameterization traps HMC at the mouth of a funnel, with low ESS, $\hat R$ well above 1, divergent trajectories and a biased estimate of the between-school spread, while the non-centered parameterization matches the exact answer. small CPU config: about 30 seconds; full config: about 2 minutes.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (a conjugate posterior and predictive, forward–backward and Viterbi on a two-step HMM, one leapfrog step and its acceptance probability), and a 60-minute blind rebuild of Metropolis–Hastings, the effective sample size and HMC.

## notation

| symbol | meaning |
|---|---|
| $\Phi$, $\phi(x)$, $w$, $d$ | design matrix $(n, d)$, feature vector, weights, number of features |
| $\alpha$, $\beta$ | prior precision of the weights, noise precision (inverse variances) |
| $m_N$, $S_N$, $A$ | posterior mean and covariance of the weights, $A = S_N^{-1} = \alpha I + \beta\Phi^\top\Phi$ |
| $\gamma$ | effective number of parameters (topic 01); also the HMM posterior marginals $\gamma_t$ (topic 04) |
| $\phi_a(x_{S_a})$, $Z$ | a factor over the variables $S_a$; the normalizing constant |
| $X \perp Y \mid Z$ | conditional independence |
| $z_t$, $x_t$, $K$, $M$ | hidden state and observation at time $t$; numbers of states and symbols |
| $\pi$, $A$, $B$ | HMM start distribution, transition matrix $A_{ij} = P(z_{t+1} = j \mid z_t = i)$, emission matrix $B_{km}$ |
| $\alpha_t$, $\beta_t$, $c_t$, $\xi_t$, $\delta_t$ | forward and backward messages, scaling constants, pairwise posteriors, Viterbi scores |
| $\pi(x)$, $q(x' \mid x)$ | in topics 05–06: the target density (often unnormalized) and the proposal |
| $\rho_k$, $\tau$, ESS, $\hat R$ | autocorrelation at lag $k$, integrated autocorrelation time, effective sample size $n/\tau$, potential scale reduction |
| $W$, $B$ | within-chain and between-chain variance in $\hat R$ |
| $q$, $p$, $U$, $K$, $H$, $M$ | in topic 06: position, momentum, potential $-\log\pi$, kinetic energy, Hamiltonian, mass matrix |
| $\varepsilon$, $L$, $T = L\varepsilon$ | leapfrog step size, number of steps, trajectory length |
| $q(z)$, $\mathcal L(q)$ | in topic 07: the variational distribution and the ELBO |
| $\varphi_{ik}$, $m_k$, $s_k^2$, $\sigma^2$ | CAVI responsibilities, variational means and variances of the component means, prior variance |
| $\mu$, $\tau$, $\theta_j$, $\eta_j$, $\sigma_j$ | eight schools: population mean, between-school sd, school effects, non-centered effects, known standard errors |

## how this connects

- builds on: the multivariate Gaussian, Markov chains and Monte Carlo (ch 04), Bayes' rule, KL divergence and entropy (ch 05), forward-mode autodiff (ch 03), Adam (ch 06), the `Estimator` protocol and model selection (ch 07), Cholesky factors (ch 02), log-space arithmetic (ch 00).
- used later in: Gaussian processes and Bayesian optimization (ch 09), variational autoencoders and the ELBO (ch 21), score-based and diffusion models, which sample with Langevin dynamics and train on an ELBO (ch 22), uncertainty, ensembles and calibration (ch 28 · topic 05), causal graphs and the PC algorithm, and Kalman filters and state-space models (ch 29 electives), and EM-style latent-variable training throughout.

## core references

- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, ch. 3, 8, 10, 11, 13. [microsoft.com/en-us/research/publication/pattern-recognition-machine-learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- Koller & Friedman (2009), *Probabilistic Graphical Models: Principles and Techniques*, MIT Press. [mitpress.mit.edu/9780262013192](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
- Murphy (2023), *Probabilistic Machine Learning: Advanced Topics*, MIT Press. [probml.github.io/pml-book/book2.html](https://probml.github.io/pml-book/book2.html)
- Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), *Bayesian Data Analysis*, 3rd ed., CRC Press. the eight schools, MCMC diagnostics. [stat.columbia.edu/~gelman/book](http://www.stat.columbia.edu/~gelman/book/)
- Neal (2011), "MCMC using Hamiltonian dynamics", in *Handbook of Markov Chain Monte Carlo*. [arXiv:1206.1901](https://arxiv.org/abs/1206.1901)
- Blei, Kucukelbir & McAuliffe (2017), "Variational inference: a review for statisticians", *JASA* 112(518). [arXiv:1601.00670](https://arxiv.org/abs/1601.00670)
