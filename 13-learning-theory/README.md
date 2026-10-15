# 13 · learning theory

> understand when and why learning from finite data works, and where the theory stops.

you've trained dozens of models and watched them generalize, or not, on held-out data. this chapter asks the question underneath: why should fitting a finite sample tell you anything about data you haven't seen? the answers are some of the most beautiful arguments in machine learning: a union bound over hypotheses (PAC learning), a count of how many ways a class can label a sample (the VC dimension), a measure of how well it can fit pure noise (Rademacher complexity), a game against an adversary that needs no probability at all (online learning), and the information a learner adds to a prior (PAC-Bayes). you'll implement every one as a calculator, then do what most courses skip: **plug in real numbers and compare each bound with the error you actually measure**. some bounds are nearly exact (thresholds on a line, the test-set bound), some are off by two orders of magnitude (the VC bound for logistic regression), and a few, computed carefully, are non-vacuous for real models (a PAC-Bayes bound around 0.2 for a 785-parameter classifier on MNIST). knowing which is which, and why, is the point.

**you will build:** a PAC sample-complexity calculator for finite classes, ERM over thresholds in $O(n\log n)$, exact and Monte Carlo failure probabilities and a measured sample complexity; realizability oracles for intervals, boxes and halfspaces (by linear programming), a guarded brute-force shattering checker, a VC-dimension search and the growth function against Sauer's lemma; a Monte Carlo Rademacher estimator, Massart's lemma, closed-form complexities of $\ell_1$/$\ell_2$ balls, the Rademacher risk bound and complexity estimates from random labels; finite-class and VC bounds with explicit constants, the exact binomial test-set bound, a VC sample-size calculator and a generalization report for any model; Hedge, online gradient descent, follow-the-leader, regret curves and both regret bounds; the binary KL and its inverse, the PAC-Bayes-kl bound, the exact risk of a Gaussian stochastic linear classifier, a PAC-Bayes bound optimized over a prior grid, a margin bound and a leave-one-out stability estimate. module: **`mlfs.theory`** (topics 01, 03, 04, 05, 06; topic 02 adds nothing to the library).
**prerequisites:** chapter 04 (probability, Monte Carlo, the binomial), chapter 05 (Hoeffding's inequality, KL divergence, confidence intervals and the bootstrap), chapter 06 (gradient descent, projections, linear programming), chapter 07 (train/test methodology, the estimator API), chapter 08 (logistic and ridge regression, the perceptron) and chapter 09 ($k$-NN, kernels, SVMs). chapter 00's stable log-sum-exp and `as_rng` are used throughout.
**time · scaffolding:** about 75 hours · L3 (independent): every function and method comes with its signature and a one-line docstring; the conventions the checks expect are in each lesson's spec and ▶ build callouts. hints have two levels. the capstone gives you the data loading and the models; you write the measurements and the bounds.
**newly unlocked library components:** nothing new from scikit-learn (it has no learning-theory module). after this chapter you may use `scipy.stats.beta.ppf` / `binom` for binomial confidence bounds and `scipy.optimize.linprog` for separability checks without re-deriving them, and later chapters use *your* `mlfs.theory` calculators (finite-class and VC bounds, Rademacher estimates, Hedge, OGD, the kl-inverse and PAC-Bayes bound) per the contract in [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [PAC learning](01-pac-learning/) | `finite_class_sample_complexity` → `mlfs.theory`, `erm_threshold`, `threshold_risk`, `threshold_failure_probability`, `estimate_failure_probability`, `empirical_sample_complexity` | for thresholds the bound $\ln(1/\delta)/\varepsilon$ is within 7% of the measured sample complexity, and 4 billion float32 thresholds cost only 8.5×; with 10% label noise the agnostic bound is 80× too pessimistic and the measured rate is $1/\varepsilon$ (slope −1.07), not $1/\varepsilon^2$ |
| 02 | [VC dimension](02-vc-dimension/) | `interval_realizable`, `rectangle_realizable`, `halfspace_realizable` (LP), `count_labelings`, `is_shattered`, `vc_dimension_search`, `sauer_bound`, `growth_lower_bound` | boxes shatter a diamond but not a square's corners; lines realize exactly $m^2 - m + 2$ labelings (92 of 1,024 for 10 points); $\mathbb 1[\sin(\omega x) > 0]$, with one parameter, shatters 15 points in float64 |
| 03 | [rademacher complexity](03-rademacher-complexity/) | `empirical_rademacher` → `mlfs.theory`, `massart_bound`, `norm_ball_rademacher`, `l2_rademacher_bound`, `rademacher_bound`, `rademacher_via_erm` | 200 thresholds fit noise far worse than 200 random hypotheses; the $\ell_1$ ball grows like $\sqrt{\ln d}$ while the $\ell_2$ ball reaches 10 at $d = 10^4$; logistic regression starts fitting random MNIST labels at Cover's $k \approx n/2$, and 1-NN fits them perfectly yet has the lowest test error |
| 04 | [bounds in practice](04-bounds-in-practice/) | `finite_class_bound`, `vc_bound` → `mlfs.theory`, `test_set_bound`, `vc_sample_size`, `generalization_report` | at $n = 16{,}000$ the VC bound for a 9-parameter model is 0.37 against a measured 0.156; quantizing MNIST weights to 4 bits makes the counting bound 0.25 (VC: 0.89; still 0.250 after a union bound over the 7 bit widths tried) at a cost of 1.2 points of error; the best of 100 coin flips beats its naive 95% bound every time |
| 05 | [online learning](05-online-learning/) | `Hedge`, `OnlineGradientDescent` → `mlfs.theory`, `follow_the_leader`, `regret_curve`, `run_hedge`, `hedge_regret_bound`, `ogd_regret_bound`, `online_least_squares` | pure coin-flip experts give the *largest* regret (slope 0.52); FTL beats Hedge on iid data (1 vs 30) and loses $T/2$ on the alternating sequence; on a latitude-sorted housing stream OGD with a constant step 0.01 has regret −651: it beats the best fixed predictor in hindsight (the theory's decaying step gets −17) |
| 06 | [stability, margins & PAC-Bayes](06-stability-margins-and-pac-bayes/) | `kl_bernoulli`, `kl_inverse`, `pac_bayes_bound` → `mlfs.theory`, `mcallester_bound`, `gibbs_error`, `isotropic_gaussian_kl`, `pac_bayes_linear`, `linear_margin_bound`, `loo_stability` | a non-vacuous PAC-Bayes bound for the Gibbs (stochastic) version of logistic regression on MNIST: 0.232 with a prior at 0 and 0.180 with a data-dependent prior (its measured test errors: 0.148 and 0.131), where the VC bound is 0.893; for the deterministic $w$ that becomes $L(w) \le 2L(Q) = 0.36$; at zero empirical risk the kl form is 7× tighter than McAllester's; random removals make 1-NN look perfectly stable at $n = 10^4$, targeted ones show it never stabilizes |

## capstone

*generalization report*: theoretical bounds vs empirical gaps for three models, written up. logistic regression, an RBF-kernel SVM and 15-nearest neighbours on MNIST odd vs even (2,000 training digits) and binarized California housing (4,000 districts): for each, the measured train–test gap with a bootstrap CI, the exact test-set bound, every uniform bound that applies (VC, bit-counting, margin, PAC-Bayes, RKHS margin), and the random-label correlation. measured: the RBF SVM generalizes best on both tasks (test error 0.035 on MNIST, 0.141 on California) and its margin bound is vacuous; the only informative training-data bounds at these sizes are for logistic regression on California: bit-counting 0.311 for the float32 classifier (measured 0.156), and PAC-Bayes 0.294 for its Gibbs classifier (measured 0.200; for $w$ itself, $2 \times 0.294$ is vacuous); with all 60,000 digits the PAC-Bayes bound reaches 0.232 for the Gibbs classifier (measured 0.149) and 0.463 for $w$ (measured 0.100), while the VC bound never drops below 0.89; the exact test-set bound is within 0.01 of every model's test error. small CPU config: the synthetic stand-ins, in seconds. full config: about 3.5 minutes.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 14 concept questions, three by-hand problems (a sample complexity and a VC bound, a Rademacher complexity and a Hedge update, a kl-inverse and a PAC-Bayes bound), and a 60-minute blind rebuild of the bound calculators, Hedge and OGD.

## notation

| symbol | meaning |
|---|---|
| $\mathcal X$, $\mathcal D$, $S$, $n$ (or $m$) | domain, data distribution, iid sample, sample size |
| $\mathcal H$, $h$, $h^\star$ | hypothesis class, a hypothesis, the best (or target) hypothesis |
| $L_{\mathcal D}(h)$ or $L(h)$, $L_S(h)$ or $\hat L(h)$ | true risk (error rate) and empirical risk |
| $\varepsilon$, $\delta$ | accuracy tolerance and failure probability ("approximately", "probably") |
| $m_{\mathcal H}(\varepsilon, \delta)$ | sample complexity |
| $\eta$ (topics 01–02) | label-noise rate |
| $\mathcal H_C$, $\Pi_{\mathcal H}(m)$, $d$ | restriction to a point set $C$, growth function, VC dimension |
| $\sigma_i$, $\hat{\mathfrak R}_S(\mathcal F)$, $\mathfrak R_n(\mathcal F)$ | Rademacher signs, empirical and expected Rademacher complexity |
| $\|\cdot\|_p$, $\|\cdot\|_q$, $B$, $R$ | a norm and its dual ($1/p + 1/q = 1$), weight-norm bound, data-norm bound |
| $\bar L(k, m, \delta)$ | the exact binomial (Clopper–Pearson) upper bound from $k$ errors in $m$ test examples |
| $p_t$, $\ell_t$, $L_{t,i}$, $N$, $T$ | Hedge's distribution, the round's losses, cumulative loss of expert $i$, number of experts, horizon |
| $\eta$ (topic 05), $\eta_t$ | learning rate (Hedge), step size of OGD at round $t$ |
| $R_T$ | regret after $T$ rounds |
| $\mathcal K$, $D$, $G$, $\Pi_{\mathcal K}$ | convex decision set, its diameter, gradient-norm bound, Euclidean projection |
| $\text{kl}(q\,\|\,p)$, $\text{kl}^{-1}(q, c)$ | binary KL divergence and its upper inverse |
| $P$, $Q$, $\text{KL}(Q\,\|\,P)$ | PAC-Bayes prior and posterior over hypotheses, their KL divergence |
| $L(Q)$, $\hat L(Q)$ | true and empirical risk of the Gibbs (stochastic) classifier |
| $\gamma$, $\phi_\gamma$ | margin, ramp loss |
| $\beta$ | uniform stability |
| $\Phi$ | standard normal CDF |

## how this connects

- builds on: Monte Carlo and the binomial (ch 04), Hoeffding's inequality, KL divergence and confidence intervals (ch 05), projected gradient descent and linear programming (ch 06), held-out evaluation and the estimator API (ch 07), logistic and ridge regression (ch 08), $k$-NN, kernels and SVMs (ch 09).
- used later in: decision stumps and AdaBoost's margins (ch 10), regularization as stabilization (ch 15), sharpness, flat minima and generalization in deep learning, including the random-label experiments (ch 19), bandits and regret (ch 23), and quantization (ch 26).

## core references

- Shalev-Shwartz & Ben-David (2014), *Understanding Machine Learning: From Theory to Algorithms*, Cambridge University Press. PAC, VC, Rademacher, stability, online learning, PAC-Bayes. [free PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
- Mohri, Rostamizadeh & Talwalkar (2018), *Foundations of Machine Learning*, 2nd ed., MIT Press. the constants used in this chapter's Rademacher, margin and stability bounds. [cs.nyu.edu/~mohri/mlbook](https://cs.nyu.edu/~mohri/mlbook/)
- Langford (2005), "Tutorial on practical prediction theory for classification", *JMLR* 6. the test-set bound, Occam bounds and the PAC-Bayes-kl bound, computed. [jmlr.org/papers/v6/langford05a](https://www.jmlr.org/papers/v6/langford05a.html)
- Cesa-Bianchi & Lugosi (2006), *Prediction, Learning, and Games*, Cambridge University Press. experts, Hedge and regret. [doi:10.1017/CBO9780511546921](https://doi.org/10.1017/CBO9780511546921)
- Hazan (2016), "Introduction to online convex optimization", *Foundations and Trends in Optimization* 2(3–4). [arXiv:1909.05207](https://arxiv.org/abs/1909.05207)
- Dziugaite & Roy (2017), "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data", *UAI*. [arXiv:1703.11008](https://arxiv.org/abs/1703.11008)
