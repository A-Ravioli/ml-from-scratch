# 13 · capstone: a generalization report

## goal

a model's test error is a measurement; a generalization bound is a promise made from the training data alone. this capstone puts the two side by side for real models on real data, the way a careful practitioner (or a reviewer) would. for three models from part II, logistic regression, an RBF-kernel SVM and 15-nearest neighbours, on two tasks, you'll **measure** the train–test gap with its uncertainty, **compute** every bound from this chapter that applies to each model (VC, bit-counting, margin, PAC-Bayes, the RKHS margin bound, the exact test-set bound), estimate each model's ability to fit noise, and write up what holds, what's vacuous, and why.

nothing here is estimated by hand: every empirical number comes from held-out data, every bound states its version, its constants, its $\delta$ and the grid it was optimized over (with the union bound paid).

## data

- **MNIST, odd vs even** via `mlfs.datasets.mnist()` (cached under `~/.cache/mlfs/`): 2,000 training digits drawn at random from the 60,000, pixels scaled to $[0, 1]$, labels $\pm1$; tested on all 10,000 official test digits. experiment C uses up to all 60,000 training digits.
- **California housing, binarized** via `mlfs.datasets.california_housing()`: is a district's median house value above the median of the training districts? 4,000 training districts, 4,000 others for testing, features standardized with a scaler fitted on the training districts.
- with `MLFS_OFFLINE=1` (tests, `verify.py`), both loaders return synthetic stand-ins and the notebook runs a small config. **tests** use only small synthetic data defined in `test_capstone.py`.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 400 train / 400 test per task (synthetic); 2 random-label draws; 300 bootstrap resamples | about 15 seconds |
| **full** (your write-up) | default | MNIST 2,000 / 10,000, California 4,000 / 4,000; 5 random-label draws; 2,000 bootstrap resamples; experiment C up to 60,000 digits | about 3.5 minutes (213 s measured end to end; experiment A takes about 155 s, most of it the SVMs' random-label fits) |

the notebook picks the config from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `measure_gap(model, X_train, y_train, X_test, y_test, *, delta=0.05, n_boot=1000, rng=None)`: fit once; train error, test error, gap, a 95% percentile-bootstrap CI for the test error (your `mlfs.stats.bootstrap_ci`), and the exact binomial test-set bound at $\delta$
   - `margin_bounds(norm_margins, *, delta=0.05, gammas=GAMMAS, sigmas=SIGMAS)`: from normalized margins $y\,u^\top\tilde x/\|\tilde x\|$ (unit-norm $u$, so $B = R = 1$): the ramp-loss margin bound minimized over the $\gamma$ grid and the PAC-Bayes-kl bound (Gibbs risk $\text{mean}\,\Phi(-m/\sigma)$, KL $= 1/(2\sigma^2)$ to the prior $\mathcal N(0, \sigma^2I)$) minimized over the $\sigma$ grid, each with a union bound over its grid
   - `linear_bounds(model, X, y, *, delta=0.05, bits=32)`: for a fitted linear classifier: VC bound with $d + 1$, bit-counting bound with $2^{32(d+1)}$ hypotheses (evaluated with the weights rounded to float32, the class it counts), and `margin_bounds` of its normalized margins ($\tilde x = (x, 1)$, $u = (w, b)/\|(w, b)\|$)
   - `svm_bounds(svc, X, y, *, delta=0.05)`: for a fitted kernel SVM: its RKHS norm $\sqrt{\alpha^\top K\alpha}$ (with $\alpha$ the `dual_coef_`), normalized margins $y f(x)/(\|(w, b)\|\,\|(\varphi(x), 1)\|)$, and the margin bound of those margins
   - `random_label_correlation(make_model, X, *, n_draws=5, rng=None)`: the mean training correlation of fresh models fitted to random $\pm1$ labels
2. the write-up section at the end of the notebook, answering its five prompts (the measurements, the bounds, models without a bound, the honest number, limits), with numbers from experiments A–C.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| margin and PAC-Bayes bounds | match direct computations with the stated grids and union bounds |
| normalized margins | negative exactly on training errors; computed from the model's own `coef_`/`intercept_` or `dual_coef_`/`intercept_` |
| random-label correlation | exactly 1 for 1-NN, about 0 for a constant predictor |
| full-config report | every number in your write-up is reproducible from the notebook |

## what the reference solution measures (full config)

**experiment A**, all bounds at $\delta = 0.05$ ("–": no bound of that kind applies; "PAC-Bayes (Gibbs)" bounds the *stochastic* Gibbs classifier $\mathcal N(u, \sigma^2I)$ at the selected $\sigma$, whose own measured test error is the next column; "for $w$" is the valid bound on the deterministic classifier, $L(w) \le 2L(Q)$, capped at 1; ρ̂: random-label correlation; proxy: train error + ρ̂ + $3\sqrt{\ln(2/\delta)/(2n)}$, *not* a certified bound):

| task | model | train | test (95% CI) | gap | test-set bound | VC | 32-bit | margin | PAC-Bayes (Gibbs) | Gibbs test | PAC-Bayes for $w$ | ρ̂ | proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | logistic | 0.040 | 0.140 (0.133–0.147) | +0.100 | 0.146 | 2.916 | 2.127 | 1.059 | 0.530 | 0.446 | 1.000 | 0.463 | 0.594 |
| MNIST | RBF SVM | 0.000 | **0.035** (0.032–0.039) | +0.035 | 0.038 | ∞ | – | 1.052 | – | – | – | 0.957 | 1.048 |
| MNIST | 15-NN | 0.061 | 0.061 (0.056–0.066) | −0.001 | 0.065 | ∞ | – | – | – | – | – | 0.204 | 0.357 |
| California | logistic | 0.152 | 0.156 (0.145–0.168) | +0.005 | 0.166 | 0.538 | 0.311 | 0.603 | **0.294** | 0.200 | 0.588 | 0.025 | 0.241 |
| California | RBF SVM | 0.089 | **0.141** (0.131–0.152) | +0.052 | 0.151 | ∞ | – | 1.034 | – | – | – | 0.379 | 0.533 |
| California | 15-NN | 0.141 | 0.169 (0.158–0.181) | +0.028 | 0.179 | ∞ | – | – | – | – | – | 0.204 | 0.410 |

read the table carefully:

- **the best model has no useful bound.** on both tasks the RBF SVM has the lowest test error, and its RKHS margin bound is vacuous (1.05 and 1.03). its normalized training margins are about 0.025 (a functional margin of 1 divided by $\|(w, b)\|\sqrt2 \approx 28\sqrt2$ on MNIST), far below every $\gamma$ in the grid, so the optimizer settles on the grid's edge, $\gamma = 1$, where the bound is still above 1. it also fits random labels almost perfectly on MNIST (ρ̂ = 0.96). the same happens to the margin bound for logistic regression on MNIST.
- **one uniform bound is informative, for the right classifier.** for logistic regression on California (9 parameters, 4,000 districts), the bit-counting bound for the float32 classifier is 0.311 (measured test error 0.156), and the PAC-Bayes bound is 0.294 for the Gibbs classifier at $\sigma = 0.079$, whose measured test error is 0.200. turned into a statement about $w$ itself, PAC-Bayes gives $2 \times 0.294 = 0.588$: vacuous. the VC bound is 0.538. on MNIST with 2,000 digits every uniform bound for logistic regression is vacuous or nearly so (PAC-Bayes 0.530 for a Gibbs classifier that errs 44.6% of the time), although its measured gap of 0.100 is the largest in the table.
- **the gaps are real but small.** most gaps exceed their sampling error (a test-error CI is about ±0.007 on MNIST and ±0.011 on California), except 15-NN on MNIST (−0.001) and logistic regression on California (+0.005, inside the CI).
- **15-NN fits noise at a fixed rate.** ρ̂ ≈ 0.20 on both tasks is what you get from including each point among its own 15 neighbours: the prediction agrees with the point's own random label iff at least 7 of the 14 others do, $2P(\text{Bin}(14, \frac12) \ge 7) - 1 = 0.209$.
- **the test-set bound wins everywhere.** at most 0.01 above the test error for every model, because it bounds one fixed classifier with 4,000–10,000 fresh examples and an exact binomial tail.

**experiment C** (logistic regression on MNIST odd vs even, the first $n$ of a fixed shuffle, tested on all 10,000 test digits):

| n | train | test | test-set bound | Gibbs test | PAC-Bayes (Gibbs) | PAC-Bayes for $w$ | margin | 32-bit | VC |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 0.000 | 0.156 | 0.162 | 0.435 | 0.569 | 1.000 | 1.123 | 4.173 | 3.341 |
| 2,000 | 0.047 | 0.135 | 0.141 | 0.447 | 0.530 | 1.000 | 1.059 | 2.134 | 2.923 |
| 8,000 | 0.088 | 0.106 | 0.111 | 0.236 | 0.393 | 0.786 | 0.903 | 1.131 | 1.865 |
| 30,000 | 0.096 | 0.099 | 0.104 | 0.155 | **0.275** | 0.550 | 0.636 | 0.635 | 1.154 |
| 60,000 | 0.098 | 0.100 | 0.105 | 0.149 | **0.232** | 0.463 | 0.515 | **0.479** | 0.893 |

what more data does: the measured gap closes from 0.156 to 0.002. the PAC-Bayes bound for the Gibbs classifier becomes informative first (below 0.4 from $n = 8000$) and reaches 0.232 at 60,000, 1.6× the Gibbs classifier's measured test error of 0.149. for $w$ itself, the PAC-Bayes route gives 0.463 and the float32 bit-counting bound 0.479, both just under 0.5 and about 4.6× the measured 0.100. the VC bound never drops below 0.89. the $\sigma$ that PAC-Bayes selects shrinks from 0.2 to 0.02 as $n$ grows: with more data the bound can afford a sharper posterior (a larger KL, $1/(2\sigma^2)$), and a sharper posterior's Gibbs risk is closer to $w$'s own error (0.149 against 0.100 at 60,000, against 0.447 vs 0.135 at 2,000).

## stretch goals

- **a data-dependent prior.** split the MNIST training set, fit a prior mean on one half (topic 06), and compute the PAC-Bayes bound on the other half. how much does it beat the prior at 0?
- **trade margin for error.** refit the SVM with smaller $C$ (0.1, 0.5) and watch the RKHS norm, the normalized margins, the margin bound and the test error. is there any $C$ for which the margin bound is non-vacuous at $n = 2000$?
- **a sample-compression bound.** an SVM is determined by its support vectors. look up the compression bound (Floyd & Warmuth, 1995; Langford, 2005, section 6) and compute it from the number of support vectors. is it tighter than the margin bound?
- **check a bound's validity by simulation.** on synthetic data where you can compute the true risk exactly (a linear model on Gaussian inputs), repeat training on fresh samples and count how often each bound fails. is the failure rate below $\delta$?
- **coverage of the bootstrap.** the bootstrap CI and the binomial bound answer slightly different questions. compare their widths, and check the bootstrap interval against the exact two-sided Clopper–Pearson interval.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Langford (2005), "Tutorial on practical prediction theory for classification", *JMLR* 6. the test-set bound, sample compression, and PAC-Bayes, all computed. [jmlr.org/papers/v6/langford05a](https://www.jmlr.org/papers/v6/langford05a.html)
- Dziugaite & Roy (2017), "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data", *UAI*. [arXiv:1703.11008](https://arxiv.org/abs/1703.11008)
- Jiang, Neyshabur, Mobahi, Krishnan & Bengio (2020), "Fantastic generalization measures and where to find them", *ICLR*. a large empirical comparison of bounds and complexity measures against measured gaps. [arXiv:1912.02178](https://arxiv.org/abs/1912.02178)
- Zhang, Bengio, Hardt, Recht & Vinyals (2017), "Understanding deep learning requires rethinking generalization", *ICLR*. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- Floyd & Warmuth (1995), "Sample compression, learnability, and the Vapnik–Chervonenkis dimension", *Machine Learning* 21(3). [doi:10.1007/BF00993593](https://doi.org/10.1007/BF00993593)
- Mohri, Rostamizadeh & Talwalkar (2018), *Foundations of Machine Learning*, 2nd ed., ch. 3, 5 and 6 (the margin bounds and kernel classes used here). [cs.nyu.edu/~mohri/mlbook](https://cs.nyu.edu/~mohri/mlbook/)
- Pace & Barry (1997), "Sparse spatial autoregressions", *Statistics & Probability Letters* 33(3). the California housing data. [doi:10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X)
