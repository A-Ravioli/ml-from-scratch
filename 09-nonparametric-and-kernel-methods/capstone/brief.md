# 09 · capstone: Bayesian optimization

## goal

tuning hyperparameters is optimizing a function you can't differentiate, can only evaluate at a cost, and whose evaluations are a little noisy. grid search spends its budget blindly and random search spends it at random (chapter 07, topic 06). **Bayesian optimization** spends it carefully: a Gaussian process models the score as a function of the hyperparameters, its posterior says where the score is probably high *and* where it's still uncertain, and expected improvement turns that into the next point to try. every evaluation updates the model, so the search gets smarter as it goes.

you'll build the whole loop on top of this chapter's library: your `GaussianProcessRegressor` and `expected_improvement` (topic 06) propose the next point, and the expensive objective is a 5-fold cross-validation of your `KernelRidge` (topic 05) on real data. then you'll measure, honestly and over several seeds, whether it beats random search with the same budget, and check the tuned model once on a locked test set.

## data

- **California housing** via `mlfs.datasets.california_housing()` (1990 census districts; 8 numeric features; target: median house value in \$100,000, capped at 5.0), cached under `~/.cache/mlfs/`. the notebook shuffles the 20,433 districts with a fixed seed, takes `N_TRAIN` for tuning (every objective evaluation is a 5-fold CV on them) and the next `N_TEST` as a locked test set, and standardizes the features with a scaler fitted on the tuning districts.
- with `MLFS_OFFLINE=1` (tests, `verify.py`), a 1,000-row synthetic stand-in whose target is *linear* in the features. the search still runs, but its conclusions don't carry over: it's a smoke test.
- **tests** use only cheap synthetic objectives defined in `test_capstone.py` (a 1-D function with two bumps, the Branin function).

## the search problem

the model is `KernelRidge(alpha, kernel="rbf", gamma)`, and the two hyperparameters are searched on a log scale: $x = (\log_{10}\alpha, \log_{10}\gamma) \in [-6, 2] \times [-4, 2]$. the objective is minus the cross-validated RMSE, so larger is better and every search *maximizes*. the box is deliberately wide: most of it is bad (very large $\alpha$ predicts the mean; very large $\gamma$ with tiny $\alpha$ memorizes), and the good region is a narrow diagonal valley.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 300 tuning / 200 test rows (synthetic); budget 12 (4 initial); 2 seeds; 6 × 6 reference grid | well under a minute |
| **full** (your write-up) | default | 1,500 tuning / 4,000 test districts; budget 30 (5 initial); 6 seeds per method; 16 × 16 reference grid | about 3 minutes (195 s measured), almost all of it in the 616 objective evaluations (each a 5-fold kernel ridge fit on 1,500 points) |

the notebook picks the config from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `propose(gp, candidates, best, *, xi=0.01)`: the index of the candidate with the highest expected improvement under a fitted GP, and the EI of every candidate
   - `bayes_opt(objective, bounds, *, n_init=5, n_iter=20, n_candidates=1000, xi=0.01, rng=None)`: the loop. `n_init` uniform random points, then `n_iter` rounds of: fit a Matérn-5/2 GP (with learned hyperparameters) to all evaluations so far *on the unit cube*, propose the best of `n_candidates` random candidates plus a cloud of local candidates around the current best, evaluate it. returns `dict(X, y, best_x, best_y, trace)` with `trace` the best value so far after each evaluation
   - `random_search(objective, bounds, *, n_iter=25, rng=None)`: the baseline, same return value
   - `krr_objective(X, y, *, cv=5, rng=None)`: a function $x \mapsto -\text{CV RMSE}$ of `KernelRidge(10**x[0], gamma=10**x[1])`, with targets centred on each training fold and the folds fixed once
   - `compare_searches(objective, bounds, *, budget=25, n_init=5, n_seeds=5, rng=None)`: both searches over `n_seeds` seeds (each seed shared by the two methods), returning their traces as `(n_seeds, budget)` arrays
2. the write-up section at the end of the notebook, answering its five prompts (the result, why it works, the baseline, the cost of choosing, limits), with numbers from experiments A–C.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `bayes_opt` on a two-bump 1-D function | within 0.01 of the global maximum in 16 evaluations, in at least 2 of 3 runs |
| `bayes_opt` on the Branin function | median regret (over 5 runs) below 0.6 with 25 evaluations, and below that of 25 random evaluations (median over 50 runs) |
| budget discipline | the objective is called exactly `n_init + n_iter` (or `n_iter`) times |
| full-config report | every number in your write-up is reproducible from the notebook |

for reference, in the full config the reference solution measures:

- **the landscape** (experiment A, 256 evaluations): the best grid point has CV RMSE **0.5976** (at $\log_{10}\alpha = \log_{10}\gamma = -1.2$); the median grid point **0.873**, the worst **4.35**. the mean predictor scores about 1.16.
- **the race** (experiment B, 6 seeds, 30 evaluations each): Bayesian optimization ends at a median best CV RMSE of **0.5980** (range 0.5975–0.6006 over seeds), random search at **0.6028** (range 0.6015–0.6138). after 15 evaluations: 0.6032 against 0.6069. all **6/6** BO runs end within 0.005 of the grid optimum, against **3/6** for random search. paired by seed (the two methods share seeds), BO's final CV RMSE is lower in **6/6** seeds, by +0.0024 to +0.0159 (mean **0.0063**). the first 5 evaluations of the two methods coincide (they share seeds, and both start with uniform draws), so the curves separate only once the GP takes over.
- **the test set** (experiment C, 4,000 districts, used once): the class defaults ($\alpha = \gamma = 1$) give test RMSE **0.733** (95% bootstrap CI 0.709–0.760); the first random-search run's choice **0.598** (0.576–0.622); the first BO run's choice ($\log_{10}\alpha = -0.75$, $\log_{10}\gamma = -1.17$) **0.594** (0.573–0.616); the mean predictor 1.166.

read those numbers carefully. tuning matters a lot: 0.73 → 0.59, a \$14,000 smaller typical error. *which* search found the tuned values matters much less here: the valley is broad enough that random search with 30 evaluations usually lands in it, and the difference between the two tuned models on the test set is far inside either CI. what BO buys is reliability and speed of getting close (every run within 0.005 of the optimum), which matters most when each evaluation costs hours instead of a third of a second.

## stretch goals

- **a harder landscape.** add a third hyperparameter (the kernel: RBF vs Laplacian, or the degree of a polynomial kernel) or shrink the budget to 15. how does the gap between the methods change? Bergstra & Bengio's argument says random search gets *relatively* better when few dimensions matter; check it.
- **the acquisition function.** replace EI with the upper confidence bound $\mu + \kappa\sigma$ for $\kappa \in \{0.5, 2, 5\}$, or vary $\xi$. plot where each one evaluates.
- **noise.** make the objective noisy (a different CV split on every call) and fit the GP's noise. what happens to "the best observed value" as the incumbent, and how would you choose the final hyperparameters instead?
- **what the GP believes.** after a BO run, plot the GP's posterior mean and standard deviation over the box next to experiment A's grid. where is it confident and wrong?
- **an SVM.** tune your `SVC`'s $C$ and $\gamma$ on 4s versus 9s from MNIST (topic 04's lab) with cross-validated accuracy as the objective. the objective is now a step function of the hyperparameters; does the GP cope?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Jones, Schonlau & Welch (1998), "Efficient global optimization of expensive black-box functions", *Journal of Global Optimization* 13: 455–492. [doi:10.1023/A:1008306431147](https://doi.org/10.1023/A:1008306431147)
- Snoek, Larochelle & Adams (2012), "Practical Bayesian optimization of machine learning algorithms", *NeurIPS*. [arXiv:1206.2944](https://arxiv.org/abs/1206.2944)
- Shahriari, Swersky, Wang, Adams & de Freitas (2016), "Taking the human out of the loop: a review of Bayesian optimization", *Proceedings of the IEEE* 104(1): 148–175. [doi:10.1109/JPROC.2015.2494218](https://doi.org/10.1109/JPROC.2015.2494218)
- Bergstra & Bengio (2012), "Random search for hyper-parameter optimization", *JMLR* 13: 281–305. [jmlr.org/papers/v13/bergstra12a](https://www.jmlr.org/papers/v13/bergstra12a.html)
- Garnett (2023), *Bayesian Optimization*, Cambridge University Press. [bayesoptbook.com](https://bayesoptbook.com/)
- Pace & Barry (1997), "Sparse spatial autoregressions", *Statistics & Probability Letters* 33(3): 291–297. the source of the California housing data. [doi:10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X)
