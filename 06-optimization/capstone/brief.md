# 06 · capstone: an optimizer benchmark suite

## goal

you've built a dozen optimizers in this chapter. which one should you use? the honest answer is "it depends", and this capstone makes you find out *on what*. you'll build a small benchmark harness: an objective for logistic regression, a runner that measures what each deterministic method really costs (function, gradient and Hessian evaluations, not just iterations), a runner for the minibatch `Optimizer` family, and the bookkeeping that turns runs into a convergence report. then you'll benchmark every optimizer in your `mlfs.optim` on standard test functions (an ill-conditioned quadratic, Rosenbrock in 2-D and 10-D, Rastrigin) and on a logistic regression with real data, with raw and with standardized features, and write up what you find.

this pulls the whole chapter together: convexity and what it buys (topic 01), the $\kappa$ and $\sqrt\kappa$ rates (topics 02–03), stochastic and adaptive methods (topic 04), Newton and quasi-Newton (topic 05), and nonconvex landscapes (topic 07). it's also your first experiment with the evaluation habits the rest of the course insists on: a fair cost measure, several seeds, and a baseline.

## data

- **test functions**: generated in the notebook (no download): a 50-D quadratic with condition number 1,000, Rosenbrock (2-D and 10-D) and Rastrigin (5-D), all from `mlfs.optim`.
- **California housing** via `mlfs.datasets.california_housing()` (1990 census, 20,433 districts after dropping rows with missing values, 8 numeric features), downloaded once and cached under `~/.cache/mlfs/`. the task is binary: is a district's median house value above the median? the features have wildly different scales (median income around 4, total rooms in the thousands), which is the point of experiment C.
- **tests** use only small synthetic logistic regressions and hand-made records.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | California's 1,000-row synthetic stand-in (its features are already on similar scales, so "raw" and "standardized" coincide); first-order methods capped at 4,000 iterations; 20 epochs, 2 seeds; 8 Rastrigin starts | a few seconds |
| **full** (your write-up) | default | the real data (20,433 rows); caps of 20,000 iterations; 25 epochs, 5 seeds; 40 Rastrigin starts | 40–90 seconds with the reference `mlfs.optim`, depending on the machine |

the notebook picks the config from `datasets.is_offline()`. the adaptive methods' learning rates differ between the two configs (the small run takes about 20 times fewer minibatch steps); tuning them is part of the stretch goals.

## deliverables

1. `capstone.py` with the four required functions implemented and their checks green:
   - `logistic_objective(X, y, alpha)` → `(f, grad, hess)`: the ridge-regularized mean logistic loss with its exact gradient and Hessian, finite for any weights
   - `run_deterministic(method, f, grad, x0, *, hess, max_iters, tol, **options)` → a record: `minimize` with counted calls, the cost (in gradient-equivalents: $n_f + n_\nabla + d\,n_H$) after every iterate, and the wall-clock time
   - `run_stochastic(make_optimizer, loss_grad, full_loss, w0, n, *, epochs, batch_size, rng)` → a record: the `zero_grad → grad → step` loop over `minibatch_indices`, with the full loss after every epoch (cost: one per pass over the data)
   - `cost_to_reach(record, f_star, rel_tol)` and `rank_methods(records, f_star, rel_tol)`: when each run first got within `rel_tol` of the optimum (relative to its starting gap), and the resulting ranking
2. the write-up section at the end of the notebook: a convergence report answering its five prompts (ranking, nonconvexity, conditioning, stochastic versus deterministic, recommendations).

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `logistic_objective` | gradient and Hessian match finite differences; no overflow at $\lvert z\rvert \sim 10^4$ |
| `run_deterministic` | counts match the optimizer's actual calls; the final cost is the total work |
| `run_stochastic` | every example once per epoch, reproducible from `rng`; Adam within $5\times10^{-3}$ of $f^*$ on the test problem in 30 epochs |
| full-config report | every number in your write-up is reproducible from the notebook |

for reference, in the full config the solution's L-BFGS reaches $10^{-8}$ relative suboptimality on 2-D Rosenbrock after about **100** gradient-equivalents (gradient descent with a line search: about **69,000**), Newton solves the quadratic in **one** step, the Hessian of the raw-feature logistic regression has condition number about **$2\times10^9$** (standardized: about **180**), and with raw features gradient descent and Nesterov don't reach even $10^{-2}$ in 2,000 iterations while Newton needs **5–6** iterations either way.

## stretch goals

- **tune fairly.** give every first-order and adaptive method a small learning-rate grid (say 5 values, log-spaced) and report each at its best, with the grid in the report. how much of experiment C's ranking survives?
- **a better Newton.** your `newton` falls back to gradient steps wherever the Hessian makes the Newton direction uphill, and on 10-D Rosenbrock it crawls on those fallbacks for about 280 steps before it reaches the region where Newton steps work and converges quadratically. pass a modified Hessian (shift its eigenvalues, or use the ch 03 capstone's Cholesky-with-shift) through the `hess` argument and compare: how many iterations does it save, and does it end at the global minimum or at the local one near $(-1, 1, \dots, 1)$?
- **time, not evaluations.** redo experiment A's ranking with wall-clock time on a larger problem (a 500-D quadratic, or logistic regression on MNIST pixels) and find the dimension where Newton stops winning.
- **noise floors.** for SGD with a constant learning rate, plot the final suboptimality against learning rate and batch size, and compare with topic 04's `sgd_noise_floor` prediction.
- **multi-start statistics.** for Rastrigin, estimate the probability of finding the global minimum as a function of the start box, with a confidence interval (chapter 05's bootstrap).

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer. the methods, and how to compare them. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Moré, Garbow & Hillstrom (1981), "Testing unconstrained optimization software", *ACM Transactions on Mathematical Software* 7(1). the classic test-function collection (Rosenbrock among them) and benchmarking practice. [doi:10.1145/355934.355936](https://doi.org/10.1145/355934.355936)
- Dolan & Moré (2002), "Benchmarking optimization software with performance profiles", *Mathematical Programming* 91. a principled way to aggregate "cost to reach a tolerance" across problems. [doi:10.1007/s101070100263](https://doi.org/10.1007/s101070100263)
- Schmidt, Schneider & Hennig (2021), "Descending through a crowded valley: benchmarking deep learning optimizers", *ICML*. what happens when you do this carefully for neural networks. [arXiv:2007.01547](https://arxiv.org/abs/2007.01547)
- Pace & Barry (1997), "Sparse spatial autoregressions", *Statistics & Probability Letters* 33(3). the source of the California housing data. [doi:10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X)
