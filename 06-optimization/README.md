# 06 · optimization

> understand why and how fast optimizers converge, and build the optimizer library you'll train everything with.

almost everything in machine learning ends in "and then minimize this". so far you've done it with whatever worked: a gradient step in chapter 03, a Newton step in its capstone. this chapter makes optimization a subject. you'll learn to tell when a problem is easy (convex) and what that guarantees; how fast gradient descent converges and why the condition number sets its speed; how momentum buys a square root of that condition number and why nothing that uses only gradients can do better; how minibatch noise and adaptive step sizes change the picture; how Newton and quasi-Newton methods use curvature; how constraints turn into prices; and what goes wrong when the landscape isn't convex at all. along the way you'll be surprised by a step-size rule that beats the "optimal" fixed step, an implicit bias that steers gradient descent towards the maximum-margin classifier, a heavy-ball method that cycles forever on a function it's provably optimal for in the quadratic case, an Adam that loses its advantage when you rotate the problem, an L-BFGS that beats full BFGS, and a 1,000-dimensional egg carton that looks like a smooth slope.

**you will build:** a numerical convexity checker (chords of sets, Jensen gaps, tangent planes, Hessian eigenvalues); `OptimResult`, Armijo backtracking and gradient descent with exact-line-search analysis; heavy-ball and Nesterov momentum, their exact rates on quadratics, and Nesterov's worst-case function; the `Param`/`Optimizer` protocol with SGD (momentum, Nesterov, weight decay), Adagrad, RMSProp, Adam and AdamW, minibatching, and the theory of SGD's noise floor; damped Newton, a Wolfe line search, BFGS, L-BFGS and a `minimize` front end; projections onto boxes, balls and the simplex, projected gradient descent, equality-constrained QPs through their KKT system and inequality-constrained ones through their dual; the Rosenbrock and Rastrigin test functions, perturbed gradient descent, loss-landscape slices and trajectory PCA. module: **`mlfs.optim`** (topics 02–07), which chapters 08, 09, 12, 14, 15 and later train with.
**prerequisites:** chapter 02 (linear solves, eigenvalues, the SVD, matrix calculus), chapter 03 (gradients and Hessians, Taylor models, smoothness, the descent lemma, convergence rates, Newton's method), chapter 04 (expectations and variances, sampling), and chapter 00's `as_rng`. chapter 05 isn't needed.
**time · scaffolding:** about 90 hours · L2 (supported): every function comes with its signature and a docstring with shapes. hints have three levels. the capstone gives you the experiments and the report; you write the harness.
**newly unlocked library components:** after this chapter you may use `scipy.optimize.minimize` (BFGS, L-BFGS-B, Newton-CG, trust regions), constrained solvers (`scipy.optimize.minimize(method="trust-constr")`, `scipy.optimize.linprog`) and `scipy.optimize.line_search` in labs and experiments, since you've built the methods they implement. inside `mlfs` modules, later chapters train with *your* optimizers through `mlfs.optim`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md): every gradient-trained model from chapter 08 on uses your `minimize`, `lbfgs`, `SGD`, `Adam` or `AdamW`, and chapter 14's `Tensor` plugs into your `Optimizer` protocol. PyTorch's `torch.optim` is unlocked with the rest of PyTorch in chapter 15.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [convex sets & functions](01-convex-sets-and-functions/) | `in_polyhedron`, `set_convexity_violation`, `jensen_gap`, `min_jensen_gap`, `first_order_gap`, `min_first_order_gap`, `min_hessian_eigenvalue`, `check_convexity` | squared error on a sigmoid and a "linear" two-layer model are nonconvex; the Jensen test catches a tiny nonconvex bump with fewer points than the Hessian test; a robust loss has round sublevel sets but isn't convex |
| 02 | [gradient descent](02-gradient-descent/) | `OptimResult`, `gradient_descent`, `backtracking_line_search` → `mlfs.optim`, `exact_step`, `steepest_descent_quadratic` | the best possible step size is still $O(\kappa)$, backtracking beats it by varying its steps, and on separable data the loss goes to 0 while $\lVert w\rVert$ grows like $\log k$ and its direction creeps to the max-margin one |
| 03 | [momentum & acceleration](03-momentum-and-acceleration/) | `heavy_ball`, `nesterov` → `mlfs.optim`, `momentum_spectral_radius`, `tuned_momentum`, `worst_case_quadratic`, `first_order_lower_bound` | 22× fewer iterations at $\kappa = 1{,}000$; Nesterov's schedule makes $f$ go *up* on 42% of its steps; "optimal" heavy ball cycles forever on a 1-D strongly convex function |
| 04 | [stochastic optimization](04-stochastic-optimization/) | `Param`, `Optimizer`, `minibatch_indices`, `SGD`, `Adagrad`, `RMSProp`, `Adam`, `AdamW` → `mlfs.optim`, `minibatch_gradient_variance`, `sgd_noise_floor` | Adam wins by ten orders of magnitude on an axis-aligned problem and loses to SGD after a rotation; reshuffling beats the noise-floor theory; SGD finds the min-norm interpolator and Adam a worse one |
| 05 | [second-order methods](05-second-order-methods/) | `newton`, `wolfe_line_search`, `bfgs_update`, `bfgs`, `two_loop`, `lbfgs`, `minimize` → `mlfs.optim` | Newton needs 6 iterations at every size but loses on time from $d \approx 200$; L-BFGS with 5 pairs beats full BFGS, and with unlimited memory it's fastest of all: BFGS loses to its one-time initial scaling, not to stale curvature |
| 06 | [constraints & duality](06-constraints-and-duality/) | `project_box`, `project_l2_ball`, `project_simplex`, `projected_gradient_descent`, `solve_eq_qp` → `mlfs.optim`, `qp_dual`, `kkt_residuals`, `solve_qp_dual` | only 2 of 40 SVM multipliers are nonzero; the simplex makes 14 of 20 weights exactly zero; multipliers predict the effect of moving a constraint to five decimals |
| 07 | [nonconvex optimization](07-nonconvex-optimization/) | `rosenbrock`, `rosenbrock_grad`, `rastrigin`, `rastrigin_grad`, `perturbed_gradient_descent`, `loss_landscape_2d` → `mlfs.optim`, `trajectory_pca` | small random kicks don't help at all on Rastrigin; 6-D Rosenbrock traps 12% of L-BFGS runs (and the classic start) in a local minimum; a random slice of 1,000-D Rastrigin shows no local minima |

## capstone

*an optimizer benchmark suite*: every optimizer on standard test functions and logistic regression, with a convergence report. you build the logistic objective with exact derivatives, a runner that counts every function, gradient and Hessian call and converts them into a common cost, a minibatch runner for the `Optimizer` family, and the cost-to-tolerance bookkeeping behind the report; then you benchmark everything on an ill-conditioned quadratic, Rosenbrock in 2-D and 10-D, Rastrigin, and logistic regression on California housing with raw features (Hessian condition number about $2\times10^9$) and standardized ones (about 180). small CPU config: a few seconds (offline). full config: 40–90 seconds, depending on the machine.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (backtracking and a zig-zag, momentum and Adam step by step, a QP through its KKT conditions and its dual), and a 60-minute blind rebuild of the stateful optimizer family.

## notation

| symbol | meaning |
|---|---|
| $f$, $f^*$, $x^*$ | the objective, its optimal value, a minimizer |
| $g$, $\nabla f(x)$, $H$, $\nabla^2f(x)$ | gradient and Hessian at $x$ |
| $L$, $\mu$, $\kappa = L/\mu$ | smoothness constant, strong-convexity constant, condition number |
| $\eta$ (`lr`), $\alpha$ | learning rate (fixed step), step length chosen by a line search |
| $\rho$, $c$ ($c_1$, $c_2$) | backtracking factor; Armijo (and Wolfe curvature) constants |
| $\beta$ (`momentum`), $\beta_k$ | momentum coefficient, constant or scheduled ($\frac{k-1}{k+2}$) |
| $v$, $m$, $\hat m$, $\hat v$ | momentum buffer; Adam's first and second moment averages and their bias-corrected versions |
| $\beta_1, \beta_2$, $\epsilon$ | Adam's averaging rates and its denominator guard (not machine epsilon, which is $\epsilon_{\text{mach}}$ here) |
| $\lambda$ (`weight_decay`) | weight decay / ridge coefficient (in the capstone the ridge coefficient is $\alpha$) |
| $b$, $B$ | batch size, a minibatch of indices |
| $s_k = x_{k+1} - x_k$, $y_k = g_{k+1} - g_k$, $H_k$ | quasi-Newton step and gradient-change pairs, inverse-Hessian approximation |
| $P_C(v)$ | Euclidean projection onto the convex set $C$ |
| $\mathcal L(x, \lambda)$, $\lambda$ (multipliers), $g(\lambda)$ | Lagrangian, Lagrange/KKT multipliers, dual function (context distinguishes $g(\lambda)$ from the gradient $g$) |
| $\rho(\cdot)$ | spectral radius of an iteration (rate on a quadratic) |
| $\xi$, $r$ | a random perturbation and its radius |

## how this connects

- builds on: linear solves, eigenvalues, the SVD and matrix calculus (ch 02, `mlfs.linalg`), gradients, Hessians, Taylor models, smoothness and Newton's method (ch 03, `mlfs.autodiff`), variance of averages and inverse-CDF sampling (ch 04), `as_rng` (ch 00).
- used later in: logistic and softmax regression, GLMs and the lasso (ch 08), the SVM dual and Gaussian-process hyperparameters with `lbfgs` (ch 09), variational inference with `Adam` (ch 12), the NumPy training loop with `minibatch_indices` and the optimizer protocol (ch 14), schedulers, gradient clipping and the LR range test (ch 15), loss landscapes and implicit bias (ch 19), RL policy optimization (ch 23–24), `ZeroAdamW` and memory accounting for optimizer state (ch 26), PGD attacks and DP-SGD (ch 28).

## core references

- Boyd & Vandenberghe (2004), *Convex Optimization*, Cambridge University Press. convex sets and functions, duality, KKT, descent and Newton methods; free online. [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/)
- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer. line searches, Newton, quasi-Newton, L-BFGS and constrained optimization. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Nesterov (2018), *Lectures on Convex Optimization*, 2nd ed., Springer. rates, lower bounds and accelerated methods. [doi:10.1007/978-3-319-91578-4](https://doi.org/10.1007/978-3-319-91578-4)
- Bottou, Curtis & Nocedal (2018), "Optimization methods for large-scale machine learning", *SIAM Review* 60(2). stochastic methods for ML. [arXiv:1606.04838](https://arxiv.org/abs/1606.04838)
- Beck (2017), *First-Order Methods in Optimization*, SIAM. projected and proximal gradient methods. [doi:10.1137/1.9781611974997](https://doi.org/10.1137/1.9781611974997)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, chapter 8 (optimization for training deep models). [deeplearningbook.org](https://www.deeplearningbook.org)
