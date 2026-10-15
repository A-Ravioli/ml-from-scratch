# 03 · capstone: an autodiff-powered newton solver

## goal

build the method that every serious solver in numerical computing descends from: Newton's method, for solving systems of equations $F(x) = 0$ and for minimizing functions $f(x)$, with every derivative it needs (Jacobians, gradients and Hessians) computed **exactly** by your own forward-mode autodiff library, `mlfs.autodiff`. then take it apart: measure its order of convergence and see when that order drops, compare it with gradient descent in iterations and wall-clock time, map out where it converges from, and use it to fit a logistic regression by maximum likelihood without writing down a single derivative.

this capstone pulls the whole chapter together: convergence orders and their estimation (topic 01), Jacobians and Hessians (topic 02), quadratic models and why Newton can head for saddles (topic 03), dual numbers, including nested ones for second derivatives (topic 04), and line searches and curvature (topic 05). it's also a preview of chapter 06, where Newton, quasi-Newton and gradient methods become the optimizer library you'll train models with.

## data

none to download. every problem is synthetic and generated in the notebook or the tests: polynomial equations and systems, the Rosenbrock function, a double well, Newton's method for $z^3 = 1$ written as a real system, and a logistic-regression dataset drawn from a known model with a fixed seed. the tests use only their own small synthetic problems.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | basin grid 40 × 40; gradient descent capped at 2,000 steps | a few seconds |
| **full** (your write-up) | default | basin grid 200 × 200; gradient descent capped at 20,000 steps | about 10–20 seconds with the reference `mlfs.autodiff`; your own `Dual` may be slower |

the notebook picks the config from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the four required functions implemented and their checks green:
   - `newton_root(F, x0, *, tol, max_iters)` → `(x, iterates)`: Newton for $F: \mathbb{R}^n \to \mathbb{R}^n$, Jacobians from your `jacobian`
   - `hessian(f, x)`: the exact Hessian from nested dual numbers, $n(n+1)/2$ evaluations
   - `newton_minimize(f, x0, *, tol, max_iters)` → `(x, iterates)`: damped Newton, with a Hessian modification that guarantees a descent direction and a backtracking (Armijo) line search that stops the run if no step gives sufficient decrease. choose `tol` above the gradient's rounding floor (about $10^{-10}$–$10^{-12}$ relative to the size of $f$'s terms); below it, the line search, not `tol`, ends the run
   - `estimate_order(errors, *, floor)`: the median of the last three order estimates, ignoring errors at the rounding floor
2. the write-up section at the end of the notebook, answering its five prompts (convergence orders, Newton versus gradient descent, safeguards, basins, and where exactness mattered).

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| Newton iterates on a 2-D system | match Newton with the exact Jacobian to $10^{-13}$ (so the Jacobian must be exact) |
| Hessians | exact to relative error $10^{-12}$, exactly symmetric |
| damped Newton on Rosenbrock from $(-1.2, 1)$ | reaches $(1, 1)$ to $10^{-8}$ in at most 40 iterations, with $f$ never increasing |
| damped Newton from an indefinite-Hessian start on a double well | ends at a minimum, not at the saddle that pure Newton finds |
| estimated orders in experiment A | about 2 at simple roots, about 1 at the double root (report them) |

for reference, the solution takes **5** Newton steps on $x^3 - 2x - 5$ from $x_0 = 3$ (order ≈ 1.92 by `estimate_order`), **35** on the double root (order 1.00), **21** damped-Newton steps on Rosenbrock against about **19,400** for gradient descent with the same line search, and **5** steps for the logistic regression.

## stretch goals

- **Newton–Krylov.** replace the explicit Jacobian in `newton_root` with your `jvp` and an iterative linear solver (conjugate gradients for symmetric systems, or GMRES), so a Newton step never forms $J$. measure the cost in forward passes per step for $n = 50$.
- **quasi-Newton.** implement the secant (Broyden) update for $J$ in `newton_root`, or BFGS for `newton_minimize`, and compare iterations and function evaluations with your exact-derivative Newton. (chapter 06 · topic 05 builds BFGS properly.)
- **Hessian-vector products.** compute $Hv$ in $n$ (or fewer) passes by applying `jvp` to a gradient computed with nested duals, and use it in a truncated-Newton method.
- **basins up close.** zoom into a boundary of the basin picture (experiment C) and count the starts that take more than 20 iterations. what's the structure?
- **multiplicity-aware Newton.** detect a multiple root from your estimated linear rate $C \approx (m-1)/m$, and switch to the modified step $x - m\,F/F'$ (topic 01, on paper 2) to restore quadratic convergence.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer, chapters 3 (line searches), 3.4 (Hessian modification) and 11 (Newton's method for nonlinear equations). [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Dennis & Schnabel (1996), *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*, SIAM. the classic treatment of Newton's method, its local convergence and its globalization. [doi:10.1137/1.9781611971200](https://doi.org/10.1137/1.9781611971200)
- Kelley (2003), *Solving Nonlinear Equations with Newton's Method*, SIAM. a short, practical guide, including Newton–Krylov methods. [doi:10.1137/1.9780898718898](https://doi.org/10.1137/1.9780898718898)
- Griewank & Walther (2008), *Evaluating Derivatives*, 2nd ed., SIAM, chapter 13 (higher derivatives by nested forward mode). [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
