# 03 · calculus & analysis

> multivariable calculus, approximation, and enough analysis to reason about convergence and smoothness.

every training run is a sequence of iterates, every loss is a function of many variables, and every optimizer acts on a local model of that function. this chapter gives you the tools to reason about all three. **analysis** answers "does it converge, how fast, and when do I stop?" (and warns you that "the last step was tiny" proves nothing). **multivariable calculus** gives you gradients, Jacobians and Hessians, and with them the geometry of loss surfaces: steepest directions, curvature, saddles. **approximation** says when a local model can be trusted, and when a perfectly smooth function's Taylor series diverges anyway. along the way you'll build an exact differentiation engine out of numbers that carry their own derivatives, and use it to drive Newton's method, the ancestor of every second-order optimizer.

**you will build:** an empirical convergence-rate estimator and three root finders; directional derivatives, numerical Jacobians and Hessians, and a critical-point classifier; Taylor polynomials, error bounds, multivariate Taylor models and a trust-radius tool; a `Dual` number class with exact `derivative`, `grad`, `jacobian` and `jvp`; Lipschitz, smoothness and strong-convexity estimators and a step-size stability finder; grid and Monte Carlo integrators in $d$ dimensions and densities of transformed random variables; $L^2$ inner products, Legendre polynomials and Fourier and Legendre projections. module: **`mlfs.autodiff`** (topics 02 and 04).
**prerequisites:** chapter 00 (NumPy, floating point, operator overloading), chapter 01 (derivatives, finite differences, integrals, Monte Carlo), and chapter 02 (matrix calculus and `mlfs.numgrad`, eigenvalues, linear solves).
**time · scaffolding:** about 90 hours · L2 (supported): every function comes with its signature and a docstring with shapes. hints have three levels. the capstone gives you the experiments and evaluation; you write the solver.
**newly unlocked library components:** after this chapter you may use `scipy.optimize.approx_fprime` and finite-difference helpers, `np.polynomial.legendre` / `np.polynomial.chebyshev`, `np.fft` for Fourier coefficients, and `scipy.integrate` quadrature (`quad`, `dblquad`) and Monte Carlo-style integration in labs and experiments, since you've built their simplest versions. forward-mode autodiff in other libraries (`jax.jvp`, `torch.func.jvp`) stays locked until chapter 14 has you build reverse mode, and the chapters before 15 stay NumPy-only. inside `mlfs` modules, later chapters use *your* derivatives through `mlfs.autodiff`, per the contracts in [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [sequences, limits & convergence](01-sequences-limits-and-convergence/) | `tail_index`, `cauchy_diameters`, `order_estimates`, `rate_estimates`, `bisection`, `newton_1d`, `secant` | the secant method's order is the golden ratio 1.618, and Newton at a double root is no faster than bisection |
| 02 | [multivariable calculus](02-multivariable-calculus/) | `directional_derivative`, `steepest_direction`, `numerical_jacobian`, `numerical_hessian` → `mlfs.autodiff`, `classify_critical_point` | "steepest descent" changes direction when you change units; Newton's direction doesn't |
| 03 | [taylor approximation](03-taylor-approximation/) | `taylor_poly_1d`, `lagrange_bound`, `loglog_slope`, `quadratic_model`, `taylor_model`, `agreement_ratio`, `trust_radius` | the Taylor series of the perfectly smooth $1/(1 + x^2)$ diverges beyond $\lvert x\rvert = 1$, and Newton walks to a saddle from half the plane |
| 04 | [forward-mode autodiff](04-forward-mode-autodiff/) | `Dual`, `dual_parts`, `derivative`, `value_and_grad`, `grad`, `jvp`, `jacobian` → `mlfs.autodiff` | exact derivatives come out tens of times *slower* than finite differences in pure Python (about 60× on the author's laptop; machine-dependent), yet finite differences are 100% wrong on a chaotic map |
| 05 | [smoothness & lipschitz continuity](05-smoothness-and-lipschitz-continuity/) | `lipschitz_estimate`, `smoothness_estimate`, `strong_convexity_estimate`, `descent_lemma_gap`, `hessian_smoothness`, `gradient_descent`, `largest_stable_step` | gradient descent survives steps 5× beyond $2/L$, and 5,000 random pairs underestimate $L$ by more than half in 50-D |
| 06 | [integration & change of variables](06-integration-and-change-of-variables/) | `midpoint_rule`, `monte_carlo_integrate`, `log_abs_det_jacobian`, `pushforward_log_density` | forgetting the Jacobian is harmless for a nonlinear shear, and the peak of $e^Z$ isn't at $e^{\text{peak of } Z}$ |
| 07 | [function spaces](07-function-spaces/) | `l2_inner`, `l2_norm`, `legendre`, `legendre_coefficients`, `legendre_series`, `fourier_coefficients`, `fourier_series` | the Gibbs overshoot never shrinks, and interpolating more points makes Runge's function worse |

**further reading (★ advanced track, optional).** CURRICULUM.md lists an advanced track of real-analysis rigor, measure theory and Lebesgue integration: proofs, not code, so it isn't built as a topic. if you want it, read Abbott (2015), *Understanding Analysis*, for rigorous single-variable analysis; Tao (2016), *Analysis I* and *II*, for a construction-first treatment; and Axler (2020), *Measure, Integration & Real Analysis* (free online, [doi:10.1007/978-3-030-33143-6](https://doi.org/10.1007/978-3-030-33143-6)), for Lebesgue integration and the $L^p$ spaces that topic 07 only sketches. chapter 04's advanced track (measure-theoretic probability) builds on it.

## capstone

*an autodiff-powered Newton solver*: root finding and minimization with your forward-mode library, plus convergence analysis. you build Newton's method for systems (exact Jacobians from your `jacobian`), exact Hessians from nested dual numbers, a damped Newton minimizer with a Hessian fix and a line search, and a robust order estimator; then you measure quadratic versus linear convergence, race Newton against gradient descent on Rosenbrock, map Newton's basins of attraction for $z^3 = 1$, and fit a logistic regression with no hand-written derivatives. small CPU config: a few seconds (offline). full config: under half a minute.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (Newton steps and orders, a Taylor model and a Newton direction, dual numbers and a transformed density), and a 60-minute blind rebuild of forward-mode autodiff.

## notation

| symbol | meaning |
|---|---|
| $a_n \to L$, $e_k = \lvert x_k - x^*\rvert$ | a sequence converging to $L$; the error of an iterative method at step $k$ |
| $q$, $C$ | order and rate of convergence, $e_{k+1} \approx Ce_k^{\,q}$ |
| $\nabla f(x)$ | gradient of $f: \mathbb{R}^n \to \mathbb{R}$, same shape as $x$ |
| $D_vf(x) = \nabla f(x)\cdot v$ | directional derivative along $v$ (not normalized) |
| $J_F(x)$ | Jacobian of $F: \mathbb{R}^n \to \mathbb{R}^m$, shape $(m, n)$, $(J_F)_{ij} = \partial F_i/\partial x_j$ |
| $H(x)$, $\nabla^2 f(x)$ | Hessian, symmetric $(n, n)$, $H_{ij} = \partial^2f/\partial x_i\partial x_j$ |
| $\lambda_{\min}, \lambda_{\max}$ | smallest and largest eigenvalues of a symmetric matrix (here usually a Hessian) |
| $T_K$, $m(x_0 + d) = f_0 + g^\top d + \tfrac12d^\top Hd$ | degree-$K$ Taylor polynomial; quadratic model at $x_0$ |
| $\rho$ | agreement ratio, actual reduction / predicted reduction |
| $a + b\varepsilon$, $\varepsilon^2 = 0$ | dual number with real part $a$ and dual part $b$ |
| $J_F\,v$ | Jacobian-vector product (one forward-mode pass) |
| $L$, $\mu$, $\kappa = L/\mu$ | Lipschitz or smoothness constant, strong-convexity constant, condition number |
| $\eta$ | step size (learning rate) |
| $p_Z$, $p_X$, $X = T(Z)$ | densities of a random variable and of its image under an invertible map |
| $\lvert\det J_T\rvert$ | volume-scaling factor of $T$ |
| $\langle f, g\rangle = \int fg$, $\lVert f\rVert$ | $L^2$ inner product and norm of functions on an interval |
| $P_k$ | Legendre polynomial of degree $k$ on $[-1, 1]$ |
| $a_k$, $b_k$ | Fourier cosine and sine coefficients on $[-\pi, \pi]$, with $f \approx a_0/2 + \sum_k(a_k\cos kx + b_k\sin kx)$ |
| $\varepsilon$ (in error analysis) | machine epsilon of double precision, the gap between 1 and the next double, $\approx 2.2 \times 10^{-16}$ (the unit roundoff is half of it); order-of-magnitude arguments round it to $10^{-16}$. context distinguishes it from the dual-number $\varepsilon$ |

## how this connects

- builds on: NumPy, floating point and operator overloading (ch 00), derivatives, finite differences, quadrature and Monte Carlo (ch 01), matrix calculus, `mlfs.numgrad`, eigenvalues, linear solves and conditioning (ch 02).
- used later in: convergence rates, line searches, Newton and quasi-Newton methods (ch 06, which builds `mlfs.optim`), the multivariate Gaussian and limit theorems (ch 04), kernel methods and RKHSs (ch 09), HMC with `mlfs.autodiff.grad` (ch 12), reverse-mode autodiff and backpropagation (ch 14), learning-rate tuning and gradient clipping (ch 15), spectral bias and loss-landscape curvature (ch 19), normalizing flows (ch 21), and flow matching (ch 22).

## core references

- Abbott (2015), *Understanding Analysis*, 2nd ed., Springer. limits, Cauchy sequences, series and completeness. [doi:10.1007/978-1-4939-2712-8](https://doi.org/10.1007/978-1-4939-2712-8)
- Hubbard & Hubbard (2015), *Vector Calculus, Linear Algebra, and Differential Forms: A Unified Approach*, 5th ed., Matrix Editions. derivatives as linear maps, Taylor's theorem in several variables, and change of variables.
- Deisenroth, Faisal & Ong (2020), *Mathematics for Machine Learning*, Cambridge University Press, chapters 5–6. vector calculus and probability with ML examples; free online. [mml-book.github.io](https://mml-book.github.io)
- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer. rates of convergence, finite differences, trust regions and Newton's method. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Griewank & Walther (2008), *Evaluating Derivatives*, 2nd ed., SIAM. the reference on automatic differentiation. [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
- Trefethen (2019), *Approximation Theory and Approximation Practice*, extended ed., SIAM. polynomial approximation, Runge and Gibbs. [doi:10.1137/1.9781611975949](https://doi.org/10.1137/1.9781611975949)
