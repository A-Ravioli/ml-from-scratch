# 01 · math onramp

> the precalculus, calculus, and probability needed for part I, learned by computing.

you don't need a calculus course to start this course. you need a handful of ideas (exponents and logs, slopes and areas, vectors and matrices, averages and error bars) and the confidence to use them. this chapter teaches them the way a programmer learns best: by building each one from arithmetic and checking it against NumPy. you'll compute $e^x$ from an infinite sum and $\log x$ with Newton's method, take derivatives with two function values and areas with thin strips, watch a matrix bend a circle into an ellipse, estimate $\pi$ by throwing darts, and finally put it all together to train your first machine learning model with a gradient you derived yourself. along the way you'll keep running into the same two lessons that matter for the rest of the course: **floating point has limits** (smaller steps aren't always better), and **every estimate deserves an error bar**.

**you will build:** `exp` and `log` from a series and Newton's method; finite-difference derivatives (forward, backward, central, second) and a derivative checker; Riemann, trapezoid and Simpson integrators with a convergence-order meter; dot products, angles, matrix-vector and matrix-matrix products, and a 2D transformation visualizer; dice by counting and by simulation, Monte Carlo estimates with standard errors, and $\pi$ from darts; 1-D linear regression by hand-derived gradient descent and in closed form. module: **`mlfs.numerics`** (topics 02, 03, 05): `derivative`, `check_derivative`, `trapezoid`, `simpson`, `mc_estimate`.
**prerequisites:** high school algebra (powers, functions, lines, a little trigonometry) and chapter 00 (Python, NumPy arrays and vectorization, floating point, plotting, seeds). no calculus or probability course is assumed.
**time · scaffolding:** about 70 hours · L1 (guided): every function comes with its signature, a detailed docstring with an example, and step-by-step comments inside its body. hints have three levels. the capstone gives you most of the code, and you fill in the core.
**newly unlocked library components:** after this chapter you may use `np.exp`, `np.log`, `np.log2`, `np.log10` (topic 01), `scipy.integrate.trapezoid` and `scipy.integrate.simpson` (topic 03), `@`, `np.dot`, `np.matmul` and `np.linalg.norm` on vectors (topic 04), and `np.polyfit` for straight-line fits (topic 06) freely in labs and experiments. finite-difference derivatives, quadrature and Monte Carlo estimates in later `mlfs` modules go through *your* `mlfs.numerics`, per the contracts in [`CONTRACTS.md`](../CONTRACTS.md). (see AUTHORING.md §9.)

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [functions, exponents & logs](01-functions-exponents-and-logs/) | `compose`, `bisect_inverse`, `exp_series`, range-reduced `exp`, `newton_log_steps`, `log` | the raw series for $e^x$ *never* reaches 10 digits for $x \le -9$, no matter how many terms; and $n^{0.1}$ only overtakes $\log n$ at $n \approx 3 \times 10^{15}$ |
| 02 | [derivatives](02-derivatives/) | `secant_slope`, `derivative`, `product_rule`, `chain_rule`, `check_derivative`, `second_derivative` → `mlfs.numerics` | the error is U-shaped in $h$: below $10^{-5}$, smaller steps make central differences *worse*; and each chain-rule bug passes a check at special points |
| 03 | [integrals](03-integrals/) | `riemann_sum`, `trapezoid`, `simpson`, `cumulative_trapezoid`, `convergence_order` → `mlfs.numerics` | on a periodic function the "second-order" trapezoid rule hits machine precision with 16 strips and beats Simpson by 10,000×; on $\sqrt x$ Simpson loses its edge entirely |
| 04 | [vectors & matrices, a first look](04-vectors-and-matrices-a-first-look/) | `dot`, `norm`, `angle`, `project`, `matvec`, `matmul`, `rotation`/`scaling`/`shear`, `plot_transform`, `det2`, `polygon_area` | a shear visibly stretches the circle but doesn't change its area at all, and its stretch factors are the golden ratio |
| 05 | [probability by simulation](05-probability-by-simulation/) | `dice_sum_pmf`, `roll_dice`, `empirical_pmf`, `expectation`, `variance`, `mean_and_se`, `mc_estimate` → `mlfs.numerics`, `estimate_pi` | 23 people suffice for a 50% shared birthday; and "95%" error bars for a rare event contain the truth only 38% of the time |
| 06 | [your first ML model](06-your-first-ml-model/) | `predict`, `mse_loss`, `mse_gradient`, `gradient_descent`, `closed_form_fit`, `standardize`, `unstandardize_params` | at the largest *safe* learning rate, 1,000 steps on raw house sizes leave the intercept where it started; standardized, 30 steps get the exact answer |

## capstone

*predict house prices*: a linear model of California house prices with eight features, trained by gradient descent with a gradient you derive by hand, checked against the exact least-squares answer from the normal equations, and evaluated on held-out districts against a baseline with an error bar. small CPU config: the synthetic stand-in (offline) in about a second. full config: the real 20,433 districts in a few seconds.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (Newton and series; a derivative, an integral and a gradient step; a matrix and a standard error), and a 60-minute blind rebuild of `mlfs.numerics` and the gradient descent loop.

## notation

| symbol | meaning |
|---|---|
| $f \circ g$ | composition: $(f \circ g)(x) = f(g(x))$, $g$ first |
| $f^{-1}$ | inverse function: $f^{-1}(f(x)) = x$ |
| $e$, $e^x$, $\exp x$ | Euler's number $2.71828\ldots$ and the exponential function |
| $\log x$ | the **natural** logarithm (base $e$); other bases are written $\log_2$, $\log_{10}$ |
| $k!$ | factorial $1 \cdot 2 \cdots k$, with $0! = 1$ |
| $f'(x)$, $\frac{df}{dx}$ | derivative of $f$ at $x$; $f''$ is the second derivative |
| $\frac{\partial L}{\partial w}$ | partial derivative: derivative with respect to $w$, other variables held fixed |
| $\nabla L$ | gradient: the vector of all partial derivatives |
| $h$ | step size of a finite difference, or strip width of a quadrature rule |
| $\int_a^b f(x)\,dx$ | definite integral: the (signed) area under $f$ from $a$ to $b$ |
| $n$ | number of strips (quadrature), samples (Monte Carlo), or examples (data) |
| $p$ | order of convergence: error $\approx C/n^p$ |
| $u, v, x$ (vectors) | lists of numbers; $u \cdot v$ dot product, $\|v\|$ length (2-norm) |
| $e_1, e_2$ | standard basis vectors $(1, 0)$ and $(0, 1)$ |
| $A, B$ (matrices) | $Av$ matrix times vector; $AB$ "first $B$, then $A$"; $\det A$ determinant |
| $R_\theta$ | rotation by $\theta$ radians counterclockwise |
| $P(\cdot)$, $E[X]$, $\operatorname{Var}[X]$ | probability, expected value, variance |
| $\sigma$, $s$ | true standard deviation, sample standard deviation (divides by $n - 1$) |
| $\bar x$ | sample mean $\frac1n\sum_i x_i$ |
| se | standard error of an estimate, $s/\sqrt n$ for a mean |
| $\mathbf 1\{\cdot\}$ | indicator: 1 if the condition holds, 0 otherwise |
| $\hat y$, $w$, $b$ | prediction, weight (slope), bias (intercept) of a linear model |
| $L(w, b)$ | loss: mean squared error $\frac1n\sum_i(\hat y_i - y_i)^2$ |
| $r_i$ | residual $\hat y_i - y_i$ |
| $\eta$ | learning rate |
| $\varepsilon$ | unit roundoff of double precision, $\approx 1.1 \times 10^{-16}$ |

## how this connects

- builds on: Python, NumPy, floating point, plotting and reproducibility (ch 00).
- used later in: linear algebra (ch 02) picks up vectors and matrices where topic 04 stops, and its `mlfs.numgrad` extends your finite differences to gradients; multivariable calculus and autodiff (ch 03) extend derivatives; probability and statistics (ch 04–05) formalize the simulations of topic 05; optimization (ch 06) develops the gradient descent of topic 06; and every model from chapter 07 on is trained by a loop like the one in the capstone, and evaluated against a baseline with an error bar.

## core references

- Strang (1991), *Calculus*, Wellesley-Cambridge Press. a complete calculus course with an eye to applications, free on MIT OpenCourseWare. companion for topics 01–03.
- Deisenroth, Faisal & Ong (2020), *Mathematics for Machine Learning*, Cambridge University Press. vectors, matrices, calculus and probability with ML in mind; free online. [doi:10.1017/9781108679930](https://doi.org/10.1017/9781108679930)
- Blitzstein & Hwang (2019), *Introduction to Probability*, 2nd ed., CRC Press. companion for topic 05, free online. [doi:10.1201/9780429428357](https://doi.org/10.1201/9780429428357)
- Süli & Mayers (2003), *An Introduction to Numerical Analysis*, Cambridge University Press. Newton's method, finite differences and quadrature, with proofs. [doi:10.1017/CBO9780511801181](https://doi.org/10.1017/CBO9780511801181)
- 3Blue1Brown (Grant Sanderson), *Essence of Calculus* and *Essence of Linear Algebra* (video series). the best visual intuition for topics 02–04. [3blue1brown.com](https://www.3blue1brown.com)
