# 00 · python & numpy for ML

> write fast, correct, vectorized numerical code and test it.

machine learning is numerical programming. a model is a few arrays of numbers, training is a loop of arithmetic on those arrays, and almost every bug you'll ever chase is a numerical one: a shape that broadcast the wrong way, a number too big for a float, a loop that takes an hour instead of a second, a test that compared floats with `==`. this chapter teaches the craft that everything else stands on. you'll learn how Python objects and operators work (by building a `Vector` class whose `u @ v` you wrote yourself), why NumPy is a hundred to a thousand times faster than a Python loop and how to think in whole arrays, how floating point numbers really work and how `softmax` breaks when you ignore them, how to make plots that tell the truth, and how to write tests that catch the bugs your eyes miss. by the end you'll have written the first module of the course's library and a working image-filtering toolkit, and you'll never write `np.exp(x) / np.exp(x).sum()` again.

**you will build:** a `Vector` class with operator overloading and a `LinearModel` dataclass that trains itself; batched dot products, one-hot encoding, moving averages, standardization and a loop-free pairwise distance matrix; a float decomposer, machine epsilon, a cancellation-free quadratic solver, and stable `sigmoid`, `softplus`, `log_sigmoid`, `logsumexp`, `softmax` and `log_softmax`; a histogram from scratch and the course's three plotting helpers; seeded random generators, an `assert_close` with informative diffs, a random tester and a `git bisect`-style binary search. module: **`mlfs.utils`** (topics 02–05).
**prerequisites:** basic Python (functions, loops, lists, `if` statements) and high school math (powers, logarithms, the quadratic formula, the Pythagorean theorem). no NumPy, linear algebra or calculus is assumed. if Python itself is new, work through chapters 1–5 and 8 (exceptions) of the [official tutorial](https://docs.python.org/3/tutorial/) first; the "python toolkit" in topic 01's lesson covers the rest (f-strings, `isinstance`, passing functions around, decorators, `with` blocks).
**time · scaffolding:** about 50 hours · L1 (guided): every function comes with its signature, a detailed docstring and step-by-step comments inside its body. hints have three levels. the capstone gives you most of the code and asks you to write the core.
**newly unlocked library components:** after this chapter you may use NumPy's vectorized operations, reductions and indexing freely, along with `np.histogram`/`ax.hist`, `scipy.special.logsumexp`/`softmax`/`expit`/`log_expit`, `np.testing.assert_allclose`, `torchvision.utils.make_grid`-style tiling, and `scipy.signal`/`scipy.ndimage` convolutions in labs and experiments, since you've built each of them. inside `mlfs` modules, later chapters use *your* versions through `mlfs.utils`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [python for numerical work](01-python-for-numerical-work/) | `dot`, `norm`, a `Vector` class with operator overloading, a `LinearModel` dataclass that trains by gradient descent | your `dot` beats NumPy on 3-element vectors, then loses by 200–600× at 10,000; and looping over a NumPy array is *slower* than looping over a list |
| 02 | [numpy arrays & vectorization](02-numpy-arrays-and-vectorization/) | `batch_dot`, `row_norms`, `one_hot`, `moving_average`, `standardize`, `pairwise_differences`, `pairwise_distances` → `mlfs.utils` | ~900× faster than a double loop; the "fully vectorized" broadcast version is no faster than a row loop; and the fast formula returns garbage for points far from the origin |
| 03 | [floating point & stability](03-floating-point-and-stability/) | `decompose`, `machine_epsilon`, `quadratic_roots`, `sigmoid`, `softplus`, `log_sigmoid`, `logsumexp`, `softmax`, `log_softmax` → `mlfs.utils` | naive softmax breaks at logits of about 89 in `float32`; the textbook variance formula returns −128 for data with variance 1 |
| 04 | [visualizing data](04-visualizing-data/) | `plot_curves`, `histogram`, `make_grid`, `image_grid`, `make_meshgrid`, `plot_decision_boundary` → `mlfs.utils` | the "worse" model on a linear axis is the better one on a log axis; four datasets with identical statistics look nothing alike; 100% training accuracy draws a boundary full of islands |
| 05 | [testing, debugging & reproducibility](05-testing-debugging-and-reproducibility/) | `as_rng`, `set_seed`, `assert_close` → `mlfs.utils`, `find_counterexample`, `first_bad` | two invisible bugs caught instantly by `assert_close`; a bug that survives 1,000 random tests; a "6% improvement" that wins on exactly half the seeds |

## capstone

*image filters from scratch*: a vectorized 2-D convolution with three padding modes, then box and Gaussian blur, unsharp-mask sharpening, the Sobel operator and a blur-differentiate-threshold edge detector, applied to the Olivetti faces and CIFAR-10 photos and scored on synthetic images with known edges. you'll measure how much blur best removes noise (on held-out images, with error bars) and why edge detection needs blurring first. small CPU config: synthetic stand-ins, seconds. full config: the real images, seconds once downloaded. the convolution comes back in chapter 16, where the kernels are learned.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (broadcasting shapes and a distance matrix, a stable log-sum-exp and softmax, a histogram and a bisection), and a 60-minute blind rebuild of the stable numerics of topic 03.

## notation

| symbol | meaning |
|---|---|
| $x, u, v$ (lowercase) | vectors (1-D arrays); $x_i$ is the $i$-th entry |
| $X, A$ (uppercase) | matrices (2-D arrays); a dataset $X$ has shape $(n, d)$: one row per example, one column per feature |
| $n$, $d$, $m$ | number of examples, number of features, number of points in a second set |
| `(n,)`, `(n, d)` | array shapes in code; `(n,)` is a 1-D array, `(n, 1)` a column |
| $u \cdot v$, `u @ v` | dot product $\sum_i u_iv_i$ |
| $\lVert x\rVert$ | Euclidean norm $\sqrt{x \cdot x}$ |
| $D_{ij} = \lVert x_i - y_j\rVert$ | pairwise distance matrix |
| $\bar x$, $\sigma$ | mean and standard deviation (dividing by $n$ unless stated) |
| $\varepsilon$ | machine epsilon, $2^{-52} \approx 2.2 \times 10^{-16}$ for `float64` |
| $\text{fl}(\cdot)$ | the floating point result of an operation, $\text{fl}(a \circ b) = (a \circ b)(1 + \delta)$, $\lvert\delta\rvert \le \varepsilon/2$ |
| $\sigma(x)$ | the sigmoid $1/(1 + e^{-x})$ (context distinguishes it from a standard deviation) |
| $\operatorname{softplus}(x)$ | $\log(1 + e^{x})$ |
| $\operatorname{logsumexp}(x)$ | $\log\sum_i e^{x_i}$ |
| $\operatorname{softmax}(x)_i$ | $e^{x_i}/\sum_j e^{x_j}$ |
| rtol, atol | relative and absolute tolerances: $\lvert a - e\rvert \le \text{atol} + \text{rtol}\,\lvert e\rvert$ |
| $\eta$ | learning rate (step size) |
| $K$, $*$ | a convolution kernel, and convolution $(I * K)$ (capstone) |
| indices | math counts from 1 ($x_1$ is the first entry); NumPy counts from 0 (`x[0]`) |

## how this connects

- builds on: basic Python and high school math.
- used later in: everything. specifically: chapter 01 (math onramp) uses NumPy arrays and `as_rng` for its Monte Carlo estimates; chapter 02 (linear algebra) leans on broadcasting, `pairwise_distances`-style nearest neighbours and floating point error; chapters 04–05 use `logsumexp` and `log1p` for probabilities and entropies; chapter 07 builds on `standardize`, `one_hot`, multi-seed evaluation and `plot_curves`; chapters 08–10 draw with `plot_decision_boundary` and compute losses with `sigmoid`, `log_sigmoid` and `log_softmax`; chapter 14 grows your `Vector` into an autograd `Tensor`; chapter 16 learns the kernels of the capstone's convolution; chapter 18's attention is a `softmax` and chapter 26 returns to `float16`, overflow and loss scaling.

## core references

- Harris et al. (2020), "Array programming with NumPy", *Nature* 585: 357–362. [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- VanderPlas (2016), *Python Data Science Handbook*, O'Reilly, chapters 2 and 4 (NumPy and matplotlib). [free online](https://jakevdp.github.io/PythonDataScienceHandbook/)
- Ramalho (2022), *Fluent Python*, 2nd ed., O'Reilly: Python's data model, classes and operator overloading.
- Goldberg (1991), "What every computer scientist should know about floating-point arithmetic", *ACM Computing Surveys* 23(1): 5–48. [doi:10.1145/103162.103163](https://doi.org/10.1145/103162.103163)
- Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, chapters 1–4. [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
- Wilke (2019), *Fundamentals of Data Visualization*, O'Reilly. [free online](https://clauswilke.com/dataviz/)
