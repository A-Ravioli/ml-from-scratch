# 00 · python & numpy for ML · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. `v` is an instance of your `Vector` class. what does Python do, step by step, to evaluate `3 * v`?

<details><summary>answer</summary>

it calls `int.__mul__(3, v)` first. the integer doesn't know about `Vector`, so it returns `NotImplemented`, and Python then tries the reflected method on the right operand, `v.__rmul__(3)`. if `Vector` doesn't define `__rmul__`, the expression raises `TypeError`, even though `v * 3` works.

</details>

2. why should an operator method like `__add__` return `NotImplemented` for an operand type it doesn't handle, instead of `False` or raising an error?

<details><summary>answer</summary>

so that the other operand gets a chance: Python then calls the other object's reflected method (`__radd__`), and only raises `TypeError` if that fails too. returning `NotImplemented` lets independently written classes cooperate.

</details>

3. give two reasons a NumPy array of a million floats is much faster to sum than a Python list of the same floats.

<details><summary>answer</summary>

(1) memory layout: the array is one contiguous block of raw 8-byte doubles, while the list holds pointers to separate 24-byte float objects scattered in memory. (2) who runs the loop: `np.sum` runs a compiled loop (with vector instructions) over raw numbers, while a Python loop pays interpreter overhead (type checks, object creation) on every element.

</details>

4. `X` has shape `(200, 5)`. what are the shapes of `X.mean(axis=0)`, `X.mean(axis=1)` and `X.mean(axis=0, keepdims=True)`, and which one do you subtract from `X` to center each feature?

<details><summary>answer</summary>

`(5,)`, `(200,)` and `(1, 5)`. to center features, subtract the column means: either `(5,)` or `(1, 5)` broadcasts against `(200, 5)`.

</details>

5. which of these give a view of `a`, and which a copy: `a[2:5]`, `a[[2, 3, 4]]`, `a[a > 0]`, `a[:, 0]`, `a.T`?

<details><summary>answer</summary>

views: `a[2:5]`, `a[:, 0]`, `a.T` (basic slicing and transposes never copy). copies: `a[[2, 3, 4]]` and `a[a > 0]` (fancy and boolean indexing always copy).

</details>

6. what is the shape of `x + y` when `x` has shape `(3,)` and `y` has shape `(3, 1)`? why is this a dangerous source of bugs?

<details><summary>answer</summary>

`(3, 3)`: aligned from the right, `(1, 3)` and `(3, 1)` broadcast to every pairwise sum. it's dangerous because it raises no error: code that mixes up a vector and a column silently computes an `(n, n)` matrix instead of an `(n,)` vector.

</details>

7. `pairwise_distances` uses $\|x\|^2 - 2x\cdot y + \|y\|^2$. why is it faster than building the array of differences, and what two numerical precautions does it need?

<details><summary>answer</summary>

it never creates the $(n, m, d)$ difference array, and almost all its work is one matrix product `X @ Y.T`, which BLAS runs near the CPU's peak. precautions: clip the squared distances at 0 before the square root (rounding can make them slightly negative), and center the data when the points are far from the origin (otherwise catastrophic cancellation destroys the small distances).

</details>

8. roughly how many significant decimal digits do `float64` and `float32` hold, and what is machine epsilon for each?

<details><summary>answer</summary>

about 16 and about 7. $\varepsilon = 2^{-52} \approx 2.2 \times 10^{-16}$ and $2^{-23} \approx 1.2 \times 10^{-7}$.

</details>

9. what is catastrophic cancellation? give an example and its fix.

<details><summary>answer</summary>

subtracting two nearly equal numbers that carry rounding errors: the leading (correct) digits cancel, leaving mostly noise, so the relative error is multiplied by $\frac{|a| + |b|}{|a - b|}$. examples: the small root of $x^2 + 10^8x + 1$ from the school formula (fix: compute the other root with no cancellation and use $x_1x_2 = c/a$); the variance as $\overline{x^2} - \bar x^2$ (fix: subtract the mean first); $\log(1 + t)$ for tiny $t$ (fix: `np.log1p`).

</details>

10. why does subtracting the maximum make softmax stable, and why doesn't it change the result?

<details><summary>answer</summary>

after the shift, every exponent is $\le 0$, so no exponential overflows, and the largest one is exactly $e^0 = 1$, so the denominator is at least 1. it doesn't change the result because $e^{-m}$ multiplies both the numerator and the denominator and cancels.

</details>

11. why compute a classification loss with `log_softmax(x)` rather than `np.log(softmax(x))`?

<details><summary>answer</summary>

$\log\operatorname{softmax}(x)_i = x_i - \operatorname{logsumexp}(x)$ is a subtraction of ordinary numbers and is accurate for any logits. `np.log(softmax(x))` returns $\log 0 = -\infty$ as soon as a probability underflows (a confident wrong prediction), giving an infinite loss and `nan` gradients.

</details>

12. why should training losses usually be plotted on a logarithmic y axis?

<details><summary>answer</summary>

losses shrink by large factors over training. on a linear axis the late phase (say 0.03 versus 0.015, a factor of 2) is compressed into a sliver at the bottom and looks flat; a log axis gives every factor of 10 the same height, so improvements and plateaus are visible at every stage.

</details>

13. what goes wrong with a histogram that has too few bins? too many?

<details><summary>answer</summary>

too few: each bin averages over a wide range and real structure (two peaks, a gap) disappears (bias). too many: each bin expects only a few values, and the counts fluctuate by about the square root of their size, so the picture fills with spurious peaks (variance). try several bin counts and trust only features that persist.

</details>

14. why must `as_rng(g)` return the same `Generator` object `g` rather than a copy?

<details><summary>answer</summary>

so that one random stream flows through the whole program. with copies, two helpers given the same generator would draw identical "random" numbers, and the caller's generator would never advance.

</details>

15. state the tolerance rule of `assert_close`. why does it check shapes exactly before comparing values? how many test runs does a binary search need to find the bad commit among 1,000?

<details><summary>answer</summary>

$|a - e| \le \text{atol} + \text{rtol}\cdot|e|$ elementwise (with `nan` matching `nan` and infinities matching only themselves). shapes are checked first because broadcasting would otherwise compare mismatched arrays (a `(3,)` against a `(3, 1)` becomes a `(3, 3)` comparison), hiding shape bugs. binary search needs $\lceil\log_2 1000\rceil = 10$ tests (plus one to confirm the newest commit is bad).

</details>

## 2. by hand (no computer)

1. **shapes and distances.** (a) give the shape of each result, or say it's an error: `np.ones((4, 1, 3)) + np.ones((5, 3))`; `np.ones((2, 3)) * np.ones((3, 2))`; `v[:, None] - v[None, :]` for `v` of shape `(6,)`. (b) compute the distance matrix between the rows of $X = \begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}$ and $Y = \begin{pmatrix}0 & 0\\ 1 & 1\\ 2 & 0\end{pmatrix}$ using $\|x\|^2 - 2x\cdot y + \|y\|^2$.

<details><summary>solution</summary>

(a) `(4, 5, 3)`; an error (aligned from the right, 3 vs 2 and 2 vs 3, neither is 1); `(6, 6)`.

(b) squared norms: $(1, 1)$ for $X$ (a column), $(0, 2, 4)$ for $Y$ (a row). $XY^\top = \begin{pmatrix}0 & 1 & 2\\ 0 & 1 & 0\end{pmatrix}$. so

$$D^2 = \begin{pmatrix}1 & 1 & 1\\ 1 & 1 & 1\end{pmatrix} - 2\begin{pmatrix}0 & 1 & 2\\ 0 & 1 & 0\end{pmatrix} + \begin{pmatrix}0 & 2 & 4\\ 0 & 2 & 4\end{pmatrix} = \begin{pmatrix}1 & 1 & 1\\ 1 & 1 & 5\end{pmatrix}, \qquad D = \begin{pmatrix}1 & 1 & 1\\ 1 & 1 & \sqrt5\end{pmatrix}.$$

check: $(0, 1)$ to $(2, 0)$ is $\sqrt{4 + 1} = \sqrt 5$. ✓

</details>

2. **stable numerics.** (a) compute $\operatorname{logsumexp}$, softmax and log-softmax of $x = (500,\ 500 + \ln 3)$ the stable way, and say what the naive code returns. (b) compute $\log\sigma(-800)$ with the stable formula. (c) how many `float64` numbers lie in $[1, 2)$?

<details><summary>solution</summary>

(a) $m = 500 + \ln 3$, so $x - m = (-\ln 3, 0)$ and the exponentials are $(1/3, 1)$, summing to $4/3$. $\operatorname{logsumexp}(x) = 500 + \ln 3 + \ln\tfrac43 = 500 + \ln 4 \approx 501.386$. softmax $= (1/4, 3/4)$. log-softmax $= x - \operatorname{logsumexp}(x) = (-\ln 4,\ \ln 3 - \ln 4) \approx (-1.386, -0.288)$. the naive code computes $e^{500} \approx 1.4 \times 10^{217}$ and $e^{501.1}$, which still fit in a `float64`, so it happens to work here. any logit above about 709.8 overflows to `inf`, and then it returns `nan` (in `float32`, the limit is about 88.7).

(b) $\log\sigma(x) = -\operatorname{softplus}(-x) = -(\max(800, 0) + \log(1 + e^{-800})) = -800 - \text{(about } 10^{-348}\text{, which underflows to 0)} = -800$. the naive $\log(1/(1 + e^{800}))$ overflows and returns $-\infty$ (or $\log 0$).

(c) $2^{52}$: every pattern of the 52 fraction bits with exponent 0.

</details>

3. **a histogram and a bisection.** (a) compute `histogram([0.5, 1, 2.5, 3, 3, 4], bins=2)`: the edges and the counts. (b) 16 versions, numbered 0–15; the bug was introduced in version 11. trace `first_bad(16, is_bad)` and count the calls to `is_bad`.

<details><summary>solution</summary>

(a) the range is $[0.5, 4]$, so the edges are $(0.5, 2.25, 4)$. bin $[0.5, 2.25)$ holds $0.5, 1$; bin $[2.25, 4]$ (closed on the right) holds $2.5, 3, 3, 4$. counts $(2, 4)$.

(b) check version 15: bad. $[lo, hi] = [0, 15]$: $mid = 7$ good → $lo = 8$; $mid = 11$ bad → $hi = 11$; $mid = 9$ good → $lo = 10$; $mid = 10$ good → $lo = 11$. now $lo = hi = 11$. five calls, within $\lceil\log_2 16\rceil + 1 = 5$.

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **floating point & stability**, the numerics every later chapter depends on. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=00-python-and-numpy-for-ml/03-floating-point-and-stability
mkdir -p /tmp/rebuild-03
git show HEAD:$T/exercise.py > /tmp/rebuild-03/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-03/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-03/solutions/` and run `python scripts/stubify.py /tmp/rebuild-03`.) open `/tmp/rebuild-03/exercise.py` with `marimo edit`, implement `decompose`, `machine_epsilon`, `quadratic_roots`, `sigmoid`, `softplus`, `log_sigmoid`, `logsumexp`, `softmax` and `log_softmax` from memory, and run the checks against your copy:

```bash
pytest /tmp/rebuild-03/test_exercise.py
```

you pass if every check is green within the hour.

**harder variant** (for testing out of the chapter with confidence): also rebuild `pairwise_distances` (topic 02, stage 4) and `assert_close` (topic 05, stage 2) the same way, each in its own scratch directory, and pass these within 90 minutes in total:

```bash
pytest /tmp/rebuild-02/test_exercise.py -k stage4
pytest /tmp/rebuild-05/test_exercise.py -k stage2
```

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each lettered part separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: questions 1–3 are topic 01, 4–7 topic 02, 8–11 topic 03, 12–13 topic 04, and 14–15 topic 05.
