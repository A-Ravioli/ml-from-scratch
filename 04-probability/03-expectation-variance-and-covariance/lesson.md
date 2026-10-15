# expectation, variance & covariance

## why this matters

a distribution is a whole function, but you almost always summarize it by two or three numbers: where it's centered, how spread out it is, and how its coordinates move together. training loss is an average. batch normalization keeps running means and variances of every activation. Adam keeps running averages of gradients and their squares. standardizing features divides by a standard deviation. PCA (ch 11) is an eigendecomposition of a covariance matrix. a Monte Carlo estimate is a mean, and its error bar is a standard deviation divided by $\sqrt n$.

these summaries are easy to define and surprisingly easy to compute wrongly. every statistics textbook gives the shortcut $\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$. it's a correct identity. it's also a numerical trap: on data like timestamps, sensor readings or prices, which sit far from zero, it returns garbage, including *negative* variances, and spreadsheet programs shipped with exactly this bug for years. in this topic you'll build the summaries properly: exact expectations, Monte Carlo ones with honest error bars, a variance that survives floating point, a streaming version that sees each data point once, and covariance matrices.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. for a fair die, is $\mathbb{E}[X^2]$ equal to $\mathbb{E}[X]^2 = 12.25$? bigger? smaller?
2. you estimate $\mathbb{E}[X]$ by averaging $n$ samples. to make the error 10 times smaller, how many more samples do you need?
3. the four numbers $10^9 + 4$, $10^9 + 7$, $10^9 + 13$, $10^9 + 16$ have variance $22.5$. what does `np.mean(x**2) - np.mean(x)**2` return for them?
4. two machines each computed the mean and variance of half of a dataset. what do you need from each to get the variance of the whole dataset, without seeing the data again?
5. $X$ is a random vector with covariance matrix $\Sigma$. what's the covariance matrix of $AX + b$?

## prerequisites check

- [ ] you can compute probabilities from a pmf, and know what independence means (→ ch 04 · topic 01)
- [ ] you can draw samples from common distributions (→ ch 04 · topic 02)
- [ ] you know floating-point numbers carry about 16 significant digits, and what catastrophic cancellation is (→ ch 00 · topic 03)
- [ ] you can multiply matrices and know what a symmetric positive semidefinite matrix is (→ ch 02 · topics 02 and 05)

## core ideas

### idea 1: expectation, exactly and by Monte Carlo

the **expectation** (or mean) of a discrete random variable is its probability-weighted average,

$$\mathbb{E}[X] = \sum_x x\,P(X = x),$$

and for a continuous one, $\int x\,f(x)\,dx$. it's the balance point of the distribution: put weight $P(X = x)$ at each $x$ on a ruler, and the ruler balances at $\mathbb{E}[X]$. a fair die balances at $3.5$, a value it never takes.

you often want the expectation of a *function* of $X$: the expected loss, the expected squared error, $\mathbb{E}[X^2]$. you could find the distribution of $Y = g(X)$ first (a pushforward, topic 01) and then average. but you don't have to:

$$\mathbb{E}[g(X)] = \sum_x g(x)\,P(X = x).$$

apply $g$ to the values and keep the original weights. this is the **law of the unconscious statistician** (LOTUS), named because people use it without noticing it needs proof. it does (on paper 1 sketches why): it's exactly the pushforward, with the bookkeeping done implicitly.

the most important property of expectation is **linearity**: $\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c$, always, whether or not $X$ and $Y$ are independent. but expectation does *not* commute with nonlinear functions: $\mathbb{E}[g(X)] \ne g(\mathbb{E}[X])$ in general. for a convex $g$ like $x^2$ or $e^x$, **Jensen's inequality** says $\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])$. that's warm-up 1: $\mathbb{E}[X^2] = 91/6 \approx 15.17 > 12.25$. the gap is exactly the variance.

when the sum or integral is intractable (most of the time in ML), you **estimate** it. draw $X_1, \dots, X_n$ independently and average:

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n g(X_i), \qquad \mathbb{E}[\hat\mu_n] = \mathbb{E}[g(X)], \qquad \operatorname{sd}(\hat\mu_n) = \frac{\sigma}{\sqrt n},$$

where $\sigma$ is the standard deviation of $g(X)$. this is **Monte Carlo**. the estimate is unbiased, and its typical error, the **standard error**, shrinks like $1/\sqrt n$. that's warm-up 2: ten times more accuracy costs a hundred times more samples. you don't know $\sigma$, so you plug in the sample standard deviation $s$: the standard error is $s/\sqrt n$. topic 05 shows why $\hat\mu_n \pm 1.96\, s/\sqrt n$ is a 95% interval.

> **▶ build: stage 1.** implement `expectation(values, probs, f=None)` (LOTUS) and `mc_estimate(samples)`, which returns the mean and its standard error $s/\sqrt n$ (with $s$ using `ddof=1`), in `exercise.py` (`marimo edit exercise.py`).

### idea 2: variance, and the formula that fails

the **variance** measures spread: the expected squared distance from the mean,

$$\operatorname{Var}(X) = \mathbb{E}\big[(X - \mu)^2\big], \qquad \mu = \mathbb{E}[X],$$

and the **standard deviation** $\sigma = \sqrt{\operatorname{Var}(X)}$ is in the same units as $X$. two facts you'll use constantly: $\operatorname{Var}(aX + b) = a^2\operatorname{Var}(X)$ (shifting changes nothing, scaling scales), and for *independent* $X, Y$, $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$. that second fact is where the $\sigma/\sqrt n$ above comes from: the sum of $n$ independent copies has variance $n\sigma^2$, and dividing by $n$ divides the variance by $n^2$.

expanding the square gives the shortcut formula

$$\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2.$$

on paper, it's the same thing. in floating point, it's a disaster whenever the mean is large compared with the spread. take warm-up 3. the values are near $10^9$, so their squares are near $10^{18}$. a double holds about 16 significant digits, so numbers near $10^{18}$ are stored with a spacing of 128. $\mathbb{E}[X^2]$ and $\mathbb{E}[X]^2$ are two huge numbers that agree in their first 16 digits. their difference, the variance $22.5$, lives entirely in the digits that were rounded away. NumPy returns $-128$: a negative variance.

the relative error of the shortcut formula is roughly $\varepsilon\,\mu^2/\sigma^2$ (with $\varepsilon \approx 10^{-16}$). for $\mu/\sigma = 10^8$ that's 100%. this is **catastrophic cancellation**: subtracting nearly equal numbers doesn't create error, it *exposes* the rounding error already in them.

the fix is to subtract the mean *before* squaring. the **two-pass algorithm** computes $\bar x$ in a first pass, then $\frac{1}{n}\sum(x_i - \bar x)^2$ in a second. the deviations $x_i - \bar x$ are small numbers (4.5 or so, not $10^9$), squaring them loses nothing, and the result is accurate to nearly full precision.

one more choice: divide by $n$ or by $n - 1$? dividing by $n$ gives the variance of the data themselves (the **population** variance, `ddof=0`). if the data are a sample and you want to estimate the variance of the distribution they came from, divide by $n - 1$ (`ddof=1`, **Bessel's correction**): the deviations are measured from $\bar x$, which is fitted to the same data and so is closer to them than the true $\mu$ is. on paper 2 shows that $n - 1$ makes the estimate unbiased.

> **🔮 predict:** at what offset $\mu$ does the shortcut formula first return a *negative* variance for unit-variance data in float64? and in float32? (the lab measures it.)

> **▶ build: stage 2.** implement `naive_variance(x)` (the shortcut, exactly as written, so you can watch it fail) and `two_pass_variance(x, ddof=0)`.

### idea 3: streaming statistics, Welford and Chan

two passes are fine for an array in memory. but often the data arrive one at a time, or in batches, and you can't (or don't want to) keep them: activations during training, a sensor stream, a dataset too big for RAM. you want to keep a small summary and update it as each batch arrives.

the naive streaming approach keeps $\sum x$ and $\sum x^2$, and that's the shortcut formula again, with the same catastrophic cancellation. the stable approach keeps three numbers: the count $n$, the running mean $\bar x$, and the running **sum of squared deviations** from the mean,

$$M_2 = \sum_i (x_i - \bar x)^2, \qquad \operatorname{Var} = M_2/n.$$

when a new value $x$ arrives, Welford (1962) showed how to update all three:

$$n \leftarrow n + 1,\qquad \delta = x - \bar x_{\text{old}},\qquad \bar x \leftarrow \bar x_{\text{old}} + \frac{\delta}{n},\qquad M_2 \leftarrow M_2 + \delta\,(x - \bar x_{\text{new}}).$$

all the arithmetic is on deviations, never on raw squares, so it's stable.

a batch is just as easy, and the formula for merging two summaries answers warm-up 4. say summary $A$ has $(n_A, \bar x_A, M_{2,A})$ and batch $B$ has $(n_B, \bar x_B, M_{2,B})$, computed from the batch with two passes. with $\delta = \bar x_B - \bar x_A$ and $n = n_A + n_B$, Chan, Golub & LeVeque (1983) give

$$\bar x = \bar x_A + \delta\,\frac{n_B}{n}, \qquad M_2 = M_{2,A} + M_{2,B} + \delta^2\,\frac{n_A n_B}{n}.$$

the last term accounts for the two groups having different means: the spread of the union is the spread within each group plus the spread *between* them. (on paper 3 derives it.) Welford's update is the special case $n_B = 1$, $\bar x_B = x$, $M_{2,B} = 0$. and because merging is associative, you can split a dataset across machines, summarize each piece, and merge the summaries in any order. that's how parallel and distributed statistics are computed.

your `RunningStats` works on samples of any shape: `RunningStats(shape=(3,))` tracks the mean and variance of each of three coordinates, which is exactly what batch normalization's running statistics are.

> **🔮 predict:** you feed the same 300 numbers to two `RunningStats`, one as batches of 50 then 250, the other as 100 then 200. will the final variances agree to the last bit?

> **▶ build: stage 3.** implement `RunningStats.__init__(shape=())` and `RunningStats.update(x)`, which accepts one sample of shape `shape` or a batch `(k, *shape)` and merges it with Chan's formula. `var` and `std` are given.

### idea 4: covariance and correlation

with two random variables, the question is how they move together. the **covariance** is

$$\operatorname{Cov}(X, Y) = \mathbb{E}\big[(X - \mu_X)(Y - \mu_Y)\big].$$

it's positive when $X$ tends to be above its mean at the same time as $Y$, negative when they move oppositely, and zero when neither tendency wins. $\operatorname{Cov}(X, X) = \operatorname{Var}(X)$. if $X$ and $Y$ are independent, their covariance is zero. the converse fails: topic 01's $Y = X^2$ with $X$ uniform on $\{-1, 0, 1\}$ is uncorrelated but completely dependent. covariance only sees linear association.

covariance has units ($X$'s units times $Y$'s), so its size is hard to interpret. dividing by the standard deviations gives the dimensionless **correlation**

$$\rho_{XY} = \frac{\operatorname{Cov}(X, Y)}{\sigma_X\sigma_Y} \in [-1, 1],$$

with $\pm 1$ exactly when $Y$ is a linear function of $X$.

for a random vector $X \in \mathbb{R}^d$, collect all the pairwise covariances into the **covariance matrix** $\Sigma_{ij} = \operatorname{Cov}(X_i, X_j)$, or in matrix form

$$\Sigma = \mathbb{E}\big[(X - \mu)(X - \mu)^\top\big].$$

it's symmetric, and it's positive semidefinite: for any vector $v$, $v^\top\Sigma v = \operatorname{Var}(v^\top X) \ge 0$. that one line is also how $\Sigma$ transforms. for $Y = AX + b$, $Y - \mathbb{E}[Y] = A(X - \mu)$, so

$$\operatorname{Cov}(AX + b) = A\,\Sigma\,A^\top,$$

which answers warm-up 5. it's the rule behind sampling correlated Gaussians (topic 04), propagating uncertainty through a linear model, and whitening data.

from data, a matrix $X$ with $n$ rows (samples) and $d$ columns (variables), the **sample covariance** is

$$\hat\Sigma = \frac{1}{n - 1}X_c^\top X_c, \qquad X_c = X - \mathbf{1}\bar x^\top,$$

where $X_c$ is the data with each column's mean subtracted. that's the two-pass algorithm in matrix form, and the same cancellation trap awaits anyone who computes $\frac{1}{n}X^\top X - \bar x\bar x^\top$ instead.

> **▶ build: stage 4.** implement `covariance(X, *, ddof=1)` (two passes, like `np.cov(X, rowvar=False)`) and `correlation(X)`.

## worked example

**a fair die.** with LOTUS, $\mathbb{E}[X] = \frac{1 + \cdots + 6}{6} = 3.5$ and $\mathbb{E}[X^2] = \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6}$, so $\operatorname{Var}(X) = \frac{91}{6} - \frac{49}{4} = \frac{35}{12} \approx 2.917$. (with small exact numbers the shortcut is harmless.)

**the offset data.** $x = 10^9 + (4, 7, 13, 16)$. two passes: $\bar x = 10^9 + 10$, deviations $(-6, -3, 3, 6)$, squares $(36, 9, 9, 36)$ summing to $90$. so the population variance is $90/4 = 22.5$, and the sample variance is $90/3 = 30$. the shortcut formula in float64 returns $-128$.

**merging two halves with Chan's formula.** split the deviations into $A = (4, 7)$ and $B = (13, 16)$ (drop the $10^9$; shifting doesn't change variances, and the merge works the same either way). $A$: $n_A = 2$, $\bar x_A = 5.5$, $M_{2,A} = 1.5^2 + 1.5^2 = 4.5$. $B$: $n_B = 2$, $\bar x_B = 14.5$, $M_{2,B} = 4.5$. then $\delta = 9$, $n = 4$:

$$\bar x = 5.5 + 9 \cdot \tfrac24 = 10, \qquad M_2 = 4.5 + 4.5 + 81 \cdot \tfrac{2 \cdot 2}{4} = 90,$$

the same $M_2 = 90$ as the two-pass computation. the "between groups" term, 81, is almost all of it.

**a covariance matrix.** four points in 2-D, $(1, 2), (2, 1), (3, 5), (6, 4)$. column means $(3, 3)$. centered: $(-2, -1), (-1, -2), (0, 2), (3, 1)$. sums of products: $\sum x^2 = 4 + 1 + 0 + 9 = 14$, $\sum y^2 = 1 + 4 + 4 + 1 = 10$, $\sum xy = 2 + 2 + 0 + 3 = 7$. dividing by $n - 1 = 3$:

$$\hat\Sigma = \begin{pmatrix}14/3 & 7/3\\ 7/3 & 10/3\end{pmatrix}, \qquad \hat\rho = \frac{7}{\sqrt{14 \cdot 10}} \approx 0.59.$$

the tests check the die moments, $22.5$ and $30$, and this covariance matrix.

## on paper

1. **LOTUS from the pushforward.** let $Y = g(X)$ with $X$ discrete. starting from $\mathbb{E}[Y] = \sum_y y\,P(Y = y)$ and $P(Y = y) = \sum_{x: g(x) = y}P(X = x)$, show that $\mathbb{E}[Y] = \sum_x g(x)P(X = x)$.

<details><summary>solution</summary>

$\sum_y y\sum_{x:\,g(x) = y}P(X = x) = \sum_y\sum_{x:\,g(x) = y}g(x)P(X = x)$, since $y = g(x)$ inside the inner sum. every $x$ appears in exactly one inner sum (the one for $y = g(x)$), so the double sum is a single sum over all $x$: $\sum_x g(x)P(X = x)$.

</details>

2. **Bessel's correction.** let $X_1, \dots, X_n$ be independent with mean $\mu$ and variance $\sigma^2$. show that $\mathbb{E}\big[\sum_i(X_i - \bar X)^2\big] = (n - 1)\sigma^2$. (hint: write $X_i - \bar X = (X_i - \mu) - (\bar X - \mu)$ and use $\operatorname{Var}(\bar X) = \sigma^2/n$.)

<details><summary>solution</summary>

$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - 2(\bar X - \mu)\sum_i(X_i - \mu) + n(\bar X - \mu)^2 = \sum_i(X_i - \mu)^2 - n(\bar X - \mu)^2$, using $\sum_i(X_i - \mu) = n(\bar X - \mu)$. taking expectations: $n\sigma^2 - n\cdot\sigma^2/n = (n - 1)\sigma^2$. so dividing by $n - 1$ gives an unbiased estimate of $\sigma^2$, and dividing by $n$ underestimates it by a factor $(n-1)/n$.

</details>

3. **Chan's merge.** let group $A$ have $n_A$ points with mean $\bar x_A$ and $M_{2,A} = \sum_{i \in A}(x_i - \bar x_A)^2$, and similarly for $B$. show that the combined $M_2 = \sum_{i \in A \cup B}(x_i - \bar x)^2$ equals $M_{2,A} + M_{2,B} + \delta^2 n_A n_B/n$ with $\delta = \bar x_B - \bar x_A$.

<details><summary>solution</summary>

for any group $G$ and any number $c$: $\sum_{i \in G}(x_i - c)^2 = M_{2,G} + n_G(\bar x_G - c)^2$ (expand around $\bar x_G$; the cross term vanishes). apply it to $A$ and $B$ with $c = \bar x$: $M_2 = M_{2,A} + n_A(\bar x_A - \bar x)^2 + M_{2,B} + n_B(\bar x_B - \bar x)^2$. since $\bar x = (n_A\bar x_A + n_B\bar x_B)/n$, $\bar x_A - \bar x = -\delta n_B/n$ and $\bar x_B - \bar x = \delta n_A/n$. so the extra terms are $\delta^2(n_A n_B^2 + n_B n_A^2)/n^2 = \delta^2 n_A n_B/n$.

</details>

4. **covariance under linear maps.** show that $\operatorname{Cov}(AX + b) = A\Sigma A^\top$, and conclude that every covariance matrix is positive semidefinite. when is it positive *definite*?

<details><summary>solution</summary>

$\mathbb{E}[AX + b] = A\mu + b$, so $\operatorname{Cov}(AX + b) = \mathbb{E}[A(X - \mu)(X - \mu)^\top A^\top] = A\Sigma A^\top$ (expectation is linear, and $A$ is constant). with $A = v^\top$ (a row), $v^\top\Sigma v = \operatorname{Var}(v^\top X) \ge 0$. it's positive definite unless some $v \ne 0$ has $\operatorname{Var}(v^\top X) = 0$, meaning some linear combination of the coordinates is constant: the distribution lives on a lower-dimensional (affine) subspace. a sample covariance from $n \le d$ points is always singular for this reason (its rank is at most $n - 1$).

</details>

## experiment

`lab.py` runs three experiments on your code:

1. the shortcut formula, two passes and your `RunningStats`, on data with a growing offset, in float64 and float32.
2. Monte Carlo error bars: how the error of `mc_estimate` shrinks with $n$, and whether its intervals deserve to be trusted.
3. the eigenvalues of your sample covariance matrix, when the true covariance is the identity.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. why is $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ true even for dependent $X, Y$, while $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$ needs more?

<details><summary>answer</summary>

expectation is a sum (or integral) weighted by the joint distribution, and sums split. variance is an expectation of a *square*, and $(a + b)^2$ has a cross term: $\operatorname{Var}(X + Y) = \operatorname{Var}X + \operatorname{Var}Y + 2\operatorname{Cov}(X, Y)$. the extra condition is zero covariance (independence is more than enough).

</details>

2. the two-pass algorithm also subtracts nearly equal numbers ($x_i - \bar x$ with both near $10^9$). why isn't that catastrophic?

<details><summary>answer</summary>

the inputs $x_i$ are exact (they're the data), and $\bar x$ is accurate to about $10^{-16}$ relative, an absolute error near $10^{-7}$. so each deviation is accurate to about $10^{-7}$ absolute, tiny compared with a deviation of size 5. the shortcut formula subtracts two *derived* quantities of size $10^{18}$, each carrying rounding errors of size $\sim 100$: errors far bigger than the answer.

</details>

3. a Monte Carlo estimate with 10,000 samples has standard error 0.02. how many samples do you need for 0.005?

<details><summary>answer</summary>

the standard error scales like $1/\sqrt n$, so a 4× smaller error needs $16 \times$ the samples: 160,000.

</details>

4. why does Chan's merge need $\bar x_A$ and $\bar x_B$, not just the two variances and counts?

<details><summary>answer</summary>

the combined spread includes the spread *between* the groups, $\delta^2 n_An_B/n$, which depends on how far apart their means are. two groups with variance 1 each could have a combined variance of 1 (same mean) or of 100 (means 20 apart).

</details>

5. a feature has correlation 0.0 with the label. can it still be useful for predicting the label?

<details><summary>answer</summary>

yes. correlation measures *linear* association only. the label could be $x^2$, or depend on $x$ through an interaction with another feature. trees and neural networks (ch 10, 14) pick up such dependencies; a linear model on $x$ alone can't.

</details>

6. you computed a $50 \times 50$ sample covariance from 30 data points and `cholesky` fails. bug?

<details><summary>answer</summary>

no. with $n = 30$ points the centered data matrix has rank at most 29, so $\hat\Sigma = X_c^\top X_c/(n - 1)$ has rank at most 29 < 50: it's singular, positive *semi*definite, and Cholesky needs positive definite. you need more data, or a regularized estimate like $\hat\Sigma + \lambda I$ (ch 08 and 11).

</details>

7. batch normalization keeps a running mean and variance for each channel during training. which of your stages is that?

<details><summary>answer</summary>

stage 3: `RunningStats(shape=(n_channels,))` with a batch update per step. (real batch norm uses an exponential moving average instead of an exact running mean, so that old batches from early in training are forgotten. the stability issue is the same.)

</details>

## common bugs

- **the shortcut formula in disguise.** keeping running `sum_x` and `sum_x2` *is* the shortcut formula. symptom: accurate variances on centered test data, garbage (or negative values) on real data with an offset.
- **`f(E[X])` instead of `E[f(X)]`.** applying the function to the mean instead of to each value. symptom: right answers for linear `f` only.
- **standard deviation vs standard error.** the spread of the data is $s$; the uncertainty of their mean is $s/\sqrt n$. symptom: error bars that don't shrink when you collect more data.
- **ddof confusion.** `np.var` defaults to `ddof=0`, `np.cov` to `ddof=1`. symptom: answers off by a factor $(n-1)/n$, invisible for large $n$ and glaring for small $n$.
- **centering rows instead of columns.** `X - X.mean(axis=1, keepdims=True)` subtracts each *sample's* mean. symptom: a covariance matrix that disagrees with `np.cov(X, rowvar=False)`.
- **using the updated count in the merge.** the cross term is $\delta^2 n_A n_B/n$ with $n_A$ the count *before* the batch. symptom: variances that are slightly too big, by an amount that depends on how you batch.

## going deeper

- Welford (1962), "Note on a method for calculating corrected sums of squares and products", *Technometrics* 4(3): 419–420. [doi:10.1080/00401706.1962.10490022](https://doi.org/10.1080/00401706.1962.10490022)
- Chan, Golub & LeVeque (1983), "Algorithms for computing the sample variance: analysis and recommendations", *The American Statistician* 37(3): 242–247. compares the shortcut, two-pass and updating algorithms, with error bounds. [doi:10.1080/00031305.1983.10483115](https://doi.org/10.1080/00031305.1983.10483115)
- Blitzstein & Hwang (2019), *Introduction to Probability*, 2nd ed., CRC Press, chapters 4 and 7: expectation, LOTUS, variance, covariance and correlation. free at [probabilitybook.net](https://probabilitybook.net).
- Wasserman (2004), *All of Statistics*, Springer, chapter 3: expectation, variance, covariance and the sample versions, in 20 pages.

## connections

- ← builds on: pmfs and independence (topic 01), samplers to generate Monte Carlo draws (topic 02), floating point (ch 00 · topic 03), symmetric positive semidefinite matrices (ch 02).
- → used in: the multivariate Gaussian's mean and covariance (topic 04), the CLT's $\sigma/\sqrt n$ (topic 05), concentration bounds that use the variance (topic 06), estimators and their bias and variance (ch 05), standardization (ch 07), PCA (ch 11), batch normalization and Adam's moment estimates (ch 15).
