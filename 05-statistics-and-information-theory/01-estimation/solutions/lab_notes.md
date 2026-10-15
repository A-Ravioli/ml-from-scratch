# estimation · lab notes

## experiment 1: divide by $n - 1$, $n$, or $n + 1$?

**what people usually predict:** $n - 1$ wins, because it's unbiased and it's what every textbook and `np.var(ddof=1)` recommend. maybe the three are about equal.

**what happens (n = 10, true variance 1, 40,000 repetitions per divisor):**

| distribution | MSE at c = 9 (unbiased) | c = 10 (MLE) | c = 11 | best c on the grid |
|---|---|---|---|---|
| uniform (kurtosis 1.8) | 0.102 | 0.093 | 0.101 | 10 |
| normal (kurtosis 3) | 0.220 | 0.190 | 0.182 | 11 |
| laplace (kurtosis 6) | 0.523 | 0.434 | 0.377 | 14 |

the unbiased divisor is the worst of the three for normal and Laplace data. for normal data the simulated curve lies on your exact formula, with its minimum at $c = 11$, 18% below the unbiased estimator. for Laplace data the best divisor is 14, and the unbiased estimator's MSE (0.52) is about 50% above the best (0.34).

**why:** dividing by a larger $c$ multiplies every estimate by $(n-1)/c < 1$: it shrinks toward 0. that adds a bias proportional to the shrinkage, and cuts the standard deviation by the same factor. near $c = n - 1$ the bias costs almost nothing (it enters squared) while the variance saving is first-order, so a little shrinkage always helps. how much depends on how noisy $SS = \sum(x_i - \bar x)^2$ is. in general $\operatorname{Var}(SS) = (n-1)\sigma^4\big((n-1)\kappa - (n-3)\big)/n$, where $\kappa$ is the kurtosis, and the MSE-optimal divisor is

$$c^\star = (n - 1) + \frac{\operatorname{Var}(SS)}{(n - 1)\sigma^4},$$

which is $n + 1$ for the normal ($\kappa = 3$), 9.92 for the uniform and 13.7 for the Laplace at $n = 10$, matching the simulation. heavy tails make $SS$ noisier, so they reward more shrinkage. lesson: "unbiased" is a property, not a virtue, and how much bias is worth paying depends on the data.

## experiment 2: Stein's paradox

**what people usually predict:** James–Stein helps when $\theta$ happens to be near 0 (where you're shrinking toward) and hurts elsewhere. for unrelated quantities it seems absurd that it could help at all.

**what happens (d = 10):** the MLE's total MSE is 10 everywhere, as it must be. James–Stein's is 2.0 at $\theta = 0$, 4.6 at $\lVert\theta\rVert \approx 2$, 8.5 at 6.2, and still 9.6 at 12.5: below 10 at *every* $\lVert\theta\rVert$, approaching it from below. with $d = 50$ the gain is enormous (2.0 instead of 50 at the origin, 38 at $\lVert\theta\rVert = 12$). with $d = 2$ the two curves coincide (the shrinkage factor $d - 2$ is zero). with $d = 1$ "James–Stein" *expands* the estimate, $X + 1/X$, and its MSE is infinite for **every** $\theta$, because $\mathbb{E}[1/X^2] = \infty$ whenever $X$ has positive density at 0. the simulation shows tens of thousands near the origin and modest numbers far from it only because, far out, the draws that land near 0 are too rare to appear in 4,000 repetitions: a simulated average can't see an infinite expectation dominated by rare events (topic 02's lab makes the same point).

**why:** Stein's lemma gives the risk exactly: $\mathbb{E}\lVert\hat\theta_{JS} - \theta\rVert^2 = d - (d - 2)^2\,\mathbb{E}\big[1/\lVert X\rVert^2\big]$, and the subtracted term is positive whenever $d \ge 3$. so the MLE is **inadmissible** in three or more dimensions: James–Stein beats it for every $\theta$. the gain is largest near the shrinkage target and fades as $\lVert X\rVert^2$ grows, because then $(d-2)/\lVert X\rVert^2$ is small and the estimator barely shrinks. the paradox dissolves once you notice that the *total* MSE is what improves. a single coordinate can get worse (a coordinate with a large $|\theta_j|$ is pulled toward 0 along with the rest), but the shared shrinkage factor is estimated from all $d$ coordinates at once: $\lVert X\rVert^2 \approx \lVert\theta\rVert^2 + d$ tells you how much of the spread is noise. the same idea, "borrowing strength" across many related estimates, drives empirical Bayes, hierarchical models (ch 12) and ridge regression (ch 08). the target needn't be 0: shrinking toward the grand mean (with $d - 3$ in place of $d - 2$) is the version Efron and Morris used for baseball.

## experiment 3: consistency, and how fast

**what people usually predict:** all five are consistent except perhaps the median; everything converges at the usual $1/n$ rate.

**what happens (2,000 repetitions per $n$, $n = 4$ to 1024):**

| estimator | MSE at n = 4 | MSE at n = 1024 | fitted rate |
|---|---|---|---|
| 2 · mean | 8.2e-02 | 3.4e-04 | −0.99 |
| max (MLE) | 6.4e-02 | 1.8e-06 | −1.91 |
| (n+1)/n · max | 4.0e-02 | 1.1e-06 | −1.92 |
| 2 · median | 1.4e-01 | 1.0e-03 | −0.90 |
| 2 · first draw | 3.4e-01 | 3.3e-01 | 0.00 |

"twice the first draw" is unbiased and never improves: inconsistent. the mean and median converge at rate 1 (the median's fitted slope is a bit shallower at small $n$ because its MSE is $\approx \frac{1}{n+2}$, not $\propto \frac1n$). the max-based estimators converge at rate 2: at $n = 1024$ their MSE is about 300 times smaller than the mean's. (the fitted slope is a little above −2 because $n(n+2)$ isn't exactly $n^2$ at small $n$.)

**why:** an unbiased estimator that ignores most of the data has a variance that can't shrink; consistency needs the variance (and bias) to go to zero. the mean averages $n$ independent errors, so its variance is $\sigma^2/n$: rate 1, the rate of every "smooth" estimator in a regular model. the max is different in kind. the largest of $n$ uniforms sits about $\theta/(n+1)$ below $\theta$, so its *error* (not squared error) is of order $1/n$, and its MSE of order $1/n^2$. this is possible because the uniform family is not regular: the support $[0, \theta]$ moves with $\theta$, so one draw near the edge rules out every smaller $\theta$ outright. the Cramér–Rao bound in topic 06, which says no unbiased estimator beats rate 1, assumes the support doesn't depend on the parameter, and the uniform violates exactly that. the median loses to the mean here because the uniform has light tails: the median ignores how far points are from the center, which is a waste when there are no outliers. for heavy-tailed data (Laplace, Cauchy) the ranking reverses.
