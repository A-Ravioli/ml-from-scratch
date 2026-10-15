# distributions & sampling · lab notes

## experiment 1: QQ plots against NumPy

**what people usually predict:** that the correct samplers give straight diagonal lines (true), and that the broken normal gives a visibly curved, wiggly or asymmetric plot.

**what happens:** all seven correct samplers sit on the diagonal, with means and standard deviations matching NumPy's to two or three decimals. the Poisson plots are staircases on the diagonal, and for poisson(4) a few points sit one step off it. the broken normal is *also* a perfectly straight line, just at the wrong angle: slope about 0.71, standard deviation 0.7096 instead of 1.

**why:** dropping the 2 turns $R = \sqrt{-2\ln U_1}$ into $R/\sqrt 2$, so the sampler returns $Z/\sqrt2$: an exact normal with variance 1/2. a QQ plot of one normal against another is always straight, with slope equal to the ratio of standard deviations. that's the useful part: a QQ plot doesn't only say "wrong", it says *how*. a shift moves the line off the diagonal, a wrong scale tilts it, wrong tails bend its ends, and skew bends it into an S or a banana. a histogram of the broken sampler would show a perfectly good bell curve, and unless you overlaid the right density at the right scale you'd never notice. that's why the tests compare against the exact CDF rather than eyeballing a shape. the Poisson staircases are just discreteness: many percentiles share the same integer value, in both samples. the occasional point one step off the diagonal is a percentile that lands right at a jump (say, the 25th percentile when $P(X \le 2)$ is almost exactly 0.24): sampling noise in either sample decides which side of the jump it falls on. that's expected for discrete data and doesn't indicate a bug; a real bug shifts or tilts the whole staircase.

## experiment 2: rejection sampling in high dimensions

**what people usually predict:** a 10% wider proposal sounds like a near-perfect fit, so people expect acceptance rates around 80–90% regardless of dimension, maybe declining slowly.

**what happens:** the measured acceptance rate tracks $1.1^{-d}$ exactly: 0.91 in 1-D, 0.38 in 10-D, 0.15 in 20-D, 0.0085 in 50-D. at $d = 100$ it would be $7 \times 10^{-5}$, about 14,000 proposals per accepted sample. the accepted draws are still correct (their mean $\|x\|^2/d$ stays near 1).

**why:** densities in $d$ dimensions are products of $d$ one-dimensional densities, so a per-coordinate mismatch of 10% compounds into a factor $1.1^d$. the tightest envelope must cover the target at its peak (the origin), where $p/q = \sigma^d$. but in high dimensions almost no probability mass is near the origin: the target's samples live on a thin shell at radius $\sqrt d$, and the proposal's live on a shell at radius $1.1\sqrt d$. the envelope is enormous almost everywhere either distribution actually puts its mass. this is the curse of dimensionality for exact samplers, and it's why Bayesian inference for models with more than a handful of parameters uses Markov chain Monte Carlo (ch 12), which builds on the Markov chains of topic 07 and never needs a global envelope. topic 04's lab shows the shell from the other side.

## experiment 3: the cost of exactness

**what people usually predict:** "normal: 2, since Box–Muller uses two uniforms", and something like "Poisson: 1" for every rate.

**what happens:**

| sampler | uniforms per draw |
|---|---|
| normal | 1.00 |
| exponential | 1.00 |
| bernoulli(0.3) | 1.00 |
| gamma(3) | 2.02 |
| gamma(0.5) | 3.06 |
| beta(2, 5) | 4.05 |
| poisson(5) | 5.98 |
| poisson(30) | 31.09 |
| poisson(31), poisson(1000) | 1.00 |

**why:** Box–Muller turns two uniforms into *two* normals, so it's exactly one per draw (if you use both outputs). Marsaglia–Tsang spends one normal (one uniform) plus one uniform per attempt, and accepts about 99% of attempts for shape 3, so a bit over 2. the boost for shape 0.5 adds one uniform on top of a Gamma(1.5). a Beta is two Gammas. Knuth's Poisson method simulates every arrival in the window, so it needs $\lambda + 1$ uniforms on average: the count, plus the one that crosses the threshold. at $\lambda = 31$ your code switches to the table, which costs one uniform and a binary search, at any $\lambda$. Knuth's method is still worth having: it needs no table and no special functions, it's exact for any small $\lambda$, and for a fresh $\lambda$ on every call (common in simulations where the rate itself is random) building a table each time would cost more than it saves. uniform counts aren't the whole story (a uniform costs a few nanoseconds; logs, cosines and table lookups cost more), but they show where the work goes.
