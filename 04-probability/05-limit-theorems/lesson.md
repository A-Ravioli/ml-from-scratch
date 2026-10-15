# limit theorems

## why this matters

you train a classifier and it gets 912 of 1000 test examples right. you report "91.2% accuracy". how much of that number is the model, and how much is the luck of which 1000 examples you drew? your minibatch gradient is an average of 64 per-example gradients. how close is it to the true gradient, and in what way is it wrong? a Monte Carlo estimate averages a million samples. why does it work at all, and how do you put an error bar on it?

two theorems answer all three questions. the **law of large numbers** (LLN) says that averages of independent draws settle down to the expected value. the **central limit theorem** (CLT) says *how* they settle: the error of an average of $n$ draws is approximately Gaussian, with a standard deviation that shrinks like $\sigma/\sqrt n$, almost regardless of what the individual draws look like. that's where every "± 1.96 standard errors" comes from.

"almost regardless" is doing a lot of work, though. both theorems have conditions, and real ML quantities violate them more often than you'd think: importance weights, rewards in reinforcement learning, per-example losses with outliers and gradient noise can all have heavy tails. in this topic you'll build a simulator that measures how close averages get to their limits, and in the lab you'll find a distribution where the average never settles at all.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. you flip a fair coin 1000 times. what's the probability that the number of heads is between 480 and 520?
2. the standard Cauchy distribution is a bell-shaped curve centered at 0. you track the running average of Cauchy draws. does it settle down to 0?
3. the average of 30 exponential draws: does its distribution look Gaussian yet? what about 30 draws from a distribution that's 0 with probability 0.99 and 100 with probability 0.01?
4. the LLN says the fraction of heads converges to 1/2. does the *number* of heads minus 500 (out of 1000), or minus $n/2$ in general, converge to 0?
5. averages converge to a Gaussian "in distribution". does that mean the averages themselves converge to some particular random number?

## prerequisites check

- [ ] you know expectation and variance, and that $\operatorname{Var}(\bar X_n) = \sigma^2/n$ for i.i.d. draws (→ ch 04 · topic 03)
- [ ] you can sample from common distributions and pass a `rng` around (→ ch 04 · topic 02)
- [ ] you know what a CDF is, and the standard normal CDF $\Phi$ (→ ch 04 · topic 02)
- [ ] you know what $\varepsilon$–$N$ convergence of a sequence of numbers means (→ ch 03 · topic 01)

## core ideas

### idea 1: the law of large numbers

let $X_1, X_2, \dots$ be independent draws from a distribution with mean $\mu$ and variance $\sigma^2$, and let

$$\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i$$

be the average of the first $n$. by topic 03, $\mathbb{E}[\bar X_n] = \mu$ and $\operatorname{Var}(\bar X_n) = \sigma^2/n$. the expected value is right on target, and the spread around it shrinks to zero. that already suggests $\bar X_n \to \mu$. Chebyshev's inequality (which topic 06 proves) turns it into a statement:

$$P\big(|\bar X_n - \mu| > \varepsilon\big) \le \frac{\sigma^2}{n\varepsilon^2} \xrightarrow{n \to \infty} 0 \quad \text{for every } \varepsilon > 0.$$

that's the **weak law of large numbers**: the probability of being off by more than any fixed $\varepsilon$ goes to zero. (the condition $\sigma^2 < \infty$ makes the proof one line; the weak law actually only needs a finite mean.) the **strong law** says more: with probability 1, the sequence $\bar X_1, \bar X_2, \dots$ converges to $\mu$, like an ordinary sequence of numbers. pick one infinite path of draws and its running average settles at $\mu$ and stays there.

the strong law needs exactly one condition: $\mathbb{E}|X| < \infty$. when the mean doesn't exist, nothing settles. that's warm-up 2. the **Cauchy** distribution, with density $\frac{1}{\pi(1 + x^2)}$, looks like a bell curve with fatter tails, but its tails are so fat that $\int|x|\,f(x)\,dx = \infty$. the running average of Cauchy draws lurches around forever: every so often a draw so huge arrives that it drags the average somewhere new. (on paper 3 shows something stronger: the average of $n$ Cauchy draws has exactly the same distribution as a single draw.)

and warm-up 4: the LLN is about the *fraction*. the count of heads minus $n/2$ does not converge to 0. its typical size grows like $\sqrt n/2$. the fraction converges because you divide that growing error by the even faster-growing $n$.

> **🔮 predict:** a Pareto distribution with tail exponent 1.5 has a finite mean (3) but an infinite variance. does the strong law hold for it? would you expect its running average to look like a normal distribution's, settling smoothly, or different? (the lab runs both, with the Cauchy for comparison.)

to look at one path you need running averages, and to look at the distribution of $\bar X_n$ you need many independent averages. you'll build both. a **sampler** in this topic is any function `sampler(size, rng)` that returns i.i.d. draws with shape `size`.

> **▶ build: stage 1.** implement `running_mean(x)` and `sample_means(sampler, n, n_reps, *, rng=None)` (one `(n_reps, n)` draw, averaged along each row) in `exercise.py` (`marimo edit exercise.py`).

### idea 2: measuring the distance to a distribution

to check a claim like "$\bar X_n$ is approximately Gaussian", you need a way to measure how far a sample is from a distribution. the cleanest is to compare CDFs. the **empirical CDF** of a sample $z_1, \dots, z_n$ is the fraction of points at or below $t$:

$$F_n(t) = \frac{1}{n}\#\{i : z_i \le t\}.$$

it's a staircase that jumps by $1/n$ at each data point. the **Kolmogorov–Smirnov (KS) distance** to a continuous CDF $F$ is the largest vertical gap between them:

$$D_n = \sup_t\,|F_n(t) - F(t)|.$$

you can't check every $t$, but you don't need to. between data points $F_n$ is flat and $F$ is increasing, so the gap is largest right next to a jump. with the points sorted, $z_{(1)} \le \cdots \le z_{(n)}$, the staircase is at height $(i-1)/n$ just before $z_{(i)}$ and at height $i/n$ just after, so

$$D_n = \max_{i}\,\max\Big(\frac{i}{n} - F(z_{(i)}),\ F(z_{(i)}) - \frac{i - 1}{n}\Big).$$

checking only one side of each jump is the classic bug: for the two points $(-1, 1)$ the largest gap is just *before* the jump at 1, where the staircase is still at 1/2 and $\Phi(1) = 0.841$.

for the standard normal you need $\Phi$. it has no closed form, but it's a rescaled **error function**, which every math library has: $\Phi(z) = \tfrac12\big(1 + \operatorname{erf}(z/\sqrt2)\big)$.

even a sample from exactly the right distribution has $D_n > 0$, of typical size about $0.87/\sqrt n$. so when you measure "how Gaussian" something is with $n_{\text{reps}}$ samples, distances below a few times $1/\sqrt{n_{\text{reps}}}$ are noise.

> **▶ build: stage 2.** implement `normal_cdf(z)` (elementwise, with `math.erf`) and `ks_distance(z)`, the KS distance between a sample and $\mathcal{N}(0, 1)$.

### idea 3: the central limit theorem

the LLN says $\bar X_n - \mu \to 0$. zoom in on that shrinking error by multiplying by $\sqrt n$, which exactly cancels the $\sigma/\sqrt n$ shrinkage, and something remarkable happens. if the $X_i$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2$, then

$$Z_n = \frac{\sqrt n\,(\bar X_n - \mu)}{\sigma} \ \xrightarrow{d}\ \mathcal{N}(0, 1),$$

meaning $P(Z_n \le t) \to \Phi(t)$ for every $t$. whatever the shape of the original distribution (skewed, discrete, bounded, bimodal), the shape of the average's error converges to the same bell curve. only $\mu$ and $\sigma$ survive; every other detail is averaged away. on paper 2 sketches why: adding independent variables multiplies their characteristic functions, and after rescaling, only the second-order term of each survives.

warm-up 1 is the CLT in action. the number of heads $S$ has mean 500 and standard deviation $\sqrt{1000/4} \approx 15.81$. so

$$P(480 \le S \le 520) \approx P\Big(|Z| \le \frac{20.5}{15.81}\Big) = 2\Phi(1.297) - 1 \approx 0.805,$$

where the $.5$ (a "continuity correction") accounts for $S$ being an integer. the exact binomial answer is $0.8052$. a 4% band around 500 contains 80% of the probability.

**how fast?** "converges" alone tells you nothing about $n = 30$. the **Berry–Esseen theorem** gives a guarantee: if $\rho = \mathbb{E}|X - \mu|^3 < \infty$, then for every $n$,

$$\sup_t\,\big|P(Z_n \le t) - \Phi(t)\big| \le \frac{C\rho}{\sigma^3\sqrt n}, \qquad C \le 0.4748.$$

two lessons. the error shrinks like $1/\sqrt n$: to halve it, quadruple $n$. and the constant depends on $\rho/\sigma^3$, a measure of how skewed and heavy-tailed the distribution is. a symmetric, light-tailed distribution like the uniform looks Gaussian after a handful of terms. a skewed one like the exponential takes longer. and warm-up 3's distribution (0 with probability 0.99, 100 with probability 0.01) has $\rho/\sigma^3 \approx 9.9$: with 30 draws, the most likely sum is 0, and the average's distribution is a lopsided handful of spikes, nothing like a bell.

**when it fails.** the CLT needs $\sigma^2 < \infty$. if the tails are heavy enough that the variance is infinite but the mean is finite (a Pareto distribution with tail exponent between 1 and 2), the LLN still holds but the $\sqrt n$ scaling is wrong and the limit isn't Gaussian. for the Cauchy, there's no mean at all, and the average never concentrates. the lab measures all of this with your simulator.

> **🔮 predict:** for the mean of $n$ exponential draws, the KS distance to the normal is 0.159 at $n = 1$. what is it at $n = 4$, $16$ and $64$? (the tests compute it exactly.)

> **▶ build: stage 3.** implement `clt_distances(sampler, mu, sigma, ns, n_reps, *, rng=None)`, which standardizes `n_reps` sample means for each `n` and measures their KS distance to $\mathcal{N}(0, 1)$, and `berry_esseen_bound(n, sigma, rho, C=0.4748)`.

### idea 4: modes of convergence

"$\bar X_n$ converges" has several different meanings, and the two theorems use different ones. for random variables $Y_n$ and a limit $Y$:

| mode | definition | example |
|---|---|---|
| **almost surely** | $P(Y_n \to Y) = 1$: each path converges, as a sequence of numbers | strong LLN |
| **in probability** | $P(\lvert Y_n - Y\rvert > \varepsilon) \to 0$ for every $\varepsilon > 0$ | weak LLN |
| **in mean square** | $\mathbb{E}[(Y_n - Y)^2] \to 0$ | $\bar X_n \to \mu$, since $\mathbb{E}[(\bar X_n - \mu)^2] = \sigma^2/n$ |
| **in distribution** | $P(Y_n \le t) \to P(Y \le t)$ at every continuity point $t$ | CLT |

almost sure convergence and mean-square convergence each imply convergence in probability, which implies convergence in distribution. none of the arrows reverse in general.

convergence in distribution is the odd one out: it's a statement about *distributions*, not about the random variables themselves. that's warm-up 5. $Z_n$ doesn't converge to any particular random number. $Z_{1000}$ and $Z_{2000}$ are highly dependent but different, and the sequence $Z_1, Z_2, \dots$ wanders forever. only the *shape* of its distribution settles down. an extreme example: if $Z \sim \mathcal{N}(0, 1)$ and $Y_n = -Z$ for every $n$, then $Y_n \to Z$ in distribution (both are standard normal), but $|Y_n - Z| = 2|Z|$ never shrinks.

convergence in probability is the one you can measure directly by simulation: estimate $P(|\bar X_n - \mu| > \varepsilon)$ for a growing list of $n$ and watch it go to zero. for moderate $n$ the CLT suggests it behaves like $2\Phi(-\varepsilon\sqrt n/\sigma)$. careful, though: for a fixed $\varepsilon$ and large $n$, the event $|\bar X_n - \mu| > \varepsilon$ is many standard errors out, deep in the tail, where (as the lab shows) the normal approximation is not reliable. the true decay is exponential in $n$, $e^{-nI(\varepsilon) + o(n)}$, with a rate $I(\varepsilon)$ that depends on the whole distribution, not just $\sigma$ (it's a KL divergence for coin flips). topic 06 computes that rate, and asks how fast you can *guarantee* the decay without knowing the distribution.

> **▶ build: stage 4.** implement `deviation_probability(sampler, mu, eps, ns, n_reps, *, rng=None)`: Monte Carlo estimates of $P(|\bar X_n - \mu| > \varepsilon)$, with a strict inequality, for each $n$.

## worked example

**running means.** for the draws $(2, 4, 6, 8, 0)$, the running sums are $2, 6, 12, 20, 20$, and dividing by $1, 2, 3, 4, 5$ gives running means $2, 3, 4, 5, 4$.

**a KS distance by hand.** for the sample $(1, -1)$: sorted, $z_{(1)} = -1$ and $z_{(2)} = 1$, with $\Phi(-1) \approx 0.1587$ and $\Phi(1) \approx 0.8413$. the four candidate gaps are $\tfrac12 - 0.1587 = 0.3413$, $0.1587 - 0 = 0.1587$, $1 - 0.8413 = 0.1587$, and $0.8413 - \tfrac12 = 0.3413$. so $D = 0.3413$. a single point at 0 gives $D = \tfrac12$.

**the CLT for exponentials.** for Exponential(1), $\mu = \sigma = 1$ and $\rho = \mathbb{E}|X - 1|^3 = 12/e - 2 \approx 2.415$. the sum of $n$ exponentials is Gamma($n$), so the exact distance of $Z_n$ from $\mathcal{N}(0, 1)$ can be computed:

| $n$ | exact KS distance | Berry–Esseen bound $0.4748\,\rho/\sqrt n$ |
|---|---|---|
| 1 | 0.159 | 1.146 |
| 4 | 0.067 | 0.573 |
| 16 | 0.033 | 0.287 |
| 64 | 0.017 | 0.143 |

each quadrupling of $n$ halves the distance, exactly the $1/\sqrt n$ rate, and the bound, while correct, is about 8 times too pessimistic.

**the weak law for a coin.** $P(|\bar X_8 - \tfrac12| > \tfrac18)$ for 8 fair flips: the average is $k/8$, and $|k/8 - 1/2| > 1/8$ means $k \le 2$ or $k \ge 6$. so the probability is $2\cdot\frac{1 + 8 + 28}{256} = \frac{74}{256} \approx 0.289$. with $\ge$ instead of $>$, $k = 3$ and $k = 5$ also count, giving $\frac{186}{256} \approx 0.727$. on a discrete distribution, the strict inequality matters.

the tests check the running means, both KS distances, the exponential distances, and the coin probabilities.

## on paper

1. **the weak law.** assuming Chebyshev's inequality, $P(|Y - \mathbb{E}Y| > \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$, prove the weak LLN for i.i.d. draws with finite variance. how large must $n$ be to guarantee $P(|\bar X_n - \mu| > 0.01) \le 0.05$ for a fair coin?

<details><summary>solution</summary>

$\operatorname{Var}(\bar X_n) = \sigma^2/n$, so $P(|\bar X_n - \mu| > \varepsilon) \le \sigma^2/(n\varepsilon^2) \to 0$. for a coin, $\sigma^2 = 1/4$: $\frac{1/4}{n \cdot 10^{-4}} \le 0.05$ needs $n \ge 50{,}000$. (the CLT says about $9{,}600$ is actually enough, since $1.96^2 \cdot 0.25/10^{-4} \approx 9604$. Chebyshev is a guarantee for *every* distribution with that variance, so it's loose for any particular one. topic 06 is about tighter guarantees.)

</details>

2. **why the CLT holds (sketch).** let $Y_i = (X_i - \mu)/\sigma$, so $\mathbb{E}Y_i = 0$ and $\mathbb{E}Y_i^2 = 1$, and let $\varphi(t) = \mathbb{E}[e^{itY}]$ be the characteristic function. show that $Z_n = \frac{1}{\sqrt n}\sum Y_i$ has characteristic function $\varphi(t/\sqrt n)^n$, expand $\varphi(s) = 1 - s^2/2 + o(s^2)$, and find the limit. (the standard normal's characteristic function is $e^{-t^2/2}$, and characteristic functions determine distributions.)

<details><summary>solution</summary>

independence makes the expectation of a product of $e^{itY_j/\sqrt n}$ factor: $\mathbb{E}[e^{itZ_n}] = \prod_j\mathbb{E}[e^{itY_j/\sqrt n}] = \varphi(t/\sqrt n)^n$. the expansion comes from $\varphi(0) = 1$, $\varphi'(0) = i\mathbb{E}Y = 0$, $\varphi''(0) = -\mathbb{E}Y^2 = -1$. then $\varphi(t/\sqrt n)^n = \big(1 - \frac{t^2}{2n} + o(1/n)\big)^n \to e^{-t^2/2}$. every feature of the distribution beyond its first two moments sits in the $o(1/n)$ term, which vanishes in the limit. that's why the limit is universal. (making "characteristic functions determine the limit" rigorous is Lévy's continuity theorem.)

</details>

3. **the Cauchy average.** the standard Cauchy has characteristic function $e^{-|t|}$. show that $\bar X_n$ of $n$ i.i.d. standard Cauchy draws is again standard Cauchy. why doesn't this contradict the LLN or the CLT?

<details><summary>solution</summary>

$\mathbb{E}[e^{it\bar X_n}] = \varphi(t/n)^n = (e^{-|t|/n})^n = e^{-|t|}$, the characteristic function of a standard Cauchy. so averaging any number of Cauchy draws gives you exactly one Cauchy draw's worth of uncertainty. no contradiction: the LLN needs $\mathbb{E}|X| < \infty$ and the CLT needs $\sigma^2 < \infty$, and the Cauchy has neither. the expansion in on paper 2 fails because $\varphi(s) = 1 - |s| + \dots$ has a first-order term, so the right scaling is $1/n$, not $1/\sqrt n$, and the limit is Cauchy, not normal.

</details>

4. **in distribution, not in probability.** let $Z \sim \mathcal{N}(0, 1)$ and $Y_n = (-1)^n Z$. show that $Y_n$ converges to $Z$ in distribution but not in probability.

<details><summary>solution</summary>

$-Z$ has the same distribution as $Z$ (the normal is symmetric), so every $Y_n$ is standard normal and the CDFs converge trivially. but for odd $n$, $|Y_n - Z| = 2|Z|$, so $P(|Y_n - Z| > \varepsilon) = P(|Z| > \varepsilon/2)$, which doesn't depend on $n$ and isn't 0. convergence in distribution says nothing about the joint behavior of $Y_n$ and $Z$.

</details>

## experiment

`lab.py` runs three experiments with your simulator:

1. running averages for several distributions, Cauchy included.
2. the CLT across distributions: how fast the standardized mean approaches the normal, measured with your `clt_distances`, against the Berry–Esseen bound.
3. the CLT in the tails: how good is the normal approximation for the probability of a *rare* deviation?

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. why does $\sqrt n$ appear in the CLT, and not $n$ or $\log n$?

<details><summary>answer</summary>

$\bar X_n - \mu$ has standard deviation $\sigma/\sqrt n$. multiplying by $\sqrt n$ is the unique scaling that keeps the spread fixed: with a smaller factor everything collapses to 0 (that's the LLN), with a larger one it blows up. for infinite-variance distributions the right scaling is different, which is one way to see why the CLT fails for them.

</details>

2. your test accuracy is 0.912 on 1000 examples. what's the standard error, and what does the CLT let you say?

<details><summary>answer</summary>

each example is a Bernoulli with $p \approx 0.912$, so the standard error is $\sqrt{0.912 \cdot 0.088/1000} \approx 0.009$. by the CLT the estimate is approximately normal around the true accuracy, so $0.912 \pm 1.96 \cdot 0.009$, about $[0.894, 0.930]$, is an approximate 95% interval. a rival model at 0.905 is well within the noise. (chapter 05 makes this rigorous, chapter 07 makes it routine.)

</details>

3. does the strong LLN say that the fraction of heads will eventually equal exactly 1/2?

<details><summary>answer</summary>

no. it says the fraction converges to 1/2: for every $\varepsilon$, it eventually stays within $\varepsilon$ of 1/2. it can equal exactly 1/2 only at even $n$, and the probability of exact equality at $n$ flips goes to zero like $1/\sqrt n$.

</details>

4. why does a KS distance computed from `n_reps` samples never reach zero, even for an exactly Gaussian quantity?

<details><summary>answer</summary>

the empirical CDF of any finite sample is a staircase, and it fluctuates around the true CDF by about $1/\sqrt{n_{\text{reps}}}$ (typical $D \approx 0.87/\sqrt{n_{\text{reps}}}$). measured distances below that floor carry no information about the CLT.

</details>

5. a minibatch gradient averages $B$ per-example gradients. what do the LLN and CLT say about it, and when would you distrust them?

<details><summary>answer</summary>

the minibatch gradient is an unbiased estimate of the full gradient (sampling uniformly), its noise shrinks like $1/\sqrt B$, and for large $B$ the noise is approximately Gaussian. distrust the Gaussian picture when per-example gradients are heavy-tailed (a few examples with enormous gradients, common in language models and RL); then the noise is spiky, and gradient clipping (ch 15) exists partly for this reason.

</details>

6. which mode of convergence does each of these use: (a) "the running average of my simulation converged", (b) "the test statistic is asymptotically normal"?

<details><summary>answer</summary>

(a) almost sure convergence: one path settling down, the strong law. (b) convergence in distribution: the CLT, which makes no claim about any single sequence settling.

</details>

7. the Berry–Esseen bound for exponentials is 8 times larger than the true distance. is it useless?

<details><summary>answer</summary>

no: it holds for *every* distribution with the same $\rho/\sigma^3$, with no further assumptions, and it gives the right rate, $1/\sqrt n$. it's the worst case over a huge family. for a specific distribution you can do better by simulation (stage 3) or exact computation. the bound tells you when you don't need to check.

</details>

## common bugs

- **dividing by $n$ instead of by the running count** in `running_mean`. symptom: a "running average" that starts near 0 and grows towards the mean.
- **averaging over the wrong axis.** `x.mean(axis=0)` averages *across* repetitions. symptom: sample means with almost no spread.
- **standardizing with the variance.** $(\bar X_n - \mu)/\sigma^2$ instead of $/\sigma$. symptom: CLT distances that never shrink for any distribution with $\sigma \ne 1$.
- **one-sided KS.** checking only $i/n - F(z_{(i)})$. symptom: distances that are too small for samples shifted to the right, and a mismatch with `scipy.stats.kstest`.
- **$\ge$ vs $>$.** with discrete data the boundary has positive probability. symptom: deviation probabilities that are too big for coins and dice.
- **re-seeding inside a loop.** creating `np.random.default_rng(seed)` for every $n$ reuses the same random numbers, so results for different $n$ are correlated. symptom: suspiciously smooth curves (not wrong exactly, but misleading).

## going deeper

- Wasserman (2004), *All of Statistics*, Springer, chapter 5: convergence of random variables, the LLN and the CLT, in a dozen pages.
- Feller (1971), *An Introduction to Probability Theory and Its Applications, vol. II*, 2nd ed., Wiley, chapters VIII and XVI: the CLT, Berry–Esseen, and stable laws.
- Shevtsova (2011), "On the absolute constants in the Berry–Esseen type inequalities for identically distributed summands", [arXiv:1111.6554](https://arxiv.org/abs/1111.6554). where $C \le 0.4748$ comes from.
- Durrett (2019), *Probability: Theory and Examples*, 5th ed., Cambridge University Press, chapters 2–3: the measure-theoretic proofs of both laws and the CLT. free from the author's website.
- Taleb (2020), *Statistical Consequences of Fat Tails*, STEM Academic Press, chapters 3–4: what goes wrong with averages when tails are heavy, with many simulations. [arXiv:2001.10488](https://arxiv.org/abs/2001.10488)

## connections

- ← builds on: sampling (topic 02), variance of a mean (topic 03), the normal CDF (topic 02), limits (ch 03 · topic 01).
- → used in: concentration inequalities, which make the LLN quantitative for finite $n$ (topic 06), the ergodic theorem for Markov chains (topic 07), confidence intervals and tests (ch 05), evaluation with uncertainty (ch 07), generalization bounds (ch 13), and SGD noise (ch 06, 15).
