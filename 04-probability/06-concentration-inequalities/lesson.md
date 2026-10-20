# concentration inequalities

## why this matters

how many labeled test examples do you need before you can trust a measured accuracy to within one percentage point? an A/B test has run for 10,000 users: is the difference real? a bandit algorithm must decide which of five ads to show next: how optimistic should it be about the one it has barely tried? and the deepest version: a model gets 2% training error. what can you say, with a guarantee, about its error on data it hasn't seen? (that's chapter 13.)

the central limit theorem gives approximate answers, but topic 05's lab showed its weak spot: it's an asymptotic statement, and it can be badly wrong in the tails, which is exactly where "with 99% confidence" lives. **concentration inequalities** give a different kind of answer: guaranteed upper bounds on the probability that an average strays far from its mean, valid for *every* $n$ and *every* distribution satisfying a few simple assumptions (bounded, or known variance). you pay for the guarantee with some looseness. this topic is about how much: you'll build the five classic bounds, from the crude (Markov) to the nearly exact (Chernoff), and in the lab you'll measure each one against the truth.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. the average income in a town is \$50,000. what's the largest possible fraction of people earning at least \$500,000?
2. for a normal distribution, 5% of the probability lies more than 2 standard deviations from the mean. what's the most it could be for *any* distribution with a finite variance?
3. you want to estimate the bias of a coin to within $\pm 0.01$, with a guarantee that holds with probability 99%. how many flips do you need?
4. you're estimating a click-through rate around 1% (so the variance $p(1-p)$ is tiny). does knowing the variance is small let you use fewer samples than question 3's answer?
5. you evaluate 1000 different models on the same 1000 test examples and report the best one's accuracy. is that number trustworthy?

## prerequisites check

- [ ] you know expectation, variance, and linearity of expectation (→ ch 04 · topic 03)
- [ ] you know the weak law of large numbers and $\operatorname{Var}(\bar X_n) = \sigma^2/n$ (→ ch 04 · topic 05)
- [ ] you can manipulate $e^x$ and $\log x$, and use $1 + x \le e^x$ (→ ch 01)
- [ ] you know what convexity means for a function of one variable (→ ch 03 · topic 05)

## core ideas

### idea 1: Markov and Chebyshev

start with the crudest possible information: $X \ge 0$ and you know its mean. **Markov's inequality** says

$$P(X \ge t) \le \frac{\mathbb{E}[X]}{t} \qquad \text{for every } t > 0.$$

the proof is a picture. the indicator $\mathbb{1}[X \ge t]$ is 1 when $X \ge t$ and 0 otherwise, and since $X \ge 0$, it's always at most $X/t$ (on paper 1). take expectations of both sides and you're done. that's warm-up 1: at most $50{,}000/500{,}000 = 10\%$ of people can earn \$500,000 or more. if more did, they alone would push the average above \$50,000.

Markov is **tight**: no better bound is possible from the mean alone. the distribution that puts mass $\mathbb{E}[X]/t$ at exactly $t$ and the rest at 0 has the given mean and meets the bound with equality. but for most distributions it's very loose, and it only bounds one tail, at a rate of $1/t$.

apply Markov to a better random variable and you get a better bound. $(X - \mu)^2$ is non-negative with mean $\sigma^2$, and $|X - \mu| \ge t$ exactly when $(X - \mu)^2 \ge t^2$, so

$$P(|X - \mu| \ge t) \le \frac{\sigma^2}{t^2}.$$

that's **Chebyshev's inequality**. with $t = k\sigma$ it says at most $1/k^2$ of the probability lies more than $k$ standard deviations out. warm-up 2: at most $1/4 = 25\%$ lies beyond $2\sigma$, five times the normal's 5%, and that's tight too (a three-point distribution achieves it). for an average of $n$ i.i.d. draws, $\sigma^2$ becomes $\sigma^2/n$, which gives the one-line proof of the weak LLN you used in topic 05.

every bound in this topic returns $\min(1, \cdot)$: a "bound" of 2.5 on a probability is true but says nothing.

> **▶ build: stage 1.** implement `markov_bound(mean, t)` and `chebyshev_bound(var, t)` in `exercise.py` (`marimo edit exercise.py`). both clip at 1 and work elementwise on arrays.

### idea 2: exponential bounds and Hoeffding

Chebyshev's bound on an average shrinks like $1/n$. but topic 05 suggested the truth shrinks like $e^{-cn}$, much faster. the trick for getting exponential bounds is due to Chernoff: apply Markov to $e^{\lambda S}$ for a sum $S$, and then choose the best $\lambda > 0$:

$$P(S \ge s) = P(e^{\lambda S} \ge e^{\lambda s}) \le e^{-\lambda s}\,\mathbb{E}[e^{\lambda S}] = e^{-\lambda s}\prod_{i=1}^n\mathbb{E}[e^{\lambda X_i}].$$

the last step is where **independence** is essential: the expectation of a product of independent factors is the product of expectations. now you need a bound on each factor $\mathbb{E}[e^{\lambda X_i}]$, the moment generating function. for a variable confined to $[a, b]$, **Hoeffding's lemma** gives $\mathbb{E}[e^{\lambda(X - \mu)}] \le e^{\lambda^2(b - a)^2/8}$: a bounded variable's tails are no heavier than a Gaussian's with standard deviation $(b - a)/2$. plug in, optimize over $\lambda$ (on paper 2), and add the mirror-image bound for the lower tail:

$$P\big(|\bar X_n - \mu| \ge t\big) \le 2\exp\Big(-\frac{2nt^2}{(b - a)^2}\Big).$$

that's **Hoeffding's inequality**. it needs only independence and boundedness: no knowledge of the variance, no assumption about the shape. and it decays exponentially in $n$ and in $t^2$, like a Gaussian tail.

because it's explicit, you can invert it: to guarantee $P(|\bar X_n - \mu| \ge t) \le \delta$, solve $2e^{-2nt^2/(b-a)^2} \le \delta$ for $n$:

$$n \ge \frac{(b - a)^2\log(2/\delta)}{2t^2}.$$

warm-up 3: for a coin ($[a, b] = [0, 1]$), $t = 0.01$, $\delta = 0.01$, that's $n \ge \log(200)/0.0002 \approx 26{,}492$ flips. the CLT's approximate answer is about $1.96^2/(4 \cdot 10^{-4}) = 9{,}604$ for 95%, or about $16{,}600$ for 99%. Hoeffding's guarantee costs roughly 60% more samples than the approximation, which is a fair price for "holds for every $n$, no asymptotics". note how $\delta$ enters: only through $\log(1/\delta)$. demanding 99.99% instead of 99% costs less than double the samples.

> **🔮 predict:** Hoeffding's bound for 100 fair coin flips and $t = 0.1$ is $2e^{-2} \approx 0.27$. what is the true probability that the fraction of heads is at least 0.1 away from 1/2? (the lab and the worked example find out.)

> **▶ build: stage 2.** implement `hoeffding_bound(n, t, a=0.0, b=1.0)` and `hoeffding_sample_size(t, delta, a=0.0, b=1.0)`, the smallest integer $n$ that makes the bound at most $\delta$.

### idea 3: Bernstein, when the variance is small

Hoeffding uses only the range $[a, b]$, so it treats every variable in $[0, 1]$ as if it could be a fair coin, the highest-variance case. a click that happens 1% of the time has variance $0.0099$, about 25 times smaller than a fair coin's $0.25$, and Hoeffding can't see that. warm-up 4 says you should be able to do better, and **Bernstein's inequality** does. if $|X_i - \mu| \le c$ and $\operatorname{Var}(X_i) = \sigma^2$, then

$$P\big(|\bar X_n - \mu| \ge t\big) \le 2\exp\Big(-\frac{nt^2}{2\sigma^2 + \tfrac23 ct}\Big).$$

look at the two regimes of the denominator. for small deviations ($ct \ll \sigma^2$), it's about $2\exp(-nt^2/2\sigma^2)$: exactly the Gaussian tail with the *true* variance, as if the CLT were exact. for large deviations ($ct \gg \sigma^2$), it's about $2\exp(-3nt/2c)$: exponential in $t$ rather than $t^2$, which reflects that a sum of rare events has a Poisson-like, heavier tail than a Gaussian.

for the 1% click rate, the difference is dramatic. with $n = 100$ and $t = 0.1$, Hoeffding gives 0.27 and Bernstein gives $1.7 \times 10^{-5}$. when the variance is small compared with the range squared, which is the usual situation for rare events, error rates near 0, and losses that are usually small, Bernstein is the bound to use.

> **▶ build: stage 3.** implement `bernstein_bound(n, t, var, c)`.

### idea 4: the Chernoff bound for coins, and the truth

for the most important special case, an average of Bernoulli($p$) variables (errors, clicks, heads), you don't need a generic bound on the moment generating function: you can compute it exactly, $\mathbb{E}[e^{\lambda X}] = 1 - p + pe^\lambda$. carrying out the Chernoff method with the exact MGF and the best $\lambda$ (on paper 3) gives

$$P(\bar X_n \ge p + t) \le \exp\big(-n\,\mathrm{KL}(p + t \,\|\, p)\big), \qquad \mathrm{KL}(q \,\|\, p) = q\log\frac qp + (1 - q)\log\frac{1-q}{1-p}.$$

$\mathrm{KL}(q\,\|\,p)$ is the **Kullback–Leibler divergence** between a coin with bias $q$ and one with bias $p$, which you'll meet properly in chapter 05. it measures how distinguishable the two coins are, and it's always at least $2(q - p)^2$ (Pinsker's inequality), so this bound is never worse than one-sided Hoeffding, $e^{-2nt^2}$. it's also the *right* exponent: the true tail probability is $e^{-n\mathrm{KL}}$ times a factor that only shrinks polynomially in $n$, so no bound can have a better exponential rate (this is Cramér's theorem from large-deviations theory). this bound is one-sided; for the lower tail, use $\mathrm{KL}(p - t\,\|\,p)$.

two edge cases: if $p + t > 1$ the event is impossible and the bound is 0. if $p + t = 1$, the event is "every draw is 1", with probability exactly $p^n$, and since $\mathrm{KL}(1\,\|\,p) = \log(1/p)$ (using $0\log 0 = 0$), the bound gives exactly $p^n$.

to judge a bound you need the truth to compare with. for a simulation, that's the **empirical tail**: the fraction of simulated averages at least $t$ from the mean. you'll write it to work for a whole array of thresholds $t$ at once.

> **🔮 predict:** for 100 fair coins and a deviation of 0.1 (one-sided), order these from smallest to largest: the true probability, Chernoff, one-sided Hoeffding $e^{-2nt^2}$, Chebyshev.

> **▶ build: stage 4.** implement `chernoff_bernoulli_bound(n, p, t)` (one-sided, with both edge cases) and `empirical_tail(samples, mu, t, two_sided=True)`.

## worked example

**Markov and Chebyshev.** a non-negative $X$ with mean 2: $P(X \ge 10) \le 2/10 = 0.2$. with variance 4, $P(|X - 2| \ge 4) \le 4/16 = 0.25$.

**Hoeffding.** 100 draws in $[0, 1]$, $t = 0.1$: $2\exp(-2 \cdot 100 \cdot 0.01) = 2e^{-2} \approx 0.2707$. sample size for $t = 0.05$, $\delta = 0.05$: $\log(40)/(2 \cdot 0.0025) = 3.689/0.005 = 737.8$, so $n = 738$.

**Bernstein vs Hoeffding for a rare event.** Bernoulli(0.01): $\sigma^2 = 0.0099$, $c = 0.99$. with $n = 100$, $t = 0.1$:

$$2\exp\Big(-\frac{100 \cdot 0.01}{2 \cdot 0.0099 + \tfrac23 \cdot 0.99 \cdot 0.1}\Big) = 2\exp\Big(-\frac{1}{0.0858}\Big) \approx 1.7 \times 10^{-5},$$

against Hoeffding's 0.27. the truth, $P(\text{at least 11 successes in 100})$, is about $6 \times 10^{-9}$.

**Chernoff for a fair coin.** $n = 100$, $p = 0.5$, $t = 0.1$:

$$\mathrm{KL}(0.6\,\|\,0.5) = 0.6\log 1.2 + 0.4\log 0.8 \approx 0.10939 - 0.08926 = 0.02014,$$

so the bound is $e^{-2.014} \approx 0.1335$. one-sided Hoeffding gives $e^{-2} \approx 0.1353$ (barely worse, since $2t^2 = 0.02$ is close to the KL), and the exact $P(\text{at least 60 heads}) \approx 0.0284$. all the exponential bounds are within a factor of about 5 of the truth here, which, for a tail probability, is close.

**an edge case.** 10 Bernoulli(0.3) draws, $t = 0.7$: the event is "all ten are 1", probability $0.3^{10} \approx 5.9 \times 10^{-6}$, and the Chernoff bound $e^{-10\log(1/0.3)} = 0.3^{10}$ is exact.

**an empirical tail.** samples $(0, 1, 2, 3, 4)$ with $\mu = 2$: the deviations are $(2, 1, 0, 1, 2)$, so the fraction with deviation $\ge 1$ is $4/5$, and with deviation $\ge 1.5$ it's $2/5$.

the tests check every number here.

## on paper

1. **Markov and Chebyshev.** prove $\mathbb{1}[X \ge t] \le X/t$ for $X \ge 0$, $t > 0$, and deduce Markov's inequality. then deduce Chebyshev's.

<details><summary>solution</summary>

if $X \ge t$, the left side is 1 and $X/t \ge 1$. if $X < t$, the left side is 0 and $X/t \ge 0$ since $X \ge 0$. taking expectations (which preserve inequalities), $P(X \ge t) = \mathbb{E}\,\mathbb{1}[X \ge t] \le \mathbb{E}[X]/t$. for Chebyshev, apply Markov to $Y = (X - \mu)^2 \ge 0$ at level $t^2$: $P(|X - \mu| \ge t) = P(Y \ge t^2) \le \mathbb{E}[Y]/t^2 = \sigma^2/t^2$.

</details>

2. **Hoeffding from the lemma.** using $\mathbb{E}[e^{\lambda(X_i - \mu)}] \le e^{\lambda^2(b-a)^2/8}$, show $P(\bar X_n - \mu \ge t) \le \exp(-2nt^2/(b-a)^2)$.

<details><summary>solution</summary>

with $S = \sum_i(X_i - \mu)$: $P(\bar X_n - \mu \ge t) = P(S \ge nt) \le e^{-\lambda nt}\prod_i\mathbb{E}[e^{\lambda(X_i - \mu)}] \le \exp(-\lambda nt + n\lambda^2(b-a)^2/8)$ for every $\lambda > 0$. the exponent is a quadratic in $\lambda$, minimized at $\lambda = 4t/(b-a)^2$, where it equals $-2nt^2/(b-a)^2$. the lower tail is the same argument applied to $-X_i$, and the factor 2 comes from adding the two tails (a union bound).

</details>

3. **the Chernoff bound for coins.** for $S$ a sum of $n$ Bernoulli($p$), show $P(S \ge nq) \le \exp(-n\,\mathrm{KL}(q\,\|\,p))$ for $p < q < 1$. (minimize $e^{-\lambda q}(1 - p + pe^\lambda)$ over $\lambda > 0$.)

<details><summary>solution</summary>

the Chernoff method gives $P(S \ge nq) \le [e^{-\lambda q}(1 - p + pe^\lambda)]^n$. take logs, $g(\lambda) = -\lambda q + \log(1 - p + pe^\lambda)$, and set $g'(\lambda) = -q + \frac{pe^\lambda}{1 - p + pe^\lambda} = 0$, so $e^\lambda = \frac{q(1 - p)}{p(1 - q)}$ (which is $> 1$, so $\lambda > 0$, because $q > p$). then $1 - p + pe^\lambda = \frac{1-p}{1-q}$, and $g = -q\log\frac{q(1-p)}{p(1-q)} + \log\frac{1-p}{1-q} = -q\log\frac qp - (1 - q)\log\frac{1-q}{1-p} = -\mathrm{KL}(q\,\|\,p)$.

</details>

4. **the union bound (warm-up 5).** $k$ models each have true accuracy $\mu_j$ and are evaluated on the same $n$ test examples. show that with probability at least $1 - \delta$, *every* measured accuracy is within $t = \sqrt{\log(2k/\delta)/(2n)}$ of its true value. what is $t$ for $k = 1000$, $n = 1000$, $\delta = 0.05$, and what does it say about reporting the best of the 1000?

<details><summary>solution</summary>

for each model, Hoeffding gives $P(|\hat\mu_j - \mu_j| \ge t) \le 2e^{-2nt^2}$ (each measured accuracy is an average of $n$ independent 0/1 outcomes, even though the models share the test set). the probability that *some* model deviates is at most the sum, $2ke^{-2nt^2}$ (union bound: $P(\cup A_j) \le \sum P(A_j)$, no independence needed). setting it to $\delta$ gives $t$. for $k = n = 1000$: $t = \sqrt{\log(40{,}000)/2000} \approx 0.073$. so the best model's measured accuracy can be inflated by up to about 7 points by luck alone. picking the best of many on the test set turns the test set into a training set. this is the core of chapter 07's advice about validation sets, and the $\log k$ dependence is the seed of the generalization bounds in chapter 13.

</details>

## experiment

`lab.py` runs three experiments with your bounds:

1. every bound against the true tail of an average of coin flips, with knobs for $p$ and $n$.
2. sample-size calculators: how many test examples do you need, according to each bound and to the CLT?
3. the winner's curse: evaluate many equally good models on one test set and look at the best score.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. why does Markov's inequality need $X \ge 0$?

<details><summary>answer</summary>

without it, $\mathbb{1}[X \ge t] \le X/t$ fails for negative $X$, and the conclusion is false: a variable equal to $-100$ or $+100$ with equal probability has mean 0, but $P(X \ge 50) = 1/2$, not $\le 0$.

</details>

2. Chebyshev is tight. so why is it so loose for averages?

<details><summary>answer</summary>

it's tight over *all* distributions with a given variance. the distribution that achieves it puts its mass at three points. an average of many independent draws is never like that: by the CLT it's nearly Gaussian, with tails decaying like $e^{-t^2/2}$, far below $1/t^2$. Chebyshev ignores independence and the averaging structure; exponential bounds exploit it.

</details>

3. what does Hoeffding need that Chebyshev doesn't, and vice versa?

<details><summary>answer</summary>

Hoeffding needs bounded variables (in a known range $[a, b]$) and full independence, but not the variance. Chebyshev needs only a finite variance (known), and for averages only pairwise uncorrelated draws. a heavy-tailed but finite-variance variable gets Chebyshev but not Hoeffding.

</details>

4. you want to guarantee an accuracy estimate within $\pm 0.005$ with 95% confidence. how many test examples, by Hoeffding? what if you halve the tolerance?

<details><summary>answer</summary>

$n \ge \log(40)/(2 \cdot 0.005^2) \approx 73{,}778$. halving $t$ multiplies $n$ by 4: about 295,000. precise accuracy estimates are expensive, which is why small differences between models on standard test sets are often not meaningful.

</details>

5. when is Bernstein *worse* than Hoeffding?

<details><summary>answer</summary>

when the variance is near its maximum for the range, like a fair coin ($\sigma^2 = 0.25$, $c = 0.5$ in $[0, 1]$): Bernstein's denominator $2\sigma^2 + \tfrac23ct = 0.5 + t/3$ exceeds Hoeffding's effective $1/2$, so its exponent is slightly worse. Bernstein pays a little for its generality and wins big only when $\sigma^2 \ll (b - a)^2/4$.

</details>

6. Chernoff's KL bound is described as having the "right exponent". what does that mean, and what's still lost?

<details><summary>answer</summary>

$P(\bar X_n \ge p + t) = e^{-n\mathrm{KL}(p+t\|p) + o(n)}$: the bound's exponential rate in $n$ matches the truth exactly. what's lost is a polynomial factor (of order $1/\sqrt n$): the worked example's bound, 0.13, is about 5 times the truth, 0.028, and that ratio grows slowly with $n$.

</details>

7. the UCB bandit algorithm (ch 23) plays the arm with the largest $\hat\mu_j + \sqrt{2\log T/n_j}$. where does the square root come from?

<details><summary>answer</summary>

from inverting Hoeffding: with $n_j$ plays of an arm with rewards in $[0, 1]$, $\hat\mu_j + \sqrt{\log(1/\delta)/(2n_j)}$ is an upper confidence bound holding with probability $1 - \delta$. UCB chooses $\delta$ shrinking with time ($\delta \approx T^{-4}$ gives $\sqrt{2\log T/n_j}$), so every arm's optimism decays as it's tried, and untried arms stay attractive.

</details>

## common bugs

- **forgetting to clip at 1.** a "probability bound" of 3.7 is true but useless, and it breaks log-scale plots and sample-size searches.
- **dropping the factor 2** in two-sided Hoeffding or Bernstein. symptom: bounds that are violated by simulations at small $n$.
- **$(b - a)$ instead of $(b - a)^2$.** symptom: correct numbers for $[0, 1]$ data, wrong ones for any other range.
- **rounding the sample size to nearest instead of up.** symptom: a "guarantee" that's just barely violated at the $n$ you return.
- **$\log 0$ in the KL edge cases.** `q * np.log(q / p)` is `nan` at $q = 0$, and `(1 - q) * np.log(...)` at $q = 1$. use $0\log 0 = 0$ explicitly.
- **$>$ instead of $\ge$ in the empirical tail.** the bounds are for $P(\cdot \ge t)$. with discrete averages the boundary has positive probability, so the comparison can look wrong at exactly the thresholds you check.

## going deeper

- Boucheron, Lugosi & Massart (2013), *Concentration Inequalities: A Nonasymptotic Theory of Independence*, Oxford University Press, chapters 2–3: everything here and much more, with clean proofs. [doi:10.1093/acprof:oso/9780199535255.001.0001](https://doi.org/10.1093/acprof:oso/9780199535255.001.0001)
- Hoeffding (1963), "Probability inequalities for sums of bounded random variables", *Journal of the American Statistical Association* 58(301): 13–30. [doi:10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830)
- Chernoff (1952), "A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations", *The Annals of Mathematical Statistics* 23(4): 493–507. [doi:10.1214/aoms/1177729330](https://doi.org/10.1214/aoms/1177729330)
- Vershynin (2018), *High-Dimensional Probability*, Cambridge University Press, chapter 2: concentration of sums, sub-Gaussian and sub-exponential variables (the modern language for Hoeffding and Bernstein). free from the author's website.
- Mitzenmacher & Upfal (2017), *Probability and Computing*, 2nd ed., Cambridge University Press, chapter 4: Chernoff bounds, with applications to algorithms.

## connections

- ← builds on: expectation and variance (topic 03), the weak LLN and CLT (topic 05), exponentials and convexity (ch 01, ch 03).
- → used in: KL divergence and large deviations (ch 05), honest evaluation and test-set sizes (ch 07), PAC learning and generalization bounds, which are Hoeffding plus a union bound over hypotheses (ch 13), UCB and bandits (ch 23), and random projections and high-dimensional geometry (ch 11).
