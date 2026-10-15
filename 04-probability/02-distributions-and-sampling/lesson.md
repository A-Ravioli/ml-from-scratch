# distributions & sampling

## why this matters

your computer can't flip a coin. what it can do is produce a long, deterministic stream of bits that passes every statistical test anyone has thought of, and turn 53 of them into a number $u$ that looks uniformly distributed on $[0, 1)$. that is the *only* randomness you get. every other random thing in machine learning is built on top of it: the Gaussian initialization of a network's weights, the shuffling of minibatches, dropout masks, the noise in a diffusion model, the next token a language model samples, the data points a Monte Carlo estimate averages over.

so "sample from a distribution" really means "find a function that turns uniform numbers into numbers with the distribution you want". sometimes that function is a one-liner. sometimes it has a loop that throws proposals away. if you don't know how these functions work, you can't sample from a distribution you invented, you can't tell a correct sampler from a subtly broken one, and the reparameterization trick that makes variational autoencoders trainable (ch 21) will look like magic instead of like `normal()` from this topic.

in this topic you'll build `mlfs.sampling`, a small sampling library, with one strict rule: **the only source of randomness is `rng.random()`**. you'll derive exponentials from uniforms, coins and dice from uniforms, Gaussians from pairs of uniforms, and finally samplers that work for almost any density.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. $U$ is uniform on $(0, 1)$. what is the distribution of $-\ln U$? (guess its shape: where is it concentrated, how far does it reach?)
2. how would you use one uniform number to simulate a biased coin with $P(\text{heads}) = 0.3$? a loaded die with probabilities $(0.2, 0.5, 0.3)$?
3. the normal distribution's CDF $\Phi$ has no closed-form inverse. so how does anyone turn uniforms into normals exactly?
4. a sampler proposes points and accepts each with some probability. if the "envelope" is 10 times the target density's total mass, roughly what fraction of proposals survive?
5. you can sample Gamma distributions. how could you get a Beta, which lives on $(0, 1)$, out of them?

## prerequisites check

- [ ] you know what a pmf is and how conditioning works (→ ch 04 · topic 01)
- [ ] you can describe a CDF $F(x) = P(X \le x)$ and a density $f = F'$ (→ ch 01 · topic 05)
- [ ] you can change variables in a double integral, Jacobian determinant included (→ ch 03 · topic 06)
- [ ] you can work with `np.cumsum`, `np.searchsorted`, boolean masks, and `size=` arguments (→ ch 00 · topic 02)

## core ideas

### idea 1: the inverse-CDF method

here's the most useful fact in this topic. let $F$ be a continuous, strictly increasing CDF and $U \sim \text{Uniform}(0, 1)$. then

$$X = F^{-1}(U) \quad \text{has CDF } F.$$

the proof is one line: $P(X \le x) = P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$, because $F$ is increasing and $P(U \le t) = t$. picture it: the CDF's graph squeezes the uniform vertical axis into the horizontal axis, and steep parts of $F$ (where the density is high) catch many uniforms in a narrow range of $x$.

**exponential.** $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$. solving $u = F(x)$ gives

$$F^{-1}(u) = -\frac{\ln(1 - u)}{\lambda}.$$

and since $1 - U$ is also uniform, $-\ln(U)/\lambda$ works just as well. that answers warm-up 1: $-\ln U$ is Exponential(1), piled up near 0 with a tail that reaches arbitrarily far (the largest of a million draws is about $\ln 10^6 \approx 14$). the exponential is the distribution of waiting times for events that happen at a constant rate $\lambda$ with no memory: the time until the next radioactive decay, the next customer, the next request to a server. its mean is $1/\lambda$.

**the floating-point catch.** `rng.random()` returns numbers of the form $k/2^{53}$ for $k = 0, 1, \dots, 2^{53} - 1$. that's $[0, 1)$, so $0$ is possible (once in $9 \times 10^{15}$ draws, which is rare, but you'll make billions of draws over this course). $-\ln 0 = \infty$, and so are lots of inverse CDFs at 0 or 1. your `inverse_cdf_sample` guarantees `inv_cdf` only ever sees the open interval $(0, 1)$, by replacing an exact zero with $2^{-54}$. (don't add $2^{-54}$ to *everything*: the top value $1 - 2^{-53}$ would round up to exactly 1.)

**the rule.** every function in this notebook takes `rng=None`, an int seed, or a `np.random.Generator`, and starts with `rng = np.random.default_rng(rng)`. a Generator passes through unchanged, so two calls with the same Generator continue one stream (different draws), while two calls with the same int seed repeat the same draws. that's the course-wide convention (and it's how you make experiments reproducible without any global state).

> **▶ build: stage 1.** implement `uniform(low, high, size, *, rng)`, `inverse_cdf_sample(inv_cdf, size, *, rng)` and `exponential(rate, size, *, rng)` in `exercise.py` (`marimo edit exercise.py`). `size=None` returns a single Python float; an int or a tuple returns an array of that shape.

### idea 2: discrete distributions

the inverse-CDF idea works for discrete variables too, with a staircase instead of a smooth curve. for a variable taking values $0, 1, \dots, k-1$ with probabilities $p_0, \dots, p_{k-1}$, cut $[0, 1)$ into consecutive intervals of lengths $p_0, p_1, \dots$, and return the index of the interval $U$ lands in. the interval boundaries are the cumulative sums $c_i = p_0 + \cdots + p_i$, so the answer is "the first $i$ with $U < c_i$", which is exactly what `np.searchsorted(c, u, side="right")` computes for a whole array of $u$ at once.

that's warm-up 2. a biased coin (a **Bernoulli**(p) variable) is the case $k = 2$: return 1 if $u < p$, else 0. a loaded die is a **categorical** distribution: with $c = (0.2, 0.7, 1.0)$, a draw of $u = 0.65$ lands in the second interval and returns index 1.

two details matter. a category with $p_i = 0$ has an interval of length zero, so it must never be returned: with `side="right"`, $u = c_{i-1} = c_i$ moves past it. and `np.cumsum` rounds: `np.cumsum([0.1] * 10)[-1]` is `0.9999999999999999`, so a $u$ above that falls off the end. scale $u$ by $c_{k-1}$ (or clip the index) and the problem disappears.

**Poisson.** the number of events in a fixed window, when events happen independently at a constant rate, is Poisson: $P(N = n) = e^{-\lambda}\lambda^n/n!$, with mean and variance both $\lambda$. it counts typos per page, photons per pixel, clicks per minute. there's a beautiful exact sampler built from idea 1. events at rate 1 arrive after Exponential(1) gaps $E_i = -\ln U_i$, and $N$ is the number of arrivals before time $\lambda$:

$$N = \max\{n : E_1 + \cdots + E_n < \lambda\} = \max\{n : U_1 U_2 \cdots U_n > e^{-\lambda}\}.$$

so multiply uniforms together until the product drops to $e^{-\lambda}$ or below, and count the factors *before* that happens. that's **Knuth's method**. it costs about $\lambda + 1$ uniforms per draw, which is fine for small $\lambda$ but wasteful for $\lambda = 1000$. for large $\lambda$ you'll use the discrete inverse CDF instead: tabulate the pmf for $k = 0, \dots, K$ with $K$ far enough into the tail that the rest has probability below $10^{-30}$, and hand the table to your `categorical`.

> **🔮 predict:** how many calls' worth of uniforms does Knuth's method use on average for one Poisson(50) draw? (the lab counts.)

> **▶ build: stage 2.** implement `bernoulli(p, size, *, rng)`, `categorical(probs, size, *, rng)` and `poisson(lam, size, *, rng)`. these return integers (a Python int for `size=None`).

### idea 3: Box–Muller, normals from pairs of uniforms

the inverse-CDF method needs $F^{-1}$, and the normal distribution's $\Phi^{-1}$ has no closed form. you could approximate it numerically (libraries do), but there's a cleaner trick, and it's a good example of a general strategy: when one variable is hard, look at two.

take two independent standard normals $Z_1, Z_2$. their joint density is

$$\frac{1}{2\pi}e^{-(z_1^2 + z_2^2)/2},$$

which depends only on the distance from the origin. switch to polar coordinates $(r, \theta)$. the density is rotationally symmetric, so $\theta$ is uniform on $[0, 2\pi)$. and the squared radius $R^2 = Z_1^2 + Z_2^2$ turns out to be Exponential with rate $1/2$ (you'll show this on paper). both of those are easy with idea 1:

$$R = \sqrt{-2\ln U_1}, \qquad \theta = 2\pi U_2, \qquad Z_1 = R\cos\theta, \quad Z_2 = R\sin\theta.$$

that's the **Box–Muller transform** (Box & Muller, 1958), and it answers warm-up 3: you get *two* exact, independent normals from two uniforms. use both. and $\ln U_1$ needs $U_1 > 0$: since `rng.random()` can return 0 but never 1, take $U_1 = 1 - $ `rng.random()`.

for a general normal with mean $\mu$ and standard deviation $\sigma$, scale and shift: $X = \mu + \sigma Z$. (in `normal(loc, scale)`, `scale` is the standard deviation, not the variance, as in NumPy.)

> **🔮 predict:** $Z_1$ and $Z_2$ share the same radius $R$. are they really independent, or is a large $|Z_1|$ more likely to come with a large $|Z_2|$?

> **▶ build: stage 3.** implement `box_muller(u1, u2)` (elementwise on arrays) and `normal(loc, scale, size, *, rng)`, which uses both outputs of each pair: $n$ draws need $\lceil n/2 \rceil$ pairs.

### idea 4: rejection sampling

for most densities you'll meet, you have neither a closed-form $F^{-1}$ nor a clever trick. often you don't even know the normalizing constant: you can compute $\tilde p(x) = Z\,p(x)$ but not $Z$. (that's the normal situation for posterior distributions in chapter 12.) **rejection sampling** works anyway.

pick a **proposal** distribution $q$ that you *can* sample, and a constant $M$ with

$$\tilde p(x) \le M\,q(x) \quad \text{for every } x.$$

the graph of $Mq$ is an envelope over the graph of $\tilde p$. now:

1. draw $x \sim q$ and $u \sim \text{Uniform}(0, 1)$;
2. accept $x$ if $u < \tilde p(x)/(M q(x))$, otherwise throw it away and try again.

why it works: the pair $(x, u\,Mq(x))$ is a uniformly random point under the envelope's graph. accepting exactly when that point also lies under $\tilde p$'s graph leaves you with a uniform point under $\tilde p$, and the $x$-coordinate of a uniform point under a density's graph has that density. (on paper 3 does this with integrals.) the normalizing constant never appears, because only the ratio $\tilde p/(Mq)$ is used.

the price is the rejected proposals. the probability of accepting is

$$P(\text{accept}) = \int q(x)\,\frac{\tilde p(x)}{M q(x)}\,dx = \frac{Z}{M},$$

so with a normalized target ($Z = 1$) you accept a fraction $1/M$ of proposals. that's warm-up 4: an envelope 10 times too big wastes 90% of the work. a good proposal hugs the target.

work in logs: $\tilde p$ and $q$ are often tiny, and log-densities are what you'll have anyway. accept when $\ln u < \ln\tilde p(x) - \ln M - \ln q(x)$. and check the envelope: if you ever see $\ln\tilde p(x) > \ln M + \ln q(x)$, the sampler is producing the wrong distribution (it under-samples the region where the envelope is too low), so raise an error instead of returning garbage.

> **🔮 predict:** you sample a 100-dimensional standard normal by rejection, using a normal proposal that is just 10% wider in every direction. what fraction of proposals is accepted? (the lab finds out.)

> **▶ build: stage 4.** implement `rejection_sample(log_target, sample_proposal, log_proposal, log_M, size, *, rng)`. propose in batches, accept with one uniform each, and return exactly `size` accepted draws.

### idea 5: Gamma and Beta

the **Gamma**(shape $k$, scale $\theta$) distribution has density $\propto x^{k-1}e^{-x/\theta}$ on $x > 0$, mean $k\theta$ and variance $k\theta^2$. for integer $k$ it's the waiting time until the $k$-th event of a rate-$1/\theta$ process, a sum of $k$ exponentials. but $k$ can be any positive number, and Gammas show up everywhere: as priors for rates and precisions, as the distribution of a variance estimate (the chi-square is a Gamma), and as the building block for Betas and Dirichlets.

summing exponentials only works for integer shapes and costs $k$ uniforms. **Marsaglia–Tsang** (2000) is a rejection sampler that's fast for every shape $k \ge 1$. set $d = k - 1/3$ and $c = 1/\sqrt{9d}$, and propose

$$v = (1 + cz)^3 \quad \text{with } z \sim \mathcal{N}(0, 1).$$

a cube of a shifted, scaled normal is already very close to a Gamma(k)/d variable (it's the Wilson–Hilferty approximation, run backwards). reject if $v \le 0$, draw $u$ uniform, and accept $dv$ if

$$\ln u < \tfrac12 z^2 + d - dv + d\ln v.$$

acceptance rates are above 95% for every $k \ge 1$, so on average each draw costs about one normal and one uniform. for $k < 1$, use a "boost": if $g \sim \text{Gamma}(k + 1)$ and $u$ is uniform, then $g\,u^{1/k} \sim \text{Gamma}(k)$. multiply by $\theta$ at the end.

the **Beta**(a, b) distribution lives on $(0, 1)$, with density $\propto x^{a-1}(1-x)^{b-1}$ and mean $a/(a+b)$. it's the standard distribution for an unknown probability: chapter 05 uses it as the prior (and posterior) for a coin's bias. warm-up 5: if $X \sim \text{Gamma}(a)$ and $Y \sim \text{Gamma}(b)$ are independent (same scale), then

$$\frac{X}{X + Y} \sim \text{Beta}(a, b).$$

that's one line on top of your `gamma`.

here's a summary of what your library now covers:

| family | parameters | mean | variance | models |
|---|---|---|---|---|
| Uniform | low $a$, high $b$ | $(a+b)/2$ | $(b-a)^2/12$ | "no idea, within limits" |
| Exponential | rate $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | memoryless waiting times |
| Bernoulli | $p$ | $p$ | $p(1-p)$ | one yes/no outcome |
| Categorical | $p_0, \dots, p_{k-1}$ | | | one of $k$ classes, the next token |
| Poisson | $\lambda$ | $\lambda$ | $\lambda$ | counts of rare independent events |
| Normal | $\mu$, $\sigma$ | $\mu$ | $\sigma^2$ | sums of many small effects (topic 05) |
| Gamma | shape $k$, scale $\theta$ | $k\theta$ | $k\theta^2$ | waiting for $k$ events; positive quantities |
| Beta | $a$, $b$ | $\frac{a}{a+b}$ | $\frac{ab}{(a+b)^2(a+b+1)}$ | unknown probabilities |

> **▶ build: stage 5.** implement `gamma(shape, scale, size, *, rng)` with Marsaglia–Tsang (using *your* `normal` and `uniform`) plus the boost for shape < 1, and `beta(a, b, size, *, rng)` from two of your Gammas.

## worked example

**exponential.** rate $\lambda = 2$, $u = 0.3$: $x = -\ln(1 - 0.3)/2 = -\ln(0.7)/2 \approx 0.1783$.

**categorical.** probabilities $(0.2, 0.5, 0.3)$ have cumulative sums $(0.2, 0.7, 1.0)$. $u = 0.65$ is at least $0.2$ but below $0.7$, so the draw is index 1. $u = 0.7$ exactly is *not* below $0.7$, so it gives index 2 (`side="right"`).

**Poisson by Knuth.** $\lambda = 1$, so the threshold is $e^{-1} \approx 0.3679$. uniforms $0.6, 0.5, \dots$: after one factor the product is $0.6 > 0.3679$ (count 1). after two it's $0.3 \le 0.3679$: stop. the draw is $N = 1$.

**Box–Muller.** $u_1 = 0.25$, $u_2 = 0.125$:

$$r = \sqrt{-2\ln 0.25} = \sqrt{2.7726} \approx 1.6651, \qquad \theta = 2\pi \cdot 0.125 = \pi/4,$$

$$z_1 = r\cos(\pi/4) \approx 1.1774, \qquad z_2 = r\sin(\pi/4) \approx 1.1774.$$

**Marsaglia–Tsang.** shape $k = 3$: $d = 8/3 \approx 2.6667$, $c = 1/\sqrt{24} \approx 0.2041$. with $z = 0.5$: $v = (1 + 0.1021)^3 \approx 1.3385 > 0$. the right side of the test is $0.125 + 2.6667 - 2.6667 \cdot 1.3385 + 2.6667 \ln 1.3385 \approx -0.0002$, and $u = 0.5$ gives $\ln u \approx -0.693 < -0.0002$, so accept: the draw is $dv \approx 3.569$.

**the rejection envelope used in the tests.** the Beta(2, 5) density is $30x(1-x)^4$. its maximum is at $x = 1/5$, where it equals $30 \cdot 0.2 \cdot 0.8^4 \approx 2.4576$. with a Uniform(0, 1) proposal ($q = 1$), any $M \ge 2.4576$ is a valid envelope, and $M = 2.5$ accepts $1/2.5 = 40\%$ of proposals.

the tests check the Box–Muller numbers exactly, and every sampler against its exact distribution.

## on paper

1. **the inverse-CDF method, discrete case.** let $X$ take values $0, \dots, k-1$ with cumulative sums $c_i$ (and $c_{-1} = 0$). show that "the smallest $i$ with $U < c_i$" has $P(\text{index} = i) = p_i$.

<details><summary>solution</summary>

the index is $i$ exactly when $c_{i-1} \le U < c_i$ (all earlier boundaries are $\le U$, and $c_i$ is the first one above it). $U$ is uniform, so this has probability $c_i - c_{i-1} = p_i$. a zero-probability category has $c_{i-1} = c_i$, an empty interval, so it's never returned.

</details>

2. **Box–Muller.** let $Z_1, Z_2$ be independent standard normals and $S = Z_1^2 + Z_2^2$. using polar coordinates, show that $P(S > s) = e^{-s/2}$, so $S \sim$ Exponential(rate $1/2$), and that the angle is uniform and independent of $S$. conclude that $\sqrt{-2\ln U_1}$ has the distribution of $R = \sqrt{S}$.

<details><summary>solution</summary>

$P(S > s) = \iint_{z_1^2 + z_2^2 > s}\frac{1}{2\pi}e^{-(z_1^2+z_2^2)/2}\,dz_1\,dz_2$. in polar coordinates ($dz_1\,dz_2 = r\,dr\,d\theta$) this is $\int_0^{2\pi}\frac{d\theta}{2\pi}\int_{\sqrt s}^\infty r e^{-r^2/2}\,dr = 1 \cdot e^{-s/2}$. the integrand factors into a function of $\theta$ (constant) times a function of $r$, so $\theta$ is uniform on $[0, 2\pi)$ and independent of $r$. for the sampler: $P(-2\ln U_1 > s) = P(U_1 < e^{-s/2}) = e^{-s/2}$, the same tail, so $-2\ln U_1$ has the distribution of $S$ and its square root has the distribution of $R$.

</details>

3. **rejection sampling.** with target $p = \tilde p/Z$, proposal $q$ and $\tilde p \le Mq$, show that $P(\text{accept}) = Z/M$ and that the density of an accepted $x$ is $p(x)$.

<details><summary>solution</summary>

$P(x \in dx,\ \text{accept}) = q(x)\,dx \cdot \frac{\tilde p(x)}{Mq(x)} = \frac{\tilde p(x)}{M}dx$. integrating over $x$: $P(\text{accept}) = \frac{1}{M}\int\tilde p = Z/M$. dividing, $P(x \in dx \mid \text{accept}) = \frac{\tilde p(x)/M}{Z/M}dx = p(x)\,dx$. the number of proposals until the first acceptance is geometric with mean $M/Z$.

</details>

4. **the boost.** show that if $g \sim \text{Gamma}(k + 1)$ and $u \sim \text{Uniform}(0, 1)$ are independent, then $P(g\,u^{1/k} \le x)$ has density $\propto x^{k-1}e^{-x}$. (hint: condition on $g$; $P(g u^{1/k} \le x \mid g) = \min(1, (x/g)^k)$.)

<details><summary>solution</summary>

$P(gu^{1/k} \le x) = \int_0^\infty \min(1, (x/g)^k)\frac{g^k e^{-g}}{\Gamma(k+1)}dg = \int_0^x \frac{g^ke^{-g}}{\Gamma(k+1)}dg + \frac{x^k}{\Gamma(k+1)}\int_x^\infty e^{-g}\,dg$. differentiate in $x$: $\frac{x^ke^{-x}}{\Gamma(k+1)} + \frac{kx^{k-1}e^{-x}}{\Gamma(k+1)} - \frac{x^ke^{-x}}{\Gamma(k+1)} = \frac{x^{k-1}e^{-x}}{\Gamma(k)}$, the Gamma(k) density.

</details>

## experiment

`lab.py` runs three experiments on your samplers:

1. QQ plots of every sampler against NumPy's own, including one deliberately broken sampler so you know what a bug looks like.
2. rejection sampling as the dimension grows.
3. the cost of exactness: how many uniforms each of your samplers burns per draw.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. why does `inverse_cdf_sample` have to avoid $u = 0$ but `categorical` doesn't?

<details><summary>answer</summary>

many continuous inverse CDFs blow up at 0 or 1 ($-\ln 0$, $\Phi^{-1}(0)$). the discrete inverse CDF is a lookup, defined for every $u \in [0, 1)$: $u = 0$ just returns the first category with positive probability.

</details>

2. Box–Muller makes two normals from two uniforms. why is throwing away $z_2$ a waste but not an error?

<details><summary>answer</summary>

$z_1$ alone is a perfectly good standard normal, so discarding $z_2$ gives correct draws at twice the cost. $z_1$ and $z_2$ are independent (on paper 2), so keeping both is also correct.

</details>

3. your rejection sampler accepts 2% of proposals. name two ways to improve it.

<details><summary>answer</summary>

lower $M$ to the true maximum of $\tilde p/q$ (if your bound is loose), or pick a proposal shaped more like the target (similar location, similar width, tails at least as heavy). the acceptance rate is $Z/M$, so both shrink the wasted area between envelope and target.

</details>

4. why must the proposal's tails be at least as heavy as the target's?

<details><summary>answer</summary>

if $\tilde p/q \to \infty$ in the tails, no finite $M$ satisfies $\tilde p \le Mq$. a normal proposal can't cover a Cauchy target, for example. the envelope check in your code catches the symptom when a proposal lands in the uncovered region.

</details>

5. why does Knuth's method get slow for large $\lambda$, and why does a table fix it?

<details><summary>answer</summary>

it simulates every arrival, about $\lambda + 1$ uniforms per draw. the table does one uniform and one binary search per draw, at the cost of building a table of size $O(\lambda + \sqrt\lambda)$ once.

</details>

6. `normal(loc=1, scale=3)` returns draws with variance 9, not 3. why is that the convention?

<details><summary>answer</summary>

scale parameters multiply the variable: $X = \mu + \sigma Z$. the standard deviation scales like the variable; the variance scales like its square. NumPy, SciPy and PyTorch all use the standard deviation for this argument.

</details>

7. Marsaglia–Tsang is a rejection sampler. what's the proposal, and why is its acceptance rate so high?

<details><summary>answer</summary>

the proposal is the distribution of $dv = d(1 + cz)^3$ with $z$ normal. by the Wilson–Hilferty approximation, the cube root of a Gamma variable is nearly normal, so the proposal is already almost exactly a Gamma(k) distribution. little area lies between envelope and target, so few proposals are rejected.

</details>

## common bugs

- **dropping the 2 in Box–Muller.** `np.sqrt(-np.log(u1))` gives normals with standard deviation $1/\sqrt2 \approx 0.707$. symptom: a QQ line with the wrong slope.
- **`u = rng.random()` straight into a log.** works for $9 \times 10^{15}$ draws, then returns `inf` once and poisons an average. use $1 - u$ or replace exact zeros.
- **an off-by-one in Knuth.** counting the factor that pushed the product *below* the threshold gives a mean of $\lambda + 1$.
- **`np.cumsum(probs)` not ending at 1.** `searchsorted` then returns $k$ (out of range) for $u$ in the gap. symptom: an `IndexError` once in a long while.
- **re-seeding inside a function.** `np.random.default_rng(0)` inside your sampler makes every call return the same draws. symptom: two calls with the same Generator are identical.
- **accepting with $\tilde p/q$ instead of $\tilde p/(Mq)$.** symptom: wrong distribution, or "acceptance probabilities" above 1. your envelope check is there to catch the second.

## going deeper

- Devroye (1986), *Non-Uniform Random Variate Generation*, Springer. the encyclopedia of exact samplers, free online from the author at [luc.devroye.org/rnbookindex.html](http://luc.devroye.org/rnbookindex.html). chapter II covers inversion and rejection.
- Box & Muller (1958), "A note on the generation of random normal deviates", *The Annals of Mathematical Statistics* 29(2): 610–611. [doi:10.1214/aoms/1177706645](https://doi.org/10.1214/aoms/1177706645)
- Marsaglia & Tsang (2000), "A simple method for generating gamma variables", *ACM Transactions on Mathematical Software* 26(3): 363–372. [doi:10.1145/358407.358414](https://doi.org/10.1145/358407.358414)
- Knuth (1997), *The Art of Computer Programming, vol. 2: Seminumerical Algorithms*, 3rd ed., Addison-Wesley, §3.4.1: the Poisson method used here, and much more on generating random numbers.
- O'Neill (2014), "PCG: a family of simple fast space-efficient statistically good algorithms for random number generation", Harvey Mudd College technical report HMC-CS-2014-0905. the bit generator behind `np.random.default_rng`.
- Robert & Casella (2004), *Monte Carlo Statistical Methods*, 2nd ed., Springer, chapter 2: inversion and accept–reject, with proofs.

## connections

- ← builds on: pmfs and conditioning (topic 01), CDFs and densities (ch 01), change of variables (ch 03 · topic 06), vectorized NumPy (ch 00).
- → used in: running moments of your samples (topic 03), Gaussian sampling through Cholesky (topic 04), the CLT simulator (topic 05), Markov chain simulation (topic 07), the capstone's text generator (`categorical`), weight initialization (ch 14), the reparameterization trick (ch 21), MCMC, which replaces rejection when the dimension is high (ch 12), and token sampling in language models (ch 17–18).
