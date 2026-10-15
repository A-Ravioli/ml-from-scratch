# 04 · probability · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. what's the difference between $P(A \mid B)$ and $P(B \mid A)$? give a numerical example where one is near 1 and the other near 0.

<details><summary>answer</summary>

$P(A \mid B)$ measures $A$ within the world where $B$ happened; $P(B \mid A)$ the reverse. Bayes' rule links them through the base rates: $P(A \mid B) = P(B \mid A)P(A)/P(B)$. with $A$ = "has the disease" (prevalence 0.1%) and $B$ = "tests positive" with a test that's 99% sensitive and 2% false-positive, $P(B \mid A) = 0.99$ but $P(A \mid B) \approx 0.047$ (topic 01).

</details>

2. in Monty Hall, why does it matter whether the host knows where the car is?

<details><summary>answer</summary>

evidence enters through the likelihood $P(\text{observation} \mid \text{hypothesis})$. an informed host's choice depends on the car's location (he must avoid it), so it carries information and switching wins 2/3. an ignorant host's choice doesn't depend on the car, so the only information is "the car isn't behind the opened door", and switching wins 1/2 (topic 01).

</details>

3. why does $X = F^{-1}(U)$ with $U$ uniform have CDF $F$? what goes wrong if `rng.random()` returns exactly 0?

<details><summary>answer</summary>

$P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$, because $F$ is increasing and $U$ is uniform. many inverse CDFs are infinite at 0 (e.g. $-\log u$ for the exponential), and `rng.random()` returns values in $[0, 1)$, so an exact 0 must be mapped into $(0, 1)$ first (topic 02).

</details>

4. rejection sampling with envelope $Mq \ge \tilde p$: what fraction of proposals is accepted, and why does this collapse in high dimensions?

<details><summary>answer</summary>

$Z/M$, where $Z = \int\tilde p$ ($1/M$ for a normalized target). in $d$ dimensions, a per-coordinate mismatch between proposal and target compounds multiplicatively, so $M$ grows exponentially in $d$: a normal proposal just 10% wider than a normal target needs $M = 1.1^d$ (topic 02).

</details>

5. why does $\mathbb{E}[X^2] - \mathbb{E}[X]^2$ fail in floating point, and what do two-pass and Welford's method do instead?

<details><summary>answer</summary>

when the mean is large compared with the spread, both terms are huge and nearly equal, and their difference (the variance) lives in digits that rounding has already destroyed; the relative error is about $\varepsilon\mu^2/\sigma^2$, and the result can even be negative. the stable methods only square *deviations from the mean*, which are small numbers (topic 03).

</details>

6. how do you merge the running statistics $(n_A, \bar x_A, M_{2,A})$ and $(n_B, \bar x_B, M_{2,B})$ of two batches?

<details><summary>answer</summary>

with $\delta = \bar x_B - \bar x_A$ and $n = n_A + n_B$: $\bar x = \bar x_A + \delta n_B/n$ and $M_2 = M_{2,A} + M_{2,B} + \delta^2 n_An_B/n$. the last term is the spread *between* the two groups (topic 03).

</details>

7. if $X$ has covariance $\Sigma$, what's the covariance of $AX + b$? why is every covariance matrix positive semidefinite?

<details><summary>answer</summary>

$A\Sigma A^\top$. with $A = v^\top$, $v^\top\Sigma v = \operatorname{Var}(v^\top X) \ge 0$ for every $v$ (topic 03).

</details>

8. how do you sample $\mathcal{N}(\mu, \Sigma)$ from standard normals, and how do you compute its log density without inverting $\Sigma$?

<details><summary>answer</summary>

with the Cholesky factor $\Sigma = LL^\top$: $x = \mu + Lz$ (for a batch $Z$ of shape $(n, d)$, $\mu + ZL^\top$). for the density, solve $Lw = x - \mu$ by forward substitution, so the Mahalanobis term is $\|w\|^2$, and $\log|\Sigma| = 2\sum_i\log L_{ii}$ (topic 04).

</details>

9. for a jointly Gaussian $(x_a, x_b)$, what's the distribution of $x_b$ given $x_a = v$? which part depends on $v$?

<details><summary>answer</summary>

Gaussian with mean $\mu_b + \Sigma_{ba}\Sigma_{aa}^{-1}(v - \mu_a)$ and covariance $\Sigma_{bb} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}$. only the mean depends on $v$; the conditional covariance doesn't (topic 04).

</details>

10. what's the difference between the weak law of large numbers, the strong law, and the central limit theorem? which fails for the Cauchy distribution?

<details><summary>answer</summary>

weak LLN: $P(|\bar X_n - \mu| > \varepsilon) \to 0$ (convergence in probability). strong LLN: $\bar X_n \to \mu$ along almost every path. CLT: $\sqrt n(\bar X_n - \mu)/\sigma \to \mathcal{N}(0, 1)$ in distribution. the Cauchy has no mean, so all three fail: the average of $n$ Cauchy draws is again standard Cauchy (topic 05).

</details>

11. what does the Berry–Esseen theorem add to the CLT, and why can the normal approximation still be badly off for $P(Z_n > 3)$ at $n = 30$?

<details><summary>answer</summary>

a rate: the CDF error is at most $C\rho/(\sigma^3\sqrt n)$ for every $n$. but that's an *absolute* error, of order 0.1 at $n = 30$, which says nothing about a probability of 0.001. in the tails the relative error converges much more slowly, with a skewness correction growing with the threshold: for exponential draws at $n = 30$, $P(Z > 3)$ is 3 times the normal value (topic 05).

</details>

12. state Hoeffding's inequality. how many samples guarantee an estimate of a probability within $\pm 0.05$ with confidence 95%?

<details><summary>answer</summary>

for independent draws in $[a, b]$, $P(|\bar X_n - \mu| \ge t) \le 2\exp(-2nt^2/(b-a)^2)$. setting it to $0.05$ with $t = 0.05$ on $[0, 1]$: $n \ge \log(40)/(2 \cdot 0.0025) \approx 738$ (topic 06).

</details>

13. when is Bernstein's inequality much better than Hoeffding's?

<details><summary>answer</summary>

when the variance is small compared with the squared range, e.g. rare events: for Bernoulli(0.01) Hoeffding acts as if the variance were 1/4, while Bernstein uses the true 0.0099. for 100 draws and $t = 0.1$, Hoeffding gives 0.27 and Bernstein $1.7 \times 10^{-5}$ (topic 06).

</details>

14. what's a stationary distribution of a Markov chain, how do you compute it, and when is it unique?

<details><summary>answer</summary>

a distribution with $\pi P = \pi$: a left eigenvector of $P$ for eigenvalue 1. solve $(P^\top - I)\pi = 0$ together with $\sum_i\pi_i = 1$ (for example by least squares on the stacked system). it's unique if the chain is irreducible; $p_t \to \pi$ from every start if it's also aperiodic (topic 07).

</details>

15. what does the spectral gap tell you about a Markov chain, and what does a gap of zero mean?

<details><summary>answer</summary>

$\gamma = 1 - |\lambda_2|$ sets the rate at which the chain forgets its start: the distance to stationarity decays like $|\lambda_2|^t$, so the mixing time is roughly $\log(1/\varepsilon)/\gamma$. a gap of 0 means another eigenvalue of modulus 1: the chain is reducible (a second eigenvalue 1) or periodic (e.g. $-1$), and $p_t$ doesn't converge from every start (topic 07).

</details>

## 2. by hand (no computer)

1. **Bayes, twice.** a factory has two machines. machine A makes 70% of the parts and 2% of its parts are defective; machine B makes 30% and 10% of its parts are defective. (a) what fraction of all parts is defective? (b) a part is defective: what's the probability it came from B? (c) a second, independent part from the *same* machine is also defective: now what's the probability the machine is B?

<details><summary>solution</summary>

(a) total probability: $0.7 \cdot 0.02 + 0.3 \cdot 0.1 = 0.014 + 0.03 = 0.044$. (b) $P(B \mid D) = 0.03/0.044 \approx 0.682$. (c) odds form: prior odds B:A $= 0.3:0.7$, each defect multiplies by the likelihood ratio $0.1/0.02 = 5$, so after two defects the odds are $0.3 \cdot 25 : 0.7 = 7.5 : 0.7$, and $P(B \mid DD) = 7.5/8.2 \approx 0.915$. (equivalently, update the posterior from (b) once more: $0.682 \cdot 5 : 0.318 = 3.41 : 0.318$.)

</details>

2. **a Gaussian, conditioned.** $x = (x_1, x_2) \sim \mathcal{N}\big((1, 0),\ \Sigma\big)$ with $\Sigma = \begin{pmatrix}9 & 3\\ 3 & 2\end{pmatrix}$. (a) find the Cholesky factor $L$ and use it to turn $z = (1, 1)$ into a sample. (b) find the distribution of $x_2$ given $x_1 = 4$. (c) compute $\log p(x)$ at $x = (1, 0)$.

<details><summary>solution</summary>

(a) $L_{11} = 3$, $L_{21} = 3/3 = 1$, $L_{22} = \sqrt{2 - 1} = 1$, so $L = \begin{pmatrix}3&0\\1&1\end{pmatrix}$ and $x = \mu + Lz = (1 + 3,\ 0 + 1 + 1) = (4, 2)$. (b) $\mu_{2|1} = 0 + \tfrac39(4 - 1) = 1$, $\Sigma_{2|1} = 2 - \tfrac{3^2}{9} = 1$, so $x_2 \mid x_1 = 4 \sim \mathcal{N}(1, 1)$. (c) at the mean the Mahalanobis term is 0, and $\log|\Sigma| = 2(\log 3 + \log 1) = \log 9$ (check: $\det\Sigma = 18 - 9 = 9$), so $\log p = -\tfrac12(\log 9 + 2\log 2\pi) = -\log 3 - \log 2\pi \approx -1.0986 - 1.8379 = -2.9365$.

</details>

3. **a chain and a bound.** (a) find the stationary distribution, the eigenvalues and the spectral gap of $P = \begin{pmatrix}0.7 & 0.3\\ 0.2 & 0.8\end{pmatrix}$. starting from state 0, after how many steps is the distribution within total variation $0.01$ of stationarity? (b) you flip a coin 400 times. bound $P(|\hat p - p| \ge 0.1)$ with Chebyshev (using $p(1-p) \le 1/4$) and with Hoeffding, and compare with the CLT approximation.

<details><summary>solution</summary>

(a) flow balance $0.3\pi_0 = 0.2\pi_1$ gives $\pi = (0.4, 0.6)$. eigenvalues 1 and $\operatorname{tr}P - 1 = 0.5$, gap $0.5$. from state 0, $p_t - \pi = (0.6, -0.6)\cdot 0.5^t$, so TV $= 0.6 \cdot 0.5^t$: $0.6 \cdot 0.5^5 = 0.019$, $0.6 \cdot 0.5^6 = 0.0094$, so $t = 6$. (b) Chebyshev: $\frac{1/4}{400 \cdot 0.01} = 0.0625$. Hoeffding: $2e^{-2 \cdot 400 \cdot 0.01} = 2e^{-8} \approx 6.7 \times 10^{-4}$. CLT: the standard error is at most $\sqrt{0.25/400} = 0.025$, so $t$ is 4 standard errors and $P \approx 2\Phi(-4) \approx 6 \times 10^{-5}$. Chebyshev is 1000 times looser than the truth, Hoeffding about 10 times.

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **distributions & sampling**, the core of `mlfs.sampling`. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=04-probability/02-distributions-and-sampling
mkdir -p /tmp/rebuild-0402
git show HEAD:$T/exercise.py > /tmp/rebuild-0402/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-0402/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-0402/solutions/` and run `python scripts/stubify.py /tmp/rebuild-0402`.) open `/tmp/rebuild-0402/exercise.py` with `marimo edit`, implement `uniform`, `inverse_cdf_sample`, `exponential`, `bernoulli`, `categorical`, `poisson`, `box_muller`, `normal` and `rejection_sample` from memory, using nothing but `rng.random()`, and run the checks against your copy:

```bash
pytest /tmp/rebuild-0402/test_exercise.py -k "stage1 or stage2 or stage3 or stage4"
```

you pass if every one of those checks is green within the hour.

**harder variant** (for testing out of the chapter with confidence): also rebuild Marsaglia–Tsang `gamma` and `beta` (stage 5 of the same topic) and `MultivariateNormal` (topic 04, in its own scratch directory), and pass these within 90 minutes in total:

```bash
pytest /tmp/rebuild-0402/test_exercise.py
pytest /tmp/rebuild-0404/test_exercise.py
```

(run them from the repo root: topic 04's notebook imports `mlfs.linalg` and `mlfs.sampling`, which the tests resolve to the reference solutions, so only the class you rebuilt is being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each part of a problem separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the topic, and the lessons are organized by the same ideas.
