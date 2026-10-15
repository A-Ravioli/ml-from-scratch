# limit theorems · lab notes

## experiment 1: do running averages settle?

**what people usually predict:** all three settle; the Cauchy, being "a bell curve centered at 0", maybe a bit more slowly.

**what happens:** the normal paths end within about 0.004 of 0, all six of them. the Pareto paths end near 3 but scattered (2.92 to 3.28 in one run), and their curves show visible jumps. the Cauchy paths end all over the place (from about −4.2 to +7.0 in one run) and never stop moving: long calm stretches, then a sudden leap, at every scale of $n$, including after 100,000 draws.

**why:** a running average jumps when a single draw is large compared with the sum so far, roughly when $|x_n| \gtrsim n$. for the Cauchy, $P(|X| > n) \approx 2/(\pi n)$, and $\sum_n 1/n$ diverges, so such draws keep arriving forever (on average, one per factor-of-$e$ increase in $n$). there's no mean for the average to converge to: on paper 3 shows the average of $n$ Cauchy draws is exactly as spread out as one draw. the Pareto with tail exponent 1.5 has a mean, so the strong LLN holds and every path converges to 3. but its variance is infinite, so the convergence is slower than $1/\sqrt n$ (the error shrinks like $n^{-1/3}$) and is driven by rare large draws: a ragged staircase rather than the normal's smooth funnel. heavy-tailed quantities like these are common in ML: importance sampling weights, returns in RL, the sizes of gradient spikes, and word frequencies all have Pareto-like tails.

## experiment 2: how fast does the CLT kick in?

**what people usually predict:** "about 30 draws for everything" (a rule of thumb many people learned), with the Cauchy perhaps a bit slower.

**what happens (KS distance to the normal at n = 1, 4, 16, 64, 256):**

| distribution | n=1 | n=4 | n=16 | n=64 | n=256 |
|---|---|---|---|---|---|
| uniform(0, 1) | 0.060 | 0.015 | 0.013 | 0.012 | 0.013 |
| exponential(1) | 0.159 | 0.071 | 0.041 | 0.037 | 0.015 |
| bernoulli(0.5) | 0.346 | 0.188 | 0.104 | 0.052 | 0.029 |
| bernoulli(0.01) | 0.529 | 0.537 | 0.506 | 0.332 | 0.176 |
| lognormal(0, 1) | 0.228 | 0.120 | 0.083 | 0.044 | 0.038 |
| cauchy | 0.136 | 0.234 | 0.341 | 0.402 | 0.454 |

(distances around 0.013 are the noise floor of 4,000 repetitions.) the uniform is indistinguishable from normal by $n = 4$. the mean of 64 bernoulli(0.01) draws is still far off (0.33): its histogram is a few spikes, mostly at "zero successes". the Cauchy gets *worse* with $n$. every Berry–Esseen bound holds, typically 3 to 10 times above the measured distance.

**why:** the speed is governed by $\rho/\sigma^3$, which measures skewness and tail weight: $1/32 \div (1/12)^{3/2} \approx 1.3$ for the uniform, 2.4 for the exponential, 1 for a fair coin, and 9.9 for bernoulli(0.01). the fair coin converges slowly for a different reason: it's discrete, and the KS distance between any lattice distribution and a continuous one is at least about half the largest jump, which shrinks only like $0.4/\sqrt n$. bernoulli(0.01) combines both problems: with $n = 64$, the expected number of successes is 0.64, so the "average" is mostly 0 and occasionally 1/64 or 2/64. "n ≥ 30" is a rule of thumb for mildly skewed data, nothing more: for rare events, the relevant count is the expected number of successes, $np$, not $n$. the Cauchy has no $\sigma$. the mean of $n$ draws stays standard Cauchy, so $\sqrt n\,\bar X_n$ is a Cauchy with scale $\sqrt n$, spreading out without limit, and its KS distance to $\mathcal{N}(0, 1)$ climbs towards 1.

## experiment 3: the CLT in the tails

**what people usually predict:** with $n = 30$, the normal approximation is "good", so both tails should be near 0.00135, within maybe 10–20%.

**what happens:** at $n = 30$ the true $P(Z > 3)$ is 0.0042, **3.1 times** the normal value, while $P(Z < -3)$ is 0.00008, **17 times smaller**. even at $n = 1000$ the upper tail is 29% too heavy and the lower one 26% too light. both tails are within about 10% only around $n = 10{,}000$. the simulation agrees with the exact values.

**why:** Berry–Esseen controls the *absolute* error of the CDF, which at $n = 30$ is up to about 0.2 (the actual worst-case gap is closer to 0.03). an absolute error of a few hundredths says nothing about a probability of 0.001, where what matters is the *relative* error. in the tails, the CLT's convergence is much slower. the leading correction to the normal approximation is a skewness term (the Edgeworth expansion) of size $\gamma/(6\sqrt n)\cdot(t^2 - 1)\varphi(t)$, with skewness $\gamma = 2$ for the exponential. it grows with $t$, so the further into the tail you look, the bigger $n$ must be. the exponential is right-skewed (its sums have a long right tail and a hard floor at 0), so the right tail of $Z_n$ is heavier than normal and the left lighter. the practical lesson: be very skeptical of tiny p-values and "six sigma" claims computed from a normal approximation to a skewed statistic. the bulk of the distribution converges fast; the tails, where those claims live, converge last. concentration inequalities (topic 06) give guarantees in the tails that hold for every $n$.
