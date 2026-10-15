# maximum likelihood & MAP · lab notes

## experiment 1: how much is a prior worth?

**what people usually predict:** a wrong prior always hurts; the MLE, being "objective", is safer. or the opposite: priors help a little, everywhere.

**what happens (true p = 0.3, RMSE over 3,000 coins):**

| prior strength m | n = 1 | n = 3 | n = 10 | n = 30 | n = 100 | n = 300 | n = 1000 |
|---|---|---|---|---|---|---|---|
| 0 (MLE) | 0.457 | 0.260 | 0.142 | 0.083 | 0.046 | 0.026 | 0.015 |
| 2 | 0.202 | 0.175 | 0.122 | 0.079 | 0.045 | 0.026 | 0.015 |
| 10 | 0.186 | 0.165 | 0.120 | 0.081 | 0.045 | 0.026 | 0.015 |
| 50 | 0.196 | 0.189 | 0.167 | 0.129 | 0.073 | 0.036 | 0.017 |
| 200 | 0.199 | 0.197 | 0.190 | 0.174 | 0.134 | 0.081 | 0.036 |

a prior centered at the *wrong* value 0.5 beats the MLE for small $n$: with $m = 10$ it has less than half the MLE's error at $n = 1$ and still wins at $n = 10$ (0.120 against 0.142). the weak priors ($m = 2, 10$) converge to the MLE by $n \approx 30$ and never hurt noticeably. the strong ones ($m = 50, 200$) cost a lot for hundreds of flips: at $n = 100$ the $m = 200$ prior's error is three times the MLE's.

**why:** the posterior mean is $\frac{m/2 + k}{m + n} = w \cdot 0.5 + (1 - w)\hat p$ with $w = \frac{m}{m + n}$. its MSE is $w^2(0.5 - p)^2 + (1 - w)^2\frac{p(1 - p)}{n}$: squared bias from the prior's error, plus the MLE's variance shrunk by $(1 - w)^2$. when $n$ is small, the MLE's variance $p(1 - p)/n$ is huge (at $n = 1$ the MLE is 0 or 1), so trading it for a bounded bias is a bargain, even toward a wrong center. comparing the two MSEs exactly, the prior wins whenever $n\,\big(m(0.5 - p)^2 - 2p(1 - p)\big) < m\,p(1 - p)$. two regimes follow. if $m(0.5 - p)^2 \le 2p(1 - p)$, which here means $m \le 10.5$, the prior helps at *every* $n$: its bias shrinks like $m/n$, fast enough that it never costs more than the variance it removes (that's why the $m = 2$ and $m = 10$ rows never rise above the MLE). a stronger prior helps only while $n < \frac{m\,p(1-p)}{m(0.5 - p)^2 - 2p(1-p)}$: about 6.6 flips for $m = 50$ and 5.5 for $m = 200$, matching the crossovers in the table. after that it hurts, for a long time, because its weight $w$ only decays like $m/n$. at $p = 0.5$ every prior helps at every $n$ (no bias at all); at $p = 0.05$ the wrong prior's bias is large and the strong priors hurt badly. a prior is worth $m$ observations, and it's worth having when your guess is closer to the truth than $m$ coin flips would get you.

## experiment 2: can maximum likelihood be infinitely wrong?

**what people usually predict:** the MLE is a bit worse than Laplace's rule at small $n$ and the same at large $n$; nothing dramatic.

**what happens ($p = 0.05$; the best possible expected log loss is the entropy, 0.1985 nats):**

| n | fraction with q̂ = 0 | exact 0.95ⁿ | MLE mean log loss | MLE, finite cases only | Laplace |
|---|---|---|---|---|---|
| 5 | 0.769 | 0.774 | inf | 0.320 | 0.281 |
| 20 | 0.361 | 0.358 | inf | 0.211 | 0.217 |
| 50 | 0.077 | 0.077 | inf | 0.206 | 0.207 |
| 100 | 0.009 | 0.006 | inf | 0.204 | 0.203 |
| 200 | 0.000 | 0.000 | 0.201 | 0.201 | 0.201 |
| 1000 | 0.000 | 0.000 | 0.199 | 0.199 | 0.199 |

the MLE predicts probability 0 for the rare event in 36% of datasets at $n = 20$, and each of those costs infinite log loss the first time the event happens. Laplace's rule is always finite and, at $n = 5$, much better even than the MLE's *finite* cases.

**why:** whenever all $n$ observations are 0 (probability $0.95^n > 0$ for every $n$), the MLE is exactly 0, and $-p\log 0 = \infty$. any positive probability of an infinite loss makes the expected loss infinite, for every $n$. the simulation at $n = 200$ and $1000$ shows finite numbers only because $0.95^{200} \approx 3.5 \times 10^{-5}$ and none of the 4,000 datasets hit the bad case. that's a lesson in itself: a simulation can't see an expectation dominated by rare events. the "finite cases only" column is misleading for a related reason: it conditions on having seen at least one 1, which removes exactly the datasets where the MLE fails, so at $n = 20$ the MLE even looks *better* than Laplace. Laplace's $\frac{k+1}{n+2}$ is the posterior mean under a uniform Beta(1, 1) prior: one pseudo-success and one pseudo-failure, which keep every prediction strictly between 0 and 1. the same fix appears as add-$k$ smoothing in n-gram models (ch 04 capstone), label smoothing in classifiers (ch 15), and the small $\varepsilon$ in `log(p + eps)` everywhere.

## experiment 3: do priors matter in the end?

**what people usually predict:** a strong wrong prior keeps a visible pull even after 1000 flips; the two-humped prior gives a two-humped posterior forever.

**what happens (true p = 0.3; 95% credible intervals):**

| n | uniform | Beta(40, 10) | two humps |
|---|---|---|---|
| 0 | (0.025, 0.975) | (0.680, 0.898) | (0.227, 0.773) |
| 10 | (0.308, 0.833) | (0.653, 0.864) | (0.279, 0.770) |
| 100 | (0.237, 0.417) | (0.401, 0.560) | (0.248, 0.376) |
| 1000 | (0.282, 0.339) | (0.305, 0.362) | (0.282, 0.337) |

after 10 flips (6 heads, an unlucky run) the two-humped posterior bets on the *wrong* hump near 0.7 and the confident prior hasn't moved much. by 100 flips the uniform and two-humped posteriors agree; the Beta(40, 10) one is still well off. by 1000 all three are narrow Gaussians, nearly on top of the dashed normal approximation, and the uniform and two-humped intervals agree to 0.002. the Beta(40, 10) interval is still shifted by about 0.023.

**why:** the log-posterior is $\ell(p) + \log p_0(p)$. the log-likelihood is a sum of $n$ terms, so it grows like $n$ and gets sharper (curvature $n/(p(1-p))$); the log-prior doesn't change. so the likelihood eventually dominates, and near its peak it's a quadratic, which makes the posterior Gaussian with standard deviation $\sqrt{\hat p(1 - \hat p)/n}$, whatever the prior's shape: that's the **Bernstein–von Mises theorem**. the Beta(40, 10) prior is worth 50 flips with 40 heads, so after 1000 real flips with 310 heads its posterior mean is $\frac{40 + 310}{1050} = 0.333$ instead of 0.31: diluted, but not gone. a prior that's zero on part of the space is different in kind: no amount of data can move posterior mass into a region with prior 0, because $\log 0 = -\infty$. "never assign probability exactly 0 to something possible" (Cromwell's rule) is the Bayesian cousin of experiment 2.
