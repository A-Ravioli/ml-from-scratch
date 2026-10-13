# concentration inequalities · lab notes

## experiment 1: every bound against the truth

**what people usually predict:** that the bounds are "conservative", maybe 2–3 times the truth, and that a fancier bound is always better than a simpler one.

**what happens:** with $n = 100$ fair coins, every bound is useless (clipped near 1) for small $t$ and they spread apart quickly as $t$ grows:

| $t$ | exact | Chebyshev | Hoeffding | Bernstein | Chernoff |
|---|---|---|---|---|---|
| 0.10 | 0.057 | 4.4× | 4.8× | 5.4× | 4.7× |
| 0.15 | 0.0035 | 32× | 6.3× | 9.5× | 5.9× |
| 0.20 | 7.9e-5 | 800× | 8.6× | 22× | 6.8× |
| 0.30 | 1.1e-9 | 2.5e7× | 27× | 550× | 7.6× |

with $p = 0.01$ (and $n = 100$) the picture flips: Hoeffding is the *worst* bound at small $t$ (12.6× at $t = 0.02$, worse than Chebyshev's 3.1×), and at $t = 0.1$ it's off by a factor of $4 \times 10^7$, while Bernstein is off by 2,800 and Chernoff by 7.

**why:** Chebyshev decays like $1/t^2$, the exponential bounds like $e^{-ct^2}$. for small $t$ the polynomial is smaller (and Chebyshev uses the exact variance), for large $t$ the exponential wins by an astronomical margin. Hoeffding only knows the range $[0, 1]$, so it gives the same curve for every $p$: it effectively assumes the worst-case variance 1/4. at $p = 0.01$ the real variance is 25 times smaller, and Hoeffding's exponent is 25 times too small, which compounds into many orders of magnitude. Bernstein uses the variance and does far better, but its denominator's $\tfrac23 ct$ term makes it decay only like $e^{-ct}$ for large $t$. Chernoff uses the exact moment generating function, so it has the *correct* exponent $n\,\mathrm{KL}$: on a log plot its curve runs parallel to the truth, a roughly constant factor of 5–25 above it (that factor is the polynomial prefactor that exponential bounds can't capture, and it grows slowly with $n$). the takeaway: there's no single best bound. choose by what you know (range only, variance, the exact distribution family) and by where you need it (moderate or extreme deviations).

## experiment 2: how many test examples?

**what people usually predict:** a few hundred to a few thousand, roughly the same for every accuracy.

**what happens** (tolerance ±0.01, 95% confidence):

| method | accuracy 0.5 | accuracy 0.9 | accuracy 0.99 |
|---|---|---|---|
| Chebyshev | 50,000 | 18,000 | 1,980 |
| Hoeffding | 18,445 | 18,445 | 18,445 |
| Bernstein | 18,568 | 6,862 | 974 |
| Chernoff (two-sided) | 18,444 | 6,641 | 768 |
| CLT approximation | 9,604 | 3,458 | 381 |
| exact (binomial) | 9,602 | 3,482 | 401 |

for a 99%-accurate model, Hoeffding demands 18,445 examples while about 400 actually suffice.

**why:** Hoeffding depends only on the range, so its answer can't change with the accuracy. the variance of a 0/1 outcome, $a(1 - a)$, is 0.25 at $a = 0.5$ but 0.0099 at $a = 0.99$, and the variance-aware methods scale with it. the CLT approximation is excellent here (within 5% of exact) because $n$ is large and the deviation moderate, but it's an approximation, with no guarantee: at $a = 0.99$ it's slightly too optimistic (381 vs 401). Chebyshev, Hoeffding, Bernstein and Chernoff are guarantees, and each costs a factor of 2 or more over the exact answer. the catch with variance-aware bounds is that the variance depends on the accuracy you're trying to estimate. honest practice: bound it (e.g. use $a(1 - a) \le 1/4$, which gives back Hoeffding-like answers), or use an *empirical* Bernstein bound that plugs in the sample variance with a small correction (Maurer & Pontil, 2009), or use exact binomial confidence intervals (chapter 05).

reference: Maurer & Pontil (2009), "Empirical Bernstein bounds and sample variance penalization", COLT 2009. [arXiv:0907.3740](https://arxiv.org/abs/0907.3740)

## experiment 3: the winner's curse

**what people usually predict:** with every model at 0.80, the best of 1000 should score around 0.81, "maybe 0.82 with luck".

**what happens:** the best of 10 models averages 0.820; the best of 100, 0.831; the best of 1000, 0.840; the best of 10,000, 0.847. the luckiest run reached 0.861. all of it is noise: every model is exactly 0.80. the union-bound limit $0.80 + t_k$ (0.843 for one model, 0.873 for 1000, 0.880 for 10,000) is never exceeded, as guaranteed, and grows at the same slow rate.

**why:** each measured accuracy has standard deviation $\sqrt{0.8 \cdot 0.2/1000} \approx 0.0126$. the measured +0.047 for $k = 10{,}000$ is 3.7 of those standard deviations. the classical asymptotic for the maximum of $k$ independent normals is $\sqrt{2\log k}$ standard deviations, which is 4.3 at $k = 10{,}000$: it overestimates at any practical $k$ (the correction is a subtracted $\log\log k$ term), but it gets the growth rate right. the union bound's $t_k = \sqrt{\log(2k/\delta)/(2n)}$ has the same $\sqrt{\log k}$ growth. that slow growth is good news and bad news. good: you can compare thousands of models on a fixed test set and pay only logarithmically (this is the heart of the PAC bounds in chapter 13). bad: you do pay, and the payment is invisible in the reported number. a leaderboard where hundreds of teams submit to the same test set, a hyperparameter search scored on test data, or a paper reporting its best of many variants all inflate scores this way. the fix is procedural: select on a validation set, then evaluate the single chosen model once on a held-out test set (chapter 07).
