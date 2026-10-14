# confidence intervals & the bootstrap · lab notes

## experiment 1: do 95% intervals cover 95% of the time?

**what people usually predict:** the normal interval undercovers a bit for skewed data at small $n$; the bootstrap, which "makes no assumptions", fixes it; BCa is best.

**what happens (mean of lognormal(0, 1) data, 1,000 datasets per cell, 1,000 resamples per interval):**

| n | normal | percentile bootstrap | BCa bootstrap |
|---|---|---|---|
| 10 | 0.833 (width 2.21) | 0.819 (width 1.91) | 0.822 (width 2.12) |
| 30 | 0.882 (width 1.37) | 0.864 (width 1.30) | 0.875 (width 1.47) |
| 100 | 0.910 (width 0.81) | 0.910 (width 0.77) | 0.915 (width 0.85) |

every method misses about one time in six at $n = 10$, and even at $n = 100$ the "95%" intervals cover only about 91%. the bootstrap is no better than the normal interval; the percentile version is slightly narrower and slightly worse. at $n = 10$ the normal interval lies entirely *below* the true mean 19% of the time and entirely above it 0.2% of the time. for comparison (same settings): with normal data at $n = 10$ the three methods cover 0.92, 0.91 and 0.90, and with exponential data 0.86, 0.87 and 0.87. by $n = 100$ both are within about 1.5 points of 0.95.

**why:** lognormal data have a long right tail. a sample of 10 usually contains none of the rare huge values that pull the population mean up to $e^{0.5} = 1.65$, so its mean is usually too low, *and* its standard deviation is too low (the missing values are what would make it large). the interval is then too short and sits too low: that's the 19% of misses below. when the sample does catch a huge value, the mean overshoots but $s$ is huge too, so the interval is wide and still covers. hence the lopsided misses. the bootstrap can't help: it resamples the same 10 points, so it can't know about the tail they missed. its variance estimate uses the ddof-0 $\hat\sigma$, a factor $\sqrt{(n-1)/n}$ narrower than $s$, which costs it a little more. BCa corrects for skewness *visible in the sample*, which at $n = 10$ is too noisy to estimate well. more resamples change nothing: the limit is the information in $n$ points. the practical lesson: for heavy-tailed metrics (revenue per user, latency, per-example losses with outliers), nominal 95% intervals from small samples are optimistic, and you should say so, transform the data (e.g. a log scale), or collect more.

## experiment 2: when the bootstrap fails

**what people usually predict:** a bit below 95%, since the max is biased low. maybe 80–90%.

**what happens (n = 50, 300 datasets):** the actual coverage of both the percentile and BCa intervals is **0.000**. in the histogram, 62.5% of the bootstrap replicates sit exactly on the sample max (theory: $1 - (1 - 1/50)^{50} = 0.636$), and the rest pile up on the next few order statistics. the true sampling distribution is smooth and runs right up to $\theta = 1$.

**why:** every bootstrap replicate of the max is one of the observed values, so it's at most the sample max, which is strictly less than $\theta$. any interval made of quantiles of the replicates lies entirely below $\theta$: coverage zero, whatever $B$ is. the bootstrap's core assumption is that the relationship between $\hat F$ and a resample mimics the relationship between $F$ and the sample. for smooth statistics like the mean, small perturbations of the distribution cause small changes, and the analogy holds. the max depends on the extreme edge of the distribution, exactly where $\hat F$ (which has nothing beyond the largest point) and $F$ differ most. the spike shows the mismatch: a resample contains the sample max with probability 0.63, while a fresh sample hits *any* particular value with probability 0. a correct interval uses the exact pivot from topic 01: $M/\theta$ has CDF $t^n$, so $P\big((\alpha/2)^{1/n} \le M/\theta \le (1 - \alpha/2)^{1/n}\big) = 1 - \alpha$, giving $\theta \in \big[M/(1 - \alpha/2)^{1/n},\ M/(\alpha/2)^{1/n}\big]$. (the "m-out-of-n" bootstrap, resampling fewer than $n$ points, is the general repair.)

## experiment 3: paired data, resampled two ways

**what people usually predict:** both methods give similar intervals; the unpaired one is maybe a bit wider.

**what happens (correlation 0.9, 300 test sets of 200 examples):**

| bootstrap | coverage of the true gap 0.05 | mean width | fraction of intervals excluding 0 |
|---|---|---|---|
| paired | 0.947 | 0.123 | 0.337 |
| unpaired | 1.000 | 0.390 | 0.000 |

the unpaired interval is 3.2 times wider, covers 100% of the time (overcautious, not "safe"), and *never* detects that B is better. the paired interval has the right coverage and detects the improvement in a third of the test sets. at correlation 0 the two agree.

**why:** $\operatorname{Var}(\bar a - \bar b) = \frac1n(\sigma_a^2 + \sigma_b^2 - 2\operatorname{Cov}(a, b))$. resampling examples keeps each example's two scores together, so the replicates carry the covariance. resampling the two arrays independently destroys it and estimates $\frac1n(\sigma_a^2 + \sigma_b^2)$: with $\rho = 0.9$ that's 10 times the true variance, $\sqrt{10} = 3.2$ times the width. the unpaired analysis is wrong, in the conservative direction: it's not that it lies, it's that it throws away the most useful fact about the experiment, that both models saw the same examples. most of the variation in per-example loss is "this example is hard", which affects both models equally and cancels in the difference. this is why model comparisons should always be paired on a shared test set, and it's the central design choice of this chapter's capstone.
