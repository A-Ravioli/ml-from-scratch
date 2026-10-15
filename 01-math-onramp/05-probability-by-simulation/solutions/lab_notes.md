# probability by simulation · lab notes

## experiment 1: the birthday problem

**what people usually predict:** a small chance for 23 people, often 5–10% ("23 out of 365"). for 90%, guesses of 150–300 people are common.

**what happens:** with 23 people the simulation gives about **0.504 ± 0.005** (10,000 rooms), against the exact **0.507**. the chance passes 90% at **41** people, and by 60 people it's above 99%. every simulated point on the curve sits within its error bars of the exact curve (with 27 points and ±2 se bars, one or two may poke just outside: that's expected about 5% of the time).

**why:** what matters is the number of *pairs*, not the number of people. 23 people make $\binom{23}{2} = 253$ pairs, each with a $1/365$ chance of matching, so you expect $253/365 \approx 0.69$ matching pairs. the exact computation multiplies the chances that each new person avoids all the earlier birthdays, $\frac{364}{365} \cdot \frac{363}{365} \cdots$, and those factors shrink faster than intuition expects. the error bar is what lets you compare the simulation with the exact answer at all: $0.504$ against $0.507$ is a difference of less than one standard error, so the two agree. (your own run, with your own implementation, may land 1–2 standard errors away from the exact value; that still counts as agreement. only differences of 3 or more standard errors should make you suspect a bug.) without the $\pm 0.005$, you couldn't tell whether $0.504$ was a bug or noise. the same reasoning, "is the difference bigger than the uncertainty?", decides whether one model really beats another, from chapter 07 on.

## experiment 2: what does one more digit cost?

**what people usually predict:** that more darts help steadily, maybe "10 times more darts, 10 times more accurate". for 3 decimal places, guesses of 10,000 to 100,000 darts.

**what happens:** the average error falls along a straight line with fitted slope about **−0.5** (around $-0.52$ in this run), lying on top of the reported standard error and the $1.64/\sqrt n$ line: about 0.4 with 10 darts, 0.01 with 10,000, and 0.001 with a million. getting the typical error below 0.0005 takes about **7 million** darts, and each further digit costs 100 times more.

**why:** each dart's score (4 or 0) has standard deviation $\sigma = \sqrt{\pi(4 - \pi)} \approx 1.64$, and averaging $n$ independent scores divides the standard deviation by $\sqrt n$. a slope of $-\frac12$ on a log-log plot is exactly $n^{-1/2}$. (the typical absolute error is about $0.8\,\sigma/\sqrt n$, because the average of $|\text{bell-shaped error}|$ is $\sqrt{2/\pi} \approx 0.8$ standard deviations.) compared with topic 03's trapezoid rule ($1/n^2$) this is terribly slow for a 1-D integral. Monte Carlo's strength is that $\sigma/\sqrt n$ doesn't care about the dimension: the same number of darts gives the same accuracy in 2 dimensions or in 2,000, where a grid with even 2 points per side would need $2^{2000}$ points. that's why every high-dimensional average in ML (over the weights of a Bayesian model in chapter 12, or the noise of a diffusion model in chapter 22) is computed by Monte Carlo, and why reducing $\sigma$ itself ("variance reduction") is a whole field.

## experiment 3: do error bars tell the truth?

**what people usually predict:** about 95% in every case ("that's what 95% means"), or slightly less for small $n$.

**what happens:**

| true $p$ | trials $n$ | intervals containing $p$ | runs with zero events |
|---|---|---|---|
| 0.5 | 50 | 92.7% | 0% |
| 0.5 | 5000 | 95.5% | 0% |
| 0.01 | 50 | **38.4%** | 61.7% |
| 0.01 | 5000 | 94.6% | 0% |

for the common event, the rule of thumb works (a little under 95% with $n = 50$). for the rare event with 50 trials, the "95%" interval contains the truth only **38%** of the time.

**why:** with $p = 0.01$ and $n = 50$, the event doesn't happen at all in $0.99^{50} \approx 61\%$ of runs. then every value is 0, so the estimate is 0, the sample standard deviation is 0, and the interval is $[0, 0]$: perfectly confident, and wrong. those runs are all misses. the "±1.96 standard errors" rule assumes the estimate has a bell-shaped distribution around the truth, which is true when the expected number of events, $np$, is reasonably large (here $np = 0.5$, far too small; with $n = 5000$, $np = 50$ and the rule works again). the lesson: an error bar is a model of your uncertainty, and models can fail. be most suspicious exactly when it's most tempting to trust them: tiny estimated errors from few samples, and rare events. chapter 05 builds intervals (like the Wilson interval and the bootstrap) that behave better, and the "rule of three" says that if you saw zero events in $n$ trials, a 95% upper bound for $p$ is about $3/n$, here $0.06$.
