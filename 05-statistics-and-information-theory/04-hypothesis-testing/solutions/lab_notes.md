# hypothesis testing · lab notes

## experiment 1: a p-hacking simulator

**what people usually predict:** each habit adds a little, maybe a point or two each; all three together perhaps 10–15%.

**what happens (10 outcomes; ± is the Monte Carlo standard error):**

| analysis | studies simulated | fraction of null studies with a "discovery" |
|---|---|---|
| honest: 1 outcome, n = 30, no tricks | 1,000 | 0.058 ± 0.007 |
| 10 outcomes | 400 | 0.393 ± 0.024 |
| optional stopping (10 looks) | 1,000 | 0.185 ± 0.012 |
| outlier rule | 1,000 | 0.108 ± 0.010 |
| all three | 200 | 0.980 ± 0.010 |

the honest analysis is at the nominal 5% (0.058 is about one standard error above it). ten outcomes give 40%, matching $1 - 0.95^{10} = 0.401$. optional stopping alone nearly quadruples the false-positive rate. with all three habits, 98% of studies of a treatment that does *nothing* report a significant effect.

**why:** every habit adds tests, and the researcher reports the best one. ten outcomes are ten independent chances. optional stopping gives ten correlated chances: the p-value of a null experiment wanders as data arrive, and stopping the first time it dips below 0.05 catches its lowest point (on paper 4: with unlimited peeking the rate goes to 1). the outlier rule is a second, highly correlated analysis of the same data, and that one extra look roughly doubles the rate, to about 11% (an exact $t$-test gives the same, 0.11). a simulation row has its own error bar: with only 200 studies it would be about ±0.02, enough to turn 0.11 into 0.15 by chance, which is why this row uses 1,000 studies and the table shows the standard errors. combined, the habits multiply: 10 outcomes × 10 looks × 2 versions is 200 implicit tests. Bonferroni or BH can fix "10 outcomes" if all ten are reported and corrected for, because then the number of tests is known. they can't fix optional stopping or post-hoc analysis choices, because the number of implicit tests isn't written down anywhere. those need pre-registration (fix the outcome, the sample size and the analysis in advance) or methods built for sequential looks. in ML the same habits appear as "we tried several seeds and report the best", "we stopped training when the test score peaked", and "we dropped the benchmarks where we didn't win".

## experiment 2: power, and the winner's curse

**what people usually predict:** 20 per group should detect a 0.3 sd effect most of the time; significant studies report roughly the true effect.

**what happens (true effect 0.3 sd, 200 studies per row):**

| n per group | power | mean estimated effect among significant studies |
|---|---|---|
| 20 | 0.130 | 0.805 (2.7× the truth) |
| 50 | 0.280 | 0.505 (1.7×) |
| 100 | 0.550 | 0.417 (1.4×) |
| 200 | 0.845 | 0.328 (1.1×) |

with 20 per group the test detects the effect only 13% of the time, and when it does, the estimated effect averages 0.8, almost three times the truth. the p-value histogram with no effect is flat; with an effect of 0.5 sd it piles up near 0, but still has a long tail: many real effects give $p > 0.05$.

**why:** under the null the p-value is uniform (on paper 1), so a flat histogram is what a correct test looks like, and $p = 0.84$ is exactly as likely as $p = 0.04$. power depends on the effect in units of the standard error, $\delta/\sqrt{2/n}$: 0.95 at $n = 20$, far below the $\approx 2.8$ needed for 80% power, which takes about 175 per group. the winner's curse is selection: with $n = 20$, the estimate's standard error is 0.32, so it must land above roughly $1.96 \times 0.32 = 0.62$ to be significant. conditioning on significance throws away every run where noise pulled the estimate down, so the survivors overestimate the effect, the more so the lower the power. a literature that publishes only significant results, or a leaderboard where only "wins" get written up, inherits that bias: early, small studies report the biggest effects, and replications shrink them.

## experiment 3: a thousand tests at once

**what people usually predict:** without correction a handful of false discoveries; Bonferroni finds most real effects; BH is about the same.

**what happens (1000 hypotheses, 50 real, averages over 100 screens):**

| procedure | real effects found | false discoveries | false discovery proportion | P(at least one false) |
|---|---|---|---|---|
| no correction | 37.1 of 50 | 47.5 | 0.559 | 1.00 |
| Bonferroni | 7.9 of 50 | 0.0 | 0.007 | 0.04 |
| Benjamini–Hochberg | 16.4 of 50 | 0.9 | 0.048 | 0.61 |

without correction, more than half of the "discoveries" are false: 950 nulls at 5% give about 47 false positives, more than the 37 real effects found. Bonferroni almost never makes a false discovery, but finds only 8 of the 50 real effects. BH finds twice as many with a false discovery proportion of 4.8%, but it makes at least one false discovery in 61% of screens.

**why:** each procedure keeps a different promise. no correction controls the error rate *per test*: 5% of nulls are rejected, and with 950 nulls that's dozens. Bonferroni controls the *family-wise* rate, $P(\text{any false}) \le 0.05$ (measured 0.04), by demanding $p \le 0.00005$, a $z$-score of about 3.9, which only the largest effects reach. BH controls the expected *proportion* of false discoveries, at most $\frac{m_0}{m}q = 0.0475$ here (measured 0.048), so it accepts a few false positives in exchange for many more true ones. it adapts: its effective threshold, $kq/m$ for the $k$ it lands on, is about 17 times Bonferroni's here. use Bonferroni (or the uniformly better Holm procedure) when any single false claim is costly, such as a confirmatory claim in a paper or a safety check. use BH for screening, where discoveries will be followed up anyway. one practical gotcha surfaced in the setup: Monte Carlo p-values can't go below $1/(1 + B)$, so with 200 permutations Bonferroni at $m = 1000$ can *never* reject. large-scale testing with permutation tests needs many permutations, or exact or asymptotic p-values.
