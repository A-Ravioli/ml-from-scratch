# experimental rigor

## why this matters

you add a trick to your training recipe, run it once, and test accuracy goes from 93.0% to 93.4%. it feels like progress. but if you rerun the *old* recipe with a different seed you might get 93.5%. a single run is a sample of size one from a distribution you haven't looked at.

this isn't a hypothetical worry. Henderson et al. (2018) ran the same RL algorithm with the same hyperparameters on two sets of five seeds and got learning curves that a t-test called significantly different. Bouthillier et al. (2021) measured how much a benchmark score moves when you change only the data split, only the data order or only the initialization, and found each of these alone moves scores by amounts comparable to many published improvements. Melis, Dyer & Blunsom (2018) showed that a carefully tuned plain LSTM beat several newer architectures. Lipton & Steinhardt (2018) list "failure to identify the sources of empirical gains" as one of the troubling trends of the field.

none of this needs new mathematics: you've built bootstrap intervals and permutation tests (ch 05), paired comparisons (ch 29) and random search (ch 15). what's missing is the habit, and small tools that make it cheap: seeds for every configuration, means with intervals, seed-by-seed comparisons, ablation tables with uncertainty, and tuning budgets. this topic builds them as `mlfs.research`, which the rest of the chapter and your thesis use for every number.

## warm-up: predict first

1. two configurations, one run each: 93.2% and 92.7%. with five seeds per configuration, how often do you think the order would flip?
2. a 95% interval for the mean of 5 runs is $\bar x \pm c \cdot s/\sqrt{5}$. what is $c$?
3. you ran method A and method B on the same 5 seeds (same data order, same subset). should you compare them seed by seed, or as two independent groups? does it matter?
4. you tune your new method with 40 random-search trials and the baseline with 5. the new method wins by a point. what would you need to know before believing it?
5. an ablation removes component $X$ and accuracy changes by $-0.3 \pm 0.5$ points. does $X$ matter?

## prerequisites check

- [ ] you can compute a percentile bootstrap interval (→ ch 05 · topic 03)
- [ ] you can run a permutation test, paired or unpaired (→ ch 05 · topic 04)
- [ ] you know why multiple comparisons inflate false positives (→ ch 05 · topic 04)
- [ ] you've run random search and logged experiments (→ ch 15 · topic 06)
- [ ] optional: you've built ch 29's paired-comparison toolkit, whose sign-flip test and Holm correction this topic cites

## core ideas

### idea 1: a result is a distribution over seeds

a training run has many sources of randomness: the initial weights, the order in which minibatches arrive, which examples land in the training subset or split, dropout masks, data augmentation, and on GPUs non-deterministic kernels that add rounding noise. one "seed" usually sets several of these at once. the number you report from one run is one draw from the distribution these sources induce, and the honest object to report is that distribution (or at least its mean and an interval).

two levels of noise are easy to confuse. the **test set** is a sample of examples, so even a fixed model's accuracy has sampling error (that's what ch 27's item-level bootstrap measured). the **training run** is a sample of seeds, so the model itself varies. a bootstrap over test items with one trained model says nothing about the second kind. for claims about a *method* ("dropout helps"), the unit of replication is the whole training run, and the noise you must beat is seed-to-seed.

which sources matter most? Bouthillier et al. (2021) varied them one at a time: no single source dominates everywhere, data sampling is often among the largest, and fixing one seed hides variance that a reader re-running your code will see. Summers & Dinneen (2021) found that changing a *single* source, down to one bit of one initial weight, produced about as much variability as changing all of them, because training amplifies tiny differences. so vary *all* the sources together across seeds, and vary them one at a time only to learn where the variance comes from.

the first tool is trivial but removes excuses: a runner that calls your experiment once per seed and collects the results per metric.

> **spec.** `run_seeds(fn, seeds=(0, 1, 2, 3, 4))`: calls `fn(seed)` exactly once per seed, in the order given (any iterable). if every result is a scalar (a Python or NumPy number, or a 0-d array), return a float ndarray `(n_seeds,)` in seed order. if every result is a dict of scalars with the same keys, return a dict `key → float ndarray (n_seeds,)`, keys in the first result's order. a `nan` result is a crashed run: keep it. `ValueError` for no seeds, a mix of dicts and scalars, dicts with different keys, or non-scalar values.

> **🔮 predict:** you train a small MNIST network 20 times, varying only the initialization, then 20 times varying only the minibatch order, then only which 5,000 training images it sees. which source moves test accuracy the most? and how does each compare with varying all three together?

> **▶ build: stage 1.** implement `run_seeds`.

### idea 2: an interval for the mean of a few runs

with $n$ runs $x_1, \dots, x_n$, the mean $\bar x$ estimates the expected score of the method and its standard error is $s/\sqrt n$, where $s$ is the sample standard deviation with $n - 1$ in the denominator. if the runs are roughly normal, $(\bar x - \mu)/(s/\sqrt n)$ follows a Student $t$ distribution with $n - 1$ degrees of freedom, so

$$\bar x \pm t_{1 - \alpha/2,\, n - 1}\,\frac{s}{\sqrt n}$$

covers the true mean $\mu$ with probability $1 - \alpha$. at $n = 5$, $t_{0.975, 4} = 2.776$, not 1.96: using the normal quantile with five runs gives an interval that covers only about 88% of the time, because $s$ itself is a noisy estimate from five values and the $t$ distribution's heavy tails pay for that.

the bootstrap (your `mlfs.stats.bootstrap_ci`) avoids the normality assumption, but five values are a poor picture of their distribution, so at five runs its interval is too narrow (ch 29's capstone measured the coverage). a good default for "a handful of seeds" is $t$; use the bootstrap for skewed metrics with enough runs.

keep two numbers apart in your reports. the **standard deviation** $s$ describes how much *one run* varies: it's what a reader re-running your code once should expect. the **interval** describes how well you know the *mean*: it shrinks as $1/\sqrt n$. "93.1 ± 0.6" means nothing until you say which one it is and how many runs it came from.

> **spec.** `mean_ci(values, *, alpha=0.05, method='t', rng=None)` → `(mean, lo, hi)` as floats. `'t'`: the interval above, $s$ with `ddof=1`, the quantile from `scipy.stats.t.ppf(1 - alpha/2, n - 1)`. `'bootstrap'`: `lo, hi = mlfs.stats.bootstrap_ci(values, np.mean, alpha=alpha, rng=rng)`. `ValueError` for fewer than 2 values, any non-finite value, or another method.

> **▶ build: stage 2.** implement `mean_ci`.

### idea 3: comparing two methods, seed by seed

the question is almost never "what is method A's accuracy" but "is A better than B, and by how much?". if you ran both on the same seeds, so that seed $i$ fixed the same data order and subset for both, then $a_i$ and $b_i$ share part of their noise: an unlucky subset hurts both. the paired difference $d_i = a_i - b_i$ cancels the shared part:

$$\operatorname{Var}(a_i - b_i) = \operatorname{Var}(a_i) + \operatorname{Var}(b_i) - 2\operatorname{Cov}(a_i, b_i).$$

when the covariance is large, pairing can shrink the noise by a factor of several, and a real effect that's invisible in two overlapping intervals shows up clearly in the interval of the differences. pair when the seeds really are shared; compare unpaired when they aren't (a number from a paper, two codebases), and never zip lists of runs whose seeds you haven't matched.

the comparison reports three things: the **difference** (the effect size, in the metric's units), its **interval**, and a **p-value**. the interval is the headline: it says how big the effect could plausibly be, which a p-value can't. here the interval is a bootstrap over seeds and the p-value a permutation test, both from ch 05. two caveats you already know. with five paired seeds there are only $2^5 = 32$ sign patterns, so no two-sided permutation p-value can go below $2/32 = 0.0625$ (ch 29's `min_p_value`): five seeds can never be "significant at 0.05" by this test. and every extra comparison is another chance at a false positive: correct for them (Bonferroni or Benjamini-Hochberg from ch 05, Holm from ch 29), and decide which comparisons are primary *before* you run (Dror et al., 2018).

> **spec.** `compare_runs(a, b, *, paired=True, alpha=0.05, n_permutations=10000, rng=None)` → a dict with `diff`, `ci` (a `(lo, hi)` tuple), `p_value`, `n_a`, `n_b` and `paired`. normalize `rng` once with `mlfs.utils.as_rng` and draw the interval first, then the test, from that generator. **paired**: `ValueError` unless `len(a) == len(b)`; `diff = mean(a - b)`; `ci = bootstrap_ci(a - b, np.mean, alpha=alpha, rng=g)`; `p_value` from `permutation_test(a, b, n_permutations=n_permutations, paired=True, rng=g)`. **unpaired**: `diff = mean(a) - mean(b)`; `ci` a percentile bootstrap that resamples `a` and `b` independently (your `bootstrap_ci` resamples one array, so give it a statistic that draws a fresh resample of `b` each time it's called); `p_value` from `permutation_test(..., paired=False, ...)`. `ValueError` if either side has fewer than 2 runs or any value isn't finite.

> **▶ build: stage 3.** implement `compare_runs`.

### idea 4: ablations

an ablation asks *which part of the recipe is responsible for the result*. the standard form removes one component at a time from the full system and measures the drop (leave-one-out); the alternative adds components one at a time to a baseline. the two can disagree when components interact (two regularizers that each help alone may be redundant together), and when they do, say so: report both, or the full factorial grid for a small number of components.

an ablation table is a set of comparisons against the full system, so it needs everything idea 3 does: the same seeds for every row, a mean and interval per row, and the **paired difference** to the baseline with its interval. a row whose difference interval straddles zero is "no evidence this component matters at this budget", not "this component doesn't matter". the table below is the format `ablation_table` produces; it's Markdown, so it drops straight into a report, and the best row is bold so a reader's eye goes there first.

| config | mean ± 95% CI | n | Δ vs full ± 95% CI |
|---|---|---|---|
| **full** | **0.910 ± 0.025** | **3** | **(baseline)** |
| no aug | 0.880 ± 0.050 | 3 | -0.030 ± 0.043 |

the intervals are $t$ intervals (idea 2), so they're deterministic: the same results always render the same table. the difference column uses the $t$ interval of the per-seed differences, which is why the table demands the same number of runs in every row.

> **spec.** `ablation_table(results, *, baseline=None, alpha=0.05, higher_is_better=True, fmt='{:.3f}')`: `results` maps a config name to its per-seed values (rows in that order). returns the Markdown table as a string: a header `| config | mean ± L CI | n |` with `L` = `f"{100 * (1 - alpha):g}%"`, plus `Δ vs {baseline} ± L CI |` when a baseline is given; the separator `|---|---|…`; one row per config with `fmt(mean) ± fmt(half-width)` of `mean_ci(values, alpha=alpha)` and the number of runs; with a baseline, `(baseline)` in its own row and `fmt(mean d) ± fmt(half-width)` of `mean_ci(d, alpha=alpha)` for the per-seed differences `d = row − baseline` elsewhere, with a leading `+` unless the number starts with `-`. every cell of the best row (highest mean, or lowest if `not higher_is_better`; every tied row) is wrapped in `**…**`. `ValueError` for empty results, an unknown baseline, a row with a different number of runs than the baseline, or a row with fewer than 2 runs.

> **🔮 predict:** you ablate three components of a recipe on 20 seeds and get a clear order of importance. if you had run each row of the table only once (one seed per table), how often would the single-run table rank the three components in the same order?

> **▶ build: stage 4.** implement `ablation_table`.

### idea 5: strong baselines and tuning budgets

a comparison is only as strong as its weakest tuned arm. if your method got 40 hyperparameter trials and the baseline 5, you've measured tuning effort as much as methods. Melis et al. (2018) is the classic demonstration; Bouthillier et al. (2021) add that the hyperparameter search itself is a source of variance, so even "tuned" is a random variable.

Dodge et al. (2019) turned this into a reporting tool: **expected validation performance as a function of budget**. suppose you ran $n$ random-search trials with validation scores $v_{(1)} \le \dots \le v_{(n)}$. if you had run only $k$ trials, drawn from the same search, the best of them would be $v_{(i)}$ with probability

$$P(\max = v_{(i)}) = \left(\tfrac{i}{n}\right)^k - \left(\tfrac{i - 1}{n}\right)^k,$$

because "the best of $k$ draws is at most $v_{(i)}$" means all $k$ draws land among the $i$ smallest. so the expected best of $k$ trials is

$$\mathbb{E}[\max_k] = \sum_{i=1}^{n} v_{(i)} \left[\left(\tfrac{i}{n}\right)^k - \left(\tfrac{i-1}{n}\right)^k\right],$$

drawing with replacement from the empirical distribution of your trials. plot it against $k$ for each method: the curves show who wins at every budget, and where they cross. at $k = 1$ it's the mean trial (how good the method is with a random hyperparameter, a measure of robustness); as $k$ grows it climbs to the best trial you observed.

the same thinking applies to compute. report the budget of every comparison in a unit that matters (trials, GPU-hours, FLOPs: ch 26), match it between arms, and when you can't, say which arm got more. ch 29's capstone showed the *unit* can flip a winner too.

> **spec.** `expected_best(scores, k, *, higher_is_better=True)`: the formula above over the observed trial scores (any order), for an integer $k \ge 1$ or an integer array of them (returns a float, or an array of `k`'s shape). with `higher_is_better=False` the best is the minimum: sort the scores in descending order instead. `ValueError` for no scores, non-finite scores, or a `k` that isn't a positive integer.

> **🔮 predict:** SGD with momentum and Adam, each tuned over the learning rate by random search on a log scale. which one is better with 3 trials? with 30?

> **▶ build: stage 5.** implement `expected_best`.

## worked example

**five runs.** test accuracies of one configuration over seeds 0–4: $(1, 2, 3, 4, 5)$ in some unit. mean 3, $s = \sqrt{2.5} = 1.581$, standard error $1.581/\sqrt 5 = 0.707$, $t_{0.975,4} = 2.776$: interval $3 \pm 1.963 = [1.037, 4.963]$. with 1.96 you'd get $[1.614, 4.386]$, too narrow.

**an ablation table.** full recipe: $(0.90, 0.92, 0.91)$; without augmentation: $(0.88, 0.90, 0.86)$ on the same three seeds. full: mean 0.910, $s = 0.010$, half-width $4.303 \times 0.010/\sqrt 3 = 0.025$. no aug: mean 0.880, $s = 0.020$, half-width 0.050. paired differences $(-0.02, -0.02, -0.05)$: mean $-0.030$, $s = 0.0173$, half-width $4.303 \times 0.0173/\sqrt 3 = 0.043$. so the table reads "−0.030 ± 0.043": three seeds aren't enough to say augmentation matters. that's exactly the table in idea 4.

**expected best.** three trials scored $(1, 2, 3)$. the best of $k = 2$ draws is 1 with probability $1/9$, 2 with $4/9 - 1/9 = 3/9$, 3 with $1 - 4/9 = 5/9$: expected $22/9 = 2.444$. with $k = 3$: $72/27 = 2.667$, still below the observed maximum, because draws with replacement can repeat. your code must reproduce all three.

## on paper

1. **why $t$ and not $z$?** for $n = 5$ normal runs, the normal-quantile interval $\bar x \pm 1.96\, s/\sqrt n$ covers less than 95% of the time. use the fact that $(\bar x - \mu)/(s/\sqrt n) \sim t_4$ to compute its actual coverage. (you'll need $P(|t_4| \le 1.96)$; the $t_4$ CDF at 1.96 is about 0.939.)

<details><summary>solution</summary>

the interval covers $\mu$ exactly when $|\bar x - \mu|/(s/\sqrt n) \le 1.96$, i.e. $|T| \le 1.96$ with $T \sim t_4$. $P(|T| \le 1.96) = 2 \times 0.939 - 1 = 0.878$: about 88% coverage, so roughly one "95%" interval in eight misses instead of one in twenty. at $n = 20$ the $t_{19}$ CDF at 1.96 is about 0.968, coverage 93.5%; the problem fades as $n$ grows.

</details>

2. **how much does pairing buy?** two methods each have seed-to-seed standard deviation $\sigma = 0.01$, and the correlation between $a_i$ and $b_i$ on the same seed is $\rho$. what's the standard deviation of $d_i = a_i - b_i$ for $\rho = 0$, $0.5$ and $0.9$? how many paired seeds give the same interval width as 20 unpaired seeds per arm when $\rho = 0.9$?

<details><summary>solution</summary>

$\operatorname{Var}(d) = 2\sigma^2(1 - \rho)$, so $\operatorname{sd}(d) = 0.01\sqrt{2(1 - \rho)}$: 0.0141, 0.0100 and 0.0045. unpaired with 20 per arm, the standard error of $\bar a - \bar b$ is $0.01\sqrt{2/20} = 0.00316$. paired with $n$ seeds it's $0.0045/\sqrt n$; setting them equal gives $n = 2$ (ignoring the $t$ quantile, which at 2 seeds is 12.7 and makes you want at least 5 or so).

</details>

3. **the expected-best formula.** derive $P(\max_k = v_{(i)})$ for $k$ draws with replacement from $n$ distinct values, and check that the probabilities sum to 1. then show that $\mathbb{E}[\max_1]$ is the mean of the trials.

<details><summary>solution</summary>

$P(\max_k \le v_{(i)}) = P(\text{every draw is among the } i \text{ smallest}) = (i/n)^k$, since the draws are independent and each lands among the $i$ smallest with probability $i/n$. then $P(\max_k = v_{(i)}) = P(\max_k \le v_{(i)}) - P(\max_k \le v_{(i-1)}) = (i/n)^k - ((i-1)/n)^k$. the sum telescopes to $(n/n)^k - 0 = 1$. for $k = 1$ each weight is $i/n - (i-1)/n = 1/n$, so the expectation is the plain mean.

</details>

## experiment

the lab trains a small MLP on 5,000 MNIST digits, about a tenth of a second per run on a laptop CPU, so that it can afford a few hundred runs. it measures how much each source of randomness (initialization, data order, training subset) moves test accuracy on its own and together; ablates three parts of the training recipe on 20 paired seeds and asks how often a single-seed ablation table would have told the same story; and tunes SGD and Adam by random search over the learning rate, then uses your `expected_best` to see who wins at each budget.

## check your understanding

1. a paper reports "93.1 ± 0.4 (5 runs)". what two things could "± 0.4" mean?

<details><summary>answer</summary>

the sd of one run, or the half-width of a 95% interval for the mean, which at 5 runs is $2.776/\sqrt 5 = 1.24$ times the sd. say which, and how many runs.

</details>

2. why does `run_seeds` keep `nan` results instead of dropping them?

<details><summary>answer</summary>

a `nan` is usually a diverged or crashed run. dropping it silently reports the method as more reliable than it is: a method that diverges in 1 of 5 seeds has a real failure rate, and the reader needs to know. `mean_ci` and `compare_runs` then refuse to average it away, which forces you to decide (and report) what to do.

</details>

3. you compare 6 variants of your method against the baseline and report the one with p = 0.03. what's wrong, and what should you do?

<details><summary>answer</summary>

with 6 comparisons, the chance that at least one reaches p < 0.05 under the null is up to $1 - 0.95^6 = 26\%$. correct for the family (Holm or Bonferroni: here the corrected threshold is about 0.0083 for the smallest p), or designate one comparison as primary before running, and report all six either way.

</details>

4. a paired comparison of 5 seeds gives an interval of $[+0.2, +1.1]$ points and a permutation p of 0.06. is the effect real?

<details><summary>answer</summary>

the interval is the better summary: every value in it is a positive effect of a few tenths of a point to a point. the p-value can't go below 0.0625 two-sided with 5 paired seeds, so it carries almost no information here. say "an improvement of 0.65 points (95% CI 0.2 to 1.1, 5 paired seeds)", and note that the bootstrap interval at 5 seeds is likely too narrow; run more seeds if the claim matters.

</details>

5. why report `expected_best` at $k = 1$, though nobody tunes with one trial?

<details><summary>answer</summary>

it's the average score of a random configuration from the search space: how sensitive the method is to its hyperparameters, which is what a reader with less compute will experience.

</details>

## common bugs

- **zipping runs by position, not by seed.** a list of runs for A in seed order 0–4 and for B in the order they finished compares seed 3 with seed 0; a paired test then sees noise instead of shared noise. key runs by seed.
- **1.96 with five runs.** coverage drops to 88%. use the $t$ quantile with $n - 1$ degrees of freedom.
- **the unpaired bootstrap that only resamples one side.** the interval then ignores the other mean's noise and comes out too narrow; the check with a noisy `b` catches it.
- **one global seed for a whole sweep.** `torch.manual_seed(0)` once at the top, and every configuration's "seeds" are whatever the previous run left behind: results depend on the order you ran them in. seed each run from its own seed.
- **expected-best without replacement.** "the best of $k$ of my $n$ trials" sounds like sampling without replacement, but Dodge et al.'s estimator draws with replacement, so at $k = n$ it's below the maximum you observed.

## going deeper

- Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), "Deep reinforcement learning that matters", *AAAI*. seeds, implementations and evaluation in deep RL. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
- Bouthillier et al. (2021), "Accounting for variance in machine learning benchmarks", *MLSys*. sources of variance measured one at a time, and how many runs a comparison needs. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Summers & Dinneen (2021), "Nondeterminism and instability in neural network optimization", *ICML*. every source of nondeterminism has a similar effect, down to a single bit. [arXiv:2103.04514](https://arxiv.org/abs/2103.04514)
- Dodge, Gururangan, Card, Schwartz & Smith (2019), "Show your work: improved reporting of experimental results", *EMNLP*. expected validation performance. [arXiv:1909.03004](https://arxiv.org/abs/1909.03004)
- Melis, Dyer & Blunsom (2018), "On the state of the art of evaluation in neural language models", *ICLR*. tuned baselines. [arXiv:1707.05589](https://arxiv.org/abs/1707.05589)
- Lipton & Steinhardt (2018), "Troubling trends in machine learning scholarship". [arXiv:1807.03341](https://arxiv.org/abs/1807.03341)
- Dror, Baumer, Shlomov & Reichart (2018), "The hitchhiker's guide to testing statistical significance in natural language processing", *ACL*. [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/)
- Agarwal, Schwarzer, Castro, Courville & Bellemare (2021), "Deep reinforcement learning at the edge of the statistical precipice", *NeurIPS*. interval estimates and robust aggregates for few runs. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
- Colas, Sigaud & Oudeyer (2018), "How many random seeds? Statistical power analysis in deep reinforcement learning experiments". [arXiv:1806.08295](https://arxiv.org/abs/1806.08295)
- Wasserman, *All of Statistics* (2004), chapters 6–8 and 10: intervals, the bootstrap and testing.

## connections

- ← builds on: ch 05 · topics 03–04 (bootstrap intervals, permutation tests, multiple comparisons), ch 07 (honest evaluation), ch 15 · topic 06 (random search, experiment logs), ch 29's capstone (paired differences, the sign-flip test's floor, Holm).
- → used in: topic 02's equivalence checks build on the same intervals; topic 04's power analysis sizes the seeds you'll need; topic 05's figures show these intervals; the thesis capstone renders every number in your paper through `mlfs.research`; R1–R3 and any project you run afterwards.
