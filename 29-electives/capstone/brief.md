# 29 · capstone: the elective portfolio

## goal

every elective block ends with a small project, and every project has the same shape: a method you built against a classical baseline, on a world you can only partly control. PC and IDA against a naive regression. replay against fine-tuning. a selective SSM against a transformer, K-FAC against Adam, a quantum kernel against an RBF kernel. the projects differ in everything except the question a reader will ask about each: *is the method actually better, by how much, and how sure are you?*

this capstone answers that question the same way for every elective you completed. you build a small **fair-comparison toolkit** (pairing by seed, an interval for the paired difference, an exact sign-flip test, Holm's correction across claims, a budget table with explicit units, and a structured portfolio entry), then you write one entry and a short report per elective. the toolkit is deliberately small; the hard part is the judgement in the reports: whether your baseline is the one a skeptic would pick, whether the budget is matched in the unit that matters, and whether the simulator you built favours your method.

nothing here is heavy to run. the expensive work is the block projects you've already done; this notebook only consumes their per-seed scores.

## what you build

all of it in `capstone.py`, with its checks green. the spec, function by function:

- **stage 1 · `paired_differences(method_scores, baseline_scores)`** → `(seeds, d)`. both arguments are `{seed: score}` dicts, or both are sequences indexed by seed (index $i$ = seed $i$). `seeds` is the sorted list of seeds; `d` is a float array of `method − baseline` per seed, in that order. raise `ValueError` when the seed sets differ, the lengths differ, one argument is a dict and the other isn't, any score isn't finite, or there are no scores. never drop, truncate or zip silently.
- **stage 2 · `paired_ci(d, *, alpha=0.05, method='t', n_boot=10000, rng=None)`** → `(lo, hi)` for the mean of `d`. `'t'`: $\bar d \pm t_{1-\alpha/2,\,n-1}\,s_d/\sqrt n$ with $s_d$ using $n - 1$ (`scipy.stats.t.ppf` is allowed: you met the $t$ correction in ch 05 · topic 03). `'bootstrap'`: return your `mlfs.stats.bootstrap_ci(d, np.mean, n_boot=n_boot, alpha=alpha, rng=rng)`. fewer than 2 values, or another method, raise `ValueError`.
- **stage 3 · `sign_flip_test(d, *, alternative='two-sided', n_permutations=10000, rng=None)`** → p. the statistic is the mean of `d`. if $2^n \le$ `n_permutations`, enumerate **all** $2^n$ sign patterns and return the exact fraction at least as extreme as the observed mean (the observed pattern counts; compare with a tolerance $10^{-12}\max(1, |\bar d|)$ in the direction that counts ties as extreme). otherwise return the p-value of your `mlfs.stats.permutation_test(d, np.zeros(n), n_permutations=..., alternative=..., paired=True, rng=rng)`. `alternative` is `'two-sided'` ($|T^*| \ge |\bar d|$), `'greater'` or `'less'`; anything else raises `ValueError`. **`min_p_value(n, alternative='two-sided')`** → the smallest p the exact test can return: $2/2^n$ two-sided, $1/2^n$ one-sided.
- **stage 4 · `holm(p_values, alpha=0.05)`** → `(reject, adjusted)`, both in the input's order: adjusted $\tilde p_{(i)} = \max_{j \le i}\min\{1, (m - j + 1)\,p_{(j)}\}$ over the sorted p-values, and `reject = adjusted <= alpha` (equivalent to the step-down procedure).
- **stage 5 · `memory_cost(components, unit='bytes')`**: `components` is a list of `(count, dtype)` pairs (any dtype `np.dtype` accepts). `unit='floats'` returns the number of stored values whatever their type (the continual block's `n_floats()`); `unit='bytes'` returns $\sum$ count × itemsize; other units raise `ValueError`. **`budget_table(methods, budget, *, unit='bytes', higher_is_better=True)`**: `methods` maps a name to `{'memory': components, 'scores': per-seed scores}`. returns `{'rows', 'winner', 'budget', 'unit'}`: one row dict per method with `name`, `cost`, `fits` (`cost <= budget`), `mean`, `sd` (`ddof=1`; `nan` for one seed) and `n`; fitting rows first, best mean first within each group; `winner` is the best fitting method's name, or `None` if nothing fits.
- **stage 6 · `portfolio_entry(block, question, method_name, method_scores, baseline_name, baseline_scores, *, metric, higher_is_better=True, alpha=0.05, ci_method='t', n_permutations=10000, budget=None, notes='', rng=None)`** → a dict with at least `block`, `question`, `metric`, `higher_is_better`, `method`, `baseline`, `seeds`, `n`, `method_mean`, `baseline_mean`, `diff` (mean of method − baseline), `ci` (from `paired_ci` with `ci_method`), `alpha`, `ci_method`, `p` (two-sided `sign_flip_test`), `min_p` (`min_p_value(n)`), `verdict`, `budget` and `notes`. the scores go through `paired_differences`. `verdict` reads the interval: `'better'` if it lies entirely on the good side of 0 (above for higher-is-better, below for lower-is-better), `'worse'` if entirely on the bad side, `'inconclusive'` otherwise. **`render_entry(entry)`** → markdown in this format (the budget and notes lines only when given):

```text
### {block}

**question.** {question}

| | {metric} (mean over n = {n} paired seeds) |
|---|---|
| {method} | {method_mean:.3f} |
| {baseline} (baseline) | {baseline_mean:.3f} |
| difference (method − baseline) | {diff:+.3f}, {100(1 − alpha)}% CI [{lo:+.3f}, {hi:+.3f}] ({ci_method}) |
| sign-flip p (two-sided) | {p:.3g} (smallest possible with n = {n}: {min_p:.3g}) |

**verdict:** {verdict} ({higher|lower} is better).

**budget:** {budget}

{notes}
```

## data

none of its own. the tests use synthetic scores with fixed seeds. your entries use the per-seed scores from your block projects: each project's lab prints or stores them; rerun it with several seeds if it doesn't.

## configurations

| config | where | what runs | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | the toolkit on synthetic scores, the coverage demo, your entries | a few seconds |
| **full** (your write-up) | default | the same notebook; the numbers come from your block projects' full configs, run beforehand (topic 04's project about a minute, topic 14's about two) | a few seconds here, plus your projects |

there's nothing heavy to switch: the notebook itself never trains anything.

## deliverables

1. `capstone.py` with the six stages implemented and their checks green.
2. one `portfolio_entry` per elective you completed (at least two), built from your own per-seed scores, rendered in the notebook. for a budgeted comparison, a `budget_table` as well, in the unit you argue for (and in the other one, if it changes the winner).
3. the write-up at the end of the notebook: for each elective, the five prompts (question and baseline; fairness and budget; result; the setup's thumb on the scale; what would change the verdict), then the portfolio-level Holm correction.

## two worked entries

the reference solution includes these, from the two blocks finished first, with numbers measured by the reference implementations on the cached real data.

**causal inference (topics 01–04).** topic 04's project builds a 7-variable linear world on real California-housing covariates (shuffled per column, 5,000 districts per draw) with a true effect of $T$ on $Y$ of 0.96. over 20 draws (seeds 100–119), PC ($\alpha = 0.01$) plus IDA estimates **0.960 ± 0.018** (sd over draws), recovering the true CPDAG in 20 of 20 draws; the naive regression of $Y$ on $T$ gives **1.120** (income confounds it by $0.4 \times 0.8/2 = 0.16$). as a portfolio entry on absolute error (lower is better): 0.015 against 0.160, difference −0.145, 95% t interval [−0.151, −0.138], Monte Carlo sign-flip p = 0.0001 over 20 paired draws; verdict **better**. the caveat that belongs in the report: keep the covariates' real dependence and a near-violation of faithfulness makes PC orient a false v-structure at $T$ in half the draws, and IDA drops to 0.816 ± 0.147.

**continual learning (topics 12–14).** topic 14's project: class-incremental split MNIST, a memory budget of 39,200 stored numbers. a per-class Gaussian generator (15,680 float32s) reaches ACC **0.780 ± 0.018** over seeds 0–4 against fine-tuning's **0.196 ± 0.000**, and a plain 50-image reservoir filling the budget **0.565 ± 0.010**. as an entry: difference +0.584, 95% t interval [+0.562, +0.606], verdict **better**, and an exact sign-flip p of **0.0625**, which is the smallest p five seeds can produce (two-sided), so the interval carries this claim, not the p-value. the budget table shows the unit deciding the winner: in stored numbers the generator wins; in bytes, 200 uint8 images (156,800 bytes, the size of 39,200 float32s) fit and reach 0.804 ± 0.019, slightly ahead of the generator and within the seed noise.

Holm over the two primary p-values (0.0001 and 0.0625) keeps the causal claim (adjusted 0.0002) and not the continual one (0.0625), which is why each report must say what its interval shows.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass (`python -m pytest test_capstone.py`) |
| pairing | pairs by seed, not by order; refuses every unpairable input |
| intervals and tests | exact on hand examples; the $t$ interval covers 95% ± 1.2% over 3,000 simulated 5-seed experiments; the sign-flip test's size under the null is ≤ 6.5% at $\alpha = 0.05$ (exact size 4.7% at $n = 8$) |
| Holm | matches the definition on random inputs, including ties; never rejects less than Bonferroni; family-wise error ≤ 6% over 4,000 simulated families of 5 true nulls |
| budgets and entries | the unit flips the winner on the continual block's numbers; verdicts follow the interval in both directions; false "better/worse" claims 2.5–8% of the time under the null |
| the write-up | one entry per completed elective (≥ 2), every number with its interval and number of seeds, the budget's unit stated |

## stretch goals

- **power before you run.** from a pilot of 3 seeds, estimate how many seeds your project needs to detect half the observed difference with 80% power (simulate from the pilot's differences), then check the estimate with the extra seeds.
- **a second baseline per project.** "Y on T and everything else" for causal inference, EWC for continual learning, a seasonal-naive forecast or an LSTM for state-space models: report each with Holm across that entry's comparisons.
- **interval coverage at your seed count.** repeat this notebook's coverage demo with the distribution of *your* project's paired differences (resample them), and report how far the bootstrap's coverage falls below 95%.
- **a compute-matched table.** many electives trade memory for compute (replay doubles each step's cost). add a second budget column in FLOPs or wall-clock seconds and see whether the winner changes again.
- **one more elective.** a third block adds a third entry, and makes Holm bite harder.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Holm (1979), "A simple sequentially rejective multiple test procedure", *Scandinavian Journal of Statistics* 6(2), 65–70. [jstor.org/stable/4615733](https://www.jstor.org/stable/4615733)
- Fisher (1935), *The Design of Experiments*, Oliver & Boyd, chapter III: the randomization (sign-flip) test on Darwin's paired plants.
- Good (2005), *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed., Springer. [doi:10.1007/b138696](https://doi.org/10.1007/b138696)
- Dror, Baumer, Shlomov & Reichart (2018), "The hitchhiker's guide to testing statistical significance in natural language processing", *ACL*. [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/)
- Bouthillier et al. (2021), "Accounting for variance in machine learning benchmarks", *MLSys*. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Agarwal, Schwarzer, Castro, Courville & Bellemare (2021), "Deep reinforcement learning at the edge of the statistical precipice", *NeurIPS*. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
