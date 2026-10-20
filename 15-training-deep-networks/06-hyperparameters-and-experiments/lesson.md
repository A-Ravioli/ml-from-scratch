# hyperparameters & experiments

## why this matters

count the knobs this chapter has added: the initialization scheme, the normalization layer, the dropout rate, the weight decay, label smoothing, the augmentations, the learning rate, its schedule and warmup, the batch size, the gradient-clipping threshold, and the patience of early stopping. add the architecture (depth, width) and you have a dozen numbers to choose before training even starts, each of which can make the difference between a model that works and one that doesn't. trying them by hand, one at a time, is slow and biased towards whatever you tried first.

this topic builds the standard tools for doing it systematically, and the habits that keep the result honest. **random search** is the baseline that beats the intuitive alternative (grid search) for a reason you'll derive. **successive halving** and **hyperband** stop wasting compute on configurations that are clearly bad, by training many configurations briefly and only the promising ones fully. and an **experiment log** records every run, so that the final comparison uses all the evidence, including the uncomfortable fact the lab demonstrates: the difference between two random seeds of the *same* configuration is often as large as the improvement you're trying to measure.

## warm-up: predict first

1. you have budget for 9 training runs and two hyperparameters, one of which barely matters. a 3×3 grid tests how many distinct values of the one that matters? 9 random configurations test how many?
2. a learning rate is drawn uniformly from $[10^{-5}, 10^{-1}]$. what fraction of the draws are below $10^{-2}$?
3. you train 27 configurations for 1 epoch each, keep the best 9 for 3 epochs, the best 3 for 9 epochs and the best 1 for 27 epochs. how many epochs in total? how many would 27 full runs cost?
4. you run 8 configurations and report the best validation accuracy. is that number an unbiased estimate of how well the chosen configuration will do on new data?
5. the same configuration, trained with 10 different seeds, reaches validation accuracies between 89.8% and 91.6%. a different configuration gets 90.9% with one seed. is it better?

## prerequisites check

- [ ] you know why a validation set (not the test set) is used for choosing hyperparameters, and what nested cross-validation fixes (→ ch 07 · topic 06)
- [ ] you can compute a bootstrap confidence interval and a standard error (→ ch 05 · topic 03)
- [ ] you know what each of this chapter's knobs does (→ topics 01–05)
- [ ] you can seed a training run through `rng` arguments (→ ch 00 · topic 05)

## core ideas

### idea 1: random search

every search in this topic talks to the problem through two functions, so it works for any model in any framework:

- **`sample_config(rng) -> dict`** draws one configuration, e.g. `{"lr": 10 ** rng.uniform(-5, -1), "width": rng.choice([32, 64, 128])}`.
- **`evaluate(config, budget) -> float`** trains it with a given amount of resources (epochs, steps, examples) and returns a loss to minimize, usually a validation loss or an error rate.

each search returns `(best_config, best_loss, history)`, where `history` lists every evaluation as a `(config, budget, loss)` tuple.

**random search** (`random_search(sample_config, evaluate, n_trials=20, budget=1, rng=None)`) draws `n_trials` configurations in turn from one generator, evaluates each at the same budget, and keeps the lowest loss. it seems naive next to a grid, and it's better than one, for a reason Bergstra & Bengio (2012) pointed out: in most problems only a few hyperparameters matter much. a $3 \times 3$ grid over (learning rate, dropout) spends 9 runs to test only 3 distinct learning rates; 9 random configurations test 9 (warm-up 1). when you don't know which hyperparameters matter, random search hedges automatically. it also has a clean guarantee: if the top 5% of the space counts as "good", $n$ random trials find a good one with probability $1 - 0.95^n$, which is 95% at $n = 59$, whatever the dimension (on paper 1).

**sample on the right scale.** learning rates, weight decays and regularization strengths matter multiplicatively: $10^{-4}$ versus $10^{-3}$ is as big a change as $10^{-2}$ versus $10^{-1}$. sample their logarithm uniformly (`10 ** rng.uniform(-5, -1)`), or 90% of your trials land between $10^{-2}$ and $10^{-1}$ (warm-up 2).

> **▶ build: stage 1.** implement `random_search`. losses are stored as Python floats; ties go to the first.

### idea 2: successive halving

most configurations in a random search are bad, and most bad configurations look bad early. **successive halving** (Jamieson & Talwalkar, 2016) exploits that: train many configurations with a small budget, keep the best fraction, give them more budget, and repeat. with $n$ configurations, a minimum budget $r$ and a reduction factor $\eta$ (3 is standard), **rung** $i$ evaluates the survivors at budget $r\eta^i$ and promotes the best $\lfloor n_i/\eta\rfloor$ (at least one) to the next rung:

| rung | configurations | budget each | cost |
|---|---|---|---|
| 0 | 27 | 1 | 27 |
| 1 | 9 | 3 | 27 |
| 2 | 3 | 9 | 27 |
| 3 | 1 | 27 | 27 |

every rung costs about the same, so the whole search costs $108$ budget units, the price of 4 full runs, for a look at 27 configurations (warm-up 3). successive halving runs rungs until a single configuration is left; with $n = 10$ and $\eta = 3$ that's rungs of 10, 3 and 1, and with $n = 2$ a single rung (since $\lfloor 2/3\rfloor = 0$). the returned best is the survivor of the last rung, with its loss at the largest budget.

write the rung loop once as **`run_bracket(configs, evaluate, n_rungs=..., min_budget=..., eta=..., history=...)`**: it evaluates the configurations at rung $i$'s budget (`rung_budget`, given, returns $r\eta^i$ as an int when it's a whole number), appends every evaluation to `history`, promotes the best $\lfloor n/\eta\rfloor$ (best first) unless it's the last rung, and returns the last rung's `[(config, loss)]`. `successive_halving` samples its configurations and calls it; hyperband will call it once per bracket.

the weakness is the first rung. a configuration that learns slowly but ends up best (a small learning rate, heavy regularization) can look bad after one epoch and be eliminated: learning curves cross. how aggressively to cut is itself a choice, and that's what hyperband hedges.

> **🔮 predict:** with 81 configurations and $\eta = 3$, how many rungs does successive halving run, and what's the total cost in units of the minimum budget?

> **▶ build: stage 2.** implement `run_bracket` and `successive_halving`.

### idea 3: hyperband

**hyperband** (Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar, 2017) runs successive halving several times, from very aggressive (many configurations, a tiny starting budget) to not at all (a few configurations at the full budget: plain random search), and returns the best of all. with a maximum budget $R$ (`max_budget`) and $\eta$:

$$s_\text{max} = \lfloor\log_\eta R\rfloor,\qquad\text{bracket } s = s_\text{max}, \dots, 0:\quad n_s = \Big\lceil\frac{(s_\text{max} + 1)\,\eta^s}{s + 1}\Big\rceil\text{ configurations, starting at } r_s = R\,\eta^{-s},$$

each run as successive halving with $s + 1$ rungs, so that every bracket ends at budget $R$. every bracket costs roughly the same, $(s_\text{max} + 1)R$. for $R = 81$, $\eta = 3$:

| bracket $s$ | configurations $n_s$ | rungs (configurations @ budget) |
|---|---|---|
| 4 | 81 | 81@1, 27@3, 9@9, 3@27, 1@81 |
| 3 | 34 | 34@3, 11@9, 3@27, 1@81 |
| 2 | 15 | 15@9, 5@27, 1@81 |
| 1 | 8 | 8@27, 2@81 |
| 0 | 5 | 5@81 |

143 configurations, 206 evaluations, 1,902 budget units, the cost of about 23 full runs. compute $s_\text{max}$ with integers (the largest $s$ with $\eta^s \le R$), not with a floating-point logarithm, which can round $\log_3 81$ to $3.9999$. when $R$ isn't a power of $\eta$, budgets are fractions ($R = 10$ starts its first bracket at $10/9$); `evaluate` must accept them (round them if it needs whole epochs). brackets run from $s_\text{max}$ down, each sampling its configurations from the same generator when it starts. hyperband's best is the lowest loss among evaluations at the full budget $R$: losses at different budgets aren't comparable (a configuration evaluated after one epoch hasn't finished learning).

> **▶ build: stage 3.** implement `hyperband(sample_config, evaluate, *, max_budget=81, eta=3, rng=None)`.

### idea 4: experiments you can trust

a search produces numbers; turning them into a conclusion needs bookkeeping and statistics.

**log every run.** keep a record of every configuration you ran, with its seed and its metrics, including the failures. it's how you avoid re-running things, how you notice that "the best" was a lucky seed, and how you write an honest report. your **`ExperimentLog(path=None)`** is a minimal tracker: `log(config, metrics, seed=None)` appends a record `{"id", "config", "seed", "metrics"}` (with plain-dict copies, and metrics as plain floats) and, if a path was given, writes it as one line of JSON to that file, so that runs survive a crashed notebook; opening an existing file loads its runs and continues the ids. `best(metric, mode='min')` returns the best record. `group(metric)` collects the runs of each distinct configuration (treating `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` as the same) and returns their count, mean and sample standard deviation (`ddof=1`; 0.0 for a single run), in order of first appearance. tools like MLflow and Weights & Biases do the same with a server and a dashboard.

**seeds are part of the experiment.** the seed of a training run picks the initial weights and the minibatch order, and two seeds of the same configuration can differ by as much as two configurations do (the lab measures it). so a single run per configuration can't rank configurations whose true difference is smaller than the seed-to-seed spread. run several seeds of the finalists, report the mean with its standard error or a confidence interval, and say how many seeds.

**the winner's curse.** the best of $k$ noisy measurements is biased upwards: if each configuration's validation score is its true quality plus noise, the maximum picks configurations that were lucky as well as good. the expected maximum of $k$ independent Gaussian noises with standard deviation $\sigma$ grows like $\sigma\sqrt{2\ln k}$ (an upper bound, reached only asymptotically: for $k = 8$ the exact value is $1.42\sigma$, not $\sqrt{2\ln 8}\,\sigma = 2.04\sigma$; on paper 4). re-evaluate the winner with fresh seeds (and, at the very end, once on the test set) to get an unbiased number. this is the same reason chapter 07 reserved the test set and introduced nested cross-validation.

> **🔮 predict:** eight configurations of *exactly* equal quality are each trained once, and the best validation accuracy is reported. if one run's accuracy varies by a standard deviation of 0.5 points from seed to seed, how much better than its true quality will the reported winner look?

> **▶ build: stage 4.** implement `ExperimentLog`.

## worked example

**hyperband with $R = 9$, $\eta = 3$.** the largest $s$ with $3^s \le 9$ is $s_\text{max} = 2$.

| bracket $s$ | $n_s = \lceil 3 \cdot 3^s/(s + 1)\rceil$ | starting budget $9/3^s$ | rungs | cost |
|---|---|---|---|---|
| 2 | $\lceil 27/3\rceil = 9$ | 1 | 9@1, 3@3, 1@9 | 9 + 9 + 9 = 27 |
| 1 | $\lceil 9/2\rceil = 5$ | 3 | 5@3, 1@9 | 15 + 9 = 24 |
| 0 | $\lceil 3/1\rceil = 3$ | 9 | 3@9 | 27 |

in total 17 configurations, 22 evaluations (13 + 6 + 3) and 78 budget units. the same 78 units buy random search 8 configurations at the full budget. in the first bracket the 3 survivors of rung 0 are the 3 lowest losses at budget 1 ($\lfloor 9/3\rfloor = 3$), and the single survivor of rung 1 is the lowest of those 3 at budget 3. the best of the 5 evaluations at budget 9 (one per bracket for $s = 2$ and $s = 1$, three for $s = 0$) is the answer. the tests check this arithmetic, the $R = 81$ table and the order in which configurations are sampled.

**a seed comparison.** two learning rates, 10 seeds each, give mean validation accuracies of 0.9033 (std 0.0056) and 0.9058 (std 0.0099). the difference, 0.0025, has a standard error of about $\sqrt{0.0056^2/10 + 0.0099^2/10} = 0.0036$: it is not distinguishable from zero with 10 seeds. comparing one run of each would pick the "right" winner barely more often than a coin (the lab counts 55% of the 100 possible pairs).

## on paper

1. **random search's guarantee.** if a fraction $q$ of the search space is "good", what's the probability that $n$ random trials include at least one good configuration? how many trials give 95% for $q = 0.05$?

<details><summary>solution</summary>

each trial misses with probability $1 - q$, independently, so all $n$ miss with probability $(1 - q)^n$ and at least one hits with probability $1 - (1 - q)^n$. for $q = 0.05$: $0.95^n \le 0.05$ needs $n \ge \ln 0.05/\ln 0.95 = 58.4$, so 59 trials. the dimension of the space doesn't appear: that's why random search scales to many hyperparameters while a grid's cost grows exponentially.

</details>

2. **grids waste trials.** a function of two hyperparameters depends only on the first. compare the number of distinct values of the first that a $k \times k$ grid and $k^2$ random trials evaluate.

<details><summary>solution</summary>

the grid tests $k$ distinct values of the first hyperparameter, each $k$ times (the repeats only vary the irrelevant one). $k^2$ random trials draw $k^2$ distinct values (with probability 1, for a continuous distribution). for $k = 3$: 3 versus 9. in $d$ dimensions with one that matters, a grid of $k^d$ trials still tests only $k$ values.

</details>

3. **the cost of successive halving.** with $n = \eta^m$ configurations and minimum budget $r$, show that every rung costs $nr$ and the whole search costs $nr(m + 1)$. compare with evaluating all $n$ at the full budget $r\eta^m$.

<details><summary>solution</summary>

rung $i$ has $n\eta^{-i}$ configurations at budget $r\eta^i$: cost $nr$. there are $m + 1$ rungs ($i = 0, \dots, m$), so $nr(m + 1)$ in total. evaluating all at the full budget costs $n \cdot r\eta^m = nr \cdot n$. the ratio is $(m + 1)/n = (\log_\eta n + 1)/n$: for $n = 81$, $\eta = 3$, 5/81, about 16 times cheaper.

</details>

4. **the winner's curse.** $k$ configurations have the same true validation loss $\mu$; each measured loss is $\mu + \epsilon_i$ with $\epsilon_i \sim N(0, \sigma^2)$ independent. what's the expected *minimum* measured loss for $k = 2$? (for $k = 8$ it's about $\mu - 1.42\sigma$.)

<details><summary>solution</summary>

for two independent standard normals, $E[\max(Z_1, Z_2)] = 1/\sqrt\pi \approx 0.564$ (because $\max = \frac{Z_1 + Z_2}{2} + \frac{|Z_1 - Z_2|}{2}$ and $Z_1 - Z_2 \sim N(0, 2)$, so $E|Z_1 - Z_2| = 2/\sqrt\pi$). by symmetry the expected minimum loss is $\mu - 0.564\sigma$: even with two identical configurations, the winner looks better than it is. with $k = 8$, $1.42\sigma$. a search that picks the best of many configurations reports an optimistic number unless the winner is re-evaluated on fresh seeds and data.

</details>

## experiment

`lab.py` uses your `ExperimentLog`, `random_search` and `hyperband` on a small MNIST network:

1. two learning rates, 10 seeds each: how often does a single run pick the right one?
2. where the seed-to-seed variance comes from: the initialization or the minibatch order.
3. hyperband against random search with the same compute, repeated with three search seeds, with the winners re-trained on fresh seeds.

## check your understanding

1. why does random search beat grid search in practice, even though the grid covers the space "evenly"?

<details><summary>answer</summary>

because of low effective dimensionality: usually only a few hyperparameters matter much, and you don't know which. a grid evenly covers *combinations*, but projects onto each important axis as only $k$ distinct values, repeated; random search tests a new value of every hyperparameter in every trial.

</details>

2. why is a learning rate sampled log-uniformly while a dropout rate is usually sampled uniformly?

<details><summary>answer</summary>

a learning rate's effect is multiplicative: halving it from $10^{-2}$ or from $10^{-4}$ are similar changes, so the natural scale is $\log\eta$. a dropout rate lives on a bounded interval where differences are roughly additive (0.1 versus 0.2 is similar to 0.4 versus 0.5), so a uniform draw on $[0, 0.6]$ is reasonable.

</details>

3. what's the main failure mode of successive halving, and how does hyperband hedge it?

<details><summary>answer</summary>

eliminating slow starters: configurations that look bad at a small budget but would be best at the full budget (learning curves cross). hyperband runs several brackets with different starting budgets, down to one that evaluates everything at the full budget (random search), so if early stopping is misleading for this problem, the conservative brackets still find good configurations.

</details>

4. why does hyperband choose its best among the full-budget evaluations only?

<details><summary>answer</summary>

a loss after one epoch and a loss after 81 epochs measure different things; a configuration's early loss is almost always worse than its final loss. comparing across budgets would favour whatever got the most training, not the best configuration. every bracket ends at the full budget, so there are always full-budget evaluations to choose from.

</details>

5. your search's best configuration scored 0.277 in the search and 0.297 when re-trained with fresh seeds. was the re-training worse, or the search optimistic?

<details><summary>answer</summary>

most likely the search was optimistic: the winner of several noisy evaluations is selected partly for its luck (the winner's curse), so its search score is biased low (for a loss). the fresh-seed number is the unbiased estimate of how that configuration performs.

</details>

6. why does `ExperimentLog` write each run to the file immediately rather than saving everything at the end?

<details><summary>answer</summary>

so that a crash, a kernel restart or an interrupted search doesn't lose the runs already completed (which may represent hours of compute). an append-only log is also a simple audit trail: you can't quietly forget the runs that went badly.

</details>

## common bugs

- **uniform sampling of scale parameters.** symptom: almost every trial has a learning rate within a factor of 10 of the maximum, and the search "finds" that large learning rates are best.
- **floating-point $s_\text{max}$.** `int(math.log(81, 3))` is 3 on many machines, not 4. compute it with integer powers.
- **choosing across budgets.** symptom: hyperband returns a configuration that was only evaluated at the full budget because everything it competed with was judged at budget 1. pick among full-budget evaluations.
- **a new generator per sample.** calling `as_rng(seed)` inside the loop gives the same configuration every trial.
- **one seed per configuration, reported as a ranking.** symptom: conclusions that flip when someone reruns your code. report the spread, or the standard error.
- **selecting on the test set.** symptom: an optimistic final number that doesn't hold up on new data. search on validation; touch the test set once.

## going deeper

- Bergstra & Bengio (2012), "Random search for hyper-parameter optimization", *JMLR* 13: 281–305. [jmlr.org/papers/v13/bergstra12a](https://jmlr.org/papers/v13/bergstra12a.html)
- Jamieson & Talwalkar (2016), "Non-stochastic best arm identification and hyperparameter optimization", *AISTATS*. successive halving. [arXiv:1502.07943](https://arxiv.org/abs/1502.07943)
- Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2017), "Hyperband: a novel bandit-based approach to hyperparameter optimization", *JMLR* 18(185): 1–52. [arXiv:1603.06560](https://arxiv.org/abs/1603.06560)
- Falkner, Klein & Hutter (2018), "BOHB: robust and efficient hyperparameter optimization at scale", *ICML*. hyperband with Bayesian optimization instead of random sampling. [arXiv:1807.01774](https://arxiv.org/abs/1807.01774)
- Bouthillier, Delaunay, Bronzi, Trofimov, Nichyporuk, Szeto, Sepah, Raff, Madan, Michalski, Arbel, Pal, Varoquaux & Vincent (2021), "Accounting for variance in machine learning benchmarks", *MLSys*. how large the sources of variance are, and how to compare fairly. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), "Deep reinforcement learning that matters", *AAAI*. seeds that change conclusions. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)

## connections

- ← builds on: validation sets, nested CV and grid search (ch 07 · topic 06), confidence intervals and standard errors (ch 05), every knob in topics 01–05, bandits (successive halving is a best-arm identification algorithm; ch 23 · topic 01).
- → used in: the capstone's honest report (several seeds, a baseline, an interval), CIFAR-10 ablations (ch 16), the multi-seed comparisons of `mlfs.research` (ch 30), and every "which is better?" question from here on.
