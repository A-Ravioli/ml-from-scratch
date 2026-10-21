# thesis project · lab notes

numbers below are from the reference solution; every experiment is a simulation with a fixed seed (20,000 simulated plans unless stated), so they're exactly reproducible. the lab runs in under a second. the thesis plan's estimates are invented but typical; the history in experiment 3 is synthetic.

## experiment 1: the plan everyone writes

**what people predict.** that the likely plan (18 weeks) has something like a 30–50% chance, since "likely" sounds like "more likely than not"; and that a couple of weeks of padding fixes it.

**what happens.** P(done by 18 weeks) = **0.0005**: about one plan in 2,000. the simulated finish date has median 24.9 weeks, 80% quantile **27.3** and 95% quantile 29.5. the padding needed: +4 weeks gives 12%, +7 gives 51%, +9 gives 77%, +12 gives 97%. criticality: read & annotate, proposal, main experiments, figures, write paper and revise are on the critical path in 100% of simulations, reproduce baseline in 99%, the second dataset in 58% and the ablations in 42% (they run in parallel and trade places), the pilot in 1%.

**why.** every task's estimate is right-skewed: the pessimistic value is much further from the likely one than the optimistic value is (reproduce baseline is (2, 3, 8): finishing by 3 weeks has probability 1/6, and its mean is 4.33). along a critical path of eight tasks, the gaps between the likely values and the means add up to about 7 weeks, which is where the median of 24.9 comes from. merge bias adds a little more where two branches meet (ablations and second dataset). so the likely plan isn't a plan you'll probably meet; it's the plan in which every task has a good week, and the probability of that is a product of many smallish numbers. the criticality index agrees with the likely plan's critical path here, with one useful extra: the ablations and the second dataset share the risk, so shortening only one of them barely helps; cutting the second dataset (58% critical) and simplifying the ablations is the scope cut that moves the date.

## experiment 2: work in parallel, finish later?

**what people predict.** that parallel tasks finish when one task would, "3 weeks"; at most a small delay.

**what happens.**

| parallel tasks | likely plan (weeks) | P(done by 3) | median | 80% sure |
|---|---|---|---|---|
| 1 | 3 | 0.252 | 3.55 | 4.45 |
| 2 | 3 | 0.0626 | 4.12 | 4.89 |
| 4 | 3 | 0.00425 | 4.62 | 5.19 |
| 8 | 3 | 0.00005 | 5.01 | 5.42 |

the likely plan says 3 weeks for every $k$; the probability of meeting it falls as $(1/4)^k$ (on paper, problem 2: 0.25, 0.0625, 0.0039, $1.5 \times 10^{-5}$), and the median finish moves from 3.55 to 5.01 weeks.

**why.** the block finishes when its slowest task does, and the maximum of $k$ independent right-skewed durations creeps towards the pessimistic end as $k$ grows (*merge bias*). each extra parallel task is one more chance that something hits its tail. running tasks in parallel is still much faster than running them in series (which would take about $k$ times 3.67 weeks on average); the lesson is to plan the parallel block for its 80% date, not its likely one. a thesis plan that waits for "all experiments" before writing starts puts a merge point right before its longest task, which is one more reason to write while you work.

## experiment 3: calibrating on your own history

**what people predict.** that a median ratio of 1.6 turns 18 weeks into about 29 (18 × 1.6), and that a handful of logged tasks is enough to calibrate.

**what happens.** with 15 logged tasks (median ratio 1.67), the forecast gives P(done by 18 weeks) = **0** (none of 20,000 simulations), a median of **35.7** weeks, an 80% date of **40.6** and a 95% date of 45.6. across 20 other synthetic histories of 15 tasks from the same distribution, the 80% date ranges from 27.5 to 51.1 weeks; with 5 tasks from 28.6 to 65.2; with 40 tasks from 32.5 to 41.7 (the original 5- and 40-task histories give 80% dates of 39.4 and 37.0 weeks, run separately with the lab's code).

**why.** the forecast is well beyond 18 × 1.6 for two reasons. the ratios are skewed (log-normal): their *mean* is about 1.77 even though their median is 1.6, and the plan's total adds up per-task durations, so it scales with the mean, not the median. and each task draws its own ratio, so the plan inherits the tails and merge bias of experiments 1 and 2. the instability with short histories is ordinary sampling noise: 5 ratios are a poor picture of a skewed distribution, and one lucky or unlucky task moves the forecast by weeks. the 40-task forecasts agree to within about ±5 weeks. as for the risk in the explain prompt: coursework tasks are short, well specified and familiar, so their ratios are smaller and less spread out than research tasks'. a forecast calibrated on them is too optimistic, which is why the lesson asks you to start logging thesis tasks in week one. the synthetic history here is loosely modelled on Buehler et al.'s thesis-writers (34 days predicted, 56 taken: a ratio of about 1.6); your own numbers are the ones that matter.
