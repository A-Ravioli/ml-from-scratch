# thesis project

## why this matters

the thesis is the last project of the course and the first that's entirely yours: a question you chose (topic 04), answered by original work, written up as a paper (topic 05), and, if it turns out well, submitted to a workshop. the capstone describes the deliverables and how they're judged. this topic is about the part that sinks more theses than bad ideas do: **time**.

almost everyone underestimates how long their own work will take, even when they know they usually do. Buehler, Griffin & Ross (1994) asked students how long their honours thesis would take: they predicted about 34 days on average and took about 56, later than even their own "worst case" predictions on average. the reasons are structural, not moral. tasks have long right tails (a baseline that won't reproduce, a bug that takes a week), the tail of one task delays everything after it, and work done in parallel finishes when its *slowest* branch does. a plan made of "likely" durations is a plan for the best day of every task at once.

project managers have had tools for this since the late 1950s: the **critical path method** (Kelley & Walker, 1959), **three-point estimates** (PERT; Malcolm et al., 1959), and Monte Carlo simulation of the plan. this topic builds them for a thesis-sized plan, and one more: a forecast calibrated on *your own* history of estimates and actual durations. by the end you'll have a plan that tells you, honestly, when you'll finish, which tasks to protect, and when to cut scope.

## warm-up: predict first

1. a plan of 8 tasks in a row, each estimated as "likely 2 weeks, but anywhere from 1.5 to 5". you set the deadline at 16 weeks. roughly what's the chance you make it: 50%, 25%, or under 5%?
2. four experiments that each "likely take 3 weeks" run in parallel. when are all four done, likely?
3. your pilot has 4 weeks of slack in the plan. what does that mean, and which tasks have none?
4. you've logged 15 finished tasks; on average they took 1.6× your estimate. how should that change your thesis deadline?

## prerequisites check

- [ ] you can sort a task graph topologically with Kahn's algorithm (→ topic 01, `reading_order`)
- [ ] you have a proposal with experiments, checkpoints and a budget (→ topic 04)
- [ ] you can estimate a probability and quantiles by Monte Carlo (→ ch 01 · topic 05, ch 05 · topic 01)
- [ ] you know what a distribution's mode, median and mean are, and why they differ for skewed distributions (→ ch 04)

## core ideas

### idea 1: the plan as a graph, and its critical path

a plan is a set of **tasks**, each with a duration and a list of tasks that must finish before it can start. "write the paper" waits for "figures", which waits for "ablations" and "second dataset", which wait for "main experiments", and so on. that's a directed acyclic graph, and two passes over it in topological order answer the planning questions.

the **forward pass** computes, for each task in topological order, its earliest start (the latest finish among the tasks it waits for, or 0) and earliest finish (start + duration). the plan's total length is the largest finish. the **backward pass**, in reverse topological order, computes each task's latest finish that doesn't delay the whole plan: the plan's total for tasks nothing waits on, otherwise the smallest latest *start* among the tasks waiting on it. a task's **slack** is latest finish − earliest finish: how long it can slip for free. the tasks with zero slack form the **critical path**, a chain from a first task to a last one whose durations add up to the total. a week lost on the critical path is a week lost on the thesis; a week lost on a task with four weeks of slack costs nothing.

the practical uses are immediate: protect the critical path (start it first, give it your best hours, look for ways to shorten it), and fill slack with work that can move. it's also the first place to look when cutting scope: a task on the critical path that isn't essential (a second dataset, say) is the most valuable thing to drop.

> **spec.** `topo_order(tasks)`: `tasks` maps a name to a dict with an optional `after` list. returns all names, each after every task in its `after` list, taking the alphabetically first ready task at each step (repeated entries in `after` count once). `ValueError` for an unknown task in an `after` list, or a cycle (naming the tasks in it). `critical_path(tasks)`: every task also has `weeks`, a positive number (not a bool; `ValueError` otherwise). returns a dict with `start` and `finish` (earliest, per task), `slack` (latest finish − earliest finish), `total` (the largest finish) and `path`: the critical path from first task to last, found by starting at the alphabetically first task whose finish equals the total and repeatedly stepping to the alphabetically first task in its `after` list whose finish equals its start (use `np.isclose` for these comparisons).

> **▶ build: stage 1.** implement `topo_order` and `critical_path`.

### idea 2: uncertain durations

a single number per task hides what you actually know: "probably 3 weeks, at best 2, and if the baseline doesn't reproduce, 8". **three-point estimates** keep it: optimistic $o$, most likely $m$ and pessimistic $p$. the simplest distribution with those three numbers is the **triangular** distribution, with density rising linearly from $o$ to a peak at $m$ and falling linearly to $p$. two of its properties explain most schedule overruns:

- **the mode isn't the mean.** research tasks are right-skewed ($p - m \gg m - o$), and then the mean, $(o + m + p)/3$, sits well above $m$, and so does the median. a task estimated at $(1, 2, 4)$ finishes by its likely value only with probability $(m - o)/(p - o) = 1/3$. a chain of such tasks adds up the gaps between means and modes.
- **merge bias.** a task that waits for several others starts when the *last* of them finishes, and the maximum of several uncertain durations is larger than any one of them is likely to be. four parallel tasks that each finish by their likely value with probability 1/4 all finish by it with probability $(1/4)^4 \approx 0.004$ (MacCrimmon & Ryavec, 1964, showed that the original PERT method's analytic shortcut ignores exactly this).

neither effect has a tidy formula for a whole graph, but Monte Carlo is easy: draw every task's duration, run the forward pass, repeat 10,000 times, and read off the probability of finishing by a deadline and the quantiles of the finish date. the backward pass, run per simulation, also tells you each task's **criticality**: the fraction of simulations in which it was on the critical path. it ranks the tasks to watch better than the single likely plan, because a task with slack in the likely plan can be critical whenever its tail is long.

> **spec.** `simulate_makespan(tasks, durations)`: `durations` maps every task (exactly the tasks in `tasks`; `ValueError` otherwise) to a 1-D array of the same length $S$ (`ValueError` otherwise), one sampled duration per simulation. run the forward and backward passes of idea 1 on all simulations at once (vectorized over $S$). returns `(makespan, critical)`: the total of each simulation (shape $(S,)$), and a dict mapping each task to a bool array, True where its slack is 0 (within $10^{-9} \max(1, \text{makespan})$). `schedule_risk(tasks, deadline, *, n_sims=10000, rng=None)`: every task has an `estimate` $(o, m, p)$ with $0 < o \le m \le p$ (`ValueError` otherwise). with `rng = as_rng(rng)`, for each task *in the order of `tasks`*, draw `rng.triangular(o, m, p, size=n_sims)` (or, if $o = p$, use the constant $o$ without drawing). returns a dict with `p_on_time` (fraction of makespans $\le$ `deadline`), `quantiles` (`{0.5: ..., 0.8: ..., 0.95: ...}` of the makespan, `np.quantile`), `criticality` (per task, the fraction of simulations in which it's critical) and `likely_total` (`critical_path`'s total with every task at $m$).

> **🔮 predict:** a ten-task thesis plan, 18 weeks if every task takes its likely duration. what's the probability of finishing in 18 weeks?

> **▶ build: stage 2.** implement `simulate_makespan` and `schedule_risk`.

### idea 3: calibrate on your own history

three-point estimates help, but they're still your estimates, made from the "inside view": thinking about this task's steps. Kahneman and Tversky's remedy for the planning fallacy is the **outside view**: ignore the details of the task and ask how long tasks *like* it took in the past. Flyvbjerg (2006) turned it into *reference class forecasting* for large projects; Spolsky (2007) described a version for individual programmers, *evidence-based scheduling*, which is what you'll build. the idea fits in one line: log every task you finish as (estimate, actual), and forecast each new task's duration as its estimate times a ratio drawn at random from your log.

the resampling does three things a fixed correction factor can't. it captures your *typical* bias (if you usually take 1.6× your estimate, so will the forecast), its *spread* (some tasks take 0.8×, some 3×), and its tail, without assuming a distribution. run through the task graph, it gives the same outputs as idea 2: a probability of finishing by a date and quantiles of the finish date. it needs a history (a dozen tasks is a start; the lab shows how much the forecast moves with 5 vs 15 vs 40) from similar work: estimates you made for coursework calibrate research tasks only loosely, so log thesis tasks from the first week.

> **spec.** `forecast(tasks, history, deadline, *, n_sims=10000, rng=None)`: every task has a single positive `estimate` (not a bool; `ValueError` otherwise). `history` is a list of `(estimate, actual)` pairs, both positive (`ValueError` otherwise, or if it's empty); the ratios are `actual / estimate`. with `rng = as_rng(rng)`, for each task in the order of `tasks`, draw `rng.choice(ratios, size=n_sims, replace=True)` and multiply by the task's estimate; run `simulate_makespan`. returns a dict with `p_on_time`, `quantiles` (as in `schedule_risk`), `naive_total` (`critical_path`'s total at the estimates) and `median_ratio`.

> **🔮 predict:** your history says your tasks take a median 1.6× their estimate. with an 18-week plan, when are you 80% sure to finish?

> **▶ build: stage 3.** implement `forecast`.

### the shape of a thesis

the tools above are only useful inside a working routine. a few habits that keep a thesis on track, each tied to something you've built:

- **a written plan from week one**, as a task graph with three-point estimates, rebuilt whenever reality diverges. the proposal's checkpoints (topic 04) are its milestones.
- **a weekly log**: what you did, what you learned, every finished task's (estimate, actual), and next week's plan. it feeds `forecast`, and it's your lab notebook when you write the paper.
- **write while you work.** the paper's introduction is the proposal; its method section is written when the method is implemented; its figures are generated by code from the run registry (the capstone builds it), never pasted by hand. a thesis whose writing starts after the experiments has a writing task at the end of the critical path with a very long tail.
- **cut scope early, on the critical path.** when the 80% date passes the deadline, drop or shrink a critical task that isn't essential (a second dataset, an extra ablation), not seeds (topic 04), and not the write-up.
- **know what "done" means**: the capstone's rubric. a thesis that answers its question with a clear negative result, honestly reported, is done; one that keeps adding experiments in search of a positive one isn't.

## worked example

**the critical path.** read (4 weeks) → reproduce (6, after read) and pilot (2, after read) → experiments (8, after both) → write (4) and figures (1), both after experiments. forward pass: read 0–4; reproduce 4–10; pilot 4–6; experiments 10–18; write 18–22; figures 18–19. total 22 weeks. backward pass: write and figures must finish by 22, so experiments by 18 (write's latest start); reproduce and pilot by 10. slack: pilot 10 − 6 = 4, figures 22 − 19 = 3, everything else 0. critical path: read → reproduce → experiments → write.

**a triangular task.** estimate $(1, 2, 4)$: $P(T \le 2) = (2 - 1)/(4 - 1) = 1/3$. its median $x$ satisfies $1 - (4 - x)^2/((4 - 1)(4 - 2)) = 1/2$, so $x = 4 - \sqrt 3 = 2.268$; its mean is $7/3 = 2.333$. two such tasks in parallel are both done by week 2 with probability $1/9$.

**a forecast.** a 4-week task and a history with ratios 1 and 2: the forecast duration is 4 or 8 weeks with equal probability, so $P(\text{done by } 6) = 0.5$. two 1-week tasks in a row with ratios 1 and 3: both must draw 1 to finish by week 2, probability $1/4$. your code must reproduce all of these.

## on paper

1. **the triangular distribution.** for the triangular distribution on $[o, p]$ with mode $m$, show that $P(T \le m) = (m - o)/(p - o)$ and that $E[T] = (o + m + p)/3$. for $(2, 3, 8)$ (the lab's "reproduce baseline"), how much later than the likely value is the mean?

<details><summary>solution</summary>

the density is a triangle of height $h = 2/(p - o)$ (so its area is 1) with its peak at $m$. the area left of $m$ is $\tfrac12 (m - o) h = (m - o)/(p - o)$. the mean of a triangle's distribution is its centroid, the average of the three vertices' positions: $(o + m + p)/3$ (integrate $t f(t)$ over the two linear pieces to check). for $(2, 3, 8)$: $P(T \le 3) = 1/6$ and the mean is $13/3 = 4.33$ weeks, 1.33 weeks after the likely 3. summed over a critical path of eight such tasks, those gaps are most of the lab's overrun.

</details>

2. **merge bias.** $k$ independent parallel tasks each finish by time $m$ with probability $F(m)$. what's the probability all $k$ do? for estimates $(2, 3, 6)$, evaluate it for $k = 1, 2, 4, 8$.

<details><summary>solution</summary>

independence gives $F(m)^k$. with $F(3) = (3 - 2)/(6 - 2) = 1/4$: $0.25$, $0.0625$, $0.0039$, $1.5 \times 10^{-5}$. the likely plan says "3 weeks" for every $k$; the probability that it's right collapses geometrically, and the finish date's quantiles drift later as $k$ grows. (the lab's simulated values: 0.252, 0.0626, 0.00425, 0.00005.)

</details>

3. **why slack is zero along a chain.** show that the critical path's durations sum to the plan's total, and that every task on it has zero slack.

<details><summary>solution</summary>

the path starts at a task whose finish equals the total and steps back to a predecessor whose finish equals the current task's start; it ends at a task with start 0. along it, each task's start is its predecessor's finish, so the durations telescope: $\sum (\text{finish} - \text{start}) = \text{total} - 0$. for slack: the last task has latest finish = total = its finish. if a task on the path has latest finish = finish, its predecessor on the path has latest finish $\le$ the task's latest start = its start = the predecessor's finish; and latest finish $\ge$ finish always (the backward pass only ever subtracts real durations from the total), so it's equal, and by induction every task on the path has zero slack.

</details>

## experiment

the lab runs your planner on a ten-task thesis plan with realistic three-point estimates: how likely is the "likely" plan, which tasks are critical in most simulations, and how much padding buys 80% confidence. then it shows merge bias with 1 to 8 parallel tasks, and forecasts the plan from a (synthetic) history of estimates and actual durations, with a slider for how long that history is.

## check your understanding

1. a task has 4 weeks of slack in the likely plan. can it still delay the thesis?

<details><summary>answer</summary>

yes: slack is computed with likely durations. if the task's pessimistic duration exceeds its likely one by more than 4 weeks, or the tasks it runs alongside finish early, it can become critical. the Monte Carlo criticality index measures how often that happens (the lab's pilot is critical in about 1% of simulations; with a longer tail it would be more).

</details>

2. why does running experiments in parallel help even though merge bias makes the parallel block finish later than its likely duration?

<details><summary>answer</summary>

the alternative is running them in series, whose finish is the *sum* of their durations, not the maximum. parallel is still much faster; merge bias only says the parallel block finishes later than the likely value of any one branch. plan for the parallel block's 80% date, not its likely date.

</details>

3. your 80% finish date is 6 weeks past the deadline. list the options in the order you'd consider them.

<details><summary>answer</summary>

shorten the critical path (cut or shrink an inessential task on it, start a critical task earlier, run a critical task in parallel with something that doesn't depend on it), then cut non-critical scope that will become critical, then negotiate the deadline. don't cut seeds or the write-up, and don't "plan to work harder": the plan already assumed your usual pace.

</details>

4. why resample ratios from your history rather than multiply every estimate by the history's mean ratio?

<details><summary>answer</summary>

a fixed factor shifts the forecast but gives it no spread: every task overruns by exactly the average, and the forecast is a single date with false certainty. resampling keeps the variability of your overruns (and their skew), which is what makes some finish dates likely and others not, and it lets merge bias act on realistic durations.

</details>

5. a friend's history says ratios have a median of 1.0. should they plan with their estimates?

<details><summary>answer</summary>

check the spread first: a median of 1 with ratios from 0.5 to 4 still makes long tails, and the forecast's 80% date can be well past the naive total because of the right tail and merge bias. also check the reference class: if the history is from short, familiar tasks and the thesis is long and novel, the history probably understates the overrun.

</details>

## common bugs

- **adding branch durations instead of taking the maximum.** a task starts when the *last* of its predecessors finishes: `np.maximum`, not `+`.
- **the backward pass in forward order.** latest finishes depend on successors, so they must be computed in reverse topological order.
- **one ratio for the whole plan.** drawing a single ratio per simulation and multiplying the whole plan by it makes every task overrun together; draw per task.
- **looping over simulations in Python.** 10,000 simulations × 10 tasks is fine vectorized over simulations (`np.maximum` on arrays) and slow as nested loops.
- **planning with the mode.** "likely" is the most common outcome of one task, not the expected total of a plan. report quantiles of the simulated finish date.
- **a deadline set by the plan's likely total.** the lab shows it's met about once in two thousand plans (P = 0.0005).

## going deeper

- Buehler, Griffin & Ross (1994), "Exploring the 'planning fallacy': why people underestimate their task completion times", *Journal of Personality and Social Psychology* 67(3): 366–381. [doi:10.1037/0022-3514.67.3.366](https://doi.org/10.1037/0022-3514.67.3.366)
- Kelley & Walker (1959), "Critical-path planning and scheduling", *Proceedings of the Eastern Joint Computer Conference*, 160–173. [doi:10.1145/1460299.1460318](https://doi.org/10.1145/1460299.1460318)
- Malcolm, Roseboom, Clark & Fazar (1959), "Application of a technique for research and development program evaluation", *Operations Research* 7(5): 646–669. the original PERT paper. [doi:10.1287/opre.7.5.646](https://doi.org/10.1287/opre.7.5.646)
- MacCrimmon & Ryavec (1964), "An analytical study of the PERT assumptions", *Operations Research* 12(1): 16–37. merge bias. [doi:10.1287/opre.12.1.16](https://doi.org/10.1287/opre.12.1.16)
- Flyvbjerg (2006), "From Nobel Prize to project management: getting risks right", *Project Management Journal* 37(3): 5–15. reference class forecasting.
- Spolsky (2007), "Evidence based scheduling". [joelonsoftware.com/2007/10/26/evidence-based-scheduling](https://www.joelonsoftware.com/2007/10/26/evidence-based-scheduling/)
- Kahneman (2011), *Thinking, Fast and Slow*, Farrar, Straus and Giroux, chapter 23 ("the outside view").

## connections

- ← builds on: topic 01 (topological order with Kahn's algorithm), topic 04 (the proposal's experiments, checkpoints and budget), ch 01 · topic 05 and ch 05 (Monte Carlo estimates and quantiles), ch 04 (skewed distributions).
- → used in: the thesis capstone, whose brief asks for a plan built with these tools and whose tooling (project skeleton, run registry, generated report) keeps the thesis reproducible; and every project after the course.
