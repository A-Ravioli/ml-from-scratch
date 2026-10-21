# finding questions

## why this matters

the hardest decision in a research project comes first, and it's made with the least information: *what question to work on*. most failed student projects fail here, in one of three ways. the question is too broad to have an answer ("explore how normalization affects training"). it has an answer, but not one the budget can reach (detecting a 0.2-point improvement with three seeds, or a scaling law with a laptop). or it was chosen because a quick pilot looked spectacular, and the spectacular part was luck.

Hamming (1986) asked his colleagues at Bell Labs what the important problems in their field were, and then why they weren't working on them. "important" isn't the only criterion, though: a thesis question must also be *answerable by you, with your compute, in your time*, and it must be stated so that an experiment could prove it wrong. this topic is about getting from a gap you noticed (a reproduction that only partly held, an anomaly in a lab, a limitation a paper admits) to a question of that kind, and about writing it as a two-page proposal whose falsifiability, statistical power and compute budget you can check mechanically before you spend a week on it.

## warm-up: predict first

1. "we will explore whether data augmentation improves robustness." what's wrong with this as a hypothesis?
2. your pilot shows a paired seed-to-seed sd of 0.65 points. to detect a true improvement of 0.2 points with 80% power, how many paired seeds do you need: 5, 20, or 70?
3. you have a week of laptop time. what's the largest language model you can train *compute-optimally*: 5M, 50M or 500M parameters?
4. you pilot 20 ideas with one seed each and pick the best. it showed +1 point. what do you expect when you rerun it with 20 seeds?

## prerequisites check

- [ ] you've written a reproducibility report, or know what its discrepancy section contains (→ topic 02, R1)
- [ ] you can compute a paired $t$ interval and know what power is (→ ch 05 · topic 04, topic 03)
- [ ] you know $C \approx 6ND$ and the Chinchilla rule, and have `mlfs.dltheory.compute_optimal` (→ ch 19 · topic 05)
- [ ] you know what FLOP/s and utilization mean for a training run (→ ch 26 · topic 01)

## core ideas

### idea 1: from a gap to a falsifiable hypothesis

good questions rarely arrive from nowhere; they arrive from friction. some reliable sources:

- **a reproduction that only partly held.** the effect was real but smaller, or appeared only at one learning rate (topic 02's lab). *why* is a question.
- **an anomaly in your own experiments.** in topic 03's lab, varying only the data order produced the heaviest tail of bad runs. is that a property of momentum SGD at that learning rate? of the dataset? at what scale does it go away?
- **a paper's limitations and unstated assumptions.** your pass-3 notes (topic 01) list them. "we only test on image classification" is an invitation.
- **moving a claim to a new regime**: smaller data, longer sequences, another modality, a different optimizer.

a question becomes a project when it's sharpened into a **hypothesis**: a statement that a specific experiment could show to be false. Popper's criterion (1959) is the useful one here: a claim that no observation could contradict says nothing. "we will explore whether label smoothing helps" can't fail; "label smoothing of 0.1 raises the test accuracy of a 2-layer MLP on 5,000 MNIST images by at least 0.3 points after 3 epochs" can. the second names everything the experiment needs:

- the **method** and the **baseline** it's compared against (not itself);
- the **metric** and the **setting** (data, model, budget);
- the **direction** of the effect and the **smallest effect that matters** (`min_effect`), chosen for what it would mean, not from a pilot (idea 2 shows why);
- what result would **refute** it, stated in advance;
- the **number of seeds** you plan to run.

writing all this *before* the experiment is a light form of pre-registration (Nosek et al., 2018). it protects you from the two classic ways of fooling yourself: changing the hypothesis to fit the results (HARKing, Kerr, 1998) and trying analyses until one works (Gelman & Loken's "garden of forking paths", 2013).

> **spec.** `check_hypothesis(h)` → a list of problem strings, each starting with the field and `': '` (empty = falsifiable as stated). `statement`, `metric`, `method`, `baseline`, `setting` and `refuted_if` must be non-empty strings; the statement must not match `VAGUE` (a regex given in the setup cell: activity words like *explore*, *investigate*, *study*, *understand*, and hedges like *may*, *might*, *could*); the problem names the matches, lowercased and sorted. `baseline` must differ from `method` (ignoring case and surrounding space); `direction` is `'greater'` or `'less'`; `min_effect` is a finite positive number (not a bool); `n_seeds` an integer $\ge 2$ (not a bool). one problem per broken field; never modify `h`.

> **▶ build: stage 1.** implement `check_hypothesis`.

### idea 2: power, and why pilots lie

a falsifiable hypothesis can still be untestable in practice. **power** is the probability that your experiment detects an effect of a given size if it's really there. for a paired comparison over $n$ seeds with differences of standard deviation $\sigma_d$ and true mean $\Delta$, the $t$ statistic $\bar d / (s_d/\sqrt n)$ follows a **noncentral** $t$ distribution with $n - 1$ degrees of freedom and noncentrality $\lambda = \Delta\sqrt n/\sigma_d$. the one-sided test at level $\alpha$ rejects when the statistic exceeds $t_{1-\alpha, n-1}$, so

$$\text{power}(n) = P\big(T_{n-1,\lambda} > t_{1-\alpha,\,n-1}\big),$$

and the number of seeds you need is the smallest $n$ with power at least 80% (or whatever you choose). the effect enters as $\Delta/\sigma_d$, so halving the effect you care about roughly quadruples the seeds. this differs from topic 02's `seeds_needed` only in the question: there you wanted to show a result *equals* the paper's within a margin; here, that an effect *exists*.

where does $\sigma_d$ come from? a **pilot**: a few paired runs whose only job is to measure the noise. it's tempting to also use the pilot's *effect* as the `min_effect`. don't. small, noisy experiments that happen to clear a bar overestimate effects systematically, the **winner's curse** (Button et al., 2013; Gelman & Carlin, 2014, call the ratio of the estimate to the truth a *type M* error). the same happens when you choose among ideas: pilot 20 ideas with one seed each and the best-looking one is mostly the luckiest one. if true effects scatter with sd $\tau$ and a pilot's noise has sd $\sigma$, the expected true effect given an observation $x$ is only $\tfrac{\tau^2}{\tau^2 + \sigma^2}\,x$ (on paper, problem 3). Sculley et al. (2018) argue that the field as a whole selects its published results this way.

> **spec.** `paired_power(sd_diff, effect, n, *, alpha=0.05, alternative='greater')`: the power above, with `scipy.stats.nct` (noncentrality `effect * sqrt(n) / sd_diff`, `n - 1` dof). `'less'` uses the lower tail, $P(T < -t_{1-\alpha})$; `'two-sided'` sums $P(T > t_{1-\alpha/2}) + P(T < -t_{1-\alpha/2})$. `ValueError` for `n < 2`, `sd_diff <= 0` or another alternative. `seeds_for_power(sd_diff, min_effect, *, alpha=0.05, power=0.8, alternative='greater', max_seeds=1000)`: the smallest $n \ge 2$ (up to and including `max_seeds`) with `paired_power(...) >= power`; `ValueError` if none, or if `power` isn't in $(0, 1)$.

> **🔮 predict:** with a paired sd of 0.65 points, what's the power of a 3-seed comparison to detect a true 0.3-point improvement?

> **▶ build: stage 2.** implement `paired_power` and `seeds_for_power`.

### idea 3: compute, before you commit

the second feasibility check is compute. for a dense transformer, training costs about $C \approx 6ND$ floating-point operations for $N$ parameters and $D$ tokens (2 for the forward multiply-adds, 4 for the backward; Kaplan et al., 2020; ch 19 · topic 05). a machine with peak throughput $F$ FLOP/s, running at **utilization** $u$ (the fraction of peak a real training loop achieves; 30–50% is typical for well-tuned GPU code, and the lab measures what plain matrix multiplies reach on your laptop), takes

$$\text{hours} = \frac{6ND}{u F \cdot 3600}.$$

inverting it tells you what you can afford. Hoffmann et al. (2022) found that for a fixed $C$, loss is minimized with roughly $D \approx 20N$, which your `mlfs.dltheory.compute_optimal` implements: $N = \sqrt{C/(6 \cdot 20)}$. so a budget in hours becomes a largest sensible model. the 6ND rule counts only the big matrix multiplies (no attention scores, data loading, evaluation or checkpointing), so real runs take longer; treat it as a lower bound.

> **spec.** `run_hours(n_params, n_tokens, *, flops_per_second, utilization=0.4)` → $6ND/(uF)/3600$ as a float. `largest_model(hours, *, flops_per_second, utilization=0.4, tokens_per_param=20.0)` → `(n_params, n_tokens)` as floats from `compute_optimal(hours * 3600 * flops_per_second * utilization, tokens_per_param=tokens_per_param)`. both raise `ValueError` for non-positive sizes, hours or throughput, or a utilization outside $(0, 1]$.

> **🔮 predict:** with a week of your laptop, how big is the largest compute-optimal language model, and how long would GPT-2 small (124M parameters) take at 20 tokens per parameter?

> **▶ build: stage 3.** implement `run_hours` and `largest_model`.

### idea 4: the budget plan

an experiment is many runs: configurations × seeds, for each experiment in the plan (pilot, main comparison, ablations, a sanity check at a second scale). the plan's total compute is the sum, and it will be wrong in one direction: runs crash, bugs force reruns, a learning rate turns out to need tuning, the pilot reveals more noise than you hoped. people underestimate how long their own tasks take even when they know about the bias (the *planning fallacy*, Buehler, Griffin & Ross, 1994; topic 06 measures it on schedules). a fixed **safety factor** (2× is a sensible default for a first project) is crude but works better than optimism: if the plan doesn't fit with it, cut configurations, not seeds.

> **spec.** `budget_plan(experiments, budget_hours, *, safety=2.0)`: each experiment is a dict with a unique, non-empty string `name`, integer `configs >= 1` and `seeds >= 1`, and `hours_per_run > 0`. returns a dict with `rows` (one `{'name', 'runs', 'hours'}` per experiment, in order, `runs = configs × seeds`, `hours = runs × hours_per_run`), `total` (summed hours), `with_safety` (`total × safety`), `budget` and `fits` (`with_safety <= budget_hours`). `ValueError` for `experiments` that isn't a list of dicts, a bad experiment, duplicate names, a `budget_hours` or `safety` that isn't a number, `budget_hours <= 0` or `safety < 1`.

> **▶ build: stage 4.** implement `budget_plan`.

### idea 5: the two-page proposal

the output of this topic is a proposal of about two pages (≈ 1,000 words) that someone else could judge. a good template is the **Heilmeier catechism**, the questions George Heilmeier asked of every proposed project at DARPA: what are you trying to do (in plain words)? how is it done today, and what are its limits? what's new in your approach, and why do you think it will work? who cares, and what difference will it make? what are the risks? how much will it cost, and how long will it take? what are the mid-term and final checks of success?

in this template they become fields: `title`, `goal`, `state_of_the_art`, `approach`, `impact` (the prose sections), `risks` and `checkpoints` (lists), `timeline_weeks`, the `hypothesis` of idea 1, the `pilot_sd` that sizes it, and the `experiments` and `budget_hours` of idea 4. a checker can't tell you whether the question is *interesting*: that's for you and whoever reads the proposal. but it can tell you, in a second, that the proposal is incomplete, too long, unfalsifiable, underpowered or over budget, which is where proposals usually go wrong.

> **spec.** `check_proposal(proposal, *, max_words=1000, alpha=0.05, power=0.8, safety=2.0)` → a list of problem strings, each starting with a field and `': '`. `SECTIONS` (given: `title`, `goal`, `state_of_the_art`, `approach`, `impact`) must be non-empty strings; `risks` and `checkpoints` non-empty lists of non-empty strings; `timeline_weeks` a positive number (not a bool). `length`: the words in the sections, risks and checkpoints together must not exceed `max_words`. `hypothesis`: `'hypothesis: missing'` if it isn't a dict, else each `check_hypothesis` problem prefixed `hypothesis.`. `pilot_sd` must be a positive number. `power`: only if the hypothesis has no problems and `pilot_sd` is valid, `seeds_for_power(pilot_sd, min_effect, alpha=alpha, power=power)` must not exceed `n_seeds` (the message says how many are needed). `budget`: `budget_plan(experiments, budget_hours, safety=safety)` must fit, and an invalid plan is a `budget` problem, not a crash. never modify the proposal.

> **▶ build: stage 5.** implement `check_proposal`.

## worked example

**a hypothesis.** the statement "we will explore whether label smoothing helps" is flagged for the word *explore*, and a hypothesis that consists of nothing else fails on every other field too: no method, baseline, metric, setting, direction, effect size, refutation criterion or seed count. the version in idea 1, with method "label smoothing 0.1", baseline "no label smoothing", direction `'greater'`, `min_effect` 0.3, a refutation criterion and 30 seeds, passes.

**power.** with $\sigma_d = 1$ and $\Delta = 0.5$ (one-sided, $\alpha = 0.05$): $n = 26$ gives power 0.798 and $n = 27$ gives 0.812, so 27 seeds. two-sided needs 34. only $\Delta/\sigma_d$ matters: $\sigma_d = 0.6$, $\Delta = 0.3$ also needs 27.

**compute.** $N = 10^6$ parameters on $D = 2 \times 10^7$ tokens is $1.2 \times 10^{14}$ FLOPs; at half of $10^{12}$ FLOP/s that's 240 s, or 0.0667 h. one hour at 40% of $10^{12}$ FLOP/s is $1.44 \times 10^{15}$ FLOPs, so the compute-optimal model has $\sqrt{1.44 \times 10^{15}/120} = 3.46$M parameters and 69M tokens.

**a plan.** a pilot (2 configurations × 3 seeds × 0.5 h = 3 h) and a main experiment (4 × 10 × 0.25 h = 10 h): 13 h, 26 h with the safety factor. it fits a 30-hour budget, not a 25-hour one. your code must reproduce all of these.

## on paper

1. **the normal approximation to power.** replacing the noncentral $t$ by a normal, show that the seeds needed for one-sided power $1 - \beta$ are about $n \approx \big((z_{1-\alpha} + z_{1-\beta})\,\sigma_d/\Delta\big)^2$, and evaluate it for $\sigma_d/\Delta = 2$. why is the exact answer (27) a little larger?

<details><summary>solution</summary>

with known $\sigma_d$ the statistic is $Z = \bar d\sqrt n/\sigma_d \sim N(\lambda, 1)$ with $\lambda = \Delta\sqrt n/\sigma_d$. the test rejects when $Z > z_{1-\alpha}$, with probability $\Phi(\lambda - z_{1-\alpha})$. setting it to $1 - \beta$ gives $\lambda = z_{1-\alpha} + z_{1-\beta}$, i.e. $n = ((z_{1-\alpha} + z_{1-\beta})\sigma_d/\Delta)^2 = ((1.645 + 0.842) \times 2)^2 = 24.7$, so 25. the exact answer is larger because $s_d$ is estimated: the $t$ critical value exceeds $z_{1-\alpha}$, and the noncentral $t$ is wider than the normal. the gap is a couple of seeds here and much larger at small $n$.

</details>

2. **Chinchilla by hand.** with $C = 6ND$ and $D = kN$, solve for $N$ and $D$. a laptop week at 460 GFLOP/s effective: what are they for $k = 20$?

<details><summary>solution</summary>

$C = 6kN^2$, so $N = \sqrt{C/(6k)}$ and $D = kN$. a week is 604,800 s, so $C = 604{,}800 \times 4.6 \times 10^{11} = 2.8 \times 10^{17}$ FLOPs; $N = \sqrt{2.8 \times 10^{17}/120} = 4.8 \times 10^7$, about 48M parameters, trained on about 960M tokens. (that's roughly GPT-2 small's size class at a fraction of its data.)

</details>

3. **the winner's curse.** true effects are $\theta \sim N(0, \tau^2)$ and a pilot observes $x = \theta + \varepsilon$, $\varepsilon \sim N(0, \sigma^2)$. show that $E[\theta \mid x] = \frac{\tau^2}{\tau^2 + \sigma^2}\,x$. with $\tau = 0.2$ and $\sigma = 0.5$, how much of a winning pilot's advantage is real?

<details><summary>solution</summary>

$(\theta, x)$ is jointly Gaussian with $\operatorname{Var}(x) = \tau^2 + \sigma^2$ and $\operatorname{Cov}(\theta, x) = \tau^2$, so $E[\theta \mid x] = \frac{\operatorname{Cov}(\theta, x)}{\operatorname{Var}(x)}\,x = \frac{\tau^2}{\tau^2 + \sigma^2}\,x$ (ch 04's conditional Gaussian). with $\tau = 0.2$, $\sigma = 0.5$: $0.04/0.29 = 0.14$. selecting the best of many pilots doesn't change this conditional expectation, it just picks a large $x$: a winner that showed +1 point is worth about +0.14 in expectation. the noisier the pilot relative to the spread of true effects, the more it lies.

</details>

## experiment

the lab sizes a real question with a pilot on MNIST (label smoothing on a small MLP), turning the measured noise into a seed budget with your power functions; measures your laptop's matrix-multiply throughput and asks your compute functions what it can afford; and simulates 2,000 research projects that each pick the best of 20 one-seed pilots, to see how much of the winner's advantage survives a proper follow-up.

## check your understanding

1. why is `min_effect` "the smallest effect that would matter" rather than "the effect we expect"?

<details><summary>answer</summary>

the experiment is sized to detect `min_effect`: if the true effect is larger, it's detected even more easily. sizing for the expected effect (often an optimistic pilot estimate) leaves you underpowered for every effect between the important minimum and your optimism. and "what would matter" is decided by the question (would a 0.1-point gain change anyone's practice?), not by the data.

</details>

2. a 3-seed comparison finds no significant difference. a reviewer concludes the method doesn't help. what's your reply, with numbers?

<details><summary>answer</summary>

with a paired sd like the lab's 0.65 points, 3 seeds have about 14% power for a 0.3-point effect and 24% for 0.5 points: the experiment would usually miss a real effect of that size. "not significant" from an underpowered test is absence of evidence, not evidence of absence; report the interval, which will be wide.

</details>

3. your proposal passes every check, but needs 31 seeds per configuration and has 4 configurations. what do you cut first?

<details><summary>answer</summary>

configurations, not seeds: an underpowered comparison of 4 configurations answers nothing, while a well-powered comparison of the 2 that matter answers the question. if even 2 configurations don't fit, raise `min_effect` only if a larger effect is still worth knowing about, and say so; otherwise change the question.

</details>

4. why does the proposal include `checkpoints`?

<details><summary>answer</summary>

they're Heilmeier's "mid-term exams": points where you can see, early, whether the project is on track (the pilot measured the noise; the baseline reproduces the course's number; the first full comparison ran). without them, a project that's going wrong looks fine until the end.

</details>

5. you tried 12 small ideas this month and one gave a significant improvement (p = 0.03). should you write it up?

<details><summary>answer</summary>

not yet. with 12 tries, a p = 0.03 is unremarkable (the chance of at least one p < 0.05 among 12 true nulls is 46%), and the winner's estimate is inflated. treat it as a pilot: state a hypothesis with a `min_effect` chosen for what matters, size a fresh experiment with `seeds_for_power`, and run it on new seeds.

</details>

## common bugs

- **the central $t$ for power.** using $P(T > t_c - \lambda)$ with a central $t$ is close but not right, especially at small $n$; use `scipy.stats.nct`.
- **the pilot's effect as `min_effect`.** it's inflated (idea 2), so the experiment comes out underpowered. choose `min_effect` for what matters.
- **power with the pilot's seeds reused.** the confirming experiment must use fresh seeds; otherwise the pilot's luck is in the confirmation.
- **forgetting the safety factor**, or applying it per experiment and then again to the total. apply it once, to the total.
- **counting words only in the prose.** a "two-page" proposal with a 600-word list of risks isn't two pages.
- **crashing on a bad plan.** `check_proposal` should report a malformed experiment as a `budget` problem, so the author sees every problem at once.

## going deeper

- Hamming (1986), "You and your research", a talk at Bellcore. [cs.virginia.edu/~robins/YouAndYourResearch.html](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html)
- the Heilmeier catechism, DARPA. [darpa.mil/work-with-us/heilmeier-catechism](https://www.darpa.mil/work-with-us/heilmeier-catechism)
- Schulman (2020), "An opinionated guide to ML research". [joschu.net/blog/opinionated-guide-ml-research.html](http://joschu.net/blog/opinionated-guide-ml-research.html)
- Popper (1959), *The Logic of Scientific Discovery*, Hutchinson (English translation of *Logik der Forschung*, 1934). falsifiability.
- Nosek, Ebersole, DeHaven & Mellor (2018), "The preregistration revolution", *PNAS* 115(11): 2600–2606. [doi:10.1073/pnas.1708274114](https://doi.org/10.1073/pnas.1708274114)
- Kerr (1998), "HARKing: hypothesizing after the results are known", *Personality and Social Psychology Review* 2(3): 196–217. [doi:10.1207/s15327957pspr0203_4](https://doi.org/10.1207/s15327957pspr0203_4)
- Button et al. (2013), "Power failure: why small sample size undermines the reliability of neuroscience", *Nature Reviews Neuroscience* 14: 365–376. [doi:10.1038/nrn3475](https://doi.org/10.1038/nrn3475)
- Gelman & Carlin (2014), "Beyond power calculations: assessing type S (sign) and type M (magnitude) errors", *Perspectives on Psychological Science* 9(6): 641–651. [doi:10.1177/1745691614551642](https://doi.org/10.1177/1745691614551642)
- Sculley, Snoek, Rahimi & Wiltschko (2018), "Winner's curse? On pace, progress, and empirical rigor", *ICLR 2018 workshop track* (on OpenReview).
- Hoffmann et al. (2022), "Training compute-optimal large language models". [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Kaplan et al. (2020), "Scaling laws for neural language models". [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

## connections

- ← builds on: topic 01 (limitations and assumptions from your notes), topic 02 (reproduction discrepancies as questions), topic 03 (seeds and intervals), ch 05 · topic 04 (power), ch 19 · topic 05 (`compute_optimal`), ch 26 · topic 01 (throughput and utilization).
- → used in: topic 05 (the proposal is the first draft of the paper's introduction), topic 06 (the proposal's experiments and checkpoints become the thesis schedule), and the thesis capstone, which starts from a checked proposal.
