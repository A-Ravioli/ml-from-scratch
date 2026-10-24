# alignment

## why this matters

every other topic in this chapter assumed you knew what you wanted from the model and asked whether it delivered. alignment asks the prior question: does the objective you wrote down capture what you want at all? usually it doesn't, quite. a reward is a proxy, a measurable stand-in for an intent that's hard to measure, and a capable optimiser finds wherever the proxy and the intent come apart. in a boat-racing game, an RL agent rewarded for hitting score targets learned to circle a lagoon forever, catching fire and crashing, because the targets respawned, and never finished the race (Clark & Amodei, 2016). Krakovna et al. (2020) collected dozens of such cases of **specification gaming**. in chapter 27's lab you watched a language model optimised against a learned reward model write "thee thy thee thee thee": **reward hacking** of the same kind, with the reward model as the flawed specification.

these failures aren't bugs in the optimiser; they're the optimiser working. that makes them hard to fix by optimising better, and it's why alignment research focuses on three things this topic touches: understanding how optimisation pressure turns small proxy errors into large failures (Goodhart's law, made quantitative), designing rewards that can't be gamed (and knowing when that's possible), and **evaluations** that measure the behaviour you care about directly, so that failures show up before deployment. the later questions, how to supervise models that are better than us at the task (**scalable oversight**: debate, recursive reward modelling, weak-to-strong generalisation), build on the same ideas and are introduced at the end.

## warm-up: predict first

1. best-of-64 sampling picks the highest-scoring of 64 responses according to a reward model. how far (in KL) does that move the output distribution from the base model?
2. a reward model's error is usually small but occasionally enormous. as you optimise harder against it, does the true quality keep improving, plateau, or fall?
3. a game gives 1 point every time you enter a bonus square and 10 for finishing. a patient agent (it barely discounts the future) and an impatient one: which is more likely to circle the bonus square forever?
4. you can measure how often an agent reaches its goal on 200 test layouts. it succeeds on 106. what's a 95% interval for its success rate?

## prerequisites check

- [ ] you know the KL-regularised RLHF objective and its closed-form optimum $\pi^* \propto \pi_\text{ref}\, e^{r/\beta}$, and best-of-$n$'s KL (→ ch 27 · 04)
- [ ] you know value iteration, policy evaluation and discounting (→ ch 23 · 02–03, `mlfs.rl`)
- [ ] you know reward shaping in RL (→ ch 23)
- [ ] you can compute bootstrap and paired bootstrap intervals for accuracies (→ ch 27 · 06, `mlfs.llm.accuracy_ci`, `paired_accuracy_diff`)
- [ ] you know order statistics: the distribution of the maximum of $n$ samples (→ ch 04)

## core ideas

### idea 1: optimisation pressure and Goodhart's law

"when a measure becomes a target, it ceases to be a good measure." Manheim & Garrabrant (2018) split Goodhart's law into mechanisms. **regressional**: the proxy is the true value plus independent noise, so the items that score highest on the proxy are partly selected for their noise, and the true value of the selection is less than its proxy value (but still improves). **extremal**: in the tails, the relationship between proxy and truth breaks down entirely, and optimising pushes you into the tails. **causal**: the proxy is correlated with the goal through something you can manipulate directly (response length, a bonus square).

to study these exactly, take a finite set of responses $i$ with reference probabilities $\pi_\text{ref}(i)$, a **proxy** reward $r(i)$ and a **true** value $u(i)$ that the optimiser never sees. the KL-regularised objective of RLHF, $\max_\pi \mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \| \pi_\text{ref})$, has the closed-form maximiser (chapter 27, on paper 1)

$$\pi_\beta(i) = \frac{\pi_\text{ref}(i)\, e^{r(i)/\beta}}{\sum_j \pi_\text{ref}(j)\, e^{r(j)/\beta}}.$$

lowering $\beta$ dials up optimisation pressure, and the KL from the reference measures how much was applied. plotting $\mathbb{E}_{\pi_\beta}[u]$ against $\mathrm{KL}(\pi_\beta \| \pi_\text{ref})$ gives the curve Gao, Schulman & Hilton (2023) measured for real reward models: the proxy rises steadily; the true value rises, and then, depending on the kind of error, keeps rising slowly, stalls, or falls. computing $e^{r/\beta}$ for small $\beta$ overflows unless you subtract the largest log-weight first; responses the reference never produces keep probability zero.

> **spec.** `kl_regularized_policy(ref_probs, rewards, beta)`: the distribution $\pi_\beta$ above as an ndarray (computed stably for large $r/\beta$; zero where `ref_probs` is zero); `ValueError` for `beta <= 0`. `policy_stats(policy, ref_probs, proxy, true)`: `{"kl": KL(policy || ref) in nats (0 log 0 = 0), "proxy": E_policy[proxy], "true": E_policy[true]}`, Python floats.

> **🔮 predict:** a proxy equals the true value plus an error that is usually tiny but, for one response in a thousand, enormous. as $\beta$ falls and the KL grows, what happens to the expected true value?

> **▶ build: stage 1.** implement `kl_regularized_policy` and `policy_stats`.

### idea 2: best-of-n, exactly

the simplest optimiser is best-of-$n$: draw $n$ responses from the reference and keep the one the proxy likes best. on a finite set it has an exact distribution. sort the proxy values; let $F(v) = \sum_{j: r_j \le v} \pi_\text{ref}(j)$ be the reference probability of scoring at most $v$, and $F(v^-)$ of scoring strictly less. the best of $n$ independent draws has score exactly $v$ with probability $F(v)^n - F(v^-)^n$, and if several responses share that score, the winner is each of them in proportion to its reference probability:

$$P_n(i) = \big(F(r_i)^n - F(r_i^-)^n\big)\,\frac{\pi_\text{ref}(i)}{\sum_{j: r_j = r_i} \pi_\text{ref}(j)}.$$

for continuous scores its KL from the reference is $\log n - (n-1)/n$ (chapter 27's `best_of_n_kl`, warm-up 1: 3.17 nats for $n = 64$), which you can check against the exact distribution. one difference from $\pi_\beta$ matters in the lab: best-of-$n$ uses only the *ranks* of the proxy, not its values. a response whose proxy score is wrong by a factor of 100 is ranked first, but no more first than a response that's wrong by a factor of 2; under $e^{r/\beta}$ it takes all the probability mass.

> **spec.** `best_of_n_distribution(ref_probs, scores, n)`: the ndarray $P_n$ above; `ValueError` for $n < 1$.

> **▶ build: stage 2.** implement `best_of_n_distribution`.

### idea 3: specification gaming in an MDP

reward hacking needs no learned reward model: a hand-written reward with a loophole is enough. take a corridor of $L$ cells, with the agent starting at 0, actions left and right, and a goal at $L - 1$ worth $G$ (entering it ends the episode). the designer wants the agent to reach the goal, and to help it learn adds a **bonus** $B$ for entering a cell $b$ along the way. the bonus respawns: every entry pays. the **proxy** reward is goal plus bonuses; the **true** reward is the goal alone.

compare two plans from cell $b - 1$, with discount $\gamma$. shuttling in and out of $b$ forever earns $B + \gamma^2 B + \gamma^4 B + \dots = B/(1 - \gamma^2)$. walking to the goal earns the bonus once and then $G$ after $L - 1 - b$ more steps, $B + \gamma^{L-1-b} G$. the agent games the reward exactly when

$$B > B^\star(\gamma) = G\,\gamma^{L-1-b}\,\frac{1 - \gamma^2}{\gamma^2}$$

(on paper 2). the threshold is small for an *impatient* agent ($\gamma$ small: the goal is far away and heavily discounted) and also for a very *patient* one ($\gamma \to 1$: an infinite stream of small bonuses is worth more than any finite prize). somewhere in between the bonus is hardest to abuse, but no discount factor makes it safe.

the fix isn't a smaller bonus; it's a bonus with the right *form*. **potential-based shaping** (Ng, Harada & Russell, 1999) pays $F(s, s') = \gamma\Phi(s') - \Phi(s)$ for a potential $\Phi$ over states. around any loop these payments telescope to (nearly) zero, so they can't create new optimal behaviour: the optimal policy under $R + F$ is exactly the optimal policy under $R$. paying $\Phi(b) = B$ as a potential means earning $\gamma B$ on the way into the cell and paying $B$ back on the way out. the lab's third agent plans with it.

> **spec.** `corridor_mdp(length, bonus_pos, bonus, *, goal_reward=10.0)`: tabular `(P (S, 2, S), R_proxy (S, 2), R_true (S, 2))` with $S$ = `length`, action 0 = left (staying put at cell 0), 1 = right, the goal $S-1$ absorbing with zero reward, `R_true` = `goal_reward` on the step into the goal, and `R_proxy` = `R_true` plus `bonus` on every step that enters `bonus_pos` from another cell; `ValueError` unless `length >= 3` and `0 < bonus_pos < length - 1`. `gaming_gap(P, R_proxy, R_true, gamma=0.95, start=0)`: plans with `mlfs.rl.value_iteration` on the proxy, evaluates that policy on the true reward with `mlfs.rl.policy_evaluation`, and returns `{"policy": proxy-optimal policy, "true_value": its true value at start, "optimal_value": the true-optimal value at start, "regret": their difference}`. `rollout_states(P, policy, start, horizon)`: the list `[start, s_1, ..., s_horizon]` visited by following the policy's argmax action in a deterministic MDP.

> **🔮 predict:** in a corridor of 10 cells with the bonus at cell 2 and $G = 10$, at which $\gamma$ is the gaming threshold highest, and how high is it?

> **▶ build: stage 3.** implement `corridor_mdp`, `gaming_gap` and `rollout_states`.

### idea 4: evals

the corridor agent that circles the bonus gets more *proxy* reward than the honest one. any metric computed from the training reward says it's doing better. the only way to see the failure is to measure the behaviour you actually want, on cases the agent wasn't optimised for: an **evaluation**. dangerous-capability and safety evaluations of frontier models are elaborate versions of this (Shevlane et al., 2023): a suite of test cases, each with a check that says pass or fail, grouped into categories, reported with uncertainty.

three habits from chapter 27's evaluation topic carry over unchanged. report every pass rate with an interval (a bootstrap over cases); break the results down by category, because an average hides the slice where the failures are; and compare agents with a **paired** difference over the same cases, which cancels the variation between cases and is much tighter than comparing two separate intervals. (warm-up 4: 106 of 200 is 0.53, with a 95% interval of about $[0.46, 0.60]$: 200 cases pin a rate down to about ±7 points.) what an eval can't do is tell you about behaviour it doesn't test: a 100% pass rate on 200 corridors says nothing about a corridor with two bonus cells.

> **spec.** `eval_suite(agent, cases, *, n_boot=2000, rng=None)`: `cases` is a list of `(category, check)` with `check(agent) -> bool`, each check called exactly once; returns `{category: (rate, lo, hi)}` for every category plus `"overall"`, each from `mlfs.llm.accuracy_ci(..., n_boot=n_boot, rng=rng)` with one generator (`mlfs.utils.as_rng(rng)`) shared by all of them, used for the categories in sorted order and then for `"overall"`; `ValueError` for no cases. `compare_agents(agent_a, agent_b, cases, *, n_boot=2000, rng=None)`: `mlfs.llm.paired_accuracy_diff` of a's and b's pass/fail over the same cases: `(diff, lo, hi)`.

> **▶ build: stage 4.** implement `eval_suite` and `compare_agents`.

### beyond this topic: scalable oversight

every fix above relied on someone knowing the true objective: the gray "optimise $u$" curve, the true corridor reward, the eval's checks. for tasks where the model is more capable than its overseers that stops being true, and alignment research tries to amplify weak supervision instead. **debate** (Irving, Christiano & Amodei, 2018) has two models argue and a human judge the argument, betting that it's easier to judge than to produce. **recursive reward modelling** uses models to help humans evaluate harder tasks. **weak-to-strong generalisation** (Burns et al., 2023) asks how much of a strong model's capability a weak supervisor's labels can elicit, measured by the *performance gap recovered*, $\text{PGR} = (\text{acc}_{w2s} - \text{acc}_{weak}) / (\text{acc}_{strong} - \text{acc}_{weak})$. none of these is solved; all of them need the measurement discipline of this chapter.

## worked example

**KL-regularised optimisation.** reference $(0.5, 0.3, 0.2)$, proxy rewards $(0, 1, 2)$, $\beta = 0.5$: weights $0.5 e^0, 0.3 e^2, 0.2 e^4 = 0.5, 2.217, 10.920$, normalised to $\pi = (0.037, 0.163, 0.801)$. with true values $(2, 0, -5)$ the optimiser has moved almost all mass onto the response with the worst true value. **statistics:** for $\pi = (0.5, 0.5, 0)$ and reference $(0.25, 0.25, 0.5)$, $\mathrm{KL} = 2 \cdot 0.5 \log 2 = \log 2 = 0.693$ nats.

**best-of-2.** reference $(0.5, 0.3, 0.2)$ with scores $(1, 3, 2)$. the best response (score 3) wins unless both draws miss it: $1 - 0.7^2 = 0.51$. the middle one wins when the best is $\le 2$ but not both $\le 1$: $0.7^2 - 0.5^2 = 0.24$. the worst wins only if both draws are it: $0.5^2 = 0.25$.

**the gaming threshold.** $L = 8$, $b = 2$, $G = 10$, $\gamma = 0.9$: $B^\star = 10 \cdot 0.9^5 \cdot 0.19 / 0.81 = 1.385$. a bonus of $1.25$ (10% below) leaves the agent honest; $1.52$ (10% above) makes it loop forever, true value 0 instead of $10 \cdot 0.9^6 = 5.31$.

**an eval.** an agent passes 8 of 10 easy cases and 15 of 30 hard ones: rates 0.80, 0.50 and 0.575 overall, the easy one with the widest interval.

your functions must reproduce all of these.

## on paper

1. **why $\pi_\beta$ maximises the regularised objective.** show that $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi\|\pi_\text{ref}) = \beta \log Z - \beta\,\mathrm{KL}(\pi \| \pi_\beta)$, where $Z = \sum_j \pi_\text{ref}(j) e^{r_j/\beta}$.

<details><summary>solution</summary>

$\log \pi_\beta(i) = \log\pi_\text{ref}(i) + r_i/\beta - \log Z$, so $r_i = \beta(\log\pi_\beta(i) - \log\pi_\text{ref}(i) + \log Z)$. then $\mathbb{E}_\pi[r] - \beta\,\mathbb{E}_\pi[\log\pi - \log\pi_\text{ref}] = \beta\,\mathbb{E}_\pi[\log\pi_\beta - \log\pi] + \beta\log Z$. the expectation is $-\mathrm{KL}(\pi\|\pi_\beta) \le 0$, zero exactly at $\pi = \pi_\beta$.

</details>

2. **the gaming threshold.** derive $B^\star(\gamma)$, and show that it tends to 0 both as $\gamma \to 0$ and as $\gamma \to 1$ (for $L - 1 - b \ge 3$).

<details><summary>solution</summary>

from $b - 1$, looping is worth $B/(1-\gamma^2)$ and going is worth $B + \gamma^{L-1-b} G$ (the bonus on the first step, then the goal). looping wins when $B\big(\frac{1}{1-\gamma^2} - 1\big) = B\frac{\gamma^2}{1-\gamma^2} > \gamma^{L-1-b}G$. as $\gamma \to 1$, $(1-\gamma^2) \to 0$; as $\gamma \to 0$, $\gamma^{L-1-b}/\gamma^2 = \gamma^{L-3-b} \to 0$. (other mixed strategies, loop for a while then go, are never optimal in a stationary MDP: if looping beats going once, it beats going every time.)

</details>

3. **shaping can't be gamed.** with $F(s, s') = \gamma\Phi(s') - \Phi(s)$, show that the discounted sum of $F$ along any trajectory $s_0, s_1, \dots$ is $-\Phi(s_0) + \lim_t \gamma^t \Phi(s_t)$, and conclude that every policy's value changes by the same $-\Phi(s_0)$.

<details><summary>solution</summary>

$\sum_{t=0}^{T-1}\gamma^t(\gamma\Phi(s_{t+1}) - \Phi(s_t)) = \gamma^T\Phi(s_T) - \Phi(s_0)$ (the sum telescopes). for bounded $\Phi$ and $\gamma < 1$ the first term vanishes, so $V^\pi_{R+F}(s_0) = V^\pi_R(s_0) - \Phi(s_0)$ for every policy $\pi$: the ranking of policies, and hence the optimal policy, is unchanged. the corridor's respawning bonus is not of this form: around the loop $b-1 \to b \to b-1$ it pays $B$ and takes nothing back.

</details>

4. **why heavy-tailed errors break $\pi_\beta$ but not best-of-$n$ (at first).** a proxy is $r = u + \varepsilon$. if one response has $\varepsilon = 100$ and $\pi_\text{ref} = 10^{-4}$, what weight does $\pi_\beta$ give it at $\beta = 1$, compared with a typical response with $r \approx 2$? what's its probability under best-of-32?

<details><summary>solution</summary>

$\pi_\beta$ gives it weight $10^{-4} e^{100/\beta}$ against roughly $e^{2/\beta}$ for all the typical responses together, so it takes essentially all the mass whenever $e^{98/\beta} > 10^4$, i.e. for any $\beta$ below about 10, whatever its true value. under best-of-32 it wins only if it's drawn at least once, probability $1 - (1 - 10^{-4})^{32} \approx 0.003$; best-of-$n$ only cares that it's ranked first. as $n$ grows into the thousands, best-of-$n$ draws the outliers too and falls the same way (the lab sees this past $n = 32$).

</details>

## experiment

the lab sweeps optimisation pressure against three kinds of flawed proxy and plots true against proxy value as a function of KL, for the KL-regularised policy and for best-of-$n$; maps where the corridor agent starts gaming as a function of bonus size and discount, against the threshold you derived; and runs a 200-corridor behavioural eval of three agents that plan with the proxy, the true reward and a potential-based shaping of the same bonus.

## check your understanding

1. why doesn't a more powerful optimiser fix specification gaming?

<details><summary>answer</summary>

gaming is the optimiser finding high-reward behaviour that the designer didn't intend; a better optimiser finds more of it. the fix has to come from the objective (a better specification, a KL leash to a trusted reference, shaping that can't be exploited) or from oversight (evals, human review), not from optimisation.

</details>

2. what does the KL penalty in RLHF buy, in terms of Goodhart's law?

<details><summary>answer</summary>

it limits optimisation pressure: the policy stays near the reference, where the reward model was trained and its errors are small, instead of travelling to the regions where the proxy and the truth have come apart. it trades some proxy reward for staying where the proxy can be trusted; choosing $\beta$ chooses where on the Goodhart curve you stop.

</details>

3. a reward model is 99% accurate on held-out comparisons. why can optimising against it still fail badly?

<details><summary>answer</summary>

held-out accuracy is measured on the distribution it was trained on; the optimiser seeks out inputs where it's wrong, often far from that distribution. the 1% of errors may be concentrated exactly where a strong optimiser goes (the tails), and that's all it needs.

</details>

4. why is potential-based shaping safe while the respawning bonus isn't, even though both pay $B$ for reaching the cell?

<details><summary>answer</summary>

shaping pays for *progress* in a potential that has to be paid back when you lose it, so no cycle can earn net reward; the respawning bonus pays for *events*, and a cycle can trigger the event repeatedly. the general lesson: reward outcomes and states, not repeatable actions, or make the reward a difference of potentials.

</details>

5. your eval shows the proxy agent passing 34% of corridors with large bonuses and 100% with small ones. what would you report, and what would you still not know?

<details><summary>answer</summary>

both slices with intervals, plus the overall rate and a paired comparison with a reference agent; the slice is where the failure lives. you still wouldn't know about layouts the eval doesn't include (two bonuses, a moving goal), and the result says nothing about why the agent fails unless you look at its rollouts.

</details>

6. what does weak-to-strong generalisation measure, and why is it a stand-in for the superhuman case? a weak supervisor labels a task with 70% accuracy; a strong model trained on ground truth reaches 90%, and trained on the weak labels it reaches 82%. compute the PGR, and say what a PGR of 0, of 1 and of a negative number would mean.

<details><summary>answer</summary>

how much of a strong model's ability can be elicited by training it on labels from a weaker supervisor. $\text{PGR} = (82 - 70)/(90 - 70) = 0.6$: the strong student recovers 60% of the gap between its teacher and its own ceiling. PGR 0 means it just imitates the weak labels (errors included); 1 means the weak labels elicited everything the strong model could do; negative means training on weak labels made it *worse* than the weak supervisor. it can beat its teacher because the teacher's errors are partly noise that the strong model's simpler, more general solution doesn't fit (Burns et al. report substantial PGRs on NLP classification tasks and much lower ones on chess puzzles and reward modelling). it's an analogy we can run today, a small model supervising a large one, for the future problem of humans supervising models smarter than them: our labels are the weak teacher.

</details>

## common bugs

- **overflow in $e^{r/\beta}$.** subtract the maximum log-weight before exponentiating.
- **best-of-$n$ off by one level**: $F(v)^n - F(v^-)^n$ needs the mass strictly below $v$, and ties are split in proportion to the reference probabilities, not equally.
- **a bonus that pays for leaving a cell, or only for being in it.** decide what event the reward pays for; the lab's bonus pays for *entering*.
- **evaluating the proxy policy with the proxy reward.** the regret is measured under the *true* reward; under the proxy, the gaming agent looks best.
- **unpaired comparisons.** two overlapping intervals don't mean no difference; compare agents on the same cases.

## going deeper

- Krakovna et al. (2020), "Specification gaming: the flip side of AI ingenuity", DeepMind (with a list of examples). [deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
- Amodei et al. (2016), "Concrete problems in AI safety". [arXiv:1606.06565](https://arxiv.org/abs/1606.06565)
- Manheim & Garrabrant (2018), "Categorizing variants of Goodhart's law". [arXiv:1803.04585](https://arxiv.org/abs/1803.04585)
- Gao, Schulman & Hilton (2023), "Scaling laws for reward model overoptimization", *ICML*. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
- Skalse et al. (2022), "Defining and characterizing reward hacking", *NeurIPS*. [arXiv:2209.13085](https://arxiv.org/abs/2209.13085)
- Ng, Harada & Russell (1999), "Policy invariance under reward transformations: theory and application to reward shaping", *ICML*. [people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Irving, Christiano & Amodei (2018), "AI safety via debate". [arXiv:1805.00899](https://arxiv.org/abs/1805.00899)
- Burns et al. (2023), "Weak-to-strong generalization: eliciting strong capabilities with weak supervision". [arXiv:2312.09390](https://arxiv.org/abs/2312.09390)
- Shevlane et al. (2023), "Model evaluation for extreme risks". [arXiv:2305.15324](https://arxiv.org/abs/2305.15324)

## connections

- ← builds on: the RLHF objective, best-of-$n$ and reward over-optimisation (ch 27 · 04), the eval harness and paired intervals (ch 27 · 06), MDPs, value iteration and shaping (ch 23), adversarial optimisation against a model (topic 04).
- → used in: the capstone asks the interpretability question behind alignment (what algorithm did the network actually learn?); chapter 30's thesis projects on alignment start from `kl_regularized_policy`-style exact experiments, chapter 27's DPO and reward models, and evals with intervals.
