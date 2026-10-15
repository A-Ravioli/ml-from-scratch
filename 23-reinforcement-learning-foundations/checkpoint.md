# 23 · reinforcement learning foundations · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. what is pseudo-regret, and why does constant-ε ε-greedy have linear regret even after it has identified the best arm?

<details><summary>answer</summary>

$\bar R_T = \sum_t(\mu^* - \mu_{A_t}) = \sum_a\Delta_aN_a(T)$: the cost of your choices, measured with the arms' means rather than the noisy rewards. ε-greedy keeps playing a uniformly random arm a fraction ε of the time forever, paying about $\varepsilon\bar\Delta$ per round, so its regret grows like $\varepsilon\bar\Delta T$. (topic 01, ideas 1–2)

</details>

2. UCB has never pulled arm 2. what does it do, and why? what does the $\ln t$ in its bonus achieve?

<details><summary>answer</summary>

it pulls it: with $N_a = 0$ the confidence bonus is infinite. the slowly growing $\ln t$ makes every neglected arm's bonus creep up until it's tried again, so no arm is abandoned forever, while well-sampled arms' bonuses shrink like $1/\sqrt{N_a}$. (topic 01, idea 3)

</details>

3. Thompson sampling with posteriors Beta(3, 2) and Beta(2, 3): does it play arm 0 always, 60% of the time, or something else?

<details><summary>answer</summary>

with the probability that arm 0 is the better one under the posterior, $P(\theta_0 > \theta_1) = 53/70 \approx 0.757$: probability matching, not the posterior mean (60%) and not greedy (100%). (topic 01, idea 4)

</details>

4. `step` returns `terminated` and `truncated` separately. what does each mean for a TD target?

<details><summary>answer</summary>

terminated: the next state is terminal and worth 0, so the target is $r$. truncated: an outside limit (usually time) cut the episode; the next state still has a future, so the target is $r + \gamma V(s')$. treating truncation as termination makes states near the time limit look better (or worse) than they are. (topic 02, idea 1; topic 05, idea 1 and lab)

</details>

5. write the Bellman expectation equation for $V^\pi$ in matrix form. when is $I - \gamma P_\pi$ invertible?

<details><summary>answer</summary>

$V^\pi = R_\pi + \gamma P_\pi V^\pi$, i.e. $(I - \gamma P_\pi)V^\pi = R_\pi$. for $\gamma < 1$ always ($\gamma P_\pi$ has spectral radius at most $\gamma$); with $\gamma = 1$ the absorbing terminal states make it singular, and you solve on the non-terminal states, which works when the policy reaches a terminal state with probability 1. (topic 02, idea 4)

</details>

6. in an episodic gridworld, you change the reward from $-1$ per step to $+1$ per step. what happens to the optimal policy, and why doesn't the "adding a constant changes nothing" argument apply?

<details><summary>answer</summary>

the agent now prefers never to reach the goal. the constant-shift invariance needs every trajectory to collect the same number of (discounted) rewards, which holds in continuing tasks; in episodic tasks the terminal state's value (0) doesn't shift, so episode length changes the comparison. (topic 02, on paper 4 and lab)

</details>

7. state the policy improvement theorem. why does policy iteration terminate?

<details><summary>answer</summary>

if $\pi'$ is greedy with respect to $Q^\pi$, then $V^{\pi'} \ge V^\pi$ in every state, strictly somewhere unless $\pi$ is already optimal. each improvement step therefore produces a strictly better deterministic policy until it can't, and there are finitely many deterministic policies. (topic 03, idea 2)

</details>

8. value iteration stops when $\|V_{k+1} - V_k\|_\infty < \epsilon$. how far from $V^*$ can $V_{k+1}$ still be? why can the greedy policy be optimal long before that?

<details><summary>answer</summary>

up to $\gamma\epsilon/(1 - \gamma)$ (from the $\gamma$-contraction). the greedy policy only needs the *ordering* of actions' values in each state to be right, and the large, slowly converging part of the values (shared across actions) cancels in the argmax. (topic 03, idea 3 and lab)

</details>

9. first-visit versus every-visit Monte Carlo: which is unbiased? why does model-free control estimate $Q$ rather than $V$?

<details><summary>answer</summary>

first-visit: each episode gives one independent sample of $G$ given $S_t = s$. every-visit averages correlated, nested returns and is biased for finite $n$ (but consistent). $Q$ because improving from $V$ needs the model to look one step ahead; with $Q$ the greedy action is an argmax. (topic 04, ideas 2–3)

</details>

10. ordinary importance sampling is unbiased. why can it be useless, and what does weighted importance sampling trade for its stability?

<details><summary>answer</summary>

the product of ratios can have huge (even infinite) variance, so most runs land far below the truth and rare runs far above it. weighted IS divides by the sum of the weights, which bounds its error but makes it biased for finite $n$ (with one sample it's just that sample's return). (topic 04, idea 4 and lab)

</details>

11. on the cliff, Q-learning learns the optimal path but earns less reward while learning than SARSA. explain.

<details><summary>answer</summary>

Q-learning's target uses the max, so it learns the greedy policy's values, which walk the edge; but the agent behaves ε-greedily and a random step from the edge falls off. SARSA's target uses the action actually taken, so it learns the values of the ε-greedy policy, which avoid the edge. (topic 05, idea 2 and lab)

</details>

12. what's the n-step return, and why is an intermediate $n$ (or $\lambda$) usually best?

<details><summary>answer</summary>

$G_{t:t+n} = R_{t+1} + \dots + \gamma^{n-1}R_{t+n} + \gamma^nQ(S_{t+n}, A_{t+n})$ (no bootstrap if the episode terminated first). small $n$ relies on the current estimates (bias, slow propagation); large $n$ uses more random rewards (variance). the best trade-off is in between and depends on the step size. (topic 05, ideas 4–5)

</details>

13. why is semi-gradient TD called "semi"? name the three ingredients of the deadly triad.

<details><summary>answer</summary>

it differentiates the prediction but treats the bootstrapped target as a constant, so it's not the gradient of any fixed loss. function approximation, bootstrapping and off-policy training: together they can make the expected update $w \leftarrow w + \alpha(b - Aw)$ unstable ($A$ with an eigenvalue of negative real part), as in Baird's counterexample. (topic 06, ideas 3–4)

</details>

14. tile coding: what sets how far a single update generalizes, and what sets how finely the value function can vary?

<details><summary>answer</summary>

the tile width sets the reach of generalization (about one tile on either side); the number of offset tilings sets the resolution (the function can change every tile width / number of tilings). (topic 06, idea 1 and lab)

</details>

## 2. by hand (no computer)

1. **policy evaluation and one improvement.** a $1\times3$ corridor (states 0, 1, 2; goal 2; actions up, right, down, left; moving off the grid stays put; $-1$ per step; $\gamma = 1$). (a) compute $V$ of the uniformly random policy. (b) compute $Q(1, a)$ for all four actions. (c) what policy does one greedy improvement give, and what is its value?

<details><summary>solution</summary>

(a) $V(0) = -1 + \tfrac34V(0) + \tfrac14V(1)$ and $V(1) = -1 + \tfrac14V(0) + \tfrac12V(1)$ with $V(2) = 0$: $V(0) = V(1) - 4$, then $\tfrac12V(1) = -1 + \tfrac14(V(1) - 4)$, so $V(1) = -8$, $V(0) = -12$. (b) $Q(1, \text{up}) = Q(1, \text{down}) = -1 + V(1) = -9$, $Q(1, \text{right}) = -1 + 0 = -1$, $Q(1, \text{left}) = -1 + V(0) = -13$. (c) "right" in states 0 and 1 (in state 0: right $-9$, others $-13$), with value $(-2, -1, 0)$: optimal in one step. (topic 02 worked example, topic 03 worked example)

</details>

2. **TD updates.** a chain $0 \to 1 \to 2 \to$ end with rewards $0, 0, 1$, one action, $\gamma = 1$, $\alpha = 0.5$, all values 0. (a) one episode of TD(0): what are $V(0), V(1), V(2)$? (b) a second episode? (c) one episode of TD($\lambda$) with accumulating traces and $\lambda = 0.5$?

<details><summary>solution</summary>

(a) the first two TD errors are 0; the last is $1 - 0 = 1$: $V = (0, 0, 0.5)$. (b) $\delta_0 = 0 + V(1) - V(0) = 0$; $\delta_1 = 0 + V(2) - V(1) = 0.5$, so $V(1) = 0.25$; $\delta_2 = 1 - 0.5 = 0.5$, so $V(2) = 0.75$: $V = (0, 0.25, 0.75)$. (c) the only nonzero error, $\delta = 1$, arrives with traces $(\lambda^2, \lambda, 1) = (0.25, 0.5, 1)$: $V = 0.5\cdot(0.25, 0.5, 1) = (0.125, 0.25, 0.5)$. (topic 05, worked example)

</details>

3. **importance sampling.** a two-step environment: from state 0 any action leads to state 1 (reward 0); in state 1, action 0 ends with $+1$, action 1 ends with 0. the target policy always takes action 0; the behaviour policy takes action 0 with probability $0.8$ in state 1. (a) what's $Q^\pi(0, a)$? (b) for a visit to $(0, a)$, what values can the ordinary-IS term $\rho G$ take, with what probabilities? check that its mean is right. (c) its variance?

<details><summary>solution</summary>

(a) $Q^\pi(0, a) = 1$ (then action 0 at state 1 earns $+1$). (b) $\rho = \pi/b$ for the action at state 1: $1/0.8 = 1.25$ with $G = 1$ (probability $0.8$), or $0$ (probability $0.2$). mean $0.8 \cdot 1.25 = 1$. (c) $\mathbb E[(\rho G)^2] = 0.8 \cdot 1.5625 = 1.25$, so the variance is $1.25 - 1 = 0.25$. with $b(0\mid1) = 0.5$ it would be $0.5\cdot4 - 1 = 1$: the further the behaviour from the target, the larger the variance. (topic 04, idea 4)

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **the TD control family** of topic 05: `CliffWalking`, `td0_prediction`, `sarsa`, `q_learning`, `expected_sarsa` and `n_step_sarsa`. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=23-reinforcement-learning-foundations/05-temporal-difference-learning
mkdir -p /tmp/rebuild-2305
git show HEAD:$T/exercise.py > /tmp/rebuild-2305/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-2305/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-2305/solutions/` and run `python scripts/stubify.py /tmp/rebuild-2305`.) open `/tmp/rebuild-2305/exercise.py` with `marimo edit`, implement the stages from memory, and run the checks against your copy:

```bash
pytest /tmp/rebuild-2305/test_exercise.py -k "stage1 or stage2 or stage3 or stage4"
```

you pass if every check is green within the hour. (the checks verify each update rule exactly on tiny environments with $\alpha = 1$, require `n_step_sarsa` with $n = 1$ to reproduce your `sarsa` exactly, check bootstrapping on truncation, and check learning on the cliff, so "almost right" fails.)

**harder variant** (for testing out of the chapter with confidence): in a further 60 minutes, rebuild topic 02's `GridWorld` and `policy_evaluation`, and topic 03's `policy_iteration` and `value_iteration`, each in its own scratch directory, and pass:

```bash
pytest /tmp/rebuild-2302/test_exercise.py -k "stage2 or stage3 or stage4"
pytest /tmp/rebuild-2303/test_exercise.py -k "stage1 or stage2 or stage3"
```

(run them from the repo root: the notebooks import earlier topics through `mlfs.rl`, which the tests resolve to the reference solutions, so only the code you rebuilt is being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (14 concept questions + 3 problems, counting each lettered part separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the topic and idea it comes from.
