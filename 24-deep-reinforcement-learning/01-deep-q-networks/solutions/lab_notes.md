# deep Q-networks · lab notes

numbers below are from the reference solution with the lab's default settings (15,000 steps per run, seeds 0–2, one CPU thread). your numbers will differ in detail; the pattern shouldn't.

## experiment 1: take the plumbing away

**what people usually predict:** replay is the big one (it's the famous ingredient, and correlated data is the textbook problem), the target network is a minor stabilizer, and an agent without either "learns slower".

**what happens:** greedy returns (5 episodes per seed, mean ± standard error over 3 seeds):

| agent | greedy return | per seed | $\max_aQ(s_0, a)$ |
|---|---|---|---|
| DQN | 274 ± 113 | 161, 500, 162 | about 82 |
| no target network | 9 ± 0 | 10, 9, 9 | about $2\times10^6$ |
| no replay (64-transition buffer) | 194 ± 153 | 17, 66, 500 | about 80 |
| neither | 9 ± 0 | 10, 9, 9 | about $1.5\times10^6$ |

without a target network the agent never learns anything (9 steps is what you get by pushing one way until the pole falls) and its Q-values explode to millions, four orders of magnitude above the largest value any state can have, $1/(1 - \gamma) = 100$. without replay the agent still learns, and one of its seeds balances for the full 500 steps; the difference from full DQN is well inside the seed-to-seed noise of three runs.

**why:** the target network breaks a feedback loop. with `target_update=1` the target $r + \gamma\max_{a'}Q_\theta(s', a')$ moves with every gradient step. CartPole's consecutive states are almost identical, so raising $Q_\theta(s, a)$ also raises $Q_\theta(s', \cdot)$ by nearly as much (the network generalizes between them), which raises the target by $\gamma$ times that, which calls for a bigger $Q_\theta(s, a)$ next step. once the per-step growth of the target outpaces the regression, the values run away geometrically: the deadly triad (function approximation, bootstrapping, off-policy data) with nothing to stop it. a smaller learning rate only slows the escape; the loop is in the targets, not the step size. a frozen target turns each period into a supervised regression with fixed labels, and the values can then grow by at most one Bellman backup per period, which is also why the healthy agents' estimates are still around 80 after 15,000 steps (on paper 4).

replay matters less here because CartPole is a small, smooth problem: the last 64 transitions span several episodes' worth of a short-lived pole early on, the state is 4-dimensional, and the policy only has to learn one thing. the correlation problem bites when the state distribution drifts within an episode (the network forgets the parts of the state space it isn't visiting), when episodes are long, and when the network is big and the data expensive; that's the Atari regime, where Mnih et al.'s own ablation found that removing replay hurt far more than removing the target network. with 500-step episodes late in training, the no-replay agent is already more erratic here (its training curve drops back twice).

the general lesson: an ablation result is about a task, a budget and a set of hyperparameters, not about an algorithm in general. three seeds is also too few for a confident ranking of DQN against no-replay; the standard errors overlap completely.

## experiment 2: how good are the value estimates?

**what people usually predict:** that DQN overestimates (maximization bias is the famous failure) and that double DQN fixes it.

**what happens:** the estimates are poor in both directions, and double DQN doesn't make them better:

| agent | seed | mean true return | mean estimate | estimate − true | max estimate |
|---|---|---|---|---|---|
| DQN | 0 | 49.3 | 62.5 | +13.2 | 83.4 |
| DQN | 1 | 100.0 | 83.0 | −17.0 | 83.6 |
| DQN | 2 | 48.2 | 72.7 | +24.6 | 82.1 |
| double DQN | 0 | 34.7 | 90.9 | +56.1 | 93.5 |
| double DQN | 1 | 57.4 | 41.9 | −15.5 | 82.5 |
| double DQN | 2 | 51.7 | 68.0 | +16.4 | 82.1 |

the policy that balances for 500 steps (DQN seed 1) is *under*estimated (83 against a true 100). the policies that fall are *over*estimated, and worst near the end of an episode: with a few steps left to live, the network still predicts values of 40–90. no estimate reaches 100 in these runs (in longer runs with a larger learning rate the reference solution has produced estimates above 120, which can only be overestimation).

**why:** two forces pull in opposite directions. propagation is slow: with a target update every 100 gradient steps and 14,000 gradient steps, the values have had about 140 backups, so they can't yet represent returns near 100 (the bound after $k$ backups is $(1 - \gamma^k)/(1 - \gamma) \approx 75$; generalization lets the network get a little beyond it). that caps the estimates around 80 and makes good policies look worse than they are. at the same time, the network barely distinguishes "pole nearly vertical" from "pole about to fall": nearly all of its training data comes from states where things are fine, the failure states are rare, and the max in the target prefers whichever action looks best under the network's errors. so imminent failure is underweighted and bad policies look better than they are.

double DQN removes the *compounding of maximization bias*, which on a two-action task with deterministic dynamics and $\gamma = 0.99$ isn't the main source of error. the remaining errors (slow propagation, poor resolution of rare failure states, the policy changing under the values) affect both agents alike, and with three seeds per agent the differences between them are noise. van Hasselt et al.'s large overestimates appear with many actions, stochastic rewards and long training; the point here is that the value estimates of a learning agent shouldn't be trusted as values, even when the greedy policy is good. evaluate policies by running them.
