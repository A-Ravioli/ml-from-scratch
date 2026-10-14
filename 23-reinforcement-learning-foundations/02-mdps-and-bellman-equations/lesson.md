# MDPs & bellman equations

## why this matters

a bandit forgets everything between rounds: your choice now doesn't change what you can do next. almost nothing interesting works like that. a move in chess changes the board; a robot's step changes where it stands; a recommendation changes what the user will click tomorrow. actions have *consequences*, and a good action now can be one that pays nothing immediately but leads somewhere valuable.

to reason about that you need two things. first, a precise model of "an environment with state": the **Markov decision process** (MDP), which every algorithm in this chapter and the next assumes. second, a way to say how good a state is when the payoff is spread over the future: the **value function**, and the recursive equation it satisfies, the **Bellman equation**. that one equation is the backbone of reinforcement learning: dynamic programming solves it with a model (topic 03), Monte Carlo and TD methods estimate it from experience (topics 04–05), and DQN and actor–critic methods approximate it with neural networks (ch 24).

this topic also builds the plumbing. you'll write the environment protocol that every environment in chapters 23 and 24 follows (the one popularized by OpenAI Gym and its successor Gymnasium), a gridworld that exposes both the protocol *and* its full model as arrays, a function that runs an episode, and the solver for the Bellman equation of a fixed policy.

## warm-up: predict first

1. an agent pays $-1$ per step until it reaches a goal. a random walker starts 6 steps away on a $4\times4$ grid. how many steps does it take on average: about 6, about 20, or about 60?
2. with discount $\gamma = 0.5$, how much is a reward of 1 received ten steps from now worth today?
3. you add $+2$ to every reward of that grid (so each step pays $+1$). does the best policy change?
4. an episode ends because the time limit ran out, not because the agent reached the goal. should the value of the last state be zero?
5. how many unknowns and how many equations does it take to compute the values of a fixed policy on a 16-state grid?

## prerequisites check

- [ ] you can work with a row-stochastic transition matrix and simulate a Markov chain (→ ch 04 · topic 07)
- [ ] you can solve a linear system and know when $I - A$ is invertible (→ ch 02 · topic 03)
- [ ] you know geometric series, $\sum_{k\ge0}\gamma^k = 1/(1 - \gamma)$ for $|\gamma| < 1$ (→ ch 01)
- [ ] you've built bandit agents with `select_arm`/`update` and know why exploration matters (→ topic 01)

## core ideas

### idea 1: agents, environments, and the protocol

reinforcement learning splits the world into an **agent**, which chooses actions, and an **environment**, which responds. at each step $t$ the agent sees an **observation** $S_t$, picks an **action** $A_t$, and the environment returns a **reward** $R_{t+1}$ and the next observation $S_{t+1}$. the interaction runs in **episodes** that start from a reset and end at some point.

Gymnasium (the maintained version of OpenAI Gym, Brockman et al., 2016) standardized this as two methods:

```python
obs, info = env.reset(seed=None)
obs, reward, terminated, truncated, info = env.step(action)
```

two flags, not one, because there are two very different reasons for an episode to stop. **terminated** means the environment reached a terminal state: the goal, a crash, a lost game. nothing follows, so the future is worth zero. **truncated** means an outside limit cut the episode short, typically a time limit. the state you stopped in is an ordinary state with a future you just didn't get to see. mixing these up is one of the most common bugs in RL code (warm-up 4: no), and topic 05 will make you treat them differently.

environments describe what they accept and produce with **spaces**: `Discrete(n)` for the integers $0, \dots, n - 1$ (gridworld cells, actions) and `Box(low, high)` for real vectors in a box (a car's position and velocity, topic 06). an agent can ask `env.action_space.n` for the number of actions or `env.observation_space.shape` for the size of an observation without knowing anything else about the environment.

finally, randomness. an environment owns a generator, `self.np_random`. `reset(seed=s)` re-creates it from the seed, so an experiment can be replayed exactly; `reset()` without a seed keeps the existing stream, so consecutive episodes differ. that's Gymnasium's rule, and the course's convention: no global random state anywhere.

> **spec.** `Discrete(n)`: attribute `n` (an int) and `shape = ()`; `sample(rng=None)` returns a Python int drawn with `as_rng(rng)`; `contains(x)` is True only for integers (Python or NumPy ints, or 0-d integer arrays; never bools or floats) in $[0, n)$. `Box(low, high, shape=None)`: `shape` is a tuple (inferred by broadcasting `low` and `high` when `None`), and `low`, `high` are float arrays of that shape; `sample(rng=None)` is uniform in the box; `contains(x)` checks the shape and the bounds. `Env.reset(*, seed=None)` sets `self.np_random = np.random.default_rng(seed)` if a seed is given or no generator exists yet, and otherwise leaves it alone. subclasses call it first in their own `reset`.

> **▶ build: stage 1.** implement `Discrete`, `Box` and `Env.reset`.

### idea 2: Markov decision processes

an MDP is a tuple $(\mathcal S, \mathcal A, P, R, \gamma)$: states, actions, transition probabilities $P(s' \mid s, a)$, expected rewards $R(s, a) = \mathbb E[R_{t+1} \mid S_t = s, A_t = a]$, and a discount factor $\gamma \in [0, 1]$. the defining assumption is the **Markov property**: the next state and reward depend only on the current state and action, not on how you got there. the state summarizes the past.

for small problems you can write the whole MDP down as arrays. that's what **tabular** means in this chapter:

- `P` with shape $(S, A, S)$: `P[s, a, s2]` is $P(s_2 \mid s, a)$, so every `P[s, a, :]` sums to 1;
- `R` with shape $(S, A)$: the expected immediate reward.

terminal states fit in by being **absorbing**: every action loops back to the same state with reward 0. then "the episode ended" and "you stay in a state worth nothing forever" are the same thing mathematically, and all formulas work without special cases.

the **gridworld** is the classic test bed. the agent lives on a `height × width` grid; state $s = \text{row}\cdot\text{width} + \text{col}$; actions 0, 1, 2, 3 move up, right, down, left. moving off the grid or into a wall leaves you where you are. every step costs `step_reward` (default $-1$), so maximizing reward means reaching the goal fast. with `slip > 0` the floor is icy: with probability `slip`, your action is replaced by a uniformly random one (which may happen to be the one you chose), so

$$P(\cdot \mid s, a) = (1 - \text{slip})\,\delta_{\text{next}(s, a)} + \frac{\text{slip}}{4}\sum_{b}\delta_{\text{next}(s, b)}.$$

> **spec.** `GridWorld(height=4, width=4, *, start=(0, 0), goal=None, walls=(), slip=0.0, step_reward=-1.0, max_steps=100)` stores its arguments (`goal` defaults to the bottom-right cell) and builds `n_states`, `n_actions = 4`, `observation_space = Discrete(n_states)`, `action_space = Discrete(4)`, `start_state` (an int), `terminal` (a bool array $(S,)$, True only at the goal), `P` and `R` as above. every action in a non-terminal state costs `step_reward`, including the step *into* the goal. the goal is absorbing with reward 0; so are wall cells (they are states nobody can reach). a start or goal outside the grid, on a wall, or equal to each other raises `ValueError`.

> **▶ build: stage 2.** implement `GridWorld.__init__`.

### idea 3: returns, policies, and episodes

the agent wants a lot of reward, but rewards arrive over time. the **return** from step $t$ is the discounted sum of what follows:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3} + \dots = \sum_{k\ge0}\gamma^kR_{t+k+1} = R_{t+1} + \gamma\,G_{t+1}.$$

the last form, a return is the next reward plus the discounted return from the next step, is the seed of everything that follows. why discount? partly because uncertain or delayed rewards are worth less (and money now earns interest). mostly for mathematical convenience: with $\gamma < 1$ and bounded rewards, the sum is finite even if the episode never ends, and $|G_t| \le r_{\max}/(1 - \gamma)$. a useful reading of $\gamma$ is the **effective horizon** $1/(1 - \gamma)$: rewards much further away than that barely count. $\gamma = 0.9$ looks about 10 steps ahead, $\gamma = 0.99$ about 100; with $\gamma = 0.5$ a reward ten steps away is worth $0.5^{10} \approx 0.001$ of one now (warm-up 2). episodic tasks that are sure to end can use $\gamma = 1$.

a **policy** is how the agent chooses. tabular algorithms represent it as an $(S, A)$ array of probabilities $\pi(a \mid s)$; code that just runs episodes only needs a callable `policy(obs) -> action`. running an episode is the loop from idea 1, and you'll write it once as `rollout`.

> **spec.** `GridWorld.reset(*, seed=None)` calls `Env.reset`, sets `self.state = self.start_state` and the step counter `self.t = 0`, and returns `(self.state, {})`. `GridWorld.step(action)` raises `ValueError` for an action outside the action space; otherwise it draws $s' \sim$ `P[self.state, action]` using `self.np_random`, increments `self.t`, and returns `(s', float(R[s, action]), terminated, truncated, {})` with `terminated = terminal[s']` and `truncated` True when `t` reaches `max_steps` (`None`: never) without terminating: if the goal is reached on the last allowed step, the episode *terminated* and `truncated` stays False, so at most one flag is True. (Gymnasium's `TimeLimit` wrapper can set both; code that consumes either convention should check `terminated` first.) obs are Python ints, flags Python bools. `rollout(env, policy, *, max_steps=None, seed=None)` resets with the seed, steps until terminated or truncated or `max_steps` actions (which counts as truncated), and returns a dict with `observations` (a list of $T + 1$), `actions` and `rewards` (lists of $T$), `terminated`, `truncated` and `return` (the undiscounted sum, a float).

> **🔮 predict:** a uniformly random policy on the $4\times4$ grid, starting in the corner opposite the goal, with the default `max_steps=100`. what's the average return over many rollouts, and is it the same as the true expected return of the random walk?

> **▶ build: stage 3.** implement `GridWorld.reset`, `GridWorld.step` and `rollout`.

### idea 4: value functions and the Bellman equation

the **state-value function** of a policy is its expected return from each state, and the **action-value function** the expected return after taking a specific first action:

$$V^\pi(s) = \mathbb E_\pi\big[G_t \mid S_t = s\big],\qquad Q^\pi(s, a) = \mathbb E_\pi\big[G_t \mid S_t = s, A_t = a\big].$$

take expectations of $G_t = R_{t+1} + \gamma G_{t+1}$. the Markov property says that once you know $S_{t+1}$, the expected future is $V^\pi(S_{t+1})$, whatever came before. so

$$Q^\pi(s, a) = R(s, a) + \gamma\sum_{s'}P(s' \mid s, a)\,V^\pi(s'),\qquad V^\pi(s) = \sum_a\pi(a \mid s)\,Q^\pi(s, a).$$

combining them gives the **Bellman expectation equation**:

$$V^\pi(s) = \sum_a\pi(a \mid s)\Big[R(s, a) + \gamma\sum_{s'}P(s' \mid s, a)V^\pi(s')\Big].$$

it's a consistency condition: the value of a state is the expected immediate reward plus the discounted value of where you land. with $P_\pi(s, s') = \sum_a\pi(a\mid s)P(s'\mid s, a)$ and $R_\pi(s) = \sum_a\pi(a\mid s)R(s, a)$ it's a linear system,

$$V^\pi = R_\pi + \gamma P_\pi V^\pi \quad\Longleftrightarrow\quad (I - \gamma P_\pi)V^\pi = R_\pi,$$

$S$ equations in $S$ unknowns (warm-up 5). for $\gamma < 1$ the matrix is always invertible. a direct solve costs $O(S^3)$, fine for 16 states, not for a million. the alternative is to iterate the equation as an update, $V_{k+1} = R_\pi + \gamma P_\pi V_k$, starting from $V_0 = 0$. the update is a **contraction** in the max norm: $\|V_{k+1} - V^\pi\|_\infty \le \gamma\|V_k - V^\pi\|_\infty$ (on paper 3), so it converges geometrically from any start, at rate $\gamma$. with $\gamma = 1$ it still converges when every episode ends (the "effective" contraction comes from reaching the terminal state), but a policy that never terminates has $V = -\infty$ and the iteration runs forever: hence a `max_iters` cap. iteration also needs only matrix–vector products, which is the form that generalizes to sampled and approximate versions in the rest of the chapter.

> **spec.** `policy_evaluation(P, R, policy, gamma=0.99, *, tol=1e-8, max_iters=10000)`: start from zeros, apply $V \leftarrow R_\pi + \gamma P_\pi V$ until the largest absolute change is below `tol` or `max_iters` updates have been made, and return $V$ $(S,)$. (`np.einsum("sa,sat->st", policy, P)` builds $P_\pi$.) `q_from_v(P, R, V, gamma=0.99)` returns $R + \gamma PV$ with shape $(S, A)$.

> **🔮 predict:** on a $6\times6$ grid, the random policy's value in the far corner with $\gamma = 0.5$, $0.9$, $0.99$ and $1$. which of them still tells you anything about where the goal is?

> **▶ build: stage 4.** implement `policy_evaluation` and `q_from_v`.

## worked example

a $1\times3$ corridor, `GridWorld(1, 3)`: states 0, 1, 2, goal at 2, start at 0, reward $-1$ per step. the random policy picks each of the 4 actions with probability $\tfrac14$. from state 0, only "right" moves (to 1); up, down and left bump into the edge. from state 1, right reaches the goal, left goes back to 0, up and down stay. with $\gamma = 1$ and $V(2) = 0$, the Bellman equations are

$$V(0) = -1 + \tfrac34V(0) + \tfrac14V(1),\qquad V(1) = -1 + \tfrac14V(0) + \tfrac12V(1) + \tfrac14\cdot0.$$

the first gives $V(0) = V(1) - 4$; substituting into the second, $\tfrac12V(1) = -1 + \tfrac14(V(1) - 4)$, so $V(1) = -8$ and $V(0) = -12$. a random walker from the far end needs 12 steps on average to cover a distance of 2.

the action values follow from `q_from_v`: $Q(1, \text{right}) = -1 + V(2) = -1$, $Q(1, \text{left}) = -1 + V(0) = -13$, $Q(1, \text{up}) = Q(1, \text{down}) = -1 + V(1) = -9$, and averaging with the policy, $\tfrac14(-1 - 13 - 9 - 9) = -8 = V(1)$. your `policy_evaluation` and `q_from_v` must reproduce these numbers.

## on paper

1. **the Bellman equation.** starting from $G_t = R_{t+1} + \gamma G_{t+1}$ and the Markov property, derive $V^\pi(s) = \sum_a\pi(a\mid s)\big[R(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')\big]$.

<details><summary>solution</summary>

$V^\pi(s) = \mathbb E[R_{t+1} \mid S_t = s] + \gamma\,\mathbb E[G_{t+1} \mid S_t = s]$. condition on the action and next state: $\mathbb E[G_{t+1} \mid S_t = s] = \sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s, a)\,\mathbb E[G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$. by the Markov property (and because the policy depends only on the current state), the last expectation doesn't depend on $s$ or $a$: it's $\mathbb E[G_{t+1} \mid S_{t+1} = s'] = V^\pi(s')$. the first term is $\sum_a\pi(a\mid s)R(s, a)$. put them together.

</details>

2. **a slippery corridor.** redo the worked example's $V(1)$ for the *deterministic* policy "always right" with `slip=0.5`, $\gamma = 1$.

<details><summary>solution</summary>

from state 1 the intended move happens with probability $0.5 + 0.5\cdot\tfrac14 = 0.625$; left with $0.125$; up or down (stay) with $0.25$. from state 0: right with $0.625$, otherwise stay. so $V(0) = -1 + 0.375V(0) + 0.625V(1)$ and $V(1) = -1 + 0.125V(0) + 0.25V(1)$. the first gives $V(0) = V(1) - 1.6$; the second, $0.75V(1) = -1 + 0.125(V(1) - 1.6)$, so $V(1) = -1.2/0.625 = -1.92$ and $V(0) = -3.52$.

</details>

3. **contraction.** let $T_\pi V = R_\pi + \gamma P_\pi V$. show $\|T_\pi V - T_\pi U\|_\infty \le \gamma\|V - U\|_\infty$, and conclude that iterative policy evaluation converges to the unique solution for $\gamma < 1$.

<details><summary>solution</summary>

$T_\pi V - T_\pi U = \gamma P_\pi(V - U)$. each row of $P_\pi$ is a probability vector, so each entry of $P_\pi(V - U)$ is an average of entries of $V - U$, at most $\|V - U\|_\infty$ in absolute value. so $T_\pi$ is a $\gamma$-contraction, and by Banach's fixed-point theorem it has exactly one fixed point $V^\pi$, with $\|V_k - V^\pi\|_\infty \le \gamma^k\|V_0 - V^\pi\|_\infty$.

</details>

4. **shifting rewards.** (a) in a task that never ends, show that adding a constant $c$ to every reward adds $c/(1 - \gamma)$ to every value, so no policy comparison changes. (b) on a $5\times5$ gridworld (goal in the opposite corner), compare "walk straight to the goal (8 steps)" and "walk into the top wall forever" with $\gamma = 0.95$, with rewards $-1$ and then $+1$ per step.

<details><summary>solution</summary>

(a) $G_t' = \sum_k\gamma^k(R_{t+k+1} + c) = G_t + c/(1 - \gamma)$ for every trajectory, so every value shifts by the same amount. (b) straight: $\mp\sum_{k=0}^{7}0.95^k = \mp6.73$; forever: $\mp1/(1 - 0.95) = \mp20$. with $-1$ per step the straight walk is better ($-6.7 > -20$); with $+1$ per step the agent prefers never to finish ($20 > 6.7$). in an episodic task the shift isn't uniform, because episodes of different lengths collect different numbers of $c$'s: the terminal state, worth 0, is the thing that changes value relative to the rest (warm-up 3).

</details>

## experiment

`lab.py` investigates three things with your code:

1. what the random policy's values look like on a $6\times6$ grid as you vary $\gamma$.
2. which of three policies is best when every step costs $-1$, and when every step pays $+1$.
3. whether averaging `rollout` returns agrees with `policy_evaluation`, with and without a time limit.

## check your understanding

1. why does `step` return two flags instead of a single `done`?

<details><summary>answer</summary>

because they mean different things for learning. terminated: the future is worth 0 (the state is absorbing). truncated: the episode was cut off, but the state still has a future value. an agent that treats a time-out like a terminal state learns that states near the time limit are worth less, which isn't a property of the environment.

</details>

2. the Markov property is a property of the *state*. give an example where a natural observation isn't Markov, and how to fix it.

<details><summary>answer</summary>

a single frame of a video game: from one image you can't tell which way a ball is moving. stacking the last few frames (as DQN does, ch 24), or adding the velocity to the observation (as mountain car does, topic 06), restores the Markov property.

</details>

3. (warm-up 1.) the random walker on the $4\times4$ grid is 6 steps from the goal. why does it need about 60 steps on average?

<details><summary>answer</summary>

a random walk wanders: half of its moves go away from the goal or into walls, and it revisits states many times. `policy_evaluation` gives $V = -59.4$ at the start: about 60 steps.

</details>

4. why must the iteration in `policy_evaluation` have a `max_iters` cap even though it's a contraction?

<details><summary>answer</summary>

it's a contraction only for $\gamma < 1$ (or for $\gamma = 1$ when every policy reaches a terminal state). with $\gamma = 1$ and a policy that never terminates (walking into a wall forever), $V = -\infty$ and the iteration decreases by 1 forever. even for $\gamma$ close to 1 it needs about $\ln(\text{error}/\text{tol})/(1 - \gamma)$ iterations.

</details>

5. what's the relationship between $V^\pi$ and $Q^\pi$, in both directions?

<details><summary>answer</summary>

$V^\pi(s) = \sum_a\pi(a\mid s)Q^\pi(s, a)$ (average over the policy's first action), and $Q^\pi(s, a) = R(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')$ (look one step ahead with the model). the second direction needs the model $P$, $R$; the first doesn't. that asymmetry is why model-free control (topics 04–05) learns $Q$ rather than $V$.

</details>

6. you set `slip=1.0`. what does the gridworld become, and what does that do to every policy's value?

<details><summary>answer</summary>

every action is replaced by a uniformly random one, so all policies behave like the random policy and have the same value. the actions no longer matter: it's a Markov chain, not a decision process.

</details>

## common bugs

- **row versus column.** symptom: "right" moves you down. `s = row * width + col`, so moving right is `s + 1` and moving down is `s + width`.
- **forgetting absorbing terminals.** symptom: `P[goal]` is all zeros and the rows don't sum to 1, or values leak past the goal. make terminal states loop to themselves with reward 0.
- **reseeding on every reset.** symptom: every episode is identical. only create a new generator when `seed` is given (or none exists yet).
- **`done = terminated or truncated` too early.** symptom: everything works now and breaks in topic 05. keep both flags all the way through `rollout`.
- **`P @ V` on the wrong axis.** symptom: `q_from_v` returns the wrong shape or nonsense. with `P` of shape $(S, A, S)$, `P @ V` contracts the last axis, giving $(S, A)$.
- **no iteration cap.** symptom: `policy_evaluation` hangs on a policy that never reaches the goal with $\gamma = 1$.

## going deeper

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, chapter 3 (finite MDPs) and §4.1 (policy evaluation). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Puterman (1994), *Markov Decision Processes: Discrete Stochastic Dynamic Programming*, Wiley. the rigorous theory. [doi:10.1002/9780470316887](https://doi.org/10.1002/9780470316887)
- Bellman (1957), *Dynamic Programming*, Princeton University Press. where the equation comes from.
- Brockman et al. (2016), "OpenAI Gym". the environment interface. [arXiv:1606.01540](https://arxiv.org/abs/1606.01540)
- Towers et al. (2024), "Gymnasium: a standard interface for reinforcement learning environments". the two-flag `step` and seeding rules used here. [arXiv:2407.17032](https://arxiv.org/abs/2407.17032)
- Pardo, Tavakoli, Levdik & Kormushev (2018), "Time limits in reinforcement learning", *ICML*. why truncation isn't termination. [arXiv:1712.00378](https://arxiv.org/abs/1712.00378)

## connections

- ← builds on: Markov chains and transition matrices (ch 04 · topic 07), linear solves (ch 02 · topic 03), the exploration problem (topic 01), `as_rng` (ch 00).
- → used in: policy and value iteration (topic 03), every environment in this chapter (`Blackjack`, `CliffWalking`, `MountainCar` and your capstone game subclass `Env`), Monte Carlo and TD estimation of $V^\pi$ (topics 04–05), the replay buffers, DQN targets and GAE of ch 24, which all distinguish `terminated` from `truncated`.
