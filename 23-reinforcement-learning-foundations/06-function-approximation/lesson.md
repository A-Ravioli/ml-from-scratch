# function approximation

## why this matters

every method so far keeps one number per state (or state–action pair). that works for 48 cliff cells and 360 blackjack hands. it doesn't work for a car whose state is a real-valued position and velocity: there are infinitely many states, you'll never visit the same one twice, and a table learns nothing about a state it hasn't seen. the fix is the one supervised learning made routine: represent the value function as a parametrized function, $\hat v(s; w)$ or $\hat q(s, a; w)$, and let what you learn about one state *generalize* to similar ones.

that sounds like a drop-in replacement, and partly it is: this topic builds a linear SARSA agent with tile-coded features that learns to drive an underpowered car up a mountain in a hundred episodes. but function approximation also breaks things that tabular RL took for granted. the TD update isn't the gradient of anything; the values it converges to aren't the best fit; and when it's combined with bootstrapping and off-policy data it can diverge, even on a seven-state problem where the right answer is all zeros. that combination, the **deadly triad**, is exactly what deep Q-networks use (ch 24), and the reason they need target networks and replay buffers.

## warm-up: predict first

1. a $10\times10$ grid of tiles over a 2-D state space, and you update the value of one state. how many other states' values change?
2. 8 grids, each offset slightly, each $8\times8$. same question. how finely can the values vary?
3. the car's engine can't climb the hill directly. with no random exploration at all ($\varepsilon = 0$), can a learning agent still find the solution?
4. the TD update with a linear $\hat v$ moves $w$ along $\nabla\hat v(S_t)$ by the TD error. is that gradient descent on some loss?
5. seven states, every reward 0, values linear in 8 weights. can semi-gradient TD, with exact expected updates and no noise, make the weights go to infinity?

## prerequisites check

- [ ] you know SARSA, Q-learning, TD(0) and the role of the TD error (→ topic 05)
- [ ] you know linear regression and stochastic gradient descent (→ ch 06 · topic 04, ch 08)
- [ ] you know that a linear iteration $w \leftarrow w + \alpha(b - Aw)$ converges iff every eigenvalue of $\alpha A$ lies where $|1 - \alpha\lambda| < 1$, e.g. all real parts positive for small $\alpha$ (→ ch 02 · topic 05, ch 06 · topic 03)
- [ ] you can work with `Box` observation spaces (→ topic 02)

## core ideas

### idea 1: features, and tile coding

a **linear** value function is $\hat v(s; w) = w^\top x(s) = \sum_iw_ix_i(s)$ for a feature vector $x(s)$. everything interesting is in the features. they decide what generalizes to what: two states with similar features get similar values, whether or not you've visited both.

**tile coding** (also called CMAC, Albus 1975) is the classic choice for low-dimensional continuous states. cover the state space with a grid of tiles; the feature of each tile is 1 if the state lies in it, 0 otherwise. one grid gives a piecewise-constant function: an update to one state changes the value of every state in the same tile and no other (warm-up 1). coarse tiles generalize broadly but can't represent detail; fine tiles represent detail but generalize to almost nothing, so you need data in every tile.

tile coding gets both by using several grids (**tilings**), each **offset** by a fraction of a tile. a state activates exactly one tile per tiling, so $x(s)$ has exactly $n_{\text{tilings}}$ ones. two states share as many active tiles as there are tilings in which they fall in the same tile, so the generalization between them decays smoothly with distance, out to about one tile width. and because the tilings' boundaries are staggered, the value function can change every $1/n_{\text{tilings}}$ of a tile width: the resolution of a grid $n_{\text{tilings}}$ times finer (warm-up 2). the feature vector is sparse, so you never build it: you compute the list of active indices, and $\hat v(s) = \sum_{i\in\text{active}}w_i$.

how should the tilings be offset? shifting every tiling by the same amount along the diagonal makes the generalization pattern diagonal. Sutton & Barto (§9.5.4) recommend asymmetric displacements: tiling $i$ shifted by $i\cdot(1, 3, 5, \dots)/n_{\text{tilings}}$ tile widths (mod 1) in dimensions $1, 2, 3, \dots$.

> **spec.** `TileCoder(low, high, *, n_tilings=8, tiles_per_dim=8, rng=None)`: `low` and `high` become float arrays of shape $(d,)$; the tile width is `width = (high - low) / (tiles_per_dim - 1)`, so that `tiles_per_dim` tiles cover the box even after an offset of up to one tile. `offsets` has shape `(n_tilings, d)`, in units of tile widths: `(i * (2j + 1) / n_tilings) % 1` for tiling $i$ and dimension $j$ when `rng` is None, and `as_rng(rng).random((n_tilings, d))` otherwise. `n_features = n_tilings * tiles_per_dim ** d`. calling the coder on an observation clips it to the box, computes $u = (x - \text{low})/\text{width}$, the tile coordinates `floor(u + offsets[i])` (capped at `tiles_per_dim - 1`) in each tiling, and returns the int array of `i * tiles_per_dim ** d + ravel_multi_index(coords)` for $i = 0, \dots, n_{\text{tilings}} - 1$.

> **▶ build: stage 1.** implement `TileCoder`.

### idea 2: mountain car

**mountain car** (Moore, 1990; Sutton & Barto, Example 10.1): a car sits in a valley between two hills and must reach the top of the right one. its engine is too weak to drive straight up; it has to back up the left hill first and use gravity to build momentum. the state is position $x \in [-1.2, 0.5]$ and velocity $\dot x \in [-0.07, 0.07]$; the three actions are full reverse, coast and full forward ($a = 0, 1, 2$). the dynamics:

$$\dot x_{t+1} = \operatorname{clip}\big(\dot x_t + 0.001(a_t - 1) - 0.0025\cos(3x_t),\ \pm0.07\big),\qquad x_{t+1} = \operatorname{clip}(x_t + \dot x_{t+1},\ -1.2, 0.5),$$

with the velocity reset to 0 when the car hits the left wall. each step costs $-1$; reaching $x = 0.5$ ends the episode. episodes start at rest at a random position in $[-0.6, -0.4)$.

it's hard for two reasons. the obvious greedy behaviour (drive towards the goal) never works: success requires first moving *away* from the goal. and the reward gives no hint: every step costs $-1$ until, suddenly, the episode ends. a random policy almost never gets there within 200 steps. what saves a value-based learner is **optimistic initialization**: starting at $\hat q = 0$ when the true values are around $-100$ makes every untried action look better than every tried one, so the agent systematically tries new things, even with $\varepsilon = 0$ (warm-up 3).

> **spec.** `MountainCar(*, max_steps=200)` subclasses `Env`: `observation_space = Box([-1.2, -0.07], [0.5, 0.07])`, `action_space = Discrete(3)`. `reset` calls `Env.reset`, draws the position with `self.np_random.uniform(-0.6, -0.4)`, sets `self.state = np.array([x, 0.0])` and the step counter `self.t = 0`, and returns `(self.state.copy(), {})`. `step` (invalid actions raise `ValueError`) applies the dynamics above, increments `t`, and returns `(new state copy, -1.0, terminated, truncated, {})` with `terminated = x >= 0.5` and `truncated` when `t` reaches `max_steps`. always return a *fresh* array: anything that stores observations (ch 24's replay buffer) breaks if you hand out the same array every step.

> **🔮 predict:** the hand-coded policy "push in the direction you're moving" (forward if $\dot x \ge 0$, else reverse). how many steps does it need from the usual start: about 50, about 120, or never?

> **▶ build: stage 2.** implement `MountainCar`.

### idea 3: semi-gradient SARSA

if someone handed you the true values $v_\pi(s)$, you'd fit $w$ by stochastic gradient descent on the squared error: $w \leftarrow w + \alpha\,[v_\pi(S_t) - \hat v(S_t; w)]\,\nabla_w\hat v(S_t; w)$. nobody hands you $v_\pi$, so substitute a target, exactly as in the tabular case: a Monte Carlo return $G_t$ (then it really is SGD, on noisy but unbiased targets), or a bootstrapped TD target $R_{t+1} + \gamma\hat v(S_{t+1}; w)$. the TD target depends on $w$ too, but the update ignores that dependence: it differentiates only the prediction, not the target. that's why it's called a **semi-gradient** method (warm-up 4: it isn't the gradient of any fixed objective). it's still the method everyone uses: it learns faster than the true gradient of the TD error, and with linear features on-policy it provably converges (Tsitsiklis & Van Roy, 1997).

with linear features $\nabla_w\hat v = x(s)$, so the update touches only the active features. for control, keep one weight vector per action, $\hat q(s, a) = \sum_{i\in\text{active}(s)}w_{i,a}$, and use the SARSA target:

$$\delta = R_{t+1} + \gamma\hat q(S_{t+1}, A_{t+1}) - \hat q(S_t, A_t),\qquad w_{i,A_t} \leftarrow w_{i,A_t} + \frac{\alpha}{n_{\text{active}}}\,\delta\quad\text{for each active } i.$$

dividing by the number of active features makes $\alpha$ mean the same thing as in the tabular case: $\alpha = 1$ moves $\hat q(S_t, A_t)$ all the way to its target in one update (Sutton & Barto use $\alpha = 0.5/8$ per weight for 8 tilings, i.e. $\alpha = 0.5$ here).

> **spec.** `linear_sarsa(env, featurizer, n_episodes, *, alpha=0.1, gamma=1.0, epsilon=0.0, rng=None)`: `featurizer(obs)` returns active indices and has `n_features`; $w$ starts at zeros of shape `(n_features, n_actions)`. same structure as your `sarsa` (env seeding, first action right after reset, next action after each step unless terminated, `epsilon_greedy` on $\hat q(s, \cdot)$ = `w[idx].sum(axis=0)`), with the update above. returns `(w, episode_returns)`.

> **▶ build: stage 3.** implement `linear_sarsa`.

### idea 4: the deadly triad

where does semi-gradient TD converge? take expectations of the TD(0) update with linear features, states drawn from a distribution $d$ (a diagonal matrix $D$), and transitions from the policy being evaluated, $P$:

$$\mathbb E[\Delta w] = \alpha\,\Phi^\top D\big(r + \gamma P\Phi w - \Phi w\big) = \alpha(b - Aw),\qquad A = \Phi^\top D(\Phi - \gamma P\Phi),\quad b = \Phi^\top Dr,$$

where $\Phi$ stacks the feature vectors as rows. it's a linear iteration. if it converges, it converges to the **TD fixed point** $w_{TD} = A^{-1}b$ (not the least-squares fit of $v_\pi$; its error is within a factor $1/(1 - \gamma)$ of the best possible). it converges for small $\alpha$ iff every eigenvalue of $A$ has a positive real part. when $d$ is the **on-policy** distribution (the stationary distribution of $P$ itself) with $d > 0$ everywhere and $\Phi$ of full column rank, $A$ is positive definite and all is well (Tsitsiklis & Van Roy, 1997). (without those conditions $A$ can be singular, as in the lab's on-policy run on Baird's example, where $d$ sits on one state: then some directions just don't move.) with tabular features ($\Phi = I$), $A = D(I - \gamma P)$ is also stable for any $d$ with positive entries. but if the states are weighted by a different distribution, as happens when you learn off-policy from another policy's data, $A$ can have eigenvalues with negative real parts, and the weights diverge.

Sutton & Barto call the combination the **deadly triad**: function approximation, bootstrapping, and off-policy training. any two are fine; all three can diverge. Baird's counterexample (1995) is the minimal demonstration: seven states, all rewards zero, a linear parametrization that can represent the true values (all zero) exactly, and yet expected off-policy semi-gradient TD sends $\|w\| \to \infty$ (warm-up 5: yes). there's no noise to blame: it's the geometry of $A$. remedies exist (gradient-TD methods, emphatic TD, or giving up one leg of the triad), and deep RL's tricks (target networks, replay of recent data, double Q-learning) are partly ways to keep this instability at bay.

> **spec.** `td_matrices(Phi, P, r, d, gamma)`: `Phi` $(S, k)$, `P` $(S, S)$, `r` $(S,)$, `d` $(S,)$; returns `(A, b)` as above. `expected_td(Phi, P, r, d, gamma, *, alpha=0.01, n_iters=1000, w0=None)` iterates $w \leftarrow w + \alpha(b - Aw)$ from `w0` (zeros by default; never modify the caller's array) and returns all iterates as an array of shape `(n_iters + 1, k)`.

> **🔮 predict:** in Baird's counterexample with $\gamma = 0.99$, does the on-policy weighting (all weight on the state the target policy visits) converge? does a smaller $\gamma$ change anything for the off-policy weighting?

> **▶ build: stage 4.** implement `td_matrices` and `expected_td`.

## worked example

**tile coding in 1-D.** `TileCoder(0, 1, n_tilings=2, tiles_per_dim=5)`: width $0.25$, offsets $(0, 0.5)$, 10 features. $x = 0.3$ is $u = 1.2$ widths in: tiling 0 is in tile $\lfloor1.2\rfloor = 1$, tiling 1 in tile $\lfloor1.7\rfloor = 1$, which is feature $5 + 1 = 6$. so `tc(0.3) = [1, 6]`. for $x = 0.4$ ($u = 1.6$): $\lfloor1.6\rfloor = 1$ and $\lfloor2.1\rfloor = 2$, so `[1, 7]`: the two points share one of their two tiles.

**one step of mountain car.** from $x = -0.5$ at rest, full forward: $\dot x' = 0 + 0.001 - 0.0025\cos(-1.5) = 0.001 - 0.000177 = 0.000823$, and $x' = -0.5 + 0.000823 = -0.499177$. the hill pushes back almost as hard as the engine.

**a linear SARSA update.** two states with active features $\{0, 1\}$ and $\{1, 2\}$ (they share feature 1), $\alpha = 0.5$, all weights 0. the first transition has $\delta = 0 + \hat q(s_1, a') - 0 = 0$: nothing changes. the second, from $s_1$ with action 1, terminates with reward $+1$: $\delta = 1$, so $w_{1,1}$ and $w_{2,1}$ each grow by $\alpha/2 = 0.25$, and $\hat q(s_1, 1) = 0.5$. because feature 1 is shared, $\hat q(s_0, 1)$ also became $0.25$, without ever being updated: that's generalization.

**the TD matrices with tabular features.** with $\Phi = I$: $A = D(I - \gamma P)$ and $b = Dr$, so $A^{-1}b = (I - \gamma P)^{-1}r = v_\pi$, whatever $d$ is. your code must reproduce all of these.

## on paper

1. **generalization of tile coding.** in 1-D with $n$ tilings, uniform offsets $i/n$, and tile width $h$, show that two points at distance $\Delta < h$ share about $n(1 - \Delta/h)$ active tiles. what does a one-sample update at $x$ do to $\hat v(x + \Delta)$?

<details><summary>solution</summary>

in each tiling, the tile boundaries fall at $x$-positions spaced $h$ apart, and $x$ and $x + \Delta$ are separated by a boundary iff one falls in the interval $(x, x + \Delta]$, which has probability $\Delta/h$ over the offsets; with $n$ evenly spread offsets that's about $n\Delta/h$ tilings. so they share about $n(1 - \Delta/h)$ tiles. an update of $\hat v(x)$ by $\alpha\delta$ (spread as $\alpha\delta/n$ per active weight) moves $\hat v(x + \Delta)$ by $\alpha\delta(1 - \Delta/h)$: a triangular "tent" of width $2h$.

</details>

2. **the semi-gradient isn't a gradient.** for one transition $s \to s'$ with reward $r$, write $J(w) = \frac12(r + \gamma w^\top x' - w^\top x)^2$. compute $\nabla J$ and compare it with the semi-gradient TD update direction $\delta x$.

<details><summary>solution</summary>

$\nabla J = \delta(\gamma x' - x)$, so gradient descent moves along $-\nabla J = \delta(x - \gamma x')$: it also pushes $\hat v(s')$ towards $\hat v(s)$ ("residual gradient", Baird 1995). semi-gradient TD moves along $\delta x$ only. the semi-gradient direction isn't $\nabla$ of anything in general: its Jacobian $-x(x - \gamma x')^\top$ isn't symmetric.

</details>

3. **why tabular TD is safe.** with $\Phi = I$, $A = D(I - \gamma P)$ for a row-stochastic $P$ and positive $d$. show that $A$'s eigenvalues have positive real parts. (hint: $I - \gamma P$ is strictly diagonally dominant by rows; what does multiplying by a positive diagonal $D$ do?)

<details><summary>solution</summary>

row $s$ of $D(I - \gamma P)$ has diagonal $d_s(1 - \gamma P_{ss})$ and off-diagonal absolute sum $d_s\gamma\sum_{s'\ne s}P_{ss'} = d_s\gamma(1 - P_{ss})$; the difference is $d_s(1 - \gamma) > 0$. by Gershgorin's theorem every eigenvalue lies in a disc centred at a positive diagonal entry with a smaller radius, so its real part is positive.

</details>

4. **Baird's divergence.** in Baird's example all rewards are 0 and every transition (under the target policy) goes to state 7. write the off-policy expected update for $w_7$ and $w_8$ alone (assume $w_1 = \dots = w_6 = c$). explain in words why increasing $w_8$ to fix state 7 makes the error at states 1–6 grow.

<details><summary>solution</summary>

the TD error at a state $s \in \{1..6\}$ is $\gamma\hat v(7) - \hat v(s) = \gamma(w_7 + 2w_8) - (2c + w_8)$, and at state 7, $(\gamma - 1)(w_7 + 2w_8)$. with uniform weighting, the six states 1–6 dominate: their TD errors push $w_8$ (shared by all of them with weight 1) and the $w_s$ up whenever $\hat v(7)$ is large, while only one state (with a small error $(1 - \gamma)\hat v(7)$) pushes $\hat v(7)$ down. but raising $w_8$ raises $\hat v(7)$ twice as fast as it raises $\hat v(1..6)$, so the targets $\gamma\hat v(7)$ run away from the predictions: each step to fix states 1–6 makes their target larger. under the on-policy weighting, state 7's own update dominates and pulls $\hat v(7)$ to 0.

</details>

## experiment

`lab.py` investigates three things with your code:

1. `linear_sarsa` with tile coding on mountain car, with $\varepsilon = 0$, for several step sizes, and the cost-to-go it learns.
2. coarse, fine and multi-tiling coders fitting a sine from 40 samples.
3. Baird's counterexample: off-policy versus on-policy weighting versus tabular features, as $\gamma$ varies.

## check your understanding

1. why divide the step size by the number of active features?

<details><summary>answer</summary>

every active weight moves by $\alpha\delta/n$, so the prediction $\hat q(s, a) = \sum_{\text{active}}w$ moves by exactly $\alpha\delta$, the tabular step. without the division, 8 tilings would effectively use $8\alpha$ and overshoot.

</details>

2. (the predict callout in idea 2.) how long does "push in the direction you're moving" take?

<details><summary>answer</summary>

about 120 steps from the usual start (the checks see 116–124): it pumps energy into the oscillation like someone on a swing. a learned policy does slightly better by choosing *when* to switch.

</details>

3. optimistic initial values drive exploration in mountain car. why does that stop working with function approximation in general?

<details><summary>answer</summary>

generalization spreads updates to unvisited states: once nearby states have been updated to realistic (negative) values, the "optimism" of an unvisited state is gone before it was ever tried. with tile coding the reach is about one tile, so states far from anything visited stay optimistic for a while; with a neural network everything is coupled and initial optimism is lost almost immediately.

</details>

4. the TD fixed point isn't the least-squares fit of $v_\pi$. why might it still be preferable?

<details><summary>answer</summary>

the least-squares fit needs the true values (Monte Carlo returns, high variance). TD's fixed point comes from low-variance one-step targets, and its error is bounded by $1/(1 - \gamma)$ times the best achievable (Tsitsiklis & Van Roy, 1997); in practice it often learns faster from limited data.

</details>

5. which leg of the deadly triad does each of these give up: Monte Carlo targets, on-policy SARSA, tabular Q-learning?

<details><summary>answer</summary>

Monte Carlo targets give up bootstrapping; on-policy SARSA gives up off-policy training; tabular Q-learning gives up function approximation. Q-learning *with* function approximation (DQN) has all three.

</details>

6. (the predict callout in idea 4.) does a smaller $\gamma$ help in Baird's example?

<details><summary>answer</summary>

yes: for $\gamma$ below about 0.88 the off-policy $A$ has no eigenvalue with negative real part and the iteration doesn't diverge (lab, experiment 3). bootstrapping matters less when the bootstrapped term is discounted more, and with $\gamma = 0$ there's no bootstrapping at all.

</details>

## common bugs

- **the wrong tile width.** symptom: the top corner of the box maps to an index out of range, or tilings don't cover the box. with offsets up to one tile, the width must be `(high - low) / (tiles_per_dim - 1)`.
- **not clipping observations.** symptom: negative or too-large indices for states slightly outside the box.
- **reusing the observation array.** symptom: stored trajectories all show the final state. return `self.state.copy()`.
- **forgetting $\alpha/n$.** symptom: mountain car diverges or oscillates with step sizes that should work.
- **updating with the next state's features.** symptom: learning goes nowhere. keep the current state's active indices until after the update.
- **treating the time limit as termination.** symptom: values near $-200$ look better than they are. bootstrap on truncation (topic 05).

## going deeper

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, chapters 9 (on-policy prediction with approximation, tile coding §9.5.4), 10 (on-policy control, mountain car) and 11 (off-policy methods with approximation, Baird's counterexample, the deadly triad). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Tsitsiklis & Van Roy (1997), "An analysis of temporal-difference learning with function approximation", *IEEE Transactions on Automatic Control* 42(5). convergence and the fixed-point bound. [doi:10.1109/9.580874](https://doi.org/10.1109/9.580874)
- Baird (1995), "Residual algorithms: reinforcement learning with function approximation", *ICML*. the counterexample. [doi:10.1016/B978-1-55860-377-6.50013-X](https://doi.org/10.1016/B978-1-55860-377-6.50013-X)
- Sutton (1996), "Generalization in reinforcement learning: successful examples using sparse coarse coding", *NeurIPS*. tile coding on mountain car and other control problems.
- Moore (1990), *Efficient Memory-Based Learning for Robot Control*, PhD thesis, University of Cambridge. where mountain car comes from.
- van Hasselt, Doron, Strub, Hessel, Sonnerat & Modayil (2018), "Deep reinforcement learning and the deadly triad". how often the triad bites in deep RL. [arXiv:1812.02648](https://arxiv.org/abs/1812.02648)

## connections

- ← builds on: TD learning and SARSA (topic 05), `Box` spaces (topic 02), SGD (ch 06 · topic 04), linear models (ch 08), eigenvalues and linear iterations (ch 02, ch 06).
- → used in: the capstone's approximate agent, DQN with neural features, target networks and replay (ch 24 · topic 01), linear and neural critics in actor–critic methods (ch 24 · topic 03), offline RL's extrapolation problems (ch 24 · topic 07).
