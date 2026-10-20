# MDPs & bellman equations · lab notes

## experiment 1: how far does the agent see?

**what people usually predict:** that the values grow smoothly towards the goal for every $\gamma$, and that $V(\text{far corner})$ is about minus the distance to the goal (10) with $\gamma = 1$.

**what happens:** with $\gamma = 0.5$, the far corner is worth **−2.00**, the cell next to the goal **−1.70**: almost the whole map is a flat $-2$, and only the two or three cells nearest the goal differ. with $\gamma = 0.9$: **−9.92** versus **−5.86**. with $\gamma = 0.99$: **−67.9** versus **−29.6**, a clear gradient across the whole grid. with $\gamma = 1$: **−170.3** in the far corner and **−70.0** next to the goal.

**why:** with a reward of $-1$ per step, $V$ is minus the expected *discounted* number of steps, which can never be below $-\sum_k\gamma^k = -1/(1 - \gamma)$. for small $\gamma$, the random walker from a distant state almost never reaches the goal within the effective horizon $1/(1 - \gamma)$ (2 steps for $\gamma = 0.5$), so its value sits at the floor: from the agent's point of view, a state 4 steps from the goal and one 10 steps away look identical. the goal only becomes visible in the values once the horizon is comparable with the time the policy needs to get there. with $\gamma = 1$ there's no floor, and the values measure the random walk's expected hitting time, which is far longer than the distance: even the cell *next* to the goal is worth $-70$, because the random walker steps into the goal only a quarter of the time and otherwise wanders off for a long time. the lesson for later topics: $\gamma$ is part of the problem definition, not just a convergence trick, and too small a $\gamma$ makes distant goals invisible.

## experiment 2: pay the agent to walk

**what people usually predict:** that the direct policy stays best, because adding a constant to every reward "doesn't change the optimal policy".

**what happens:** with $-1$ per step: direct **−6.73**, wall **−20.00**, random **−18.32**; direct is best. with $+1$ per step every number flips sign, and **wall** (+20.00) is best: the agent prefers to bump into a wall forever rather than reach the goal.

**why:** the goal (and every terminal state) is worth exactly 0 and isn't shifted, while every non-terminal step is. a policy's value is (reward per step) × (expected discounted number of steps before termination), and the second factor is a property of the policy alone: 6.73 for direct, $1/(1 - \gamma) = 20$ for a policy that never terminates. so changing the sign of the per-step reward reverses the ranking. the invariance to constant shifts (on paper 4) needs every trajectory to collect the same number of discounted rewards, which holds only in continuing tasks. in episodic tasks, a positive per-step reward makes termination something to avoid. that's a classic reward-design bug: "+1 for every frame you survive" teaches an agent to stall, not to win.

## experiment 3: rollouts versus the Bellman equation

**what people usually predict:** that both averages match $V(\text{start})$ up to noise, since rollouts sample exactly the returns $V$ averages.

**what happens:** `policy_evaluation` gives $V(\text{start}) = -59.43$. without a time limit the 2,000 rollouts average **−60.20 ± 1.16**: consistent. with the default `max_steps = 100` they average **−51.15 ± 0.69**: about 8 too high, 12 standard errors away. **16.2%** of the episodes hit the limit, and the histogram shows them as a spike at exactly $-100$; without the limit the left tail continues past $-300$.

**why:** the random walk's return distribution is heavily skewed: most episodes end within 50 steps, but a long tail wanders for hundreds. truncation replaces every return below $-100$ by exactly $-100$, so the truncated average is biased towards zero, and no number of rollouts fixes that. the Bellman equation describes the time-limit-free MDP; a time limit turns it into a different, non-stationary problem (the time step becomes part of the state). an agent that treats the truncated states as terminal (value 0) learns that states visited late in an episode are better than they are; that's why `step` returns `truncated` separately, and why topic 05's TD methods bootstrap from the next state's value on truncation but not on termination.
