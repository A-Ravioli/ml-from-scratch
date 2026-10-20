# temporal-difference learning · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`CliffWalking.__init__`: call `super().__init__(4, 12, start=(3, 0), goal=(3, 11), max_steps=1000)`, which builds `P` and `R` for an ordinary grid, then patch them: for every non-terminal, non-cliff state and action whose (deterministic) next state is a cliff cell, send it to `start_state` with reward $-100$. the cliff cells are $3 \cdot 12 + c$ for $c = 1..10$. `td0_prediction` is one loop: step, compute the target (no bootstrap only if `terminated`), update, move on.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
self.cliff = [3 * 12 + c for c in range(1, 11)]
for s in range(self.n_states):
    if s in self.cliff or self.terminal[s]:
        continue
    for a in range(self.n_actions):
        if int(np.argmax(self.P[s, a])) in self.cliff:
            self.P[s, a] = 0.0
            self.P[s, a, self.start_state] = 1.0
            self.R[s, a] = -100.0
for c in self.cliff:
    self.P[c] = 0.0; self.P[c, :, c] = 1.0; self.R[c] = 0.0

# td0_prediction, inside the episode loop
s2, r, terminated, truncated, _ = env.step(policy(s))
target = r if terminated else r + gamma * V[s2]
V[s] += alpha * (target - V[s])
s, done = s2, terminated or truncated
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

SARSA carries the action across steps: pick `a` after `reset`, and inside the loop pick `a2` for the next state *before* updating (it's in the target), then set `a = a2`. Q-learning picks its action at the top of the loop and never needs the next one. keep a running `total` of rewards for `episode_returns`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# SARSA
a = epsilon_greedy(Q[s], epsilon, rng=rng)
while not done:
    s2, r, terminated, truncated, _ = env.step(a)
    total += r
    if terminated:
        target, a2 = r, None
    else:
        a2 = epsilon_greedy(Q[s2], epsilon, rng=rng)
        target = r + gamma * Q[s2, a2]
    Q[s, a] += alpha * (target - Q[s, a])
    s, a, done = s2, a2, terminated or truncated

# Q-learning's target
target = r if terminated else r + gamma * np.max(Q[s2])
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

copy your Q-learning and change only the target. build the ε-greedy probabilities of the next state's actions as a vector: `epsilon / nA` everywhere, plus `(1 - epsilon)` divided equally among the actions equal to the max.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
q = Q[s2]
greedy = q == q.max()
probs = np.full(nA, epsilon / nA) + (1 - epsilon) * greedy / greedy.sum()
target = r if terminated else r + gamma * probs @ q
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

store the episode as lists `S`, `A`, `Rw` with `Rw[0]` a dummy, so that `Rw[i]` is $R_i$; `T = np.inf` until the episode ends. at step `t`: if `t < T`, take `A[t]`, append the next state and reward, set `T = t + 1` if it ended, and append the next action unless it terminated. then update the pair at `tau = t - n + 1` if `tau >= 0`; stop after updating `tau == T - 1`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
tau = t - n + 1
if tau >= 0:
    m = int(min(tau + n, T))
    G = sum(gamma ** (i - tau - 1) * Rw[i] for i in range(tau + 1, m + 1))
    if m < T or not terminated:          # bootstrap unless the return reached a terminal state
        G += gamma ** (m - tau) * Q[S[m], A[m]]
    Q[S[tau], A[tau]] += alpha * (G - Q[S[tau], A[tau]])
if tau == T - 1:
    break
t += 1
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

it's your `td0_prediction` with three extra lines per step: decay the whole trace vector by `gamma * lam`, add 1 at the current state, and update *every* state by `alpha * delta * e` instead of just the current one. zero the trace at the start of each episode.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
e = np.zeros_like(V)
while not done:
    s2, r, terminated, truncated, _ = env.step(policy(s))
    delta = (r if terminated else r + gamma * V[s2]) - V[s]
    e *= gamma * lam
    e[s] += 1.0
    V += alpha * delta * e
    s, done = s2, terminated or truncated
```

</details>
