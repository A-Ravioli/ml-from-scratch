# monte carlo methods · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

write the small pieces first and build `step` from them: `hand_value` (sum the cards; if there's an ace and `sum + 10 <= 21`, add 10 and report a usable ace), `draw`, and `observe` (the obs for `self.player` and `self.dealer[0]`, with the player's total clipped to 21). `reset` calls `super().reset(seed=seed)` first, then deals four cards in the order the spec gives. in `step`, hitting is "append a card, check for a bust"; sticking is "the dealer draws while below 17, then compare".

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def step(self, action):
    if not self.action_space.contains(action):
        raise ValueError(...)
    if action == 1:
        self.player.append(self.draw())
        total, _ = self.hand_value(self.player)
        if total > 21:
            return self.observe(), -1.0, True, False, {}
        return self.observe(), 0.0, False, False, {}
    while self.hand_value(self.dealer)[0] < 17:
        self.dealer.append(self.draw())
    mine, theirs = self.hand_value(self.player)[0], self.hand_value(self.dealer)[0]
    reward = 1.0 if theirs > 21 or mine > theirs else (0.0 if mine == theirs else -1.0)
    return self.observe(), reward, True, False, {}
```

`encode` is one line of arithmetic; `decode` inverts it with `//` and `%`.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`mc_prediction` has two phases per episode: generate (record the states and rewards until `terminated or truncated`), then learn (walk backwards accumulating `G = gamma * G + rewards[t]`). for first-visit, find each state's first time step before the backward pass, e.g. with a dict and `setdefault`. remember the seeding rule: `env.reset(seed=int(rng.integers(2**31)))` on the first episode only.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
first = {}
for t, s in enumerate(states):
    first.setdefault(s, t)
G = 0.0
for t in range(len(states) - 1, -1, -1):
    G = gamma * G + rewards[t]
    s = states[t]
    if not first_visit or first[s] == t:
        N[s] += 1
        V[s] += (G - V[s]) / N[s]
```

`epsilon_greedy` is topic 01's `select_arm` as a function: `rng = as_rng(rng)`, then the same two random decisions.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

it's `mc_prediction` with (state, action) pairs instead of states, and actions chosen by `epsilon_greedy(Q[obs], epsilon, rng=rng)` from the *current* `Q`. append `float(sum(rewards))` to the returns list after each episode, before the update.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
while not done:
    a = epsilon_greedy(Q[obs], epsilon, rng=rng)
    obs2, r, terminated, truncated, _ = env.step(a)
    pairs.append((obs, a)); rewards.append(r)
    obs, done = obs2, terminated or truncated
episode_returns.append(float(sum(rewards)))
# then the backward pass of stage 2, keyed by pairs[t], updating N[s, a] and Q[s, a]
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

generate an episode with the behaviour policy (`int(rng.choice(A, p=behavior_policy[obs]))`), then walk backwards with two running quantities: the return `G` and the weight `W`, starting at 1. at step $t$, first update $Q(S_t, A_t)$ with the current `W` (which covers only the actions *after* $t$), then multiply `W` by `target[s, a] / behavior[s, a]`. weighted IS keeps cumulative weights `C`; ordinary IS keeps visit counts and averages `W * G`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
G, W = 0.0, 1.0
for t in range(len(pairs) - 1, -1, -1):
    s, a = pairs[t]
    G = gamma * G + rewards[t]
    if weighted:
        if W == 0.0:
            break
        C[s, a] += W
        Q[s, a] += W / C[s, a] * (G - Q[s, a])
    else:
        C[s, a] += 1
        Q[s, a] += (W * G - Q[s, a]) / C[s, a]
    W *= target_policy[s, a] / behavior_policy[s, a]
```

</details>
