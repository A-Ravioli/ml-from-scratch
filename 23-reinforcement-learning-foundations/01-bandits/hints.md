# bandits · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

a bandit is just three attributes and a generator: `self.means = np.asarray(probs, dtype=float)`, `self.n_arms = len(self.means)`, `self.optimal_arm = int(np.argmax(self.means))`, `self.rng = as_rng(rng)`. a Bernoulli reward is "a uniform draw below the arm's probability". in `run_bandit`, the regret of one round is `max(means) - means[arm]`, whatever the reward was; accumulate it at the end with `np.cumsum`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
rewards, gaps = np.zeros(n_steps), np.zeros(n_steps)
best = np.max(bandit.means)
for t in range(n_steps):
    arm = agent.select_arm()
    r = bandit.pull(arm)
    agent.update(arm, r)
    rewards[t], gaps[t] = r, best - bandit.means[arm]
return rewards, np.cumsum(gaps)
```

`pull` returns `float(self.rng.random() < self.means[arm])` (Bernoulli) or `float(self.rng.normal(self.means[arm], self.std))` (Gaussian).

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

two random decisions, both from `self.rng`: first whether to explore (`self.rng.random() < self.epsilon`), then which arm. when exploring, any arm is fine, including the greedy one. when exploiting, collect *all* arms that tie for the max and pick one of them at random.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if self.rng.random() < self.epsilon:
    return int(self.rng.integers(self.n_arms))
best = np.flatnonzero(self.Q == self.Q.max())
return int(self.rng.choice(best))
```

and `update`: `self.N[arm] += 1; self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]`.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

handle the untried arms first (`np.flatnonzero(self.N == 0)`; return the first one). after that every `N` is at least 1, so the bonus is finite. the clock is `t = self.N.sum()`, the number of pulls made so far.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
untried = np.flatnonzero(self.N == 0)
if len(untried):
    return int(untried[0])
t = self.N.sum()
return int(np.argmax(self.Q + self.c * np.sqrt(np.log(t) / self.N)))
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`self.rng.beta(self.alpha, self.beta)` draws one sample per arm when given arrays. play the argmax of the *samples*, not of the posterior means $\alpha/(\alpha + \beta)$. for `lai_robbins_constant`, loop over the arms, skip any with mean $\ge \mu^*$, and write the Bernoulli KL with a special case for $\mu_a = 0$ ($0\ln 0 = 0$).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
best = mu.max()
total = 0.0
for m in mu:
    if m >= best:
        continue
    kl = (m * np.log(m / best) if m > 0 else 0.0) + (1 - m) * np.log((1 - m) / (1 - best))
    total += (best - m) / kl
```

if `best == 1` the second log is infinite and that arm contributes 0: wrap the line in `np.errstate(divide="ignore")`.

</details>
