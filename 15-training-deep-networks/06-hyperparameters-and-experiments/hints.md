# hyperparameters & experiments · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`rng = as_rng(rng)` once, then a loop of `n_trials`: `config = sample_config(rng)`, `loss = float(evaluate(config, budget))`, append `(config, budget, loss)` to `history`. the best entry is `min(history, key=lambda h: h[2])` (which keeps the first of equal losses).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
rng = as_rng(rng)
history = []
for _ in range(n_trials):
    config = sample_config(rng)
    history.append((config, budget, float(evaluate(config, budget))))
best = min(history, key=lambda h: h[2])
return best[0], best[2], history
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`run_bracket`: `survivors = list(configs)`; for each rung `i`, compute `budget = rung_budget(min_budget, eta, i)`, score every survivor (appending to `history`), and unless it's the last rung keep the `max(len(scored) // eta, 1)` lowest losses, in order. `successive_halving`: sample `n_configs` configs, count the rungs (start at 1 and keep dividing `m = n_configs` by `eta` while `m // eta >= 1`), call `run_bracket`, and return the best of its last rung.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
survivors = list(configs)
for i in range(n_rungs):
    budget = rung_budget(min_budget, eta, i)
    scored = []
    for config in survivors:
        loss = float(evaluate(config, budget))
        history.append((config, budget, loss))
        scored.append((config, loss))
    if i < n_rungs - 1:
        keep = max(len(scored) // eta, 1)
        survivors = [c for c, _ in sorted(scored, key=lambda s: s[1])[:keep]]
return scored
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

find `s_max` with integers: start at 0 and increase it while `eta ** (s_max + 1) <= max_budget`. then loop `for s in range(s_max, -1, -1)`: `n = math.ceil((s_max + 1) * eta ** s / (s + 1))`, sample `n` configs, and call `run_bracket(configs, evaluate, n_rungs=s + 1, min_budget=max_budget / eta ** s, eta=eta, history=history)`. collect every bracket's returned last rung (all at `max_budget`) and pick the lowest loss.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
history, finalists = [], []
for s in range(s_max, -1, -1):
    n = math.ceil((s_max + 1) * eta ** s / (s + 1))
    configs = [sample_config(rng) for _ in range(n)]
    finalists += run_bracket(configs, evaluate, n_rungs=s + 1, min_budget=max_budget / eta ** s,
                             eta=eta, history=history)
best_config, best_loss = min(finalists, key=lambda f: f[1])
return best_config, best_loss, history
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`__init__`: `self.path = Path(path)` (or `None`), `self.runs = []`, and if the file exists read it line by line with `json.loads`. `log`: build the record with `dict(config)` and `{k: float(v) ...}`, append it, and if there's a path open the file in mode `"a"` and write `json.dumps(record) + "\n"`. `group`: use `json.dumps(config, sort_keys=True)` as a dictionary key so key order doesn't matter, collecting `(config, [values])`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def group(self, metric):
    groups = {}
    for r in self.runs:
        key = json.dumps(r["config"], sort_keys=True)
        groups.setdefault(key, (r["config"], []))[1].append(r["metrics"][metric])
    out = []
    for config, values in groups.values():
        v = np.asarray(values, dtype=float)
        out.append({"config": config, "n": len(v), "mean": float(v.mean()),
                    "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0})
    return out
```

</details>
