# probability & random variables · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

build the new dict in a loop over `pmf.items()`: convert each probability with `float(p)`, check it, and keep it only if it's positive. check the total *after* the loop. `np.isfinite(p)` is false for `nan` and `inf`.

</details>

<details><summary>hint 2: the approach</summary>

`pmf(x)` is a dict lookup with a default: `self.probs.get(x, 0.0)`. for `prob(event)`, branch on `callable(event)`. for a predicate, sum `p` over the items where `event(x)` is true. for a collection, turn it into a `set` first (so duplicates count once) and sum `self.probs.get(x, 0.0)` over it.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
probs = {}
for x, p in dict(pmf).items():
    p = float(p)
    if not np.isfinite(p) or p < 0:
        raise ValueError(f"P({x!r}) = {p} is not a valid probability")
    if p > 0:
        probs[x] = p
if abs(sum(probs.values()) - 1.0) > 1e-9:
    raise ValueError("probabilities must sum to 1")
self.probs = probs
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

for `pushforward`, loop over `X.probs.items()`, compute `y = f(x)`, and *accumulate* `p` into a dict under key `y`. two different `x` can share a `y`.

</details>

<details><summary>hint 2: the approach</summary>

`product` is a dict comprehension over two loops: key `(x, y)`, value `px * py`. then `add` doesn't need any loop of its own: it's the pushforward of `product(X, Y)` through the function that adds the two coordinates of a pair.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def add(X, Y):
    return pushforward(product(X, Y), lambda xy: xy[0] + xy[1])
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

two steps: keep only the outcomes inside the event, then divide each kept probability by the total you kept. that total is `X.prob(event)`.

</details>

<details><summary>hint 2: the approach</summary>

it's convenient to turn a collection event into a predicate first, so the rest of the code handles one case: `inside = event if callable(event) else (lambda x: x in set(event))`. compute `p_event = X.prob(inside)`, raise `ValueError` if it's 0, and return `Discrete({x: p / p_event for x, p in X.probs.items() if inside(x)})`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
s = None if callable(event) else set(event)
inside = event if s is None else (lambda x: x in s)
p_event = X.prob(inside)
if p_event == 0:
    raise ValueError("cannot condition on an event of probability 0")
return Discrete({x: p / p_event for x, p in X.probs.items() if inside(x)})
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

a marginal is a pushforward through a projection. you already wrote `pushforward`: what function maps the tuple `t` to its `idx`-th coordinate?

</details>

<details><summary>hint 2: the approach</summary>

`marginal`: if `idx` is a tuple (or list), project with `lambda t: tuple(t[i] for i in idx)`, otherwise with `lambda t: t[idx]`. `is_independent`: compute `PX = marginal(J, 0)` and `PY = marginal(J, 1)`, then loop over every `x` in `PX.probs` and every `y` in `PY.probs` and compare `J.pmf((x, y))` with `PX.pmf(x) * PY.pmf(y)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
PX, PY = marginal(J, 0), marginal(J, 1)
return all(abs(J.pmf((x, y)) - px * py) <= tol
           for x, px in PX.probs.items() for y, py in PY.probs.items())
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`chain` looks just like `product`, except the second factor depends on the first: the inner loop runs over `likelihood(h).probs.items()`.

</details>

<details><summary>hint 2: the approach</summary>

the joint from `chain` has outcomes `(h, d)`. conditioning it on "the data equals `observed`" keeps the pairs with `hd[1] == observed`. then the marginal over coordinate 0 forgets the data and leaves a distribution over hypotheses. `condition` already raises if the observation is impossible.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def bayes(prior, likelihood, observed):
    joint = chain(prior, likelihood)
    return marginal(condition(joint, lambda hd: hd[1] == observed), 0)
```

</details>
