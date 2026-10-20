# language modeling · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

make `counts = [defaultdict(Counter) for _ in range(n)]`, then for each sequence build `padded = ["<s>"] * (n - 1) + list(seq) + ["</s>"]` and loop `t` from `n - 1` to `len(padded) - 1`. at each `t`, every order `j` gets one count: `counts[j][tuple(padded[t - j:t])][padded[t]] += 1` (for `j = 0` the slice is empty, so the context is `()`). the vocabulary is every token you saw plus `"</s>"` and `"<unk>"`; sort it with `key=str` so mixed token types still sort.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
self.counts_ = [dict(c) for c in counts]
self.totals_ = [{ctx: sum(c.values()) for ctx, c in cj.items()} for cj in self.counts_]
self.vocab_ = sorted(vocab | {"</s>", "<unk>"}, key=str)
return self
```

build `counts` fresh inside `fit`, so a second call replaces the first.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

only the highest order is used: `self.counts_[self.n - 1].get(ctx)` is the Counter for this context, or `None` if it was never seen. the numerator is that count of `token` (0 if absent) plus `add_k`; the denominator is `self.totals_[self.n - 1].get(ctx, 0) + add_k * len(self.vocab_)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
c = self.counts_[self.n - 1].get(ctx)
num = (c.get(token, 0) if c else 0) + self.add_k
return num / (self.totals_[self.n - 1].get(ctx, 0) + self.add_k * len(self.vocab_))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`prob_interpolated`: start with the add-k unigram `p` (order 0, context `()`), add `lam[0] * p`, then for `j = 1 .. n-1` look at `sub = ctx[len(ctx) - j:]`. if `totals_[j]` has it, replace `p` by the ML estimate `counts_[j][sub].get(token, 0) / total`; if not, keep `p` as it is (that's the backoff). either way add `lam[j] * p`. `perplexity`: pad each sequence like `fit` does, sum `logprob(padded[:t], padded[t])` over every predicted position, count the positions, and return `exp(-sum / count)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
lam = [1.0 / self.n] * self.n if self.lambdas is None else [float(x) for x in self.lambdas]
# ... validate: n weights, non-negative, summing to 1 (ValueError otherwise)
p = (self.counts_[0][()].get(token, 0) + self.add_k) / (self.totals_[0][()] + self.add_k * len(self.vocab_))
total = lam[0] * p
for j in range(1, self.n):
    sub = ctx[len(ctx) - j:]
    if self.totals_[j].get(sub, 0):
        p = self.counts_[j][sub].get(token, 0) / self.totals_[j][sub]
    total += lam[j] * p
return total
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

first build the matrix `P` of shape (held-out tokens, n): for each order `j`, set `model.lambdas = np.eye(n)[j]` and collect `model.prob_interpolated(model._context(padded[:t]), token)` for every held-out position, with `token` replaced by `"<unk>"` when `not model._known(token)`. then EM is three lines of NumPy: `mix = P @ lam`, record `np.log(mix).sum()`, and `lam = (P * lam / mix[:, None]).mean(axis=0)`. stop after `n_iters` or when the gain is below `tol * len(P)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
lam = np.full(n, 1.0 / n)
history = []
for _ in range(n_iters):
    mix = P @ lam
    history.append(float(np.log(mix).sum()))
    if len(history) > 1 and history[-1] - history[-2] < tol * len(P):
        break
    lam = (P * lam / mix[:, None]).mean(axis=0)
model.lambdas = lam
return lam, history
```

raise `ValueError` first if `model.smoothing != "interpolated"`.

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

candidates are `[t for t in self.vocab_ if t != "<unk>"]`. at each step compute `logprob(history, t)` for every candidate, turn it into probabilities with `np.exp(logp - logp.max())` (`-inf` becomes 0, which is fine), normalize, and draw an index with `rng.choice(len(cands), p=p)`, where `rng = as_rng(rng)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
tok = cands[int(rng.choice(len(cands), p=p / p.sum()))]
out.append(tok)
history = [] if tok == "</s>" else history + [tok]
```

start with `history = list(context)`.

</details>
