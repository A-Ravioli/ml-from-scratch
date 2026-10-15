# matrix factorization · hints

open a hint only after you've been stuck for a while. this block is spec-driven: there's one hint per stage, and it points at the idea, not the code.

## stage 1

<details><summary>hint</summary>

the split is one boolean mask, `test = as_rng(rng).random(len(ratings)) < test_frac`, applied to all three arrays. for the biases, `np.bincount(items, weights=residuals, minlength=n_items)` sums residuals per item in one call, and `np.bincount(items, minlength=n_items)` counts ratings per item; divide the first by `lam + ` the second. compute the item residuals with the current user biases, then the user residuals with the item biases you just computed.

</details>

## stage 2

<details><summary>hint</summary>

group the rating indices by item once before the loop (`np.argsort(items, kind="stable")` split at the cumulative `np.bincount` counts, or a list of `np.flatnonzero(items == i)`), and the same for users. for one item: `X = np.c_[user_factors_[users[idx]], np.ones(len(idx))]`, `y = ratings[idx] - mu_ - user_bias_[users[idx]]`, `w = np.linalg.solve(X.T @ X + lam * np.eye(rank + 1), X.T @ y)`, then `item_factors_[i], item_bias_[i] = w[:-1], w[-1]`. an empty `idx` gives `w = 0` automatically. write `loss` first and assert it never goes up while you debug.

</details>

## stage 3

<details><summary>hint</summary>

in `step`, copy the two factor rows (`p = user_factors_[u].copy()`) before changing anything, compute `e` from the old values, then write all four updates from `e`, `p` and `q`. in `fit`, draw the initial factors exactly as in `ALS` and then, per epoch, `for t in rng.permutation(len(ratings)): self.step(users[t], items[t], ratings[t])`, using the same `rng` object (don't make a new one per epoch).

</details>

## stage 4

<details><summary>hint</summary>

this is one iteration of the loop body from your ALS user sweep, for a single user whose ratings you're given: the same `X`, `y` and `np.linalg.solve`, with `model.item_factors_`, `model.item_bias_` and `model.mu_`. with `items` empty, `X` has shape `(0, rank + 1)`, `X.T @ X` is all zeros, and the solve returns zeros, so you don't need a special case (but build `X` with `np.ones((len(items), rank + 1))` so the shape is right).

</details>
