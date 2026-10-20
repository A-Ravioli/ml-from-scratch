# catastrophic forgetting · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

flatten once: `flat = np.asarray(X).reshape(len(X), -1).astype(np.float32)`. `split_tasks`: for each group, `keep = np.isin(y, group)` and `torch.from_numpy(flat[keep].copy())` (the copy makes the array contiguous and writable). `permuted_tasks`: make one generator `rng = np.random.default_rng(seed)`, then `perms = [np.arange(d)] + [rng.permutation(d) for _ in range(n_tasks - 1)]`; each task is `flat[:, perm]`. since the permutations come from the seed alone, a second call with the test set reproduces them.

</details>

## stage 2

<details><summary>hint</summary>

convert with `R = np.asarray(R, dtype=float)` and `T = R.shape[0]`. ACC is `R[-1].mean()`; BWT is `np.mean(R[-1, :-1] - np.diag(R)[:-1])`; FWT averages `R[j - 1, j] - b[j]` over `j = 1..T-1`; forgetting averages `R[:T - 1, j].max() - R[T - 1, j]` over `j = 0..T-2`. wrap each in `float(...)` and raise `ValueError` first if `T < 2`.

</details>

## stage 3

<details><summary>hint</summary>

`scenario_logits`: build a boolean mask `torch.ones(n_classes, dtype=torch.bool)`, set `mask[list(classes)] = False`, and return `logits.masked_fill(mask, float("-inf"))` (it broadcasts over the batch). `evaluate`: remember `was = model.training`, call `model.eval()`, loop `for i in range(0, n, batch_size)` under `torch.no_grad()`, count `(out.argmax(1) == y_batch).sum()`, then `model.train(was)` and return `correct / n`.

</details>

## stage 4

<details><summary>hint</summary>

create the optimizer and `g = torch.Generator().manual_seed(seed)` once, before the epoch loop; each epoch, `order = torch.randperm(n, generator=g)` and step through `order[i:i + batch_size]`. for replay, `extra = replay()` and, if it isn't `None`, `torch.cat` it onto `x` and `y` before the forward pass. the loss is `F.cross_entropy(scenario_logits(model(x), task["classes"], scenario), y)`, plus `penalty(model)`. record `loss.item()` after the step, and call `after_step(model)` right after `opt.step()`.

</details>

## stage 5

<details><summary>hint</summary>

evaluate every test task once before training for the baseline. in the loop, the default branch is `train_task(model, task, scenario=scenario, **{"seed": t, **train_kwargs})` (so a caller's own `seed` would win), otherwise `train_fn(model, task, t)`. then `R[t] = [evaluate(model, tt, scenario=scenario) for tt in test_tasks]`, every column, not just `j <= t`.

</details>
