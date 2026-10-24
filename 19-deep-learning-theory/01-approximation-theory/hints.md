# approximation theory · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`slopes = np.diff(y) / np.diff(x)`, `coef = np.concatenate([[slopes[0]], np.diff(slopes)])`. build `nn.Sequential(nn.Linear(1, n - 1), nn.ReLU(), nn.Linear(n - 1, 1))` and, under `torch.no_grad()`, set the hidden weights to 1, the hidden biases to `-x[:-1]`, the output weights to `coef` (shape `(1, n - 1)`) and the output bias to `y[0]`. check `np.all(np.diff(x) > 0)` first.

</details>

## stage 2

<details><summary>hint</summary>

layers: `Linear(1, 2)`, then `k - 1` times `Linear(2, 2)`, a `ReLU` after each, and `Linear(2, 1)`. weights: first layer `[[1], [1]]` with biases `[0, -0.5]`; every middle layer `[[2, -4], [2, -4]]` with biases `[0, -0.5]` (it computes the tent of the previous output and splits it again); output `[[2, -4]]` with bias 0.

</details>

## stage 3

<details><summary>hint</summary>

`x = torch.linspace(lo, hi, n_grid, dtype=torch.float64).reshape(-1, 1)`; if `isinstance(f, nn.Module)`, `f = copy.deepcopy(f).double()`. with `v` the outputs as a NumPy array, `d1 = np.diff(v)`, `d2 = np.diff(d1)`, `bent = np.abs(d2) > tol * np.abs(d1).max()` (return 1 if that max is 0). the number of runs is the number of `True`s that follow a `False`, plus one if `bent[0]` is `True`: `np.sum(bent[1:] & ~bent[:-1]) + bent[0]`.

</details>

## stage 4

<details><summary>hint</summary>

`mlp_param_count` is the formula of idea 4. for `width_for_budget`, walk `w = 1, 2, ...`, keep the best `|P(w) - budget|` (update only on a strict improvement, so ties keep the smaller width), and stop once `P(w)` exceeds the budget: `P` is increasing, so nothing further can be closer.

</details>

## stage 5

<details><summary>hint</summary>

create `torch.optim.Adam(model.parameters(), lr=lr)` and `g = torch.Generator().manual_seed(seed)`; call `model.train()`. each step: set every `param_group["lr"]` to `lr * 0.5 * (1 + math.cos(math.pi * t / steps))`, draw `idx = torch.randint(0, n, (min(batch_size, n),), generator=g)`, compute the mean of the squared differences (reshape predictions and targets to the same shape), `zero_grad`, `backward`, `step`, and append `float(loss.detach())`. finish with `model.eval()`.

</details>
