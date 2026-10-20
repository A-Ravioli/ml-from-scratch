# replay & architectural approaches · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

preallocate `self.x = torch.zeros((capacity, *x.shape[1:]), dtype=x.dtype)` and an int64 `self.y` on the first `add`, and keep `self.size`. loop over the rows: `self.n_seen += 1`; if `size < capacity`, write to slot `size` and increment it; otherwise `j = int(self.rng.integers(0, self.n_seen))` and write only if `j < capacity`. `sample(n)`: `idx = self.rng.choice(self.size, size=min(n, self.size), replace=False)`, then index both tensors with `torch.from_numpy(idx)`.

</details>

## stage 2

<details><summary>hint</summary>

keep a dict `seen` of per-class counts and compute stored counts with `torch.unique(self.y[:size], return_counts=True)`. when full: with `largest = max(counts.values())`, if the incoming class has fewer than `largest`, pick a class among those with count `largest` (`rng.integers` over the list), then a random slot among `torch.nonzero(self.y[:size] == victim)`; otherwise keep the example with probability `counts[c] / seen[c]` (`rng.random() < ...`) and overwrite a random slot of class `c`. `seen[c]` must already include the incoming example.

</details>

## stage 3

<details><summary>hint</summary>

`fit`: for `c in torch.unique(y).tolist()`, `xc = x[y == c]`, `means[c] = xc.mean(0)`, `stds[c] = xc.std(0, unbiased=False).clamp_min(min_std)`. `sample(n)`: `classes = torch.tensor(sorted(self.means))`, `y = classes[torch.randint(len(classes), (n,), generator=self.gen)]`, stack the per-example means and stds, and add `std * torch.randn(shape, generator=self.gen)`.

</details>

## stage 4

<details><summary>hint</summary>

build `self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))` and `self.head`. with one `g = torch.Generator().manual_seed(seed)`, for each layer make `masks = torch.zeros(n_tasks, h)` and set `masks[t, torch.randperm(h, generator=g)[:k]] = 1.0` per task, then `self.register_buffer(f"mask{i}", masks)`; `masks()` returns them with `getattr`. in `forward`, `h = torch.relu(layer(h)) * mask[t]` for each layer, then `self.head(h)`.

</details>
