# attribution & probing · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

remember `was = model.training`, call `model.eval()`, and restore with `model.train(was)` in a `finally`. make a leaf copy `x = x.detach().clone().requires_grad_(True)`, run `out = model(x)`, turn the target into a `(B,)` tensor with `torch.as_tensor(target).long().expand(out.shape[0])`, pick one logit per row with `out.gather(1, t[:, None]).sum()` (rows don't interact, so the sum's gradient is each row's own gradient), and get `torch.autograd.grad(picked, x)[0].abs()`.

</details>

## stage 2

<details><summary>hint</summary>

loop `k = 0..n_steps`: `xk = (base + k / n * (x - base)).requires_grad_(True)`, the gradient of the picked logits w.r.t. `xk` as in stage 1, and accumulate `w_k * grad` with `w_k = 0.5 / n` at the two ends and `1 / n` inside. return `(x - base) * total`. `torch.as_tensor(baseline).to(x).expand_as(x)` handles `(D,)` and `(B, D)` baselines alike. check yourself on the worked example before anything else.

</details>

## stage 3

<details><summary>hint</summary>

`uniq, inverse = np.unique(types, return_inverse=True)`, then `table = rng.integers(0, n_classes, len(uniq))` and `table[inverse]`. for the selectivity, concatenate the train and test types, call `control_labels` ONCE on the concatenation, and split the result back at `len(train_labels)`; then two calls of `linear_probe`.

</details>

## stage 4

<details><summary>hint</summary>

`scipy.stats.rankdata` gives average ranks; Spearman is the Pearson correlation of the ranks. for `randomize_layers`: `clone = copy.deepcopy(model)`, `layers = [m for m in clone.modules() if list(m.parameters(recurse=False))]`, and for each of the last `n`: draw a seed with `torch.randint(0, 2**62, (1,), generator=generator)`, then `with torch.random.fork_rng(devices=[]): torch.manual_seed(seed); m.reset_parameters()` (so the global RNG isn't disturbed). `cascading_randomization` calls `attribute` once on the model and once per $k$.

</details>
