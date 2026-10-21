# diffusion (DDPM) · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

linear: `betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)`. cosine: `u = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps`, `f = torch.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2`, `ab = f / f[0]`, `betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)`. then store `betas.float()`, `(1 - betas).float()` and `torch.cumprod(1 - betas, 0).float()`. in `q_sample`, index with `t` and reshape: `ab = self.alpha_bars.to(x0.device)[t].view((-1,) + (1,) * (x0.dim() - 1))`.

</details>

## stage 2

<details><summary>hint</summary>

features: `half = dim // 2`, `freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)`, `args = 1000 * t[:, None] * freqs[None]`, `torch.cat([torch.cos(args), torch.sin(args)], 1)`, and `F.pad(emb, (0, 1))` for odd `dim`. `TimeMLP`: `c = time_emb(t)` (+ `label_emb(y)`, filling `y` with `n_classes` when it's `None`); `h = inp(x)`; for each layer, `h = h + lin(F.silu(h + cond_proj(c)))`; return `out(F.silu(h))`. one `nn.Linear(time_dim, hidden_dim)` per layer for the conditioning.

</details>

## stage 3

<details><summary>hint</summary>

draw the two random tensors in the order the spec gives (steps, then noise), both from `generator` on the CPU, then move them to `x0`'s device; build $x_t$ with your `q_sample`; the model sees the step divided by `n_steps`. the training loop has the same shape as every loop in this chapter (e.g. topic 02's `fit_dsm`): put the model in train mode, draw each minibatch's indices from the same generator you pass to the loss.

</details>

## stage 4

<details><summary>hint</summary>

copy the schedule to the CPU once (as float64) and use plain python floats inside the loop, so the sampler works whatever device the model is on. remember the model's mode before switching to eval and restore it in a `finally`; run everything under `torch.no_grad()`; and draw the per-step noise only for $i > 0$, so the draws line up with the spec's order.

</details>

## stage 5

<details><summary>hint</summary>

the whole difficulty is bookkeeping the skip connections. keep a list: push the feature map after the input convolution, after every residual block of the down path and after every downsampling; on the way up, every residual block pops one entry and concatenates it to its input, so the input width of each up block is (current width + popped width). that means each up level needs one more block than the corresponding down level. downsample with a stride-2 convolution, upsample with nearest-neighbour interpolation followed by a convolution, and add the (time + class) embedding inside each block. zero-initializing the final convolution makes the untrained network predict $\epsilon = 0$.

</details>
