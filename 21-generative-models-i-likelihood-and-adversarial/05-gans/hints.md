# GANs · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

with `F.softplus`: `discriminator_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()`; the non-saturating generator loss is `F.softplus(-fake_logits).mean()` and the saturating one is `-F.softplus(fake_logits).mean()` (because $\log(1 - \sigma(\ell)) = -\operatorname{softplus}(\ell)$).

</details>

## stage 2

<details><summary>hint</summary>

```python
z1 = torch.randn((B, latent_dim), generator=generator)
z2 = torch.randn((B, latent_dim), generator=generator)
opt_d.zero_grad()
d_loss = discriminator_loss(D(real).reshape(B), D(G(z1).detach()).reshape(B))
d_loss.backward(); opt_d.step()
opt_g.zero_grad()
g_loss = generator_loss(D(G(z2)).reshape(B), non_saturating=non_saturating)
g_loss.backward(); opt_g.step()
return d_loss.item(), g_loss.item()
```

(move `z1`, `z2` to `real.device` if you train on a GPU.)

</details>

## stage 3

<details><summary>hint</summary>

generator: `s = img_size // 4`; `Linear(latent_dim, 4b·s·s, bias=False), BatchNorm1d, ReLU`, reshape to `(B, 4b, s, s)`, then `ConvTranspose2d(4b, 2b, 4, stride=2, padding=1, bias=False), BatchNorm2d, ReLU, ConvTranspose2d(2b, b, 4, 2, 1, bias=False), BatchNorm2d, ReLU, Conv2d(b, C, 3, padding=1), Tanh`. discriminator: `Conv2d(C, b, 4, 2, 1), LeakyReLU(0.2), Conv2d(b, 2b, 4, 2, 1, bias=False), BatchNorm2d, LeakyReLU(0.2), Flatten, Linear(2b·s·s, 1)` and `.view(-1)` on the output. check `img_size % 4` first and call `dcgan_init(self)` at the end of each `__init__`.

</details>

## stage 4

<details><summary>hint</summary>

the distance matrix `d = np.sqrt(((x[:, None, :] - c[None, :, :])**2).sum(-1))` is `(n, K)`; `nearest = d.argmin(1)`; `good = d[np.arange(n), nearest] <= n_std * std`; `counts = np.bincount(nearest[good], minlength=K)`; covered modes are `(counts >= min_frac * n).sum()`, and the quality fraction is `good.mean()`.

</details>
