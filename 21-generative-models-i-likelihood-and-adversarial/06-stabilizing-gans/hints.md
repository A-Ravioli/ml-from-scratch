# stabilizing GANs · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

the losses are one line each: `fake_scores.mean() - real_scores.mean()` and `-fake_scores.mean()`. for clipping, loop over `module.parameters()` inside `torch.no_grad()` and call `p.clamp_(-c, c)` (the trailing underscore means in place), then `return module`.

</details>

## stage 2

<details><summary>hint</summary>

```python
B = real.shape[0]
eps = torch.rand((B,) + (1,) * (real.dim() - 1), generator=generator)
x_hat = (eps * real + (1 - eps) * fake).detach().requires_grad_(True)
(grad,) = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)
return ((grad.reshape(B, -1).norm(dim=1) - 1) ** 2).mean()
```

summing the critic's outputs before `grad` gives each sample's gradient with respect to its own input, because the samples don't interact.

</details>

## stage 3

<details><summary>hint</summary>

`w = weight.reshape(weight.shape[0], -1)`; inside `torch.no_grad()`: `v = F.normalize(w.T @ u, dim=0)`, then the loop `v = F.normalize(w.T @ u, dim=0); u = F.normalize(w @ v, dim=0)`; outside it: `sigma = torch.dot(u, w @ v)` and `return weight / sigma, u, sigma`. in `SNLinear`, register `self.register_buffer("u", F.normalize(torch.randn(out_features), dim=0))`, and in `forward`, `w, u, _ = spectral_normalize(self.weight, self.u, self.n_power_iterations)`, `if self.training: self.u.copy_(u)`, then `F.linear(x, w, self.bias)`.

</details>

## stage 4

<details><summary>hint</summary>

two `torch.optim.Adam(..., lr=lr, betas=betas)`. per generator step: a loop of `n_critic` iterations that each call `sample_real()`, draw `z` (shape `(B, latent_dim)`, from `generator`), compute `fake = G(z).detach()`, the scores, `gp = gradient_penalty(critic, real, fake, generator=generator)` (after drawing `z`, so the random draws come in the specified order), and step the critic on `critic_loss + gp_lambda * gp`; then draw a fresh `z` and step the generator on `wgan_generator_loss(critic(G(z)))`. append the last critic step's `c_loss.item()`, `gp.item()` and `-c_loss.item()` to the three lists.

</details>
