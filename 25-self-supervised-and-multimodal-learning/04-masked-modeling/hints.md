# masked modeling · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`x = imgs.reshape(B, C, h, p, w, p)`, then `torch.einsum("nchpwq->nhwpqc", x).reshape(B, h * w, p * p * C)`. `unpatchify` runs the same steps backwards: `reshape(B, h, h, p, p, C)`, `einsum("nhwpqc->nchpwq")`, `reshape(B, C, h * p, h * p)`, with `h = math.isqrt(N)`.

</details>

## stage 2

<details><summary>hint</summary>

`n_keep = math.floor(N * (1 - mask_ratio) + 0.5)`. `noise = torch.rand(B, N, generator=generator).to(x.device)`; `ids_shuffle = noise.argsort(1)`; `ids_restore = ids_shuffle.argsort(1)`; `x_visible = torch.gather(x, 1, ids_shuffle[:, :n_keep, None].expand(-1, -1, D))`. for the mask: `mask = torch.ones(B, N); mask[:, :n_keep] = 0; mask = torch.gather(mask, 1, ids_restore)`.

</details>

## stage 3

<details><summary>hint</summary>

register the positional tables with `self.register_buffer("pos_embed", sinusoidal_positions(N, encoder_dim).unsqueeze(0))` so they move with `.to(device)` but aren't trained. blocks: `nn.ModuleList(TransformerBlock(encoder_dim, encoder_heads) for _ in range(encoder_depth))`. `encode`: `t = self.patch_embed(patchify(x, p)) + self.pos_embed`, every block, then `self.norm(t).mean(1)`.

</details>

## stage 4

<details><summary>hint</summary>

after the encoder: `y = self.decoder_embed(self.norm(t))`; `y = torch.cat([y, self.mask_token.expand(B, N - n_keep, -1)], 1)`; `y = torch.gather(y, 1, ids_restore[..., None].expand(-1, -1, y.shape[-1]))`; `y = y + self.decoder_pos_embed`; decoder blocks; `pred = self.decoder_pred(self.decoder_norm(y))`. loss: `per_patch = ((pred - target) ** 2).mean(-1)`, then `(per_patch * mask).sum() / mask.sum()`.

</details>

## stage 5

<details><summary>hint</summary>

it's `train_simclr`'s loop without augmentations: `torch.manual_seed(seed)`, `rng = np.random.default_rng(seed)`, `opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)`, and per batch `loss = model(X[perm[s:s + bs]].to(device))[0]`.

</details>
