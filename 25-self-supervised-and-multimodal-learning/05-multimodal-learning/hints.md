# multimodal learning · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

```python
logits = logit_scale * F.normalize(image_emb, dim=1) @ F.normalize(text_emb, dim=1).T
t = torch.arange(len(logits))
return (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2
```

</details>

## stage 2

<details><summary>hint</summary>

`real = tokens != self.pad_id`; `x = self.tok_emb(tokens) + self.pos_emb[:T]`; for each block, `x = blk(x, mask=real[:, None, None, :])`; `x = self.ln_f(x)`; then the masked mean `(x * real[..., None]).sum(1) / real.sum(1, keepdim=True).clamp_min(1)`. register the positions with `self.register_buffer("pos_emb", sinusoidal_positions(max_len, d_model))`.

</details>

## stage 3

<details><summary>hint</summary>

`self.log_scale = nn.Parameter(torch.tensor(math.log(1 / init_temperature)))`. `encode_image` is `F.normalize(self.image_proj(self.image_encoder(images)), dim=1)`, and the same for text. in `forward`, return the scale as `self.log_scale.exp().clamp(max=100.0)` (no `.item()`, no `.detach()`), so it keeps its gradient.

</details>

## stage 4

<details><summary>hint</summary>

`zero_shot_predict`: `(F.normalize(img, dim=1) @ F.normalize(cls, dim=1).T).argmax(1)`. `class_embeddings`: remember `was_training = model.training`, call `model.eval()`, and inside `torch.no_grad()` loop over the names: `emb = model.encode_text(tokenize([t.format(name) for t in templates]))`, then `F.normalize(F.normalize(emb, dim=1).mean(0), dim=0)`. stack, and `model.train(was_training)` in a `finally:`.

</details>

## stage 5

<details><summary>hint</summary>

the same skeleton as `train_mae`: `torch.manual_seed(seed)`, `rng = np.random.default_rng(seed)`, `Adam(model.parameters(), lr=lr)`; per epoch `model.train()` and `perm = rng.permutation(n)`; per batch `idx = perm[s:s + bs]`, `img, txt, scale = model(X[idx].to(device), T[idx].to(device))`, `loss = clip_loss(img, txt, scale)`. record `loss.item()` and `scale.item()`.

</details>
