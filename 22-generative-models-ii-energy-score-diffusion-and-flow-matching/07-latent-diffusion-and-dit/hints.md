# latent diffusion & DiT · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

patchify is two reshapes and a permute: `x.reshape(B, C, H // p, p, W // p, p)` gives axes `(B, C, gh, i, gw, j)`; permute to `(B, gh, gw, i, j, C)` and reshape to `(B, gh * gw, p * p * C)`. unpatchify reverses it: `tokens.reshape(B, g, g, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)` with `g = math.isqrt(N)`. positions: `pe = sinusoidal_positions(grid_size, d_model // 2)`, rows `r = arange(g).repeat_interleave(g)`, columns `c = arange(g).repeat(g)`, and `torch.cat([pe[r], pe[c]], 1)`.

</details>

## stage 2

<details><summary>hint</summary>

the modulation network maps `c` (B, d) to six (B, d) vectors; insert a token axis so they broadcast over the N tokens of `x` (B, N, d). each branch is: normalize, modulate with `(1 + scale)` and `shift`, apply attention or the MLP, multiply by the gate, add to the residual stream. with the modulation layer's weights and bias at zero, every gate is zero and the block returns its input.

</details>

## stage 3

<details><summary>hint</summary>

assemble it in the order data flows: patch tokens plus the (buffered) positions; a conditioning vector from the time embedding plus the class row; the blocks, each given that vector; then a final non-affine LayerNorm modulated by `ada_out`'s shift and scale (the same `h (1 + scale) + shift` as in the blocks), `out`, and `unpatchify`. zero-initialize both `ada_out`'s linear layer and `out`, then check that a fresh model returns exactly zero.

</details>

## stage 4

<details><summary>hint</summary>

remember `was = vae.training`, call `vae.eval()`, and restore with `vae.train(was)` in a `finally`. inside `torch.no_grad()`: for each batch, `x, y = batch[0], batch[1]`, `mu, _ = vae.encode(x.to(device))`, append `mu.cpu()` and `torch.as_tensor(y).long()`; return the two concatenations. decoding: loop over `range(0, len(z), batch_size)` and append `vae.decode(z[i:i + batch_size].to(device) * scale).cpu()`.

</details>
