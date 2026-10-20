# RNNs & BPTT · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

allocate `hs = np.empty((B, T, H))` with `H = W_hh.shape[0]`, set `h = h0`, and loop over `t`: compute the new `h` from `x[:, t]` (shape `(B, D)`) and the old `h` (shape `(B, H)`), then store it in `hs[:, t]`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
h = h0
for t in range(T):
    h = np.tanh(x[:, t] @ W_xh.T + h @ W_hh.T + b)
    hs[:, t] = h
return hs
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

keep `dh`, the gradient arriving at $h_t$ from step $t + 1$ (zeros before the loop). for `t` from `T - 1` down to 0: add `dhs[:, t]`, multiply by `1 - hs[:, t] ** 2` to get `da`, use `h_prev = hs[:, t - 1]` (or `h0` when `t == 0`) for the recurrent weight gradient, and set `dh = da @ W_hh` for the next iteration. all the weight gradients are `+=`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for t in reversed(range(T)):
    dh = dh + dhs[:, t]
    da = dh * (1.0 - hs[:, t] ** 2)
    h_prev = hs[:, t - 1] if t > 0 else h0
    dW_xh += da.T @ x[:, t]
    dW_hh += da.T @ h_prev
    db += da.sum(axis=0)
    dx[:, t] = da @ W_xh
    dh = da @ W_hh
return {"dW_xh": dW_xh, "dW_hh": dW_hh, "db": db, "dh0": dh, "dx": dx}
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`self.ih = nn.ModuleList(nn.Linear(input_size if k == 0 else hidden_size, hidden_size) for k in range(num_layers))`, and `self.hh` likewise with `nn.Linear(hidden_size, hidden_size)`. in `forward`, loop over layers outside and time inside: `pre = self.ih[k](layer_in)` once, then `h = act(pre[:, t] + self.hh[k](h))` for each `t`, collecting the `h`s.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
layer_in, finals = x, []
for k in range(self.num_layers):
    h, outs = h0[k], []
    pre = self.ih[k](layer_in)
    for t in range(T):
        h = act(pre[:, t] + self.hh[k](h))
        outs.append(h)
    layer_in = torch.stack(outs, dim=1)
    finals.append(h)
return layer_in, torch.stack(finals, dim=0)
```

with `h0 = x.new_zeros(self.num_layers, B, self.hidden_size)` when it's `None`.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

repeat the forward pass of one layer by hand, appending every `h` to a list `hs` (don't stack them: autograd needs the individual tensors). `torch.autograd.grad(hs[-1].sum(), hs)` returns a tuple with one gradient per hidden state, each `(B, H)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
grads = torch.autograd.grad(hs[-1].sum(), hs)
return np.array([g.norm(dim=1).mean().item() for g in grads])
```

use `torch.tanh` or `torch.relu` according to `model.nonlinearity`.

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`batchify`: `n = len(ids) // batch_size`, then `ids[:n * batch_size].view(batch_size, n)` (row-major `view` makes each row a contiguous stretch). `tbptt_batches`: loop `i` over `range(0, N - 1, seq_len)` with `L = min(seq_len, N - 1 - i)`. `train_tbptt`: one Adam optimizer; for each epoch `h = None`, then for each chunk: forward with `h`, `h = h.detach()`, cross-entropy on the flattened logits and targets, backward, optional clipping, step.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for x, y in tbptt_batches(data, seq_len):
    logits, h = model(x, h)
    h = h.detach()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    if clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    losses.append(loss.item())
```

</details>
