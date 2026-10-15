# LSTMs & GRUs · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`self.ih = nn.Linear(input_size, 4 * hidden_size)` and `self.hh = nn.Linear(hidden_size, 4 * hidden_size)`, then `with torch.no_grad(): self.ih.bias[hidden_size:2 * hidden_size].fill_(1.0)`. in `forward`, `(self.ih(x) + self.hh(h)).chunk(4, dim=-1)` gives `i, f, g, o` in PyTorch's order.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if state is None:
    zeros = x.new_zeros(x.shape[0], self.hidden_size)
    state = (zeros, zeros)
h, c = state
i, f, g, o = (self.ih(x) + self.hh(h)).chunk(4, dim=-1)
c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
h = torch.sigmoid(o) * torch.tanh(c)
return h, c
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

chunk the two linear outputs separately: `xr, xz, xn = self.ih(x).chunk(3, dim=-1)` and `hr, hz, hn = self.hh(h).chunk(3, dim=-1)`. $r$ and $z$ are sigmoids of `xr + hr` and `xz + hz`; the reset gate multiplies `hn` (which already contains `hh`'s bias for that chunk).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
r = torch.sigmoid(xr + hr)
z = torch.sigmoid(xz + hz)
n = torch.tanh(xn + r * hn)
return (1 - z) * n + z * h
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

copy the structure of your topic-04 `RNN.forward`, replacing the update line with a call to the layer's cell. for the LSTM, carry `(h, c)` for each layer, starting from `state[0][k], state[1][k]`, and collect both final values. stack the finals with `torch.stack` to get `(num_layers, B, H)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
layer_in, hs, cs = x, [], []
for k, cell in enumerate(self.cells):
    h, c = state[0][k], state[1][k]
    outs = []
    for t in range(T):
        h, c = cell(layer_in[:, t], (h, c))
        outs.append(h)
    layer_in = torch.stack(outs, dim=1)
    hs.append(h)
    cs.append(c)
return layer_in, (torch.stack(hs), torch.stack(cs))
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`recall_batch`: `x = torch.randint(K, 2 * K, (B, delay + 1), generator=generator)`, then overwrite `x[:, 0]` with `torch.randint(0, K, (B,), generator=generator)` (in that order); `y = torch.full_like(x, -100)` and set its last column. `SeqTagger`: a dict `{"rnn": RNN, "gru": GRU, "lstm": LSTM}` picks the class. `copy_accuracy`: remember `model.training`, call `model.eval()`, predict under `torch.no_grad()`, restore in a `finally`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
with torch.no_grad():
    pred = model(x).argmax(-1)
mask = y != -100
return (pred[mask] == y[mask]).float().mean().item()
```

</details>
