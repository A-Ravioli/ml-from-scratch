# classic architectures · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`__init__`: two `nn.Conv2d(..., 5)` layers without padding and three `nn.Linear` layers; the first linear layer takes `16 * 5 * 5 = 400` inputs. `forward`: if the input is smaller than $32 \times 32$, `F.pad(x, (left, right, top, bottom))` with the missing pixels split between the two sides (2 and 2 for a $28 \times 28$ digit). then `F.max_pool2d(F.relu(self.conv1(x)), 2)`, the same with `conv2`, `torch.flatten(x, 1)`, and the three linear layers with ReLUs between them but not after the last.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
h, w = x.shape[-2:]
if h < 32 or w < 32:
    ph, pw = max(32 - h, 0), max(32 - w, 0)
    x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
x = F.max_pool2d(F.relu(self.conv1(x)), 2)
x = F.max_pool2d(F.relu(self.conv2(x)), 2)
x = torch.flatten(x, 1)
x = F.relu(self.fc1(x))
x = F.relu(self.fc2(x))
return self.fc3(x)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

build a Python list of layers: for each width, twice append `nn.Conv2d(c_in, w, 3, padding=1, bias=not batch_norm)`, then `nn.BatchNorm2d(w)` if `batch_norm`, then `nn.ReLU()`, updating `c_in = w`; after the two convs append `nn.MaxPool2d(2)`. wrap the list in `nn.Sequential` as `self.features`. the head is `self.pool = nn.AdaptiveAvgPool2d(1)` and `self.fc = nn.Linear(widths[-1], num_classes)`; `forward` flattens the pooled map before `fc`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
layers, c_in = [], in_channels
for w in widths:
    for _ in range(2):
        layers.append(nn.Conv2d(c_in, w, 3, padding=1, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(w))
        layers.append(nn.ReLU())
        c_in = w
    layers.append(nn.MaxPool2d(2))
self.features = nn.Sequential(*layers)
self.pool = nn.AdaptiveAvgPool2d(1)
self.fc = nn.Linear(c_in, num_classes)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

make a dict `names = {module: name for name, module in model.named_modules()}` first. register a hook `hook(module, inputs, output)` on every `nn.Conv2d` and `nn.Linear` (keep the handles that `register_forward_hook` returns); inside it, take `out = output[0]` (one example) and append a row: conv MACs are `out.numel() * (module.in_channels // module.groups) * kh * kw`, linear MACs are `out.numel() * module.in_features`. then run `model(torch.zeros(1, *input_shape))` in eval mode under `torch.no_grad()`, and in a `finally:` remove every handle and restore `model.train(was_training)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def hook(module, inputs, output):
    out = output[0]
    params = sum(p.numel() for p in module.parameters(recurse=False))
    if isinstance(module, nn.Conv2d):
        kh, kw = module.kernel_size
        macs = out.numel() * (module.in_channels // module.groups) * kh * kw
    else:
        macs = out.numel() * module.in_features
    rows.append({"name": names[module], "out_shape": tuple(out.shape), "params": int(params), "macs": int(macs)})

hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
```

</details>
