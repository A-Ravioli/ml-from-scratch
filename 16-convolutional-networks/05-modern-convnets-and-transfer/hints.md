# modern convnets & transfer · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

the depthwise layer is `nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=in_channels, bias=False)`; the pointwise layer a $1 \times 1$ `nn.Conv2d(in_channels, out_channels, 1, bias=False)`. a `nn.BatchNorm2d` after each, with the right number of channels, and a ReLU after each batch norm.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def forward(self, x):
    return F.relu(self.bn2(self.pointwise(F.relu(self.bn1(self.depthwise(x))))))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)` (with bias), `self.norm = nn.LayerNorm(dim, eps=1e-6)`, `self.pwconv1 = nn.Linear(dim, expansion * dim)`, `self.pwconv2 = nn.Linear(expansion * dim, dim)`, `self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))`. in `forward`, `.permute(0, 2, 3, 1)` after the depthwise conv puts the channels last, where `LayerNorm` and `Linear` act; `.permute(0, 3, 1, 2)` puts them back before the residual addition.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
h = self.dwconv(x).permute(0, 2, 3, 1)
h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
return x + (self.gamma * h).permute(0, 3, 1, 2)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`replace_head`: pick the attribute name (`"fc"` if `hasattr(model, "fc")`, else `"head"`, else raise `AttributeError`), read the old layer with `getattr`, and `setattr` a new `nn.Linear(old.in_features, num_classes, device=old.weight.device, dtype=old.weight.dtype)`. `freeze_backbone`: loop over `model.named_parameters()` and set `p.requires_grad` to whether the name is `head_name` or starts with `head_name + "."`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for name, p in model.named_parameters():
    p.requires_grad = name == head_name or name.startswith(head_name + ".")
return model
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

after `replace_head` and `model.to(device)`, collect `head_params = list(getattr(model, head_name).parameters())` and their ids. linear: `freeze_backbone(model, head_name=head_name)` and `groups = [{"params": head_params, "lr": lr}]`. full: set every parameter's `requires_grad = True` and make two groups, the backbone (every parameter whose id isn't a head id) at `lr * backbone_lr_mult` and the head at `lr`. `torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)`. each epoch, set the modes first (linear: `model.eval()` then `head.train()`; full: `model.train()`), then the usual loop, accumulating `loss.item() * len(x)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for _ in range(epochs):
    if mode == "linear":
        model.eval()
        head.train()
    else:
        model.train()
    total, n = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total, n = total + loss.item() * len(x), n + len(x)
    history["train_loss"].append(total / max(n, 1))
```

</details>
