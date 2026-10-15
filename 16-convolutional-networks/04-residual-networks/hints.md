# residual networks · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)`, `conv2` the same from `out_channels` with stride 1, a `nn.BatchNorm2d(out_channels)` after each. start with `self.shortcut = nn.Sequential()` (an empty `Sequential` returns its input) and replace it with `nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))` only when `residual` is true and the shape changes. in `forward`, compute the branch, add `self.shortcut(x)` if `self.residual`, and apply the ReLU last.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def forward(self, x):
    out = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))
    if self.residual:
        out = out + self.shortcut(x)
    return torch.relu(out)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

build the stem from `small_input`, then loop over `enumerate(block_counts)`: stage `i` has width `base_width * 2 ** i`, and block `j` of it has stride 2 if `i > 0 and j == 0`, else 1; keep a running `c_in` that becomes the stage's width after its first block. collect each stage's blocks in an `nn.Sequential`, and all stages in `self.layers = nn.Sequential(*stages)`. `self.feature_dim` is the last stage's width. initialize every `nn.Conv2d` in `self.modules()` with `nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")`. `forward_features` is stem → layers → pool → `torch.flatten(..., 1)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
stages, c_in = [], base_width
for i, n_blocks in enumerate(block_counts):
    c_out = base_width * 2 ** i
    blocks = []
    for j in range(n_blocks):
        blocks.append(BasicBlock(c_in, c_out, 2 if (i > 0 and j == 0) else 1, residual=residual))
        c_in = c_out
    stages.append(nn.Sequential(*blocks))
self.layers = nn.Sequential(*stages)
self.pool = nn.AdaptiveAvgPool2d(1)
self.feature_dim = c_in
self.fc = nn.Linear(self.feature_dim, num_classes)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

three convolutions: `conv1` $1 \times 1$ from `in_channels` to `mid_channels`, `conv2` $3 \times 3$ with the stride and padding 1, `conv3` $1 \times 1$ to `out_channels = mid_channels * expansion`; all without bias, each followed by its batch norm. the shortcut rule is the basic block's, with `out_channels` in place of the block's width. ReLUs after `bn1` and `bn2`, and after the addition.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def forward(self, x):
    out = torch.relu(self.bn1(self.conv1(x)))
    out = torch.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.residual:
        out = out + self.shortcut(x)
    return torch.relu(out)
```

</details>
