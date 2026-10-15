import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # residual networks · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. a deeper network can represent
    everything a shallower one can, yet plain deep networks train *worse*. the fix that
    made networks of 100+ layers routine is one line: add the input back. you'll build
    the residual block, the ResNet around it (the backbone chapters 18, 21, 25 and 28 reuse),
    and the bottleneck block that makes very deep ResNets affordable.

    | stage | you build |
    |---|---|
    | 1 | `BasicBlock`: two 3×3 convs with batch norm, plus a shortcut (identity or 1×1 projection) |
    | 2 | `ResNet`, `resnet18`: a stem, four stages of blocks, global average pooling, a linear head; `forward_features` |
    | 3 | `Bottleneck`: 1×1 reduce → 3×3 → 1×1 expand, the block of ResNet-50 and deeper |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class BasicBlock(nn.Module):
    """two 3x3 conv-BN layers; output relu(F(x) + shortcut(x)), or relu(F(x)) when residual=False."""

    def __init__(self, in_channels, out_channels, stride=1, *, residual=True):
        """stage 1: conv1 (3x3, stride, no bias), bn1, conv2 (3x3, no bias), bn2; shortcut = 1x1 conv + BN if the shape changes, else identity."""
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if residual and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """stage 1: relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x)); no addition when residual=False."""
        out = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))
        if self.residual:
            out = out + self.shortcut(x)
        return torch.relu(out)


@app.cell(hide_code=True)
def _():
    checks.run(1, BasicBlock=BasicBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ResNet(nn.Module):
    """stem -> stages of BasicBlocks (width doubles, resolution halves from stage 2 on) -> global average pool -> fc."""

    def __init__(self, block_counts=(2, 2, 2, 2), num_classes=10, *, in_channels=3, base_width=64, residual=True,
                 small_input=True):
        """stage 2: stem (3x3/1 conv-BN-ReLU if small_input, else 7x7/2 conv-BN-ReLU + 3x3/2 maxpool), self.layers, self.fc; feature_dim."""
        super().__init__()
        w = base_width
        if small_input:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, w, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(w), nn.ReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, w, 7, stride=2, padding=3, bias=False),
                                      nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        stages, c_in = [], w
        for i, n_blocks in enumerate(block_counts):
            c_out = base_width * 2 ** i
            blocks = []
            for j in range(n_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(BasicBlock(c_in, c_out, stride, residual=residual))
                c_in = c_out
            stages.append(nn.Sequential(*blocks))
        self.layers = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = c_in
        self.fc = nn.Linear(self.feature_dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward_features(self, x):
        """stage 2: the pooled features (B, feature_dim), before the head."""
        return torch.flatten(self.pool(self.layers(self.stem(x))), 1)

    def forward(self, x):
        """stage 2: logits (B, num_classes) = fc(forward_features(x))."""
        return self.fc(self.forward_features(x))


@app.function
def resnet18(num_classes=10, *, in_channels=3, residual=True):
    """stage 2: ResNet with block counts (2, 2, 2, 2) and base width 64, CIFAR stem."""
    return ResNet((2, 2, 2, 2), num_classes, in_channels=in_channels, residual=residual)


@app.cell(hide_code=True)
def _():
    checks.run(2, ResNet=ResNet, resnet18=resnet18, BasicBlock=BasicBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 (stride) -> 1x1 expand to mid_channels * expansion, each with BN; projection shortcut when the shape changes."""

    def __init__(self, in_channels, mid_channels, stride=1, *, expansion=4, residual=True):
        """stage 3: conv1 (1x1), bn1, conv2 (3x3, stride), bn2, conv3 (1x1 -> mid*expansion), bn3, shortcut; out_channels attribute."""
        super().__init__()
        out_channels = mid_channels * expansion
        self.residual, self.out_channels = residual, out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if residual and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """stage 3: relu(bn3(conv3(relu(bn2(conv2(relu(bn1(conv1(x)))))))) + shortcut(x))."""
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out = out + self.shortcut(x)
        return torch.relu(out)


@app.cell(hide_code=True)
def _():
    checks.run(3, Bottleneck=Bottleneck, BasicBlock=BasicBlock)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.cnn import resnet18` gives you this network (the chapter's capstone trains
    it on CIFAR-10, and chapters 18, 21, 25 and 28 reuse it as a backbone), and you may use
    `torchvision.models.resnet18`/`resnet50` and their blocks: you've built them.
    """)
    return


if __name__ == "__main__":
    app.run()
